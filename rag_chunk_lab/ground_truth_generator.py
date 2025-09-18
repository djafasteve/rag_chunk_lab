"""
Ground Truth Generator - Génération automatique de paires Q&A à partir de documents

Ce module utilise des LLMs (Ollama local ou Azure OpenAI) pour générer automatiquement
des paires question/réponse expertes à partir de documents juridiques ou techniques.
"""

import os
import json
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm

from .utils import load_document
from .chunkers import fixed_chunks


class LLMClient:
    """Interface pour différents clients LLM"""

    def __init__(self, provider: str, model: str, **kwargs):
        self.provider = provider
        self.model = model

        if provider == "ollama":
            self._init_ollama(**kwargs)
        elif provider == "azure":
            self._init_azure(**kwargs)
        else:
            raise ValueError(f"Provider {provider} not supported")

    def _init_ollama(self, base_url: str = "http://localhost:11434", **kwargs):
        """Initialize Ollama client"""
        try:
            import requests
            self.ollama_url = base_url.rstrip('/')
            # Test connection
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code != 200:
                raise ConnectionError("Cannot connect to Ollama")
        except Exception as e:
            raise ConnectionError(f"Ollama connection failed: {e}")

    def _init_azure(self, **kwargs):
        """Initialize Azure OpenAI client"""
        try:
            from langchain_openai import AzureChatOpenAI

            self.azure_client = AzureChatOpenAI(
                model=self.model,
                azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
                api_key=os.getenv('AZURE_OPENAI_API_KEY'),
                api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
                temperature=0.1,
                max_tokens=2000
            )
        except Exception as e:
            raise ConnectionError(f"Azure OpenAI connection failed: {e}")

    def generate(self, prompt: str) -> str:
        """Generate text using the configured LLM"""
        if self.provider == "ollama":
            return self._generate_ollama(prompt)
        elif self.provider == "azure":
            return self._generate_azure(prompt)

    def _generate_ollama(self, prompt: str) -> str:
        """Generate using Ollama"""
        import requests

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 500
            }
        }

        response = requests.post(f"{self.ollama_url}/api/generate", json=payload, timeout=60)
        if response.status_code == 200:
            return response.json()["response"].strip()
        else:
            raise Exception(f"Ollama generation failed: {response.text}")

    def _generate_azure(self, prompt: str) -> str:
        """Generate using Azure OpenAI"""
        from langchain_core.messages import HumanMessage

        response = self.azure_client.invoke([HumanMessage(content=prompt)])
        return response.content.strip()


class GroundTruthGenerator:
    """Générateur de ground truth automatique"""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

        # Prompt pour génération de questions expertes
        self.qa_generation_prompt = """
Tu es un expert juridique spécialisé dans l'analyse de documents techniques et réglementaires.
À partir du texte suivant, génère UNE question très spécifique et technique qui nécessite une expertise approfondie pour y répondre.

RÈGLES IMPORTANTES:
1. La question doit être PRÉCISE et technique, pas généraliste
2. Elle doit nécessiter une connaissance experte du domaine
3. La réponse doit être EXACTEMENT le texte fourni, sans modification
4. Focus sur les détails spécifiques, références d'articles, procédures, etc.

TEXTE SOURCE:
{text}

MÉTADONNÉES:
- Document: {document_name}
- Page: {page_number}

Réponds UNIQUEMENT au format JSON suivant:
{{"question": "ta question experte ici", "answer": "le texte source exactement tel qu'il est fourni"}}
"""

    def generate_from_folder(self,
                           folder_path: str,
                           output_path: str,
                           questions_per_doc: int = 10,
                           min_text_length: int = 200,
                           max_text_length: int = 800) -> str:
        """
        Génère un dataset de ground truth à partir d'un dossier de documents

        Args:
            folder_path: Chemin vers le dossier contenant les documents
            output_path: Chemin du fichier JSONL de sortie
            questions_per_doc: Nombre de questions par document
            min_text_length: Longueur minimale du texte pour générer une question
            max_text_length: Longueur maximale du texte

        Returns:
            Chemin du fichier généré
        """
        folder_path = Path(folder_path)
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        # Trouver tous les documents
        doc_files = []
        for ext in ['*.pdf', '*.txt', '*.md']:
            doc_files.extend(folder_path.glob(ext))

        if not doc_files:
            raise FileNotFoundError(f"No supported documents found in {folder_path}")

        print(f"📁 Found {len(doc_files)} documents in {folder_path}")

        all_qa_pairs = []

        # Traiter chaque document
        with tqdm(doc_files, desc="🔄 Processing documents", unit="doc") as doc_bar:
            for doc_file in doc_bar:
                doc_bar.set_description(f"📄 Processing {doc_file.name}")

                try:
                    qa_pairs = self._process_document(
                        doc_file,
                        questions_per_doc,
                        min_text_length,
                        max_text_length
                    )
                    all_qa_pairs.extend(qa_pairs)
                    print(f"  ✅ Generated {len(qa_pairs)} Q&A pairs from {doc_file.name}")

                except Exception as e:
                    print(f"  ❌ Error processing {doc_file.name}: {e}")
                    continue

        # Sauvegarder le dataset
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for qa in all_qa_pairs:
                f.write(json.dumps(qa, ensure_ascii=False) + '\n')

        print(f"\n🎉 Generated {len(all_qa_pairs)} expert Q&A pairs")
        print(f"💾 Saved to: {output_path}")

        return str(output_path)

    def _process_document(self,
                         doc_path: Path,
                         questions_count: int,
                         min_length: int,
                         max_length: int) -> List[Dict]:
        """Traite un document pour générer des paires Q&A"""

        # Charger le document
        pages = load_document(str(doc_path))

        # Créer des chunks de taille appropriée
        chunks = fixed_chunks(pages,
                            size_tokens=400,
                            overlap_tokens=50,
                            doc_id=doc_path.stem)

        # Filtrer les chunks par longueur
        valid_chunks = []
        for chunk in chunks:
            text_length = len(chunk['text'])
            if min_length <= text_length <= max_length:
                valid_chunks.append(chunk)

        if len(valid_chunks) < questions_count:
            print(f"  ⚠️  Only {len(valid_chunks)} valid chunks, generating {len(valid_chunks)} questions")
            questions_count = len(valid_chunks)

        # Sélectionner des chunks aléatoires
        selected_chunks = random.sample(valid_chunks, questions_count)

        qa_pairs = []

        # Générer Q&A pour chaque chunk
        with tqdm(selected_chunks, desc=f"  💭 Generating Q&A", leave=False, unit="chunk") as chunk_bar:
            for chunk in chunk_bar:
                try:
                    qa_pair = self._generate_qa_pair(chunk, doc_path.name)
                    if qa_pair:
                        qa_pairs.append(qa_pair)
                except Exception as e:
                    print(f"    ❌ Failed to generate Q&A: {e}")
                    continue

        return qa_pairs

    def _generate_qa_pair(self, chunk: Dict, document_name: str) -> Optional[Dict]:
        """Génère une paire Q&A à partir d'un chunk"""

        # Préparer le prompt
        prompt = self.qa_generation_prompt.format(
            text=chunk['text'],
            document_name=document_name,
            page_number=chunk.get('page', 'N/A')
        )

        # Générer avec le LLM
        try:
            response = self.llm.generate(prompt)

            # Parser la réponse JSON
            if response.strip().startswith('{'):
                qa_data = json.loads(response.strip())
            else:
                # Essayer de trouver le JSON dans la réponse
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    qa_data = json.loads(json_match.group())
                else:
                    raise ValueError("No JSON found in response")

            # Validation
            if not isinstance(qa_data, dict) or 'question' not in qa_data or 'answer' not in qa_data:
                raise ValueError("Invalid JSON structure")

            # Construire la paire finale avec métadonnées
            return {
                "question": qa_data["question"].strip(),
                "answer": qa_data["answer"].strip(),
                "source_document": document_name,
                "page": chunk.get('page', 'N/A'),
                "doc_section": chunk.get('section', ''),
                "chunk_id": chunk.get('chunk_id', ''),
                "generated_by": f"{self.llm.provider}:{self.llm.model}"
            }

        except Exception as e:
            print(f"      Error parsing LLM response: {e}")
            return None


def create_llm_client(provider: str, model: str = None, **kwargs) -> LLMClient:
    """
    Crée un client LLM selon le provider

    Args:
        provider: 'ollama' ou 'azure'
        model: nom du modèle (défaut: mistral:7b pour Ollama, variable d'env pour Azure)
        **kwargs: paramètres additionnels
    """
    if provider == "ollama":
        model = model or "mistral:7b"
    elif provider == "azure":
        model = model or os.getenv('AZURE_OPENAI_DEPLOYMENT', 'gpt-4o-mini')

    return LLMClient(provider, model, **kwargs)