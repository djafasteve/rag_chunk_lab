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
    """Générateur de ground truth automatique avec support multilingue"""

    def __init__(self, llm_client: LLMClient, language: str = 'fr', question_style: str = 'standard'):
        self.llm = llm_client
        self.language = language
        self.question_style = question_style

        # Prompts multilingues pour génération de questions expertes - style standard
        self.qa_generation_prompts_standard = {
            'fr': """
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
""",
            'en': """
You are a legal expert specialized in analyzing technical and regulatory documents.
From the following text, generate ONE very specific and technical question that requires deep expertise to answer.

IMPORTANT RULES:
1. The question must be PRECISE and technical, not generalist
2. It must require expert knowledge of the domain
3. The answer must be EXACTLY the provided text, without modification
4. Focus on specific details, article references, procedures, etc.

SOURCE TEXT:
{text}

METADATA:
- Document: {document_name}
- Page: {page_number}

Respond ONLY in the following JSON format:
{{"question": "your expert question here", "answer": "the source text exactly as provided"}}
""",
            'es': """
Eres un experto legal especializado en el análisis de documentos técnicos y reglamentarios.
A partir del siguiente texto, genera UNA pregunta muy específica y técnica que requiera experiencia profunda para responder.

REGLAS IMPORTANTES:
1. La pregunta debe ser PRECISA y técnica, no generalista
2. Debe requerir conocimiento experto del dominio
3. La respuesta debe ser EXACTAMENTE el texto proporcionado, sin modificación
4. Enfócate en detalles específicos, referencias de artículos, procedimientos, etc.

TEXTO FUENTE:
{text}

METADATOS:
- Documento: {document_name}
- Página: {page_number}

Responde ÚnicAMENTE en el siguiente formato JSON:
{{"question": "tu pregunta experta aquí", "answer": "el texto fuente exactamente como se proporcionó"}}
"""
        }

        # Prompts pour questions avec mots-clés minimaux (plus réalistes)
        self.qa_generation_prompts_minimal = {
            'fr': """
Tu es un expert juridique. À partir du texte suivant, génère UNE question NATURELLE qu'un utilisateur réel poserait.

RÈGLES CRITIQUES:
1. La question doit être formulée comme un utilisateur NON-EXPERT la poserait
2. NE PAS utiliser les mots-clés exacts ou termes techniques du texte
3. Utiliser un langage courant, synonymes, reformulations
4. La question doit être réaliste (ce qu'on demanderait vraiment à un chatbot juridique)
5. La réponse reste EXACTEMENT le texte fourni

EXEMPLES DE REFORMULATION:
- "Article 123" → "règle concernant...", "que dit la loi sur..."
- "Infraction" → "problème", "violation", "délit"
- "Procédure" → "comment faire", "les étapes", "le processus"

TEXTE SOURCE:
{text}

MÉTADONNÉES:
- Document: {document_name}
- Page: {page_number}

Réponds UNIQUEMENT au format JSON:
{{"question": "question reformulée sans mots-clés du texte", "answer": "le texte source exactement"}}
""",
            'en': """
You are a legal expert. From the following text, generate ONE NATURAL question that a real user would ask.

CRITICAL RULES:
1. Question must be formulated as a NON-EXPERT user would ask
2. DO NOT use exact keywords or technical terms from the text
3. Use common language, synonyms, reformulations
4. Question must be realistic (what one would actually ask a legal chatbot)
5. Answer remains EXACTLY the provided text

REFORMULATION EXAMPLES:
- "Article 123" → "rule about...", "what does the law say about..."
- "Violation" → "problem", "breach", "offense"
- "Procedure" → "how to", "the steps", "the process"

SOURCE TEXT:
{text}

METADATA:
- Document: {document_name}
- Page: {page_number}

Respond ONLY in JSON format:
{{"question": "reformulated question without text keywords", "answer": "the source text exactly"}}
""",
            'es': """
Eres un experto legal. A partir del siguiente texto, genera UNA pregunta NATURAL que un usuario real haría.

REGLAS CRÍTICAS:
1. La pregunta debe formularse como la haría un usuario NO-EXPERTO
2. NO usar palabras clave exactas o términos técnicos del texto
3. Usar lenguaje común, sinónimos, reformulaciones
4. La pregunta debe ser realista (lo que realmente se preguntaría a un chatbot legal)
5. La respuesta sigue siendo EXACTAMENTE el texto proporcionado

EJEMPLOS DE REFORMULACIÓN:
- "Artículo 123" → "regla sobre...", "qué dice la ley sobre..."
- "Infracción" → "problema", "violación", "delito"
- "Procedimiento" → "cómo hacer", "los pasos", "el proceso"

TEXTO FUENTE:
{text}

METADATOS:
- Documento: {document_name}
- Página: {page_number}

Responde ÚnicAMENTE en formato JSON:
{{"question": "pregunta reformulada sin palabras clave del texto", "answer": "el texto fuente exactamente"}}
"""
        }

        # Sélectionner le prompt selon le style demandé
        if question_style == 'minimal-keywords':
            prompts = self.qa_generation_prompts_minimal
        else:
            prompts = self.qa_generation_prompts_standard

        self.qa_generation_prompt = prompts.get(language, prompts['fr'])

    def generate_from_folder(self,
                           folder_path: str,
                           output_path: str,
                           questions_per_doc: int = 10,
                           min_text_length: int = 200,
                           max_text_length: int = 800,
                           allow_reuse: bool = False) -> str:
        """
        Génère un dataset de ground truth à partir d'un dossier de documents

        Args:
            folder_path: Chemin vers le dossier contenant les documents
            output_path: Chemin du fichier JSONL de sortie
            questions_per_doc: Nombre de questions par document
            min_text_length: Longueur minimale du texte pour générer une question
            max_text_length: Longueur maximale du texte
            allow_reuse: Permet de réutiliser les chunks pour générer plus de questions

        Returns:
            Chemin du fichier généré
        """
        folder_path = Path(folder_path)
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        # Trouver tous les documents supportés
        doc_files = []
        supported_extensions = ['*.pdf', '*.docx', '*.doc', '*.txt', '*.md']
        for ext in supported_extensions:
            doc_files.extend(folder_path.glob(ext))

        if not doc_files:
            raise FileNotFoundError(f"No supported documents found in {folder_path}")

        print(f"📁 Found {len(doc_files)} documents in {folder_path}")

        all_qa_pairs = []

        # Traiter chaque document pour générer exactement questions_per_doc questions par document
        print(f"🎯 Objectif: {questions_per_doc} questions par document supporté")

        with tqdm(doc_files, desc="🔄 Processing documents", unit="doc") as doc_bar:
            for doc_file in doc_bar:
                doc_bar.set_description(f"📄 Processing {doc_file.name}")

                try:
                    qa_pairs = self._process_document(
                        doc_file,
                        questions_per_doc,
                        min_text_length,
                        max_text_length,
                        allow_reuse
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

    def _generate_balanced(self,
                          doc_files: List[Path],
                          total_questions_per_doc: int,
                          min_length: int,
                          max_length: int,
                          allow_reuse: bool) -> List[Dict]:
        """Génère des questions en équilibrant entre tous les documents"""

        # Phase 1: Analyser tous les documents pour compter les chunks valides
        doc_analysis = []
        total_valid_chunks = 0

        print(f"🔍 Phase 1: Analyse des {len(doc_files)} documents...")

        for doc_file in tqdm(doc_files, desc="Analyse", unit="doc"):
            try:
                pages = load_document(str(doc_file))
                chunks = fixed_chunks(pages, size_tokens=400, overlap_tokens=50, doc_id=doc_file.stem)

                valid_chunks = []
                for chunk in chunks:
                    text_length = len(chunk['text'])
                    if min_length <= text_length <= max_length:
                        valid_chunks.append(chunk)

                doc_info = {
                    'file': doc_file,
                    'valid_chunks': valid_chunks,
                    'chunk_count': len(valid_chunks)
                }
                doc_analysis.append(doc_info)
                total_valid_chunks += len(valid_chunks)

                print(f"  📄 {doc_file.name}: {len(valid_chunks)} chunks valides")

            except Exception as e:
                print(f"  ❌ Erreur analyse {doc_file.name}: {e}")
                continue

        if total_valid_chunks == 0:
            print(f"❌ Aucun chunk valide trouvé dans les documents")
            return []

        # Phase 2: Calculer la répartition équilibrée
        total_questions_wanted = total_questions_per_doc * len(doc_analysis)

        print(f"⚖️ Phase 2: Répartition de {total_questions_wanted} questions sur {len(doc_analysis)} documents")

        # Répartition proportionnelle avec minimum garanti
        min_questions_per_doc = max(1, total_questions_wanted // (len(doc_analysis) * 3))  # Au moins 1/3 de la moyenne

        questions_allocation = []
        remaining_questions = total_questions_wanted

        for doc_info in doc_analysis:
            # Répartition proportionnelle au nombre de chunks, avec minimum
            if total_valid_chunks > 0:
                proportional = int((doc_info['chunk_count'] / total_valid_chunks) * total_questions_wanted)
                allocated = max(min_questions_per_doc, proportional)
            else:
                allocated = min_questions_per_doc

            # Ne pas dépasser ce qui est disponible (sauf si allow_reuse)
            if not allow_reuse:
                allocated = min(allocated, doc_info['chunk_count'])

            questions_allocation.append(allocated)
            remaining_questions -= allocated

        # Distribuer les questions restantes équitablement
        while remaining_questions > 0:
            for i in range(len(questions_allocation)):
                if remaining_questions <= 0:
                    break
                questions_allocation[i] += 1
                remaining_questions -= 1

        # Afficher la répartition
        for i, doc_info in enumerate(doc_analysis):
            allocation = questions_allocation[i]
            print(f"  🎯 {doc_info['file'].name}: {allocation} questions ({doc_info['chunk_count']} chunks)")

        # Phase 3: Générer les questions selon la répartition
        print(f"📝 Phase 3: Génération des questions...")

        all_qa_pairs = []

        for i, doc_info in enumerate(doc_analysis):
            questions_for_this_doc = questions_allocation[i]

            if questions_for_this_doc == 0:
                continue

            print(f"📄 Génération de {questions_for_this_doc} questions pour {doc_info['file'].name}")

            try:
                qa_pairs = self._generate_questions_for_chunks(
                    doc_info['valid_chunks'],
                    questions_for_this_doc,
                    doc_info['file'].name,
                    allow_reuse
                )
                all_qa_pairs.extend(qa_pairs)
                print(f"  ✅ Généré {len(qa_pairs)} Q&A pour {doc_info['file'].name}")

            except Exception as e:
                print(f"  ❌ Erreur génération {doc_info['file'].name}: {e}")
                continue

        return all_qa_pairs

    def _generate_questions_for_chunks(self,
                                     valid_chunks: List[Dict],
                                     questions_count: int,
                                     document_name: str,
                                     allow_reuse: bool) -> List[Dict]:
        """Génère des questions à partir d'une liste de chunks"""

        if len(valid_chunks) == 0:
            return []

        # Sélectionner les chunks
        if len(valid_chunks) < questions_count:
            if allow_reuse:
                selected_chunks = random.choices(valid_chunks, k=questions_count)
            else:
                selected_chunks = valid_chunks
        else:
            selected_chunks = random.sample(valid_chunks, questions_count)

        qa_pairs = []

        # Générer Q&A pour chaque chunk sélectionné
        for chunk in tqdm(selected_chunks, desc=f"  Q&A {document_name[:20]}...", leave=False, unit="chunk"):
            try:
                qa_pair = self._generate_qa_pair(chunk, document_name)
                if qa_pair:
                    qa_pairs.append(qa_pair)
            except Exception as e:
                continue  # Ignore les erreurs individuelles

        return qa_pairs

    def _process_document(self,
                         doc_path: Path,
                         questions_count: int,
                         min_length: int,
                         max_length: int,
                         allow_reuse: bool = False) -> List[Dict]:
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
            if allow_reuse:
                print(f"  🔄 Reusing {len(valid_chunks)} chunks to generate {questions_count} questions")
                # Réutiliser les chunks avec répétition pour atteindre le nombre demandé
                selected_chunks = random.choices(valid_chunks, k=questions_count)
            else:
                print(f"  ⚠️  Only {len(valid_chunks)} valid chunks, generating {len(valid_chunks)} questions")
                print(f"      Utilisez --allow-reuse pour générer plus de questions en réutilisant les chunks")
                questions_count = len(valid_chunks)
                # Utiliser tous les chunks disponibles
                selected_chunks = valid_chunks
        else:
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