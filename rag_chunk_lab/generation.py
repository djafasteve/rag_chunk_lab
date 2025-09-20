from typing import List, Dict
import re
import os
from .config import AZURE_CONFIG

try:
    from openai import AzureOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

def extractive_answer(question: str, passages: List[Dict], max_sentences: int = 4) -> str:
    q_terms = set(re.findall(r"\w+", question.lower()))
    scored_sents = []
    for p in passages:
        for sent in re.split(r"(?<=[.!?])\s+", p['text']):
            terms = set(re.findall(r"\w+", sent.lower()))
            score = len(q_terms & terms)
            scored_sents.append((score, sent))
    scored_sents.sort(reverse=True, key=lambda x: x[0])
    picked = [s for sc,s in scored_sents if s.strip()][:max_sentences]
    if not picked and passages:
        return passages[0]['text'][:600]
    return ' '.join(picked) if picked else ''

def llm_answer(question: str, passages: List[Dict]) -> str:
    if not OPENAI_AVAILABLE:
        return "Erreur: librairie openai non installée. Installer avec: pip install openai"
    
    if not AZURE_CONFIG.api_key or not AZURE_CONFIG.endpoint:
        return "Erreur: configuration Azure OpenAI manquante. Vérifiez AZURE_OPENAI_API_KEY et AZURE_OPENAI_ENDPOINT"
    
    try:
        client = AzureOpenAI(
            api_key=AZURE_CONFIG.api_key,
            api_version=AZURE_CONFIG.api_version,
            azure_endpoint=AZURE_CONFIG.endpoint
        )
        
        context = "\n\n".join([
            f"Document {p['meta']['doc_id']}, Page {p['meta']['page']}, Section: {p['meta']['section_title']}\n{p['text']}"
            for p in passages[:5]
        ])
        
        prompt = f"""Basé sur le contexte suivant, réponds à la question de manière précise et concise.

Contexte:
{context}

Question: {question}

Réponse:"""
        
        response = client.chat.completions.create(
            model=AZURE_CONFIG.deployment,
            messages=[
                {"role": "system", "content": "Tu es un assistant qui répond aux questions basées sur des documents fournis. Réponds de manière précise et concise en te basant uniquement sur le contexte fourni."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.1
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        return f"Erreur lors de l'appel Azure OpenAI: {str(e)}"

def get_azure_embedding(text: str) -> List[float]:
    """Génère un embedding à partir d'un texte via Azure OpenAI"""
    if not OPENAI_AVAILABLE:
        raise ValueError("Librairie openai non installée. Installer avec: pip install openai")
    
    if not AZURE_CONFIG.api_key or not AZURE_CONFIG.endpoint:
        raise ValueError("Configuration Azure OpenAI manquante. Vérifiez AZURE_OPENAI_API_KEY et AZURE_OPENAI_ENDPOINT")
    
    try:
        client = AzureOpenAI(
            api_key=AZURE_CONFIG.api_key,
            api_version=AZURE_CONFIG.api_version,
            azure_endpoint=AZURE_CONFIG.endpoint
        )
        
        response = client.embeddings.create(
            input=text,
            model=AZURE_CONFIG.embedding_deployment
        )
        
        return response.data[0].embedding
        
    except Exception as e:
        raise ValueError(f"Erreur lors de l'appel d'embedding Azure OpenAI: {str(e)}")

def build_answer_payload(pipeline: str, question: str, candidates: List[Dict], max_sentences: int, use_llm: bool = False):
    if use_llm:
        answer = llm_answer(question, candidates)
    else:
        answer = extractive_answer(question, candidates, max_sentences=max_sentences)
    
    sources = [{
        'score': c['score'],
        'doc_id': c['meta']['doc_id'],
        'page': c['meta']['page'],
        'start': c['meta']['start'],
        'end': c['meta']['end'],
        'section_title': c['meta']['section_title'],
        'source_file': c['meta'].get('source_file', c['meta']['doc_id']),
        'snippet': c['text'][:300]
    } for c in candidates]
    return {'pipeline': pipeline, 'answer': answer, 'sources': sources}
