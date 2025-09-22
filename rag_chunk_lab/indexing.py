from typing import List, Dict, Optional, Tuple
import os, joblib, json
import numpy as np
from functools import lru_cache
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .utils import save_json
from .generation import get_azure_embedding, get_azure_embeddings_batch
from .config import AZURE_CONFIG
from .monitoring import monitor_performance

try:
    from sentence_transformers import SentenceTransformer
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False

@lru_cache(maxsize=1)
def get_sentence_transformer():
    """Cache singleton pour SentenceTransformer - évite le rechargement du modèle 1.1GB"""
    if not SEMANTIC_AVAILABLE:
        raise ImportError("sentence-transformers non installé. Installer avec: pip install sentence-transformers")

    print("🧠 Chargement du modèle SentenceTransformer (une seule fois)...")
    return SentenceTransformer('dangvantuan/sentence-camembert-large')

@lru_cache(maxsize=10)
def load_index_data(doc_id: str, pipeline_name: str, data_dir: str):
    """Cache LRU pour les données d'index - évite les reloads JSON répétitifs"""
    base = f"{data_dir}/{doc_id}/{pipeline_name}"

    if not os.path.exists(base):
        raise FileNotFoundError(f"Pipeline '{pipeline_name}' non trouvé pour le document '{doc_id}'. Vérifiez que l'ingestion a réussi pour ce pipeline.")

    with open(f"{base}/chunks_texts.json", 'r', encoding='utf-8') as f:
        texts = json.load(f)
    with open(f"{base}/chunks_meta.json", 'r', encoding='utf-8') as f:
        meta = json.load(f)

    return texts, meta

try:
    from openai import AzureOpenAI
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

@monitor_performance("build_index")
def build_index(doc_id: str, pipeline_name: str, chunks: List[Dict], data_dir: str):
    os.makedirs(f"{data_dir}/{doc_id}/{pipeline_name}", exist_ok=True)
    texts = [c['text'] for c in chunks]
    meta = [{k: c[k] for k in ['doc_id','page','start','end','section_title','source_file']} for c in chunks]
    save_json(f"{data_dir}/{doc_id}/{pipeline_name}/chunks_meta.json", meta)
    save_json(f"{data_dir}/{doc_id}/{pipeline_name}/chunks_texts.json", texts)

    if pipeline_name == 'semantic':
        # Pipeline sémantique : utilise des embeddings au lieu de TF-IDF
        build_semantic_index(doc_id, texts, data_dir)
    elif pipeline_name == 'azure_semantic':
        # Pipeline sémantique Azure : utilise Azure OpenAI embeddings
        build_azure_semantic_index(doc_id, texts, data_dir)
    else:
        # Pipelines classiques : TF-IDF
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf = vectorizer.fit_transform(texts)
        joblib.dump(vectorizer, f"{data_dir}/{doc_id}/{pipeline_name}/tfidf_vectorizer.joblib")
        joblib.dump(tfidf, f"{data_dir}/{doc_id}/{pipeline_name}/tfidf_matrix.joblib")

@monitor_performance("build_semantic_index")
def build_semantic_index(doc_id: str, texts: List[str], data_dir: str):
    """Construit un index sémantique avec des embeddings locaux"""
    print(f"🧠 Génération des embeddings sémantiques locaux ({len(texts)} chunks)...")

    # Utilise le modèle en cache singleton
    model = get_sentence_transformer()

    # Génération des embeddings (peut prendre quelques secondes) - OPTIMISATION: float32
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype(np.float32)  # 50% moins de mémoire

    # Sauvegarde des embeddings et du modèle
    np.save(f"{data_dir}/{doc_id}/semantic/embeddings.npy", embeddings)
    model.save(f"{data_dir}/{doc_id}/semantic/model/")

    print(f"✅ Embeddings sémantiques sauvegardés ({embeddings.shape})")

@monitor_performance("build_azure_semantic_index")
def build_azure_semantic_index(doc_id: str, texts: List[str], data_dir: str):
    """Construit un index sémantique avec Azure OpenAI embeddings"""
    if not AZURE_AVAILABLE:
        raise ImportError("openai non installé. Installer avec: pip install openai")

    if not AZURE_CONFIG.api_key or not AZURE_CONFIG.endpoint:
        raise ValueError("Configuration Azure OpenAI manquante. Vérifiez AZURE_OPENAI_API_KEY et AZURE_OPENAI_ENDPOINT")

    print(f"☁️ Génération des embeddings Azure OpenAI ({len(texts)} chunks)...")

    # OPTIMISATION: Génération par batch au lieu d'appels individuels
    try:
        embeddings = get_azure_embeddings_batch(texts, batch_size=100)
        embeddings_array = np.array(embeddings, dtype=np.float32)  # 50% moins de mémoire
    except Exception as e:
        print(f"⚠️ Échec du batch, retour aux appels individuels: {e}")
        # Fallback vers l'ancienne méthode si le batch échoue
        embeddings = []
        for i, text in enumerate(texts):
            if i % 10 == 0:
                print(f"  Progression: {i+1}/{len(texts)}")
            try:
                embedding = get_azure_embedding(text)
                embeddings.append(embedding)
            except Exception as e:
                print(f"Erreur embedding chunk {i}: {e}")
                # Fallback: embedding zéro
                embeddings.append([0.0] * 1536)  # Dimension standard Azure embeddings
        embeddings_array = np.array(embeddings, dtype=np.float32)  # Cohérence float32

    # Sauvegarde des embeddings
    os.makedirs(f"{data_dir}/{doc_id}/azure_semantic", exist_ok=True)
    np.save(f"{data_dir}/{doc_id}/azure_semantic/embeddings.npy", embeddings_array)

    print(f"✅ Embeddings Azure sauvegardés ({embeddings_array.shape})")

def retrieve(doc_id: str, pipeline_name: str, query: str, top_k: int, data_dir: str):
    # Utilise le cache LRU pour charger les données d'index
    texts, meta = load_index_data(doc_id, pipeline_name, data_dir)

    if pipeline_name == 'semantic':
        # Recherche sémantique locale
        return semantic_retrieve(doc_id, query, top_k, data_dir, texts, meta)
    elif pipeline_name == 'azure_semantic':
        # Recherche sémantique Azure
        return azure_semantic_retrieve(doc_id, query, top_k, data_dir, texts, meta)
    else:
        # Recherche TF-IDF classique
        base = f"{data_dir}/{doc_id}/{pipeline_name}"
        vectorizer = joblib.load(f"{base}/tfidf_vectorizer.joblib")
        tfidf = joblib.load(f"{base}/tfidf_matrix.joblib")
        q = vectorizer.transform([query])
        sims = (q @ tfidf.T).toarray().ravel()
        idxs = np.argsort(-sims)[:top_k]
        results = []
        for i in idxs:
            results.append({'score': float(sims[i]), 'text': texts[i], 'meta': meta[i]})
        return results

def semantic_retrieve(doc_id: str, query: str, top_k: int, data_dir: str, texts: List[str], meta: List[Dict]):
    """Recherche sémantique avec embeddings locaux"""
    base = f"{data_dir}/{doc_id}/semantic"

    # Utilise le modèle en cache singleton au lieu de recharger depuis le disque
    model = get_sentence_transformer()
    chunk_embeddings = np.load(f"{base}/embeddings.npy")

    # Embedding de la requête
    query_embedding = model.encode([query], convert_to_numpy=True)

    # Calcul de similarité cosinus
    similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]

    # Top-k résultats
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for i in top_indices:
        results.append({
            'score': float(similarities[i]),
            'text': texts[i],
            'meta': meta[i]
        })

    return results

def azure_semantic_retrieve(doc_id: str, query: str, top_k: int, data_dir: str, texts: List[str], meta: List[Dict]):
    """Recherche sémantique avec Azure OpenAI embeddings"""
    if not AZURE_AVAILABLE:
        raise ImportError("openai non installé.")

    base = f"{data_dir}/{doc_id}/azure_semantic"

    # Chargement des embeddings des chunks
    chunk_embeddings = np.load(f"{base}/embeddings.npy")

    # Embedding de la requête via Azure
    try:
        query_embedding_list = get_azure_embedding(query)
        query_embedding = np.array([query_embedding_list])
    except Exception as e:
        raise ValueError(f"Erreur génération embedding requête: {e}")

    # Calcul de similarité cosinus
    similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]

    # Top-k résultats
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for i in top_indices:
        results.append({
            'score': float(similarities[i]),
            'text': texts[i],
            'meta': meta[i]
        })

    return results
