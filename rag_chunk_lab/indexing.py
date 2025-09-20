from typing import List, Dict
import os, joblib, json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .utils import save_json
from .generation import get_azure_embedding
from .config import AZURE_CONFIG

try:
    from sentence_transformers import SentenceTransformer
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False

try:
    from openai import AzureOpenAI
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

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

def build_semantic_index(doc_id: str, texts: List[str], data_dir: str):
    """Construit un index sémantique avec des embeddings locaux"""
    if not SEMANTIC_AVAILABLE:
        raise ImportError("sentence-transformers non installé. Installer avec: pip install sentence-transformers")

    print(f"🧠 Génération des embeddings sémantiques locaux ({len(texts)} chunks)...")

    # Modèle français optimisé pour la recherche sémantique
    model = SentenceTransformer('dangvantuan/sentence-camembert-large')

    # Génération des embeddings (peut prendre quelques secondes)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # Sauvegarde des embeddings et du modèle
    np.save(f"{data_dir}/{doc_id}/semantic/embeddings.npy", embeddings)
    model.save(f"{data_dir}/{doc_id}/semantic/model/")

    print(f"✅ Embeddings sémantiques sauvegardés ({embeddings.shape})")

def build_azure_semantic_index(doc_id: str, texts: List[str], data_dir: str):
    """Construit un index sémantique avec Azure OpenAI embeddings"""
    if not AZURE_AVAILABLE:
        raise ImportError("openai non installé. Installer avec: pip install openai")

    if not AZURE_CONFIG.api_key or not AZURE_CONFIG.endpoint:
        raise ValueError("Configuration Azure OpenAI manquante. Vérifiez AZURE_OPENAI_API_KEY et AZURE_OPENAI_ENDPOINT")

    print(f"☁️ Génération des embeddings Azure OpenAI ({len(texts)} chunks)...")

    # Génération des embeddings via Azure OpenAI
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

    embeddings_array = np.array(embeddings)

    # Sauvegarde des embeddings
    os.makedirs(f"{data_dir}/{doc_id}/azure_semantic", exist_ok=True)
    np.save(f"{data_dir}/{doc_id}/azure_semantic/embeddings.npy", embeddings_array)

    print(f"✅ Embeddings Azure sauvegardés ({embeddings_array.shape})")

def retrieve(doc_id: str, pipeline_name: str, query: str, top_k: int, data_dir: str):
    base = f"{data_dir}/{doc_id}/{pipeline_name}"

    # Vérifier que le pipeline existe
    if not os.path.exists(base):
        raise FileNotFoundError(f"Pipeline '{pipeline_name}' non trouvé pour le document '{doc_id}'. Vérifiez que l'ingestion a réussi pour ce pipeline.")

    # Chargement des métadonnées et textes (commun)
    with open(f"{base}/chunks_texts.json", 'r', encoding='utf-8') as f:
        texts = json.load(f)
    with open(f"{base}/chunks_meta.json", 'r', encoding='utf-8') as f:
        meta = json.load(f)

    if pipeline_name == 'semantic':
        # Recherche sémantique locale
        return semantic_retrieve(doc_id, query, top_k, data_dir, texts, meta)
    elif pipeline_name == 'azure_semantic':
        # Recherche sémantique Azure
        return azure_semantic_retrieve(doc_id, query, top_k, data_dir, texts, meta)
    else:
        # Recherche TF-IDF classique
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
    if not SEMANTIC_AVAILABLE:
        raise ImportError("sentence-transformers non installé.")

    base = f"{data_dir}/{doc_id}/semantic"

    # Chargement du modèle et des embeddings
    model = SentenceTransformer(f"{base}/model/")
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
