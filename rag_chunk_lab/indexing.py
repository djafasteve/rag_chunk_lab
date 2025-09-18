from typing import List, Dict
import os, joblib, json
from sklearn.feature_extraction.text import TfidfVectorizer
from .utils import save_json

def build_index(doc_id: str, pipeline_name: str, chunks: List[Dict], data_dir: str):
    os.makedirs(f"{data_dir}/{doc_id}/{pipeline_name}", exist_ok=True)
    texts = [c['text'] for c in chunks]
    meta = [{k: c[k] for k in ['doc_id','page','start','end','section_title']} for c in chunks]
    save_json(f"{data_dir}/{doc_id}/{pipeline_name}/chunks_meta.json", meta)
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = vectorizer.fit_transform(texts)
    joblib.dump(vectorizer, f"{data_dir}/{doc_id}/{pipeline_name}/tfidf_vectorizer.joblib")
    joblib.dump(tfidf, f"{data_dir}/{doc_id}/{pipeline_name}/tfidf_matrix.joblib")
    save_json(f"{data_dir}/{doc_id}/{pipeline_name}/chunks_texts.json", texts)

def retrieve(doc_id: str, pipeline_name: str, query: str, top_k: int, data_dir: str):
    import numpy as np
    base = f"{data_dir}/{doc_id}/{pipeline_name}"
    vectorizer = joblib.load(f"{base}/tfidf_vectorizer.joblib")
    tfidf = joblib.load(f"{base}/tfidf_matrix.joblib")
    with open(f"{base}/chunks_texts.json", 'r', encoding='utf-8') as f:
        texts = json.load(f)
    with open(f"{base}/chunks_meta.json", 'r', encoding='utf-8') as f:
        meta = json.load(f)
    q = vectorizer.transform([query])
    sims = (q @ tfidf.T).toarray().ravel()
    idxs = np.argsort(-sims)[:top_k]
    results = []
    for i in idxs:
        results.append({'score': float(sims[i]), 'text': texts[i], 'meta': meta[i]})
    return results
