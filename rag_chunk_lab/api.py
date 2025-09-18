from fastapi import FastAPI, Query
from .config import DEFAULTS
from .retrieval import get_candidates
from .generation import build_answer_payload
import os

app = FastAPI(title='RAG Chunk Lab API')
DATA_DIR = os.environ.get('RAG_LAB_DATA', 'data')

@app.get('/ask')
def ask(doc_id: str = Query(...), question: str = Query(...), top_k: int = Query(DEFAULTS.top_k), use_llm: bool = Query(False)):
    results = []
    for pipe in ['fixed', 'structure', 'sliding']:
        cands = get_candidates(doc_id, question, pipe, top_k, DATA_DIR)
        payload = build_answer_payload(pipe, question, cands, max_sentences=DEFAULTS.max_sentences, use_llm=use_llm)
        results.append(payload)
    return {'doc_id': doc_id, 'question': question, 'results': results}
