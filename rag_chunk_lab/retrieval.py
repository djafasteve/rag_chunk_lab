from typing import Dict, List
from .indexing import retrieve

def get_candidates(doc_id: str, query: str, pipeline: str, top_k: int, data_dir: str) -> List[Dict]:
    return retrieve(doc_id, pipeline, query, top_k, data_dir)
