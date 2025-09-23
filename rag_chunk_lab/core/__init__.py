"""
Core RAG functionality - Chunking, Indexing, Generation, Retrieval
"""

from .chunkers import fixed_chunks, structure_aware_chunks, sliding_window_chunks, semantic_chunks, azure_semantic_chunks
from .indexing import build_index
from .generation import build_answer_payload
from .retrieval import get_candidates
from .hierarchical_chunking import HierarchicalChunker

__all__ = [
    'fixed_chunks',
    'structure_aware_chunks',
    'sliding_window_chunks',
    'semantic_chunks',
    'azure_semantic_chunks',
    'build_index',
    'build_answer_payload',
    'get_candidates',
    'HierarchicalChunker'
]