from typing import List, Dict, Tuple
from .utils import tokenize_words, join_tokens, iter_headings

try:
    from sentence_transformers import SentenceTransformer
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False

def tokenize_pages_once(pages: List[Dict]) -> List[Tuple[Dict, List[str]]]:
    """OPTIMISATION: Tokenise toutes les pages une seule fois pour réutilisation"""
    return [(page, tokenize_words(page.get('text') or '')) for page in pages]

def fixed_chunks(pages: List[Dict], size_tokens: int, overlap_tokens: int, doc_id: str) -> List[Dict]:
    chunks: List[Dict] = []
    # OPTIMISATION: Tokenise une seule fois par page
    tokenized_pages = tokenize_pages_once(pages)

    for p, tokens in tokenized_pages:
        start = 0
        while start < len(tokens):
            end = min(start + size_tokens, len(tokens))
            text = join_tokens(tokens[start:end])
            chunks.append({
                'doc_id': doc_id,
                'page': p['page'],
                'start': start,
                'end': end,
                'section_title': None,
                'source_file': p.get('source_file', doc_id),
                'text': text
            })
            if end == len(tokens):
                break
            start = max(end - overlap_tokens, 0)
    return chunks

def structure_aware_chunks(pages: List[Dict], size_tokens: int, overlap_tokens: int, doc_id: str) -> List[Dict]:
    chunks: List[Dict] = []
    for p in pages:
        text = p.get('text') or ''
        boundaries = [(s,e,title) for s,e,title in iter_headings(text)]
        boundaries.sort()
        segments = []
        if not boundaries:
            segments = [(0, len(text), None)]
        else:
            for i, (s,e,title) in enumerate(boundaries):
                seg_start = e
                seg_end = boundaries[i+1][0] if i+1 < len(boundaries) else len(text)
                segments.append((seg_start, seg_end, title))
        for seg_start, seg_end, title in segments:
            seg_txt = text[seg_start:seg_end]
            tokens = tokenize_words(seg_txt)
            start = 0
            while start < len(tokens):
                end = min(start + size_tokens, len(tokens))
                ctext = join_tokens(tokens[start:end])
                chunks.append({
                    'doc_id': doc_id,
                    'page': p['page'],
                    'start': start,
                    'end': end,
                    'section_title': title,
                    'source_file': p.get('source_file', doc_id),
                    'text': ctext
                })
                if end == len(tokens):
                    break
                start = max(end - overlap_tokens, 0)
    return chunks

def sliding_window_chunks(pages: List[Dict], window: int, stride: int, doc_id: str) -> List[Dict]:
    chunks: List[Dict] = []
    # OPTIMISATION: Tokenise une seule fois par page
    tokenized_pages = tokenize_pages_once(pages)

    for p, tokens in tokenized_pages:
        if not tokens:
            continue
        start = 0
        while start < len(tokens):
            end = min(start + window, len(tokens))
            text = join_tokens(tokens[start:end])
            chunks.append({
                'doc_id': doc_id,
                'page': p['page'],
                'start': start,
                'end': end,
                'section_title': None,
                'source_file': p.get('source_file', doc_id),
                'text': text
            })
            if end == len(tokens):
                break
            start = start + stride
    return chunks

def semantic_chunks(pages: List[Dict], size_tokens: int, overlap_tokens: int, doc_id: str) -> List[Dict]:
    """
    Pipeline sémantique : utilise les mêmes chunks que fixed_chunks
    mais optimise pour la recherche sémantique avec des embeddings locaux.

    Avantages:
    - Comprend le sens des mots (synonymes, paraphrases)
    - Meilleure recherche en langage naturel
    - Trouve des passages pertinents même avec vocabulaire différent
    """
    if not SEMANTIC_AVAILABLE:
        raise ImportError("sentence-transformers non installé. Installer avec: pip install sentence-transformers")

    # Utilise la même logique de chunking que fixed_chunks pour cohérence
    return fixed_chunks(pages, size_tokens, overlap_tokens, doc_id)

def azure_semantic_chunks(pages: List[Dict], size_tokens: int, overlap_tokens: int, doc_id: str) -> List[Dict]:
    """
    Pipeline sémantique Azure : utilise les mêmes chunks que fixed_chunks
    mais optimise pour la recherche sémantique avec Azure OpenAI embeddings.

    Avantages:
    - Embeddings de qualité professionnelle Azure OpenAI
    - Meilleure compréhension contextuelle pour documents juridiques
    - Recherche multilingue optimisée
    - Pas besoin de modèle local lourd
    """
    # Pas besoin de vérifier SEMANTIC_AVAILABLE car utilise Azure OpenAI
    # La vérification se fait dans build_azure_semantic_index

    # Utilise la même logique de chunking que fixed_chunks pour cohérence
    return fixed_chunks(pages, size_tokens, overlap_tokens, doc_id)
