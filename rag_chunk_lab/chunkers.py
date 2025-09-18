from typing import List, Dict
from .utils import tokenize_words, join_tokens, iter_headings

def fixed_chunks(pages: List[Dict], size_tokens: int, overlap_tokens: int, doc_id: str) -> List[Dict]:
    chunks: List[Dict] = []
    for p in pages:
        tokens = tokenize_words(p.get('text') or '')
        start = 0
        while start < len(tokens):
            end = min(start + size_tokens, len(tokens))
            text = join_tokens(tokens[start:end])
            chunks.append({'doc_id': doc_id, 'page': p['page'], 'start': start, 'end': end, 'section_title': None, 'text': text})
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
                chunks.append({'doc_id': doc_id, 'page': p['page'], 'start': start, 'end': end, 'section_title': title, 'text': ctext})
                if end == len(tokens):
                    break
                start = max(end - overlap_tokens, 0)
    return chunks

def sliding_window_chunks(pages: List[Dict], window: int, stride: int, doc_id: str) -> List[Dict]:
    chunks: List[Dict] = []
    for p in pages:
        tokens = tokenize_words(p.get('text') or '')
        if not tokens:
            continue
        start = 0
        while start < len(tokens):
            end = min(start + window, len(tokens))
            text = join_tokens(tokens[start:end])
            chunks.append({'doc_id': doc_id, 'page': p['page'], 'start': start, 'end': end, 'section_title': None, 'text': text})
            if end == len(tokens):
                break
            start = start + stride
    return chunks
