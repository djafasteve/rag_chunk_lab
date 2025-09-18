from typing import List, Dict
from pypdf import PdfReader
import re, os, json

def load_document(path: str) -> List[Dict]:
    # Load PDF or text/markdown. Returns list of pages [{'page': n, 'text': '...'}]
    ext = os.path.splitext(path)[1].lower()
    pages = []
    if ext == '.pdf':
        reader = PdfReader(path)
        for i, page in enumerate(reader.pages):
            try:
                txt = page.extract_text() or ''
            except Exception:
                txt = ''
            pages.append({'page': i+1, 'text': txt})
    else:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            txt = f.read()
        pages = [{'page': 1, 'text': txt}]
    return pages

def tokenize_words(text: str) -> List[str]:
    return re.findall(r"\w+|\S", text, re.UNICODE)

def join_tokens(tokens: List[str]) -> str:
    out = []
    for i, t in enumerate(tokens):
        if i > 0 and not re.match(r"\W$", tokens[i-1]) and not re.match(r"\W$", t):
            out.append(' ')
        out.append(t)
    return ''.join(out)

def iter_headings(text: str):
    # Yield (start, end, title) for simple headings common in regulatory docs
    pattern = re.compile(r"(?im)^(?P<title>\s*(Article\s+\d+\b|\bSection\s+\d+\b|\bChapitre\s+\d+\b|[A-Z][A-Z0-9 \-]{4,}))\s*$")
    for m in pattern.finditer(text):
        yield m.start(), m.end(), m.group('title').strip()

def save_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
