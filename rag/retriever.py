from pathlib import Path
from typing import List, Dict, Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader


def _load_markdown(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)


def _load_file(path: Path) -> str:
    if path.suffix.lower() == ".md":
        return _load_markdown(path)
    if path.suffix.lower() == ".pdf":
        return _load_pdf(path)
    return ""


def load_corpus(corpus_path: str) -> List[Dict[str, Any]]:
    root = Path(corpus_path)
    documents: List[Dict[str, Any]] = []
    for path in sorted(root.glob("*")):
        if path.suffix.lower() not in {".md", ".pdf"}:
            continue
        text = _load_file(path)
        if not text.strip():
            continue
        documents.append({
            "title": path.stem,
            "path": str(path),
            "text": text,
        })
    return documents


def build_retriever(corpus_path: str) -> Dict[str, Any]:
    documents = load_corpus(corpus_path)
    vectorizer = TfidfVectorizer(stop_words=None)
    corpus_texts = [doc["text"] for doc in documents]
    matrix = vectorizer.fit_transform(corpus_texts) if corpus_texts else None
    return {
        "documents": documents,
        "vectorizer": vectorizer,
        "matrix": matrix,
    }


def retrieve(retriever: Dict[str, Any], query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    if not query or len(retriever["documents"]) == 0 or retriever["matrix"] is None:
        return []
    query_vector = retriever["vectorizer"].transform([query])
    scores = cosine_similarity(query_vector, retriever["matrix"]).flatten()
    ranking = scores.argsort()[::-1][:top_k]
    results: List[Dict[str, Any]] = []
    for idx in ranking:
        doc = retriever["documents"][idx]
        results.append({
            "title": doc["title"],
            "path": doc["path"],
            "score": float(scores[idx]),
            "text": doc["text"],
        })
    return results
