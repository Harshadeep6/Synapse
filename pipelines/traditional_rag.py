"""
Traditional RAG pipeline: chunk -> embed -> ChromaDB -> retrieve -> synthesize.
"""
import time
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(__file__).parent.parent / "data" / "filings"

_model = None
_collections = {}  # cache: label -> chromadb collection


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


def _chroma_name(label: str) -> str:
    """Sanitize label to a valid ChromaDB collection name."""
    import re
    name = re.sub(r"[^A-Za-z0-9._-]", "_", label)
    name = re.sub(r"[_.\-]+", "_", name).strip("_.-")
    name = name[:512]
    if len(name) < 3:
        name = (name + "___")[:3]
    return name


def build_index(label: str) -> chromadb.Collection:
    """Build (or return cached) ChromaDB collection for a filing."""
    if label in _collections:
        return _collections[label]

    txt_path = DATA_DIR / f"{label}.txt"
    pdf_path = DATA_DIR / f"{label}.pdf"

    # If .txt is missing but .pdf exists (e.g. user upload), extract text now
    if not txt_path.exists() and pdf_path.exists():
        import PyPDF2
        reader = PyPDF2.PdfReader(str(pdf_path))
        text = "\n\n".join(p.extract_text() or "" for p in reader.pages)
        txt_path.write_text(text, encoding="utf-8")

    if not txt_path.exists():
        raise FileNotFoundError(f"Filing not found: {txt_path}")

    text = txt_path.read_text(encoding="utf-8")
    chunks = _chunk_text(text)

    model = _get_model()
    embeddings = model.encode(chunks, show_progress_bar=False).tolist()

    client = chromadb.Client()
    collection = client.create_collection(_chroma_name(label))
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"{label}_chunk_{i}" for i in range(len(chunks))],
    )
    _collections[label] = collection
    return collection


def query(label: str, question: str, n_results: int = 4) -> dict:
    """
    Retrieve top chunks and synthesize an answer via Groq.
    Returns dict with keys: answer, chunks, latency_s
    """
    t0 = time.time()

    collection = build_index(label)
    model = _get_model()
    q_emb = model.encode([question]).tolist()

    results = collection.query(query_embeddings=q_emb, n_results=n_results)
    chunks = results["documents"][0]
    distances = results["distances"][0]
    # ChromaDB returns L2 distances; convert to a 0-1 similarity score
    similarities = [round(1 / (1 + d), 3) for d in distances]

    context = "\n\n---\n\n".join(chunks)
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    resp = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a financial analyst. Answer the question using ONLY "
                    "the provided context. Be concise and cite specific figures."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ],
        max_tokens=400,
    )
    answer = resp.choices[0].message.content.strip()
    latency = round(time.time() - t0, 2)

    return {
        "answer": answer,
        "chunks": [{"text": c, "similarity": s} for c, s in zip(chunks, similarities)],
        "latency_s": latency,
    }
