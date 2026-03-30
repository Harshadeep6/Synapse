"""
PageIndex RAG pipeline: submit doc -> poll ready -> query -> get answer.
Tree index is cached to disk so documents are only submitted once.
"""
import os
import time
import json
import hashlib
import pathlib
from dotenv import load_dotenv
from pageindex import PageIndexClient

load_dotenv()

DATA_DIR = pathlib.Path(__file__).parent.parent / "data" / "filings"
CACHE_DIR = pathlib.Path(__file__).parent.parent / "data" / "pageindex_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_client: PageIndexClient | None = None


def _get_client() -> PageIndexClient:
    global _client
    if _client is None:
        _client = PageIndexClient(api_key=os.getenv("PAGEINDEX_API_KEY"))
    return _client


def _cache_path(label: str) -> pathlib.Path:
    return CACHE_DIR / f"{label}.json"


def _load_cache(label: str) -> dict | None:
    p = _cache_path(label)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def _save_cache(label: str, data: dict):
    _cache_path(label).write_text(json.dumps(data, indent=2), encoding="utf-8")


def _pdf_hash(pdf_path: pathlib.Path) -> str:
    """MD5 of first 64 KB — fast fingerprint to match against existing docs."""
    data = pdf_path.read_bytes()[:65536]
    return hashlib.md5(data).hexdigest()


def _find_existing_doc(pdf_path: pathlib.Path, client: "PageIndexClient") -> str | None:
    """
    Check if this PDF is already indexed under any doc_id by comparing filename.
    PageIndex list_documents() returns metadata we can match against.
    """
    try:
        docs = client.list_documents()
        # list_documents may return a dict with a list, or a list directly
        if isinstance(docs, dict):
            docs = docs.get("documents") or docs.get("data") or docs.get("results") or []
        pdf_name = pdf_path.name
        for doc in docs:
            if isinstance(doc, dict):
                name = doc.get("filename") or doc.get("name") or doc.get("file_name") or ""
                if name == pdf_name or name.endswith(pdf_name):
                    return doc.get("doc_id") or doc.get("id") or ""
    except Exception:
        pass
    return None


def get_doc_id(label: str) -> str:
    """
    Submit document to PageIndex if not already cached, then return its doc_id.
    Polls until indexing is complete.
    """
    cache = _load_cache(label)
    if cache and cache.get("doc_id"):
        return cache["doc_id"]

    pdf_path = DATA_DIR / f"{label}.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}. Run download_filings.py first.")

    client = _get_client()

    # Before submitting, check if this file is already on PageIndex (avoids hitting quota)
    existing_id = _find_existing_doc(pdf_path, client)
    if existing_id:
        print(f"  Reusing existing PageIndex doc_id: {existing_id}")
        tree = client.get_tree(existing_id, node_summary=True)
        _save_cache(label, {"doc_id": existing_id, "tree": tree})
        return existing_id

    print(f"  Submitting {label} to PageIndex API...")
    try:
        result = client.submit_document(file_path=str(pdf_path))
    except Exception as e:
        if "LimitReached" in str(e):
            raise RuntimeError(
                "PageIndex free-tier page limit reached. "
                "Try uploading a shorter document (fewer pages), "
                "or use one of the pre-indexed filings (Apple / Microsoft)."
            ) from e
        raise
    doc_id = result["doc_id"]
    print(f"  doc_id: {doc_id} | Waiting for indexing to complete...")

    # Poll until ready
    while not client.is_retrieval_ready(doc_id):
        print("  ... still indexing, waiting 10s ...")
        time.sleep(10)

    print(f"  Indexing complete for {label}.")

    # Fetch and cache the tree
    tree = client.get_tree(doc_id, node_summary=True)
    _save_cache(label, {"doc_id": doc_id, "tree": tree})
    return doc_id


def get_tree(label: str) -> list[dict]:
    """Return cached tree nodes list (or fetch after ensuring doc is indexed)."""
    cache = _load_cache(label)
    if cache and cache.get("tree"):
        raw = cache["tree"]
        # Nodes are under the "result" key
        return raw.get("result") or raw.get("nodes") or []
    get_doc_id(label)
    raw = _load_cache(label)["tree"]
    return raw.get("result") or raw.get("nodes") or []


def query(label: str, question: str) -> dict:
    """
    Run a question against a PageIndex document.
    Returns dict with keys: answer, nodes_visited, retrieval_raw, latency_s
    """
    t0 = time.time()

    client = _get_client()
    doc_id = get_doc_id(label)

    # Submit retrieval query
    retrieval_resp = client.submit_query(doc_id=doc_id, query=question)
    retrieval_id = retrieval_resp["retrieval_id"]

    # Poll for retrieval result
    retrieval_result = None
    for _ in range(30):  # max ~60s
        r = client.get_retrieval(retrieval_id)
        if r.get("status") == "completed":
            retrieval_result = r
            break
        time.sleep(2)

    if retrieval_result is None:
        raise TimeoutError("PageIndex retrieval did not complete in time.")

    # Extract retrieved nodes/sections for display
    nodes_visited = _extract_nodes(retrieval_result)

    # Get final answer via chat_completions scoped to this doc
    chat_resp = client.chat_completions(
        messages=[{"role": "user", "content": question}],
        doc_id=doc_id,
        enable_citations=True,
    )

    answer = _extract_answer(chat_resp)
    latency = round(time.time() - t0, 2)

    return {
        "answer": answer,
        "nodes_visited": nodes_visited,
        "retrieval_raw": retrieval_result,
        "latency_s": latency,
    }


def _extract_nodes(retrieval_result: dict) -> list[dict]:
    """Pull out the retrieved sections/nodes for display in the UI."""
    nodes = []
    # PageIndex returns retrieved content under various keys depending on version
    for key in ("results", "retrieved_nodes", "nodes", "chunks", "content"):
        if key in retrieval_result and retrieval_result[key]:
            items = retrieval_result[key]
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        nodes.append({
                            "title": item.get("title") or item.get("node_title") or "Section",
                            "text": item.get("content") or item.get("text") or str(item)[:300],
                            "node_id": item.get("node_id") or item.get("id") or "",
                        })
                    else:
                        nodes.append({"title": "Section", "text": str(item)[:300], "node_id": ""})
            break
    # Fallback: show raw keys if nothing matched
    if not nodes:
        nodes.append({
            "title": "Retrieved Context",
            "text": str(retrieval_result)[:500],
            "node_id": "",
        })
    return nodes


def _extract_answer(chat_resp: dict) -> str:
    """Extract answer string from chat_completions response."""
    try:
        return chat_resp["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError):
        return str(chat_resp)
