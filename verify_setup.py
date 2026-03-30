"""
Step 1 checkpoint: verify all API clients work.
Run: python verify_setup.py
"""
import os
from dotenv import load_dotenv

load_dotenv()

def check_pageindex():
    from pageindex import PageIndexClient
    api_key = os.getenv("PAGEINDEX_API_KEY")
    if not api_key or api_key == "your_pageindex_api_key_here":
        print("❌ PageIndex: PAGEINDEX_API_KEY not set in .env")
        return False
    client = PageIndexClient(api_key=api_key)
    # List documents to verify auth (lightweight call)
    docs = client.list_documents()
    print(f"✅ PageIndex: connected. Documents in account: {len(docs) if isinstance(docs, list) else 'OK'}")
    return True

def check_groq():
    from groq import Groq
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        print("❌ Groq: GROQ_API_KEY not set in .env")
        return False
    client = Groq(api_key=api_key)
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": "Say 'hello from Groq' and nothing else."}],
        max_tokens=20,
    )
    print(f"✅ Groq: {resp.choices[0].message.content.strip()}")
    return True

def check_chromadb():
    import chromadb
    client = chromadb.Client()
    col = client.create_collection("test")
    col.add(documents=["hello world"], ids=["1"])
    results = col.query(query_texts=["hello"], n_results=1)
    print(f"✅ ChromaDB: in-memory store works. Query result: {results['documents'][0][0]}")
    client.delete_collection("test")
    return True

def check_sentence_transformers():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb = model.encode(["hello world"])
    print(f"✅ SentenceTransformers: model loaded. Embedding dim: {emb.shape[1]}")
    return True

if __name__ == "__main__":
    print("=== EarningsLens — Step 1 Checkpoint ===\n")
    results = []
    results.append(check_chromadb())
    results.append(check_sentence_transformers())
    results.append(check_groq())
    results.append(check_pageindex())
    print(f"\n{'All checks passed!' if all(results) else 'Some checks failed — see above.'}")
