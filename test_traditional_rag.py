"""
Step 3 checkpoint: verify traditional RAG pipeline works.
Run: python test_traditional_rag.py
"""
import sys
sys.path.insert(0, ".")
from pipelines.traditional_rag import query

print("=== Step 3 Checkpoint: Traditional RAG ===\n")
print("Building index for AAPL_FY2024 (embedding ~400 chunks, ~20s first run)...")

result = query("AAPL_FY2024", "What was Apple's net income for fiscal year 2024?")

print(f"\nAnswer:\n{result['answer']}")
print(f"\nLatency: {result['latency_s']}s")
print(f"\nTop {len(result['chunks'])} retrieved chunks:")
for i, chunk in enumerate(result["chunks"], 1):
    preview = chunk["text"][:120].replace("\n", " ")
    print(f"  [{i}] similarity={chunk['similarity']} | {preview}...")
