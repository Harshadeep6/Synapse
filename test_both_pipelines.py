"""
Step 5 checkpoint: both pipelines answer the same question side-by-side.
Run: python test_both_pipelines.py
"""
import sys
sys.path.insert(0, ".")
from pipelines.traditional_rag import query as trad_query
from pipelines.pageindex_rag import query as pi_query

LABEL = "AAPL_FY2024"
QUESTIONS = [
    "What was Apple's net income for fiscal year 2024?",
    "What are the primary risk factors that could cause Apple's revenue to decline?",
]

print("=== Step 5 Checkpoint: Side-by-Side Comparison ===\n")

for question in QUESTIONS:
    print("=" * 70)
    print(f"Q: {question}")
    print("=" * 70)

    print("\n[Traditional RAG] running...")
    trad = trad_query(LABEL, question)
    print(f"  Answer: {trad['answer']}")
    print(f"  Latency: {trad['latency_s']}s | Chunks retrieved: {len(trad['chunks'])}")

    print("\n[PageIndex RAG] running...")
    pi = pi_query(LABEL, question)
    print(f"  Answer: {pi['answer']}")
    print(f"  Latency: {pi['latency_s']}s | Nodes visited: {len(pi['nodes_visited'])}")
    print()
