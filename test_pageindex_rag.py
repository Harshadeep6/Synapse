"""
Step 4 checkpoint: verify PageIndex pipeline works.
Run: python test_pageindex_rag.py
"""
import sys
sys.path.insert(0, ".")
from pipelines.pageindex_rag import get_doc_id, get_tree, query
import json

print("=== Step 4 Checkpoint: PageIndex RAG ===\n")
print("Submitting AAPL_FY2024 to PageIndex (cached after first run)...")

doc_id = get_doc_id("AAPL_FY2024")
print(f"doc_id: {doc_id}\n")

top_nodes = get_tree("AAPL_FY2024")
print(f"Tree top-level nodes ({len(top_nodes)} found):")
for node in top_nodes[:8]:
    title = node.get("title") or node.get("node_title") or str(node)[:60]
    nid = node.get("node_id", "")
    print(f"  [{nid}] {title}")

print("\nRunning query: 'What was Apple's net income for fiscal year 2024?'")
result = query("AAPL_FY2024", "What was Apple's net income for fiscal year 2024?")

print(f"\nAnswer:\n{result['answer']}")
print(f"\nLatency: {result['latency_s']}s")
print(f"\nNodes visited ({len(result['nodes_visited'])}):")
for node in result["nodes_visited"]:
    preview = node["text"][:120].replace("\n", " ")
    print(f"  [{node['node_id']}] {node['title']}: {preview}...")
