"""
Step 6 checkpoint: load question bank and verify structure.
Run: python test_questions.py
"""
import json
from pathlib import Path

questions = json.loads(Path("data/questions.json").read_text(encoding="utf-8"))

print("=== Step 6 Checkpoint: Question Bank ===\n")
print(f"Total questions: {len(questions)}\n")

by_type = {}
by_difficulty = {}
for q in questions:
    by_type[q["type"]] = by_type.get(q["type"], 0) + 1
    by_difficulty[q["difficulty"]] = by_difficulty.get(q["difficulty"], 0) + 1

print("By type:")
for t, n in by_type.items():
    print(f"  {t}: {n}")

print("\nBy difficulty:")
for d, n in by_difficulty.items():
    print(f"  {d}: {n}")

print("\nAll questions:")
for q in questions:
    print(f"  [{q['id']}] ({q['difficulty']}/{q['type']}) {q['question'][:80]}...")
    print(f"       Truth: {q['ground_truth'][:80]}")
    print()
