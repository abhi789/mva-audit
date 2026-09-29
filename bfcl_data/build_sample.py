"""
Build a 100-instance subset of BFCL v3 Simple Function Call.
First 100 cases (deterministic ordering) — matches JSONSchemaBench n=100.
"""
import json
from pathlib import Path

BASE = Path(__file__).parent

# Load JSONL-style data (BFCL files are one JSON per line)
with open(BASE / "BFCL_v3_simple.json") as f:
    instances = [json.loads(line) for line in f if line.strip()]
with open(BASE / "BFCL_v3_simple_answers.json") as f:
    answers = {json.loads(line)["id"]: json.loads(line)["ground_truth"] for line in f if line.strip()}

print(f"Total simple instances: {len(instances)}; answer keys: {len(answers)}")

# Sample first 100 (deterministic, no random)
N = 100
selected = instances[:N]

out = []
for inst in selected:
    sid = inst["id"]
    out.append({
        "instance_id": sid,
        "question": inst["question"][0][0]["content"],  # user message in 1st turn
        "function": inst["function"][0],  # single function for simple FC
        "ground_truth": answers.get(sid, []),
    })

with open(BASE / "bfcl_sample_100.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"Wrote bfcl_sample_100.json")

# Quick sanity check: function names + question previews
print("\nFirst 3 instances:")
for inst in out[:3]:
    fn = inst["function"]["name"]
    print(f"  {inst['instance_id']}: q={inst['question'][:80]!r}  fn={fn}")
