"""
Build a 100-instance stratified sample of JSONSchemaBench (Geng et al. 2025, arXiv:2501.10868):
25 from Github_easy, 25 Github_medium, 25 Glaiveai2K, 25 JsonSchemaStore.
Deterministic seed; outputs to jsonschema_sample_100.json.
"""
import json
import random
from pathlib import Path
import pyarrow.parquet as pq

BASE = Path(__file__).parent
random.seed(20260518)

SPLITS = ["Github_easy", "Github_medium", "Glaiveai2K", "JsonSchemaStore"]
N_PER_SPLIT = 25

instances = []
for split in SPLITS:
    t = pq.read_table(BASE / f"{split}_test.parquet").to_pandas()
    rows = t.to_dict("records")
    sampled = random.sample(rows, min(N_PER_SPLIT, len(rows)))
    for r in sampled:
        instances.append({
            "instance_id": f"{split}__{r['unique_id']}",
            "split": split,
            "json_schema": r["json_schema"],
        })

print(f"Total instances: {len(instances)}")
with open(BASE / "jsonschema_sample_100.json", "w") as f:
    json.dump(instances, f, indent=2)
print(f"Wrote jsonschema_sample_100.json")

# Print a sample schema for sanity
print("\nSample schema (first instance):")
print(instances[0]["instance_id"])
print(instances[0]["json_schema"][:300])
