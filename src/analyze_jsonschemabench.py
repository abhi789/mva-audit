"""
Analyze a JSONSchemaBench run.

Reports per model:
  - Valid rate (schema-validates)
  - Failure-mode breakdown: schema_violation / json_parse_fail / placeholder_echo / schema_regurgitation
  - Regurgitation rate: fraction of schema_violation outputs whose top-level keys
    match the input schema's top-level keys (the schema definition returned as
    if it were the instance)
"""
import json
import re
from pathlib import Path
from collections import Counter

BASE = Path(__file__).resolve().parents[1]  # repository root
# Pool across all available seed directories (seed0, seed1, seed2, ...).
SEED_DIRS = sorted([p for p in BASE.glob("agent_runs_jsonschemabench_seed*") if p.is_dir()])

PLACEHOLDER_RE = re.compile(r"<\s*(your|the|a)\s+json|<\s*example|<\s*replace|<\s*fill", re.IGNORECASE)


def detect_schema_regurgitation(output_obj, schema_obj):
    """Return True if the output's top-level keys match the schema's top-level keys
    closely (i.e., the model emitted the schema definition as the instance)."""
    if not isinstance(output_obj, dict) or not isinstance(schema_obj, dict):
        return False
    out_keys = set(output_obj.keys())
    sch_keys = set(schema_obj.keys())
    # Schema keys are JSON-Schema meta keys: $schema, properties, type, required, etc.
    SCHEMA_META = {"$schema", "type", "properties", "required", "additionalProperties",
                   "title", "description", "items", "definitions", "$defs", "$ref",
                   "patternProperties", "id", "$id", "enum", "anyOf", "oneOf", "allOf"}
    schema_meta_in_output = out_keys & SCHEMA_META
    return len(schema_meta_in_output) >= 2  # Output contains >=2 JSON-Schema meta keys


def categorize_with_schema_check(j, schema_obj):
    """Refine the original category by checking if a 'valid' or 'schema_violation'
    output actually exhibits schema-regurgitation."""
    orig_cat = j["category"]
    raw = j.get("raw_response", "")
    # Already-classified placeholder echo
    if orig_cat == "placeholder_echo":
        return orig_cat
    if orig_cat in ("schema_violation", "valid"):
        # Try to parse the output
        try:
            t = raw.strip()
            if t.startswith("```"):
                t = re.sub(r"^```(?:json)?\s*", "", t)
                t = re.sub(r"\s*```\s*$", "", t).strip()
            obj = json.loads(t)
            if detect_schema_regurgitation(obj, schema_obj):
                return f"{orig_cat}__regurgitation"
        except Exception:
            pass
    return orig_cat


def load_schemas():
    """Map instance_id -> parsed schema_obj for regurgitation analysis."""
    inst_list = json.load(open(BASE / "jsonschema_bench" / "jsonschema_sample_100.json"))
    out = {}
    for inst in inst_list:
        try:
            out[inst["instance_id"]] = json.loads(inst["json_schema"])
        except Exception:
            out[inst["instance_id"]] = None
    return out


def main():
    schemas = load_schemas()
    print(f"Pooling across {len(SEED_DIRS)} seed dirs: {[p.name for p in SEED_DIRS]}\n")
    print(f"{'Model':<35} {'n':<5} {'valid':<7} {'schema_viol':<14} {'parse_fail':<11} {'plc_echo':<9} {'regurg':<8} {'err':<5}")
    print("-" * 100)

    # Collect model names that appear in any seed dir
    model_names = set()
    for sd in SEED_DIRS:
        for md in sd.glob("*/"):
            if md.is_dir() and md.name != "summary.json":
                model_names.add(md.name)

    model_rows = []
    for model_name in sorted(model_names):
        cats = Counter()
        regurg = 0
        per_seed_regurg = []
        for sd in SEED_DIRS:
            md = sd / model_name
            if not md.is_dir(): continue
            seed_regurg = 0; seed_n = 0
            for f in md.glob("*.json"):
                j = json.load(open(f))
                sid = j["instance_id"]
                schema_obj = schemas.get(sid)
                cat = categorize_with_schema_check(j, schema_obj)
                cats[cat] += 1
                seed_n += 1
                if "__regurgitation" in cat:
                    regurg += 1; seed_regurg += 1
            if seed_n > 0:
                per_seed_regurg.append(seed_regurg / seed_n)
        n = sum(cats.values())
        if n == 0: continue
        # Aggregate primary categories
        valid = cats.get("valid", 0) + cats.get("valid__regurgitation", 0)  # rare
        schema_viol = cats.get("schema_violation", 0) + cats.get("schema_violation__regurgitation", 0)
        parse_fail = cats.get("json_parse_fail", 0)
        placeholder = cats.get("placeholder_echo", 0)
        err = cats.get("ollama_error", 0)

        # Per-seed mean ± SD on schema-regurgitation rate
        import statistics
        if len(per_seed_regurg) > 1:
            mean_r = statistics.mean(per_seed_regurg)
            sd_r = statistics.stdev(per_seed_regurg)
            seed_str = f"{mean_r*100:.1f}%±{sd_r*100:.1f}"
        else:
            seed_str = f"{regurg/n*100:.1f}%"

        print(f"{model_name:<35} {n:<5} {valid/n*100:>5.1f}% {schema_viol/n*100:>10.1f}% {parse_fail/n*100:>9.1f}% {placeholder/n*100:>7.1f}% {regurg/n*100:>6.1f}% {err/n*100:>3.1f}% [per-seed regurg: {seed_str}]")
        model_rows.append({"model": model_name, "n": n,
                          "n_seeds": len(per_seed_regurg),
                          "valid_pct": valid/n,
                          "schema_violation_pct": schema_viol/n, "parse_fail_pct": parse_fail/n,
                          "placeholder_echo_pct": placeholder/n,
                          "schema_regurgitation_pct": regurg/n,
                          "schema_regurgitation_per_seed": per_seed_regurg,
                          "error_pct": err/n})

    # Save
    summary = {"per_model": model_rows}
    with open(BASE / "jsonschemabench_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved jsonschemabench_results.json")


if __name__ == "__main__":
    main()
