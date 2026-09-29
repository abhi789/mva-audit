"""
Analyze BFCL Simple Function Call cohort run, 3-seed pooled.

Per model:
  - valid_correct_fn rate (function call emitted with the correct function name)
  - schema_regurgitation rate (function DEFINITION emitted instead of a call)
  - other failure modes
  - per-seed schema-regurgitation mean ± SD

Cross-walk: Meditron's medical Task B + tool-router placeholder-echo + JSONSchemaBench
schema-regurgitation + BFCL schema-regurgitation = same model-level "regurgitate prompt
content verbatim" signature, surface differs by what's in the prompt's literal template.
"""
import json, statistics
from pathlib import Path
from collections import Counter

BASE = Path(__file__).resolve().parents[1]  # repository root
SEED_DIRS = sorted([p for p in BASE.glob("agent_runs_bfcl_seed*") if p.is_dir()])


def main():
    print(f"Pooling across {len(SEED_DIRS)} seed dirs: {[p.name for p in SEED_DIRS]}\n")
    print(f"{'Model':<35} {'n':<5} {'correct_fn':<12} {'regurg':<8} {'parse_fail':<11} {'wrong_fn':<10} {'placeholder':<12} {'other/err':<10} {'per-seed regurg':<20}")
    print("-" * 130)

    # Collect model names appearing in any seed dir
    model_names = set()
    for sd in SEED_DIRS:
        for md in sd.glob("*/"):
            if md.is_dir() and md.name != "summary.json":
                model_names.add(md.name)

    model_rows = []
    for model_name in sorted(model_names):
        cats = Counter()
        per_seed_regurg = []
        for sd in SEED_DIRS:
            md = sd / model_name
            if not md.is_dir(): continue
            seed_regurg = 0; seed_n = 0
            for f in md.glob("*.json"):
                j = json.load(open(f))
                c = j["category"]
                cats[c] += 1
                seed_n += 1
                if c == "schema_regurgitation":
                    seed_regurg += 1
            if seed_n > 0:
                per_seed_regurg.append(seed_regurg / seed_n)
        n = sum(cats.values())
        if n == 0: continue
        correct = cats.get("valid_correct_fn", 0)
        regurg = cats.get("schema_regurgitation", 0)
        parse_fail = cats.get("json_parse_fail", 0)
        wrong_fn = cats.get("valid_wrong_fn", 0)
        placeholder = cats.get("placeholder_echo", 0)
        other = cats.get("other_invalid", 0) + cats.get("ollama_error", 0) + cats.get("empty", 0)

        if len(per_seed_regurg) > 1:
            seed_str = f"{statistics.mean(per_seed_regurg)*100:.1f}%±{statistics.stdev(per_seed_regurg)*100:.1f}"
        else:
            seed_str = f"{regurg/n*100:.1f}%"

        print(f"{model_name:<35} {n:<5} {correct/n*100:>9.1f}% {regurg/n*100:>6.1f}% {parse_fail/n*100:>9.1f}% {wrong_fn/n*100:>8.1f}% {placeholder/n*100:>10.1f}% {other/n*100:>8.1f}% {seed_str}")
        model_rows.append({"model": model_name, "n": n,
                          "n_seeds": len(per_seed_regurg),
                          "valid_correct_fn_pct": correct/n,
                          "schema_regurgitation_pct": regurg/n,
                          "schema_regurgitation_per_seed": per_seed_regurg,
                          "parse_fail_pct": parse_fail/n,
                          "valid_wrong_fn_pct": wrong_fn/n,
                          "placeholder_echo_pct": placeholder/n,
                          "other_invalid_pct": other/n})

    print()
    print("=" * 80)
    print("4-surface cross-walk for Meditron")
    print("=" * 80)
    print("\nMeditron's failure surface across the 4 evaluation environments:\n")
    print(f"  Medical Task B (ICU):  85% schema-invalid; 63% placeholder-echo of failures (`<one of 6 actions>`)")
    print(f"  Tool-router (NLP):     45% schema-invalid; 100% placeholder-echo of failures (`<one of 6 tool names>`)")
    print(f"  JSONSchemaBench:       ~75-80% schema-regurgitation (emits schema definition)")
    for r in model_rows:
        if "meditron" in r["model"].lower():
            print(f"  BFCL Simple FC:        {r['schema_regurgitation_pct']*100:.1f}% schema-regurgitation (emits function definition)")
    print(f"\n  Underlying behavior: regurgitate prompt content verbatim.")
    print(f"  Surface differs by what the prompt offers as a literal template.")

    with open(BASE / "bfcl_results.json", "w") as f:
        json.dump({"per_model": model_rows, "n_seed_dirs": len(SEED_DIRS)}, f, indent=2)
    print(f"\nSaved bfcl_results.json")


if __name__ == "__main__":
    main()
