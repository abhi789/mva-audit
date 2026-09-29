"""
ICU model-vs-baseline comparisons.

  - Baselines:
      * majority-class: always continue_monitoring
      * NEWS2 rule: NEWS2 >= 5 -> escalate_to_attending, else continue_monitoring.
        The NEWS2 score is each scenario's vital_snapshot.news2_score
        (computed at extraction time).
  - For each model, score every window the model evaluated.
  - McNemar exact paired test on (model_correct, baseline_correct) pairs,
    with Holm-Bonferroni correction over the model x baseline family.

Output: vs_baselines_icu_results.json + stdout summary.
"""
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
import stats_tools as st

BASE = Path(__file__).resolve().parents[1]  # repository root
SEEDS = [0, 1, 2]
SCENARIO_DIR = BASE / "icu_data_v1" / "scenarios"

MODELS = [
    ("gp",      "gemma3n_e2b"),
    ("medical", "medgemma1.5_4b-it-q8_0"),
    ("medical", "adrienbrault__biomistral-7b_Q4_K_M"),
    ("medical", "meditron_7b-q4_K_M"),
    ("base",    "mistral_7b-instruct-v0.2-q4_K_M"),
    ("base",    "llama2_7b-chat-q4_K_M"),
]

ICU_ACTIONS = [
    "continue_monitoring", "recheck_vitals", "adjust_positioning_or_oxygen",
    "fluid_or_vasopressor_adjustment", "escalate_to_attending", "transfer_higher_acuity",
]


def load_scenarios():
    out = {}
    for p in sorted(SCENARIO_DIR.glob("*.json")):
        with open(p) as f:
            out[p.stem] = json.load(f)
    return out


def baseline_majority(scenario_window):
    """Always predict continue_monitoring (the modal GT class)."""
    return "continue_monitoring"


def baseline_classifier_rule(scenario_window):
    """NEWS2 binary classifier-rule. Mirrors gt_icu_v1 R2c with no other
    inputs (i.e., what you would do given ONLY the upstream signal that
    the LLM agent also has access to)."""
    news2 = scenario_window["vital_snapshot"].get("news2_score", 0)
    if news2 >= 5:
        return "escalate_to_attending"
    return "continue_monitoring"


def collect_model_decisions(model_subdir, scenarios, seed):
    """For seed `seed`, walk agent_runs_icu_active_sim_seed<seed>/<model>/
    and return list of (key, model_action, gt_action, window, sid)."""
    run_dir = BASE / f"agent_runs_icu_active_sim_seed{seed}" / model_subdir
    out = []
    if not run_dir.exists():
        return out
    for jp in sorted(run_dir.glob("*.json")):
        try:
            with open(jp) as f:
                d = json.load(f)
        except Exception:
            continue
        sid = d.get("scenario_id", jp.stem)
        scn = scenarios.get(sid)
        if scn is None:
            continue
        windows = scn["window_stream"]
        wnum_to_idx = {w["window_number"]: i for i, w in enumerate(windows)}
        for w in d.get("window_trace", []):
            wn = w.get("window_number")
            if wn is None or wn not in wnum_to_idx:
                continue
            idx = wnum_to_idx[wn]
            sw = windows[idx]
            gt = sw["ground_truth"]["action"]
            out.append({
                "key": (sid, wn, seed),
                "model_action": w.get("agent_action"),
                "gt": gt,
                "window": sw,
            })
    return out


def score_baseline_on(decisions, baseline_fn):
    return [1 if baseline_fn(d["window"]) == d["gt"] else 0 for d in decisions]


def score_model_on(decisions):
    return [1 if d["model_action"] == d["gt"] else 0 for d in decisions]


def main():
    scenarios = load_scenarios()
    n_scenarios = len(scenarios)
    print(f"Loaded {n_scenarios} scenarios from {SCENARIO_DIR}")
    print(f"Seeds: {SEEDS}")
    print()

    # Per-model decisions, pooled across seeds
    per_model = {}
    for label, model in MODELS:
        all_decisions = []
        for seed in SEEDS:
            all_decisions.extend(collect_model_decisions(model, scenarios, seed))
        per_model[model] = {"label": label, "decisions": all_decisions}
        print(f"  {model:<42s} n_decisions={len(all_decisions):,}")
    print()

    # Compute accuracies + McNemar tests
    rows = []
    all_pvals = []  # (test_id, pval) for Holm
    for model, store in per_model.items():
        decs = store["decisions"]
        n = len(decs)
        if n == 0:
            continue
        model_correct = score_model_on(decs)
        n_model_correct = sum(model_correct)
        acc_model = n_model_correct / n
        ci_model = st.wilson_ci(n_model_correct, n)

        maj_correct = score_baseline_on(decs, baseline_majority)
        rule_correct = score_baseline_on(decs, baseline_classifier_rule)
        acc_maj  = sum(maj_correct) / n
        acc_rule = sum(rule_correct) / n

        mcn_vs_maj  = st.mcnemar_paired(model_correct, maj_correct)
        mcn_vs_rule = st.mcnemar_paired(model_correct, rule_correct)

        all_pvals.append((f"{model}_vs_majority", mcn_vs_maj["p"]))
        all_pvals.append((f"{model}_vs_classifier_rule", mcn_vs_rule["p"]))

        rows.append({
            "model": model,
            "label": store["label"],
            "n": n,
            "acc_model": acc_model,
            "wilson_95_model": ci_model,
            "acc_majority": acc_maj,
            "acc_classifier_rule": acc_rule,
            "mcnemar_vs_majority": mcn_vs_maj,
            "mcnemar_vs_classifier_rule": mcn_vs_rule,
        })

    # Holm correction over the 12-test family
    holm = st.holm_correction(all_pvals)
    holm_lookup = {h["label"]: h for h in holm}

    # Print summary
    print(f"{'Model':<42s}{'n':>6s}{'acc':>8s}{'maj':>8s}{'rule':>8s}{'p_vs_maj':>14s}{'p_vs_rule':>14s}")
    print("-" * 100)
    for r in rows:
        m = r["model"]
        p_maj  = holm_lookup[f"{m}_vs_majority"]
        p_rule = holm_lookup[f"{m}_vs_classifier_rule"]
        marker_m = "*" if p_maj["significant"] else " "
        marker_r = "*" if p_rule["significant"] else " "
        print(f"{m:<42s}{r['n']:>6d}{r['acc_model']:>8.3f}"
              f"{r['acc_majority']:>8.3f}{r['acc_classifier_rule']:>8.3f}"
              f"  p={p_maj['p']:.1e}{marker_m}"
              f"  p={p_rule['p']:.1e}{marker_r}")
    print()
    print("Holm-corrected (alpha=0.05). * = reject null after correction (family of 12 tests).")
    print()
    for h in holm:
        sig = "REJECT" if h["significant"] else "  ns  "
        print(f"  {h['label']:<60s} p={h['p']:.2e}  "
              f"thresh={h['threshold']:.2e}  rank={h['rank']:>2d}  {sig}")

    # Save
    out = {
        "generated_at": datetime.now().isoformat(),
        "n_scenarios": n_scenarios,
        "seeds": SEEDS,
        "rows": rows,
        "holm": holm,
    }
    with open(BASE / "vs_baselines_icu_results.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved vs_baselines_icu_results.json")


if __name__ == "__main__":
    main()
