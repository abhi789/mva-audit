"""
Rule-based ground-truth scorer for wearable agent traces (version 1).

The reference action per cycle is derived from the RPE trajectory and session
context using thresholds grounded in published ergonomics (Borg CR-10 semantics,
the Kuber et al. 2024 medium-fatigue intervention window, NIOSH-LE / ACGIH TLV
proxies, and a passive-exoskeleton feasibility constraint). The evaluated
reference labels use score_agent_v2.gt_action_v2; this module is retained
because stats_tools.py imports it.

Scoring: per-cycle exact-match accuracy + a near-miss partial-credit metric.
"""
import json
from pathlib import Path
from collections import defaultdict, Counter

BASE = Path(__file__).resolve().parents[1]  # repository root
V3_DIR = BASE / "agent_data_v3"
RUN_DIR = BASE / "agent_runs"
OUT = RUN_DIR / "scoring.json"

ACTION_SEVERITY = {
    "continue_monitoring": 0,
    "suggest_micro_break": 1,
    "suggest_posture_change": 1,
    "suggest_task_modification": 2,
    "recommend_offline_refit": 2,
    "escalate_end_session": 3,
}


def gt_action(stream, idx, exoskeleton):
    """Return the rule-based correct action for cycle index `idx`."""
    cur = stream[idx]["ground_truth"]
    rb, rl = cur["RPE_back"], cur["RPE_legs"]
    cur_max = max(rb, rl)
    diff = abs(rb - rl)

    # Trend: change in max-RPE over last 2 cycles
    if idx >= 2:
        prev_max = max(stream[idx - 2]["ground_truth"]["RPE_back"],
                       stream[idx - 2]["ground_truth"]["RPE_legs"])
        delta_2 = cur_max - prev_max
    else:
        delta_2 = 0

    # Acute spike: jumped >=2 in one cycle into >=5
    if idx >= 1:
        prev = max(stream[idx - 1]["ground_truth"]["RPE_back"],
                   stream[idx - 1]["ground_truth"]["RPE_legs"])
        if cur_max - prev >= 2 and cur_max >= 5:
            return "escalate_end_session"

    # Hard severity gates
    if cur_max >= 7:
        return "escalate_end_session"
    if cur_max == 6 and delta_2 >= 1:
        return "suggest_task_modification"

    # Posture / device-driven actions on sustained divergence
    if diff >= 3 and cur_max >= 4:
        if rb > rl and exoskeleton == "with_exoskeleton":
            return "recommend_offline_refit"  # back-dominant under exo
        return "suggest_posture_change"

    # Medium fatigue
    if cur_max >= 5:
        return "suggest_task_modification"
    if cur_max in (3, 4) and delta_2 >= 1:
        return "suggest_micro_break"
    if cur_max in (3, 4):
        return "suggest_micro_break"  # sustained medium → break

    # Low fatigue
    return "continue_monitoring"


def score(predicted, gt):
    if predicted == gt:
        return 1.0, "exact"
    if predicted is None:
        return 0.0, "missing"
    sev_p = ACTION_SEVERITY.get(predicted, -1)
    sev_g = ACTION_SEVERITY[gt]
    if sev_p == sev_g:
        return 0.75, "same_severity"
    if abs(sev_p - sev_g) == 1:
        return 0.4, "adjacent_severity"
    return 0.0, "wrong"


def main():
    if not RUN_DIR.exists():
        print("agent_runs/ not found — run run_agent.py first.")
        return

    # Load all v3 scenarios (full streams)
    scenarios = {}
    for p in (V3_DIR / "scenarios").glob("*.json"):
        with open(p) as f:
            scenarios[p.stem] = json.load(f)

    rows_by_model = defaultdict(list)
    confusion = defaultdict(Counter)  # by_model -> Counter((gt, pred))

    for model_dir in sorted(RUN_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        model = model_dir.name.replace("_", ":")
        for trace_file in sorted(model_dir.glob("*.json")):
            with open(trace_file) as f:
                run = json.load(f)
            sid = run["scenario_id"]
            scenario = scenarios[sid]
            stream = scenario["cycle_stream"]
            exo = scenario["exoskeleton"]

            scored = []
            for entry in run["cycle_trace"]:
                if "_note" in entry:
                    continue
                idx = entry["cycle_number"] - 1
                if idx >= len(stream):
                    continue
                gt = gt_action(stream, idx, exo)
                pred = entry.get("agent_action")
                pts, kind = score(pred, gt)
                scored.append(dict(
                    cycle=entry["cycle_number"],
                    rpe_back=entry["rpe_back"],
                    rpe_legs=entry["rpe_legs"],
                    ground_truth_action=gt,
                    agent_action=pred,
                    score=pts,
                    match_kind=kind,
                ))
                confusion[model][(gt, pred or "PARSE_ERROR")] += 1

            n = len(scored)
            exact = sum(1 for s in scored if s["match_kind"] == "exact")
            partial = sum(s["score"] for s in scored)
            rows_by_model[model].append(dict(
                scenario_id=sid,
                n_cycles_scored=n,
                exact_matches=exact,
                exact_accuracy=round(exact / n, 3) if n else 0,
                partial_credit_accuracy=round(partial / n, 3) if n else 0,
                mean_latency_seconds=run.get("mean_latency_seconds"),
                cycle_scores=scored,
            ))

    summary = {}
    for model, rows in rows_by_model.items():
        all_scored = [c for r in rows for c in r["cycle_scores"]]
        n = len(all_scored)
        if not n:
            continue
        ex = sum(1 for c in all_scored if c["match_kind"] == "exact")
        pa = sum(c["score"] for c in all_scored)
        kinds = Counter(c["match_kind"] for c in all_scored)
        summary[model] = dict(
            total_cycles_scored=n,
            exact_accuracy=round(ex / n, 3),
            partial_credit_accuracy=round(pa / n, 3),
            match_kind_breakdown=dict(kinds),
            mean_latency=round(
                sum(r["mean_latency_seconds"] or 0 for r in rows) / len(rows), 2),
            per_scenario=[{k: v for k, v in r.items() if k != "cycle_scores"}
                          for r in rows],
        )

    out = dict(
        scoring_method="rule_based_from_RPE_and_context",
        action_severity=ACTION_SEVERITY,
        per_model=summary,
    )
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)

    # --- Report ---
    print(f"\nWrote {OUT.relative_to(BASE)}\n")
    print("=== Per-model overall ===")
    for model, s in summary.items():
        print(f"\n{model}")
        print(f"  cycles scored      : {s['total_cycles_scored']}")
        print(f"  exact accuracy     : {s['exact_accuracy']}")
        print(f"  partial-credit acc : {s['partial_credit_accuracy']}")
        print(f"  mean latency       : {s['mean_latency']}s")
        print(f"  match kinds        : {s['match_kind_breakdown']}")

    print("\n=== Per-scenario exact accuracy ===")
    print(f"{'scenario':45s}  " + "  ".join(f"{m:14s}" for m in summary))
    by_scenario = defaultdict(dict)
    for model, s in summary.items():
        for r in s["per_scenario"]:
            by_scenario[r["scenario_id"]][model] = r["exact_accuracy"]
    for sid, m_acc in sorted(by_scenario.items()):
        cells = "  ".join(f"{m_acc.get(m, '-'):>14}" for m in summary)
        print(f"{sid:45s}  {cells}")

    print("\n=== Confusion (top mismatches per model) ===")
    for model, conf in confusion.items():
        print(f"\n{model}")
        wrong = [(k, v) for k, v in conf.items() if k[0] != k[1]]
        wrong.sort(key=lambda x: -x[1])
        for (gt, pred), n in wrong[:8]:
            print(f"  GT={gt:30s} PRED={pred:30s} n={n}")


if __name__ == "__main__":
    main()
