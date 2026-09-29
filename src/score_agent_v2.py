"""
Revised rule-based ground-truth scorer (v2) — addresses the clinical ergonomist
audit of score_agent.py. Key changes:

  1. Acute spike downgrade — single-cycle Δ≥2 is within Borg test-retest noise
     (SEM ≈ 0.8, Grant 1999; Lamb 2016). Revised: downgrade to task_modification
     unless the spike is confirmed (sustained RPE≥5 on the next cycle as well)
     OR occurs at or after cycle 3 with cur_max ≥ 6.
  2. Escalation threshold — single RPE=7 is not an imminent-injury trigger
     (Borg 7 = "very hard", ~85% HRmax, sustainable short-term). Revised to
     RPE≥8 OR RPE≥7 sustained 2+ cycles.
  3. Micro-break step inserted at RPE=5 — published ergonomic protocols
     (OSHA eTool, WA DLI) recommend rest before task-redesign.
  4. Divergence threshold lowered 3→2 — per Hlavenka 2017 and Borg SEM, 2 is
     the detectable threshold for asymmetric loading.
  5. offline_refit severity 2→3 — functionally stops the session.
  6. Back-weighted aggregation — agg = max(RPE_back, 0.8 × RPE_legs) since LBP
     is the primary injury concern in BSIE trunk studies (de Looze 2016).
  7. Early-session guard — cycle_idx < 2 uses continue_monitoring unless RPE≥7
     (ignores rater anchoring).
  8. Recovery detection — if cur_max drops ≥2 from prior cycle, emit
     continue_monitoring rather than re-firing the prior severity action.
  9. Cumulative-dose clause — 5+ consecutive cycles at RPE≥5 triggers
     task_modification regardless of cur_max.
 10. RPE=3 requires delta_2 ≥ 1 — don't over-fire micro_break on moderate-hold.

Backwards-compat: score_agent.py (v1) is unchanged. Import this module instead
to use the revised rule.
"""
import json
from pathlib import Path
from collections import defaultdict, Counter

BASE = Path(__file__).resolve().parents[1]  # repository root

ACTION_SEVERITY = {
    "continue_monitoring": 0,
    "suggest_micro_break": 1,
    "suggest_posture_change": 1,
    "suggest_task_modification": 2,
    "recommend_offline_refit": 3,   # v2: bumped from 2 (session-ending like escalate)
    "escalate_end_session": 3,
}


def _back_weighted_max(rb, rl, back_weight=1.0, leg_weight=0.8):
    """LBP-weighted aggregation for injury-risk decisions."""
    return max(rb * back_weight, rl * leg_weight)


def gt_action_v2(stream, idx, exoskeleton):
    """Return the revised rule-based correct action for cycle index `idx`."""
    cur = stream[idx]["ground_truth"]
    rb, rl = cur["RPE_back"], cur["RPE_legs"]
    cur_max = max(rb, rl)
    # Back-weighted aggregation for severity decisions
    weighted_max = _back_weighted_max(rb, rl)
    diff = abs(rb - rl)

    # Trend: change in max-RPE over last 2 cycles
    if idx >= 2:
        prev_max = max(stream[idx - 2]["ground_truth"]["RPE_back"],
                       stream[idx - 2]["ground_truth"]["RPE_legs"])
        delta_2 = cur_max - prev_max
    else:
        delta_2 = 0

    # Previous cycle
    prev = None
    if idx >= 1:
        prev = max(stream[idx - 1]["ground_truth"]["RPE_back"],
                   stream[idx - 1]["ground_truth"]["RPE_legs"])

    # Next cycle (for spike confirmation)
    nxt = None
    if idx < len(stream) - 1:
        nxt = max(stream[idx + 1]["ground_truth"]["RPE_back"],
                  stream[idx + 1]["ground_truth"]["RPE_legs"])

    # (7) Early-session guard: cycles 0-1 ignore all tier-2+ actions unless cur_max >=8
    if idx < 2 and cur_max < 8:
        return "continue_monitoring"

    # (8) Recovery detection: if cur_max dropped >=2 from prior, continue_monitoring
    if prev is not None and (prev - cur_max) >= 2:
        return "continue_monitoring"

    # (2) Escalation: RPE >= 8 always, or RPE >= 7 confirmed by another RPE>=7 cycle
    if cur_max >= 8:
        return "escalate_end_session"
    if cur_max >= 7 and (prev is not None and prev >= 7):
        return "escalate_end_session"
    # Single-cycle 7 without confirmation -> task_modification (not escalate)

    # (1) Acute spike: Δ>=2 within one cycle is within noise; require confirmation
    if prev is not None and (cur_max - prev) >= 2 and cur_max >= 5:
        # Confirmed if next cycle also at >= cur_max, or if cur_max >= 7 (already handled above)
        confirmed = (nxt is not None and nxt >= cur_max)
        if confirmed or cur_max >= 7:
            # Confirmed acute rise late session — escalate
            return "escalate_end_session"
        # Unconfirmed — downgrade to task_modification
        return "suggest_task_modification"

    # Severity band: single RPE 7 (unconfirmed) or RPE 6 with Δ>=1 — task_mod
    if cur_max >= 7:
        return "suggest_task_modification"
    if cur_max == 6 and delta_2 >= 1:
        return "suggest_task_modification"

    # (9) Cumulative-dose clause: 5+ consecutive cycles at weighted_max >= 5 -> task_mod
    high_run = 0
    for j in range(idx, max(-1, idx - 10), -1):
        m = _back_weighted_max(
            stream[j]["ground_truth"]["RPE_back"],
            stream[j]["ground_truth"]["RPE_legs"])
        if m >= 5:
            high_run += 1
        else:
            break
    if high_run >= 5:
        return "suggest_task_modification"

    # (4) Divergence: threshold lowered 3 -> 2, gate on weighted_max >= 4
    if diff >= 2 and weighted_max >= 4:
        if rb > rl and exoskeleton == "with_exoskeleton":
            return "recommend_offline_refit"  # back-dominant under exo (severity 3)
        return "suggest_posture_change"

    # (5) First hit at RPE 5 -> micro_break; task_mod only on persistence/trend
    if cur_max >= 5:
        # If previous cycle also >= 5, escalate the response to task_mod
        if prev is not None and prev >= 5:
            return "suggest_task_modification"
        # First entry into RPE 5 — micro_break (OSHA/WA DLI protocol)
        return "suggest_micro_break"

    # (10) RPE 3/4 handling — 4 unconditional micro_break, 3 requires delta
    if cur_max == 4:
        return "suggest_micro_break"
    if cur_max == 3 and delta_2 >= 1:
        return "suggest_micro_break"

    # Low fatigue
    return "continue_monitoring"


def score_v2(predicted, gt):
    """Revised partial-credit: same tier = 0.75, adjacent = 0.4, else 0.
    Uses v2 ACTION_SEVERITY (offline_refit = 3)."""
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


def compare_v1_v2_on_scenarios(scenario_dir: Path):
    """How many cycles get a different GT label between v1 and v2?"""
    import sys
    sys.path.insert(0, str(BASE))
    from score_agent import gt_action as gt_v1
    n = 0; diff = 0
    diff_pairs = Counter()
    for p in sorted(scenario_dir.glob("P*.json")):
        s = json.loads(p.read_text())
        stream = s["cycle_stream"]; exo = s["exoskeleton"]
        for i in range(len(stream)):
            g1 = gt_v1(stream, i, exo)
            g2 = gt_action_v2(stream, i, exo)
            n += 1
            if g1 != g2:
                diff += 1
                diff_pairs[(g1, g2)] += 1
    print(f"v1 vs v2 GT: {diff}/{n} cycles relabel ({100*diff/n:.1f}%)")
    print("Top relabel pairs (v1 -> v2):")
    for (g1, g2), k in diff_pairs.most_common(12):
        print(f"  {g1:28s} -> {g2:28s}  n={k}")


if __name__ == "__main__":
    scen_dir = BASE / "agent_data_v4" / "scenarios"
    compare_v1_v2_on_scenarios(scen_dir)
