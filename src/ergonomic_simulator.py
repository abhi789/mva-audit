"""
Ergonomic task-cycle simulator — Python port of the "Tool" sheet from
Cumulative Analysis_1.xlsx.

Given proposed task cycle parameters (duration + reps per phase),
returns predicted time-weighted muscle activity per region for:
  - with exoskeleton (E) vs without (NE)
  - symmetric (S) vs asymmetric (A) posture
plus the predicted exoskeleton efficacy (% reduction NE -> E).

Base data: posture-split mean muscle activity per phase, extracted from
rows 53-72 of the "Tool" sheet (which are themselves cohort means across
the 14 participants × multiple trials).
"""
from typing import Dict, Literal

# Base data: MUSCLE_ACTIVITY[muscle][phase][posture][exo_condition]
# Source: Cumulative Analysis_1.xlsx, Tool sheet rows 53-72
# Columns: F=Asym×E, G=Asym×NE, H=Sym×E, I=Sym×NE

_BASE = {
    "LBF": {
        "SS":  {"A": {"E": 0.053, "NE": 0.040}, "S": {"E": 0.043, "NE": 0.045}},
        "B":   {"A": {"E": 0.252, "NE": 0.290}, "S": {"E": 0.211, "NE": 0.239}},
        "SUS": {"A": {"E": 0.204, "NE": 0.224}, "S": {"E": 0.169, "NE": 0.165}},
        "R":   {"A": {"E": 0.368, "NE": 0.426}, "S": {"E": 0.401, "NE": 0.406}},
        "SE":  {"A": {"E": 0.041, "NE": 0.038}, "S": {"E": 0.042, "NE": 0.043}},
    },
    "LES": {
        "SS":  {"A": {"E": 0.047, "NE": 0.066}, "S": {"E": 0.082, "NE": 0.095}},
        "B":   {"A": {"E": 0.362, "NE": 0.437}, "S": {"E": 0.455, "NE": 0.588}},
        "SUS": {"A": {"E": 0.243, "NE": 0.298}, "S": {"E": 0.277, "NE": 0.398}},
        "R":   {"A": {"E": 0.507, "NE": 0.502}, "S": {"E": 0.784, "NE": 0.717}},
        "SE":  {"A": {"E": 0.054, "NE": 0.069}, "S": {"E": 0.100, "NE": 0.110}},
    },
    "RBF": {
        "SS":  {"A": {"E": 0.036, "NE": 0.064}, "S": {"E": 0.044, "NE": 0.068}},
        "B":   {"A": {"E": 0.206, "NE": 0.225}, "S": {"E": 0.191, "NE": 0.209}},
        "SUS": {"A": {"E": 0.147, "NE": 0.175}, "S": {"E": 0.131, "NE": 0.152}},
        "R":   {"A": {"E": 0.326, "NE": 0.322}, "S": {"E": 0.403, "NE": 0.449}},
        "SE":  {"A": {"E": 0.034, "NE": 0.058}, "S": {"E": 0.043, "NE": 0.064}},
    },
    "RES": {
        "SS":  {"A": {"E": 0.058, "NE": 0.074}, "S": {"E": 0.116, "NE": 0.091}},
        "B":   {"A": {"E": 0.476, "NE": 0.513}, "S": {"E": 0.530, "NE": 0.554}},
        "SUS": {"A": {"E": 0.291, "NE": 0.333}, "S": {"E": 0.256, "NE": 0.342}},
        "R":   {"A": {"E": 0.660, "NE": 0.657}, "S": {"E": 0.735, "NE": 0.622}},
        "SE":  {"A": {"E": 0.090, "NE": 0.077}, "S": {"E": 0.121, "NE": 0.114}},
    },
}

# Baseline (as run in the original experiment): SS=15 s, B=~3, SUS=30, R=~3, SE=15, Rest=15
# (Figure 2 of Kuber et al. 2024.) The Excel defaults use a longer variant (75s SUS,
# 50s SE). We treat the experiment values as the reference baseline.
BASELINE_PARAMS = {
    "SS":   {"duration": 15, "reps": 1},
    "B":    {"duration": 3,  "reps": 1},
    "SUS":  {"duration": 30, "reps": 1},
    "R":    {"duration": 3,  "reps": 1},
    "SE":   {"duration": 15, "reps": 1},
    "Rest": {"duration": 15, "reps": 1},
}
PHASES = ["SS", "B", "SUS", "R", "SE", "Rest"]
MUSCLES_BACK = ["LES", "RES"]
MUSCLES_LEGS = ["LBF", "RBF"]


def simulate(params: Dict[str, Dict[str, float]],
             posture: Literal["symmetric", "asymmetric"]) -> Dict[str, float]:
    """
    Compute predicted time-weighted muscle activity for the given parameters.

    params: {phase: {"duration": s, "reps": n}} for phases SS, B, SUS, R, SE, Rest.
    posture: "symmetric" or "asymmetric"
    returns: dict with back_E, back_NE, leg_E, leg_NE (time-weighted mean activity)
             and efficacy_back_pct, efficacy_leg_pct (% reduction from NE to E).
    """
    p = "A" if posture == "asymmetric" else "S"

    total_weight = 0.0
    sum_act = {"back_E": 0.0, "back_NE": 0.0, "leg_E": 0.0, "leg_NE": 0.0}

    for phase in PHASES:
        cfg = params.get(phase, BASELINE_PARAMS[phase])
        w = cfg["duration"] * cfg["reps"]
        total_weight += w
        if phase == "Rest":
            # During rest, muscle activity is ~0 — don't add to numerator,
            # but rest duration still stretches the cycle so it reduces
            # the weighted average (lower load per total cycle time).
            continue
        for m in MUSCLES_BACK:
            sum_act["back_E"]  += _BASE[m][phase][p]["E"]  * w
            sum_act["back_NE"] += _BASE[m][phase][p]["NE"] * w
        for m in MUSCLES_LEGS:
            sum_act["leg_E"]  += _BASE[m][phase][p]["E"]  * w
            sum_act["leg_NE"] += _BASE[m][phase][p]["NE"] * w

    # Divide by total cycle time × 2 muscles per region
    for k in sum_act:
        sum_act[k] = sum_act[k] / (total_weight * 2) if total_weight else 0.0

    def eff_pct(e, ne):
        return 100.0 * (ne - e) / ne if ne > 0 else 0.0

    return {
        **sum_act,
        "efficacy_back_pct": round(eff_pct(sum_act["back_E"], sum_act["back_NE"]), 1),
        "efficacy_leg_pct":  round(eff_pct(sum_act["leg_E"], sum_act["leg_NE"]), 1),
        "cycle_time_seconds": total_weight,
    }


def compare_to_baseline(modified_params: Dict[str, Dict[str, float]],
                        posture: str) -> Dict[str, float]:
    """Return predicted change vs baseline."""
    base = simulate(BASELINE_PARAMS, posture)
    mod = simulate(modified_params, posture)
    return {
        "baseline": {k: round(v, 3) if isinstance(v, float) else v for k, v in base.items()},
        "modified": {k: round(v, 3) if isinstance(v, float) else v for k, v in mod.items()},
        "back_activity_change_pct": round(100 * (mod["back_E"] - base["back_E"]) / base["back_E"], 1),
        "leg_activity_change_pct":  round(100 * (mod["leg_E"]  - base["leg_E"])  / base["leg_E"],  1),
    }


if __name__ == "__main__":
    import json
    print("Baseline (experiment default: SS=15, B=3, SUS=30, R=3, SE=15, Rest=15):")
    for posture in ["symmetric", "asymmetric"]:
        print(f"\n  {posture}:")
        print("  " + json.dumps(simulate(BASELINE_PARAMS, posture), indent=2).replace("\n", "\n  "))

    print("\n\n--- Example: reduce sustained-bend from 30s to 15s ---")
    modified = dict(BASELINE_PARAMS)
    modified["SUS"] = {"duration": 15, "reps": 1}
    r = compare_to_baseline(modified, "symmetric")
    print(json.dumps(r, indent=2))

    print("\n--- Example: add more rest (30s vs 15s), keep rest at 15s ---")
    modified = dict(BASELINE_PARAMS)
    modified["Rest"] = {"duration": 30, "reps": 1}
    r = compare_to_baseline(modified, "asymmetric")
    print(json.dumps(r, indent=2))
