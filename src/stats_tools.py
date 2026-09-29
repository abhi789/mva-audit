"""
Statistical infrastructure for agent-evaluation reporting.

Provides:
  - wilson_ci(k, n): exact 95% Wilson score interval for a binomial proportion
  - bootstrap_ci(values, n_iter): paired bootstrap CI for a mean
  - mcnemar_paired(a_correct, b_correct): paired-model comparison test
  - holm_correction(pvals): Holm-Bonferroni FWER correction
  - score_pure_rule(scenarios): what a trivial Python if-else baseline scores
  - score_majority_class(scenarios): always predict continue_monitoring
  - score_random(scenarios, seed): uniform over the 6 actions
"""
import json
import math
import sys
import random
from pathlib import Path
from typing import List, Dict, Sequence, Tuple
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))
import score_agent

BASE = Path(__file__).resolve().parents[1]  # repository root
ACTIONS = [
    "continue_monitoring", "suggest_micro_break", "suggest_posture_change",
    "suggest_task_modification", "escalate_end_session", "recommend_offline_refit",
]


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score 95% CI for k successes out of n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def bootstrap_ci(values: Sequence[float], n_iter: int = 10_000,
                 alpha: float = 0.05, seed: int = 42) -> Tuple[float, float]:
    """Percentile bootstrap CI for the mean of `values`."""
    rng = random.Random(seed)
    vals = list(values)
    n = len(vals)
    if n == 0:
        return (0.0, 0.0)
    means = []
    for _ in range(n_iter):
        s = [vals[rng.randrange(n)] for _ in range(n)]
        means.append(sum(s) / n)
    means.sort()
    lo_i = int(alpha / 2 * n_iter)
    hi_i = int((1 - alpha / 2) * n_iter)
    return (means[lo_i], means[hi_i])


def mcnemar_paired(a_correct: Sequence[int], b_correct: Sequence[int]) -> Dict:
    """McNemar's test for paired binary outcomes (same cycles)."""
    n10 = sum(1 for a, b in zip(a_correct, b_correct) if a and not b)
    n01 = sum(1 for a, b in zip(a_correct, b_correct) if b and not a)
    # Continuity correction
    if n10 + n01 == 0:
        return dict(chi2=0.0, p=1.0, n10=0, n01=0)
    chi2 = (abs(n10 - n01) - 1) ** 2 / (n10 + n01)
    # Chi2(df=1) tail via survival: 1 - Phi(sqrt(chi2))*2+1 ≈ erfc(sqrt(chi2/2))
    p = math.erfc(math.sqrt(chi2 / 2))
    return dict(chi2=chi2, p=p, n10=n10, n01=n01)


def holm_correction(pvals: List[Tuple[str, float]], alpha: float = 0.05) -> List[Dict]:
    """Holm-Bonferroni correction. Input: list of (label, p). Output: annotated list."""
    sorted_p = sorted(pvals, key=lambda x: x[1])
    m = len(sorted_p)
    results = []
    stop = False
    for rank, (label, p) in enumerate(sorted_p):
        threshold = alpha / (m - rank)
        significant = (not stop) and (p <= threshold)
        if not significant:
            stop = True
        results.append(dict(label=label, p=p, threshold=threshold,
                            rank=rank + 1, significant=significant))
    # Reorder to original order
    order = {label: i for i, (label, _) in enumerate(pvals)}
    results.sort(key=lambda x: order[x["label"]])
    return results


# ---------- Naive baselines ----------

def load_scenario(scenario_dir: Path, sid: str) -> Dict:
    return json.loads((scenario_dir / f"{sid}.json").read_text())


def score_dataset(run_gt_func, scenario_ids: List[str],
                  scenario_dir: Path) -> Dict:
    """Run `run_gt_func(stream, idx, exo) -> action` and score against gt_action."""
    n = 0; exact = 0; partial_sum = 0.0
    correct_vec = []
    conf_matrix = Counter()
    for sid in scenario_ids:
        scen = load_scenario(scenario_dir, sid)
        stream = scen["cycle_stream"]; exo = scen["exoskeleton"]
        for idx in range(len(stream)):
            gt = score_agent.gt_action(stream, idx, exo)
            pred = run_gt_func(stream, idx, exo)
            pts, kind = score_agent.score(pred, gt)
            exact += (1 if kind == "exact" else 0)
            partial_sum += pts
            correct_vec.append(1 if kind == "exact" else 0)
            conf_matrix[(gt, pred)] += 1
            n += 1
    lo, hi = wilson_ci(exact, n)
    return dict(
        n=n, exact=exact, exact_accuracy=exact / n if n else 0,
        partial_credit_accuracy=partial_sum / n if n else 0,
        ci95=(lo, hi), correct_vector=correct_vec,
        confusion=dict(("%s->%s" % k, v) for k, v in conf_matrix.most_common(15)),
    )


def baseline_majority(stream, idx, exo):
    return "continue_monitoring"


def baseline_random(seed=42):
    rng = random.Random(seed)
    def f(stream, idx, exo):
        return rng.choice(ACTIONS)
    return f


def baseline_pure_rule(stream, idx, exo):
    """Perfect score by construction: uses the GT rule as the predictor."""
    return score_agent.gt_action(stream, idx, exo)


def baseline_classifier_only(stream, idx, exo):
    """Simple rule based on classifier output alone (no RPE access).
    Approximates what a non-LLM rule engine reading fatigue_monitor would do.
    Handles both v4 data (has tertiary) and v5 data (binary only).
    """
    fp = stream[idx]["fatigue_prediction"]
    bb = fp["back_binary"]["predicted_level"]
    lb = fp["leg_binary"]["predicted_level"]
    bb_conf = fp["back_binary"]["confidence"]
    lb_conf = fp["leg_binary"]["confidence"]

    # If tertiary fields exist (v4), use them for escalation
    if "back_tertiary" in fp:
        bt = fp["back_tertiary"]["predicted_level"]
        lt = fp["leg_tertiary"]["predicted_level"]
        bt_conf = fp["back_tertiary"]["confidence"]
        lt_conf = fp["leg_tertiary"]["confidence"]
        if (bt == "H" and bt_conf >= 0.70) or (lt == "H" and lt_conf >= 0.70):
            return "escalate_end_session"
        if bt == "M" or lt == "M" or bb == "M" or lb == "M":
            return "suggest_micro_break"
        return "continue_monitoring"

    # v5 binary-only fallback: M with high confidence on both regions -> escalate
    if (bb == "M" and bb_conf >= 0.85 and lb == "M" and lb_conf >= 0.85):
        return "escalate_end_session"
    if bb == "M" or lb == "M":
        return "suggest_micro_break"
    return "continue_monitoring"


if __name__ == "__main__":
    # Usage: python3 stats_tools.py <scenario_dir>
    scen_dir = Path(sys.argv[1] if len(sys.argv) > 1
                    else "agent_data_v3/scenarios")
    scenario_ids = [
        "P03_symmetric_with_exoskeleton",
        "P04_symmetric_with_exoskeleton",
        "P05_asymmetric_without_exoskeleton",
        "P09_symmetric_with_exoskeleton",
        "P14_asymmetric_without_exoskeleton",
    ]
    print(f"Evaluating baselines on {scen_dir} × {len(scenario_ids)} scenarios\n")
    for name, fn in [
        ("majority-class (always continue)", baseline_majority),
        ("random-uniform (seed=42)",         baseline_random(42)),
        ("pure Python if-else (GT rule)",    baseline_pure_rule),
        ("classifier-only rule",             baseline_classifier_only),
    ]:
        r = score_dataset(fn, scenario_ids, scen_dir)
        print(f"  {name}")
        print(f"    n={r['n']}  exact={r['exact_accuracy']:.3f}  "
              f"partial={r['partial_credit_accuracy']:.3f}  "
              f"95% CI=[{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]")
        print(f"    top confusion: {list(r['confusion'].items())[:3]}\n")
