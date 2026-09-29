"""Minimum Viable Audit: rescore saved runs and compute the paper's measures.

For each evaluation group and model this computes
  validity   V/N and invalid-response categories,
  k          number of distinct valid actions,
  H          Shannon entropy of the valid-action distribution (bits),
  d          dominance, the share of valid actions taken by the modal action,
  recall     per reference class,
  M          concentration criterion count over
             (k <= 2, H <= 1, d >= 0.80, zero recall on a supported class);
             M >= 2 flags concentration (paper Section 2).

No inference is run. Run directories are the ones written by the runners in
src/; missing directories are skipped. Usage:

    python src/mva_audit.py                 # audit runs under the repository root
    MVA_ROOT=/path/to/runs python src/mva_audit.py

Writes results/audited_results_local.json and results/run_manifest_local.json.
Per-record rows (which contain scenario identifiers) go to
runs_index/record_index.json, which is git-ignored.
"""
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("MVA_ROOT", REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import score_agent_v2  # noqa: E402
from analyze_jsonschemabench import categorize_with_schema_check, load_schemas  # noqa: E402

ICU = ["continue_monitoring", "recheck_vitals", "adjust_positioning_or_oxygen",
       "fluid_or_vasopressor_adjustment", "escalate_to_attending", "transfer_higher_acuity"]
WEAR = list(score_agent_v2.ACTION_SEVERITY)
TOOLS = ["calculator", "web_search", "calendar", "email", "file_search", "code_execution"]


def mva_metrics(rows, actions):
    """MVA measures for one condition. rows: dicts with 'pred', 'gt', 'category'."""
    n = len(rows)
    counts = Counter(r["pred"] for r in rows if r["pred"] in actions)
    n_valid = sum(counts.values())
    gt = Counter(r["gt"] for r in rows)
    hit = Counter(r["gt"] for r in rows if r["pred"] == r["gt"])
    H = -sum(v / n_valid * math.log2(v / n_valid) for v in counts.values()) if n_valid else None
    dom = max(counts.values()) / n_valid if n_valid else None
    recall = {a: hit[a] / gt[a] if gt[a] else None for a in actions}
    criteria = [
        len(counts) <= 2 if n_valid else None,
        H <= 1 if H is not None else None,
        dom >= 0.8 if dom is not None else None,
        any(v == 0 for v in recall.values() if v is not None),
    ]
    return dict(
        n=n, correct=sum(hit.values()), accuracy=sum(hit.values()) / n,
        n_valid=n_valid, invalid=n - n_valid, invalid_pct=100 * (n - n_valid) / n,
        counts=dict(counts), gt_counts=dict(gt), recall=recall,
        k=len(counts), H=H, dom=dom,
        M=sum(x is True for x in criteria) if n_valid else None, criteria=criteria,
        categories=dict(Counter(r["category"] for r in rows)),
    )


class Auditor:
    def __init__(self):
        self.rows, self.results, self.manifest = {}, {}, []
        self.scenario_cache = {}
        self.schemas = None

    def _read(self, path):
        return json.loads(path.read_bytes())

    def _wearable_gt(self, sid, cycle_number, scenario_dir):
        sp = ROOT / scenario_dir / (sid + ".json")
        if sp not in self.scenario_cache:
            self.scenario_cache[sp] = self._read(sp)
        sc = self.scenario_cache[sp]
        stream = sc["cycle_stream"]
        idx = next(k for k, c in enumerate(stream) if c["cycle_number"] == cycle_number)
        return score_agent_v2.gt_action_v2(stream, idx, sc["exoskeleton"])

    def _records(self, j, f, kind, scenario_dir):
        raw = lambda x: hashlib.sha256(str(x.get("raw_response", "")).encode()).hexdigest()
        if kind in ("icu", "wear"):
            out = []
            trace = j.get("window_trace" if kind == "icu" else "cycle_trace", [])
            for i, w in enumerate(trace):
                if "agent_action" not in w:
                    continue
                sid = j.get("scenario_id", f.stem)
                gt = (self._wearable_gt(sid, w["cycle_number"], scenario_dir)
                      if kind == "wear" else w["gt_action_v1"])
                out.append(dict(sid=sid, unit=w.get("cycle_number", i), pred=w["agent_action"],
                                gt=gt, category=w.get("failure_category", "unknown"),
                                raw_sha256=raw(w)))
            return out
        if kind == "router" and "agent_tool" in j:
            return [dict(sid=j["scenario_id"], unit=0, pred=j["agent_tool"], gt=j["gt_tool"],
                         category=j.get("failure_category", "unknown"), raw_sha256=raw(j))]
        if kind in ("bfcl", "jsb") and "category" in j:
            if kind == "jsb":
                if self.schemas is None:
                    self.schemas = load_schemas()
                cat = categorize_with_schema_check(j, self.schemas.get(j.get("instance_id")))
            else:
                cat = j["category"]
            return [dict(sid=j.get("instance_id", j.get("scenario_id", f.stem)), category=cat,
                         original_category=j["category"], raw_sha256=raw(j))]
        return []

    def group(self, name, paths, kind, scenario_dir="agent_data_v5/scenarios"):
        by_model, files = defaultdict(list), defaultdict(int)
        for path in paths:
            d = ROOT / path
            if not d.exists():
                continue
            for model_dir in sorted(p for p in d.iterdir() if p.is_dir()):
                for f in sorted(model_dir.glob("*.json")):
                    j = self._read(f)
                    if not isinstance(j, dict):
                        continue
                    recs = self._records(j, f, kind, scenario_dir)
                    for r in recs:
                        r.update(source=str(f.relative_to(ROOT)), seed_directory=path)
                    if recs:
                        by_model[model_dir.name] += recs
                        files[model_dir.name] += 1
        for model, rows in by_model.items():
            key = f"{name}/{model}"
            self.rows[key] = rows
            if kind in ("icu", "wear", "router"):
                actions = {"icu": ICU, "wear": WEAR, "router": TOOLS}[kind]
                r = mva_metrics(rows, actions)
                r["per_seed"] = {p: mva_metrics([x for x in rows if x["seed_directory"] == p], actions)
                                 for p in paths if any(x["seed_directory"] == p for x in rows)}
            else:
                cats = Counter(x["category"] for x in rows)
                n = len(rows)
                ok = (sum(x["original_category"] == "valid" for x in rows) if kind == "jsb"
                      else cats["valid_correct_fn"])
                r = dict(n=n, categories=dict(cats), success=ok, success_pct=100 * ok / n,
                         copy=sum("regurgitation" in x["category"] or x["category"] == "placeholder_echo"
                                  for x in rows))
            r.update(model=model, kind=kind)
            self.results[key] = r
            self.manifest.append(dict(group=name, model=model, directories=paths,
                                      files=files[model], records=len(rows), kind=kind))

    def bootstrap(self, seed=20260926, reps=10000):
        """Scenario-cluster bootstrap: resample cases, keeping each case's seeds together."""
        rng = np.random.default_rng(seed)
        for key, r in self.results.items():
            if r["kind"] not in ("icu", "router"):
                continue
            bucket = defaultdict(list)
            for x in self.rows[key]:
                bucket[x["sid"]].append(float(x["pred"] == x["gt"]))
            vals = np.array([np.mean(v) for v in bucket.values()])
            boots = vals[rng.integers(0, len(vals), (reps, len(vals)))].mean(axis=1)
            r["scenario_cluster_bootstrap95"] = [float(v) for v in np.quantile(boots, [0.025, 0.975])]
            r["unique_scenarios"] = len(vals)


def seeds(pattern, n=3):
    return [pattern.format(s=s) for s in range(n)]


def main():
    a = Auditor()
    a.group("icu", seeds("agent_runs_icu_active_sim_seed{s}"), "icu")
    for tag in ["phi4mini", "openbiollm", "aloe", "medgemma_q4", "meditron3", "q8"]:
        a.group("icu", seeds(f"agent_runs_icu_active_sim_{tag}_seed{{s}}"), "icu")
    for prefix in ["agent_runs_binary_active_sim_v5_seed{s}", "agent_runs_medical_v5_seed{s}",
                   "agent_runs_medical_v5_seed{s}_basemodels", "agent_runs_binary_active_sim_v5_openbiollm_seed{s}",
                   "agent_runs_binary_active_sim_v5_aloe_seed{s}", "agent_runs_binary_active_sim_v5_seed{s}_phi4mini",
                   "agent_runs_binary_active_sim_v5_meditron3_seed{s}"]:
        a.group("wearable", seeds(prefix), "wear")
    for tag in ["", "_q8", "_medgemma_q4", "_gemma3n_e4b", "_meditron3", "_exp2", "_exp4", "_exp8",
                "_med42v2", "_biomedllama3"]:
        a.group("router" + tag, seeds(f"agent_runs_toolrouter{tag}_seed{{s}}"), "router")
    for surface, kind in [("bfcl", "bfcl"), ("jsonschemabench", "jsb")]:
        for tag in ["", "_v8", "_v9_exp5", "_q8", "_meditron3", "_perturbed"]:
            a.group(surface + tag, seeds(f"agent_runs_{surface}{tag}_seed{{s}}"), kind)
    for variant in ["bullet", "header", "reverse"]:
        a.group("format_" + variant, seeds(f"agent_runs_toolrouter_perturb_cohort_v16/{variant}_seed{{s}}"), "router")
    # Joint reasoning-field + schema intervention (Section 5.2).
    a.group("cot_schema", seeds("agent_runs_icu_cot_cot_constrained_seed{s}"), "icu")
    a.group("wearable_cot_schema", seeds("agent_runs_p9cot_v5_seed{s}_constrained"), "wear")
    # Additional ICU data: MIMIC-IV scenarios and the held-out eICU draw.
    a.group("mimic", seeds("agent_runs_icu_active_sim_mimic_seed{s}"), "icu")
    a.group("heldout", ["agent_runs_icu_active_sim_heldout_seed0"], "icu")
    a.bootstrap()

    out = REPO / "results"
    out.mkdir(exist_ok=True)
    (out / "audited_results_local.json").write_text(json.dumps(a.results, indent=2) + "\n")
    (out / "run_manifest_local.json").write_text(json.dumps(a.manifest, indent=2) + "\n")
    idx = REPO / "runs_index"
    idx.mkdir(exist_ok=True)
    (idx / "record_index.json").write_text(json.dumps(a.rows, indent=2) + "\n")

    print(f"{len(a.results)} model-condition groups audited")
    for key, r in a.results.items():
        if r["kind"] in ("icu", "wear", "router"):
            print(f"{key:70s} acc={r['accuracy']:.3f} valid={r['n_valid']}/{r['n']} "
                  f"k={r['k']} H={r['H'] if r['H'] is None else round(r['H'], 2)} M={r['M']}")


if __name__ == "__main__":
    main()
