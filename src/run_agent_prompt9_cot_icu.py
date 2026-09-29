"""
ICU joint intervention: reasoning field + schema-constrained decoding
(paper Section 5.2, Appendix E).

Differences from run_agent_prompt9_icu.py:
  - The system prompt asks for reasoning before the action.
  - The output schema adds a 'thinking' field filled before 'action'.
  - Domain rules are restated more compactly.
  - AGENT_CONSTRAINED is forced on (schema-guided decoding via Ollama's
    format= field with the JSON schema).

Reuses call_ollama, parse_response and the intervention-candidate block from
run_agent_prompt9_icu. Single seed by default (seed 0); set AGENT_SEED to vary.
"""
import os
import json
import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
import run_agent_prompt9_icu as p9_icu

BASE = Path(__file__).resolve().parents[1]  # repository root
_DATA_VER = os.environ.get("AGENT_DATA_DIR", "icu_data_v1")
DATA_DIR = BASE / _DATA_VER
_OUT_SUFFIX = os.environ.get("AGENT_OUT_SUFFIX", "_cot_constrained")
OUT_DIR = BASE / f"agent_runs_icu_cot{_OUT_SUFFIX}"
OUT_DIR.mkdir(exist_ok=True)

_DEFAULT_MODELS = [
    "medgemma1.5:4b-it-q8_0",
    "adrienbrault/biomistral-7b:Q4_K_M",
    "meditron:7b-q4_K_M",
]
_MODELS_ENV = os.environ.get("AGENT_MODELS")
MODELS = [m.strip() for m in _MODELS_ENV.split(",") if m.strip()] if _MODELS_ENV else _DEFAULT_MODELS

_DEFAULT_SCENARIOS = sorted(p.stem for p in (BASE / "icu_data_v1" / "scenarios").glob("*.json"))
_SCENARIOS_ENV = os.environ.get("AGENT_SCENARIOS")
SCENARIOS = [s.strip() for s in _SCENARIOS_ENV.split(",") if s.strip()] if _SCENARIOS_ENV else _DEFAULT_SCENARIOS

ACTION_SPACE = p9_icu.ACTION_SPACE  # unchanged
CANDIDATE_PARAMS = p9_icu.CANDIDATE_PARAMS  # unchanged

# ---- Forced schema-constrained decoding for this ablation ----
# Ollama's `format` field accepts a JSON schema object. We add 'thinking'
# as a required field so the model is forced to populate it.
ACTION_SCHEMA_COT = {
    "type": "object",
    "properties": {
        "thinking": {"type": "string"},
        "action": {"type": "string", "enum": ACTION_SPACE},
        "suggestion": {"type": "string"},
        "rationale": {"type": "string"},
        "intervention_option": {"type": "string", "enum": ["A", "B", "C"]},
    },
    "required": ["thinking", "action", "suggestion", "rationale"],
}

SYSTEM_PROMPT_COT = """You are an ICU monitoring agent reviewing 1-hour observation windows from an adult ICU patient. Pick exactly one of six interventions per window.

DOMAIN FACTS
MAP < 65 sustained drives end-organ hypoperfusion; lactate rises as a downstream marker. SpO2 < 92 triggers escalation of oxygen support. Fluid responsiveness is testable bedside with a 250-500 mL bolus and ~10-15 min reassessment. Refractory shock = norepinephrine OR epinephrine >= 0.25 ug/kg/min for >= 4 h with vasopressin on board AND MAP < 65 persists.

DECISION PROCESS - REASON FIRST, THEN COMMIT
For each window, write 2-3 sentences of step-by-step reasoning in the 'thinking' field:
  1. Read the deterioration level and confidence. Combine: high-confidence D means real deterioration; low-confidence D is classifier noise.
  2. Read the vital snapshot (MAP, SpO2, lactate, vasopressors). Compare to trend and intervention history.
  3. Decide whether to continue, recheck vitals, adjust positioning/oxygen, titrate fluids/pressors, escalate to attending, or transfer.
Only after writing the thinking, commit to one action.

DECISION RULES
  - S with conf >= 0.75 AND vitals within normal ranges: continue_monitoring.
  - D with conf < 0.55: classifier noise; treat as no-signal.
  - D with conf >= 0.85 OR D with conf >= 0.70 sustained for 2+ windows: act.
  - MAP < 65 sustained AND fluid bolus given: escalate_to_attending.
  - Norepinephrine OR epinephrine >= 0.25 ug/kg/min for >= 4 h + vasopressin + MAP < 65: transfer_higher_acuity.
  - SpO2 < 92: adjust_positioning_or_oxygen.
  - MAP 65-70 trending down without recent bolus: fluid_or_vasopressor_adjustment (select option A, B, or C).
  - Single ambiguous reading or threshold value: recheck_vitals.

AVAILABLE ACTIONS (pick exactly one)
  continue_monitoring, recheck_vitals, adjust_positioning_or_oxygen,
  fluid_or_vasopressor_adjustment (select A/B/C), escalate_to_attending,
  transfer_higher_acuity.

OUTPUT SCHEMA — STRICTLY THIS JSON OBJECT
{
  "thinking": "<2-3 sentences of step-by-step reasoning>",
  "action": "<one of the 6 actions>",
  "suggestion": "<concrete bedside advice>",
  "rationale": "<one sentence tying action to signals: level, conf, trend, vitals>",
  "intervention_option": "<A, B, or C; required if action is fluid_or_vasopressor_adjustment>"
}
"""


import urllib.request
import urllib.error

OLLAMA_URL = "http://localhost:11434/api/generate"

_SEED_ENV = os.environ.get("AGENT_SEED")
SEED = int(_SEED_ENV) if _SEED_ENV not in (None, "") else 0


def call_ollama_cot(model, prompt, timeout=180, seed=None):
    """Forces schema-guided decoding with the CoT-extended schema."""
    options = {"temperature": 0.2, "num_predict": 600}
    if seed is not None:
        options["seed"] = int(seed)
    body = json.dumps({
        "model": model, "system": SYSTEM_PROMPT_COT, "prompt": prompt,
        "format": ACTION_SCHEMA_COT, "stream": False, "think": False,
        "options": options,
    }).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        latency = time.time() - t0
        return data.get("response", ""), latency, None
    except urllib.error.URLError as e:
        return "", time.time() - t0, str(e)


def run_scenario_cot(model, scenario, seed=None):
    windows = scenario["window_stream"]
    history = []
    trace = []
    candidate_block = p9_icu.build_candidate_block()
    base_seed = SEED if seed is None else seed
    for w in windows:
        prompt = p9_icu.build_user_prompt(w, history, candidate_block)
        window_seed = (base_seed + int(w["window_number"])) if base_seed is not None else None
        response, latency, err = call_ollama_cot(model, prompt, seed=window_seed)
        if err:
            action_strict = action_lenient = suggestion = rationale = opt = None
            failure_category = "transport-error"
            thinking = ""
            parse_err = err
        else:
            # Parse with the same logic as headline runs
            (action_strict, action_lenient, suggestion, rationale, opt,
             failure_category, parse_err) = p9_icu.parse_response(response)
            # Also extract thinking field if present
            try:
                obj = json.loads(response)
                thinking = obj.get("thinking", "") if isinstance(obj, dict) else ""
            except Exception:
                thinking = ""
        gt = w["ground_truth"]
        intervention_result = None
        if action_strict == "fluid_or_vasopressor_adjustment":
            intervention_result = p9_icu.annotate_intervention(opt)
        trace.append({
            "window_number": w["window_number"],
            "gt_action_v1": gt["action"],
            "gt_rule_fired": gt.get("rule", None),
            "deterioration_predicted": {
                "level": w["deterioration_prediction"]["predicted_level"],
                "confidence": w["deterioration_prediction"]["confidence"],
                "trend": w["deterioration_prediction"].get("trend", "n/a"),
            },
            "agent_thinking": thinking,
            "agent_action": action_strict,
            "agent_action_lenient": action_lenient,
            "failure_category": failure_category,
            "agent_suggestion": suggestion,
            "agent_rationale": rationale,
            "agent_selected_option": opt,
            "intervention_annotation": intervention_result,
            "raw_response": response,
            "seed": window_seed,
            "latency_seconds": round(latency, 2),
            "error": parse_err,
        })
        if action_strict and action_strict != "continue_monitoring" and not parse_err:
            history.append({"window_number": w["window_number"], "action": action_strict})
        if action_strict == "transfer_higher_acuity":
            trace.append({"_note": "agent recommended transfer_higher_acuity -- stopping run early"})
            break
    return trace


def main():
    scenarios = {}
    for sid in SCENARIOS:
        with open(BASE / "icu_data_v1" / "scenarios" / f"{sid}.json") as f:
            scenarios[sid] = json.load(f)
    summary = []
    for model in MODELS:
        model_dir = OUT_DIR / model.replace(":", "_").replace("/", "__")
        model_dir.mkdir(exist_ok=True)
        for sid, scenario in scenarios.items():
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {model} on {sid} (CoT+schema)...", flush=True)
            t0 = time.time()
            trace = run_scenario_cot(model, scenario)
            elapsed = time.time() - t0
            actions = [t.get("agent_action") for t in trace if "agent_action" in t]
            errors = [t for t in trace if t.get("error")]
            mean_lat = sum(t.get("latency_seconds", 0) for t in trace) / max(1, len(trace))
            print(f"  done in {elapsed:.1f}s  actions={dict((a, actions.count(a)) for a in set(actions) if a)}  "
                  f"errors={len(errors)}  mean_lat={mean_lat:.2f}s")
            out = dict(
                model=model, scenario_id=sid,
                patientunitstayid=scenario.get("patientunitstayid"),
                care_unit_type=scenario.get("care_unit_type"),
                n_windows=scenario["n_windows"],
                base_seed=SEED,
                total_seconds=round(elapsed, 1),
                mean_latency_seconds=round(mean_lat, 2),
                data_version=_DATA_VER,
                ablation="cot_plus_schema",
                window_trace=trace,
            )
            with open(model_dir / f"{sid}.json", "w") as f:
                json.dump(out, f, indent=2)
            summary.append(dict(
                model=model, scenario_id=sid, n_windows=scenario["n_windows"],
                actions_distribution=dict((a, actions.count(a)) for a in set(actions) if a),
                errors=len(errors),
                mean_latency_seconds=round(mean_lat, 2),
                total_seconds=round(elapsed, 1),
            ))
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(dict(
            ablation="cot_plus_schema",
            scenarios=SCENARIOS, models=MODELS,
            data_version=_DATA_VER,
            candidate_params=CANDIDATE_PARAMS,
            runs=summary,
            generated_at=datetime.now().isoformat(),
        ), f, indent=2)


if __name__ == "__main__":
    main()
