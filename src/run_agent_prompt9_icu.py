"""
ICU monitoring runner (baseline prompt, paper Section 3.1).

Structural port of the wearable runner (run_agent_prompt9.py): the wearable's
binary fatigue level + confidence is replaced by an ICU deterioration signal
(S=stable / D=deteriorating) + confidence, posture context becomes care-unit and
admission context, and the six wearable actions become the six ICU actions whose
reference rules are implemented in extract_icu_scenarios.py (paper Appendix A).
Prompt wording was fixed before the evaluated runs.

  1. Only the overall (level, confidence) pair is passed, matching the wearable's
     binary back/leg fields.
  2. Three fluid / vasopressor / oxygen intervention candidates are pre-computed
     per window with predicted MAP and SpO2 deltas (Marik fluid-responsiveness
     prior). If the model selects fluid_or_vasopressor_adjustment it names one.
  3. Rules are stated as general principles over (level, confidence, trend,
     history).
"""
import os
import json
import time
import sys
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime

BASE = Path(__file__).resolve().parents[1]  # repository root
_DATA_VER = os.environ.get("AGENT_DATA_DIR", "icu_data_v1")
DATA_DIR = BASE / _DATA_VER
_OUT_SUFFIX = os.environ.get("AGENT_OUT_SUFFIX", "")
OUT_DIR = BASE / f"agent_runs_icu_active_sim{_OUT_SUFFIX}"
OUT_DIR.mkdir(exist_ok=True)

_SEED_ENV = os.environ.get("AGENT_SEED")
SEED = int(_SEED_ENV) if _SEED_ENV not in (None, "") else None

OLLAMA_URL = "http://localhost:11434/api/generate"
_DEFAULT_MODELS = [
    "gemma3n:e2b",
    "medgemma1.5:4b-it-q8_0",
    "adrienbrault/biomistral-7b:Q4_K_M",
    "meditron:7b-q4_K_M",
    "mistral:7b-instruct-v0.2-q4_K_M",
    "llama2:7b-chat-q4_K_M",
]
_MODELS_ENV = os.environ.get("AGENT_MODELS")
MODELS = [m.strip() for m in _MODELS_ENV.split(",") if m.strip()] if _MODELS_ENV else _DEFAULT_MODELS

_DEFAULT_SCENARIOS = ["icu_001", "icu_002", "icu_003"]
_SCENARIOS_ENV = os.environ.get("AGENT_SCENARIOS")
SCENARIOS = [s.strip() for s in _SCENARIOS_ENV.split(",") if s.strip()] if _SCENARIOS_ENV else _DEFAULT_SCENARIOS

ACTION_SPACE = [
    "continue_monitoring",
    "recheck_vitals",
    "adjust_positioning_or_oxygen",
    "fluid_or_vasopressor_adjustment",
    "escalate_to_attending",
    "transfer_higher_acuity",
]

# Three canonical fluid/vasopressor candidates. Predicted MAP deltas grounded in
# Marik fluid-response criterion (ΔSV +10-15% per 500 mL crystalloid → ~5-8 mmHg
# ΔMAP in fluid-responsive subset); see gt_icu_v1.md §3 R3 citations.
CANDIDATE_PARAMS = {
    "A_conservative": {"intervention": "250 mL crystalloid bolus over 10 min",
                       "pred_map_delta_mmhg": 3, "pred_spo2_delta_pp": 0,
                       "time_to_effect_min": 10},
    "B_moderate":     {"intervention": "500 mL crystalloid bolus over 15 min",
                       "pred_map_delta_mmhg": 6, "pred_spo2_delta_pp": 0,
                       "time_to_effect_min": 15},
    "C_pressor":      {"intervention": "norepinephrine titration +0.05 ug/kg/min",
                       "pred_map_delta_mmhg": 8, "pred_spo2_delta_pp": 0,
                       "time_to_effect_min": 5},
}


SYSTEM_PROMPT = """You are an ICU monitoring agent reviewing 1-hour observation windows from an adult ICU patient. Your output goes to the bedside clinical team — be clinically helpful, not just categorical.

HOW TO READ THE INPUT
For each observation window you receive TWO combined pieces of information:
  1. level: S (stable) or D (deteriorating)
  2. confidence: 0.0-1.0, calibrated — "conf=0.80" means this label is correct ~80% of the time
Plus:
  • trend: increasing / stable / decreasing risk over last 3 observation windows
  • window number and ICU hour (session context)
  • intervention history (what was already escalated in prior windows)
  • current vital snapshot (MAP, SpO2, HR, RR, temp, lactate if recent)

NEVER READ THE LEVEL ALONE. The level and the confidence are one combined signal.

Interpret the combined (level, confidence) signal as follows:
  • S with conf >= 0.80     -> strongly stable. Do not intervene.
  • S with conf 0.60-0.80   -> probably stable. Do not intervene on this alone.
  • S with conf < 0.60      -> the classifier is unsure -- treat as "no signal either way" and use trend.
  • D with conf >= 0.85     -> strong signal of real deterioration. Appropriate to act.
  • D with conf 0.70-0.85   -> plausible deterioration -- intervene ONLY if supported by increasing trend OR >=2 prior windows at D.
  • D with conf 0.55-0.70   -> borderline. Treat essentially as "maybe" -- do not intervene on this alone. Wait for a second confirming window.
  • D with conf < 0.55      -> low-quality positive. Ignore this reading; it is almost certainly a classifier artifact, especially early in the ICU stay.

Three-signal rule for acting: before ANY intervention beyond continue_monitoring, one of the following must hold:
  (a) current window has D with conf >= 0.85, OR
  (b) current window has D with conf >= 0.70 AND vital snapshot shows MAP < 65 or SpO2 < 92, OR
  (c) D for >= 2 consecutive windows with conf >= 0.70 on each, OR
  (d) trend is "increasing" AND the current window has D at conf >= 0.70.
If none of (a)-(d) hold, default to continue_monitoring even if the classifier shows D.

Escalation rule: escalate_to_attending only when EITHER
  • D with conf >= 0.85 AND trend is increasing AND MAP < 65 or lactate >= 2 mmol/L with vasopressor on board, OR
  • a prior intervention was applied >= 2 windows ago AND signals remain above the action-threshold AND trend has not turned decreasing.

Transfer to higher acuity only when refractory shock criteria are met: norepinephrine OR epinephrine >= 0.25 ug/kg/min for >= 4 h AND vasopressin on board AND MAP < 65 persists.

THE TASK (steady-state observation window structure)
Each window is 1 hour. We summarize end-of-window vitals + 1h and 6h trends. Lab values (lactate, creatinine) reflect the most recent draw within 2 h (6 h fallback).

PHYSIOLOGY FACTS (broadly applicable, not patient-specific)
  - MAP < 65 mmHg sustained drives end-organ hypoperfusion; lactate rises as a downstream marker.
  - SpO2 < 92% triggers escalation of oxygen support before invasive maneuvers.
  - Fluid responsiveness is testable bedside with a 250-500 mL bolus and ~10-15 min reassessment.
  - Vasopressor escalation past norepinephrine 0.25 ug/kg/min with persistent hypotension defines refractoriness.
  - Lactate elevation with vasopressor requirement after adequate fluid resuscitation defines Sepsis-3 septic shock.

DIVERGENCE RULE (adjust_positioning_or_oxygen only)
Position / oxygen adjustment applies when the deterioration signal is dominantly RESPIRATORY (low SpO2 or rising FiO2) without circulatory failure:
  • SpO2 < 92 on current FiO2 with MAP >= 65 sustained, for >= 1 window.
Do NOT call adjust_positioning_or_oxygen when MAP is below threshold or when shock-state signals (lactate elevation, vasopressor requirement) are present -- those need higher-priority actions.

AVAILABLE ACTIONS (pick exactly one, listed lowest-acuity to highest)
  • continue_monitoring -- no intervention needed
  • recheck_vitals -- request one more measurement before deciding; for ambiguous or single-outlier readings
  • adjust_positioning_or_oxygen -- bedside maneuver (head-of-bed elevation, FiO2 titration, prone positioning)
  • fluid_or_vasopressor_adjustment -- titrate a fluid bolus or vasopressor dose; REQUIRES you to select one of the candidate parameter options listed in the user prompt (by letter: A, B, or C). Pick the least-aggressive option that addresses the deterioration pattern.
  • escalate_to_attending -- alert attending physician; sustained deterioration despite initial resuscitation OR sepsis criteria OR worsening early-warning score
  • transfer_higher_acuity -- transfer to higher-acuity unit / ECMO consult / CRRT initiation; refractory shock or intervention beyond unit capability

OUTPUT SCHEMA
Return ONLY a JSON object:
{
  "action": "<one of the 6 actions>",
  "suggestion": "<concrete 1-sentence advice the bedside team can act on immediately>",
  "rationale": "<one sentence citing which signals drove the choice: level, conf, trend, vitals, history>",
  "intervention_option": "<A, B, or C -- required if action is fluid_or_vasopressor_adjustment, else omit>"
}
"""


def build_candidate_block():
    """Pre-formatted candidate-intervention block. Same shape as wearable's
    build_candidate_block(posture) but using Marik-criterion-derived deltas
    rather than a physiological simulator. Per gt_icu_v1.md §3 R3."""
    lines = []
    for letter, (name, params) in zip("ABC", CANDIDATE_PARAMS.items()):
        lines.append(
            f"  Option {letter} ({name}: {params['intervention']}): "
            f"predicted MAP {params['pred_map_delta_mmhg']:+d} mmHg, "
            f"SpO2 {params['pred_spo2_delta_pp']:+d} pp, "
            f"effect by ~{params['time_to_effect_min']} min"
        )
    return ("Intervention candidates (Marik fluid-response priors; gt_icu_v1.md "
            "§R3 citations):\n" + "\n".join(lines))


def build_user_prompt(window_record, history, candidate_block):
    dp = window_record["deterioration_prediction"]
    vs = window_record["vital_snapshot"]
    ctx = window_record.get("session_context", {})

    history_str = "none yet"
    if history:
        last = history[-1]
        history_str = (f"{len(history)} prior interventions; most recent was "
                       f"'{last['action']}' at window {last['window_number']}")

    payload = {
        "window_number": window_record["window_number"],
        "icu_hour": window_record["elapsed_seconds"] // 3600,
        "session_context": ctx,
        "deterioration_signal": {
            "level": dp["predicted_level"],          # "S" or "D"
            "confidence": round(dp["confidence"], 2),
            "trend": dp.get("trend", "n/a"),
        },
        "vital_snapshot": {
            "map_mmhg": vs.get("map_mmhg"),
            "spo2_pct": vs.get("spo2_pct"),
            "fio2_frac": vs.get("fio2_frac"),
            "hr_bpm": vs.get("hr_bpm"),
            "rr_per_min": vs.get("rr_per_min"),
            "temp_celsius": vs.get("temp_celsius"),
            "lactate_mmol_per_l": vs.get("lactate_mmol_per_l"),
            "lactate_age_hours": vs.get("lactate_age_hours"),
            "vasopressors_active": vs.get("vasopressors_active", []),
        },
        "intervention_history": history_str,
    }
    return ("Current state (binary deterioration signal + vital snapshot + context):\n"
            + json.dumps(payload, indent=2)
            + "\n\n" + candidate_block
            + "\n\nRead (level, confidence) TOGETHER as one signal using the "
              "table in the system prompt. Before recommending anything beyond "
              "continue_monitoring, verify one of the three-signal rule "
              "conditions (a)-(d) is met. If not, pick continue_monitoring. "
              "Respond ONLY with the JSON object.")


_CONSTRAINED = os.environ.get("AGENT_CONSTRAINED", "").lower() in ("1", "true", "yes")
_ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ACTION_SPACE,
        },
        "suggestion": {"type": "string"},
        "rationale": {"type": "string"},
        "intervention_option": {"type": "string", "enum": ["A", "B", "C"]},
    },
    "required": ["action", "suggestion", "rationale"],
}


def call_ollama(model, system, prompt, timeout=120, seed=None):
    options = {"temperature": 0.2, "num_predict": 400}
    if seed is not None:
        options["seed"] = int(seed)
    fmt = _ACTION_SCHEMA if _CONSTRAINED else "json"
    body = json.dumps({
        "model": model, "system": system, "prompt": prompt,
        "format": fmt, "stream": False, "think": False,
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


def _coerce_str(x):
    """Coerce any model output value to a string. Mirrors wearable
    parse_response handling of Meditron's dict-typed action field."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return json.dumps(x) if not isinstance(x, (int, float, bool)) else str(x)


def _extract_dict_wrap_action(action_dict):
    """If action is dict-wrapped (e.g., {'type': 'continue_monitoring'} or
    {'name': '...'}), try to lift out the inner action token. Returns
    None if no inner string field maps to a valid action."""
    if not isinstance(action_dict, dict):
        return None
    for k in ("type", "name", "action", "value", "label"):
        v = action_dict.get(k)
        if isinstance(v, str) and v.strip() in ACTION_SPACE:
            return v.strip()
    return None


def _edit_distance_match(action_str, max_dist=2):
    """Recover near-miss action tokens (e.g., 'adjustmment' -> 'adjustment').
    Returns None if no enum entry within edit distance `max_dist`."""
    from difflib import get_close_matches
    matches = get_close_matches(action_str, ACTION_SPACE, n=1, cutoff=0.85)
    return matches[0] if matches else None


def _categorize_failure(text, obj, action_raw):
    """Returns one of:
      valid / dict-wrap / null-action / placeholder /
      hallucinated-schema / near-miss / unknown-action /
      non-dict / non-json / other-malformed.
    See the failure-mode categories in paper Appendix B."""
    if text == "" or text is None:
        return "empty-response"
    if obj is None:
        return "non-json"
    if not isinstance(obj, dict):
        return "non-dict"
    if action_raw is None:
        return "null-action"
    if isinstance(action_raw, dict):
        return "dict-wrap"
    if not isinstance(action_raw, str):
        return "other-malformed"
    s = action_raw.strip()
    # Placeholder text from the prompt's OUTPUT SCHEMA block
    if "<" in s and ">" in s:
        return "placeholder"
    # Empty or one-word non-enum
    if s in ACTION_SPACE:
        return "valid"
    # Edit-distance lenient
    if _edit_distance_match(s):
        return "near-miss"
    # If the response looks like a totally different schema (e.g., contains
    # FHIR-ish keys at top level), flag as hallucinated-schema
    fhir_ish = {"resourceType", "code", "coding", "subject", "encounter"}
    if any(k in obj for k in fhir_ish):
        return "hallucinated-schema"
    return "unknown-action"


def parse_response(text):
    """Returns: (action_strict, action_lenient, suggestion, rationale,
                 intervention_option, failure_category, error_msg).
    action_strict is used for reported accuracy;
    action_lenient recovers dict-wraps and edit-distance ≤2 near-misses
    for the supporting 'recoverable failure' analysis."""
    obj = None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        return (None, None, None, None, None,
                "non-json", f"json_parse_error: {e}")
    except Exception as e:
        return (None, None, None, None, None,
                "other-malformed", f"parse_error: {type(e).__name__}: {e}")

    if not isinstance(obj, dict):
        return (None, None, None, None, None,
                "non-dict", f"non_dict_response: {type(obj).__name__}")

    action_raw = obj.get("action")
    suggestion = _coerce_str(obj.get("suggestion", "")).strip()
    rationale = _coerce_str(obj.get("rationale", "")).strip()
    opt_raw = obj.get("intervention_option")
    opt = _coerce_str(opt_raw).strip().upper() if opt_raw else None
    if opt and opt not in ("A", "B", "C"):
        opt = None

    category = _categorize_failure(text, obj, action_raw)

    # Strict: only a clean string in ACTION_SPACE counts
    if category == "valid":
        return (action_raw.strip(), action_raw.strip(),
                suggestion, rationale, opt, "valid", None)

    # Lenient recovery paths
    if category == "dict-wrap":
        recovered = _extract_dict_wrap_action(action_raw)
        return (None, recovered, suggestion, rationale, opt,
                "dict-wrap",
                f"non_string_action: dict={json.dumps(action_raw)!r}")
    if category == "near-miss":
        recovered = _edit_distance_match(action_raw.strip())
        return (None, recovered, suggestion, rationale, opt,
                "near-miss",
                f"near_miss_action: {action_raw!r} -> {recovered!r}")

    # No recovery — strict fail
    if category == "null-action":
        err = "non_string_action: NoneType"
    elif category == "placeholder":
        err = f"placeholder_text: {action_raw!r}"
    elif category == "hallucinated-schema":
        err = f"hallucinated_schema: keys={list(obj.keys())}"
    elif category == "unknown-action":
        err = f"unknown_action: {action_raw!r}"
    else:
        err = f"{category}: {action_raw!r}"
    return (None, None, suggestion, rationale, opt, category, err)


def annotate_intervention(opt):
    """Record which intervention candidate the agent picked + its Marik-prior
    predicted deltas. Mirrors wearable annotate_simulator()."""
    if opt is None:
        return None
    letter_to_name = {"A": "A_conservative", "B": "B_moderate", "C": "C_pressor"}
    name = letter_to_name.get(opt)
    if name is None:
        return {"intervention_error": f"unknown option {opt}"}
    p = CANDIDATE_PARAMS[name]
    return {
        "selected_option": opt,
        "selected_name": name,
        "intervention": p["intervention"],
        "predicted_map_delta_mmhg": p["pred_map_delta_mmhg"],
        "predicted_spo2_delta_pp": p["pred_spo2_delta_pp"],
        "time_to_effect_min": p["time_to_effect_min"],
    }


def run_scenario(model, scenario, seed=None):
    windows = scenario["window_stream"]
    history = []
    trace = []
    candidate_block = build_candidate_block()
    base_seed = SEED if seed is None else seed

    for w in windows:
        prompt = build_user_prompt(w, history, candidate_block)
        window_seed = (base_seed + int(w["window_number"])) if base_seed is not None else None
        response, latency, err = call_ollama(model, SYSTEM_PROMPT, prompt, seed=window_seed)
        if err:
            action_strict = action_lenient = None
            suggestion = rationale = None
            opt = None
            failure_category = "transport-error"
            parse_err = err
        else:
            (action_strict, action_lenient, suggestion, rationale, opt,
             failure_category, parse_err) = parse_response(response)
        gt = w["ground_truth"]

        intervention_result = None
        if action_strict == "fluid_or_vasopressor_adjustment":
            intervention_result = annotate_intervention(opt)

        trace.append({
            "window_number": w["window_number"],
            "gt_action_v1": gt["action"],
            "gt_rule_fired": gt.get("rule", None),
            "deterioration_predicted": {
                "level": w["deterioration_prediction"]["predicted_level"],
                "confidence": w["deterioration_prediction"]["confidence"],
                "trend": w["deterioration_prediction"].get("trend", "n/a"),
            },
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
        with open(DATA_DIR / "scenarios" / f"{sid}.json") as f:
            scenarios[sid] = json.load(f)

    summary = []
    for model in MODELS:
        model_dir = OUT_DIR / model.replace(":", "_").replace("/", "__")
        model_dir.mkdir(exist_ok=True)
        for sid, scenario in scenarios.items():
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] {model} on {sid} "
                  f"({scenario['n_windows']} windows)...", flush=True)
            t0 = time.time()
            trace = run_scenario(model, scenario)
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
            scenarios=SCENARIOS, models=MODELS,
            data_version=_DATA_VER,
            candidate_params=CANDIDATE_PARAMS,
            runs=summary,
            generated_at=datetime.now().isoformat(),
        ), f, indent=2)

    print("\n=== Summary ===")
    for r in summary:
        print(f"  {r['model']:40s} {r['scenario_id']:14s} "
              f"actions={r['actions_distribution']} errs={r['errors']} "
              f"lat={r['mean_latency_seconds']}s")


if __name__ == "__main__":
    main()
