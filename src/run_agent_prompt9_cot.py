"""
Wearable joint intervention: reasoning field (+ schema-constrained decoding
when AGENT_CONSTRAINED=1), paper Section 5.2.

Differences from the baseline wearable prompt (run_agent_prompt9.py):
  - The system prompt asks for reasoning before the action.
  - The output schema adds a 'thinking' field filled before 'action'.
  - Domain rules are restated more compactly.

Reuses call_ollama, parse_response, simulator candidates and environment-variable
overrides from run_agent_prompt9.
"""
import os, json, sys, time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
import run_agent_prompt9 as p9

BASE = Path(__file__).resolve().parents[1]  # repository root
_DATA_VER = os.environ.get("AGENT_DATA_DIR", "agent_data_v5")
DATA_DIR = BASE / _DATA_VER
_OUT_SUFFIX = os.environ.get("AGENT_OUT_SUFFIX", "_v5_cot")
OUT_DIR = BASE / f"agent_runs_p9cot{_OUT_SUFFIX}"
OUT_DIR.mkdir(exist_ok=True)

_DEFAULT_MODELS = ["meditron:7b-q4_K_M"]
_MODELS_ENV = os.environ.get("AGENT_MODELS")
MODELS = [m.strip() for m in _MODELS_ENV.split(",") if m.strip()] if _MODELS_ENV else _DEFAULT_MODELS

_DEFAULT_SCENARIOS = [
    "P03_symmetric_with_exoskeleton", "P04_symmetric_with_exoskeleton",
    "P05_asymmetric_without_exoskeleton", "P09_symmetric_with_exoskeleton",
    "P14_asymmetric_without_exoskeleton", "P08_symmetric_with_exoskeleton",
    "P13_asymmetric_with_exoskeleton", "P03_symmetric_without_exoskeleton",
    "P07_symmetric_without_exoskeleton", "P06_asymmetric_with_exoskeleton",
]
_SCENARIOS_ENV = os.environ.get("AGENT_SCENARIOS")
SCENARIOS = [s.strip() for s in _SCENARIOS_ENV.split(",") if s.strip()] if _SCENARIOS_ENV else _DEFAULT_SCENARIOS


SYSTEM_PROMPT_COT = """You are an occupational safety agent monitoring a worker on repetitive trunk-flexion task cycles. Pick exactly one of six interventions per cycle.

DOMAIN FACTS
Sustained trunk flexion fatigues erector spinae (back) and biceps femoris (legs). Passive back-support exoskeletons reduce back load but can shift demand to the legs. Fatigue is reversible in the medium-load window; once high-risk, the session must end.

DECISION PROCESS — REASON FIRST, THEN COMMIT
For each cycle, write 2-3 sentences of step-by-step reasoning in the 'thinking' field:
  1. Read both region levels and confidences. Combine: high-confidence M means real fatigue; low-confidence M is classifier noise.
  2. Compare to recent trend and prior interventions.
  3. Decide whether to do nothing, a soft intervention (micro-break, posture, task modification), an escalation, or an offline refit.
Only after that, commit to one action.

DECISION RULES
  - L with conf >= 0.75 on both regions: continue_monitoring.
  - M with conf < 0.55: classifier noise; treat as no-signal.
  - M with conf >= 0.85 on either region OR M with conf >= 0.70 sustained for 2+ cycles: act.
  - Both regions M with conf >= 0.85 AND increasing trend: escalate_end_session.
  - Back >> legs (and exoskeleton worn): recommend_offline_refit.
  - Back vs legs differ clearly with both signals confident: suggest_posture_change.
  - Otherwise prefer suggest_micro_break (mild, first-line) before suggest_task_modification.

ACTIONS (pick exactly one for the 'action' field)
continue_monitoring | suggest_micro_break | suggest_posture_change | suggest_task_modification | escalate_end_session | recommend_offline_refit

OUTPUT SCHEMA — return ONLY a JSON object with these fields in this order:
{
  "thinking": "<2-3 sentences citing level, conf, trend, history>",
  "action": "<one of the 6 valid actions>",
  "modification_option": "<A, B, or C if action is suggest_task_modification, else omit>",
  "suggestion": "<concrete 1-sentence advice for the worker>"
}
"""


def build_user_prompt_cot(cycle_record, history, scenario, candidate_block):
    fp = cycle_record["fatigue_prediction"]
    bb = fp["back_binary"]; lb = fp["leg_binary"]
    ctx = cycle_record["session_context"]
    history_str = "none yet"
    if history:
        last = history[-1]
        history_str = (f"{len(history)} prior interventions; most recent "
                       f"'{last['action']}' at cycle {last['cycle_number']}")
    payload = {
        "cycle_number": cycle_record["cycle_number"],
        "elapsed_seconds": cycle_record["elapsed_seconds"],
        "session_context": ctx,
        "back_fatigue": {"level": bb["predicted_level"],
                         "confidence": round(bb["confidence"], 2),
                         "trend": bb.get("trend", "n/a")},
        "leg_fatigue":  {"level": lb["predicted_level"],
                         "confidence": round(lb["confidence"], 2),
                         "trend": lb.get("trend", "n/a")},
        "intervention_history": history_str,
    }
    return ("Per-cycle state:\n" + json.dumps(payload, indent=2)
            + "\n\n" + candidate_block
            + "\n\nReason step-by-step in 'thinking', then commit to one action. "
              "Respond ONLY with the JSON object.")


def run_scenario_cot(model, scenario):
    cycles = scenario["cycle_stream"]
    history = []
    trace = []
    posture = scenario["posture"]
    candidate_block = p9.build_candidate_block(posture)
    base_seed = p9.SEED
    for c in cycles:
        prompt = build_user_prompt_cot(c, history, scenario, candidate_block)
        cycle_seed = (base_seed + int(c["cycle_number"])) if base_seed is not None else None
        response, latency, err = p9.call_ollama(model, SYSTEM_PROMPT_COT, prompt, seed=cycle_seed)
        if err:
            action, suggestion, rationale, mod_opt, parse_err = None, None, None, None, err
        else:
            action, suggestion, rationale, mod_opt, parse_err = p9.parse_response(response)
        gt = c["ground_truth"]
        sim_result = p9.annotate_simulator(mod_opt, posture) if action == "suggest_task_modification" else None
        trace.append({
            "cycle_number": c["cycle_number"],
            "rpe_back": gt["RPE_back"], "rpe_legs": gt["RPE_legs"],
            "fatigue_predicted": {
                "back_binary": c["fatigue_prediction"]["back_binary"]["predicted_level"],
                "back_binary_conf": c["fatigue_prediction"]["back_binary"]["confidence"],
                "leg_binary": c["fatigue_prediction"]["leg_binary"]["predicted_level"],
                "leg_binary_conf": c["fatigue_prediction"]["leg_binary"]["confidence"],
            },
            "agent_action": action,
            "agent_suggestion": suggestion,
            "agent_rationale": rationale,
            "agent_selected_option": mod_opt,
            "simulator_annotation": sim_result,
            "raw_response": response,
            "seed": cycle_seed,
            "latency_seconds": round(latency, 2),
            "error": parse_err,
        })
        if action and action != "continue_monitoring" and not parse_err:
            history.append({"cycle_number": c["cycle_number"], "action": action})
        if action == "escalate_end_session":
            trace.append({"_note": "agent escalated end_session — stopping run early"})
            break
    return trace


def main():
    scenarios = {}
    for sid in SCENARIOS:
        with open(DATA_DIR / "scenarios" / f"{sid}.json") as f:
            scenarios[sid] = json.load(f)
    summary = []
    for model in MODELS:
        safe = model.replace(":", "_").replace("/", "__")
        model_dir = OUT_DIR / safe
        model_dir.mkdir(exist_ok=True)
        for sid, scenario in scenarios.items():
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] {model} on {sid} "
                  f"({scenario['n_cycles']} cycles)...", flush=True)
            t0 = time.time()
            trace = run_scenario_cot(model, scenario)
            elapsed = time.time() - t0
            actions = [t.get("agent_action") for t in trace if "agent_action" in t]
            errors = [t for t in trace if t.get("error")]
            mean_lat = sum(t.get("latency_seconds", 0) for t in trace) / max(1, len(trace))
            print(f"  done in {elapsed:.1f}s actions={dict((a, actions.count(a)) for a in set(actions) if a)} "
                  f"errors={len(errors)} lat={mean_lat:.2f}s")
            out = dict(model=model, scenario_id=sid,
                       participant_id=scenario["participant_id"],
                       posture=scenario["posture"], exoskeleton=scenario["exoskeleton"],
                       n_cycles=scenario["n_cycles"], base_seed=p9.SEED,
                       total_seconds=round(elapsed, 1),
                       mean_latency_seconds=round(mean_lat, 2),
                       data_version=_DATA_VER, prompt_variant="9-cot",
                       cycle_trace=trace)
            with open(model_dir / f"{sid}.json", "w") as f:
                json.dump(out, f, indent=2)
            summary.append(dict(model=model, scenario_id=sid, n_cycles=scenario["n_cycles"],
                                actions_distribution=dict((a, actions.count(a)) for a in set(actions) if a),
                                errors=len(errors), mean_latency_seconds=round(mean_lat, 2),
                                total_seconds=round(elapsed, 1)))
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(dict(scenarios=SCENARIOS, models=MODELS, data_version=_DATA_VER,
                       prompt_variant="9-cot", runs=summary,
                       generated_at=datetime.now().isoformat()), f, indent=2)


if __name__ == "__main__":
    main()
