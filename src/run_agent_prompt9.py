"""
Prompt 9: binary-only + generalized + active simulator.

Design decisions vs prior prompts:
  1. DROP tertiary classifier fields. Tertiary models have LOSO accuracy 0.51-0.65
     (below majority-class floor), so surfacing them to the agent adds pure noise.
     Agent sees only binary {L, M} + calibrated confidence + trend + history +
     elapsed time.

  2. ACTIVE simulator. For every cycle, we pre-compute 3 sensible task-modification
     candidate param sets and their predicted back/leg load changes. These appear
     in the user prompt. If the agent picks suggest_task_modification, it selects
     one of Options A/B/C by name. This turns the simulator from post-hoc annotation
     into an in-the-loop option presenter.

  3. GENERALIZED rules. Rules are stated as principles that would transfer to any
     noisy-classifier + RPE-target setup, not as dataset-specific thresholds. The
     agent reasons over (level, confidence, trend, duration, session context)
     rather than matching magic numbers.

  4. Evaluation unchanged — same GT, same baselines, same McNemar + Wilson CIs
     via rescore_all.py.
"""
import os
import json
import time
import sys
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from ergonomic_simulator import compare_to_baseline, BASELINE_PARAMS

BASE = Path(__file__).resolve().parents[1]  # repository root
_DATA_VER = os.environ.get("AGENT_DATA_DIR", "agent_data_v4")
V3_DIR = BASE / _DATA_VER
_OUT_SUFFIX = os.environ.get("AGENT_OUT_SUFFIX", "")
OUT_DIR = BASE / f"agent_runs_binary_active_sim{_OUT_SUFFIX}"
OUT_DIR.mkdir(exist_ok=True)

_SEED_ENV = os.environ.get("AGENT_SEED")
SEED = int(_SEED_ENV) if _SEED_ENV not in (None, "") else None

OLLAMA_URL = "http://localhost:11434/api/generate"
_DEFAULT_MODELS = [
    "gemma3n:e2b", "gemma3n:e4b",
    "gemma4:e2b",  "gemma4:e4b",
    "qwen3.5:2b",  "qwen3.5:4b"
]
_MODELS_ENV = os.environ.get("AGENT_MODELS")
MODELS = [m.strip() for m in _MODELS_ENV.split(",") if m.strip()] if _MODELS_ENV else _DEFAULT_MODELS

_DEFAULT_SCENARIOS = [
    "P03_symmetric_with_exoskeleton",
    "P04_symmetric_with_exoskeleton",
    "P05_asymmetric_without_exoskeleton",
    "P09_symmetric_with_exoskeleton",
    "P14_asymmetric_without_exoskeleton",
]
_SCENARIOS_ENV = os.environ.get("AGENT_SCENARIOS")
SCENARIOS = [s.strip() for s in _SCENARIOS_ENV.split(",") if s.strip()] if _SCENARIOS_ENV else _DEFAULT_SCENARIOS

ACTION_SPACE = [
    "continue_monitoring",
    "suggest_micro_break",
    "suggest_posture_change",
    "suggest_task_modification",
    "escalate_end_session",
    "recommend_offline_refit",
]

# Three canonical task-mod proposals the simulator evaluates once per (scenario,
# cycle). These are mild / medium / aggressive trade-offs between cycle time
# overhead and load reduction. Simulator runs on the *scenario's posture*.
CANDIDATE_PARAMS = {
    "A_mild":      {"SUS": 25},                  # just shave SUS, no time cost
    "B_medium":    {"SUS": 22, "Rest": 20},      # moderate SUS cut + modest rest
    "C_aggressive":{"SUS": 20, "Rest": 25},      # larger SUS cut + longer rest
}


SYSTEM_PROMPT = """You are an occupational safety agent monitoring a worker doing repetitive trunk-flexion task cycles. The worker may be wearing a passive back-support exoskeleton (BackX Model AC). Your output goes to the worker directly — be clinically helpful, not just categorical.

HOW TO READ THE INPUT
Each region (back, legs) gives you TWO pieces of information that you MUST combine:
  1. level: L (low fatigue) or M (medium-or-above fatigue)
  2. confidence: 0.0-1.0, calibrated — "conf=0.80" means this label is correct ~80% of the time
Plus:
  • trend: increasing / stable / decreasing over last 3 cycles
  • cycle number and elapsed seconds (session context)
  • intervention history (what you already suggested in prior cycles)

NEVER READ THE LEVEL ALONE. The level and the confidence are one combined signal.

Interpret the combined (level, confidence) signal as follows:
  • L with conf ≥ 0.80     → strongly low fatigue. Do not intervene on this region.
  • L with conf 0.60-0.80  → probably low. Do not intervene on this region alone.
  • L with conf < 0.60     → the classifier is unsure — treat as "no signal either way" and use trend.
  • M with conf ≥ 0.85     → strong signal of real medium fatigue. Appropriate to act.
  • M with conf 0.70-0.85  → plausible fatigue — intervene ONLY if supported by increasing trend OR ≥2 prior cycles at M.
  • M with conf 0.55-0.70  → borderline. Treat essentially as "maybe" — do not intervene on this alone. Wait for a second confirming cycle.
  • M with conf < 0.55     → low-quality positive. Ignore this reading; it is almost certainly a classifier artifact, especially early in the session.

Three-signal rule for acting: before ANY intervention beyond continue_monitoring, one of the following must hold:
  (a) at least one region has M with conf ≥ 0.85 on this cycle, OR
  (b) both regions have M with conf ≥ 0.70 on this cycle, OR
  (c) at least one region has been M for ≥ 2 consecutive cycles with conf ≥ 0.70 on each, OR
  (d) trend is "increasing" AND the current cycle has M at conf ≥ 0.70.
If none of (a)-(d) hold, default to continue_monitoring even if the classifier shows M.

Escalation rule: escalate_end_session only when EITHER
  • both regions hit M with conf ≥ 0.85 AND trend is increasing in at least one region, OR
  • your last intervention was ≥ 2 cycles ago AND signals are still above the action-threshold from the three-signal rule AND trend has not turned decreasing.

THE TASK (steady-state cycle structure, in seconds)
  SS (stand-still start): 15   B (bend down): 3   SUS (sustained bent hold): 30   R (retract): 3   SE (stand-still end): 15   Rest: 15
Total cycle ≈ 81 s. Sustained bend is the hardest phase.

ERGONOMICS FACTS (device- and cohort-independent)
  - Sustained trunk flexion at ~45° fatigues the erector spinae (back) and biceps femoris (legs).
  - Passive back-support exoskeletons reduce back load but can shift demand to the legs.
  - The BackX is passive-mechanical — spring tension is set at donning, not adjusted mid-task.
  - Fatigue is reversible in the "medium" window. Once the worker is in the high-risk state, no in-task intervention fixes it: the session must end.

DIVERGENCE RULE (posture_change only)
Posture change applies ONLY when back and legs differ clearly AND both signals are confident:
  • one region is L with conf ≥ 0.75 AND the other region is M with conf ≥ 0.75, for ≥ 2 consecutive cycles.
Do NOT call posture_change when one side is M at low confidence — that's classifier noise, not a posture problem.

AVAILABLE ACTIONS (pick exactly one)
  • continue_monitoring — no intervention needed
  • suggest_micro_break — 15-30 s rest, first-line for rising medium fatigue in one region
  • suggest_posture_change — symmetric ↔ asymmetric swap; ONLY when both-region divergence is genuine (principle 3)
  • suggest_task_modification — reduce cycle intensity; REQUIRES you to select one of the candidate parameter options listed in the user prompt (by letter: A, B, or C). Pick the least-aggressive option that addresses the fatigue pattern.
  • escalate_end_session — recommend stopping. Use when sustained medium in both regions with increasing trend, or when prior interventions didn't help, or late in session with strong fatigue signal.
  • recommend_offline_refit — end session + book exoskeleton refit; only if exoskeleton is worn AND back is persistently worse than legs.

OUTPUT SCHEMA
Return ONLY a JSON object:
{
  "action": "<one of the 6 actions>",
  "suggestion": "<concrete 1-sentence advice the worker can act on immediately>",
  "rationale": "<one sentence citing which signals drove the choice: level, conf, trend, duration, elapsed, history>",
  "modification_option": "<A, B, or C — required if action is suggest_task_modification, else omit>"
}
"""


def build_candidate_block(posture):
    """Pre-compute simulator outputs for the 3 candidates, for this posture."""
    lines = []
    for letter, (name, params) in zip("ABC", CANDIDATE_PARAMS.items()):
        full = {ph: dict(cfg) for ph, cfg in BASELINE_PARAMS.items()}
        for k, v in params.items():
            full[k]["duration"] = v
        try:
            res = compare_to_baseline(full, posture)
            dback = res["back_activity_change_pct"]
            dleg = res["leg_activity_change_pct"]
            dtime = res["modified"]["cycle_time_seconds"] - res["baseline"]["cycle_time_seconds"]
            params_str = ", ".join(f"{k}={v}s" for k, v in params.items())
            lines.append(
                f"  Option {letter} ({name}: {params_str}): "
                f"predicted back load {dback:+.1f}%, leg load {dleg:+.1f}%, "
                f"cycle time change {dtime:+.0f} s"
            )
        except Exception as e:
            lines.append(f"  Option {letter} (sim error: {e})")
    return "Task-modification candidates already evaluated by the simulator:\n" + "\n".join(lines)


def build_user_prompt(cycle_record, history, session_context, candidate_block):
    fp = cycle_record["fatigue_prediction"]
    # BINARY ONLY — we are deliberately NOT passing tertiary fields
    bb = fp["back_binary"]; lb = fp["leg_binary"]
    ctx = cycle_record["session_context"]

    history_str = "none yet"
    if history:
        last = history[-1]
        history_str = (f"{len(history)} prior interventions; most recent was "
                       f"'{last['action']}' at cycle {last['cycle_number']}")

    payload = {
        "cycle_number": cycle_record["cycle_number"],
        "elapsed_seconds": cycle_record["elapsed_seconds"],
        "session_context": ctx,
        "back_fatigue": {
            "level": bb["predicted_level"],
            "confidence": round(bb["confidence"], 2),
            "trend": bb.get("trend", "n/a"),
        },
        "leg_fatigue": {
            "level": lb["predicted_level"],
            "confidence": round(lb["confidence"], 2),
            "trend": lb.get("trend", "n/a"),
        },
        "intervention_history": history_str,
    }
    return ("Current state (binary fatigue + context):\n"
            + json.dumps(payload, indent=2)
            + "\n\n" + candidate_block
            + "\n\nFor each region, read (level, confidence) TOGETHER as one signal using the "
              "table in the system prompt. Before recommending anything beyond continue_monitoring, "
              "verify one of the three-signal rule conditions (a)-(d) is met. If not, pick "
              "continue_monitoring. Respond ONLY with the JSON object.")


_CONSTRAINED = os.environ.get("AGENT_CONSTRAINED", "").lower() in ("1", "true", "yes")
_ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": [
                "continue_monitoring", "suggest_micro_break", "suggest_posture_change",
                "suggest_task_modification", "escalate_end_session", "recommend_offline_refit",
            ],
        },
        "suggestion": {"type": "string"},
        "rationale": {"type": "string"},
        "modification_option": {"type": "string", "enum": ["A", "B", "C"]},
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
    """Coerce any model output value to a string for downstream string ops.
    Models like Meditron sometimes return dicts where strings are expected — we
    log this as a parse_err but don't crash."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return json.dumps(x) if not isinstance(x, (int, float, bool)) else str(x)


def parse_response(text):
    try:
        obj = json.loads(text)
        if not isinstance(obj, dict):
            return None, None, None, None, f"non_dict_response: {type(obj).__name__}"
        action_raw = obj.get("action", "")
        action = _coerce_str(action_raw).strip()
        suggestion = _coerce_str(obj.get("suggestion", "")).strip()
        rationale = _coerce_str(obj.get("rationale", "")).strip()
        mod_opt_raw = obj.get("modification_option")
        mod_opt = _coerce_str(mod_opt_raw).strip().upper() if mod_opt_raw else None
        if mod_opt and mod_opt not in ("A", "B", "C"):
            mod_opt = None
        # If the model wrapped the action in a non-string (dict / list), record
        # that as a schema violation rather than treating the json-stringified
        # form as a valid action.
        if not isinstance(action_raw, str):
            return None, suggestion, rationale, mod_opt, (
                f"non_string_action: {type(action_raw).__name__}={action!r}"
            )
        if action in ACTION_SPACE:
            return action, suggestion, rationale, mod_opt, None
        return action, suggestion, rationale, mod_opt, f"unknown_action: {action!r}"
    except json.JSONDecodeError as e:
        return None, None, None, None, f"json_parse_error: {e}"
    except Exception as e:
        return None, None, None, None, f"parse_error: {type(e).__name__}: {e}"


def annotate_simulator(mod_opt, posture):
    """Record which candidate the agent picked + the simulator's prediction."""
    if mod_opt is None:
        return None
    letter_to_name = {"A": "A_mild", "B": "B_medium", "C": "C_aggressive"}
    name = letter_to_name.get(mod_opt)
    if name is None:
        return {"simulator_error": f"unknown option {mod_opt}"}
    params = CANDIDATE_PARAMS[name]
    full = {ph: dict(cfg) for ph, cfg in BASELINE_PARAMS.items()}
    for k, v in params.items():
        full[k]["duration"] = v
    try:
        res = compare_to_baseline(full, posture)
        return {
            "selected_option": mod_opt,
            "selected_name": name,
            "params": params,
            "predicted_back_change_pct": res["back_activity_change_pct"],
            "predicted_leg_change_pct": res["leg_activity_change_pct"],
            "cycle_time_change_s": res["modified"]["cycle_time_seconds"]
                                   - res["baseline"]["cycle_time_seconds"],
        }
    except Exception as e:
        return {"simulator_error": str(e)}


def run_scenario(model, scenario, seed=None):
    cycles = scenario["cycle_stream"]
    history = []
    trace = []
    posture = scenario["posture"]
    # Compute candidate block once per scenario (baseline cycle structure
    # doesn't change within a scenario)
    candidate_block = build_candidate_block(posture)
    base_seed = SEED if seed is None else seed

    for c in cycles:
        prompt = build_user_prompt(c, history, scenario, candidate_block)
        # Per-cycle seed = base_seed + cycle_number → deterministic but distinct
        # per cycle, so we don't repeat the same generation 17 times in a row.
        cycle_seed = (base_seed + int(c["cycle_number"])) if base_seed is not None else None
        response, latency, err = call_ollama(model, SYSTEM_PROMPT, prompt, seed=cycle_seed)
        if err:
            action, suggestion, rationale, mod_opt, parse_err = None, None, None, None, err
        else:
            action, suggestion, rationale, mod_opt, parse_err = parse_response(response)
        gt = c["ground_truth"]

        sim_result = None
        if action == "suggest_task_modification":
            sim_result = annotate_simulator(mod_opt, posture)

        trace.append({
            "cycle_number": c["cycle_number"],
            "rpe_back": gt["RPE_back"],
            "rpe_legs": gt["RPE_legs"],
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
        with open(V3_DIR / "scenarios" / f"{sid}.json") as f:
            scenarios[sid] = json.load(f)

    summary = []
    for model in MODELS:
        model_dir = OUT_DIR / model.replace(":", "_").replace("/", "__")
        model_dir.mkdir(exist_ok=True)
        for sid, scenario in scenarios.items():
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] {model} on {sid} "
                  f"({scenario['n_cycles']} cycles)...", flush=True)
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
                participant_id=scenario["participant_id"],
                posture=scenario["posture"], exoskeleton=scenario["exoskeleton"],
                v3_tags=scenario.get("tags", []),
                n_cycles=scenario["n_cycles"],
                base_seed=SEED,
                total_seconds=round(elapsed, 1),
                mean_latency_seconds=round(mean_lat, 2),
                data_version=_DATA_VER,
                cycle_trace=trace,
            )
            with open(model_dir / f"{sid}.json", "w") as f:
                json.dump(out, f, indent=2)
            summary.append(dict(
                model=model, scenario_id=sid, n_cycles=scenario["n_cycles"],
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
        print(f"  {r['model']:15s} {r['scenario_id']:45s} "
              f"actions={r['actions_distribution']} errs={r['errors']} "
              f"lat={r['mean_latency_seconds']}s")


if __name__ == "__main__":
    main()
