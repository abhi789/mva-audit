"""
Run prompt 9 (winning variant) with MEDICAL fine-tuned models on the full
10-scenario bank. Reuses all prompt 9 logic; only MODELS and OUT_DIR change.

Models tested:
  - medgemma1.5:4b-it-q8_0   — Google medical Gemma 3 fine-tune (Q8 numeric fidelity)
  - adrienbrault/biomistral-7b:Q4_K_M — PubMed-pretrained Mistral
  - meditron:7b-q4_K_M       — Llama-2 + clinical guidelines (WHO/CDC/NICE)
"""
import os, json, sys, time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
import run_agent_prompt9 as p9

BASE = Path(__file__).resolve().parents[1]  # repository root
_DATA_VER = os.environ.get("AGENT_DATA_DIR", "agent_data_v5")
DATA_DIR = BASE / _DATA_VER
_OUT_SUFFIX = os.environ.get("AGENT_OUT_SUFFIX", "_v5")
OUT_DIR = BASE / f"agent_runs_medical{_OUT_SUFFIX}"
OUT_DIR.mkdir(exist_ok=True)

_DEFAULT_MODELS = [
    "medgemma1.5:4b-it-q8_0",
    "adrienbrault/biomistral-7b:Q4_K_M",
    "meditron:7b-q4_K_M",
]
_MODELS_ENV = os.environ.get("AGENT_MODELS")
MODELS = [m.strip() for m in _MODELS_ENV.split(",") if m.strip()] if _MODELS_ENV else _DEFAULT_MODELS

_DEFAULT_SCENARIOS = [
    "P03_symmetric_with_exoskeleton",
    "P04_symmetric_with_exoskeleton",
    "P05_asymmetric_without_exoskeleton",
    "P09_symmetric_with_exoskeleton",
    "P14_asymmetric_without_exoskeleton",
    "P08_symmetric_with_exoskeleton",
    "P13_asymmetric_with_exoskeleton",
    "P03_symmetric_without_exoskeleton",
    "P07_symmetric_without_exoskeleton",
    "P06_asymmetric_with_exoskeleton",
]
_SCENARIOS_ENV = os.environ.get("AGENT_SCENARIOS")
SCENARIOS = [s.strip() for s in _SCENARIOS_ENV.split(",") if s.strip()] if _SCENARIOS_ENV else _DEFAULT_SCENARIOS


def main():
    scenarios = {}
    for sid in SCENARIOS:
        with open(DATA_DIR / "scenarios" / f"{sid}.json") as f:
            scenarios[sid] = json.load(f)

    summary = []
    for model in MODELS:
        # Sanitize model name for dir (slashes/colons)
        safe = model.replace(":", "_").replace("/", "__")
        model_dir = OUT_DIR / safe
        model_dir.mkdir(exist_ok=True)
        for sid, scenario in scenarios.items():
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] {model} on {sid} "
                  f"({scenario['n_cycles']} cycles)...", flush=True)
            t0 = time.time()
            trace = p9.run_scenario(model, scenario)
            elapsed = time.time() - t0
            actions = [t.get("agent_action") for t in trace if "agent_action" in t]
            errors = [t for t in trace if t.get("error")]
            mean_lat = sum(t.get("latency_seconds", 0) for t in trace) / max(1, len(trace))
            print(f"  done in {elapsed:.1f}s actions={dict((a, actions.count(a)) for a in set(actions) if a)} "
                  f"errors={len(errors)} lat={mean_lat:.2f}s")
            out = dict(
                model=model, scenario_id=sid,
                participant_id=scenario["participant_id"],
                posture=scenario["posture"], exoskeleton=scenario["exoskeleton"],
                v4rev_tags=scenario.get("tags_v4rev", []),
                n_cycles=scenario["n_cycles"],
                base_seed=p9.SEED,
                total_seconds=round(elapsed, 1),
                mean_latency_seconds=round(mean_lat, 2),
                data_version=_DATA_VER,
                cycle_trace=trace,
            )
            with open(model_dir / f"{sid}.json", "w") as f:
                json.dump(out, f, indent=2)
            summary.append(dict(model=model, scenario_id=sid, n_cycles=scenario["n_cycles"],
                                actions_distribution=dict((a, actions.count(a)) for a in set(actions) if a),
                                errors=len(errors),
                                mean_latency_seconds=round(mean_lat, 2),
                                total_seconds=round(elapsed, 1)))

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(dict(scenarios=SCENARIOS, models=MODELS, data_version=_DATA_VER,
                       runs=summary, generated_at=datetime.now().isoformat()), f, indent=2)


if __name__ == "__main__":
    main()
