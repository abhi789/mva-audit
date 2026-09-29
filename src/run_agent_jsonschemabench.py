"""
JSONSchemaBench runner — cross-domain validation for the Meditron placeholder-echo
schema-fragility claim (Geng et al. 2025, arXiv:2501.10868).

Same Ollama infrastructure as the medical / tool-router runs. For each scenario,
the model is shown the JSON schema and asked to produce a valid JSON instance
conforming to it. We then categorize the output:

  * valid       — JSON parses + schema-validates
  * json_parse_fail — output is not valid JSON
  * schema_violation — output parses as JSON but fails schema validation
  * placeholder_echo — output literally echoes the prompt's placeholder text
                       ("<your JSON instance here>", "<example>" etc.)
  * empty           — output is empty or whitespace

The cross-domain claim we want to test: Meditron's placeholder-echo failure mode
on medical Task B + NLP tool-router should reproduce on JSONSchemaBench.
"""
import os
import json
import re
import time
import sys
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime
import jsonschema

BASE = Path(__file__).resolve().parents[1]  # repository root
SC_FILE = BASE / os.environ.get("JSB_DATA", "jsonschema_bench/jsonschema_sample_100.json")

_OUT_SUFFIX = os.environ.get("AGENT_OUT_SUFFIX", "")
OUT_DIR = BASE / f"agent_runs_jsonschemabench{_OUT_SUFFIX}"
OUT_DIR.mkdir(exist_ok=True)

_SEED_ENV = os.environ.get("AGENT_SEED")
SEED = int(_SEED_ENV) if _SEED_ENV not in (None, "") else None

OLLAMA_URL = "http://localhost:11434/api/generate"
_DEFAULT_MODELS = [
    "medgemma1.5:4b-it-q8_0",
    "adrienbrault/biomistral-7b:Q4_K_M",
    "meditron:7b-q4_K_M",
    "mistral:7b-instruct-v0.2-q4_K_M",
]
_MODELS_ENV = os.environ.get("AGENT_MODELS")
MODELS = [m.strip() for m in _MODELS_ENV.split(",") if m.strip()] if _MODELS_ENV else _DEFAULT_MODELS

INSTANCES = json.load(open(SC_FILE))

# Placeholder-echo detection: literal phrases from the prompt skeleton
PLACEHOLDER_PATTERNS = [
    r"<your\s+json\s+instance",
    r"<json\s+instance\s+here>",
    r"<example>",
    r"<your\s+output>",
    r"<the\s+(json|object|instance)\s+(here|conforming)",
    r"<a\s+valid\s+json",
    r"<replace\s+with",
    r"<fill\s+in",
]
PLACEHOLDER_RE = re.compile("|".join(PLACEHOLDER_PATTERNS), re.IGNORECASE)

SYSTEM_PROMPT = """You are a JSON-generation assistant. The user provides a JSON Schema. Your job is to output one valid JSON instance that conforms to the schema. Return only the JSON object, no explanation, no markdown fences."""


def build_user_prompt(schema_text):
    return f"""Given the following JSON Schema, produce one valid JSON instance that conforms to it. Return only the JSON object, no preamble or markdown.

SCHEMA:
{schema_text}

YOUR JSON INSTANCE:
<your JSON instance here>"""


def call_ollama(model, system, prompt, timeout=120, seed=None):
    t0 = time.time()
    body = {
        "model": model,
        "system": system,
        "prompt": prompt,
        "stream": False,
        "format": "json",  # Match the medical-task runner — force JSON-mode
        "options": {"temperature": 0.2},
    }
    if seed is not None:
        body["options"]["seed"] = seed
    req = urllib.request.Request(
        OLLAMA_URL, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read())
            return d.get("response", ""), time.time() - t0, None
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        return "", time.time() - t0, f"{type(e).__name__}: {e}"
    except Exception as e:
        return "", time.time() - t0, f"{type(e).__name__}: {e}"


def categorize_output(text, schema_obj):
    """Return one of: valid | json_parse_fail | schema_violation | placeholder_echo | empty."""
    t = (text or "").strip()
    if not t:
        return "empty", None
    # Strip common JSON code-fence wrappers
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```\s*$", "", t)
        t = t.strip()
    # Placeholder echo check (before parse attempt — many models emit literal text)
    if PLACEHOLDER_RE.search(t):
        return "placeholder_echo", None
    # JSON parse attempt
    try:
        obj = json.loads(t)
    except Exception:
        return "json_parse_fail", None
    # Schema validation
    try:
        jsonschema.validate(instance=obj, schema=schema_obj)
        return "valid", obj
    except jsonschema.exceptions.ValidationError:
        return "schema_violation", obj
    except jsonschema.exceptions.SchemaError:
        # Schema itself malformed (rare; treat as valid output if JSON parses)
        return "valid", obj


def run_one(model, scenario, seed=None):
    try:
        schema_obj = json.loads(scenario["json_schema"])
    except Exception:
        schema_obj = None
    prompt = build_user_prompt(scenario["json_schema"])
    text, lat, err = call_ollama(model, SYSTEM_PROMPT, prompt, seed=seed)
    if err:
        return {
            "instance_id": scenario["instance_id"],
            "split": scenario["split"],
            "category": "ollama_error",
            "raw_response": text,
            "error": err,
            "latency_seconds": round(lat, 2),
            "seed": seed,
        }
    cat, parsed = (categorize_output(text, schema_obj)
                   if schema_obj is not None
                   else ("schema_undefined", None))
    return {
        "instance_id": scenario["instance_id"],
        "split": scenario["split"],
        "category": cat,
        "raw_response": text,
        "raw_response_preview": text[:300],
        "parsed_output": parsed,
        "error": None,
        "latency_seconds": round(lat, 2),
        "seed": seed,
    }


def main():
    print(f"JSONSchemaBench: {len(MODELS)} models x {len(INSTANCES)} instances; seed={SEED}")
    summary = []
    for model in MODELS:
        model_safe = model.replace(":", "_").replace("/", "__")
        model_dir = OUT_DIR / model_safe
        model_dir.mkdir(exist_ok=True)
        for inst in INSTANCES:
            t0 = time.time()
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {model} on {inst['instance_id']}...")
            result = run_one(model, inst, seed=SEED)
            elapsed = time.time() - t0
            with open(model_dir / f"{inst['instance_id'].replace('/', '_')}.json", "w") as f:
                json.dump(result, f, indent=2)
            summary.append({
                "model": model,
                "instance_id": inst["instance_id"],
                "split": inst["split"],
                "category": result["category"],
                "latency": result["latency_seconds"],
            })
            print(f"  done in {elapsed:.1f}s  cat={result['category']}")
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump({"models": MODELS, "n_instances": len(INSTANCES), "seed": SEED,
                   "runs": summary, "generated_at": datetime.now().isoformat()}, f, indent=2)
    print(f"\nWrote {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
