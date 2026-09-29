"""
BFCL Simple Function Call runner — public-benchmark validation for the claim-A
(context-induced engagement) cross-domain finding.

Each instance: a user question + ONE function definition (name, description, parameters
with required/optional). The model is asked to emit a single function call as JSON:
  {"name": "<function_name>", "arguments": {"<param>": <value>, ...}}

We categorize each output:
  * valid_correct_fn        — parses + correct function name
  * valid_wrong_fn          — parses + wrong function name (irrelevant; in single-fn case)
  * schema_regurgitation    — model emitted the function DEFINITION rather than a call
  * placeholder_echo        — literal echo of prompt-skeleton placeholders
  * json_parse_fail         — output is not valid JSON
  * other_invalid           — JSON but not a function-call shape

In the SIMPLE FC subset, only ONE function is available per scenario, so "wrong function"
is unlikely. The key signal is whether the model engages at all (produces a valid call) vs
regurgitating the function definition (Meditron-style) vs failing to parse.

For the prototype cross-walk, we additionally measure:
  * argument value match — for valid_correct_fn outputs, do the parameter values match GT?
"""
import os, json, re, time, sys, urllib.request, urllib.error
from pathlib import Path
from datetime import datetime

BASE = Path(__file__).resolve().parents[1]  # repository root
SC_FILE = BASE / os.environ.get("BFCL_DATA", "bfcl_data/bfcl_sample_100.json")

_OUT_SUFFIX = os.environ.get("AGENT_OUT_SUFFIX", "")
OUT_DIR = BASE / f"agent_runs_bfcl{_OUT_SUFFIX}"
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

PLACEHOLDER_RE = re.compile(r"<\s*(your|the|function|name|param|argument|fn)\s*(call|name|here|placeholder)", re.IGNORECASE)

SYSTEM_PROMPT = """You are a function-calling assistant. The user gives you a question and ONE function definition. Your job is to output a JSON object representing the call to that function with the correct argument values. Return only the JSON, no preamble."""


def build_user_prompt(instance):
    fn = instance["function"]
    fn_json = json.dumps(fn, indent=2)
    return f"""USER QUESTION:
{instance['question']}

AVAILABLE FUNCTION (one function; you must call it):
{fn_json}

Output a JSON object with this exact shape:
{{"name": "<the function name from above>", "arguments": {{"<param_name>": <value>, ...}}}}

Return only the JSON object, no explanation."""


def call_ollama(model, system, prompt, timeout=120, seed=None):
    t0 = time.time()
    body = {"model": model, "system": system, "prompt": prompt, "stream": False,
            "format": "json", "options": {"temperature": 0.2}}
    if seed is not None:
        body["options"]["seed"] = seed
    req = urllib.request.Request(OLLAMA_URL, data=json.dumps(body).encode("utf-8"),
                                  headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read())
            return d.get("response", ""), time.time() - t0, None
    except Exception as e:
        return "", time.time() - t0, f"{type(e).__name__}: {e}"


def detect_function_definition_regurg(obj, ground_truth_fn_name):
    """Did the model emit the function DEFINITION (parameters schema) rather than a call?"""
    if not isinstance(obj, dict):
        return False
    # Schema-meta keys: 'parameters', 'description', 'required', 'type', 'properties'
    # The function CALL should have 'name' + 'arguments' (or function_name).
    SCHEMA_META = {"parameters", "description", "required", "type", "properties", "items"}
    meta_at_top = set(obj.keys()) & SCHEMA_META
    if len(meta_at_top) >= 1:
        return True
    # Also catch: nested {"name": fn_name, "parameters": {...}} which is definition-shape
    if "name" in obj and "parameters" in obj:
        return True
    return False


def categorize(text, instance):
    fn_name = instance["function"]["name"]
    raw = (text or "").strip()
    if not raw:
        return "empty", None, False
    # Strip code fences
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```\s*$", "", raw).strip()
    if PLACEHOLDER_RE.search(raw):
        return "placeholder_echo", None, False
    try:
        obj = json.loads(raw)
    except Exception:
        return "json_parse_fail", None, False
    # Schema-regurgitation check
    if detect_function_definition_regurg(obj, fn_name):
        return "schema_regurgitation", obj, False
    # Function-call shape: {"name": "...", "arguments": {...}}
    if isinstance(obj, dict):
        emitted_name = obj.get("name") or obj.get("function_name") or obj.get("function")
        if isinstance(emitted_name, str):
            if emitted_name == fn_name:
                return "valid_correct_fn", obj, True
            else:
                # Try fuzzy match (some models include namespace)
                if fn_name.split(".")[-1] in emitted_name:
                    return "valid_correct_fn", obj, True
                return "valid_wrong_fn", obj, False
    return "other_invalid", obj, False


def run_one(model, instance, seed=None):
    prompt = build_user_prompt(instance)
    text, lat, err = call_ollama(model, SYSTEM_PROMPT, prompt, seed=seed)
    if err:
        return {"instance_id": instance["instance_id"], "category": "ollama_error",
                "raw_response": text, "error": err, "latency_seconds": round(lat, 2), "seed": seed}
    cat, parsed, correct_name = categorize(text, instance)
    return {"instance_id": instance["instance_id"], "category": cat,
            "raw_response": text, "raw_response_preview": text[:300],
            "parsed_output": parsed, "correct_function_name": correct_name,
            "ground_truth_fn": instance["function"]["name"],
            "error": None, "latency_seconds": round(lat, 2), "seed": seed}


def main():
    print(f"BFCL Simple FC: {len(MODELS)} models x {len(INSTANCES)} instances; seed={SEED}")
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
            with open(model_dir / f"{inst['instance_id']}.json", "w") as f:
                json.dump(result, f, indent=2)
            summary.append({"model": model, "instance_id": inst["instance_id"],
                           "category": result["category"], "latency": result["latency_seconds"]})
            print(f"  done in {elapsed:.1f}s  cat={result['category']}")
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump({"models": MODELS, "n_instances": len(INSTANCES), "seed": SEED,
                   "runs": summary, "generated_at": datetime.now().isoformat()}, f, indent=2)
    print(f"\nWrote {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
