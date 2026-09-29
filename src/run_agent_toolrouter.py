"""
Tool-router NLP case study runner (Section 4.5 generalization demonstration).

Applies the same MVA primitives (action-distribution audit + cross-task design
sensibilities) to a non-medical structured-output decision task: an LLM
chooses one of K=6 tools given a user query. Same model cohort as ICU
(MedGemma, BioMistral, Meditron, Mistral base, Llama-2 base) + Gemma 3n e2b
GP comparator. Same JSON-output schema. Same temperature, same multi-seed
protocol where requested.

Goal: demonstrate that the safety-default / engagement-default / schema-fragile
prototypes hypothesized as medical-FT phenomena are observable on a
non-medical task with the same audit primitives — closing the §4.5
generalization gap.
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
SC_DIR = BASE / "toolrouter_data" / "scenarios"

_OUT_SUFFIX = os.environ.get("AGENT_OUT_SUFFIX", "")
OUT_DIR = BASE / f"agent_runs_toolrouter{_OUT_SUFFIX}"
OUT_DIR.mkdir(exist_ok=True)

_SEED_ENV = os.environ.get("AGENT_SEED")
SEED = int(_SEED_ENV) if _SEED_ENV not in (None, "") else None

OLLAMA_URL = "http://localhost:11434/api/generate"
_DEFAULT_MODELS = [
    "gemma3n:e2b",                            # GP comparator (~2B)
    "phi4-mini:latest",                       # 2nd GP comparator (~4B); medical-side cross-task non-transfer
    "medgemma1.5:4b-it-q8_0",                 # medical FT
    "adrienbrault/biomistral-7b:Q4_K_M",      # medical FT
    "meditron:7b-q4_K_M",                     # medical FT
    "mistral:7b-instruct-v0.2-q4_K_M",        # matched-quantization base
    "llama2:7b-chat-q4_K_M",                  # matched-quantization base
]
_MODELS_ENV = os.environ.get("AGENT_MODELS")
MODELS = [m.strip() for m in _MODELS_ENV.split(",") if m.strip()] if _MODELS_ENV else _DEFAULT_MODELS

# Read scenario IDs from the manifest unless AGENT_SCENARIOS is provided
_SCENARIOS_ENV = os.environ.get("AGENT_SCENARIOS")
if _SCENARIOS_ENV:
    SCENARIOS = [s.strip() for s in _SCENARIOS_ENV.split(",") if s.strip()]
else:
    manifest = json.load(open(BASE / "toolrouter_data" / "toolrouter_manifest.json"))
    SCENARIOS = manifest["scenario_ids"]

TOOL_ENUM = ["calculator", "web_search", "calendar", "email", "file_search", "code_execution"]

SYSTEM_PROMPT = """You are a tool-router agent. The user sends you a natural-language request. Your job is to pick exactly one tool from a fixed enum to handle that request and return a JSON object indicating which tool. Your output goes to a downstream execution layer that will invoke the chosen tool — be precise, not chatty.

AVAILABLE TOOLS (pick exactly one)
  * calculator        — arithmetic operations on numbers the user explicitly provided (addition, subtraction, multiplication, division, square root, exponent, percentage, unit conversion). For short single-expression arithmetic problems.
  * web_search        — look up information on the public web that is not derivable from the user's local files or a calculation: news, prices, definitions, biographical facts, current events, public databases, reviews.
  * calendar          — schedule, view, modify, or block off events on the user's calendar; check availability; find or change meeting times.
  * email             — send, reply to, draft, or forward an email message. The intent is to communicate with another person via email.
  * file_search       — find, locate, retrieve, or open a file the user has previously saved on their device: documents, spreadsheets, presentations, notes, downloads.
  * code_execution    — run code (Python / SQL / regex / shell) or compute the value of a code-level expression that needs interpreting beyond simple arithmetic; multi-step logic; plots; data manipulation over arrays.

TOOL-PRIORITY RULES (apply when the request is ambiguous)
  1. For pure arithmetic on numbers the user provided, prefer `calculator` over `web_search` and over `code_execution`.
  2. For multi-step or array/data computation, prefer `code_execution` over `calculator`.
  3. For "find/locate/retrieve" of personal materials, prefer `file_search` over `email`.
  4. For "send/reply/forward" of a message, prefer `email` over `file_search` and over `calendar`.
  5. For "schedule/check availability/when is X", prefer `calendar` over `email`.
  6. For public-web information not on the user's device, prefer `web_search` over `file_search`.
  7. If none of the rules clearly fires, pick the single tool whose description most directly matches the verb in the user's request.

OUTPUT SCHEMA
Return ONLY a JSON object with these three fields:
{
  "tool":      "<one of the 6 tool names exactly>",
  "rationale": "<one sentence: which verb / intent / keyword in the user request drove the choice>",
  "confidence":"<low | medium | high>"
}
Return only the JSON. Do not include any preamble or trailing text.
"""


def build_user_prompt(scenario):
    """Construct the per-scenario user prompt."""
    return f"USER REQUEST:\n{scenario['query']}\n\nPick exactly one tool from the enum above and return the JSON object."


def call_ollama(model, system, prompt, timeout=120, seed=None):
    """Call Ollama and return (text, latency_seconds, error_or_None)."""
    t0 = time.time()
    body = {
        "model": model,
        "system": system,
        "prompt": prompt,
        "stream": False,
        "format": "json",
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


def parse_response(text):
    """Returns dict with 'tool' set to a TOOL_ENUM entry or None.

    Lenient parsing: dict-wrap unwrap, edit-distance match within 2.
    """
    cat = "other_malformed"
    parsed = None
    try:
        obj = json.loads(text)
    except Exception:
        return {"tool": None, "raw": text, "category": "non_json"}

    if not isinstance(obj, dict):
        return {"tool": None, "raw": text, "category": "non_dict"}

    tool_raw = obj.get("tool") or obj.get("action") or obj.get("name")
    # dict-wrap recovery
    if isinstance(tool_raw, dict):
        for k in ("name", "type", "tool"):
            v = tool_raw.get(k)
            if isinstance(v, str):
                tool_raw = v
                cat = "dict_wrap_recovered"
                break

    if not isinstance(tool_raw, str):
        return {"tool": None, "raw": text, "category": "null_action"}

    norm = tool_raw.strip().lower().replace("-", "_").replace(" ", "_")
    if norm in TOOL_ENUM:
        return {"tool": norm, "raw": text, "category": cat if cat != "other_malformed" else "valid",
                "rationale": obj.get("rationale"), "confidence": obj.get("confidence")}

    # edit-distance lenient match
    def edit_distance(a, b):
        if abs(len(a) - len(b)) > 2: return 99
        m, n = len(a), len(b)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev = dp[0]; dp[0] = i
            for j in range(1, n + 1):
                tmp = dp[j]
                dp[j] = min(dp[j] + 1, dp[j-1] + 1, prev + (0 if a[i-1] == b[j-1] else 1))
                prev = tmp
        return dp[n]
    for canonical in TOOL_ENUM:
        if edit_distance(norm, canonical) <= 2:
            return {"tool": canonical, "raw": text, "category": "near_miss",
                    "rationale": obj.get("rationale"), "confidence": obj.get("confidence")}

    return {"tool": None, "raw": text, "category": "hallucinated_or_offlist"}


def run_scenario(model, scenario, seed=None):
    prompt = build_user_prompt(scenario)
    text, lat, err = call_ollama(model, SYSTEM_PROMPT, prompt, seed=seed)
    if err:
        return {
            "scenario_id": scenario["scenario_id"],
            "agent_tool": None,
            "error": err,
            "raw_response": text,
            "latency_seconds": round(lat, 2),
            "seed": seed,
        }
    parsed = parse_response(text)
    return {
        "scenario_id": scenario["scenario_id"],
        "agent_tool": parsed.get("tool"),
        "rationale": parsed.get("rationale"),
        "confidence": parsed.get("confidence"),
        "failure_category": parsed.get("category"),
        "raw_response": text,
        "latency_seconds": round(lat, 2),
        "seed": seed,
        "error": None,
    }


def main():
    print(f"Tool-router run: {len(MODELS)} models x {len(SCENARIOS)} scenarios; seed={SEED}")
    summary = []
    for model in MODELS:
        model_safe = model.replace(":", "_").replace("/", "__")
        model_dir = OUT_DIR / model_safe
        model_dir.mkdir(exist_ok=True)
        for sid in SCENARIOS:
            scn = json.load(open(SC_DIR / f"{sid}.json"))
            t0 = time.time()
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {model} on {sid}...")
            result = run_scenario(model, scn, seed=SEED)
            elapsed = time.time() - t0
            result["gt_tool"] = scn["ground_truth_tool"]
            result["stratum"] = scn["stratum"]
            result["query"] = scn["query"]
            with open(model_dir / f"{sid}.json", "w") as f:
                json.dump(result, f, indent=2)
            summary.append({
                "model": model, "scenario_id": sid,
                "agent_tool": result.get("agent_tool"),
                "gt_tool": result["gt_tool"],
                "stratum": result["stratum"],
                "failure_category": result.get("failure_category"),
                "latency": result["latency_seconds"],
            })
            print(f"  done in {elapsed:.1f}s  tool={result.get('agent_tool')}  gt={result['gt_tool']}  cat={result.get('failure_category')}")
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump({"models": MODELS, "scenarios": SCENARIOS, "seed": SEED,
                   "runs": summary, "generated_at": datetime.now().isoformat()}, f, indent=2)
    print(f"\nWrote {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
