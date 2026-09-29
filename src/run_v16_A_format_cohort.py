#!/usr/bin/env python3
"""Tool-router format perturbations (paper Section 4.4, Appendix E).

Runs the router task under three prompt-format variants (bullet, header,
reverse tool order) for the listed models, 3 seeds x 50 requests each.
Set AGENT_MODELS to override the model list, e.g.
    AGENT_MODELS=meditron:7b-q8_0 python src/run_v16_A_format_cohort.py
reproduces the Meditron-7B Q8 format grid.

Output: agent_runs_toolrouter_perturb_cohort_v16/<variant>_seed<S>/<model>/<sid>.json
"""
import json
import os
import time
import sys
import urllib.request
import urllib.error
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]  # repository root
OLLAMA_URL = "http://localhost:11434/api/generate"
OUT_DIR = BASE / "agent_runs_toolrouter_perturb_cohort_v16"
OUT_DIR.mkdir(exist_ok=True)
TOOL_ENUM = ["calculator", "web_search", "calendar", "email", "file_search", "code_execution"]

# Main router cohort
COHORT_MODELS = [
    "gemma3n:e2b",
    "gemma3n:e4b",
    "phi4-mini:latest",
    "medgemma1.5:4b-it-q8_0",
    "adrienbrault/biomistral-7b:Q4_K_M",
    "meditron:7b-q4_K_M",
    "mistral:7b-instruct-v0.2-q4_K_M",
    "llama2:7b-chat-q4_K_M",
    # Additional models (tag names vary by local install):
    # "openbiollm:8b-q4",  # name varies; uncomment if pulled locally
    # "aloe:7b-q4",        # name varies; uncomment if pulled locally
]
_MODELS_ENV = os.environ.get("AGENT_MODELS")
if _MODELS_ENV:
    COHORT_MODELS = [m.strip() for m in _MODELS_ENV.split(",") if m.strip()]

ORIGINAL_SYSTEM_PROMPT = """You are a tool-router agent. The user sends you a natural-language request. Your job is to pick exactly one tool from a fixed enum to handle that request and return a JSON object indicating which tool. Your output goes to a downstream execution layer that will invoke the chosen tool — be precise, not chatty.

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

def perturb_bullet(p): return p.replace("  * ", "  - ")
def perturb_header(p):
    p = p.replace("AVAILABLE TOOLS (pick exactly one)", "# Available Tools")
    p = p.replace("TOOL-PRIORITY RULES (apply when the request is ambiguous)", "# Tool Priority Rules")
    p = p.replace("OUTPUT SCHEMA", "# Output Schema")
    return p
def perturb_reverse_tools(p):
    lines = p.split("\n")
    tool_lines = []; start_idx = None
    for i, line in enumerate(lines):
        if line.startswith("  * ") and any(t in line for t in TOOL_ENUM):
            if start_idx is None: start_idx = i
            tool_lines.append(line)
    if start_idx is None: return p
    end_idx = start_idx + len(tool_lines)
    lines[start_idx:end_idx] = list(reversed(tool_lines))
    return "\n".join(lines)

PERTURBATIONS = {"bullet": perturb_bullet, "header": perturb_header, "reverse": perturb_reverse_tools}

def call_ollama(model, system, prompt, timeout=120, seed=None):
    t0 = time.time()
    body = {"model": model, "system": system, "prompt": prompt, "stream": False,
            "format": "json", "options": {"temperature": 0.2}}
    if seed is not None: body["options"]["seed"] = seed
    req = urllib.request.Request(OLLAMA_URL, data=json.dumps(body).encode("utf-8"),
                                  headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read())
            return d.get("response", ""), time.time()-t0, None
    except Exception as e:
        return "", time.time()-t0, f"{type(e).__name__}: {e}"

def parse_tool(text):
    if not text: return None
    try: obj = json.loads(text)
    except Exception: return None
    if not isinstance(obj, dict): return None
    t = obj.get("tool") or obj.get("action") or obj.get("name")
    if isinstance(t, dict):
        for k in ("name", "type", "tool"):
            v = t.get(k)
            if isinstance(v, str): t = v; break
    if not isinstance(t, str): return None
    t = t.strip().lower()
    return t if t in TOOL_ENUM else None

def load_scenarios():
    scen_dir = BASE / "toolrouter_data" / "scenarios"
    return [json.load(open(f)) for f in sorted(scen_dir.glob("tr_*.json"))]

def main():
    scenarios = load_scenarios()
    print(f"[A] Loaded {len(scenarios)} scenarios; {len(COHORT_MODELS)} cohort models; 3 perturbations × 3 seeds")
    if not scenarios:
        print("ERROR: no scenarios"); return
    seeds = [0, 1, 2]
    perturbations = list(PERTURBATIONS.keys())
    summary = {}
    for pname in perturbations:
        system_prompt = PERTURBATIONS[pname](ORIGINAL_SYSTEM_PROMPT)
        print(f"\n=== Perturbation: {pname} ===")
        summary[pname] = {}
        for model in COHORT_MODELS:
            model_safe = model.replace("/", "__").replace(":", "_")
            cycles = {"n": 0, "correct": 0}
            for seed in seeds:
                seed_dir = OUT_DIR / f"{pname}_seed{seed}" / model_safe
                seed_dir.mkdir(parents=True, exist_ok=True)
                for sc in scenarios:
                    sid = sc.get("scenario_id") or sc.get("id")
                    out_file = seed_dir / f"{sid}.json"
                    if out_file.exists(): continue
                    user_prompt = f"USER REQUEST:\n{sc.get('query') or sc.get('user_request', '')}\n\nPick exactly one tool from the enum above and return the JSON object."
                    text, lat, err = call_ollama(model, system_prompt, user_prompt, seed=seed)
                    tool = parse_tool(text)
                    gt = sc.get("gt_tool") or sc.get("ground_truth_tool")
                    rec = {"scenario_id": sid, "perturbation": pname, "model": model, "seed": seed,
                           "agent_tool": tool, "gt_tool": gt, "raw_response": text, "latency": lat, "error": err}
                    out_file.write_text(json.dumps(rec, indent=2))
                    cycles["n"] += 1
                    if tool == gt: cycles["correct"] += 1
                    if cycles["n"] % 50 == 0:
                        print(f"  {pname}/{model_safe}/seed{seed}: {cycles['n']} done, acc={cycles['correct']/cycles['n']:.3f}")
            summary[pname][model] = cycles
            print(f"  Final {model}: {cycles['correct']}/{cycles['n']} = {cycles['correct']/max(1,cycles['n']):.3f}")
    out_path = OUT_DIR / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_path}")

if __name__ == "__main__":
    main()
