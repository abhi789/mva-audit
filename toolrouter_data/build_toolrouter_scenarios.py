"""
Build the 50-scenario tool-router dataset.

K=6 tool enum: {calculator, web_search, calendar, email, file_search, code_execution}.

Two strata:
  - UNAMBIGUOUS (n=30): 5 scenarios per tool. Each scenario is a natural-language
    user query where exactly one tool is clearly correct under the tool descriptions.
    GT = the obvious tool.
  - AMBIGUOUS (n=20): scenarios where two tools are both plausible (e.g., "what's
    2+2 — can you look it up?" -> calculator OR web_search; "find my notes on X" ->
    file_search OR email). GT = the "primary" tool by the tool-description rule:
       calculator beats web_search for arithmetic-only;
       file_search beats email for "find/locate" intent;
       calendar beats email for "schedule/when/meeting" intent;
       email beats calendar for "send/reply" intent;
       code_execution beats calculator for multi-step logic / scripting;
       web_search beats every other tool when the answer is on the public web
         and not derivable from local resources.

The point of the ambiguous stratum is to STRESS-TEST whether models lock to
one tool (engagement-default) or distribute across plausible candidates.
A model that emits the same tool on 90% of ambiguous scenarios is exhibiting
the engagement-default failure mode.
"""
import json
from pathlib import Path

BASE = Path(__file__).parent
OUT = BASE / "scenarios"
OUT.mkdir(exist_ok=True)

TOOL_ENUM = ["calculator", "web_search", "calendar", "email", "file_search", "code_execution"]

# 5 unambiguous per tool = 30 scenarios
UNAMBIGUOUS = [
    # calculator (5)
    ("tr_001", "calculator", "What is 1247 multiplied by 38?"),
    ("tr_002", "calculator", "Compute the square root of 7921."),
    ("tr_003", "calculator", "If I invest $5000 at 4.5% annual interest, what's the simple interest after 3 years?"),
    ("tr_004", "calculator", "What is 15% of 280?"),
    ("tr_005", "calculator", "Convert 73 degrees Fahrenheit to Celsius."),
    # web_search (5)
    ("tr_006", "web_search", "Who won the 2024 Nobel Prize in Physics?"),
    ("tr_007", "web_search", "What is the current population of Tokyo?"),
    ("tr_008", "web_search", "When did the iPhone 17 launch?"),
    ("tr_009", "web_search", "Find me reviews of the restaurant Eleven Madison Park."),
    ("tr_010", "web_search", "What's the weather forecast for Mumbai tomorrow?"),
    # calendar (5)
    ("tr_011", "calendar", "Schedule a 30-minute meeting with Priya on Thursday at 2pm."),
    ("tr_012", "calendar", "What's on my calendar for next Tuesday?"),
    ("tr_013", "calendar", "Block off my afternoon for deep work on Friday."),
    ("tr_014", "calendar", "When is my next dentist appointment?"),
    ("tr_015", "calendar", "Move my 3pm meeting to 4pm."),
    # email (5)
    ("tr_016", "email", "Send a reply to Marcus thanking him for the proposal."),
    ("tr_017", "email", "Draft an email to the team announcing the new policy."),
    ("tr_018", "email", "Forward the contract to legal@company.com."),
    ("tr_019", "email", "Send a follow-up to the candidate from yesterday's interview."),
    ("tr_020", "email", "Reply to Sarah's email about the budget revision."),
    # file_search (5)
    ("tr_021", "file_search", "Find the Q3 financial report I saved last month."),
    ("tr_022", "file_search", "Locate all PDF files I downloaded from the conference."),
    ("tr_023", "file_search", "Open my notes from the design review meeting."),
    ("tr_024", "file_search", "Search my documents folder for the presentation about onboarding."),
    ("tr_025", "file_search", "Find the spreadsheet with last year's expenses."),
    # code_execution (5)
    ("tr_026", "code_execution", "Run this Python script that processes the CSV: `import pandas as pd; df = pd.read_csv('data.csv'); print(df.describe())`."),
    ("tr_027", "code_execution", "Execute the following SQL query against the database: SELECT user_id, COUNT(*) FROM events GROUP BY user_id LIMIT 10."),
    ("tr_028", "code_execution", "Test this regex against the strings I'll send next: `^[A-Z]{2}\\d{4}$`."),
    ("tr_029", "code_execution", "Plot a histogram of the values [1,1,2,3,3,3,4,5,5,5,5,6] using matplotlib."),
    ("tr_030", "code_execution", "Run a Fibonacci function in Python that returns the 30th term."),
]

# 20 ambiguous scenarios — two tools plausible; GT = primary per the rule above
AMBIGUOUS = [
    ("tr_031", "calculator",       "What's 2+2 — can you look that up for me?",
        "calculator beats web_search for arithmetic-only"),
    ("tr_032", "calculator",       "Convert 100 USD to EUR at current rates — I just need the number.",
        "Arithmetic conversion with implied current-rate; calculator if rate is known, web_search otherwise. Tightly arithmetic intent."),
    ("tr_033", "code_execution",   "Multiply these two matrices for me.",
        "code_execution beats calculator for multi-element math"),
    ("tr_034", "code_execution",   "Compute the mean and standard deviation of these 50 numbers.",
        "code_execution preferred for statistical reduction over a long list"),
    ("tr_035", "file_search",      "Find my notes on the kickoff meeting.",
        "file_search beats email for 'find/locate' intent on documents"),
    ("tr_036", "email",            "Reply to Jane with the file I just found.",
        "email beats file_search for 'reply' intent"),
    ("tr_037", "calendar",         "When is the meeting with Acme — and email me a reminder.",
        "calendar primary (lookup); email is a follow-up step"),
    ("tr_038", "email",            "Send the schedule to the team for the new quarter.",
        "email beats calendar for 'send' intent; the calendar object is content"),
    ("tr_039", "web_search",       "What did the CEO say in the latest earnings call?",
        "web_search beats file_search; public-web information"),
    ("tr_040", "file_search",      "I had a doc explaining this — can you pull it up?",
        "file_search beats web_search; 'I had a doc' implies local"),
    ("tr_041", "web_search",       "What's the latest version of React?",
        "web_search beats code_execution; lookup not execution"),
    ("tr_042", "code_execution",   "Tell me what `np.dot([1,2,3], [4,5,6])` evaluates to.",
        "code_execution beats web_search; the answer is reproducible by execution"),
    ("tr_043", "calendar",         "Am I free Thursday afternoon for a coffee with Sam?",
        "calendar beats email; calendar lookup, not draft"),
    ("tr_044", "email",            "Tell Sam I'll meet them at 3pm Thursday.",
        "email beats calendar; communication intent"),
    ("tr_045", "file_search",      "Find my response to the customer's complaint from last week.",
        "file_search beats email; 'find my response' implies retrieval"),
    ("tr_046", "email",            "Resend the response I drafted last week to the customer.",
        "email beats file_search; 'resend' is communication intent"),
    ("tr_047", "calculator",       "What is 17 squared plus 23 squared?",
        "calculator beats code_execution; simple arithmetic"),
    ("tr_048", "code_execution",   "Run a script that computes 17 squared plus 23 squared and writes the result to a file.",
        "code_execution beats calculator; multi-step including I/O"),
    ("tr_049", "web_search",       "Look up the current ICU mortality rate in the US.",
        "web_search beats file_search; statistic from the web"),
    ("tr_050", "file_search",      "Find the email Marcus sent me about the contract.",
        "file_search beats email; 'find' beats 'send' in email-search vs email-compose distinction. (Note: email-search would be a 7th tool we don't have; file_search is the catch-all retrieval tool for our 6-enum.)"),
]

scenarios = []
for sid, gt, query in UNAMBIGUOUS:
    scenarios.append({
        "scenario_id": sid,
        "stratum": "unambiguous",
        "ground_truth_tool": gt,
        "query": query,
    })
for entry in AMBIGUOUS:
    sid, gt, query, rationale = entry
    scenarios.append({
        "scenario_id": sid,
        "stratum": "ambiguous",
        "ground_truth_tool": gt,
        "query": query,
        "gt_rationale": rationale,
    })

# Save per-scenario JSON
for sc in scenarios:
    with open(OUT / f"{sc['scenario_id']}.json", "w") as f:
        json.dump(sc, f, indent=2)

# Save manifest with class counts
from collections import Counter
gt_dist = Counter(sc["ground_truth_tool"] for sc in scenarios)
stratum_dist = Counter(sc["stratum"] for sc in scenarios)
manifest = {
    "n_total": len(scenarios),
    "tool_enum": TOOL_ENUM,
    "stratum_counts": dict(stratum_dist),
    "gt_distribution": dict(gt_dist),
    "scenario_ids": [sc["scenario_id"] for sc in scenarios],
}
with open(BASE / "toolrouter_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

print(f"Wrote {len(scenarios)} scenarios to {OUT}")
print(f"GT distribution: {dict(gt_dist)}")
print(f"Stratum distribution: {dict(stratum_dist)}")
