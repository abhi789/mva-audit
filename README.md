# The Minimum Viable Audit

Code, prompts, rule mappings, the authored router task and aggregate results for

> **The Minimum Viable Audit: Action Defaults and Prompt Copying in LLM Decision Agents.**
> Abhineet Rajendra Kulkarni, Pranav Madhav Kuber and Anurag Bihani. AACL-IJCNLP 2026.

The Minimum Viable Audit (MVA) checks a fixed-menu decision agent beyond accuracy. It looks at:
- output validity;
- the distribution of valid actions: count *k*, entropy *H* and dominance *d*;
- recall for each class;
- copied prompt content;

and then tests the same checkpoint on an unrelated structured task. `src/mva_audit.py` implements the measures and the concentration flag. A condition is flagged when *M* ≥ 2 of these four criteria hold: *k* ≤ 2, *H* ≤ 1 bit, *d* ≥ 0.80, zero recall on a supported class.

## What is included

| Path | Contents |
|---|---|
| `src/mva_audit.py` | MVA measures, the concentration flag and scenario-cluster bootstrap intervals, recomputed from saved runs |
| `src/run_agent_toolrouter.py` | Non-medical six-tool router (Section 3.1) |
| `src/run_v16_A_format_cohort.py` | Router format perturbations: bullet, header, reversed tool order (Section 4.4) |
| `src/run_agent_prompt9_icu.py` | ICU monitoring task, baseline prompt |
| `src/run_agent_prompt9.py`, `src/run_agent_prompt9_medical.py`, `src/ergonomic_simulator.py` | Wearable fatigue task (general-purpose and medical model runners) |
| `src/run_agent_prompt9_cot_icu.py`, `src/run_agent_prompt9_cot.py` | Joint reasoning-field + schema intervention (Section 5.2) |
| `src/run_agent_bfcl.py`, `src/run_agent_jsonschemabench.py`, `src/analyze_*.py` | Public-subset adaptations and copying detectors (Appendix B) |
| `src/extract_icu_scenarios.py`, `src/extract_icu_scenarios_mimic.py`, `src/load_*.py` | ICU scenario extraction and reference rules, `gt_icu_v1` (Appendix A) |
| `src/score_agent_v2.py` | Wearable reference rules, `gt_action_v2` (Appendix A) |
| `src/compare_vs_baselines_icu.py`, `src/build_*_baseline*.py`, `src/stats_tools.py` | NEWS2, MLP and XGBoost baselines (Appendix C) |
| `toolrouter_data/` | The 50 authored router requests and their reference tools |
| `bfcl_data/`, `jsonschema_bench/` | Sampling scripts and the IDs of the 100 sampled instances |
| `prompts/` | ICU and router system prompts as evaluated |
| `results/` | Aggregate results: per-condition counts, MVA measures, baselines, paired router statistics and threshold sensitivity |

## What is not included

- **ICU scenarios and model outputs.** The ICU scenarios come from eICU-CRD v2.0 and MIMIC-IV v3.1, and model outputs on them contain record-level clinical values. Under the PhysioNet data use agreements they are not redistributed. The extraction code regenerates them for credentialed users, and derived record-level outputs are available on request to credentialed PhysioNet users.
- **Wearable data.** The wearable scenarios are derived from the human-subjects study of Kuber et al. (2024) and are not redistributed.
- **Public benchmark content.** Download BFCL v3 and JSONSchemaBench from their original sources and rebuild the samples with the included scripts. Only the sampled instance IDs are included.

## Setup

- Python 3.9+ and [Ollama](https://github.com/ollama/ollama). All inference runs locally through Ollama's HTTP API at `localhost:11434`.
- Install the Python dependencies:

```sh
pip install -r requirements.txt
```

The models, with their Ollama tags (paper Appendix H):

| Model | Ollama tag |
|---|---|
| MedGemma 1.5 4B Q8 / Q4 | `medgemma1.5:4b-it-q8_0` / `medgemma1.5:4b-it-q4_K_M` |
| BioMistral 7B | `adrienbrault/biomistral-7b:Q4_K_M` |
| Meditron 7B Q4 / Q8 | `meditron:7b-q4_K_M` / `meditron:7b-q8_0` |
| Mistral Instruct v0.2 | `mistral:7b-instruct-v0.2-q4_K_M` |
| Llama-2 Chat | `llama2:7b-chat-q4_K_M` |
| Gemma 3n E2B / E4B | `gemma3n:e2b` / `gemma3n:e4b` |
| Phi-4-mini | `phi4-mini:latest` |
| OpenBioLLM 8B | `koesn/llama3-openbiollm-8b:Q4_K_M` |
| Aloe-Qwen 7B | `hf.co/mradermacher/Qwen2.5-Aloe-Beta-7B-i1-GGUF:Q4_K_M` |
| Meditron3 8B | `hf.co/mradermacher/Meditron3-8B-GGUF:Q4_K_M` |

Runs use temperature 0.2 and seeds 0, 1 and 2. The historical runs used Ollama 0.6.x; the Meditron Q8 format extension used Ollama 0.20.7. Runtime versions can affect outputs (paper Appendix D).

## Running

All scripts run from the repository root and write to `agent_runs_*` directories there. Runners are configured through environment variables:

| Variable | Sets |
|---|---|
| `AGENT_MODELS` | Comma-separated Ollama tags |
| `AGENT_SEED` | Seed |
| `AGENT_OUT_SUFFIX` | Output-directory suffix |
| `AGENT_DATA_DIR` | Scenario directory |
| `AGENT_SCENARIOS` | Comma-separated scenario IDs |

**Router** (no restricted data needed):

```sh
for s in 0 1 2; do
  AGENT_SEED=$s AGENT_OUT_SUFFIX="_seed$s" python src/run_agent_toolrouter.py
done
```

**Format perturbations.** This example reproduces the Meditron-7B Q8 grid, including the reversed tool order:

```sh
AGENT_MODELS=meditron:7b-q8_0 python src/run_v16_A_format_cohort.py
```

**Public subsets.** Place `BFCL_v3_simple.json` and `BFCL_v3_simple_answers.json` in `bfcl_data/`, and the JSONSchemaBench test parquet files in `jsonschema_bench/`. Then:

```sh
python bfcl_data/build_sample.py
python jsonschema_bench/build_sample.py
for s in 0 1 2; do
  AGENT_SEED=$s AGENT_OUT_SUFFIX="_seed$s" python src/run_agent_bfcl.py
  AGENT_SEED=$s AGENT_OUT_SUFFIX="_seed$s" python src/run_agent_jsonschemabench.py
done
```

Check that the rebuilt samples match `sample_ids.json`.

**ICU** (requires credentialed PhysioNet access to eICU-CRD v2.0):

1. Place the eICU-CRD CSV files in `raw_data/eicu-collaborative-research-database-2.0/`.
2. Build the local database and extract the scenarios:

```sh
python src/load_eicu_to_sqlite.py
python src/load_vitalperiodic.py
python src/extract_icu_scenarios.py              # writes icu_data_v1/
```

3. Run the models:

```sh
for s in 0 1 2; do
  AGENT_SEED=$s AGENT_OUT_SUFFIX="_seed$s" python src/run_agent_prompt9_icu.py
done
```

For MIMIC-IV v3.1, place the files in `~/mimic-iv/`, run `src/extract_icu_scenarios_mimic.py`, then run the ICU runner with `AGENT_DATA_DIR=icu_data_mimic_v1` and `AGENT_OUT_SUFFIX=_mimic_seed<s>`.

**Baselines:**

```sh
python src/build_mlp_baseline.py
python src/build_xgboost_baseline_taskB.py
python src/compare_vs_baselines_icu.py
```

**Audit:**

```sh
python src/mva_audit.py
```

This writes `results/audited_results_local.json`, which can be compared with the published `results/audited_results.json`. Missing run directories are skipped. Per-record rows, which include scenario identifiers, are written to the git-ignored `runs_index/`.

Bootstrap intervals share one random-number stream across conditions. An audit of a different set of conditions can therefore shift interval endpoints slightly; point estimates are unaffected.

## Results files

`results/audited_results.json` holds the audited metrics behind the paper's tables. Keys are `<condition>/<model>`, for example `icu/medgemma1.5_4b-it-q8_0` or `format_reverse/meditron_7b-q4_K_M`. Each entry has:
- the attempt count, accuracy, valid and invalid counts, and invalid-response categories;
- action and reference counts, recall per class;
- *k*, *H*, *d*, the four criteria and *M*;
- per-seed metrics;
- where applicable, scenario-cluster bootstrap intervals.

## Citation

```bibtex
@inproceedings{kulkarni2026mva,
  title     = {The Minimum Viable Audit: Action Defaults and Prompt Copying in {LLM} Decision Agents},
  author    = {Kulkarni, Abhineet Rajendra and Kuber, Pranav Madhav and Bihani, Anurag},
  booktitle = {Proceedings of AACL-IJCNLP 2026},
  year      = {2026}
}
```

## Data and use

- **ICU data:** eICU-CRD (Pollard et al., 2018) and MIMIC-IV (Johnson et al., 2023) are available through [PhysioNet](https://physionet.org) under credentialed access.
- **Research use only.** This code is for research evaluation and is not a clinical decision tool.
