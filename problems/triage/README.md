# Triage (MIMIC-IV-ED, ESI 1–5)

Code for the triage experiments in the paper. Seeds, validator, data preparation
and the ESI reference KB are unchanged from the first release; this update adds
the helper used for all reported runs, the selected finalists, and the
evaluation/analysis scripts.

## Task model

Every candidate program calls the task model only through `helper.py`.
Configure it with environment variables (or a `.env` file):

| Variable | Meaning |
|---|---|
| `TRIAGE_ESI_INFERENCE_BASE_URL` (or `TRIAGE_LLM_BASE_URL`) | OpenAI-compatible endpoint |
| `TRIAGE_ESI_API_KEY` (or `OPENAI_API_KEY`) | API key |
| `TRIAGE_ESI_FORCE_MODEL` | model used for every call, overriding `models.json` |
| `TRIAGE_ESI_LLM_MAX_CONCURRENCY` | parallel requests per event loop (default 8) |
| `TRIAGE_ESI_LLM_MAX_RETRIES` | retries on 429/5xx/timeouts (default 6) |
| `TRIAGE_ESI_LLM_STRICT` | `1` raises `LLMRequestError` instead of returning an empty answer |
| `TRIAGE_ESI_TOKEN_USAGE_PATH` | optional JSONL log of per-call token usage |

Paper runs used `openai/gpt-oss-120b` and GPT-5.6 Luna (`openai/gpt-5.6-luna`
via OpenRouter); either list the model in `models.json` or set
`TRIAGE_ESI_FORCE_MODEL`. Calls use temperature 0.

## Data

MIMIC-IV-ED requires PhysioNet credentialed access. Build the splits with
`data/prepare_mimic_esi.py`:

| Dataset name | File | Use |
|---|---|---|
| `mimic_esi_train` | `mimic_esi_train.jsonl` (25,330) | evolution; class-balanced batches of 96 per evaluation |
| `mimic_esi_train_evolution` | `mimic_esi_train_evolution.jsonl` (460, balanced) | audit: selects the baseline and finalists |
| `mimic_esi_article_like_test` | `mimic_esi_article_like_test.jsonl` (2,000, natural distribution) | held-out test, reported results only |
 No patient data is stored in
this repository.

## Finalists

`finalists/` holds the programs reported in the paper (each is a standalone
`entrypoint(patients)` that imports `helper`):

| File | Task model | Search |
|---|---|---|
| `gptoss_018_0.6193_3222dd01.py` | gpt-oss-120b | single search |
| `luna_run1.py`, `luna_run2.py`, `luna_run3.py` | GPT-5.6 Luna | three independent searches |

The baseline (`Base`) is the best seed on the audit split,
`initial_programs/article_prompt_clinical.py`.

## Reproducing the test-set tables

From `run_scripts/triage/`:

```bash
# per-case predictions + aggregate metrics (fitness, exact, range, under/over/severe undertriage, ESI-1 recall)
python evaluate_programs.py --program-dir ../../problems/triage/finalists \
  --dataset mimic_esi_article_like_test --repeats 1 --output-dir out/finalists
python evaluate_programs.py --program-dir ../../problems/triage/initial_programs \
  --include-stem article_prompt_clinical --dataset mimic_esi_article_like_test --repeats 1 --output-dir out/base

# weighted kappa, high-acuity sensitivity, confusion matrices, paired bootstrap vs. the first --pred
python analyze_predictions.py --bootstrap 5000 \
  --pred base=out/base/predictions/article_prompt_clinical_r1.jsonl \
  --pred run1=out/finalists/predictions/luna_run1_r1.jsonl \
  --out out/analysis.json
```

`export_top_programs.py` exports the top candidates from a GigaEvo Redis
database (requires `gigaevo-core`). Its `--max-generation` filter uses GigaEvo's
`Program.generation`, which is lineage depth rather than the engine iteration.
