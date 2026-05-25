# med_pneu_v12_27b_r128_iv README

Prompt-evolution task for binary chest X-ray classification with MedGemma VLM.

The experiment evaluates prompts on `problems/med_pneu_v12_27b_r128_iv/data/pneumoniamnist_subset`:

- `0`: normal
- `1`: pneumonia

Candidate programs in `initial_programs/*.py` should only return a prompt string from `entrypoint()`. The evaluator fixes model/runtime settings in `helper.py` and environment variables.

## Ubuntu Environment

Run from the repository root.

```bash
cd /path/to/evolve
conda env create -f problems/med_pneu_v12_27b_r128_iv/environment.yml
conda activate EvolvePy312
```

If the environment already exists:

```bash
conda env update -n EvolvePy312 -f problems/med_pneu_v12_27b_r128_iv/environment.yml --prune
conda activate EvolvePy312
```

The quick tests rely mainly on `openai`, `httpx`, `pillow`, `numpy`, `pyyaml`, and the local `gigaevo` package. The task-local `environment.yml` is the preferred reproducible setup.

## Credentials

Preferred local option: write a fresh key to task-local `kv.py`.

```bash
cat > problems/med_pneu_v12_27b_r128_iv/kv.py <<'PY'
base_url = "https://inference.airi.net:46783/v1"
api_key = "PASTE_FRESH_KEY_HERE"
PY
```

Alternative via environment variables:

```bash
export MEDGEMMA_BASE_URL='https://inference.airi.net:46783/v1'
export MEDGEMMA_API_KEY='PASTE_FRESH_KEY_HERE'
```

OpenAI-compatible names also work:

```bash
export OPENAI_BASE_URL='https://inference.airi.net:46783/v1'
export OPENAI_API_KEY='PASTE_FRESH_KEY_HERE'
```

## Models

Default 27B multimodal model:

```bash
export MEDGEMMA_MODEL='Google/Medgemma-27b-it'
```

4B compatibility check:

```bash
export MEDGEMMA_MODEL='Google/Medgemma-1.5-4b-it'
```

Use multimodal models only. Do not use `27b-text-it` for this task because images are part of the input.

Both 27B and 4B use the same image input size in this task:

```bash
export MEDGEMMA_IMAGE_SIZE='896'
```

`MEDGEMMA_RESIZE` is also supported, but prefer `MEDGEMMA_IMAGE_SIZE` for clarity.

## Smoke Tests

Fast data check:

```bash
python problems/med_pneu_v12_27b_r128_iv/debug_run_initials.py --quick
```

Expected: dataset exists, `label_map.json` exists, 100 rows detected, sampled files present.

One VLM request per initial prompt with 27B:

```bash
export MEDGEMMA_MODEL='Google/Medgemma-27b-it'
export MEDGEMMA_IMAGE_SIZE='896'
python problems/med_pneu_v12_27b_r128_iv/debug_run_initials.py --smoke-vlm
```

One VLM request per initial prompt with 4B:

```bash
export MEDGEMMA_MODEL='Google/Medgemma-1.5-4b-it'
export MEDGEMMA_IMAGE_SIZE='896'
python problems/med_pneu_v12_27b_r128_iv/debug_run_initials.py --smoke-vlm
```

Short validation run over the first few images:

```bash
export PNEUMONIA_NUM_IMAGES='4'
export MEDGEMMA_MODEL='Google/Medgemma-27b-it'
export MEDGEMMA_IMAGE_SIZE='896'
python problems/med_pneu_v12_27b_r128_iv/debug_run_initials.py
unset PNEUMONIA_NUM_IMAGES
```

## Full Initials Run

```bash
export MEDGEMMA_MODEL='Google/Medgemma-27b-it'
export MEDGEMMA_IMAGE_SIZE='896'
python problems/med_pneu_v12_27b_r128_iv/debug_run_initials.py
```

Generated artifacts:

- `problems/med_pneu_v12_27b_r128_iv/data/pneumoniamnist_subset/predictions_latest.jsonl`
- `problems/med_pneu_v12_27b_r128_iv/data/pneumoniamnist_subset/metrics_latest.json`
- `problems/med_pneu_v12_27b_r128_iv/data/pneumoniamnist_subset/last_validator_error.txt`

## GigaEvo Run Sketch

Use the normal repo entrypoint and select this problem. Exact overrides depend on the active experiment config, but the important part is to pin the model and image size in the process environment:

```bash
export MEDGEMMA_MODEL='Google/Medgemma-27b-it'
export MEDGEMMA_IMAGE_SIZE='896'
python run.py problem.name=med_pneu_v12_27b_r128_iv
```

Before long runs, always confirm:

- `--quick` passes;
- `--smoke-vlm` returns model responses, not credential errors;
- the model printed by the smoke script is the intended 27B or 4B multimodal model;
- `metrics_latest.json` records `"resize": 896`.
