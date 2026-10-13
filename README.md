# LLM-Guided Evolution for Medical Decision Pipelines

Three [GigaEvo](https://github.com/FusionBrainLab/gigaevo-core) tasks applying MAP-Elites–based LLM program evolution to medical decision-making.

## Repository layout

```
problems/          task definitions (helper, validate, context, metrics, initial programs)
  medical_dialogue/   differential diagnosis via multi-turn patient dialogue
  image_classification/  chest X-ray classification (PneumoniaMNIST, MedGemma 1.5 4B / 27B)
  triage/             emergency triage ESI-level prediction (MIMIC-IV derived)

run_scripts/       execution scripts and algorithm configs
  medical_dialogue/   run.sh  +  config/algorithm/single_island_2d.yaml
  image_classification/  setup_ubuntu.sh  +  run_experiment.py  +  run_matrix.py
  triage/             config/algorithm/triage_esi.yaml  (two-island MAP-Elites)

shared/
  batch_composer.py   adaptive anchor + hard-buffer + random batch sampling
  lineage_fitness.py  exponentially weighted ancestor fitness blending
```

## Tasks

### medical_dialogue
Evolves a decision policy for a doctor agent that asks clarifying questions and then diagnoses. Fitness = fraction of correct final diagnoses. Secondary behavior dimension: average doctor token count (encourages concise consultations). Two-island MAP-Elites with `single_island_2d` config.

**Requirements:** two OpenAI-compatible endpoints (evolution LLM + doctor/patient LLMs), HuggingFace token for tokenizer, [MedIQ dataset](https://github.com/stellalisy/mediQ).

### image_classification
Evolves instruction prompts and decoding settings for MedGemma 1.5 4B and MedGemma 27B to classify PneumoniaMNIST chest X-rays as normal or pneumonia. The experiment covers 28, 64, 128, and 224 pixel inputs, a fixed balanced 100-image training subset, and the 624-image held-out test split. Fitness is accuracy on the balanced training subset, where it equals balanced accuracy. Search uses single-island MAP-Elites.

**Requirements:** Ubuntu 24.04 / Python 3.12 and OpenAI-compatible endpoints for the selected MedGemma evaluator and GPT-OSS-120B mutator. The setup script obtains the pinned GigaEvo engine; the data preparation script downloads and verifies the official dataset files. Dataset images and model weights are obtained separately. This task uses disk storage and does not require a Redis service.

See the [PneumoniaMNIST guide](problems/image_classification/README.md) for Ubuntu setup, data preparation, credentials, search, resume, evaluation, and dataset attribution.

### triage
Evolves an ESI-level (1–5) triage scoring program. Fitness = `0.65 × exact_accuracy + 0.25 × range_accuracy + 0.10 × recall_ESI1`. Two-island MAP-Elites: `exact_range_island` (exact × range) and `fitness_safety_island` (fitness × ESI-1 recall).

**Requirements:** OpenAI-compatible LLM endpoint (`TRIAGE_LLM_BASE_URL`, `TRIAGE_MODEL_NAME`), MIMIC-IV ESI data (PhysioNet credentials required — run `data/prepare_mimic_esi.py`).

## Quick start

1. Clone [gigaevo-core](https://github.com/FusionBrainLab/gigaevo-core) as a sibling directory.
2. Start Redis locally.
3. Configure env vars in the relevant `run_scripts/` entry point and run it.

For **PneumoniaMNIST / MedGemma**, follow the [task-specific Ubuntu quick start](problems/image_classification/README.md#ubuntu-quick-start). It installs the required engine version and dependencies.

## Shared utilities

`BatchComposer` keeps evaluation noise low by mixing a fixed anchor set, a hard-buffer of high-error cases, and fresh random samples each generation. `lineage_fitness.py` blends raw fitness with exponentially weighted ancestor scores to detect anomalous drops.
