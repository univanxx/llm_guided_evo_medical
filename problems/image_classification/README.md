# MedGemma pipeline evolution on PneumoniaMNIST

This experiment uses an evolutionary search to optimize the instruction prompt and decoding settings of a medical vision-language model. Each candidate is a Python program that returns a prompt and sampling parameters. Model weights remain fixed. The task is binary chest X-ray classification: **0 = normal**, **1 = pneumonia**.

The experiment covers [MedGemma 1.5 4B IT](https://huggingface.co/google/medgemma-1.5-4b-it) and [MedGemma 27B IT](https://huggingface.co/google/medgemma-27b-it), each at input resolutions **28, 64, 128, and 224**. The 27B model must be the multimodal variant. A separate GPT-OSS-120B endpoint proposes candidate programs. Both endpoints use the OpenAI-compatible chat-completions protocol.

## Data

[PneumoniaMNIST](https://medmnist.com/) contains grayscale pediatric chest X-rays. Its public split has 4,708 training, 524 validation, and 624 test images. This experiment uses a fixed balanced subset of **100 training images (50 per class)** and all **624 test images (234 normal, 390 pneumonia)**. Validation images are not used by the search. See the [dataset release](https://zenodo.org/records/10519652) for download and licensing information.

The same image indices are used for both models and all four resolutions. `manifests/` defines the selected indices, labels, and image checksums. `prepare_data.py` downloads the resolution-specific arrays and verifies selected image pixels before producing local PNGs. Images and model weights are obtained separately. Before inference, each image is converted to RGB and resized with bicubic interpolation to a 896 × 896 canvas.

PneumoniaMNIST images use CC BY 4.0. See [dataset attribution and references](DATA_ATTRIBUTION.md) for source credits, licensing, and the transformations performed by this experiment.

## Search and evaluation

The search uses the pinned [GigaEvo engine](https://github.com/FusionBrainLab/gigaevo-core/tree/2fccecc62481b4b9c5d4be9b2f1a6d2cf84769e4) with a single MAP-Elites island, a one-dimensional fitness archive with 150 bins, and a maximum archive size of 75. Ten initial programs are followed by 232 proposed descendants. Search seeds 43, 44, and 45 define three independent searches per model/resolution pair. Parent evaluations are refreshed during selection.

Candidates optimize the prompt together with `temperature` in `{0, 0.1, 0.3, 0.6}` and `top_p` in `{0.9, 1.0}`. Temperature zero is normalized to `top_p=1`. Model, resolution, preprocessing, and the 256-token response limit are fixed outside candidate programs. The mutator uses temperature 0.6 and an 8,192-token limit.

Fitness is accuracy on the balanced 100-image training subset, where it equals balanced accuracy. The validator accepts a JSON object containing matching `class_id` and `class_name` values. Invalid answers count as errors of their true class. The best initial candidate and best descendant are selected independently using training fitness, then each is evaluated three times on test. Test responses are never used for selection. Per-pass outputs include accuracy, macro-F1, sensitivity, specificity, and invalid-answer counts; the summary command also calculates pneumonia F1 and balanced accuracy from saved predictions.

## Ubuntu quick start

Use Ubuntu 24.04 with Python 3.12. Commands below run from the repository root. API access to the two model endpoints is required for actual inference; the experiment client itself does not require a GPU.

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv screen
bash run_scripts/image_classification/setup_ubuntu.sh
source run_scripts/image_classification/.venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1

python -B run_scripts/image_classification/prepare_data.py --download
python -B -m pytest run_scripts/image_classification/tests -q -p no:cacheprovider
```

The setup script downloads the pinned engine into `../gigaevo-core` and creates a task-specific virtual environment using the versions in `constraints-ubuntu-py312.txt`. Set `GIGAEVO_DIR` to use another engine location. The dataset is prepared in `problems/image_classification/data`. To prepare only one resolution, add `--resolutions 128` and use a separate `--output` directory. Pass that directory to the launcher with `--data-root`.

To check the entire search and completed-run resume through a local test server, run the following command after preparing the data. It uses deterministic mock responses and does not measure model accuracy or require API credentials:

```bash
python -B run_scripts/image_classification/smoke_offline.py \
  --output-root run_scripts/image_classification/outputs/offline-smoke
```

Configure credentials in the task-local `.env` file:

```bash
cp run_scripts/image_classification/.env.example run_scripts/image_classification/.env
chmod 600 run_scripts/image_classification/.env
# Edit .env and set EVALUATOR_API_KEY and MUTATOR_API_KEY.
```

The endpoint URLs below are examples for local inference servers. Replace them with your serving endpoints. Set `--evaluator-model` and `--mutator-model` when the server exposes different model identifiers. An API server that does not require authentication can use a nonempty placeholder key.

Run a short end-to-end check with one initial candidate, one descendant, two balanced train images, and two test images per pass:

```bash
python -B run_scripts/image_classification/run_experiment.py \
  --model 4b --resolution 128 --smoke \
  --evaluator-base-url http://127.0.0.1:8000/v1 \
  --mutator-base-url http://127.0.0.1:8001/v1 \
  --run-dir run_scripts/image_classification/outputs/smoke-4b-r128
```

Use `--model 27b` for the other model. A successful smoke run ends with `status: completed` in `run_status.json` and creates `paired_evaluation/summary.json`.

## Full experiments

A single full search:

```bash
python -B run_scripts/image_classification/run_experiment.py \
  --model 27b --resolution 128 --seed 43 \
  --evaluator-base-url http://127.0.0.1:8000/v1 \
  --mutator-base-url http://127.0.0.1:8001/v1 \
  --run-dir run_scripts/image_classification/outputs/medgemma-27b-r128-s43
```

The matrix launcher runs 24 searches sequentially: two models × four resolutions × three seeds. Its endpoint must serve the model selected by each request. Preview the commands with `--dry-run`. For server-specific model identifiers, set `--evaluator-model-4b` and `--evaluator-model-27b` on the matrix launcher. Run long jobs in a detached session:

```bash
screen -dmS medgemma-search bash -c '
  source run_scripts/image_classification/.venv/bin/activate
  export PYTHONDONTWRITEBYTECODE=1
  python -B run_scripts/image_classification/run_matrix.py \
    --output-root run_scripts/image_classification/outputs/matrix \
    --evaluator-base-url http://127.0.0.1:8000/v1 \
    --mutator-base-url http://127.0.0.1:8001/v1 \
    > run_scripts/image_classification/matrix.log 2>&1
'
screen -r medgemma-search
```

Detach from the console with `Ctrl-A`, then `D`. To continue an interrupted search, repeat its command with `--resume`. For a matrix, pass `--resume` to the matrix launcher. Completed image responses from the same interrupted evaluation are reused; new evaluation passes receive fresh responses.

## Results

Each run directory contains:

| Path | Contents |
| --- | --- |
| `run_config.json` | Model, resolution, search seed, protocol, and dataset identity |
| `gigaevo_config.yaml` | Effective engine configuration |
| `run_status.json` | Progress and final status |
| `candidates/`, `candidates.json` | Candidate programs, prompt/decoder payloads, and train scores |
| `best_initial/`, `best_evolved/` | Train-selected winners for the two roles |
| `paired_evaluation/selection.json` | Fixed selection and test settings |
| `paired_evaluation/summary.json` | Test results for both selected roles |
| `paired_evaluation/*/final_evaluation/` | Per-image predictions and per-pass metrics |
| `gateway_evaluator/`, `gateway_mutator/` | Request-attempt status and reported token usage |

To select a pair across three searches using train scores, then run its test evaluation:

```bash
python -B run_scripts/image_classification/evaluate.py \
  --evolution-run run_scripts/image_classification/outputs/matrix/medgemma-27b-r128-s43 \
  --evolution-run run_scripts/image_classification/outputs/matrix/medgemma-27b-r128-s44 \
  --evolution-run run_scripts/image_classification/outputs/matrix/medgemma-27b-r128-s45 \
  --final-repeats 5 \
  --run-dir run_scripts/image_classification/outputs/selected-27b-r128
```

Within a search, equal fitness is resolved by the latest stored update counter, then program identifier. Across searches, equal train fitness is resolved by the lower search seed. The command above performs a new test evaluation. Use `--skip-final-evaluation` on the original search commands if only the cross-search pair is needed.

To evaluate one exported candidate directly, use `python -B run_scripts/image_classification/evaluate.py --candidate PATH --model 4b --resolution 128 --split test --repeats 3 --run-dir OUTPUT` with the corresponding endpoint options.

Aggregate the individual search results without additional inference:

```bash
python -B run_scripts/image_classification/summarize_results.py \
  run_scripts/image_classification/outputs/matrix/medgemma-* \
  --output run_scripts/image_classification/outputs/tables
```

`per_search.csv` averages test repetitions within each search. `conditions.csv` reports the mean and sample standard deviation across independent searches for each model, resolution, and role. Metric values are fractions; multiply by 100 for percentages.
