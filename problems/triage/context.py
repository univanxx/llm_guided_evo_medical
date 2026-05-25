import json
import os
import random
import secrets
import sys
from collections import Counter
from pathlib import Path


def _find_data_dir() -> Path:
    env_data_dir = os.environ.get("TRIAGE_ESI_DATA_DIR")
    if env_data_dir:
        return Path(env_data_dir).expanduser()

    candidates = []
    module_file = globals().get("__file__")
    if module_file:
        candidates.append(Path(module_file).resolve().parent / "data")

    cwd = Path.cwd().resolve()
    candidates.extend(
        [
            cwd / "data",
            cwd / "problems" / "triage_esi" / "data",
        ]
    )

    for entry in sys.path:
        if not entry:
            continue
        base = Path(entry).resolve()
        candidates.extend(
            [
                base / "data",
                base / "problems" / "triage_esi" / "data",
            ]
        )

    for candidate in candidates:
        if (
            (candidate / "mimic_esi_train_evolution.jsonl").exists()
            or (candidate / "mimic_esi_article_like_test.jsonl").exists()
        ):
            return candidate

    return Path("data")


DATA_DIR = _find_data_dir()

DATASET_ALIASES = {
    "mimic_esi": DATA_DIR / "mimic_esi_train_evolution.jsonl",
    "mimic_esi_train_evolution": DATA_DIR / "mimic_esi_train_evolution.jsonl",
    "mimic_esi_train": DATA_DIR / "mimic_esi_train.jsonl",
    "mimic_esi_article_candidate_2200": DATA_DIR
    / "mimic_esi_article_candidate_2200.jsonl",
    "mimic_esi_article_like_test": DATA_DIR / "mimic_esi_article_like_test.jsonl",
    "mimic_esi_all": DATA_DIR / "mimic_esi_all.jsonl",
}

DEFAULT_DATASET = "mimic_esi_train"
DEFAULT_K_PER_CLASS = 24


def _resolve_dataset_path(path: str | Path | None = None) -> Path:
    if path is None:
        path = os.environ.get("TRIAGE_ESI_DATASET_PATH") or os.environ.get(
            "TRIAGE_ESI_DATASET", DEFAULT_DATASET
        )
    return DATASET_ALIASES.get(str(path), Path(path)).expanduser()


def _record_to_patient(row: dict) -> dict:
    return {
        "patient": {
            "case_id": row["case_id"],
            "source": row.get("source"),
            "hpi": row["hpi"],
            "patient_info": row["patient_info"],
            "initial_vitals": row.get("initial_vitals", ""),
            "chief_complaint": row.get("chief_complaint", ""),
            "true_acuity": str(row["true_acuity"]),
        }
    }


def _load_jsonl(path: Path) -> list[dict]:
    dataset = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            dataset.append(_record_to_patient(json.loads(line)))
    return dataset


def _sample_k_per_class(dataset: list[dict]) -> list[dict]:
    k_raw = os.environ.get("TRIAGE_ESI_K_PER_CLASS", str(DEFAULT_K_PER_CLASS))
    k = int(k_raw)
    if k <= 0:
        return dataset

    by_label: dict[str, list[dict]] = {}
    for sample in dataset:
        label = sample["patient"]["true_acuity"]
        by_label.setdefault(label, []).append(sample)

    seed_raw = os.environ.get("TRIAGE_ESI_SAMPLE_SEED")
    rng = random.Random(int(seed_raw)) if seed_raw else random.Random(secrets.randbits(64))

    sampled: list[dict] = []
    for label in sorted(by_label):
        items = by_label[label]
        sampled.extend(rng.sample(items, min(k, len(items))))
    rng.shuffle(sampled)
    return sampled


def _maybe_sample(dataset: list[dict], path: Path) -> list[dict]:
    should_stratify = (
        path.name == "mimic_esi_train.jsonl"
        and os.environ.get("TRIAGE_ESI_DISABLE_STRATIFIED_SAMPLING") != "1"
    )
    if should_stratify:
        dataset = _sample_k_per_class(dataset)

    limit_raw = os.environ.get("TRIAGE_ESI_MAX_SAMPLES")
    if not limit_raw:
        return dataset

    limit = int(limit_raw)
    if limit <= 0 or limit >= len(dataset):
        return dataset

    seed = int(os.environ.get("TRIAGE_ESI_SAMPLE_SEED", "20260515"))
    rng = random.Random(seed)
    return rng.sample(dataset, limit)


def load_eval_dataset(path: str | Path | None = None):
    """Load an ESI benchmark split in the patient/true_acuity contract."""
    path = _resolve_dataset_path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. "
            "Run problems/triage_esi/data/prepare_mimic_esi.py first."
        )

    dataset = _maybe_sample(_load_jsonl(path), path)
    counts = Counter(sample["patient"]["true_acuity"] for sample in dataset)
    return dataset, {
        "path": str(path),
        "n": len(dataset),
        "label_counts": dict(sorted(counts.items())),
    }


def build_context() -> list[dict]:
    dataset, _ = load_eval_dataset()
    return [{"patient": sample["patient"]} for sample in dataset]
