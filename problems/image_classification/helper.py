"""Shared candidate contract, fixed experiment settings, data loading, and prediction parsing."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import runpy
import uuid
from pathlib import Path
from typing import Any

from PIL import Image, __version__ as pillow_version, features

TASK_ROOT = Path(__file__).resolve().parent
MODEL_IDS = {"4b": "google/medgemma-1.5-4b-it", "27b": "google/medgemma-27b-it"}
LABELS = {0: "normal", 1: "pneumonia"}
TEMPERATURES = (0.0, 0.1, 0.3, 0.6)
TOP_PS = (0.9, 1.0)
TOP_KS = (-1, 20, 50)


def get_default_settings() -> dict[str, Any]:
    """Read researcher-controlled settings; candidates cannot choose model, data, or limits."""
    model = os.getenv("PNEUMONIA_MODEL", "4b")
    resolution = int(os.getenv("PNEUMONIA_RESOLUTION", "28"))
    if model not in MODEL_IDS or resolution not in (28, 64, 128, 224):
        raise ValueError("PNEUMONIA_MODEL must be 4b or 27b and resolution must be 28, 64, 128, or 224")
    limit = os.getenv("PNEUMONIA_MAX_IMAGES") or os.getenv("PNEUMONIA_NUM_IMAGES")
    seed = os.getenv("PNEUMONIA_SEED")
    data_root = Path(os.getenv("PNEUMONIA_DATA_ROOT", str(TASK_ROOT / "data")))
    settings = {"model_choice": model, "model": os.getenv("MEDGEMMA_MODEL", MODEL_IDS[model]), "resolution": resolution, "resize": 896, "max_tokens": 256, "concurrency": int(os.getenv("MEDGEMMA_CONCURRENCY", "8")), "num_images": int(limit) if limit else None, "repeats": int(os.getenv("PNEUMONIA_REPEATS", "1")), "seed": int(seed) if seed else None, "http_timeout": float(os.getenv("MEDGEMMA_HTTP_TIMEOUT_SECONDS") or os.getenv("PNEUMONIA_REQUEST_TIMEOUT") or "16200"), "base_url": os.getenv("MEDGEMMA_BASE_URL", ""), "api_key": os.getenv("MEDGEMMA_API_KEY", ""), "data_root": str(data_root / f"r{resolution}"), "output_root": os.getenv("PNEUMONIA_RUN_DIR", str(TASK_ROOT.parents[2] / "runs" / "standalone")), "search_space": os.getenv("PNEUMONIA_SEARCH_SPACE", "joint"), "top_k_enabled": os.getenv("PNEUMONIA_ENABLE_TOP_K", "0").lower() in ("1", "true"), "run_id": os.getenv("PNEUMONIA_RUN_ID", "standalone"), "evaluation_instance_id": os.getenv("PNEUMONIA_EVALUATION_INSTANCE_ID") or os.getenv("GIGAEVO_PROGRAM_ID")}
    if settings["concurrency"] < 1 or settings["repeats"] < 1 or settings["http_timeout"] <= 0 or settings["num_images"] is not None and settings["num_images"] < 1:
        raise ValueError("Concurrency, repeats, timeout, and optional image limit must be positive")
    if settings["search_space"] not in ("joint", "prompt-only"):
        raise ValueError("PNEUMONIA_SEARCH_SPACE must be joint or prompt-only")
    return settings


def normalize_program_output(obj: Any, *, top_k_enabled: bool | None = None, search_space: str | None = None) -> dict[str, Any]:
    """Validate the exact search space and collapse all zero-temperature settings to greedy."""
    if top_k_enabled is None:
        top_k_enabled = os.getenv("PNEUMONIA_ENABLE_TOP_K", "0").lower() in ("1", "true")
    if search_space is None:
        search_space = os.getenv("PNEUMONIA_SEARCH_SPACE", "joint")
    if search_space not in ("joint", "prompt-only"):
        raise ValueError("PNEUMONIA_SEARCH_SPACE must be joint or prompt-only")
    if isinstance(obj, str):
        obj = {"prompt": obj}
    if not isinstance(obj, dict) or set(obj) - {"prompt", "sampling"}:
        raise ValueError("Candidate must be a prompt string or an object containing only prompt and sampling")
    prompt = obj.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Candidate prompt must be a nonempty string")
    sampling = obj.get("sampling", {})
    allowed = {"temperature", "top_p", "top_k"} if top_k_enabled else {"temperature", "top_p"}
    if not isinstance(sampling, dict) or set(sampling) - allowed:
        raise ValueError(f"Unsupported sampling keys; allowed keys: {sorted(allowed)}")
    temperature = sampling.get("temperature", 0.0)
    top_p = sampling.get("top_p", 1.0)
    if type(temperature) not in (int, float) or temperature not in TEMPERATURES:
        raise ValueError(f"temperature must be one of {TEMPERATURES}")
    if type(top_p) not in (int, float) or top_p not in TOP_PS:
        raise ValueError(f"top_p must be one of {TOP_PS}")
    top_k = sampling.get("top_k", -1)
    if type(top_k) is not int or top_k not in TOP_KS:
        raise ValueError(f"top_k must be one of {TOP_KS}")
    if search_space == "prompt-only" and (temperature != 0.0 or top_p != 1.0 or top_k != -1):
        raise ValueError("prompt-only control requires temperature=0, top_p=1, and no top_k restriction")
    canonical = {"temperature": float(temperature), "top_p": float(top_p) if temperature else 1.0}
    if top_k_enabled and temperature:
        canonical["top_k"] = top_k
    return {"prompt": prompt, "sampling": canonical}


def candidate_id(candidate: dict[str, Any]) -> str:
    """Hash the canonical prompt and decoder jointly for stable audit identifiers."""
    return hashlib.sha256(json.dumps(candidate, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def load_candidate(path: str | Path) -> dict[str, Any]:
    """Load an explicit trusted Python candidate or an exported JSON configuration."""
    path = Path(path).resolve()
    if path.suffix.lower() == ".json":
        output = json.loads(path.read_text(encoding="utf-8"))
    elif path.suffix.lower() == ".py":
        module = runpy.run_path(str(path))
        output = module["entrypoint"]()
    else:
        raise ValueError("Candidate path must end in .py or .json")
    return normalize_program_output(output)


def load_dataset(subset_dir: str | Path, num_images: int | None = None) -> list[dict[str, Any]]:
    """Load the frozen manifest; small smoke subsets alternate classes deterministically."""
    subset_dir = Path(subset_dir)
    rows = [json.loads(line) for line in (subset_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("Dataset manifest is empty")
    if num_images is not None and num_images < len(rows):
        grouped = [[row for row in rows if row["label_id"] == label] for label in LABELS]
        rows = [group[index] for index in range(max(map(len, grouped))) for group in grouped if index < len(group)][:num_images]
    return rows


def image_data_url(path: str, expected_sha256: str | None = None) -> str:
    """Verify each frozen source and optionally reuse its exact prepared payload across workers."""
    data = Path(path).read_bytes()
    actual_sha256 = hashlib.sha256(data).hexdigest()
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError(f"Frozen image SHA-256 mismatch for {Path(path).name}")
    cache_root = os.getenv("PNEUMONIA_IMAGE_CACHE")
    if not cache_root:
        return _encode_image_data(data)
    transform = {"source_sha256": actual_sha256, "format_version": 1, "resize": 896, "mode": "RGB", "resample": "BICUBIC", "pad": [0, 0, 0], "format": "PNG", "optimize": True, "pillow": pillow_version, "zlib": features.version_codec("zlib")}
    key = hashlib.sha256(json.dumps(transform, sort_keys=True, separators=(",", ":")).encode("ascii")).hexdigest()
    cache = Path(cache_root) / (key + ".json")
    try:
        record = json.loads(cache.read_text(encoding="utf-8"))
        payload = record["payload"]
        if record["key"] == key and isinstance(payload, str) and payload.startswith("data:image/png;base64,") and hashlib.sha256(payload.encode("ascii")).hexdigest() == record["payload_sha256"]:
            return payload
    except (OSError, ValueError, KeyError, TypeError, UnicodeError):
        pass
    payload = _encode_image_data(data)
    record = {"key": key, "payload_sha256": hashlib.sha256(payload.encode("ascii")).hexdigest(), "payload": payload}
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_name(f".{cache.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(record, separators=(",", ":")), encoding="utf-8")
        os.replace(temporary, cache)
    finally:
        temporary.unlink(missing_ok=True)
    return payload


def _encode_image_data(data: bytes) -> str:
    """Apply the unchanged scientific image transform on a cache miss."""
    with Image.open(io.BytesIO(data)) as source:
        image = source.convert("RGB")
    scale = min(896 / image.width, 896 / image.height)
    resized = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), resample=Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (896, 896), (0, 0, 0))
    canvas.paste(resized, ((896 - resized.width) // 2, (896 - resized.height) // 2))
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _unique_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate JSON keys instead of silently selecting the last value."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def extract_json(text: str) -> dict[str, Any]:
    """Extract complete JSON objects, allowing repeated identical objects but no repair."""
    text = text.strip()
    decoder = json.JSONDecoder(object_pairs_hook=_unique_json_keys)
    try:
        obj = decoder.decode(text)
    except json.JSONDecodeError:
        objects = []
        start = text.find("{")
        while start >= 0:
            # An invalid outer object must not expose a valid-looking nested fragment.
            try:
                value, end = decoder.raw_decode(text, start)
            except json.JSONDecodeError as error:
                unfinished = error.pos == len(text) or error.msg.startswith("Unterminated string")
                if not objects or not unfinished:
                    raise
                # Preserve a complete answer already accepted by the legacy parser; never repair its tail.
                break
            objects.append(value)
            start = text.find("{", end)
        if not objects:
            raise ValueError("No JSON object found")
        signatures = {json.dumps(value, sort_keys=True, ensure_ascii=False) for value in objects}
        if len(signatures) != 1:
            raise ValueError("Conflicting JSON objects in model output")
        obj = objects[0]
    if not isinstance(obj, dict):
        raise ValueError("Prediction must be a JSON object")
    return obj


def parse_prediction(text: str) -> dict[str, Any]:
    """Require matching binary labels, normalizing the exact normal alias in the initial prompt."""
    prediction = extract_json(text)
    label = prediction.get("class_id")
    if type(label) is not int or label not in LABELS:
        raise ValueError("class_id must be integer 0 or 1")
    name = prediction.get("class_name")
    if label == 0 and name == "normal (no pneumonia)":
        name = "normal"
    if name != LABELS[label]:
        raise ValueError("class_name must match class_id")
    return {"class_id": label, "class_name": LABELS[label]}
