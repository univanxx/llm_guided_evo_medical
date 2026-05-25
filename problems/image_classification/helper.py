"""med_pneu_v6 helper

Helper utilities for the EV task: MedGemma PneumoniaMNIST (binary CXR classification).

Design goals
------------
- Keep candidate programs *simple*: they mainly mutate a PROMPT string.
- Provide robust I/O: strict JSON parsing, image preprocessing, and OpenAI-compatible calls.
- Make evaluation reproducible and easy to run outside EV.

Credentials
-----------
Provide either:
- env vars: MEDGEMMA_BASE_URL + MEDGEMMA_API_KEY (task-local names)
  or OPENAI_BASE_URL + OPENAI_API_KEY (GigaEvo/LangChain-friendly)
or
- a kv.py next to this file with: base_url = "..."; api_key = "..."

Environment knobs (optional)
----------------------------
These affect evaluation but are NOT allowed to be changed by evolved programs.

- MEDGEMMA_MODEL: model name (default: Google/Medgemma-1.5-4b-it)
- MEDGEMMA_IMAGE_SIZE (or MEDGEMMA_RESIZE): int square size for resize+pad (default: 896; 0 disables)
- MEDGEMMA_TEMPERATURE: float (default: 0.0)
- MEDGEMMA_MAX_TOKENS: int (default: 256)
- MEDGEMMA_CONCURRENCY: int (default: 8)
- MEDGEMMA_INSECURE_SSL: "1" disables TLS cert verification (default: 0)
- PNEUMONIA_SUBSET_DIR: path to subset dir (default: data/pneumoniamnist_subset)
- PNEUMONIA_NUM_IMAGES / MEDGEMMA_NUM_IMAGES: int; how many images to evaluate (default: unset -> ALL)
"""

from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image


# Binary labels for PneumoniaMNIST (MedMNIST).
PNEUMONIA_LABEL_MAP: Dict[int, str] = {0: "normal", 1: "pneumonia"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _parse_optional_int(x: str | None) -> int | None:
    if x is None:
        return None
    x = str(x).strip()
    if not x:
        return None
    try:
        v = int(x)
    except Exception:
        return None
    if v <= 0:
        return None
    return v


def _default_subset_dir() -> str:
    """Default experiment subset: task-local data/pneumoniamnist_subset."""
    task_root = Path(__file__).resolve().parent
    return str((task_root / "data" / "pneumoniamnist_subset").resolve())


# --- Fixed model + vision preprocessing for evaluation ---
#
# We *intentionally* keep these out of candidate programs so the evolutionary loop
# can only mutate the prompt (and not change the underlying model or image resize).
#
# MedGemma 1.5 4B IT expects images normalized to 896×896. (See model card / docs.)


#FIXED_MEDGEMMA_MODEL: str = os.getenv("MEDGEMMA_MODEL", "Google/Medgemma-1.5-4b-it") #!!!
FIXED_MEDGEMMA_MODEL: str = os.getenv("MEDGEMMA_MODEL", "Google/Medgemma-27b-it") #!!!

FIXED_IMAGE_SIZE: int = int(
    os.getenv("MEDGEMMA_IMAGE_SIZE")
    or os.getenv("MEDGEMMA_RESIZE")
    or "896"
)


@dataclass(frozen=True)
class ManifestRow:
    rel_path: str
    label_id: int
    label_name: str
    split: str
    original_index: int


def load_kv_if_present() -> Tuple[str | None, str | None]:
    """Return (base_url, api_key) from env or kv.py.

    Why this exists
    ---------------
    In GigaEvo the validator is executed in a separate process (exec_runner). That
    process does NOT reliably "see" Hydra objects (e.g. llm=airi_med) and therefore
    cannot always access your base_url/api_key from config. So we load credentials
    from:
      1) MEDGEMMA_BASE_URL + MEDGEMMA_API_KEY (task-local)
      2) OPENAI_BASE_URL/OPENAI_API_BASE + OPENAI_API_KEY (common OpenAI-compatible vars)
      3) kv.py next to this task file (task folder)

    Fallbacks
    ---------
    If base_url is missing, we can fall back to MEDGEMMA_FALLBACK_BASE_URL
    (defaults to Airi inference endpoint). If api_key is missing, we fall back to
    MEDGEMMA_FALLBACK_API_KEY (defaults to "DUMMY"), because many OpenAI-compatible
    servers accept any non-empty key.
    """
    env_base = (
        os.getenv("MEDGEMMA_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("OPENAI_API_BASE")
    )
    env_key = os.getenv("MEDGEMMA_API_KEY") or os.getenv("OPENAI_API_KEY")

    fallback_base = os.getenv("MEDGEMMA_FALLBACK_BASE_URL", "")
    fallback_key = os.getenv("MEDGEMMA_FALLBACK_API_KEY", "DUMMY")

    # If at least something is present, use fallbacks for the rest.
    if env_base or env_key:
        return (env_base or fallback_base), (env_key or fallback_key)

    # Finally, try kv.py next to this task (do NOT commit real secrets).
    try:
        from kv import base_url, api_key  # type: ignore

        return str(base_url), str(api_key)
    except Exception:
        return None, None


def get_default_settings() -> Dict[str, Any]:
    """Defaults for evaluation.

    NOTE: These settings are applied by the validator and are *not* overridable
    from candidate program output.
    """
    return {
        "model": FIXED_MEDGEMMA_MODEL,
        "temperature": float(os.getenv("MEDGEMMA_TEMPERATURE", "0.0")),
        "max_tokens": int(os.getenv("MEDGEMMA_MAX_TOKENS", "256")),
        "resize": FIXED_IMAGE_SIZE,
        "concurrency": int(os.getenv("MEDGEMMA_CONCURRENCY", "8")),
        "insecure_ssl": os.getenv("MEDGEMMA_INSECURE_SSL", "0") in {"1", "true", "True"},
        "subset_dir": os.getenv("PNEUMONIA_SUBSET_DIR") or _default_subset_dir(),
        # Evaluate a limited number of images if requested (useful for quick debug).
        # None means: evaluate ALL images from manifest.jsonl.
        "num_images": _parse_optional_int(
            os.getenv("PNEUMONIA_NUM_IMAGES")
            or os.getenv("MEDGEMMA_NUM_IMAGES")
            or os.getenv("NUM_IMAGES")
        ),
    }


def normalize_program_output(obj: Any) -> Dict[str, Any]:
    """
    Normalize candidate program output to a dict:
      - If obj is a string -> {"prompt": obj}
      - If obj is a dict -> returned as-is (shallow copy)
    """
    if isinstance(obj, str):
        return {"prompt": obj}
    if isinstance(obj, dict):
        return dict(obj)
    raise TypeError(f"Program output must be str or dict, got: {type(obj)}")


def resize_and_pad_to_square(img: Image.Image, size: int) -> Image.Image:
    """Resize keeping aspect ratio; pad with black to size×size."""
    if size <= 0:
        return img.convert("RGB")

    img = img.convert("RGB")
    w, h = img.size
    if w == size and h == size:
        return img

    scale = min(size / w, size / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = img.resize((nw, nh), resample=Image.BICUBIC)

    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    left = (size - nw) // 2
    top = (size - nh) // 2
    canvas.paste(resized, (left, top))
    return canvas


def pil_to_data_url(img: Image.Image, fmt: str = "PNG", jpeg_quality: int = 95) -> str:
    """Encode PIL image as a data URL for OpenAI-compatible multimodal API."""
    fmt = fmt.upper()
    if fmt not in {"PNG", "JPEG", "JPG"}:
        raise ValueError("fmt must be PNG or JPEG")

    import io

    bio = io.BytesIO()
    if fmt in {"JPEG", "JPG"}:
        img.save(bio, format="JPEG", quality=jpeg_quality, optimize=True)
        mime = "image/jpeg"
    else:
        img.save(bio, format="PNG", optimize=True)
        mime = "image/png"

    b64 = base64.b64encode(bio.getvalue()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def extract_json(text: str) -> Dict[str, Any]:
    """
    Extract a single JSON object from a model response.
    Accepts:
      - pure JSON
      - JSON embedded in surrounding text
    """
    text = (text or "").strip()
    try:
        val = json.loads(text)
        if isinstance(val, dict):
            return val
    except Exception:
        pass

    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError("No JSON object found in model output")
    val = json.loads(m.group(0))
    if not isinstance(val, dict):
        raise ValueError("Extracted JSON is not an object")
    return val


def _parse_label_dir_name(name: str, label_map: Dict[int, str]) -> Tuple[int, str] | None:
    m = re.match(r"^(\d+)[_-](.+)$", name)
    if m:
        return int(m.group(1)), str(m.group(2)).strip()

    if name.isdigit():
        label_id = int(name)
        return label_id, label_map.get(label_id, name)

    lowered = name.strip().lower()
    for label_id, label_name in sorted(label_map.items()):
        if lowered == str(label_name).strip().lower():
            return label_id, label_name
    return None


def _parse_original_index(filename: str, fallback: int) -> int:
    m = re.search(r"_(\d+)_y\d+(?:\.[^.]+)?$", filename)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    return fallback


def _infer_split(filename: str) -> str:
    m = re.match(r"^([A-Za-z0-9]+)_", filename)
    if m:
        return str(m.group(1))
    return ""


def load_dataset(subset_dir: str | Path) -> Tuple[List[ManifestRow], Dict[int, str]]:
    """Load a subset by scanning class folders; manifest.jsonl is optional."""
    subset_dir = Path(subset_dir)
    if not subset_dir.exists():
        raise FileNotFoundError(f"subset_dir not found: {subset_dir}")
    if not subset_dir.is_dir():
        raise NotADirectoryError(f"subset_dir is not a directory: {subset_dir}")

    label_map = PNEUMONIA_LABEL_MAP.copy()
    label_map_path = subset_dir / "label_map.json"
    if label_map_path.exists():
        raw = json.loads(label_map_path.read_text(encoding="utf-8"))
        label_map = {int(k): str(v) for k, v in raw.items()}

    rows: List[ManifestRow] = []
    next_fallback_index = 0

    for child in sorted(subset_dir.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue

        parsed = _parse_label_dir_name(child.name, label_map)
        if parsed is None:
            continue

        label_id, inferred_label_name = parsed
        label_name = label_map.get(label_id, inferred_label_name)
        label_map[label_id] = label_name

        for img_path in sorted(child.rglob("*"), key=lambda p: p.as_posix()):
            if not img_path.is_file():
                continue
            if img_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue

            rows.append(
                ManifestRow(
                    rel_path=img_path.relative_to(subset_dir).as_posix(),
                    label_id=int(label_id),
                    label_name=str(label_name),
                    split=_infer_split(img_path.name),
                    original_index=_parse_original_index(img_path.name, next_fallback_index),
                )
            )
            next_fallback_index += 1

    if not rows:
        raise FileNotFoundError(
            f"No labeled image files found in {subset_dir}. Expected class subdirectories like 00_normal and 01_pneumonia."
        )

    return rows, label_map


def build_prompt(label_map: Dict[int, str], extra: str = "") -> str:
    """
    Baseline prompt builder.

    Candidate programs are free to ignore this function and return a custom prompt,
    but using it keeps the class schema consistent.

    The output contract is strict JSON:
      {"class_id": int, "class_name": str}
    """
    items = "\n".join([f"- {k}: {v}" for k, v in sorted(label_map.items())]).strip()
    extra = (extra or "").strip()
    if extra:
        extra = "\n\n" + extra

    return (
        "You are a medical image classifier. "
        "Given a chest X-ray image, choose EXACTLY ONE class id from the list below.\n\n"
        "Classes (id: name):\n"
        f"{items}\n\n"
        "Return ONLY a JSON object with keys: class_id (int), class_name (string). "
        "No extra keys, no markdown, no explanation."
        f"{extra}"
    )
