"""
Validator for EV task: MedGemma PneumoniaMNIST (binary CXR classification).

Expected candidate program output
--------------------------------
Candidate programs in initial_programs/ should define:

    def entrypoint():
        return PROMPT  # str

The evolutionary loop is intended to mutate *only* PROMPT. Runtime parameters
(model name, image resize, etc.) are fixed in helper.py / env vars and are NOT
accepted from candidate program output.

The validator calls MedGemma on the local subset and computes accuracy.

Return value
------------
validate(...) returns a dict with at least:
  - fitness: float  (accuracy in [0, 1])
  - is_valid: int   (1 if candidate output is usable, else 0)

Extra diagnostics are also returned (counts, confusion matrix, etc.).
"""

from __future__ import annotations

import os
import json
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

from helper import (
    build_prompt,
    extract_json,
    get_default_settings,
    load_dataset,
    load_kv_if_present,
    normalize_program_output,
    pil_to_data_url,
    resize_and_pad_to_square,
)

from PIL import Image


# Module-level cache: (abs_path, resize) -> data_url
_DATA_URL_CACHE: Dict[Tuple[str, int], str] = {}


def _get_data_url(img_path: Path, resize: int) -> str:
    key = (str(img_path.resolve()), int(resize))
    if key in _DATA_URL_CACHE:
        return _DATA_URL_CACHE[key]

    img = Image.open(img_path).convert("RGB")
    if resize and resize > 0:
        img = resize_and_pad_to_square(img, size=int(resize))
    data_url = pil_to_data_url(img, fmt="PNG")
    _DATA_URL_CACHE[key] = data_url
    return data_url


def _build_messages(prompt: str, data_url: str) -> List[dict]:
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        },
    ]


def _score_results(results: List[dict]) -> Dict[str, Any]:
    # Only count items that produced a prediction.
    total = 0
    correct = 0
    failures = 0
    parse_failures = 0
    conf = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}

    for r in results:
        if r.get("error"):
            failures += 1
            continue
        total += 1
        if r.get("parse_error"):
            parse_failures += 1
            continue
        t = int(r["true_id"])
        p = int(r["pred_id"])
        conf[(t, p)] = conf.get((t, p), 0) + 1
        if int(r.get("correct", 0)) == 1:
            correct += 1

    acc = (correct / total) if total else 0.0
    parse_rate = ((total - parse_failures) / total) if total else 0.0

    return {
        "fitness": float(acc),
        "accuracy": float(acc),
        "is_valid": 1,
        "total": int(total),
        "correct": int(correct),
        "failures": int(failures),
        "parse_failures": int(parse_failures),
        "parse_success_rate": float(parse_rate),
        "confusion_matrix": {
            "tn": int(conf.get((0, 0), 0)),
            "fp": int(conf.get((0, 1), 0)),
            "fn": int(conf.get((1, 0), 0)),
            "tp": int(conf.get((1, 1), 0)),
        },
    }


def _eval_with_async_openai(
    *, prompt: str, rows: list, subset_dir: Path, settings: dict, base_url: str, api_key: str
) -> List[dict]:
    """
    Preferred path: AsyncOpenAI + asyncio.Semaphore.
    We keep it inside a function to avoid importing asyncio at module import time.
    """
    import asyncio
    from openai import AsyncOpenAI, DefaultAsyncHttpxClient

    async def run_async() -> List[dict]:
        client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            http_client=DefaultAsyncHttpxClient(verify=not settings["insecure_ssl"]),
        )
        sem = asyncio.Semaphore(int(settings["concurrency"]))

        async def one(row) -> dict:
            img_path = subset_dir / row.rel_path
            if not img_path.exists():
                return {"rel_path": row.rel_path, "true_id": row.label_id, "true_name": row.label_name, "error": "missing_file"}

            data_url = _get_data_url(img_path, resize=int(settings["resize"]))

            async with sem:
                try:
                    resp = await client.chat.completions.create(
                        model=settings["model"],
                        temperature=float(settings["temperature"]),
                        max_tokens=int(settings.get("max_tokens", 256)),
                        messages=_build_messages(prompt, data_url),
                    )
                    text = resp.choices[0].message.content or ""
                    try:
                        pred = extract_json(text)
                        pred_id = int(pred.get("class_id"))
                        pred_name = str(pred.get("class_name", ""))
                        return {
                            "rel_path": row.rel_path,
                            "true_id": row.label_id,
                            "true_name": row.label_name,
                            "pred_id": pred_id,
                            "pred_name": pred_name,
                            "correct": 1 if pred_id == row.label_id else 0,
                            "raw": text,
                        }
                    except Exception as e:
                        return {
                            "rel_path": row.rel_path,
                            "true_id": row.label_id,
                            "true_name": row.label_name,
                            "parse_error": repr(e),
                            "raw": text,
                        }
                except Exception as e:
                    return {
                        "rel_path": row.rel_path,
                        "true_id": row.label_id,
                        "true_name": row.label_name,
                        "error": repr(e),
                    }

        tasks = [asyncio.create_task(one(r)) for r in rows]
        results: List[dict] = []
        for fut in asyncio.as_completed(tasks):
            results.append(await fut)
        return results

    return asyncio.run(run_async())


def _eval_with_threads(
    *, prompt: str, rows: list, subset_dir: Path, settings: dict, base_url: str, api_key: str
) -> List[dict]:
    """Fallback path: OpenAI sync client + ThreadPoolExecutor."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from openai import OpenAI, DefaultHttpxClient

    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
        http_client=DefaultHttpxClient(verify=not settings["insecure_ssl"]),
    )

    def one(row) -> dict:
        img_path = subset_dir / row.rel_path
        if not img_path.exists():
            return {"rel_path": row.rel_path, "true_id": row.label_id, "true_name": row.label_name, "error": "missing_file"}

        data_url = _get_data_url(img_path, resize=int(settings["resize"]))

        try:
            resp = client.chat.completions.create(
                model=settings["model"],
                temperature=float(settings["temperature"]),
                max_tokens=int(settings.get("max_tokens", 256)),
                messages=_build_messages(prompt, data_url),
            )
            text = resp.choices[0].message.content or ""
            try:
                pred = extract_json(text)
                pred_id = int(pred.get("class_id"))
                pred_name = str(pred.get("class_name", ""))
                return {
                    "rel_path": row.rel_path,
                    "true_id": row.label_id,
                    "true_name": row.label_name,
                    "pred_id": pred_id,
                    "pred_name": pred_name,
                    "correct": 1 if pred_id == row.label_id else 0,
                    "raw": text,
                }
            except Exception as e:
                return {
                    "rel_path": row.rel_path,
                    "true_id": row.label_id,
                    "true_name": row.label_name,
                    "parse_error": repr(e),
                    "raw": text,
                }
        except Exception as e:
            return {
                "rel_path": row.rel_path,
                "true_id": row.label_id,
                "true_name": row.label_name,
                "error": repr(e),
            }

    results: List[dict] = []
    with ThreadPoolExecutor(max_workers=int(settings["concurrency"])) as ex:
        futs = [ex.submit(one, r) for r in rows]
        for fut in as_completed(futs):
            results.append(fut.result())
    return results


def _guess_task_root() -> Path:
    # Prefer explicit env from the runner if present.
    for k in ("GIGAEVO_PROBLEM_DIR", "GIGA_EVO_PROBLEM_DIR", "PROBLEM_DIR"):
        v = os.getenv(k)
        if v:
            try:
                return Path(v).resolve()
            except Exception:
                pass
    # Next, try to anchor on helper.py (usually has __file__ even under exec_runner).
    try:
        import helper as _h  # type: ignore

        if getattr(_h, "__file__", None):
            return Path(_h.__file__).resolve().parent
    except Exception:
        pass
    # Fallback: current working dir.
    return Path.cwd().resolve()


def _write_text_safely(path: Path, text: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    except Exception:
        pass


def _write_debug_artifacts(subset_dir: Path, payload: dict) -> None:
    # metrics_latest.json: can contain arbitrary debug info (strings, dicts).
    _write_text_safely(subset_dir / "metrics_latest.json", json.dumps(payload, ensure_ascii=False, indent=2))


def _write_last_error(subset_dir: Path, message: str) -> None:
    _write_text_safely(subset_dir / "last_validator_error.txt", message)


def validate(program_output: Any) -> Dict[str, float]:
    """
    Main EV hook. Computes accuracy on the local PneumoniaMNIST subset.

    Contract with GigaEvo
    ---------------------
    The runner expects Dict[str, float] (see Box[dict[str,float]]). That means:
      - ONLY floats in the returned dict
      - any verbose debug / errors MUST go to files (metrics_latest.json, last_validator_error.txt, ...)
    """
    # --- Normalize candidate output ---
    try:
        out = normalize_program_output(program_output)
    except Exception as e:
        # We don't know subset_dir yet, so log to CWD.
        _write_last_error(Path.cwd(), f"bad_program_output: {e!r}\n{traceback.format_exc()}")
        return {"fitness": 0.0, "is_valid": 0.0, "parse_success_rate": 0.0}

    prompt = out.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        _write_last_error(Path.cwd(), "prompt must be a non-empty string\n")
        return {"fitness": 0.0, "is_valid": 0.0, "parse_success_rate": 0.0}

    # --- Settings ---
    # IMPORTANT: evolved programs are ONLY allowed to mutate the *prompt*.
    # We intentionally do NOT allow candidate output to override model / resize
    # (or any other runtime parameters), to prevent evaluation-time "cheating".
    settings = get_default_settings()

    # Resolve subset_dir robustly (no reliance on __file__).
    subset_dir = Path(settings["subset_dir"])
    if not subset_dir.is_absolute():
        task_root = _guess_task_root()
        subset_dir = (task_root / subset_dir).resolve()
    settings["subset_dir"] = str(subset_dir)

    # --- Load dataset ---
    try:
        rows, label_map = load_dataset(subset_dir)
    except Exception as e:
        _write_last_error(subset_dir, f"load_dataset_failed: {e!r}\n{traceback.format_exc()}")
        _write_debug_artifacts(subset_dir, {"ok": False, "stage": "load_dataset", "error": repr(e), "settings": settings})
        return {"fitness": 0.0, "is_valid": 0.0, "parse_success_rate": 0.0}

    # Optional: allow adding the schema via an environment variable only.
    # (kept for backwards compatibility / debugging)
    if os.getenv("PNEUMONIA_APPEND_CLASS_SCHEMA", "0") in {"1", "true", "True"}:
        prompt = build_prompt(label_map, extra=prompt)

    # Optional: evaluate only a subset of images (useful for quick debug).
    num_images = settings.get("num_images")
    try:
        if num_images is not None:
            n = int(num_images)
            if n > 0:
                rows = rows[:n]
    except Exception:
        pass

    # --- Credentials ---
    base_url, api_key = load_kv_if_present()
    if not base_url or not api_key:
        _write_last_error(
            subset_dir,
            "missing_credentials: set OPENAI_BASE_URL/OPENAI_API_BASE + OPENAI_API_KEY "
            "(or MEDGEMMA_*), or provide kv.py next to this task file.\n",
        )
        _write_debug_artifacts(subset_dir, {"ok": False, "stage": "credentials", "settings": settings})
        return {"fitness": 0.0, "is_valid": 0.0, "parse_success_rate": 0.0}

    # --- Run inference ---
    try:
        try:
            results = _eval_with_async_openai(
                prompt=prompt, rows=rows, subset_dir=subset_dir, settings=settings, base_url=base_url, api_key=api_key
            )
        except Exception:
            results = _eval_with_threads(
                prompt=prompt, rows=rows, subset_dir=subset_dir, settings=settings, base_url=base_url, api_key=api_key
            )
    except Exception as e:
        _write_last_error(subset_dir, f"eval_failed: {e!r}\n{traceback.format_exc()}")
        _write_debug_artifacts(subset_dir, {"ok": False, "stage": "eval", "error": repr(e), "settings": settings})
        return {"fitness": 0.0, "is_valid": 0.0, "parse_success_rate": 0.0}

    # --- Score (parse errors count as wrong because total is incremented before parse check) ---
    scored_any = _score_results(results)
    fitness = float(scored_any.get("accuracy", 0.0))
    parse_success_rate = float(scored_any.get("parse_success_rate", 0.0))

    # Persist per-image predictions for offline debugging.
    try:
        out_path = subset_dir / "predictions_latest.jsonl"
        out_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in results) + "\n", encoding="utf-8")
    except Exception:
        out_path = None

    # Rich debug payload (does not go to EV return value).
    debug_payload = {
        "ok": True,
        "metrics": scored_any,
        "settings": settings,
        "base_url": base_url,
        "predictions_path": str(out_path) if out_path else None,
        "notes": {
            "ev_return_contract": "Dict[str, float] only; see validate()",
            "parse_errors_count_as_wrong": True,
        },
    }
    _write_debug_artifacts(subset_dir, debug_payload)

    # Return only floats (EV contract).
    return {
        "fitness": fitness,
        "is_valid": 1.0,
        "parse_success_rate": parse_success_rate,
    }
