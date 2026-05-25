#!/usr/bin/env python3
"""
Smoke-test for initial programs.

Usage:
  python problems/med_pneu_v12/debug_run_initials.py --quick
  python problems/med_pneu_v12/debug_run_initials.py --smoke-vlm
  python problems/med_pneu_v12/debug_run_initials.py

What it does:
  - loads each program from initial_programs/*.py
  - runs entrypoint() to get candidate output
  - runs validate(candidate_output)
  - prints (fitness, is_valid, parse_success_rate)

Artifacts:
  - <subset_dir>/predictions_latest.jsonl
  - <subset_dir>/metrics_latest.json
  - <subset_dir>/last_validator_error.txt (only if something failed)

Tip:
  The validator runs in-process here. During full GigaEvo runs the validator is
  executed in a separate process (exec_runner), so make sure base_url/api_key are
  available via env (OPENAI_BASE_URL + OPENAI_API_KEY) or kv.py.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

from validate import _eval_with_async_openai, _eval_with_threads, validate  # noqa: E402
from helper import get_default_settings, load_dataset, load_kv_if_present, normalize_program_output  # noqa: E402


def _load_module(path: Path):
    name = path.stem
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def _resolve_subset_dir() -> Path:
    settings = get_default_settings()
    subset_dir = Path(settings["subset_dir"])
    if not subset_dir.is_absolute():
        subset_dir = (Path(__file__).resolve().parent / subset_dir).resolve()
    return subset_dir


def _run_quick_check() -> int:
    subset_dir = _resolve_subset_dir()
    manifest_path = subset_dir / "manifest.jsonl"
    label_map_path = subset_dir / "label_map.json"

    print("=== med_pneu_v12: quick data check ===")
    print(f"OPENAI_BASE_URL: {os.getenv('OPENAI_BASE_URL')}")
    print(f"OPENAI_API_KEY set: {bool(os.getenv('OPENAI_API_KEY'))}")
    print(f"Subset dir env: {os.getenv('PNEUMONIA_SUBSET_DIR')}")
    print(f"Num images env: {os.getenv('PNEUMONIA_NUM_IMAGES') or os.getenv('MEDGEMMA_NUM_IMAGES')}")
    print(f"Subset dir resolved: {subset_dir}")
    print(f"Subset dir exists: {subset_dir.exists()}")
    print(f"manifest.jsonl exists (optional): {manifest_path.exists()}")
    print(f"label_map.json exists: {label_map_path.exists()}")
    print()

    try:
        rows, label_map = load_dataset(subset_dir)
    except Exception as e:
        print(f"Dataset check FAILED: {e!r}")
        return 1

    print(f"Rows detected: {len(rows)}")
    print(f"Labels: {label_map}")

    sample_rows = rows[:3]
    missing = 0
    for row in sample_rows:
        img_path = subset_dir / row.rel_path
        exists = img_path.exists()
        print(f"- {row.rel_path} -> exists={exists}, label={row.label_name}")
        if not exists:
            missing += 1

    if missing:
        print()
        print(f"Quick check FAILED: {missing} sampled image(s) are missing.")
        return 1

    print()
    print("Quick check OK: dataset scan succeeded and sampled files are present.")
    return 0


def _run_smoke_vlm() -> int:
    subset_dir = _resolve_subset_dir()
    rows, _label_map = load_dataset(subset_dir)
    if not rows:
        print("Smoke VLM FAILED: dataset scan returned 0 rows.")
        return 1

    base_url, api_key = load_kv_if_present()
    if not base_url or not api_key:
        print("Smoke VLM FAILED: missing OPENAI/MEDGEMMA credentials or kv.py settings.")
        return 1

    sample_row = rows[0]
    settings = get_default_settings()
    settings["subset_dir"] = str(subset_dir)
    settings["concurrency"] = 1
    settings["num_images"] = 1

    print("=== med_pneu_v12: one-request-per-candidate smoke test ===")
    print(f"Subset dir resolved: {subset_dir}")
    print(f"Sample image: {sample_row.rel_path}")
    print(f"Model: {settings['model']}")
    print()

    base = Path(__file__).resolve().parent
    programs_dir = base / "initial_programs"
    py_files = sorted(p for p in programs_dir.glob("*.py") if p.is_file() and p.name != "__init__.py")

    failures = 0
    for path in py_files:
        try:
            mod = _load_module(path)
            if not hasattr(mod, "entrypoint"):
                print(f"{path.name:<28} -> SKIP no entrypoint()")
                failures += 1
                continue

            raw = mod.entrypoint()
            normalized = normalize_program_output(raw)
            prompt = normalized.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                print(f"{path.name:<28} -> FAIL empty prompt")
                failures += 1
                continue

            try:
                results = _eval_with_async_openai(
                    prompt=prompt,
                    rows=[sample_row],
                    subset_dir=subset_dir,
                    settings=settings,
                    base_url=base_url,
                    api_key=api_key,
                )
            except Exception:
                results = _eval_with_threads(
                    prompt=prompt,
                    rows=[sample_row],
                    subset_dir=subset_dir,
                    settings=settings,
                    base_url=base_url,
                    api_key=api_key,
                )

            result = results[0]
            if result.get("error"):
                print(f"{path.name:<28} -> REQUEST_ERROR {result['error']}")
                failures += 1
                continue

            if result.get("parse_error"):
                raw_preview = str(result.get("raw", "")).replace("\n", " ").strip()[:120]
                print(f"{path.name:<28} -> RESPONSE_OK parse_error={result['parse_error']} raw={raw_preview!r}")
                continue

            print(
                f"{path.name:<28} -> RESPONSE_OK pred_id={result.get('pred_id')} "
                f"pred_name={result.get('pred_name')!r} correct={result.get('correct')}"
            )
        except Exception as e:
            print(f"{path.name:<28} -> EXCEPTION {e!r}")
            failures += 1

    print()
    if failures:
        print(f"Smoke VLM finished with {failures} failure(s).")
        return 1

    print("Smoke VLM OK: every initial candidate got one successful model response.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test med_pneu_v12 initial programs.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Only verify that data/pneumoniamnist_subset is readable and sample images are present.",
    )
    parser.add_argument(
        "--smoke-vlm",
        action="store_true",
        help="For each initial candidate, send exactly one request on one image and print the model response status.",
    )
    args = parser.parse_args()

    if args.quick:
        raise SystemExit(_run_quick_check())
    if args.smoke_vlm:
        raise SystemExit(_run_smoke_vlm())

    subset_dir = _resolve_subset_dir()
    print("=== med_pneu_v12: initial programs smoke-test ===")
    print(f"OPENAI_BASE_URL: {os.getenv('OPENAI_BASE_URL')}")
    print(f"OPENAI_API_KEY set: {bool(os.getenv('OPENAI_API_KEY'))}")
    print(f"Subset dir env: {os.getenv('PNEUMONIA_SUBSET_DIR')}")
    print(f"Num images env: {os.getenv('PNEUMONIA_NUM_IMAGES') or os.getenv('MEDGEMMA_NUM_IMAGES')}")
    print(f"Subset dir resolved: {subset_dir}")
    print()

    base = Path(__file__).resolve().parent
    programs_dir = base / "initial_programs"
    py_files = sorted(p for p in programs_dir.glob("*.py") if p.is_file() and p.name != "__init__.py")

    for path in py_files:
        mod = _load_module(path)
        if not hasattr(mod, "entrypoint"):
            continue
        try:
            out = mod.entrypoint()
            res = validate(out)
        except Exception as e:
            print(f"{path.name:<28} -> EXCEPTION {e!r}")
            continue

        fitness = res.get("fitness")
        is_valid = res.get("is_valid")
        psr = res.get("parse_success_rate")
        print(f"{path.name:<28} -> {{'fitness': {fitness}, 'is_valid': {is_valid}, 'parse_success_rate': {psr}}}")

    print()
    print("Inspect:")
    print("  - <subset_dir>/predictions_latest.jsonl (per-image outputs)")
    print("  - <subset_dir>/metrics_latest.json (metrics+debug+settings)")
    print("  - <subset_dir>/last_validator_error.txt (if something failed)")


if __name__ == "__main__":
    main()
