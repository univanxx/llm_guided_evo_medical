#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import inspect
import json
from pathlib import Path
from statistics import mean, stdev
import sys
from typing import Any

from common import METRICS, PROBLEM_DIR, read_csv, write_csv, write_json

if str(PROBLEM_DIR) not in sys.path:
    sys.path.insert(0, str(PROBLEM_DIR))

from context import load_eval_dataset  # noqa: E402
from validate import validate  # noqa: E402


def load_module(path: Path):
    resolved = path.resolve()
    name = f"triage_eval_{resolved.stem}_{abs(hash(str(resolved)))}"
    spec = importlib.util.spec_from_file_location(name, resolved)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {resolved}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def run_program(path: Path, patients: list[dict]) -> dict[str, Any]:
    module = load_module(path)
    entrypoint = module.entrypoint
    if inspect.iscoroutinefunction(entrypoint):
        result = await entrypoint(patients)
    else:
        result = await asyncio.to_thread(entrypoint, patients)
    if inspect.isawaitable(result):
        result = await result
    if not isinstance(result, dict):
        raise TypeError(f"{path}: entrypoint returned {type(result).__name__}")
    return result


def write_predictions(path: Path, patients: list[dict], result: dict[str, Any]) -> None:
    """Persist per-case predictions for confusion matrices, kappa and bootstrap."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row, pred in zip(patients, result.get("results", []), strict=False):
            patient = row["patient"]
            raw = pred.get("predicted_acuity") if isinstance(pred, dict) else None
            f.write(
                json.dumps(
                    {
                        "case_id": patient.get("case_id"),
                        "true_acuity": str(patient["true_acuity"]),
                        "predicted_acuity": None if raw is None else str(raw).strip(),
                    }
                )
                + "\n"
            )


async def evaluate(args: argparse.Namespace) -> None:
    included = set(args.include_stem)
    programs = sorted(
        path
        for path in args.program_dir.glob("*.py")
        if path.name != "__init__.py" and (not included or path.stem in included)
    )
    if not programs:
        raise FileNotFoundError(f"No programs found in {args.program_dir}")
    patients, meta = load_eval_dataset(args.dataset)
    runs_path = args.output_dir / "evaluation_runs.csv"
    rows: list[dict[str, Any]] = (
        list(read_csv(runs_path)) if args.resume and runs_path.is_file() else []
    )
    completed = {
        (str(row.get("program_path", "")), int(row.get("repeat", 0)))
        for row in rows
        if row.get("dataset") == args.dataset
    }
    for path in programs:
        code = path.read_text(encoding="utf-8")
        for repeat in range(1, args.repeats + 1):
            key = (str(path.resolve()), repeat)
            if key in completed:
                print(
                    f"[RESUME] {path.name} repeat {repeat}/{args.repeats} already complete"
                )
                continue
            print(
                f"[EVAL] {path.name} repeat {repeat}/{args.repeats} "
                f"on {len(patients)} {args.dataset} cases"
            )
            result = await run_program(path, patients)
            write_predictions(
                args.output_dir / "predictions" / f"{path.stem}_r{repeat}.jsonl",
                patients,
                result,
            )
            rows.append(
                {
                    "program": path.stem,
                    "program_path": str(path.resolve()),
                    "dataset": args.dataset,
                    "repeat": repeat,
                    "n_patients": len(patients),
                    "execution_time_sec": float(result.get("execution_time_sec", 0)),
                    "uses_helper": int("from helper import" in code or "import helper" in code),
                    "direct_openai_reference": int("AsyncOpenAI" in code or "OpenAI(" in code),
                    **validate(patients, result),
                }
            )
            # Persist every completed repeat.  A provider/network interruption
            # can therefore resume without paying for successful repeats again.
            args.output_dir.mkdir(parents=True, exist_ok=True)
            write_csv(runs_path, rows)
            completed.add(key)

    summaries: list[dict[str, Any]] = []
    for path in programs:
        group = [row for row in rows if row["program_path"] == str(path.resolve())]
        summary: dict[str, Any] = {
            "program": path.stem,
            "program_path": str(path.resolve()),
            "dataset": args.dataset,
            "n_patients": len(patients),
            "repeats": len(group),
            "uses_helper": group[0]["uses_helper"],
            "direct_openai_reference": group[0]["direct_openai_reference"],
        }
        for metric in (*METRICS, "execution_time_sec"):
            values = [float(row[metric]) for row in group]
            summary[f"{metric}_mean"] = mean(values)
            summary[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        summaries.append(summary)
    summaries.sort(
        key=lambda row: (
            float(row["fitness_mean"]),
            float(row["exact_accuracy_mean"]),
        ),
        reverse=True,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "evaluation_runs.csv", rows)
    write_csv(args.output_dir / "evaluation_summary.csv", summaries)
    write_json(
        args.output_dir / "evaluation_manifest.json",
        {
            "dataset": meta,
            "repeats": args.repeats,
            "program_dir": str(args.program_dir),
            "program_count": len(programs),
        },
    )
    print(f"[DONE] wrote evaluation to {args.output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--program-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--include-stem", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be >= 1")
    asyncio.run(evaluate(args))


if __name__ == "__main__":
    main()
