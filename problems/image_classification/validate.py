"""Evaluate joint prompt/decoder candidates identically on frozen train or held-out test images.

The local gateway owns outage retries and token monitoring. Transport failures abort
evaluation and never become candidate fitness. Parse errors count as wrong predictions.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

if __package__:
    from .helper import candidate_id, get_default_settings, image_data_url, load_dataset, normalize_program_output, parse_prediction
else:
    from helper import candidate_id, get_default_settings, image_data_url, load_dataset, normalize_program_output, parse_prediction


class InferenceInfrastructureError(RuntimeError):
    """Signal a resumable provider/gateway failure that must never become fitness."""

    def __init__(self, message: str, resume_reason: str = "inference_infrastructure"):
        super().__init__(message)
        self.resume_reason = resume_reason


def _write_json(path: Path, value: Any) -> None:
    """Atomically publish a complete audit artifact without partially written JSON."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as output:
            output.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _evaluation_identity(candidate: dict[str, Any], settings: dict[str, Any], split: str, repeat: int, rows: list[dict[str, Any]], manifest_sha256: str) -> dict[str, Any]:
    """Build a portable key for safely reusing completed per-image responses."""
    value = {
        "schema_version": 1,
        "run_id": settings["run_id"],
        "evaluation_instance_id": settings.get("evaluation_instance_id"),
        "candidate_id": candidate_id(candidate),
        "split": split,
        "repeat": repeat,
        "model": settings["model"],
        "resolution": settings["resolution"],
        "inference_seed": settings["seed"],
        "max_tokens": settings["max_tokens"],
        "dataset_manifest_sha256": manifest_sha256,
        "selected_images": [{"rel_path": row["rel_path"], "sha256": row["sha256"], "original_index": row["original_index"]} for row in rows],
    }
    canonical = json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    value["sha256"] = hashlib.sha256(canonical).hexdigest()
    return value


def _prior_attempts(output: Path, output_root: Path, split: str) -> list[Path]:
    """Find earlier explicit repeat attempts or native evaluation directories."""
    if output.parent.name == "final_evaluation":
        base = output.name.split("_resume_", 1)[0]
        return [path for path in output.parent.glob(base + "*") if path.is_dir() and path != output]
    evaluations = output_root / "evaluations"
    return [path for path in evaluations.glob(split + "_*") if path.is_dir() and path != output] if evaluations.exists() else []


def _load_reusable_rows(attempts: list[Path], identity: dict[str, Any], rows: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Merge durable rows from every compatible attempt of one logical pass."""
    expected = {row["rel_path"]: row for row in rows}
    recovered: dict[str, dict[str, Any]] = {}
    sources = []
    for attempt in sorted(attempts, key=lambda path: path.stat().st_mtime_ns, reverse=True):
        config_path = attempt / "config.json"
        predictions_path = attempt / "predictions.jsonl"
        status_path = attempt / "status.json"
        if not config_path.is_file() or not predictions_path.is_file() or not status_path.is_file():
            continue
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if config.get("evaluation_identity") != identity:
            continue
        if status.get("status") not in {"running", "resumable", "complete"}:
            continue
        lines = predictions_path.read_text(encoding="utf-8").splitlines()
        if status.get("status") == "complete" and not any(line.strip() for line in lines):
            raise ValueError(f"Complete evaluation has an empty predictions journal: {predictions_path}")
        accepted = 0
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                result = json.loads(line)
            except json.JSONDecodeError:
                if line_number == len(lines):
                    break
                raise ValueError(f"Corrupt predictions journal {predictions_path} at line {line_number}")
            source = expected.get(result.get("rel_path"))
            valid = source is not None and all(
                (
                    result.get("candidate_id") == identity["candidate_id"],
                    result.get("split") == identity["split"],
                    result.get("repeat") == identity["repeat"],
                    result.get("model") == identity["model"],
                    result.get("resolution") == identity["resolution"],
                    result.get("original_index") == source["original_index"],
                    result.get("image_sha256") == source["sha256"],
                    isinstance(result.get("raw"), str),
                    type(result.get("correct")) is int,
                )
            )
            if valid:
                previous = recovered.get(source["rel_path"])
                if previous is not None and previous != result:
                    raise ValueError(f"Conflicting prediction rows for {source['rel_path']} in logical pass {identity.get('evaluation_instance_id')}")
                recovered[source["rel_path"]] = result
                accepted += 1
        if accepted:
            sources.append(str(attempt.resolve()))
    return recovered, sources


def _gateway_resume_reason(error: HTTPError) -> str:
    """Extract the local gateway's stable resumable code without exposing its body."""
    try:
        payload = json.loads(error.read())
        code = payload.get("error", {}).get("code")
    except (OSError, ValueError, TypeError, AttributeError):
        code = None
    return str(code) if code == "retry_budget_exhausted" else "inference_http_" + str(error.code)


def _generation_seed(settings: dict[str, Any], row: dict[str, Any], split: str, repeat: int) -> int | None:
    """Use a candidate-independent seed schedule only when backend support was enabled."""
    if settings["seed"] is None:
        return None
    split_offset = 0 if split == "train" else 100000000
    return (settings["seed"] + split_offset + repeat * 1000000 + row["original_index"]) % 2147483647


def _request_prediction(row: dict[str, Any], candidate: dict[str, Any], settings: dict[str, Any], split: str, repeat: int, evaluation_id: str, config_id: str) -> dict[str, Any]:
    """Send one logical image request through the gateway and keep response usage and identity."""
    image_id = f"r{settings['resolution']}/{split}/{row['rel_path']}"
    headers = {"Authorization": "Bearer " + settings["api_key"], "Content-Type": "application/json", "X-Evolve-Role": "evaluator", "X-Evolve-Stage": split, "X-Evolve-Run-Id": settings["run_id"], "X-Evolve-Evaluation-Id": evaluation_id, "X-Evolve-Candidate-Id": config_id, "X-Evolve-Image-Id": image_id, "X-Evolve-Repeat": str(repeat)}
    content = [{"type": "text", "text": candidate["prompt"]}, {"type": "image_url", "image_url": {"url": image_data_url(str(Path(settings["data_root"]) / split / row["rel_path"]), row["sha256"])}}]
    payload = {"model": settings["model"], "messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": content}], "max_tokens": settings["max_tokens"], **candidate["sampling"]}
    seed = _generation_seed(settings, row, split, repeat)
    if seed is not None:
        payload["seed"] = seed
    request = Request(settings["base_url"].rstrip("/") + "/chat/completions", data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    started = time.monotonic()
    try:
        with urlopen(request, timeout=settings["http_timeout"]) as response:
            result = json.load(response)
    except HTTPError as error:
        reason = _gateway_resume_reason(error)
        if reason == "retry_budget_exhausted":
            raise InferenceInfrastructureError(f"Inference infrastructure failed with HTTP {error.code}; evaluation {evaluation_id} has no fitness", reason) from None
        raise RuntimeError(f"Inference request was rejected with HTTP {error.code}; evaluation {evaluation_id} has no fitness") from None
    except (URLError, TimeoutError, OSError) as error:
        raise InferenceInfrastructureError(f"Inference infrastructure failed ({type(error).__name__}); evaluation {evaluation_id} has no fitness", type(error).__name__) from None
    try:
        choice = result["choices"][0]
        text = choice["message"].get("content") or ""
        if not isinstance(text, str):
            raise TypeError("Expected string message content")
    except (KeyError, IndexError, TypeError, AttributeError) as error:
        raise RuntimeError(f"Malformed inference response; evaluation {evaluation_id} has no fitness") from error
    output = {"run_id": settings["run_id"], "evaluation_id": evaluation_id, "candidate_id": config_id, "image_id": image_id, "model": settings["model"], "resolution": settings["resolution"], "split": split, "repeat": repeat, "rel_path": row["rel_path"], "original_index": row["original_index"], "image_sha256": row["sha256"], "true_id": row["label_id"], "true_name": row["label_name"], "sampling": candidate["sampling"], "seed": seed, "response_id": result.get("id"), "response_model": result.get("model"), "system_fingerprint": result.get("system_fingerprint"), "finish_reason": choice.get("finish_reason"), "usage": result.get("usage"), "elapsed_seconds": time.monotonic() - started, "raw": text, "correct": 0}
    try:
        prediction = parse_prediction(text)
        output.update({"pred_id": prediction["class_id"], "pred_name": prediction["class_name"], "correct": int(prediction["class_id"] == row["label_id"])})
    except (ValueError, TypeError) as error:
        output["parse_error"] = str(error)
    return output


def score_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate accuracy and class metrics over all images, penalizing invalid output."""
    total = len(results)
    correct = sum(row["correct"] for row in results)
    parse_failures = sum("parse_error" in row for row in results)
    confusion = {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
    names = {(0, 0): "tn", (0, 1): "fp", (1, 0): "fn", (1, 1): "tp"}
    for row in results:
        if "pred_id" in row:
            confusion[names[(row["true_id"], row["pred_id"])]] += 1
    counts = {label: sum(row["true_id"] == label for row in results) for label in (0, 1)}
    sensitivity = confusion["tp"] / counts[1] if counts[1] else None
    specificity = confusion["tn"] / counts[0] if counts[0] else None
    f1s = []
    for label in (0, 1):
        true_positive = sum(row["true_id"] == label and row.get("pred_id") == label for row in results)
        false_positive = sum(row["true_id"] != label and row.get("pred_id") == label for row in results)
        false_negative = counts[label] - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        f1s.append(2 * true_positive / denominator if denominator else 0.0)
    usages = [row.get("usage") if isinstance(row.get("usage"), dict) else {} for row in results]
    input_known = [usage["prompt_tokens"] for usage in usages if isinstance(usage.get("prompt_tokens"), int)]
    output_known = [usage["completion_tokens"] for usage in usages if isinstance(usage.get("completion_tokens"), int)]
    complete_usage = sum(isinstance(usage.get("prompt_tokens"), int) and isinstance(usage.get("completion_tokens"), int) for usage in usages)
    return {"fitness": correct / total, "accuracy": correct / total, "is_valid": 1.0, "parse_success_rate": (total - parse_failures) / total, "total": total, "correct": correct, "parse_failures": parse_failures, "failures": 0, "confusion_matrix": confusion, "sensitivity": sensitivity, "specificity": specificity, "macro_f1": sum(f1s) / 2, "input_tokens_known": sum(input_known), "output_tokens_known": sum(output_known), "requests_with_usage": complete_usage, "requests_missing_usage": total - complete_usage, "requests_missing_input_usage": total - len(input_known), "requests_missing_output_usage": total - len(output_known)}


def evaluate_candidate(program_output: Any, split: str = "train", settings: dict[str, Any] | None = None, output_dir: str | Path | None = None, repeat: int = 0) -> dict[str, Any]:
    """Run one frozen pass and persist configuration, every prediction, and completion status."""
    settings = dict(settings or get_default_settings())
    candidate = normalize_program_output(program_output, top_k_enabled=settings["top_k_enabled"], search_space=settings["search_space"])
    if split not in ("train", "test"):
        raise ValueError("split must be train or test")
    if not settings["base_url"] or not settings["api_key"]:
        raise RuntimeError("Set MEDGEMMA_BASE_URL and MEDGEMMA_API_KEY to the monitored gateway")
    subset_dir = Path(settings["data_root"]) / split
    rows = load_dataset(subset_dir, settings["num_images"])
    config_id = candidate_id(candidate)
    evaluation_id = uuid.uuid4().hex
    output_root = Path(settings["output_root"])
    output = Path(output_dir) if output_dir else output_root / "evaluations" / f"{split}_{evaluation_id}"
    manifest_sha256 = hashlib.sha256((subset_dir / "manifest.jsonl").read_bytes()).hexdigest()
    evaluation_identity = _evaluation_identity(candidate, settings, split, repeat, rows, manifest_sha256)
    reusable, reused_sources = ({}, []) if evaluation_identity["evaluation_instance_id"] is None else _load_reusable_rows(_prior_attempts(output, output_root, split), evaluation_identity, rows)
    output.mkdir(parents=True, exist_ok=False)
    safe_settings = {key: value for key, value in settings.items() if key != "api_key"}
    config = {"run_id": settings["run_id"], "evaluation_id": evaluation_id, "candidate_id": config_id, "split": split, "repeat": repeat, "candidate": candidate, "settings": safe_settings, "dataset_manifest_sha256": manifest_sha256, "evaluation_identity": evaluation_identity, "selected_image_paths": [row["rel_path"] for row in rows], "reused_image_paths": [row["rel_path"] for row in rows if row["rel_path"] in reusable], "reused_from_attempts": reused_sources, "parse_errors_count_as_wrong": True, "transport_errors_abort": True}
    _write_json(output / "config.json", config)
    _write_json(output / "candidate.json", candidate)
    results_by_path = dict(reusable)
    pending_rows = [row for row in rows if row["rel_path"] not in results_by_path]
    _write_json(output / "status.json", {"status": "running", "evaluation_id": evaluation_id, "completed_images": len(results_by_path), "reused_images": len(reusable), "target_images": len(rows), "pending_image_paths": [row["rel_path"] for row in pending_rows]})
    started = time.monotonic()
    failure: Exception | None = None
    try:
        with (output / "predictions.jsonl").open("w", encoding="utf-8") as predictions:
            for row in rows:
                if row["rel_path"] in reusable:
                    predictions.write(json.dumps(reusable[row["rel_path"]], ensure_ascii=False) + "\n")
            predictions.flush()
            os.fsync(predictions.fileno())
            with ThreadPoolExecutor(max_workers=settings["concurrency"]) as executor:
                futures = [executor.submit(_request_prediction, row, candidate, settings, split, repeat, evaluation_id, config_id) for row in pending_rows]
                for future in as_completed(futures):
                    try:
                        result = future.result()
                    except Exception as error:
                        if failure is None:
                            failure = error
                            for pending in futures:
                                pending.cancel()
                        continue
                    results_by_path[result["rel_path"]] = result
                    predictions.write(json.dumps(result, ensure_ascii=False) + "\n")
                    predictions.flush()
                    os.fsync(predictions.fileno())
                    pending = [row["rel_path"] for row in rows if row["rel_path"] not in results_by_path]
                    _write_json(output / "status.json", {"status": "running", "evaluation_id": evaluation_id, "completed_images": len(results_by_path), "reused_images": len(reusable), "target_images": len(rows), "pending_image_paths": pending})
        if failure is not None:
            raise failure
    except Exception as error:
        resumable = isinstance(error, InferenceInfrastructureError)
        pending = [row["rel_path"] for row in rows if row["rel_path"] not in results_by_path]
        _write_json(output / "status.json", {"status": "resumable" if resumable else "failed", "resume_reason": getattr(error, "resume_reason", None), "evaluation_id": evaluation_id, "completed_images": len(results_by_path), "reused_images": len(reusable), "target_images": len(rows), "pending_image_paths": pending, "error": str(error), "elapsed_seconds": time.monotonic() - started, "fitness": None})
        raise
    results = [results_by_path[row["rel_path"]] for row in rows]
    metrics = score_results(results)
    metrics.update({"evaluation_id": evaluation_id, "candidate_id": config_id, "split": split, "repeat": repeat, "elapsed_seconds": time.monotonic() - started, "output_dir": str(output.resolve()), "reused_images": len(reusable), "fresh_images": len(pending_rows)})
    _write_json(output / "metrics.json", metrics)
    _write_json(output / "status.json", {"status": "complete", "evaluation_id": evaluation_id, "completed_images": len(results), "reused_images": len(reusable), "target_images": len(rows), "pending_image_paths": []})
    return metrics


def validate(program_output: Any, evaluation_instance_id: str | None = None) -> dict[str, float]:
    """Return GigaEvo metrics using train only; malformed candidates fail without API calls."""
    settings = get_default_settings()
    if evaluation_instance_id is not None:
        settings["evaluation_instance_id"] = evaluation_instance_id
    try:
        candidate = normalize_program_output(program_output, top_k_enabled=settings["top_k_enabled"], search_space=settings["search_space"])
    except (ValueError, TypeError):
        return {"fitness": 0.0, "is_valid": 0.0, "parse_success_rate": 0.0}
    passes = [evaluate_candidate(candidate, split="train", settings=settings, repeat=repeat) for repeat in range(settings["repeats"])]
    return {metric: sum(result[metric] for result in passes) / len(passes) for metric in ("fitness", "is_valid", "parse_success_rate")}
