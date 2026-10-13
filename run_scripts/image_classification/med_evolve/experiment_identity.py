"""Content identities for portable, comparable Pneumonia experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


TASK_IDENTITY_SCHEMA = 1
PROTOCOL_IDENTITY_SCHEMA = 1


def _sha256(path: Path) -> str:
    """Hash one immutable source or manifest file."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_dataset_manifest(data_root: Path, resolution: int, split: str) -> dict[str, Any]:
    """Verify every selected PNG against the frozen manifest before reuse or inference."""
    directory = data_root / f"r{resolution}" / split
    manifest = directory / "manifest.jsonl"
    rows = []
    seen = set()
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            relative = Path(row["rel_path"])
            expected = row["sha256"]
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise ValueError(f"Invalid {split} manifest row {line_number}") from error
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() in seen or not isinstance(expected, str):
            raise ValueError(f"Unsafe or duplicate {split} manifest row {line_number}")
        seen.add(relative.as_posix())
        image = directory / relative
        if not image.is_file() or _sha256(image) != expected:
            raise ValueError(f"Frozen {split} image differs from manifest at row {line_number}: {relative.as_posix()}")
        rows.append({"rel_path": relative.as_posix(), "sha256": expected})
    if not rows:
        raise ValueError(f"Frozen {split} manifest is empty")
    canonical = json.dumps(rows, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return {"images": len(rows), "sha256": hashlib.sha256(canonical).hexdigest()}


def task_identity(gigaevo_dir: Path, problem_dir: Path, data_root: Path, resolution: int) -> dict[str, Any]:
    """Fingerprint the validator, seed population and frozen train/test manifests."""
    gigaevo_dir = Path(gigaevo_dir).resolve()
    problem_dir = Path(problem_dir).resolve()
    data_root = Path(data_root).resolve()
    project_root = Path(__file__).resolve().parents[1]
    sources = [
        ("project/upstream_version.json", project_root / "upstream_version.json"),
        ("task/helper.py", problem_dir / "helper.py"),
        ("task/validate.py", problem_dir / "validate.py"),
        ("task/metrics.yaml", problem_dir / "metrics.yaml"),
        ("task/task_description.txt", problem_dir / "task_description.txt"),
        ("task/data/provenance.json", data_root / "provenance.json"),
        ("task/data/initial_prompts.json", data_root / "initial_prompts.json"),
        ("task/data/train/manifest.jsonl", data_root / f"r{resolution}" / "train" / "manifest.jsonl"),
        ("task/data/test/manifest.jsonl", data_root / f"r{resolution}" / "test" / "manifest.jsonl"),
    ]
    sources.extend(
        (f"task/initial_programs/{path.name}", path)
        for path in sorted((problem_dir / "initial_programs").glob("*.py"), key=lambda value: value.name)
    )
    missing = [label for label, path in sources if not path.is_file()]
    if missing:
        raise ValueError("Task identity inputs are missing: " + ", ".join(missing))
    files = {label: _sha256(path) for label, path in sources}
    datasets = {split: _verify_dataset_manifest(data_root, resolution, split) for split in ("train", "test")}
    canonical = json.dumps({"files": files, "datasets": datasets}, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return {"schema_version": TASK_IDENTITY_SCHEMA, "resolution": resolution, "files": files, "datasets": datasets, "sha256": hashlib.sha256(canonical).hexdigest()}


def protocol_identity(args: Any) -> dict[str, Any]:
    """Fingerprint the scientific search protocol while excluding paths and search seed."""
    values = {
        "mutations": args.mutations,
        "parallel": args.parallel,
        "initial_candidates": args.initial_candidates,
        "mutator_temperature": args.mutator_temperature,
        "mutator_max_tokens": args.mutator_max_tokens,
        "optimization_mode": args.optimization_mode,
        "enable_top_k": args.enable_top_k,
        "train_repeats": args.repeats,
        "train_max_images": args.max_images,
        "final_repeats": args.final_repeats,
        "final_max_images": args.final_max_images,
    }
    canonical = json.dumps(values, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return {"schema_version": PROTOCOL_IDENTITY_SCHEMA, "values": values, "sha256": hashlib.sha256(canonical).hexdigest()}


def protocol_with_mutation_target(identity: dict[str, Any], mutations: int) -> dict[str, Any]:
    """Return a protocol identity for an explicitly recorded nondecreasing extension."""
    values = dict(identity["values"])
    values["mutations"] = mutations
    canonical = json.dumps(values, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return {"schema_version": PROTOCOL_IDENTITY_SCHEMA, "values": values, "sha256": hashlib.sha256(canonical).hexdigest()}


def effective_protocol(run_dir: Path, config: dict[str, Any]) -> dict[str, Any] | None:
    """Apply validated append-only mutation extensions to a saved protocol identity."""
    identity = config.get("protocol_identity")
    if not isinstance(identity, dict) or not isinstance(identity.get("values"), dict):
        return None
    effective = identity
    history = Path(run_dir) / "resume_history.jsonl"
    if not history.exists():
        return effective
    prior = identity["values"].get("mutations")
    for line_number, line in enumerate(history.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            previous = record["previous_mutations"]
            requested = record["requested_mutations"]
            recorded_identity = record["protocol_identity"]
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise ValueError(f"Invalid resume history at line {line_number}") from error
        if type(requested) is not int or type(previous) is not int or type(prior) is not int or previous != prior or requested < prior:
            raise ValueError(f"Nonmonotonic resume history at line {line_number}")
        expected_identity = protocol_with_mutation_target(effective, requested)
        if recorded_identity != expected_identity:
            raise ValueError(f"Resume protocol identity mismatch at line {line_number}")
        effective = expected_identity
        prior = requested
    return effective
