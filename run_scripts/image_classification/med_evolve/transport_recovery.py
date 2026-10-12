"""Audit and requeue evaluations proven to have failed on a local connection reset."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any

from gigaevo.programs.core_types import StageState
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState


VALIDATOR_STAGE = "CallValidatorFunction"
RESET_TRACEBACK = re.compile(r"user_code\.InferenceInfrastructureError: Inference infrastructure failed \(ConnectionResetError\); evaluation ([0-9a-f]+) has no fitness(?:\n|$)")


def _canonical(value: Any) -> bytes:
    """Encode JSON-native evidence deterministically for optimistic preconditions."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _program_fingerprint(program: Program) -> str:
    """Fingerprint the complete trusted Program serialization before recovery."""
    return hashlib.sha256(_canonical(program.to_dict())).hexdigest()


def _plan_checksum(plan: dict[str, Any]) -> str:
    """Bind apply to the exact audited plan without including its checksum field."""
    return hashlib.sha256(_canonical({key: value for key, value in plan.items() if key != "plan_sha256"})).hexdigest()


def connection_reset_evaluation(program: Program) -> str | None:
    """Return the failed evaluation id only for the decoded validator StageError shape."""
    result = program.stage_results.get(VALIDATOR_STAGE)
    if result is None or result.status != StageState.FAILED or result.error is None:
        return None
    if result.error.stage not in {"OutageAwareValidator", VALIDATOR_STAGE}:
        return None
    traceback = result.error.traceback
    if not isinstance(traceback, str):
        return None
    matches = RESET_TRACEBACK.findall(traceback)
    return matches[-1] if len(matches) == 1 else None


def _descendants(program_id: str, programs: Mapping[str, Program]) -> list[str]:
    """Compute transitive descendants from both lineage directions without trusting one index alone."""
    reverse = {candidate.id: set(candidate.lineage.children) for candidate in programs.values()}
    for candidate in programs.values():
        for parent in candidate.lineage.parents:
            reverse.setdefault(parent, set()).add(candidate.id)
    found = set()
    pending = list(reverse.get(program_id, ()))
    while pending:
        child = pending.pop()
        if child in found:
            continue
        found.add(child)
        pending.extend(reverse.get(child, ()))
    return sorted(found)


async def audit_transport_failures(storage: Any, archives: Mapping[str, Any], candidate_ids: set[str] | None = None) -> dict[str, Any]:
    """Build a read-only recovery plan from decoded, authoritative Program objects."""
    loaded = await storage.get_all()
    programs = {program.id: program for program in loaded}
    if candidate_ids is not None:
        missing = set(candidate_ids) - programs.keys()
        if missing:
            raise ValueError(f"Candidate IDs missing from authoritative storage: {sorted(missing)}")
    archive_members = {name: set(await archive.get_all_elites()) for name, archive in archives.items()}
    findings = []
    for program in sorted(programs.values(), key=lambda item: (item.iteration, item.id)):
        if candidate_ids is not None and program.id not in candidate_ids:
            continue
        evaluation_id = connection_reset_evaluation(program)
        if evaluation_id is None:
            continue
        marker = program.metadata.get("pneumonia_evaluation_pass")
        findings.append({"id": program.id, "iteration": program.iteration, "prior_state": program.state.value, "prior_metrics": dict(program.metrics), "atomic_counter": program.atomic_counter, "program_sha256": _program_fingerprint(program), "evaluation_id": evaluation_id, "evaluation_pass": dict(marker) if isinstance(marker, dict) else None, "archive_locations": sorted(name for name, members in archive_members.items() if program.id in members), "descendants": _descendants(program.id, programs)})
    plan = {"schema_version": 1, "program_count": len(programs), "findings": findings}
    plan["plan_sha256"] = _plan_checksum(plan)
    return plan


async def apply_transport_recovery(plan: dict[str, Any], storage: Any, archives: Mapping[str, Any], *, stopped: bool, expected_plan_sha256: str) -> dict[str, Any]:
    """Apply an audited plan after checking stopped-run and exact-state preconditions."""
    if not stopped:
        raise RuntimeError("Transport recovery requires a stopped child and supervisor")
    if plan.get("plan_sha256") != expected_plan_sha256 or _plan_checksum(plan) != expected_plan_sha256:
        raise RuntimeError("Recovery plan checksum mismatch")
    findings = plan.get("findings")
    if not isinstance(findings, list):
        raise ValueError("Recovery plan findings must be a list")
    current = {program.id: program for program in await storage.mget([finding["id"] for finding in findings])}
    if set(current) != {finding["id"] for finding in findings}:
        raise RuntimeError("Recovery target disappeared after audit")
    for finding in findings:
        program = current[finding["id"]]
        if _program_fingerprint(program) != finding["program_sha256"]:
            raise RuntimeError(f"Recovery preimage changed for {program.id}")
        if connection_reset_evaluation(program) != finding["evaluation_id"]:
            raise RuntimeError(f"Transport evidence changed for {program.id}")
        actual_locations = []
        for name, archive in archives.items():
            if program.id in await archive.get_all_elites():
                actual_locations.append(name)
        actual_locations.sort()
        if actual_locations != finding["archive_locations"]:
            raise RuntimeError(f"Archive membership changed for {program.id}")
    applied = []
    for finding in findings:
        program = current[finding["id"]]
        removed = []
        for name, archive in archives.items():
            if await archive.remove_elite_by_id(program.id):
                removed.append(name)
        if sorted(removed) != finding["archive_locations"]:
            raise RuntimeError(f"Archive removal mismatch for {program.id}")
        marker = program.metadata.get("pneumonia_evaluation_pass")
        program.state = ProgramState.QUEUED
        program.metrics = {}
        program.stage_results = {}
        program.metadata.pop("mutation_context", None)
        program.metadata["evaluation_interrupted"] = True
        program.metadata["transport_recovery"] = {"reason": "ConnectionResetError", "evaluation_id": finding["evaluation_id"], "prior_state": finding["prior_state"], "prior_atomic_counter": finding["atomic_counter"]}
        if isinstance(marker, dict):
            marker = dict(marker)
            marker["state"] = "running"
            program.metadata["pneumonia_evaluation_pass"] = marker
        await storage.add(program)
        applied.append({"id": program.id, "prior_state": finding["prior_state"], "removed_archives": sorted(removed), "descendants": finding["descendants"]})
    return {"schema_version": 1, "plan_sha256": expected_plan_sha256, "applied": applied}
