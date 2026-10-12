"""Select initial and evolved candidates on train only, then evaluate both on held-out test data."""

from copy import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path


COMPATIBILITY_FIELDS = ("model", "resolution", "evaluator_provider", "evaluator_model", "evaluator_base_url", "mutator_provider", "mutator_model", "max_images", "repeats", "search_space", "inference_seed", "enable_top_k", "task_identity")
SEARCH_PROTOCOL_FIELDS = ("mutations", "parallel", "initial_candidates", "mutator_temperature", "mutator_max_tokens", "optimization_mode", "enable_top_k", "train_repeats", "train_max_images")


def read_json(path):
    """Read a persisted JSON artifact without executing candidate Python code."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def experiment_contract(config, run_dir=None):
    """Compare evaluation conditions while allowing independent search seeds and run directories."""
    contract = {key: config.get(key) for key in COMPATIBILITY_FIELDS}
    contract["search_space"] = config.get("search_space", config.get("optimization_mode", "joint").replace("_", "-"))
    contract["evaluator_base_url"] = config.get("evaluator_base_url") or {"compatible": "http://127.0.0.1:8000/v1", "openrouter": "https://openrouter.ai/api/v1"}.get(config.get("evaluator_provider"))
    from med_evolve.experiment_identity import effective_protocol

    protocol = effective_protocol(Path(run_dir), config) if run_dir is not None else config.get("protocol_identity")
    values = protocol.get("values", {}) if isinstance(protocol, dict) else {}
    contract["search_protocol"] = {key: values.get(key, config.get(key)) for key in SEARCH_PROTOCOL_FIELDS}
    return contract


def semantic_pair_contract(contract):
    """Remove informational absolute paths before deciding whether a moved pair can resume."""
    value = copy(contract)
    selection = copy(value.get("selection", {}))
    selection.pop("source_run_dirs", None)
    roles = {}
    for role, source in selection.get("roles", {}).items():
        source = copy(source)
        source.pop("source_run_dir", None)
        source.pop("source_candidate_path", None)
        roles[role] = source
    selection["roles"] = roles
    value["selection"] = selection
    return value


def source_record(run_dir, config, record):
    """Resolve one validated train record and its exact exported prompt and decoder."""
    candidate_path = run_dir / "candidates" / f"{record.get('iteration', 0):04d}_{record['id']}" / "candidate.json"
    if not candidate_path.exists():
        candidate_path = Path(record["path"]) / "candidate.json"
        if not candidate_path.is_absolute():
            candidate_path = run_dir / candidate_path
    candidate = read_json(candidate_path)
    revision = record.get("atomic_counter")
    revision_source = "exported_atomic_counter"
    if revision is None:
        native_path = run_dir / "storage" / "pneumonia" / "programs" / (record["id"] + ".json")
        if native_path.exists():
            revision = read_json(native_path)["atomic_counter"]
            revision_source = "native_storage_atomic_counter"
        else:
            revision = record.get("iteration", 0)
            revision_source = "legacy_iteration_fallback"
    return {"run_id": str(config.get("run_id", run_dir.name)), "search_seed": int(config["seed"]), "source_run_dir": str(run_dir), "program_id": record["id"], "train_fitness": record["metrics"]["fitness"], "generation": record.get("generation", 1), "parents": record.get("parents", []), "atomic_counter": revision, "revision_source": revision_source, "iteration": record.get("iteration", 0), "state": record["state"], "candidate": candidate, "candidate_sha256": hashlib.sha256(json.dumps(candidate, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest(), "source_candidate_path": str(candidate_path)}


def select_pair(run_dirs):
    """Independently select train-best initial and descendant candidates across compatible runs."""
    directories = sorted({Path(directory).resolve() for directory in run_dirs}, key=str)
    if not directories:
        raise ValueError("At least one evolution run is required")
    contracts = []
    per_run = {"initial": [], "evolved": []}
    for directory in directories:
        config = read_json(directory / "run_config.json")
        contract = experiment_contract(config, directory)
        if len(directories) > 1 and (contract.get("task_identity") is None or any(value is None for value in contract["search_protocol"].values())):
            raise ValueError("Legacy evolution runs without task/protocol identity cannot be pooled")
        if contracts and contract != contracts[0]:
            raise ValueError("Evolution runs have incompatible model, data, provider, or train evaluation settings")
        contracts.append(contract)
        groups = {"initial": [], "evolved": []}
        for record in read_json(directory / "candidates.json"):
            metrics = record.get("metrics", {})
            fitness = metrics.get("fitness")
            if record.get("state") not in {"done", "discarded"} or metrics.get("is_valid", 0) <= 0 or not isinstance(fitness, (int, float)) or not math.isfinite(fitness):
                continue
            initial = record.get("generation", 1) == 1 or not record.get("parents")
            groups["initial" if initial else "evolved"].append(source_record(directory, config, record))
        for role, records in groups.items():
            if records:
                per_run[role].append(max(records, key=lambda row: (row["train_fitness"], row["atomic_counter"], row["program_id"])))
    if any(not records for records in per_run.values()):
        raise ValueError("Both a valid initial candidate and a valid evolved descendant are required")
    roles = {role: max(records, key=lambda row: (row["train_fitness"], -row["search_seed"], row["program_id"])) for role, records in per_run.items()}
    return {"schema_version": 1, "experiment": contracts[0], "source_run_dirs": [str(directory) for directory in directories], "selection_split": "train", "within_run_tie_break": "atomic_counter descending, then program_id descending", "across_run_tie_break": "search seed ascending, then program_id descending", "roles": roles}


def evaluate_pair(args, gateway, selection=None):
    """Freeze a train-selected pair and resume its independent test passes through one root gateway."""
    from med_evolve.integration import write_json
    from med_evolve.launcher import configure_evaluator, evaluate_saved, save_manifest

    selection = selection or select_pair(args.evolution_run or [args.run_dir])
    if args.model != selection["experiment"]["model"] or args.resolution != selection["experiment"]["resolution"] or args.evaluator_model != selection["experiment"]["evaluator_model"]:
        raise ValueError("Final evaluator model and resolution must match the selected evolution runs")
    destination = args.run_dir / "paired_evaluation"
    contract = {"selection": selection, "test_settings": {"model": args.model, "resolution": args.resolution, "evaluator_model": args.evaluator_model, "evaluator_provider": args.evaluator_provider, "evaluator_base_url": args.evaluator_base_url or selection["experiment"]["evaluator_base_url"], "max_images": args.final_max_images, "repeats": args.final_repeats, "inference_seed": args.inference_seed, "enable_top_k": args.enable_top_k}}
    contract_path = destination / "selection.json"
    if contract_path.exists() and (not args.resume or semantic_pair_contract(read_json(contract_path)) != semantic_pair_contract(contract)):
        raise ValueError("Existing paired evaluation differs; resume identical final settings or use a new run directory")
    write_json(contract_path, contract)
    parent_status_path = args.run_dir / "run_status.json"
    parent_status = read_json(parent_status_path) if parent_status_path.exists() else {}
    parent_status.update({"phase": "final_evaluation", "evolution_status": "not_run" if args.evolution_run else "completed", "status": "running", "pid": os.getpid(), "run_id": args.run_id, "model": args.model, "resolution": args.resolution, "final_repeats": args.final_repeats, "final_max_images": args.final_max_images, "completed_final_roles": []})
    reports = {}
    try:
        for role, source in selection["roles"].items():
            role_args = copy(args)
            role_args.run_dir = destination / role
            role_args.run_id = args.run_id
            role_args.evaluate = role_args.run_dir / "selected_candidate.json"
            role_args.evolution_run = None
            role_args.split = "test"
            role_args.max_images = args.final_max_images
            role_args.repeats = args.final_repeats
            role_args.evaluation_instance_id = source["program_id"]
            role_args.validator_timeout = role_args.http_timeout * role_args.max_images * role_args.repeats + 600
            write_json(role_args.evaluate, source["candidate"])
            save_manifest(role_args)
            configure_evaluator(role_args, gateway.base_url)

            def progress(current, selected_role=role):
                """Forward each role's heartbeat without announcing overall completion early."""
                parent_status.update({"status": "waiting" if current["status"] == "waiting" else "running", "final_role": selected_role, "completed_final_repeats": current.get("completed_repeats", 0), "target_final_repeats": args.final_repeats, "updated_at": datetime.now(timezone.utc).isoformat()})
                write_json(parent_status_path, parent_status)

            result = evaluate_saved(role_args, gateway, progress=progress)
            reports[role] = {"selection": source, "test": result, "independent_role_evaluation": True}
            parent_status["completed_final_roles"].append(role)
            write_json(destination / "summary.json", {"selection_split": "train", "evaluation_split": "test", "roles": reports})
    except BaseException as error:
        resumable = getattr(error, "resume_reason", None) is not None
        parent_status.update({"status": "resumable" if resumable else "failed", "phase": "resumable_final_evaluation" if resumable else "failed_final_evaluation", "resume_reason": getattr(error, "resume_reason", None), "resume_required": resumable, "updated_at": datetime.now(timezone.utc).isoformat()})
        write_json(parent_status_path, parent_status)
        raise
    parent_status.update({"status": "completed", "phase": "completed", "final_role": None, "updated_at": datetime.now(timezone.utc).isoformat()})
    write_json(parent_status_path, parent_status)
    return {"selection_split": "train", "evaluation_split": "test", "roles": reports}
