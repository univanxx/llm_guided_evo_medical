"""Run or resume native GigaEvo search and evaluate complete Pneumonia candidates."""

from __future__ import annotations

import argparse
import asyncio
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import hashlib
import os
from pathlib import Path
import random
import sys
import threading
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ROOT.parents[1]
PROVIDERS = {"compatible": ("http://127.0.0.1:8000/v1", "INFERENCE_API_KEY"), "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY")}
MODEL_IDS = {"4b": "google/medgemma-1.5-4b-it", "27b": "google/medgemma-27b-it"}


def validated_base_url(value: str) -> str:
    """Reject endpoint forms that could persist or transmit embedded credentials."""
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Provider base URL must be HTTP(S) without credentials, query, or fragment")
    return value


def parser() -> argparse.ArgumentParser:
    """Define shared experiment, provider, retry, and evaluation CLI settings."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model", choices=MODEL_IDS)
    result.add_argument("--resolution", choices=(28, 64, 128, 224), type=int)
    result.add_argument("--run-dir", type=Path, required=True)
    result.add_argument("--smoke", action="store_true", help="Short defaults: one seed, one mutant, and two balanced train images")
    result.add_argument("--resume", action="store_true")
    result.add_argument("--seed", type=int, default=42, help="Search and mutation RNG seed")
    result.add_argument("--inference-seed", type=int, help="Optional backend generation seed; independent of search seed")
    result.add_argument("--generations", type=int, default=30, help="Historical budget conversion: (generations-1)*8 descendants; current upstream uses steady state")
    result.add_argument("--mutations", type=int, help="Explicit native descendant budget, overriding --generations")
    result.add_argument("--parallel", type=int, help="Concurrent mutants and candidate DAGs; default 1")
    result.add_argument("--initial-candidates", type=int, help="Number of deterministic seed programs; default 10, smoke 1")
    result.add_argument("--repeats", type=int, default=1)
    result.add_argument("--optimization-mode", "--search-space", type=lambda value: value.replace("-", "_"), choices=("joint", "prompt_only"), default="joint")
    result.add_argument("--enable-top-k", "--allow-top-k", action="store_true", help="Allow top_k only after confirming support on the selected backend")
    result.add_argument("--max-images", type=int, help="Balanced subset size; default full train=100 or full test=624")
    result.add_argument("--mutator-temperature", type=float, default=0.6)
    result.add_argument("--mutator-max-tokens", type=int, default=8192)
    result.add_argument("--credentials-file", type=Path, help="Read API credentials from a dotenv file")
    result.add_argument("--resolve-ip", help="Optional HTTPS DNS override retaining TLS hostname verification")
    result.add_argument("--retry-budget-seconds", type=float, default=15300, help="4h15m availability budget per request")
    result.add_argument("--request-timeout-seconds", type=float, default=180)
    evaluation = result.add_mutually_exclusive_group()
    evaluation.add_argument("--evaluate", type=Path, help="Evaluate an exported JSON or Python candidate instead of running evolution")
    evaluation.add_argument("--evolution-run", type=Path, action="append", help="Select initial/evolved winners across these compatible train runs and evaluate the pair")
    result.add_argument("--skip-final-evaluation", action="store_true", help="Finish after search; useful when selecting across several runs later")
    result.add_argument("--final-repeats", type=int, help="Test repeats for each selected role; full default 3, smoke default 2")
    result.add_argument("--final-max-images", type=int, help="Held-out images per role and repeat; full default 624, smoke uses max-images")
    result.add_argument("--split", choices=("train", "test"), default="test")
    result.add_argument("--gigaevo-dir", type=Path, default=Path(os.environ.get("GIGAEVO_DIR", str(REPOSITORY_ROOT.parent / "gigaevo-core"))))
    result.add_argument("--data-root", type=Path, help="External PneumoniaMNIST data directory containing provenance.json and r28/r64/r128/r224")
    for role in ("mutator", "evaluator"):
        result.add_argument(f"--{role}-provider", choices=PROVIDERS)
        result.add_argument(f"--{role}-base-url")
        result.add_argument(f"--{role}-model")
        result.add_argument(f"--{role}-key-env")
    return result


def read_credentials(path: Path | None) -> dict[str, str]:
    """Read dotenv credentials without executing a credential file."""
    from dotenv import dotenv_values

    values = {key: value for key, value in dotenv_values(ROOT / ".env").items() if value}
    if path:
        values.update({key: value for key, value in dotenv_values(path).items() if value})
    return values


def prepare_arguments(args: argparse.Namespace) -> argparse.Namespace:
    """Resolve defaults and time budgets before connecting to either provider."""
    args.run_dir = args.run_dir.resolve()
    args.gigaevo_dir = args.gigaevo_dir.resolve()
    args.problem_dir = REPOSITORY_ROOT / "problems" / "image_classification"
    if args.data_root is None:
        args.data_root = args.problem_dir / "data"
    args.data_root = args.data_root.resolve()
    args.run_id = args.run_dir.name
    if args.evolution_run:
        from med_evolve.final_selection import select_pair

        args.pair_selection = select_pair(args.evolution_run)
        source = args.pair_selection["experiment"]
        for key in ("model", "resolution", "evaluator_provider", "evaluator_model", "evaluator_base_url", "mutator_provider", "mutator_model"):
            explicit = getattr(args, key)
            if explicit is not None and explicit != source[key]:
                raise ValueError(f"Paired test {key} must match the selected evolution runs")
            setattr(args, key, source[key])
        args.optimization_mode = source["search_space"].replace("-", "_")
        args.enable_top_k = source["enable_top_k"]
        args.inference_seed = source["inference_seed"]
        if args.split != "test":
            raise ValueError("Paired final evaluation uses held-out test only")
    args.model = args.model or "4b"
    args.resolution = args.resolution or 28
    args.mutator_provider = args.mutator_provider or "compatible"
    args.evaluator_provider = args.evaluator_provider or "compatible"
    args.mutations = args.mutations if args.mutations is not None else (1 if args.smoke else (args.generations - 1) * 8)
    args.parallel = args.parallel if args.parallel is not None else 1
    args.initial_candidates = args.initial_candidates if args.initial_candidates is not None else (1 if args.smoke else 10)
    args.max_images = args.max_images if args.max_images is not None else (2 if args.smoke else (624 if (args.evaluate or args.evolution_run) and args.split == "test" else 100))
    args.final_repeats = args.final_repeats if args.final_repeats is not None else (2 if args.smoke else 3)
    args.final_max_images = args.final_max_images if args.final_max_images is not None else (args.max_images if args.smoke or args.evolution_run else 624)
    args.mutator_model = args.mutator_model or ("gpt-oss-120b" if args.mutator_provider == "compatible" else "openai/gpt-oss-120b")
    if args.evaluator_provider == "openrouter" and not args.evaluator_model:
        raise ValueError("--evaluator-model is required for OpenRouter; select an available vision model explicitly")
    args.evaluator_model = args.evaluator_model or MODEL_IDS[args.model]
    if args.mutations < 0 or args.parallel < 1 or not 1 <= args.initial_candidates <= 10 or args.repeats < 1 or args.max_images < 2 or args.max_images % 2:
        raise ValueError("Require mutations>=0, parallel/repeats>=1, 1-10 initial candidates, and an even max-images>=2")
    if args.retry_budget_seconds <= 0 or args.request_timeout_seconds <= 0:
        raise ValueError("Retry budget and request timeout must be positive")
    if args.final_repeats < 1 or not 2 <= args.final_max_images <= 624 or args.final_max_images % 2:
        raise ValueError("Require final-repeats>=1 and even final-max-images between 2 and 624")
    for role in ("mutator", "evaluator"):
        provider = getattr(args, f"{role}_provider")
        validated_base_url(getattr(args, f"{role}_base_url") or PROVIDERS[provider][0])
    from med_evolve.experiment_identity import protocol_identity, task_identity

    args.task_identity = task_identity(args.gigaevo_dir, args.problem_dir, args.data_root, args.resolution)
    args.protocol_identity = protocol_identity(args)
    args.http_timeout = args.retry_budget_seconds + args.request_timeout_seconds + 120
    # Each image can independently encounter a full outage; outer limits must not truncate retries.
    args.validator_timeout = args.http_timeout * args.max_images * args.repeats + 600
    return args


def configure_evaluator(args: argparse.Namespace, base_url: str) -> None:
    """Pass experiment-owned settings to validators and their worker subprocesses."""
    values = {"PNEUMONIA_MODEL": args.model, "PNEUMONIA_RESOLUTION": str(args.resolution), "PNEUMONIA_DATA_ROOT": str(args.data_root), "PNEUMONIA_RUN_DIR": str(args.run_dir), "PNEUMONIA_RUN_ID": args.run_id, "PNEUMONIA_MAX_IMAGES": str(args.max_images), "PNEUMONIA_REPEATS": str(args.repeats), "MEDGEMMA_BASE_URL": base_url, "MEDGEMMA_API_KEY": "gateway-local", "MEDGEMMA_MODEL": args.evaluator_model, "MEDGEMMA_HTTP_TIMEOUT_SECONDS": str(args.http_timeout), "PNEUMONIA_REQUEST_TIMEOUT": str(args.http_timeout)}
    os.environ.update(values)
    os.environ["PNEUMONIA_OPTIMIZATION_MODE"] = args.optimization_mode
    os.environ["PNEUMONIA_SEARCH_SPACE"] = args.optimization_mode.replace("_", "-")
    os.environ["PNEUMONIA_ENABLE_TOP_K"] = "1" if args.enable_top_k else "0"
    evaluation_instance_id = getattr(args, "evaluation_instance_id", None)
    if evaluation_instance_id is None:
        os.environ.pop("PNEUMONIA_EVALUATION_INSTANCE_ID", None)
    else:
        os.environ["PNEUMONIA_EVALUATION_INSTANCE_ID"] = str(evaluation_instance_id)
    if args.inference_seed is None:
        os.environ.pop("PNEUMONIA_SEED", None)
    else:
        os.environ["PNEUMONIA_SEED"] = str(args.inference_seed)


def start_role_gateway(stack: ExitStack, args: argparse.Namespace, role: str, credentials: dict[str, str]):
    """Start a monitored gateway for one independently selected provider."""
    from med_evolve.gateway import GatewayConfig, start_gateway

    provider = getattr(args, f"{role}_provider")
    default_url, default_key = PROVIDERS[provider]
    key_name = getattr(args, f"{role}_key_env") or (f"{role.upper()}_API_KEY" if provider == "compatible" else default_key)
    key = os.environ.get(key_name) or credentials.get(key_name)
    if not key:
        raise ValueError(f"Set {key_name} or supply --credentials-file for the {role} provider")
    bind_port = 0
    config = GatewayConfig(upstream_base_url=getattr(args, f"{role}_base_url") or default_url, api_key=key, output_dir=args.run_dir / f"gateway_{role}", run_id=args.run_id, retry_budget_seconds=args.retry_budget_seconds, request_timeout_seconds=args.request_timeout_seconds, dns_override=args.resolve_ip if provider == "compatible" else None, provider_kind=provider, default_role=role, bind_port=bind_port)
    return stack.enter_context(start_gateway(config))


def save_manifest(args: argparse.Namespace) -> None:
    """Save the nonsecret experiment contract and reject incompatible resumes."""
    from med_evolve.integration import write_json
    from med_evolve.experiment_identity import effective_protocol, protocol_identity, task_identity

    mutator_base_url = validated_base_url(args.mutator_base_url or PROVIDERS[args.mutator_provider][0])
    evaluator_base_url = validated_base_url(args.evaluator_base_url or PROVIDERS[args.evaluator_provider][0])
    current_task_identity = task_identity(args.gigaevo_dir, args.problem_dir, args.data_root, args.resolution)
    current_protocol = protocol_identity(args)
    args.task_identity = current_task_identity
    args.protocol_identity = current_protocol
    fields = ("model", "resolution", "seed", "inference_seed", "repeats", "max_images", "parallel", "mutator_provider", "mutator_model", "evaluator_provider", "evaluator_model", "mutator_temperature", "mutator_max_tokens", "optimization_mode", "enable_top_k", "final_repeats", "final_max_images")
    contract = {key: getattr(args, key) for key in fields}
    contract.update({"search_space": args.optimization_mode.replace("_", "-"), "split": args.split if (args.evaluate or args.evolution_run) else "train", "evaluation_source_sha256": hashlib.sha256(args.evaluate.read_bytes()).hexdigest() if args.evaluate else None, "initial_candidates": args.initial_candidates, "mutator_base_url": mutator_base_url, "evaluator_base_url": evaluator_base_url, "task_identity": current_task_identity})
    path = args.run_dir / "run_config.json"
    previous = {}
    if path.exists():
        if not args.resume:
            raise ValueError("Existing run_config.json: use --resume or a new --run-dir")
        previous = json.loads(path.read_text(encoding="utf-8"))
        if any(previous.get(key) != value for key, value in contract.items()):
            raise ValueError("Resume settings differ from saved run_config.json")
        saved_protocol = effective_protocol(args.run_dir, previous)
        if saved_protocol is None:
            raise ValueError("Existing run_config.json lacks versioned protocol identity; use a new run directory")
        saved_values = saved_protocol["values"]
        current_values = current_protocol["values"]
        immutable_protocol = set(current_values) - {"mutations"}
        if any(saved_values.get(key) != current_values.get(key) for key in immutable_protocol):
            raise ValueError("Resume protocol differs from saved run_config.json")
        previous_target = saved_values.get("mutations")
        if type(previous_target) is not int or args.mutations < previous_target:
            raise ValueError("Resume cannot reduce the saved mutation target")
        if args.mutations > previous_target:
            history = args.run_dir / "resume_history.jsonl"
            history.parent.mkdir(parents=True, exist_ok=True)
            record = {"timestamp": datetime.now(timezone.utc).isoformat(), "previous_mutations": previous_target, "requested_mutations": args.mutations, "protocol_identity": current_protocol}
            with history.open("a", encoding="utf-8") as output:
                output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                output.flush()
                os.fsync(output.fileno())
        return
    contract.update({"run_id": args.run_id, "created_at": datetime.now(timezone.utc).isoformat(), "engine": "GigaEvo native SteadyStateEvolutionEngine / MAP-Elites", "mutations": args.mutations, "protocol_identity": current_protocol, "historical_generations_budget": args.generations, "parallel": args.parallel, "initial_candidates": args.initial_candidates, "retry_budget_seconds": args.retry_budget_seconds, "request_timeout_seconds": args.request_timeout_seconds, "http_timeout_seconds": args.http_timeout, "validator_timeout_seconds": args.validator_timeout, "gigaevo_dir": str(args.gigaevo_dir), "problem_dir": str(args.problem_dir), "data_root": str(args.data_root), "mutator_base_url": mutator_base_url, "evaluator_base_url": evaluator_base_url, "dns_override": args.resolve_ip, "evaluation_source": str(args.evaluate.resolve()) if args.evaluate else None})
    contract.update({"automatic_final_evaluation": not args.skip_final_evaluation and not args.evaluate and not args.evolution_run, "final_repeats": args.final_repeats, "final_max_images": args.final_max_images, "evolution_source_runs": [str(path.resolve()) for path in args.evolution_run] if args.evolution_run else None})
    write_json(path, contract)


def record_runtime(args: argparse.Namespace) -> None:
    """Append secret-free execution scheduling without changing scientific identity."""
    helper = args.problem_dir / "helper.py"
    raw_concurrency = os.environ.get("MEDGEMMA_CONCURRENCY", "8")
    try:
        image_concurrency = int(raw_concurrency)
    except ValueError as error:
        raise ValueError("MEDGEMMA_CONCURRENCY must be an integer") from error
    if not 1 <= image_concurrency <= 16:
        raise ValueError("MEDGEMMA_CONCURRENCY must be between 1 and 16")
    cache_dir = os.environ.get("PNEUMONIA_IMAGE_CACHE", "")
    if cache_dir and not Path(cache_dir).is_absolute():
        raise ValueError("PNEUMONIA_IMAGE_CACHE must be empty or an absolute path")
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "resume": bool(args.resume),
        "image_concurrency": image_concurrency,
        "image_cache_dir": cache_dir,
        "helper_sha256": hashlib.sha256(helper.read_bytes()).hexdigest(),
        "task_identity_sha256": args.task_identity["sha256"],
        "protocol_identity_sha256": args.protocol_identity["sha256"],
    }
    history = args.run_dir / "runtime_history.jsonl"
    with history.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        output.flush()
        os.fsync(output.fileno())


def completed_repeat(run_dir: Path, repeat: int, *, split: str | None = None, candidate: str | None = None, total: int | None = None, evaluation_instance_id: str | None = None) -> dict | None:
    """Find a completed original or resumed pass so another resume does not rebill it."""
    base = run_dir / "final_evaluation"
    name = f"repeat_{repeat}"
    paths = [path / "metrics.json" for path in base.glob(name + "*") if path.name == name or path.name.startswith(name + "_resume_")]
    for path in sorted((path for path in paths if path.is_file()), key=lambda path: path.stat().st_mtime_ns, reverse=True):
        if split is None and candidate is None and total is None and evaluation_instance_id is None:
            try:
                metrics = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if metrics.get("repeat") == repeat:
                return metrics
            continue
        try:
            metrics = json.loads(path.read_text(encoding="utf-8"))
            status = json.loads((path.parent / "status.json").read_text(encoding="utf-8"))
            config = json.loads((path.parent / "config.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        identity = config.get("evaluation_identity", {})
        try:
            prediction_rows = [json.loads(line) for line in (path.parent / "predictions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        except (OSError, json.JSONDecodeError):
            continue
        selected = identity.get("selected_images") if isinstance(identity, dict) else None
        expected_images = {row.get("rel_path"): row for row in selected} if isinstance(selected, list) and all(isinstance(row, dict) for row in selected) else {}
        actual_images = {row.get("rel_path"): row for row in prediction_rows if isinstance(row, dict) and isinstance(row.get("rel_path"), str)}
        complete_rows = len(prediction_rows) == len(actual_images) == total and set(actual_images) == set(expected_images)
        if complete_rows:
            complete_rows = all(row.get("candidate_id") == candidate and row.get("split") == split and row.get("repeat") == repeat and row.get("image_sha256") == expected_images[relative].get("sha256") for relative, row in actual_images.items())
        checks = (
            metrics.get("repeat") == repeat,
            status.get("status") == "complete",
            split is None or metrics.get("split") == split == identity.get("split"),
            candidate is None or metrics.get("candidate_id") == candidate == identity.get("candidate_id"),
            total is None or metrics.get("total") == total == status.get("target_images") == status.get("completed_images"),
            evaluation_instance_id is None or identity.get("evaluation_instance_id") == evaluation_instance_id,
            total is None or complete_rows,
        )
        if all(checks):
            return metrics
    return None


def evaluate_saved(args: argparse.Namespace, gateway, progress=None) -> dict:
    """Re-read the complete candidate and evaluate it through the same validator."""
    from med_evolve.integration import write_json
    from problems.image_classification.helper import candidate_id, load_candidate, load_dataset
    from problems.image_classification.validate import evaluate_candidate
    from med_evolve.integration import gateway_failure

    candidate = load_candidate(args.evaluate)
    config_id = candidate_id(candidate)
    evaluation_instance_id = str(getattr(args, "evaluation_instance_id", None) or ("saved:" + config_id))
    os.environ["PNEUMONIA_EVALUATION_INSTANCE_ID"] = evaluation_instance_id
    expected_total = len(load_dataset(args.data_root / f"r{args.resolution}" / args.split, args.max_images))
    write_json(args.run_dir / "candidate.json", candidate)
    results = []
    reused_repeats = []
    stop = threading.Event()

    def heartbeat() -> None:
        """Keep final evaluation visible during long provider outages."""
        while not stop.is_set():
            with gateway.engine.cooldown_lock:
                waiting = bool(gateway.engine.cooldowns)
            current = {"status": "waiting" if waiting else "running", "pid": os.getpid(), "run_id": args.run_id, "model": args.model, "resolution": args.resolution, "completed_repeats": len(results), "target_repeats": args.repeats, "updated_at": datetime.now(timezone.utc).isoformat()}
            write_json(args.run_dir / "run_status.json", current)
            if progress:
                progress(current)
            stop.wait(5)

    monitor = threading.Thread(target=heartbeat, name="evaluation-heartbeat", daemon=True)
    monitor.start()
    try:
        for repeat in range(args.repeats):
            output_dir = args.run_dir / "final_evaluation" / f"repeat_{repeat}"
            completed = completed_repeat(args.run_dir, repeat, split=args.split, candidate=config_id, total=expected_total, evaluation_instance_id=evaluation_instance_id) if args.resume else None
            if completed is not None:
                results.append(completed)
                reused_repeats.append(repeat)
            else:
                if output_dir.exists():
                    # Retain the incomplete pass; a resumed attempt gets a new directory.
                    output_dir = output_dir.with_name(output_dir.name + "_resume_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f"))
                results.append(evaluate_candidate(candidate, split=args.split, output_dir=output_dir, repeat=repeat))
            failure = gateway_failure([gateway])
            if failure is not None:
                raise failure
    except Exception as exc:
        failure = gateway_failure([gateway])
        if failure is not None:
            raise failure from exc
        raise
    finally:
        stop.set()
        monitor.join()
    summary = {"candidate": candidate, "split": args.split, "repeats": args.repeats, "reused_completed_repeats": reused_repeats, "results": results}
    write_json(args.run_dir / "summary.json", summary)
    current = {"status": "completed", "pid": os.getpid(), "run_id": args.run_id, "model": args.model, "resolution": args.resolution, "completed_repeats": len(results), "updated_at": datetime.now(timezone.utc).isoformat()}
    write_json(args.run_dir / "run_status.json", current)
    if progress:
        progress(current)
    return summary


def main(argv: list[str] | None = None) -> int:
    """Create isolated gateways, run native evolution, and persist its outcome."""
    args = prepare_arguments(parser().parse_args(argv))
    sys.dont_write_bytecode = True
    from med_evolve.upstream_integrity import verify_upstream

    integrity_manifest = ROOT / "gigaevo_upstream_manifest.json"
    verify_upstream(args.gigaevo_dir, integrity_manifest)
    sys.path.insert(0, str(args.gigaevo_dir))
    from loguru import logger
    from omegaconf import OmegaConf
    import numpy as np
    from med_evolve.integration import build_config, run_native_engine, write_json
    from med_evolve.final_selection import evaluate_pair
    from med_evolve.logging_utils import configure_logging

    args.run_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.run_dir / "run_status.json"
    previous_status = json.loads(status_path.read_text(encoding="utf-8")) if args.resume and status_path.exists() else {}
    completed_search = bool(args.resume and not args.evaluate and not args.evolution_run and (args.run_dir / "candidates.json").exists() and (previous_status.get("evolution_status") == "completed" or previous_status.get("status") == "completed") and previous_status.get("mutations_created", -1) >= args.mutations)
    configure_logging(args.run_dir)
    random.seed(args.seed)
    np.random.seed(args.seed)
    preflight_complete = False
    try:
        save_manifest(args)
        record_runtime(args)
        preflight_complete = True
        credentials = read_credentials(args.credentials_file)
        starting_status = dict(previous_status)
        starting_status.update({"status": "starting", "pid": os.getpid(), "run_id": args.run_id, "model": args.model, "resolution": args.resolution, "resume_required": False, "updated_at": datetime.now(timezone.utc).isoformat()})
        write_json(args.run_dir / "run_status.json", starting_status)
        with ExitStack() as stack:
            evaluator_gateway = start_role_gateway(stack, args, "evaluator", credentials)
            configure_evaluator(args, evaluator_gateway.base_url)
            if args.evolution_run:
                evaluate_pair(args, evaluator_gateway, args.pair_selection)
            elif args.evaluate:
                evaluate_saved(args, evaluator_gateway)
            else:
                if completed_search:
                    previous_status.update({"status": "completed" if args.skip_final_evaluation else "running", "phase": "completed" if args.skip_final_evaluation else "evolution_completed", "evolution_status": "completed", "pid": os.getpid(), "updated_at": datetime.now(timezone.utc).isoformat()})
                    write_json(status_path, previous_status)
                    logger.info("Reusing completed native search; no mutation or train evaluation is needed")
                else:
                    mutator_gateway = start_role_gateway(stack, args, "mutator", credentials)
                    cfg = build_config(args, mutator_gateway.base_url)
                    (args.run_dir / "gigaevo_config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=False), encoding="utf-8")
                    os.environ["GIGAEVO_LLM_IO_DUMP_DIR"] = str(args.run_dir / "llm_io")
                    logger.info("Native steady-state MAP-Elites: {} initial + {} descendants; model={}, resolution={}", args.initial_candidates, args.mutations, args.model, args.resolution)
                    asyncio.run(run_native_engine(args, cfg, [mutator_gateway, evaluator_gateway]))
                if not args.skip_final_evaluation:
                    evaluate_pair(args, evaluator_gateway)
        verify_upstream(args.gigaevo_dir, integrity_manifest)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        if not preflight_complete:
            logger.error("Experiment preflight rejected: {}", exc)
            return 1
        path = args.run_dir / "run_status.json"
        status = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        resume_reason = getattr(exc, "resume_reason", None)
        resumable = resume_reason is not None
        if status.get("evolution_status") == "completed" and not args.skip_final_evaluation:
            status["phase"] = "resumable_final_evaluation" if resumable else "failed_final_evaluation"
        status.update({"status": "resumable" if resumable else "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", "resume_reason": resume_reason, "resume_required": resumable or isinstance(exc, KeyboardInterrupt), "error": str(exc), "updated_at": datetime.now(timezone.utc).isoformat()})
        write_json(path, status)
        logger.error("Experiment stopped: {}", exc)
        verify_upstream(args.gigaevo_dir, integrity_manifest)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
