"""Adapt the native GigaEvo engine to one resumable Pneumonia experiment."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import time
from typing import Any
import uuid

from med_evolve.windows_compat import install_windows_flock

install_windows_flock()

from gigaevo.entrypoint.default_pipelines import DefaultPipelineBuilder
from gigaevo.evolution.engine import SteadyStateEvolutionEngine
from gigaevo.evolution.engine.refresh import ParentRefresher
from gigaevo.llm.models import MultiModelRouter
from gigaevo.programs.core_types import ProgramStageResult
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState
from gigaevo.problems.context import ProblemContext
from gigaevo.programs.stages.python_executors.execution import CallProgramFunction
from gigaevo.programs.stages.cache_handler import NO_CACHE
from gigaevo.programs.stages.validator_metadata import ProgramMetadataValidatorStage
from loguru import logger
from med_evolve.storage import repair_interrupted_archive_members


GATEWAYS: list[Any] = []


class ResumableExperimentError(RuntimeError):
    """Stop cleanly after retry exhaustion while preserving durable native state."""

    def __init__(self, message: str, resume_reason: str = "gateway_stopped"):
        super().__init__(message)
        self.resume_reason = resume_reason


EVALUATION_FAILURES: list[ResumableExperimentError] = []


def gateway_failure(gateways: list[Any]) -> ResumableExperimentError | None:
    """Preserve the primary gateway failure even when a worker error wins the race."""
    for gateway in gateways:
        if gateway.fatal_event.is_set():
            reason = getattr(gateway, "fatal_reason", None) or "gateway stopped"
            return ResumableExperimentError(f"Inference gateway stopped: {reason}; resume this run after service recovery", getattr(gateway, "resume_reason", "gateway_stopped"))
    return None


_TRANSPORT_STAGE_ERRORS = {"ConnectionResetError", "BrokenPipeError", "IncompleteReadError", "TimeoutError"}


def evaluation_transport_failure(result: Any) -> ResumableExperimentError | None:
    """Recognize validator infrastructure failures without widening candidate rejection."""
    if not isinstance(result, ProgramStageResult) or result.error is None:
        return None
    error = result.error
    if error.type in _TRANSPORT_STAGE_ERRORS:
        return ResumableExperimentError(f"Validator transport failed ({error.type}); resume the incomplete evaluation", "evaluation_transport")
    details = "\n".join(value for value in (error.message, error.traceback) if isinstance(value, str))
    if error.type == "SubprocessError" and "InferenceInfrastructureError" in details:
        return ResumableExperimentError("Validator reported inference infrastructure failure; resume the incomplete evaluation", "evaluation_transport")
    return None


def infrastructure_failure(gateways: list[Any]) -> ResumableExperimentError | None:
    """Expose either gateway or validator transport failure to engine supervision."""
    return gateway_failure(gateways) or (EVALUATION_FAILURES[0] if EVALUATION_FAILURES else None)


def write_json(path: Path, value: Any) -> None:
    """Atomically publish a UTF-8 status or experiment artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class GatewayModelRouter(MultiModelRouter):
    """Retain upstream mutation and token accounting without its 10-second probe."""

    def _verify_models(self) -> None:
        """Delegate availability checks to the gateway's four-hour retry policy."""
        logger.info("Model availability is checked on inference requests by the retry gateway")


class OutageAwareEvolutionEngine(SteadyStateEvolutionEngine):
    """Keep the native search while extending its hardcoded parent refresh limit."""

    def __init__(self, *, parent_refresh_timeout: float, **kwargs: Any):
        """Give native just-in-time parent refresh the same budget as its DAG."""
        super().__init__(**kwargs)
        self._parent_refresher = ParentRefresher(storage=self.storage, poll_interval=0.2, timeout_seconds=parent_refresh_timeout)


class PneumoniaProblemContext(ProblemContext):
    """Describe the active decoder search space without rewriting shared task files."""

    @property
    def task_description(self) -> str:
        """Append researcher-selected control and optional top-k constraints."""
        text = super().task_description
        if os.environ.get("PNEUMONIA_SEARCH_SPACE") == "prompt-only":
            text += "\nACTIVE CONTROL: Optimize only prompt. Every candidate must keep temperature=0 and top_p=1. Do not change sampling settings."
        if os.environ.get("PNEUMONIA_ENABLE_TOP_K") == "1":
            text += "\nACTIVE EXTENSION: top_k is enabled; when temperature>0 sampling may also include top_k in {-1,20,50}."
        return text


class OutageAwareValidator(ProgramMetadataValidatorStage):
    """Keep infrastructure failures pending until the supervisor saves the run."""

    cache_handler = NO_CACHE

    def __init__(self, *, storage: Any, **kwargs: Any):
        super().__init__(**kwargs)
        self.storage = storage

    async def begin_evaluation_pass(self, program: Program) -> str:
        """Persist one pass id before inference and reuse it only after interruption."""
        previous = program.metadata.get("pneumonia_evaluation_pass")
        interrupted = bool(program.metadata.pop("evaluation_interrupted", False))
        if interrupted and isinstance(previous, dict) and isinstance(previous.get("id"), str):
            marker = dict(previous)
            marker["state"] = "running"
        else:
            prior_ordinal = previous.get("ordinal", 0) if isinstance(previous, dict) else 0
            ordinal = prior_ordinal + 1 if type(prior_ordinal) is int and prior_ordinal >= 0 else 1
            marker = {"id": f"{program.id}:train:{ordinal}", "ordinal": ordinal, "state": "running"}
        program.metadata["pneumonia_evaluation_pass"] = marker
        await self.storage.update(program)
        return marker["id"]

    def _build_call(self, program: Program) -> tuple[Any, dict[str, Any]]:
        """Pass the durable logical-pass id through the upstream call-argument hook."""
        marker = program.metadata.get("pneumonia_evaluation_pass")
        if not isinstance(marker, dict) or not isinstance(marker.get("id"), str):
            raise RuntimeError("Pneumonia evaluation pass was not persisted before inference")
        args, kwargs = super()._build_call(program)
        return args, {**kwargs, "evaluation_instance_id": marker["id"]}

    async def compute(self, program: Program) -> Any:
        """Retry transport failures in one logical pass and never score them."""
        await self.begin_evaluation_pass(program)
        retry_budget = float(os.environ.get("PNEUMONIA_TRANSPORT_RETRY_BUDGET_SECONDS") or os.environ.get("MEDGEMMA_HTTP_TIMEOUT_SECONDS") or "43200")
        if retry_budget < 0:
            raise ValueError("PNEUMONIA_TRANSPORT_RETRY_BUDGET_SECONDS must be non-negative")
        retry_base = float(os.environ.get("PNEUMONIA_TRANSPORT_RETRY_BASE_SECONDS", "1"))
        if retry_base < 0:
            raise ValueError("PNEUMONIA_TRANSPORT_RETRY_BASE_SECONDS must be non-negative")
        started = time.monotonic()
        attempt = 0
        while True:
            result = await super().compute(program)
            failure = gateway_failure(GATEWAYS)
            if failure is None:
                failure = evaluation_transport_failure(result)
                if failure is not None and time.monotonic() - started < retry_budget:
                    attempt += 1
                    delay = min(30.0, retry_base * (2 ** min(attempt - 1, 5)))
                    logger.warning("Validator transport retry {} for program {} in {:.1f}s; logical pass {} is preserved", attempt, program.id[:8], delay, program.metadata["pneumonia_evaluation_pass"]["id"])
                    await asyncio.sleep(delay)
                    continue
            if failure is not None:
                if not EVALUATION_FAILURES:
                    EVALUATION_FAILURES.append(failure)
                # Stage.execute converts ordinary exceptions into scientific failures.
                # Stay pending until run_native_engine supervision raises the signal.
                await asyncio.Future()
            if isinstance(result, ProgramStageResult):
                return result
            marker = dict(program.metadata["pneumonia_evaluation_pass"])
            marker["state"] = "complete"
            program.metadata["pneumonia_evaluation_pass"] = marker
            await self.storage.update(program)
            return result


class PneumoniaPipeline(DefaultPipelineBuilder):
    """Use the upstream evaluation DAG without paid insight or memory stages."""

    def __init__(self, ctx: Any, *, stage_timeout: float, dag_timeout: float, max_parallel: int = 2):
        """Give validators outage-sized budgets but keep candidate execution short."""
        super().__init__(ctx, stage_timeout=stage_timeout, dag_timeout=dag_timeout, max_parallel=max_parallel, include_legacy_feedback=False, archive_gate_enabled=False)
        self.replace_stage("CallProgramFunction", lambda: CallProgramFunction(function_name="entrypoint", python_path=[ctx.problem_ctx.problem_dir.resolve()], timeout=30, max_memory_mb=512))
        self.replace_stage("CallValidatorFunction", lambda: OutageAwareValidator(storage=ctx.storage, path=ctx.problem_ctx.problem_dir / "validate.py", function_name="validate", timeout=stage_timeout, max_memory_mb=2048))


class SeedLoader:
    """Load the archived initial prompts in deterministic order, optionally one."""

    def __init__(self, problem_dir: str, limit: int | None = None):
        """Bind the shared task directory and smoke-test seed limit."""
        self.problem_dir = Path(problem_dir)
        self.limit = limit

    async def load(self, storage: Any) -> list[Program]:
        """Persist each seed without silently swallowing missing or invalid files."""
        paths = sorted((self.problem_dir / "initial_programs").glob("*.py"))
        if self.limit:
            paths = paths[:self.limit]
        if not paths:
            raise ValueError("No initial programs found")
        programs = []
        for ordinal, path in enumerate(paths):
            program = Program(code=path.read_text(encoding="utf-8"), iteration=ordinal)
            program.metadata = {"source": "initial_program", "strategy_name": path.stem, "file_path": str(path)}
            await storage.add(program)
            programs.append(program)
        return programs


def build_config(args: Any, mutator_url: str) -> Any:
    """Compose the current upstream configuration and replace integration nodes."""
    from hydra import compose, initialize_config_dir
    from hydra.core.hydra_config import HydraConfig
    from omegaconf import OmegaConf, open_dict
    from gigaevo.config.resolvers import register_resolvers

    register_resolvers()
    with initialize_config_dir(version_base=None, config_dir=str(args.gigaevo_dir / "config")):
        cfg = compose(config_name="config", overrides=["memory=none", "pipeline=guided", "storage=disk"], return_hydra_config=True)
    with open_dict(cfg):
        cfg.hydra.runtime.output_dir = str(args.run_dir)
        cfg.hydra.runtime.cwd = str(args.gigaevo_dir)
        cfg.problem.name = "pneumonia"
        cfg.problem.dir = str(args.problem_dir)
        cfg.problem_context._target_ = "med_evolve.integration.PneumoniaProblemContext"
        cfg.model_name = args.mutator_model
        cfg.llm_base_url = mutator_url
        cfg.temperature = args.mutator_temperature
        cfg.max_tokens = args.mutator_max_tokens
        cfg.request_timeout = args.http_timeout
        cfg.llm._target_ = "med_evolve.integration.GatewayModelRouter"
        cfg.llm.structured_output_method = "auto"
        cfg.llm.models[0].api_key = "gateway-local"
        cfg.llm.models[0].max_retries = 0
        cfg.llm.models[0].seed = args.seed
        cfg.llm.models[0].default_headers = {"X-Evolve-Role": "mutator", "X-Evolve-Stage": "mutation", "X-Evolve-Run-Id": args.run_id}
        cfg.llm_max_concurrent = args.parallel
        cfg.max_mutants = args.mutations
        cfg.max_in_flight = args.parallel
        cfg.num_parents = 1 if args.smoke else 2
        cfg.island_max_size = 75
        cfg.primary_resolution = 150
        cfg.enable_migration = False
        cfg.max_concurrent_dags = args.parallel
        cfg.logging.level = "INFO"
        cfg.log_rotation = "10 MB"
        cfg.log_retention = 2
        cfg.runner_poll_interval = 0.2
        cfg.loop_interval = 0.2
        cfg.stage_timeout = args.validator_timeout
        cfg.dag_timeout = args.validator_timeout + 300
        cfg.post_cap_drain_grace_s = None
        cfg.max_consecutive_mutation_failures = 3
        cfg.evolution_engine._target_ = "med_evolve.integration.OutageAwareEvolutionEngine"
        cfg.evolution_engine.parent_refresh_timeout = args.validator_timeout + 600
        cfg.archive_gate_enabled = False
        cfg.pipeline = {"id": "custom", "reads_external_memory": False, "routes_program_metadata": True, "archive_gate_mode": "none"}
        cfg.pipeline_builder = {"_target_": "med_evolve.integration.PneumoniaPipeline", "ctx": "${evolution_context}", "stage_timeout": "${stage_timeout}", "dag_timeout": "${dag_timeout}", "max_parallel": 2}
        cfg.program_loader = {"_target_": "med_evolve.integration.SeedLoader", "problem_dir": "${problem.dir}", "limit": args.initial_candidates}
        cfg.live_frontier_compare.enabled = False
        cfg.redis.resume = args.resume
        cfg.program_storage._target_ = "med_evolve.storage.RecoverableDiskProgramStorage"
        cfg.archive_storage_factory._target_ = "med_evolve.storage.AtomicDiskArchiveStorageFactory"
    HydraConfig.instance().set_config(cfg)
    return OmegaConf.masked_copy(cfg, [key for key in cfg if key != "hydra"])


async def requeue_interrupted(storage: Any, program_ids: set[str]) -> None:
    """Reset interrupted evaluations so resume retries them without bad fitness."""
    for program in await storage.mget(sorted(program_ids)):
        if program is None or program.state not in {ProgramState.RUNNING, ProgramState.QUEUED}:
            continue
        program.state = ProgramState.QUEUED
        program.metrics = {}
        program.stage_results = {}
        program.metadata.pop("mutation_context", None)
        program.metadata["evaluation_interrupted"] = True
        await storage.add(program)


async def export_programs(storage: Any, run_dir: Path) -> dict[str, Any]:
    """Export every program and the best full inference configuration."""
    from problems.image_classification.helper import normalize_program_output

    programs = sorted(await storage.get_all(), key=lambda program: program.iteration)
    summary = []
    eligible = []
    role_winners = {}
    for program in programs:
        destination = run_dir / "candidates" / f"{program.iteration:04d}_{program.id}"
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "program.py").write_text(program.code, encoding="utf-8")
        origin = "initial" if program.generation == 1 or not program.lineage.parents else "evolved"
        record = {"id": program.id, "iteration": program.iteration, "generation": program.generation, "atomic_counter": program.atomic_counter, "origin": origin, "state": program.state.value, "metrics": program.metrics, "parents": program.lineage.parents, "path": str(destination)}
        write_json(destination / "metrics.json", record)
        if program.metrics.get("is_valid", 0) > 0:
            # Export the actual evaluated payload, never execute candidate code again.
            stage = program.stage_results.get("CallProgramFunction")
            if stage is None or stage.output is None:
                raise RuntimeError(f"Valid program {program.id} has no persisted candidate payload")
            candidate = normalize_program_output(stage.output.data)
            write_json(destination / "candidate.json", candidate)
            if program.state == ProgramState.DONE:
                eligible.append((program.metrics.get("fitness", float("-inf")), -program.iteration, program, candidate))
            if program.state in {ProgramState.DONE, ProgramState.DISCARDED}:
                ranking = (program.metrics.get("fitness", float("-inf")), program.atomic_counter, program.id)
                if origin not in role_winners or ranking > role_winners[origin][0]:
                    role_winners[origin] = (ranking, program, candidate, record)
        summary.append(record)
    write_json(run_dir / "candidates.json", summary)
    best_record: dict[str, Any] = {"candidate_count": len(programs), "valid_candidate_count": sum(program.metrics.get("is_valid", 0) > 0 for program in programs), "archived_candidate_count": len(eligible)}
    if eligible:
        _, _, best, candidate = max(eligible, key=lambda item: (item[0], item[1]))
        best_dir = run_dir / "best"
        best_dir.mkdir(exist_ok=True)
        (best_dir / "program.py").write_text(best.code, encoding="utf-8")
        write_json(best_dir / "candidate.json", candidate)
        best_record.update({"program_id": best.id, "metrics": best.metrics, "candidate": candidate})
        write_json(best_dir / "metrics.json", best_record)
    for role, (_, program, candidate, record) in role_winners.items():
        role_dir = run_dir / f"best_{role}"
        role_dir.mkdir(exist_ok=True)
        (role_dir / "program.py").write_text(program.code, encoding="utf-8")
        write_json(role_dir / "candidate.json", candidate)
        write_json(role_dir / "metrics.json", record)
        best_record[f"best_{role}"] = {"program_id": program.id, "train_fitness": program.metrics["fitness"], "path": str(role_dir)}
    write_json(run_dir / "summary.json", best_record)
    return best_record


async def run_native_engine(args: Any, cfg: Any, gateways: list[Any]) -> dict[str, Any]:
    """Supervise GigaEvo's native engine, preserve failures, and export artifacts."""
    from datetime import datetime, timezone
    from hydra.utils import instantiate
    from gigaevo.config.validation import validate_algorithm_requirements, validate_archive_gate_pipeline_compat, validate_memory_pipeline_compat, validate_program_format_pipeline_compat
    from gigaevo.programs.stages.python_executors.wrapper import default_exec_runner_pool

    GATEWAYS[:] = gateways
    EVALUATION_FAILURES.clear()
    for validate in (validate_algorithm_requirements, validate_archive_gate_pipeline_compat, validate_memory_pipeline_compat, validate_program_format_pipeline_compat):
        validate(cfg)
    instances = instantiate(cfg, _recursive_=True)
    storage, runner, engine, writer = instances.program_storage, instances.dag_runner, instances.evolution_engine, instances.writer
    status: dict[str, Any] = {"status": "starting", "phase": "evolution", "evolution_status": "running", "pid": os.getpid(), "run_id": args.run_id, "model": args.model, "resolution": args.resolution, "started_at": datetime.now(timezone.utc).isoformat(), "target_mutations": args.mutations, "target_candidates": args.mutations + args.initial_candidates}
    failed = False
    pending_ids: set[str] = set()
    supervision = None

    async def publish_status() -> None:
        """Publish liveness and native engine progress without running inference."""
        status["updated_at"] = datetime.now(timezone.utc).isoformat()
        status["candidates"] = await storage.size()
        status["done_candidates"] = await storage.count_by_status("done")
        status["discarded_candidates"] = await storage.count_by_status("discarded")
        status["completed_candidates"] = status["done_candidates"] + status["discarded_candidates"]
        status["mutations_created"] = engine.metrics.mutations_created
        status["parent_refreshes"] = engine.metrics.submitted_for_refresh
        write_json(args.run_dir / "run_status.json", status)

    async def supervise() -> None:
        """Stop on infrastructure failure and otherwise update a five-second heartbeat."""
        while True:
            waiting = []
            failure = infrastructure_failure(gateways)
            if failure is not None:
                raise failure
            for gateway in gateways:
                with gateway.engine.cooldown_lock:
                    if gateway.engine.cooldowns:
                        waiting.append(gateway.engine.config.default_role)
            status["status"] = "waiting" if waiting else "running"
            status["waiting_for"] = waiting
            await publish_status()
            await asyncio.sleep(5)

    try:
        await storage.acquire_instance_lock()
        if await storage.has_data():
            if not args.resume:
                raise RuntimeError("Run directory already contains an experiment; use --resume or a new directory")
            await storage.recover_stranded_programs()
            await engine.restore_state()
            await repair_interrupted_archive_members(engine.strategy, storage)
            await engine.strategy.restore_state()
        else:
            programs = await instances.program_loader.load(storage)
            engine.metrics.iteration = len(programs)
        status["status"] = "running"
        await publish_status()
        runner.start()
        engine.start()
        supervision = asyncio.create_task(supervise())
        done, _ = await asyncio.wait([engine.task, runner.task, supervision], return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
        if runner.task in done:
            raise RuntimeError("GigaEvo DAG runner stopped unexpectedly")
        status["evolution_status"] = "completed"
        status["status"] = "completed" if args.skip_final_evaluation else "running"
        status["phase"] = "completed" if args.skip_final_evaluation else "evolution_completed"
    except BaseException as exc:
        failed = True
        if isinstance(exc, Exception):
            exc = infrastructure_failure(gateways) or exc
        resumable = getattr(exc, "resume_reason", None) is not None
        status["status"] = "resumable" if resumable else "interrupted" if isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError)) else "failed"
        status["evolution_status"] = status["status"]
        status["error"] = str(exc)
        if resumable:
            status["resume_reason"] = exc.resume_reason
            status["resume_required"] = True
        pending_ids = set(await storage.get_ids_by_status(ProgramState.RUNNING.value)) | set(await storage.get_ids_by_status(ProgramState.QUEUED.value))
        raise exc
    finally:
        if supervision:
            supervision.cancel()
            await asyncio.gather(supervision, return_exceptions=True)
        await asyncio.gather(engine.stop(), runner.stop(), return_exceptions=True)
        await default_exec_runner_pool().shutdown()
        if failed:
            await requeue_interrupted(storage, pending_ids)
        try:
            summary = await export_programs(storage, args.run_dir)
            status["best"] = summary
            await publish_status()
        finally:
            await storage.close()
            writer.close()
            GATEWAYS.clear()
            EVALUATION_FAILURES.clear()
    return summary
