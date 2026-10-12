"""Check experiment budgets, secret loading, resume contracts, and exact exports offline."""

import asyncio
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from med_evolve.launcher import completed_repeat, parser, prepare_arguments, read_credentials, save_manifest


def arguments(tmp_path, *extra):
    """Construct real CLI settings without contacting model providers."""
    return prepare_arguments(parser().parse_args(["--run-dir", str(tmp_path), *extra]))


def test_smoke_and_historical_budgets(tmp_path):
    """The short run stays tiny and the full run keeps the archival descendant budget."""
    full = arguments(tmp_path)
    smoke = arguments(tmp_path, "--smoke")
    explicit = arguments(tmp_path / "explicit", "--smoke", "--initial-candidates", "2")
    assert (full.initial_candidates, full.mutations, full.parallel, full.max_images) == (10, 232, 1, 100)
    assert (smoke.initial_candidates, smoke.mutations, smoke.parallel, smoke.max_images) == (1, 1, 1, 2)
    assert explicit.initial_candidates == 2
    assert full.retry_budget_seconds > 4 * 3600
    assert full.http_timeout > full.retry_budget_seconds + full.request_timeout_seconds
    assert full.validator_timeout > full.max_images * full.http_timeout


def test_default_data_root_is_inside_published_task(tmp_path):
    """Keep the default dataset self-contained within the published task tree."""
    args = arguments(tmp_path)
    assert args.data_root == ROOT.parents[1] / "problems" / "image_classification" / "data"


def test_credentials_are_dotenv_only(tmp_path):
    """Read only explicit dotenv values without importing or executing user files."""
    credential = tmp_path / "credentials.env"
    credential.write_text("EVALUATOR_API_KEY=fixture-key\n", encoding="utf-8")
    assert read_credentials(credential)["EVALUATOR_API_KEY"] == "fixture-key"


def test_resume_rejects_changed_sampling_space(tmp_path):
    """A resumed archive must not silently change model/data/search-space identity."""
    args = arguments(tmp_path)
    save_manifest(args)
    args.resume = True
    save_manifest(args)
    args.enable_top_k = True
    with pytest.raises(ValueError, match="Resume settings differ"):
        save_manifest(args)


@pytest.mark.parametrize("change", ["endpoint", "initial_population"])
def test_resume_rejects_changed_provider_or_seed_population(tmp_path, change):
    """Provider routing and the initial population belong to the archive identity."""
    args = arguments(tmp_path, "--smoke")
    save_manifest(args)
    args.resume = True
    if change == "endpoint":
        args.evaluator_base_url = "https://different.example/v1"
    else:
        args.initial_candidates = 2
    with pytest.raises(ValueError, match="Resume settings differ"):
        save_manifest(args)


def test_final_resume_reuses_completed_resumed_pass(tmp_path):
    """A pass completed by an earlier resume remains reusable after a later failure."""
    path = tmp_path / "final_evaluation" / "repeat_0_resume_20260910" / "metrics.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"repeat": 0, "fitness": 0.5}), encoding="utf-8")
    assert completed_repeat(tmp_path, 0) == {"repeat": 0, "fitness": 0.5}
    assert completed_repeat(tmp_path, 1) is None


def test_final_evaluation_cannot_hide_terminal_telemetry_failure(tmp_path, monkeypatch):
    """A paid response is retained, but a gateway audit failure must prevent completion."""
    import threading
    from med_evolve import integration
    from problems.image_classification import validate
    from med_evolve.launcher import evaluate_saved

    candidate_path = tmp_path / "input.json"
    candidate_path.write_text(json.dumps({"prompt": "Classify the radiograph.", "sampling": {"temperature": 0, "top_p": 1}}), encoding="utf-8")
    args = arguments(tmp_path, "--smoke", "--evaluate", str(candidate_path))
    fatal = threading.Event()
    fatal.set()
    gateway = SimpleNamespace(fatal_event=fatal, fatal_reason="telemetry failure", engine=SimpleNamespace(cooldown_lock=threading.Lock(), cooldowns={}))
    calls = []

    def completed_response(*args, **kwargs):
        """Represent a completed provider response whose audit write then failed."""
        calls.append(kwargs["repeat"])
        return {"repeat": kwargs["repeat"], "fitness": 0.5}

    monkeypatch.setattr(validate, "evaluate_candidate", completed_response)
    with pytest.raises(RuntimeError, match="telemetry failure"):
        evaluate_saved(args, gateway)
    assert calls == [0]
    assert not (tmp_path / "summary.json").exists()


def test_config_extends_every_outer_inference_timeout(tmp_path):
    """Native DAG, SDK, parent-refresh, and terminal-drain budgets must preserve retries."""
    from gigaevo.programs.stages.cache_handler import NO_CACHE
    from med_evolve.integration import OutageAwareValidator, build_config

    args = arguments(tmp_path, "--smoke", "--initial-candidates", "2")
    config = build_config(args, "http://127.0.0.1:9000/v1")
    assert config.program_loader.limit == 2
    assert config.stopper.max_mutants == 1
    assert config.llm.models[0].max_retries == 0
    assert config.request_timeout == args.http_timeout
    assert config.stage_timeout > args.http_timeout
    assert config.dag_timeout > config.stage_timeout
    assert config.evolution_engine.parent_refresh_timeout > config.dag_timeout
    assert config.post_cap_drain_grace_s is None
    assert config.memory.capabilities.read is False
    assert config.memory.capabilities.write is False
    assert OutageAwareValidator.cache_handler is NO_CACHE


def test_validator_pass_id_distinguishes_refresh_from_crash_window(tmp_path):
    """A recovered RUNNING program keeps its pass; a normal refresh gets a new one."""
    from med_evolve.integration import OutageAwareValidator
    from gigaevo.programs.program import Program
    from gigaevo.programs.program_state import ProgramState

    class Storage:
        async def update(self, program):
            return None

    program = Program(code="def entrypoint(): return {}", state=ProgramState.RUNNING)
    stage = OutageAwareValidator(
        storage=Storage(),
        path=ROOT.parents[1] / "problems" / "image_classification" / "validate.py",
        function_name="validate",
        timeout=30,
    )
    first = asyncio.run(stage.begin_evaluation_pass(program))
    marker = dict(program.metadata["pneumonia_evaluation_pass"])
    marker["state"] = "complete"
    program.metadata["pneumonia_evaluation_pass"] = marker
    program.metadata["evaluation_interrupted"] = True
    recovered = asyncio.run(stage.begin_evaluation_pass(program))
    normal_refresh = asyncio.run(stage.begin_evaluation_pass(program))
    assert recovered == first
    assert normal_refresh != first and normal_refresh.endswith(":train:2")
    from gigaevo.programs.stages.common import Box

    stage.attach_inputs({"payload": Box(data={})})
    call_args, call_kwargs = stage._build_call(program)
    assert call_args == [{}]
    assert call_kwargs == {"evaluation_instance_id": normal_refresh}


def test_failed_validator_pass_stays_running_for_exact_resume(tmp_path, monkeypatch):
    """A validator contract failure must not publish an incomplete pass as reusable."""
    from gigaevo.programs.core_types import ProgramStageResult, StageError
    from gigaevo.programs.program import Program
    from gigaevo.programs.stages.validator_metadata import ProgramMetadataValidatorStage
    from med_evolve.integration import OutageAwareValidator

    class Storage:
        async def update(self, program):
            return None

    failure = ProgramStageResult.failure(error=StageError(type="ValidatorContractError", message="invalid", stage="validator"))

    async def failed_compute(self, program):
        return failure

    monkeypatch.setattr(ProgramMetadataValidatorStage, "compute", failed_compute)
    program = Program(code="pass")
    stage = OutageAwareValidator(storage=Storage(), path=ROOT.parents[1] / "problems" / "image_classification" / "validate.py", function_name="validate", timeout=30)
    assert asyncio.run(stage.compute(program)) is failure
    assert program.metadata["pneumonia_evaluation_pass"]["state"] == "running"


def test_export_uses_evaluated_payload_not_python_execution(tmp_path):
    """Persist the exact prompt and decoder even if re-executing source would fail."""
    from gigaevo.programs.core_types import ProgramStageResult
    from gigaevo.programs.program import Program
    from gigaevo.programs.program_state import ProgramState
    from gigaevo.programs.stages.common import Box
    from med_evolve.integration import export_programs

    candidate = {"prompt": "Classify the radiograph.", "sampling": {"temperature": 0.3, "top_p": 0.9}}
    program = Program(code="raise AssertionError('must not execute during export')", state=ProgramState.DONE, metrics={"fitness": 0.5, "is_valid": 1})
    program.stage_results["CallProgramFunction"] = ProgramStageResult.success(output=Box(data=candidate))

    async def get_all():
        """Return one already evaluated native program."""
        return [program]

    asyncio.run(export_programs(SimpleNamespace(get_all=get_all), tmp_path))
    assert json.loads((tmp_path / "best" / "candidate.json").read_text(encoding="utf-8")) == candidate


def test_native_config_builds_every_stage_without_network(tmp_path):
    """Build actual upstream objects so API signature drift fails before paid inference."""
    from hydra.utils import instantiate
    from med_evolve.integration import build_config

    config = build_config(arguments(tmp_path, "--smoke"), "http://127.0.0.1:1/v1")
    instances = instantiate(config, _recursive_=True)
    try:
        dag = instances.dag_blueprint.build(instances.evolution_engine.state, instances.writer)
        dag.teardown()
        assert instances.llm._structured_output_method == "auto"
    finally:
        asyncio.run(instances.program_storage.close())
        instances.writer.close()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific compatibility bridge")
def test_windows_bridge_uses_real_file_locks(tmp_path):
    """A second descriptor must fail a nonblocking lock until the first unlocks."""
    from med_evolve.windows_compat import install_windows_flock

    install_windows_flock()
    import fcntl

    path = tmp_path / "lock"
    with path.open("a+b") as first, path.open("a+b") as second:
        fcntl.flock(first.fileno(), fcntl.LOCK_EX)
        with pytest.raises(BlockingIOError):
            fcntl.flock(second.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(first.fileno(), fcntl.LOCK_UN)
        fcntl.flock(second.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(second.fileno(), fcntl.LOCK_UN)


def test_gateway_failure_cannot_reach_fitness(monkeypatch):
    """A terminal infrastructure failure keeps the validator pending until supervisor cancellation."""
    import threading
    from gigaevo.programs.program import Program
    from gigaevo.programs.stages.validator_metadata import ProgramMetadataValidatorStage
    from med_evolve.integration import GATEWAYS, OutageAwareValidator

    async def failed_compute(self, program):
        """Stand in for a completed subprocess that reported an infrastructure error."""
        return object()

    monkeypatch.setattr(ProgramMetadataValidatorStage, "compute", failed_compute)
    event = threading.Event()
    event.set()
    GATEWAYS[:] = [SimpleNamespace(fatal_event=event)]

    async def verify_pending():
        """Only cancellation may finish an outage-blocked validator."""
        stage = OutageAwareValidator(storage=SimpleNamespace(update=lambda program: asyncio.sleep(0)), path=ROOT.parents[1] / "problems" / "image_classification" / "validate.py", function_name="validate", timeout=99999)
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(stage.compute(Program(code="pass")), timeout=0.02)

    try:
        asyncio.run(verify_pending())
    finally:
        GATEWAYS.clear()


def test_interrupted_candidate_is_requeued_without_fitness(tmp_path):
    """Persisted interrupted work resumes from the same candidate without an outage score."""
    from gigaevo.database.disk_program_storage import DiskProgramStorage, DiskProgramStorageConfig
    from gigaevo.programs.program import Program
    from gigaevo.programs.program_state import ProgramState
    from med_evolve.integration import requeue_interrupted

    async def exercise_storage():
        """Round-trip the native durable state through the interruption adapter."""
        storage = DiskProgramStorage(DiskProgramStorageConfig(root_dir=str(tmp_path), key_prefix="pneumonia"))
        program = Program(code="def entrypoint(): return 'prompt'", state=ProgramState.RUNNING, metrics={"fitness": 0.5})
        await storage.add(program)
        await requeue_interrupted(storage, {program.id})
        recovered = await storage.get(program.id)
        assert recovered.state == ProgramState.QUEUED
        assert recovered.metrics == {}
        assert recovered.stage_results == {}
        await storage.close()

    asyncio.run(exercise_storage())
