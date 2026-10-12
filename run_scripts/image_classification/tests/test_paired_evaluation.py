"""Exercise automatic train-selected paired testing and interruption recovery without external APIs."""

import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from med_evolve import integration
from gigaevo.programs.core_types import ProgramStageResult
from gigaevo.programs.program import Lineage, Program
from gigaevo.programs.program_state import ProgramState
from gigaevo.programs.stages.common import Box
from problems.image_classification import validate
from med_evolve.launcher import main, parser, prepare_arguments


@pytest.fixture
def paired_runtime(monkeypatch):
    """Use the real launcher, gateway, task HTTP and exports around one bounded fake search."""
    state = SimpleNamespace(requests=[], fail_evolved_test=False, search_calls=0, stage="train")

    class Handler(BaseHTTPRequestHandler):
        """Return audited mock image predictions or an intentional final-only failure."""

        def do_POST(self):
            """Capture each real HTTP attempt with its stage and candidate role."""
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            # Private X-Evolve audit headers correctly stop at the gateway boundary.
            stage = state.stage
            prompt = payload["messages"][1]["content"][0]["text"]
            role = "evolved" if "evolved" in prompt else "initial"
            state.requests.append({"stage": stage, "role": role, "model": payload["model"]})
            failure = state.fail_evolved_test and stage == "test" and role == "evolved"
            prediction = json.dumps({"class_id": 0, "class_name": "normal"})
            response = {"error": {"message": "Intentional offline candidate rejection"}} if failure else {"id": "paired-mock", "model": payload["model"], "choices": [{"message": {"content": prediction}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}}
            body = json.dumps(response).encode("utf-8")
            self.send_response(400 if failure else 200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            """Suppress local HTTP access logs."""
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state.base_url = f"http://127.0.0.1:{server.server_port}/v1"

    async def searched(args, cfg, gateways):
        """Evaluate one initial and one descendant on train and export actual native Program objects."""
        state.search_calls += 1
        state.stage = "train"
        programs = []
        for ordinal, role in enumerate(("initial", "evolved")):
            candidate = {"prompt": f"Classify this image: {role} candidate.", "sampling": {"temperature": 0.3 if ordinal else 0.0, "top_p": 0.9 if ordinal else 1.0}}
            metrics = await asyncio.to_thread(validate.evaluate_candidate, candidate, split="train")
            program = Program(code="raise AssertionError('export must not execute Python')", iteration=ordinal, atomic_counter=10 + ordinal, state=ProgramState.DISCARDED if ordinal else ProgramState.DONE, lineage=Lineage(parents=[programs[0].id] if ordinal else [], generation=ordinal + 1), metrics={key: metrics[key] for key in ("fitness", "is_valid", "parse_success_rate")})
            program.stage_results["CallProgramFunction"] = ProgramStageResult.success(output=Box(data=candidate))
            programs.append(program)

        async def get_all():
            """Expose already evaluated programs to the production exporter."""
            return programs

        summary = await integration.export_programs(SimpleNamespace(get_all=get_all), args.run_dir)
        integration.write_json(args.run_dir / "run_status.json", {"status": "completed" if args.skip_final_evaluation else "running", "evolution_status": "completed", "phase": "evolution_completed", "mutations_created": 1, "completed_candidates": 2, "run_id": args.run_id})
        state.stage = "test"
        return summary

    monkeypatch.setattr(integration, "run_native_engine", searched)
    with patch.dict(os.environ):
        os.environ["EVALUATOR_API_KEY"] = "offline-mock-key"
        os.environ["MUTATOR_API_KEY"] = "offline-mock-key"
        os.environ["MEDGEMMA_CONCURRENCY"] = "2"
        try:
            yield state
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


def command(directory, runtime, *extra):
    """Build a small production CLI invocation using only the loopback endpoint."""
    return ["--run-dir", str(directory), "--model", "4b", "--resolution", "128", "--smoke", "--max-images", "2", "--final-repeats", "1", "--retry-budget-seconds", "0.3", "--request-timeout-seconds", "1", "--evaluator-base-url", runtime.base_url, "--mutator-base-url", runtime.base_url, *extra]


def test_paired_default_budgets_and_skip_flag(tmp_path):
    """Use the user's two smoke and three full test repeats, with explicit overrides available."""
    full = prepare_arguments(parser().parse_args(["--run-dir", str(tmp_path)]))
    smoke = prepare_arguments(parser().parse_args(["--run-dir", str(tmp_path), "--smoke", "--max-images", "4"]))
    override = prepare_arguments(parser().parse_args(["--run-dir", str(tmp_path), "--final-repeats", "7", "--final-max-images", "8", "--skip-final-evaluation"]))
    assert (full.final_repeats, full.final_max_images, full.skip_final_evaluation) == (3, 624, False)
    assert (smoke.final_repeats, smoke.final_max_images) == (2, 4)
    assert (override.final_repeats, override.final_max_images, override.skip_final_evaluation) == (7, 8, True)


def test_auto_paired_evaluation_aggregates_once_and_completed_resume_is_free(tmp_path, paired_runtime):
    """Train and both final roles share one ledger; completed resume calls neither search nor HTTP."""
    assert main(command(tmp_path, paired_runtime)) == 0
    stages = [(row["stage"], row["role"]) for row in paired_runtime.requests]
    assert stages.count(("train", "initial")) == stages.count(("train", "evolved")) == 2
    assert stages.count(("test", "initial")) == stages.count(("test", "evolved")) == 2
    assert stages[:4] == [("train", "initial")] * 2 + [("train", "evolved")] * 2
    totals = json.loads((tmp_path / "gateway_evaluator/gateway_tokens.json").read_text())["totals"]
    assert (totals["attempts"], totals["input_tokens"], totals["output_tokens"]) == (8, 88, 56)
    events = [json.loads(line) for line in (tmp_path / "gateway_evaluator/gateway_events.jsonl").read_text().splitlines()]
    attempts = [event for event in events if event["event"] == "request_attempt_started"]
    assert sum(event["stage"] == "train" for event in attempts) == 4
    assert sum(event["stage"] == "test" for event in attempts) == 4
    assert not list((tmp_path / "paired_evaluation").rglob("gateway_tokens.json"))
    pair = json.loads((tmp_path / "paired_evaluation/summary.json").read_text())
    assert pair["roles"]["evolved"]["selection"]["state"] == "discarded"
    assert pair["roles"]["evolved"]["test"]["candidate"]["sampling"] == {"temperature": 0.3, "top_p": 0.9}
    assert (tmp_path / "best_initial/candidate.json").exists()
    assert (tmp_path / "best_evolved/candidate.json").exists()
    assert main(command(tmp_path, paired_runtime, "--resume")) == 0
    assert paired_runtime.search_calls == 1
    assert len(paired_runtime.requests) == 8
    resumed = json.loads((tmp_path / "paired_evaluation/summary.json").read_text())
    assert all(role["test"]["reused_completed_repeats"] == [0] for role in resumed["roles"].values())
    status = json.loads((tmp_path / "run_status.json").read_text())
    assert (status["status"], status["evolution_status"], status["completed_final_roles"]) == ("completed", "completed", ["initial", "evolved"])
    assert main(command(tmp_path, paired_runtime, "--resume", "--final-repeats", "2")) == 1
    assert paired_runtime.search_calls == 1
    assert len(paired_runtime.requests) == 8


def test_failed_final_role_resumes_without_repeating_train_or_completed_role(tmp_path, paired_runtime):
    """Retain a successful first role when the second fails, then resume only the unfinished role."""
    paired_runtime.fail_evolved_test = True
    assert main(command(tmp_path, paired_runtime)) == 1
    status = json.loads((tmp_path / "run_status.json").read_text())
    assert (status["phase"], status["evolution_status"], status["completed_final_roles"]) == ("failed_final_evaluation", "completed", ["initial"])
    assert (tmp_path / "paired_evaluation/initial/final_evaluation/repeat_0/metrics.json").exists()
    assert not (tmp_path / "paired_evaluation/evolved/final_evaluation/repeat_0/metrics.json").exists()
    before = len(paired_runtime.requests)
    paired_runtime.fail_evolved_test = False
    assert main(command(tmp_path, paired_runtime, "--resume")) == 0
    assert paired_runtime.search_calls == 1
    assert [(row["stage"], row["role"]) for row in paired_runtime.requests[before:]] == [("test", "evolved")] * 2
    pair = json.loads((tmp_path / "paired_evaluation/summary.json").read_text())
    assert pair["roles"]["initial"]["test"]["reused_completed_repeats"] == [0]
    assert pair["roles"]["evolved"]["test"]["reused_completed_repeats"] == []


def test_skip_then_standalone_pair_uses_source_model_and_no_new_search(tmp_path, paired_runtime):
    """Preserve search-only operation and later run the historical pair selector via evaluate.py."""
    source = tmp_path / "source"
    assert main(command(source, paired_runtime, "--skip-final-evaluation")) == 0
    assert len(paired_runtime.requests) == 4
    assert not (source / "paired_evaluation").exists()
    from evaluate import main as evaluate

    destination = tmp_path / "standalone"
    assert evaluate(["--evolution-run", str(source), "--run-dir", str(destination), "--final-repeats", "1", "--final-max-images", "2", "--retry-budget-seconds", "0.3", "--request-timeout-seconds", "1"]) == 0
    assert paired_runtime.search_calls == 1
    assert len(paired_runtime.requests) == 8
    config = json.loads((destination / "run_config.json").read_text())
    assert (config["model"], config["resolution"], config["evaluator_model"]) == ("4b", 128, "google/medgemma-1.5-4b-it")


def test_failed_search_never_starts_final_evaluation(tmp_path, paired_runtime, monkeypatch):
    """An interrupted search must not consume held-out evaluation requests."""
    async def failed_search(*args):
        """Represent an evolution failure before any result is available."""
        raise RuntimeError("Intentional offline search failure")

    monkeypatch.setattr(integration, "run_native_engine", failed_search)
    assert main(command(tmp_path, paired_runtime)) == 1
    assert not paired_runtime.requests
    assert not (tmp_path / "paired_evaluation").exists()
