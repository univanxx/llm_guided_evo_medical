"""Verify historical train-only paired selection without executing code or model calls."""

import json
from pathlib import Path

import pytest

from med_evolve.final_selection import select_pair


def write_json(path, value):
    """Create a portable test artifact with the same JSON shape as an exported run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def candidate_record(program_id, fitness, *, generation=1, parents=None, iteration=0, atomic_counter=1, state="done", is_valid=1, **extra):
    """Create a candidate record with explicit lineage independent of seed ordinal."""
    return {"id": program_id, "iteration": iteration, "generation": generation, "parents": parents or [], "atomic_counter": atomic_counter, "state": state, "metrics": {"fitness": fitness, "is_valid": is_valid}, **extra}


def make_run(tmp_path, run_id, records, *, configs=None, **settings):
    """Save real exported candidates while keeping source Python deliberately nonexecutable."""
    run_dir = tmp_path / run_id
    config = {"run_id": run_id, "model": "4b", "resolution": 128, "evaluator_provider": "compatible", "evaluator_model": "google/medgemma-1.5-4b-it", "evaluator_base_url": "http://127.0.0.1:8000/v1", "mutator_provider": "compatible", "mutator_model": "gpt-oss-120b", "max_images": 100, "repeats": 1, "search_space": "joint", "inference_seed": None, "enable_top_k": False, "seed": 42, **settings}
    config.setdefault("task_identity", {"schema_version": 1, "sha256": "same-frozen-task"})
    protocol_values = {"mutations": config.get("mutations", 8), "parallel": config.get("parallel", 2), "initial_candidates": config.get("initial_candidates", 10), "mutator_temperature": config.get("mutator_temperature", 0.6), "mutator_max_tokens": config.get("mutator_max_tokens", 8192), "optimization_mode": config.get("optimization_mode", config["search_space"].replace("-", "_")), "enable_top_k": config["enable_top_k"], "train_repeats": config["repeats"], "train_max_images": config["max_images"], "final_repeats": 3, "final_max_images": 624}
    config.setdefault("protocol_identity", {"schema_version": 1, "values": protocol_values, "sha256": "test-protocol"})
    write_json(run_dir / "run_config.json", config)
    write_json(run_dir / "run_status.json", {"status": "completed"})
    for record in records:
        directory = run_dir / "candidates" / f"{record['iteration']:04d}_{record['id']}"
        record["path"] = str(directory)
        candidate = (configs or {}).get(record["id"], {"prompt": f"Prompt for {record['id']}", "sampling": {"temperature": 0.0, "top_p": 1.0}})
        write_json(directory / "candidate.json", candidate)
        (directory / "program.py").write_text("raise RuntimeError('Selection must not execute candidate Python')\n", encoding="utf-8")
    write_json(run_dir / "candidates.json", records)
    return run_dir


def test_all_seed_ordinals_and_discarded_descendant_remain_eligible(tmp_path):
    """Later seeds and valid discarded descendants are distinct even with inherited source metadata."""
    records = [candidate_record(f"seed-{index}", index / 10, iteration=index, atomic_counter=index, state="discarded" if index == 9 else "done") for index in range(10)]
    records.append(candidate_record("child", 0.8, generation=2, parents=["seed-9"], iteration=10, state="discarded", metadata={"source": "initial_program"}, source="initial_program"))
    run = make_run(tmp_path, "run", records)
    selected = select_pair([run])["roles"]
    assert selected["initial"]["program_id"] == "seed-9"
    assert selected["evolved"]["program_id"] == "child"
    assert selected["evolved"]["train_fitness"] < selected["initial"]["train_fitness"]


def test_independent_roles_and_historical_ties_across_three_runs(tmp_path):
    """Within-run update order and cross-run lower search seeds define deterministic ties."""
    runs = [make_run(tmp_path, "run-001", [candidate_record("seed-one", 0.6), candidate_record("child-old", 0.9, generation=2, parents=["seed-one"], atomic_counter=9), candidate_record("child-new-a", 0.9, generation=2, parents=["seed-one"], atomic_counter=10), candidate_record("child-new-z", 0.9, generation=2, parents=["seed-one"], atomic_counter=10)], seed=45)]
    runs.append(make_run(tmp_path, "run-002", [candidate_record("seed-two", 0.7), candidate_record("child-two", 0.8, generation=2, parents=["seed-two"])], seed=43))
    runs.append(make_run(tmp_path, "run-003", [candidate_record("seed-three", 0.7), candidate_record("child-three", 0.7, generation=2, parents=["seed-three"])], seed=44))
    selected = select_pair(list(reversed(runs)))["roles"]
    assert selected["initial"]["run_id"] == "run-002"
    assert selected["initial"]["program_id"] == "seed-two"
    assert selected["evolved"]["run_id"] == "run-001"
    assert selected["evolved"]["program_id"] == "child-new-z"


def test_test_results_never_rank_candidates_and_sampling_is_preserved(tmp_path, monkeypatch):
    """Only saved train fitness selects the exact JSON payload, regardless of held-out results."""
    candidate = {"prompt": "Exact saved medical prompt", "sampling": {"temperature": 0.7, "top_p": 0.85, "top_k": 20}}
    records = [candidate_record("seed", 0.5), candidate_record("train-best", 0.8, generation=2, parents=["seed"]), candidate_record("test-best", 0.7, generation=2, parents=["seed"])]
    run = make_run(tmp_path, "run", records, configs={"train-best": candidate}, enable_top_k=True)
    write_json(run / "paired_evaluation" / "metrics.json", {"train-best": {"test_fitness": 0.0}, "test-best": {"test_fitness": 1.0}})
    original_read = Path.read_text

    def read_without_test(path, *args, **kwargs):
        """Fail if selection attempts to inspect held-out evaluation artifacts."""
        assert "paired_evaluation" not in path.parts
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_without_test)
    evolved = select_pair([run])["roles"]["evolved"]
    assert evolved["program_id"] == "train-best"
    assert evolved["candidate"] == candidate
    assert len(evolved["candidate_sha256"]) == 64
    assert Path(evolved["source_candidate_path"]).read_text() == json.dumps(candidate)


@pytest.mark.parametrize("bad", [{"state": "failed"}, {"state": "running"}, {"state": "queued"}, {"is_valid": 0}, {"fitness": float("nan")}, {"fitness": float("inf")}])
def test_invalid_or_unfinished_candidates_cannot_win(tmp_path, bad):
    """Invalid, incomplete and nonfinite metric records never beat proven valid descendants."""
    values = {"fitness": 1.0, **bad}
    run = make_run(tmp_path, "run", [candidate_record("seed", 0.5), candidate_record("valid", 0.6, generation=2, parents=["seed"]), candidate_record("bad", generation=2, parents=["seed"], **values)])
    assert select_pair([run])["roles"]["evolved"]["program_id"] == "valid"


@pytest.mark.parametrize("missing", ["initial", "evolved"])
def test_missing_group_fails_instead_of_substituting_global_best(tmp_path, missing):
    """An absent group is a selection error before any final inference is attempted."""
    record = candidate_record("only", 0.9, generation=2 if missing == "initial" else 1, parents=["missing-parent"] if missing == "initial" else [])
    run = make_run(tmp_path, "run", [record])
    with pytest.raises((ValueError, RuntimeError)):
        select_pair([run])


@pytest.mark.parametrize("field,value", [("model", "27b"), ("resolution", 224), ("evaluator_provider", "openrouter"), ("evaluator_model", "different/model"), ("evaluator_base_url", "https://different.invalid/v1"), ("mutator_provider", "openrouter"), ("mutator_model", "different/mutator"), ("max_images", 4), ("repeats", 3), ("search_space", "prompt_only"), ("inference_seed", 7), ("enable_top_k", True)])
def test_incompatible_experimental_conditions_cannot_be_pooled(tmp_path, field, value):
    """The historical multi-run pool may vary search seed but cannot mix conditions."""
    records = lambda: [candidate_record("seed", 0.5), candidate_record("child", 0.6, generation=2, parents=["seed"])]
    first = make_run(tmp_path, "run-001", records())
    second = make_run(tmp_path, "run-002", records(), **{field: value})
    with pytest.raises((ValueError, RuntimeError)):
        select_pair([first, second])


def test_independent_search_seeds_can_share_one_selection_pool(tmp_path):
    """Repeated evolution seeds remain poolable under otherwise identical conditions."""
    first = make_run(tmp_path, "run-001", [candidate_record("seed-a", 0.6), candidate_record("child-a", 0.7, generation=2, parents=["seed-a"])], seed=41)
    second = make_run(tmp_path, "run-002", [candidate_record("seed-b", 0.5), candidate_record("child-b", 0.8, generation=2, parents=["seed-b"])], seed=43)
    selected = select_pair([first, second])["roles"]
    assert selected["initial"]["program_id"] == "seed-a"
    assert selected["evolved"]["program_id"] == "child-b"


def test_legacy_exports_recover_update_order_from_native_storage(tmp_path):
    """Existing exports use persisted atomic counters before falling back to iteration."""
    records = [candidate_record("seed", 0.5), candidate_record("earlier-iteration", 0.8, generation=2, parents=["seed"], iteration=1), candidate_record("later-iteration", 0.8, generation=2, parents=["seed"], iteration=2)]
    for record in records:
        record.pop("atomic_counter")
    run = make_run(tmp_path, "run", records)
    write_json(run / "storage" / "pneumonia" / "programs" / "earlier-iteration.json", {"atomic_counter": 30})
    write_json(run / "storage" / "pneumonia" / "programs" / "later-iteration.json", {"atomic_counter": 10})
    assert select_pair([run])["roles"]["evolved"]["program_id"] == "earlier-iteration"


def test_downloaded_run_uses_local_payload_despite_remote_absolute_paths(tmp_path):
    """A fetched run remains selectable on either OS after the original server paths disappear."""
    records = [candidate_record("seed", 0.5), candidate_record("child", 0.6, generation=2, parents=["seed"], iteration=1)]
    run = make_run(tmp_path, "downloaded-run", records)
    for record in records:
        record["path"] = "/unavailable-original-server/candidates/" + record["id"]
    write_json(run / "candidates.json", records)
    selected = select_pair([run])["roles"]
    assert selected["evolved"]["program_id"] == "child"
    assert Path(selected["evolved"]["source_candidate_path"]).is_file()
    assert Path(selected["evolved"]["source_candidate_path"]).is_relative_to(run)
