"""Verify model/resolution CLI choices through real local HTTP and persisted evaluation artifacts."""

import base64
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from med_evolve.launcher import configure_evaluator, parser, prepare_arguments, save_manifest
from med_evolve.integration import build_config
from problems.image_classification import helper, validate


EXPECTED_MODELS = {"4b": "google/medgemma-1.5-4b-it", "27b": "google/medgemma-27b-it"}
COMBINATIONS = [(model, resolution) for resolution in (28, 64, 128, 224) for model in ("4b", "27b")]


@pytest.fixture(scope="module")
def local_endpoint():
    """Serve deterministic responses and retain actual request bodies without external inference."""
    received = []

    class Handler(BaseHTTPRequestHandler):
        """Capture the real validator HTTP request and echo its selected model."""

        def do_POST(self):
            """Return a predictable valid response for the audited source class."""
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            headers = {key.lower(): value for key, value in self.headers.items()}
            received.append({"path": self.path, "payload": payload, "headers": headers})
            positive = "/01_pneumonia/" in headers["x-evolve-image-id"]
            content = json.dumps({"class_id": int(positive), "class_name": "pneumonia" if positive else "normal"})
            body = json.dumps({"id": "local-options-test", "model": payload["model"], "choices": [{"message": {"content": content}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            """Suppress access logging from the local test server."""
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield SimpleNamespace(base_url=f"http://127.0.0.1:{server.server_port}/v1", received=received)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.fixture(autouse=True)
def isolated_experiment_environment():
    """Restore all launcher environment writes after each test while retaining the image cache."""
    with patch.dict(os.environ):
        for key in list(os.environ):
            if key.startswith(("PNEUMONIA_", "MEDGEMMA_", "EVOLVE_")):
                os.environ.pop(key)
        os.environ["MEDGEMMA_CONCURRENCY"] = "2"
        yield


def check_selected_options(directory, model, resolution, endpoint, override=None):
    """Follow CLI selection to the wire and compare transmitted pixels with the chosen source."""
    extra = ["--evaluator-model", override] if override else []
    args = prepare_arguments(parser().parse_args(["--model", model, "--resolution", str(resolution), "--run-dir", str(directory), "--smoke", "--max-images", "2", *extra]))
    configure_evaluator(args, endpoint.base_url)
    save_manifest(args)
    settings = helper.get_default_settings()
    expected_model = override or EXPECTED_MODELS[model]
    expected_root = args.data_root / f"r{resolution}"
    assert settings["model_choice"] == model
    assert settings["model"] == expected_model
    assert settings["resolution"] == resolution
    assert Path(settings["data_root"]) == expected_root
    assert settings["resize"] == 896
    native_config = build_config(args, endpoint.base_url)
    assert native_config.llm.models[0].model == "gpt-oss-120b"
    assert native_config.llm.models[0].default_headers["X-Evolve-Role"] == "mutator"
    before = len(endpoint.received)
    candidate = {"prompt": "Classify this chest radiograph using class_id and class_name.", "sampling": {"temperature": 0.3, "top_p": 0.9}}
    metrics = validate.evaluate_candidate(candidate, split="train", output_dir=directory / "evaluation")
    requests = endpoint.received[before:]
    config = json.loads((directory / "evaluation" / "config.json").read_text(encoding="utf-8"))
    manifest = json.loads((directory / "run_config.json").read_text(encoding="utf-8"))
    predictions = [json.loads(line) for line in (directory / "evaluation" / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(requests) == len(predictions) == metrics["total"] == 2
    assert metrics["accuracy"] == metrics["parse_success_rate"] == 1
    assert {prediction["true_id"] for prediction in predictions} == {0, 1}
    assert (manifest["model"], manifest["resolution"], manifest["evaluator_model"]) == (model, resolution, expected_model)
    assert manifest["mutator_model"] == "gpt-oss-120b"
    assert config["settings"]["model"] == expected_model
    assert config["settings"]["resolution"] == resolution
    assert Path(config["settings"]["data_root"]) == expected_root
    assert set(config["selected_image_paths"]) == {row["rel_path"] for row in predictions}
    assert config["dataset_manifest_sha256"] == hashlib.sha256((expected_root / "train" / "manifest.jsonl").read_bytes()).hexdigest()
    request_by_image = {request["headers"]["x-evolve-image-id"]: request for request in requests}
    evidence = []
    for prediction in predictions:
        assert (prediction["model"], prediction["response_model"], prediction["resolution"]) == (expected_model, expected_model, resolution)
        assert prediction["image_id"] == f"r{resolution}/train/{prediction['rel_path']}"
        source_path = expected_root / "train" / prediction["rel_path"]
        source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
        assert prediction["image_sha256"] == source_hash
        request = request_by_image[prediction["image_id"]]
        assert request["path"] == "/v1/chat/completions"
        assert request["headers"]["x-evolve-role"] == "evaluator"
        assert request["payload"]["model"] == expected_model
        assert request["payload"]["model"] != native_config.llm.models[0].model
        assert request["payload"]["temperature"] == 0.3
        assert request["payload"]["top_p"] == 0.9
        content = request["payload"]["messages"][1]["content"]
        assert content[0]["text"] == candidate["prompt"]
        prefix, encoded = content[1]["image_url"]["url"].split(",", 1)
        assert prefix == "data:image/png;base64"
        with Image.open(source_path) as source, Image.open(io.BytesIO(base64.b64decode(encoded))) as transmitted:
            assert source.size == (resolution, resolution)
            assert transmitted.size == (896, 896)
            expected_pixels = source.convert("RGB").resize((896, 896), Image.Resampling.BICUBIC)
            assert transmitted.convert("RGB").tobytes() == expected_pixels.tobytes()
            evidence.append({"rel_path": prediction["rel_path"], "source_path": str(source_path), "source_sha256": source_hash, "source_size": list(source.size), "transmitted_size": list(transmitted.size), "transmitted_pixel_sha256": hashlib.sha256(transmitted.tobytes()).hexdigest(), "request_model": request["payload"]["model"], "request_role": request["headers"]["x-evolve-role"]})
    (directory / "options_verification.json").write_text(json.dumps({"model_choice": model, "resolution": resolution, "evaluator_model": expected_model, "mutator_model": native_config.llm.models[0].model, "images": evidence}, indent=2) + "\n", encoding="utf-8")
    return sorted(evidence, key=lambda row: row["rel_path"])


@pytest.mark.parametrize("model,resolution", COMBINATIONS, ids=[f"{model}-r{resolution}" for model, resolution in COMBINATIONS])
def test_all_model_resolution_options_reach_http_and_artifacts(tmp_path, local_endpoint, model, resolution):
    """Exercise each of the eight public options against frozen data and actual HTTP serialization."""
    check_selected_options(tmp_path / "run", model, resolution, local_endpoint)


def test_switching_options_in_one_process_keeps_data_and_model_separate(tmp_path, local_endpoint):
    """Change both choices repeatedly without clearing the image cache or process environment."""
    sequence = [("4b", 28), ("27b", 64), ("4b", 128), ("27b", 224), ("27b", 28), ("4b", 64), ("27b", 128), ("4b", 224), ("4b", 28)]
    records = [check_selected_options(tmp_path / f"step_{index}", model, resolution, local_endpoint) for index, (model, resolution) in enumerate(sequence)]
    for image_index in (0, 1):
        assert len({rows[image_index]["source_path"] for rows in records[:4]}) == 4
        assert len({rows[image_index]["source_sha256"] for rows in records[:4]}) == 4
        assert len({rows[image_index]["transmitted_pixel_sha256"] for rows in records[:4]}) == 4
    assert records[0] == records[-1]


def test_explicit_evaluator_model_override_keeps_selected_data(tmp_path, local_endpoint):
    """An explicit provider model ID overrides only evaluator routing, preserving 4B/r64 experiment labels."""
    check_selected_options(tmp_path / "override", "4b", 64, local_endpoint, override="provider/explicit-vision-model")


@pytest.mark.parametrize("invalid", [("--model", "unknown"), ("--resolution", "32"), ("--resolution", "invalid")])
def test_unsupported_cli_options_fail_before_http(tmp_path, local_endpoint, invalid):
    """Reject unknown choices and malformed resolution values before evaluation can issue requests."""
    before = len(local_endpoint.received)
    with pytest.raises(SystemExit) as error:
        parser().parse_args(["--run-dir", str(tmp_path), *invalid])
    assert error.value.code == 2
    assert len(local_endpoint.received) == before
