"""Verify the unified task contract, archived data, request settings, and fatal transport behavior."""

import hashlib
import json
import os
import runpy
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT.parents[1] / "problems" / "image_classification"
DATA_ROOT = TASK / "data"
sys.path.insert(0, str(ROOT))
from problems.image_classification import helper, validate


class TaskTests(unittest.TestCase):
    """Exercise contract and transport invariants without external model availability."""

    def test_candidate_contract(self):
        """Whitelist exact values, reject runtime overrides, and normalize greedy candidates."""
        expected = {"prompt": "x", "sampling": {"temperature": 0.0, "top_p": 1.0}}
        self.assertEqual(helper.normalize_program_output("x"), expected)
        self.assertEqual(helper.normalize_program_output({"prompt": "x", "sampling": {"temperature": 0, "top_p": 0.9}}), expected)
        for invalid in ({"prompt": "x", "model": "other"}, {"prompt": ""}, {"prompt": "x", "sampling": {"temperature": True}}, {"prompt": "x", "sampling": {"temperature": 0.2}}, {"prompt": "x", "sampling": {"max_tokens": 10}}, {"prompt": "x", "sampling": {"top_k": 20}}, {"prompt": "x", "sampling": {"top_p": float("nan")}}):
            with self.assertRaises(ValueError):
                helper.normalize_program_output(invalid, top_k_enabled=False)
        self.assertEqual(helper.normalize_program_output({"prompt": "x", "sampling": {"temperature": .3, "top_p": .9, "top_k": 20}}, top_k_enabled=True)["sampling"]["top_k"], 20)

    def test_prompt_only_control(self):
        """Enforce the exact greedy baseline before canonicalization in prompt-only runs."""
        with patch.dict(os.environ, {"PNEUMONIA_SEARCH_SPACE": "prompt-only", "PNEUMONIA_ENABLE_TOP_K": "1"}):
            settings = helper.get_default_settings()
            self.assertEqual(settings["search_space"], "prompt-only")
            self.assertEqual(helper.normalize_program_output("x")["sampling"], {"temperature": 0.0, "top_p": 1.0})
            for sampling in ({"temperature": .3}, {"temperature": 0, "top_p": .9}, {"temperature": 0, "top_k": 20}):
                with self.assertRaises(ValueError):
                    helper.normalize_program_output({"prompt": "x", "sampling": sampling})
                self.assertEqual(validate.validate({"prompt": "x", "sampling": sampling})["is_valid"], 0)

    def test_dataset_and_initials(self):
        """Verify copied image content and preservation of the original ten prompt strings."""
        reference = json.loads((DATA_ROOT / "initial_prompts.json").read_text(encoding="utf-8"))
        initials = list((TASK / "initial_programs").glob("*.py"))
        self.assertEqual(len(initials), 10)
        for path in initials:
            prompt = runpy.run_path(str(path))["entrypoint"]()["prompt"]
            self.assertEqual(hashlib.sha256(prompt.encode("utf-8")).hexdigest(), reference[path.name]["prompt_sha256"])
        for resolution in (28, 64, 128, 224):
            for split, count in (("train", 100), ("test", 624)):
                folder = DATA_ROOT / f"r{resolution}" / split
                rows = helper.load_dataset(folder)
                self.assertEqual(len(rows), count)
                self.assertEqual([row["label_id"] for row in helper.load_dataset(folder, 2)], [0, 1])
                for row in rows:
                    self.assertEqual(hashlib.sha256((folder / row["rel_path"]).read_bytes()).hexdigest(), row["sha256"])

    def test_prediction_parser_and_scoring(self):
        """Invalid class outputs remain in the denominator and penalize sensitivity."""
        self.assertEqual(helper.parse_prediction('```json\n{"class_id":1,"class_name":"pneumonia"}\n```')["class_id"], 1)
        for response in ('{"class_id":2,"class_name":"pneumonia"}', '{"class_id":true,"class_name":"pneumonia"}', '{"class_id":1,"class_name":"normal"}', '{"class_id":"1","class_name":"pneumonia"}'):
            with self.assertRaises(ValueError):
                helper.parse_prediction(response)
        result = validate.score_results([{"true_id": 0, "pred_id": 0, "correct": 1, "usage": None}, {"true_id": 1, "correct": 0, "parse_error": "invalid", "usage": None}])
        self.assertEqual(result["fitness"], .5)
        self.assertEqual(result["sensitivity"], 0)
        self.assertEqual(result["requests_missing_usage"], 2)

    def test_exact_normal_alias(self):
        """Accept the original prompt's exact normal synonym without weakening class consistency."""
        self.assertEqual(helper.parse_prediction('{"class_id":0,"class_name":"normal (no pneumonia)"}'), {"class_id": 0, "class_name": "normal"})
        for response in ('{"class_id":1,"class_name":"normal (no pneumonia)"}', '{"class_id":0,"class_name":"normal (healthy)"}'):
            with self.assertRaises(ValueError):
                helper.parse_prediction(response)

    def test_repeated_json_and_ambiguous_fragments(self):
        """Recover identical full answers while rejecting conflicts and truncated nested fragments."""
        normal = '{"class_id":0,"class_name":"normal"}'
        duplicate = f"Output: `{normal}`<unused95>```json\n{normal}\n```"
        self.assertEqual(helper.parse_prediction(duplicate), {"class_id": 0, "class_name": "normal"})
        incomplete_tail = f"Output: `{normal}`<unused95>```json\n" + '{"class_id":0,"class_name'
        self.assertEqual(helper.parse_prediction(incomplete_tail)["class_id"], 0)
        self.assertEqual(helper.parse_prediction(normal + '\n{"class_id":0,')["class_id"], 0)
        escaped = '{"class_id":0,"class_name":"normal","note":"quoted braces } { and \\"word\\"","nested":{"a":1}}'
        self.assertEqual(helper.parse_prediction(escaped + "\n" + escaped)["class_id"], 0)
        invalid = [normal + '\n{"class_id":1,"class_name":"pneumonia"}', normal + '\n{"class_id":false,"class_name":"normal"}', normal + '\n{"class_id":0,"class_name":"pneumonia"}', normal + '{bad}', normal + '{bad}{"class_id":1,"class_name":"pneumonia"}', '{"class_id":', '{"wrapper":' + normal, '{"wrapper":' + normal + '}', '{"class_id":0,"class_id":1,"class_name":"pneumonia"}']
        for response in invalid:
            with self.assertRaises(ValueError):
                helper.parse_prediction(response)

    def test_same_transport_and_fatal_errors(self):
        """Both splits send decoder fields and identities, while HTTP errors never yield fitness."""
        received = []
        state = {"fail": False, "malformed": False}

        class Handler(BaseHTTPRequestHandler):
            """Act as an OpenAI-compatible gateway for contract verification."""

            def do_POST(self):
                """Record model request and return deterministic JSON or a fatal transport error."""
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                received.append((payload, dict(self.headers)))
                if state["fail"]:
                    self.send_response(503)
                    self.end_headers()
                    return
                pneumonia = "pneumonia/" in self.headers["X-Evolve-Image-Id"]
                content = json.dumps({"class_id": int(pneumonia), "class_name": "pneumonia" if pneumonia else "normal"})
                result = {"id": "mock", "model": payload["model"], "choices": [{"message": {"content": content}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 17, "completion_tokens": 5, "total_tokens": 22}}
                if state["malformed"]:
                    result["choices"] = [{}]
                body = json.dumps(result).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                """Keep mock request logs out of test output."""
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as folder:
                settings = helper.get_default_settings()
                settings.update({"base_url": f"http://127.0.0.1:{server.server_port}/v1", "api_key": "mock", "num_images": 2, "concurrency": 2, "seed": 123, "output_root": folder})
                candidate = {"prompt": "Classify.", "sampling": {"temperature": .3, "top_p": .9}}
                for split in ("train", "test"):
                    result = validate.evaluate_candidate(candidate, split=split, settings=settings)
                    self.assertEqual(result["fitness"], 1)
                    self.assertEqual(result["input_tokens_known"], 34)
                    self.assertEqual(result["output_tokens_known"], 10)
                    self.assertEqual(result["requests_missing_usage"], 0)
                    directory = Path(result["output_dir"])
                    self.assertEqual(json.loads((directory / "candidate.json").read_text()), candidate)
                    self.assertNotIn("api_key", json.loads((directory / "config.json").read_text())["settings"])
                    self.assertEqual(len((directory / "predictions.jsonl").read_text().splitlines()), 2)
                for payload, headers in received:
                    self.assertEqual(payload["temperature"], .3)
                    self.assertEqual(payload["top_p"], .9)
                    self.assertEqual(payload["max_tokens"], 256)
                    self.assertIn("seed", payload)
                    self.assertEqual(headers["X-Evolve-Role"], "evaluator")
                state["fail"] = True
                with self.assertRaises(RuntimeError):
                    validate.evaluate_candidate(candidate, settings=settings)
                failed = [directory for directory in (Path(folder) / "evaluations").iterdir() if json.loads((directory / "status.json").read_text())["status"] == "failed"]
                self.assertEqual(len(failed), 1)
                self.assertFalse((failed[0] / "metrics.json").exists())
                state.update({"fail": False, "malformed": True})
                with self.assertRaises(RuntimeError):
                    validate.evaluate_candidate(candidate, settings=settings)
                failed = [directory for directory in (Path(folder) / "evaluations").iterdir() if json.loads((directory / "status.json").read_text())["status"] == "failed"]
                self.assertEqual(len(failed), 2)
                self.assertTrue(all(not (directory / "metrics.json").exists() for directory in failed))
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main(verbosity=2)
