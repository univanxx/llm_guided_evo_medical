"""Exercise the native search, HTTP gateways, selection and resume using local test responses."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent


def main():
    """Run all requested model/resolution settings without contacting an inference provider."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=("4b", "27b"), default=["4b", "27b"])
    parser.add_argument("--resolutions", nargs="+", type=int, choices=(28, 64, 128, 224), default=[28, 64, 128, 224])
    args = parser.parse_args()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    calls = []

    class Handler(BaseHTTPRequestHandler):
        """Return deterministic schema-valid mock completions over real loopback HTTP."""

        def do_POST(self):
            """Handle the ordinary chat-completions wire format used by both gateways."""
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            mutation = payload["model"] == "mock-mutator"
            calls.append({"model": payload["model"], "mutation": mutation})
            if mutation:
                code = 'def entrypoint():\n    return {"prompt": "Classify this chest X-ray. Return class_id and class_name as JSON.", "sampling": {"temperature": 0.3, "top_p": 0.9}}\n'
                content = {"archetype": "Precision Optimization", "justification": "A deterministic mutation fixture for integration testing.", "insights_used": [], "insight_ids_used": [], "base_parent": 1, "card_ids_used": [], "changes": [], "code": code}
            else:
                content = {"class_id": 0, "class_name": "normal"}
            message = {"role": "assistant", "content": json.dumps(content)}
            if mutation and payload.get("tools"):
                message = {"role": "assistant", "content": None, "tool_calls": [{"id": "mock-tool", "type": "function", "function": {"name": payload["tools"][0]["function"]["name"], "arguments": json.dumps(content)}}]}
            body = json.dumps({"id": "mock-completion", "object": "chat.completion", "created": int(time.time()), "model": payload["model"], "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if "tool_calls" in message else "stop"}], "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *unused):
            """Keep HTTP access logs out of the smoke report."""
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    endpoint = f"http://127.0.0.1:{server.server_port}/v1"
    environment = dict(os.environ)
    environment.update({"EVALUATOR_API_KEY": "offline-fixture", "MUTATOR_API_KEY": "offline-fixture", "MEDGEMMA_CONCURRENCY": "2", "PYTHONDONTWRITEBYTECODE": "1", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"})
    report = []
    try:
        for model in args.models:
            for resolution in args.resolutions:
                run = output / f"medgemma-{model}-r{resolution}-s43"
                command = [sys.executable, "-B", str(ROOT / "run_experiment.py"), "--model", model, "--resolution", str(resolution), "--seed", "43", "--smoke", "--final-repeats", "1", "--mutator-model", "mock-mutator", "--evaluator-base-url", endpoint, "--mutator-base-url", endpoint, "--retry-budget-seconds", "5", "--request-timeout-seconds", "10", "--run-dir", str(run)]
                started = time.monotonic()
                with (output / f"{model}-r{resolution}.log").open("w") as log:
                    subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=300, check=True)
                    before = len(calls)
                    subprocess.run([*command, "--resume"], env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=120, check=True)
                if len(calls) != before:
                    raise AssertionError("A completed resume issued new inference calls")
                status = json.loads((run / "run_status.json").read_text())
                pair = json.loads((run / "paired_evaluation/summary.json").read_text())
                if status["status"] != "completed" or set(pair["roles"]) != {"initial", "evolved"}:
                    raise AssertionError("Native search did not complete both final roles")
                item = {"model": model, "resolution": resolution, "status": status["status"], "mutations": status["mutations_created"], "resume_new_requests": 0, "elapsed_seconds": round(time.monotonic() - started, 2)}
                report.append(item)
                (output / "smoke_report.json").write_text(json.dumps({"inference": "local_mock", "runs": report, "requests": len(calls)}, indent=2) + "\n")
                print(json.dumps(item), flush=True)
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


if __name__ == "__main__":
    main()
