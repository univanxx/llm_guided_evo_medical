"""Proxy model requests through four-hour outages and audit reported token usage.

Only hashes and small experiment identifiers are persisted; prompts, images and
credentials remain in memory. Retries of ambiguous network failures are at-least-
once: their unknown token consumption is reported explicitly, never as zero.
"""

from __future__ import annotations

import hashlib
import http.client
import ipaddress
import json
import os
import random
import socket
import ssl
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


@dataclass
class GatewayConfig:
    """Configure an isolated provider gateway; give each instance its own directory."""

    api_key: str = field(repr=False)
    output_dir: Path
    run_id: str
    upstream_base_url: str = "http://127.0.0.1:8000/v1"
    provider_kind: str = "compatible"
    default_role: str = "unknown"
    retry_budget_seconds: float = 4 * 3600 + 15 * 60
    request_timeout_seconds: float = 180
    initial_backoff_seconds: float = 2
    max_backoff_seconds: float = 120
    dns_override: str | None = None
    openrouter_provider: dict | None = None
    bind_port: int = 0

    def __post_init__(self):
        """Reject unusable settings before starting any upstream requests."""
        self.output_dir = Path(self.output_dir)
        parsed = urlsplit(self.upstream_base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("upstream_base_url must be an HTTP(S) API base URL without credentials/query")
        if self.dns_override and parsed.scheme != "https":
            raise ValueError("The optional DNS override requires HTTPS")
        if self.dns_override:
            try:
                ipaddress.ip_address(self.dns_override)
            except ValueError as error:
                raise ValueError("dns_override must be a literal IP address") from error
        if type(self.bind_port) is not int or self.bind_port < 0 or self.bind_port > 65535:
            raise ValueError("bind_port must be an integer from 0 through 65535")
        if min(self.retry_budget_seconds, self.request_timeout_seconds, self.initial_backoff_seconds, self.max_backoff_seconds) <= 0:
            raise ValueError("Timeouts and backoff must be positive")


@dataclass
class UpstreamResponse:
    """Carry the complete upstream reply independently of its JSON validity."""

    status: int
    headers: dict
    body: bytes


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """Connect to a configured address while preserving TLS hostname checks."""

    def __init__(self, host, port, address, timeout):
        """Keep the actual hostname for SNI and certificate verification."""
        super().__init__(host, port=port, timeout=timeout, context=ssl.create_default_context())
        self.address = address

    def connect(self):
        """Override only this connection, without modifying process-wide DNS."""
        raw_socket = socket.create_connection((self.address, self.port), self.timeout)
        try:
            self.sock = self._context.wrap_socket(raw_socket, server_hostname=self.host)
        except BaseException:
            raw_socket.close()
            raise


class HTTPTransport:
    """Issue one HTTP request with no hidden SDK retries."""

    def __init__(self, config):
        """Remember credentials in memory and parse the provider endpoint once."""
        self.config = config
        self.url = urlsplit(config.upstream_base_url)

    def __call__(self, method, suffix, body, timeout):
        """Return status, safe response headers and bytes for exactly one attempt."""
        if self.config.dns_override:
            connection = _PinnedHTTPSConnection(self.url.hostname, self.url.port, self.config.dns_override, timeout)
        elif self.url.scheme == "https":
            connection = http.client.HTTPSConnection(self.url.hostname, self.url.port, timeout=timeout, context=ssl.create_default_context())
        else:
            connection = http.client.HTTPConnection(self.url.hostname, self.url.port, timeout=timeout)
        headers = {"Content-Type": "application/json", "Authorization": "Bearer " + self.config.api_key, "Accept": "application/json"}
        try:
            connection.request(method, self.url.path.rstrip("/") + suffix, body=body, headers=headers)
            response = connection.getresponse()
            return UpstreamResponse(response.status, {key.lower(): value for key, value in response.getheaders() if key.lower() in {"content-type", "retry-after", "x-request-id", "request-id"}}, response.read())
        finally:
            connection.close()


def _timestamp():
    """Use UTC timestamps for portable experiment logs."""
    return datetime.now(timezone.utc).isoformat()


def _number(value):
    """Return numeric usage only; missing values have the explicit value None."""
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0 else None


def _usage(payload):
    """Normalize provider totals without adding reasoning or cache tokens twice."""
    usage = payload.get("usage") if isinstance(payload, dict) else None
    if not isinstance(usage, dict):
        usage = {}
    input_tokens = _number(usage.get("prompt_tokens", usage.get("input_tokens")))
    output_tokens = _number(usage.get("completion_tokens", usage.get("output_tokens")))
    total_tokens = _number(usage.get("total_tokens"))
    return {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": total_tokens, "cost": _number(usage.get("cost")), "raw_usage": usage or None}


def _empty_counts():
    """Initialize aggregates with unknown token totals rather than false zeros."""
    return {"attempts": 0, "http_responses": 0, "network_failures": 0, "successful_http_responses": 0, "missing_usage_responses": 0, "unresolved_attempts": 0, "ambiguous_attempts": 0, "input_tokens": None, "output_tokens": None, "total_tokens": None, "cost": None, "missing_input_tokens": 0, "missing_output_tokens": 0, "missing_total_tokens": 0, "missing_cost": 0}


def _valid_choices(payload):
    """Require a usable chat envelope before returning a provider HTTP success."""
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return False
    message = choices[0].get("message")
    if not isinstance(message, dict):
        return False
    return isinstance(message.get("content"), str) or (message.get("content") is None and isinstance(message.get("tool_calls"), list) and bool(message["tool_calls"]))


class GatewayLedgerIntegrityError(ValueError):
    """Report an internal JSONL corruption that cannot be treated as a torn tail."""


class TokenLedger:
    """Persist a thread-safe audit log plus cumulative reported-token counters."""

    def __init__(self, config):
        """Resume aggregates for the same run from the append-only event log."""
        self.config = config
        self.lock = threading.Lock()
        config.output_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = config.output_dir / "gateway_events.jsonl"
        self.summary_path = config.output_dir / "gateway_tokens.json"
        self.summary = {"schema_version": 1, "run_id": config.run_id, "provider_kind": config.provider_kind, "upstream_host": urlsplit(config.upstream_base_url).hostname, "totals": _empty_counts(), "by_role_model_stage": {}, "pending": {}, "active_requests": 0, "waiting_requests": 0, "totals_exact": True}
        torn_final = None
        if self.events_path.exists():
            torn_final = self._load_existing_events()
        unresolved = [{"request_id": request_id, "attempt": item["attempts"], "role": item.get("role"), "model": item.get("model"), "stage": item.get("stage")} for request_id, item in self.summary["pending"].items() if item["attempts"] > item.get("last_recorded_attempt", 0)]
        self.record({"event": "gateway_started", "retry_budget_seconds": config.retry_budget_seconds, "abandoned_request_ids": list(self.summary["pending"]), "unresolved_attempts": unresolved, "torn_final_record_discarded": torn_final})

    def _publish_summary(self):
        """Atomically publish the current summary and flush it before replacement."""
        temporary = self.summary_path.with_suffix(".json.tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as output:
                output.write(json.dumps(self.summary, ensure_ascii=True, indent=2, allow_nan=False) + "\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, self.summary_path)
        finally:
            temporary.unlink(missing_ok=True)

    def _integrity_failure(self, line_number, byte_offset, error):
        """Publish an inspection-required summary before refusing corrupted history."""
        self.summary["integrity_error"] = {"type": "MalformedInternalJSONL", "line_number": line_number, "byte_offset": byte_offset, "detail": type(error).__name__}
        self.summary["totals_exact"] = False
        self.summary["state"] = "inspection_required"
        self.summary["updated_at"] = _timestamp()
        for counts in [self.summary["totals"], *self.summary["by_role_model_stage"].values()]:
            for key in ("input_tokens", "output_tokens", "total_tokens", "cost"):
                counts[key] = None
        self._publish_summary()
        return GatewayLedgerIntegrityError("Malformed internal gateway ledger record at line " + str(line_number))

    def _load_existing_events(self):
        """Accept only an incomplete final record and reject all internal corruption."""
        raw = self.events_path.read_bytes()
        lines = raw.splitlines(keepends=True)
        offset = 0
        for index, line in enumerate(lines):
            final = index == len(lines) - 1
            complete = line.endswith(b"\n")
            try:
                event = json.loads(line)
                if not isinstance(event, dict) or not isinstance(event.get("event"), str) or not isinstance(event.get("timestamp"), str):
                    raise ValueError("Invalid event shape")
            except (UnicodeError, ValueError) as error:
                if final and not complete:
                    with self.events_path.open("rb+") as existing:
                        existing.truncate(offset)
                        existing.flush()
                        os.fsync(existing.fileno())
                    return {"line_number": index + 1, "discarded_bytes": len(line), "reason": type(error).__name__}
                raise self._integrity_failure(index + 1, offset, error)
            if event.get("run_id") != self.config.run_id:
                raise ValueError("Gateway output directory already belongs to another run")
            try:
                self._aggregate(event)
            except (KeyError, TypeError) as error:
                raise self._integrity_failure(index + 1, offset, error)
            offset += len(line)
        if raw and lines and not lines[-1].endswith(b"\n"):
            with self.events_path.open("ab") as existing:
                existing.write(b"\n")
                existing.flush()
                os.fsync(existing.fileno())
        return None

    def _aggregate(self, event):
        """Add usage from each actual attempt, including failed HTTP responses."""
        event_type = event.get("event")
        pending = self.summary["pending"]
        request_id = event.get("request_id")
        if event_type == "gateway_started":
            for attempt in event.get("unresolved_attempts", []):
                self.summary["totals"]["unresolved_attempts"] += 1
                group = json.dumps([attempt.get("role"), attempt.get("model"), attempt.get("stage")], ensure_ascii=True)
                self.summary["by_role_model_stage"].setdefault(group, _empty_counts())["unresolved_attempts"] += 1
            pending.clear()
            self.summary["state"] = "running"
            self.summary.pop("fatal_reason", None)
            self.summary.pop("fatal_code", None)
        elif event_type == "gateway_stopped":
            self.summary["state"] = "stopped"
            self.summary["fatal_reason"] = event.get("fatal_reason")
            self.summary["fatal_code"] = event.get("fatal_code")
        elif event_type == "request_started":
            pending[request_id] = {key: event.get(key) for key in ("role", "model", "stage", "candidate_id", "evaluation_id", "image_id", "request_sha256")}
            pending[request_id].update({"started_at": event["timestamp"], "state": "in_flight", "attempts": 0, "budget_remaining_seconds": event.get("retry_budget_seconds"), "budget_updated_at": event["timestamp"]})
        elif event_type in {"request_completed", "request_interrupted"}:
            pending.pop(request_id, None)
            if event_type == "request_interrupted" and event.get("fatal_reason"):
                self.summary["state"] = "interrupted"
                self.summary["fatal_reason"] = event["fatal_reason"]
        elif request_id in pending:
            pending[request_id]["last_event_at"] = event["timestamp"]
            if event_type == "request_attempt_started":
                pending[request_id].update({"state": "in_flight", "attempts": event["attempt"]})
            elif event_type == "upstream_attempt":
                pending[request_id]["last_recorded_attempt"] = event["attempt"]
            elif event_type == "retry_wait":
                pending[request_id].update({"state": "waiting", "last_retry_at": event["timestamp"], "budget_remaining_seconds": event["budget_remaining_seconds"], "budget_updated_at": event["timestamp"], "delay_seconds": event["delay_seconds"]})
        self.summary["active_requests"] = len(pending)
        self.summary["waiting_requests"] = sum(item["state"] == "waiting" for item in pending.values())
        if event.get("event") != "upstream_attempt":
            return
        group = json.dumps([event.get("role"), event.get("model"), event.get("stage")], ensure_ascii=True)
        group_counts = self.summary["by_role_model_stage"].setdefault(group, _empty_counts())
        for counts in (self.summary["totals"], group_counts):
            counts["attempts"] += 1
            if event["status"] is None:
                counts["network_failures"] += 1
            else:
                counts["http_responses"] += 1
                counts["successful_http_responses"] += int(200 <= event["status"] < 300)
            if event.get("is_inference"):
                counts["ambiguous_attempts"] += int(bool(event.get("ambiguous_remote_execution")))
                counts["missing_usage_responses"] += int(event.get("raw_usage") is None)
                for key in ("input_tokens", "output_tokens", "total_tokens", "cost"):
                    value = event.get(key)
                    if value is None:
                        counts["missing_" + key] += 1
                    else:
                        counts[key] = (counts[key] or 0) + value

    def record(self, event):
        """Write the durable event before atomically replacing the summary file."""
        event = {"timestamp": _timestamp(), "run_id": self.config.run_id, "provider_kind": self.config.provider_kind, **event}
        with self.lock:
            with self.events_path.open("a", encoding="utf-8") as output:
                output.write(json.dumps(event, ensure_ascii=True, allow_nan=False) + "\n")
                output.flush()
                os.fsync(output.fileno())
            self._aggregate(event)
            self.summary["updated_at"] = event["timestamp"]
            self._publish_summary()


class GatewayEngine:
    """Apply bounded retries, outage coordination and accounting to each call."""

    def __init__(self, config, transport=None, clock=time.monotonic, sleeper=None, jitter=None):
        """Allow deterministic clocks and transports for four-hour outage tests."""
        self.config = config
        self.transport = transport or HTTPTransport(config)
        self.clock = clock
        self.stop_event = threading.Event()
        self.fatal_event = threading.Event()
        self.fatal_reason = None
        self.fatal_code = None
        self.telemetry_failed = False
        self.sleeper = sleeper or self.stop_event.wait
        self.jitter = jitter or (lambda delay: random.uniform(0.8, 1.2) * delay)
        self.ledger = TokenLedger(config)
        self.cooldown_lock = threading.Lock()
        self.cooldowns = {}

    def _fatal(self, reason, code="gateway_stopped"):
        """Expose sanitized infrastructure failures for the experiment supervisor."""
        if not self.fatal_event.is_set():
            self.fatal_reason = reason
            self.fatal_code = code
        self.fatal_event.set()
        self.stop_event.set()

    def _record(self, event):
        """Never replay a completed upstream call because its log write failed."""
        try:
            self.ledger.record(event)
        except Exception as error:
            self.telemetry_failed = True
            self._fatal("Token telemetry failed: " + type(error).__name__, "telemetry_failed")
            print("[gateway] Token telemetry failed; stopping new upstream requests.", file=sys.stderr, flush=True)

    def _acquire(self, model, deadline):
        """Allow only one recovery probe per unavailable model and gateway."""
        while not self.stop_event.is_set() and self.clock() < deadline:
            with self.cooldown_lock:
                state = self.cooldowns.get(model)
                if state is None:
                    return True
                remaining = state["until"] - self.clock()
                if remaining <= 0 and not state["probing"]:
                    state["probing"] = True
                    return True
                delay = min(max(remaining, 0.1), 1, deadline - self.clock())
            if delay > 0:
                self.sleeper(delay)
        return False

    def _cooldown(self, model, delay):
        """Share the next retry time so concurrent requests do not create a storm."""
        with self.cooldown_lock:
            self.cooldowns[model] = {"until": self.clock() + delay, "probing": False}

    def _recovered(self, model):
        """Release queued requests once one request confirms provider recovery."""
        with self.cooldown_lock:
            return self.cooldowns.pop(model, None) is not None

    def _retryable(self, response, payload):
        """Distinguish temporary capacity/model availability from bad requests."""
        status = self._effective_status(response, payload)
        if status in {408, 429} or 500 <= status <= 599:
            return True
        # Serving endpoints can temporarily remove healthy models during outages.
        if status == 404 and self.config.provider_kind == "compatible":
            return True
        if status not in {200, 400, 404, 409, 424}:
            return False
        error = payload.get("error", payload) if isinstance(payload, dict) else {}
        message = str(error.get("message", "") if isinstance(error, dict) else error).lower()
        return any(phrase in message for phrase in ("temporarily unavailable", "currently unavailable", "model unavailable", "model is unavailable", "model is loading", "model not loaded", "model is not loaded", "model is not ready", "no available worker", "no available endpoint", "overloaded"))

    def _effective_status(self, response, payload):
        """Interpret provider error envelopes that were transported with HTTP 200."""
        error = payload.get("error") if isinstance(payload, dict) else None
        code = error.get("code") if isinstance(error, dict) else None
        if isinstance(code, str) and len(code) == 3 and code.isascii() and code.isdecimal():
            code = int(code)
        if 200 <= response.status < 300 and isinstance(code, int) and 400 <= code <= 599:
            return code
        return response.status

    def _delay(self, attempt, response):
        """Honor Retry-After and otherwise use bounded exponential jitter."""
        base_delay = min(self.config.max_backoff_seconds, self.config.initial_backoff_seconds * 2 ** min(attempt - 1, 16))
        delay = min(self.config.max_backoff_seconds, self.jitter(base_delay))
        value = response.headers.get("retry-after") if response else None
        if value:
            try:
                retry_after = float(value)
            except ValueError:
                try:
                    retry_after = (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds()
                except (TypeError, ValueError, OverflowError):
                    retry_after = 0
            delay = max(delay, retry_after)
        return max(delay, 0)

    def _error(self, status, code, message):
        """Return an OpenAI-shaped local error without exposing upstream secrets."""
        return UpstreamResponse(status, {"content-type": "application/json"}, json.dumps({"error": {"type": "gateway_error", "code": code, "message": message}}).encode())

    def execute(self, method, suffix, body, headers=None):
        """Keep a pending request alive until success, fatal failure or its deadline."""
        headers = {key.lower(): value for key, value in (headers or {}).items()}
        is_inference = method == "POST"
        try:
            payload = json.loads(body) if is_inference else {}
        except (ValueError, UnicodeError):
            return self._error(400, "invalid_json", "Request must contain valid JSON")
        if not isinstance(payload, dict) or payload.get("stream"):
            return self._error(400, "unsupported_request", "Gateway supports nonstreaming JSON chat requests")
        if self.config.provider_kind == "openrouter" and is_inference:
            if not isinstance(payload.get("provider", {}), dict):
                return self._error(400, "invalid_provider", "provider must be a JSON object")
            provider = {**(self.config.openrouter_provider or {}), **payload.get("provider", {})}
            provider["require_parameters"] = True
            payload["provider"] = provider
            body = json.dumps(payload, ensure_ascii=True).encode()
        metadata = {"request_id": uuid.uuid4().hex, "request_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode()).hexdigest(), "model": str(payload.get("model", "models"))[:200], "role": str(headers.get("x-evolve-role", self.config.default_role))[:100], "is_inference": is_inference}
        for key in ("stage", "candidate-id", "evaluation-id", "image-id", "repeat"):
            value = headers.get("x-evolve-" + key)
            if value is not None:
                metadata[key.replace("-", "_")] = str(value)[:200]
        if "x-evolve-run-id" in headers:
            metadata["client_run_id"] = str(headers["x-evolve-run-id"])[:200]
        deadline = self.clock() + self.config.retry_budget_seconds
        self._record({"event": "request_started", **metadata, "retry_budget_seconds": self.config.retry_budget_seconds})
        attempt = 0
        while self._acquire(metadata["model"], deadline):
            attempt += 1
            response, response_payload, failure_type = None, None, None
            started = self.clock()
            self._record({"event": "request_attempt_started", **metadata, "attempt": attempt})
            if self.stop_event.is_set():
                break
            try:
                response = self.transport(method, suffix, body, min(self.config.request_timeout_seconds, max(0.001, deadline - started)))
            except (OSError, http.client.HTTPException) as error:
                failure_type = type(error).__name__
            if response is not None:
                try:
                    response_payload = json.loads(response.body)
                except (ValueError, UnicodeError):
                    failure_type = "InvalidUpstreamJSON"
                if 200 <= response.status < 300 and isinstance(response_payload, dict):
                    if "error" in response_payload:
                        failure_type = "UpstreamErrorEnvelope"
                    elif is_inference and not _valid_choices(response_payload):
                        failure_type = "InvalidUpstreamChoices"
                    elif not is_inference and not isinstance(response_payload.get("data"), list):
                        failure_type = "MissingUpstreamModels"
                elif 200 <= response.status < 300 and response_payload is not None:
                    failure_type = "InvalidUpstreamShape"
            usage = _usage(response_payload)
            self._record({"event": "upstream_attempt", **metadata, "attempt": attempt, "status": response.status if response else None, "effective_status": self._effective_status(response, response_payload) if response else None, "duration_seconds": round(self.clock() - started, 3), "failure_type": failure_type, "ambiguous_remote_execution": response is None, "provider_request_id": response.headers.get("x-request-id", response.headers.get("request-id")) if response else None, **usage})
            retryable = response is None or self._retryable(response, response_payload)
            if response is not None and not retryable:
                recovered = self._recovered(metadata["model"])
                effective_status = self._effective_status(response, response_payload)
                if effective_status in {401, 402, 403, 404}:
                    self._fatal("Upstream HTTP " + str(effective_status) + " for " + metadata["role"] + "/" + metadata["model"], "upstream_http_" + str(effective_status))
                elif 200 <= response.status < 300 and failure_type:
                    self._fatal("Upstream response failure " + failure_type + " for " + metadata["role"] + "/" + metadata["model"], failure_type)
                if recovered and 200 <= effective_status < 300 and not failure_type:
                    self._record({"event": "provider_recovered", **metadata, "attempt": attempt})
                    print("[gateway] Provider recovered: " + metadata["role"] + "/" + metadata["model"], flush=True)
                self._record({"event": "request_completed", **metadata, "attempts": attempt, "status": response.status})
                if 200 <= response.status < 300 and failure_type:
                    return self._error(effective_status if effective_status >= 400 else 502, failure_type, self.fatal_reason or "Provider returned an error envelope")
                return response
            if self.stop_event.is_set():
                break
            delay = min(self._delay(attempt, response), max(0, deadline - self.clock()))
            self._cooldown(metadata["model"], delay)
            self._record({"event": "retry_wait", **metadata, "attempt": attempt, "delay_seconds": round(delay, 3), "budget_remaining_seconds": round(max(0, deadline - self.clock()), 3)})
            if attempt == 1 or attempt % 10 == 0:
                print("[gateway] Waiting for " + metadata["role"] + "/" + metadata["model"] + "; retry budget remaining " + str(round(max(0, deadline - self.clock()))) + " s", flush=True)
        exhausted = not self.stop_event.is_set()
        reason = "retry_budget_exhausted" if exhausted else "gateway_stopped"
        fatal_reason = "Retry budget exhausted for " + metadata["role"] + "/" + metadata["model"] if exhausted else None
        self._record({"event": "request_interrupted", **metadata, "attempts": attempt, "reason": reason, "fatal_reason": fatal_reason})
        if exhausted:
            self._fatal(fatal_reason, "retry_budget_exhausted")
        return self._error(503, reason, self.fatal_reason or "Gateway stopped; pending request can be resumed by the experiment launcher")


class GatewayHTTPServer(ThreadingHTTPServer):
    """Accept image-request bursts without the default five-connection backlog."""

    request_queue_size = 128
    daemon_threads = True


class GatewayHandle:
    """Expose the local API address and terminal provider status to the launcher."""

    def __init__(self, server, engine):
        """Create an inspection handle without starting extra background work."""
        self.server = server
        self.engine = engine
        self.base_url = "http://127.0.0.1:" + str(server.server_address[1]) + "/v1"
        self.fatal_event = engine.fatal_event

    @property
    def fatal_reason(self):
        """Describe infrastructure failure without returning provider bodies."""
        return self.engine.fatal_reason

    @property
    def resume_reason(self):
        """Keep actual retry exhaustion distinct from other terminal gateway failures."""
        return self.engine.fatal_code or "gateway_stopped"

    @property
    def telemetry_failed(self):
        """Allow the launcher to mark an incompletely accounted run as failed."""
        return self.engine.telemetry_failed


@contextmanager
def start_gateway(config, transport=None):
    """Serve a loopback-only OpenAI endpoint until the surrounding run finishes."""
    engine = GatewayEngine(config, transport=transport)

    class Handler(BaseHTTPRequestHandler):
        """Adapt HTTP requests without using BaseHTTPRequestHandler's raw logs."""

        def log_message(self, format, *args):
            """Suppress request/body/header logging at the HTTP server layer."""
            return

        def do_GET(self):
            """Forward only the model-list endpoint."""
            self._serve("GET")

        def do_POST(self):
            """Forward only nonstreaming chat completions."""
            self._serve("POST")

        def _serve(self, method):
            """Return the completed response even if a downstream client disconnects."""
            suffix = self.path.removeprefix("/v1")
            if (method, suffix) not in {("GET", "/models"), ("POST", "/chat/completions")}:
                response = engine._error(404, "unsupported_route", "Supported routes: /v1/models and /v1/chat/completions")
            else:
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    length = -1
                if not 0 <= length <= 64 * 1024 * 1024:
                    response = engine._error(413, "invalid_size", "Request body limit is 64 MiB")
                else:
                    response = engine.execute(method, suffix, self.rfile.read(length) if method == "POST" else None, dict(self.headers))
            try:
                self.send_response(response.status)
                self.send_header("Content-Type", response.headers.get("content-type", "application/json"))
                self.send_header("Content-Length", str(len(response.body)))
                if engine.telemetry_failed:
                    self.send_header("X-Evolve-Telemetry-Error", "true")
                self.end_headers()
                self.wfile.write(response.body)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                engine._record({"event": "downstream_disconnected", "status": response.status})

    server = GatewayHTTPServer(("127.0.0.1", config.bind_port), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True, name="evolve-gateway")
    worker.start()
    handle = GatewayHandle(server, engine)
    try:
        yield handle
    finally:
        engine.stop_event.set()
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)
        engine._record({"event": "gateway_stopped", "fatal_reason": engine.fatal_reason, "fatal_code": engine.fatal_code})
