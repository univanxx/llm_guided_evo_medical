"""Exercise prolonged outages and usage accounting without remote model calls."""

import json
import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path
from urllib.request import Request, urlopen
from unittest.mock import MagicMock, patch

from med_evolve.gateway import GatewayConfig, GatewayEngine, UpstreamResponse, _PinnedHTTPSConnection, start_gateway


class FakeClock:
    """Advance a virtual clock immediately for multi-hour outage tests."""

    def __init__(self):
        """Start virtual monotonic time at zero."""
        self.now = 0.0

    def __call__(self):
        """Return current virtual time."""
        return self.now

    def sleep(self, seconds):
        """Advance time without making the test wait."""
        self.now += seconds


def reply(status=200, usage=None, error=None, headers=None):
    """Build a realistic nonstreaming provider completion or error."""
    body = {"error": {"message": error}} if error else {"choices": [{"message": {"content": "test"}}]}
    if usage is not None:
        body["usage"] = usage
    return UpstreamResponse(status, headers or {}, json.dumps(body).encode())


class GatewayTests(unittest.TestCase):
    """Verify bounded resilience, provider compatibility, privacy and token sums."""

    def setUp(self):
        """Isolate the persistent ledger for every test."""
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = GatewayConfig(api_key="secret-key-for-test", output_dir=Path(self.temp.name), run_id="test-run")
        self.body = json.dumps({"model": "medgemma", "messages": [{"role": "user", "content": "PRIVATE_IMAGE_AND_PROMPT"}], "temperature": 0.3}).encode()

    def engine(self, transport, clock=None):
        """Create a deterministic engine with no random delay variation."""
        clock = clock or FakeClock()
        return GatewayEngine(self.config, transport=transport, clock=clock, sleeper=clock.sleep, jitter=lambda value: value)

    def events(self):
        """Read the persisted audit events for assertions."""
        return [json.loads(line) for line in (Path(self.temp.name) / "gateway_events.jsonl").read_text().splitlines()]

    def test_recovers_after_four_hour_outage_and_counts_all_attempts(self):
        """A model unavailable for four hours still succeeds within default budget."""
        clock = FakeClock()
        calls = []

        def transport(method, suffix, body, timeout):
            """Report unavailable until four virtual hours have elapsed."""
            calls.append(clock())
            if clock() < 4 * 3600:
                return reply(503, error="temporarily unavailable")
            return reply(usage={"prompt_tokens": 17, "completion_tokens": 3, "total_tokens": 20})

        engine = self.engine(transport, clock)
        response = engine.execute("POST", "/chat/completions", self.body, {"X-Evolve-Role": "evaluator", "X-Evolve-Candidate-Id": "candidate-1"})
        self.assertEqual(response.status, 200)
        self.assertGreaterEqual(clock(), 4 * 3600)
        self.assertLess(clock(), self.config.retry_budget_seconds)
        self.assertFalse(engine.fatal_event.is_set())
        totals = engine.ledger.summary["totals"]
        self.assertEqual(totals["input_tokens"], 17)
        self.assertEqual(totals["total_tokens"], 20)
        self.assertEqual(totals["missing_input_tokens"], len(calls) - 1)
        self.assertTrue(any(event["event"] == "provider_recovered" for event in self.events()))

    def test_exhaustion_stops_subsequent_requests(self):
        """An outage beyond the configured budget becomes a supervisor-visible fatal."""
        self.config.retry_budget_seconds = 5
        calls = []
        engine = self.engine(lambda *args: calls.append(args) or reply(503, error="unavailable"))
        self.assertEqual(engine.execute("POST", "/chat/completions", self.body).status, 503)
        previous_count = len(calls)
        self.assertTrue(engine.fatal_event.is_set())
        self.assertEqual(engine.execute("POST", "/chat/completions", self.body).status, 503)
        self.assertEqual(len(calls), previous_count)

    def test_usage_counts_even_when_content_cannot_be_parsed_downstream(self):
        """Content formatting errors cannot erase tokens already consumed upstream."""
        engine = self.engine(lambda *args: reply(usage={"input_tokens": 10, "output_tokens": 4, "total_tokens": 14, "completion_tokens_details": {"reasoning_tokens": 2}, "cost": 0.005}))
        engine.execute("POST", "/chat/completions", self.body)
        totals = engine.ledger.summary["totals"]
        self.assertEqual(totals["output_tokens"], 4)
        self.assertEqual(totals["total_tokens"], 14)
        self.assertEqual(totals["cost"], 0.005)
        raw_log = (Path(self.temp.name) / "gateway_events.jsonl").read_text()
        self.assertNotIn("PRIVATE_IMAGE_AND_PROMPT", raw_log)
        self.assertNotIn(self.config.api_key, raw_log)
        self.assertIn("reasoning_tokens", raw_log)

    def test_missing_usage_and_network_failure_are_unknown(self):
        """Unknown consumption stays null and retry ambiguity is recorded explicitly."""
        calls = []

        def transport(*args):
            """Lose the first connection and omit usage from the completed response."""
            calls.append(1)
            if len(calls) == 1:
                raise TimeoutError("Do not log the provider exception body")
            return reply()

        engine = self.engine(transport)
        engine.execute("POST", "/chat/completions", self.body)
        totals = engine.ledger.summary["totals"]
        self.assertIsNone(totals["input_tokens"])
        self.assertEqual(totals["missing_input_tokens"], 2)
        self.assertEqual(totals["network_failures"], 1)
        self.assertTrue(any(event.get("ambiguous_remote_execution") for event in self.events()))

    def test_auth_fatal_but_candidate_400_is_not(self):
        """Authentication fails immediately while malformed candidate parameters stay local."""
        responses = iter([reply(400, error="invalid temperature"), reply(401, error="bad api key")])
        engine = self.engine(lambda *args: next(responses))
        self.assertEqual(engine.execute("POST", "/chat/completions", self.body).status, 400)
        self.assertFalse(engine.fatal_event.is_set())
        self.assertEqual(engine.execute("POST", "/chat/completions", self.body).status, 401)
        self.assertTrue(engine.fatal_event.is_set())

    def test_temporary_model_404_and_retry_after(self):
        """Explicit model loading honors Retry-After rather than becoming a typo failure."""
        responses = iter([reply(404, error="model is loading", headers={"retry-after": "25"}), reply(usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})])
        clock = FakeClock()
        engine = self.engine(lambda *args: next(responses), clock)
        self.assertEqual(engine.execute("POST", "/chat/completions", self.body).status, 200)
        self.assertEqual(clock(), 25)

    def test_openrouter_requires_sampling_support_and_handles_200_errors(self):
        """Strict parameter routing and embedded rate limits work with OpenRouter."""
        self.config.provider_kind = "openrouter"
        self.config.upstream_base_url = "https://openrouter.ai/api/v1"
        calls = []

        def transport(method, suffix, body, timeout):
            """Return a rate limit in a 200 envelope before a successful completion."""
            calls.append(json.loads(body))
            if len(calls) == 1:
                return UpstreamResponse(200, {}, b'{"error":{"code":429,"message":"rate limited"}}')
            return reply(usage={"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3})

        engine = self.engine(transport)
        self.assertEqual(engine.execute("POST", "/chat/completions", self.body).status, 200)
        self.assertTrue(all(call["provider"]["require_parameters"] for call in calls))
        self.assertEqual(engine.ledger.summary["totals"]["input_tokens"], 2)

    def test_invalid_upstream_body_is_not_replayed(self):
        """An HTTP 200 with a broken body stops instead of paying for a hidden repeat."""
        calls = []
        engine = self.engine(lambda *args: calls.append(1) or UpstreamResponse(200, {}, b"not-json"))
        self.assertEqual(engine.execute("POST", "/chat/completions", self.body).status, 502)
        self.assertEqual(len(calls), 1)
        self.assertTrue(engine.fatal_event.is_set())
        self.assertIsNone(engine.ledger.summary["totals"]["input_tokens"])

    def test_missing_choices_still_accounts_usage_and_stops(self):
        """Malformed response envelopes retain billed usage and cannot masquerade as success."""
        engine = self.engine(lambda *args: UpstreamResponse(200, {}, b'{"usage":{"prompt_tokens":13,"completion_tokens":2,"total_tokens":15}}'))
        self.assertEqual(engine.execute("POST", "/chat/completions", self.body).status, 502)
        self.assertTrue(engine.fatal_event.is_set())
        self.assertEqual(engine.ledger.summary["totals"]["total_tokens"], 15)

    def test_invalid_choice_shapes_cannot_become_bad_candidate_fitness(self):
        """Malformed chat envelopes fail infrastructure validation before downstream scoring."""
        for choices in ([{}], "invalid", [{"message": {"content": 42}}], [{"message": {"content": None}}]):
            with self.subTest(choices=choices):
                engine = self.engine(lambda *args: UpstreamResponse(200, {}, json.dumps({"choices": choices, "usage": {"prompt_tokens": 1}}).encode()))
                self.assertEqual(engine.execute("POST", "/chat/completions", self.body).status, 502)
                self.assertTrue(engine.fatal_event.is_set())

    def test_failed_http_attempts_with_usage_are_included(self):
        """A paid failed attempt and its retry both contribute to reported token totals."""
        responses = iter([reply(503, error="unavailable", usage={"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5}), reply(usage={"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9})])
        engine = self.engine(lambda *args: next(responses))
        engine.execute("POST", "/chat/completions", self.body)
        self.assertEqual(engine.ledger.summary["totals"]["input_tokens"], 11)
        self.assertEqual(engine.ledger.summary["totals"]["total_tokens"], 14)

    def test_telemetry_failure_never_repeats_completed_request(self):
        """A disk error after completion forwards that completion and stops future calls."""
        calls = []
        engine = self.engine(lambda *args: calls.append(1) or reply(usage={"prompt_tokens": 1}))
        original_record = engine.ledger.record

        def broken_record(event):
            """Simulate the log filesystem failing precisely after the remote response."""
            if event["event"] == "upstream_attempt":
                raise OSError("disk full")
            original_record(event)

        engine.ledger.record = broken_record
        self.assertEqual(engine.execute("POST", "/chat/completions", self.body).status, 200)
        self.assertEqual(len(calls), 1)
        self.assertTrue(engine.telemetry_failed)
        self.assertTrue(engine.fatal_event.is_set())

    def test_resume_restores_prior_usage(self):
        """Restarting the gateway in the same run retains cumulative usage."""
        transport = lambda *args: reply(usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5})
        self.engine(transport).execute("POST", "/chat/completions", self.body)
        resumed = self.engine(transport)
        resumed.execute("POST", "/chat/completions", self.body)
        self.assertEqual(resumed.ledger.summary["totals"]["total_tokens"], 10)

    def test_pending_requests_visible_and_cleared_after_restart(self):
        """A monitor sees waiting work while a restart records abandoned request IDs."""
        engine = self.engine(lambda *args: reply())
        engine.ledger.record({"event": "request_started", "request_id": "pending-1", "model": "medgemma", "role": "evaluator", "retry_budget_seconds": 15300})
        engine.ledger.record({"event": "retry_wait", "request_id": "pending-1", "budget_remaining_seconds": 15000, "delay_seconds": 120})
        self.assertEqual(engine.ledger.summary["active_requests"], 1)
        self.assertEqual(engine.ledger.summary["waiting_requests"], 1)
        resumed = self.engine(lambda *args: reply())
        self.assertEqual(resumed.ledger.summary["active_requests"], 0)
        self.assertEqual(self.events()[-1]["abandoned_request_ids"], ["pending-1"])

    def test_crash_during_request_preserves_unknown_consumption_once(self):
        """A torn response event leaves an unresolved attempt without inventing billing."""
        engine = self.engine(lambda *args: reply())
        metadata = {"request_id": "crashed-1", "model": "medgemma", "role": "evaluator", "stage": "test"}
        engine.ledger.record({"event": "request_started", **metadata, "retry_budget_seconds": 15300})
        engine.ledger.record({"event": "request_attempt_started", **metadata, "attempt": 1})
        with engine.ledger.events_path.open("a", encoding="utf-8") as output:
            output.write('{"event":"upstream_attempt"')
        resumed = self.engine(lambda *args: reply())
        totals = resumed.ledger.summary["totals"]
        self.assertEqual(totals["unresolved_attempts"], 1)
        self.assertEqual(totals["attempts"], 0)
        self.assertEqual(totals["http_responses"], 0)
        self.assertIsNone(totals["input_tokens"])
        restarted_again = self.engine(lambda *args: reply())
        self.assertEqual(restarted_again.ledger.summary["totals"]["unresolved_attempts"], 1)

    def test_crash_after_accounted_response_does_not_add_unknown_attempt(self):
        """Missing request-completed bookkeeping cannot duplicate already accounted usage."""
        engine = self.engine(lambda *args: reply())
        metadata = {"request_id": "accounted-1", "model": "medgemma", "role": "evaluator", "is_inference": True}
        engine.ledger.record({"event": "request_started", **metadata, "retry_budget_seconds": 15300})
        engine.ledger.record({"event": "request_attempt_started", **metadata, "attempt": 1})
        engine.ledger.record({"event": "upstream_attempt", **metadata, "attempt": 1, "status": 200, "input_tokens": 8, "output_tokens": 2, "total_tokens": 10, "raw_usage": {"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10}})
        resumed = self.engine(lambda *args: reply())
        self.assertEqual(resumed.ledger.summary["totals"]["unresolved_attempts"], 0)
        self.assertEqual(resumed.ledger.summary["totals"]["total_tokens"], 10)

    def test_concurrent_recovery_has_one_probe(self):
        """Only one caller probes an unavailable model after its shared cooldown."""
        engine = GatewayEngine(self.config, transport=lambda *args: reply())
        engine._cooldown("medgemma", 0)
        self.assertTrue(engine._acquire("medgemma", time.monotonic() + 1))
        results = []
        waiter = threading.Thread(target=lambda: results.append(engine._acquire("medgemma", time.monotonic() + 1)))
        waiter.start()
        time.sleep(0.03)
        self.assertEqual(results, [])
        engine._recovered("medgemma")
        waiter.join(2)
        self.assertEqual(results, [True])

    def test_loopback_http_interface(self):
        """A real local HTTP client receives the proxy response and role metadata."""
        with start_gateway(self.config, transport=lambda *args: reply(usage={"prompt_tokens": 7, "completion_tokens": 1, "total_tokens": 8})) as gateway:
            request = Request(gateway.base_url + "/chat/completions", data=self.body, headers={"Content-Type": "application/json", "X-Evolve-Role": "mutator", "X-Evolve-Image-Id": "42"})
            with urlopen(request, timeout=2) as response:
                self.assertEqual(response.status, 200)
            self.assertFalse(gateway.fatal_event.is_set())
        self.assertTrue(any(event.get("role") == "mutator" and event.get("image_id") == "42" for event in self.events()))

    def test_explicit_guarded_port_is_bound_without_dynamic_fallback(self):
        """The OS guard can couple an approved destination port to the local server."""
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        config = GatewayConfig(api_key="test", output_dir=Path(self.temp.name), run_id="fixed-port", bind_port=port)
        with start_gateway(config, transport=lambda *args: reply()) as gateway:
            self.assertEqual(gateway.server.server_address, ("127.0.0.1", port))

    def test_dns_override_must_be_a_literal_address(self):
        """Runtime DNS denial cannot be bypassed by passing another hostname as a pin."""
        with self.assertRaises(ValueError):
            GatewayConfig(api_key="test", output_dir=Path(self.temp.name), run_id="test-run", upstream_base_url="https://inference.example/v1", dns_override="other.example")

    def test_dns_override_requires_https(self):
        """An address override requires a TLS connection."""
        with self.assertRaises(ValueError):
            GatewayConfig(api_key="test", output_dir=Path(self.temp.name), run_id="test-run", upstream_base_url="http://inference.example/v1", dns_override="192.0.2.10")

    def test_dns_override_preserves_tls_server_hostname(self):
        """Pinned inference connections retain the real domain for SNI and certificate checks."""
        connection = _PinnedHTTPSConnection("inference.example", None, "192.0.2.10", 5)
        self.assertTrue(connection._context.check_hostname)
        context = MagicMock()
        raw_socket = MagicMock()
        connection._context = context
        with patch("med_evolve.gateway.socket.create_connection", return_value=raw_socket) as connect:
            connection.connect()
        connect.assert_called_once_with(("192.0.2.10", 443), 5)
        context.wrap_socket.assert_called_once_with(raw_socket, server_hostname="inference.example")


if __name__ == "__main__":
    unittest.main()
