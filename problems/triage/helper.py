from __future__ import annotations

import asyncio
import functools
import inspect
import json
import os
from pathlib import Path
import random
import time
import weakref

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    DefaultAsyncHttpxClient,
    NotFoundError,
    RateLimitError,
)


load_dotenv()
# Base URL of the OpenAI-compatible endpoint serving the task model.
INFERENCE_BASE_URL = os.environ.get("TRIAGE_ESI_INFERENCE_BASE_URL") or os.environ.get(
    "TRIAGE_LLM_BASE_URL", ""
)
API_KEY = os.environ.get("TRIAGE_ESI_API_KEY") or os.environ.get("OPENAI_API_KEY", "")

_MODELS_CONFIG_PATH = Path(__file__).parent / "models.json"
_ESI_KB_PATH = Path(__file__).parent / "data" / "esi_reference_kb.json"
_DEFAULT_MODEL = os.environ.get("TRIAGE_MODEL_NAME", "")
_ESI_KB_CACHE: list[dict[str, str]] | None = None
_ESI_KB_TEXT_CACHE: str | None = None
_LOOP_CLIENTS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_LOOP_LIMITERS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_UNAVAILABLE_MODELS: set[str] = set()
_FORCED_MODEL_NOTICES: set[tuple[str, str]] = set()


def append_jsonl(path: str | Path, payload: dict) -> None:
    """Append one JSON line with a single write (safe across subprocesses)."""
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(payload, ensure_ascii=False, default=str) + "\n").encode()
    fd = os.open(target, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(fd, line)
    finally:
        os.close(fd)


class LLMRequestError(RuntimeError):
    """Raised in strict mode when an expert-model request fails."""


def _load_models() -> list[str]:
    if _MODELS_CONFIG_PATH.is_file():
        try:
            data = json.loads(_MODELS_CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(data, list) and data:
                return data
        except Exception:
            pass
    return [_DEFAULT_MODEL]


def get_models() -> list[str]:
    return _load_models()


def get_model(index: int = 0) -> str:
    models = _load_models()
    return models[index % len(models)]


def load_esi_kb() -> list[dict[str, str]]:
    global _ESI_KB_CACHE
    if _ESI_KB_CACHE is not None:
        return _ESI_KB_CACHE
    try:
        data = json.loads(_ESI_KB_PATH.read_text(encoding="utf-8"))
        _ESI_KB_CACHE = (
            [
                row
                for row in data
                if isinstance(row, dict) and "title" in row and "text" in row
            ]
            if isinstance(data, list)
            else []
        )
    except Exception:
        _ESI_KB_CACHE = []
    return _ESI_KB_CACHE


def load_esi_kb_text(max_chars: int = 120_000) -> str:
    global _ESI_KB_TEXT_CACHE
    if _ESI_KB_TEXT_CACHE is not None and len(_ESI_KB_TEXT_CACHE) <= max_chars:
        return _ESI_KB_TEXT_CACHE
    parts: list[str] = []
    total = 0
    for row in load_esi_kb():
        section = f"### {row['title']}\n{row['text']}"
        if total + len(section) > max_chars:
            break
        parts.append(section)
        total += len(section)
    text = "\n\n".join(parts)
    if max_chars >= 120_000:
        _ESI_KB_TEXT_CACHE = text
    return text


MODEL = get_model(0)


def _new_client() -> AsyncOpenAI:
    headers = {}
    if referer := os.environ.get("OPENROUTER_HTTP_REFERER", "").strip():
        headers["HTTP-Referer"] = referer
    if title := os.environ.get("OPENROUTER_APP_TITLE", "").strip():
        headers["X-Title"] = title
    verify_ssl = os.environ.get("TRIAGE_ESI_VERIFY_SSL", "1").lower() not in {
        "0",
        "false",
        "no",
    }
    # Scope proxying to the expert client. This is intentionally separate from
    # HTTPS_PROXY so other endpoints can keep using a direct route.
    proxy_url = os.environ.get("OPENROUTER_PROXY_URL", "").strip()
    http_kwargs = {"verify": verify_ssl}
    if proxy_url:
        http_kwargs["proxy"] = proxy_url
    return AsyncOpenAI(
        base_url=INFERENCE_BASE_URL,
        api_key=API_KEY,
        max_retries=0,
        timeout=float(os.environ.get("TRIAGE_ESI_LLM_TIMEOUT_SEC", "180")),
        default_headers=headers or None,
        http_client=DefaultAsyncHttpxClient(**http_kwargs),
    )


def _record_usage(model: str, response) -> None:
    path = os.environ.get("TRIAGE_ESI_TOKEN_USAGE_PATH", "").strip()
    if not path:
        return
    usage = getattr(response, "usage", None)
    input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
    if not total_tokens:
        total_tokens = input_tokens + output_tokens
    details = getattr(usage, "completion_tokens_details", None)
    append_jsonl(
        path,
        {
            "timestamp": time.time(),
            "role": "expert",
            "phase": os.environ.get("TRIAGE_ESI_TOKEN_PHASE", "evolution"),
            "experiment_run": os.environ.get("GIGAEVO_EXPERIMENT_RUN", ""),
            "model": getattr(response, "model", None) or model,
            "status": "ok" if usage is not None else "missing_usage",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "reasoning_tokens": int(getattr(details, "reasoning_tokens", 0) or 0),
            "total_tokens": total_tokens,
        },
    )


class _TrackedCompletions:
    def __init__(self, target):
        self._target = target

    async def create(self, *args, **kwargs):
        requested = _requested_model(kwargs.get("model"))
        kwargs["model"] = requested
        retries = max(0, int(os.environ.get("TRIAGE_ESI_LLM_MAX_RETRIES", "6")))
        base_delay = max(
            0.1, float(os.environ.get("TRIAGE_ESI_LLM_RETRY_BASE_SEC", "1"))
        )
        max_delay = max(
            base_delay, float(os.environ.get("TRIAGE_ESI_LLM_RETRY_MAX_SEC", "30"))
        )
        for attempt in range(retries + 1):
            try:
                async with _loop_limiter():
                    response = await self._target.create(*args, **kwargs)
                _record_usage(requested, response)
                return response
            except Exception as exc:
                if not _retryable(exc) or attempt >= retries:
                    _record_failure(requested, exc)
                    if _strict_errors():
                        raise LLMRequestError(str(exc)) from exc
                    raise
                delay = min(max_delay, base_delay * (2**attempt)) * random.uniform(
                    0.75, 1.25
                )
                print(
                    f"[LLM RETRY] {requested} {attempt + 1}/{retries}, "
                    f"sleep={delay:.1f}s"
                )
                await asyncio.sleep(delay)
        raise AssertionError("unreachable")

    def __getattr__(self, name):
        return getattr(self._target, name)


class _TrackedChat:
    def __init__(self, target):
        self._target = target
        self.completions = _TrackedCompletions(target.completions)

    def __getattr__(self, name):
        return getattr(self._target, name)


class _TrackedClient:
    def __init__(self, target):
        self._target = target
        self.chat = _TrackedChat(target.chat)

    def __getattr__(self, name):
        return getattr(self._target, name)


# Compatibility for evolved programs that import helper.client directly.
client = _TrackedClient(_new_client())


def _loop_client() -> AsyncOpenAI:
    loop = asyncio.get_running_loop()
    current = _LOOP_CLIENTS.get(loop)
    if current is None:
        current = _new_client()
        _LOOP_CLIENTS[loop] = current
    return current


def _loop_limiter() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    limiter = _LOOP_LIMITERS.get(loop)
    if limiter is None:
        limiter = asyncio.Semaphore(
            max(1, int(os.environ.get("TRIAGE_ESI_LLM_MAX_CONCURRENCY", "8")))
        )
        _LOOP_LIMITERS[loop] = limiter
    return limiter


def _strict_errors() -> bool:
    return os.environ.get("TRIAGE_ESI_LLM_STRICT", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _requested_model(model: str | None) -> str:
    original = model or MODEL
    forced = os.environ.get("TRIAGE_ESI_FORCE_MODEL", "").strip()
    requested = forced or original
    if not forced and requested not in set(get_models()):
        raise LLMRequestError(f"Model {requested!r} is not listed in {_MODELS_CONFIG_PATH}")
    if forced and original != forced and (original, forced) not in _FORCED_MODEL_NOTICES:
        _FORCED_MODEL_NOTICES.add((original, forced))
        print(f"[LLM MODEL OVERRIDE] {original} -> {forced}")
    return requested


def _retryable(exc: Exception) -> bool:
    return isinstance(exc, (RateLimitError, APITimeoutError, APIConnectionError)) or (
        isinstance(exc, APIStatusError) and exc.status_code >= 500
    )


def _record_failure(model: str, exc: Exception) -> None:
    path = os.environ.get("TRIAGE_ESI_TOKEN_USAGE_PATH", "").strip()
    if path:
        append_jsonl(
            path,
            {
                "timestamp": time.time(),
                "role": "expert",
                "phase": os.environ.get("TRIAGE_ESI_TOKEN_PHASE", "evolution"),
                "experiment_run": os.environ.get("GIGAEVO_EXPERIMENT_RUN", ""),
                "model": model,
                "status": "failed",
                "input_tokens": 0,
                "output_tokens": 0,
                "reasoning_tokens": 0,
                "total_tokens": 0,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )


async def _chat_completion(
    prompt: str,
    system_prompt: str,
    *,
    model: str | None,
    response_format: dict | None = None,
) -> str | None:
    requested = _requested_model(model)
    if requested in _UNAVAILABLE_MODELS:
        raise LLMRequestError(f"Model {requested!r} is unavailable")
    if not API_KEY:
        raise LLMRequestError("TRIAGE_ESI_API_KEY/OPENAI_API_KEY is not set")

    retries = max(0, int(os.environ.get("TRIAGE_ESI_LLM_MAX_RETRIES", "6")))
    base_delay = max(
        0.1, float(os.environ.get("TRIAGE_ESI_LLM_RETRY_BASE_SEC", "1"))
    )
    max_delay = max(
        base_delay, float(os.environ.get("TRIAGE_ESI_LLM_RETRY_MAX_SEC", "30"))
    )
    # OpenAI-compatible backends reject json_object mode unless at least one
    # input message explicitly mentions JSON.  Evolved programs control both
    # prompts, so enforce this provider requirement centrally rather than
    # allowing an otherwise valid candidate to fail every patient request.
    if response_format and response_format.get("type") == "json_object":
        combined = f"{system_prompt}\n{prompt}".lower()
        if "json" not in combined:
            system_prompt = f"{system_prompt}\nReturn a valid JSON object only."

    kwargs = {
        "model": requested,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        # OpenRouter otherwise reserves the model's full output allowance
        # (65,536 tokens for Luna) when checking the key budget.  Triage calls
        # produce short labels/JSON, so a bounded, configurable ceiling avoids
        # spurious HTTP 402 errors without truncating normal outputs.
        "max_tokens": max(
            1, int(os.environ.get("TRIAGE_ESI_LLM_MAX_TOKENS", "4096"))
        ),
    }
    if response_format is not None:
        kwargs["response_format"] = response_format

    for attempt in range(retries + 1):
        try:
            async with _loop_limiter():
                response = await _loop_client().chat.completions.create(**kwargs)
            _record_usage(requested, response)
            content = response.choices[0].message.content
            return content.strip() if content else ""
        except NotFoundError as exc:
            _UNAVAILABLE_MODELS.add(requested)
            _record_failure(requested, exc)
            if _strict_errors():
                raise LLMRequestError(str(exc)) from exc
            return None
        except Exception as exc:
            if not _retryable(exc) or attempt >= retries:
                _record_failure(requested, exc)
                if _strict_errors():
                    raise LLMRequestError(str(exc)) from exc
                print(f"LLM error: {exc}")
                return None
            delay = min(max_delay, base_delay * (2**attempt)) * random.uniform(0.75, 1.25)
            print(
                f"[LLM RETRY] {requested} {attempt + 1}/{retries}, "
                f"sleep={delay:.1f}s"
            )
            await asyncio.sleep(delay)
    return None


def measure_time(result_key: str = "execution_time_sec"):
    def decorator(func):
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                start = time.perf_counter()
                result = await func(*args, **kwargs)
                if isinstance(result, dict):
                    result[result_key] = time.perf_counter() - start
                return result

            return async_wrapper

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            if isinstance(result, dict):
                result[result_key] = time.perf_counter() - start
            return result

        return sync_wrapper

    return decorator


async def ask_llm(
    prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    model: str | None = None,
) -> str:
    return (await _chat_completion(prompt, system_prompt, model=model)) or ""


async def ask_llm_json(
    prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    model: str | None = None,
) -> dict:
    raw = await _chat_completion(
        prompt,
        system_prompt,
        model=model,
        response_format={"type": "json_object"},
    )
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        import re

        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
        return {}
