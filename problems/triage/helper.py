from openai import AsyncOpenAI, DefaultAsyncHttpxClient
import functools
import inspect
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv


# Set TRIAGE_LLM_BASE_URL to the base URL of your OpenAI-compatible inference endpoint
INFERENCE_BASE_URL = os.environ.get("TRIAGE_LLM_BASE_URL", "")
load_dotenv()
API_KEY = os.environ.get("OPENAI_API_KEY", "")

_MODELS_CONFIG_PATH = Path(__file__).parent / "models.json"
_ESI_KB_PATH = Path(__file__).parent / "data" / "esi_reference_kb.json"
_DEFAULT_MODEL = os.environ.get("TRIAGE_MODEL_NAME", "")
_ESI_KB_CACHE: list[dict[str, str]] | None = None
_ESI_KB_TEXT_CACHE: str | None = None


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
    """Load the local ESI/triage reference KB for candidate programs."""
    global _ESI_KB_CACHE
    if _ESI_KB_CACHE is not None:
        return _ESI_KB_CACHE

    try:
        data = json.loads(_ESI_KB_PATH.read_text(encoding="utf-8"))
        if isinstance(data, list):
            _ESI_KB_CACHE = [
                row
                for row in data
                if isinstance(row, dict) and "title" in row and "text" in row
            ]
        else:
            _ESI_KB_CACHE = []
    except Exception:
        _ESI_KB_CACHE = []
    return _ESI_KB_CACHE


def load_esi_kb_text(max_chars: int = 120_000) -> str:
    """Return the local ESI/triage reference KB as prompt-ready text."""
    global _ESI_KB_TEXT_CACHE
    if _ESI_KB_TEXT_CACHE is not None and len(_ESI_KB_TEXT_CACHE) <= max_chars:
        return _ESI_KB_TEXT_CACHE

    parts = []
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

client = AsyncOpenAI(
    base_url=INFERENCE_BASE_URL,
    api_key=API_KEY,
    http_client=DefaultAsyncHttpxClient(verify=False),
)


def measure_time(result_key: str = "execution_time_sec"):
    def decorator(func):
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                start = time.perf_counter()
                result = await func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                if isinstance(result, dict):
                    result[result_key] = elapsed
                return result

            return async_wrapper

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            if isinstance(result, dict):
                result[result_key] = elapsed
            return result

        return sync_wrapper

    return decorator


async def ask_llm(
    prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    model: str | None = None,
) -> str:
    if not API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set")
    try:
        response = await client.chat.completions.create(
            model=model or MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        print(f"LLM error: {exc}")
        return ""


async def ask_llm_json(
    prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    model: str | None = None,
) -> dict:
    if not API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set")
    try:
        response = await client.chat.completions.create(
            model=model or MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content.strip()
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
    except Exception as exc:
        print(f"LLM JSON error: {exc}")
        return {}
