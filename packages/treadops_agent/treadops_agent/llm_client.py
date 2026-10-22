"""Portable OpenAI-compatible chat client (ported from AI Cortex `app/ai/llm_client.py`).

Works against OpenAI, Azure OpenAI, LiteLLM, DeepSeek, vLLM, Ollama (/v1) — anything that speaks the
OpenAI chat-completions protocol. Only dependency: ``pip install "openai>=1.55"``.

Config (env, or pass explicitly):
    LLM_BASE_URL     e.g. https://api.openai.com/v1
    LLM_API_KEY
    LLM_MAX_RETRIES  default 2 (0 is honoured)

Transports:
    complete()           one non-streaming call -> full text. Prefer this when you don't render tokens:
                         non-streaming responses carry prompt_tokens_details (cached-token telemetry);
                         many gateways omit it on streams.
    chat_stream()        token-by-token generator of str.
    chat_tools()         non-streaming tool-use -> assistant message dict (incl. tool_calls).
    chat_tools_stream()  streamed tool-use: yields {"type": "content"|"tool_call"|"message", ...}.

After every call `client.last_usage` holds
    {prompt_tokens, completion_tokens, total_tokens, cached_prompt_tokens}
(`prompt_tokens` is inclusive of the cached subset). Zeros when the gateway reports nothing.

Thread-safety: the underlying SDK client / connection pool is shared process-wide and thread-safe, but
`last_usage` lives on the LLMClient *instance* — build one LLMClient per request/task (cheap), don't
share one across concurrent calls if you read `last_usage`.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable, Iterator
from typing import Any

__all__ = [
    "LLMClient",
    "build_llm_client",
    "complete_text",
    "normalize_usage",
    "reset_sdk_clients",
]

# ── Token usage parsing ──────────────────────────────────────────────────────────────────────────
# Three provider shapes for "cached input tokens", in priority order:
#   usage.prompt_tokens_details.cached_tokens   (OpenAI / Azure / LiteLLM)
#   usage.prompt_cache_hit_tokens               (DeepSeek native)
#   usage.cache_read_input_tokens               (Anthropic, incl. via LiteLLM)
# Total by contract: never raises, never returns None; unknown shapes yield zeros.

_FLAT_CACHED_KEYS = ("prompt_cache_hit_tokens", "cache_read_input_tokens")


def _get(obj: Any, key: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _int(value: Any) -> int:
    if value is None or isinstance(value, bool):
        return 0
    try:
        out = int(value)
    except (TypeError, ValueError):
        return 0
    return out if out > 0 else 0


def _cached_tokens(usage: Any) -> int:
    cached = _int(_get(_get(usage, "prompt_tokens_details"), "cached_tokens"))
    if cached:
        return cached
    for key in _FLAT_CACHED_KEYS:
        cached = _int(_get(usage, key))
        if cached:
            return cached
    return 0


def normalize_usage(usage: Any) -> dict[str, int]:
    """`{prompt_tokens, completion_tokens, total_tokens, cached_prompt_tokens}` from any provider."""
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cached_prompt_tokens": 0}
    prompt = _int(_get(usage, "prompt_tokens"))
    completion = _int(_get(usage, "completion_tokens"))
    total = _int(_get(usage, "total_tokens")) or (prompt + completion)
    # Clamp: cached is a subset of prompt, so it can never exceed it.
    cached = min(_cached_tokens(usage), prompt)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "cached_prompt_tokens": cached,
    }


# ── Shared SDK clients (connection reuse) ────────────────────────────────────────────────────────
# Building a fresh `OpenAI` per request throws away the HTTPS pool: ~3.5s cold vs ~0.7s warm on the
# gateway this was measured against. So the pool is process-wide, keyed on what distinguishes one.

_SDK_CLIENTS: dict[tuple[str, str, int], Any] = {}
_SDK_LOCK = threading.Lock()


def _shared_sdk_client(base_url: str, api_key: str, max_retries: int) -> Any:
    key = (base_url, api_key, max_retries)
    client = _SDK_CLIENTS.get(key)
    if client is not None:
        return client
    with _SDK_LOCK:
        client = _SDK_CLIENTS.get(key)  # re-check inside the lock
        if client is None:
            from openai import OpenAI

            client = OpenAI(base_url=base_url, api_key=api_key, max_retries=max_retries)
            _SDK_CLIENTS[key] = client
    return client


def reset_sdk_clients() -> None:
    """Drop the cached SDK clients (tests; or after a base-url/key change at runtime)."""
    with _SDK_LOCK:
        _SDK_CLIENTS.clear()


# ── The client ───────────────────────────────────────────────────────────────────────────────────


class LLMClient:
    """Thin wrapper over OpenAI-compatible chat completions."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        sdk_client: Any | None = None,  # injectable for tests
        max_retries: int = 2,
        collapse_non_streaming: bool = True,
    ) -> None:
        self.base_url = base_url
        self._api_key = api_key
        self._client = sdk_client
        # False -> complete() drains chat_stream instead (escape hatch for a gateway that
        # misbehaves on non-streaming calls).
        self._collapse_non_streaming = collapse_non_streaming
        # SDK-level retries with exponential backoff + jitter on 429 / 5xx / timeout.
        self._max_retries = max_retries
        # Provider-reported usage for the most recent call (see module docstring).
        self.last_usage: dict[str, int] | None = None

    def _ensure_client(self) -> Any:
        if self._client is None:
            self._client = _shared_sdk_client(self.base_url, self._api_key, self._max_retries)
        return self._client

    def _client_for(self, timeout: float | None, max_retries: int | None) -> Any:
        """The shared client, or a per-call copy (same connection pool) with its own timeout/retries.

        Without a timeout the SDK waits 600s per attempt, and a non-streaming call must finish the
        WHOLE reply inside it. For streams, `timeout` bounds each wait for the next chunk, not the
        whole reply. None for both returns the shared client unchanged.
        """
        client = self._ensure_client()
        opts: dict[str, Any] = {}
        if timeout is not None:
            opts["timeout"] = timeout
        if max_retries is not None:
            opts["max_retries"] = max_retries
        return client.with_options(**opts) if opts else client

    # -- non-streaming text ------------------------------------------------------------------

    def complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> str:
        """One non-streaming completion -> the whole answer text."""
        self.last_usage = None
        if not self._collapse_non_streaming:
            return "".join(
                self.chat_stream(
                    model, messages, temperature=temperature, max_tokens=max_tokens,
                    timeout=timeout, max_retries=max_retries,
                )
            )
        kwargs: dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        resp = self._client_for(timeout, max_retries).chat.completions.create(**kwargs)
        usage = getattr(resp, "usage", None)
        if usage is not None:
            self.last_usage = normalize_usage(usage)
        choices = getattr(resp, "choices", None)
        if not choices:
            return ""
        return str(getattr(choices[0].message, "content", "") or "")

    # -- streaming text ----------------------------------------------------------------------

    def chat_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> Iterator[str]:
        """Yield answer tokens for a streamed chat completion."""
        self.last_usage = None
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},  # final usage-only chunk carries the counts
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        stream = self._client_for(timeout, max_retries).chat.completions.create(**kwargs)
        for chunk in stream:
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                self.last_usage = normalize_usage(usage)
            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            content = getattr(delta, "content", None) if delta is not None else None
            if content:
                yield content

    # -- tool use ----------------------------------------------------------------------------

    def chat_tools(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        tool_choice: str | dict[str, Any] = "auto",
        temperature: float = 0.2,
        max_tokens: int = 1024,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> dict[str, Any]:
        """Non-streaming tool-use completion -> the assistant message dict (incl. `tool_calls`).

        An endpoint without tool support raises here (SDK error) — catch it and fall back to a
        prompt-based (ReAct-style) loop if you need to support such gateways.
        """
        self.last_usage = None
        resp = self._client_for(timeout, max_retries).chat.completions.create(
            model=model, messages=messages, tools=tools, tool_choice=tool_choice,
            temperature=temperature, max_tokens=max_tokens,
        )
        usage = getattr(resp, "usage", None)
        if usage is not None:
            self.last_usage = normalize_usage(usage)
        return resp.choices[0].message.model_dump()

    def chat_tools_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        tool_choice: str | dict[str, Any] = "auto",
        temperature: float = 0.2,
        max_tokens: int = 1024,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Streamed tool-use: the answer as it is written, or the tools it calls.

        Yields
            {"type": "content", "text": str}     each answer delta
            {"type": "tool_call"}                once, when the first tool-call delta arrives
            {"type": "message", "message": {...}}  last; the assembled assistant message, same shape
                                                   as `chat_tools` (tool_calls stitched from deltas)

        Lets an agent's final round BE its answer instead of a decide-round plus a second call.
        """
        self.last_usage = None
        stream = self._client_for(timeout, max_retries).chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )
        content: list[str] = []
        calls: dict[int, dict[str, Any]] = {}
        announced = False
        for chunk in stream:
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                self.last_usage = normalize_usage(usage)
            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            if delta is None:
                continue
            text = getattr(delta, "content", None)
            if text:
                content.append(text)
                yield {"type": "content", "text": text}
            for part in getattr(delta, "tool_calls", None) or []:
                index = int(getattr(part, "index", None) or 0)
                call = calls.setdefault(
                    index, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
                )
                if getattr(part, "id", None):
                    call["id"] = part.id
                fn = getattr(part, "function", None)
                if fn is not None:
                    call["function"]["name"] += getattr(fn, "name", None) or ""
                    call["function"]["arguments"] += getattr(fn, "arguments", None) or ""
                if not announced:
                    announced = True
                    yield {"type": "tool_call"}
        message: dict[str, Any] = {"role": "assistant", "content": "".join(content)}
        if calls:
            message["tool_calls"] = [
                {**call, "id": call["id"] or f"call_{index}"} for index, call in sorted(calls.items())
            ]
        yield {"type": "message", "message": message}


# ── Construction + helpers ───────────────────────────────────────────────────────────────────────


def build_llm_client(
    base_url: str | None = None,
    api_key: str | None = None,
    *,
    max_retries: int | None = None,
    collapse_non_streaming: bool = True,
) -> LLMClient:
    """Build a client from arguments, falling back to LLM_BASE_URL / LLM_API_KEY / LLM_MAX_RETRIES."""
    base_url = base_url or os.environ.get("LLM_BASE_URL", "")
    api_key = api_key or os.environ.get("LLM_API_KEY", "")
    if not base_url:
        raise ValueError("LLM base URL not set: pass base_url or set LLM_BASE_URL")
    if max_retries is None:
        raw = os.environ.get("LLM_MAX_RETRIES")
        # Explicit None-check, not `or 2`: a configured 0 (no retries) must be honoured.
        max_retries = int(raw) if raw not in (None, "") else 2
    return LLMClient(
        base_url, api_key, max_retries=max_retries, collapse_non_streaming=collapse_non_streaming
    )


def complete_text(
    llm: Any,
    model: str,
    messages: list[dict[str, Any]],
    *,
    temperature: float = 0.2,
    max_tokens: int | None = None,
    on_chunk: Callable[[], None] | None = None,
) -> str:
    """Full answer as one string via `llm.complete` if the client has it, else by draining
    `llm.chat_stream`. `on_chunk` fires per streamed chunk on the fallback path only."""
    complete = getattr(llm, "complete", None)
    if callable(complete):
        return complete(model, messages, temperature=temperature, max_tokens=max_tokens)
    parts: list[str] = []
    for chunk in llm.chat_stream(model, messages, temperature=temperature, max_tokens=max_tokens):
        parts.append(chunk)
        if on_chunk is not None:
            on_chunk()
    return "".join(parts)
