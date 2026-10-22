#!/usr/bin/env python3
"""MLOps ACC: Ark adapter, request logs, Motor lifecycle and dynamic registration."""

from __future__ import annotations

import argparse
import asyncio
import codecs
import json
import logging
import os
import re
import shutil
import signal
import sys
import threading
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager, contextmanager, suppress
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import anyio
import httpx
import uvicorn
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

# Ark HTTP protocol adapter

LOGGER = logging.getLogger("ark-pd-http-adapter")


FORWARDED_REQUEST_HEADERS = ("x-request-id", "x-tt-logid", "traceparent")


FORWARDED_RESPONSE_HEADERS = ("x-request-id", "x-tt-logid", "traceparent")


INTERNAL_REQUEST_FIELDS = ("kv_transfer_params", "ec_transfer_params")


ARK_REASONING_EFFORTS = (
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
)


ARK_EFFORT_TO_VLLM = {
    "none": "none",
    "minimal": "none",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "max",
    "max": "max",
}


class RequestValidationError(ValueError):
    """Raised when the Ark request cannot safely be normalized."""


class UpstreamProtocolError(ValueError):
    """Raised when the native proxy returns an unusable streaming response."""


@dataclass(frozen=True)
class AdapterSettings:
    upstream: str = "http://127.0.0.1:8090"
    served_model: str = "deepseek_v4"
    expected_prefill: int = 1
    expected_decode: int = 4
    connect_timeout_seconds: float = 5.0
    write_timeout_seconds: float = 60.0

    def normalized_upstream(self) -> str:
        return self.upstream.rstrip("/")


def _error_payload(message: str, *, error_type: str, code: str) -> dict[str, Any]:
    return {
        "error": {
            "message": message,
            "type": error_type,
            "param": None,
            "code": code,
        }
    }


def _safe_request_id(value: str | None) -> str:
    if value:
        sanitized = "".join(ch for ch in value if ch.isprintable()).strip()
        if sanitized:
            return sanitized[:128]
    return f"adapter-{uuid.uuid4().hex}"


def _normalize_tool_arguments(function: dict[str, Any]) -> None:
    arguments = function.get("arguments")
    if arguments is None:
        function["arguments"] = "{}"
        return
    if isinstance(arguments, (dict, list)):
        function["arguments"] = json.dumps(
            arguments,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return
    if not isinstance(arguments, str):
        function["arguments"] = "{}"
        return
    try:
        json.loads(arguments)
    except (TypeError, ValueError):
        # Ark middleware can truncate historical tool arguments.  Keeping the
        # malformed value makes vLLM reject the whole follow-up request.
        function["arguments"] = "{}"


def normalize_chat_request(
    payload: Any,
    served_model: str,
) -> dict[str, Any]:
    """Normalize an Ark chat request for native vLLM chat completions.

    The object is mutated in place to avoid copying large prompts.
    """
    if not isinstance(payload, dict):
        raise RequestValidationError("request body must be a JSON object")

    messages = payload.get("messages")
    if not isinstance(messages, list):
        raise RequestValidationError("messages must be a JSON array")

    requested_stream = payload.get("stream", False)
    if not isinstance(requested_stream, bool):
        raise RequestValidationError("stream must be a boolean")

    payload["model"] = served_model
    payload["stream"] = requested_stream
    if requested_stream:
        stream_options = payload.get("stream_options")
        if stream_options is None:
            stream_options = {}
        if not isinstance(stream_options, dict):
            raise RequestValidationError("stream_options must be a JSON object")
        stream_options["include_usage"] = True
        payload["stream_options"] = stream_options
    else:
        payload.pop("stream_options", None)

    if (
        payload.get("max_completion_tokens") is None
        and payload.get("max_tokens") is None
        and payload.get("max_new_tokens") is not None
    ):
        payload["max_tokens"] = payload["max_new_tokens"]
    payload.pop("max_new_tokens", None)

    for field in INTERNAL_REQUEST_FIELDS:
        payload.pop(field, None)

    thinking = payload.pop("thinking", None)
    thinking_type = None
    if thinking is not None:
        if not isinstance(thinking, dict):
            raise RequestValidationError("thinking must be a JSON object")
        thinking_type = thinking.get("type")
        if thinking_type not in ("enabled", "disabled"):
            raise RequestValidationError("thinking.type must be 'enabled' or 'disabled'")
    reasoning_effort = payload.get("reasoning_effort")
    if reasoning_effort is not None:
        if reasoning_effort not in ARK_REASONING_EFFORTS:
            raise RequestValidationError("reasoning_effort must be one of: " + ", ".join(ARK_REASONING_EFFORTS))
        if thinking_type == "disabled" and reasoning_effort not in ("none", "minimal"):
            raise RequestValidationError("thinking.type='disabled' conflicts with non-minimal reasoning_effort")
        payload["reasoning_effort"] = ARK_EFFORT_TO_VLLM[reasoning_effort]
    elif thinking_type is not None:
        payload["reasoning_effort"] = "max" if thinking_type == "enabled" else "none"
    else:
        payload["reasoning_effort"] = "max"

    for message in messages:
        if not isinstance(message, dict):
            raise RequestValidationError("every message must be a JSON object")
        if message.get("name") is None:
            message.pop("name", None)
        tool_calls = message.get("tool_calls")
        if tool_calls is None:
            continue
        if not isinstance(tool_calls, list):
            raise RequestValidationError("message.tool_calls must be a JSON array")
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                raise RequestValidationError("every message.tool_calls item must be a JSON object")
            function = tool_call.get("function")
            if function is None:
                continue
            if not isinstance(function, dict):
                raise RequestValidationError("message.tool_calls[].function must be a JSON object")
            _normalize_tool_arguments(function)

    tools = payload.get("tools")
    if tools is not None and not isinstance(tools, list):
        raise RequestValidationError("tools must be a JSON array")
    if tools and payload.get("tool_choice") is None:
        payload["tool_choice"] = "auto"

    payload.pop("chat_template", None)
    payload.pop("chat_template_kwargs", None)

    if payload.get("max_tokens") is None and payload.get("max_completion_tokens") is not None:
        payload["max_tokens"] = payload["max_completion_tokens"]
    payload.pop("max_completion_tokens", None)

    max_tokens_limit = int(os.getenv("XLLM_MAX_NEW_TOKENS", "393216"))
    request_max_tokens = payload.get("max_tokens")
    payload["max_tokens"] = (
        min(request_max_tokens, max_tokens_limit)
        if request_max_tokens is not None and request_max_tokens > 0
        else max_tokens_limit
    )

    return payload


def _translate_reasoning_fields(document: Any) -> Any:
    if not isinstance(document, dict):
        return document
    choices = document.get("choices")
    if not isinstance(choices, list):
        return document
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        for field in ("delta", "message"):
            item = choice.get(field)
            if not isinstance(item, dict) or "reasoning" not in item:
                continue
            if "reasoning_content" not in item:
                item["reasoning_content"] = item["reasoning"]
            item.pop("reasoning", None)
    return document


def translate_json_response(body: bytes) -> bytes:
    try:
        document = json.loads(body)
    except (TypeError, ValueError):
        return body
    document = _translate_reasoning_fields(document)
    return json.dumps(
        document,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def translate_sse_line(line: str) -> str:
    if not line.startswith("data:"):
        return line
    data = line[5:].strip()
    if not data or data == "[DONE]":
        return line
    try:
        document = json.loads(data)
    except (TypeError, ValueError):
        return line
    document = _translate_reasoning_fields(document)
    return "data: " + json.dumps(
        document,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _merge_text_fragment(
    target: dict[str, Any],
    field: str,
    value: Any,
) -> None:
    if value is None:
        return
    if not isinstance(value, str):
        raise UpstreamProtocolError(f"delta.{field} must be a string or null")
    fragments = target.setdefault(field, [])
    if not isinstance(fragments, list):
        raise UpstreamProtocolError(f"delta.{field} changed type while streaming")
    fragments.append(value)


def _merge_function_fragment(
    target: dict[str, Any],
    fragment: Any,
    *,
    field: str,
) -> None:
    if fragment is None:
        return
    if not isinstance(fragment, dict):
        raise UpstreamProtocolError(f"delta.{field} must be a JSON object")
    for name, value in fragment.items():
        if value is None:
            continue
        if name == "arguments":
            if not isinstance(value, str):
                raise UpstreamProtocolError(f"delta.{field}.{name} must be a string or null")
            fragments = target.setdefault(name, [])
            if not isinstance(fragments, list):
                raise UpstreamProtocolError(f"delta.{field}.{name} changed type while streaming")
            fragments.append(value)
        elif name == "name":
            if not isinstance(value, str):
                raise UpstreamProtocolError(f"delta.{field}.{name} must be a string or null")
            # GLM's terminal tool-call chunk can repeat the full function
            # name while only `arguments` is a continuation.
            if name not in target or (not target[name] and value):
                target[name] = value
        else:
            target[name] = value


def _merge_tool_call_fragments(
    target: dict[int, dict[str, Any]],
    fragments: Any,
) -> None:
    if fragments is None:
        return
    if not isinstance(fragments, list):
        raise UpstreamProtocolError("delta.tool_calls must be a JSON array")
    for position, fragment in enumerate(fragments):
        if not isinstance(fragment, dict):
            raise UpstreamProtocolError("every delta.tool_calls item must be a JSON object")
        index = fragment.get("index", position)
        if isinstance(index, bool) or not isinstance(index, int):
            raise UpstreamProtocolError("delta.tool_calls[].index must be an integer")
        tool_call = target.setdefault(index, {})
        for name, value in fragment.items():
            if name in ("index", "function") or value is None:
                continue
            if name in ("id", "type") and name in tool_call and not value:
                continue
            tool_call[name] = value
        function = fragment.get("function")
        if function is not None:
            function_target = tool_call.setdefault("function", {})
            if not isinstance(function_target, dict):
                raise UpstreamProtocolError("delta.tool_calls[].function changed type while streaming")
            _merge_function_fragment(
                function_target,
                function,
                field="tool_calls[].function",
            )


def _merge_logprobs(current: Any, fragment: Any) -> Any:
    if fragment is None:
        return current
    if current is None:
        if not isinstance(fragment, dict):
            return fragment
        return {name: list(value) if isinstance(value, list) else value for name, value in fragment.items()}
    if not isinstance(current, dict) or not isinstance(fragment, dict):
        return fragment
    for name, value in fragment.items():
        if name in ("content", "refusal") and isinstance(value, list):
            existing = current.get(name)
            if not isinstance(existing, list):
                existing = []
                current[name] = existing
            existing.extend(value)
        elif value is not None:
            current[name] = value
    return current


def _finalize_function(function: dict[str, Any]) -> dict[str, Any]:
    finalized = dict(function)
    arguments = finalized.get("arguments")
    if isinstance(arguments, list):
        finalized["arguments"] = "".join(arguments)
    return finalized


def _finalize_tool_calls(
    tool_calls: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    finalized: list[dict[str, Any]] = []
    for tool_index in sorted(tool_calls):
        tool_call = dict(tool_calls[tool_index])
        function = tool_call.get("function")
        if isinstance(function, dict):
            tool_call["function"] = _finalize_function(function)
        finalized.append(tool_call)
    return finalized


class _ChatCompletionAccumulator:
    """Incrementally turn vLLM chat-completion chunks into one response."""

    _TOP_LEVEL_FIELDS = (
        "id",
        "created",
        "model",
        "service_tier",
        "system_fingerprint",
    )

    def __init__(self) -> None:
        self.metadata: dict[str, Any] = {}
        self.choices: dict[int, dict[str, Any]] = {}
        self.usage: Any = None
        self.saw_document = False

    def add(self, document: Any) -> None:
        if not isinstance(document, dict):
            raise UpstreamProtocolError("SSE data must be a JSON object")
        if document.get("error") is not None:
            raise UpstreamProtocolError("upstream returned an error event")

        self.saw_document = True
        document = _translate_reasoning_fields(document)
        for field in self._TOP_LEVEL_FIELDS:
            if field in document and (
                field not in self.metadata or (self.metadata[field] is None and document[field] is not None)
            ):
                self.metadata[field] = document[field]
        if document.get("usage") is not None:
            self.usage = document["usage"]

        raw_choices = document.get("choices")
        if raw_choices is None:
            return
        if not isinstance(raw_choices, list):
            raise UpstreamProtocolError("choices must be a JSON array")
        for raw_choice in raw_choices:
            if not isinstance(raw_choice, dict):
                raise UpstreamProtocolError("every choices item must be a JSON object")
            index = raw_choice.get("index", 0)
            if isinstance(index, bool) or not isinstance(index, int):
                raise UpstreamProtocolError("choice.index must be an integer")
            choice = self.choices.setdefault(
                index,
                {
                    "message": {"role": "assistant"},
                    "tool_calls": {},
                    "function_call": {},
                    "content_seen": False,
                    "reasoning_seen": False,
                    "refusal_seen": False,
                    "logprobs": None,
                    "finish_reason": None,
                    "extras": {},
                },
            )

            delta = raw_choice.get("delta")
            if delta is None:
                delta = raw_choice.get("message")
            if delta is not None and not isinstance(delta, dict):
                raise UpstreamProtocolError("choice.delta must be a JSON object or null")
            if isinstance(delta, dict):
                role = delta.get("role")
                if role is not None:
                    if not isinstance(role, str):
                        raise UpstreamProtocolError("delta.role must be a string or null")
                    choice["message"]["role"] = role
                if "content" in delta and delta["content"] is not None:
                    choice["content_seen"] = True
                    _merge_text_fragment(
                        choice["message"],
                        "content",
                        delta["content"],
                    )
                if "reasoning_content" in delta and delta["reasoning_content"] is not None:
                    choice["reasoning_seen"] = True
                    _merge_text_fragment(
                        choice["message"],
                        "reasoning_content",
                        delta["reasoning_content"],
                    )
                if "refusal" in delta and delta["refusal"] is not None:
                    choice["refusal_seen"] = True
                    _merge_text_fragment(
                        choice["message"],
                        "refusal",
                        delta["refusal"],
                    )
                _merge_tool_call_fragments(
                    choice["tool_calls"],
                    delta.get("tool_calls"),
                )
                _merge_function_fragment(
                    choice["function_call"],
                    delta.get("function_call"),
                    field="function_call",
                )
                if delta.get("audio") is not None:
                    choice["message"]["audio"] = delta["audio"]

            choice["logprobs"] = _merge_logprobs(
                choice["logprobs"],
                raw_choice.get("logprobs"),
            )
            if raw_choice.get("finish_reason") is not None:
                choice["finish_reason"] = raw_choice["finish_reason"]
            for field in ("stop_reason",):
                if raw_choice.get(field) is not None:
                    choice["extras"][field] = raw_choice[field]
            token_ids = raw_choice.get("token_ids")
            if token_ids is not None:
                if not isinstance(token_ids, list):
                    raise UpstreamProtocolError("choice.token_ids must be a JSON array or null")
                choice["extras"].setdefault("token_ids", []).extend(token_ids)

    def build(self) -> dict[str, Any]:
        if not self.saw_document:
            raise UpstreamProtocolError("upstream SSE contained no JSON data")
        if not self.choices:
            raise UpstreamProtocolError("upstream SSE contained no choices")
        if self.usage is None:
            raise UpstreamProtocolError("upstream SSE contained no final usage")

        result = dict(self.metadata)
        result["object"] = "chat.completion"
        result_choices: list[dict[str, Any]] = []
        for index in sorted(self.choices):
            state = self.choices[index]
            if state["finish_reason"] is None:
                raise UpstreamProtocolError(f"upstream SSE choice {index} did not finish")
            message = {
                name: value
                for name, value in state["message"].items()
                if name not in ("content", "reasoning_content", "refusal")
            }
            for field, seen_field in (
                ("content", "content_seen"),
                ("reasoning_content", "reasoning_seen"),
                ("refusal", "refusal_seen"),
            ):
                if state[seen_field]:
                    message[field] = "".join(state["message"].get(field, []))
            if not state["content_seen"]:
                message["content"] = None
            if state["tool_calls"]:
                message["tool_calls"] = _finalize_tool_calls(state["tool_calls"])
            if state["function_call"]:
                message["function_call"] = _finalize_function(state["function_call"])

            choice = {
                "index": index,
                "message": message,
                "logprobs": state["logprobs"],
                "finish_reason": state["finish_reason"],
            }
            choice.update(state["extras"])
            result_choices.append(choice)
        result["choices"] = result_choices
        result["usage"] = self.usage
        return result


def aggregate_chat_completion_chunks(
    documents: list[dict[str, Any]],
) -> dict[str, Any]:
    accumulator = _ChatCompletionAccumulator()
    for document in documents:
        accumulator.add(document)
    return accumulator.build()


def _copy_request_headers(headers: Mapping[str, str]) -> dict[str, str]:
    copied: dict[str, str] = {}
    for name in FORWARDED_REQUEST_HEADERS:
        value = headers.get(name)
        if value:
            copied[name] = value
    return copied


def _copy_response_headers(headers: Mapping[str, str]) -> dict[str, str]:
    copied: dict[str, str] = {}
    for name in FORWARDED_RESPONSE_HEADERS:
        value = headers.get(name)
        if value:
            copied[name] = value
    return copied


async def _translated_sse_stream(
    response: httpx.Response,
    request_id: str,
    started_at: float,
) -> AsyncIterator[bytes]:
    try:
        async for line in response.aiter_lines():
            translated = translate_sse_line(line)
            yield (translated + "\n").encode("utf-8")
    except asyncio.CancelledError:
        raise
    except (httpx.HTTPError, OSError) as error:
        LOGGER.warning(
            "request_id=%s streaming_closed error_type=%s",
            request_id,
            type(error).__name__,
        )
        raise
    finally:
        # Starlette cancels the streaming task when the downstream disconnects.
        # Shield the pool release so a cancelled request cannot leak the
        # upstream connection.
        with anyio.CancelScope(shield=True):
            await response.aclose()
        LOGGER.info(
            "request_id=%s completed elapsed_ms=%.1f",
            request_id,
            (time.monotonic() - started_at) * 1000.0,
        )


async def _aggregate_sse_response(
    response: httpx.Response,
    request_id: str,
    started_at: float,
) -> dict[str, Any]:
    accumulator = _ChatCompletionAccumulator()
    saw_done = False
    try:
        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data:
                continue
            if data == "[DONE]":
                saw_done = True
                break
            try:
                document = json.loads(data)
            except (TypeError, ValueError) as error:
                raise UpstreamProtocolError("upstream SSE contained invalid JSON data") from error
            accumulator.add(document)
        if not saw_done:
            raise UpstreamProtocolError("upstream SSE ended before [DONE]")
        return accumulator.build()
    finally:
        with anyio.CancelScope(shield=True):
            await response.aclose()
        LOGGER.info(
            "request_id=%s completed elapsed_ms=%.1f",
            request_id,
            (time.monotonic() - started_at) * 1000.0,
        )


async def _wait_for_http_disconnect(
    receive: Callable[[], Awaitable[Mapping[str, Any]]],
) -> None:
    while True:
        message = await receive()
        if message.get("type") == "http.disconnect":
            return


async def _send_until_disconnect(
    send: Awaitable[httpx.Response],
    receive: Callable[[], Awaitable[Mapping[str, Any]]],
) -> httpx.Response | None:
    """Send upstream while the downstream is still waiting for response headers."""
    send_task = asyncio.create_task(send)
    disconnect_task = asyncio.create_task(_wait_for_http_disconnect(receive))
    try:
        done, _ = await asyncio.wait(
            (send_task, disconnect_task),
            return_when=asyncio.FIRST_COMPLETED,
        )
    except asyncio.CancelledError:
        send_task.cancel()
        disconnect_task.cancel()
        with anyio.CancelScope(shield=True):
            await asyncio.gather(
                send_task,
                disconnect_task,
                return_exceptions=True,
            )
        raise

    if disconnect_task in done:
        if not send_task.done():
            send_task.cancel()
        try:
            response = await send_task
        except (asyncio.CancelledError, httpx.HTTPError):
            response = None
        if response is not None:
            with anyio.CancelScope(shield=True):
                await response.aclose()
        return None

    disconnect_task.cancel()
    with suppress(asyncio.CancelledError):
        await disconnect_task
    return await send_task


async def _aggregate_until_disconnect(
    response: httpx.Response,
    receive: Callable[[], Awaitable[Mapping[str, Any]]],
    request_id: str,
    started_at: float,
) -> dict[str, Any] | None:
    aggregate_task = asyncio.create_task(_aggregate_sse_response(response, request_id, started_at))
    disconnect_task = asyncio.create_task(_wait_for_http_disconnect(receive))
    try:
        done, _ = await asyncio.wait(
            (aggregate_task, disconnect_task),
            return_when=asyncio.FIRST_COMPLETED,
        )
    except asyncio.CancelledError:
        aggregate_task.cancel()
        disconnect_task.cancel()
        with anyio.CancelScope(shield=True):
            await asyncio.gather(
                aggregate_task,
                disconnect_task,
                return_exceptions=True,
            )
        raise

    if disconnect_task in done:
        if not aggregate_task.done():
            aggregate_task.cancel()
        with anyio.CancelScope(shield=True):
            await asyncio.gather(aggregate_task, return_exceptions=True)
        return None

    disconnect_task.cancel()
    with suppress(asyncio.CancelledError):
        await disconnect_task
    return await aggregate_task


def create_app(
    settings: AdapterSettings,
    *,
    readiness_check: Callable[[], Awaitable[Response]] | None = None,
) -> FastAPI:
    timeout = httpx.Timeout(
        connect=settings.connect_timeout_seconds,
        read=None,
        write=settings.write_timeout_seconds,
        pool=settings.connect_timeout_seconds,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.client = httpx.AsyncClient(
            base_url=settings.normalized_upstream(),
            timeout=timeout,
            limits=httpx.Limits(
                max_connections=4096,
                max_keepalive_connections=1024,
            ),
        )
        try:
            yield
        finally:
            await app.state.client.aclose()

    app = FastAPI(lifespan=lifespan)

    @app.get("/livez")
    async def livez() -> dict[str, str]:
        return {"status": "ok"}

    async def readiness() -> Response:
        if readiness_check is not None:
            return await readiness_check()
        try:
            response = await app.state.client.get("/healthcheck", timeout=5.0)
            response.raise_for_status()
            document = response.json()
            ready = (
                isinstance(document, dict)
                and document.get("status") == "ok"
                and document.get("prefill_instances") == settings.expected_prefill
                and document.get("decode_instances") == settings.expected_decode
            )
            if not ready:
                return JSONResponse(
                    status_code=503,
                    content={
                        "status": "not_ready",
                        "reason": "unexpected_proxy_topology",
                        "prefill_instances": document.get("prefill_instances") if isinstance(document, dict) else None,
                        "decode_instances": document.get("decode_instances") if isinstance(document, dict) else None,
                    },
                )
            return JSONResponse(
                content={
                    "status": "ok",
                    "prefill_instances": settings.expected_prefill,
                    "decode_instances": settings.expected_decode,
                }
            )
        except (httpx.HTTPError, TypeError, ValueError) as error:
            LOGGER.warning("readiness_failed error_type=%s", type(error).__name__)
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "reason": "proxy_unavailable"},
            )

    app.add_api_route("/readyz", readiness, methods=["GET"])
    app.add_api_route("/healthcheck", readiness, methods=["GET"])

    # Keep the adapter at the HTTP boundary. Native vLLM owns the DeepSeek-V4
    # chat template, reasoning parser, and tool-call parser.
    @app.post("/inner/api/v3/responses/create")
    @app.post("/inner/api/v3/chat/completions")
    async def chat_completions(request: Request) -> Response:
        started_at = time.monotonic()
        request_id = _safe_request_id(request.headers.get("x-request-id"))
        content_type = request.headers.get("content-type", "")
        if not content_type.lower().startswith("application/json"):
            return JSONResponse(
                status_code=415,
                content=_error_payload(
                    "Content-Type must be application/json",
                    error_type="invalid_request_error",
                    code="unsupported_media_type",
                ),
            )
        try:
            payload = await request.json()
            requested_stream = payload.get("stream", False) if isinstance(payload, dict) else False
            payload = normalize_chat_request(
                payload,
                settings.served_model,
            )
        except (
            json.JSONDecodeError,
            UnicodeDecodeError,
            RequestValidationError,
            ValueError,
        ) as error:
            LOGGER.info(
                "request_id=%s rejected error_type=%s",
                request_id,
                type(error).__name__,
            )
            return JSONResponse(
                status_code=400,
                content=_error_payload(
                    str(error) or "invalid JSON request",
                    error_type="invalid_request_error",
                    code="invalid_request",
                ),
            )

        headers = _copy_request_headers(request.headers)
        headers.setdefault("x-request-id", request_id)
        LOGGER.info("request_id=%s accepted", request_id)
        try:
            upstream_request = app.state.client.build_request(
                "POST",
                "/v1/chat/completions",
                json=payload,
                headers=headers,
            )
            upstream_response = await _send_until_disconnect(
                app.state.client.send(upstream_request, stream=True),
                request.receive,
            )
        except httpx.HTTPError as error:
            LOGGER.warning(
                "request_id=%s upstream_unavailable error_type=%s",
                request_id,
                type(error).__name__,
            )
            return JSONResponse(
                status_code=503,
                content=_error_payload(
                    "PD proxy is unavailable",
                    error_type="server_error",
                    code="upstream_unavailable",
                ),
            )

        if upstream_response is None:
            LOGGER.info(
                "request_id=%s downstream_disconnected_before_headers elapsed_ms=%.1f",
                request_id,
                (time.monotonic() - started_at) * 1000.0,
            )
            return Response(status_code=499)

        response_headers = _copy_response_headers(upstream_response.headers)
        response_headers.setdefault("x-request-id", request_id)
        upstream_content_type = upstream_response.headers.get(
            "content-type",
            "application/octet-stream",
        )
        if upstream_response.status_code < 200 or upstream_response.status_code >= 300:
            body = await upstream_response.aread()
            await upstream_response.aclose()
            LOGGER.info(
                "request_id=%s upstream_status=%s elapsed_ms=%.1f",
                request_id,
                upstream_response.status_code,
                (time.monotonic() - started_at) * 1000.0,
            )
            return Response(
                content=body,
                status_code=upstream_response.status_code,
                media_type=upstream_content_type.split(";", 1)[0],
                headers=response_headers,
            )

        if "text/event-stream" in upstream_content_type.lower():
            if requested_stream:
                return StreamingResponse(
                    _translated_sse_stream(
                        upstream_response,
                        request_id,
                        started_at,
                    ),
                    status_code=upstream_response.status_code,
                    media_type="text/event-stream",
                    headers=response_headers,
                )
            try:
                document = await _aggregate_until_disconnect(
                    upstream_response,
                    request.receive,
                    request_id,
                    started_at,
                )
            except (httpx.HTTPError, OSError, UpstreamProtocolError) as error:
                LOGGER.warning(
                    "request_id=%s upstream_stream_invalid error_type=%s",
                    request_id,
                    type(error).__name__,
                )
                return JSONResponse(
                    status_code=502,
                    content=_error_payload(
                        "PD proxy returned an invalid streaming response",
                        error_type="server_error",
                        code="upstream_stream_invalid",
                    ),
                    headers=response_headers,
                )
            if document is None:
                LOGGER.info(
                    "request_id=%s downstream_disconnected_during_response elapsed_ms=%.1f",
                    request_id,
                    (time.monotonic() - started_at) * 1000.0,
                )
                return Response(status_code=499)
            return JSONResponse(
                content=document,
                status_code=upstream_response.status_code,
                headers=response_headers,
            )

        body = await upstream_response.aread()
        await upstream_response.aclose()
        LOGGER.info(
            "request_id=%s completed status=%s elapsed_ms=%.1f",
            request_id,
            upstream_response.status_code,
            (time.monotonic() - started_at) * 1000.0,
        )
        if "application/json" in upstream_content_type.lower():
            body = translate_json_response(body)
        return Response(
            content=body,
            status_code=upstream_response.status_code,
            media_type=upstream_content_type.split(";", 1)[0],
            headers=response_headers,
        )

    return app


# Asynchronous request logging

REQUEST_LOG = logging.getLogger("ark_adapter.request_log")


TZ = timezone(timedelta(hours=8))


MIB = 1024 * 1024


OWNER = "ark-request-log-v1"


def now():
    return datetime.now(TZ).isoformat(timespec="milliseconds")


def encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


class Record:
    def __init__(self, request_id, path, method):
        self.started = time.monotonic()
        self.key = uuid.uuid4().hex
        received_at = now()
        self.date = received_at[:10]
        slug = re.sub(r"[^A-Za-z0-9_.-]", "_", request_id)[:80].strip(".") or "request"
        self.dirname = slug + "__" + self.key
        self.pending = []
        self.version = 0
        self.written_version = -1
        self.captured_bytes = 0
        self.status = dict(
            owner=OWNER,
            request_id=request_id,
            attempt_id=self.key,
            path=path,
            method=method,
            state="received",
            received_at=received_at,
            first_output_at=None,
            last_output_at=None,
            finished_at=None,
            elapsed_ms=None,
            http_status=None,
            usage=None,
            finish_reasons={},
            output_events=0,
            logging_complete=True,
            logging_errors=[],
        )


class RequestLogWriter:
    def __init__(
        self,
        root,
        interval=0.2,
        max_request_bytes=32 * MIB,
        max_pending_bytes=64 * MIB,
        max_records=1024,
        retention_days=3,
        max_disk_bytes=5 * 1024 * MIB,
    ):
        self.root = Path(root)
        self.interval = interval
        self.max_request_bytes = max_request_bytes
        self.max_pending_bytes = max_pending_bytes
        self.max_records = max_records
        self.retention_days = retention_days
        self.max_disk_bytes = max_disk_bytes
        self.records = {}
        self.pending_bytes = 0
        self.disk_bytes = 0
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name="request-log-writer", daemon=True)
        self.thread.start()

    def start(self, request_id, path, method):
        with self.lock:
            if len(self.records) >= self.max_records:
                REQUEST_LOG.error("request logging skipped: too many active records")
                return None
            r = Record(request_id, path, method)
            self.records[r.key] = r
            return r

    def update(self, r, **fields):
        with self.lock:
            r.status.update(fields)
            r.version += 1

    def mark(self, r, reason):
        with self.lock:
            if reason not in r.status["logging_errors"]:
                r.status["logging_errors"].append(reason)
                REQUEST_LOG.warning("request_id=%s incomplete request log: %s", r.status["request_id"], reason)
            r.status["logging_complete"] = False
            r.version += 1

    def append(self, r, filename, value):
        data = encode(value) + b"\n"
        with self.lock:
            over = (
                r.captured_bytes + len(data) > self.max_request_bytes
                or self.pending_bytes + len(data) > self.max_pending_bytes
                or self.disk_bytes + self.pending_bytes + len(data) > self.max_disk_bytes
            )
            if not over:
                r.pending.append((filename, data))
                r.captured_bytes += len(data)
                self.pending_bytes += len(data)
                r.version += 1
        if over:
            self.mark(r, "capture_budget_exceeded")

    def finish(self, r, state):
        self.update(r, state=state, finished_at=now(), elapsed_ms=round((time.monotonic() - r.started) * 1000, 3))

    @staticmethod
    def _atomic(path, data):
        tmp = path.with_suffix(path.suffix + ".tmp")
        with os.fdopen(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "wb") as f:
            f.write(data)
        tmp.chmod(0o600)
        tmp.replace(path)

    def _flush(self):
        with self.lock:
            batch = []
            for r in self.records.values():
                if r.version == r.written_version:
                    continue
                pending, r.pending = r.pending, []
                batch.append((r, pending, encode(r.status), r.version, r.status["finished_at"]))
            # Detached buffers remain charged until writes finish, bounding both batches.
        for r, pending, status, version, finished in batch:
            path = self.root / r.date / r.dirname
            try:
                path.mkdir(parents=True, exist_ok=True, mode=0o700)
                path.chmod(0o700)
                for filename, data in pending:
                    if filename == "input.json":
                        self._atomic(path / filename, data)
                    else:
                        with os.fdopen(
                            os.open(path / filename, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600), "ab"
                        ) as f:
                            f.write(data)
                        (path / filename).chmod(0o600)
                    self.disk_bytes += len(data)
                self._atomic(path / "status.json", status)
            except Exception as exc:
                # Never spill prompt/response text or credentials into process logs.
                self.mark(r, "disk_write_failed:" + type(exc).__name__)
                REQUEST_LOG.error(
                    "request_id=%s request log write failed (%s)", r.status["request_id"], type(exc).__name__
                )
            with self.lock:
                self.pending_bytes -= sum(len(data) for _, data in pending)
                r.written_version = version
                if finished and not r.pending:
                    self.records.pop(r.key, None)

    def _cleanup(self):
        """Only delete owned, terminal records. Never touch active/unknown files."""
        candidates = []
        total = 0
        with self.lock:
            active = {r.dirname for r in self.records.values()}
        for date_dir in self.root.glob("????-??-??"):
            if date_dir.is_symlink() or not date_dir.is_dir():
                continue
            for directory in date_dir.iterdir():
                if directory.is_symlink() or not directory.is_dir():
                    continue
                size = sum(p.stat().st_size for p in directory.iterdir() if p.is_file() and not p.is_symlink())
                total += size
                try:
                    status = json.loads((directory / "status.json").read_text())
                    if directory.name not in active and status.get("owner") == OWNER and status.get("finished_at"):
                        finished = datetime.fromisoformat(status["finished_at"]).timestamp()
                        candidates.append((finished, directory, size))
                except (OSError, ValueError, TypeError):
                    continue
        for finished, directory, size in sorted(candidates):
            if time.time() - finished > self.retention_days * 86400 or total > self.max_disk_bytes:
                shutil.rmtree(directory)
                total -= size
        self.disk_bytes = total

    def _run(self):
        cleanup_at = 0
        while True:
            try:
                if time.monotonic() >= cleanup_at:
                    self._cleanup()
                    cleanup_at = time.monotonic() + 60
                self._flush()
            except Exception as exc:
                REQUEST_LOG.error("request log writer failed (%s)", type(exc).__name__)
            if self.stop.is_set():
                self._flush()
                return
            self.stop.wait(self.interval)

    def close(self):
        self.stop.set()
        self.thread.join(timeout=5)


class ResponseCapture:
    def __init__(self, writer, record, limit=4 * MIB):
        self.writer, self.record, self.limit = writer, record, limit
        self.decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self.buffer = ""
        self.sse = False
        self.done = False
        self.error = False
        self.disabled = False
        self.seq = 0

    def document(self, data, kind="chunk"):
        self.seq += 1
        stamp = now()
        r = self.record
        self.writer.append(r, "output.jsonl", dict(seq=self.seq, timestamp=stamp, type=kind, data=data))
        fields = dict(output_events=self.seq)
        if isinstance(data, dict):
            if data.get("error"):
                self.error = True
            if data.get("usage") is not None:
                fields["usage"] = data["usage"]
            meaningful = False
            reasons = dict(r.status["finish_reasons"])
            for choice in data.get("choices", []) or []:
                delta = choice.get("delta", choice.get("message", {})) or {}
                meaningful |= any(delta.get(k) for k in ("content", "reasoning_content", "tool_calls", "function_call"))
                if choice.get("finish_reason") is not None:
                    reasons[str(choice.get("index", 0))] = choice["finish_reason"]
            fields["finish_reasons"] = reasons
            if meaningful:
                fields["last_output_at"] = stamp
                if r.status["first_output_at"] is None:
                    fields["first_output_at"] = stamp
        self.writer.update(r, **fields)

    def feed(self, body, final=False):
        if self.disabled:
            return
        self.buffer += self.decoder.decode(body, final=final)
        if self.sse:
            # Accept CRLF, CR and LF without changing the forwarded response bytes.
            while True:
                match = re.search(r"\r\n\r\n|\n\n|\r\r", self.buffer)
                if not match:
                    break
                event, self.buffer = self.buffer[: match.start()], self.buffer[match.end() :]
                payload = "\n".join(line[5:].lstrip(" ") for line in event.splitlines() if line.startswith("data:"))
                if not payload:
                    continue
                if payload == "[DONE]":
                    self.done = True
                    self.document(None, "done")
                else:
                    try:
                        data = json.loads(payload)
                    except ValueError:
                        self.writer.mark(self.record, "invalid_sse_json")
                        data = {"raw_data": payload}
                    self.document(data)
            if final and self.buffer.strip():
                self.writer.mark(self.record, "unterminated_sse_event")
        elif final:
            try:
                self.document(json.loads(self.buffer), "response")
            except ValueError:
                self.document({"raw_body": self.buffer}, "response")
            self.buffer = ""
        if len(self.buffer.encode("utf-8")) > self.limit:
            self.writer.mark(self.record, "response_event_too_large")
            self.buffer = ""
            self.disabled = True


class RequestLoggingMiddleware:
    def __init__(self, app, writer, input_limit=16 * MIB):
        self.app, self.writer, self.input_limit = app, writer, input_limit

    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan":
            try:
                return await self.app(scope, receive, send)
            finally:
                await asyncio.to_thread(self.writer.close)
        if (
            scope["type"] != "http"
            or scope.get("method") != "POST"
            or scope.get("path") not in ("/inner/api/v3/chat/completions", "/inner/api/v3/responses/create")
        ):
            return await self.app(scope, receive, send)
        headers = list(scope.get("headers", []))
        supplied = next((v.decode("latin1") for k, v in headers if k.lower() == b"x-request-id"), "")
        request_id = "".join(c for c in supplied if c.isprintable()).strip()[:128] or "adapter-" + uuid.uuid4().hex
        scope = dict(
            scope,
            headers=[(k, v) for k, v in headers if k.lower() != b"x-request-id"]
            + [(b"x-request-id", request_id.encode("latin1"))],
        )
        r = self.writer.start(request_id, scope["path"], scope["method"])
        if r is None:
            return await self.app(scope, receive, send)
        capture = ResponseCapture(self.writer, r)
        input_body = bytearray()
        input_truncated = False
        disconnected = False
        response_ended = False
        status_code = 0

        def guard(fn, *args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                self.writer.mark(r, "capture_failed:" + type(exc).__name__)

        async def logged_receive():
            nonlocal disconnected, input_truncated
            message = await receive()
            if message["type"] == "http.disconnect":
                disconnected = True
            elif message["type"] == "http.request":
                body = message.get("body", b"")
                if not input_truncated:
                    if len(input_body) + len(body) > self.input_limit:
                        input_truncated = True
                        input_body.clear()
                        self.writer.mark(r, "input_too_large")
                    else:
                        input_body.extend(body)
                if not message.get("more_body", False):
                    if input_truncated:
                        document = {"truncated": True, "reason": "input_too_large"}
                    else:
                        try:
                            document = {"body": json.loads(input_body)}
                        except ValueError:
                            document = {"raw_body": input_body.decode("utf-8", "replace")}
                    guard(self.writer.append, r, "input.json", document)
                    input_body.clear()
            return message

        async def logged_send(message):
            nonlocal response_ended, status_code
            await send(message)
            if message["type"] == "http.response.start":
                status_code = message["status"]
                content_type = next(
                    (v.lower() for k, v in message.get("headers", []) if k.lower() == b"content-type"), b""
                )
                capture.sse = b"text/event-stream" in content_type
                self.writer.update(r, http_status=status_code, state="streaming" if capture.sse else "responding")
            elif message["type"] == "http.response.body":
                response_ended = not message.get("more_body", False)
                guard(capture.feed, message.get("body", b""), final=response_ended)

        state = "incomplete"
        try:
            await self.app(scope, logged_receive, logged_send)
            if status_code == 499 or (disconnected and not response_ended):
                state = "client_disconnected"
            elif status_code >= 400 or capture.error:
                state = "error"
            elif response_ended and (not capture.sse or capture.done):
                state = "completed"
        except asyncio.CancelledError:
            state = "client_disconnected" if disconnected else "cancelled"
            raise
        except OSError:
            state = "client_disconnected" if disconnected else "transport_error"
            raise
        except Exception:
            state = "error"
            raise
        finally:
            self.writer.finish(r, state)


# Motor lifecycle and MLOps discovery

LOG = logging.getLogger("acc")


def port_list(value):
    """Accept a single port, inclusive range or comma-separated ports."""
    ports = set()
    for item in value.split(","):
        bounds = item.strip().split("-")
        if len(bounds) not in (1, 2):
            raise ValueError("Ports must use N or N-M, separated by commas")
        first, last = int(bounds[0]), int(bounds[-1])
        if not 1 <= first <= last <= 65535:
            raise ValueError("Ports must be within 1..65535")
        ports.update(range(first, last + 1))
    return tuple(sorted(ports))


@dataclass(frozen=True)
class Settings:
    discovery_path: Path = Path("/etc/config/discovery/application-endpoints.yaml")
    application: str = ""
    prefill_workerset: str = "prefill"
    decode_workerset: str = "decode"
    prefill_role: str = "prefill"
    decode_role: str = "decode"
    prefill_ports: tuple = (7100,)
    decode_ports: tuple = tuple(range(7100, 7116))
    interval: float = 5.0
    health_timeout: float = 2.0
    adapter_host: str = "0.0.0.0"
    adapter_port: int = 8005
    motor_upstream: str = "http://127.0.0.1:1025"
    motor_management: str = "http://127.0.0.1:1026"
    served_model: str = "dsv4"
    log_dir: Path = Path("/app/pure_vllm/logs")
    request_log_retention_days: float = 3.0
    request_log_max_gib: float = 5.0
    shutdown_timeout: float = 30.0

    def validate(self):
        if not all((self.prefill_workerset, self.decode_workerset, self.prefill_role, self.decode_role)):
            raise ValueError("WorkerSet and backend Role names must be nonempty")
        if (self.prefill_workerset, self.prefill_role) == (self.decode_workerset, self.decode_role):
            raise ValueError("Prefill and Decode WorkerSet/Role pairs must be distinct")
        if not 1 <= self.adapter_port <= 65535:
            raise ValueError("Listener port must be within 1..65535")
        for value in (
            self.interval,
            self.health_timeout,
            self.shutdown_timeout,
            self.request_log_retention_days,
            self.request_log_max_gib,
        ):
            if not 0 < value < float("inf"):
                raise ValueError("Intervals, timeouts and log limits must be finite and positive")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = Settings()
    for field in defaults.__dataclass_fields__:
        default = getattr(defaults, field)
        convert = port_list if field.endswith("_ports") else type(default)
        env = "APPLICATION_NAME" if field == "application" else "ACC_" + field.upper()
        value = os.environ.get(env, default)
        parser.add_argument(
            "--" + field.replace("_", "-"), type=convert, default=value, help=f"Environment: {env}; default: {default}"
        )
    settings = Settings(**vars(parser.parse_args(argv)))
    settings.validate()
    return settings


def read_endpoints(settings):
    document = yaml.safe_load(settings.discovery_path.read_text())
    if not isinstance(document, dict):
        raise ValueError("Discovery must contain an application mapping")
    application = settings.application
    if not application:
        if len(document) != 1:
            raise ValueError("APPLICATION_NAME is required for multi-application discovery")
        application = next(iter(document))
    app = document.get(application, {})
    if not isinstance(app, dict):
        raise ValueError("Application discovery must be a WorkerSet mapping")
    result = {}
    for role, workerset, name, ports in (
        ("prefill", settings.prefill_workerset, settings.prefill_role, settings.prefill_ports),
        ("decode", settings.decode_workerset, settings.decode_role, settings.decode_ports),
    ):
        # A removed WorkerSet/Role is an empty pool; a malformed value is an error.
        roles = app.get(workerset, {})
        if not isinstance(roles, dict):
            raise ValueError(f"WorkerSet {workerset} must contain a Role mapping")
        pods = roles.get(name, [])
        if not isinstance(pods, list):
            raise ValueError(f"Role {name} must contain a list")
        endpoints = set()
        for pod in pods:
            if not isinstance(pod, dict):
                raise ValueError(f"Invalid Pod entry in Role {name}")
            ip = pod.get("ip")
            if not ip:  # Pending Pods may not have an IP yet.
                continue
            if not isinstance(ip, str):
                raise ValueError(f"Invalid Pod IP in Role {name}")
            endpoints.update((ip, port) for port in ports)
        result[role] = endpoints
    return result


def endpoint_url(ip, port):
    host = f"[{ip}]" if ":" in ip else ip
    return f"http://{host}:{port}"


def motor_instance(settings, role, ip, ports):
    # Stable across discovery reordering and ACC restarts. Each Pod is one Motor
    # instance; the endpoint id is its local API index, not the global EP rank.
    identity = f"{settings.application}:{role}:{ip}"
    instance_id = uuid.uuid5(uuid.NAMESPACE_URL, identity).int & ((1 << 63) - 1)
    configured_ports = settings.prefill_ports if role == "prefill" else settings.decode_ports
    return {
        "id": instance_id,
        "role": role,
        "dispatch_capabilities": ["prefill_handoff_decode"],
        "engine_type": "vllm",
        "job_name": f"{role}-{ip}",
        "model_name": settings.served_model,
        "endpoints": {
            ip: {
                str(index): {"id": index, "ip": ip, "business_port": str(port), "mgmt_port": str(port)}
                for index, port in enumerate(configured_ports)
                if port in ports
            }
        },
    }


async def healthy(client, endpoint):
    try:
        response = await client.get(endpoint_url(*endpoint) + "/health")
        return response.status_code == 200
    except httpx.RequestError:
        return False


async def refresh_instances(client, settings, event, instances):
    response = await client.post(
        settings.motor_management + "/instances/refresh",
        json={"event": event, "instances": instances},
        timeout=30,
    )
    response.raise_for_status()
    if response.json()["status"] != "success":
        raise RuntimeError(f"Motor rejected instance {event}: {response.text}")
    LOG.info("Motor %s: %s", event, [instance["job_name"] for instance in instances])


async def sync_once(registered, client, settings):
    desired = {}
    for role, endpoints in read_endpoints(settings).items():
        endpoints = sorted(endpoints)
        checks = await asyncio.gather(*(healthy(client, endpoint) for endpoint in endpoints))
        pods = {}
        for (ip, port), ok in zip(endpoints, checks):
            if ok:
                pods.setdefault(ip, set()).add(port)
        for ip, ports in pods.items():
            instance = motor_instance(settings, role, ip, ports)
            desired[instance["id"]] = instance

    # Remove stale or changed registrations before adding their replacements.
    # Update our snapshot only after Motor acknowledges each operation.
    removed = [value for key, value in registered.items() if desired.get(key) != value]
    if removed:
        await refresh_instances(client, settings, "del", removed)
        for instance in removed:
            registered.pop(instance["id"])
    added = [value for key, value in desired.items() if registered.get(key) != value]
    if added:
        await refresh_instances(client, settings, "add", added)
        registered.update((instance["id"], instance) for instance in added)


async def watch(registered, client, settings):
    while True:
        try:
            await sync_once(registered, client, settings)
        except (OSError, ValueError, RuntimeError, httpx.HTTPError, yaml.YAMLError) as error:
            LOG.warning("Discovery/Motor sync pending: %s", error)
        await asyncio.sleep(settings.interval)


async def motor_readiness(registered, client, settings):
    counts = {
        role: sum(
            len(endpoints)
            for instance in registered.values()
            if instance["role"] == role
            for endpoints in instance["endpoints"].values()
        )
        for role in ("prefill", "decode")
    }
    ready = False
    try:
        response = await client.get(settings.motor_management + "/readiness")
        ready = response.status_code == 200 and response.json().get("ready") is True
    except (httpx.HTTPError, ValueError):
        pass
    ready = ready and bool(counts["prefill"] and counts["decode"])
    return JSONResponse(
        {
            "status": "ok" if ready else "not_ready",
            "prefill_endpoints": counts["prefill"],
            "decode_endpoints": counts["decode"],
        },
        status_code=200 if ready else 503,
    )


class ManagedServer(uvicorn.Server):
    async def serve(self, sockets=None):
        try:
            await super().serve(sockets=sockets)
        except SystemExit as error:
            # Keep bind failures inside the supervisor so Motor is shut down.
            raise RuntimeError("ACC listener failed to start") from error

    @contextmanager
    def capture_signals(self):
        # ACC forwards termination to the Motor process group as well.
        yield


async def stop_motor(process, timeout):
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        await asyncio.wait_for(process.wait(), timeout=timeout)
    except TimeoutError:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        await process.wait()


async def main(settings):
    settings.validate()
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            RotatingFileHandler(settings.log_dir / "acc.log", maxBytes=20 * 1024**2, backupCount=3),
        ],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    registered = {}
    async with httpx.AsyncClient(timeout=settings.health_timeout, trust_env=False) as client:

        async def readiness():
            return await motor_readiness(registered, client, settings)

        ark = create_app(
            AdapterSettings(upstream=settings.motor_upstream, served_model=settings.served_model),
            readiness_check=readiness,
        )
        ark = RequestLoggingMiddleware(
            ark,
            RequestLogWriter(
                settings.log_dir / "requests",
                retention_days=settings.request_log_retention_days,
                max_disk_bytes=int(settings.request_log_max_gib * 1024**3),
            ),
        )
        server = ManagedServer(
            uvicorn.Config(
                ark,
                host=settings.adapter_host,
                port=settings.adapter_port,
                access_log=False,
                timeout_graceful_shutdown=settings.shutdown_timeout,
            )
        )
        loop = asyncio.get_running_loop()
        stopped = asyncio.Event()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stopped.set)
        motor_env = dict(os.environ)
        motor_env.setdefault("USER_CONFIG_PATH", str(Path(__file__).with_name("motor_config.json")))
        motor = await asyncio.create_subprocess_exec(
            sys.executable,
            "-u",
            "-m",
            "motor.coordinator.main",
            env=motor_env,
            start_new_session=True,
        )
        LOG.info(
            "Motor PID=%s upstream=%s management=%s", motor.pid, settings.motor_upstream, settings.motor_management
        )
        serving = asyncio.create_task(server.serve())
        watching = asyncio.create_task(watch(registered, client, settings))
        motor_exit = asyncio.create_task(motor.wait())
        stopping = asyncio.create_task(stopped.wait())
        try:
            done, _ = await asyncio.wait([serving, watching, motor_exit, stopping], return_when=asyncio.FIRST_COMPLETED)
            if stopping not in done:
                for task in done:
                    task.result()
                raise RuntimeError(f"ACC or Motor stopped unexpectedly (Motor exit={motor.returncode})")
        finally:
            watching.cancel()
            stopping.cancel()
            server.should_exit = True
            # Stop the HTTP listener before Motor so current requests can finish.
            try:
                await asyncio.wait_for(serving, timeout=settings.shutdown_timeout + 5)
            finally:
                await stop_motor(motor, settings.shutdown_timeout)
                await asyncio.gather(watching, motor_exit, stopping, return_exceptions=True)
                for sig in (signal.SIGINT, signal.SIGTERM):
                    loop.remove_signal_handler(sig)


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
