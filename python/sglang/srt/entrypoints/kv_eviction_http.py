"""HTTP glue for explicit KV eviction, shared by /generate and chat completions."""

from __future__ import annotations

from http import HTTPStatus
from typing import Any, Awaitable, Callable, Dict, Optional

from fastapi.responses import ORJSONResponse

from sglang.srt.managers.kv_eviction import KvEvictionError, parse_kv_eviction_request
from sglang.srt.managers.kv_eviction_registry import CallResult


def validate_http_kv_eviction(
    kv_eviction: Any,
    session_params: Any,
    *,
    stream: bool,
    n: int = 1,
    has_exact_input_ids: bool,
) -> None:
    """Request-shape rules that do not need scheduler state. Raises KvEvictionError."""
    spec = parse_kv_eviction_request(kv_eviction)
    if stream:
        raise KvEvictionError("kv_eviction requires a non-streaming request")
    if n != 1:
        raise KvEvictionError("kv_eviction requires n=1")
    if not has_exact_input_ids:
        raise KvEvictionError(
            "kv_eviction requires exact pre-tokenized input (input_ids)"
        )
    if not isinstance(session_params, dict) or session_params.get("id") != spec.cache_id:
        raise KvEvictionError(
            "kv_eviction requires session_params.id to equal kv_eviction.cache_id"
        )
    for key in ("offset", "replace", "drop_previous_output"):
        if session_params.get(key):
            raise KvEvictionError(f"kv_eviction does not support session_params.{key}")
    if session_params.get("rid") is not None:
        raise KvEvictionError("kv_eviction does not support session_params.rid")


def abort_result(ret: Dict[str, Any]) -> Optional[CallResult]:
    """Map a scheduler abort carried in a normal output to an HTTP error."""
    finish_reason = (ret.get("meta_info") or {}).get("finish_reason") or {}
    if finish_reason.get("type") != "abort":
        return None
    status = int(finish_reason.get("status_code") or HTTPStatus.INTERNAL_SERVER_ERROR)
    return status, {
        "error": {
            "message": finish_reason.get("message", "request aborted"),
            "type": finish_reason.get("err_type") or "AbortError",
            "code": status,
        }
    }


def public_kv_eviction_fields(meta_info: Dict[str, Any]) -> Dict[str, Any]:
    """Split terminal kv_eviction metadata into public response fields."""
    info = meta_info.get("kv_eviction")
    if info is None:
        return {}
    info = dict(info)
    return {
        "prompt_token_ids": info.pop("prompt_token_ids"),
        "compaction_events": info.pop("compaction_events"),
        "compaction_replay_mode": info.pop("compaction_replay_mode"),
        "kv_eviction": info,
    }


def error_result(e: Exception) -> CallResult:
    status = int(getattr(e, "status_code", HTTPStatus.BAD_REQUEST))
    return status, {"error": {"message": str(e), "code": status}}


async def run_kv_eviction_call(
    tokenizer_manager,
    *,
    kv_eviction: Dict[str, Any],
    body: Any,
    producer: Callable[[int], Awaitable[CallResult]],
) -> ORJSONResponse:
    """Execute one kv_eviction call exactly once and return its HTTP response."""
    try:
        status, content = await tokenizer_manager.kv_eviction_registry.run(
            cache_id=kv_eviction["cache_id"],
            call_id=kv_eviction["call_id"],
            body=body,
            producer=producer,
        )
    except KvEvictionError as e:
        status, content = error_result(e)
    return ORJSONResponse(content=content, status_code=status)
