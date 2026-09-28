"""Exactly-once execution of explicit KV-eviction calls at the HTTP layer.

A kv_eviction call mutates session KV, so a client retry must never execute
it twice. The scheduler's state chain already refuses a replayed call (its
``expected_state_id`` is stale after the first execution); this registry makes
retries *useful*: an identical retry joins the in-flight call or gets the
cached response, a changed body under the same ``call_id`` is a conflict, and
the producer is shielded so an HTTP disconnect cannot strand a half-applied
mutation.

The registry lives in the tokenizer-manager process, which is why
``--enable-kv-eviction`` requires ``--tokenizer-worker-num 1``.
"""

from __future__ import annotations

import asyncio
import dataclasses
from http import HTTPStatus
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple

from sglang.srt.managers.kv_eviction import KvEvictionError, canonical_body_digest

# (status_code, json-serializable body)
CallResult = Tuple[int, Any]


@dataclasses.dataclass
class _ActiveCall:
    call_id: str
    digest: str
    task: "asyncio.Task[CallResult]"


@dataclasses.dataclass
class _CacheEntry:
    generation: int
    active: Optional[_ActiveCall] = None
    last_call_id: Optional[str] = None
    last_digest: Optional[str] = None
    last_result: Optional[CallResult] = None


class KvEvictionCallRegistry:
    def __init__(self) -> None:
        self._entries: Dict[str, _CacheEntry] = {}
        self._generations: Dict[str, int] = {}

    def generation(self, cache_id: str) -> int:
        return self._generations.get(cache_id, 0)

    def invalidate(self, cache_id: str) -> None:
        """Start a new session incarnation (on open/close of ``cache_id``).

        In-flight producers keep running to a terminal result, but their
        result is no longer cached and their dispatch is refused if they have
        not reached the scheduler yet.
        """
        self._generations[cache_id] = self.generation(cache_id) + 1
        self._entries.pop(cache_id, None)

    def check_dispatch(self, cache_id: str, generation: Optional[int]) -> None:
        if generation is not None and generation != self.generation(cache_id):
            raise KvEvictionError(
                f"kv_eviction session {cache_id} was reopened or closed while "
                "this call was pending",
                HTTPStatus.CONFLICT,
            )

    async def run(
        self,
        *,
        cache_id: str,
        call_id: str,
        body: Any,
        producer: Callable[[int], Awaitable[CallResult]],
    ) -> CallResult:
        digest = canonical_body_digest(body)
        generation = self.generation(cache_id)
        entry = self._entries.get(cache_id)
        if entry is None or entry.generation != generation:
            entry = _CacheEntry(generation=generation)
            self._entries[cache_id] = entry

        active = entry.active
        if active is not None:
            if active.call_id == call_id and active.digest == digest:
                return await asyncio.shield(active.task)
            raise KvEvictionError(
                f"kv_eviction session {cache_id} already has an active call "
                f"{active.call_id}",
                HTTPStatus.CONFLICT,
            )
        if entry.last_call_id == call_id:
            if entry.last_digest == digest:
                return entry.last_result
            raise KvEvictionError(
                f"call_id {call_id} was already used with a different request body",
                HTTPStatus.CONFLICT,
            )

        task = asyncio.ensure_future(producer(generation))
        entry.active = _ActiveCall(call_id=call_id, digest=digest, task=task)

        def _on_done(t: "asyncio.Task[CallResult]") -> None:
            if entry.active is not None and entry.active.task is t:
                entry.active = None
            if t.cancelled() or t.exception() is not None:
                return
            status, _ = t.result()
            current = self._entries.get(cache_id)
            if status == HTTPStatus.OK and current is entry:
                entry.last_call_id = call_id
                entry.last_digest = digest
                entry.last_result = t.result()

        task.add_done_callback(_on_done)
        return await asyncio.shield(task)
