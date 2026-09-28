"""Client-driven explicit KV eviction for streaming sessions.

The client owns the policy (which token spans to forget); this module owns the
mechanism: strict request validation, the per-session state chain, the
logical-position map, and the in-place splice of a session's KV row.

Coordinates. Every call addresses the session's *resident* sequence: the
previous call's post-eviction prompt followed by its generated tokens. Spans
are half-open ``[start, end)`` indices into that sequence, sorted and
non-overlapping. Surviving KV is never recomputed and keeps the logical (RoPE)
position it was computed at; tokens computed after an eviction continue from
the largest logical position so far, so a single per-request offset
(``position_offset``) maps every newly computed physical position to its
logical one.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from http import HTTPStatus
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import torch

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ReqKvInfo
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache

KV_EVICTION_VERSION = 1
KV_EVICTION_REPLAY_MODE = "prefill_trim"

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_REQUEST_FIELDS = frozenset(
    {
        "version",
        "cache_id",
        "call_id",
        "call_index",
        "expected_state_id",
        "protected_prefix_len",
        "evict_spans",
        "evict_group_ids",
    }
)
_STATE_DOMAIN = "sglang.kv_eviction.state.v1"
_EVENT_DOMAIN = "sglang.kv_eviction.event.v1"


class KvEvictionError(ValueError):
    """A rejected explicit-eviction call. ``status_code`` is the HTTP status."""

    def __init__(self, message: str, status_code: int = HTTPStatus.BAD_REQUEST):
        super().__init__(message)
        self.status_code = int(status_code)


def _conflict(message: str) -> KvEvictionError:
    return KvEvictionError(message, HTTPStatus.CONFLICT)


def _is_int(value: Any) -> bool:
    return type(value) is int


def _canonical_hash(domain: str, payload: Dict[str, Any]) -> str:
    body = json.dumps(
        {"domain": domain, **payload}, sort_keys=True, separators=(",", ":")
    )
    return "sha256:" + hashlib.sha256(body.encode()).hexdigest()


def canonical_body_digest(body: Any) -> str:
    """Digest of a request body, used to detect a call_id reused with changes."""
    return _canonical_hash(
        "sglang.kv_eviction.request_body.v1", {"body": body}
    )


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class KvEvictionRequest:
    version: int
    cache_id: str
    call_id: str
    call_index: int
    expected_state_id: Optional[str]
    protected_prefix_len: int
    evict_spans: Tuple[Tuple[int, int], ...]
    evict_group_ids: Tuple[int, ...]

    @property
    def tokens_evicted(self) -> int:
        return sum(end - start for start, end in self.evict_spans)


def _parse_span(raw: Any, index: int) -> Tuple[int, int]:
    if isinstance(raw, dict):
        if set(raw) != {"start", "end"}:
            raise KvEvictionError(
                f"evict_spans[{index}] must have exactly the keys start and end"
            )
        start, end = raw["start"], raw["end"]
    elif isinstance(raw, (list, tuple)) and len(raw) == 2:
        start, end = raw
    else:
        raise KvEvictionError(
            f"evict_spans[{index}] must be [start, end] or {{start, end}}"
        )
    if not (_is_int(start) and _is_int(end)):
        raise KvEvictionError(f"evict_spans[{index}] bounds must be integers")
    if start < 0 or end <= start:
        raise KvEvictionError(
            f"evict_spans[{index}]=[{start}, {end}) must satisfy 0 <= start < end"
        )
    return start, end


def parse_kv_eviction_request(raw: Any) -> KvEvictionRequest:
    """Strictly validate the public ``kv_eviction`` request object."""
    if isinstance(raw, KvEvictionRequest):
        return raw
    if not isinstance(raw, dict):
        raise KvEvictionError("kv_eviction must be an object")
    unknown = set(raw) - _REQUEST_FIELDS
    if unknown:
        raise KvEvictionError(f"kv_eviction has unknown fields: {sorted(unknown)}")
    missing = {
        "version",
        "cache_id",
        "call_id",
        "call_index",
        "protected_prefix_len",
    } - set(raw)
    if missing:
        raise KvEvictionError(f"kv_eviction is missing fields: {sorted(missing)}")

    version = raw["version"]
    if not _is_int(version) or version != KV_EVICTION_VERSION:
        raise KvEvictionError(f"kv_eviction.version must be {KV_EVICTION_VERSION}")
    for key in ("cache_id", "call_id"):
        value = raw[key]
        if not isinstance(value, str) or not _ID_RE.match(value):
            raise KvEvictionError(
                f"kv_eviction.{key} must match {_ID_RE.pattern}, got {value!r}"
            )
    call_index = raw["call_index"]
    if not _is_int(call_index) or call_index < 0:
        raise KvEvictionError("kv_eviction.call_index must be a non-negative integer")
    protected = raw["protected_prefix_len"]
    if not _is_int(protected) or protected < 0:
        raise KvEvictionError(
            "kv_eviction.protected_prefix_len must be a non-negative integer"
        )

    expected = raw.get("expected_state_id")
    if call_index == 0:
        if expected is not None:
            raise KvEvictionError("call_index=0 requires expected_state_id=null")
    elif not isinstance(expected, str) or not _HASH_RE.match(expected):
        raise KvEvictionError(
            "call_index>0 requires expected_state_id to be a lowercase sha256:<hex>"
        )

    raw_spans = raw.get("evict_spans", [])
    if raw_spans is None:
        raw_spans = []
    if not isinstance(raw_spans, (list, tuple)):
        raise KvEvictionError("kv_eviction.evict_spans must be a list")
    spans = tuple(_parse_span(s, i) for i, s in enumerate(raw_spans))
    for i in range(1, len(spans)):
        if spans[i][0] < spans[i - 1][1]:
            raise KvEvictionError(
                "kv_eviction.evict_spans must be sorted and non-overlapping; "
                f"span {i} starts at {spans[i][0]} before span {i - 1} ends at "
                f"{spans[i - 1][1]}"
            )
    if spans and spans[0][0] < protected:
        raise KvEvictionError(
            f"evict span [{spans[0][0]}, {spans[0][1]}) overlaps the protected "
            f"prefix of length {protected}"
        )
    if call_index == 0 and spans:
        raise KvEvictionError("call_index=0 cannot evict")

    raw_groups = raw.get("evict_group_ids", [])
    if raw_groups is None:
        raw_groups = []
    if not isinstance(raw_groups, (list, tuple)) or not all(
        _is_int(g) and g >= 0 for g in raw_groups
    ):
        raise KvEvictionError(
            "kv_eviction.evict_group_ids must be a list of non-negative integers"
        )
    groups = tuple(raw_groups)
    if any(b <= a for a, b in zip(groups, groups[1:])):
        raise KvEvictionError("kv_eviction.evict_group_ids must be sorted and unique")
    if groups and not spans:
        raise KvEvictionError("evict_group_ids given without any evict_spans")

    return KvEvictionRequest(
        version=version,
        cache_id=raw["cache_id"],
        call_id=raw["call_id"],
        call_index=call_index,
        expected_state_id=expected,
        protected_prefix_len=protected,
        evict_spans=spans,
        evict_group_ids=groups,
    )


# ---------------------------------------------------------------------------
# Logical-position map
# ---------------------------------------------------------------------------
# A map is a list of runs ``[phys_start, logical_start, length]`` covering the
# physical sequence [0, n) contiguously; within a run logical = physical + delta.


PositionMap = List[List[int]]


def position_map_len(runs: PositionMap) -> int:
    return runs[-1][0] + runs[-1][2] if runs else 0


def extend_position_map(runs: PositionMap, num_tokens: int, offset: int) -> PositionMap:
    """Append ``num_tokens`` physical positions at logical = physical + offset."""
    if num_tokens <= 0:
        return [list(r) for r in runs]
    out = [list(r) for r in runs]
    phys = position_map_len(out)
    if out and out[-1][1] - out[-1][0] == offset:
        out[-1][2] += num_tokens
    else:
        out.append([phys, phys + offset, num_tokens])
    return out


def evict_from_position_map(
    runs: PositionMap, spans: Sequence[Tuple[int, int]]
) -> PositionMap:
    """Drop physical spans and renumber survivors' physical coordinates."""
    out: PositionMap = []
    removed_before = 0
    span_idx = 0
    for phys_start, logical_start, length in runs:
        cursor = phys_start
        run_end = phys_start + length
        while cursor < run_end:
            while span_idx < len(spans) and spans[span_idx][1] <= cursor:
                removed_before += spans[span_idx][1] - spans[span_idx][0]
                span_idx += 1
            if span_idx < len(spans) and spans[span_idx][0] <= cursor:
                # Inside a span: skip to its end (or the run end).
                cursor = min(spans[span_idx][1], run_end)
                continue
            piece_end = run_end
            if span_idx < len(spans):
                piece_end = min(piece_end, spans[span_idx][0])
            new_phys = cursor - removed_before
            new_logical = logical_start + (cursor - phys_start)
            piece_len = piece_end - cursor
            if out and out[-1][0] + out[-1][2] == new_phys and (
                out[-1][1] - out[-1][0] == new_logical - new_phys
            ):
                out[-1][2] += piece_len
            else:
                out.append([new_phys, new_logical, piece_len])
            cursor = piece_end
    return out


def logical_positions(runs: PositionMap) -> List[int]:
    return [
        logical_start + i
        for _, logical_start, length in runs
        for i in range(length)
    ]


# ---------------------------------------------------------------------------
# State chain
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class KvEvictionState:
    cache_id: str
    call_index: int
    parent_state_id: Optional[str]
    protected_prefix_len: int
    position_offset: int
    resident_tokens: int
    physical_tokens: int
    position_map: PositionMap
    last_event_id: Optional[str]
    state_id: str = ""

    def _hash_payload(self) -> Dict[str, Any]:
        return {
            "version": KV_EVICTION_VERSION,
            "cache_id": self.cache_id,
            "call_index": self.call_index,
            "parent_state_id": self.parent_state_id,
            "protected_prefix_len": self.protected_prefix_len,
            "position_offset": self.position_offset,
            "resident_tokens": self.resident_tokens,
            "physical_tokens": self.physical_tokens,
            "position_map": self.position_map,
            "last_event_id": self.last_event_id,
        }

    def seal(self) -> "KvEvictionState":
        self.state_id = _canonical_hash(_STATE_DOMAIN, self._hash_payload())
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {**self._hash_payload(), "state_id": self.state_id}


def compute_event_id(
    *,
    cache_id: str,
    call_id: str,
    call_index: int,
    parent_event_id: Optional[str],
    evicted_group_ids: Sequence[int],
    evicted_spans: Sequence[Tuple[int, int]],
    tokens_evicted: int,
    position_offset_after: int,
) -> str:
    return _canonical_hash(
        _EVENT_DOMAIN,
        {
            "version": KV_EVICTION_VERSION,
            "cache_id": cache_id,
            "call_id": call_id,
            "call_index": call_index,
            "parent_event_id": parent_event_id,
            "evicted_group_ids": list(evicted_group_ids),
            "evicted_spans": [list(s) for s in evicted_spans],
            "tokens_evicted": tokens_evicted,
            "position_offset_after": position_offset_after,
        },
    )


@dataclasses.dataclass
class KvEvictionSessionState:
    """Per-session explicit-eviction bookkeeping owned by the scheduler."""

    state: Optional[KvEvictionState] = None
    # Set when the session's KV no longer matches ``state`` (e.g. a call was
    # aborted after its eviction was applied). Every later call is refused.
    invalid_reason: Optional[str] = None
    # The call currently executing on this session, if any.
    pending: Optional["KvEvictionPlan"] = None


@dataclasses.dataclass
class KvEvictionPlan:
    """An admitted call: the validated request plus what admission decided."""

    spec: KvEvictionRequest
    parent_state: Optional[KvEvictionState]
    # Pre-eviction resident length / materialized-KV length.
    resident_tokens_before: int
    physical_tokens_before: int
    new_input_len: int
    # Post-eviction prompt (survivors + appended input) and bookkeeping.
    position_offset: int = 0
    position_map_after_eviction: PositionMap = dataclasses.field(default_factory=list)
    compaction_event: Optional[Dict[str, Any]] = None
    event: Optional[Dict[str, Any]] = None
    # Evidence from the physical splice.
    applied: bool = False
    slots_freed: int = 0
    retained_slots_unchanged: Optional[bool] = None
    # prefix length observed when the session row was restored for prefill.
    matched_prefix_len: Optional[int] = None

    @property
    def expected_prefix_len(self) -> int:
        """Materialized KV that must be reused (not recomputed) after eviction."""
        return self.physical_tokens_before - self.spec.tokens_evicted


def validate_call(
    session_state: KvEvictionSessionState,
    spec: KvEvictionRequest,
    *,
    session_has_history: bool,
    resident_tokens: int,
    physical_tokens: int,
    new_input_len: int,
    context_len: int,
    max_new_tokens: Optional[int],
) -> None:
    """Check one call against the session's authoritative state."""
    if session_state.invalid_reason is not None:
        raise _conflict(
            f"kv_eviction session {spec.cache_id} is invalidated: "
            f"{session_state.invalid_reason}. Close and reopen the session."
        )
    if session_state.pending is not None:
        raise _conflict(
            f"kv_eviction session {spec.cache_id} already has an active call "
            f"{session_state.pending.spec.call_id}"
        )
    state = session_state.state
    if spec.call_index == 0:
        if state is not None or session_has_history:
            raise _conflict(
                "call_index=0 requires an empty, freshly opened session; "
                f"session {spec.cache_id} already has history"
            )
    else:
        if state is None:
            raise _conflict(
                f"call_index={spec.call_index} but session {spec.cache_id} has "
                "no committed kv_eviction state (expected call_index=0)"
            )
        if spec.call_index != state.call_index + 1:
            raise _conflict(
                f"call_index={spec.call_index} does not follow committed "
                f"call_index={state.call_index}"
            )
        if spec.expected_state_id != state.state_id:
            raise _conflict(
                f"expected_state_id {spec.expected_state_id} does not match the "
                f"resident state {state.state_id}"
            )
        if spec.protected_prefix_len != state.protected_prefix_len:
            raise KvEvictionError(
                "protected_prefix_len cannot change within a session "
                f"({spec.protected_prefix_len} != {state.protected_prefix_len})"
            )
        if resident_tokens != state.resident_tokens:
            raise _conflict(
                f"assembled resident length {resident_tokens} does not match "
                f"state resident_tokens={state.resident_tokens}"
            )
        if physical_tokens != state.physical_tokens:
            raise _conflict(
                f"session KV holds {physical_tokens} materialized tokens but the "
                f"state records {state.physical_tokens}"
            )

    if new_input_len <= 0 and resident_tokens == physical_tokens:
        raise KvEvictionError("a kv_eviction call must append at least one token")
    for start, end in spec.evict_spans:
        if end > physical_tokens:
            raise KvEvictionError(
                f"evict span [{start}, {end}) extends past the {physical_tokens} "
                "tokens with materialized KV"
            )

    offset_before = state.position_offset if state is not None else 0
    offset_after = offset_before + spec.tokens_evicted
    prompt_len_after = resident_tokens - spec.tokens_evicted + new_input_len
    logical_end = offset_after + prompt_len_after + (max_new_tokens or 0)
    if offset_after + prompt_len_after >= context_len or (
        max_new_tokens is not None and logical_end > context_len
    ):
        raise KvEvictionError(
            f"logical length {logical_end} (position_offset={offset_after}, "
            f"prompt={prompt_len_after}, max_new_tokens={max_new_tokens}) exceeds "
            f"the model context length {context_len}"
        )


def remove_spans(tokens: Sequence[int], spans: Sequence[Tuple[int, int]]) -> List[int]:
    out: List[int] = []
    cursor = 0
    for start, end in spans:
        out.extend(tokens[cursor:start])
        cursor = end
    out.extend(tokens[cursor:])
    return out


def kept_indices(total_len: int, spans: Sequence[Tuple[int, int]]) -> List[int]:
    out: List[int] = []
    cursor = 0
    for start, end in spans:
        out.extend(range(cursor, start))
        cursor = end
    out.extend(range(cursor, total_len))
    return out


def build_plan(
    session_state: KvEvictionSessionState,
    spec: KvEvictionRequest,
    *,
    assembled_prompt: Sequence[int],
    resident_tokens: int,
    physical_tokens: int,
) -> KvEvictionPlan:
    """Plan a validated call: post-eviction prompt, positions and events."""
    parent = session_state.state
    new_input_len = len(assembled_prompt) - resident_tokens
    plan = KvEvictionPlan(
        spec=spec,
        parent_state=parent,
        resident_tokens_before=resident_tokens,
        physical_tokens_before=physical_tokens,
        new_input_len=new_input_len,
    )
    offset_before = parent.position_offset if parent is not None else 0
    runs_before = (
        parent.position_map if parent is not None else []
    )
    assert position_map_len(runs_before) == resident_tokens
    plan.position_offset = offset_before + spec.tokens_evicted
    plan.position_map_after_eviction = evict_from_position_map(
        runs_before, spec.evict_spans
    )
    if spec.evict_spans:
        parent_event_id = parent.last_event_id if parent is not None else None
        event_id = compute_event_id(
            cache_id=spec.cache_id,
            call_id=spec.call_id,
            call_index=spec.call_index,
            parent_event_id=parent_event_id,
            evicted_group_ids=spec.evict_group_ids,
            evicted_spans=spec.evict_spans,
            tokens_evicted=spec.tokens_evicted,
            position_offset_after=plan.position_offset,
        )
        plan.event = {
            "version": KV_EVICTION_VERSION,
            "event_id": event_id,
            "parent_event_id": parent_event_id,
            "evicted_group_ids": list(spec.evict_group_ids),
            "evicted_spans": [list(s) for s in spec.evict_spans],
            "tokens_evicted": spec.tokens_evicted,
            "position_offset_after": plan.position_offset,
        }
        # vLLM-compatible single coalesced admission event. Indices address the
        # assembled pre-eviction prompt (resident + appended input).
        kept = kept_indices(len(assembled_prompt), spec.evict_spans)
        plan.compaction_event = {
            "num_output_tokens_at_compaction": 0,
            "tokens_evicted": spec.tokens_evicted,
            "position_offset_after": plan.position_offset,
            "num_prompt_tokens": len(kept),
            "evict_start": spec.evict_spans[0][0],
            "new_user_fragment_len": new_input_len,
            "kept_indices": kept,
            "kept_token_ids": [int(assembled_prompt[i]) for i in kept],
            "event_kind": 0,
        }
    return plan


def finalize_call(
    session_state: KvEvictionSessionState,
    plan: KvEvictionPlan,
    *,
    prompt_ids: Sequence[int],
    resident_tokens_after: int,
    physical_tokens_after: int,
    cached_tokens: int,
) -> Dict[str, Any]:
    """Commit the post-call state and return the response metadata."""
    spec = plan.spec
    parent = plan.parent_state
    runs = extend_position_map(
        plan.position_map_after_eviction,
        resident_tokens_after - position_map_len(plan.position_map_after_eviction),
        plan.position_offset,
    )
    state = KvEvictionState(
        cache_id=spec.cache_id,
        call_index=spec.call_index,
        parent_state_id=parent.state_id if parent is not None else None,
        protected_prefix_len=spec.protected_prefix_len,
        position_offset=plan.position_offset,
        resident_tokens=resident_tokens_after,
        physical_tokens=physical_tokens_after,
        position_map=runs,
        last_event_id=(
            plan.event["event_id"]
            if plan.event is not None
            else (parent.last_event_id if parent is not None else None)
        ),
    ).seal()
    session_state.state = state
    session_state.pending = None

    matched = plan.matched_prefix_len
    expected = plan.expected_prefix_len if spec.call_index > 0 else 0
    prompt_len = len(prompt_ids)
    reused = matched if matched is not None else 0
    retained_prefilled = max(0, expected - reused)
    evidence = {
        "slots_freed": plan.slots_freed,
        "retained_slots_unchanged": (
            plan.retained_slots_unchanged
            if plan.retained_slots_unchanged is not None
            else True
        ),
        "cached_tokens": cached_tokens,
        "reused_tokens": reused,
        "new_tokens_prefilled": prompt_len - reused,
        "retained_tokens_prefilled": retained_prefilled,
    }
    return {
        "cache_state": state.to_dict(),
        "event": plan.event,
        "evidence": evidence,
        "call_id": spec.call_id,
        "prompt_token_ids": [int(t) for t in prompt_ids],
        "compaction_events": (
            [plan.compaction_event] if plan.compaction_event is not None else []
        ),
        "compaction_replay_mode": KV_EVICTION_REPLAY_MODE,
    }


# ---------------------------------------------------------------------------
# Physical KV-row splice
# ---------------------------------------------------------------------------


def splice_kv_row(
    tree_cache: "BasePrefixCache",
    kv: "ReqKvInfo",
    spans: Sequence[Tuple[int, int]],
) -> Tuple[int, bool]:
    """Remove ``spans`` from a session's req_to_token row in place.

    Surviving slot ids are moved (not copied as KV) to a dense prefix of the
    row; removed slots are returned to the allocator exactly once. Returns
    ``(slots_freed, survivors_unchanged)``.
    """
    if not spans:
        return 0, True
    committed = kv.kv_committed_len
    assert kv.holds_kv and spans[-1][1] <= committed
    assert kv.kv_allocated_len == committed, (
        "kv_eviction expects no over-allocated tail on an idle session row "
        f"({kv.kv_allocated_len=} {committed=})"
    )
    pool = tree_cache.req_to_token_pool
    row = pool.req_to_token[kv.req_pool_idx]
    old = row[:committed].clone()
    device = old.device
    keep = torch.ones(committed, dtype=torch.bool, device=device)
    for start, end in spans:
        keep[start:end] = False
    survivors = old[keep]
    removed = old[~keep]
    new_len = int(survivors.numel())
    row[:new_len] = survivors
    row[new_len:committed] = 0
    unchanged = bool(torch.equal(row[:new_len], survivors))
    tree_cache.token_to_kv_pool_allocator.free(removed.to(torch.int64))
    kv.kv_committed_len = new_len
    kv.kv_allocated_len = new_len
    kv.clamp_evicted_seqlens(new_len)
    return int(removed.numel()), unchanged


def reject_unsupported_server_args(server_args: Any) -> None:
    """Refuse configurations whose position/cache semantics were not proven."""
    errors = []
    if not getattr(server_args, "enable_streaming_session", False):
        errors.append("requires --enable-streaming-session")
    if getattr(server_args, "page_size", 1) not in (None, 1):
        errors.append("requires --page-size 1")
    for flag in ("tp_size", "pp_size", "dp_size"):
        if (getattr(server_args, flag, 1) or 1) != 1:
            errors.append(f"requires --{flag.replace('_', '-')} 1")
    if (getattr(server_args, "tokenizer_worker_num", 1) or 1) != 1:
        errors.append("requires --tokenizer-worker-num 1")
    if getattr(server_args, "disable_radix_cache", False):
        errors.append("requires the radix cache (drop --disable-radix-cache)")
    if getattr(server_args, "enable_session_radix_cache", False):
        errors.append("is incompatible with --enable-session-radix-cache")
    if getattr(server_args, "speculative_algorithm", None):
        errors.append("is incompatible with speculative decoding")
    if getattr(server_args, "disaggregation_mode", "null") not in (None, "null"):
        errors.append("is incompatible with prefill/decode disaggregation")
    if getattr(server_args, "enable_hierarchical_cache", False):
        errors.append("is incompatible with HiCache")
    if getattr(server_args, "enable_lmcache", False):
        errors.append("is incompatible with LMCache")
    if getattr(server_args, "enable_dp_attention", False):
        errors.append("is incompatible with DP attention")
    if errors:
        raise ValueError("--enable-kv-eviction " + "; ".join(errors))
