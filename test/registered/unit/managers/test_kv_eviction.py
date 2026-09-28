import asyncio
from http import HTTPStatus
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.managers.kv_eviction import (
    KvEvictionError,
    KvEvictionSessionState,
    build_plan,
    evict_from_position_map,
    extend_position_map,
    finalize_call,
    kept_indices,
    logical_positions,
    parse_kv_eviction_request,
    position_map_len,
    remove_spans,
    splice_kv_row,
    validate_call,
)
from sglang.srt.managers.kv_eviction_registry import KvEvictionCallRegistry
from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

STATE = "sha256:" + "a" * 64


def _raw(**overrides):
    raw = {
        "version": 1,
        "cache_id": "ep-1",
        "call_id": "ep-1-c0",
        "call_index": 0,
        "expected_state_id": None,
        "protected_prefix_len": 4,
        "evict_spans": [],
        "evict_group_ids": [],
    }
    raw.update(overrides)
    return raw


# -- request parsing ---------------------------------------------------------


def test_parse_accepts_multiple_spans_in_both_forms():
    spec = parse_kv_eviction_request(
        _raw(
            call_index=3,
            expected_state_id=STATE,
            evict_spans=[[32, 64], {"start": 256, "end": 512}, [617, 632]],
        )
    )
    assert spec.evict_spans == ((32, 64), (256, 512), (617, 632))
    assert spec.tokens_evicted == 32 + 256 + 15
    assert spec.evict_group_ids == ()


@pytest.mark.parametrize(
    "overrides, fragment",
    [
        ({"extra": 1}, "unknown fields"),
        ({"version": 2}, "version"),
        ({"version": True}, "version"),
        ({"cache_id": "bad id"}, "cache_id"),
        ({"call_index": -1}, "call_index"),
        ({"expected_state_id": STATE}, "expected_state_id=null"),
        ({"call_index": 1}, "expected_state_id"),
        ({"call_index": 1, "expected_state_id": "sha256:ABC"}, "expected_state_id"),
        ({"evict_spans": [[8, 9]]}, "cannot evict"),
        (
            {"call_index": 1, "expected_state_id": STATE, "evict_spans": [[2, 9]]},
            "protected prefix",
        ),
        (
            {
                "call_index": 1,
                "expected_state_id": STATE,
                "evict_spans": [[10, 20], [15, 30]],
            },
            "non-overlapping",
        ),
        (
            {
                "call_index": 1,
                "expected_state_id": STATE,
                "evict_spans": [[20, 30], [10, 12]],
            },
            "sorted",
        ),
        (
            {"call_index": 1, "expected_state_id": STATE, "evict_spans": [[9, 9]]},
            "start < end",
        ),
        (
            {"call_index": 1, "expected_state_id": STATE, "evict_spans": [[9, 12.0]]},
            "integers",
        ),
        (
            {
                "call_index": 1,
                "expected_state_id": STATE,
                "evict_spans": [[9, 12]],
                "evict_group_ids": [2, 1],
            },
            "sorted and unique",
        ),
        ({"evict_group_ids": [1]}, "without any evict_spans"),
    ],
)
def test_parse_rejects(overrides, fragment):
    with pytest.raises(KvEvictionError, match=fragment):
        parse_kv_eviction_request(_raw(**overrides))


def test_adjacent_spans_are_allowed():
    spec = parse_kv_eviction_request(
        _raw(call_index=1, expected_state_id=STATE, evict_spans=[[4, 8], [8, 10]])
    )
    assert spec.tokens_evicted == 6


# -- position map ------------------------------------------------------------


def test_position_map_multi_span_keeps_original_positions():
    runs = extend_position_map([], 1024, 0)
    spans = [(32, 64), (256, 512), (617, 632)]
    after = evict_from_position_map(runs, spans)
    kept = kept_indices(1024, spans)
    # Survivors keep the logical position they had before eviction.
    assert logical_positions(after) == kept
    assert position_map_len(after) == 1024 - (32 + 256 + 15)
    # New tokens continue from the largest logical position so far.
    grown = extend_position_map(after, 3, 32 + 256 + 15)
    assert logical_positions(grown)[-3:] == [1024, 1025, 1026]


def test_position_map_repeated_evictions_compose():
    runs = extend_position_map([], 100, 0)
    runs = evict_from_position_map(runs, [(10, 20)])  # offset 10
    runs = extend_position_map(runs, 10, 10)  # logical 100..109
    # Physical coordinates of the current sequence: evict two survivor pieces.
    runs = evict_from_position_map(runs, [(5, 15), (80, 85)])
    expected = [
        p
        for p in list(range(0, 10)) + list(range(20, 110))
        if p not in set(range(5, 10)) | set(range(20, 25)) | set(range(90, 95))
    ]
    assert logical_positions(runs) == expected


def test_remove_spans_and_kept_indices_agree():
    tokens = list(range(100, 140))
    spans = [(4, 8), (20, 30)]
    kept = kept_indices(len(tokens), spans)
    assert remove_spans(tokens, spans) == [tokens[i] for i in kept]


# -- validation / plan / finalize ---------------------------------------------


def _run_call(state, spec_raw, *, assembled, resident, physical, new_len):
    spec = parse_kv_eviction_request(spec_raw)
    validate_call(
        state,
        spec,
        session_has_history=state.state is not None,
        resident_tokens=resident,
        physical_tokens=physical,
        new_input_len=new_len,
        context_len=4096,
        max_new_tokens=8,
    )
    plan = build_plan(
        state,
        spec,
        assembled_prompt=assembled,
        resident_tokens=resident,
        physical_tokens=physical,
    )
    state.pending = plan
    return spec, plan


def test_call_chain_state_and_events():
    state = KvEvictionSessionState()
    # Call 0: 10 prompt tokens, 5 generated (last one not yet materialized).
    prompt0 = list(range(10))
    _, plan0 = _run_call(
        state, _raw(), assembled=prompt0, resident=0, physical=0, new_len=10
    )
    plan0.matched_prefix_len = 0
    info0 = finalize_call(
        state,
        plan0,
        prompt_ids=prompt0,
        resident_tokens_after=15,
        physical_tokens_after=14,
        cached_tokens=0,
    )
    s0 = info0["cache_state"]
    assert s0["call_index"] == 0 and s0["parent_state_id"] is None
    assert info0["compaction_events"] == [] and info0["event"] is None

    # Call 1: evict two spans of the resident 15, append 3 new tokens.
    resident = list(range(15))
    assembled = resident + [90, 91, 92]
    spec_raw = _raw(
        call_id="ep-1-c1",
        call_index=1,
        expected_state_id=s0["state_id"],
        evict_spans=[[4, 6], [8, 11]],
        evict_group_ids=[0, 2],
    )
    spec1, plan1 = _run_call(
        state, spec_raw, assembled=assembled, resident=15, physical=14, new_len=3
    )
    assert plan1.position_offset == 5
    assert plan1.expected_prefix_len == 9
    event = plan1.compaction_event
    assert event["tokens_evicted"] == 5 and event["evict_start"] == 4
    assert event["kept_token_ids"] == remove_spans(assembled, spec1.evict_spans)
    assert event["num_prompt_tokens"] == len(assembled) - 5
    assert event["new_user_fragment_len"] == 3

    plan1.matched_prefix_len = 9
    prompt1 = remove_spans(assembled, spec1.evict_spans)
    info1 = finalize_call(
        state,
        plan1,
        prompt_ids=prompt1,
        resident_tokens_after=len(prompt1) + 4,
        physical_tokens_after=len(prompt1) + 3,
        cached_tokens=9,
    )
    s1 = info1["cache_state"]
    assert s1["parent_state_id"] == s0["state_id"]
    assert s1["position_offset"] == 5
    assert s1["last_event_id"] == info1["event"]["event_id"]
    assert info1["evidence"]["retained_tokens_prefilled"] == 0
    assert info1["evidence"]["new_tokens_prefilled"] == len(prompt1) - 9
    positions = logical_positions(s1["position_map"])
    assert positions[:9] == [0, 1, 2, 3, 6, 7, 11, 12, 13]
    assert positions[9:] == list(range(14, 14 + len(positions) - 9))


def test_validate_rejects_stale_and_out_of_order():
    state = KvEvictionSessionState()
    _, plan0 = _run_call(
        state, _raw(), assembled=list(range(10)), resident=0, physical=0, new_len=10
    )
    info0 = finalize_call(
        state,
        plan0,
        prompt_ids=list(range(10)),
        resident_tokens_after=12,
        physical_tokens_after=11,
        cached_tokens=0,
    )
    sid = info0["cache_state"]["state_id"]

    def check(spec_raw, **kw):
        args = dict(resident=12, physical=11, new_len=2)
        args.update(kw)
        with pytest.raises(KvEvictionError) as exc:
            _run_call(state, spec_raw, assembled=[0] * 14, **args)
        return exc.value

    later = dict(call_id="c1", call_index=1, expected_state_id=sid)
    assert check(_raw(**{**later, "expected_state_id": STATE})).status_code == 409
    assert check(_raw(**{**later, "call_index": 2})).status_code == 409
    assert check(_raw(**later), resident=13).status_code == 409
    assert "protected_prefix_len" in str(
        check(_raw(**{**later, "protected_prefix_len": 5}))
    )
    # Cannot evict the deferred (not yet materialized) token at index 11.
    assert "materialized" in str(check(_raw(**{**later, "evict_spans": [[8, 12]]})))
    # Cannot claim a session that already has history.
    with pytest.raises(KvEvictionError, match="empty"):
        validate_call(
            KvEvictionSessionState(),
            parse_kv_eviction_request(_raw()),
            session_has_history=True,
            resident_tokens=0,
            physical_tokens=0,
            new_input_len=1,
            context_len=100,
            max_new_tokens=1,
        )


def test_validate_enforces_logical_context_length():
    state = KvEvictionSessionState()
    spec = parse_kv_eviction_request(_raw())
    with pytest.raises(KvEvictionError, match="context length"):
        validate_call(
            state,
            spec,
            session_has_history=False,
            resident_tokens=0,
            physical_tokens=0,
            new_input_len=90,
            context_len=100,
            max_new_tokens=20,
        )


def test_invalidated_session_refuses_calls():
    state = KvEvictionSessionState(invalid_reason="boom")
    with pytest.raises(KvEvictionError, match="invalidated") as exc:
        validate_call(
            state,
            parse_kv_eviction_request(_raw()),
            session_has_history=False,
            resident_tokens=0,
            physical_tokens=0,
            new_input_len=1,
            context_len=100,
            max_new_tokens=1,
        )
    assert exc.value.status_code == HTTPStatus.CONFLICT


# -- physical splice -----------------------------------------------------------


class _Allocator:
    def __init__(self):
        self.freed = []

    def free(self, indices):
        self.freed.append(indices.clone())


def test_splice_kv_row_moves_survivor_slots_and_frees_removed_once():
    req_to_token = torch.zeros((2, 32), dtype=torch.int32)
    slots = torch.arange(100, 120, dtype=torch.int32)
    req_to_token[1, :20] = slots
    alloc = _Allocator()
    cache = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
        token_to_kv_pool_allocator=alloc,
    )
    kv = ReqKvInfo(req_pool_idx=1, kv_committed_len=20, kv_allocated_len=20)
    freed, unchanged = splice_kv_row(cache, kv, [(2, 5), (10, 12), (18, 20)])
    assert freed == 7 and unchanged
    survivors = [100, 101, 105, 106, 107, 108, 109, 112, 113, 114, 115, 116, 117]
    assert req_to_token[1, :13].tolist() == survivors
    assert kv.kv_committed_len == kv.kv_allocated_len == 13
    [freed_slots] = alloc.freed
    assert sorted(freed_slots.tolist()) == [102, 103, 104, 110, 111, 118, 119]
    # The other row is untouched.
    assert req_to_token[0].abs().sum() == 0


# -- exactly-once registry -----------------------------------------------------


def test_registry_joins_replays_and_conflicts():
    async def scenario():
        registry = KvEvictionCallRegistry()
        calls = []
        gate = asyncio.Event()

        async def producer(generation):
            calls.append(generation)
            await gate.wait()
            return HTTPStatus.OK, {"ok": len(calls)}

        body = {"x": 1}
        first = asyncio.ensure_future(
            registry.run(cache_id="c", call_id="k0", body=body, producer=producer)
        )
        await asyncio.sleep(0)
        joined = asyncio.ensure_future(
            registry.run(cache_id="c", call_id="k0", body=body, producer=producer)
        )
        await asyncio.sleep(0)
        with pytest.raises(KvEvictionError) as busy:
            await registry.run(cache_id="c", call_id="k1", body=body, producer=producer)
        assert busy.value.status_code == HTTPStatus.CONFLICT
        gate.set()
        assert await first == await joined == (HTTPStatus.OK, {"ok": 1})
        # Identical retry after completion replays the cached response.
        assert await registry.run(
            cache_id="c", call_id="k0", body=body, producer=producer
        ) == (HTTPStatus.OK, {"ok": 1})
        with pytest.raises(KvEvictionError, match="different request body"):
            await registry.run(
                cache_id="c", call_id="k0", body={"x": 2}, producer=producer
            )
        assert calls == [0]
        # A reopen starts a new incarnation: stale dispatch is refused.
        registry.invalidate("c")
        with pytest.raises(KvEvictionError, match="reopened"):
            registry.check_dispatch("c", 0)
        registry.check_dispatch("c", 1)

    asyncio.run(scenario())


def test_registry_cancelled_waiter_does_not_cancel_producer():
    async def scenario():
        registry = KvEvictionCallRegistry()
        done = asyncio.Event()

        async def producer(generation):
            await asyncio.sleep(0.01)
            done.set()
            return HTTPStatus.OK, {}

        waiter = asyncio.ensure_future(
            registry.run(cache_id="c", call_id="k0", body={}, producer=producer)
        )
        await asyncio.sleep(0)
        waiter.cancel()
        await asyncio.wait_for(done.wait(), 1)

    asyncio.run(scenario())
