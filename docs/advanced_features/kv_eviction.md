# Explicit KV eviction (streaming sessions)

`--enable-kv-eviction` lets a client remove arbitrary token spans from a
streaming session's resident KV cache. The client decides *what* to forget;
the server splices the removed slots out of the session's KV row in place.
Surviving KV is never recomputed and keeps the logical (RoPE) position it was
computed at.

## Launch

```bash
python -m sglang.launch_server --model-path Qwen/Qwen3-4B \
  --enable-streaming-session --enable-kv-eviction --page-size 1
```

Required: `--page-size 1`, TP/PP/DP = 1, one tokenizer worker, the radix
cache enabled. Rejected with: speculative decoding, PD disaggregation,
HiCache, LMCache, DP attention, `--enable-session-radix-cache`.

## Session lifecycle

| Endpoint | Body | Notes |
|---|---|---|
| `POST /open_session` | `{"session_id", "capacity_of_str_len", "streaming": true, "timeout"}` | The id becomes the `cache_id`. Each open starts a new incarnation. |
| `POST /session_status` | `{"session_id"}` | Scheduler-authoritative: `present`, `inflight`, `deferred`, `kv_eviction_state`, `kv_eviction_invalid_reason`. |
| `POST /close_session` | `{"session_id", "wait_timeout"}` | `200 {"status":"closed"}` only once the scheduler confirms absence; otherwise `409` with `closing`/`timeout`. |

## A call

Both `/generate` and `/v1/chat/completions` accept a `kv_eviction` object.
Requirements: exact token input (`input_ids`), non-streaming, `n=1`,
`session_params.id == cache_id`, `rid == call_id` (or unset).

```json
{
  "input_ids": [ ...tokens appended by this call... ],
  "session_params": {"id": "ep-1"},
  "kv_eviction": {
    "version": 1,
    "cache_id": "ep-1",
    "call_id": "ep-1-c4",
    "call_index": 4,
    "expected_state_id": "sha256:...",
    "protected_prefix_len": 32,
    "evict_spans": [[32, 64], [256, 512], [617, 632]],
    "evict_group_ids": [0, 1, 2]
  }
}
```

* Spans are half-open `[start, end)` indices into the **resident sequence** —
  the previous call's returned `prompt_token_ids` followed by its output
  tokens. Any number of spans; sorted, non-overlapping, `start >=
  protected_prefix_len`, `end <= cache_state.physical_tokens` (the last
  sampled token has no KV yet and cannot be evicted). `{"start","end"}`
  objects are accepted too.
* `evict_group_ids` is optional client metadata, echoed in the event.
* `call_index=0` claims an empty, freshly opened session, cannot evict, and
  sends `expected_state_id=null`. Call 0 bypasses radix prefix sharing so the
  session exclusively owns every KV slot it may later free.
* Every later call sends `call_index = previous + 1` and the previous
  `cache_state.state_id` as `expected_state_id`.

## Response (additions)

```json
{
  "prompt_token_ids": [ ...authoritative post-eviction prompt... ],
  "kv_eviction": {
    "call_id": "ep-1-c4",
    "cache_state": {
      "state_id": "sha256:...", "parent_state_id": "sha256:...",
      "call_index": 4, "protected_prefix_len": 32,
      "position_offset": 303, "resident_tokens": 1180, "physical_tokens": 1179,
      "position_map": [[0, 0, 32], [32, 64, 192], [224, 512, 105], [329, 632, 851]],
      "last_event_id": "sha256:..."
    },
    "event": {"event_id", "parent_event_id", "evicted_spans", "evicted_group_ids",
              "tokens_evicted", "position_offset_after"},
    "evidence": {"slots_freed", "retained_slots_unchanged", "cached_tokens",
                 "reused_tokens", "new_tokens_prefilled", "retained_tokens_prefilled"}
  },
  "compaction_events": [{ "num_output_tokens_at_compaction": 0, "tokens_evicted",
    "position_offset_after", "num_prompt_tokens", "evict_start",
    "new_user_fragment_len", "kept_indices", "kept_token_ids", "event_kind": 0 }],
  "compaction_replay_mode": "prefill_trim"
}
```

* `position_map` runs are `[phys_start, logical_start, length]` covering the
  resident sequence: the logical position of every surviving token. New tokens
  are computed at `physical + position_offset`.
* `compaction_events` holds exactly one coalesced admission event per evicting
  call (empty otherwise). `kept_indices` address the assembled pre-eviction
  prompt (resident + appended input); `kept_token_ids == prompt_token_ids`.
* `evidence.retained_tokens_prefilled` is `0` and `slots_freed ==
  tokens_evicted` on every successful call.

## Semantics worth knowing

* Surviving KV was computed while the evicted tokens were still visible. A
  plain forward over the surviving tokens is therefore **not** equivalent;
  an exact reference drops the evicted positions from a live KV cache and
  continues at the logical positions (see
  `test/manual/kv_eviction/e2e_kv_eviction.py`).
* Retries: an identical body under the same `call_id` joins the in-flight call
  or replays its cached response; a different body is `409`; another call
  while one is active is `409`. The mutation is shielded from HTTP
  disconnects. The cache is process-local (one tokenizer worker); route every
  call of a session to the same server.
* A stale `expected_state_id` / `call_index` is `409` and changes nothing. If a
  call is aborted after its KV was spliced, the session is invalidated (every
  later call is `409`); close and reopen it.
