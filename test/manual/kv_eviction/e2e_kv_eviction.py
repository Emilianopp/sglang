"""End-to-end validation of --enable-kv-eviction against a live server.

Runs several concurrent streaming sessions with multi-span evictions, then
checks, per call:
  * the authoritative post-eviction prompt == resident + new with spans removed
  * evidence: slots_freed == tokens_evicted, no retained token re-prefilled
  * the state/event hash chain and the vLLM-compatible compaction event
  * engine output logprobs against a HuggingFace reference that reproduces the
    engine semantics exactly: KV of evicted positions is dropped from a live
    cache (survivors keep KV computed with full context) and new tokens run at
    their logical positions. A "renumbered" reference (physical positions for
    new tokens) is reported as a negative control.
Plus the retry / conflict / close protocol.

Usage: python e2e_kv_eviction.py --url http://127.0.0.1:30000 --model <path>
"""

import argparse
import concurrent.futures as cf
import json
import math
import random
import sys
import time

import requests
import torch

FAILURES = []


def check(cond, msg):
    if not cond:
        FAILURES.append(msg)
        print("FAIL:", msg, flush=True)


def post(url, path, body, timeout=600):
    r = requests.post(url + path, json=body, timeout=timeout)
    try:
        data = r.json()
    except Exception:
        data = r.text
    return r.status_code, data


def remove_spans(tokens, spans):
    out, cur = [], 0
    for s, e in spans:
        out.extend(tokens[cur:s])
        cur = e
    out.extend(tokens[cur:])
    return out


def user_turn(tok, text):
    return tok.encode(
        f"<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )


def run_session(url, tok, sid, n_calls, seed, max_new, temperature=0.0, evict=True):
    """Drive one session; return the recorded calls for offline reference."""
    rng = random.Random(seed)
    post(url, "/close_session", {"session_id": sid})
    code, _ = post(
        url,
        "/open_session",
        {"session_id": sid, "capacity_of_str_len": 1 << 20, "streaming": True},
    )
    check(code == 200, f"{sid}: open_session -> {code}")
    system = tok.encode(
        "<|im_start|>system\nYou are a terse assistant.<|im_end|>\n",
        add_special_tokens=False,
    )
    protected = len(system)
    first = system + tok.encode(
        "<|im_start|>user\nCount from one upward, one number per line, and "
        "mention a fruit each line.<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    records = []
    state_id = None
    resident = []
    physical = 0
    for k in range(n_calls):
        new_ids = first if k == 0 else user_turn(tok, f"Continue, turn {k}. Be brief.")
        spans = []
        if k > 0 and evict:
            # 1-3 random disjoint spans inside [protected, physical).
            lo, hi = protected, physical
            cuts = sorted(rng.sample(range(lo, hi), min(hi - lo, 2 * rng.randint(1, 3))))
            for a, b in zip(cuts[0::2], cuts[1::2]):
                if b > a:
                    spans.append([a, b])
        kv = {
            "version": 1,
            "cache_id": sid,
            "call_id": f"{sid}-c{k}",
            "call_index": k,
            "expected_state_id": state_id,
            "protected_prefix_len": protected,
            "evict_spans": spans,
            "evict_group_ids": list(range(len(spans))),
        }
        body = {
            "input_ids": new_ids,
            "sampling_params": {"temperature": temperature, "max_new_tokens": max_new},
            "return_logprob": True,
            "session_params": {"id": sid},
            "kv_eviction": kv,
        }
        code, resp = post(url, "/generate", body)
        check(code == 200, f"{sid} call {k}: status {code} {resp}")
        if code != 200:
            break
        # Retry of the identical body must replay the same response.
        code2, resp2 = post(url, "/generate", body)
        check(
            code2 == 200
            and resp2["kv_eviction"]["cache_state"]["state_id"]
            == resp["kv_eviction"]["cache_state"]["state_id"],
            f"{sid} call {k}: identical retry did not replay ({code2})",
        )
        info = resp["kv_eviction"]
        prompt = resp["prompt_token_ids"]
        expect_prompt = remove_spans(resident + new_ids, spans)
        check(prompt == expect_prompt, f"{sid} call {k}: prompt_token_ids mismatch")
        out_ids = resp["output_ids"]
        ev = info["evidence"]
        evicted = sum(e - s for s, e in spans)
        check(ev["slots_freed"] == evicted, f"{sid} call {k}: slots_freed {ev}")
        check(ev["retained_tokens_prefilled"] == 0, f"{sid} call {k}: re-prefill {ev}")
        check(ev["retained_slots_unchanged"], f"{sid} call {k}: survivors changed")
        if k > 0:
            check(
                ev["reused_tokens"] == physical - evicted,
                f"{sid} call {k}: reused {ev['reused_tokens']} != {physical - evicted}",
            )
        st = info["cache_state"]
        check(st["parent_state_id"] == state_id, f"{sid} call {k}: broken state chain")
        check(st["resident_tokens"] == len(prompt) + len(out_ids), f"{sid} {k}: resident")
        # The last sampled token has KV only if the engine already ran it
        # (overlap scheduling launches one more decode before seeing the stop).
        check(
            st["physical_tokens"] in (st["resident_tokens"] - 1, st["resident_tokens"]),
            f"{sid} call {k}: physical {st['physical_tokens']} vs resident "
            f"{st['resident_tokens']}",
        )
        events = resp["compaction_events"]
        if spans:
            check(len(events) == 1, f"{sid} call {k}: expected one coalesced event")
            e = events[0]
            check(e["kept_token_ids"] == prompt, f"{sid} call {k}: kept_token_ids")
            check(e["tokens_evicted"] == evicted, f"{sid} call {k}: tokens_evicted")
            check(
                e["position_offset_after"] == st["position_offset"],
                f"{sid} call {k}: offset mismatch",
            )
        else:
            check(events == [], f"{sid} call {k}: unexpected event")
        engine_lp = [x[0] for x in resp["meta_info"]["output_token_logprobs"]]
        records.append(
            dict(
                call=k,
                new_ids=new_ids,
                spans=spans,
                prompt=prompt,
                out_ids=out_ids,
                engine_lp=engine_lp,
                offset=st["position_offset"],
                physical_after=st["physical_tokens"],
                resident_after=st["resident_tokens"],
                position_map=st["position_map"],
            )
        )
        state_id = st["state_id"]
        resident = prompt + out_ids
        physical = st["physical_tokens"]

    # Protocol checks on the final state.
    if records:
        k = len(records)
        stale = {
            "input_ids": user_turn(tok, "stale"),
            "sampling_params": {"temperature": 0, "max_new_tokens": 2},
            "session_params": {"id": sid},
            "kv_eviction": {
                "version": 1,
                "cache_id": sid,
                "call_id": f"{sid}-stale",
                "call_index": k,
                "expected_state_id": "sha256:" + "0" * 64,
                "protected_prefix_len": protected,
                "evict_spans": [],
            },
        }
        code, _ = post(url, "/generate", stale)
        check(code == 409, f"{sid}: stale expected_state_id -> {code}, want 409")
        reuse = dict(stale)
        reuse["kv_eviction"] = dict(stale["kv_eviction"], call_id=f"{sid}-c0")
        code, _ = post(url, "/generate", reuse)
        check(code == 409, f"{sid}: reused call_id with new body -> {code}, want 409")
        code, status = post(url, "/session_status", {"session_id": sid})
        check(
            code == 200 and status["kv_eviction_state"]["state_id"] == state_id,
            f"{sid}: session_status after rejected calls {status}",
        )
        code, closed = post(url, "/close_session", {"session_id": sid, "wait_timeout": 5})
        check(code == 200 and closed.get("status") == "closed", f"{sid}: close {closed}")
        code, status = post(url, "/session_status", {"session_id": sid})
        check(code == 200 and status["present"] is False, f"{sid}: still present")
    return records


def prune_cache(cache, keep):
    """Drop evicted positions from attention KV only.

    Hybrid linear-attention layers (Qwen3.5 Gated DeltaNet) keep their
    recurrent/conv state untouched -- exactly what the server does.
    """
    from transformers.cache_utils import LinearAttentionCacheLayerMixin

    idx = torch.tensor(keep, device="cuda", dtype=torch.long)
    if hasattr(cache, "layers"):
        for layer in cache.layers:
            if isinstance(layer, LinearAttentionCacheLayerMixin) and not hasattr(
                layer, "values"
            ):
                continue
            if getattr(layer, "keys", None) is None or layer.keys.numel() == 0:
                continue
            layer.keys = layer.keys.index_select(-2, idx)
            layer.values = layer.values.index_select(-2, idx)
    else:
        for i in range(len(cache.key_cache)):
            cache.key_cache[i] = cache.key_cache[i].index_select(-2, idx)
            cache.value_cache[i] = cache.value_cache[i].index_select(-2, idx)


@torch.no_grad()
def reference_logprobs(model, records, renumber):
    """Replay a session in HF with engine semantics; return per-call logprobs."""
    from transformers import DynamicCache

    cache = DynamicCache(config=model.config)
    phys_logical = []  # logical position of each cached (physical) token
    deferred = None  # (token, logical) sampled but not yet in cache
    results = []
    for rec in records:
        if rec["spans"]:
            resident_len = len(phys_logical) + (1 if deferred else 0)
            keep = [
                i
                for i in range(len(phys_logical))
                if not any(s <= i < e for s, e in rec["spans"])
            ]
            prune_cache(cache, keep)
            phys_logical = [phys_logical[i] for i in keep]
            assert resident_len >= len(phys_logical)
        offset = rec["offset"]
        feed = ([deferred[0]] if deferred else []) + rec["new_ids"]
        out = rec["out_ids"]
        # Mirror the engine: materialize the last token iff it did.
        materialized = rec["physical_after"] == rec["resident_after"]
        feed_all = feed + (out if materialized else out[:-1])
        start = len(phys_logical)
        phys = list(range(start, start + len(feed_all)))
        logical = [p if renumber else p + offset for p in phys]
        ids = torch.tensor([feed_all], device="cuda")
        pos = torch.tensor([logical], device="cuda")
        cpos = torch.tensor(phys, device="cuda")
        logits = model(
            input_ids=ids,
            position_ids=pos,
            cache_position=cpos,
            past_key_values=cache,
            use_cache=True,
        ).logits[0].float()
        lp = torch.log_softmax(logits, -1)
        first = len(feed) - 1
        ref = [lp[first + j, t].item() for j, t in enumerate(out)]
        argmax_ok = [
            int(lp[first + j].argmax().item() == t) for j, t in enumerate(out)
        ]
        results.append((ref, argmax_ok))
        phys_logical.extend(logical)
        assert len(phys_logical) == rec["physical_after"], (
            len(phys_logical),
            rec["physical_after"],
        )
        deferred = (
            None if materialized else (out[-1], phys[-1] + 1 + (0 if renumber else offset))
        )
    return results


def expand_positions(runs):
    return [logical + i for _, logical, length in runs for i in range(length)]


@torch.no_grad()
def reprefill_logprobs(model, records):
    """Control: recompute survivors from scratch at their logical positions.

    Surviving tokens no longer see the evicted context (and on hybrid models
    the recurrent state is rebuilt from survivors only) -- the semantics the
    engine deliberately does NOT implement.
    """
    results = []
    for rec in records:
        tokens = rec["prompt"] + rec["out_ids"]
        pos = expand_positions(rec["position_map"])[: len(tokens)]
        assert len(pos) == len(tokens)
        logits = model(
            input_ids=torch.tensor([tokens], device="cuda"),
            position_ids=torch.tensor([pos], device="cuda"),
            use_cache=False,
        ).logits[0].float()
        lp = torch.log_softmax(logits, -1)
        first = len(rec["prompt"]) - 1
        ref = [lp[first + j, t].item() for j, t in enumerate(rec["out_ids"])]
        top1 = [int(lp[first + j].argmax().item() == t) for j, t in enumerate(rec["out_ids"])]
        results.append((ref, top1))
    return results


_NEVER = 1 << 30
_FLEX_STATE = {"block_mask": None}


def _kv_death_flex_attention(module, query, key, value, attention_mask, **kwargs):
    """Attention layer = flex_attention with the session's death-position mask."""
    from transformers.integrations.flex_attention import flex_attention_forward

    kwargs.pop("dropout", None)
    return flex_attention_forward(
        module, query, key, value, _FLEX_STATE["block_mask"], **kwargs
    )


def enable_kv_death_flex(model):
    """Register and switch the model's attention layers to the death-mask flex path."""
    from transformers import AttentionInterface

    AttentionInterface.register("kv_death_flex", _kv_death_flex_attention)
    model.set_attn_implementation("kv_death_flex")


def build_chronological_stream(records):
    """Chronological token stream of one session + per-token call/death indices.

    S = prompt_0 + out_0 + new_1 + out_1 + ...; S index == engine logical
    position. query_call[i] = call in which token i was computed (a deferred
    final token is computed at the start of the next call); death[j] = call
    whose admission evicted token j.
    """
    S, query_call, death, resident, score_idx = [], [], [], [], []
    pending = None
    for k, rec in enumerate(records):
        if rec["spans"]:
            evicted = {resident[i] for a, b in rec["spans"] for i in range(a, b)}
            for j in evicted:
                death[j] = k
            resident = [j for j in resident if j not in evicted]
        if pending is not None:
            query_call[pending] = k
            pending = None
        for t in rec["new_ids"]:
            S.append(t), query_call.append(k), death.append(_NEVER)
            resident.append(len(S) - 1)
        start = len(S)
        for t in rec["out_ids"]:
            S.append(t), query_call.append(k), death.append(_NEVER)
            resident.append(len(S) - 1)
        if rec["physical_after"] != rec["resident_after"]:
            pending = len(S) - 1
            query_call[pending] = k + 1
        score_idx.append(list(range(start - 1, start - 1 + len(rec["out_ids"]))))
        assert len(resident) == rec["resident_after"], (len(resident), rec["resident_after"])
    return S, query_call, death, score_idx


@torch.no_grad()
def flex_logprobs(model, records):
    """Trainer-style reference: ONE forward over the chronological stream.

    Full-attention layers use flex_attention with key j visible to query i iff
    j <= i and query_call[i] < death[j]; linear-attention (DeltaNet) layers see
    the whole unpruned stream. Should equal the incremental replay exactly (up
    to kernel numerics).
    """
    from torch.nn.attention.flex_attention import create_block_mask

    S, query_call, death, score_idx = build_chronological_stream(records)
    n = len(S)
    qc = torch.tensor(query_call, device="cuda", dtype=torch.int32)
    dth = torch.tensor(death, device="cuda", dtype=torch.int32)

    def mask_mod(b, h, q_idx, kv_idx):
        return (kv_idx <= q_idx) & (qc[q_idx] < dth[kv_idx])

    _FLEX_STATE["block_mask"] = create_block_mask(
        mask_mod, B=None, H=None, Q_LEN=n, KV_LEN=n, device="cuda"
    )
    logits = model(
        input_ids=torch.tensor([S], device="cuda"),
        position_ids=torch.arange(n, device="cuda")[None],
        use_cache=False,
    ).logits[0].float()
    lp = torch.log_softmax(logits, -1)
    results = []
    for rec, idxs in zip(records, score_idx):
        ref = [lp[i, t].item() for i, t in zip(idxs, rec["out_ids"])]
        top1 = [int(lp[i].argmax().item() == t) for i, t in zip(idxs, rec["out_ids"])]
        results.append((ref, top1))
    return results


def k3(eng_lp, ref_lp):
    """Schulman k3 for KL(engine || reference) on engine-sampled tokens."""
    lr = ref_lp - eng_lp
    return math.exp(lr) - 1.0 - lr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--sessions", type=int, default=4)
    ap.add_argument(
        "--control-sessions",
        type=int,
        default=0,
        help="extra sessions that append every call but never evict (depth-matched noise floor)",
    )
    ap.add_argument("--calls", type=int, default=6)
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument("--tag", default="")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--dump", default=None, help="write per-token logprobs here")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    t0 = time.time()
    n_total = args.sessions + args.control_sessions
    with cf.ThreadPoolExecutor(n_total) as pool:
        futs = [
            pool.submit(
                run_session,
                args.url,
                tok,
                f"kve-{args.tag}-{i}",
                args.calls,
                i,
                args.max_new,
                args.temperature,
                i < args.sessions,
            )
            for i in range(n_total)
        ]
        all_records = [f.result() for f in futs]
    print(f"[{args.tag}] sessions done in {time.time() - t0:.1f}s", flush=True)

    # Chat endpoint smoke: one call 0 through /v1/chat/completions.
    sid = f"kve-{args.tag}-chat"
    post(args.url, "/open_session", {"session_id": sid, "capacity_of_str_len": 1 << 20, "streaming": True})
    ids = tok.encode("<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)
    code, chat = post(
        args.url,
        "/v1/chat/completions",
        {
            "model": "m",
            "messages": [{"role": "user", "content": "Hi"}],
            "input_ids": ids,
            "return_token_ids": True,
            "max_tokens": 8,
            "temperature": 0,
            "session_params": {"id": sid},
            "kv_eviction": {
                "version": 1, "cache_id": sid, "call_id": f"{sid}-c0",
                "call_index": 0, "expected_state_id": None, "protected_prefix_len": 0,
            },
        },
    )
    check(code == 200 and chat.get("kv_eviction", {}).get("cache_state", {}).get("call_index") == 0,
          f"chat kv_eviction call 0 -> {code} {str(chat)[:300]}")
    check(code == 200 and chat.get("prompt_token_ids") == ids, "chat prompt_token_ids echo")
    post(args.url, "/close_session", {"session_id": sid, "wait_timeout": 5})

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
    except (ValueError, KeyError):
        # Multimodal checkpoints (e.g. Qwen3.5 *ForConditionalGeneration).
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(
            args.model, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
    model = model.cuda().eval()
    variants = ("exact", "flex", "renumbered", "reprefill")
    all_refs = []
    for records in all_records:
        all_refs.append(
            {
                "exact": reference_logprobs(model, records, renumber=False),
                "renumbered": reference_logprobs(model, records, renumber=True),
                "reprefill": reprefill_logprobs(model, records),
            }
        )
    # Same weights, attention layers switched to the death-mask flex kernel.
    enable_kv_death_flex(model)
    for records, refs in zip(all_records, all_refs):
        refs["flex"] = flex_logprobs(model, records)
    rows = []
    for si, (records, refs) in enumerate(zip(all_records, all_refs)):
        for ci, rec in enumerate(records):
            if rec["spans"]:
                group = "evicting"
            elif rec["call"] == 0:
                group = "call0"
            else:
                group = "later_no_eviction"
            for j, (tok_id, eng) in enumerate(zip(rec["out_ids"], rec["engine_lp"])):
                row = dict(session=si, call=rec["call"], group=group, j=j,
                           token=tok_id, engine_lp=eng,
                           depth=len(rec["prompt"]) + j,
                           chrono=rec["offset"] + len(rec["prompt"]) + j)
                for v in variants:
                    ref, top1 = refs[v][ci]
                    row[f"{v}_lp"] = ref[j]
                    row[f"{v}_top1"] = top1[j]
                rows.append(row)
    if args.dump:
        with open(args.dump, "w") as f:
            json.dump(rows, f)

    mean = lambda xs: sum(xs) / max(1, len(xs))
    breakdown = {}
    for group in ("call0", "later_no_eviction", "evicting"):
        sel = [r for r in rows if r["group"] == group]
        breakdown[group] = {
            "tokens": len(sel),
            # physical context length (what the kernels see) and logical position
            "mean_physical_depth": mean([r["depth"] for r in sel]),
            "mean_logical_position": mean([r["chrono"] for r in sel]),
        }
        for v in variants:
            breakdown[group][v] = {
                "k3_kl": mean([k3(r["engine_lp"], r[f"{v}_lp"]) for r in sel]),
                "mean_abs_dlogprob": mean([abs(r[f"{v}_lp"] - r["engine_lp"]) for r in sel]),
                "mean_signed_dlogprob": mean([r[f"{v}_lp"] - r["engine_lp"] for r in sel]),
                "max_abs_dlogprob": max([abs(r[f"{v}_lp"] - r["engine_lp"]) for r in sel] or [0]),
                "top1_agreement": mean([r[f"{v}_top1"] for r in sel]),
            }
    exact_all = [abs(r["exact_lp"] - r["engine_lp"]) for r in rows]
    # Direct check that the trainer-style flex formulation == incremental replay.
    for group in breakdown:
        sel = [r for r in rows if r["group"] == group]
        breakdown[group]["flex_vs_exact_mean_abs"] = mean(
            [abs(r["flex_lp"] - r["exact_lp"]) for r in sel]
        )
        breakdown[group]["flex_vs_exact_max_abs"] = max(
            [abs(r["flex_lp"] - r["exact_lp"]) for r in sel] or [0]
        )
    res = {
        "tag": args.tag,
        "temperature": args.temperature,
        "tokens_compared": len(rows),
        "mean_abs_dlogprob_correct": mean(exact_all),
        "breakdown": breakdown,
        "calls": sum(len(r) for r in all_records),
        "evicting_calls": sum(1 for r in all_records for c in r if c["spans"]),
        "failures": FAILURES,
    }
    print("RESULT", json.dumps(res), flush=True)
    check(res["mean_abs_dlogprob_correct"] < 0.05, "logprob mismatch vs exact reference")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
