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


def run_session(url, tok, sid, n_calls, seed, max_new):
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
        if k > 0:
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
            "sampling_params": {"temperature": 0, "max_new_tokens": max_new},
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
        check(
            st["physical_tokens"] == st["resident_tokens"] - 1,
            f"{sid} call {k}: physical {st['physical_tokens']} != resident - 1",
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
    idx = torch.tensor(keep, device="cuda", dtype=torch.long)
    if hasattr(cache, "layers"):
        for layer in cache.layers:
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

    cache = DynamicCache()
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
        feed_all = feed + out[:-1]
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
        deferred = (out[-1], phys[-1] + 1 + (0 if renumber else offset))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--sessions", type=int, default=4)
    ap.add_argument("--calls", type=int, default=6)
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    t0 = time.time()
    with cf.ThreadPoolExecutor(args.sessions) as pool:
        futs = [
            pool.submit(
                run_session, args.url, tok, f"kve-{args.tag}-{i}", args.calls, i, args.max_new
            )
            for i in range(args.sessions)
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

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).cuda().eval()
    summary = {"correct": [], "renumbered": [], "top1": [], "top1_renum": []}
    for records in all_records:
        good = reference_logprobs(model, records, renumber=False)
        bad = reference_logprobs(model, records, renumber=True)
        for rec, (ref, top1), (ref_bad, top1_bad) in zip(records, good, bad):
            eng = rec["engine_lp"]
            d = [abs(a - b) for a, b in zip(eng, ref)]
            d_bad = [abs(a - b) for a, b in zip(eng, ref_bad)]
            summary["correct"].extend(d)
            summary["top1"].extend(top1)
            if rec["call"] > 0 and rec["offset"] > 0:
                summary["renumbered"].extend(d_bad)
                summary["top1_renum"].extend(top1_bad)
    mean = lambda xs: sum(xs) / max(1, len(xs))
    res = {
        "tag": args.tag,
        "tokens_compared": len(summary["correct"]),
        "mean_abs_dlogprob_correct": mean(summary["correct"]),
        "max_abs_dlogprob_correct": max(summary["correct"] or [0]),
        "top1_agreement_correct": mean(summary["top1"]),
        "mean_abs_dlogprob_renumbered_control": mean(summary["renumbered"]),
        "top1_agreement_renumbered_control": mean(summary["top1_renum"]),
        "calls": sum(len(r) for r in all_records),
        "evicting_calls": sum(1 for r in all_records for c in r if c["spans"]),
        "failures": FAILURES,
    }
    print("RESULT", json.dumps(res), flush=True)
    check(res["mean_abs_dlogprob_correct"] < 0.05, "logprob mismatch vs exact reference")
    check(res["top1_agreement_correct"] > 0.97, "top-1 disagreement vs exact reference")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
