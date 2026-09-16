//! Every way a rank leaves bootstrap without keeping its graft: fail, discard and gap retry.

use std::collections::{HashMap, VecDeque};
use std::time::Instant;

use tracing::{debug, warn};

use super::{apply_batch, LateJoin, ObligationBatch, PumpState};
use crate::state::kv_events::bootstrap::{BootstrapState, RankOutcome};
use crate::state::kv_events::tree::KvWorkerId;
use crate::state::kv_events::wire::KvEventBatch;

/// Give up on bootstrapping `rank`: release whatever it held and mark it
/// [`BootstrapState::Failed`].
///
/// `discard_held` is for a queue whose contents describe a cache that no
/// longer exists (a publisher reset): drop it and resume from the live stream.
/// Otherwise the intact queue is replayed as live deltas.
pub(super) fn fail_rank(
    st: &PumpState<'_>,
    held: &mut HashMap<KvWorkerId, VecDeque<(i64, KvEventBatch)>>,
    rank: &KvWorkerId,
    discard_held: bool,
    outcome: RankOutcome,
) {
    // Only transition ranks that are actually mid-bootstrap; an
    // AbandonBootstrap racing a successful ApplySnapshot must not undo it.
    //
    // This gate is also what keeps the rank tally once-per-rank: a second
    // attempt to fail an already-resolved rank returns before recording.
    if st.bootstrap.state_of(rank) != Some(BootstrapState::Pending) {
        held.remove(rank);
        return;
    }
    discard_graft(st, rank, outcome);
    let queue = held.remove(rank).unwrap_or_default();
    if discard_held {
        return;
    }
    for (seq, batch) in queue {
        apply_batch(st.tree, st.cursors, st.tally, rank, seq, &batch);
    }
}

/// Mark `rank` [`BootstrapState::Failed`], tally `outcome`, and drop
/// everything a snapshot contributed for it: tree carriers and cursor.
fn discard_graft(st: &PumpState<'_>, rank: &KvWorkerId, outcome: RankOutcome) {
    st.bootstrap.set(rank, BootstrapState::Failed);
    st.bootstrap.record_rank_outcome(outcome);
    st.tree.clear_worker(rank);
    st.cursors.lock().remove(rank);
}

/// Discard a graft whose live stream did not join its watermark, then hand the
/// rank back for one more sweep, or fail it cold when no retry is allowed.
///
/// A gap is the costliest failure: a snapshot was fetched, grafted, then thrown
/// away. A fresher snapshot usually splices, so the tracker allows one retry per
/// rank.
///
/// PRECONDITION: every live batch the rank has received is in `held` and none
/// has been applied. That is what lets a retry start from exactly the shape a
/// first attempt does — no tree state, no cursor, every delta held. Applying
/// the queue first would leave live deltas UNDER the retry's graft, whose seeded
/// cursor then filters their later removals as already reflected (a permanent
/// false hit on the success path), and any failure path's `clear_worker` would
/// wipe them.
///
/// The verdict is tallied only when final: a granted retry records nothing
/// here, and the retry's own resolution records the rank's one `RankOutcome`.
///
/// The retry is stamped now and marked [`LateJoin::Refused`], so only a sweep
/// that asks for an export newer than the gap can take it.
pub(super) fn resolve_gap(
    st: &PumpState<'_>,
    held: &mut HashMap<KvWorkerId, VecDeque<(i64, KvEventBatch)>>,
    rank: &KvWorkerId,
) {
    // `Failed` is the state a gap leaves, and the only one `retry_after_gap`
    // grants from. Set immediately before it, so `/readyz` has no room to read
    // a retried rank as terminal.
    st.bootstrap.set(rank, BootstrapState::Failed);
    if let Some(obligation) = st.bootstrap.retry_after_gap(rank) {
        let batch = ObligationBatch {
            obligations: vec![obligation],
            holding_since: Instant::now(),
            late_join: LateJoin::Refused,
        };
        match st.bootstrap_tx.try_send(batch) {
            Ok(()) => {
                // The discarded graft goes, untallied; `held` stays for the
                // retry's graft.
                st.tree.clear_worker(rank);
                st.cursors.lock().remove(rank);
                debug!(worker = ?rank, "kv-bootstrap: gapped rank re-queued for another sweep");
                return;
            }
            // A `Pending` rank nobody owns holds its batches until the
            // per-rank cap overflows, so fail it cold instead.
            Err(e) => {
                warn!("kv-bootstrap: could not re-queue gapped rank ({e}); leaving it cold");
            }
        }
    }
    discard_graft(st, rank, RankOutcome::Gap);
    for (seq, batch) in held.remove(rank).unwrap_or_default() {
        apply_batch(st.tree, st.cursors, st.tally, rank, seq, &batch);
    }
}

/// Resolve a gap a peer witnessed for a grafted rank that never proved its
/// splice, through the same [`resolve_gap`] the first-batch check uses.
///
/// Not [`fail_rank`]: that one only acts on a rank still
/// [`BootstrapState::Pending`], and this rank is `Recovered` — it was grafted
/// and is serving, so it holds no batches.
pub(super) fn demote_unproven_rank(
    st: &PumpState<'_>,
    held: &mut HashMap<KvWorkerId, VecDeque<(i64, KvEventBatch)>>,
    rank: &KvWorkerId,
) {
    // Only a live, still-`Recovered` rank is ours to demote.
    if st.bootstrap.state_of(rank) != Some(BootstrapState::Recovered)
        || !st.live_workers.lock().contains(rank)
    {
        return;
    }
    resolve_gap(st, held, rank);
}

#[cfg(test)]
mod tests {
    use super::super::test_support::*;
    use super::super::*;
    use crate::state::kv_events::wire::BlockRemoved;

    /// A hole between the snapshot watermark and the live stream means a delta
    /// was lost. Grafting anyway could leave a permanently stale entry, so the
    /// rank drops to cold: snapshot state cleared, live deltas still applied.
    ///
    /// This is the *deferred* path — the snapshot is grafted before any batch
    /// for the rank has arrived, so continuity can only be judged later. The
    /// pump's biased select drains the control channel first, which makes this
    /// the ordering that occurs naturally.
    #[tokio::test]
    async fn pump_detects_deferred_sequence_gap_and_runs_cold() {
        let id = worker_id("http://w1", 0);
        let tracker = pending_tracker(std::slice::from_ref(&id));
        let h = spawn_pump_with_bootstrap(std::slice::from_ref(&id), tracker.clone());

        // Watermark 5, but the first live batch is seq 9 — 6..8 were lost.
        h.tx.send(WorkerEvent::Batch {
            worker: id.clone(),
            seq: 9,
            batch: batch(vec![stored(None, vec![42])]),
        })
        .await
        .unwrap();
        h.ctrl_tx
            .send(PumpControl::ApplySnapshot {
                obligations: obligations(&tracker, std::slice::from_ref(&id)),
                vetted: Box::new(vetted_for(&id, 5)),
            })
            .await
            .unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        // A gapped rank is handed back for its one retry, so it is Pending.
        assert_eq!(tracker.state_of(&id), Some(BootstrapState::Pending));
        assert!(
            !h.tree
                .match_prefix(None, &[100, 200])
                .workers()
                .contains(&id),
            "snapshot state must be discarded for a gapped rank",
        );
        // The live batch is held for the retry's graft, like any Pending rank's;
        // `abandoned_gap_retry_replays_every_held_batch` covers its release.
        assert_eq!(h.tree.match_prefix(None, &[42]).matched_blocks, 0);
        assert!(h.cursors.lock().get(&id).is_none());
        // Readiness waits: the gapped rank is Pending for its retry, so the
        // tracker is unsettled until that resolves or the deadline expires.
        // Warming the tree is preferred over opening `/readyz` on a cold rank.
        assert!(!tracker.settled());
    }

    /// A gap is the costliest failure — a snapshot was fetched, grafted, then
    /// thrown away. So the rank is handed back for one more sweep instead of
    /// staying cold with budget unspent. The one-retry cap is covered by
    /// `a_second_gap_fails_the_rank_cold_and_replays_its_batches`.
    #[tokio::test]
    async fn pump_requeues_a_gapped_rank_once() {
        let id = worker_id("http://w1", 0);
        let tracker = pending_tracker(std::slice::from_ref(&id));
        let mut h = spawn_pump_with_bootstrap(std::slice::from_ref(&id), tracker.clone());

        // Watermark 5, first live batch seq 9 — 6..8 lost, so this gaps.
        h.tx.send(WorkerEvent::Batch {
            worker: id.clone(),
            seq: 9,
            batch: batch(vec![stored(None, vec![42])]),
        })
        .await
        .unwrap();
        h.ctrl_tx
            .send(PumpControl::ApplySnapshot {
                obligations: obligations(&tracker, std::slice::from_ref(&id)),
                vetted: Box::new(vetted_for(&id, 5)),
            })
            .await
            .unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        let requeued = h.bootstrap_rx.try_recv().expect("gapped rank re-queued");
        assert_eq!(requeued.obligations.len(), 1);
        assert_eq!(
            requeued.obligations[0].0, id,
            "the gapped rank itself is handed back",
        );
        assert_eq!(
            requeued.late_join,
            LateJoin::Refused,
            "a retry must not be spent on a snapshot fetched before the gap",
        );
        assert_eq!(
            tracker.state_of(&id),
            Some(BootstrapState::Pending),
            "back to Pending so the next sweep may graft onto it",
        );
    }

    /// A retried rank must reach its second graft in the same shape as its
    /// first: nothing applied, every delta held. If the first attempt's held
    /// batches were applied before the retry, the retry's graft lands on top of
    /// them and its seeded cursor then filters their later removals as already
    /// reflected — so a block the engine has evicted stays attributed to the
    /// rank for good.
    #[tokio::test]
    async fn gap_retry_graft_does_not_resurrect_a_block_the_stream_removed() {
        let id = worker_id("http://w1", 0);
        let tracker = pending_tracker(std::slice::from_ref(&id));
        let mut h = spawn_pump_with_bootstrap(std::slice::from_ref(&id), tracker.clone());

        // seq 7 stores X. Held, because the rank is Pending.
        h.tx.send(WorkerEvent::Batch {
            worker: id.clone(),
            seq: 7,
            batch: batch(vec![stored(None, vec![500])]),
        })
        .await
        .unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;
        // First graft watermarked 5: seq 6 is missing, so this gaps and retries.
        h.ctrl_tx
            .send(PumpControl::ApplySnapshot {
                obligations: obligations(&tracker, std::slice::from_ref(&id)),
                vetted: Box::new(vetted_for(&id, 5)),
            })
            .await
            .unwrap();
        let retry = h.bootstrap_rx.recv().await.expect("gapped rank re-queued");

        // seq 8 evicts X while the retry is in flight.
        h.tx.send(WorkerEvent::Batch {
            worker: id.clone(),
            seq: 8,
            batch: batch(vec![KvCacheEvent::BlockRemoved(BlockRemoved {
                block_hashes: vec![500],
                medium: None,
            })]),
        })
        .await
        .unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;
        // The retry's peer had applied both, so its export no longer holds X.
        h.ctrl_tx
            .send(PumpControl::ApplySnapshot {
                obligations: retry.obligations,
                vetted: Box::new(vetted_for(&id, 8)),
            })
            .await
            .unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert_eq!(tracker.state_of(&id), Some(BootstrapState::Recovered));
        assert!(
            !h.tree.match_prefix(None, &[500]).workers().contains(&id),
            "the engine evicted X at seq 8; the rank must not still own it",
        );
        assert_eq!(rank_count(&tracker, "warm"), 1);
        assert_eq!(
            rank_count(&tracker, "gap"),
            0,
            "a retried gap is not a verdict; the retry's outcome is the rank's one count",
        );
    }

    /// A retry that fails must not cost the rank the deltas it held before it:
    /// abandoning replays every batch the rank has received, both attempts'.
    #[tokio::test]
    async fn abandoned_gap_retry_replays_every_held_batch() {
        let id = worker_id("http://w1", 0);
        let tracker = pending_tracker(std::slice::from_ref(&id));
        let mut h = spawn_pump_with_bootstrap(std::slice::from_ref(&id), tracker.clone());

        h.tx.send(WorkerEvent::Batch {
            worker: id.clone(),
            seq: 7,
            batch: batch(vec![stored(None, vec![500])]),
        })
        .await
        .unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;
        h.ctrl_tx
            .send(PumpControl::ApplySnapshot {
                obligations: obligations(&tracker, std::slice::from_ref(&id)),
                vetted: Box::new(vetted_for(&id, 5)),
            })
            .await
            .unwrap();
        let retry = h.bootstrap_rx.recv().await.expect("gapped rank re-queued");
        h.tx.send(WorkerEvent::Batch {
            worker: id.clone(),
            seq: 8,
            batch: batch(vec![stored(None, vec![600])]),
        })
        .await
        .unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;
        h.ctrl_tx
            .send(PumpControl::AbandonBootstrap {
                obligations: retry.obligations,
            })
            .await
            .unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert_eq!(tracker.state_of(&id), Some(BootstrapState::Failed));
        for block in [500, 600] {
            assert!(
                h.tree.match_prefix(None, &[block]).workers().contains(&id),
                "live block {block} must survive the retry's abandonment",
            );
        }
        assert_eq!(h.cursors.lock().get(&id).copied(), Some(8));
        assert_eq!(rank_count(&tracker, "abandoned"), 1);
        assert_eq!(rank_count(&tracker, "gap"), 0);
    }

    /// Same gap, detected on the *immediate* path: the batch is already held
    /// when the snapshot arrives, so the watermark can be checked at graft time.
    #[tokio::test]
    async fn pump_detects_held_queue_sequence_gap_and_runs_cold() {
        let id = worker_id("http://w1", 0);
        let tracker = pending_tracker(std::slice::from_ref(&id));
        let h = spawn_pump_with_bootstrap(std::slice::from_ref(&id), tracker.clone());

        h.tx.send(WorkerEvent::Batch {
            worker: id.clone(),
            seq: 9,
            batch: batch(vec![stored(None, vec![42])]),
        })
        .await
        .unwrap();
        // Let the pump receive and hold the batch before the snapshot lands.
        // The pump's biased select would otherwise take the control message
        // first, which is the deferred path covered by the test above.
        tokio::time::sleep(Duration::from_millis(50)).await;

        h.ctrl_tx
            .send(PumpControl::ApplySnapshot {
                obligations: obligations(&tracker, std::slice::from_ref(&id)),
                vetted: Box::new(vetted_for(&id, 5)),
            })
            .await
            .unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert_eq!(tracker.state_of(&id), Some(BootstrapState::Pending));
        assert!(
            !h.tree
                .match_prefix(None, &[100, 200])
                .workers()
                .contains(&id),
            "snapshot state must be discarded for a gapped rank",
        );
        // Still held for the retry's graft, not replayed under it.
        assert_eq!(h.tree.match_prefix(None, &[42]).matched_blocks, 0);
    }

    /// With the retry already spent, a gap resolves cold: graft discarded, live
    /// deltas replayed, one `gap`.
    #[tokio::test]
    async fn a_second_gap_fails_the_rank_cold_and_replays_its_batches() {
        let id = worker_id("http://w1", 0);
        let tracker = pending_tracker(std::slice::from_ref(&id));
        // Spend the one retry up front.
        tracker.set(&id, BootstrapState::Failed);
        let retried = tracker
            .retry_after_gap(&id)
            .expect("first retry is granted");
        let mut h = spawn_pump_with_bootstrap(std::slice::from_ref(&id), tracker.clone());

        h.tx.send(WorkerEvent::Batch {
            worker: id.clone(),
            seq: 9,
            batch: batch(vec![stored(None, vec![42])]),
        })
        .await
        .unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;
        h.ctrl_tx
            .send(PumpControl::ApplySnapshot {
                obligations: vec![retried],
                vetted: Box::new(vetted_for(&id, 5)),
            })
            .await
            .unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert_eq!(tracker.state_of(&id), Some(BootstrapState::Failed));
        assert!(h.bootstrap_rx.try_recv().is_err(), "no second retry");
        assert!(!h
            .tree
            .match_prefix(None, &[100, 200])
            .workers()
            .contains(&id));
        assert!(h.tree.match_prefix(None, &[42]).workers().contains(&id));
        assert_eq!(rank_count(&tracker, "gap"), 1);
    }

    /// No peer could supply a snapshot: held deltas must still be released, or
    /// the rank would buffer forever and never settle.
    #[tokio::test]
    async fn pump_abandon_bootstrap_releases_held_batches() {
        let id = worker_id("http://w1", 0);
        let tracker = pending_tracker(std::slice::from_ref(&id));
        let h = spawn_pump_with_bootstrap(std::slice::from_ref(&id), tracker.clone());

        h.tx.send(WorkerEvent::Batch {
            worker: id.clone(),
            seq: 3,
            batch: batch(vec![stored(None, vec![55, 66])]),
        })
        .await
        .unwrap();
        // Held first, so it is the abandon's replay that releases it.
        tokio::time::sleep(Duration::from_millis(50)).await;
        h.ctrl_tx
            .send(PumpControl::AbandonBootstrap {
                obligations: obligations(&tracker, std::slice::from_ref(&id)),
            })
            .await
            .unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert_eq!(tracker.state_of(&id), Some(BootstrapState::Failed));
        let m = h.tree.match_prefix(None, &[55, 66]);
        assert_eq!(
            m.matched_blocks, 2,
            "held batches must be released, not lost"
        );
        assert!(m.workers().contains(&id));
        assert_eq!(h.cursors.lock().get(&id).copied(), Some(3));
        assert!(tracker.settled());
    }

    /// Overflowing the hold-back queue abandons bootstrap, and the rank resumes
    /// live WITHOUT losing what it held: the queue reached the cap intact, so
    /// discarding it would drop every block stored while waiting — blocks the
    /// engine never re-announces.
    #[tokio::test]
    async fn pump_overflowing_held_queue_abandons_bootstrap() {
        let id = worker_id("http://w1", 0);
        let tracker = pending_tracker(std::slice::from_ref(&id));
        let h = spawn_pump_with_bootstrap(std::slice::from_ref(&id), tracker.clone());

        for seq in 1..=(PENDING_BATCH_LIMIT as i64 + 2) {
            h.tx.send(WorkerEvent::Batch {
                worker: id.clone(),
                seq,
                batch: batch(vec![stored(None, vec![seq])]),
            })
            .await
            .unwrap();
        }
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert_eq!(tracker.state_of(&id), Some(BootstrapState::Failed));
        // The held prefix is replayed, then the batch that tripped the limit
        // and everything after it apply directly.
        let last = PENDING_BATCH_LIMIT as i64 + 2;
        for seq in [1, PENDING_BATCH_LIMIT as i64, last] {
            assert!(
                h.tree.match_prefix(None, &[seq]).workers().contains(&id),
                "batch {seq} must be applied across the overflow",
            );
        }
        assert_eq!(h.cursors.lock().get(&id).copied(), Some(last));
        assert_eq!(
            h.tally.batches_lost(),
            0,
            "the replay is contiguous, so no sequence gap may be recorded",
        );
        assert!(tracker.settled());
    }

    /// A reset while still Pending must bail the rank to cold.
    ///
    /// Post-reset seq 1 can never exceed the peer's watermark, so the gap check
    /// passes trivially, the stale tree is kept, and every real delta is filtered.
    #[tokio::test]
    async fn pump_publisher_reset_while_pending_abandons_bootstrap() {
        let id = worker_id("http://w1", 0);
        let tracker = pending_tracker(std::slice::from_ref(&id));
        let h = spawn_pump_with_bootstrap(std::slice::from_ref(&id), tracker.clone());

        h.tx.send(WorkerEvent::Batch {
            worker: id.clone(),
            seq: 900,
            batch: batch(vec![stored(None, vec![33])]),
        })
        .await
        .unwrap();
        h.tx.send(WorkerEvent::PublisherReset { worker: id.clone() })
            .await
            .unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert_eq!(
            tracker.state_of(&id),
            Some(BootstrapState::Failed),
            "a reset mid-bootstrap must abandon, not wait for a snapshot it can no \
             longer splice",
        );
        assert!(tracker.settled());
    }

    /// A terminal rank is immutable. An Abandon arriving after a successful
    /// graft must not wipe the rank's tree and demote it to Failed.
    #[tokio::test]
    async fn pump_abandon_after_successful_graft_does_not_undo_it() {
        let id = worker_id("http://w1", 0);
        let tracker = pending_tracker(std::slice::from_ref(&id));
        let obs = obligations(&tracker, std::slice::from_ref(&id));
        let h = spawn_pump_with_bootstrap(std::slice::from_ref(&id), tracker.clone());

        h.ctrl_tx
            .send(PumpControl::ApplySnapshot {
                obligations: obs.clone(),
                vetted: Box::new(vetted_for(&id, 5)),
            })
            .await
            .unwrap();
        // A second bootstrap task for the same obligation set gives up.
        h.ctrl_tx
            .send(PumpControl::AbandonBootstrap { obligations: obs })
            .await
            .unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert_eq!(
            tracker.state_of(&id),
            Some(BootstrapState::Recovered),
            "a late Abandon must not demote an already-Recovered rank",
        );
        assert!(
            h.tree
                .match_prefix(None, &[100, 200])
                .workers()
                .contains(&id),
            "a late Abandon must not wipe a grafted tree",
        );
        assert_eq!(h.cursors.lock().get(&id).copied(), Some(5));
    }

    /// The stale-incarnation guard on the ABANDON arm, mirroring the one
    /// already covered on the ApplySnapshot arm.
    #[tokio::test]
    async fn pump_abandon_from_a_stale_incarnation_is_ignored() {
        let id = worker_id("http://w1", 0);
        let tracker = pending_tracker(std::slice::from_ref(&id));
        let stale = obligations(&tracker, std::slice::from_ref(&id));
        let h = spawn_pump_with_bootstrap(std::slice::from_ref(&id), tracker.clone());

        // Remove + re-add: a new incarnation, Pending again.
        tracker.forget(std::slice::from_ref(&id));
        let fresh = tracker.register(std::slice::from_ref(&id));
        assert_ne!(stale[0].1, fresh[0].1);

        // The previous incarnation's task gives up.
        h.ctrl_tx
            .send(PumpControl::AbandonBootstrap { obligations: stale })
            .await
            .unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert_eq!(
            tracker.state_of(&id),
            Some(BootstrapState::Pending),
            "a stale Abandon must not force the new incarnation cold",
        );
    }
}
