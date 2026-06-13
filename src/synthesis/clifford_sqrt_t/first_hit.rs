//! First-hit pipeline: lde sweep, prefix-split search, parallel-LDE
//! speculation, and the deep-ε precision routing.

use super::*;

/// MPFR-precision column-1 of `U_L† · target` as the alignment vector
/// `v_inner` — the deep-ε replacement for the f64
/// `prefix_dag_times_target_q` → `unitary_to_uv_zeta` chain. Why: the f64
/// product's ~1e-16 error matches the radial cap width ε² at ε = 1e-8
/// and displaces the constructed cap, and no enumeration bound recovers
/// a solution the cap no longer contains. `U_L` is exact ring data and
/// `target` exact f64 data, so the product carries full `prec` bits.
pub(crate) fn prefix_residual_uv_mpfr(u_l: &U2Q, target: &Mat2, prec: u32) -> [rug::Float; 4] {
    use rug::ops::Pow;
    use rug::Float as RF;
    // ζ^i = e^{iπ/8}: cos/sin tables at prec.
    let pi = RF::with_val(prec, rug::float::Constant::Pi);
    let cosv: [RF; 8] = std::array::from_fn(|i| {
        (RF::with_val(prec, &pi * (i as u32)) / 8u32).cos()
    });
    let sinv: [RF; 8] = std::array::from_fn(|i| {
        (RF::with_val(prec, &pi * (i as u32)) / 8u32).sin()
    });
    // (re, im) of a ZZeta numerator at prec. Prefix coefficients are
    // far inside i64 at any production lde; debug-guarded.
    let zz = |z: &crate::rings::ZZeta| -> (RF, RF) {
        let mut re = RF::with_val(prec, 0.0);
        let mut im = RF::with_val(prec, 0.0);
        for i in 0..8 {
            let c = crate::synthesis::lattice::lll::i256_to_f64(z.coeff(i));
            if c != 0.0 {
                re += RF::with_val(prec, &cosv[i] * c);
                im += RF::with_val(prec, &sinv[i] * c);
            }
        }
        (re, im)
    };
    // 1/√2^k at prec.
    let scale = RF::with_val(prec, RF::with_val(prec, 2.0).sqrt().pow(u_l.k)).recip();
    // U†'s row i is [conj(U[0][i]), conj(U[1][i])]; m_inner column 1:
    // mᵢ = Σⱼ conj(U[j][i])·t[j][0]. (a − bi)(c + di) = (ac+bd) + (ad−bc)i.
    let col = |z1: (RF, RF), z2: (RF, RF)| -> (RF, RF) {
        let (a1, b1) = z1;
        let (a2, b2) = z2;
        let (c1, d1) = (target[0][0].re, target[0][0].im);
        let (c2, d2) = (target[1][0].re, target[1][0].im);
        let re = RF::with_val(prec, &a1 * c1) + RF::with_val(prec, &b1 * d1)
            + RF::with_val(prec, &a2 * c2) + RF::with_val(prec, &b2 * d2);
        let im = RF::with_val(prec, &a1 * d1) - RF::with_val(prec, &b1 * c1)
            + RF::with_val(prec, &a2 * d2) - RF::with_val(prec, &b2 * c2);
        (re, im)
    };
    let (m00_re, m00_im) = col(zz(&u_l.u11), zz(&u_l.u21));
    let (m10_re, m10_im) = col(zz(&u_l.u12), zz(&u_l.u22));
    [
        m00_re * &scale,
        m00_im * &scale,
        m10_re * &scale,
        m10_im * &scale,
    ]
}

/// Rotate the complex pairs (v[0]+i·v[1], v[2]+i·v[3]) by e^{iπj/16}
/// in MPFR — the parity-branch rotation, applied AFTER exact v
/// derivation so the odd branch's cap is built from uncorrupted
/// geometry (the scalar rotation commutes with the prefix product).
pub(crate) fn rotate_uv_by_zeta32_mpfr(v: [rug::Float; 4], j: u32, prec: u32) -> [rug::Float; 4] {
    use rug::Float as RF;
    if j == 0 {
        return v;
    }
    let ang = RF::with_val(prec, rug::float::Constant::Pi) * j / 16u32;
    let c = ang.clone().cos();
    let s = ang.sin();
    let [a, b, x, y] = v;
    [
        RF::with_val(prec, &a * &c) - RF::with_val(prec, &b * &s),
        RF::with_val(prec, &a * &s) + RF::with_val(prec, &b * &c),
        RF::with_val(prec, &x * &c) - RF::with_val(prec, &y * &s),
        RF::with_val(prec, &x * &s) + RF::with_val(prec, &y * &c),
    ]
}

/// Deep-ε-aware find_aligned_lattice_points router. At ε ≤ 2e-8 the radial cap width ε²/4
/// sits under the f64 ULP at unit scale, so an f64 y-chain corrupts Q,
/// the cap center, and the Cholesky factor — and an f64 prefix product
/// additionally displaces the cap itself ([`prefix_residual_uv_mpfr`]).
/// Those ε route through the MPFR entry with `v` derived from the most
/// exact source available; above 2e-8 the f64 path is safe and ~free.
#[allow(clippy::too_many_arguments)]
pub(crate) fn find_aligned_lattice_points_auto_prec<F>(
    scratch: &mut IntScratch16,
    v: [f64; 4],
    deep_v_src: Option<(&U2Q, &Mat2)>,
    rot_src: Option<&(Mat2, u32)>,
    k: u32,
    eps: f64,
    max_leaf_checks: u64,
    budget_hit: &std::sync::atomic::AtomicBool,
    should_stop: F,
    external_abort: Option<&std::sync::atomic::AtomicBool>,
    consumed: Option<&std::sync::atomic::AtomicU64>,
) -> Vec<[i64; 16]>
where
    F: Fn(&[i64; 16]) -> bool + Sync,
{
    if eps <= 2e-8 {
        let prec = scratch.prec_q;
        // Derive v from the most exact source available. With a
        // rot_src present, the caller's f64 `v` and `target` are the
        // ROTATED (f64-corrupted) forms — rebuild from the unrotated
        // original and rotate exactly in MPFR.
        let v_mpfr: [rug::Float; 4] = match (deep_v_src, rot_src) {
            (Some((u_l, _rotated)), Some((orig, j))) => {
                rotate_uv_by_zeta32_mpfr(prefix_residual_uv_mpfr(u_l, orig, prec), *j, prec)
            }
            (Some((u_l, target)), None) => prefix_residual_uv_mpfr(u_l, target, prec),
            (None, Some((orig, j))) => {
                let base: [rug::Float; 4] = [
                    rug::Float::with_val(prec, orig[0][0].re),
                    rug::Float::with_val(prec, orig[0][0].im),
                    rug::Float::with_val(prec, orig[1][0].re),
                    rug::Float::with_val(prec, orig[1][0].im),
                ];
                rotate_uv_by_zeta32_mpfr(base, *j, prec)
            }
            (None, None) => std::array::from_fn(|i| rug::Float::with_val(prec, v[i])),
        };
        let y_mpfr = uv_to_lattice_y_zeta_mpfr(&v_mpfr, k, prec);
        find_aligned_lattice_points_mpfr(
            scratch, &y_mpfr, &v_mpfr, k, eps, max_leaf_checks, budget_hit,
            should_stop, external_abort, consumed,
        )
    } else {
        let y = uv_to_lattice_y_zeta(v, k);
        find_aligned_lattice_points_with_stop(
            scratch, &y, k, eps, max_leaf_checks, budget_hit, should_stop,
            external_abort, consumed,
        )
    }
}

/// Two-pass leaf-budget strategy: pass 1 bails fast on doomed lde levels;
/// budget-hit lde levels are queued for pass 2 with a much larger cap.
/// Preserves completeness — a budget-hit lde is never skipped.
pub(crate) const PASS1_CAP: u64 = 100_000_000;
pub(crate) const PASS2_CAP: u64 = 4_000_000_000;

/// RAII enabler for MPFR prune verification, needed below 2e-8 where
/// the f64 partial-Euclidean prune suffers catastrophic cancellation
/// and silently drops valid candidates. Restores the prior global flag
/// on drop (even on early returns / panics) so other paths are
/// unaffected.
pub(crate) struct VerifyGuard {
    restore_to: bool,
    changed: bool,
}

impl VerifyGuard {
    pub(crate) fn enable_for(epsilon: f64) -> Self {
        use crate::synthesis::lattice_zeta::{set_verify_prune_mpfr, verify_prune_mpfr};
        let was_on = verify_prune_mpfr();
        let need = epsilon < 2e-8;
        if need && !was_on {
            set_verify_prune_mpfr(true);
        }
        VerifyGuard { restore_to: was_on, changed: need && !was_on }
    }
}

impl Drop for VerifyGuard {
    fn drop(&mut self) {
        if self.changed {
            crate::synthesis::lattice_zeta::set_verify_prune_mpfr(self.restore_to);
        }
    }
}

/// Per-prefix pass-1 leaf budget; scaled with ε since the post-LLL
/// SE region grows exponentially in lde_inner.
pub(crate) fn pass1_prefix_leaf_cap_for(epsilon: f64) -> u64 {
    if epsilon <= 1e-8 {
        100_000_000
    } else if epsilon <= 1e-7 {
        25_000_000
    } else {
        PASS1_PREFIX_LEAF_CAP
    }
}

pub(crate) fn pass2_prefix_leaf_cap_for(epsilon: f64) -> u64 {
    if epsilon <= 1e-8 {
        500_000_000
    } else if epsilon <= 1e-7 {
        50_000_000
    } else {
        PASS2_PREFIX_LEAF_CAP
    }
}

pub(crate) const PASS1_PREFIX_LEAF_CAP: u64 = 5_000_000;
pub(crate) const PASS2_PREFIX_LEAF_CAP: u64 = 10_000_000;

/// Rayon `with_min_len` for `prefix_split_search_q`'s optimize-mode
/// prefix par_iter. `0` = `usable.len() / n_threads` chunking. Do NOT
/// set `1`: per-job `map_init` scratch construction nests stolen
/// `per_prefix` frames on rayon's 2 MiB pool workers and overflows the
/// stack (coarse chunking survives because job count stays ≈ n_threads,
/// bounding the nesting). The cheap-prefix serialization issue is handled
/// by [`OPTIMAL_PREFIX_INTERLEAVE`] instead.
pub(crate) const OPTIMAL_PAR_MIN_LEN: usize = 0;

/// Transpose-interleave the cost-sorted prefix list across rayon chunks
/// (chunk j gets cost ranks j, j+t, j+2t, …). Plain `len/n_threads`
/// chunking hands all the cheapest prefixes to one chunk, serializing
/// exactly the prefixes most likely to set the incumbent; interleaving
/// runs the t cheapest in parallel first so later prefixes see maximal
/// pruning. Stack-safe, unlike `with_min_len(1)`.
pub(crate) const OPTIMAL_PREFIX_INTERLEAVE: bool = true;

/// Frontier dispatch mode: strict floor-priority pull-queue (workers
/// take the lowest-floor unstarted unit from an atomic cursor) instead
/// of pre-chunked `par_iter`. Under a deadline, chunked dispatch makes
/// the started-set a scheduling draw — the main source of anytime-cost
/// variance; the queue makes it a deterministic prefix of the floor
/// order. `CYCLOSYNTH_FRONTIER_QUEUE=0` restores chunked dispatch.
pub(crate) static FRONTIER_QUEUE_DISPATCH: std::sync::LazyLock<bool> =
    std::sync::LazyLock::new(|| {
        !matches!(std::env::var("CYCLOSYNTH_FRONTIER_QUEUE").as_deref(), Ok("0"))
    });

/// Compute `U_L† · target` as a continuous Mat2 (`U_L` exact `U2Q`,
/// `target` float `Mat2`). Mirrors the 8D helper
/// `clifford_t::prefix_dag_times_target`.
pub(crate) fn prefix_dag_times_target_q(u_l: &U2Q, target: &Mat2) -> Mat2 {
    let u_f = u_l.to_float();
    // (U_L†)[i][j] = conj(U_L[j][i])
    let ud00 = Complex64::new(u_f[0][0].re, -u_f[0][0].im);
    let ud01 = Complex64::new(u_f[1][0].re, -u_f[1][0].im);
    let ud10 = Complex64::new(u_f[0][1].re, -u_f[0][1].im);
    let ud11 = Complex64::new(u_f[1][1].re, -u_f[1][1].im);
    [
        [
            ud00 * target[0][0] + ud01 * target[1][0],
            ud00 * target[0][1] + ud01 * target[1][1],
        ],
        [
            ud10 * target[0][0] + ud11 * target[1][0],
            ud10 * target[0][1] + ud11 * target[1][1],
        ],
    ]
}


/// Options tail of [`SynthesizerQ::prefix_split_search_q`]: everything
/// beyond (target, lde, m) that only some callers set. Defaults are the
/// plain first-hit configuration.
pub(crate) struct PrefixSplitOpts<'a> {
    /// Override the configured det-phase filter (enum-grid arms pass
    /// their own); `None` = the synthesizer's filter.
    pub(crate) dr_filter_override: Option<&'a [u32]>,
    /// Per-prefix SE leaf budget.
    pub(crate) per_prefix_cap: u64,
    /// Cross-branch winner abort signal (concurrent-lde dispatch).
    pub(crate) external_abort: Option<&'a AtomicBool>,
    /// Shared consumed-nodes counter (next-lde launch trigger).
    pub(crate) consumed: Option<&'a std::sync::atomic::AtomicU64>,
    /// Force min-cost (true) or first-hit (false) reduction; `None` =
    /// the synthesizer's `optimize_cost`.
    pub(crate) cost_min_override: Option<bool>,
    /// Cross-call shared incumbent for the cost prune.
    pub(crate) shared_best_cost: Option<&'a std::sync::atomic::AtomicUsize>,
}

impl SynthesizerQ {
    /// Run windows of `parallel_lde_window` lde levels concurrently from
    /// `start_k` upward; the first find aborts in-flight peers. This drops
    /// hard-target wall from "sum of no-solution burns + find" to "find
    /// alone", paid for by thread dilution on easy targets — hence enabled
    /// only where hard targets overshoot the predicted lde. Task i > 0
    /// gates on its predecessor burning `parallel_lde_trigger_nodes`
    /// without finding (0 = launch immediately) AND on predecessor finish:
    /// a level can complete below the trigger, and a successor polling a
    /// permanently-stopped counter would deadlock.
    ///
    /// Returns `(find, pass-2 queue, unclear-below-find levels)`. The
    /// third element is conservative: a non-finding window peer below
    /// the find may have been aborted mid-walk or never launched —
    /// indistinguishable here from a clean exhaust — so every one is
    /// reported.
    pub(crate) fn parallel_lde_sweep(
        &self,
        target: &Mat2,
        m_split: u32,
        start_k: u32,
    ) -> (Option<SynthResultQ>, Vec<u32>, Vec<u32>) {
        let trace = crate::synthesis::diag::trace_enabled();
        let cross_lde_abort = AtomicBool::new(false);
        let window_size: u32 = self.parallel_lde_window.max(1);
        let trigger_nodes = self.parallel_lde_trigger_nodes;
        let pass2_queue: Mutex<Vec<u32>> = Mutex::new(Vec::new());
        let mut k_cursor = start_k;

        while k_cursor <= self.effective_max_lde()
            && !cross_lde_abort.load(Ordering::Relaxed)
        {
            let window_end = (k_cursor + window_size - 1).min(self.max_lde);
            let lde_window: Vec<u32> = (k_cursor..=window_end).collect();
            if trace {
                eprintln!("[zeta] dc m={m_split} pass1 parallel-lde window={:?} dispatching ...", lde_window);
            }
            let t_window = std::time::Instant::now();

            let consumed_counters: Vec<std::sync::Arc<std::sync::atomic::AtomicU64>> =
                (0..lde_window.len())
                    .map(|_| std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)))
                    .collect();
            let finished_flags: Vec<std::sync::Arc<AtomicBool>> = (0..lde_window.len())
                .map(|_| std::sync::Arc::new(AtomicBool::new(false)))
                .collect();
            let results: Mutex<Vec<(u32, Option<SynthResultQ>, bool)>> =
                Mutex::new(Vec::new());
            std::thread::scope(|s| {
                for (i, &k) in lde_window.iter().enumerate() {
                    let results_ref = &results;
                    let abort_ref = &cross_lde_abort;
                    let pass2_ref = &pass2_queue;
                    let my_consumed = consumed_counters[i].clone();
                    let my_finished = finished_flags[i].clone();
                    let predecessor_consumed =
                        if i > 0 { Some(consumed_counters[i - 1].clone()) } else { None };
                    let predecessor_finished =
                        if i > 0 { Some(finished_flags[i - 1].clone()) } else { None };
                    s.spawn(move || {
                        // RAII: mark finished on EVERY exit path (normal,
                        // abort, panic) so a successor's gate can never
                        // be stranded.
                        struct FinishedGuard(std::sync::Arc<AtomicBool>);
                        impl Drop for FinishedGuard {
                            fn drop(&mut self) {
                                self.0.store(true, Ordering::Release);
                            }
                        }
                        let _finished_guard = FinishedGuard(my_finished);
                        if i > 0 && trigger_nodes > 0 {
                            let pred = predecessor_consumed.as_ref().unwrap();
                            let pred_done = predecessor_finished.as_ref().unwrap();
                            loop {
                                if abort_ref.load(Ordering::Relaxed) { return; }
                                if pred.load(Ordering::Relaxed) >= trigger_nodes { break; }
                                if pred_done.load(Ordering::Acquire) { break; }
                                std::thread::sleep(std::time::Duration::from_millis(50));
                            }
                            if abort_ref.load(Ordering::Relaxed) { return; }
                        }
                        let t_k = std::time::Instant::now();
                        // Pass shared signals only when they can fire:
                        // the walker pays a contended atomic per
                        // recurse-enter if either is Some.
                        let abort_opt = if window_size > 1 { Some(abort_ref) } else { None };
                        let consumed_opt = if trigger_nodes > 0 {
                            Some(my_consumed.as_ref())
                        } else {
                            None
                        };
                        let (result, budget_hit) = self.prefix_split_search_q(
                            target, k, m_split,
                            PrefixSplitOpts {
                                dr_filter_override: None,
                                per_prefix_cap: pass1_prefix_leaf_cap_for(self.epsilon),
                                external_abort: abort_opt,
                                consumed: consumed_opt,
                                cost_min_override: None,
                                shared_best_cost: None,
                            },
                        );
                        let dt = t_k.elapsed().as_secs_f64() * 1000.0;
                        if let Some(ref r) = result {
                            abort_ref.store(true, Ordering::Relaxed);
                            if trace {
                                eprintln!("[zeta] dc lde={k:>2} m={m_split} pass1  FOUND  dist={:.3e}  t={:.0}ms  (consumed={})",
                                    r.distance, dt, my_consumed.load(Ordering::Relaxed));
                            }
                        } else if trace {
                            eprintln!("[zeta] dc lde={k:>2} m={m_split} pass1  none{}  t={:.0}ms  (consumed={})",
                                if budget_hit { " (budget hit)" } else { "" }, dt,
                                my_consumed.load(Ordering::Relaxed));
                        }
                        if result.is_none() && budget_hit {
                            pass2_ref.lock().unwrap().push(k);
                        }
                        results_ref.lock().unwrap().push((k, result, budget_hit));
                    });
                }
            });
            // Lowest-lde finder wins (minimum-circuit semantics).
            let mut found_results: Vec<(u32, SynthResultQ)> = results
                .into_inner()
                .unwrap()
                .into_iter()
                .filter_map(|(k, r, _)| r.map(|x| (k, x)))
                .collect();
            found_results.sort_by_key(|(k, _)| *k);

            if let Some((found_k, r)) = found_results.into_iter().next() {
                if trace {
                    eprintln!("[zeta] dc parallel-lde window wall  t={:.0}ms",
                        t_window.elapsed().as_secs_f64() * 1000.0);
                }
                let queue = pass2_queue.into_inner().unwrap();
                let unclear_below: Vec<u32> = queue
                    .iter()
                    .copied()
                    .chain(lde_window.iter().copied())
                    .filter(|&k| k < found_k)
                    .collect();
                return (Some(r), queue, unclear_below);
            }
            k_cursor = window_end + 1;
        }
        (None, pass2_queue.into_inner().unwrap(), Vec::new())
    }

    /// Pass-2 retries for the dc dispatcher: only levels where pass 1
    /// hit budget without finding (every other level was exhausted — no
    /// solution exists there). Returns the find plus the levels that hit
    /// budget AGAIN, which the caller reports as unclear.
    pub(crate) fn retry_budget_truncated_levels(
        &self,
        target: &Mat2,
        m_split: u32,
        mut queue: Vec<u32>,
    ) -> (Option<SynthResultQ>, Vec<u32>) {
        queue.sort_unstable();
        let trace = crate::synthesis::diag::trace_enabled();
        let mut still_truncated: Vec<u32> = Vec::new();
        for k in queue {
            if k > self.effective_max_lde() {
                break;
            }
            let t_k = std::time::Instant::now();
            if trace {
                eprintln!("[zeta] dc lde={k:>2} m={m_split} pass2 dispatching ...");
            }
            let (result, budget_hit) = self.prefix_split_search_q(
                target, k, m_split,
                PrefixSplitOpts {
                    dr_filter_override: None,
                    per_prefix_cap: pass2_prefix_leaf_cap_for(self.epsilon),
                    external_abort: None,
                    consumed: None,
                    cost_min_override: None,
                    shared_best_cost: None,
                },
            );
            if let Some(r) = result {
                if trace {
                    eprintln!("[zeta] dc lde={k:>2} m={m_split} pass2  FOUND  dist={:.3e}  t={:.0}ms",
                        r.distance, t_k.elapsed().as_secs_f64() * 1000.0);
                }
                return (Some(r), still_truncated);
            }
            if budget_hit {
                still_truncated.push(k);
            }
            if trace {
                eprintln!("[zeta] dc lde={k:>2} m={m_split} pass2  none   t={:.0}ms",
                    t_k.elapsed().as_secs_f64() * 1000.0);
            }
        }
        (None, still_truncated)
    }

    /// `max_lde` clamped by the live cross-parity incumbent when present
    /// (lde ≤ cost + 1 staircase bound); the incumbent tightens
    /// concurrently as the peer branch finds circuits.
    pub(crate) fn effective_max_lde(&self) -> u32 {
        let mut m = self.max_lde;
        if let Some(best) = &self.global_best_cost {
            let c = best.load(std::sync::atomic::Ordering::Relaxed);
            if c != usize::MAX {
                let c32 = c.min(u32::MAX as usize - 1) as u32;
                m = m.min(c32.saturating_add(1));
            }
        }
        m
    }

    /// [`Self::synthesize`] with an optional truncation out-param: a find
    /// at level `fl` short-circuits the pass-2 retry queue, so a
    /// truncated-and-never-cleared level below `fl` may still hold a
    /// solution. `unclear_out` receives exactly those levels so the
    /// cost-optimal enum stage can add them to its (lde, m) grid.
    pub(crate) fn synthesize_with_unverified_levels(
        &self,
        target: Mat2,
        mut unclear_out: Option<&mut Vec<u32>>,
    ) -> Option<SynthResultQ> {
        use crate::synthesis::diag;
        crate::synthesis::ensure_rayon_stack();

        // Land the det exactly on a ζ₁₆ power first (lossless, see
        // `project_det_to_zeta_coset`) — generic U(2) inputs otherwise
        // carry a residual phase no completion can absorb.
        let target = project_det_to_zeta_coset(&target);

        if self.optimize_cost {
            return self.synthesize_optimal(target);
        }

        let trace = diag::trace_enabled();
        if trace {
            diag::reset_all();
        }

        let _verify_guard = VerifyGuard::enable_for(self.epsilon);

        let d = det_phase_of(&target);
        let v = unitary_to_uv_zeta(&target);

        let mut scratch: Option<Box<IntScratch16>> = None;

        let lattice_start = lattice_lde_estimate(self.epsilon)
            .saturating_sub(2)
            .max(BRUTE_LIMIT + 1)
            .max(self.min_lde);

        // `should_stop` short-circuits the walker on the first ε-close
        // leaf; optimize_cost returns false unconditionally so every
        // ε-close leaf is enumerated and check_sols picks the cheapest.
        let epsilon = self.epsilon;
        let use_f64_gs = self.use_f64_gs;
        let bkz_block_size = self.bkz_block_size;
        let optimize_cost = self.optimize_cost;
        let try_lattice_k = |k: u32,
                             budget: u64,
                             scratch: &mut Option<Box<IntScratch16>>|
         -> (Vec<[i64; 16]>, bool) {
            let s = scratch
                .get_or_insert_with(|| {
                    let mut sb = Box::new(IntScratch16::new(epsilon));
                    sb.use_f64_gs = use_f64_gs;
                    sb.bkz_block_size = bkz_block_size;
                    sb
                });
            let budget_hit = AtomicBool::new(false);
            let should_stop = |x: &[i64; 16]| -> bool {
                if optimize_cost { return false; }
                let cand = solution_to_u2q_with_det_phase(x, k, d);
                diamond_distance_u2q_float(&cand, &target) < epsilon
            };
            let sols = find_aligned_lattice_points_auto_prec(
                s.as_mut(), v, None, self.deep_rot_src.as_ref(), k, epsilon, budget, &budget_hit, should_stop, None, None,
            );
            (sols, budget_hit.load(std::sync::atomic::Ordering::Relaxed))
        };

        let check_sols = |sols: &[[i64; 16]], k: u32| -> Option<SynthResultQ> {
            let cands = sols.iter().map(|sol| (solution_to_u2q_with_det_phase(sol, k, d), k));
            self.pick_min_cost_result(cands, &target, !optimize_cost).map(|(_, r)| r)
        };

        // Brute regime: iterate every k for exact small-T Clifford+√T finds.
        let zd = Complex64::from_polar(1.0, d as f64 * PI / 8.0);
        for k in self.min_lde..=BRUTE_LIMIT.min(self.max_lde) {
            let t_k = std::time::Instant::now();
            let shell = brute_shell_cached(k);
            let thr = brute_prefilter_threshold(self.epsilon);
            let close: Vec<[i64; 16]> = shell
                .sols
                .iter()
                .zip(&shell.mats)
                .filter(|(_, m)| brute_dist_est(m, zd, &target) < thr)
                .map(|(s, _)| *s)
                .collect();
            let r = check_sols(&close, k);
            if trace {
                eprintln!("[zeta] brute lde={k:>2}  sols={:>7} close={:>3}  {}  t={:.0}ms",
                    shell.sols.len(), close.len(),
                    if r.is_some() { "FOUND" } else { "none " },
                    t_k.elapsed().as_secs_f64() * 1000.0);
            }
            if let Some(r) = r {
                if trace {
                    diag::dump_zeta(&diag::snapshot(),
                        &format!("synthesize ε={:.0e} k={k}", self.epsilon));
                }
                return Some(r);
            }
        }

        // 2-pass dispatcher: pass 1 bails fast on doomed levels;
        // budget-hit levels are requeued at the pass-2 cap so a
        // budget-truncated lde is never silently skipped (min-lde
        // correctness) while easy targets stay cheap.
        if let Some(m_split) = self.prefix_split_m {
            // Budget-hit levels the pass-2 queue will never retry (it
            // covers the main sweep, not this fallback; a find aborts
            // the queue) — reported through `unclear_out`.
            let mut unverified_small: Vec<u32> = Vec::new();
            // Sequential small-k pass: prefix_split_search_q cannot help for k <= m_split
            // (lde_inner ≤ 0). These are typically few levels near lattice_start.
            for k in lattice_start..=m_split.min(self.max_lde) {
                let t_k = std::time::Instant::now();
                let (sols, small_budget_hit) = try_lattice_k(k, PASS1_CAP, &mut scratch);
                if let Some(r) = check_sols(&sols, k) {
                    if trace {
                        eprintln!("[zeta] dc lde={k:>2} (single fallback)  FOUND  dist={:.3e}  t={:.0}ms",
                            r.distance, t_k.elapsed().as_secs_f64() * 1000.0);
                    }
                    if let Some(out) = unclear_out.as_deref_mut() {
                        out.extend(unverified_small.iter().copied());
                    }
                    return Some(r);
                }
                if small_budget_hit {
                    unverified_small.push(k);
                }
                if trace {
                    eprintln!("[zeta] dc lde={k:>2} (single fallback)  none   t={:.0}ms",
                        t_k.elapsed().as_secs_f64() * 1000.0);
                }
            }

            use std::sync::Mutex;
            let pass2_collector: Mutex<Vec<u32>> = Mutex::new(Vec::new());

            // window == 1 uses a plain sequential loop: the shared
            // consumed-counter alone costs a large fraction of wall on
            // shallow-ε million-node walks, and concurrent-lde dispatch
            // only pays where no-solution levels burn seconds (deep ε).
            if self.parallel_lde_window <= 1 {
                for k in (m_split + 1).max(lattice_start)..=self.max_lde {
                    if k > self.effective_max_lde() {
                        break;
                    }
                    let t_k = std::time::Instant::now();
                    let (result, budget_hit) = self.prefix_split_search_q(
                        &target, k, m_split,
                        PrefixSplitOpts {
                            dr_filter_override: None,
                            per_prefix_cap: pass1_prefix_leaf_cap_for(self.epsilon),
                            external_abort: None,
                            consumed: None,
                            cost_min_override: None,
                            shared_best_cost: None,
                        },
                    );
                    if let Some(r) = result {
                        if trace {
                            eprintln!("[zeta] dc lde={k:>2} m={m_split} pass1  FOUND  dist={:.3e}  t={:.0}ms",
                                r.distance, t_k.elapsed().as_secs_f64() * 1000.0);
                        }
                        // Find at k short-circuits the pass-2 retries:
                        // every queued (budget-hit) level < k stays
                        // unverified — report it for the enum grid.
                        if let Some(out) = unclear_out.as_deref_mut() {
                            out.extend(unverified_small.iter().copied());
                            out.extend(pass2_collector.lock().unwrap().iter().copied());
                        }
                        return Some(r);
                    }
                    if trace {
                        eprintln!("[zeta] dc lde={k:>2} m={m_split} pass1  none{}  t={:.0}ms",
                            if budget_hit { " (budget hit)" } else { "" },
                            t_k.elapsed().as_secs_f64() * 1000.0);
                    }
                    if budget_hit { pass2_collector.lock().unwrap().push(k); }
                }
                let (found, still_truncated) = self.retry_budget_truncated_levels(
                    &target, m_split, pass2_collector.into_inner().unwrap(),
                );
                if let Some(r) = found {
                    if let Some(out) = unclear_out.as_deref_mut() {
                        out.extend(unverified_small.iter().copied());
                        out.extend(still_truncated.iter().copied());
                    }
                    return Some(r);
                }
                return None;
            }

            let start_k = (m_split + 1).max(lattice_start);
            let (found, pass2_queue, unclear_below) =
                self.parallel_lde_sweep(&target, m_split, start_k);
            if let Some(r) = found {
                if let Some(out) = unclear_out.as_deref_mut() {
                    out.extend(unverified_small.iter().copied());
                    out.extend(unclear_below);
                }
                return Some(r);
            }
            let (found, still_truncated) =
                self.retry_budget_truncated_levels(&target, m_split, pass2_queue);
            if let Some(r) = found {
                if let Some(out) = unclear_out.as_deref_mut() {
                    out.extend(unverified_small.iter().copied());
                    out.extend(still_truncated.iter().copied());
                }
                return Some(r);
            }
            return None;
        }

        // Lattice regime, Pass 1: aggressive budget cap. k's that hit the
        // budget without finding a sol get queued for Pass 2.
        let mut pass2_queue: Vec<u32> = Vec::new();
        for k in lattice_start..=self.max_lde {
            if k > self.effective_max_lde() {
                break;
            }
            let t_k = std::time::Instant::now();
            let (sols, budget_was_hit) = try_lattice_k(k, PASS1_CAP, &mut scratch);
            if let Some(r) = check_sols(&sols, k) {
                if trace {
                    eprintln!("[zeta] pass1 lde={k:>2}  FOUND  dist={:.3e}  t={:.0}ms",
                        r.distance, t_k.elapsed().as_secs_f64() * 1000.0);
                    diag::dump_zeta(&diag::snapshot(),
                        &format!("synthesize ε={:.0e} k={k} (pass1)", self.epsilon));
                }
                // Queued (budget-hit) levels < k never get their pass-2
                // retry — same upward-bias class as the dc dispatcher.
                if let Some(out) = unclear_out.as_deref_mut() {
                    out.extend(pass2_queue.iter().copied());
                }
                return Some(r);
            }
            if trace {
                eprintln!("[zeta] pass1 lde={k:>2}  none{}  t={:.0}ms",
                    if budget_was_hit { " (budget hit)" } else { "" },
                    t_k.elapsed().as_secs_f64() * 1000.0);
            }
            if budget_was_hit {
                pass2_queue.push(k);
            }
        }

        // Lattice regime, Pass 2: only retry the k's that Pass 1
        // budget-hit. Guarantees no completeness loss vs single-pass-at-
        // PASS2_CAP, while skipping k's where Pass 1 was already
        // exhaustive.
        let mut still_truncated: Vec<u32> = Vec::new();
        for k in pass2_queue {
            if k > self.effective_max_lde() {
                break;
            }
            let t_k = std::time::Instant::now();
            let (sols, budget_hit2) = try_lattice_k(k, PASS2_CAP, &mut scratch);
            if let Some(r) = check_sols(&sols, k) {
                if trace {
                    eprintln!("[zeta] pass2 lde={k:>2}  FOUND  dist={:.3e}  t={:.0}ms",
                        r.distance, t_k.elapsed().as_secs_f64() * 1000.0);
                    diag::dump_zeta(&diag::snapshot(),
                        &format!("synthesize ε={:.0e} k={k} (pass2)", self.epsilon));
                }
                if let Some(out) = unclear_out.as_deref_mut() {
                    out.extend(still_truncated.iter().copied());
                }
                return Some(r);
            }
            if budget_hit2 {
                still_truncated.push(k);
            }
            if trace {
                eprintln!("[zeta] pass2 lde={k:>2}  none   t={:.0}ms",
                    t_k.elapsed().as_secs_f64() * 1000.0);
            }
        }

        if trace {
            diag::dump_zeta(&diag::snapshot(),
                &format!("synthesize ε={:.0e} (no sol)", self.epsilon));
        }
        None
    }

    /// Z[ζ_16] analog of Clifford+T's `prefix_split_search`: for each prefix
    /// `U_L ∈ L_m^Q`, search the inner factor at `lde_total − k_prefix` and
    /// compose; `d_R = (d_target − d_L) mod 16` parametrises the inner
    /// reconstruction so `U_L · U_R` matches the target's det phase.
    /// The returned bool reports any budget-hit prefix so the 2-pass
    /// dispatcher knows a deeper retry could still find something.
    /// Prefixes run under rayon with per-worker scratch; nested SE
    /// parallelism over-subscribes the pool, which work-stealing handles.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prefix_split_search_q(
        &self,
        target: &Mat2,
        lde_total: u32,
        m_split: u32,
        opts: PrefixSplitOpts<'_>,
    ) -> (Option<SynthResultQ>, bool) {
        let PrefixSplitOpts {
            dr_filter_override,
            per_prefix_cap,
            external_abort,
            consumed,
            cost_min_override,
            shared_best_cost,
        } = opts;
        use rayon::prelude::*;
        use crate::synthesis::diag;

        let prefixes = build_fgkm_prefix_set(m_split);
        let q_cost_x2 = self.q_cost_x2;
        let prefix_costs: Vec<usize> = build_fgkm_prefix_gate_counts(m_split)
            .iter()
            .map(|&(t, q)| 2 * t + q_cost_x2 * q)
            .collect();
        let d_target = det_phase_of(target);
        let epsilon = self.epsilon;
        let use_f64_gs = self.use_f64_gs;
        let bkz_block_size = self.bkz_block_size;

        // Shared across all prefix workers: any prefix that hits its
        // SE-leaf budget without finding sets this. The 2-pass dispatcher
        // uses it to decide if a pass2 retry is warranted.
        let any_budget_hit = Arc::new(AtomicBool::new(false));

        // Pre-filter the prefixes once: drop those whose lde already
        // exceeds lde_total (lde_inner would be ≤ 0), and drop those whose
        // required d_R isn't in the allowed-offsets set. Each entry
        // carries its precomputed decomposed cost for Stage-3 ranking
        // + heuristic pruning.
        let inner_det_phase_filter: &[u32] = dr_filter_override.unwrap_or(&self.inner_det_phase_filter);
        let mut cand_idx: Vec<(usize, usize)> = prefixes
            .iter()
            .enumerate()
            .filter(|(_, u_l)| u_l.k < lde_total)
            .filter(|(_, u_l)| {
                if inner_det_phase_filter.is_empty() {
                    return true;
                }
                let d_l = det_phase_of(&u_l.to_float());
                let d_r = ((d_target as i32 - d_l as i32).rem_euclid(16)) as u32;
                inner_det_phase_filter.contains(&d_r)
            })
            .map(|(i, _)| (i, prefix_costs[i]))
            .collect();

        // Right-coset dedup: one min-cost rep per (orbit, k) class of
        // the usable set. The per-prefix cap scales by the dedup ratio
        // so the total leaf budget per orbit is invariant — without
        // that, the rep gets ONE cap-bounded draw where the orbit had
        // `ratio` independent ones, and the racy leaf-visit order can
        // flip a near-cap find to budget-hit. Exhausted walks (the
        // common no-solution case) are unaffected, preserving the wall
        // win.
        let mut per_prefix_cap = per_prefix_cap;
        if *ZETA_COSET_DEDUP && cand_idx.len() > 1 {
            let pre = cand_idx.len();
            let keys = build_fgkm_prefix_coset_keys(m_split);
            let mask = coset_keep_mask(&cand_idx, &keys);
            let mut it = mask.iter();
            cand_idx.retain(|_| *it.next().unwrap());
            let post = cand_idx.len().max(1);
            let ratio = (pre.div_ceil(post)) as u64;
            per_prefix_cap = per_prefix_cap.saturating_mul(ratio.max(1));
        }

        let mut usable: Vec<(&U2Q, usize)> = cand_idx
            .into_iter()
            .map(|(i, c)| (&prefixes[i], c))
            .collect();

        if usable.is_empty() {
            return (None, false);
        }

        let optimize_cost = cost_min_override.unwrap_or(self.optimize_cost);

        // Optimal mode sorts cheapest-first so the shared incumbent
        // drops quickly; first-hit keeps k_prefix-desc (small lde_inner =
        // fast bail or hit).
        let n_threads = rayon::current_num_threads().max(1);
        if optimize_cost {
            usable.sort_by_key(|(_, c)| *c);
            if OPTIMAL_PREFIX_INTERLEAVE {
                usable = crate::synthesis::stride_interleave(&usable, n_threads);
            }
        } else {
            usable.sort_by(|(a, _), (b, _)| b.k.cmp(&a.k));
        }

        let chunk = (usable.len() / n_threads).max(1);
        let opt_chunk = if OPTIMAL_PAR_MIN_LEN == 0 { chunk } else { OPTIMAL_PAR_MIN_LEN };

        // Node-level incumbent abort: a watcher flags in-flight prefixes
        // whose static floor (cost(U_L) + class_cost_lb(d_R)) can no
        // longer beat the incumbent, killing hopeless walks mid-tree —
        // the leaf-level check alone never fires on walks that produce
        // no ε-close leaf. Sound: only cuts walks whose every candidate
        // costs ≥ the incumbent.
        let watches: Vec<PrefixWatch> = if optimize_cost {
            usable
                .iter()
                .map(|&(u_l, c)| {
                    let d_l = det_phase_of(&u_l.to_float());
                    let d_r = ((d_target as i32 - d_l as i32).rem_euclid(16)) as u32;
                    PrefixWatch {
                        abort: AtomicBool::new(false),
                        active: AtomicBool::new(false),
                        floor: c.saturating_add(
                            crate::synthesis::cost_bound::class_cost_lb_half_units(d_r),
                        ),
                    }
                })
                .collect()
        } else {
            Vec::new()
        };

        // Shared best-cost tracker; a caller-supplied atomic lets all
        // concurrent prefix_split_search_q calls prune against one (pre-seeded)
        // global incumbent.
        let local_best_cost = std::sync::atomic::AtomicUsize::new(usize::MAX);
        let best_cost: &std::sync::atomic::AtomicUsize =
            shared_best_cost.unwrap_or(&local_best_cost);

        let per_prefix = |scratch: &mut IntScratch16,
                          idx: usize,
                          entry: &(&U2Q, usize)|
         -> Option<(usize, SynthResultQ)> {
            let (u_l, u_l_cost) = (entry.0, entry.1);
            let k_prefix = u_l.k;
            let lde_inner = lde_total - k_prefix;

            let d_l = det_phase_of(&u_l.to_float());
            let d_r = ((d_target as i32 - d_l as i32).rem_euclid(16)) as u32;

            // Heuristic prune: U_R can cancel parts of U_L, so this can in
            // principle miss the optimum.
            if optimize_cost {
                let cur_best = best_cost.load(std::sync::atomic::Ordering::Relaxed);
                // Sound because syllable costs are additive in normal
                // form: any U cheaper than `best` is reachable through
                // its canonical prefix with cost ≤ best − LB(suffix).
                // Only the det-phase Q-parity bound is a valid suffix
                // LB — L(lde_inner) is NOT: the lde_inner shell contains
                // √2-scaled images of every lower-lde suffix, which can
                // cost far less.
                let suffix_lb =
                    crate::synthesis::cost_bound::class_cost_lb_half_units(d_r);
                if u_l_cost.saturating_add(suffix_lb) > cur_best {
                    return None;
                }
            }

            let m_inner = prefix_dag_times_target_q(u_l, target);
            let v_inner = unitary_to_uv_zeta(&m_inner);

            let budget_hit = AtomicBool::new(false);
            let u_l_local = *u_l;
            let target_local = *target;
            let capture = diag::capture_enabled();
            let suffix_floor =
                crate::synthesis::cost_bound::class_cost_lb_half_units(d_r);
            let should_stop = |x: &[i64; 16]| -> bool {
                if optimize_cost {
                    // Stop the walk once the incumbent reaches this
                    // prefix's floor — only skips candidates costing ≥
                    // the incumbent (checked at leaf hits, free).
                    return best_cost.load(std::sync::atomic::Ordering::Relaxed)
                        <= u_l_cost.saturating_add(suffix_floor);
                }
                let u_r = solution_to_u2q_with_det_phase(x, lde_inner, d_r);
                let u_full = u_l_local * u_r;
                let hit = diamond_distance_u2q_float(&u_full, &target_local) < epsilon;
                if hit && capture {
                    diag::try_capture(diag::CapturedFind {
                        x_inner: *x, lde_inner, lde_total, d_r, d_l,
                    });
                }
                hit
            };

            // Optimize mode routes the walker's abort signal through this
            // prefix's own flag (set by the incumbent watcher; it also
            // mirrors `external_abort` if the caller passed one).
            // First-hit mode passes the caller's signal straight through.
            let walk_abort: Option<&AtomicBool> = if optimize_cost {
                let w = &watches[idx];
                w.active.store(true, Ordering::Relaxed);
                Some(&w.abort)
            } else {
                external_abort
            };

            let sols = find_aligned_lattice_points_auto_prec(
                scratch, v_inner, Some((u_l, target)), self.deep_rot_src.as_ref(), lde_inner, epsilon,
                per_prefix_cap, &budget_hit, should_stop,
                walk_abort, consumed,
            );
            if optimize_cost {
                watches[idx].active.store(false, Ordering::Relaxed);
            }

            if budget_hit.load(std::sync::atomic::Ordering::Relaxed) {
                any_budget_hit.store(true, std::sync::atomic::Ordering::Relaxed);
            }

            // First-hit returns the first ε-close sol; optimal keeps the
            // min-cost one and publishes it for the prefix prune.
            let mut best: Option<(usize, SynthResultQ)> = None;
            for sol in &sols {
                let u_r = solution_to_u2q_with_det_phase(sol, lde_inner, d_r);
                let u_full = u_l_local * u_r;
                let dist = diamond_distance_u2q_float(&u_full, target);
                if dist < epsilon {
                    let gates = BlochDecomposer.decompose(&u_full);
                    let cost = gates_cost(&gates, q_cost_x2);
                    let result = SynthResultQ {
                        gates: Some(gates),
                        lde: lde_total,
                        distance: dist,
                    };
                    if !optimize_cost {
                        return Some((cost, result));
                    }
                    match &best {
                        Some((bcost, _)) if *bcost <= cost => {}
                        _ => best = Some((cost, result)),
                    }
                }
            }
            if optimize_cost {
                if let Some((c, _)) = &best {
                    // Relaxed is enough: the prune is a heuristic.
                    best_cost.fetch_min(*c, std::sync::atomic::Ordering::Relaxed);
                }
            }
            best
        };

        // Boxed so the per-worker scratch lives on the heap — rayon's
        // in-place execution can run these closures on the caller's
        // (possibly small) thread stack.
        let make_scratch = || {
            let mut s = Box::new(IntScratch16::new(epsilon));
            s.use_f64_gs = use_f64_gs;
            s.bkz_block_size = bkz_block_size;
            s
        };

        let result_pair: Option<(usize, SynthResultQ)> = if optimize_cost {
            // Min-cost reduce across prefixes; the scoped watcher kills
            // walks whose floor can no longer beat the incumbent, plus
            // everything once a cross-branch peer wins.
            with_incumbent_watcher(
                &watches,
                best_cost,
                || external_abort.map(|a| a.load(Ordering::Relaxed)).unwrap_or(false),
                |_| {},
                || {
                    usable
                        .par_iter()
                        .enumerate()
                        .with_min_len(opt_chunk)
                        .map_init(make_scratch, |s, (i, e)| per_prefix(s, i, e))
                        .reduce(
                            || None,
                            |a, b| match (a, b) {
                                (None, x) | (x, None) => x,
                                (Some(a), Some(b)) => Some(if a.0 <= b.0 { a } else { b }),
                            },
                        )
                },
            )
        } else {
            // First-hit: abort other prefixes as soon as one finds.
            usable
                .par_iter()
                .enumerate()
                .with_min_len(chunk)
                .map_init(make_scratch, |s, (i, e)| per_prefix(s, i, e))
                .find_map_any(|x| x)
        };
        let result = result_pair.map(|(_, r)| r);

        let budget_hit = any_budget_hit.load(std::sync::atomic::Ordering::Relaxed);
        (result, budget_hit)
    }

}
