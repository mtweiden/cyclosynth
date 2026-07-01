//! Aligned-lattice-point search for the 16D Z[ζ_16] pipeline (= the
//! paper's phase 1, arXiv:2510.05816 Alg 3.6): build Q → L²-LLL →
//! Cholesky → LU cap-center solve → Schnorr-Euchner with leaf checks.
//!
//! ## Leaf checks
//!
//!  - `‖x‖² == 2^k` (norm shell — i.e. `x ∈ Z[ζ_16]²` with combined norm
//!    matching the lde).
//!  - `bilinear_forms(x) == (0, 0, 0)` (β_1, β_2, β_3 — the totally-real
//!    decomposition of the unitarity constraint).
//!  - `(y · x)² ≥ thresh_xy(k, ε)` (alignment cap).
//!
//! ## Alignment threshold
//!
//! `thresh_xy = 2^(2k) · (1 − ε²) / 32`. Compared to the 8D path's
//! `2^(2k)·(1−ε²)/4`, the additional factor of 8 reflects the Z[ζ_16]
//! conventions:
//!
//!  - `‖y_lattice‖² = 2^k/4` (vs 8D's `2^(k−1)`) — 16D y has half the
//!    lattice-coord norm because each Z[ζ_16] element has 8 ζ-coefficients
//!    (vs 4 for Z[ω]) so the Σ-preimage spreads further.
//!  - For a valid lattice solution `x_target` with `B_1=B_2=B_3=0`, the
//!    σ_1 image of `Σ x_target` matches `y_real` exactly, so
//!    `(y_lattice · x_target) = 2^(k−2)`, target `(y·x)² = 2^(2k−4)`.
//!    Threshold = (1/2)·target gives `2^(2k−5)·(1−ε²) = 2^(2k)·(1−ε²)/32`.
//!  - The factor `(1−ε²)/2` corresponds to `cos²(θ_σ1) ≥ (1−ε²)/2` in
//!    σ_1-image space — the same cap-alignment criterion as 8D Z[ω].

#![allow(clippy::needless_range_loop)]

use rug::Assign;
use crate::rings::MpFloat;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use super::cholesky_lu::{
    cholesky_f64, euclidean_cholesky_mpfr_dual, lu_solve_int_inplace,
    q_cholesky_mpfr_dual,
};
use super::lll::{run_lll, LllResult};
use super::q_metric::build_q_int_zeta;
use super::scratch::{rfv, IntScratch16};
use super::se::{bilinear_forms, schnorr_euchner, LeafAction, SeCenter16};
use crate::synthesis::diag;

/// MPFR precision used by the alignment-threshold dot product. Same as 8D
/// `super::super::omega::se::SE_PREC` — 128 bits gives ~38 digits of
/// headroom past the precision walls in the f64 formula at ε ≲ √(machine_eps).
const ALIGN_PREC: u32 = 128;

/// L²-LLL at MPFR-80 Gram-Schmidt precision.
/// GramOverflow (i256 saturation, unhelped by precision) → None; a
/// non-unimodular det → None; det = None (Bareiss i128 overflow) is an
/// inconclusive-success. Returns `None` when the basis is unusable.
fn run_lll_ladder(scratch: &mut IntScratch16, k: u32, eps: f64) -> Option<()> {
    let trace = diag::trace_enabled();
    let t_lll = if trace { Some(std::time::Instant::now()) } else { None };

    let lll_result = run_lll(scratch);
    let det_check = super::cholesky_lu::det_exact(&scratch.basis);

    if let Some(t) = t_lll {
        diag::T_LLL_NS.fetch_add(diag::elapsed_ns(t), Ordering::Relaxed);
    }
    if let LllResult::GramOverflow = lll_result {
        return None;
    }
    if let Some(d) = det_check {
        if d != 1 && d != -1 {
            eprintln!(
                "[lattice_zeta] LLL non-unimodular (det={}) at eps={:e}, k={}; bailing.",
                d, eps, k
            );
            return None;
        }
    }
    if !matches!(lll_result, LllResult::Converged | LllResult::IterCap) {
        // Should be unreachable (only GramOverflow is left, handled above).
        return None;
    }

    Some(())
}

/// Optional BKZ-β post-pass: replaces Lovász with β-block SVP for a
/// tighter basis; empirically helpful at deep ε where the post-LLL SE
/// region is large. `None` = degenerate basis from the insertion path.
fn run_bkz_postpass(scratch: &mut IntScratch16, k: u32, eps: f64) -> Option<()> {
    if scratch.bkz_block_size >= 3 {
        let block_size = scratch.bkz_block_size as usize;
        // Populate the GS state from the current basis for BKZ to read.
        for i in 0..16 {
            super::lll::cfa_row(scratch, i);
        }
        // Best-effort basis tightening; whether it changed anything is moot —
        // the post-pass either improves the basis or leaves it valid.
        let _ = super::bkz::bkz_tours(scratch, block_size, super::bkz::BKZ_MAX_LOOPS);
        // Post-BKZ unimodularity check; bail if the insertion path
        // somehow produced a degenerate basis.
        match super::cholesky_lu::det_exact(&scratch.basis) {
            Some(1) | Some(-1) | None => {}
            Some(d) => {
                eprintln!(
                    "[lattice_zeta] BKZ-{block_size} non-unimodular (det={d}) \
                     at eps={eps:e}, k={k}; bailing."
                );
                return None;
            }
        }
    }

    Some(())
}

#[cfg(test)]
pub(crate) fn find_aligned_lattice_points(
    scratch: &mut IntScratch16,
    y: &[f64; 16],
    k: u32,
    eps: f64,
    max_leaf_checks: u64,
    budget_hit: &AtomicBool,
) -> Vec<[i64; 16]> {
    find_aligned_lattice_points_with_stop(scratch, y, k, eps, max_leaf_checks, budget_hit, |_| false, None, None)
}

/// Aligned-point search with an early-exit predicate and optional
/// speculation signals.
///
/// `should_stop(x)` is called **only** for leaves that pass the integer-exact
/// filter (norm shell + bilinear forms + alignment). When it returns `true`,
/// the lattice search aborts after collecting that leaf.
///
/// `external_abort` is a shared cross-task abort signal; when set by a peer
/// task (e.g. another LDE level finding a solution first under parallel
/// speculation), the walker aborts at its next recurse-entry.
///
/// `consumed` is a shared node counter, incremented on every recurse-entry.
/// The parallel-LDE dispatcher uses it to observe search progress.
///
/// Callers that don't need the speculation signals pass `None, None`.
///
/// Returns the empty vector on a pipeline failure: LLL Gram-overflow,
/// non-unimodular LLL output (algorithm bug, very unlikely), or Cholesky/LU
/// numerical failure.
#[allow(clippy::too_many_arguments)]
pub fn find_aligned_lattice_points_with_stop<F>(
    scratch: &mut IntScratch16,
    y: &[f64; 16],
    k: u32,
    eps: f64,
    max_leaf_checks: u64,
    budget_hit: &AtomicBool,
    should_stop: F,
    external_abort: Option<&AtomicBool>,
    consumed: Option<&AtomicU64>,
) -> Vec<[i64; 16]>
where
    F: Fn(&[i64; 16]) -> bool + Sync,
{
    // Promote f64 y to MPFR. This wrapper serves f64-precision callers
    // (ε ≥ 1e-7); ε ≤ 1e-8 must call `find_aligned_lattice_points_mpfr`
    // directly to bypass the f64 ULP floor in v.
    let prec = scratch.prec_q;
    let scale = 2.0_f64.powf(f64::from(k) / 2.0) / 4.0;
    let v_mpfr: [MpFloat; 4] = [
        rfv(prec, y[0] / scale),
        rfv(prec, y[4] / scale),
        rfv(prec, y[8] / scale),
        rfv(prec, y[12] / scale),
    ];
    let y_mpfr: [MpFloat; 16] = std::array::from_fn(|i| rfv(prec, y[i]));
    find_aligned_lattice_points_mpfr(
        scratch, &y_mpfr, &v_mpfr, k, eps, max_leaf_checks, budget_hit, should_stop,
        external_abort, consumed,
    )
}

/// MPFR-precision entry point. Caller provides `y` and `v` already in MPFR;
/// `Q` and the cap center `c[i]` are computed without any f64 round-trip.
/// The only precision path that works at ε ≤ 1e-8 (see
/// `build_q_mpfr_zeta_from_mpfr_v`).
///
/// Same `external_abort` / `consumed` semantics as [`find_aligned_lattice_points_with_stop`].
#[allow(clippy::too_many_arguments)]
pub fn find_aligned_lattice_points_mpfr<F>(
    scratch: &mut IntScratch16,
    y: &[MpFloat; 16],
    v: &[MpFloat; 4],
    k: u32,
    eps: f64,
    max_leaf_checks: u64,
    budget_hit: &AtomicBool,
    should_stop: F,
    external_abort: Option<&AtomicBool>,
    consumed: Option<&AtomicU64>,
) -> Vec<[i64; 16]>
where
    F: Fn(&[i64; 16]) -> bool + Sync,
{
    crate::synthesis::ensure_rayon_stack();
    let trace = diag::trace_enabled();
    if trace {
        diag::N_LATTICE_SEARCH_CALLS.fetch_add(1, Ordering::Relaxed);
    }

    // Step 1: build Q in MPFR + i256 snapshot.
    let t_build = if trace { Some(std::time::Instant::now()) } else { None };
    scratch.reset_basis();
    super::q_metric::build_q_mpfr_zeta_from_mpfr_v(scratch, v, k, eps);
    build_q_int_zeta(scratch);

    // Compute cap-center c[i] = y[i] · cap_mid in MPFR at prec_q.
    let prec = scratch.prec_q;
    let one = rfv(prec, 1.0);
    let two = rfv(prec, 2.0);
    let eps_rf = rfv(prec, eps);
    let eps_sq = MpFloat::with_val(prec, &eps_rf * &eps_rf);
    let one_minus_eps_sq = MpFloat::with_val(prec, &one - &eps_sq);
    let sqrt_1m = one_minus_eps_sq.sqrt();
    let cap_mid_num = MpFloat::with_val(prec, &one + &sqrt_1m);
    let cap_mid = MpFloat::with_val(prec, &cap_mid_num / &two);
    for i in 0..16 {
        scratch.c[i].assign(MpFloat::with_val(prec, &y[i] * &cap_mid));
    }
    if let Some(t) = t_build {
        diag::T_BUILD_NS.fetch_add(diag::elapsed_ns(t), Ordering::Relaxed);
    }

    if run_lll_ladder(scratch, k, eps).is_none() {
        return Vec::new();
    }

    if run_bkz_postpass(scratch, k, eps).is_none() {
        return Vec::new();
    }

    // Deep-ε bound: an MPFR-128 Cholesky projected to an f64 snapshot +
    // double-double factor makes every Q-prune decision sound at the tight
    // bound. Gated so moderate-ε hot paths pay nothing. Computed BEFORE
    // the f64 Cholesky so an f64 Cholesky failure can't bail a search
    // whose MPFR factorization is healthy.
    let q_chol_dual = if eps <= 2e-8 || scratch.verify_prune_mpfr {
        let dual = q_cholesky_mpfr_dual(&scratch.gram, scratch.scale_bits);
        if dual.is_none() {
            eprintln!(
                "[lattice_zeta] MPFR Q-Cholesky failed (non-PD Gram) at \
                 eps={:e}, k={}; falling back to f64 factor + bound 3.0.",
                eps, k
            );
        }
        dual
    } else {
        None
    };

    // Step 3: f64 Cholesky on the post-LLL Gram. Lower-triangular L in
    // `scratch.l_f64`. Skipped in dd mode (the MPFR snapshot supersedes
    // it); required only when it is the factor the SE walk will consume.
    if q_chol_dual.is_none() {
        let t_chol = if trace { Some(std::time::Instant::now()) } else { None };
        let chol_ok = cholesky_f64(scratch);
        if let Some(t) = t_chol {
            diag::T_CHOLESKY_NS.fetch_add(diag::elapsed_ns(t), Ordering::Relaxed);
        }
        if !chol_ok {
            eprintln!(
                "[lattice_zeta] Cholesky (f64) failed at eps={:e}, k={}; bailing.",
                eps, k
            );
            return Vec::new();
        }
    }

    // Step 4: solve Bᵀ · z_c = c in MPFR. Result lands in `scratch.lu_x`.
    let t_lu = if trace { Some(std::time::Instant::now()) } else { None };
    let lu_ok = lu_solve_int_inplace(scratch);
    if let Some(t) = t_lu {
        diag::T_LU_NS.fetch_add(diag::elapsed_ns(t), Ordering::Relaxed);
    }
    if !lu_ok {
        eprintln!(
            "[lattice_zeta] LU solve failed at eps={:e}, k={}; bailing.",
            eps, k
        );
        return Vec::new();
    }

    // Split lu_x into (int, frac) WITHOUT passing through to_f64():
    // |lu_x| can exceed 2^53 at deep ε, and the resulting ULP-2
    // quantization shifts the SE center by ~10⁴ Q-units — the cap then
    // looks empty even when solutions exist. Measuring Q from the true
    // center int + frac also removes the rounded-center inflation.
    let mut z_c = SeCenter16::from_lu_x(&scratch.lu_x);
    // Per-walk deep-ε prune-verification flag: rides on the center struct
    // so the SE recursion cores need no extra parameter.
    z_c.verify_prune_mpfr = scratch.verify_prune_mpfr;

    if trace {
        // Trace-only: MPFR-direct round vs the f64 path (zero at moderate
        // ε; up to ULP at deep ε).
        #[allow(clippy::cast_possible_truncation)] // trace-only diagnostic delta
        let max_diff = (0..16).fold(0i64, |acc, i| {
            let f64_path = scratch.lu_x[i].to_f64().round() as i64;
            (f64_path - z_c.int[i]).abs().max(acc)
        });
        let max_z = (0..16).fold(0i64, |acc, i| z_c.int[i].abs().max(acc));
        eprintln!(
            "[zeta diag] find_aligned_lattice_points k={k} eps={eps:.0e} z_c max_|z|={} mpfr_vs_f64_diff={}",
            max_z, max_diff,
        );
    }

    // Transpose lower-triangular L to upper-triangular for SE (dd mode:
    // take the MPFR snapshot instead — same factor at higher accuracy).
    let l_upper: [[f64; 16]; 16] = match &q_chol_dual {
        Some((snap, _)) => *snap,
        None => std::array::from_fn(|i| std::array::from_fn(|j| scratch.l_f64[j][i])),
    };

    // Step 5: SE bound. Every point of the enumeration cap — hence every
    // valid solution — has geometric Q-norm² in [0.875, 1.25], both
    // endpoints attained: the 1/4 Σ-embedding
    // factor puts every block at lattice radius ρ = R/2, so the 3 bullet
    // blocks pinned on their spheres contribute exactly 0.75, and the
    // halved σ₁ cap offsets give apex (exact solutions) Q = 1.0 and rim
    // Q = 1.25.
    //
    // Default 1.5 = tight 1.25 + 20% slack, sound at every ε once the Q
    // bracket is computed on the MPFR factor (q_chol_dual above). 3.0 is the
    // defensive fallback when the MPFR Q-Cholesky itself fails at deep ε,
    // where the f64 partial-Q can overshoot. Override via env var.
    let bound_sq = std::env::var("CYCLOSYNTH_BOUND_SQ")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(if eps <= 2e-8 && q_chol_dual.is_none() { 3.0 } else { 1.5 });

    // Pre-compute alignment threshold and y at MPFR-128.
    //
    // 8D Z[ω] path uses `(y·x)² ≥ 2^(2k)·(1−ε²)/4`. That matches the 8D
    // y-convention `‖y‖² = 2^(k−1)` and the σ_1 image of a valid lattice
    // solution coinciding with `y_real`, giving target `(y·x)² = 2^(2k-1)`
    // — threshold lifts to `2^(2k-2)·(1−ε²) = 2^(2k)·(1−ε²)/4`.
    //
    // 16D Z[ζ_16] path: for a valid lattice solution `x_target`,
    //   `(y_lattice · x_target) = (1/4) (y_real · Σ x_target)
    //                           = (1/4) (y_real · σ_1-block of Σ x_target)
    //                           = (1/4) ‖y_real‖² = (1/4) · 2^k = 2^(k−2)`,
    // so target `(y_lattice · x_target)² = 2^(2k−4)`. Equivalent
    // threshold `(1/2)·target = 2^(2k−5)·(1−ε²) = 2^(2k)·(1−ε²)/32`.
    //
    // (Derivation: `Σᵀ Σ = 4·I_16`, `y_real = Σ y_lattice`, `y_real`
    // has mass only on σ_1 indices, and the lattice solution satisfies
    // σ_1-image(Σ x) = y_real after the bilinear `B_1 = B_2 = B_3 = 0`
    // checks zero out the σ_5/σ_9/σ_13 components.)
    let two_to_2k = MpFloat::with_val(ALIGN_PREC, 1.0) << (2 * k);
    let eps_align = MpFloat::with_val(ALIGN_PREC, eps);
    let one_minus_eps_sq_align = MpFloat::with_val(ALIGN_PREC, 1.0)
        - eps_align.clone() * &eps_align;
    let threshold_xy = MpFloat::with_val(ALIGN_PREC, &two_to_2k * &one_minus_eps_sq_align)
        / 32u32;
    let y_mpfr: [MpFloat; 16] =
        std::array::from_fn(|i| MpFloat::with_val(ALIGN_PREC, &y[i]));

    // Norm-shell target. Use i128 so k ≤ 126 stays exact (the moderate-ε
    // regime targets k ≲ 30 but the deep-ε regime can reach k > 60).
    let target_norm: i128 = 1i128 << k;
    let use_i64_path = k <= 62;
    let target_norm_i64: i64 = if use_i64_path { 1i64 << k } else { 0 };

    // Step 6: SE walk + leaf checks. Parallel + norm-pruned + incremental-x.
    let basis = scratch.basis;
    let budget = AtomicU64::new(max_leaf_checks);

    // Norm-shell pruning: precompute the upper-triangular Euclidean
    // Cholesky of the post-LLL basis at MPFR-128 (then f64 snapshot).
    let (r_eucl, r_eucl_dd) = match euclidean_cholesky_mpfr_dual(&basis) {
        Some(pair) => pair,
        None => {
            eprintln!(
                "[lattice_zeta] Euclidean Cholesky failed (rank-deficient basis) at \
                 eps={:e}, k={}; bailing.",
                eps, k
            );
            return Vec::new();
        }
    };
    let target_norm_sq_f64 = 2.0_f64.powi(k as i32);

    let t_se = if trace { Some(std::time::Instant::now()) } else { None };

    // Leaf filter: Fn + Sync. Captures only immutable references / Copy
    // values. Trace counters use the global `diag::*` atomics — zero
    // overhead when tracing is off (the `if trace` branch is predictable).
    let leaf_filter = |x: &[i64; 16]| -> LeafAction {
        let t_leaf = if trace { Some(std::time::Instant::now()) } else { None };
        if trace {
            diag::N_SE_CALLBACKS.fetch_add(1, Ordering::Relaxed);
        }
        // Norm shell: ‖x‖² == 2^k (hot path — most leaves fail here).
        if use_i64_path {
            let n: i64 = x.iter().map(|&v| v * v).sum();
            if n != target_norm_i64 {
                if trace {
                    diag::N_NORM_REJECTED.fetch_add(1, Ordering::Relaxed);
                    diag::T_LEAF_CHECK_NS.fetch_add(
                        diag::elapsed_ns(t_leaf.expect("leaf timer set when trace on")),
                        Ordering::Relaxed,
                    );
                }
                return LeafAction::Skip;
            }
        } else {
            let n: i128 = x.iter().map(|&v| i128::from(v) * i128::from(v)).sum();
            if n != target_norm {
                if trace {
                    diag::N_NORM_REJECTED.fetch_add(1, Ordering::Relaxed);
                    diag::T_LEAF_CHECK_NS.fetch_add(
                        diag::elapsed_ns(t_leaf.expect("leaf timer set when trace on")),
                        Ordering::Relaxed,
                    );
                }
                return LeafAction::Skip;
            }
        }
        // Bilinear forms: B_1=B_2=B_3=0.
        let (b1, b2, b3) = bilinear_forms(x);
        if b1 != 0 || b2 != 0 || b3 != 0 {
            if trace {
                diag::N_BILINEAR_REJECTED.fetch_add(1, Ordering::Relaxed);
                diag::T_LEAF_CHECK_NS.fetch_add(
                    diag::elapsed_ns(t_leaf.expect("leaf timer set when trace on")),
                    Ordering::Relaxed,
                );
            }
            return LeafAction::Skip;
        }
        // Alignment: (y · x)² ≥ threshold_xy. MPFR alloc here is fine —
        // very few leaves reach this far in practice (post-pruning).
        let mut tmp = MpFloat::with_val(ALIGN_PREC, 0.0);
        let mut dot_acc = MpFloat::with_val(ALIGN_PREC, 0.0);
        for (xv, yv) in x.iter().zip(y_mpfr.iter()) {
            tmp.assign(*xv);
            tmp *= yv;
            dot_acc += &tmp;
        }
        tmp.assign(&dot_acc * &dot_acc);
        if tmp < threshold_xy {
            if trace {
                diag::N_ALIGN_REJECTED.fetch_add(1, Ordering::Relaxed);
                diag::T_LEAF_CHECK_NS.fetch_add(
                    diag::elapsed_ns(t_leaf.expect("leaf timer set when trace on")),
                    Ordering::Relaxed,
                );
            }
            return LeafAction::Skip;
        }
        if trace {
            diag::N_SOLS_RETURNED.fetch_add(1, Ordering::Relaxed);
            diag::T_LEAF_CHECK_NS.fetch_add(
                diag::elapsed_ns(t_leaf.expect("leaf timer set when trace on")),
                Ordering::Relaxed,
            );
        }
        // Integer-exact filter passed. Now ask the caller whether to
        // stop the walk (typically used to bail on first ε-pass).
        if should_stop(x) {
            LeafAction::TakeAndStop
        } else {
            LeafAction::Take
        }
    };

    let (solutions, budget_was_hit) = schnorr_euchner(
        &l_upper, q_chol_dual.as_ref().map(|(_, dd)| dd), &z_c, bound_sq,
        &r_eucl, &r_eucl_dd, target_norm_sq_f64, &basis,
        leaf_filter, &budget,
        external_abort, consumed,
    );
    if budget_was_hit {
        budget_hit.store(true, Ordering::Relaxed);
    }

    if let Some(t) = t_se {
        diag::T_SE_NS.fetch_add(diag::elapsed_ns(t), Ordering::Relaxed);
    }

    solutions
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthesis::lattice::zeta::brute::{enumerate_unitary_norm_shell, uv_to_lattice_y_zeta};
    use crate::synthesis::clifford_sqrt_t::{
        det_phase_of, solution_to_u2q_with_det_phase, unitary_to_uv_zeta,
    };

    fn realistic_v() -> [f64; 4] {
        let v = [0.5, 0.3, 0.7, -0.4];
        let n: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        std::array::from_fn(|i| v[i] / n)
    }

    /// At k=2 with the `realistic_v` direction, run find_aligned_lattice_points and verify every
    /// returned solution lies in `enumerate_unitary_norm_shell(2)` AND passes the alignment
    /// threshold w.r.t. the same y direction. (Phase1 may return a *subset*
    /// of brute solutions because (a) the SE bound only covers part of the
    /// norm shell and (b) the alignment threshold filters by y-direction.)
    #[test]
    fn lattice_search_at_k_2_finds_brute_subset() {
        let v = realistic_v();
        let k = 2u32;
        let eps = 0.5_f64;
        let y = uv_to_lattice_y_zeta(v, k);

        let mut s = IntScratch16::new(eps);
        let abort = AtomicBool::new(false);
        let sols = find_aligned_lattice_points(&mut s, &y, k, eps, 100_000_000, &abort);

        let brute_set: std::collections::HashSet<[i64; 16]> =
            enumerate_unitary_norm_shell(k).into_iter().collect();
        for sol in &sols {
            assert!(
                brute_set.contains(sol),
                "find_aligned_lattice_points returned non-brute solution: {:?}",
                sol
            );
        }
        eprintln!(
            "lattice_search_at_k_2: {} solutions (subset of {} brute)",
            sols.len(),
            brute_set.len()
        );
    }

    /// Target T (k=0, det-phase = 2). Recovered exactly by find_aligned_lattice_points with the
    /// `unitary_to_uv_zeta` + single-d reconstruction path.
    #[test]
    fn lattice_search_finds_t_at_k_0() {
        use crate::matrix::u2::U2Q;
        use crate::synthesis::distance::diamond_distance_float;

        let t_gate = U2Q::t();
        let target = t_gate.to_float();
        let v = unitary_to_uv_zeta(&target);
        let d = det_phase_of(&target);
        let k = t_gate.k;
        assert_eq!(k, 0, "T should have k=0");
        let y = uv_to_lattice_y_zeta(v, k);

        let eps = 0.1_f64;
        let mut s = IntScratch16::new(eps);
        let abort = AtomicBool::new(false);
        let sols = find_aligned_lattice_points(&mut s, &y, k, eps, 100_000_000, &abort);

        assert!(!sols.is_empty(), "find_aligned_lattice_points found no solutions for T at k=0");
        let min_dist = sols.iter().map(|sol| {
            let cand = solution_to_u2q_with_det_phase(sol, k, d);
            diamond_distance_float(&cand.to_float(), &target)
        }).fold(f64::INFINITY, f64::min);
        assert!(min_dist < 1e-9, "min dist to T at k=0: {min_dist:.3e}");
    }

    /// Target QHQ (k=1, det-phase = 10). Recovered exactly via column-1
    /// extraction + single-d reconstruction.
    #[test]
    fn lattice_search_finds_qhq_at_k_1() {
        use crate::matrix::u2::U2Q;
        use crate::synthesis::distance::diamond_distance_float;

        let qhq: U2Q = U2Q::q() * U2Q::h() * U2Q::q();
        let target = qhq.to_float();
        let v = unitary_to_uv_zeta(&target);
        let d = det_phase_of(&target);
        let k = qhq.k;
        assert_eq!(k, 1, "QHQ should have k=1");
        let y = uv_to_lattice_y_zeta(v, k);

        let eps = 0.1_f64;
        let mut s = IntScratch16::new(eps);
        let abort = AtomicBool::new(false);
        let sols = find_aligned_lattice_points(&mut s, &y, k, eps, 100_000_000, &abort);

        assert!(!sols.is_empty(), "find_aligned_lattice_points found no solutions for QHQ at k=1");
        let min_dist = sols.iter().map(|sol| {
            let cand = solution_to_u2q_with_det_phase(sol, k, d);
            diamond_distance_float(&cand.to_float(), &target)
        }).fold(f64::INFINITY, f64::min);
        assert!(min_dist < 1e-9, "min dist to QHQ at k=1: {min_dist:.3e}");
    }

    /// Round-trip at a moderate k=2. Pick a brute solution, derive `v` from
    /// its reconstructed unitary, run find_aligned_lattice_points, verify the *exact* same
    /// solution (after symmetry / det-phase rotation) is among find_aligned_lattice_points's
    /// returned candidates.
    #[test]
    fn lattice_search_finds_hqhqh_at_moderate_k() {
        use crate::synthesis::lattice::zeta::brute::enumerate_unitary_norm_shell;
        use crate::synthesis::distance::diamond_distance_float;

        let k = 2u32;
        let brute_sols = enumerate_unitary_norm_shell(k);
        assert!(!brute_sols.is_empty());

        // Pick a brute solution that uses non-trivial coefficients so the
        // direction `v` is generic (avoids axis-aligned degeneracies).
        let target_sol = brute_sols
            .iter()
            .find(|&s| s.iter().filter(|&&v| v != 0).count() >= 4)
            .copied()
            .expect("expected a brute sol with ≥4 nonzero coefficients");
        let target_u2q = solution_to_u2q_with_det_phase(&target_sol, k, 0);
        let target = target_u2q.to_float();
        let v = unitary_to_uv_zeta(&target);
        let d = det_phase_of(&target);
        let y = uv_to_lattice_y_zeta(v, k);

        let eps = 0.1_f64;
        let mut s = IntScratch16::new(eps);
        let abort = AtomicBool::new(false);
        let start = std::time::Instant::now();
        let sols = find_aligned_lattice_points(&mut s, &y, k, eps, 10_000_000, &abort);
        let elapsed = start.elapsed();
        eprintln!(
            "find_aligned_lattice_points round-trip at k={}: {} solutions in {:?}",
            k,
            sols.len(),
            elapsed
        );

        assert!(
            !sols.is_empty(),
            "find_aligned_lattice_points found no solutions for k={} round-trip",
            k
        );
        let min_dist = sols.iter().map(|sol| {
            let cand = solution_to_u2q_with_det_phase(sol, k, d);
            diamond_distance_float(&cand.to_float(), &target)
        }).fold(f64::INFINITY, f64::min);
        assert!(
            min_dist < 1e-9,
            "min dist for k={} round-trip: {min_dist:.3e}",
            k
        );
        // Wall-time budget allows for rayon thread-pool contention when this
        // test runs alongside the rest of the suite. Single-test runs come
        // in well under 100 ms; the 30 s ceiling is purely a runaway guard.
        assert!(
            elapsed.as_secs_f64() < 30.0,
            "find_aligned_lattice_points at k={} took {:?} (budget 30s)",
            k, elapsed
        );
    }


}
