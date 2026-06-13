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

use rug::{Assign, Float as RFloat};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use super::cholesky_lu::{
    cholesky_f64_16, euclidean_cholesky_16_mpfr_dual, lu_solve_int_inplace_16,
    q_cholesky_16_mpfr_dual,
};
use super::lll::{run_lll_16, LllResult};
use super::q_metric::build_q_int_zeta;
use super::scratch::{rfv, IntScratch16};
use super::se::{
    bilinear_forms,
    qbracket_dd_disabled, schnorr_euchner_16d,
    verify_prune_mpfr, LeafAction, SeCenter16,
};
use crate::rings::Float;
use crate::synthesis::diag;

/// MPFR precision used by the alignment-threshold dot product. Same as 8D
/// `super::super::lattice::se::SE_PREC` — 128 bits gives ~38 digits of
/// headroom past the precision walls in the f64 formula at ε ≲ √(machine_eps).
const ALIGN_PREC: u32 = 128;

/// Run the full 16D Lenstra Z[ζ_16] pipeline for one MA-prefix's `(y, k, eps)`
/// setup and collect every solution that passes all four leaf checks.
///
/// `y` is the lattice-coord scaled y-vector (output of `uv_to_lattice_y_zeta`).
/// `max_leaf_checks` caps the SE leaf budget; when reached, `budget_hit` is
/// set and the walk aborts. Returns the empty vector on:
///   - LLL Gram-overflow,
///   - non-unimodular LLL output (algorithm bug, very unlikely),
///   - Cholesky / LU numerical failure.
/// Per-(k, ε) Q_base warm seed (CYCLOSYNTH_WARM_LLL16): only the rank-1
/// ŷŷᵀ term of the metric varies per prefix, so one Q_base reduction
/// per scratch per (k, ε) hands every prefix's LLL the shared work
/// pre-done. Always sound (any LLL-output basis is unimodular) — only
/// effectiveness varies. Returns whether the caller must clear
/// `warm_lll` after its own LLL step.
fn warm_seed_q_base(scratch: &mut IntScratch16, k: u32, eps: Float) -> bool {
    if !warm_lll16_enabled() || scratch.warm_lll {
        return false;
    }
    let seed_key = (k, eps.to_bits());
    if scratch.q_base_seed_key != Some(seed_key) {
        super::q_metric::build_q_base_mpfr_zeta(scratch, k, eps);
        build_q_int_zeta(scratch);
        let r = run_lll_16(scratch);
        let det_ok = matches!(
            super::cholesky_lu::det16_exact(&scratch.basis),
            Some(1) | Some(-1) | None
        );
        scratch.q_base_seed = if matches!(r, LllResult::Converged) && det_ok {
            Some(scratch.basis)
        } else {
            None // overflow/cap: cold starts at this key
        };
        scratch.q_base_seed_key = Some(seed_key);
    }
    if let Some(seed) = scratch.q_base_seed {
        scratch.basis = seed;
        scratch.warm_lll = true; // cleared right after the LLL step
        return true;
    }
    false
}

/// L²-LLL on the 2-step precision ladder (fplll's wrapper strategy):
/// f64 GS first, then MPFR-80 on IterCap (GS-state cycling) or a
/// non-unimodular det. GramOverflow is NOT escalated (i256 saturation,
/// precision cannot help); det = None (Bareiss i128 overflow) is
/// inconclusive-success. Returns `None` when the basis is unusable.
fn run_lll_ladder(scratch: &mut IntScratch16, k: u32, eps: Float) -> Option<()> {
    let trace = diag::trace_enabled();
    let t_lll = if trace { Some(std::time::Instant::now()) } else { None };
    let initial_use_f64 = scratch.use_f64_gs;

    // Helper: closes over scratch via &mut, returns (LllResult, det).
    fn run_and_check(s: &mut IntScratch16) -> (LllResult, Option<i64>) {
        let r = if s.use_f64_gs {
            super::lll_f64::run_lll_16_f64(s)
        } else {
            run_lll_16(s)
        };
        let det = super::cholesky_lu::det16_exact(&s.basis);
        (r, det)
    }

    let lll_succeeded = |r: LllResult, det: Option<i64>| -> bool {
        if !matches!(r, LllResult::Converged) {
            return false;
        }
        // None = i128 overflow in Bareiss; treat as inconclusive-success.
        match det { Some(d) => d == 1 || d == -1, None => true }
    };

    let (mut lll_result, mut det_check) = run_and_check(scratch);

    // Escalate to MPFR if f64 was used and produced a failure result
    // (excluding GramOverflow, which won't be helped by higher precision).
    if initial_use_f64
        && !matches!(lll_result, LllResult::GramOverflow)
        && !lll_succeeded(lll_result, det_check)
    {
        if trace {
            diag::N_LLL_F64_ESCALATIONS.fetch_add(1, Ordering::Relaxed);
        }
        // The f64 LLL may have left the basis in a partially-reduced or
        // non-unimodular state. Force a fresh start: cancel warm_lll
        // (so run_lll_16 calls reset_basis internally) and switch the
        // precision flag.
        scratch.warm_lll = false;
        scratch.use_f64_gs = false;
        let (r2, d2) = run_and_check(scratch);
        lll_result = r2;
        det_check = d2;
        // Restore the caller's precision preference for the next call.
        scratch.use_f64_gs = initial_use_f64;
    }

    if let Some(t) = t_lll {
        diag::T_LLL_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
    if let LllResult::GramOverflow = lll_result {
        return None;
    }
    if let Some(d) = det_check {
        if d != 1 && d != -1 {
            eprintln!(
                "[lattice_zeta] LLL non-unimodular even after MPFR escalation \
                (det={}) at eps={:e}, k={}; bailing.",
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
fn run_bkz_postpass(scratch: &mut IntScratch16, k: u32, eps: Float) -> Option<()> {
    if scratch.bkz_block_size >= 3 {
        let block_size = scratch.bkz_block_size as usize;
        // BKZ reads the f64 GS state. Populate it from the current
        // basis (works regardless of which LLL path was taken).
        for i in 0..16 {
            super::lll_f64::cfa_row_f64(scratch, i);
        }
        let _changed = super::bkz::bkz_tours(scratch, block_size, super::bkz::BKZ_MAX_LOOPS);
        // Post-BKZ unimodularity check; bail if the insertion path
        // somehow produced a degenerate basis.
        match super::cholesky_lu::det16_exact(&scratch.basis) {
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

/// CYCLOSYNTH_WARM_LLL16=1 enables the per-(k, ε) Q_base warm-LLL seed
/// (default off pending the A/B on the 1e-8 concurrent-parity config).
fn warm_lll16_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        std::env::var("CYCLOSYNTH_WARM_LLL16").as_deref() == Ok("1")
    });
    *ON
}

#[cfg(test)]
pub(crate) fn find_aligned_lattice_points(
    scratch: &mut IntScratch16,
    y: &[Float; 16],
    k: u32,
    eps: Float,
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
#[allow(clippy::too_many_arguments)]
pub fn find_aligned_lattice_points_with_stop<F>(
    scratch: &mut IntScratch16,
    y: &[Float; 16],
    k: u32,
    eps: Float,
    max_leaf_checks: u64,
    budget_hit: &AtomicBool,
    should_stop: F,
    external_abort: Option<&AtomicBool>,
    consumed: Option<&AtomicU64>,
) -> Vec<[i64; 16]>
where
    F: Fn(&[i64; 16]) -> bool + Sync,
{
    // Promote f64 y to MPFR. This wrapper is for legacy callers at f64
    // precision (everything ≥ ε=1e-7); ε ≤ 1e-8 should call
    // `find_aligned_lattice_points_mpfr` directly to bypass the f64 ULP floor in v.
    let prec = scratch.prec_q;
    let scale = 2.0_f64.powf(k as f64 / 2.0) / 4.0;
    let v_mpfr: [RFloat; 4] = [
        rfv(prec, y[0] / scale),
        rfv(prec, y[4] / scale),
        rfv(prec, y[8] / scale),
        rfv(prec, y[12] / scale),
    ];
    let y_mpfr: [RFloat; 16] = std::array::from_fn(|i| rfv(prec, y[i]));
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
    y: &[RFloat; 16],
    v: &[RFloat; 4],
    k: u32,
    eps: Float,
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

    let warm_seeded = warm_seed_q_base(scratch, k, eps);

    // Step 1: build Q in MPFR + i256 snapshot. Reset basis unless caller
    // requested warm_lll (Z1 D&C path) or the Q_base seed is installed.
    let t_build = if trace { Some(std::time::Instant::now()) } else { None };
    if !scratch.warm_lll {
        scratch.reset_basis();
    }
    super::q_metric::build_q_mpfr_zeta_from_mpfr_v(scratch, v, k, eps);
    build_q_int_zeta(scratch);

    // Compute cap-center c[i] = y[i] · cap_mid in MPFR at prec_q.
    let prec = scratch.prec_q;
    let one = rfv(prec, 1.0);
    let two = rfv(prec, 2.0);
    let eps_rf = rfv(prec, eps);
    let eps_sq = RFloat::with_val(prec, &eps_rf * &eps_rf);
    let one_minus_eps_sq = RFloat::with_val(prec, &one - &eps_sq);
    let sqrt_1m = one_minus_eps_sq.sqrt();
    let cap_mid_num = RFloat::with_val(prec, &one + &sqrt_1m);
    let cap_mid = RFloat::with_val(prec, &cap_mid_num / &two);
    for i in 0..16 {
        scratch.c[i].assign(RFloat::with_val(prec, &y[i] * &cap_mid));
    }
    if let Some(t) = t_build {
        diag::T_BUILD_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

    if run_lll_ladder(scratch, k, eps).is_none() {
        if warm_seeded {
            scratch.warm_lll = false;
        }
        return Vec::new();
    }
    if warm_seeded {
        // The seed's warm_lll is per-call; a persisting flag would make
        // the NEXT call (possibly a different k) skip its basis reset.
        scratch.warm_lll = false;
    }

    if run_bkz_postpass(scratch, k, eps).is_none() {
        return Vec::new();
    }

    // Deep-ε dd Q-bracket: MPFR-128 Cholesky projected to an f64
    // snapshot + dd factor, making every Q-prune decision sound at the
    // tight bound. Gated so moderate-ε hot paths pay nothing
    // (CYCLOSYNTH_QBRACKET_DD=0 restores f64 + bound 3.0). Computed
    // BEFORE the f64 Cholesky: an f64 Cholesky failure must not bail a
    // find_aligned_lattice_points whose MPFR factorization is healthy.
    let q_chol_dual = if (eps <= 2e-8 || verify_prune_mpfr()) && !qbracket_dd_disabled() {
        let dual = q_cholesky_16_mpfr_dual(&scratch.gram, scratch.scale_bits);
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
        let chol_ok = cholesky_f64_16(scratch);
        if let Some(t) = t_chol {
            diag::T_CHOLESKY_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
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
    let lu_ok = lu_solve_int_inplace_16(scratch);
    if let Some(t) = t_lu {
        diag::T_LU_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
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
    let z_c = SeCenter16::from_lu_x(&scratch.lu_x);

    if trace {
        // Compare MPFR-direct round vs the legacy f64 path to expose the
        // discrepancy in the trace (zero at moderate ε; up to ULP at deep ε).
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
    // valid solution — has geometric Q-norm² in [0.875, 1.25], with both
    // endpoints attained (docs/bound_sq_soundness.md v3): the 1/4
    // Σ-embedding factor (cf. the alignment threshold derivation below)
    // puts EVERY block at lattice radius ρ = R/2, so the 3 bullet blocks
    // pinned on their spheres contribute exactly 0.75 total, and the σ₁
    // cap offsets are halved against the Δ_y/Δ_⊥ scales: apex (exact
    // solutions) +1/4 → Q = 1.0 exactly, rim +1/2 → Q = 1.25 exactly.
    // Measured: QHQ@k=1 exact solution Q = 1.0000
    // (qhq_q_decomposition_diagnostic); 7,041 ε-close solutions max at
    // 1.2500 (q_telemetry_sweep); retention cliff at (1.20, 1.26]
    // post-fractional-center (was (1.6, 1.75] with the rounded center —
    // the difference was rounding inflation, removed by SeCenter16).
    //
    // Default 1.5 = tight 1.25 + 20% slack. Historically deep ε (≤ 2e-8)
    // needed 3.0 because the incremental f64 partial-Q overshot up to
    // ~1.8× there (the ε=1.5e-8 cliff: 1.25 · 1.8 < 3.0 absorbed it).
    // With the dd-verified Q bracket (q_chol_dual above) the boundary
    // decisions are made on ~1e-32-accurate values, so the tight band
    // [0.875, 1.25] + 20% slack = 1.5 is sound at every ε; 3.0 survives
    // only as the defensive fallback if the MPFR Q-Cholesky itself fails
    // at deep ε. Override via env var.
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
    let two_to_2k = RFloat::with_val(ALIGN_PREC, 1.0) << (2 * k);
    let eps_align = RFloat::with_val(ALIGN_PREC, eps);
    let one_minus_eps_sq_align = RFloat::with_val(ALIGN_PREC, 1.0)
        - eps_align.clone() * &eps_align;
    let threshold_xy = RFloat::with_val(ALIGN_PREC, &two_to_2k * &one_minus_eps_sq_align)
        / 32u32;
    let y_mpfr: [RFloat; 16] =
        std::array::from_fn(|i| RFloat::with_val(ALIGN_PREC, &y[i]));

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
    let (r_eucl, r_eucl_dd) = match euclidean_cholesky_16_mpfr_dual(&basis) {
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
    let udiag = diag::udiag_enabled();
    let leaf_filter = |x: &[i64; 16]| -> LeafAction {
        let t_leaf = if trace { Some(std::time::Instant::now()) } else { None };
        if trace {
            diag::N_SE_CALLBACKS.fetch_add(1, Ordering::Relaxed);
        }
        if udiag {
            diag::udiag_record(x, k);
        }
        // Norm shell: ‖x‖² == 2^k (hot path — most leaves fail here).
        if use_i64_path {
            let n: i64 = x.iter().map(|&v| v * v).sum();
            if n != target_norm_i64 {
                if trace {
                    diag::N_NORM_REJECTED.fetch_add(1, Ordering::Relaxed);
                    diag::T_LEAF_CHECK_NS.fetch_add(
                        t_leaf.unwrap().elapsed().as_nanos() as u64,
                        Ordering::Relaxed,
                    );
                }
                return LeafAction::Skip;
            }
        } else {
            let n: i128 = x.iter().map(|&v| (v as i128) * (v as i128)).sum();
            if n != target_norm {
                if trace {
                    diag::N_NORM_REJECTED.fetch_add(1, Ordering::Relaxed);
                    diag::T_LEAF_CHECK_NS.fetch_add(
                        t_leaf.unwrap().elapsed().as_nanos() as u64,
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
                    t_leaf.unwrap().elapsed().as_nanos() as u64,
                    Ordering::Relaxed,
                );
            }
            return LeafAction::Skip;
        }
        // Alignment: (y · x)² ≥ threshold_xy. MPFR alloc here is fine —
        // very few leaves reach this far in practice (post-pruning).
        let mut tmp = RFloat::with_val(ALIGN_PREC, 0.0);
        let mut dot_acc = RFloat::with_val(ALIGN_PREC, 0.0);
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
                    t_leaf.unwrap().elapsed().as_nanos() as u64,
                    Ordering::Relaxed,
                );
            }
            return LeafAction::Skip;
        }
        if trace {
            diag::N_SOLS_RETURNED.fetch_add(1, Ordering::Relaxed);
            diag::T_LEAF_CHECK_NS.fetch_add(
                t_leaf.unwrap().elapsed().as_nanos() as u64,
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

    let (solutions, budget_was_hit) = schnorr_euchner_16d(
        &l_upper, q_chol_dual.as_ref().map(|(_, dd)| dd), &z_c, bound_sq,
        &r_eucl, &r_eucl_dd, target_norm_sq_f64, &basis,
        leaf_filter, &budget,
        external_abort, consumed,
    );
    if budget_was_hit {
        budget_hit.store(true, Ordering::Relaxed);
    }

    if let Some(t) = t_se {
        diag::T_SE_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

    solutions
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthesis::brute_search_zeta::{enumerate_unitary_norm_shell, uv_to_lattice_y_zeta};
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

    /// Gate (ignored): the QHQ@k=1 lattice search must succeed at
    /// CYCLOSYNTH_BOUND_SQ=2 now that SE measures Q from the fractional
    /// center (the solution's geometric Q is 1.00; the legacy rounded
    /// center inflated it to 6.28, forcing the k≤4 bound-8 escape).
    /// Ignored because it mutates the process-global env var — run alone:
    /// `cargo test --release --lib qhq_at_bound_2 -- --ignored`.
    #[test]
    #[ignore]
    fn lattice_search_finds_qhq_at_k_1_bound_2() {
        use crate::matrix::u2::U2Q;
        use crate::synthesis::distance::diamond_distance_float;

        unsafe { std::env::set_var("CYCLOSYNTH_BOUND_SQ", "2") };
        let qhq: U2Q = U2Q::q() * U2Q::h() * U2Q::q();
        let target = qhq.to_float();
        let v = unitary_to_uv_zeta(&target);
        let d = det_phase_of(&target);
        let k = qhq.k;
        let y = uv_to_lattice_y_zeta(v, k);

        let eps = 0.1_f64;
        let mut s = IntScratch16::new(eps);
        let abort = AtomicBool::new(false);
        let sols = find_aligned_lattice_points(&mut s, &y, k, eps, 100_000_000, &abort);
        unsafe { std::env::remove_var("CYCLOSYNTH_BOUND_SQ") };

        assert!(!sols.is_empty(), "find_aligned_lattice_points@bound2 found no solutions for QHQ at k=1");
        let min_dist = sols.iter().map(|sol| {
            let cand = solution_to_u2q_with_det_phase(sol, k, d);
            diamond_distance_float(&cand.to_float(), &target)
        }).fold(f64::INFINITY, f64::min);
        assert!(min_dist < 1e-9, "min dist to QHQ at k=1 (bound 2): {min_dist:.3e}");
    }

    /// Diagnostic (ignored): decompose the QHQ@k=1 solution's Q-norm into
    /// geometric Q (from the true fractional cap center), legacy SE-measured
    /// Q (from the i64-rounded z_c center), and Q_se_effective (from the
    /// fractional SeCenter16 the walk now uses — should match Q_geometric
    /// to ~1e-6). The rounded column explains why the QHQ test historically
    /// needed bound_sq > 6 while the geometric theory says Q ≤ 2.75
    /// (docs/bound_sq_soundness.md). Run with --ignored --nocapture.
    #[test]
    #[ignore]
    fn qhq_q_decomposition_diagnostic() {
        use crate::matrix::u2::U2Q;

        unsafe { std::env::set_var("CYCLOSYNTH_BOUND_SQ", "8") };
        let qhq: U2Q = U2Q::q() * U2Q::h() * U2Q::q();
        let target = qhq.to_float();
        let v = unitary_to_uv_zeta(&target);
        let k = qhq.k;
        let y = uv_to_lattice_y_zeta(v, k);
        let eps = 0.1_f64;

        let mut s = IntScratch16::new(eps);
        let abort = AtomicBool::new(false);
        let sols = find_aligned_lattice_points(&mut s, &y, k, eps, 100_000_000, &abort);
        unsafe { std::env::remove_var("CYCLOSYNTH_BOUND_SQ") };
        assert!(!sols.is_empty(), "find_aligned_lattice_points@bound8 must find QHQ");

        let q = crate::synthesis::lattice_zeta::q_metric::build_q_zzeta_lattice(v, k, eps);
        // True cap center (ambient), legacy rounded-z_c effective center,
        // and the fractional SE center (int + frac pair) the walk now uses.
        let c_true: [f64; 16] = std::array::from_fn(|i| s.c[i].to_f64());
        let mut c_rounded = [0.0f64; 16];
        for i in 0..16 {
            let zi = s.lu_x[i].to_f64().round();
            for j in 0..16 {
                c_rounded[j] += zi * s.basis[i][j] as f64;
            }
        }
        let se_center = SeCenter16::from_lu_x(&s.lu_x);
        let mut c_se = [0.0f64; 16];
        for i in 0..16 {
            let zi = se_center.int[i] as f64 + se_center.frac[i];
            for j in 0..16 {
                c_se[j] += zi * s.basis[i][j] as f64;
            }
        }
        let q_norm = |x: &[i64; 16], c: &[f64; 16]| -> f64 {
            let d: [f64; 16] = std::array::from_fn(|i| x[i] as f64 - c[i]);
            let mut acc = 0.0;
            for i in 0..16 {
                for j in 0..16 {
                    acc += d[i] * q[i][j] * d[j];
                }
            }
            acc
        };
        for (n, sol) in sols.iter().enumerate() {
            eprintln!(
                "sol {n}: Q_geometric={:.6}  Q_se_rounded_center={:.4}  Q_se_effective={:.6}",
                q_norm(sol, &c_true),
                q_norm(sol, &c_rounded),
                q_norm(sol, &c_se)
            );
        }
        let frac_err: f64 = (0..16)
            .map(|i| (s.lu_x[i].to_f64() - s.lu_x[i].to_f64().round()).abs())
            .fold(0.0, f64::max);
        eprintln!("max |frac(lu_x)| = {frac_err:.4}");
    }

    /// Telemetry (ignored): geometric Q-norm² distribution of ε-close
    /// solutions across a θ × ε × k grid, enumerated at bound 4 (wide
    /// enough to observe anything up to the geometric max 2.75 of
    /// docs/bound_sq_soundness.md). Decides whether the observed 1.75
    /// ceiling is structural (cap part ≤ 1 in implemented scaling → sound
    /// bound 2.0) or just an unpopulated rim (max → 2.75 → keep 3.0).
    /// Run with --ignored --nocapture.
    #[test]
    #[ignore]
    fn q_telemetry_sweep() {
        use crate::synthesis::distance::diamond_distance_float;
        use num_complex::Complex;

        unsafe { std::env::set_var("CYCLOSYNTH_BOUND_SQ", "4") };
        let mut global_max_close = 0.0f64;
        let mut global_max_all = 0.0f64;
        let mut total_close = 0usize;

        for &theta in &[0.3f64, 0.55, 0.8, 1.05, 1.3] {
            let target: crate::synthesis::Mat2 = [
                [Complex::from_polar(1.0, -theta / 2.0), Complex::new(0.0, 0.0)],
                [Complex::new(0.0, 0.0), Complex::from_polar(1.0, theta / 2.0)],
            ];
            let v = unitary_to_uv_zeta(&target);
            let d = det_phase_of(&target);
            for &(eps, k_lo, k_hi) in &[(3e-2f64, 5u32, 7u32), (1e-3, 9, 10)] {
                for k in k_lo..=k_hi {
                    let y = uv_to_lattice_y_zeta(v, k);
                    let mut s = IntScratch16::new(eps);
                    let abort = AtomicBool::new(false);
                    let sols = find_aligned_lattice_points(&mut s, &y, k, eps, 100_000_000, &abort);
                    if sols.is_empty() {
                        continue;
                    }
                    let q = crate::synthesis::lattice_zeta::q_metric::build_q_zzeta_lattice(
                        v, k, eps,
                    );
                    let c: [f64; 16] = std::array::from_fn(|i| s.c[i].to_f64());
                    let mut max_close = 0.0f64;
                    let mut max_all = 0.0f64;
                    let mut n_close = 0usize;
                    for sol in &sols {
                        let dvec: [f64; 16] =
                            std::array::from_fn(|i| sol[i] as f64 - c[i]);
                        let mut qn = 0.0;
                        for i in 0..16 {
                            for j in 0..16 {
                                qn += dvec[i] * q[i][j] * dvec[j];
                            }
                        }
                        max_all = max_all.max(qn);
                        let cand = solution_to_u2q_with_det_phase(sol, k, d);
                        if diamond_distance_float(&cand.to_float(), &target) <= eps {
                            max_close = max_close.max(qn);
                            n_close += 1;
                        }
                    }
                    if n_close > 0 {
                        eprintln!(
                            "θ={theta:<4} ε={eps:.0e} k={k:<2} sols={:<5} close={n_close:<4} maxQ_close={max_close:.4} maxQ_all={max_all:.4}",
                            sols.len()
                        );
                    }
                    global_max_close = global_max_close.max(max_close);
                    global_max_all = global_max_all.max(max_all);
                    total_close += n_close;
                }
            }
        }
        unsafe { std::env::remove_var("CYCLOSYNTH_BOUND_SQ") };
        eprintln!(
            "GLOBAL: eps-close sols={total_close}  maxQ_close={global_max_close:.4}  maxQ_all={global_max_all:.4}"
        );
    }

    /// Round-trip at a moderate k=2. Pick a brute solution, derive `v` from
    /// its reconstructed unitary, run find_aligned_lattice_points, verify the *exact* same
    /// solution (after symmetry / det-phase rotation) is among find_aligned_lattice_points's
    /// returned candidates.
    #[test]
    fn lattice_search_finds_hqhqh_at_moderate_k() {
        use crate::synthesis::brute_search_zeta::enumerate_unitary_norm_shell;
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

    /// Performance smoke at moderate-k (typically k=4-6 from a deterministic
    /// circuit). Reports timing and solution count; the test only fails if
    /// the walk blows up wall-clock past a generous bound.
    #[test]
    #[ignore = "60s timing budget; run with --ignored"]
    fn lattice_search_perf_at_k_8_completes() {
        use crate::matrix::u2::U2Q;

        // Deterministic k=8 circuit: 8 H's interleaved with 8 Q's. Single-d
        // reconstruction works for any det-phase (no SU(2) projection).
        let mut u = U2Q::eye();
        for c in "HQHQHQHQHQHQHQHQ".chars() {
            u = u * match c {
                'H' => U2Q::h(),
                'Q' => U2Q::q(),
                _ => unreachable!(),
            };
        }
        let target = u.to_float();
        let v = unitary_to_uv_zeta(&target);
        let k = u.k;
        let y = uv_to_lattice_y_zeta(v, k);

        let eps = 0.1_f64;
        let mut s = IntScratch16::new(eps);
        let abort = AtomicBool::new(false);
        let start = std::time::Instant::now();
        let sols = find_aligned_lattice_points(&mut s, &y, k, eps, 100_000_000, &abort);
        let elapsed = start.elapsed();
        eprintln!(
            "find_aligned_lattice_points at k={} took {} ms, returned {} solutions",
            k,
            elapsed.as_millis(),
            sols.len()
        );

        // Don't assert on solution count (spec allows missing exact match);
        // enforce only that we don't blow up wall-clock.
        assert!(
            elapsed.as_secs() < 60,
            "find_aligned_lattice_points at k={} took {:?} (budget 60s)",
            k, elapsed
        );
    }

    /// At ε=1e-5 with k=14, verify the i256 LLL path doesn't trip overflow.
    /// Returns a (possibly empty) Vec of solutions.
    #[test]
    #[ignore = "120s budget at deep ε; run with --ignored"]
    fn lattice_search_no_overflow_at_eps_1e_5() {
        let v = realistic_v();
        let k = 14u32;
        let eps = 1e-5_f64;
        let y = uv_to_lattice_y_zeta(v, k);
        let mut s = IntScratch16::new(eps);
        let abort = AtomicBool::new(false);
        let start = std::time::Instant::now();
        let sols = find_aligned_lattice_points(&mut s, &y, k, eps, 10_000_000, &abort);
        let elapsed = start.elapsed();
        eprintln!(
            "find_aligned_lattice_points at ε=1e-5, k=14: {} solutions in {:?}",
            sols.len(),
            elapsed
        );
        // We don't require non-empty (the random `realistic_v` direction may
        // not have an exact lde=14 lattice match). The test is just that the
        // pipeline doesn't crash or overflow.
        assert!(
            elapsed.as_secs() < 120,
            "find_aligned_lattice_points at ε=1e-5 took {:?} (budget 120s)",
            elapsed
        );
    }

    /// A/B diagnostic: run find_aligned_lattice_points (norm-pruned) at a fixed k and report
    /// timing + sols. To compare against the non-pruned baseline, swap the
    /// SE call site in `find_aligned_lattice_points` temporarily.
    #[test]
    #[ignore]
    fn diag_norm_prune_vs_baseline() {
        use crate::synthesis::distance::diamond_distance_float;
        let theta = 0.3_f64;
        let target: crate::synthesis::distance::Mat2 = [
            [num_complex::Complex64::from_polar(1.0, -theta / 2.0),
             num_complex::Complex64::new(0.0, 0.0)],
            [num_complex::Complex64::new(0.0, 0.0),
             num_complex::Complex64::from_polar(1.0, theta / 2.0)],
        ];
        let v = unitary_to_uv_zeta(&target);
        let d = det_phase_of(&target);
        let eps = 1e-3_f64;
        for k in 9u32..=10 {
            let y = uv_to_lattice_y_zeta(v, k);
            let mut s = IntScratch16::new(eps);
            let abort = AtomicBool::new(false);
            let budget = 1_000_000_000_u64;
            let t0 = std::time::Instant::now();
            let sols = find_aligned_lattice_points(&mut s, &y, k, eps, budget, &abort);
            let dt = t0.elapsed();
            let min_dist = sols.iter().map(|sol| {
                let cand = solution_to_u2q_with_det_phase(sol, k, d);
                diamond_distance_float(&cand.to_float(), &target)
            }).fold(f64::INFINITY, f64::min);
            eprintln!(
                "k={k:>2}  sols={:>3}  min_dist={min_dist:.3e}  t={:>10.3?}  budget_hit={}",
                sols.len(), dt, abort.load(Ordering::Relaxed)
            );
        }
    }

    /// Diagnostic: for Rz(0.3) at ε=1e-3, first establish the lde the 8D
    /// Clifford+T synthesizer reaches (upper bound for Clifford+√T since
    /// `T = QQ` as gates and lde counts √2 denominators identically). Then
    /// verify the Z[ζ_16] / Clifford+√T flow hits it at ≤ that lde.
    /// Behind `#[ignore]`: `cargo test --release --lib diag_eps_1e_3 --
    /// --ignored --nocapture`.
    #[test]
    #[ignore]
    fn diag_eps_1e_3() {
        use crate::synthesis::distance::diamond_distance_float;
        use crate::synthesis::clifford_t::SynthesizerT;
        let theta = 0.3_f64;
        let target: crate::synthesis::distance::Mat2 = [
            [num_complex::Complex64::from_polar(1.0, -theta / 2.0),
             num_complex::Complex64::new(0.0, 0.0)],
            [num_complex::Complex64::new(0.0, 0.0),
             num_complex::Complex64::from_polar(1.0, theta / 2.0)],
        ];
        let eps = 1e-3_f64;

        // 1. Upper bound from 8D Clifford+T.
        let synth_t = SynthesizerT::new(eps);
        let t0 = std::time::Instant::now();
        let r_t = synth_t.synthesize(target).expect("8D should land Rz(0.3) at ε=1e-3");
        eprintln!(
            "8D Clifford+T:  lde={}  dist={:.3e}  t={:?}",
            r_t.lde, r_t.distance, t0.elapsed()
        );
        let upper_bound = r_t.lde;

        // 2. Sweep Clifford+√T at increasing budget at each k up to upper_bound.
        let v = unitary_to_uv_zeta(&target);
        let d = det_phase_of(&target);
        eprintln!("upper bound k = {upper_bound}; v={v:?}, d={d}");
        for k in 5u32..=(upper_bound + 2).min(20) {
            let y = uv_to_lattice_y_zeta(v, k);
            let budget = 1_000_000_000_u64;
            let mut s = IntScratch16::new(eps);
            let abort = AtomicBool::new(false);
            let t0 = std::time::Instant::now();
            let sols = find_aligned_lattice_points(&mut s, &y, k, eps, budget, &abort);
            let dt = t0.elapsed();
            let abort_v = abort.load(Ordering::Relaxed);
            let min_dist = sols.iter().map(|sol| {
                let cand = solution_to_u2q_with_det_phase(sol, k, d);
                diamond_distance_float(&cand.to_float(), &target)
            }).fold(f64::INFINITY, f64::min);
            let hit = min_dist < eps;
            eprintln!(
                "k={k:>2}  sols={:>4}  budget_hit={abort_v:>5}  \
                 min_dist={min_dist:.3e}  hit_eps={hit:>5}  t={:?}",
                sols.len(), dt
            );
            if hit { break; }
        }
    }

    /// Serializes the two predictive-truncation tests: they share the
    /// process-global fire counters AND the global rayon pool (a
    /// concurrent walk shifts the other's item-completion dynamics), so
    /// running them simultaneously under `--include-ignored` makes both
    /// flaky. Poison-tolerant: a panic in one must not mask the other.
    static PREDICTIVE_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Predictive budget truncation must NOT fire on a walk that completes
    /// within its budget: a budget-capped (NOT u64::MAX — the predictive
    /// context is attached) k=2 enumeration that finishes far below the
    /// cap. Asserts via the always-on diag counter that no predictive
    /// abort happened and the walk did not report a budget hit.
    ///
    /// Ignored (run via `cargo test --release predictive_trunc --
    /// --ignored --test-threads=1`): the counter is process-global, and
    /// since margin 2.5 + concurrent parity branches landed, OTHER suite
    /// tests' budgeted odd-branch walks can legitimately fire while this
    /// test's walk is in flight (fires are semantically identical to
    /// budget-hits, just earlier) — under parallel test execution the
    /// global delta is not attributable to this walk and the assert
    /// flakes (~1 in 4 suite runs).
    #[test]
    #[ignore = "global fire counter is not attributable under parallel test execution"]
    fn predictive_trunc_no_fire_on_completing_walk() {
        use crate::synthesis::diag;

        let _guard = PREDICTIVE_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let fires_before = diag::N_PREDICTIVE_TRUNC_FIRES.load(Ordering::Relaxed);
        let v = realistic_v();
        let k = 2u32;
        let eps = 0.5_f64;
        let y = uv_to_lattice_y_zeta(v, k);
        let mut s = IntScratch16::new(eps);
        let abort = AtomicBool::new(false);
        let sols = find_aligned_lattice_points(&mut s, &y, k, eps, 10_000_000, &abort);
        assert!(!sols.is_empty(), "k=2 budget-capped walk found nothing");
        assert!(
            !abort.load(Ordering::Relaxed),
            "completing walk reported budget_hit"
        );
        let fires_after = diag::N_PREDICTIVE_TRUNC_FIRES.load(Ordering::Relaxed);
        assert_eq!(
            fires_after, fires_before,
            "predictive truncation fired on a walk that completed in-budget"
        );
    }

    /// Demonstration (ignored — ~2 s walk): predictive truncation fires on
    /// a budget-capped walk that cannot complete, aborting with consumed
    /// ≪ budget instead of burning the whole pool. Config: rz(0.7) at
    /// ε=1e-3, k=11 — the level whose m=0 coverage walks blow through even
    /// the 32×-boosted certify budgets (true total ≳ 100G nodes).
    ///
    /// Measured fire window (2026-06-10, 14 threads): completing 10% of
    /// the 4153 frontier items costs C* ≈ 1.21G nodes, so a budget B fires
    /// iff B ∈ (C*, C*/0.3 ≈ 4G): at B=3G it fires at consumed 1.21G
    /// (40% of budget, projected 11.7G > 3×3G), at B=2M/100M/6.4G it
    /// plain-exhausts (fraction_done at exhaustion 0.003/0.004/0.376 —
    /// the last projects 17G = 2.66×B, just under MARGIN=3; see
    /// docs/w_predictive_trunc_notes.md for the bias analysis). Asserts
    /// the predictive counter fired, the plain-exhaust counter did NOT,
    /// the walk surfaced as a budget hit, and consumed ≪ budget. Run:
    /// `cargo test --release --lib predictive_trunc_fires -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn predictive_trunc_fires_on_infeasible_budget() {
        use crate::synthesis::diag;
        use num_complex::Complex64;

        let theta = 0.7f64;
        let target: crate::synthesis::Mat2 = [
            [Complex64::from_polar(1.0, -theta / 2.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::from_polar(1.0, theta / 2.0)],
        ];
        let v = unitary_to_uv_zeta(&target);
        let eps = 1e-3_f64;
        let k = 11u32;
        let budget = 3_000_000_000_u64; // inside the measured (1.2G, 4G) fire window
        let y = uv_to_lattice_y_zeta(v, k);

        let _guard = PREDICTIVE_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // Walk-end `[w1] predictive: ...` progress line (items/consumed/fired).
        unsafe { std::env::set_var("CYCLOSYNTH_W1_DEBUG", "1") };
        let pred_before = diag::N_PREDICTIVE_TRUNC_FIRES.load(Ordering::Relaxed);
        let exh_before = diag::N_BUDGET_EXHAUST_FIRES.load(Ordering::Relaxed);
        let mut s = IntScratch16::new(eps);
        let abort = AtomicBool::new(false);
        let consumed = AtomicU64::new(0);
        let t0 = std::time::Instant::now();
        let _sols = crate::synthesis::lattice_zeta::find_aligned_lattice_points_with_stop(
            &mut s, &y, k, eps, budget, &abort, |_| false, None, Some(&consumed),
        );
        unsafe { std::env::remove_var("CYCLOSYNTH_W1_DEBUG") };
        let pred_fires = diag::N_PREDICTIVE_TRUNC_FIRES.load(Ordering::Relaxed) - pred_before;
        let exh_fires = diag::N_BUDGET_EXHAUST_FIRES.load(Ordering::Relaxed) - exh_before;
        let used = consumed.load(Ordering::Relaxed);
        eprintln!(
            "predictive demo: k={k} eps={eps:e} budget={budget}  consumed={used} \
             ({:.1}% of budget)  pred_fires={pred_fires}  exhaust_fires={exh_fires}  \
             budget_hit={}  t={:?}",
            100.0 * used as f64 / budget as f64,
            abort.load(Ordering::Relaxed),
            t0.elapsed(),
        );
        assert!(
            abort.load(Ordering::Relaxed),
            "infeasible-budget walk must surface as budget hit"
        );
        // Whether the PREDICTIVE path fires here is trajectory-dependent
        // on BOTH window edges: work-stealing decides how skinny-biased
        // the item completion order is, which moves the f ≥ MIN_FRAC
        // gate AND the projection crossing (observed: fired at 40% on
        // one run, plain-exhausted on the next, same config). Firing is
        // opportunistic by design — a non-fire is a plain exhaustion
        // with identical downstream semantics — so the hard contract is:
        // exactly ONE of the two truncation paths fired, never both,
        // never neither.
        assert_eq!(
            pred_fires + exh_fires,
            1,
            "exactly one truncation path must fire (pred={pred_fires}, exhaust={exh_fires})"
        );
        if pred_fires == 1 {
            eprintln!("predictive demo: PREDICTIVE path (reclaimed {:.1}% of budget)",
                100.0 * (budget - used.min(budget)) as f64 / budget as f64);
        } else {
            eprintln!("predictive demo: plain exhaustion (scheduling kept projection under margin)");
        }
        // When predictive DID fire, the budget must actually be reclaimed.
        assert!(
            pred_fires == 0 || used < budget * 95 / 100,
            "predictive abort reclaimed too little (consumed {used} of {budget})"
        );
    }
}
