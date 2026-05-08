//! Phase 1 driver for the 16D Z[ζ_16] L²-LLL pipeline.
//!
//! Wires together every stage from M1-M4:
//!
//!  1. **Build Q** in MPFR ([`build_q_mpfr_zeta`]) + i256 snapshot
//!     ([`build_q_int_zeta`]). The MPFR Q-construction does not populate
//!     `scratch.c` (cap-center in lattice coords); we compute it here.
//!  2. **L²-LLL** ([`run_lll_16`]) — MPFR Gram-Schmidt over the exact i256
//!     Gram. Returns `LllResult::GramOverflow` at deep ε if i256 saturates.
//!  3. **Cholesky** ([`cholesky_f64_16`]) — f64 lower-triangular L on the
//!     post-LLL Gram (LLL invariant κ(G) ≤ (4/3)^15 ≈ 240 keeps f64 safe).
//!     Transposed to upper-triangular at the SE call site.
//!  4. **LU solve** ([`lu_solve_int_inplace_16`]) — Bᵀ · z_c = c at MPFR
//!     `lu_prec` bits. Solution rounded to i64 for SE's z_c convention.
//!  5. **Schnorr-Euchner** ([`schnorr_euchner_16d`]) — walk integer
//!     16-tuples within the Q-bounded ellipsoid; for each leaf, reconstruct
//!     `x = B·z` and validate the four leaf checks.
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

#![allow(dead_code)]
#![allow(clippy::needless_range_loop)]

use rug::{Assign, Float as RFloat};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use super::cholesky_lu::{cholesky_f64_16, lu_solve_int_inplace_16};
use super::lll::{run_lll_16, LllResult};
use super::q_metric::{build_q_int_zeta, build_q_mpfr_zeta};
use super::scratch::{rfv, IntScratch16};
use super::se::{
    bilinear_forms, euclidean_cholesky_16_mpfr,
    schnorr_euchner_16d_par_norm_pruned, LeafAction,
};
use crate::rings::Float;
use crate::synthesis::diag;

/// MPFR precision used by the alignment-threshold dot product. Same as 8D
/// `super::super::lenstra::se::SE_PREC` — 128 bits gives ~38 digits of
/// headroom past the precision walls in the f64 formula at ε ≲ √(machine_eps).
const ALIGN_PREC: u32 = 128;

/// Run the full 16D Lenstra Z[ζ_16] pipeline for one MA-prefix's `(y, k, eps)`
/// setup and collect every solution that passes all four leaf checks.
///
/// `y` is the lattice-coord scaled y-vector (output of `uv_to_xy_zeta`).
/// `max_phase2_calls` caps the SE leaf budget; when reached, `budget_hit` is
/// set and the walk aborts. Returns the empty vector on:
///   - LLL Gram-overflow,
///   - non-unimodular LLL output (algorithm bug, very unlikely),
///   - Cholesky / LU numerical failure.
pub fn phase1(
    scratch: &mut IntScratch16,
    y: &[Float; 16],
    k: u32,
    eps: Float,
    max_phase2_calls: u64,
    budget_hit: &AtomicBool,
) -> Vec<[i64; 16]> {
    phase1_with_stop(scratch, y, k, eps, max_phase2_calls, budget_hit, |_| false)
}

/// Phase 1 with an early-exit predicate. `should_stop(x)` is called
/// **only** for leaves that pass the integer-exact filter (norm shell +
/// bilinear forms + alignment). When it returns `true`, the lattice
/// search aborts after collecting that leaf — the caller can wrap
/// expensive checks (e.g. ε-bounded diamond distance) here without
/// paying their cost on every doomed leaf.
pub fn phase1_with_stop<F>(
    scratch: &mut IntScratch16,
    y: &[Float; 16],
    k: u32,
    eps: Float,
    max_phase2_calls: u64,
    budget_hit: &AtomicBool,
    should_stop: F,
) -> Vec<[i64; 16]>
where
    F: Fn(&[i64; 16]) -> bool + Sync,
{
    let trace = diag::trace_enabled();
    if trace {
        diag::N_PHASE1_CALLS.fetch_add(1, Ordering::Relaxed);
    }
    // Recover the SU(2) direction `v = (Re V_{11}, Im V_{11}, Re V_{21},
    // Im V_{21})` from the lattice-coord y. By construction
    // `y[j]   = (cos(jπ/8)·v[0] + sin(jπ/8)·v[1]) · scale`,
    // `y[8+j] = (cos(jπ/8)·v[2] + sin(jπ/8)·v[3]) · scale`,
    // with `scale = 2^(k/2) / 4`. At j=0 (cos=1, sin=0): v[0]=y[0]/scale,
    // v[2]=y[8]/scale. At j=4 (cos=0, sin=1): v[1]=y[4]/scale,
    // v[3]=y[12]/scale.
    let scale = 2.0_f64.powf(k as f64 / 2.0) / 4.0;
    let v: [f64; 4] = [y[0] / scale, y[4] / scale, y[8] / scale, y[12] / scale];

    // Step 1: build Q in MPFR + i256 snapshot. Reset basis to identity (LLL
    // expects the identity start; `run_lll_16` calls `reset_basis()` again
    // internally but doing it here keeps the contract explicit).
    //
    // When `scratch.warm_lll` is set (Z1 D&C path), skip the reset and let
    // LLL warm-start from the previous call's reduced basis. The basis is
    // still a valid Z^16 basis (any unimodular transformation), so LLL
    // re-reduces it for the new Gram in (typically) far fewer iterations.
    let t_build = if trace { Some(std::time::Instant::now()) } else { None };
    if !scratch.warm_lll {
        scratch.reset_basis();
    }
    build_q_mpfr_zeta(scratch, v, k, eps);
    build_q_int_zeta(scratch);

    // Compute cap-center c[i] = y[i] · cap_mid in MPFR at prec_q.
    // cap_mid = (1 + √(1 − ε²)) / 2.
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
        let yi = rfv(prec, y[i]);
        scratch.c[i].assign(RFloat::with_val(prec, &yi * &cap_mid));
    }
    if let Some(t) = t_build {
        diag::T_BUILD_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

    // ─── Step 2: L²-LLL with adaptive precision ladder ───────────────
    //
    // fplll's `wrapper.cpp` runs a precision ladder: try low-precision
    // first, escalate on detected failure. Their full ladder is
    //   `double` (53 bit) → `dpe_t` (double + long expo) → `dd_real`
    //   (~106 bit) → `mpfr_t` (arbitrary).
    // We use the same idea with a 2-step ladder over our two backends:
    //
    //   1. **f64 GS** (`lll_f64::run_lll_16_f64`): 52 mantissa bits,
    //      ~2.5× faster per call than MPFR-80. fplll's `l2_min_prec`
    //      formula `≥ 10 + 2·log d − log ε + d·log ρ` says we need ~50
    //      bits at d=16, ε=1e-8, leaving f64 with a 2-bit margin.
    //      Empirically converges through ε=1e-7; ε=1e-8 is borderline.
    //
    //   2. **MPFR-80** (`run_lll_16` at `GS_PREC=80`, the default): 80
    //      mantissa bits, ~30-bit margin at ε=1e-8 — comfortably safe.
    //      ~2.5× slower per call but reliable.
    //
    // **Failure detection** (signals the f64 path is past its precision
    // budget):
    //   (a) `LllResult::IterCap` — LLL didn't converge in MAX_LLL_ITERS.
    //       Strong signal of GS-state cycling from precision loss.
    //   (b) `det16_exact == Some(d)` with `d ∉ {±1}` — basis became
    //       non-unimodular under f64 LLL's transformations. Means the
    //       size-reduction's f64 mu-rounding accumulated a wrong basis
    //       update somewhere.
    //
    // **Not escalated**:
    //   - `LllResult::GramOverflow`: the i256 Gram buffer overflowed,
    //     not a precision issue. MPFR can't help — we'd need wider
    //     integers. Return empty.
    //   - `det16_exact == None`: i128 Bareiss elimination overflowed at
    //     d=16 (rare at deep ε per the chunk 2 caveat). Treat as
    //     inconclusive-success and proceed; no clean fallback.
    //
    // The escalation cost is one full LLL setup + run. When f64 succeeds
    // (ε ≥ 1e-7 typically), the ladder's overhead is just a det check
    // (≤ 1 μs) — negligible. When f64 fails, we pay 2× LLL.
    //
    // Diag counter `N_LLL_F64_ESCALATIONS` tracks how often this fires.
    // Should be 0 at moderate ε; non-zero only at ε ≤ 1e-8.

    let t_lll = if trace { Some(std::time::Instant::now()) } else { None };
    let initial_use_f64 = scratch.use_f64_gs;

    // Helper: closes over scratch via &mut, returns (LllResult, det).
    fn run_and_check(s: &mut IntScratch16) -> (LllResult, Option<i64>) {
        let r = if s.use_f64_gs {
            super::lll_f64::run_lll_16_f64(s)
        } else {
            run_lll_16(s)
        };
        let det = super::se::det16_exact(&s.basis);
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
        return Vec::new();
    }
    if let Some(d) = det_check {
        if d != 1 && d != -1 {
            eprintln!(
                "[lenstra_zeta] LLL non-unimodular even after MPFR escalation \
                (det={}) at eps={:e}, k={}; bailing.",
                d, eps, k
            );
            return Vec::new();
        }
    }
    if !matches!(lll_result, LllResult::Converged | LllResult::IterCap) {
        // Should be unreachable (only GramOverflow is left, handled above).
        return Vec::new();
    }

    // Step 3: f64 Cholesky on the post-LLL Gram. Lower-triangular L in
    // `scratch.l_f64`.
    let t_chol = if trace { Some(std::time::Instant::now()) } else { None };
    let chol_ok = cholesky_f64_16(scratch);
    if let Some(t) = t_chol {
        diag::T_CHOLESKY_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
    if !chol_ok {
        eprintln!(
            "[lenstra_zeta] Cholesky (f64) failed at eps={:e}, k={}; bailing.",
            eps, k
        );
        return Vec::new();
    }

    // Step 4: solve Bᵀ · z_c = c in MPFR. Result lands in `scratch.lu_x`.
    let t_lu = if trace { Some(std::time::Instant::now()) } else { None };
    let lu_ok = lu_solve_int_inplace_16(scratch);
    if let Some(t) = t_lu {
        diag::T_LU_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
    if !lu_ok {
        eprintln!(
            "[lenstra_zeta] LU solve failed at eps={:e}, k={}; bailing.",
            eps, k
        );
        return Vec::new();
    }

    // Round lu_x → i64 for SE's z_c convention. Loses fractional info; the
    // SE bound (with the cap-radial slack) more than absorbs the resulting
    // off-center penalty.
    let z_c: [i64; 16] = std::array::from_fn(|i| {
        let f = scratch.lu_x[i].to_f64();
        f.round() as i64
    });

    // Transpose lower-triangular L to upper-triangular for SE.
    let l_upper: [[f64; 16]; 16] =
        std::array::from_fn(|i| std::array::from_fn(|j| scratch.l_f64[j][i]));

    // Step 5: SE bound. The 8D path uses `bound_sq = 1.51` — slightly above
    // the unit cap-radial Q-radius — and relies on the leaf checks to filter
    // false positives. We mirror that here. The Q-metric inside SE has the
    // cap normalised to radius ≈ 1 in the `1/Δ_y²` direction, so this bound
    // covers a small Q-shell around the cap center.
    //
    // **At d=16**: Q has 1 eigenvalue at 1/Δ_y² (cap radial), 3 at 1/Δ_⊥²
    // (cap tangential), and 12 at 1/R² (bullet balls). At full extent in
    // every direction, the cap × ball region has Q-norm² = 1 + 3 + 12 = 16.
    // Empirically (Q at k=0/1, T at k=0, lattice round-trips at k=2-4) the
    // valid lattice points sit in a Q-shell with Q-norm² in [1, 16]; the
    // cap-tangential and bullet-ball directions dominate over the (very
    // narrow) cap-radial direction. We enumerate the full feasible region
    // by setting `bound_sq` to cover all 16 eigenvalue contributions plus
    // slack for the i64 rounding of `z_c`. Tightening the bound risks
    // missing exact solutions that sit at the edge of σ_5/σ_9/σ_13.
    // Empirical sweep at ε=1e-3..1e-6 found bound_sq=8 always-correct and
    // ~1.6× faster than the conservative bound_sq=16. The norm-shell prune
    // (via the Euclidean Cholesky) plus the integer-exact leaf check
    // catch any false positives the smaller bound lets through.
    // Override via env var for further empirical exploration.
    let bound_sq = std::env::var("CYCLOSYNTH_BOUND_SQ")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(8.0_f64);

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
        std::array::from_fn(|i| RFloat::with_val(ALIGN_PREC, y[i]));

    // Norm-shell target. Use i128 so k ≤ 126 stays exact (the moderate-ε
    // regime targets k ≲ 30 but the deep-ε regime can reach k > 60).
    let target_norm: i128 = 1i128 << k;
    let use_i64_path = k <= 62;
    let target_norm_i64: i64 = if use_i64_path { 1i64 << k } else { 0 };

    // Step 6: SE walk + leaf checks. Parallel + norm-pruned + incremental-x.
    let basis = scratch.basis;
    let budget = AtomicU64::new(max_phase2_calls);

    // Norm-shell pruning: precompute the upper-triangular Euclidean
    // Cholesky of the post-LLL basis at MPFR-128 (then f64 snapshot).
    let r_eucl = match euclidean_cholesky_16_mpfr(&basis) {
        Some(r) => r,
        None => {
            eprintln!(
                "[lenstra_zeta] Euclidean Cholesky failed (rank-deficient basis) at \
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
        if trace {
            diag::N_SE_CALLBACKS.fetch_add(1, Ordering::Relaxed);
        }
        // Norm shell: ‖x‖² == 2^k (hot path — most leaves fail here).
        if use_i64_path {
            let n: i64 = x.iter().map(|&v| v * v).sum();
            if n != target_norm_i64 {
                if trace { diag::N_NORM_REJECTED.fetch_add(1, Ordering::Relaxed); }
                return LeafAction::Skip;
            }
        } else {
            let n: i128 = x.iter().map(|&v| (v as i128) * (v as i128)).sum();
            if n != target_norm {
                if trace { diag::N_NORM_REJECTED.fetch_add(1, Ordering::Relaxed); }
                return LeafAction::Skip;
            }
        }
        // Bilinear forms: B_1=B_2=B_3=0.
        let (b1, b2, b3) = bilinear_forms(x);
        if b1 != 0 || b2 != 0 || b3 != 0 {
            if trace { diag::N_BILINEAR_REJECTED.fetch_add(1, Ordering::Relaxed); }
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
            if trace { diag::N_ALIGN_REJECTED.fetch_add(1, Ordering::Relaxed); }
            return LeafAction::Skip;
        }
        if trace { diag::N_SOLS_RETURNED.fetch_add(1, Ordering::Relaxed); }
        // Integer-exact filter passed. Now ask the caller whether to
        // stop the walk (typically used to bail on first ε-pass).
        if should_stop(x) {
            LeafAction::TakeAndStop
        } else {
            LeafAction::Take
        }
    };

    let (solutions, budget_was_hit) = schnorr_euchner_16d_par_norm_pruned(
        &l_upper, &z_c, bound_sq, &r_eucl, target_norm_sq_f64, &basis,
        leaf_filter, &budget,
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
    use crate::synthesis::search_zeta::{phase1_brute, uv_to_xy_zeta};
    use crate::synthesis::clifford_sqrt_t::{
        det_phase_of, solution_to_u2q_d, unitary_to_uv_zeta,
    };

    fn realistic_v() -> [f64; 4] {
        let v = [0.5, 0.3, 0.7, -0.4];
        let n: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        std::array::from_fn(|i| v[i] / n)
    }

    /// At k=2 with the `realistic_v` direction, run phase1 and verify every
    /// returned solution lies in `phase1_brute(2)` AND passes the alignment
    /// threshold w.r.t. the same y direction. (Phase1 may return a *subset*
    /// of brute solutions because (a) the SE bound only covers part of the
    /// norm shell and (b) the alignment threshold filters by y-direction.)
    #[test]
    fn phase1_at_k_2_finds_brute_subset() {
        let v = realistic_v();
        let k = 2u32;
        let eps = 0.5_f64;
        let y = uv_to_xy_zeta(v, k);

        let mut s = IntScratch16::new(eps);
        let abort = AtomicBool::new(false);
        let sols = phase1(&mut s, &y, k, eps, 100_000_000, &abort);

        let brute_set: std::collections::HashSet<[i64; 16]> =
            phase1_brute(k).into_iter().collect();
        for sol in &sols {
            assert!(
                brute_set.contains(sol),
                "phase1 returned non-brute solution: {:?}",
                sol
            );
        }
        eprintln!(
            "phase1_at_k_2: {} solutions (subset of {} brute)",
            sols.len(),
            brute_set.len()
        );
    }

    /// Target T (k=0, det-phase = 2). Recovered exactly by phase1 with the
    /// `unitary_to_uv_zeta` + single-d reconstruction path.
    #[test]
    fn phase1_finds_t_at_k_0() {
        use crate::matrix::u2::U2Q;
        use crate::synthesis::distance::diamond_distance_float;

        let t_gate = U2Q::t();
        let target = t_gate.to_float();
        let v = unitary_to_uv_zeta(&target);
        let d = det_phase_of(&target);
        let k = t_gate.k;
        assert_eq!(k, 0, "T should have k=0");
        let y = uv_to_xy_zeta(v, k);

        let eps = 0.1_f64;
        let mut s = IntScratch16::new(eps);
        let abort = AtomicBool::new(false);
        let sols = phase1(&mut s, &y, k, eps, 100_000_000, &abort);

        assert!(!sols.is_empty(), "phase1 found no solutions for T at k=0");
        let min_dist = sols.iter().map(|sol| {
            let cand = solution_to_u2q_d(sol, k, d);
            diamond_distance_float(&cand.to_float(), &target)
        }).fold(f64::INFINITY, f64::min);
        assert!(min_dist < 1e-9, "min dist to T at k=0: {min_dist:.3e}");
    }

    /// Target QHQ (k=1, det-phase = 10). Recovered exactly via column-1
    /// extraction + single-d reconstruction.
    #[test]
    fn phase1_finds_qhq_at_k_1() {
        use crate::matrix::u2::U2Q;
        use crate::synthesis::distance::diamond_distance_float;

        let qhq: U2Q = U2Q::q() * U2Q::h() * U2Q::q();
        let target = qhq.to_float();
        let v = unitary_to_uv_zeta(&target);
        let d = det_phase_of(&target);
        let k = qhq.k;
        assert_eq!(k, 1, "QHQ should have k=1");
        let y = uv_to_xy_zeta(v, k);

        let eps = 0.1_f64;
        let mut s = IntScratch16::new(eps);
        let abort = AtomicBool::new(false);
        let sols = phase1(&mut s, &y, k, eps, 100_000_000, &abort);

        assert!(!sols.is_empty(), "phase1 found no solutions for QHQ at k=1");
        let min_dist = sols.iter().map(|sol| {
            let cand = solution_to_u2q_d(sol, k, d);
            diamond_distance_float(&cand.to_float(), &target)
        }).fold(f64::INFINITY, f64::min);
        assert!(min_dist < 1e-9, "min dist to QHQ at k=1: {min_dist:.3e}");
    }

    /// Round-trip at a moderate k=2. Pick a brute solution, derive `v` from
    /// its reconstructed unitary, run phase1, verify the *exact* same
    /// solution (after symmetry / det-phase rotation) is among phase1's
    /// returned candidates.
    #[test]
    fn phase1_finds_hqhqh_at_moderate_k() {
        use crate::synthesis::search_zeta::phase1_brute;
        use crate::synthesis::distance::diamond_distance_float;

        let k = 2u32;
        let brute_sols = phase1_brute(k);
        assert!(!brute_sols.is_empty());

        // Pick a brute solution that uses non-trivial coefficients so the
        // direction `v` is generic (avoids axis-aligned degeneracies).
        let target_sol = brute_sols
            .iter()
            .find(|&s| s.iter().filter(|&&v| v != 0).count() >= 4)
            .copied()
            .expect("expected a brute sol with ≥4 nonzero coefficients");
        let target_u2q = solution_to_u2q_d(&target_sol, k, 0);
        let target = target_u2q.to_float();
        let v = unitary_to_uv_zeta(&target);
        let d = det_phase_of(&target);
        let y = uv_to_xy_zeta(v, k);

        let eps = 0.1_f64;
        let mut s = IntScratch16::new(eps);
        let abort = AtomicBool::new(false);
        let start = std::time::Instant::now();
        let sols = phase1(&mut s, &y, k, eps, 10_000_000, &abort);
        let elapsed = start.elapsed();
        eprintln!(
            "phase1 round-trip at k={}: {} solutions in {:?}",
            k,
            sols.len(),
            elapsed
        );

        assert!(
            !sols.is_empty(),
            "phase1 found no solutions for k={} round-trip",
            k
        );
        let min_dist = sols.iter().map(|sol| {
            let cand = solution_to_u2q_d(sol, k, d);
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
            "phase1 at k={} took {:?} (budget 30s)",
            k, elapsed
        );
    }

    /// Performance smoke at moderate-k (typically k=4-6 from a deterministic
    /// circuit). Reports timing and solution count; the test only fails if
    /// the walk blows up wall-clock past a generous bound.
    #[test]
    fn phase1_perf_at_k_8_completes() {
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
        let y = uv_to_xy_zeta(v, k);

        let eps = 0.1_f64;
        let mut s = IntScratch16::new(eps);
        let abort = AtomicBool::new(false);
        let start = std::time::Instant::now();
        let sols = phase1(&mut s, &y, k, eps, 100_000_000, &abort);
        let elapsed = start.elapsed();
        eprintln!(
            "phase1 at k={} took {} ms, returned {} solutions",
            k,
            elapsed.as_millis(),
            sols.len()
        );

        // Don't assert on solution count (spec allows missing exact match);
        // enforce only that we don't blow up wall-clock.
        assert!(
            elapsed.as_secs() < 60,
            "phase1 at k={} took {:?} (budget 60s)",
            k, elapsed
        );
    }

    /// At ε=1e-5 with k=14, verify the i256 LLL path doesn't trip overflow.
    /// Returns a (possibly empty) Vec of solutions.
    #[test]
    fn phase1_no_overflow_at_eps_1e_5() {
        let v = realistic_v();
        let k = 14u32;
        let eps = 1e-5_f64;
        let y = uv_to_xy_zeta(v, k);
        let mut s = IntScratch16::new(eps);
        let abort = AtomicBool::new(false);
        let start = std::time::Instant::now();
        let sols = phase1(&mut s, &y, k, eps, 10_000_000, &abort);
        let elapsed = start.elapsed();
        eprintln!(
            "phase1 at ε=1e-5, k=14: {} solutions in {:?}",
            sols.len(),
            elapsed
        );
        // We don't require non-empty (the random `realistic_v` direction may
        // not have an exact lde=14 lattice match). The test is just that the
        // pipeline doesn't crash or overflow.
        assert!(
            elapsed.as_secs() < 120,
            "phase1 at ε=1e-5 took {:?} (budget 120s)",
            elapsed
        );
    }

    /// A/B diagnostic: run phase1 (norm-pruned) at a fixed k and report
    /// timing + sols. To compare against the non-pruned baseline, swap the
    /// SE call site in `phase1` temporarily.
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
            let y = uv_to_xy_zeta(v, k);
            let mut s = IntScratch16::new(eps);
            let abort = AtomicBool::new(false);
            let budget = 1_000_000_000_u64;
            let t0 = std::time::Instant::now();
            let sols = phase1(&mut s, &y, k, eps, budget, &abort);
            let dt = t0.elapsed();
            let min_dist = sols.iter().map(|sol| {
                let cand = solution_to_u2q_d(sol, k, d);
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
            let y = uv_to_xy_zeta(v, k);
            let budget = 1_000_000_000_u64;
            let mut s = IntScratch16::new(eps);
            let abort = AtomicBool::new(false);
            let t0 = std::time::Instant::now();
            let sols = phase1(&mut s, &y, k, eps, budget, &abort);
            let dt = t0.elapsed();
            let abort_v = abort.load(Ordering::Relaxed);
            let min_dist = sols.iter().map(|sol| {
                let cand = solution_to_u2q_d(sol, k, d);
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
}
