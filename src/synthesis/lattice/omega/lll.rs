//! L²-LLL (Nguyen-Stehlé 2009) inner loop, plus the integer Gram update
//! helpers it relies on.
//!
//! All routines below operate on the f64 scratch fields (`r_bar`, `mu_bar`,
//! `s_bar`) and read the EXACT integer Gram (i256) on demand. Theorem 2 of
//! the paper proves f64 suffices for d ≤ 11 at (δ=0.75, η=0.55); we operate
//! at d=8 with ~18-bit precision margin.

#![allow(clippy::needless_range_loop)]

use super::scratch::IntScratch;
use crate::synthesis::lattice::common;

// ─── L²-LLL parameters & result type — moved to lattice::common ───────────────

pub use crate::synthesis::lattice::common::{
    L2_DELTA, L2_DELTA_BAR, L2_ETA, L2_ETA_BAR, LllResult, MAX_LAZY_PASSES,
};

// ─── i256 → f64 conversion (used by CFA on the exact Gram) ───────────────────

pub use crate::synthesis::lattice::common::i256_to_f64;

// ─── Cholesky Factorization Algorithm (Figure 4) ─────────────────────────────

/// Row-at-a-time Cholesky factorization (Figure 4 of Nguyen-Stehlé 2009):
/// fills `r_bar[i][*]`, `mu_bar[i][*]`, `s_bar[i][*]` in f64, reading the
/// exact Gram via `i256_to_f64`. Assumes rows 0..i are already filled —
/// the L² main loop calls this at each new κ.
#[inline]
pub fn cfa_row(scratch: &mut IntScratch, i: usize) {
    for j in 0..i {
        let mut r = i256_to_f64(scratch.gram[i][j]);
        for k in 0..j {
            r -= scratch.mu_bar[j][k] * scratch.r_bar[i][k];
        }
        scratch.r_bar[i][j] = r;
        let r_jj = scratch.r_bar[j][j];
        scratch.mu_bar[i][j] = if r_jj.abs() < 1e-300 { 0.0 } else { r / r_jj };
    }
    scratch.s_bar[i][0] = i256_to_f64(scratch.gram[i][i]);
    for j in 1..=i {
        scratch.s_bar[i][j] =
            scratch.s_bar[i][j - 1] - scratch.mu_bar[i][j - 1] * scratch.r_bar[i][j - 1];
    }
    scratch.r_bar[i][i] = scratch.s_bar[i][i];
}

/// Run CFA for all 8 rows in order.
pub fn cfa_full(scratch: &mut IntScratch) {
    for i in 0..8 {
        cfa_row(scratch, i);
    }
}

// ─── Lazy size-reduce (Figure 5) ─────────────────────────────────────────────

/// Lazy floating-point size-reduction (Figure 5 of Nguyen-Stehlé 2009).
///
/// Reduces row κ against rows 0..κ-1 such that `|μ̄_{κ,j}| ≤ η̄` for all
/// `j < κ`, where η̄ = (η + 1/2) / 2. Operates iteratively: each pass
/// computes CFA for row κ, predicts X_i = round(μ̄_{κ,i}), updates μ̄_{κ,j}
/// predictively, then applies the basis transform `b_κ -= Σ X_i b_i` and
/// updates the i256 Gram. Repeats until convergence.
///
/// Per Theorem 3 the f64 precision requirement is satisfied when rows
/// 0..κ-1 are already L³-reduced — the L² main loop maintains this.
///
/// Returns a value < MAX_LAZY_PASSES on convergence (the pass index) and
/// MAX_LAZY_PASSES on non-convergence; callers detect the latter by `==`.
/// The cap never fires in practice; it guards pathological inputs.
pub fn lazy_size_reduce(scratch: &mut IntScratch, kappa: usize) -> usize {
    let mut x = [0i64; 8];

    for pass in 0..MAX_LAZY_PASSES {
        cfa_row(scratch, kappa);

        let mut max_mu: f64 = 0.0;
        for j in 0..kappa {
            let m = scratch.mu_bar[kappa][j].abs();
            if m > max_mu {
                max_mu = m;
            }
        }
        if max_mu <= L2_ETA_BAR {
            if crate::synthesis::diag::trace_enabled() {
                crate::synthesis::diag::record_lazy_passes((pass + 1) as u64);
            }
            return pass;
        }

        // X_i = round(μ̄_{κ,i}) descending from κ-1, shrinking μ̄_{κ,j} as we go.
        for i in (0..kappa).rev() {
            let xi = scratch.mu_bar[kappa][i].round() as i64;
            x[i] = xi;
            if xi != 0 {
                let xi_f = xi as f64;
                for j in 0..i {
                    scratch.mu_bar[kappa][j] -= xi_f * scratch.mu_bar[i][j];
                }
            }
        }

        // gram_update_size_reduce encodes M·G·Mᵀ for one (k,j,r) triple; call
        // it sequentially per non-zero x[i] so the chained updates compose.
        for i in 0..kappa {
            if x[i] != 0 {
                for c in 0..8 {
                    scratch.basis[kappa][c] -= x[i] * scratch.basis[i][c];
                }
                gram_update_size_reduce(scratch, kappa, i, x[i]);
                x[i] = 0;
            }
        }
    }
    if crate::synthesis::diag::trace_enabled() {
        crate::synthesis::diag::record_lazy_passes(MAX_LAZY_PASSES as u64);
    }
    MAX_LAZY_PASSES
}

// ─── Incremental Gram update for size-reduce + swap ──────────────────────────

/// Apply the basis transform `b_k -= r·b_j` to the Gram in O(16) i256 ops
/// instead of O(8³) for a full recompute. Math: `B_new = M·B` where
/// `M = I − r·E_kj`, hence `G_new = M·G·Mᵀ`. Two-step recurrence (row-k
/// update, then column-k update).
///
/// Caller must call this AFTER updating the i64 basis row k. Idempotent for
/// r=0.
#[inline]
pub(super) fn gram_update_size_reduce(scratch: &mut IntScratch, k: usize, j: usize, r: i64) {
    common::gram_update_size_reduce(&mut scratch.gram, k, j, r);
}

/// L² INSERT operation (Figure 6 step 6 of Nguyen-Stehlé 2009): move basis
/// row `kappa_orig` to position `kappa_insert` (≤ kappa_orig). After basis +
/// Gram are rotated, the GS state for row kappa_insert is stale: caller must
/// invoke `cfa_row(scratch, kappa_insert)`. Rows above kappa_insert are
/// recomputed naturally as κ advances and lazy_size_reduce calls CFA.
#[inline]
fn basis_insert(scratch: &mut IntScratch, kappa_orig: usize, kappa_insert: usize) {
    common::basis_insert(&mut scratch.gram, &mut scratch.basis, kappa_orig, kappa_insert);
}

// ─── Full Gram computation: G = B · Q_int · Bᵀ ───────────────────────────────

/// Compute G = B · Q_int · Bᵀ entirely in i256, into `scratch.gram` (via
/// `scratch.temp_bq`). Returns `false` on Gram overflow so the caller aborts
/// to fallback.
///
/// Overflow margin: with max |Q_int| = 2^180 and post-LLL max(|B|) ≤ 2^15, G
/// entries fit ≤ 2^216 (40-bit margin to i256::MAX). Transient B-growth
/// during deep-ε swaps can breach the 2^GRAM_OVERFLOW_THRESHOLD_BITS guard.
#[inline]
pub fn compute_gram_full(scratch: &mut IntScratch) -> bool {
    common::compute_gram_full(
        &mut scratch.gram,
        &scratch.basis,
        &scratch.q_int,
        &mut scratch.temp_bq,
    )
}

/// Check whether any Gram entry exceeds the overflow threshold.
#[inline]
fn gram_overflow_check(scratch: &IntScratch) -> bool {
    common::gram_overflow_check(&scratch.gram)
}

// ─── L²-LLL main loop (Figure 6) ─────────────────────────────────────────────

/// L²-LLL (Nguyen-Stehlé 2009, Figure 6) over the 8×8 Q-metric. Builds the
/// integer Gram from `scratch.q_int` internally and records the reduced
/// basis in `scratch.basis`; GS state lives in `r_bar`/`mu_bar`/`s_bar`.
///
/// The algorithm walks rows κ = 1..8, maintaining the invariant that rows
/// 0..κ-1 are (δ, η)-L³-reduced. At each κ:
///   1. Lazily size-reduce row κ (interleaved CFA + basis reduction) until
///      `|μ̄_{κ,j}| ≤ η̄` for all j < κ.
///   2. Find the deepest insertion position κ_insert via Lovász cascade.
///   3. If κ_insert < κ, rotate the basis (and Gram) so the reduced row
///      lands at κ_insert; otherwise leave it.
///   4. Advance κ.
///
/// Per Theorem 3, f64 precision (53 mantissa bits) suffices for d=8 at
/// (δ=0.75, η=0.55): the required precision is `ℓ ≥ 5 + 2·log d − log ε +
/// d·log ρ ≈ 34 bits`, leaving ~18 bits of margin. The L³-reduction
/// invariant on the prefix is what makes the f64 GS coefficients accurate
/// enough; running CFA on an unreduced basis would suffer catastrophic
/// cancellation at deep ε.
pub fn lll_l2_8(scratch: &mut IntScratch) -> LllResult {
    lll_l2_8_seeded(scratch, None).0
}

/// `lll_l2_8` with an optional warm-start `seed` basis, which must be
/// unimodular (e.g. the reduced basis of the prefix-independent `Q_base`
/// metric at the same `(k, ε)`): any unimodular basis of ℤ⁸ reduces to the
/// SAME lattice, so downstream (det ±1, Cholesky, LU, SE) is unaffected.
/// Returns the LLL iteration count.
pub fn lll_l2_8_seeded(
    scratch: &mut IntScratch,
    seed: Option<&super::scratch::IMat8>,
) -> (LllResult, usize) {
    match seed {
        Some(b) => scratch.basis = *b,
        None => scratch.reset_basis(),
    }
    let max_iter = common::MAX_LLL_ITERS_8D;
    let mut iters: usize = 0;

    // Gram from the current basis (= Q_int when reset to identity, else B·Q_int·Bᵀ).
    if !compute_gram_full(scratch) {
        if crate::synthesis::diag::trace_enabled() {
            crate::synthesis::diag::record_lll_iters(iters as u64, max_iter as u64);
        }
        return (LllResult::GramOverflow, iters);
    }

    cfa_row(scratch, 0); // r̄_{0,0} = ‖b_0‖²
    let mut kappa = 1usize;

    while kappa < 8 && iters < max_iter {
        iters += 1;

        // Step 3: lazy size-reduce row κ. Updates basis (i64) + Gram (i256)
        // and refreshes r_bar/mu_bar/s_bar for row κ.
        let _passes = lazy_size_reduce(scratch, kappa);

        if gram_overflow_check(scratch) {
            if crate::synthesis::diag::trace_enabled() {
                crate::synthesis::diag::record_lll_iters(iters as u64, max_iter as u64);
            }
            return (LllResult::GramOverflow, iters);
        }

        // Step 4: Lovász cascade. Find deepest position κ_insert where the
        // size-reduced row κ_orig should be inserted. Use s̄_{κ-1}^{(κ_orig)}
        // (partial CFA sum at depth κ-1 for the orig-frontier row) as the
        // projected GS norm² at insertion depth κ.
        let kappa_orig = kappa;
        while kappa >= 1
            && L2_DELTA_BAR * scratch.r_bar[kappa - 1][kappa - 1]
                > scratch.s_bar[kappa_orig][kappa - 1]
        {
            if kappa <= 1 {
                kappa = 0;
                break;
            }
            kappa -= 1;
        }

        // If the insertion position is shallower than the current frontier,
        // rotate the basis and Gram so the (now-reduced) frontier row lands
        // at the new position, then recompute that row's GS state from the
        // updated Gram.
        if kappa < kappa_orig {
            basis_insert(scratch, kappa_orig, kappa);
            cfa_row(scratch, kappa);
        }
        kappa += 1;
    }

    if crate::synthesis::diag::trace_enabled() {
        crate::synthesis::diag::record_lll_iters(iters as u64, max_iter as u64);
    }
    let res = if iters >= max_iter {
        LllResult::IterCap
    } else {
        LllResult::Converged
    };
    (res, iters)
}
