//! L²-LLL (Nguyen-Stehlé 2009) inner loop, plus the integer Gram update
//! helpers it relies on.
//!
//! All routines below operate on the f64 scratch fields (`r_bar`, `mu_bar`,
//! `s_bar`) and read the EXACT integer Gram (i256) on demand. Theorem 2 of
//! the paper proves f64 suffices for d ≤ 11 at (δ=0.75, η=0.55); we operate
//! at d=8 with ~18-bit precision margin.

#![allow(clippy::needless_range_loop)]

use i256::i256;

use super::scratch::{IntScratch, GRAM_OVERFLOW_THRESHOLD_BITS};

// ─── L²-LLL parameters & result type — moved to lenstra_common ───────────────

pub use crate::synthesis::lenstra_common::{
    L2_DELTA, L2_DELTA_BAR, L2_ETA, L2_ETA_BAR, LllResult, MAX_LAZY_PASSES,
};

// ─── i256 → f64 conversion (used by CFA on the exact Gram) ───────────────────

/// Convert i256 to f64. f64 has 53 mantissa bits + 11 exponent bits (range
/// 2^±1023). Our gram values are bounded by ≈ 2^240, well within range.
/// Mantissa rounding: low bits beyond 53 are dropped (round-to-nearest-even).
/// L² requires only ≈ 20 bits of precision per Theorem 2 — f64 gives 53 with
/// no overflow risk for our magnitudes.
#[inline]
pub fn i256_to_f64(v: i256) -> f64 {
    let zero = i256::from_i64(0);
    if v == zero {
        return 0.0;
    }
    let neg = v < zero;
    let abs = if neg { -v } else { v };
    let bytes = abs.to_le_bytes();
    let l0 = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
    let l1 = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
    let l2 = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
    let l3 = u64::from_le_bytes(bytes[24..32].try_into().unwrap());
    // Combine in increasing-precision order so the accumulation rounds the
    // low bits, not the high bits.
    let result = (l0 as f64)
        + (l1 as f64) * 2f64.powi(64)
        + (l2 as f64) * 2f64.powi(128)
        + (l3 as f64) * 2f64.powi(192);
    if neg { -result } else { result }
}

// ─── Cholesky Factorization Algorithm (Figure 4) ─────────────────────────────

/// Row-at-a-time CFA (Figure 4 of Nguyen-Stehlé 2009). Computes
/// `r_bar[i][*]`, `mu_bar[i][*]`, `s_bar[i][*]` given rows 0..i are already
/// populated. All arithmetic in f64; reads gram entries via `i256_to_f64`.
///
/// Per Figure 4 (with our 0-indexed convention):
///   For j = 0..i-1:
///     r̄_{i,j} ← <b_i, b_j>    (from i256 Gram)
///     For k = 0..j-1: r̄_{i,j} ← r̄_{i,j} - μ̄_{j,k} · r̄_{i,k}
///     μ̄_{i,j} ← r̄_{i,j} / r̄_{j,j}
///   s̄_{i,0} ← <b_i, b_i>
///   For j = 1..=i: s̄_{i,j} ← s̄_{i,j-1} - μ̄_{i,j-1} · r̄_{i,j-1}
///   r̄_{i,i} ← s̄_{i,i}
///
/// IMPORTANT: assumes rows 0..i are already filled by prior `cfa_row` calls
/// (or by initial setup). The L² main loop calls this at each new κ.
#[inline]
pub fn cfa_row(scratch: &mut IntScratch, i: usize) {
    // Off-diagonal entries: j = 0..i-1
    for j in 0..i {
        let mut r = i256_to_f64(scratch.gram[i][j]);
        for k in 0..j {
            r -= scratch.mu_bar[j][k] * scratch.r_bar[i][k];
        }
        scratch.r_bar[i][j] = r;
        // μ̄_{i,j} = r̄_{i,j} / r̄_{j,j}
        let r_jj = scratch.r_bar[j][j];
        scratch.mu_bar[i][j] = if r_jj.abs() < 1e-300 { 0.0 } else { r / r_jj };
    }
    // Diagonal: s̄_{i,*} sequence, r̄_{i,i} = s̄_{i,i}
    scratch.s_bar[i][0] = i256_to_f64(scratch.gram[i][i]);
    for j in 1..=i {
        scratch.s_bar[i][j] =
            scratch.s_bar[i][j - 1] - scratch.mu_bar[i][j - 1] * scratch.r_bar[i][j - 1];
    }
    scratch.r_bar[i][i] = scratch.s_bar[i][i];
}

/// Run CFA for ALL rows 0..d. Equivalent to `cfa_row` for i = 0..d-1 in order.
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
/// Per Theorem 3 the f64 precision requirement (ℓ=52) is satisfied when
/// rows 0..κ-1 are already L³-reduced — the L² main loop maintains this
/// invariant.
///
/// Returns the number of passes used. Caller can detect non-convergence via
/// MAX_LAZY_PASSES — never expected to fire in practice; the hard bound
/// guards against pathological inputs.
pub fn lazy_size_reduce(scratch: &mut IntScratch, kappa: usize) -> usize {
    let mut x = [0i64; 8];

    for pass in 0..MAX_LAZY_PASSES {
        // Step 2: compute CFA for row κ (reads i256 Gram via i256_to_f64).
        cfa_row(scratch, kappa);

        // Step 3: convergence check.
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

        // Steps 4-5: compute X_i values descending from κ-1 to 0,
        // predictively shrinking μ̄_{κ,j} as we go down.
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

        // Step 6: apply basis update and Gram update for each non-zero X_i.
        // gram_update_size_reduce already encodes M·G·Mᵀ for one (k, j, r)
        // triple; we call it sequentially for each non-zero x[i] so the
        // chain of updates produces the correct final Gram.
        for i in 0..kappa {
            if x[i] != 0 {
                for c in 0..8 {
                    scratch.basis[kappa][c] -= x[i] * scratch.basis[i][c];
                }
                gram_update_size_reduce(scratch, kappa, i, x[i]);
                x[i] = 0; // clear for next pass
            }
        }
        // Step 7: goto step 2.
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
pub(super) fn gram_update_size_reduce(scratch: &mut IntScratch, k: usize, j: usize, r: i64) {
    if r == 0 {
        return;
    }
    let r256 = i256::from_i64(r);
    // Step 1: row k. G[k][m] := G[k][m] - r·G[j][m] for m = 0..8.
    // Snapshot row j BEFORE mutating row k (the new G[k][k] depends on G[j][k]).
    let row_j_snapshot: [i256; 8] = scratch.gram[j];
    for m in 0..8 {
        scratch.gram[k][m] -= r256 * row_j_snapshot[m];
    }
    // Step 2: column k. G[i][k] := G[i][k] - r·G[i][j] for i = 0..8.
    // For i ≠ k: G[i][j] is unchanged from before (step 1 only touched row k).
    // For i = k: G[k][j] was updated in step 1 — we use the post-update value
    // here, which gives the correct G_new[k][k] derivation.
    let mut col_j_snapshot = [i256::from_i64(0); 8];
    for i in 0..8 {
        col_j_snapshot[i] = scratch.gram[i][j];
    }
    for i in 0..8 {
        scratch.gram[i][k] -= r256 * col_j_snapshot[i];
    }
}

/// Apply the basis swap of rows a and b to the (symmetric) Gram: swap rows
/// AND columns. O(8) work.
pub(super) fn gram_update_swap(scratch: &mut IntScratch, a: usize, b: usize) {
    if a == b {
        return;
    }
    scratch.gram.swap(a, b);
    for i in 0..8 {
        scratch.gram[i].swap(a, b);
    }
}

/// L² INSERT operation (Figure 6 step 6 of Nguyen-Stehlé 2009).
///
/// Move basis row `kappa_orig` to position `kappa_insert` (≤ kappa_orig).
/// Rows kappa_insert..kappa_orig-1 shift down by one. Implemented as a chain
/// of adjacent swaps so the i256 Gram is kept consistent via
/// `gram_update_swap`. After basis + Gram are rotated, the GS state for row
/// kappa_insert is stale: caller must invoke `cfa_row(scratch, kappa_insert)`.
///
/// Cost: O(kappa_orig - kappa_insert) adjacent swaps, each O(d) for the
/// gram column swap.
fn basis_insert(scratch: &mut IntScratch, kappa_orig: usize, kappa_insert: usize) {
    debug_assert!(kappa_insert <= kappa_orig);
    let mut current = kappa_orig;
    while current > kappa_insert {
        scratch.basis.swap(current, current - 1);
        gram_update_swap(scratch, current, current - 1);
        current -= 1;
    }
    // Note: GS state (r_bar, mu_bar, s_bar) for rows kappa_insert..kappa_orig
    // is now stale. The L² main loop must refresh row kappa_insert via
    // cfa_row before the next iteration uses it. Rows above kappa_insert
    // are recomputed naturally as κ advances and lazy_size_reduce calls CFA.
}

// ─── Full Gram computation: G = B · Q_int · Bᵀ ───────────────────────────────

/// Compute G = B · Q_int · Bᵀ entirely in i256, into `scratch.gram`. Uses
/// `scratch.temp_bq` as intermediate (= B · Q_int).
///
/// **Overflow analysis**: max |Q_int| = 2^TARGET_BITS = 2^180 by `build_q_int`.
/// For typical post-LLL max(|B[i][j]|) ≤ 2^15, intermediate B·Q_int entries
/// fit ≤ 2^198, final G entries fit ≤ 2^216 → safe with 40-bit margin to
/// i256::MAX. For transient B-growth during LLL swaps, max(|B|) can spike to
/// ~2^40 at deep ε; G entries can then approach 2^260 (overflow). Returns
/// `false` if any Gram entry exceeds 2^GRAM_OVERFLOW_THRESHOLD_BITS, so the
/// LLL caller can abort and trigger fallback.
pub fn compute_gram_full(scratch: &mut IntScratch) -> bool {
    let zero = i256::from_i64(0);

    // temp_bq[i][b] = sum_a B[i][a] · Q_int[a][b]
    for i in 0..8 {
        for b in 0..8 {
            let mut acc = zero;
            for a in 0..8 {
                let bi_a = i256::from_i64(scratch.basis[i][a]);
                acc += bi_a * scratch.q_int[a][b];
            }
            scratch.temp_bq[i][b] = acc;
        }
    }

    // gram[i][j] = sum_b temp_bq[i][b] · B[j][b]
    let mut max_abs_log2: i32 = -1;
    for i in 0..8 {
        for j in 0..8 {
            let mut acc = zero;
            for b in 0..8 {
                let bj_b = i256::from_i64(scratch.basis[j][b]);
                acc += scratch.temp_bq[i][b] * bj_b;
            }
            scratch.gram[i][j] = acc;
            // Magnitude check (cheap: leading-zero count on the |i256|).
            let bits = i256_log2_ceil(&acc);
            if bits > max_abs_log2 {
                max_abs_log2 = bits;
            }
        }
    }
    max_abs_log2 <= GRAM_OVERFLOW_THRESHOLD_BITS as i32
}

/// Bit count of |v| (≈ ⌈log₂(|v|)⌉, returns -1 for v=0).
fn i256_log2_ceil(v: &i256) -> i32 {
    let zero = i256::from_i64(0);
    if *v == zero {
        return -1;
    }
    let abs = if *v < zero { -*v } else { *v };
    let bytes = abs.to_le_bytes();
    let mut leading_zeros: u32 = 0;
    for byte in bytes.iter().rev() {
        if *byte == 0 {
            leading_zeros += 8;
        } else {
            leading_zeros += byte.leading_zeros();
            break;
        }
    }
    (256 - leading_zeros as i32) - 1
}

/// Check whether any Gram entry exceeds the overflow threshold.
fn gram_overflow_check(scratch: &IntScratch) -> bool {
    let thresh = GRAM_OVERFLOW_THRESHOLD_BITS as i32;
    for i in 0..8 {
        for j in 0..8 {
            if i256_log2_ceil(&scratch.gram[i][j]) > thresh {
                return true;
            }
        }
    }
    false
}

// ─── L²-LLL main loop (Figure 6) ─────────────────────────────────────────────

/// L²-LLL (Nguyen-Stehlé 2009, Figure 6) on the 8×8 Q-metric Gram already
/// snapshotted into `scratch.gram`. Builds an LLL-reduced basis recorded in
/// `scratch.basis`; intermediate state lives in `scratch.r_bar` /
/// `mu_bar` / `s_bar` and the i256 `gram`.
///
/// The algorithm walks rows κ = 1..d, maintaining the invariant that rows
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
    scratch.reset_basis();
    let max_iter: usize = 10_000;
    let mut iters: usize = 0;

    // Step 1: compute exact integer Gram. Basis = identity → Gram = Q_int.
    if !compute_gram_full(scratch) {
        if crate::synthesis::diag::trace_enabled() {
            crate::synthesis::diag::record_lll_iters(iters as u64, max_iter as u64);
        }
        return LllResult::GramOverflow;
    }

    // Step 2: initialize r̄_{0,0} = ‖b_0‖² (CFA on row 0).
    cfa_row(scratch, 0);
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
            return LllResult::GramOverflow;
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
    if iters >= max_iter {
        LllResult::IterCap
    } else {
        LllResult::Converged
    }
}
