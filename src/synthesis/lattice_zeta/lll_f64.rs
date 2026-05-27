//! Experimental f64-only Gram-Schmidt LLL for the 16D Z[ζ_16] lattice.
//!
//! Mirrors [`super::lll`]'s structure but uses plain `f64` for the
//! Gram-Schmidt state (`r_bar_f64`, `mu_bar_f64`, `s_bar_f64` on
//! [`super::scratch::IntScratch16`]) instead of MPFR `RFloat`.
//!
//! ## Why this might work
//!
//! Theorem 2 of Nguyen-Stehlé 2009 covers d ≤ 11 in f64 at the L²
//! parameters (δ=0.75, η=0.55). At d=16 the proof's headroom disappears
//! (precision requirement ~50 bits at ε=1e-7, leaving f64 with no margin).
//! **However**, fplll's `wrapper.cpp` tries `double` first at every
//! dimension and only escalates on failure — empirically successful most
//! of the time.
//!
//! ## Why this might fail
//!
//! At deep ε the post-LLL Gram has condition number κ(G) ≈ 2^(2k_total)
//! ≈ 2^60 at ε=1e-9. The GS state of an unreduced row carries this
//! condition number; f64 then sees ~60 bits of cancellation in
//! `r_bar[i][j] − Σ μ_bar·r_bar`, leaving 53−60 = -7 useful bits.
//! Lazy size-reduce *could* iterate forever or produce wrong basis
//! transforms.
//!
//! ## How we detect failure
//!
//! The caller wraps this in a precision ladder: try f64 first; if the
//! returned basis is non-unimodular, or if `lazy_size_reduce` cycles
//! (hits MAX_LAZY_PASSES on too many κ values), fall back to MPFR.
//!
//! ## What's reused from `super::lll`
//!
//! All integer-arithmetic helpers: `gram_update_size_reduce`,
//! `gram_update_swap`, `basis_insert`, `compute_gram_full`,
//! `gram_overflow_check`, `i256_log2_ceil`. These are MPFR-free.

#![allow(clippy::needless_range_loop)]

use i256::i256;

use super::scratch::IntScratch16;
use crate::synthesis::lattice_common::{LllResult, L2_DELTA_BAR, L2_ETA_BAR, MAX_LAZY_PASSES};

/// Convert i256 to f64. Combines limbs in increasing-precision order so
/// low bits round, not high.
///
/// **Hot path notes**:
/// - Use `i256::is_negative` (a single sign-bit check) instead of
///   constructing `i256::from_i64(0)` and comparing.
/// - Skip the early-zero return; the f64 path is monomorphic and zero
///   inputs land at 0.0 naturally with no extra cost.
/// - Pull limbs via `to_ne_limbs` (direct array access) instead of
///   `to_le_bytes` + 4×`u64::from_le_bytes`.
/// - Hoist `2^64`/`2^128`/`2^192` to `const`s so the compiler folds them.
///
/// Tried (and abandoned): two's-complement direct conversion (signed
/// high limb, unsigned low limbs). Catastrophic cancellation for small
/// negative values whose high limb is `0xFF...FF` — subtracts two
/// near-equal large f64 numbers, loses all precision below ~2^140. Must
/// take abs() in i256 first to keep mantissa precision.
#[inline]
pub fn i256_to_f64(v: i256) -> f64 {
    const SCALE_64: f64 = 18446744073709551616.0; // 2^64
    const SCALE_128: f64 = SCALE_64 * SCALE_64; // 2^128
    const SCALE_192: f64 = SCALE_128 * SCALE_64; // 2^192
    let neg = v.is_negative();
    let abs = if neg { -v } else { v };
    let limbs = abs.to_ne_limbs();
    let r = (limbs[0] as f64)
        + (limbs[1] as f64) * SCALE_64
        + (limbs[2] as f64) * SCALE_128
        + (limbs[3] as f64) * SCALE_192;
    if neg {
        -r
    } else {
        r
    }
}

/// f64 CFA: same algorithm as `super::lll::cfa_row` but operates on the
/// `_f64` state buffers.
#[inline]
pub fn cfa_row_f64(scratch: &mut IntScratch16, i: usize) {
    for j in 0..i {
        let mut r = i256_to_f64(scratch.gram[i][j]);
        for k in 0..j {
            r -= scratch.mu_bar_f64[j][k] * scratch.r_bar_f64[i][k];
        }
        scratch.r_bar_f64[i][j] = r;
        let r_jj = scratch.r_bar_f64[j][j];
        scratch.mu_bar_f64[i][j] = if r_jj.abs() < 1e-300 { 0.0 } else { r / r_jj };
    }
    scratch.s_bar_f64[i][0] = i256_to_f64(scratch.gram[i][i]);
    for j in 1..=i {
        scratch.s_bar_f64[i][j] = scratch.s_bar_f64[i][j - 1]
            - scratch.mu_bar_f64[i][j - 1] * scratch.r_bar_f64[i][j - 1];
    }
    scratch.r_bar_f64[i][i] = scratch.s_bar_f64[i][i];
}

/// f64 lazy size-reduce. Returns the number of passes used.
pub fn lazy_size_reduce_f64(scratch: &mut IntScratch16, kappa: usize) -> usize {
    let mut x = [0i64; 16];
    for pass in 0..MAX_LAZY_PASSES {
        cfa_row_f64(scratch, kappa);
        let mut max_mu: f64 = 0.0;
        for j in 0..kappa {
            let m = scratch.mu_bar_f64[kappa][j].abs();
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
        for i in (0..kappa).rev() {
            let xi = scratch.mu_bar_f64[kappa][i].round() as i64;
            x[i] = xi;
            if xi != 0 {
                let xi_f = xi as f64;
                for j in 0..i {
                    scratch.mu_bar_f64[kappa][j] -= xi_f * scratch.mu_bar_f64[i][j];
                }
            }
        }
        for i in 0..kappa {
            if x[i] != 0 {
                for c in 0..16 {
                    scratch.basis[kappa][c] -= x[i] * scratch.basis[i][c];
                }
                super::lll::gram_update_size_reduce(scratch, kappa, i, x[i]);
                x[i] = 0;
            }
        }
    }
    if crate::synthesis::diag::trace_enabled() {
        crate::synthesis::diag::record_lazy_passes(MAX_LAZY_PASSES as u64);
    }
    MAX_LAZY_PASSES
}

/// f64 L²-LLL main loop. Mirrors `super::lll::lll_l2_16`.
pub fn lll_l2_16_f64(scratch: &mut IntScratch16) -> LllResult {
    let max_iter: usize = super::lll::MAX_LLL_ITERS;
    let mut iters: usize = 0;

    cfa_row_f64(scratch, 0);
    let mut kappa = 1usize;

    while kappa < 16 && iters < max_iter {
        iters += 1;

        let _passes = lazy_size_reduce_f64(scratch, kappa);

        if super::lll::gram_overflow_check(scratch) {
            if crate::synthesis::diag::trace_enabled() {
                crate::synthesis::diag::record_lll_iters(iters as u64, max_iter as u64);
            }
            return LllResult::GramOverflow;
        }

        // Lovász cascade. Walks κ down comparing `δ̄ · r̄_{κ-1,κ-1}` against
        // `s̄_{κ_orig, κ-1}` (the prefix-sum array, populated in cfa_row).
        // Empirically cascades only 1.86 levels deep on average — lazy
        // muls beat precomputing all 16 `δ̄·r̄_{i,i}` upfront.
        let kappa_orig = kappa;
        loop {
            if kappa == 0 {
                break;
            }
            let lhs = L2_DELTA_BAR * scratch.r_bar_f64[kappa - 1][kappa - 1];
            let rhs = scratch.s_bar_f64[kappa_orig][kappa - 1];
            if lhs <= rhs {
                break;
            }
            if kappa <= 1 {
                kappa = 0;
                break;
            }
            kappa -= 1;
        }

        if kappa < kappa_orig {
            super::lll::basis_insert(scratch, kappa_orig, kappa);
            cfa_row_f64(scratch, kappa);
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

/// f64 entry-point analog of `super::lll::run_lll_16`. Honors
/// `scratch.warm_lll` for the Z1 D&C warm-start path.
pub fn run_lll_16_f64(scratch: &mut IntScratch16) -> LllResult {
    if !scratch.warm_lll {
        scratch.reset_basis();
    }
    if !super::lll::compute_gram_full(scratch) {
        return LllResult::GramOverflow;
    }
    lll_l2_16_f64(scratch)
}
