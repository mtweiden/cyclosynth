//! Experimental f64 Gram-Schmidt LLL for the 16D Z[ζ_16] lattice — the
//! `super::lll` structure on the `_f64` state buffers of
//! [`super::scratch::IntScratch16`] instead of MPFR. d=16 is outside the
//! NS09 f64 proof (d ≤ 11), and at deep ε the post-LLL Gram's condition
//! number can exceed f64's mantissa, so lazy size-reduce may cycle or
//! mis-transform. The caller (`run_lll_ladder`) tries f64 first and
//! escalates to MPFR on a non-unimodular result or `IterCap`. The
//! MPFR-free integer helpers (`gram_update_size_reduce`, `basis_insert`,
//! `compute_gram_full`, `gram_overflow_check`) are reused from `super::lll`.

#![allow(clippy::needless_range_loop)]

use super::scratch::IntScratch16;
use crate::synthesis::lattice_common::{LllResult, L2_DELTA_BAR, L2_ETA_BAR, MAX_LAZY_PASSES};

pub use crate::synthesis::lattice_common::i256_to_f64;

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

/// f64 lazy size-reduce. Returns a value < MAX_LAZY_PASSES on convergence
/// (the pass index) and MAX_LAZY_PASSES on non-convergence.
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

        // Lovász cascade: walk κ down comparing `δ̄ · r̄_{κ-1,κ-1}` against
        // the prefix-sum `s̄_{κ_orig, κ-1}`. It descends ~2 levels on
        // average, so lazy muls beat precomputing all 16 `δ̄·r̄_{i,i}`.
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

/// f64 analog of `super::lll::run_lll_16`. Honors `scratch.warm_lll` for
/// the prefix-split warm-start path.
pub fn run_lll_16_f64(scratch: &mut IntScratch16) -> LllResult {
    if !scratch.warm_lll {
        scratch.reset_basis();
    }
    if !super::lll::compute_gram_full(scratch) {
        return LllResult::GramOverflow;
    }
    lll_l2_16_f64(scratch)
}
