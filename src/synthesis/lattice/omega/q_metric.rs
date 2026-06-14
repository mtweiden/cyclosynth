//! Anisotropic Q-metric construction in MPFR (paper eq 3.15) and its
//! integer-scaled snapshot used by L²-LLL.

#![allow(clippy::needless_range_loop)]

use rug::Assign;

use super::scratch::{
    compute_scale_bits, imat_zero, rfv, IntScratch, TARGET_BITS,
};
use super::scratch::{r_add, r_div, r_mul, r_sub};
use crate::rings::Float;

// ─── build_q_mpfr: anisotropic Q-metric construction in MPFR ─────────────────

/// Build the anisotropic Q matrix in MPFR (paper eq 3.15) into
/// `scratch.q_mpfr`. Also computes the cap center into `scratch.c`.
/// Q is the metric used by the LLL; the cap center is the projection of
/// the target onto the alignment direction, used by the post-LLL LU solve.
///
/// Q_base hoist: the algebraic split
///
///   Q = inv_dy_sq·ŷŷᵀ + inv_dp_sq·(P_u − ŷŷᵀ) + inv_r_sq·P_•
///     = Q_base(k, ε) + (inv_dy_sq − inv_dp_sq)/‖y‖² · y·yᵀ
///
/// makes everything except the rank-1 `y·yᵀ` term prefix-independent.
/// `build_q_base` computes the scalars + `q_base` + `cap_mid` once per
/// `(k, ε)` (cached via `scratch.q_base_key`); the per-prefix remainder is
/// just the rank-1 term over the symmetric lower triangle plus the ‖y‖²
/// and cap-center loops.
pub fn build_q_mpfr(scratch: &mut IntScratch, y: &[Float; 8], k: u32, eps: Float) {
    let prec = scratch.prec_q;

    let key = (k, eps.to_bits());
    if scratch.q_base_key != Some(key) {
        build_q_base(scratch, k, eps);
        scratch.q_base_key = Some(key);
    }

    // ── Per-prefix part: rank-1 term + cap center ──
    for i in 0..8 {
        scratch.y_rf[i].assign(rfv(prec, y[i]));
    }
    scratch.y_norm_sq.assign(0.0_f64);
    for i in 0..8 {
        r_mul!(scratch.tmp, scratch.y_rf[i], scratch.y_rf[i]);
        let acc_clone = scratch.y_norm_sq.clone();
        r_add!(scratch.y_norm_sq, acc_clone, scratch.tmp);
    }
    r_div!(scratch.inv_y_norm_sq, scratch.one, scratch.y_norm_sq);

    // s = coef_y / ‖y‖² ; Q[i][j] = q_base[i][j] + s·y_i·y_j.
    // Q is exactly symmetric (identical expressions), so compute the lower
    // triangle and mirror.
    r_mul!(scratch.tmp3, scratch.coef_y, scratch.inv_y_norm_sq);
    for i in 0..8 {
        for j in 0..=i {
            r_mul!(scratch.tmp, scratch.y_rf[i], scratch.y_rf[j]);
            r_mul!(scratch.tmp2, scratch.tmp, scratch.tmp3);
            r_add!(scratch.q_mpfr[i][j], scratch.q_base[i][j], scratch.tmp2);
            if i != j {
                // Mirror (split borrow: rows i and j are distinct).
                let (lo, hi) = scratch.q_mpfr.split_at_mut(i);
                lo[j][i].assign(&hi[0][j]);
            }
        }
    }

    // Cap center: c = cap_mid · y (cap_mid is ε-only, hoisted).
    for i in 0..8 {
        r_mul!(scratch.c[i], scratch.y_rf[i], scratch.cap_mid);
    }
}

/// Prefix-independent part of `build_q_mpfr`: the (k, ε) scalars, the
/// `q_base = inv_dp_sq·P_u + inv_r_sq·P_•` matrix, the rank-1 weight
/// `coef_y = inv_dy_sq − inv_dp_sq`, and the ε-only `cap_mid`.
/// `p_u`/`p_ub` themselves are Σ-constants filled once in
/// `IntScratch::new`.
fn build_q_base(scratch: &mut IntScratch, k: u32, eps: Float) {
    let prec = scratch.prec_q;

    // R² = 2^k. For k ≥ 64, `1u64 << k` is UB — build via f64 powi (f64 exp
    // up to 1023 covers all reasonable k).
    let r_sq_f = 2.0_f64.powi(k as i32);
    scratch.r_sq.assign(rfv(prec, r_sq_f));
    scratch.r.assign(scratch.r_sq.clone().sqrt());
    scratch.eps_rf.assign(rfv(prec, eps));

    // Δ_y = R · ε² / (2·(1 + √(1−ε²)))
    r_mul!(scratch.tmp, scratch.eps_rf, scratch.eps_rf);
    r_sub!(scratch.tmp2, scratch.one, scratch.tmp);
    let sqrt_1m = scratch.tmp2.clone().sqrt();
    r_add!(scratch.tmp2, scratch.one, sqrt_1m);
    r_mul!(scratch.tmp3, scratch.tmp2, scratch.two);
    r_mul!(scratch.acc, scratch.r, scratch.tmp);
    r_div!(scratch.delta_y, scratch.acc, scratch.tmp3);

    r_mul!(scratch.delta_perp, scratch.r, scratch.eps_rf);

    r_mul!(scratch.tmp, scratch.delta_y, scratch.delta_y);
    r_div!(scratch.inv_dy_sq, scratch.one, scratch.tmp);
    r_mul!(scratch.tmp, scratch.delta_perp, scratch.delta_perp);
    r_div!(scratch.inv_dp_sq, scratch.one, scratch.tmp);
    r_div!(scratch.inv_r_sq, scratch.one, scratch.r_sq);

    // coef_y = inv_dy_sq − inv_dp_sq (the terms differ by ≈ (4/ε)², so no
    // cancellation), q_base = inv_dp_sq·P_u + inv_r_sq·P_• (symmetric).
    r_sub!(scratch.coef_y, scratch.inv_dy_sq, scratch.inv_dp_sq);
    for i in 0..8 {
        for j in 0..=i {
            r_mul!(scratch.tmp, scratch.inv_dp_sq, scratch.p_u[i][j]);
            r_mul!(scratch.tmp2, scratch.inv_r_sq, scratch.p_ub[i][j]);
            r_add!(scratch.q_base[i][j], scratch.tmp, scratch.tmp2);
            if i != j {
                // Mirror (split borrow: rows i and j are distinct).
                let (lo, hi) = scratch.q_base.split_at_mut(i);
                lo[j][i].assign(&hi[0][j]);
            }
        }
    }

    // cap_mid = (1 + √(1−ε²))/2 — ε-only, reused by every per-prefix call.
    r_mul!(scratch.tmp, scratch.eps_rf, scratch.eps_rf);
    r_sub!(scratch.tmp2, scratch.one, scratch.tmp);
    let sqrt_1m = scratch.tmp2.clone().sqrt();
    r_add!(scratch.tmp, scratch.one, sqrt_1m);
    r_div!(scratch.cap_mid, scratch.tmp, scratch.two);
}

// ─── build_q_int: snapshot MPFR Q to scaled i256 ────────────────────────────

/// After `build_q_mpfr`, snapshot the MPFR Q into `scratch.q_int` with
/// adaptive scaling. Sets `scratch.scale_bits` to the chosen B.
///
/// Strategy: find max |Q_mpfr[i][j]|, choose B = TARGET_BITS − ⌈log₂(max)⌉,
/// then round each `S·Q[i][j]` to i256 with `S = 2^B`.
pub fn build_q_int(scratch: &mut IntScratch) {
    // Find max magnitude (lower triangle suffices — Q is symmetric).
    let mut max_log2: i32 = i32::MIN;
    for i in 0..8 {
        for j in 0..=i {
            let v = &scratch.q_mpfr[i][j];
            if v.is_zero() {
                continue;
            }
            // log2(|v|) — MpFloat exposes the binary exponent directly via
            // get_exp(): |v| ∈ [2^(e-1), 2^e). Sign does not affect the
            // exponent, so no abs() needed.
            let e = v.get_exp().unwrap_or(0);
            if e > max_log2 {
                max_log2 = e;
            }
        }
    }
    if max_log2 == i32::MIN {
        // All zero — degenerate, but produce zero matrix.
        scratch.scale_bits = TARGET_BITS as i32;
        scratch.q_int = imat_zero();
        return;
    }
    let b = compute_scale_bits(max_log2);
    scratch.scale_bits = b;
    // Q is exactly symmetric (build_q_mpfr mirrors the triangle), so
    // convert the lower triangle once and mirror the i256 (Copy) value.
    for i in 0..8 {
        for j in 0..=i {
            let v = rug_to_i256_scaled(&scratch.q_mpfr[i][j], b);
            scratch.q_int[i][j] = v;
            scratch.q_int[j][i] = v;
        }
    }
}

// ─── rug → i256 conversion (used by build_q_int) ─────────────────────────────

pub use crate::synthesis::lattice::common::rug_to_i256_scaled;
