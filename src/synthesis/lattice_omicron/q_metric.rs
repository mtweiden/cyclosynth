//! Anisotropic Q-metric construction in MPFR (paper eq 3.15) and its
//! integer-scaled snapshot used by L²-LLL.

#![allow(clippy::needless_range_loop)]

use i256::i256;
use rug::{Assign, Float as RFloat};

use super::scratch::{compute_scale_bits, imat_zero, rfv, IntScratch, TARGET_BITS};
use super::scratch::{r_add, r_div, r_mul, r_sub};
use crate::rings::{zomicron::SIGMA_INV_U, Float};

// ─── build_q_mpfr: anisotropic Q-metric construction in MPFR ─────────────────

/// Build the anisotropic Q matrix in MPFR (paper eq 3.15) into
/// `scratch.q_mpfr`. Also computes the cap center into `scratch.c`.
/// Q is the metric used by the LLL; the cap center is the projection of
/// the target onto the alignment direction, used by the post-LLL LU solve.
pub fn build_q_mpfr(scratch: &mut IntScratch, y: &[Float; 8], v: &[Float; 4], k: u32, eps: Float) {
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

    for i in 0..8 {
        scratch.y_rf[i].assign(rfv(prec, y[i]));
    }
    for i in 0..8 {
        for j in 0..8 {
            r_mul!(scratch.tmp, scratch.y_rf[i], scratch.y_rf[j]);
            scratch.yy_t[i][j].assign(&scratch.tmp);
        }
    }

    // p_u and p_ub depend only on the constant Σ matrix and are populated
    // once by `fill_p_u_p_ub` in `IntScratch::new` — nothing to recompute here.

    for i in 0..8 {
        for j in 0..8 {
            r_mul!(scratch.tmp, scratch.inv_dy_sq, scratch.yy_t[i][j]);
            r_sub!(scratch.tmp2, scratch.p_u[i][j], scratch.yy_t[i][j]);
            r_mul!(scratch.tmp3, scratch.inv_dp_sq, scratch.tmp2);
            r_mul!(scratch.acc, scratch.inv_r_sq, scratch.p_ub[i][j]);
            let tmp_clone = scratch.tmp.clone();
            r_add!(scratch.tmp, tmp_clone, scratch.tmp3);
            r_add!(scratch.q_mpfr[i][j], scratch.tmp, scratch.acc);
        }
    }

    // Cap center: c = cap_mid * R * Σ^{-1}(v, 0), where in the crate's
    // image row order (Re u, Im u, Re u•, Im u•, Re t, Im t, Re t•, Im t•)
    // the image vector is (v0, v1, 0, 0, v2, v3, 0, 0).
    // v = (Re V00, Im V00, Re V10, Im V10). For y = Σ_top^T v as built by
    // clifford_pi6::compute_y, these components are available directly.
    r_mul!(scratch.tmp, scratch.eps_rf, scratch.eps_rf);
    r_sub!(scratch.tmp2, scratch.one, scratch.tmp);
    let sqrt_1m = scratch.tmp2.clone().sqrt();
    r_add!(scratch.tmp, scratch.one, sqrt_1m);
    r_div!(scratch.cap_mid, scratch.tmp, scratch.two);

    // TODO: c_seed_i is computed in f64 before lifting to MPFR.  At deep
    // epsilon (~1e-15 and below) this could leak f64 precision into the
    // cap center.  Promote SIGMA_INV_U @ v to MPFR end-to-end if precision
    // matters.
    for i in 0..8 {
        let mut c_seed_i = 0.0_f64;
        for (j, col) in [0usize, 1, 4, 5].iter().copied().enumerate() {
            c_seed_i += SIGMA_INV_U[i][col] * v[j];
        }
        scratch.tmp.assign(rfv(prec, c_seed_i));
        r_mul!(scratch.tmp2, scratch.tmp, scratch.r);
        r_mul!(scratch.c[i], scratch.tmp2, scratch.cap_mid);
    }
}

// ─── build_q_int: snapshot MPFR Q to scaled i256 ────────────────────────────

/// After `build_q_mpfr`, snapshot the MPFR Q into `scratch.q_int` with
/// adaptive scaling. Sets `scratch.scale_bits` to the chosen B.
///
/// Strategy: find max |Q_mpfr[i][j]|, choose B = TARGET_BITS − ⌈log₂(max)⌉,
/// then round each `S·Q[i][j]` to i256 with `S = 2^B`.
pub fn build_q_int(scratch: &mut IntScratch) {
    // Find max magnitude.
    let mut max_log2: i32 = i32::MIN;
    for i in 0..8 {
        for j in 0..8 {
            let v = scratch.q_mpfr[i][j].clone().abs();
            if v.is_zero() {
                continue;
            }
            // log2(|v|) — RFloat exposes the binary exponent directly via
            // get_exp(): |v| ∈ [2^(e-1), 2^e).
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
    for i in 0..8 {
        for j in 0..8 {
            scratch.q_int[i][j] = rug_to_i256_scaled(&scratch.q_mpfr[i][j], b);
        }
    }
}

// ─── rug → i256 conversion (used by build_q_int) ─────────────────────────────

/// Round `2^shift_bits · x` to `i256`. `shift_bits` may be positive (scale
/// up) or negative (scale down). Saturates to i256 bounds (callers should
/// choose shift_bits to avoid this).
pub fn rug_to_i256_scaled(x: &RFloat, shift_bits: i32) -> i256 {
    if x.is_zero() {
        return i256::from_i64(0);
    }
    let mut scaled = x.clone();
    if shift_bits >= 0 {
        scaled <<= shift_bits as u32;
    } else {
        scaled >>= (-shift_bits) as u32;
    }
    scaled.round_mut();
    rfloat_to_i256(&scaled)
}

/// Convert an integer-valued RFloat to i256. Saturates on overflow.
fn rfloat_to_i256(x: &RFloat) -> i256 {
    use rug::integer::Order;
    let sign_neg = x.is_sign_negative();
    let abs = x.clone().abs();
    // Fast path: fits in i64.
    if abs <= rug::Float::with_val(64, i64::MAX as f64) {
        let v = abs.to_f64() as i64;
        let res = i256::from_i64(v);
        return if sign_neg { -res } else { res };
    }
    let int = match abs.to_integer() {
        Some(i) => i,
        None => return i256::from_i64(0),
    };
    if int.significant_bits() > 254 {
        return if sign_neg { i256::MIN } else { i256::MAX };
    }
    let mut limbs = [0u64; 4];
    int.write_digits(&mut limbs, Order::Lsf);
    let mut bytes = [0u8; 32];
    for (idx, limb) in limbs.iter().enumerate() {
        bytes[idx * 8..(idx + 1) * 8].copy_from_slice(&limb.to_le_bytes());
    }
    let val = i256::from_le_bytes(bytes);
    if sign_neg {
        -val
    } else {
        val
    }
}
