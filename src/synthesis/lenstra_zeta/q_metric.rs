//! 16D Q-metric construction in MPFR + integer-scaled snapshot.
//!
//! Constructs Q in lattice coordinates at MPFR precision, then snapshots to
//! i256 with adaptive scaling for the integer LLL. Z[ζ_16] analog of
//! [`super::super::lenstra::q_metric`].

#![allow(clippy::needless_range_loop)]

use i256::i256;
use rug::{Assign, Float as RFloat};
use std::f64::consts::PI;

use super::scratch::{
    compute_scale_bits, imat_zero_16, rfv, rfz, IntScratch16, TARGET_BITS,
};
use crate::rings::Float;

// ─── build_q_mpfr_zeta: 16D Q-metric construction in MPFR ────────────────────

/// Build the 16D Q-metric matrix in **lattice coordinates** for Z[ζ_16]
/// synthesis at lde `k` and precision `eps`, in MPFR at `scratch.prec_q`.
///
/// Mirrors the f64 `build_q_zzeta_lattice` (test helper at the bottom of
/// this file) but at arbitrary precision so the i256 snapshot is faithful
/// at deep ε:
///
/// ```text
/// Q[i][j] = (1/Δ_y² − 1/Δ_⊥²) · ŷ[i] · ŷ[j]
///         + (1/Δ_⊥² − 1/R²)   · P_σ1[i][j]
///         + (1/R²)             · δ_ij
/// ```
///
/// `v` is the SU(2) direction `(Re V_{11}, Im V_{11}, Re V_{21}, Im V_{21})`.
/// MPFR-precision variant: `v` is provided in MPFR, so `y` (and downstream
/// `Q` and the cap center derived from it) carry whatever precision the
/// caller gave us. At ε=1e-8 the cap-radial direction `Δ_y/R = ε²/4 ≈ 2.5e-17`
/// is below f64 ULP at unit scale (~2.2e-16); the f64-input version below
/// loses the cap localization in this regime.
pub fn build_q_mpfr_zeta_from_mpfr_v(
    scratch: &mut IntScratch16,
    v: &[rug::Float; 4],
    k: u32,
    eps: Float,
) {
    let prec = scratch.prec_q;
    let one = rfv(prec, 1.0);
    let two = rfv(prec, 2.0);

    let r_sq_f = 2.0_f64.powi(k as i32);
    let r_sq = rfv(prec, r_sq_f);
    let r = r_sq.clone().sqrt();
    let eps_rf = rfv(prec, eps);

    let eps_sq = RFloat::with_val(prec, &eps_rf * &eps_rf);
    let one_minus_eps_sq = RFloat::with_val(prec, &one - &eps_sq);
    let sqrt_1m = one_minus_eps_sq.sqrt();
    let denom_inner = RFloat::with_val(prec, &one + &sqrt_1m);
    let denom = RFloat::with_val(prec, &denom_inner * &two);
    let r_eps_sq = RFloat::with_val(prec, &r * &eps_sq);
    let delta_y = RFloat::with_val(prec, &r_eps_sq / &denom);
    let delta_perp = RFloat::with_val(prec, &r * &eps_rf);

    let dy_sq = RFloat::with_val(prec, &delta_y * &delta_y);
    let dp_sq = RFloat::with_val(prec, &delta_perp * &delta_perp);
    let inv_dy_sq = RFloat::with_val(prec, &one / &dy_sq);
    let inv_dp_sq = RFloat::with_val(prec, &one / &dp_sq);
    let inv_r_sq = RFloat::with_val(prec, &one / &r_sq);

    let coef_yy = RFloat::with_val(prec, &inv_dy_sq - &inv_dp_sq);
    let coef_p_sigma1 = RFloat::with_val(prec, &inv_dp_sq - &inv_r_sq);
    let coef_id = inv_r_sq;

    // Compute y entirely in MPFR. cos(jπ/8), sin(jπ/8) at f64 → MPFR is exact
    // (single-rounding), so the f64 cos/sin are used as the only f64 entry
    // points. The multiplication and sum are MPFR-exact at `prec` bits, so
    // y[j]'s precision matches v's.
    let mut y: [RFloat; 16] = std::array::from_fn(|_| rfz(prec));
    for j in 0..8 {
        let theta = (j as f64) * PI / 8.0;
        let c_f = theta.cos();
        let s_f = theta.sin();
        let c = rfv(prec, c_f);
        let s = rfv(prec, s_f);
        // y[j] = c·v[0] + s·v[1]
        let cv0 = RFloat::with_val(prec, &c * &v[0]);
        let sv1 = RFloat::with_val(prec, &s * &v[1]);
        y[j].assign(RFloat::with_val(prec, &cv0 + &sv1));
        // y[8+j] = c·v[2] + s·v[3]
        let cv2 = RFloat::with_val(prec, &c * &v[2]);
        let sv3 = RFloat::with_val(prec, &s * &v[3]);
        y[8 + j].assign(RFloat::with_val(prec, &cv2 + &sv3));
    }
    let mut y_norm_sq = rfz(prec);
    for i in 0..16 {
        let yi_sq = RFloat::with_val(prec, &y[i] * &y[i]);
        y_norm_sq += yi_sq;
    }
    let y_norm = y_norm_sq.clone().sqrt();
    let y_zero = y_norm_sq.is_zero();
    let mut yhat: [RFloat; 16] = std::array::from_fn(|_| rfz(prec));
    if !y_zero {
        for i in 0..16 {
            yhat[i].assign(RFloat::with_val(prec, &y[i] / &y_norm));
        }
    }

    for i in 0..16 {
        for j in 0..16 {
            let mut qij = rfz(prec);
            let yyi = RFloat::with_val(prec, &yhat[i] * &yhat[j]);
            qij += RFloat::with_val(prec, &coef_yy * &yyi);

            let same_block = (i < 8 && j < 8) || (i >= 8 && j >= 8);
            if same_block {
                let m = (i % 8) as f64 - (j % 8) as f64;
                let p_sigma1 = 0.25 * (m * PI / 8.0).cos();
                let p = rfv(prec, p_sigma1);
                qij += RFloat::with_val(prec, &coef_p_sigma1 * &p);
            }

            if i == j {
                qij += &coef_id;
            }

            scratch.q_mpfr[i][j].assign(&qij);
        }
    }
}

pub fn build_q_mpfr_zeta(scratch: &mut IntScratch16, v: [f64; 4], k: u32, eps: Float) {
    let prec = scratch.prec_q;
    let one = rfv(prec, 1.0);
    let two = rfv(prec, 2.0);

    // R² = 2^k. Use f64 powi (range 1023 covers all reasonable k).
    let r_sq_f = 2.0_f64.powi(k as i32);
    let r_sq = rfv(prec, r_sq_f);
    let r = r_sq.clone().sqrt();
    let eps_rf = rfv(prec, eps);

    // Δ_y = R · ε² / (2·(1 + √(1−ε²)))
    let eps_sq = RFloat::with_val(prec, &eps_rf * &eps_rf);
    let one_minus_eps_sq = RFloat::with_val(prec, &one - &eps_sq);
    let sqrt_1m = one_minus_eps_sq.sqrt();
    let denom_inner = RFloat::with_val(prec, &one + &sqrt_1m);
    let denom = RFloat::with_val(prec, &denom_inner * &two);
    let r_eps_sq = RFloat::with_val(prec, &r * &eps_sq);
    let delta_y = RFloat::with_val(prec, &r_eps_sq / &denom);
    let delta_perp = RFloat::with_val(prec, &r * &eps_rf);

    let dy_sq = RFloat::with_val(prec, &delta_y * &delta_y);
    let dp_sq = RFloat::with_val(prec, &delta_perp * &delta_perp);
    let inv_dy_sq = RFloat::with_val(prec, &one / &dy_sq);
    let inv_dp_sq = RFloat::with_val(prec, &one / &dp_sq);
    let inv_r_sq = RFloat::with_val(prec, &one / &r_sq);

    let coef_yy = RFloat::with_val(prec, &inv_dy_sq - &inv_dp_sq);
    let coef_p_sigma1 = RFloat::with_val(prec, &inv_dp_sq - &inv_r_sq);
    let coef_id = inv_r_sq;

    // ŷ in lattice coords: y_lattice / |y_lattice|. Compute y in MPFR using
    // f64 cos/sin (the angles are j·π/8 for j=0..7, so the f64 cos/sin
    // values are exact-ish to ~1e-16; that's enough at the prec_q scale).
    let mut y: [RFloat; 16] = std::array::from_fn(|_| rfz(prec));
    for j in 0..8 {
        let theta = (j as f64) * PI / 8.0;
        let c = theta.cos();
        let s = theta.sin();
        y[j].assign(rfv(prec, c * v[0] + s * v[1]));
        y[8 + j].assign(rfv(prec, c * v[2] + s * v[3]));
    }
    let mut y_norm_sq = rfz(prec);
    for i in 0..16 {
        let yi_sq = RFloat::with_val(prec, &y[i] * &y[i]);
        y_norm_sq += yi_sq;
    }
    let y_norm = y_norm_sq.clone().sqrt();
    let y_zero = y_norm_sq.is_zero();
    let mut yhat: [RFloat; 16] = std::array::from_fn(|_| rfz(prec));
    if !y_zero {
        for i in 0..16 {
            yhat[i].assign(RFloat::with_val(prec, &y[i] / &y_norm));
        }
    }

    for i in 0..16 {
        for j in 0..16 {
            let mut qij = rfz(prec);
            // Term 1: coef_yy · ŷ[i] · ŷ[j].
            let yyi = RFloat::with_val(prec, &yhat[i] * &yhat[j]);
            qij += RFloat::with_val(prec, &coef_yy * &yyi);

            // Term 2: coef_p_sigma1 · P_σ1[i][j]. Block-diagonal.
            let same_block = (i < 8 && j < 8) || (i >= 8 && j >= 8);
            if same_block {
                let m = (i % 8) as f64 - (j % 8) as f64;
                let p_sigma1 = 0.25 * (m * PI / 8.0).cos();
                let p = rfv(prec, p_sigma1);
                qij += RFloat::with_val(prec, &coef_p_sigma1 * &p);
            }

            // Term 3: coef_id · δ_ij.
            if i == j {
                qij += &coef_id;
            }

            scratch.q_mpfr[i][j].assign(&qij);
        }
    }
}

// ─── build_q_int_zeta: snapshot MPFR Q to scaled i256 ────────────────────────

/// Snapshot the MPFR Q into `scratch.q_int` with adaptive scaling. Sets
/// `scratch.scale_bits` such that `max(|Q_int|) ≈ 2^TARGET_BITS`.
pub fn build_q_int_zeta(scratch: &mut IntScratch16) {
    let mut max_log2: i32 = i32::MIN;
    for i in 0..16 {
        for j in 0..16 {
            let v = scratch.q_mpfr[i][j].clone().abs();
            if v.is_zero() {
                continue;
            }
            let e = v.get_exp().unwrap_or(0);
            if e > max_log2 {
                max_log2 = e;
            }
        }
    }
    if max_log2 == i32::MIN {
        scratch.scale_bits = TARGET_BITS as i32;
        scratch.q_int = imat_zero_16();
        return;
    }
    let b = compute_scale_bits(max_log2);
    scratch.scale_bits = b;
    for i in 0..16 {
        for j in 0..16 {
            scratch.q_int[i][j] = rug_to_i256_scaled(&scratch.q_mpfr[i][j], b);
        }
    }
}

/// Round `2^shift_bits · x` to `i256`. `shift_bits` may be positive or
/// negative. Saturates to i256 bounds (callers should pick shift_bits to
/// avoid this).
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
    if sign_neg { -val } else { val }
}

/// Set an MPFR variable to the value of an i256 limb-by-limb. Used by tests
/// that need to compare integer Gram entries against MPFR references.
pub fn i256_to_rfloat(v: i256, dst: &mut RFloat) {
    use gmp_mpfr_sys::{gmp, mpfr};
    use std::ptr::NonNull;
    let zero = i256::from_i64(0);
    if v == zero {
        unsafe { mpfr::set_zero(dst.as_raw_mut(), 0) };
        return;
    }
    let neg = v < zero;
    let abs = if neg { -v } else { v };
    let bytes = abs.to_le_bytes();
    let mut limbs: [gmp::limb_t; 4] = std::array::from_fn(|i| {
        let mut buf = [0u8; 8];
        buf.copy_from_slice(&bytes[i * 8..(i + 1) * 8]);
        u64::from_le_bytes(buf) as gmp::limb_t
    });
    let mut size: i32 = 4;
    while size > 0 && limbs[(size - 1) as usize] == 0 {
        size -= 1;
    }
    let signed_size = if neg { -size } else { size };
    let mpz = gmp::mpz_t {
        alloc: 0,
        size: signed_size,
        d: unsafe { NonNull::new_unchecked(limbs.as_mut_ptr()) },
    };
    unsafe {
        mpfr::set_z(dst.as_raw_mut(), &mpz as *const _, mpfr::rnd_t::RNDN);
    }
}

// ─── f64 sanity helper for tests ─────────────────────────────────────────────

/// Build the 16D Q-metric in lattice coordinates at f64 precision. Used as
/// a sanity oracle in tests against [`build_q_mpfr_zeta`]; not exercised in
/// the production pipeline (which always goes through MPFR + i256).
#[cfg(test)]
pub fn build_q_zzeta_lattice(v: [f64; 4], k: u32, eps: f64) -> [[f64; 16]; 16] {
    use crate::synthesis::search_zeta::compute_align_vec_zeta;

    let r_sq = 2.0f64.powi(k as i32);
    let r = r_sq.sqrt();
    let delta_y = r * eps * eps / (2.0 * (1.0 + (1.0 - eps * eps).sqrt()));
    let delta_perp = r * eps;
    let inv_dy_sq = 1.0 / (delta_y * delta_y);
    let inv_dp_sq = 1.0 / (delta_perp * delta_perp);
    let inv_r_sq = 1.0 / r_sq;

    let coef_yy = inv_dy_sq - inv_dp_sq;
    let coef_p_sigma1 = inv_dp_sq - inv_r_sq;
    let coef_id = inv_r_sq;

    // ŷ in lattice coords: y_lattice / |y_lattice|.
    let y = compute_align_vec_zeta(v);
    let y_norm_sq: f64 = y.iter().map(|x| x * x).sum();
    let y_norm = y_norm_sq.sqrt();
    let yhat: [f64; 16] = if y_norm > 0.0 {
        std::array::from_fn(|i| y[i] / y_norm)
    } else {
        [0.0; 16]
    };

    let mut q = [[0.0f64; 16]; 16];
    for i in 0..16 {
        for j in 0..16 {
            // Term 1: coef_yy · ŷ[i] · ŷ[j].
            q[i][j] += coef_yy * yhat[i] * yhat[j];

            // Term 2: coef_p_sigma1 · P_σ1[i][j]. Block-diagonal.
            let same_block = (i < 8 && j < 8) || (i >= 8 && j >= 8);
            if same_block {
                let m = (i % 8) as f64 - (j % 8) as f64;
                let p_sigma1 = 0.25 * (m * PI / 8.0).cos();
                q[i][j] += coef_p_sigma1 * p_sigma1;
            }

            // Term 3: coef_id · δ_ij.
            if i == j {
                q[i][j] += coef_id;
            }
        }
    }
    q
}
