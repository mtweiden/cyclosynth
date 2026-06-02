//! 16D anisotropic Q-metric construction in MPFR + integer-scaled snapshot.
//!
//! For n=12 / Z[ζ₂₄] the Gram is anisotropic (`Σ_el^T Σ_el = 4I + 2C`,
//! eigenvalues {2,6}), so we cannot use the n=4 isotropic shortcut
//! `Σ⁻¹ = ½·Σᵀ`. Instead we build the Q metric in **lattice coordinates**
//! using two pre-computed Σ-derived projectors:
//!
//!   - `p_cap[i][j]   = Σ_{r ∈ cap rows}    σ[r][i]·σ[r][j]`
//!   - `p_bullet[i][j] = Σ_{r ∈ bullet rows} σ[r][i]·σ[r][j]`
//!
//! where the cap rows are `{0, 1, 8, 9}` (Re/Im σ_1 for u₁ and u₂) and the
//! bullet rows are the remaining 12 rows of the 16-row Σ. Their sum is
//! the full lattice Gram `ΣᵀΣ = 4I₁₆ + 2C` (verified in tests).
//!
//! ## §3 invariants honored
//!
//!  1. Rank-1 term is **unnormalized `ŷŷᵀ`** scaled by `inv_dy_sq − inv_dp_sq`
//!     (not `yyᵀ/‖y‖²` from the unit-direction `ŷ`).
//!  2. Caller passes raw `y` (the lattice-coord image), NOT `y` scaled by `R/2`.
//!  3. The cap center `c[i]` is `Σ⁻¹·(v, 0)` with `Σ⁻¹ = (ΣᵀΣ)⁻¹Σᵀ`, NOT a
//!     scalar multiple of Σᵀ.
//!  4. The bound used by Schnorr-Euchner is set in `se.rs` strictly above the
//!     true Q-norm boundary derived from this metric.
//!
//! Verbatim-ported helpers (`rug_to_i256_scaled`, `i256_to_rfloat`,
//! `rfloat_to_i256`) come from `lattice_zeta/q_metric.rs`.

#![allow(clippy::needless_range_loop)]

use i256::i256;
use rug::{Assign, Float as RFloat};
use std::f64::consts::PI;

use super::scratch::{
    compute_scale_bits, imat_zero_16, rfv, rfz, IntScratch16, TARGET_BITS,
};
use crate::rings::Float;

// ─── Row ordering for the n=12 Σ ─────────────────────────────────────────────

/// Cap rows of the full 16×16 Σ: `{Re σ_1(u₁), Im σ_1(u₁), Re σ_1(u₂), Im σ_1(u₂)}`.
pub const CAP_ROWS: [usize; 4] = [0, 1, 8, 9];

/// Bullet rows of the full 16×16 Σ: the 12 rows that are not cap rows.
pub const BULLET_ROWS: [usize; 12] = [2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15];

/// Per-element +i coset representatives `{1, 17, 13, 5}` (SPEC §2).
const COSET_REPS: [u32; 4] = [1, 17, 13, 5];

/// Fill a 16×16 f64 view of Σ for n=12.
fn fill_sigma_f64(sigma: &mut [[f64; 16]; 16]) {
    // Per-element 8×8 block: rows = [Re σ_m, Im σ_m] for m ∈ COSET_REPS.
    let mut el = [[0.0f64; 8]; 8];
    for (k, &m) in COSET_REPS.iter().enumerate() {
        for j in 0..8 {
            let theta = (m as f64) * (j as f64) * PI / 12.0;
            el[2 * k][j] = theta.cos();
            el[2 * k + 1][j] = theta.sin();
        }
    }
    // Full Σ = blkdiag(Σ_el, Σ_el).
    for i in 0..8 {
        for j in 0..8 {
            sigma[i][j] = el[i][j];
            sigma[8 + i][8 + j] = el[i][j];
        }
    }
}

// ─── Q-metric construction ───────────────────────────────────────────────────

/// Build the anisotropic 16D Q-metric in MPFR and the cap center `c`.
///
/// `v = (Re V₁₁, Im V₁₁, Re V₂₁, Im V₂₁)` is the SU(2) target column.
/// `y` is the lattice-coord alignment vector (`uv_to_xy` output).
///
/// Q decomposition (§3):
///   `Q = inv_dy_sq · ŷŷᵀ + inv_dp_sq · (p_cap − ŷŷᵀ) + inv_r_sq · p_bullet`
///
/// Cap center:
///   `c = cap_mid · R · Σ⁻¹·(v, 0)` where `Σ⁻¹·(v, 0)` is computed by solving
///   `(ΣᵀΣ) · z = Σᵀ · v_padded` once at f64 precision (the f64→MPFR lift
///   adds <1 ULP error, well within the LU-solve tolerance downstream).
pub fn build_q_mpfr_zeta(
    scratch: &mut IntScratch16,
    v: [f64; 4],
    k: u32,
    eps: Float,
) {
    // Compute y from v internally (lattice-coord alignment image of v at
    // scale √(2^k)), matching the lattice_zeta::build_q_mpfr_zeta signature.
    let y = crate::synthesis::lattice_upsilon::enumerate::uv_to_xy(v, k);
    let y: [Float; 16] = y;
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

    // §3-1: rank-1 term coefficient is `inv_dy_sq − inv_dp_sq` (cap radial −
    // cap tangential, applied to the UNNORMALIZED outer product ŷŷᵀ).
    let coef_yy = RFloat::with_val(prec, &inv_dy_sq - &inv_dp_sq);
    // Cap tangential coefficient.
    let coef_p_cap = inv_dp_sq;
    // Bullet ball coefficient.
    let coef_p_bullet = inv_r_sq;

    // ŷ = y / ‖y‖.
    let mut y_norm_sq = rfz(prec);
    for i in 0..16 {
        let yi_sq = rfv(prec, y[i] * y[i]);
        y_norm_sq += yi_sq;
    }
    let y_norm = y_norm_sq.clone().sqrt();
    let y_zero = y_norm_sq.is_zero();
    let mut yhat = [0.0f64; 16];
    if !y_zero {
        let y_norm_f = y_norm.to_f64();
        for i in 0..16 {
            yhat[i] = y[i] / y_norm_f;
        }
    }

    // Pre-compute Σ-row sums for the projectors.
    let mut sigma = [[0.0f64; 16]; 16];
    fill_sigma_f64(&mut sigma);
    let mut p_cap_f = [[0.0f64; 16]; 16];
    let mut p_bullet_f = [[0.0f64; 16]; 16];
    for i in 0..16 {
        for j in 0..16 {
            let mut s_cap = 0.0f64;
            for &r in &CAP_ROWS {
                s_cap += sigma[r][i] * sigma[r][j];
            }
            let mut s_bul = 0.0f64;
            for &r in &BULLET_ROWS {
                s_bul += sigma[r][i] * sigma[r][j];
            }
            p_cap_f[i][j] = s_cap;
            p_bullet_f[i][j] = s_bul;
        }
    }

    for i in 0..16 {
        for j in 0..16 {
            let yy = yhat[i] * yhat[j];
            let p_cap = p_cap_f[i][j];
            let p_bul = p_bullet_f[i][j];

            let yy_rf = rfv(prec, yy);
            let p_cap_rf = rfv(prec, p_cap);
            let p_bul_rf = rfv(prec, p_bul);

            let t1 = RFloat::with_val(prec, &coef_yy * &yy_rf);
            let cap_minus_yy = RFloat::with_val(prec, &p_cap_rf - &yy_rf);
            let t2 = RFloat::with_val(prec, &coef_p_cap * &cap_minus_yy);
            let t3 = RFloat::with_val(prec, &coef_p_bullet * &p_bul_rf);
            let s12 = RFloat::with_val(prec, &t1 + &t2);
            scratch.q_mpfr[i][j].assign(RFloat::with_val(prec, &s12 + &t3));
        }
    }

    // ─── Cap center c = cap_mid · R · Σ⁻¹·(v, 0) ───────────────────────────
    // §3-3: Σ⁻¹ = (ΣᵀΣ)⁻¹Σᵀ (not ½Σᵀ — that's n=4-only).
    let cap_mid_num = RFloat::with_val(prec, &one + &sqrt_1m);
    let cap_mid = RFloat::with_val(prec, &cap_mid_num / &two);
    let c_lattice = sigma_inv_apply_padded_v(&sigma, v);
    for i in 0..16 {
        let ci = rfv(prec, c_lattice[i]);
        let t = RFloat::with_val(prec, &ci * &r);
        scratch.c[i].assign(RFloat::with_val(prec, &t * &cap_mid));
    }
}

/// Solve `(ΣᵀΣ) z = Σᵀ (v_pad)` where `v_pad` puts `v=(v0,v1,v2,v3)` on
/// the cap rows `{0,1,8,9}` of the 16-D real space and zero elsewhere.
/// Returns `z ∈ R^16`, the lattice-coord pullback of the target.
///
/// f64 LU with partial pivoting. For Q-metric build this is precise enough
/// (downstream the MPFR cap center is exact to whatever `prec_q` provides
/// after the multiplication by `R · cap_mid`).
fn sigma_inv_apply_padded_v(sigma: &[[f64; 16]; 16], v: [f64; 4]) -> [f64; 16] {
    // RHS: rhs[j] = Σ_i Σ[i][j] · v_pad[i] (Σᵀ · v_pad).
    let v_pad = {
        let mut p = [0.0f64; 16];
        p[0] = v[0];
        p[1] = v[1];
        p[8] = v[2];
        p[9] = v[3];
        p
    };
    let mut rhs = [0.0f64; 16];
    for j in 0..16 {
        for i in 0..16 {
            rhs[j] += sigma[i][j] * v_pad[i];
        }
    }
    // Gram matrix G = ΣᵀΣ.
    let mut g = [[0.0f64; 16]; 16];
    for i in 0..16 {
        for j in 0..16 {
            let mut s = 0.0f64;
            for r in 0..16 {
                s += sigma[r][i] * sigma[r][j];
            }
            g[i][j] = s;
        }
    }
    // Partial-pivoting LU + back-substitution on a 16×16 system.
    lu_solve_16(&mut g, &mut rhs)
}

/// In-place partial-pivoting LU on `a` (16×16), then solve `a·x = b`.
/// Returns `x`. Used for the cap-center computation only (one call per
/// phase1, off the hot LLL/SE path).
fn lu_solve_16(a: &mut [[f64; 16]; 16], b: &mut [f64; 16]) -> [f64; 16] {
    let n = 16;
    let mut piv = [0usize; 16];
    for i in 0..n {
        piv[i] = i;
    }
    for k in 0..n {
        // Find pivot.
        let mut p = k;
        let mut max = a[k][k].abs();
        for i in (k + 1)..n {
            if a[i][k].abs() > max {
                max = a[i][k].abs();
                p = i;
            }
        }
        if p != k {
            a.swap(p, k);
            b.swap(p, k);
            piv.swap(p, k);
        }
        if a[k][k].abs() < 1e-18 {
            // Singular — return zeros; the cap-center will be ill-defined
            // but the caller's threshold check will catch it.
            return [0.0; 16];
        }
        for i in (k + 1)..n {
            let f = a[i][k] / a[k][k];
            a[i][k] = f;
            for j in (k + 1)..n {
                a[i][j] -= f * a[k][j];
            }
            b[i] -= f * b[k];
        }
    }
    let mut x = [0.0f64; 16];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n {
            s -= a[i][j] * x[j];
        }
        x[i] = s / a[i][i];
    }
    x
}

// ─── build_q_int: snapshot MPFR Q to scaled i256 ────────────────────────────

/// Snapshot `scratch.q_mpfr` into `scratch.q_int` with adaptive scaling.
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

// ─── rug ↔ i256 helpers (verbatim from lattice_zeta) ─────────────────────────

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
    if sign_neg {
        -val
    } else {
        val
    }
}

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

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthesis::lattice_upsilon::sigma::{gram_int, sigma_16};

    /// `p_cap + p_bullet` equals the full integer Gram `4I + 2C` (per element,
    /// summed over the block-diagonal). This is the SPEC §4 assertion + the
    /// load-bearing fact that the projector pair decomposes the Gram.
    #[test]
    fn p_cap_plus_p_bullet_equals_gram() {
        let mut sigma = [[0.0f64; 16]; 16];
        fill_sigma_f64(&mut sigma);
        let g_expected = gram_int();
        for i in 0..16 {
            for j in 0..16 {
                let mut s_cap = 0.0;
                for &r in &CAP_ROWS {
                    s_cap += sigma[r][i] * sigma[r][j];
                }
                let mut s_bul = 0.0;
                for &r in &BULLET_ROWS {
                    s_bul += sigma[r][i] * sigma[r][j];
                }
                let sum = s_cap + s_bul;
                let expected = g_expected[i][j] as f64;
                assert!(
                    (sum - expected).abs() < 1e-10,
                    "(p_cap + p_bullet)[{i}][{j}] = {sum}, expected {expected}"
                );
            }
        }
    }

    /// `fill_sigma_f64` matches `sigma::sigma_16`.
    #[test]
    fn fill_sigma_matches_lattice_upsilon_sigma_16() {
        let mut sigma = [[0.0f64; 16]; 16];
        fill_sigma_f64(&mut sigma);
        let reference = sigma_16();
        for i in 0..16 {
            for j in 0..16 {
                assert!(
                    (sigma[i][j] - reference[i][j]).abs() < 1e-12,
                    "Σ[{i}][{j}] mismatch: got {}, expected {}",
                    sigma[i][j],
                    reference[i][j]
                );
            }
        }
    }
}
