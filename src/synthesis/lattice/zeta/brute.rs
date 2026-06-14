//! Brute-force search and y-vector helpers for the Z[ζ_16] / Clifford+√T
//! flow. Mirrors the role of [`super::super::omega::brute`] for the Z[ω] /
//! Clifford+T flow: y-vector construction (`compute_align_vec_zeta`,
//! `uv_to_lattice_y_zeta`) plus a brute-force enumerator
//! (`enumerate_unitary_norm_shell`) used as a correctness oracle for the
//! lattice pipeline in [`super`].
//!
//! Cost of `enumerate_unitary_norm_shell` is exponential in `k` (the shell at k=4 has
//! ~5·10⁸ points); useful for `k ≤ 4` for full enumeration. The L²-LLL +
//! Schnorr-Euchner port is in [`super`].

use crate::rings::MpFloat;
use std::f64::consts::PI;

use super::se::bilinear_forms;

// ─── y-vector helpers ────────────────────────────────────────────────────────

/// 4-element target direction `v` → 16D lattice-coord y, via
/// `y = Σ_fullᵀ · v_padded` (v on the σ_1 indices, zero elsewhere). By Σ
/// orthogonality the Σ-image of y is `4·target` on σ_1, zero elsewhere:
///   `y[j]   = cos(jπ/8)·v[0] + sin(jπ/8)·v[1]`  (u_1 block)
///   `y[8+j] = cos(jπ/8)·v[2] + sin(jπ/8)·v[3]`  (u_2 block)
pub fn compute_align_vec_zeta(v: [f64; 4]) -> [f64; 16] {
    let mut y = [0.0f64; 16];
    for j in 0..8 {
        let theta = (j as f64) * PI / 8.0;
        let c = theta.cos();
        let s = theta.sin();
        y[j] = c * v[0] + s * v[1];
        y[8 + j] = c * v[2] + s * v[3];
    }
    y
}

/// Scale a 4-element alignment direction `v` to the 16-element y vector
/// used by the Z[ζ_16] lattice pipeline. Convention chosen so that
/// `Σ_full · y = √(2^k) · v_padded` (target × √(2^k) on σ_1, zero on
/// σ_5/9/13); this gives ‖y‖² = 2^k/4, the 16D norm convention (the 8D
/// flow uses 2^(k−1) — they deliberately differ).
pub fn uv_to_lattice_y_zeta(v: [f64; 4], k: u32) -> [f64; 16] {
    let scale = 2.0f64.powf(k as f64 / 2.0) / 4.0;
    let raw = compute_align_vec_zeta(v);
    std::array::from_fn(|i| raw[i] * scale)
}

/// MPFR variant of [`uv_to_lattice_y_zeta`]. At deep ε the SE cap's radial
/// window (ε²/2 ≈ 5e-17 relative at 1e-8) is below an f64 ulp, so any
/// ~1e-16 error in ‖y‖ displaces the cap and causes find/miss flicker.
/// Fix: MPFR cos/sin tables, then rescale so ‖y‖ = ρ = 2^(k/2)/2 EXACTLY.
/// Norm errors then become pure direction errors, harmless to the cap.
pub fn uv_to_lattice_y_zeta_mpfr(v: &[MpFloat; 4], k: u32, prec: u32) -> [MpFloat; 16] {
    // cos/sin(jπ/8) tables at `prec` (MPFR Pi — no f64 trig roundings).
    let pi = MpFloat::with_val(prec, rug::float::Constant::Pi);
    let mut raw: [MpFloat; 16] = std::array::from_fn(|_| MpFloat::with_val(prec, 0.0));
    let mut raw_norm_sq = MpFloat::with_val(prec, 0.0);
    for j in 0..8 {
        let theta = MpFloat::with_val(prec, &pi * (j as u32)) / 8u32;
        let c = theta.clone().cos();
        let s = theta.sin();
        let cv0 = MpFloat::with_val(prec, &c * &v[0]);
        let sv1 = MpFloat::with_val(prec, &s * &v[1]);
        raw[j] = MpFloat::with_val(prec, &cv0 + &sv1);
        let cv2 = MpFloat::with_val(prec, &c * &v[2]);
        let sv3 = MpFloat::with_val(prec, &s * &v[3]);
        raw[8 + j] = MpFloat::with_val(prec, &cv2 + &sv3);
        raw_norm_sq += MpFloat::with_val(prec, &raw[j] * &raw[j]);
        raw_norm_sq += MpFloat::with_val(prec, &raw[8 + j] * &raw[8 + j]);
    }
    if raw_norm_sq.is_zero() {
        // Degenerate v: keep the legacy convention (zero y → SE walk
        // returns empty downstream).
        return raw;
    }
    // ρ = 2^(k/2) / 2 in MPFR (exact shift; √2 at prec for odd k).
    let rho = {
        let mut s = MpFloat::with_val(prec, 1.0);
        s <<= (k / 2) as i32;
        if k % 2 == 1 {
            let sqrt2 = MpFloat::with_val(prec, 2.0).sqrt();
            s *= &sqrt2;
        }
        s >>= 1;
        s
    };
    // y = raw · ρ / ‖raw‖ — exact radial-norm anchoring (‖y‖ ≡ ρ).
    let raw_norm = raw_norm_sq.sqrt();
    let scale = MpFloat::with_val(prec, &rho / &raw_norm);
    std::array::from_fn(|i| MpFloat::with_val(prec, &raw[i] * &scale))
}

// ─── Brute-force find_aligned_lattice_points ──────────────────────────────────────────────────────

/// Recursive enumerator: walks integer 16-vectors with `‖x‖² = remaining`
/// at the current recursion depth.
fn enumerate<F: FnMut(&[i64; 16])>(
    x: &mut [i64; 16],
    pos: usize,
    remaining: i64,
    cb: &mut F,
) {
    if pos == 16 {
        if remaining == 0 {
            cb(x);
        }
        return;
    }
    let bound = (remaining as f64).sqrt().floor() as i64;
    for v in -bound..=bound {
        let v2 = v * v;
        if v2 > remaining {
            continue;
        }
        x[pos] = v;
        enumerate(x, pos + 1, remaining - v2, cb);
    }
}

/// Brute-force find_aligned_lattice_points for Z[ζ_16]: enumerate all `(u_1, u_2) ∈ Z[ζ_16]²`
/// with `‖u_1‖² + ‖u_2‖² = 2^k` and `B_1 = B_2 = B_3 = 0`.
///
/// Returns 16-element integer solutions. Cost is exponential in `k`.
pub fn enumerate_unitary_norm_shell(k: u32) -> Vec<[i64; 16]> {
    assert!(k < 31, "k too large for i64 norm shell (would overflow)");
    let target_norm_sq = 1i64 << k;
    let mut x = [0i64; 16];
    let mut results = Vec::new();
    enumerate(&mut x, 0, target_norm_sq, &mut |x| {
        let (b1, b2, b3) = bilinear_forms(x);
        if b1 == 0 && b2 == 0 && b3 == 0 {
            results.push(*x);
        }
    });
    results
}
