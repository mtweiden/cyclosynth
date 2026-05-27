//! Diamond distance between unitaries, in three flavours covering the
//! precision regimes the synthesizer needs.
//!
//! All three compute `D = √(1 − |tr(A·B†)|²/4)` mathematically, but the
//! direct formula has a precision wall at `D ≈ √(machine_eps) ≈ 1.5×10⁻⁸`
//! caused by catastrophic cancellation in `1 − |tr|²/4` (both terms ≈ 1).
//! Two design choices follow:
//!
//! 1. **Algebraic reformulation**: for unitary `A`, `B`,
//!    `D² = q · (8 − q) / 16` where `q = ‖A − φB‖²_F` with optimal phase
//!    `φ = tr(AB†)/|tr|`. The Frobenius identity is non-negative by
//!    construction and matches the trace formula exactly when both inputs
//!    are unitary; it also gracefully handles approximately-unitary inputs
//!    (f64-quantized targets are non-unitary by ~10⁻¹⁶ in Frobenius norm,
//!    which can push `|tr|² > 4` and clamp the trace formula to 0).
//!    See `feedback_diamond_distance_frobenius.md` for the full proof and
//!    citations (Watrous 2018 §3.3, Bhatia 1997 Ch. IV, Higham 2002 §1.7).
//!
//! 2. **MPFR for the U2T input**: at deep `lde` (k ≳ 50), the rounding in
//!    `u2t.to_float()` can quantize each entry by up to one f64 ulp, and
//!    the rounded bits can match the target's f64 bits exactly — hiding a
//!    real precision violation. The U2T-specific entry point evaluates the
//!    ZOmega ring representation directly in MPFR, so the f64 quantization
//!    of the U2T side is bypassed entirely.

use num_complex::Complex;
use rug::{Assign, Float as RFloat};

use crate::matrix::{U2Q, U2T};
use crate::rings::types::{int_to_f64, Float};
use crate::rings::{ZOmega, ZZeta};

/// A 2×2 matrix of complex f64 values — the synthesizer's representation of
/// arbitrary unitary inputs (target matrices) and float-converted Clifford+T
/// elements alike.
pub type Mat2 = [[Complex<Float>; 2]; 2];

/// Diamond distance between two 2×2 unitaries, in pure f64 with the
/// algebraic Frobenius reformulation:
///
/// ```text
/// φ      = tr(A·B†) / |tr(A·B†)|     (optimal global phase, |φ| = 1)
/// q      = ‖A − φB‖²_F
/// D²     = q · (8 − q) / 16
/// ```
///
/// Equivalent to `1 − |tr|²/4` for exactly-unitary inputs but precision-
/// stable down to f64 epsilon (~10⁻¹⁶) and non-negative by construction
/// for any inputs.
pub fn diamond_distance_float(a: &Mat2, b: &Mat2) -> Float {
    let tr = a[0][0] * b[0][0].conj()
        + a[0][1] * b[0][1].conj()
        + a[1][0] * b[1][0].conj()
        + a[1][1] * b[1][1].conj();
    let tr_abs = tr.norm();
    // Optimal phase φ = tr / |tr|. If `tr` is degenerately zero, φ=1 (the
    // choice is irrelevant — fro_sq is large in that case, no cancellation).
    let phi = if tr_abs > 1e-300 {
        tr / tr_abs
    } else {
        Complex::new(1.0, 0.0)
    };
    let mut fro_sq: Float = 0.0;
    for i in 0..2 {
        for j in 0..2 {
            let diff = a[i][j] - phi * b[i][j];
            fro_sq += diff.norm_sqr();
        }
    }
    let d_sq = fro_sq * (8.0 - fro_sq) / 16.0;
    d_sq.max(0.0).sqrt()
}

/// MPFR-precision diamond distance via the trace formula `1 − |tr|²/4`.
/// Kept for tests as an oracle against the f64 algebraic version. Returns
/// an `f64` for caller convenience; `prec` is the working precision in
/// bits (128 is the recommended default — at that precision the trace
/// formula's cancellation is well below the noise floor for any ε we care
/// about, so the formula choice doesn't matter and we keep the original
/// mathematical statement).
pub fn diamond_distance_float_mpfr(a: &Mat2, b: &Mat2, prec: u32) -> Float {
    let mut tr_re = RFloat::with_val(prec, 0.0);
    let mut tr_im = RFloat::with_val(prec, 0.0);
    let mut tmp = RFloat::with_val(prec, 0.0);
    let mut tmp2 = RFloat::with_val(prec, 0.0);
    // tr = Σ A_ij · conj(B_ij)
    //    = Σ (a_re + i·a_im)·(b_re − i·b_im)
    //    = Σ (a_re·b_re + a_im·b_im) + i·(a_im·b_re − a_re·b_im)
    for i in 0..2 {
        for j in 0..2 {
            let a_re = RFloat::with_val(prec, a[i][j].re);
            let a_im = RFloat::with_val(prec, a[i][j].im);
            let b_re = RFloat::with_val(prec, b[i][j].re);
            let b_im = RFloat::with_val(prec, b[i][j].im);
            tmp.assign(&a_re * &b_re);
            tr_re += &tmp;
            tmp.assign(&a_im * &b_im);
            tr_re += &tmp;
            tmp.assign(&a_im * &b_re);
            tmp2.assign(&a_re * &b_im);
            tr_im += &tmp;
            tr_im -= &tmp2;
        }
    }
    // |tr|² = tr_re² + tr_im²
    tmp.assign(&tr_re * &tr_re);
    tmp2.assign(&tr_im * &tr_im);
    tmp += &tmp2;
    // d² = 1 − |tr|²/4
    let one = RFloat::with_val(prec, 1.0);
    let four = RFloat::with_val(prec, 4.0);
    let d_sq = one - tmp / four;
    if d_sq.is_sign_negative() {
        return 0.0;
    }
    d_sq.sqrt().to_f64()
}

/// Diamond distance between an exact U2T and an f64 target matrix, computed
/// at MPFR-128 **without going through `U2T::to_float()`**.
///
/// At deep `lde` (k ≳ 50), `to_float()` rounds each ZOmega entry by up to
/// one f64 ulp ≈ 2.2×10⁻¹⁶ — and crucially, the rounded bits can match the
/// target's f64 bits exactly, hiding a real precision violation. We evaluate
/// the ring representation directly in MPFR.
///
/// For ω = e^(iπ/4) we have ω = (1+i)/√2, ω² = i, ω³ = (−1+i)/√2, so a
/// `ZOmega(a,b,c,d)` at unit scale (after dividing by √2^k) evaluates as:
///
/// ```text
/// re = (a + (b − d)/√2) / √2^k
/// im = (c + (b + d)/√2) / √2^k
/// ```
///
/// Uses the [`diamond_distance_float`] Frobenius reformulation `D² = q(8−q)/16`
/// (rather than `1 − |tr|²/4`) so the result is non-negative for any inputs;
/// the trace formula clamps to 0 when the f64 target's Frobenius norm
/// deviates from 2 by ~10⁻¹⁶.
///
/// Cost: ~2 μs/call. Fires only on SE hits, ~100× per `synthesize` call.
pub(crate) fn diamond_distance_u2t_float(u: &U2T, target: &Mat2) -> Float {
    let prec: u32 = 128;
    let two = RFloat::with_val(prec, 2.0);
    let inv_sqrt2 = RFloat::with_val(prec, 1.0) / two.clone().sqrt();

    // Build inv_scale = 1/√2^k exactly in MPFR. For even k this is just
    // 2^(-k/2) (a binary shift, no precision cost). For odd k it's
    // 2^(-(k-1)/2) · (1/√2).
    let half_k = u.k / 2;
    let mut inv_scale = RFloat::with_val(prec, 1.0);
    inv_scale >>= half_k;
    if u.k % 2 == 1 {
        inv_scale *= &inv_sqrt2;
    }

    // Convert one ZOmega to a (re, im) pair at *unit* scale (already divided
    // by √2^k). For random U(2) targets the entries are O(1).
    let zomega_to_mpfr_unit = |z: &ZOmega| -> (RFloat, RFloat) {
        let a = RFloat::with_val(prec, int_to_f64(z.a));
        let b = RFloat::with_val(prec, int_to_f64(z.b));
        let c = RFloat::with_val(prec, int_to_f64(z.c));
        let d = RFloat::with_val(prec, int_to_f64(z.d));
        let bd_diff = RFloat::with_val(prec, &b - &d);
        let bd_sum = RFloat::with_val(prec, &b + &d);
        let re_unscaled = a + bd_diff * &inv_sqrt2;
        let im_unscaled = c + bd_sum * &inv_sqrt2;
        (re_unscaled * &inv_scale, im_unscaled * &inv_scale)
    };

    let u_entries = [
        zomega_to_mpfr_unit(&u.u11),
        zomega_to_mpfr_unit(&u.u12),
        zomega_to_mpfr_unit(&u.u21),
        zomega_to_mpfr_unit(&u.u22),
    ];
    let t_entries: [(RFloat, RFloat); 4] = [
        (
            RFloat::with_val(prec, target[0][0].re),
            RFloat::with_val(prec, target[0][0].im),
        ),
        (
            RFloat::with_val(prec, target[0][1].re),
            RFloat::with_val(prec, target[0][1].im),
        ),
        (
            RFloat::with_val(prec, target[1][0].re),
            RFloat::with_val(prec, target[1][0].im),
        ),
        (
            RFloat::with_val(prec, target[1][1].re),
            RFloat::with_val(prec, target[1][1].im),
        ),
    ];

    // tr = Σ u · conj(t) at unit scale, then optimal phase φ = tr/|tr|.
    let mut tr_re = RFloat::with_val(prec, 0.0);
    let mut tr_im = RFloat::with_val(prec, 0.0);
    let mut tmp = RFloat::with_val(prec, 0.0);
    let mut tmp2 = RFloat::with_val(prec, 0.0);
    for ((u_re, u_im), (t_re, t_im)) in u_entries.iter().zip(t_entries.iter()) {
        // u · conj(t) = (u_re + i u_im)(t_re − i t_im)
        //             = (u_re·t_re + u_im·t_im) + i·(u_im·t_re − u_re·t_im)
        tmp.assign(u_re * t_re);
        tr_re += &tmp;
        tmp.assign(u_im * t_im);
        tr_re += &tmp;
        tmp.assign(u_im * t_re);
        tr_im += &tmp;
        tmp.assign(u_re * t_im);
        tr_im -= &tmp;
    }
    tmp.assign(&tr_re * &tr_re);
    tmp2.assign(&tr_im * &tr_im);
    let tr_abs_sq = RFloat::with_val(prec, &tmp + &tmp2);
    let tr_abs = tr_abs_sq.sqrt();
    // φ = tr / |tr|. If |tr| is degenerate (≈ 0), φ = 1 (resulting fro_sq is
    // large; no cancellation).
    let (phi_re, phi_im) = if tr_abs > 1e-30 {
        (
            RFloat::with_val(prec, &tr_re / &tr_abs),
            RFloat::with_val(prec, &tr_im / &tr_abs),
        )
    } else {
        (RFloat::with_val(prec, 1.0), RFloat::with_val(prec, 0.0))
    };

    // fro_sq = Σ |u − φ·t|²
    //        = Σ ((u_re − (φ_re·t_re − φ_im·t_im))² + (u_im − (φ_re·t_im + φ_im·t_re))²)
    let mut fro_sq = RFloat::with_val(prec, 0.0);
    for ((u_re, u_im), (t_re, t_im)) in u_entries.iter().zip(t_entries.iter()) {
        // φ·t = (φ_re·t_re − φ_im·t_im) + i·(φ_re·t_im + φ_im·t_re)
        tmp.assign(&phi_re * t_re);
        tmp2.assign(&phi_im * t_im);
        let phi_t_re = RFloat::with_val(prec, &tmp - &tmp2);
        tmp.assign(&phi_re * t_im);
        tmp2.assign(&phi_im * t_re);
        let phi_t_im = RFloat::with_val(prec, &tmp + &tmp2);
        let diff_re = RFloat::with_val(prec, u_re - &phi_t_re);
        let diff_im = RFloat::with_val(prec, u_im - &phi_t_im);
        tmp.assign(&diff_re * &diff_re);
        tmp2.assign(&diff_im * &diff_im);
        fro_sq += &tmp;
        fro_sq += &tmp2;
    }

    // D² = fro_sq · (8 − fro_sq) / 16. Always ≥ 0 for fro_sq ≤ 8 (the
    // Frobenius distance of two near-unitary 2×2 matrices is ≤ 2√2).
    let eight = RFloat::with_val(prec, 8.0);
    let sixteen = RFloat::with_val(prec, 16.0);
    let factor = RFloat::with_val(prec, &eight - &fro_sq);
    let d_sq = RFloat::with_val(prec, &fro_sq * &factor) / sixteen;
    if d_sq.is_sign_negative() {
        return 0.0;
    }
    d_sq.sqrt().to_f64()
}

/// Cost: ~3 μs/call (vs ~2 μs for U2T — twice the coefficients).
pub fn diamond_distance_u2q_float(u: &U2Q, target: &Mat2) -> Float {
    use std::f64::consts::PI;
    let prec: u32 = 128;
    let two = RFloat::with_val(prec, 2.0);
    let inv_sqrt2 = RFloat::with_val(prec, 1.0) / two.clone().sqrt();

    // inv_scale = 1/√2^k. Same construction as U2T: half-k binary shift,
    // odd-k extra factor of 1/√2.
    let half_k = u.k / 2;
    let mut inv_scale = RFloat::with_val(prec, 1.0);
    inv_scale >>= half_k;
    if u.k % 2 == 1 {
        inv_scale *= &inv_sqrt2;
    }

    // Precompute (cos(kπ/8), sin(kπ/8)) for k = 0..7 in MPFR. f64 sin/cos
    // are accurate to ~1 ulp at these arguments; lifting to MPFR at 128
    // bits is fine since the absolute error in the basis vector is what
    // bounds the distance error.
    let basis: [(RFloat, RFloat); 8] = std::array::from_fn(|k| {
        let theta = (k as f64) * PI / 8.0;
        (
            RFloat::with_val(prec, theta.cos()),
            RFloat::with_val(prec, theta.sin()),
        )
    });

    // Convert one ZZeta to a (re, im) pair at *unit* scale (already
    // divided by √2^k). For random U(2) targets the entries are O(1).
    let zzeta_to_mpfr_unit = |z: &ZZeta| -> (RFloat, RFloat) {
        let coeffs: [Float; 8] = [
            int_to_f64(z.a),
            int_to_f64(z.b),
            int_to_f64(z.c),
            int_to_f64(z.d),
            int_to_f64(z.e),
            int_to_f64(z.f),
            int_to_f64(z.g),
            int_to_f64(z.h),
        ];
        let mut re = RFloat::with_val(prec, 0.0);
        let mut im = RFloat::with_val(prec, 0.0);
        let mut tmp = RFloat::with_val(prec, 0.0);
        for k in 0..8 {
            let c = RFloat::with_val(prec, coeffs[k]);
            tmp.assign(&c * &basis[k].0);
            re += &tmp;
            tmp.assign(&c * &basis[k].1);
            im += &tmp;
        }
        (re * &inv_scale, im * &inv_scale)
    };

    let u_entries = [
        zzeta_to_mpfr_unit(&u.u11),
        zzeta_to_mpfr_unit(&u.u12),
        zzeta_to_mpfr_unit(&u.u21),
        zzeta_to_mpfr_unit(&u.u22),
    ];
    let t_entries: [(RFloat, RFloat); 4] = [
        (
            RFloat::with_val(prec, target[0][0].re),
            RFloat::with_val(prec, target[0][0].im),
        ),
        (
            RFloat::with_val(prec, target[0][1].re),
            RFloat::with_val(prec, target[0][1].im),
        ),
        (
            RFloat::with_val(prec, target[1][0].re),
            RFloat::with_val(prec, target[1][0].im),
        ),
        (
            RFloat::with_val(prec, target[1][1].re),
            RFloat::with_val(prec, target[1][1].im),
        ),
    ];

    // tr = Σ u · conj(t), optimal phase φ = tr/|tr|.
    let mut tr_re = RFloat::with_val(prec, 0.0);
    let mut tr_im = RFloat::with_val(prec, 0.0);
    let mut tmp = RFloat::with_val(prec, 0.0);
    let mut tmp2 = RFloat::with_val(prec, 0.0);
    for ((u_re, u_im), (t_re, t_im)) in u_entries.iter().zip(t_entries.iter()) {
        tmp.assign(u_re * t_re);
        tr_re += &tmp;
        tmp.assign(u_im * t_im);
        tr_re += &tmp;
        tmp.assign(u_im * t_re);
        tr_im += &tmp;
        tmp.assign(u_re * t_im);
        tr_im -= &tmp;
    }
    tmp.assign(&tr_re * &tr_re);
    tmp2.assign(&tr_im * &tr_im);
    let tr_abs_sq = RFloat::with_val(prec, &tmp + &tmp2);
    let tr_abs = tr_abs_sq.sqrt();
    let (phi_re, phi_im) = if tr_abs > 1e-30 {
        (
            RFloat::with_val(prec, &tr_re / &tr_abs),
            RFloat::with_val(prec, &tr_im / &tr_abs),
        )
    } else {
        (RFloat::with_val(prec, 1.0), RFloat::with_val(prec, 0.0))
    };

    // fro_sq = Σ |u − φ·t|²
    let mut fro_sq = RFloat::with_val(prec, 0.0);
    for ((u_re, u_im), (t_re, t_im)) in u_entries.iter().zip(t_entries.iter()) {
        tmp.assign(&phi_re * t_re);
        tmp2.assign(&phi_im * t_im);
        let phi_t_re = RFloat::with_val(prec, &tmp - &tmp2);
        tmp.assign(&phi_re * t_im);
        tmp2.assign(&phi_im * t_re);
        let phi_t_im = RFloat::with_val(prec, &tmp + &tmp2);
        let diff_re = RFloat::with_val(prec, u_re - &phi_t_re);
        let diff_im = RFloat::with_val(prec, u_im - &phi_t_im);
        tmp.assign(&diff_re * &diff_re);
        tmp2.assign(&diff_im * &diff_im);
        fro_sq += &tmp;
        fro_sq += &tmp2;
    }

    // D² = fro_sq · (8 − fro_sq) / 16.
    let eight = RFloat::with_val(prec, 8.0);
    let sixteen = RFloat::with_val(prec, 16.0);
    let factor = RFloat::with_val(prec, &eight - &fro_sq);
    let d_sq = RFloat::with_val(prec, &fro_sq * &factor) / sixteen;
    if d_sq.is_sign_negative() {
        return 0.0;
    }
    d_sq.sqrt().to_f64()
}
