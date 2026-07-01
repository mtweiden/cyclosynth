//! Diamond distance between 2×2 unitaries, in three precision flavours.
//!
//! All compute `D = √(1 − |tr(A·B†)|²/4)`, but via the algebraic Frobenius
//! reformulation `D² = q(8−q)/16`, `q = ‖A − φB‖²_F` (optimal phase φ = tr/|tr|):
//! it is non-negative by construction and avoids the catastrophic-cancellation
//! wall of `1 − |tr|²/4` at `D ≈ √machine_eps ≈ 1.5×10⁻⁸`. The U2T/U2Q entry
//! points evaluate the ring representation directly in MPFR (bypassing
//! `to_float()`, whose f64 quantization can hide a precision violation at deep k).
//! Proof/citations: `feedback_diamond_distance_frobenius.md`.

use num_complex::Complex;
use rug::Assign;
use crate::rings::MpFloat;

use crate::matrix::{U2T, U2Q};
use crate::rings::types::int_to_f64;
use crate::rings::{ZOmega, ZZeta};

/// A 2×2 matrix of complex f64 values (target matrices and float-converted elements).
pub type Mat2 = [[Complex<f64>; 2]; 2];

/// Project a 2×2 unitary onto SU(2): `U' = U / √det(U)`, so `det(U') = 1`
/// (global phase is unobservable). The guard handles degenerate `det ≈ 0`.
pub fn to_su2(u: &Mat2) -> Mat2 {
    let det = u[0][0] * u[1][1] - u[0][1] * u[1][0];
    let s = det.sqrt();
    if s.norm() < 1e-12 {
        return *u;
    }
    [
        [u[0][0] / s, u[0][1] / s],
        [u[1][0] / s, u[1][1] / s],
    ]
}

/// Diamond distance between two 2×2 unitaries in pure f64, via the algebraic
/// Frobenius reformulation `D² = q(8−q)/16` (precision-stable to f64 epsilon).
pub fn diamond_distance_float(a: &Mat2, b: &Mat2) -> f64 {
    let tr = a[0][0] * b[0][0].conj()
        + a[0][1] * b[0][1].conj()
        + a[1][0] * b[1][0].conj()
        + a[1][1] * b[1][1].conj();
    let tr_abs = tr.norm();
    // Optimal phase φ = tr/|tr|; if tr ≈ 0 pick φ=1 (irrelevant, fro_sq is large).
    let phi = if tr_abs > 1e-300 {
        tr / tr_abs
    } else {
        Complex::new(1.0, 0.0)
    };
    let mut fro_sq: f64 = 0.0;
    for i in 0..2 {
        for j in 0..2 {
            let diff = a[i][j] - phi * b[i][j];
            fro_sq += diff.norm_sqr();
        }
    }
    let d_sq = fro_sq * (8.0 - fro_sq) / 16.0;
    d_sq.max(0.0).sqrt()
}

/// Diamond distance between an exact U2T and an f64 target at MPFR-128, evaluating
/// each ZOmega entry directly (bypassing `to_float()`, whose f64 quantization can
/// hide a precision violation at deep k) with the `D² = q(8−q)/16` reformulation.
/// ~2 μs/call. For ω = (1+i)/√2: re = (a + (b−d)/√2)/√2^k, im = (c + (b+d)/√2)/√2^k.
pub(crate) fn diamond_distance_u2t_float(u: &U2T, target: &Mat2) -> f64 {
    let prec: u32 = 128;
    let two = MpFloat::with_val(prec, 2.0);
    let inv_sqrt2 = MpFloat::with_val(prec, 1.0) / two.clone().sqrt();

    // inv_scale = 1/√2^k: half-k binary shift, plus 1/√2 for odd k.
    let half_k = u.k / 2;
    let mut inv_scale = MpFloat::with_val(prec, 1.0);
    inv_scale >>= half_k;
    if u.k % 2 == 1 {
        inv_scale *= &inv_sqrt2;
    }

    // Convert one ZOmega to a (re, im) pair at unit scale (already /√2^k).
    let zomega_to_mpfr_unit = |z: &ZOmega| -> (MpFloat, MpFloat) {
        let a = MpFloat::with_val(prec, int_to_f64(z.a));
        let b = MpFloat::with_val(prec, int_to_f64(z.b));
        let c = MpFloat::with_val(prec, int_to_f64(z.c));
        let d = MpFloat::with_val(prec, int_to_f64(z.d));
        let bd_diff = MpFloat::with_val(prec, &b - &d);
        let bd_sum = MpFloat::with_val(prec, &b + &d);
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
    let t_entries: [(MpFloat, MpFloat); 4] = [
        (MpFloat::with_val(prec, target[0][0].re), MpFloat::with_val(prec, target[0][0].im)),
        (MpFloat::with_val(prec, target[0][1].re), MpFloat::with_val(prec, target[0][1].im)),
        (MpFloat::with_val(prec, target[1][0].re), MpFloat::with_val(prec, target[1][0].im)),
        (MpFloat::with_val(prec, target[1][1].re), MpFloat::with_val(prec, target[1][1].im)),
    ];

    // tr = Σ u · conj(t) at unit scale, then optimal phase φ = tr/|tr|.
    let mut tr_re = MpFloat::with_val(prec, 0.0);
    let mut tr_im = MpFloat::with_val(prec, 0.0);
    let mut tmp = MpFloat::with_val(prec, 0.0);
    let mut tmp2 = MpFloat::with_val(prec, 0.0);
    for ((u_re, u_im), (t_re, t_im)) in u_entries.iter().zip(t_entries.iter()) {
        // u · conj(t)
        tmp.assign(u_re * t_re); tr_re += &tmp;
        tmp.assign(u_im * t_im); tr_re += &tmp;
        tmp.assign(u_im * t_re); tr_im += &tmp;
        tmp.assign(u_re * t_im); tr_im -= &tmp;
    }
    tmp.assign(&tr_re * &tr_re);
    tmp2.assign(&tr_im * &tr_im);
    let tr_abs_sq = MpFloat::with_val(prec, &tmp + &tmp2);
    let tr_abs = tr_abs_sq.sqrt();
    // φ = tr/|tr|; if |tr| ≈ 0 pick φ=1 (irrelevant, fro_sq is large).
    let (phi_re, phi_im) = if tr_abs > 1e-30 {
        (
            MpFloat::with_val(prec, &tr_re / &tr_abs),
            MpFloat::with_val(prec, &tr_im / &tr_abs),
        )
    } else {
        (MpFloat::with_val(prec, 1.0), MpFloat::with_val(prec, 0.0))
    };

    // fro_sq = Σ |u − φ·t|²
    let mut fro_sq = MpFloat::with_val(prec, 0.0);
    for ((u_re, u_im), (t_re, t_im)) in u_entries.iter().zip(t_entries.iter()) {
        tmp.assign(&phi_re * t_re);
        tmp2.assign(&phi_im * t_im);
        let phi_t_re = MpFloat::with_val(prec, &tmp - &tmp2);
        tmp.assign(&phi_re * t_im);
        tmp2.assign(&phi_im * t_re);
        let phi_t_im = MpFloat::with_val(prec, &tmp + &tmp2);
        let diff_re = MpFloat::with_val(prec, u_re - &phi_t_re);
        let diff_im = MpFloat::with_val(prec, u_im - &phi_t_im);
        tmp.assign(&diff_re * &diff_re);
        tmp2.assign(&diff_im * &diff_im);
        fro_sq += &tmp;
        fro_sq += &tmp2;
    }

    // D² = fro_sq · (8 − fro_sq) / 16.
    let eight = MpFloat::with_val(prec, 8.0);
    let sixteen = MpFloat::with_val(prec, 16.0);
    let factor = MpFloat::with_val(prec, &eight - &fro_sq);
    let d_sq = MpFloat::with_val(prec, &fro_sq * &factor) / sixteen;
    if d_sq.is_sign_negative() {
        return 0.0;
    }
    d_sq.sqrt().to_f64()
}

/// U2Q/ZZeta analogue of [`diamond_distance_u2t_float`], evaluating each ZZeta
/// entry directly (basis `ζ^k = (cos kπ/8, sin kπ/8)`). ~3 μs/call.
pub fn diamond_distance_u2q_float(u: &U2Q, target: &Mat2) -> f64 {
    use std::f64::consts::PI;
    let prec: u32 = 128;
    let two = MpFloat::with_val(prec, 2.0);
    let inv_sqrt2 = MpFloat::with_val(prec, 1.0) / two.clone().sqrt();

    // inv_scale = 1/√2^k: half-k binary shift, plus 1/√2 for odd k.
    let half_k = u.k / 2;
    let mut inv_scale = MpFloat::with_val(prec, 1.0);
    inv_scale >>= half_k;
    if u.k % 2 == 1 {
        inv_scale *= &inv_sqrt2;
    }

    // Precompute (cos(kπ/8), sin(kπ/8)) for k = 0..7 in MPFR from f64 sin/cos.
    let basis: [(MpFloat, MpFloat); 8] = std::array::from_fn(|k| {
        let theta = (k as f64) * PI / 8.0;
        (
            MpFloat::with_val(prec, theta.cos()),
            MpFloat::with_val(prec, theta.sin()),
        )
    });

    // Convert one ZZeta to a (re, im) pair at unit scale (already /√2^k).
    let zzeta_to_mpfr_unit = |z: &ZZeta| -> (MpFloat, MpFloat) {
        let coeffs: [f64; 8] = [
            int_to_f64(z.a), int_to_f64(z.b), int_to_f64(z.c), int_to_f64(z.d),
            int_to_f64(z.e), int_to_f64(z.f), int_to_f64(z.g), int_to_f64(z.h),
        ];
        let mut re = MpFloat::with_val(prec, 0.0);
        let mut im = MpFloat::with_val(prec, 0.0);
        let mut tmp = MpFloat::with_val(prec, 0.0);
        for k in 0..8 {
            let c = MpFloat::with_val(prec, coeffs[k]);
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
    let t_entries: [(MpFloat, MpFloat); 4] = [
        (MpFloat::with_val(prec, target[0][0].re), MpFloat::with_val(prec, target[0][0].im)),
        (MpFloat::with_val(prec, target[0][1].re), MpFloat::with_val(prec, target[0][1].im)),
        (MpFloat::with_val(prec, target[1][0].re), MpFloat::with_val(prec, target[1][0].im)),
        (MpFloat::with_val(prec, target[1][1].re), MpFloat::with_val(prec, target[1][1].im)),
    ];

    // tr = Σ u · conj(t), optimal phase φ = tr/|tr|.
    let mut tr_re = MpFloat::with_val(prec, 0.0);
    let mut tr_im = MpFloat::with_val(prec, 0.0);
    let mut tmp = MpFloat::with_val(prec, 0.0);
    let mut tmp2 = MpFloat::with_val(prec, 0.0);
    for ((u_re, u_im), (t_re, t_im)) in u_entries.iter().zip(t_entries.iter()) {
        tmp.assign(u_re * t_re); tr_re += &tmp;
        tmp.assign(u_im * t_im); tr_re += &tmp;
        tmp.assign(u_im * t_re); tr_im += &tmp;
        tmp.assign(u_re * t_im); tr_im -= &tmp;
    }
    tmp.assign(&tr_re * &tr_re);
    tmp2.assign(&tr_im * &tr_im);
    let tr_abs_sq = MpFloat::with_val(prec, &tmp + &tmp2);
    let tr_abs = tr_abs_sq.sqrt();
    let (phi_re, phi_im) = if tr_abs > 1e-30 {
        (
            MpFloat::with_val(prec, &tr_re / &tr_abs),
            MpFloat::with_val(prec, &tr_im / &tr_abs),
        )
    } else {
        (MpFloat::with_val(prec, 1.0), MpFloat::with_val(prec, 0.0))
    };

    // fro_sq = Σ |u − φ·t|²
    let mut fro_sq = MpFloat::with_val(prec, 0.0);
    for ((u_re, u_im), (t_re, t_im)) in u_entries.iter().zip(t_entries.iter()) {
        tmp.assign(&phi_re * t_re);
        tmp2.assign(&phi_im * t_im);
        let phi_t_re = MpFloat::with_val(prec, &tmp - &tmp2);
        tmp.assign(&phi_re * t_im);
        tmp2.assign(&phi_im * t_re);
        let phi_t_im = MpFloat::with_val(prec, &tmp + &tmp2);
        let diff_re = MpFloat::with_val(prec, u_re - &phi_t_re);
        let diff_im = MpFloat::with_val(prec, u_im - &phi_t_im);
        tmp.assign(&diff_re * &diff_re);
        tmp2.assign(&diff_im * &diff_im);
        fro_sq += &tmp;
        fro_sq += &tmp2;
    }

    // D² = fro_sq · (8 − fro_sq) / 16.
    let eight = MpFloat::with_val(prec, 8.0);
    let sixteen = MpFloat::with_val(prec, 16.0);
    let factor = MpFloat::with_val(prec, &eight - &fro_sq);
    let d_sq = MpFloat::with_val(prec, &fro_sq * &factor) / sixteen;
    if d_sq.is_sign_negative() {
        return 0.0;
    }
    d_sq.sqrt().to_f64()
}
