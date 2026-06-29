//! Aligned lattice search for Clifford+T synthesis.
//!
//! Finds integer vectors x = (a1, b1, c1, d1, a2, b2, c2, d2) satisfying:
//!   - ‖x‖² = 2^k                                         (norm constraint)
//!   - b1(a1+c1) + d1(c1-a1) + b2(a2+c2) + d2(c2-a2) = 0 (unitarity)
//!   - (x · align_vec)² ≥ 2^k · (1 − ε²)                  (alignment)
//!
//! The alignment vector is align_vec = Σ_uv^T · v, where Σ_uv maps the
//! integer lattice coordinates to the uv parameterization.
//!
//! ## The coordinate vocabulary (authoritative; cross-linked from the
//! ζ₁₆ mirror and clifford_sqrt_t)
//!
//! - **uv** (`v: [f64; 4]`): the det-normalized first column of the
//!   target unitary as reals — (Re u₁, Im u₁, Re u₂, Im u₂). All
//!   synthesis targets reduce to this direction vector.
//! - **y** (lattice y / cap-center direction): uv pushed into integer-
//!   lattice coordinates and scaled to the paper's norm convention
//!   (‖y‖² = 2^(k−1) in 8D, 2^k/4 in 16D); `uv_to_lattice_y*` builds
//!   it. The enumeration cap is centered on y.
//! - **align_vec** (`av`): the row vector whose dot with a candidate
//!   lattice point measures cap alignment; `compute_align_vec*`.

// Search functions thread several tuning knobs (norm bound, alignment
// vector, threshold, max solutions, output sink) through their signatures.
// The alternative would be a "search options" struct, which obscures the
// call site without simplifying the underlying interface.
#![allow(clippy::too_many_arguments)]

use num_complex::Complex64;
use std::f64::consts::FRAC_1_SQRT_2;
use crate::matrix::U2T;
use crate::rings::types::int_to_f64;
use crate::rings::{MpFloat, ZOmega};

// ─── Alignment vector ─────────────────────────────────────────────────────────

/// Compute the alignment vector align_vec = Σ_uv^T · v.
///
/// v = [Re(u1), Im(u1), Re(u2), Im(u2)] (4-component, unit vector).
/// align_vec is 8-component; index k corresponds to the k-th lattice coordinate
/// in the order (a1, b1, c1, d1, a2, b2, c2, d2).
///
/// Formula (from the column structure of Σ_uv):
/// ```text
///   align[0] = v[0]
///   align[1] = (v[0] + v[1]) / √2
///   align[2] = v[1]
///   align[3] = (−v[0] + v[1]) / √2
///   align[4] = v[2]
///   align[5] = (v[2] + v[3]) / √2
///   align[6] = v[3]
///   align[7] = (−v[2] + v[3]) / √2
/// ```
pub fn compute_align_vec(v: [f64; 4]) -> [f64; 8] {
    let r = FRAC_1_SQRT_2;
    [
        v[0],
        (v[0] + v[1]) * r,
        v[1],
        (-v[0] + v[1]) * r,
        v[2],
        (v[2] + v[3]) * r,
        v[3],
        (-v[2] + v[3]) * r,
    ]
}

// ─── T / T† transforms on uv ─────────────────────────────────────────────────

/// Apply T† to uv: maps uv(V) → uv(V·T†).
///
/// T† = [[1,0],[0,ω̄]]; det(V·T†) = ω̄ for SU(2) V.
/// After det normalization in unitary_to_uv (dividing first column by √det),
/// both u1 and u2 rotate by e^{iπ/8}: uv(V·T†) = e^{iπ/8} · uv(V).
pub fn apply_t_dag_to_uv(v: [f64; 4]) -> [f64; 4] {
    let (c, s) = (std::f64::consts::FRAC_PI_8.cos(), std::f64::consts::FRAC_PI_8.sin());
    [v[0]*c - v[1]*s, v[0]*s + v[1]*c, v[2]*c - v[3]*s, v[2]*s + v[3]*c]
}

/// Apply T to uv: maps uv(V) → uv(V·T).
///
/// T = [[1,0],[0,ω]]; det(V·T) = ω for SU(2) V.
/// After det normalization in unitary_to_uv (dividing first column by √det),
/// both u1 and u2 rotate by e^{-iπ/8}: uv(V·T) = e^{-iπ/8} · uv(V).
pub fn apply_t_to_uv(v: [f64; 4]) -> [f64; 4] {
    let (c, s) = (std::f64::consts::FRAC_PI_8.cos(), std::f64::consts::FRAC_PI_8.sin());
    [v[0]*c + v[1]*s, -v[0]*s + v[1]*c, v[2]*c + v[3]*s, -v[2]*s + v[3]*c]
}

/// Apply an exact U2T unitary's dagger to a uv direction vector.
///
/// Computes uv(C†·V) given uv(V) = v.
///
/// For SU(2) C (det=1) the phase cancels and the result is simply C†·v.
/// For general C (det ≠ 1) the unitary_to_uv normalization introduces an extra
/// factor: uv(C†·V) = (C†·v / √2^k) / √conj(det(C)).
/// Clifford table entries are SU(2) (det=1) so the correction is 1 there;
/// MA prefix products (H, S, T) have det ≠ 1 and require the correction.
pub fn apply_u2t_dag_to_uv(c: &U2T, v: [f64; 4]) -> [f64; 4] {
    let v1 = Complex64::new(v[0], v[1]);
    let v2 = Complex64::new(v[2], v[3]);
    let scale = 1.0 / (2.0_f64.powi(c.k as i32)).sqrt();
    let w1 = (c.u11.to_complex().conj() * v1 + c.u21.to_complex().conj() * v2) * scale;
    let w2 = (c.u12.to_complex().conj() * v1 + c.u22.to_complex().conj() * v2) * scale;
    // Correct for det(C) ≠ 1: divide by √conj(det(C)).
    let det = (c.u11.to_complex() * c.u22.to_complex()
              - c.u12.to_complex() * c.u21.to_complex()) * scale * scale;
    let sqrt_conj_det = det.conj().sqrt();
    if sqrt_conj_det.norm() > 1e-12 {
        let inv = Complex64::new(1.0, 0.0) / sqrt_conj_det;
        [
            w1.re * inv.re - w1.im * inv.im,
            w1.re * inv.im + w1.im * inv.re,
            w2.re * inv.re - w2.im * inv.im,
            w2.re * inv.im + w2.im * inv.re,
        ]
    } else {
        [w1.re, w1.im, w2.re, w2.im]
    }
}

/// MPFR analog of [`apply_u2t_dag_to_uv`]: column 1 of `C† · target` in the
/// √det-normalized uv form, from exact ring `C` and an MPFR target column
/// `v = [Re v1, Im v1, Re v2, Im v2]`. Preserves precision below the f64 ULP
/// for the deep-ε prefix-split search (prefix coefficients are far inside
/// i64 there, so `int_to_f64` is exact). `prec` is the working precision.
pub fn apply_u2t_dag_to_uv_mpfr(c: &U2T, v: &[MpFloat; 4], prec: u32) -> [MpFloat; 4] {
    let f = |x: f64| MpFloat::with_val(prec, x);
    let mul = |a: &MpFloat, b: &MpFloat| MpFloat::with_val(prec, a * b);
    let add = |a: &MpFloat, b: &MpFloat| MpFloat::with_val(prec, a + b);
    let sub = |a: &MpFloat, b: &MpFloat| MpFloat::with_val(prec, a - b);

    // ZOmega → (re, im): re = a + (b−d)/√2, im = c + (b+d)/√2.
    let r2 = MpFloat::with_val(prec, 2.0).sqrt().recip();
    let zo = |z: &ZOmega| -> (MpFloat, MpFloat) {
        let (a, b, cc, d) = (f(int_to_f64(z.a)), f(int_to_f64(z.b)), f(int_to_f64(z.c)), f(int_to_f64(z.d)));
        (add(&a, &mul(&sub(&b, &d), &r2)), add(&cc, &mul(&add(&b, &d), &r2)))
    };
    type C = (MpFloat, MpFloat);
    let cmul = |x: &C, y: &C| -> C { (sub(&mul(&x.0, &y.0), &mul(&x.1, &y.1)), add(&mul(&x.0, &y.1), &mul(&x.1, &y.0))) };
    let cmul_conj = |x: &C, y: &C| -> C { (add(&mul(&x.0, &y.0), &mul(&x.1, &y.1)), sub(&mul(&x.0, &y.1), &mul(&x.1, &y.0))) };
    let cadd = |x: &C, y: &C| -> C { (add(&x.0, &y.0), add(&x.1, &y.1)) };
    // Principal complex sqrt: √((|z|+re)/2) + sign(im)·√((|z|−re)/2)·i.
    let csqrt = |z: &C| -> C {
        let mag = add(&mul(&z.0, &z.0), &mul(&z.1, &z.1)).sqrt();
        let re = MpFloat::with_val(prec, add(&mag, &z.0) / 2.0).sqrt();
        let im = MpFloat::with_val(prec, sub(&mag, &z.0) / 2.0).sqrt();
        (re, if z.1.is_sign_negative() { -im } else { im })
    };

    let (u11, u12, u21, u22) = (zo(&c.u11), zo(&c.u12), zo(&c.u21), zo(&c.u22));
    let v1 = (v[0].clone(), v[1].clone());
    let v2 = (v[2].clone(), v[3].clone());

    // scale = 1/√2^k.
    let mut pow = MpFloat::with_val(prec, 1.0);
    pow <<= c.k / 2;
    if c.k % 2 == 1 {
        pow *= MpFloat::with_val(prec, 2.0).sqrt();
    }
    let scale = MpFloat::with_val(prec, 1.0) / pow;

    // w = C† · v, scaled.
    let w1 = cadd(&cmul_conj(&u11, &v1), &cmul_conj(&u21, &v2));
    let w2 = cadd(&cmul_conj(&u12, &v1), &cmul_conj(&u22, &v2));
    let w1 = (mul(&w1.0, &scale), mul(&w1.1, &scale));
    let w2 = (mul(&w2.0, &scale), mul(&w2.1, &scale));

    // det(C)·scale², then divide both rows by √conj(det) to land in SU(2).
    let scale2 = mul(&scale, &scale);
    let det0 = { let p = cmul(&u11, &u22); let q = cmul(&u12, &u21); (sub(&p.0, &q.0), sub(&p.1, &q.1)) };
    let det = (mul(&det0.0, &scale2), mul(&det0.1, &scale2));
    let s = csqrt(&(det.0.clone(), MpFloat::with_val(prec, -&det.1))); // √conj(det)
    let s_norm_sq = add(&mul(&s.0, &s.0), &mul(&s.1, &s.1));
    if s_norm_sq.to_f64() > 1e-24 {
        let inv_re = MpFloat::with_val(prec, &s.0 / &s_norm_sq);
        let inv_im = MpFloat::with_val(prec, &s.1 / &s_norm_sq);
        let inv = (inv_re, -inv_im);
        let r1 = cmul(&w1, &inv);
        let r2c = cmul(&w2, &inv);
        [r1.0, r1.1, r2c.0, r2c.1]
    } else {
        [w1.0, w1.1, w2.0, w2.1]
    }
}

/// Normalize a 4-vector; returns None if near-zero.
pub fn normalize4(v: [f64; 4]) -> Option<[f64; 4]> {
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm < 1e-12 {
        None
    } else {
        Some(v.map(|x| x / norm))
    }
}

// ─── Integer square root ──────────────────────────────────────────────────────

/// Floor of √n for non-negative n.
#[inline]
pub fn integer_sqrt(n: i64) -> i64 {
    if n <= 0 {
        return 0;
    }
    let mut s = (n as f64).sqrt() as i64;
    // Correct for floating-point rounding (at most a few steps).
    while s > 0 && s * s > n {
        s -= 1;
    }
    while (s + 1) * (s + 1) <= n {
        s += 1;
    }
    s
}

// ─── Alignment check & record ─────────────────────────────────────────────────

/// Record a candidate if it passes the alignment threshold.
///
/// threshold_sq = 2^k · (1 − ε²). A solution passes if (x · av)² ≥ threshold_sq.
#[inline]
fn record_if_aligned(
    a1: i64, b1: i64, c1: i64, d1: i64,
    a2: i64, b2: i64, c2: i64, d2: i64,
    av: &[f64; 8],
    thresh_sq: f64,
    out: &mut Vec<[i64; 8]>,
    max_sol: usize,
) {
    if thresh_sq > 0.0 {
        let dot = a1 as f64 * av[0]
                + b1 as f64 * av[1]
                + c1 as f64 * av[2]
                + d1 as f64 * av[3]
                + a2 as f64 * av[4]
                + b2 as f64 * av[5]
                + c2 as f64 * av[6]
                + d2 as f64 * av[7];
        if dot * dot < thresh_sq {
            return;
        }
    }
    if out.len() < max_sol {
        out.push([a1, b1, c1, d1, a2, b2, c2, d2]);
    }
}

// ─── Algebraic solver ─────────────────────────────────────────────────────────

/// Solve the remaining (b, d) pair of element `SLOT` (0 → u1's (b1, d1),
/// 1 → u2's (b2, d2)) given that element's (a, c), the fully-fixed other
/// element, and the constraints
///   b² + d² = r
///   b·A + d·B = rhs,   where A = a+c, B = c−a.
/// Records valid full 8-vectors into `out`. SLOT is const so each
/// instantiation monomorphizes to a constant-indexed loop nest.
#[inline]
#[allow(clippy::too_many_arguments)]
fn solve_bd_pair<const SLOT: usize>(
    a: i64, c: i64,
    other: [i64; 4],
    r: i64,
    rhs: i64,
    av: &[f64; 8],
    thresh_sq: f64,
    out: &mut Vec<[i64; 8]>,
    max_sol: usize,
) {
    let record = |b: i64, d: i64, out: &mut Vec<[i64; 8]>| {
        let [ao, bo, co, dd] = other;
        if SLOT == 0 {
            record_if_aligned(a, b, c, d, ao, bo, co, dd, av, thresh_sq, out, max_sol);
        } else {
            record_if_aligned(ao, bo, co, dd, a, b, c, d, av, thresh_sq, out, max_sol);
        }
    };
    let big_a = a + c;
    let big_b = c - a;

    if big_a == 0 && big_b == 0 {
        if rhs != 0 {
            return;
        }
        let max_b = integer_sqrt(r);
        for b in -max_b..=max_b {
            let d_sq = r - b * b;
            if d_sq < 0 {
                continue;
            }
            let d_abs = integer_sqrt(d_sq);
            if d_abs * d_abs != d_sq {
                continue;
            }
            for &d in &[d_abs, -d_abs] {
                if d < 0 && d_abs == 0 {
                    continue;
                }
                record(b, d, out);
                if out.len() >= max_sol {
                    return;
                }
            }
        }
        return;
    }

    if big_a == 0 {
        // d = rhs / B
        if rhs % big_b != 0 {
            return;
        }
        let d = rhs / big_b;
        let b_sq = r - d * d;
        if b_sq < 0 {
            return;
        }
        let b_abs = integer_sqrt(b_sq);
        if b_abs * b_abs != b_sq {
            return;
        }
        for &b in &[b_abs, -b_abs] {
            if b < 0 && b_abs == 0 {
                continue;
            }
            record(b, d, out);
            if out.len() >= max_sol {
                return;
            }
        }
        return;
    }

    if big_b == 0 {
        // b = rhs / A
        if rhs % big_a != 0 {
            return;
        }
        let b = rhs / big_a;
        let d_sq = r - b * b;
        if d_sq < 0 {
            return;
        }
        let d_abs = integer_sqrt(d_sq);
        if d_abs * d_abs != d_sq {
            return;
        }
        for &d in &[d_abs, -d_abs] {
            if d < 0 && d_abs == 0 {
                continue;
            }
            record(b, d, out);
            if out.len() >= max_sol {
                return;
            }
        }
        return;
    }

    // General case: from b·A + d·B = rhs and b² + d² = r:
    //   (A²+B²)·d² − 2·rhs·B·d + (rhs² − A²·r) = 0
    //   disc = 4·A²·((A²+B²)·r − rhs²)
    let s2 = big_a * big_a + big_b * big_b;
    let disc_val = s2 * r - rhs * rhs;
    if disc_val < 0 {
        return;
    }
    let disc = 4 * big_a * big_a * disc_val;
    if disc < 0 {
        return;
    }
    let sqrt_disc = integer_sqrt(disc);
    if sqrt_disc * sqrt_disc != disc {
        return;
    }

    let denom = 2 * s2;
    for sign in [1i64, -1] {
        if sign == -1 && sqrt_disc == 0 {
            continue;
        }
        let numer_d = 2 * rhs * big_b + sign * sqrt_disc;
        if numer_d % denom != 0 {
            continue;
        }
        let d = numer_d / denom;
        let numer_b = rhs - d * big_b;
        if numer_b % big_a != 0 {
            continue;
        }
        let b = numer_b / big_a;
        if b * b + d * d != r {
            continue;
        }
        record(b, d, out);
        if out.len() >= max_sol {
            return;
        }
    }
}

// ─── Search engine ────────────────────────────────────────────────────────────

use rayon::prelude::*;

/// Brute full-sphere enumeration: enumerate coordinates I0..I5 of the
/// 8-vector in that order under Cauchy–Schwarz pruning, then solve the
/// remaining (b, d) pair of element `SLOT` algebraically via
/// [`solve_bd_pair`]. The order is const-generic so each instantiation
/// monomorphizes to a constant-indexed loop nest (the two orders in use:
/// enumerate u2 and solve u1's pair, or enumerate u1 and solve u2's —
/// chosen by where the alignment energy sits). Parallelised over the
/// outer (I0, I1) pairs via rayon.
#[allow(clippy::too_many_arguments)]
fn brute_enum<
    const I0: usize, const I1: usize, const I2: usize,
    const I3: usize, const I4: usize, const I5: usize,
    const SLOT: usize,
>(
    target_norm: i64,
    av: &[f64; 8],
    thresh_sq: f64,
    max_sol: usize,
    out: &mut Vec<[i64; 8]>,
) {
    let do_prune = thresh_sq > 0.0;
    let thresh = thresh_sq.sqrt();

    // av_sq[j] = Σ av[i]² over the coordinates still open after level j
    // (the Cauchy–Schwarz prune tail).
    let order = [I0, I1, I2, I3, I4, I5];
    let mut av_sq = [0.0f64; 6];
    let mut tail: f64 = av.iter().map(|x| x * x).sum();
    for (j, &idx) in order.iter().enumerate() {
        tail -= av[idx] * av[idx];
        av_sq[j] = tail;
    }

    let max0 = integer_sqrt(target_norm);
    let pairs: Vec<(i64, i64, i64, f64)> = (-max0..=max0)
        .flat_map(|v0| {
            let rem1 = target_norm - v0 * v0;
            if rem1 < 0 { return vec![]; }
            let pdot1 = v0 as f64 * av[I0];
            if do_prune && pdot1.abs() + (rem1 as f64 * av_sq[0]).sqrt() < thresh {
                return vec![];
            }
            let max1 = integer_sqrt(rem1);
            (-max1..=max1).filter_map(|v1| {
                let rem2 = rem1 - v1 * v1;
                if rem2 < 0 { return None; }
                let pdot2 = pdot1 + v1 as f64 * av[I1];
                if do_prune && pdot2.abs() + (rem2 as f64 * av_sq[1]).sqrt() < thresh {
                    return None;
                }
                Some((v0, v1, rem2, pdot2))
            }).collect::<Vec<_>>()
        })
        .collect();

    let batches: Vec<Vec<[i64; 8]>> = pairs
        .into_par_iter()
        .filter_map(|(v0, v1, rem2, pdot2)| {
            let local = brute_enum_inner::<I0, I1, I2, I3, I4, I5, SLOT>(
                v0, v1, rem2, pdot2, av, thresh_sq, &av_sq, do_prune, thresh, max_sol,
            );
            if local.is_empty() { None } else { Some(local) }
        })
        .collect();

    for batch in batches {
        for sol in batch {
            if out.len() >= max_sol { return; }
            out.push(sol);
        }
    }
}

/// Inner 4-level nest of [`brute_enum`] for one fixed (I0, I1) pair.
#[allow(clippy::too_many_arguments)]
fn brute_enum_inner<
    const I0: usize, const I1: usize, const I2: usize,
    const I3: usize, const I4: usize, const I5: usize,
    const SLOT: usize,
>(
    v0: i64,
    v1: i64,
    rem2: i64,
    pdot2: f64,
    av: &[f64; 8],
    thresh_sq: f64,
    av_sq: &[f64; 6],
    do_prune: bool,
    thresh: f64,
    max_sol: usize,
) -> Vec<[i64; 8]> {
    let mut out: Vec<[i64; 8]> = Vec::new();
    let mut x = [0i64; 8];
    x[I0] = v0;
    x[I1] = v1;

    let max2 = integer_sqrt(rem2);
    for v2 in -max2..=max2 {
        let rem3 = rem2 - v2 * v2;
        if rem3 < 0 { continue; }
        let pdot3 = pdot2 + v2 as f64 * av[I2];
        if do_prune && pdot3.abs() + (rem3 as f64 * av_sq[2]).sqrt() < thresh { continue; }
        x[I2] = v2;

        let max3 = integer_sqrt(rem3);
        for v3 in -max3..=max3 {
            let rem4 = rem3 - v3 * v3;
            if rem4 < 0 { continue; }
            let pdot4 = pdot3 + v3 as f64 * av[I3];
            if do_prune && pdot4.abs() + (rem4 as f64 * av_sq[3]).sqrt() < thresh { continue; }
            x[I3] = v3;

            let max4 = integer_sqrt(rem4);
            for v4 in -max4..=max4 {
                let rem5 = rem4 - v4 * v4;
                if rem5 < 0 { continue; }
                let pdot5 = pdot4 + v4 as f64 * av[I4];
                if do_prune && pdot5.abs() + (rem5 as f64 * av_sq[4]).sqrt() < thresh { continue; }
                x[I4] = v4;

                let max5 = integer_sqrt(rem5);
                for v5 in -max5..=max5 {
                    let r = rem5 - v5 * v5;
                    if r < 0 { continue; }
                    let pdot6 = pdot5 + v5 as f64 * av[I5];
                    if do_prune && pdot6.abs() + (r as f64 * av_sq[5]).sqrt() < thresh { continue; }
                    x[I5] = v5;

                    // The fixed element's cross term feeds the solved
                    // pair's linear constraint.
                    let base = 4 * (1 - SLOT);
                    let rhs = -(x[base + 1] * (x[base] + x[base + 2])
                        + x[base + 3] * (x[base + 2] - x[base]));
                    let own = 4 * SLOT;
                    let other = [x[base], x[base + 1], x[base + 2], x[base + 3]];
                    solve_bd_pair::<SLOT>(
                        x[own], x[own + 2], other, r, rhs, av, thresh_sq,
                        &mut out, max_sol,
                    );
                    if out.len() >= max_sol { return out; }
                }
            }
        }
    }
    out
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Find integer lattice solutions aligned with the target uv direction.
///
/// # Arguments
/// * `v` – target uv parameterization `[Re(u1), Im(u1), Re(u2), Im(u2)]`, unit vector.
/// * `k` – denominator exponent; target norm = 2^k.
/// * `epsilon` – approximation precision; alignment threshold = 2^k · (1−ε²).
/// * `max_solutions` – stop after collecting this many solutions.
///
/// # Returns
/// Vector of `[a1, b1, c1, d1, a2, b2, c2, d2]` satisfying all constraints.
pub fn brute_aligned_search(
    v: [f64; 4],
    k: u32,
    epsilon: f64,
    max_solutions: usize,
) -> Vec<[i64; 8]> {
    if max_solutions == 0 {
        return Vec::new();
    }

    let target_norm: i64 = 1i64 << k;
    let av = compute_align_vec(v);

    let thresh_sq = if epsilon > 0.0 {
        target_norm as f64 * (1.0 - epsilon * epsilon)
    } else {
        0.0
    };

    // Choose search strategy based on where alignment energy is concentrated.
    let u1_energy: f64 = av[0]*av[0] + av[1]*av[1] + av[2]*av[2] + av[3]*av[3];
    let u2_energy: f64 = av[4]*av[4] + av[5]*av[5] + av[6]*av[6] + av[7]*av[7];

    let mut out = Vec::new();
    if u1_energy >= u2_energy {
        // Energy in u1: enumerate u1 fully, solve u2's pair.
        brute_enum::<0, 1, 2, 3, 4, 6, 1>(target_norm, &av, thresh_sq, max_solutions, &mut out);
    } else {
        brute_enum::<0, 2, 4, 6, 5, 7, 0>(target_norm, &av, thresh_sq, max_solutions, &mut out);
    }
    out
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integer_sqrt() {
        assert_eq!(integer_sqrt(0), 0);
        assert_eq!(integer_sqrt(1), 1);
        assert_eq!(integer_sqrt(3), 1);
        assert_eq!(integer_sqrt(4), 2);
        assert_eq!(integer_sqrt(9), 3);
        assert_eq!(integer_sqrt(10), 3);
        assert_eq!(integer_sqrt(16), 4);
        assert_eq!(integer_sqrt(4096), 64);
    }

    #[test]
    fn test_compute_align_vec() {
        // v = [1, 0, 0, 0] → only u1 real part active
        let v = [1.0f64, 0.0, 0.0, 0.0];
        let av = compute_align_vec(v);
        assert!((av[0] - 1.0).abs() < 1e-12, "av[0]={}", av[0]);
        assert!((av[1] - FRAC_1_SQRT_2).abs() < 1e-12, "av[1]={}", av[1]);
        assert!(av[2].abs() < 1e-12, "av[2]={}", av[2]);
        assert!((av[3] + FRAC_1_SQRT_2).abs() < 1e-12, "av[3]={}", av[3]);
        for i in 4..8 {
            assert!(av[i].abs() < 1e-12, "av[{i}]={}", av[i]);
        }
    }

    #[test]
    fn test_apply_t_transforms_roundtrip() {
        // Applying T then T† should return to original.
        let v = [0.5f64, 0.3, 0.6, 0.4];
        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        let v = v.map(|x| x / norm);

        let v_t = apply_t_to_uv(v);
        let v_back = apply_t_dag_to_uv(v_t);
        for i in 0..4 {
            assert!((v[i] - v_back[i]).abs() < 1e-12, "roundtrip failed at {i}");
        }
    }

    /// All solutions must satisfy the norm and unitarity constraints.
    fn check_solutions(sols: &[[i64; 8]], k: u32) {
        let target_norm = 1i64 << k;
        for sol in sols {
            let [a1, b1, c1, d1, a2, b2, c2, d2] = *sol;
            let norm_sq = a1*a1 + b1*b1 + c1*c1 + d1*d1 + a2*a2 + b2*b2 + c2*c2 + d2*d2;
            assert_eq!(norm_sq, target_norm, "norm_sq={norm_sq} target={target_norm} sol={sol:?}");
            let unit = b1*(a1+c1) + d1*(c1-a1) + b2*(a2+c2) + d2*(c2-a2);
            assert_eq!(unit, 0, "unitarity violated: {unit} for sol={sol:?}");
        }
    }

    #[test]
    fn test_search_k0_identity_dir() {
        let v = [1.0f64, 0.0, 0.0, 0.0];
        let sols = brute_aligned_search(v, 0, 0.0, 100);
        assert!(!sols.is_empty(), "Should find solutions at k=0");
        check_solutions(&sols, 0);
    }

    #[test]
    fn test_search_k1_no_alignment() {
        // With epsilon=0 (no alignment filter), enumerate all valid solutions at k=1.
        let v = [1.0f64, 0.0, 0.0, 0.0];
        let sols = brute_aligned_search(v, 1, 0.0, 1000);
        // ‖x‖²=2, unitarity constraint → should find several solutions
        assert!(!sols.is_empty());
        check_solutions(&sols, 1);
    }

    #[test]
    fn test_search_k2_with_alignment() {
        let v = [1.0f64, 0.0, 0.0, 0.0];
        let sols = brute_aligned_search(v, 2, 0.5, 100);
        check_solutions(&sols, 2);
        // With alignment filter, all solutions should have reasonable alignment
        let av = compute_align_vec(v);
        let thresh_sq = 4.0f64 * (1.0 - 0.25);
        for sol in &sols {
            let dot: f64 = sol.iter().zip(av.iter()).map(|(&x, &a)| x as f64 * a).sum();
            assert!(dot * dot >= thresh_sq - 1e-9, "alignment failed for {sol:?}");
        }
    }

    #[test]
    fn test_search_finds_identity() {
        // At k=0, v=[1,0,0,0], should find [±1,0,0,0,0,0,0,0].
        let v = [1.0f64, 0.0, 0.0, 0.0];
        let sols = brute_aligned_search(v, 0, 0.0, 100);
        let found = sols.iter().any(|s| *s == [1,0,0,0,0,0,0,0] || *s == [-1,0,0,0,0,0,0,0]);
        assert!(found, "Should find identity solution");
    }
}
#[cfg(test)]
mod mpfr_prefix_tests {
    use super::*;

    #[test]
    fn apply_u2t_dag_to_uv_mpfr_matches_f64() {
        let prec = 160;
        let v = normalize4([0.6, 0.1, 0.7, 0.35]).unwrap();
        let t = U2T::t();
        let prefixes = [t, t * t, t * t * t, t * t * t * t];
        for c in prefixes {
            let want = apply_u2t_dag_to_uv(&c, v);
            let vm: [MpFloat; 4] = std::array::from_fn(|i| MpFloat::with_val(prec, v[i]));
            let got = apply_u2t_dag_to_uv_mpfr(&c, &vm, prec);
            for i in 0..4 {
                assert!(
                    (got[i].to_f64() - want[i]).abs() < 1e-9,
                    "k={} entry {i}: mpfr {} vs f64 {}",
                    c.k,
                    got[i].to_f64(),
                    want[i]
                );
            }
        }
    }
}
