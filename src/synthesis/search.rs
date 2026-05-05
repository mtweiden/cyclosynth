//! Aligned lattice search for Clifford+T synthesis.
//!
//! Finds integer vectors x = (a1, b1, c1, d1, a2, b2, c2, d2) satisfying:
//!   - ‖x‖² = 2^k                                         (norm constraint)
//!   - b1(a1+c1) + d1(c1-a1) + b2(a2+c2) + d2(c2-a2) = 0 (unitarity)
//!   - (x · align_vec)² ≥ 2^k · (1 − ε²)                  (alignment)
//!
//! The alignment vector is align_vec = Σ_uv^T · v, where Σ_uv maps the
//! integer lattice coordinates to the uv parameterization
//! (Re(u1), Im(u1), Re(u2), Im(u2)) of the SU(2) matrix.

// Search functions thread many tuning knobs (norm bound, alignment vector,
// max solutions, budget caps, abort flag) through their signatures. The
// alternative would be a "search options" struct, which obscures the call
// site without simplifying the underlying interface.
#![allow(clippy::too_many_arguments)]

use num_complex::Complex64;
use std::f64::consts::FRAC_1_SQRT_2;
use crate::matrix::U2T;

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

// ─── Algebraic solvers ────────────────────────────────────────────────────────

/// Solve for (b1, d1) given all other coordinates and the constraints:
///   b1² + d1² = r
///   b1·A + d1·B = rhs,   where A = a1+c1, B = c1−a1.
///
/// Records valid solutions directly into `out`.
#[inline]
fn solve_b1d1(
    a1: i64, c1: i64,
    a2: i64, b2: i64, c2: i64, d2: i64,
    r: i64,
    rhs: i64,
    av: &[f64; 8],
    thresh_sq: f64,
    out: &mut Vec<[i64; 8]>,
    max_sol: usize,
) {
    let big_a = a1 + c1;
    let big_b = c1 - a1;

    if big_a == 0 && big_b == 0 {
        if rhs != 0 {
            return;
        }
        let max_b1 = integer_sqrt(r);
        for b1 in -max_b1..=max_b1 {
            let d1_sq = r - b1 * b1;
            if d1_sq < 0 {
                continue;
            }
            let d1_abs = integer_sqrt(d1_sq);
            if d1_abs * d1_abs != d1_sq {
                continue;
            }
            for &d1 in &[d1_abs, -d1_abs] {
                if d1 < 0 && d1_abs == 0 {
                    continue;
                }
                record_if_aligned(a1, b1, c1, d1, a2, b2, c2, d2, av, thresh_sq, out, max_sol);
                if out.len() >= max_sol {
                    return;
                }
            }
        }
        return;
    }

    if big_a == 0 {
        // d1 = rhs / B
        if rhs % big_b != 0 {
            return;
        }
        let d1 = rhs / big_b;
        let b1_sq = r - d1 * d1;
        if b1_sq < 0 {
            return;
        }
        let b1_abs = integer_sqrt(b1_sq);
        if b1_abs * b1_abs != b1_sq {
            return;
        }
        for &b1 in &[b1_abs, -b1_abs] {
            if b1 < 0 && b1_abs == 0 {
                continue;
            }
            record_if_aligned(a1, b1, c1, d1, a2, b2, c2, d2, av, thresh_sq, out, max_sol);
            if out.len() >= max_sol {
                return;
            }
        }
        return;
    }

    if big_b == 0 {
        // b1 = rhs / A
        if rhs % big_a != 0 {
            return;
        }
        let b1 = rhs / big_a;
        let d1_sq = r - b1 * b1;
        if d1_sq < 0 {
            return;
        }
        let d1_abs = integer_sqrt(d1_sq);
        if d1_abs * d1_abs != d1_sq {
            return;
        }
        for &d1 in &[d1_abs, -d1_abs] {
            if d1 < 0 && d1_abs == 0 {
                continue;
            }
            record_if_aligned(a1, b1, c1, d1, a2, b2, c2, d2, av, thresh_sq, out, max_sol);
            if out.len() >= max_sol {
                return;
            }
        }
        return;
    }

    // General case: quadratic discriminant method.
    // From b1*A + d1*B = rhs and b1² + d1² = r:
    //   (A²+B²)·d1² − 2·rhs·B·d1 + (rhs² − A²·r) = 0
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
        let numer_d1 = 2 * rhs * big_b + sign * sqrt_disc;
        if numer_d1 % denom != 0 {
            continue;
        }
        let d1 = numer_d1 / denom;
        let numer_b1 = rhs - d1 * big_b;
        if numer_b1 % big_a != 0 {
            continue;
        }
        let b1 = numer_b1 / big_a;
        if b1 * b1 + d1 * d1 != r {
            continue;
        }
        record_if_aligned(a1, b1, c1, d1, a2, b2, c2, d2, av, thresh_sq, out, max_sol);
        if out.len() >= max_sol {
            return;
        }
    }
}

/// Solve for (b2, d2) given all other coordinates and the constraints:
///   b2² + d2² = r
///   b2·A + d2·B = rhs,   where A = a2+c2, B = c2−a2.
#[inline]
fn solve_b2d2(
    a1: i64, b1: i64, c1: i64, d1: i64,
    a2: i64, c2: i64,
    r: i64,
    rhs: i64,
    av: &[f64; 8],
    thresh_sq: f64,
    out: &mut Vec<[i64; 8]>,
    max_sol: usize,
) {
    let big_a = a2 + c2;
    let big_b = c2 - a2;

    if big_a == 0 && big_b == 0 {
        if rhs != 0 {
            return;
        }
        let max_b2 = integer_sqrt(r);
        for b2 in -max_b2..=max_b2 {
            let d2_sq = r - b2 * b2;
            if d2_sq < 0 {
                continue;
            }
            let d2_abs = integer_sqrt(d2_sq);
            if d2_abs * d2_abs != d2_sq {
                continue;
            }
            for &d2 in &[d2_abs, -d2_abs] {
                if d2 < 0 && d2_abs == 0 {
                    continue;
                }
                record_if_aligned(a1, b1, c1, d1, a2, b2, c2, d2, av, thresh_sq, out, max_sol);
                if out.len() >= max_sol {
                    return;
                }
            }
        }
        return;
    }

    if big_a == 0 {
        if rhs % big_b != 0 {
            return;
        }
        let d2 = rhs / big_b;
        let b2_sq = r - d2 * d2;
        if b2_sq < 0 {
            return;
        }
        let b2_abs = integer_sqrt(b2_sq);
        if b2_abs * b2_abs != b2_sq {
            return;
        }
        for &b2 in &[b2_abs, -b2_abs] {
            if b2 < 0 && b2_abs == 0 {
                continue;
            }
            record_if_aligned(a1, b1, c1, d1, a2, b2, c2, d2, av, thresh_sq, out, max_sol);
            if out.len() >= max_sol {
                return;
            }
        }
        return;
    }

    if big_b == 0 {
        if rhs % big_a != 0 {
            return;
        }
        let b2 = rhs / big_a;
        let d2_sq = r - b2 * b2;
        if d2_sq < 0 {
            return;
        }
        let d2_abs = integer_sqrt(d2_sq);
        if d2_abs * d2_abs != d2_sq {
            return;
        }
        for &d2 in &[d2_abs, -d2_abs] {
            if d2 < 0 && d2_abs == 0 {
                continue;
            }
            record_if_aligned(a1, b1, c1, d1, a2, b2, c2, d2, av, thresh_sq, out, max_sol);
            if out.len() >= max_sol {
                return;
            }
        }
        return;
    }

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
        let numer_d2 = 2 * rhs * big_b + sign * sqrt_disc;
        if numer_d2 % denom != 0 {
            continue;
        }
        let d2 = numer_d2 / denom;
        let numer_b2 = rhs - d2 * big_b;
        if numer_b2 % big_a != 0 {
            continue;
        }
        let b2 = numer_b2 / big_a;
        if b2 * b2 + d2 * d2 != r {
            continue;
        }
        record_if_aligned(a1, b1, c1, d1, a2, b2, c2, d2, av, thresh_sq, out, max_sol);
        if out.len() >= max_sol {
            return;
        }
    }
}

// ─── Search engines ───────────────────────────────────────────────────────────

use rayon::prelude::*;

// ── fast_search (solves for b1, d1) ──────────────────────────────────────────

/// Inner body for one fixed (a1, c1) pair. Enumerates a2, c2, b2, d2 and calls
/// solve_b1d1 for the remaining pair. Returns all solutions found up to max_sol.
fn fast_search_inner(
    a1: i64,
    c1: i64,
    rem2: i64,
    pdot2: f64,
    av: &[f64; 8],
    thresh_sq: f64,
    av_sq_3: f64,
    av_sq_4: f64,
    av_sq_5: f64,
    av_sq_6: f64,
    do_prune: bool,
    thresh: f64,
    max_sol: usize,
) -> Vec<[i64; 8]> {
    let mut out: Vec<[i64; 8]> = Vec::new();

    let max_a2 = integer_sqrt(rem2);
    for a2 in -max_a2..=max_a2 {
        let rem3 = rem2 - a2 * a2;
        if rem3 < 0 { continue; }
        let pdot3 = pdot2 + a2 as f64 * av[4];
        if do_prune && pdot3.abs() + (rem3 as f64 * av_sq_3).sqrt() < thresh { continue; }

        let max_c2 = integer_sqrt(rem3);
        for c2 in -max_c2..=max_c2 {
            let rem4 = rem3 - c2 * c2;
            if rem4 < 0 { continue; }
            let pdot4 = pdot3 + c2 as f64 * av[6];
            if do_prune && pdot4.abs() + (rem4 as f64 * av_sq_4).sqrt() < thresh { continue; }

            let max_b2 = integer_sqrt(rem4);
            for b2 in -max_b2..=max_b2 {
                let rem5 = rem4 - b2 * b2;
                if rem5 < 0 { continue; }
                let pdot5 = pdot4 + b2 as f64 * av[5];
                if do_prune && pdot5.abs() + (rem5 as f64 * av_sq_5).sqrt() < thresh { continue; }

                let max_d2 = integer_sqrt(rem5);
                for d2 in -max_d2..=max_d2 {
                    let r = rem5 - d2 * d2;
                    if r < 0 { continue; }
                    let pdot6 = pdot5 + d2 as f64 * av[7];
                    if do_prune && pdot6.abs() + (r as f64 * av_sq_6).sqrt() < thresh { continue; }

                    let rhs = -(b2 * (a2 + c2) + d2 * (c2 - a2));
                    solve_b1d1(a1, c1, a2, b2, c2, d2, r, rhs, av, thresh_sq, &mut out, max_sol);
                    if out.len() >= max_sol { return out; }
                }
            }
        }
    }
    out
}

/// Full-sphere enumeration with Cauchy–Schwarz pruning, solving for (b1, d1).
///
/// Enumeration order: a1(0), c1(2), a2(4), c2(6), b2(5), d2(7) → solve b1(1), d1(3).
/// Parallelised over (a1, c1) pairs via rayon.
fn fast_search(
    target_norm: i64,
    av: &[f64; 8],
    thresh_sq: f64,
    max_sol: usize,
    out: &mut Vec<[i64; 8]>,
) {
    let do_prune = thresh_sq > 0.0;
    let thresh = thresh_sq.sqrt();

    let av_sq_all: f64 = av.iter().map(|x| x * x).sum();
    let av_sq_1 = av_sq_all - av[0] * av[0]; // after a1
    let av_sq_2 = av_sq_1  - av[2] * av[2]; // after c1
    let av_sq_3 = av_sq_2  - av[4] * av[4]; // after a2
    let av_sq_4 = av_sq_3  - av[6] * av[6]; // after c2
    let av_sq_5 = av_sq_4  - av[5] * av[5]; // after b2
    let av_sq_6 = av_sq_5  - av[7] * av[7]; // after d2 (remaining: b1, d1)

    let max_a1 = integer_sqrt(target_norm);

    let pairs: Vec<(i64, i64, i64, f64)> = (-max_a1..=max_a1)
        .flat_map(|a1| {
            let rem1 = target_norm - a1 * a1;
            if rem1 < 0 { return vec![]; }
            let pdot1 = a1 as f64 * av[0];
            if do_prune && pdot1.abs() + (rem1 as f64 * av_sq_1).sqrt() < thresh {
                return vec![];
            }
            let max_c1 = integer_sqrt(rem1);
            (-max_c1..=max_c1).filter_map(|c1| {
                let rem2 = rem1 - c1 * c1;
                if rem2 < 0 { return None; }
                let pdot2 = pdot1 + c1 as f64 * av[2];
                if do_prune && pdot2.abs() + (rem2 as f64 * av_sq_2).sqrt() < thresh {
                    return None;
                }
                Some((a1, c1, rem2, pdot2))
            }).collect::<Vec<_>>()
        })
        .collect();

    let batches: Vec<Vec<[i64; 8]>> = pairs
        .into_par_iter()
        .filter_map(|(a1, c1, rem2, pdot2)| {
            let local = fast_search_inner(
                a1, c1, rem2, pdot2, av, thresh_sq,
                av_sq_3, av_sq_4, av_sq_5, av_sq_6,
                do_prune, thresh, max_sol,
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

// ── fast_search_u1 (solves for b2, d2) ───────────────────────────────────────

/// Inner body for one fixed (a1, b1) pair. Enumerates c1, d1, a2, c2 and calls
/// solve_b2d2 for the remaining pair. Returns all solutions found up to max_sol.
fn fast_search_u1_inner(
    a1: i64,
    b1: i64,
    rem2: i64,
    pdot2: f64,
    av: &[f64; 8],
    thresh_sq: f64,
    av_sq_3: f64,
    av_sq_4: f64,
    av_sq_5: f64,
    av_sq_6: f64,
    do_prune: bool,
    thresh: f64,
    max_sol: usize,
) -> Vec<[i64; 8]> {
    let mut out: Vec<[i64; 8]> = Vec::new();

    let max_c1 = integer_sqrt(rem2);
    for c1 in -max_c1..=max_c1 {
        let rem3 = rem2 - c1 * c1;
        if rem3 < 0 { continue; }
        let pdot3 = pdot2 + c1 as f64 * av[2];
        if do_prune && pdot3.abs() + (rem3 as f64 * av_sq_3).sqrt() < thresh { continue; }

        let max_d1 = integer_sqrt(rem3);
        for d1 in -max_d1..=max_d1 {
            let rem4 = rem3 - d1 * d1;
            if rem4 < 0 { continue; }
            let pdot4 = pdot3 + d1 as f64 * av[3];
            if do_prune && pdot4.abs() + (rem4 as f64 * av_sq_4).sqrt() < thresh { continue; }

            let cross1 = b1 * (a1 + c1) + d1 * (c1 - a1);

            let max_a2 = integer_sqrt(rem4);
            for a2 in -max_a2..=max_a2 {
                let rem5 = rem4 - a2 * a2;
                if rem5 < 0 { continue; }
                let pdot5 = pdot4 + a2 as f64 * av[4];
                if do_prune && pdot5.abs() + (rem5 as f64 * av_sq_5).sqrt() < thresh { continue; }

                let max_c2 = integer_sqrt(rem5);
                for c2 in -max_c2..=max_c2 {
                    let r = rem5 - c2 * c2;
                    if r < 0 { continue; }
                    let pdot6 = pdot5 + c2 as f64 * av[6];
                    if do_prune && pdot6.abs() + (r as f64 * av_sq_6).sqrt() < thresh { continue; }

                    let rhs = -cross1;
                    solve_b2d2(a1, b1, c1, d1, a2, c2, r, rhs, av, thresh_sq, &mut out, max_sol);
                    if out.len() >= max_sol { return out; }
                }
            }
        }
    }
    out
}

/// Full-sphere enumeration with Cauchy–Schwarz pruning, solving for (b2, d2).
///
/// Preferred when alignment energy is concentrated in u1 (indices 0-3).
/// Enumeration order: a1(0), b1(1), c1(2), d1(3), a2(4), c2(6) → solve b2(5), d2(7).
/// Parallelised over (a1, b1) pairs via rayon.
fn fast_search_u1(
    target_norm: i64,
    av: &[f64; 8],
    thresh_sq: f64,
    max_sol: usize,
    out: &mut Vec<[i64; 8]>,
) {
    let do_prune = thresh_sq > 0.0;
    let thresh = thresh_sq.sqrt();

    let av_sq_all: f64 = av.iter().map(|x| x * x).sum();
    let av_sq_1 = av_sq_all - av[0] * av[0]; // after a1
    let av_sq_2 = av_sq_1  - av[1] * av[1]; // after b1
    let av_sq_3 = av_sq_2  - av[2] * av[2]; // after c1
    let av_sq_4 = av_sq_3  - av[3] * av[3]; // after d1
    let av_sq_5 = av_sq_4  - av[4] * av[4]; // after a2
    let av_sq_6 = av_sq_5  - av[6] * av[6]; // after c2 (remaining: b2, d2)

    let max_a1 = integer_sqrt(target_norm);

    let pairs: Vec<(i64, i64, i64, f64)> = (-max_a1..=max_a1)
        .flat_map(|a1| {
            let rem1 = target_norm - a1 * a1;
            if rem1 < 0 { return vec![]; }
            let pdot1 = a1 as f64 * av[0];
            if do_prune && pdot1.abs() + (rem1 as f64 * av_sq_1).sqrt() < thresh {
                return vec![];
            }
            let max_b1 = integer_sqrt(rem1);
            (-max_b1..=max_b1).filter_map(|b1| {
                let rem2 = rem1 - b1 * b1;
                if rem2 < 0 { return None; }
                let pdot2 = pdot1 + b1 as f64 * av[1];
                if do_prune && pdot2.abs() + (rem2 as f64 * av_sq_2).sqrt() < thresh {
                    return None;
                }
                Some((a1, b1, rem2, pdot2))
            }).collect::<Vec<_>>()
        })
        .collect();

    let batches: Vec<Vec<[i64; 8]>> = pairs
        .into_par_iter()
        .filter_map(|(a1, b1, rem2, pdot2)| {
            let local = fast_search_u1_inner(
                a1, b1, rem2, pdot2, av, thresh_sq,
                av_sq_3, av_sq_4, av_sq_5, av_sq_6,
                do_prune, thresh, max_sol,
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
pub fn aligned_search(
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
        fast_search_u1(target_norm, &av, thresh_sq, max_solutions, &mut out);
    } else {
        fast_search(target_norm, &av, thresh_sq, max_solutions, &mut out);
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
        let sols = aligned_search(v, 0, 0.0, 100);
        assert!(!sols.is_empty(), "Should find solutions at k=0");
        check_solutions(&sols, 0);
    }

    #[test]
    fn test_search_k1_no_alignment() {
        // With epsilon=0 (no alignment filter), enumerate all valid solutions at k=1.
        let v = [1.0f64, 0.0, 0.0, 0.0];
        let sols = aligned_search(v, 1, 0.0, 1000);
        // ‖x‖²=2, unitarity constraint → should find several solutions
        assert!(!sols.is_empty());
        check_solutions(&sols, 1);
    }

    #[test]
    fn test_search_k2_with_alignment() {
        let v = [1.0f64, 0.0, 0.0, 0.0];
        let sols = aligned_search(v, 2, 0.5, 100);
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
        let sols = aligned_search(v, 0, 0.0, 100);
        let found = sols.iter().any(|s| *s == [1,0,0,0,0,0,0,0] || *s == [-1,0,0,0,0,0,0,0]);
        assert!(found, "Should find identity solution");
    }
}