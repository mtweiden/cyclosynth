//! Clifford + R_z(π/6) synthesis over ℤ[ζ₁₂] = ℤ[i, √3].
//!
//! Finds a Clifford+R_z(π/6) circuit U such that d_diamond(U, V) < ε.
//!
//! # Ring and coordinates
//!
//! Every element of ℤ[ζ₁₂] is written as  u = a + b·i + c·√3 + d·i√3
//! with a,b,c,d ∈ ℤ.  Re(u) = a + c√3,  Im(u) = b + d√3.
//! The bullet automorphism is  u• = a - c√3 + (b - d√3)·i  (√3 ↦ −√3).
//!
//! # Integer lattice
//!
//! The unitary has SU(2)-like form  U = [[u, −t̄], [t, ū]] / √(3^k)
//! with u, t ∈ ℤ[ζ₁₂] and  |u|² + |t|² = 3^k.
//!
//! Eight-dimensional integer coordinate vector:
//!   x = (a₁,b₁,c₁,d₁, a₂,b₂,c₂,d₂)  where u has (a₁,b₁,c₁,d₁)
//!                                         and t has (a₂,b₂,c₂,d₂).
//!
//! # Quadratic form and constraints
//!
//! ‖Σx‖² = 2(a₁²+b₁²+a₂²+b₂²) + 6(c₁²+d₁²+c₂²+d₂²) = 2·3^k.
//!
//! Bilinear unitarity constraint: a₁c₁ + b₁d₁ + a₂c₂ + b₂d₂ = 0.
//!
//! Alignment vector y = (v_re, v_im, √3·v_re, √3·v_im, 0,0,0,0)
//! with threshold  (x·y)² ≥ 3^k·(1−ε²).
//!
//! # Σ matrix (8×8)
//!
//! Maps x → (Re u, Im u, Re u•, Im u•, Re t, Im t, Re t•, Im t•):
//!   Σ = block-diag(Σ_u, Σ_t),  Σ_u = [[1,0,√3,0],[0,1,0,√3],[1,0,−√3,0],[0,1,0,−√3]].
//!   ΣᵀΣ = diag(2,2,6,6, 2,2,6,6),   Σ⁻¹ = D⁻¹Σᵀ.

#![allow(clippy::too_many_arguments)]

use num_complex::Complex64;
use rayon::prelude::*;
use std::f64::consts::PI;
use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex};

use crate::synthesis::cliffords::CLIFFORD_TABLE_T;
use crate::synthesis::distance::{diamond_distance_float, Mat2};
use crate::synthesis::search::{apply_u2t_dag_to_uv, normalize4};

// ─── Constants ────────────────────────────────────────────────────────────────

const SQRT3: f64 = 1.7320508075688772935_f64;

// Rotation angle per R_z(π/6) gate in uv-space.
// R_z(π/6) has det = e^{iπ/6}, so √det = e^{iπ/12};
// the uv direction rotates by e^{iπ/12} when one gate is peeled off.
const RZ_ANGLE: f64 = PI / 12.0; // π/12

// ─── Σ matrix ─────────────────────────────────────────────────────────────────

/// 8×8 Σ matrix for ℤ[ζ₁₂]: maps integer coords x to Minkowski embedding.
///
/// Row order: (Re u, Im u, Re u•, Im u•, Re t, Im t, Re t•, Im t•).
/// Column order: (a₁,b₁,c₁,d₁, a₂,b₂,c₂,d₂).
pub fn sigma_matrix() -> [[f64; 8]; 8] {
    let s = SQRT3;
    [
        [1.0, 0.0,  s,  0.0, 0.0, 0.0, 0.0,  0.0],  // Re u
        [0.0, 1.0, 0.0,  s,  0.0, 0.0, 0.0,  0.0],  // Im u
        [1.0, 0.0, -s,  0.0, 0.0, 0.0, 0.0,  0.0],  // Re u•
        [0.0, 1.0, 0.0, -s,  0.0, 0.0, 0.0,  0.0],  // Im u•
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0,  s,   0.0],  // Re t
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,   s ],  // Im t
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -s,   0.0],  // Re t•
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,  -s ],  // Im t•
    ]
}

/// Apply Σ⁻¹ = D⁻¹ Σᵀ to an 8-vector.  D = diag(2,2,6,6,2,2,6,6).
pub fn sigma_inverse_apply(w: [f64; 8]) -> [f64; 8] {
    let sigma = sigma_matrix();
    // First compute Σᵀ w: entry j = Σ_i sigma[i][j] * w[i]
    let mut st_w = [0.0f64; 8];
    for j in 0..8 {
        for i in 0..8 {
            st_w[j] += sigma[i][j] * w[i];
        }
    }
    // Then multiply by D⁻¹ = diag(1/2,1/2,1/6,1/6,1/2,1/2,1/6,1/6)
    let d_inv = [0.5, 0.5, 1.0/6.0, 1.0/6.0, 0.5, 0.5, 1.0/6.0, 1.0/6.0];
    std::array::from_fn(|i| st_w[i] * d_inv[i])
}

// ─── 3^k helper ───────────────────────────────────────────────────────────────

/// 3^k as i64, panics for k > 40 (where 3^40 > i64::MAX).
#[inline]
fn pow3(k: u32) -> i64 {
    debug_assert!(k <= 40, "pow3: k={k} would overflow i64");
    let mut r: i64 = 1;
    for _ in 0..k { r *= 3; }
    r
}

/// Floor of √n for non-negative n (integer square root).
#[inline]
fn isqrt(n: i64) -> i64 {
    if n <= 0 { return 0; }
    let mut s = (n as f64).sqrt() as i64;
    while s > 0 && s * s > n { s -= 1; }
    while (s + 1) * (s + 1) <= n { s += 1; }
    s
}

// ─── Constraint checkers ──────────────────────────────────────────────────────

/// Check ‖Σx‖² = 2·3^k via the explicit diagonal quadratic form.
#[inline]
pub fn check_norm_eq(x: &[i64; 8], k: u32) -> bool {
    let [a1, b1, c1, d1, a2, b2, c2, d2] = *x;
    let n = 2*(a1*a1 + b1*b1 + a2*a2 + b2*b2)
          + 6*(c1*c1 + d1*d1 + c2*c2 + d2*d2);
    n == 2 * pow3(k)
}

/// Check bilinear unitarity constraint: a₁c₁ + b₁d₁ + a₂c₂ + b₂d₂ = 0.
#[inline]
pub fn check_bilinear(x: &[i64; 8]) -> bool {
    let [a1, b1, c1, d1, a2, b2, c2, d2] = *x;
    a1*c1 + b1*d1 + a2*c2 + b2*d2 == 0
}

/// Check alignment: (x·y)² ≥ 3^k·(1−ε²).
#[inline]
pub fn check_alignment(x: &[i64; 8], y: &[f64; 8], k: u32, eps_sq: f64) -> bool {
    let dot: f64 = x.iter().zip(y.iter()).map(|(&xi, &yi)| xi as f64 * yi).sum();
    let thresh = pow3(k) as f64 * (1.0 - eps_sq);
    dot * dot >= thresh
}

// ─── Alignment vector ─────────────────────────────────────────────────────────

/// Build the 8D alignment vector y from a target unit direction v = (v_re, v_im).
///
/// y = (v_re, v_im, √3·v_re, √3·v_im, 0, 0, 0, 0).
/// Satisfies  x·y = Re(u)·v_re + Im(u)·v_im  and  yᵀD⁻¹y = |v|² = 1.
pub fn compute_y(v_re: f64, v_im: f64) -> [f64; 8] {
    [v_re, v_im, SQRT3 * v_re, SQRT3 * v_im, 0.0, 0.0, 0.0, 0.0]
}

// ─── UV direction helpers ─────────────────────────────────────────────────────

/// Extract (u₁, u₂) direction from a 2×2 unitary: uv = (Re u₁, Im u₁, Re u₂, Im u₂)
/// normalized to unit length.  Divides first column by √det for SU(2) form.
pub fn unitary_to_uv_n6(v: &Mat2) -> [f64; 4] {
    let det = v[0][0] * v[1][1] - v[0][1] * v[1][0];
    let phase = det.sqrt();
    let (u1, u2) = if phase.norm() > 1e-12 {
        (v[0][0] / phase, v[1][0] / phase)
    } else {
        (v[0][0], v[1][0])
    };
    let arr = [u1.re, u1.im, u2.re, u2.im];
    normalize4(arr).unwrap_or([1.0, 0.0, 0.0, 0.0])
}

/// Rotate a uv direction by +θ (right-multiply by R_z(−θ) in gate sense).
/// Used to peel one R_z(π/6) off the right: new_v = v · e^{iπ/12}.
fn rotate_uv(v: [f64; 4], theta: f64) -> [f64; 4] {
    let (c, s) = (theta.cos(), theta.sin());
    [
        v[0]*c - v[1]*s,  v[0]*s + v[1]*c,
        v[2]*c - v[3]*s,  v[2]*s + v[3]*c,
    ]
}

/// uv direction after right-multiplying target by R_z(-π/6): search for U with U·R_z(π/6) ≈ V.
pub fn apply_rz_dag_to_uv(v: [f64; 4]) -> [f64; 4] { rotate_uv(v,  RZ_ANGLE) }

/// uv direction after right-multiplying target by R_z(+π/6): search for U with U·R_z(-π/6) ≈ V.
pub fn apply_rz_to_uv(v: [f64; 4])     -> [f64; 4] { rotate_uv(v, -RZ_ANGLE) }

// ─── Solution → float matrix ──────────────────────────────────────────────────

/// Build a float Mat2 from a lattice solution and lde k.
///
/// x = (a₁,b₁,c₁,d₁, a₂,b₂,c₂,d₂) in {1,i,√3,i√3} coords:
///   u = (a₁+c₁√3) + i(b₁+d₁√3),  t = (a₂+c₂√3) + i(b₂+d₂√3).
///   U = [[u, −t̄], [t, ū]] / √(3^k).
pub fn solution_to_mat2(x: &[i64; 8], k: u32) -> Mat2 {
    let [a1, b1, c1, d1, a2, b2, c2, d2] = x.map(|v| v as f64);
    let s = SQRT3;
    let u = Complex64::new(a1 + c1*s, b1 + d1*s);
    let t = Complex64::new(a2 + c2*s, b2 + d2*s);
    let scale = 1.0 / (pow3(k) as f64).sqrt();
    [
        [ u * scale, -t.conj() * scale ],
        [ t * scale,  u.conj() * scale ],
    ]
}

// ─── Weighted-norm + bilinear solve ──────────────────────────────────────────

/// Solve the 2×2 system: 2·b₁² + 6·d₁² = rem  AND  b₁·d₁ = K.
///
/// Returns up to 4 (b₁, d₁) pairs. Uses the substitution b₁ = K/d₁ to
/// get a quartic → quadratic in d₁²:
///   6·d₁⁴ − rem·d₁² + 2K² = 0.
fn solve_bd(rem: i64, k_val: i64) -> Vec<(i64, i64)> {
    let mut out = Vec::with_capacity(4);
    if rem < 0 { return out; }

    if k_val == 0 {
        // d₁ = 0: 2·b₁² = rem
        if rem % 2 == 0 {
            let bsq = rem / 2;
            let b = isqrt(bsq);
            if b * b == bsq {
                out.push((b, 0));
                if b != 0 { out.push((-b, 0)); }
            }
        }
        // b₁ = 0: 6·d₁² = rem
        if rem % 6 == 0 {
            let dsq = rem / 6;
            let d = isqrt(dsq);
            if d * d == dsq && d != 0 {
                out.push((0, d));
                out.push((0, -d));
            }
        }
        return out;
    }

    // k_val ≠ 0 → d₁ ≠ 0 and b₁ ≠ 0.
    // disc of quadratic in d₁²: rem² - 48·K²
    let disc = rem.checked_mul(rem).and_then(|r2| {
        k_val.checked_mul(k_val).and_then(|k2| r2.checked_sub(48 * k2))
    });
    let disc = match disc {
        Some(d) if d >= 0 => d,
        _ => return out,
    };
    let sq = isqrt(disc);
    if sq * sq != disc { return out; }

    for sign in [1i64, -1] {
        if sign == -1 && sq == 0 { break; }
        let numer = rem + sign * sq;
        if numer < 0 || numer % 12 != 0 { continue; }
        let d1sq = numer / 12;
        let d1abs = isqrt(d1sq);
        if d1abs * d1abs != d1sq || d1abs == 0 { continue; }
        for &d1 in &[d1abs, -d1abs] {
            if k_val % d1 != 0 { continue; }
            let b1 = k_val / d1;
            if 2*b1*b1 + 6*d1*d1 != rem { continue; }
            out.push((b1, d1));
        }
    }
    out
}

// ─── Phase 1 direct search ────────────────────────────────────────────────────

/// Record a candidate if it passes the alignment threshold.
#[inline]
fn record_if_aligned(
    x: [i64; 8],
    y: &[f64; 8],
    thresh_sq: f64,
    out: &mut Vec<[i64; 8]>,
    max_sol: usize,
) {
    if thresh_sq > 0.0 {
        let dot: f64 = x.iter().zip(y.iter()).map(|(&xi, &yi)| xi as f64 * yi).sum();
        if dot * dot < thresh_sq { return; }
    }
    if out.len() < max_sol { out.push(x); }
}

/// Inner enumeration for one fixed (a₂,b₂,c₂,d₂) tuple.
///
/// Enumerates (a₁,c₁) and solves (b₁,d₁) from:
///   2b₁²+6d₁² = rem_u − 2a₁² − 6c₁²
///   b₁d₁ = −(a₁c₁ + K₂)  where K₂ = a₂c₂+b₂d₂.
fn search_inner(
    a2: i64, b2: i64, c2: i64, d2: i64,
    rem_u: i64,
    y: &[f64; 8],
    thresh_sq: f64,
    thresh: f64,
    do_prune: bool,
    max_sol: usize,
) -> Vec<[i64; 8]> {
    let mut out = Vec::new();
    let k2 = a2*c2 + b2*d2;

    let max_a1 = isqrt(rem_u / 2);
    for a1 in -max_a1..=max_a1 {
        let rem1 = rem_u - 2*a1*a1;
        if rem1 < 0 { continue; }

        // Partial alignment: contribution from a₁ and c₁ only (y[1,3] are b₁,d₁ parts)
        // Prune based on (a₁+c₁·√3)·v_re:
        // After fixing a₁ alone: maximum |extra| from c₁ part ≤ |y[2]|·√(rem1/6)
        let pdot_a1 = a1 as f64 * y[0]; // a₁·v_re
        let max_c1 = isqrt(rem1 / 6);
        // Prune: even if c₁ and (b₁,d₁) align perfectly, total ≤ |pdot_a1| + bound
        if do_prune {
            let remaining_max = (rem1 as f64 / 6.0).sqrt() * y[2].abs()
                + (rem1 as f64 / 2.0).sqrt() * (y[1]*y[1] + y[3]*y[3]).sqrt();
            if pdot_a1.abs() + remaining_max < thresh { continue; }
        }

        for c1 in -max_c1..=max_c1 {
            let rem_bd = rem1 - 6*c1*c1;
            if rem_bd < 0 { continue; }

            let pdot_ac = pdot_a1 + c1 as f64 * y[2]; // (a₁+c₁√3)·v_re
            if do_prune {
                // max remaining from b₁,d₁: |(b₁+d₁√3)·v_im| ≤ √(rem_bd)·|v_im|
                // Using: (b₁+d₁√3)² ≤ (1+3)(b₁²+d₁²)·... crude but fast bound:
                // |b₁·y[1]+d₁·y[3]| ≤ √(b₁²+d₁²)·√(y[1]²+y[3]²) ≤ √(rem_bd/2)·‖y[1,3]‖
                let y_im_sq = y[1]*y[1] + y[3]*y[3];
                let bd_max = (rem_bd as f64 / 2.0).sqrt() * y_im_sq.sqrt();
                if pdot_ac.abs() + bd_max < thresh { continue; }
            }

            let k_bd = -(a1*c1 + k2);  // target for b₁d₁
            for (b1, d1) in solve_bd(rem_bd, k_bd) {
                let x = [a1, b1, c1, d1, a2, b2, c2, d2];
                record_if_aligned(x, y, thresh_sq, &mut out, max_sol);
                if out.len() >= max_sol { return out; }
            }
        }
    }
    out
}

/// Full Phase-1 search: enumerate all x with ‖Σx‖² = 2·3^k, bilinear=0, alignment.
///
/// Parallelised over (a₂,b₂) pairs via rayon. Early-exit at max_sol.
pub fn direct_search_n6(
    target_k: i64,     // 3^k (not 2·3^k)
    y: &[f64; 8],
    eps: f64,
    max_sol: usize,
) -> Vec<[i64; 8]> {
    if max_sol == 0 { return Vec::new(); }
    let norm2 = 2 * target_k;          // 2·3^k
    let thresh_sq = if eps > 0.0 {
        target_k as f64 * (1.0 - eps * eps)
    } else {
        0.0  // no alignment filter — avoids f64 rounding at the exact threshold
    };
    let do_prune = eps > 0.0;
    let thresh = thresh_sq.max(0.0).sqrt();

    let max_a2 = isqrt(norm2 / 2);
    // Build (a₂, b₂) pairs for parallel outer loop
    let pairs: Vec<(i64, i64, i64, f64)> = (-max_a2..=max_a2).flat_map(|a2| {
        let rem_a2 = norm2 - 2*a2*a2;
        if rem_a2 < 0 { return vec![]; }
        let max_b2 = isqrt(rem_a2 / 2);
        (-max_b2..=max_b2).filter_map(move |b2| {
            let rem_ab = rem_a2 - 2*b2*b2;
            if rem_ab < 0 { None } else { Some((a2, b2, rem_ab, 0.0_f64)) }
        }).collect::<Vec<_>>()
    }).collect();

    let batches: Vec<Vec<[i64; 8]>> = pairs
        .into_par_iter()
        .filter_map(|(a2, b2, rem_ab, _)| {
            let mut local: Vec<[i64; 8]> = Vec::new();
            let max_c2 = isqrt(rem_ab / 6);
            for c2 in -max_c2..=max_c2 {
                let rem_abc = rem_ab - 6*c2*c2;
                if rem_abc < 0 { continue; }
                let max_d2 = isqrt(rem_abc / 6);
                for d2 in -max_d2..=max_d2 {
                    let rem_u = rem_abc - 6*d2*d2;
                    if rem_u < 0 { continue; }
                    let batch = search_inner(
                        a2, b2, c2, d2, rem_u, y, thresh_sq, thresh, do_prune, max_sol,
                    );
                    local.extend_from_slice(&batch);
                    if local.len() >= max_sol { return Some(local); }
                }
            }
            if local.is_empty() { None } else { Some(local) }
        })
        .collect();

    let mut out = Vec::new();
    for batch in batches {
        for sol in batch {
            if out.len() >= max_sol { return out; }
            out.push(sol);
        }
    }
    out
}

// ─── SO3 float utilities ──────────────────────────────────────────────────────

type SO3f = [[f64; 3]; 3];

/// Compute the adjoint SO(3) representation of a 2×2 unitary (float).
///
/// M_{ij} = (1/2) Re(Tr(σ_i · U · σ_j · U†)),  σ_i ∈ {σ_x, σ_y, σ_z}.
fn mat_to_so3(u: &Mat2) -> SO3f {
    let zero = Complex64::new(0.0, 0.0);
    let paulis: [Mat2; 3] = [
        [[zero, Complex64::new(1.,0.)], [Complex64::new(1.,0.), zero]],
        [[zero, Complex64::new(0.,-1.)], [Complex64::new(0.,1.), zero]],
        [[Complex64::new(1.,0.), zero], [zero, Complex64::new(-1.,0.)]],
    ];
    let ud = [[u[0][0].conj(), u[1][0].conj()], [u[0][1].conj(), u[1][1].conj()]];
    let mut m = [[0.0f64; 3]; 3];
    for j in 0..3 {
        let usj = mat_mul(*u, paulis[j]);
        let usjud = mat_mul(usj, ud);
        for i in 0..3 {
            let prod = mat_mul(paulis[i], usjud);
            m[i][j] = (prod[0][0] + prod[1][1]).re / 2.0;
        }
    }
    m
}

#[inline]
fn so3_mul(a: SO3f, b: SO3f) -> SO3f {
    let mut out = [[0.0f64; 3]; 3];
    for i in 0..3 { for j in 0..3 { for k in 0..3 { out[i][j] += a[i][k] * b[k][j]; } } }
    out
}

#[inline] fn so3_rz(t: f64) -> SO3f { let (c,s)=(t.cos(),t.sin()); [[c,-s,0.],[s,c,0.],[0.,0.,1.]] }
#[inline] fn so3_rx(t: f64) -> SO3f { let (c,s)=(t.cos(),t.sin()); [[1.,0.,0.],[0.,c,-s],[0.,s,c]] }
#[inline] fn so3_ry(t: f64) -> SO3f { let (c,s)=(t.cos(),t.sin()); [[c,0.,s],[0.,1.,0.],[-s,0.,c]] }

/// How far an SO3 matrix is from being a Clifford (entries in {−1,0,1}).
#[inline]
fn clifford_dist(m: &SO3f) -> f64 {
    m.iter().flat_map(|r| r.iter())
        .map(|&v| { let n = v.round().clamp(-1., 1.); (v - n).abs() })
        .sum()
}

/// Precomputed SO3 representations of the 24 Cliffords.
static CLIFFORD_SO3: LazyLock<Vec<(SO3f, &'static str)>> =
    LazyLock::new(|| CLIFFORD_TABLE_T.iter()
        .map(|(name, u2t)| (mat_to_so3(&u2t.to_float()), *name))
        .collect());

/// Identify the Clifford gate nearest to the given SO3 matrix.
fn identify_clifford_so3(m: &SO3f) -> &'static str {
    CLIFFORD_SO3.iter()
        .map(|(cs, name)| {
            let d: f64 = (0..3).flat_map(|i| (0..3).map(move |j| (m[i][j]-cs[i][j]).powi(2))).sum();
            (d, *name)
        })
        .min_by(|(a,_),(b,_)| a.partial_cmp(b).unwrap())
        .map(|(_, n)| n).unwrap_or("I")
}

// ─── n=6 gate string decomposer ───────────────────────────────────────────────

/// Simplify a Clifford+R gate string (R = Rz(π/6)).
///
/// Identities (up to global phase):
///   RRR = S,  SS = Z,  ZZ = "",  HH = "",  XX = "",  YY = "",
///   RRRRRR = Z,  ZR = RZ, ...
pub fn simplify_n6(input: &str) -> String {
    let mut s = input.to_string();
    let mut prev = String::new();
    while s != prev {
        prev = s.clone();
        // Combinations using RRR=S, SS=Z
        s = s.replace("RRRRRR", "Z");
        s = s.replace("RRR", "S");
        s = s.replace("SS", "Z");
        s = s.replace("ZZ", "");
        // Cancellations
        s = s.replace("HH", "");
        s = s.replace("XX", "");
        s = s.replace("YY", "");
        // Commutations
        s = s.replace("SZ", "ZS");
        s = s.replace("RZ", "ZR");
    }
    s
}

/// Candidate generators used in the greedy SO3 peeling: (neg_so3, gate_removed).
///
/// Left-multiply SO3 by neg_so3; the gate being removed from the front is gate_removed.
/// Tries three axes × two step sizes (π/6 and 2π/6 = π/3), always NEGATIVE generators
/// (removing positive gates from the left of the sequence).
fn peel_candidates() -> [(SO3f, &'static str); 6] {
    let s = PI / 6.0;
    [
        (so3_rz(-s),       "R"),          // peel Rz(+π/6) → gate "R"
        (so3_rz(-2.*s),    "RR"),         // peel Rz(+π/3)
        (so3_rx(-s),       "HRH"),        // peel Rx(+π/6) = H·R·H
        (so3_rx(-2.*s),    "HRRH"),       // peel Rx(+π/3)
        (so3_ry(-s),       "SHRHSSS"),    // peel Ry(+π/6) = S·H·R·H·S†
        (so3_ry(-2.*s),    "SHRRHSSS"),   // peel Ry(+π/3) = S·H·RR·H·S†
    ]
}

/// Decompose a float matrix into a Clifford+R gate string via greedy SO3 peeling.
///
/// Iteratively left-peels one of 6 generators (Rz/Rx/Ry at ±π/6, ±π/3) that
/// most reduces the Clifford-distance of the residual. Terminates when the
/// residual is a Clifford (all entries ≈ {−1, 0, 1}).
///
/// Returns a gate string in {H, S, R, X, Y, Z}  (R = Rz(π/6)).
pub fn decompose_pi6(mat: &Mat2) -> String {
    let candidates = peel_candidates();
    let mut so3 = mat_to_so3(mat);
    let max_steps = 200;
    let mut gate_parts: Vec<&'static str> = Vec::new();

    for _ in 0..max_steps {
        if clifford_dist(&so3) < 1e-7 { break; }
        let (best_so3, best_gate) = candidates.iter()
            .map(|(neg, gs)| { let m = so3_mul(*neg, so3); (m, *gs, clifford_dist(&m)) })
            .min_by(|a, b| a.2.partial_cmp(&b.2).unwrap())
            .map(|(m, gs, _)| (m, gs))
            .unwrap();
        so3 = best_so3;
        gate_parts.push(best_gate);
    }

    let cliff = identify_clifford_so3(&so3);
    let combined: String = gate_parts.into_iter().chain(
        if cliff == "I" { None } else { Some(cliff) }
    ).collect::<Vec<_>>().join("");
    simplify_n6(&combined)
}

// ─── MA prefix set for DC search ──────────────────────────────────────────────

/// Float canonical key for deduplication of prefix matrices, phase-invariant.
fn canonical_key_f64(m: &Mat2) -> [i64; 8] {
    let flat = [m[0][0], m[0][1], m[1][0], m[1][1]];
    let (idx, _) = flat.iter().enumerate()
        .max_by(|(_, a), (_, b)| a.norm_sqr().partial_cmp(&b.norm_sqr()).unwrap())
        .unwrap();
    let piv = flat[idx];
    let rot: Vec<f64> = if piv.norm() < 1e-12 {
        flat.iter().flat_map(|c| [c.re, c.im]).collect()
    } else {
        let phase = piv / piv.norm();
        flat.iter().flat_map(|c| { let r = c / phase; [r.re, r.im] }).collect()
    };
    rot.iter().map(|x| (x * 1_000_000.0).round() as i64)
        .collect::<Vec<_>>().try_into().unwrap()
}

fn eye_mat() -> Mat2 {
    let one = Complex64::new(1.0, 0.0);
    let z   = Complex64::new(0.0, 0.0);
    [[one, z], [z, one]]
}

fn mat_dag(m: &Mat2) -> Mat2 {
    [[m[0][0].conj(), m[1][0].conj()], [m[0][1].conj(), m[1][1].conj()]]
}

/// Build the n=6 MA-like prefix set L_{k'} as (gate_string, float_Mat2) pairs.
///
/// Syllables: H·R (b=0) and H·S·R (b=1), analogous to n=4's H·T and H·S·T.
/// L_0 = {(I, "")}
/// L_{k'} (even): (HS^b R)^{k'} · Clifford for all bit patterns b.
/// L_{k'} (odd branch): R · (HS^b R)^{k'−1} · Clifford.
/// Deduplicated by canonical_key_f64.
fn build_l_pi6(k_prime: u32) -> Arc<Vec<(String, Mat2)>> {
    static CACHE: LazyLock<Mutex<HashMap<u32, Arc<Vec<(String, Mat2)>>>>> =
        LazyLock::new(|| Mutex::new(HashMap::new()));

    {
        let cache = CACHE.lock().unwrap();
        if let Some(v) = cache.get(&k_prime) { return Arc::clone(v); }
    }

    let result = Arc::new(build_l_pi6_inner(k_prime));
    CACHE.lock().unwrap().insert(k_prime, Arc::clone(&result));
    result
}

fn build_l_pi6_inner(k_prime: u32) -> Vec<(String, Mat2)> {
    if k_prime == 0 {
        return vec![("".to_string(), eye_mat())];
    }
    let h_mat   = CLIFFORD_TABLE_T.iter().find(|(n,_)| *n == "H").unwrap().1.to_float();
    let s_mat   = CLIFFORD_TABLE_T.iter().find(|(n,_)| *n == "S").unwrap().1.to_float();
    let rz_mat  = rz_pi6_mat();

    let hs0r = mat_mul(h_mat, rz_mat);              // H·R
    let hs1r = mat_mul(mat_mul(h_mat, s_mat), rz_mat); // H·S·R

    let mut candidates: Vec<(String, Mat2)> = Vec::new();
    let n_even = 1u32 << k_prime;
    for bits in 0..n_even {
        let mut m = eye_mat();
        let mut g = String::new();
        for i in 0..k_prime {
            if (bits >> i) & 1 == 1 { m = mat_mul(m, hs1r); g.push_str("HSR"); }
            else                    { m = mat_mul(m, hs0r); g.push_str("HR"); }
        }
        for (c_str, c_u2t) in CLIFFORD_TABLE_T {
            candidates.push((format!("{g}{c_str}"), mat_mul(m, c_u2t.to_float())));
        }
    }
    if k_prime >= 1 {
        let n_odd = 1u32 << (k_prime - 1);
        for bits in 0..n_odd {
            let mut m = rz_mat;
            let mut g = "R".to_string();
            for i in 0..(k_prime - 1) {
                if (bits >> i) & 1 == 1 { m = mat_mul(m, hs1r); g.push_str("HSR"); }
                else                    { m = mat_mul(m, hs0r); g.push_str("HR"); }
            }
            for (c_str, c_u2t) in CLIFFORD_TABLE_T {
                candidates.push((format!("{g}{c_str}"), mat_mul(m, c_u2t.to_float())));
            }
        }
    }
    let mut seen: std::collections::HashSet<[i64; 8]> = std::collections::HashSet::new();
    candidates.into_iter().filter(|(_, m)| seen.insert(canonical_key_f64(m))).collect()
}

// ─── Result type ─────────────────────────────────────────────────────────────

/// Result of a successful n=6 synthesis.
pub struct SynthResultPi6 {
    /// Gate string, or `None` if exact synthesis is not yet implemented.
    pub gates: Option<String>,
    /// R_z(π/6)-count (lde).
    pub lde: u32,
    /// Diamond distance to target.
    pub distance: f64,
}

// ─── Direct search branch tags ────────────────────────────────────────────────

enum Branch {
    Even,
    Rz,      // right-multiply target by R_z(-π/6) to search for U with U·R_z(π/6)≈V
    RzDag,   // right-multiply target by R_z(+π/6) to search for U with U·R_z(-π/6)≈V
    Clif(usize),
    ClifRz(usize),
    ClifRzDag(usize),
}

// ─── Synthesizer ─────────────────────────────────────────────────────────────

/// Clifford + R_z(π/6) synthesis backend over ℤ[ζ₁₂].
pub struct SynthesizerPi6 {
    /// Approximation tolerance (diamond distance).
    pub epsilon: f64,
    /// Maximum R_z(π/6)-count to search before giving up.
    pub max_lde: u32,
    /// Minimum R_z(π/6)-count to start from.
    pub min_lde: u32,
    /// Maximum lde for direct brute-force search; beyond this the DC path is used.
    pub direct_limit: u32,
}

impl SynthesizerPi6 {
    /// Create a synthesizer with sensible defaults for the given precision.
    pub fn new(epsilon: f64) -> Self {
        let (min_lde, max_lde) = if epsilon > 0.0 && epsilon < 1.0 {
            // Analogous to n=4: R_z(π/6)-count scales as ~log₃(1/ε²) ≈ 2·log₃(1/ε).
            // coefficient ≈ 2·log₂(1/ε)/log₂(3) ≈ 1.26·log₂(1/ε).
            let log2_recip = (1.0 / epsilon).log2();
            let min_lde = (1.3 * log2_recip).floor() as u32;
            let max_lde = ((2.2 * log2_recip).ceil() as u32 + 4).max(30);
            (min_lde, max_lde)
        } else {
            (0, 30)
        };
        Self { epsilon, max_lde, min_lde, direct_limit: 6 }
    }

    pub fn with_max_lde(mut self, v: u32) -> Self { self.max_lde = v; self }
    pub fn with_min_lde(mut self, v: u32) -> Self { self.min_lde = v; self }
    pub fn with_direct_limit(mut self, v: u32) -> Self { self.direct_limit = v; self }

    /// Synthesize a Clifford+R_z(π/6) circuit approximating `target`.
    pub fn synthesize(&self, target: Mat2) -> Option<SynthResultPi6> {
        let raw = unitary_to_uv_n6(&target);
        let v = normalize4(raw).unwrap_or([1.0, 0.0, 0.0, 0.0]);

        for k in self.min_lde..=self.max_lde {
            let result = if k <= self.direct_limit {
                self.direct_search(&target, v, k)
            } else {
                let k_prefix = k - self.direct_limit;
                self.dc_search(&target, v, k, k_prefix)
            };
            if result.is_some() { return result; }
        }
        None
    }

    /// Divide-and-conquer search using MA-like prefix set L_{k_prefix}.
    ///
    /// For each prefix U_L ∈ L_{k_prefix}:
    ///   1. Compute inner target V_inner = U_L† · target.
    ///   2. Extract uv direction from V_inner.
    ///   3. Run direct_search_n6 at k_inner = k − k_prefix (even branch).
    ///   4. Also try R_z(π/6) on the right (odd branch).
    fn dc_search(
        &self,
        target: &Mat2,
        _v: [f64; 4],
        k: u32,
        k_prefix: u32,
    ) -> Option<SynthResultPi6> {
        let k_inner = k - k_prefix;
        let eps = self.epsilon;
        let target_k_inner = pow3(k_inner);

        let prefixes = build_l_pi6(k_prefix);

        // Parallel search over all prefixes.
        prefixes.par_iter().find_map_any(|(prefix_gates, u_l)| {
            let u_l_dag = mat_dag(u_l);
            let m_inner = mat_mul(u_l_dag, *target);

            // Extract uv direction from float inner target.
            let v_inner = unitary_to_uv_n6(&m_inner);

            // Even branch: U_L · U_R ≈ target
            let y = compute_y(v_inner[0], v_inner[1]);
            for sol in direct_search_n6(target_k_inner, &y, eps, 1) {
                let u_r = solution_to_mat2(&sol, k_inner);
                let full = mat_mul(*u_l, u_r);
                let dist = diamond_distance_float(&full, target);
                if dist < eps {
                    let gates = simplify_n6(
                        &format!("{}{}", prefix_gates, decompose_pi6(&u_r))
                    );
                    return Some(SynthResultPi6 { gates: Some(gates), lde: k, distance: dist });
                }
            }

            // Odd branch: U_L · U_R · R ≈ target  →  search at uv(V_inner · R†)
            if k_inner > 0 {
                let v_inner_r = apply_rz_dag_to_uv(v_inner);
                let y_r = compute_y(v_inner_r[0], v_inner_r[1]);
                for sol in direct_search_n6(target_k_inner, &y_r, eps, 1) {
                    let u_r = solution_to_mat2(&sol, k_inner);
                    let full = mat_mul(*u_l, mat_mul(u_r, rz_pi6_mat()));
                    let dist = diamond_distance_float(&full, target);
                    if dist < eps {
                        let inner_str = format!("{}R", decompose_pi6(&u_r));
                        let gates = simplify_n6(&format!("{}{}", prefix_gates, inner_str));
                        return Some(SynthResultPi6 { gates: Some(gates), lde: k, distance: dist });
                    }
                }
            }
            None
        })
    }

    /// Brute-force direct search at lde `k`.
    ///
    /// Tries 3 top-level branches (even, +Rz, -Rz) and for each of the
    /// 24 Cliffords another 3 branches = 75 total. Parallelised via rayon.
    fn direct_search(&self, target: &Mat2, v: [f64; 4], k: u32) -> Option<SynthResultPi6> {
        let eps = self.epsilon;
        let target_k = pow3(k);

        // Precompute Clifford-conjugated directions.
        let clif_vs: Vec<[f64; 4]> = CLIFFORD_TABLE_T.iter()
            .map(|(_, c_u2t)| apply_u2t_dag_to_uv(c_u2t, v))
            .collect();

        // Build all (search_direction, branch_tag) pairs.
        let mut branches: Vec<([f64; 4], Branch)> = Vec::with_capacity(75);
        branches.push((v, Branch::Even));
        branches.push((apply_rz_dag_to_uv(v), Branch::Rz));
        branches.push((apply_rz_to_uv(v), Branch::RzDag));
        for i in 1..CLIFFORD_TABLE_T.len() {
            let vi = clif_vs[i];
            branches.push((vi, Branch::Clif(i)));
            branches.push((apply_rz_dag_to_uv(vi), Branch::ClifRz(i)));
            branches.push((apply_rz_to_uv(vi), Branch::ClifRzDag(i)));
        }

        branches.par_iter().find_map_any(|(v_s, tag)| {
            let y = compute_y(v_s[0], v_s[1]);
            for sol in direct_search_n6(target_k, &y, eps, 1) {
                // Build the float matrix for this inner solution.
                let u_mat = solution_to_mat2(&sol, k);

                // Compose with left Clifford and/or right R_z to reconstruct U.
                let full_mat = match tag {
                    Branch::Even => u_mat,
                    Branch::Rz   => mat_mul(u_mat, rz_pi6_mat()),
                    Branch::RzDag => mat_mul(u_mat, rz_neg_pi6_mat()),
                    Branch::Clif(i) => {
                        let c_f = CLIFFORD_TABLE_T[*i].1.to_float();
                        mat_mul(c_f, u_mat)
                    }
                    Branch::ClifRz(i) => {
                        let c_f = CLIFFORD_TABLE_T[*i].1.to_float();
                        mat_mul(c_f, mat_mul(u_mat, rz_pi6_mat()))
                    }
                    Branch::ClifRzDag(i) => {
                        let c_f = CLIFFORD_TABLE_T[*i].1.to_float();
                        mat_mul(c_f, mat_mul(u_mat, rz_neg_pi6_mat()))
                    }
                };

                let dist = diamond_distance_float(&full_mat, target);
                if dist < eps {
                    // Decompose the inner u_mat, then apply branch affixes.
                    let inner_gates = decompose_pi6(&u_mat);
                    let gates = simplify_n6(&match tag {
                        Branch::Even    => inner_gates,
                        Branch::Rz      => format!("{inner_gates}R"),
                        Branch::RzDag   => format!("{inner_gates}RRRRR"),
                        Branch::Clif(i) => {
                            format!("{}{inner_gates}", CLIFFORD_TABLE_T[*i].0)
                        }
                        Branch::ClifRz(i) => {
                            format!("{}{inner_gates}R", CLIFFORD_TABLE_T[*i].0)
                        }
                        Branch::ClifRzDag(i) => {
                            format!("{}{inner_gates}RRRRR", CLIFFORD_TABLE_T[*i].0)
                        }
                    });
                    return Some(SynthResultPi6 { gates: Some(gates), lde: k, distance: dist });
                }
            }
            None
        })
    }
}

// ─── Float gate matrices ─────────────────────────────────────────────────────

/// R_z(π/6) as a float Mat2 (up to global phase: diag(e^{-iπ/12}, e^{iπ/12})).
fn rz_pi6_mat() -> Mat2 {
    let ph = Complex64::from_polar(1.0, PI / 12.0);
    [
        [ph.conj(), Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), ph],
    ]
}

/// R_z(-π/6) as a float Mat2.
fn rz_neg_pi6_mat() -> Mat2 {
    let ph = Complex64::from_polar(1.0, PI / 12.0);
    [
        [ph, Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), ph.conj()],
    ]
}

/// Float 2×2 matrix multiplication.
fn mat_mul(a: Mat2, b: Mat2) -> Mat2 {
    [
        [
            a[0][0]*b[0][0] + a[0][1]*b[1][0],
            a[0][0]*b[0][1] + a[0][1]*b[1][1],
        ],
        [
            a[1][0]*b[0][0] + a[1][1]*b[1][0],
            a[1][0]*b[0][1] + a[1][1]*b[1][1],
        ],
    ]
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn near(a: f64, b: f64, tol: f64) -> bool { (a - b).abs() < tol }

    fn rz(theta: f64) -> Mat2 {
        [
            [Complex64::from_polar(1.0, -theta/2.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::from_polar(1.0, theta/2.0)],
        ]
    }

    // ── Σ round-trip ──────────────────────────────────────────────────────────

    #[test]
    fn sigma_inverse_roundtrip() {
        let sigma = sigma_matrix();
        // Σ · (Σ⁻¹ · w) = w for standard basis vectors.
        for col in 0..8_usize {
            let mut w = [0.0f64; 8];
            w[col] = 1.0;
            let sinv_w = sigma_inverse_apply(w);
            // Σ · sinv_w
            let mut result = [0.0f64; 8];
            for i in 0..8 {
                for j in 0..8 {
                    result[i] += sigma[i][j] * sinv_w[j];
                }
            }
            for i in 0..8 {
                let expected = if i == col { 1.0 } else { 0.0 };
                assert!(
                    near(result[i], expected, 1e-12),
                    "Σ·Σ⁻¹·e_{col} mismatch at row {i}: got {}, expected {}",
                    result[i], expected
                );
            }
        }
    }

    #[test]
    fn sigma_gram_is_diagonal() {
        let sigma = sigma_matrix();
        let expected_diag = [2.0, 2.0, 6.0, 6.0, 2.0, 2.0, 6.0, 6.0];
        for j in 0..8 {
            for k in 0..8 {
                let dot: f64 = (0..8).map(|i| sigma[i][j] * sigma[i][k]).sum();
                let expected = if j == k { expected_diag[j] } else { 0.0 };
                assert!(near(dot, expected, 1e-12),
                    "ΣᵀΣ[{j}][{k}] = {dot}, expected {expected}");
            }
        }
    }

    // ── Constraint checkers ───────────────────────────────────────────────────

    #[test]
    fn bullet_map_sanity() {
        // bullet: √3 ↦ −√3, i fixed.
        // For u = 1 + √3 = (1,0,1,0): Re(u) = 1+√3, Re(u•) = 1-√3.
        let sigma = sigma_matrix();
        let x = [1i64, 0, 1, 0, 0, 0, 0, 0];
        let mut embed = [0.0f64; 8];
        for i in 0..8 {
            for j in 0..8 { embed[i] += sigma[i][j] * x[j] as f64; }
        }
        // Row 0 = Re(u) = 1+√3, Row 2 = Re(u•) = 1-√3
        assert!(near(embed[0], 1.0 + SQRT3, 1e-12), "Re(u)={}", embed[0]);
        assert!(near(embed[2], 1.0 - SQRT3, 1e-12), "Re(u•)={}", embed[2]);

        // For u = i√3 = (0,0,0,1): Im(u) = √3, Im(u•) = -√3.
        let x2 = [0i64, 0, 0, 1, 0, 0, 0, 0];
        let mut embed2 = [0.0f64; 8];
        for i in 0..8 {
            for j in 0..8 { embed2[i] += sigma[i][j] * x2[j] as f64; }
        }
        assert!(near(embed2[1], SQRT3, 1e-12), "Im(u)={}", embed2[1]);   // Im(u) = √3
        assert!(near(embed2[3], -SQRT3, 1e-12), "Im(u•)={}", embed2[3]); // Im(u•) = -√3
    }

    #[test]
    fn check_norm_and_bilinear_on_known_point() {
        // u=(1,0,0,0), t=(0,0,0,0): |u|²+|t|²=1=3^0 → k=0.
        let x = [1i64,0,0,0, 0,0,0,0];
        assert!(check_norm_eq(&x, 0), "identity should have k=0 norm");
        assert!(check_bilinear(&x), "identity bilinear");

        // u=(0,0,1,0), t=(0,0,0,0): |u|²=3=3^1 → k=1, bilinear a₁c₁=0.
        let x1 = [0i64,0,1,0, 0,0,0,0];
        assert!(check_norm_eq(&x1, 1), "sqrt3 should have k=1 norm");
        assert!(check_bilinear(&x1), "sqrt3 bilinear");
    }

    #[test]
    fn solution_to_mat2_identity() {
        // u=1, t=0, k=0 → I.
        let x = [1i64,0,0,0, 0,0,0,0];
        let m = solution_to_mat2(&x, 0);
        assert!(near(m[0][0].re, 1.0, 1e-12));
        assert!(near(m[0][0].im, 0.0, 1e-12));
        assert!(near(m[1][1].re, 1.0, 1e-12));
        assert!(near(m[0][1].norm(), 0.0, 1e-12));
        assert!(near(m[1][0].norm(), 0.0, 1e-12));
    }

    #[test]
    fn solution_to_mat2_unitarity() {
        // Any lattice solution with |u|²+|t|²=3^k should give a unitary matrix.
        // Use u=1, t=0, k=0.
        let x = [1i64,0,0,0, 0,0,0,0];
        let m = solution_to_mat2(&x, 0);
        // U†U should be I: check by float matmul
        let (u00, u01, u10, u11) = (m[0][0], m[0][1], m[1][0], m[1][1]);
        let d00 = u00.norm_sqr() + u10.norm_sqr(); // (U†U)[0][0]
        let d11 = u01.norm_sqr() + u11.norm_sqr(); // (U†U)[1][1]
        let off = u00.conj()*u01 + u10.conj()*u11;  // (U†U)[0][1]
        assert!(near(d00, 1.0, 1e-12), "d00={d00}");
        assert!(near(d11, 1.0, 1e-12), "d11={d11}");
        assert!(near(off.norm(), 0.0, 1e-12), "off={off}");
    }

    // ── Alignment vector ──────────────────────────────────────────────────────

    #[test]
    fn y_vector_gram_norm_is_one() {
        // yᵀ D⁻¹ y = 1 for any unit v.
        let v = [0.6_f64, 0.8_f64]; // |v|=1
        let y = compute_y(v[0], v[1]);
        let d_inv = [0.5, 0.5, 1.0/6.0, 1.0/6.0, 0.5, 0.5, 1.0/6.0, 1.0/6.0];
        let norm: f64 = y.iter().zip(d_inv.iter()).map(|(yi, di)| yi*yi*di).sum();
        assert!(near(norm, 1.0, 1e-12), "yᵀD⁻¹y={norm}");
    }

    #[test]
    fn y_vector_dot_equals_re_u_dot_v() {
        // x·y = Re(u)·v_re + Im(u)·v_im for any x.
        let v = [0.6_f64, 0.8_f64];
        let y = compute_y(v[0], v[1]);
        // Use x = (a₁,b₁,c₁,d₁,...) = (2,3,1,-1,...):
        let x = [2i64, 3, 1, -1, 5, -2, 0, 1];
        let re_u = x[0] as f64 + x[2] as f64 * SQRT3; // a₁+c₁√3
        let im_u = x[1] as f64 + x[3] as f64 * SQRT3; // b₁+d₁√3
        let expected = re_u * v[0] + im_u * v[1];
        let dot: f64 = x.iter().zip(y.iter()).map(|(&xi,&yi)| xi as f64 * yi).sum();
        assert!(near(dot, expected, 1e-10), "dot={dot} expected={expected}");
    }

    // ── Search correctness ────────────────────────────────────────────────────

    #[test]
    fn search_k0_finds_identity() {
        // At k=0, norm=2·3^0=2: solutions are u with |u|²=1, t=0.
        // u ∈ {±1, ±i} → x ∈ {(±1,0,0,0,0,0,0,0), (0,±1,0,0,0,0,0,0)}.
        let y = compute_y(1.0, 0.0);
        let sols = direct_search_n6(pow3(0), &y, 0.0, 100);
        assert!(!sols.is_empty(), "should find solutions at k=0");
        for sol in &sols {
            assert!(check_norm_eq(sol, 0), "norm failed for {sol:?}");
            assert!(check_bilinear(sol), "bilinear failed for {sol:?}");
        }
        let found_id = sols.iter().any(|s| *s == [1,0,0,0,0,0,0,0] || *s == [-1,0,0,0,0,0,0,0]);
        assert!(found_id, "should find identity solution");
    }

    #[test]
    fn search_k1_norm_and_bilinear() {
        let y = compute_y(1.0, 0.0);
        let sols = direct_search_n6(pow3(1), &y, 0.0, 1000);
        assert!(!sols.is_empty(), "k=1 should have solutions");
        for sol in &sols {
            assert!(check_norm_eq(sol, 1), "norm failed: {sol:?}");
            assert!(check_bilinear(sol), "bilinear failed: {sol:?}");
        }
    }

    #[test]
    fn search_all_solutions_satisfy_constraints() {
        // k=2, with alignment filter eps=0.5
        let v = [0.8_f64, 0.6];
        let y = compute_y(v[0], v[1]);
        let sols = direct_search_n6(pow3(2), &y, 0.5, 500);
        for sol in &sols {
            assert!(check_norm_eq(sol, 2), "norm: {sol:?}");
            assert!(check_bilinear(sol), "bilinear: {sol:?}");
            let dot: f64 = sol.iter().zip(y.iter()).map(|(&x,&y)| x as f64 * y).sum();
            let thresh = pow3(2) as f64 * (1.0 - 0.25);
            assert!(dot*dot >= thresh - 1e-9, "alignment: {sol:?} dot={dot}");
        }
    }

    // ── Synthesizer smoke tests ───────────────────────────────────────────────

    #[test]
    fn synthesize_identity() {
        let id: Mat2 = [
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        ];
        let synth = SynthesizerPi6::new(0.01).with_min_lde(0);
        let result = synth.synthesize(id).expect("should synthesize identity");
        assert!(result.distance < 0.01, "distance={}", result.distance);
        assert_eq!(result.lde, 0, "identity should have lde=0");
    }

    #[test]
    fn synthesize_s_gate() {
        let s: Mat2 = [
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(0.0, 1.0)],
        ];
        let synth = SynthesizerPi6::new(0.01).with_min_lde(0);
        let result = synth.synthesize(s).expect("should synthesize S");
        assert!(result.distance < 0.01, "distance={}", result.distance);
        assert_eq!(result.lde, 0, "S is Clifford → lde=0");
    }

    #[test]
    fn synthesize_rz_pi6_gate() {
        // R_z(π/3) = diag(e^{-iπ/6}, e^{iπ/6}) — one R_z(π/6) gate.
        let target = rz(PI / 3.0);
        let synth = SynthesizerPi6::new(0.01).with_min_lde(0).with_max_lde(4);
        let result = synth.synthesize(target).expect("should synthesize Rz(π/3)");
        assert!(result.distance < 0.01, "distance={:.4e}", result.distance);
        assert!(result.lde <= 4);
        eprintln!("Rz(π/3): lde={} dist={:.4e}", result.lde, result.distance);
    }

    #[test]
    fn synthesize_rz_small_angle() {
        let target = rz(0.3_f64);
        let synth = SynthesizerPi6::new(0.05).with_min_lde(0).with_max_lde(8);
        let result = synth.synthesize(target).expect("should synthesize Rz(0.3)");
        assert!(result.distance < 0.05, "distance={:.4e}", result.distance);
        eprintln!("Rz(0.3) @ eps=0.05: lde={} dist={:.4e}", result.lde, result.distance);
    }

    // ── SO3 and decomposer ────────────────────────────────────────────────────

    #[test]
    fn so3_of_identity_is_identity() {
        let id = eye_mat();
        let m = mat_to_so3(&id);
        for i in 0..3 { for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(near(m[i][j], expected, 1e-12), "SO3(I)[{i}][{j}]={}", m[i][j]);
        }}
    }

    #[test]
    fn so3_of_rz_pi6_matches_formula() {
        let rz = rz_pi6_mat();
        let m = mat_to_so3(&rz);
        // SO3(Rz(θ)) = [[cos(θ),-sin(θ),0],[sin(θ),cos(θ),0],[0,0,1]] where θ=π/6
        let (c, s) = ((PI/6.).cos(), (PI/6.).sin());
        assert!(near(m[0][0], c, 1e-10));
        assert!(near(m[0][1], -s, 1e-10));
        assert!(near(m[1][0], s, 1e-10));
        assert!(near(m[1][1], c, 1e-10));
        assert!(near(m[2][2], 1.0, 1e-12));
        assert!(near(m[0][2], 0., 1e-12));
        assert!(near(m[2][0], 0., 1e-12));
    }

    #[test]
    fn decompose_identity_gives_empty_or_clifford() {
        let id = eye_mat();
        let g = decompose_pi6(&id);
        // Identity should decompose to "" or some Clifford equivalent of I
        let m = eval_gate_string(&g);
        let d = diamond_distance_float(&m, &id);
        assert!(d < 1e-9, "decompose(I)=\"{g}\" dist={d:.3e}");
    }

    #[test]
    fn decompose_rz_pi6_round_trip() {
        let rz = rz_pi6_mat();
        let g = decompose_pi6(&rz);
        let m = eval_gate_string(&g);
        let d = diamond_distance_float(&m, &rz);
        assert!(d < 1e-9, "decompose(Rz(π/6))=\"{g}\" dist={d:.3e}");
    }

    #[test]
    fn decompose_s_gate_round_trip() {
        let s: Mat2 = [
            [Complex64::new(1., 0.), Complex64::new(0., 0.)],
            [Complex64::new(0., 0.), Complex64::new(0., 1.)],
        ];
        let g = decompose_pi6(&s);
        let m = eval_gate_string(&g);
        let d = diamond_distance_float(&m, &s);
        assert!(d < 1e-9, "decompose(S)=\"{g}\" dist={d:.3e}");
    }

    // ── Gate string round-trip ────────────────────────────────────────────────

    #[test]
    fn synthesize_identity_gates_round_trip() {
        let id: Mat2 = eye_mat();
        let synth = SynthesizerPi6::new(0.01).with_min_lde(0);
        let result = synth.synthesize(id).expect("should synthesize identity");
        assert!(result.distance < 0.01);
        if let Some(ref g) = result.gates {
            let m = eval_gate_string(g);
            let d = diamond_distance_float(&m, &id);
            assert!(d < 0.01, "gate round-trip failed: \"{g}\" dist={d:.3e}");
        }
    }

    #[test]
    fn synthesize_rz_pi6_gates_round_trip() {
        let target = rz(PI / 3.0);
        let synth = SynthesizerPi6::new(0.01).with_min_lde(0).with_max_lde(5);
        let result = synth.synthesize(target).expect("should synthesize Rz(π/3)");
        assert!(result.distance < 0.01, "dist={:.4e}", result.distance);
        if let Some(ref g) = result.gates {
            let m = eval_gate_string(g);
            let d = diamond_distance_float(&m, &target);
            assert!(d < 0.01, "gate round-trip: \"{g}\" dist={d:.3e}");
            eprintln!("Rz(π/3): lde={} gates=\"{g}\" dist={:.4e}", result.lde, result.distance);
        }
    }

    #[test]
    fn dc_search_fires_and_finds_solution() {
        // Force DC path by setting direct_limit=2, then synthesize at k>2.
        let target = rz(0.3);
        let synth = SynthesizerPi6::new(0.1)
            .with_min_lde(0)
            .with_max_lde(12)
            .with_direct_limit(2);
        let result = synth.synthesize(target).expect("DC should find a solution");
        assert!(result.distance < 0.1, "dist={:.4e}", result.distance);
        eprintln!("DC Rz(0.3) @ eps=0.1: lde={} dist={:.4e} gates={:?}", result.lde, result.distance, result.gates);
        if let Some(ref g) = result.gates {
            let m = eval_gate_string(g);
            let d = diamond_distance_float(&m, &target);
            assert!(d < 0.1, "gate round-trip dist={d:.3e}");
        }
    }

    #[test]
    fn build_l_pi6_sizes() {
        for k_prime in 0u32..=4 {
            let l = build_l_pi6(k_prime);
            eprintln!("|L_{{{}}}| = {}", k_prime, l.len());
            assert!(!l.is_empty());
            // k'=0: only identity; k'≥1: at least 24 (one Clifford per syllable pattern)
            if k_prime == 0 { assert_eq!(l.len(), 1); }
            else { assert!(l.len() >= 24); }
        }
    }

    // ── Helper: evaluate a gate string to a float Mat2 ────────────────────────

    fn eval_gate_string(gates: &str) -> Mat2 {
        let mut m = eye_mat();
        for ch in gates.chars() {
            let g: Mat2 = match ch {
                'H' => CLIFFORD_TABLE_T.iter().find(|(n,_)| *n=="H").unwrap().1.to_float(),
                'S' => CLIFFORD_TABLE_T.iter().find(|(n,_)| *n=="S").unwrap().1.to_float(),
                'X' => CLIFFORD_TABLE_T.iter().find(|(n,_)| *n=="X").unwrap().1.to_float(),
                'Y' => CLIFFORD_TABLE_T.iter().find(|(n,_)| *n=="Y").unwrap().1.to_float(),
                'Z' => CLIFFORD_TABLE_T.iter().find(|(n,_)| *n=="Z").unwrap().1.to_float(),
                'R' => rz_pi6_mat(),
                'I' => eye_mat(),
                other => panic!("unknown gate '{other}'"),
            };
            m = mat_mul(m, g);
        }
        m
    }

    #[test]
    fn norm_eq_for_search_solutions_implies_unitarity() {
        // Any solution with check_norm_eq and check_bilinear gives a unitary matrix.
        let y = compute_y(1.0, 0.0);
        let sols = direct_search_n6(pow3(1), &y, 0.0, 50);
        for sol in &sols {
            let m = solution_to_mat2(sol, 1);
            // U†U ≈ I
            let d00 = m[0][0].norm_sqr() + m[1][0].norm_sqr();
            let d11 = m[0][1].norm_sqr() + m[1][1].norm_sqr();
            let off = m[0][0].conj()*m[0][1] + m[1][0].conj()*m[1][1];
            assert!(near(d00, 1.0, 1e-9), "d00={d00} for {sol:?}");
            assert!(near(d11, 1.0, 1e-9), "d11={d11} for {sol:?}");
            assert!(near(off.norm(), 0.0, 1e-9), "off={off} for {sol:?}");
        }
    }
}
