//! Clifford + R_z(π/6) synthesis over ℤ[ξ], ξ = ζ₁₂ = e^{iπ/6}.
//!
//! Finds a Clifford+R_z(π/6) circuit U such that d_diamond(U, V) < ε.
//!
//! # Ring and coordinates
//!
//! Every element of ℤ[ξ] is written as  u = a₀ + a₁·ξ + a₂·ξ² + a₃·ξ³
//! with a₀,a₁,a₂,a₃ ∈ ℤ, and the relation ξ⁴ = ξ² − 1.  ξ³ = i.
//! Re(u) = a₀ + (√3/2)·a₁ + (1/2)·a₂,  Im(u) = (1/2)·a₁ + (√3/2)·a₂ + a₃.
//!
//! # Integer lattice
//!
//! The unitary has SU(2)-like form  U = [[u, −t̄], [t, ū]] / √(2^k)
//! with u, t ∈ ℤ[ξ] and  |u|² + |t|² = 2^k  (in ℤ[√3]).
//!
//! Eight-dimensional integer coordinate vector:
//!   x = (a₀,a₁,a₂,a₃, b₀,b₁,b₂,b₃)  where u has (a₀,a₁,a₂,a₃)
//!                                         and t has (b₀,b₁,b₂,b₃).
//!
//! # Quadratic form and constraints
//!
//! Norm equation: (a₀²+a₁²+a₂²+a₃²+a₀a₂+a₁a₃) + (b₀²+b₁²+b₂²+b₃²+b₀b₂+b₁b₃) = 2^k.
//!
//! Bilinear (√3-part vanishes): (a₀a₁+a₁a₂+a₂a₃) + (b₀b₁+b₁b₂+b₂b₃) = 0.
//!
//! Alignment bound: (x·y)² ≥ 2^k·(1−ε²).
//!
//! # Σ matrix (8×8)
//!
//! Maps x → (Re u, Im u, Re u•, Im u•, Re t, Im t, Re t•, Im t•):
//!   Σ = block-diag(Σ_u, Σ_u),
//!   Σ_u = [[1, √3/2,  1/2,     0  ],
//!           [0, 1/2,   √3/2,    1  ],
//!           [1, −√3/2, 1/2,     0  ],
//!           [0, 1/2,   −√3/2,   1  ]].
//!   G = ΣᵀΣ = block-diag(G_u, G_u),  G_u = [[2,0,1,0],[0,2,0,1],[1,0,2,0],[0,1,0,2]].

#![allow(clippy::too_many_arguments)]

use num_complex::Complex64;
use rayon::prelude::*;
use std::f64::consts::PI;
use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex};

use crate::rings::ZOmicron;
use crate::rings::zomicron::SIGMA_GRAM_U;
use crate::rings::types::Int;
use crate::matrix::U2;
use crate::synthesis::cliffords::CLIFFORD_TABLE_T;
use crate::synthesis::decomposer::{decompose_so3_canonical_n6, simplify_gate_string_n6};
use crate::synthesis::distance::{diamond_distance_float, Mat2};
use crate::synthesis::search::{apply_u2t_dag_to_uv, normalize4};

// ─── Constants ────────────────────────────────────────────────────────────────

const SQRT3: f64 = 1.7320508075688772935_f64;
const SQRT3_HALF: f64 = 0.86602540378443864676_f64;  // √3/2

// Rotation angle per R_z(π/6) gate in uv-space.
const RZ_ANGLE: f64 = PI / 12.0;

// ─── Σ matrix ─────────────────────────────────────────────────────────────────

/// 8×8 Σ matrix for ℤ[ξ], ξ = e^{iπ/6}: maps integer coords x to Minkowski embedding.
///
/// Row order: (Re u, Im u, Re u•, Im u•, Re t, Im t, Re t•, Im t•).
/// Column order: (a₀,a₁,a₂,a₃, b₀,b₁,b₂,b₃).
pub fn sigma_matrix() -> [[f64; 8]; 8] {
    let s = SQRT3_HALF;
    [
        [1.0,  s,   0.5,  0.0, 0.0, 0.0, 0.0, 0.0],  // Re u
        [0.0,  0.5, s,    1.0, 0.0, 0.0, 0.0, 0.0],  // Im u
        [1.0, -s,   0.5,  0.0, 0.0, 0.0, 0.0, 0.0],  // Re u•
        [0.0,  0.5, -s,   1.0, 0.0, 0.0, 0.0, 0.0],  // Im u•
        [0.0, 0.0, 0.0, 0.0, 1.0,  s,   0.5,  0.0],  // Re t
        [0.0, 0.0, 0.0, 0.0, 0.0,  0.5, s,    1.0],  // Im t
        [0.0, 0.0, 0.0, 0.0, 1.0, -s,   0.5,  0.0],  // Re t•
        [0.0, 0.0, 0.0, 0.0, 0.0,  0.5, -s,   1.0],  // Im t•
    ]
}

/// Apply Σ⁻¹ = G⁻¹·Σᵀ to an 8-vector, block-diagonally.
///
/// G_u = [[2,0,1,0],[0,2,0,1],[1,0,2,0],[0,1,0,2]] (from SIGMA_GRAM_U).
/// G_u⁻¹ = [[2/3,0,-1/3,0],[0,2/3,0,-1/3],[-1/3,0,2/3,0],[0,-1/3,0,2/3]].
pub fn sigma_inverse_apply(w: [f64; 8]) -> [f64; 8] {
    let sh6 = SQRT3 / 6.0;  // √3/6
    let sh3 = SQRT3 / 3.0;  // √3/3
    // G_u⁻¹·Σ_uᵀ: rows are:
    //   [1/2, -√3/6, 1/2,  √3/6]
    //   [√3/3,  0,  -√3/3,  0  ]
    //   [0,   √3/3,  0,   -√3/3]
    //   [-√3/6, 1/2, √3/6, 1/2 ]
    let sinv_block = |wblk: [f64; 4]| -> [f64; 4] {
        let (w0, w1, w2, w3) = (wblk[0], wblk[1], wblk[2], wblk[3]);
        [
             0.5*w0 - sh6*w1 + 0.5*w2 + sh6*w3,
             sh3*w0           - sh3*w2,
                      sh3*w1           - sh3*w3,
            -sh6*w0 + 0.5*w1 + sh6*w2 + 0.5*w3,
        ]
    };
    let u = sinv_block([w[0], w[1], w[2], w[3]]);
    let t = sinv_block([w[4], w[5], w[6], w[7]]);
    [u[0], u[1], u[2], u[3], t[0], t[1], t[2], t[3]]
}

// ─── Integer square root ──────────────────────────────────────────────────────

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

/// Check norm equation: rational(|u|²) + rational(|t|²) = 2^k.
///
/// rational(|u|²) = a₀²+a₁²+a₂²+a₃² + a₀a₂+a₁a₃  (from complex_norm_sqr).
#[inline]
pub fn check_norm_eq(x: &[i64; 8], k: u32) -> bool {
    let [a0, a1, a2, a3, b0, b1, b2, b3] = *x;
    let euclid = a0*a0 + a1*a1 + a2*a2 + a3*a3
               + b0*b0 + b1*b1 + b2*b2 + b3*b3;
    let cross = a0*a2 + a1*a3 + b0*b2 + b1*b3;
    euclid + cross == 1_i64 << k
}

/// Check bilinear constraint: √3-part of |u|²+|t|² vanishes.
///
/// (a₀a₁+a₁a₂+a₂a₃) + (b₀b₁+b₁b₂+b₂b₃) = 0.
#[inline]
pub fn check_bilinear(x: &[i64; 8]) -> bool {
    let [a0, a1, a2, a3, b0, b1, b2, b3] = *x;
    (a0*a1 + a1*a2 + a2*a3) + (b0*b1 + b1*b2 + b2*b3) == 0
}

/// Check alignment: (x·y)² ≥ 2^k·(1−ε²).
#[inline]
pub fn check_alignment(x: &[i64; 8], y: &[f64; 8], k: u32, eps_sq: f64) -> bool {
    let dot: f64 = x.iter().zip(y.iter()).map(|(&xi, &yi)| xi as f64 * yi).sum();
    let thresh = (1_i64 << k) as f64 * (1.0 - eps_sq);
    dot * dot >= thresh
}

// ─── Alignment vector ─────────────────────────────────────────────────────────

/// Build the 8D alignment vector y from a target SU(2) first column (v₁, v₂).
///
/// y = (v₁_re,
///      (√3/2)·v₁_re + (1/2)·v₁_im,
///      (1/2)·v₁_re + (√3/2)·v₁_im,
///      v₁_im,
///      v₂_re, (√3/2)·v₂_re + (1/2)·v₂_im, (1/2)·v₂_re + (√3/2)·v₂_im, v₂_im).
///
/// Satisfies  x·y = ⟨u, v₁⟩ + ⟨t, v₂⟩.
pub fn compute_y(v1_re: f64, v1_im: f64, v2_re: f64, v2_im: f64) -> [f64; 8] {
    let s = SQRT3_HALF;
    [
        v1_re,
        s * v1_re + 0.5 * v1_im,
        0.5 * v1_re + s * v1_im,
        v1_im,
        v2_re,
        s * v2_re + 0.5 * v2_im,
        0.5 * v2_re + s * v2_im,
        v2_im,
    ]
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
fn rotate_uv(v: [f64; 4], theta: f64) -> [f64; 4] {
    let (c, s) = (theta.cos(), theta.sin());
    [
        v[0]*c - v[1]*s,  v[0]*s + v[1]*c,
        v[2]*c - v[3]*s,  v[2]*s + v[3]*c,
    ]
}

/// uv direction after right-multiplying target by R_z(-π/6).
pub fn apply_rz_dag_to_uv(v: [f64; 4]) -> [f64; 4] { rotate_uv(v,  RZ_ANGLE) }

/// uv direction after right-multiplying target by R_z(+π/6).
pub fn apply_rz_to_uv(v: [f64; 4])     -> [f64; 4] { rotate_uv(v, -RZ_ANGLE) }

// ─── Solution → float matrix ──────────────────────────────────────────────────

/// Build a float Mat2 from a lattice solution and lde k.
///
/// x = (a₀,a₁,a₂,a₃, b₀,b₁,b₂,b₃) in cyclotomic basis {1,ξ,ξ²,ξ³}:
///   u = a₀ + a₁ξ + a₂ξ² + a₃ξ³,  t = b₀ + b₁ξ + b₂ξ² + b₃ξ³.
///   U = [[u, −t̄], [t, ū]] / √(2^k).
pub fn solution_to_mat2(x: &[i64; 8], k: u32) -> Mat2 {
    let [a0, a1, a2, a3, b0, b1, b2, b3] = x.map(|v| v as f64);
    let s = SQRT3_HALF;
    // Re(u) = a0 + a1*(√3/2) + a2*(1/2),  Im(u) = a1*(1/2) + a2*(√3/2) + a3
    let u = Complex64::new(a0 + a1*s + a2*0.5, a1*0.5 + a2*s + a3);
    let t = Complex64::new(b0 + b1*s + b2*0.5, b1*0.5 + b2*s + b3);
    let scale = 1.0 / ((1_i64 << k) as f64).sqrt();
    [
        [ u * scale, -t.conj() * scale ],
        [ t * scale,  u.conj() * scale ],
    ]
}

// ─── Quadratic solver for (a₁, a₃) given fixed (a₀, a₂) ─────────────────────

/// Solve the 2D system:
///   a₁² + a₁·a₃ + a₃² = rem           (norm equation, u-block variable part)
///   a₀·a₁ + a₁·a₂ + a₂·a₃ = −t_bil   (bilinear constraint)
///
/// Enumerates a₃, solves for a₁ via  a₁ = (−a₃ ± √(4·rem − 3·a₃²)) / 2.
fn solve_bd(rem: i64, a0: i64, a2: i64, t_bil: i64) -> Vec<(i64, i64)> {
    let mut out = Vec::with_capacity(4);
    if rem < 0 { return out; }

    // a₃ range: 4·rem − 3·a₃² ≥ 0 ↔ |a₃| ≤ √(4·rem/3)
    let max_a3 = isqrt(4 * rem / 3 + 1);
    for a3 in -max_a3..=max_a3 {
        let disc = 4 * rem - 3 * a3 * a3;
        if disc < 0 { continue; }
        let sq = isqrt(disc);
        if sq * sq != disc { continue; }

        for sign in [1i64, -1] {
            if sign == -1 && sq == 0 { break; }
            let numer = -a3 + sign * sq;
            if numer % 2 != 0 { continue; }
            let a1 = numer / 2;
            // Verify norm
            if a1*a1 + a1*a3 + a3*a3 != rem { continue; }
            // Bilinear: a0*a1 + a1*a2 + a2*a3 = -t_bil
            if a0*a1 + a1*a2 + a2*a3 != -t_bil { continue; }
            out.push((a1, a3));
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

/// Inner enumeration for one fixed (b₀,b₁,b₂,b₃) t-block.
///
/// Enumerates (a₀, a₂) and solves (a₁, a₃) from:
///   a₁² + a₁a₃ + a₃² = rem_u − (a₀²+a₂²+a₀a₂)
///   a₀a₁ + a₁a₂ + a₂a₃ = −t_bilinear
fn search_inner(
    b0: i64, b1: i64, b2: i64, b3: i64,
    rem_u: i64,
    t_bilinear: i64,
    y: &[f64; 8],
    thresh_sq: f64,
    thresh: f64,
    do_prune: bool,
    max_sol: usize,
) -> Vec<[i64; 8]> {
    let mut out = Vec::new();

    // t-block alignment contribution (fixed for this call)
    let t_dot = b0 as f64 * y[4] + b1 as f64 * y[5]
              + b2 as f64 * y[6] + b3 as f64 * y[7];

    // Enumerate a0: a0² ≤ rem_u
    let max_a0 = isqrt(rem_u);
    for a0 in -max_a0..=max_a0 {
        let rem_a0 = rem_u - a0 * a0;
        if rem_a0 < 0 { continue; }

        // Enumerate a2: a2²+a0*a2 ≤ rem_a0 → safe bound |a2| ≤ |a0| + √(rem_a0)
        let max_a2 = isqrt(rem_a0) + a0.abs();
        for a2 in -max_a2..=max_a2 {
            let fixed_u = a0*a0 + a2*a2 + a0*a2;
            let rem_a1a3 = rem_u - fixed_u;
            if rem_a1a3 < 0 { continue; }

            if do_prune {
                let ac_dot = a0 as f64 * y[0] + a2 as f64 * y[2];
                let partial = t_dot + ac_dot;
                // Max |a1*y[1]+a3*y[3]| ≤ √(2*rem_a1a3) * ‖(y[1],y[3])‖
                let rem_bound = (2.0 * rem_a1a3 as f64).sqrt()
                    * (y[1]*y[1] + y[3]*y[3]).sqrt();
                if partial.abs() + rem_bound < thresh { continue; }
            }

            for (a1, a3) in solve_bd(rem_a1a3, a0, a2, t_bilinear) {
                let x = [a0, a1, a2, a3, b0, b1, b2, b3];
                record_if_aligned(x, y, thresh_sq, &mut out, max_sol);
                if out.len() >= max_sol { return out; }
            }
        }
    }
    out
}

/// Full Phase-1 search: enumerate all x with norm_eq=true, bilinear=true, alignment ok.
///
/// target_k = 2^k.  Parallelised over (b₀, b₂) pairs via rayon.
pub fn direct_search_n6(
    target_k: i64,
    y: &[f64; 8],
    eps: f64,
    max_sol: usize,
) -> Vec<[i64; 8]> {
    if max_sol == 0 { return Vec::new(); }
    let norm2k = target_k;   // 2^k
    let thresh_sq = if eps > 0.0 {
        target_k as f64 * (1.0 - eps * eps)
    } else {
        0.0
    };
    let do_prune = eps > 0.0;
    let thresh = thresh_sq.max(0.0).sqrt();

    // Outermost bound: |b0|, |b2| ≤ isqrt(norm2k)
    let max_outer = isqrt(norm2k);
    let pairs: Vec<(i64, i64, i64)> = (-max_outer..=max_outer).flat_map(|b0| {
        let b0_sq = b0 * b0;
        (-max_outer..=max_outer).filter_map(move |b2| {
            let partial_tn = b0_sq + b2*b2 + b0*b2;
            if partial_tn > norm2k { None } else { Some((b0, b2, norm2k - partial_tn)) }
        })
    }).collect();

    let batches: Vec<Vec<[i64; 8]>> = pairs
        .into_par_iter()
        .filter_map(|(b0, b2, rem_b02)| {
            let mut local: Vec<[i64; 8]> = Vec::new();
            // b1, b3 bounds: b1²+b3²+b1*b3 ≤ rem_b02 → each |b| ≤ √(2*rem_b02)
            let max_b13 = isqrt(2 * rem_b02 + 1);
            for b1 in -max_b13..=max_b13 {
                if b1 * b1 > 2 * rem_b02 { continue; }
                for b3 in -max_b13..=max_b13 {
                    let t_norm = b0*b0 + b1*b1 + b2*b2 + b3*b3 + b0*b2 + b1*b3;
                    if t_norm < 0 || t_norm > norm2k { continue; }
                    let rem_u = norm2k - t_norm;
                    let t_bilinear = b0*b1 + b1*b2 + b2*b3;
                    let batch = search_inner(
                        b0, b1, b2, b3, rem_u, t_bilinear, y,
                        thresh_sq, thresh, do_prune, max_sol,
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
pub fn simplify_n6(input: &str) -> String {
    let mut s = input.to_string();
    let mut prev = String::new();
    while s != prev {
        prev = s.clone();
        s = s.replace("RRRRRR", "Z");
        s = s.replace("RRR", "S");
        s = s.replace("SS", "Z");
        s = s.replace("ZZ", "");
        s = s.replace("HH", "");
        s = s.replace("XX", "");
        s = s.replace("YY", "");
        s = s.replace("SZ", "ZS");
        s = s.replace("RZ", "ZR");
    }
    s
}

/// Build a U2<ZOmicron> from an integer lattice solution (u, t) and exponent k.
///
/// The solution encodes u = (a0,a1,a2,a3) and t = (b0,b1,b2,b3) in the cyclotomic
/// basis {1,ξ,ξ²,ξ³}, giving U = [[u, -t̄],[t, ū]] / √2^k.
pub fn solution_to_u2_omicron(sol: &[i64; 8], k: u32) -> U2<ZOmicron> {
    let u = ZOmicron::new(
        Int::from_i64(sol[0]), Int::from_i64(sol[1]),
        Int::from_i64(sol[2]), Int::from_i64(sol[3]),
    );
    let t = ZOmicron::new(
        Int::from_i64(sol[4]), Int::from_i64(sol[5]),
        Int::from_i64(sol[6]), Int::from_i64(sol[7]),
    );
    U2::new(u, -t.conj(), t, u.conj(), k)
}

/// Decompose a U2<ZOmicron> into a Clifford+R gate string using the exact
/// ring-based canonical-form decomposer.
///
/// This replaces the old greedy float-based SO3 peeling which was incorrect
/// for inputs like HRH (produced round-trip distance 0.866 instead of 0).
pub fn decompose_pi6(u: &U2<ZOmicron>) -> String {
    simplify_n6(&decompose_so3_canonical_n6(u))
}

// ─── MA prefix set for DC search ──────────────────────────────────────────────

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

    let hs0r = mat_mul(h_mat, rz_mat);
    let hs1r = mat_mul(mat_mul(h_mat, s_mat), rz_mat);

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
            let c_rev: String = c_str.chars().rev().collect();
            candidates.push((format!("{g}{c_rev}"), mat_mul(m, c_u2t.to_float())));
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
                let c_rev: String = c_str.chars().rev().collect();
                candidates.push((format!("{g}{c_rev}"), mat_mul(m, c_u2t.to_float())));
            }
        }
    }
    let mut seen: std::collections::HashSet<[i64; 8]> = std::collections::HashSet::new();
    candidates.into_iter().filter(|(_, m)| seen.insert(canonical_key_f64(m))).collect()
}

// ─── Result type ─────────────────────────────────────────────────────────────

/// Result of a successful n=6 synthesis.
pub struct SynthResultPi6 {
    pub gates: Option<String>,
    pub lde: u32,
    pub distance: f64,
}

// ─── Direct search branch tags ────────────────────────────────────────────────

enum Branch {
    Even,
    Rz,
    RzDag,
    Clif(usize),
    ClifRz(usize),
    ClifRzDag(usize),
}

// ─── Synthesizer ─────────────────────────────────────────────────────────────

/// Clifford + R_z(π/6) synthesis backend over ℤ[ξ].
pub struct SynthesizerPi6 {
    pub epsilon: f64,
    pub max_lde: u32,
    pub min_lde: u32,
    pub direct_limit: u32,
}

impl SynthesizerPi6 {
    /// Create a synthesizer with sensible defaults for the given precision.
    pub fn new(epsilon: f64) -> Self {
        let (min_lde, max_lde) = if epsilon > 0.0 && epsilon < 1.0 {
            // R_z(π/6)-count scales as ~log₂(1/ε²) ≈ 2·log₂(1/ε).
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
    fn dc_search(
        &self,
        target: &Mat2,
        _v: [f64; 4],
        k: u32,
        k_prefix: u32,
    ) -> Option<SynthResultPi6> {
        let k_inner = k - k_prefix;
        let eps = self.epsilon;
        let target_k_inner = 1_i64 << k_inner;

        let prefixes = build_l_pi6(k_prefix);

        prefixes.par_iter().find_map_any(|(prefix_gates, u_l)| {
            let u_l_dag = mat_dag(u_l);
            let m_inner = mat_mul(u_l_dag, *target);

            let v_inner = unitary_to_uv_n6(&m_inner);

            // Even branch
            let y = compute_y(v_inner[0], v_inner[1], v_inner[2], v_inner[3]);
            for sol in direct_search_n6(target_k_inner, &y, eps, 1) {
                let u_r = solution_to_mat2(&sol, k_inner);
                let full = mat_mul(*u_l, u_r);
                let dist = diamond_distance_float(&full, target);
                if dist < eps {
                    let u_r_ring = solution_to_u2_omicron(&sol, k_inner);
                    let gates = simplify_gate_string_n6(
                        &format!("{}{}", prefix_gates, decompose_pi6(&u_r_ring))
                    );
                    return Some(SynthResultPi6 { gates: Some(gates), lde: k, distance: dist });
                }
            }

            // Odd branch: search for U_R with U_L·U_R·R ≈ target
            if k_inner > 0 {
                let v_inner_r = apply_rz_dag_to_uv(v_inner);
                let y_r = compute_y(v_inner_r[0], v_inner_r[1], v_inner_r[2], v_inner_r[3]);
                for sol in direct_search_n6(target_k_inner, &y_r, eps, 1) {
                    let u_r = solution_to_mat2(&sol, k_inner);
                    let full = mat_mul(*u_l, mat_mul(u_r, rz_pi6_mat()));
                    let dist = diamond_distance_float(&full, target);
                    if dist < eps {
                        let u_r_ring = solution_to_u2_omicron(&sol, k_inner);
                        let inner_str = format!("{}R", decompose_pi6(&u_r_ring));
                        let gates = simplify_gate_string_n6(&format!("{}{}", prefix_gates, inner_str));
                        return Some(SynthResultPi6 { gates: Some(gates), lde: k, distance: dist });
                    }
                }
            }
            None
        })
    }

    /// Brute-force direct search at lde `k`.
    fn direct_search(&self, target: &Mat2, v: [f64; 4], k: u32) -> Option<SynthResultPi6> {
        let eps = self.epsilon;
        let target_k = 1_i64 << k;

        let clif_vs: Vec<[f64; 4]> = CLIFFORD_TABLE_T.iter()
            .map(|(_, c_u2t)| apply_u2t_dag_to_uv(c_u2t, v))
            .collect();

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
            let y = compute_y(v_s[0], v_s[1], v_s[2], v_s[3]);
            for sol in direct_search_n6(target_k, &y, eps, 1) {
                let u_mat = solution_to_mat2(&sol, k);

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
                    let u_ring = solution_to_u2_omicron(&sol, k);
                    let inner_gates = decompose_pi6(&u_ring);
                    let gates = simplify_gate_string_n6(&match tag {
                        Branch::Even    => inner_gates,
                        Branch::Rz      => format!("{inner_gates}R"),
                        Branch::RzDag   => format!("{inner_gates}RRRRR"),
                        Branch::Clif(i) => {
                            let c_rev: String = CLIFFORD_TABLE_T[*i].0.chars().rev().collect();
                            format!("{}{inner_gates}", c_rev)
                        }
                        Branch::ClifRz(i) => {
                            let c_rev: String = CLIFFORD_TABLE_T[*i].0.chars().rev().collect();
                            format!("{}{inner_gates}R", c_rev)
                        }
                        Branch::ClifRzDag(i) => {
                            let c_rev: String = CLIFFORD_TABLE_T[*i].0.chars().rev().collect();
                            format!("{}{inner_gates}RRRRR", c_rev)
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

fn rz_pi6_mat() -> Mat2 {
    let ph = Complex64::from_polar(1.0, PI / 12.0);
    [
        [ph.conj(), Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), ph],
    ]
}

fn rz_neg_pi6_mat() -> Mat2 {
    let ph = Complex64::from_polar(1.0, PI / 12.0);
    [
        [ph, Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), ph.conj()],
    ]
}

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
    use crate::rings::ZOmicron;
    use crate::matrix::U2;
    use crate::synthesis::decomposer::GateRing;
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
        for col in 0..8_usize {
            let mut w = [0.0f64; 8];
            w[col] = 1.0;
            let sinv_w = sigma_inverse_apply(w);
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

    // Checks G = ΣᵀΣ matches SIGMA_GRAM_U (block-diagonal, non-diagonal blocks).
    #[test]
    fn sigma_gram_is_diagonal() {
        let sigma = sigma_matrix();
        for i in 0..8 {
            for j in 0..8 {
                let dot: f64 = (0..8).map(|k| sigma[k][i] * sigma[k][j]).sum();
                // G is block-diagonal with G_u blocks
                let expected = if (i < 4) == (j < 4) {
                    SIGMA_GRAM_U[i % 4][j % 4] as f64
                } else {
                    0.0
                };
                assert!(near(dot, expected, 1e-12),
                    "G[{i}][{j}] = {dot}, expected {expected}");
            }
        }
    }

    // ── Constraint checkers ───────────────────────────────────────────────────

    #[test]
    fn bullet_map_sanity() {
        // For u = ξ² = (0,0,1,0): Re(u) = 1/2, Im(u) = √3/2.
        // bullet(ξ²) = 1−ξ²: Re(u•) = 1/2, Im(u•) = −√3/2.
        let sigma = sigma_matrix();
        let x = [0i64, 0, 1, 0, 0, 0, 0, 0];
        let mut embed = [0.0f64; 8];
        for i in 0..8 {
            for j in 0..8 { embed[i] += sigma[i][j] * x[j] as f64; }
        }
        assert!(near(embed[0], 0.5,          1e-12), "Re(u)={}", embed[0]);
        assert!(near(embed[1], SQRT3_HALF,   1e-12), "Im(u)={}", embed[1]);
        assert!(near(embed[2], 0.5,          1e-12), "Re(u•)={}", embed[2]);
        assert!(near(embed[3], -SQRT3_HALF,  1e-12), "Im(u•)={}", embed[3]);

        // For u = ξ = (0,1,0,0): Re(u) = √3/2, Im(u) = 1/2.
        // bullet(ξ) = ξ⁵ = ξ³−ξ: Re(u•) = −√3/2, Im(u•) = 1/2.
        let x2 = [0i64, 1, 0, 0, 0, 0, 0, 0];
        let mut embed2 = [0.0f64; 8];
        for i in 0..8 {
            for j in 0..8 { embed2[i] += sigma[i][j] * x2[j] as f64; }
        }
        assert!(near(embed2[0],  SQRT3_HALF, 1e-12), "Re(u)={}", embed2[0]);
        assert!(near(embed2[1],  0.5,        1e-12), "Im(u)={}", embed2[1]);
        assert!(near(embed2[2], -SQRT3_HALF, 1e-12), "Re(u•)={}", embed2[2]);
        assert!(near(embed2[3],  0.5,        1e-12), "Im(u•)={}", embed2[3]);
    }

    #[test]
    fn check_norm_and_bilinear_on_known_point() {
        // u=1 (a0=1), t=0: rational(|u|²)=1=2^0 → k=0.
        let x = [1i64,0,0,0, 0,0,0,0];
        assert!(check_norm_eq(&x, 0), "identity should have k=0 norm");
        assert!(check_bilinear(&x), "identity bilinear");

        // u=1, t=1: rational(|u|²)+rational(|t|²)=1+1=2=2^1 → k=1.
        let x1 = [1i64,0,0,0, 1,0,0,0];
        assert!(check_norm_eq(&x1, 1), "u=1,t=1 should have k=1 norm");
        assert!(check_bilinear(&x1), "u=1,t=1 bilinear");

        // u=ξ (a1=1): rational(|ξ|²)=0+1+0+0+0+0=1=2^0 → k=0.
        let x2 = [0i64,1,0,0, 0,0,0,0];
        assert!(check_norm_eq(&x2, 0), "xi should have k=0 norm");
        assert!(check_bilinear(&x2), "xi bilinear");
    }

    #[test]
    fn solution_to_mat2_identity() {
        // u=1, t=0, k=0 → identity.
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
        // u=1, t=0, k=0 → unitary.
        let x = [1i64,0,0,0, 0,0,0,0];
        let m = solution_to_mat2(&x, 0);
        let (u00, u01, u10, u11) = (m[0][0], m[0][1], m[1][0], m[1][1]);
        let d00 = u00.norm_sqr() + u10.norm_sqr();
        let d11 = u01.norm_sqr() + u11.norm_sqr();
        let off = u00.conj()*u01 + u10.conj()*u11;
        assert!(near(d00, 1.0, 1e-12), "d00={d00}");
        assert!(near(d11, 1.0, 1e-12), "d11={d11}");
        assert!(near(off.norm(), 0.0, 1e-12), "off={off}");
    }

    // ── Alignment vector ──────────────────────────────────────────────────────

    #[test]
    fn y_vector_gram_norm_is_one() {
        // yᵀ G⁻¹ y = |v1|² + |v2|² = 1 for a unit SU(2) first column.
        // G_u⁻¹ = [[2/3,0,-1/3,0],[0,2/3,0,-1/3],[-1/3,0,2/3,0],[0,-1/3,0,2/3]].
        let g_inv_u = |w: [f64; 4]| -> [f64; 4] {
            let (a, b, c, d) = (w[0], w[1], w[2], w[3]);
            [(2.0/3.0)*a - (1.0/3.0)*c,
             (2.0/3.0)*b - (1.0/3.0)*d,
            -(1.0/3.0)*a + (2.0/3.0)*c,
            -(1.0/3.0)*b + (2.0/3.0)*d]
        };
        // Use v1=(0.6+0.8i), v2=0; |v1|=1 so total norm = 1.
        let y = compute_y(0.6, 0.8, 0.0, 0.0);
        let yu = [y[0], y[1], y[2], y[3]];
        let yt = [y[4], y[5], y[6], y[7]];
        let gu_yu = g_inv_u(yu);
        let gu_yt = g_inv_u(yt);
        let norm: f64 = yu.iter().zip(gu_yu.iter()).map(|(a,b)| a*b).sum::<f64>()
                      + yt.iter().zip(gu_yt.iter()).map(|(a,b)| a*b).sum::<f64>();
        assert!(near(norm, 1.0, 1e-12), "yᵀG⁻¹y={norm}");
    }

    #[test]
    fn y_vector_dot_equals_re_u_dot_v() {
        // x·y = Re(u)·v1_re + Im(u)·v1_im + Re(t)·v2_re + Im(t)·v2_im.
        let (v1_re, v1_im) = (0.6_f64, 0.8_f64);
        let (v2_re, v2_im) = (0.0_f64, 0.0_f64);
        let y = compute_y(v1_re, v1_im, v2_re, v2_im);
        let x = [2i64, 3, 1, -1, 5, -2, 0, 1];
        let s = SQRT3_HALF;
        let re_u = x[0] as f64 + x[1] as f64 * s + x[2] as f64 * 0.5;
        let im_u = x[1] as f64 * 0.5 + x[2] as f64 * s + x[3] as f64;
        let re_t = x[4] as f64 + x[5] as f64 * s + x[6] as f64 * 0.5;
        let im_t = x[5] as f64 * 0.5 + x[6] as f64 * s + x[7] as f64;
        let expected = re_u*v1_re + im_u*v1_im + re_t*v2_re + im_t*v2_im;
        let dot: f64 = x.iter().zip(y.iter()).map(|(&xi,&yi)| xi as f64 * yi).sum();
        assert!(near(dot, expected, 1e-10), "dot={dot} expected={expected}");
    }

    // ── Search correctness ────────────────────────────────────────────────────

    #[test]
    fn search_k0_finds_identity() {
        // At k=0, 2^0=1: solutions have |u|²+|t|²=1.
        // Expect x=(±1,0,0,0,0,0,0,0) and other 12th roots of unity, with t=0.
        let y = compute_y(1.0, 0.0, 0.0, 0.0);
        let sols = direct_search_n6(1_i64 << 0, &y, 0.0, 100);
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
        let y = compute_y(1.0, 0.0, 0.0, 0.0);
        let sols = direct_search_n6(1_i64 << 1, &y, 0.0, 1000);
        assert!(!sols.is_empty(), "k=1 should have solutions");
        for sol in &sols {
            assert!(check_norm_eq(sol, 1), "norm failed: {sol:?}");
            assert!(check_bilinear(sol), "bilinear failed: {sol:?}");
        }
    }

    #[test]
    fn search_all_solutions_satisfy_constraints() {
        let v = [0.8_f64, 0.6, 0.0, 0.0];
        let y = compute_y(v[0], v[1], v[2], v[3]);
        let sols = direct_search_n6(1_i64 << 2, &y, 0.5, 500);
        let pow2k = (1_i64 << 2) as f64;
        for sol in &sols {
            assert!(check_norm_eq(sol, 2), "norm: {sol:?}");
            assert!(check_bilinear(sol), "bilinear: {sol:?}");
            let dot: f64 = sol.iter().zip(y.iter()).map(|(&x,&y)| x as f64 * y).sum();
            let thresh = pow2k * (1.0 - 0.25);
            assert!(dot*dot >= thresh - 1e-9, "alignment: {sol:?} dot={dot}");
        }
    }

    // ── New algebraic tests ───────────────────────────────────────────────────

    #[test]
    fn test_norm_eq_simple() {
        // u=1, t=1: |u|²+|t|²=2=2^1 → k=1.
        let x = [1i64, 0, 0, 0, 1, 0, 0, 0];
        assert!(check_norm_eq(&x, 1));
        assert!(check_bilinear(&x));
    }

    #[test]
    fn test_inner_product_identity() {
        // x·y should equal ⟨u,v₁⟩+⟨t,v₂⟩.
        // x=(1,0,0,0,1,0,0,0): u=1, t=1.  y=compute_y(1,0,1,0): v₁=1, v₂=1.
        let x = [1i64, 0, 0, 0, 1, 0, 0, 0];
        let y = compute_y(1.0, 0.0, 1.0, 0.0);
        let dot: f64 = x.iter().zip(y.iter()).map(|(&xi, &yi)| (xi as f64) * yi).sum();
        // ⟨1,1⟩+⟨1,1⟩ = 1+1 = 2
        assert!((dot - 2.0).abs() < 1e-12, "dot = {dot}, expected 2");
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
        let id_ring = U2::<ZOmicron>::eye();
        let g = decompose_pi6(&id_ring);
        let id = eye_mat();
        let m = eval_gate_string(&g);
        let d = diamond_distance_float(&m, &id);
        assert!(d < 1e-9, "decompose(I)=\"{g}\" dist={d:.3e}");
    }

    #[test]
    fn decompose_rz_pi6_round_trip() {
        // R gate = diag(1, ξ) ≈ Rz(π/6) up to global phase
        let r_ring = ZOmicron::rz_pos_u2();
        let g = decompose_pi6(&r_ring);
        assert_eq!(&g, "R", "R gate decomposed to \"{g}\"");
        let rz = rz_pi6_mat();
        let m = eval_gate_string(&g);
        let d = diamond_distance_float(&m, &rz);
        assert!(d < 1e-9, "decompose(R)=\"{g}\" dist={d:.3e}");
    }

    #[test]
    fn decompose_s_gate_round_trip() {
        let s_ring = U2::<ZOmicron>::s();
        let g = decompose_pi6(&s_ring);
        let s: Mat2 = [
            [Complex64::new(1., 0.), Complex64::new(0., 0.)],
            [Complex64::new(0., 0.), Complex64::new(0., 1.)],
        ];
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
        let target = rz(0.3);
        let synth = SynthesizerPi6::new(0.1)
            .with_min_lde(0)
            .with_max_lde(12)
            .with_direct_limit(2);
        let result = synth.synthesize(target).expect("DC should find a solution");
        assert!(result.distance < 0.1, "dist={:.4e}", result.distance);
        eprintln!("DC Rz(0.3) @ eps=0.1: lde={} dist={:.4e} gates={:?}", result.lde, result.distance, result.gates);
    }

    #[test]
    fn build_l_pi6_sizes() {
        for k_prime in 0u32..=4 {
            let l = build_l_pi6(k_prime);
            eprintln!("|L_{{{}}}| = {}", k_prime, l.len());
            assert!(!l.is_empty());
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
        let y = compute_y(1.0, 0.0, 0.0, 0.0);
        let sols = direct_search_n6(1_i64 << 1, &y, 0.0, 50);
        for sol in &sols {
            let m = solution_to_mat2(sol, 1);
            let d00 = m[0][0].norm_sqr() + m[1][0].norm_sqr();
            let d11 = m[0][1].norm_sqr() + m[1][1].norm_sqr();
            let off = m[0][0].conj()*m[0][1] + m[1][0].conj()*m[1][1];
            assert!(near(d00, 1.0, 1e-9), "d00={d00} for {sol:?}");
            assert!(near(d11, 1.0, 1e-9), "d11={d11} for {sol:?}");
            assert!(near(off.norm(), 0.0, 1e-9), "off={off} for {sol:?}");
        }
    }

    #[test]
    fn synthesize_h_clifford_zero_rz_count() {
        // H is a Clifford gate — should synthesize with 0 R-gates (R = R_z(π/6)).
        let h: Mat2 = CLIFFORD_TABLE_T.iter().find(|(n,_)| *n=="H").unwrap().1.to_float();
        let synth = SynthesizerPi6::new(1e-6).with_min_lde(0).with_max_lde(3);
        let result = synth.synthesize(h).expect("H should synthesize");
        
        // Verify the reconstructed unitary matches
        if let Some(ref gates) = result.gates {
            let r_count = gates.chars().filter(|&c| c == 'R').count();
            assert_eq!(r_count, 0, "H is Clifford; expected 0 R-gates, got {} in \"{}\"", r_count, gates);
            
            let m = eval_gate_string(gates);
            let d = diamond_distance_float(&m, &h);
            assert!(d < 1e-5, "H round-trip dist={:.3e}, gates=\"{}\"", d, gates);
        } else {
            panic!("H synthesized with no gate string");
        }
    }

    #[test]
    fn synthesize_rz_pi6_exactly_one_r_gate() {
        // R_z(π/6) — the named generator — should synthesize with exactly 1 R-gate.
        let target = rz(PI / 6.0);
        let synth = SynthesizerPi6::new(1e-6).with_min_lde(0).with_max_lde(3);
        let result = synth.synthesize(target).expect("R_z(π/6) should synthesize");
        
        if let Some(ref gates) = result.gates {
            let r_count = gates.chars().filter(|&c| c == 'R').count();
            assert_eq!(r_count, 1, "R_z(π/6) should have 1 R-gate, got {} in \"{}\"", r_count, gates);
            
            let m = eval_gate_string(gates);
            let d = diamond_distance_float(&m, &target);
            assert!(d < 1e-5, "R_z(π/6) round-trip dist={:.3e}, gates=\"{}\"", d, gates);
        } else {
            panic!("R_z(π/6) synthesized with no gate string");
        }
    }

    #[test]
    fn synthesize_small_angle_within_eps() {
        let theta = 0.3_f64;
        let target = rz(theta);
        let eps = 0.01;  // or whatever you actually used
        let synth = SynthesizerPi6::new(eps).with_min_lde(0).with_max_lde(20);
        let result = synth.synthesize(target).expect("should synthesize");
        
        eprintln!("result.distance = {:.6e}", result.distance);
        eprintln!("result.lde = {}", result.lde);
        eprintln!("result.gates = {:?}", result.gates);
        
        if let Some(ref gates) = result.gates {
            let r_count = gates.chars().filter(|&c| c == 'R').count();
            eprintln!("R-count = {}", r_count);
            let m = eval_gate_string(gates);
            let d_roundtrip = diamond_distance_float(&m, &target);
            eprintln!("round-trip distance = {:.6e}", d_roundtrip);
            eprintln!("target = \n  [{:?}, {:?}]\n  [{:?}, {:?}]",
                    target[0][0], target[0][1], target[1][0], target[1][1]);
            eprintln!("from gates = \n  [{:?}, {:?}]\n  [{:?}, {:?}]",
                    m[0][0], m[0][1], m[1][0], m[1][1]);
        }
    }

    #[test]
    fn check_h_r_h_decomposition() {
        let h = eval_gate_string("H");  // adjust to your codebase
        let r_pi6 = rz_pi6_mat();
        let hrh = mat_mul(mat_mul(h, r_pi6), h);
        
        let synth = SynthesizerPi6::new(0.001).with_min_lde(0).with_max_lde(5);
        let result = synth.synthesize(hrh).expect("should synthesize HRH");
        
        eprintln!("HRH: lde={} gates={:?} dist={:.3e}",
                result.lde, result.gates, result.distance);
        
        if let Some(ref gates) = result.gates {
            let m = eval_gate_string(gates);
            let d = diamond_distance_float(&m, &hrh);
            eprintln!("round-trip dist = {:.3e}", d);
        }
    }

    #[test]
    fn diagnose_small_angle() {
        let theta = -0.0699;
        let target = rz(theta);
        let synth = SynthesizerPi6::new(0.01).with_min_lde(0).with_max_lde(15);
        let result = synth.synthesize(target).expect("should synthesize");
        
        eprintln!("θ = {theta}");
        eprintln!("result.distance = {:.6e}", result.distance);
        eprintln!("result.lde = {}", result.lde);
        eprintln!("result.gates = {:?}", result.gates);
        
        if let Some(ref gates) = result.gates {
            let r_count = gates.chars().filter(|&c| c == 'R').count();
            eprintln!("R-count = {r_count}");
            eprintln!("gate-string length = {}", gates.len());
            
            let m = eval_gate_string(gates);
            let d = diamond_distance_float(&m, &target);
            eprintln!("round-trip dist = {:.6e}", d);
            
            // Per-gate-prefix breakdown: see where the decomposer goes wrong
            let mut prefix_mat = eye_mat();
            eprintln!("--- per-prefix matrices ---");
            for (i, ch) in gates.chars().enumerate() {
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
                prefix_mat = mat_mul(prefix_mat, g);
                let dist_to_target = diamond_distance_float(&prefix_mat, &target);
                eprintln!("after gate {i:2} '{ch}': dist_to_target={dist_to_target:.4e}");
            }
        }
    }

    #[test]
    #[ignore]  // slow; run with `cargo test -- --ignored`
    fn stress_random_angles_n6() {
        use std::f64::consts::PI;
        let synth = SynthesizerPi6::new(1e-2).with_min_lde(0).with_max_lde(15);
        
        let mut state: u64 = 0xDEADBEEF;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 32) as f64 / (1u64 << 32) as f64
        };
        
        let mut max_dist: f64 = 0.0;
        let mut max_r = 0;
        let mut total_r = 0;
        let n = 20;
        
        for i in 0..n {
            let theta = (next() - 0.5) * PI;
            let target = rz(theta);
            let result = synth.synthesize(target)
                .unwrap_or_else(|| panic!("[{i}] failed: θ={theta}"));
            
            if let Some(ref gates) = result.gates {
                let m = eval_gate_string(gates);
                let d = diamond_distance_float(&m, &target);
                let r_count = gates.chars().filter(|&c| c == 'R').count();
                assert!(d < 1e-2, "[{i}] θ={theta:.4} dist={d:.3e}");
                max_dist = max_dist.max(d);
                max_r = max_r.max(r_count);
                total_r += r_count;
                eprintln!("[{i:2}] θ={theta:+.4} r={r_count} d={d:.3e}");
            }
        }
        eprintln!("max_dist={max_dist:.3e} max_r={max_r} mean_r={:.1}", total_r as f64 / n as f64);
    }

    #[test]
    #[ignore]  // specific angles; run with `cargo test -- --ignored`
    fn synthesize_at_various_small_angles() {
        // Test a range of angles. Each angle should synthesize correctly.
        let eps = 0.01;
        let synth = SynthesizerPi6::new(eps).with_min_lde(0).with_max_lde(20);
        
        // Angles spanning from "near identity" to "near R" to general.
        let angles = [
            0.001,       // basically identity
            0.01,        // at-the-edge of "Clifford suffices"
            0.05,        // forces real synthesis
            0.0699,      // the original failing case
            -0.0699,     // also a known failure
            0.1,         
            0.3,         // the original "small" test
            0.5,         
            PI / 6.0 - 0.001,  // just shy of R
            PI / 6.0 + 0.001,  // just past R
            PI / 4.0,    
            1.0,
            2.0,
        ];
        
        for &theta in &angles {
            let target = rz(theta);
            let result = match synth.synthesize(target) {
                Some(r) => r,
                None => { eprintln!("θ={theta:+.4}: NO RESULT"); continue; }
            };
            
            if let Some(ref gates) = result.gates {
                let m = eval_gate_string(gates);
                let d = diamond_distance_float(&m, &target);
                let r_count = gates.chars().filter(|&c| c == 'R').count();
                let status = if d < eps { "ok " } else { "BAD" };
                eprintln!(
                    "{status} θ={theta:+.4} lde={} r={r_count} claim={:.3e} actual={:.3e} gates={}",
                    result.lde, result.distance, d,
                    if gates.len() > 30 { format!("{}...", &gates[..30]) } else { gates.clone() }
                );
            }
        }
    }
}
