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
use std::collections::HashMap;
use std::f64::consts::PI;
use std::sync::{Arc, LazyLock, Mutex};

use crate::matrix::U2;
use crate::rings::types::Int;
use crate::rings::ZOmicron;
use crate::synthesis::cliffords::CLIFFORD_TABLE_T;
use crate::synthesis::decomposer::{decompose_so3_canonical_n6, simplify_gate_string_n6};
use crate::synthesis::distance::{diamond_distance_float, Mat2};
use crate::synthesis::search::{apply_u2t_dag_to_uv, normalize4};

// ─── Constants ────────────────────────────────────────────────────────────────

const SQRT3: f64 = 1.7320508075688772935_f64;
const SQRT3_HALF: f64 = 0.86602540378443864676_f64; // √3/2

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
        [1.0, s, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],  // Re u
        [0.0, 0.5, s, 1.0, 0.0, 0.0, 0.0, 0.0],  // Im u
        [1.0, -s, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0], // Re u•
        [0.0, 0.5, -s, 1.0, 0.0, 0.0, 0.0, 0.0], // Im u•
        [0.0, 0.0, 0.0, 0.0, 1.0, s, 0.5, 0.0],  // Re t
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, s, 1.0],  // Im t
        [0.0, 0.0, 0.0, 0.0, 1.0, -s, 0.5, 0.0], // Re t•
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, -s, 1.0], // Im t•
    ]
}

/// Apply Σ⁻¹ = G⁻¹·Σᵀ to an 8-vector, block-diagonally.
///
/// G_u = [[2,0,1,0],[0,2,0,1],[1,0,2,0],[0,1,0,2]] (from SIGMA_GRAM_U).
/// G_u⁻¹ = [[2/3,0,-1/3,0],[0,2/3,0,-1/3],[-1/3,0,2/3,0],[0,-1/3,0,2/3]].
pub fn sigma_inverse_apply(w: [f64; 8]) -> [f64; 8] {
    let sh6 = SQRT3 / 6.0; // √3/6
    let sh3 = SQRT3 / 3.0; // √3/3
                           // G_u⁻¹·Σ_uᵀ: rows are:
                           //   [1/2, -√3/6, 1/2,  √3/6]
                           //   [√3/3,  0,  -√3/3,  0  ]
                           //   [0,   √3/3,  0,   -√3/3]
                           //   [-√3/6, 1/2, √3/6, 1/2 ]
    let sinv_block = |wblk: [f64; 4]| -> [f64; 4] {
        let (w0, w1, w2, w3) = (wblk[0], wblk[1], wblk[2], wblk[3]);
        [
            0.5 * w0 - sh6 * w1 + 0.5 * w2 + sh6 * w3,
            sh3 * w0 - sh3 * w2,
            sh3 * w1 - sh3 * w3,
            -sh6 * w0 + 0.5 * w1 + sh6 * w2 + 0.5 * w3,
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
    if n <= 0 {
        return 0;
    }
    let mut s = (n as f64).sqrt() as i64;
    while s > 0 && s * s > n {
        s -= 1;
    }
    while (s + 1) * (s + 1) <= n {
        s += 1;
    }
    s
}

// ─── Constraint checkers ──────────────────────────────────────────────────────

/// Check norm equation: rational(|u|²) + rational(|t|²) = 2^k.
///
/// rational(|u|²) = a₀²+a₁²+a₂²+a₃² + a₀a₂+a₁a₃  (from complex_norm_sqr).
#[inline]
pub fn check_norm_eq(x: &[i64; 8], k: u32) -> bool {
    let [a0, a1, a2, a3, b0, b1, b2, b3] = *x;
    let euclid = a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3 + b0 * b0 + b1 * b1 + b2 * b2 + b3 * b3;
    let cross = a0 * a2 + a1 * a3 + b0 * b2 + b1 * b3;
    euclid + cross == 1_i64 << k
}

/// Check bilinear constraint: √3-part of |u|²+|t|² vanishes.
///
/// (a₀a₁+a₁a₂+a₂a₃) + (b₀b₁+b₁b₂+b₂b₃) = 0.
#[inline]
pub fn check_bilinear(x: &[i64; 8]) -> bool {
    let [a0, a1, a2, a3, b0, b1, b2, b3] = *x;
    (a0 * a1 + a1 * a2 + a2 * a3) + (b0 * b1 + b1 * b2 + b2 * b3) == 0
}

/// Check alignment: (x·y)² ≥ 2^k·(1−ε²).
#[inline]
pub fn check_alignment(x: &[i64; 8], y: &[f64; 8], k: u32, eps_sq: f64) -> bool {
    let dot: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| xi as f64 * yi)
        .sum();
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
        v[0] * c - v[1] * s,
        v[0] * s + v[1] * c,
        v[2] * c - v[3] * s,
        v[2] * s + v[3] * c,
    ]
}

/// uv direction after right-multiplying target by R_z(-π/6).
pub fn apply_rz_dag_to_uv(v: [f64; 4]) -> [f64; 4] {
    rotate_uv(v, RZ_ANGLE)
}

/// uv direction after right-multiplying target by R_z(+π/6).
pub fn apply_rz_to_uv(v: [f64; 4]) -> [f64; 4] {
    rotate_uv(v, -RZ_ANGLE)
}

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
    let u = Complex64::new(a0 + a1 * s + a2 * 0.5, a1 * 0.5 + a2 * s + a3);
    let t = Complex64::new(b0 + b1 * s + b2 * 0.5, b1 * 0.5 + b2 * s + b3);
    let scale = 1.0 / ((1_i64 << k) as f64).sqrt();
    [
        [u * scale, -t.conj() * scale],
        [t * scale, u.conj() * scale],
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
    if rem < 0 {
        return out;
    }

    // a₃ range: 4·rem − 3·a₃² ≥ 0 ↔ |a₃| ≤ √(4·rem/3)
    let max_a3 = isqrt(4 * rem / 3 + 1);
    for a3 in -max_a3..=max_a3 {
        let disc = 4 * rem - 3 * a3 * a3;
        if disc < 0 {
            continue;
        }
        let sq = isqrt(disc);
        if sq * sq != disc {
            continue;
        }

        for sign in [1i64, -1] {
            if sign == -1 && sq == 0 {
                break;
            }
            let numer = -a3 + sign * sq;
            if numer % 2 != 0 {
                continue;
            }
            let a1 = numer / 2;
            // Verify norm
            if a1 * a1 + a1 * a3 + a3 * a3 != rem {
                continue;
            }
            // Bilinear: a0*a1 + a1*a2 + a2*a3 = -t_bil
            if a0 * a1 + a1 * a2 + a2 * a3 != -t_bil {
                continue;
            }
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
        let dot: f64 = x
            .iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| xi as f64 * yi)
            .sum();
        if dot * dot < thresh_sq {
            return;
        }
    }
    if out.len() < max_sol {
        out.push(x);
    }
}

/// Inner enumeration for one fixed (b₀,b₁,b₂,b₃) t-block.
///
/// Enumerates (a₀, a₂) and solves (a₁, a₃) from:
///   a₁² + a₁a₃ + a₃² = rem_u − (a₀²+a₂²+a₀a₂)
///   a₀a₁ + a₁a₂ + a₂a₃ = −t_bilinear
fn search_inner(
    b0: i64,
    b1: i64,
    b2: i64,
    b3: i64,
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
    let t_dot = b0 as f64 * y[4] + b1 as f64 * y[5] + b2 as f64 * y[6] + b3 as f64 * y[7];

    // Enumerate a0: a0² ≤ rem_u
    let max_a0 = isqrt(rem_u);
    for a0 in -max_a0..=max_a0 {
        let rem_a0 = rem_u - a0 * a0;
        if rem_a0 < 0 {
            continue;
        }

        // Enumerate a2: a2²+a0*a2 ≤ rem_a0 → safe bound |a2| ≤ |a0| + √(rem_a0)
        let max_a2 = isqrt(rem_a0) + a0.abs();
        for a2 in -max_a2..=max_a2 {
            let fixed_u = a0 * a0 + a2 * a2 + a0 * a2;
            let rem_a1a3 = rem_u - fixed_u;
            if rem_a1a3 < 0 {
                continue;
            }

            if do_prune {
                let ac_dot = a0 as f64 * y[0] + a2 as f64 * y[2];
                let partial = t_dot + ac_dot;
                // Max |a1*y[1]+a3*y[3]| ≤ √(2*rem_a1a3) * ‖(y[1],y[3])‖
                let rem_bound = (2.0 * rem_a1a3 as f64).sqrt() * (y[1] * y[1] + y[3] * y[3]).sqrt();
                if partial.abs() + rem_bound < thresh {
                    continue;
                }
            }

            for (a1, a3) in solve_bd(rem_a1a3, a0, a2, t_bilinear) {
                let x = [a0, a1, a2, a3, b0, b1, b2, b3];
                record_if_aligned(x, y, thresh_sq, &mut out, max_sol);
                if out.len() >= max_sol {
                    return out;
                }
            }
        }
    }
    out
}

/// Brute-force Phase-1 search: enumerate all x satisfying norm_eq, bilinear,
/// and alignment.  Fully sequential — the caller (dc_search, direct_search)
/// is already parallelised over prefixes / branches via rayon, so adding a
/// second rayon level here causes severe nested-parallelism contention.
///
/// Enumeration order: b₀, b₂ (hex pair), b₁, b₃ (hex pair), then u-block
/// via search_inner.  Cauchy–Schwarz pruning at the (b₀,b₂) and full
/// t-block levels reduces calls to search_inner substantially.
fn brute_force_direct_search_n6(
    target_k: i64,
    y: &[f64; 8],
    eps: f64,
    max_sol: usize,
) -> Vec<[i64; 8]> {
    if max_sol == 0 {
        return Vec::new();
    }
    let norm2k = target_k;
    let thresh_sq = if eps > 0.0 {
        target_k as f64 * (1.0 - eps * eps)
    } else {
        0.0
    };
    let do_prune = eps > 0.0;
    let thresh = thresh_sq.max(0.0).sqrt();

    // Precompute suffix ‖y‖² after removing each enumerated component so the
    // Cauchy–Schwarz bound can be tightened at every level.
    // Enumeration order for t-block: b₀→y[4], b₂→y[6], b₁→y[5], b₃→y[7].
    let y_sq_all: f64 = y.iter().map(|&yi| yi * yi).sum();
    // remaining y² after fixing (b₀, b₂):
    let y_sq_rem_b02 = y_sq_all - y[4] * y[4] - y[6] * y[6];
    // remaining y² after full t-block (u-block only):
    let y_sq_rem_u = y_sq_rem_b02 - y[5] * y[5] - y[7] * y[7];

    let mut out = Vec::new();

    let max_b0 = isqrt(norm2k);
    for b0 in -max_b0..=max_b0 {
        let p0 = b0 as f64 * y[4];
        // hex_02 = b0²+b2²+b0·b2, bound on b2:
        let max_b2 = isqrt(norm2k) + b0.abs();
        for b2 in -max_b2..=max_b2 {
            let hex_02 = b0 * b0 + b2 * b2 + b0 * b2;
            if hex_02 > norm2k {
                continue;
            }
            let rem02 = norm2k - hex_02;
            let p02 = p0 + b2 as f64 * y[6];
            // CS prune after (b₀, b₂): remaining hex budget = rem02.
            // Euclidean bound: ‖x_rem‖² ≤ 2·rem02 (hex ≥ euc/2).
            if do_prune && p02.abs() + (2.0 * rem02 as f64 * y_sq_rem_b02).sqrt() < thresh {
                continue;
            }

            // b₁, b₃ must satisfy b₁²+b₃²+b₁·b₃ ≤ rem02.
            // Minimum hex_t13 over b3 is 3b₁²/4, so |b₁| ≤ 2√(rem02/3).
            let max_b1 = isqrt(4 * rem02 / 3 + 1);
            for b1 in -max_b1..=max_b1 {
                // (2·b₃+b₁)² ≤ 4·rem02 − 3·b₁²  (tight b₃ bound)
                let disc3 = 4 * rem02 - 3 * b1 * b1;
                if disc3 < 0 {
                    continue;
                }
                let half_sq = isqrt(disc3);
                let b3_lo = (-half_sq - b1) / 2 - 1;
                let b3_hi = (half_sq - b1) / 2 + 1;
                let p1 = p02 + b1 as f64 * y[5];
                for b3 in b3_lo..=b3_hi {
                    let hex_t = b0 * b0 + b1 * b1 + b2 * b2 + b3 * b3 + b0 * b2 + b1 * b3;
                    if hex_t < 0 || hex_t > norm2k {
                        continue;
                    }
                    let rem_u = norm2k - hex_t;
                    let p_t = p1 + b3 as f64 * y[7];
                    // CS prune after full t-block: only u-block remains.
                    if do_prune && p_t.abs() + (2.0 * rem_u as f64 * y_sq_rem_u).sqrt() < thresh {
                        continue;
                    }
                    let t_bilinear = b0 * b1 + b1 * b2 + b2 * b3;
                    for sol in search_inner(
                        b0, b1, b2, b3, rem_u, t_bilinear, y, thresh_sq, thresh, do_prune, max_sol,
                    ) {
                        out.push(sol);
                        if out.len() >= max_sol {
                            return out;
                        }
                    }
                }
            }
        }
    }
    out
}

/// Threshold k below which brute-force is used instead of the lattice pipeline.
pub const LATTICE_K_MIN: u32 = 7;
const DEFAULT_DC_INNER_K: u32 = 12;
const TIGHT_DC_INNER_K: u32 = 18;

fn default_dc_inner_k(epsilon: f64) -> u32 {
    if epsilon <= 1e-4 {
        TIGHT_DC_INNER_K
    } else {
        DEFAULT_DC_INNER_K
    }
}

/// Phase-1 search: enumerate x with norm_eq=true, bilinear=true, alignment ok.
///
/// target_k = 2^k.  Uses brute-force for small k (where the LLL ellipsoid is
/// too tight) and the lattice LLL+SE pipeline for large k (where brute-force
/// is exponentially slow). The lattice path expects the unscaled alignment
/// vector `y = Σ_top^T v`.
pub fn direct_search_n6(target_k: i64, y: &[f64; 8], eps: f64, max_sol: usize) -> Vec<[i64; 8]> {
    if max_sol == 0 {
        return Vec::new();
    }

    let k = (target_k as u64).trailing_zeros();

    // For small k use brute-force (faster than lattice and avoids MPFR overhead).
    if k < LATTICE_K_MIN {
        return brute_force_direct_search_n6(target_k, y, eps, max_sol);
    }

    // Large k: lattice LLL+SE pipeline.
    use crate::synthesis::lattice_omicron;
    use std::sync::atomic::AtomicBool;
    let mut scratch = lattice_omicron::LatticeScratch::new(eps);
    let budget_hit = AtomicBool::new(false);
    lattice_omicron::phase1(&mut scratch, y, k, eps, 100_000, &budget_hit)
}

// ─── SO3 float utilities ──────────────────────────────────────────────────────

type SO3f = [[f64; 3]; 3];

/// Compute the adjoint SO(3) representation of a 2×2 unitary (float).
fn mat_to_so3(u: &Mat2) -> SO3f {
    let zero = Complex64::new(0.0, 0.0);
    let paulis: [Mat2; 3] = [
        [
            [zero, Complex64::new(1., 0.)],
            [Complex64::new(1., 0.), zero],
        ],
        [
            [zero, Complex64::new(0., -1.)],
            [Complex64::new(0., 1.), zero],
        ],
        [
            [Complex64::new(1., 0.), zero],
            [zero, Complex64::new(-1., 0.)],
        ],
    ];
    let ud = [
        [u[0][0].conj(), u[1][0].conj()],
        [u[0][1].conj(), u[1][1].conj()],
    ];
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
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                out[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    out
}

#[inline]
fn so3_rz(t: f64) -> SO3f {
    let (c, s) = (t.cos(), t.sin());
    [[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]
}
#[inline]
fn so3_rx(t: f64) -> SO3f {
    let (c, s) = (t.cos(), t.sin());
    [[1., 0., 0.], [0., c, -s], [0., s, c]]
}
#[inline]
fn so3_ry(t: f64) -> SO3f {
    let (c, s) = (t.cos(), t.sin());
    [[c, 0., s], [0., 1., 0.], [-s, 0., c]]
}

/// How far an SO3 matrix is from being a Clifford (entries in {−1,0,1}).
#[inline]
fn clifford_dist(m: &SO3f) -> f64 {
    m.iter()
        .flat_map(|r| r.iter())
        .map(|&v| {
            let n = v.round().clamp(-1., 1.);
            (v - n).abs()
        })
        .sum()
}

/// Precomputed SO3 representations of the 24 Cliffords.
static CLIFFORD_SO3: LazyLock<Vec<(SO3f, &'static str)>> = LazyLock::new(|| {
    CLIFFORD_TABLE_T
        .iter()
        .map(|(name, u2t)| (mat_to_so3(&u2t.to_float()), *name))
        .collect()
});

/// Identify the Clifford gate nearest to the given SO3 matrix.
fn identify_clifford_so3(m: &SO3f) -> &'static str {
    CLIFFORD_SO3
        .iter()
        .map(|(cs, name)| {
            let d: f64 = (0..3)
                .flat_map(|i| (0..3).map(move |j| (m[i][j] - cs[i][j]).powi(2)))
                .sum();
            (d, *name)
        })
        .min_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap())
        .map(|(_, n)| n)
        .unwrap_or("I")
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
        Int::from_i64(sol[0]),
        Int::from_i64(sol[1]),
        Int::from_i64(sol[2]),
        Int::from_i64(sol[3]),
    );
    let t = ZOmicron::new(
        Int::from_i64(sol[4]),
        Int::from_i64(sol[5]),
        Int::from_i64(sol[6]),
        Int::from_i64(sol[7]),
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
    let (idx, _) = flat
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.norm_sqr().partial_cmp(&b.norm_sqr()).unwrap())
        .unwrap();
    let piv = flat[idx];
    let rot: Vec<f64> = if piv.norm() < 1e-12 {
        flat.iter().flat_map(|c| [c.re, c.im]).collect()
    } else {
        let phase = piv / piv.norm();
        flat.iter()
            .flat_map(|c| {
                let r = c / phase;
                [r.re, r.im]
            })
            .collect()
    };
    rot.iter()
        .map(|x| (x * 1_000_000.0).round() as i64)
        .collect::<Vec<_>>()
        .try_into()
        .unwrap()
}

fn eye_mat() -> Mat2 {
    let one = Complex64::new(1.0, 0.0);
    let z = Complex64::new(0.0, 0.0);
    [[one, z], [z, one]]
}

fn mat_dag(m: &Mat2) -> Mat2 {
    [
        [m[0][0].conj(), m[1][0].conj()],
        [m[0][1].conj(), m[1][1].conj()],
    ]
}

/// Build the n=6 MA-like prefix set L_{k'} as (gate_string, float_Mat2) pairs.
fn build_l_pi6(k_prime: u32) -> Arc<Vec<(String, Mat2)>> {
    static CACHE: LazyLock<Mutex<HashMap<u32, Arc<Vec<(String, Mat2)>>>>> =
        LazyLock::new(|| Mutex::new(HashMap::new()));

    {
        let cache = CACHE.lock().unwrap();
        if let Some(v) = cache.get(&k_prime) {
            return Arc::clone(v);
        }
    }

    let result = Arc::new(build_l_pi6_inner(k_prime));
    CACHE.lock().unwrap().insert(k_prime, Arc::clone(&result));
    result
}

fn build_l_pi6_inner(k_prime: u32) -> Vec<(String, Mat2)> {
    if k_prime == 0 {
        return vec![("".to_string(), eye_mat())];
    }
    let h_mat = CLIFFORD_TABLE_T
        .iter()
        .find(|(n, _)| *n == "H")
        .unwrap()
        .1
        .to_float();
    let s_mat = CLIFFORD_TABLE_T
        .iter()
        .find(|(n, _)| *n == "S")
        .unwrap()
        .1
        .to_float();
    let rz_mat = rz_pi6_mat();

    let hs0r = mat_mul(h_mat, rz_mat);
    let hs1r = mat_mul(mat_mul(h_mat, s_mat), rz_mat);

    let mut candidates: Vec<(String, Mat2)> = Vec::new();
    let n_even = 1u32 << k_prime;
    for bits in 0..n_even {
        let mut m = eye_mat();
        let mut g = String::new();
        for i in 0..k_prime {
            if (bits >> i) & 1 == 1 {
                m = mat_mul(m, hs1r);
                g.push_str("HSR");
            } else {
                m = mat_mul(m, hs0r);
                g.push_str("HR");
            }
        }
        for (c_str, c_u2t) in CLIFFORD_TABLE_T {
            push_prefix_candidate(&mut candidates, &g, c_str, mat_mul(m, c_u2t.to_float()));
        }
    }
    if k_prime >= 1 {
        let n_odd = 1u32 << (k_prime - 1);
        for bits in 0..n_odd {
            let mut m = rz_mat;
            let mut g = "R".to_string();
            for i in 0..(k_prime - 1) {
                if (bits >> i) & 1 == 1 {
                    m = mat_mul(m, hs1r);
                    g.push_str("HSR");
                } else {
                    m = mat_mul(m, hs0r);
                    g.push_str("HR");
                }
            }
            for (c_str, c_u2t) in CLIFFORD_TABLE_T {
                push_prefix_candidate(&mut candidates, &g, c_str, mat_mul(m, c_u2t.to_float()));
            }
        }
    }
    let mut seen: std::collections::HashSet<[i64; 8]> = std::collections::HashSet::new();
    candidates
        .into_iter()
        .filter(|(_, m)| seen.insert(canonical_key_f64(m)))
        .collect()
}

fn push_prefix_candidate(
    candidates: &mut Vec<(String, Mat2)>,
    body: &str,
    clifford: &str,
    m: Mat2,
) {
    let c_rev: String = clifford.chars().rev().collect();
    let forms = [
        format!("{body}{clifford}"),
        format!("{body}{c_rev}"),
        format!("{clifford}{body}"),
        format!("{c_rev}{body}"),
    ];
    for gates in forms {
        let gates = simplify_gate_string_n6(&gates);
        if diamond_distance_float(&eval_gate_string_n6_float(&gates), &m) < 1e-9 {
            candidates.push((gates, m));
            return;
        }
    }
    candidates.push((format!("{body}{clifford}"), m));
}

// ─── Result type ─────────────────────────────────────────────────────────────

/// Result of a successful n=6 synthesis.
pub struct SynthResultPi6 {
    pub gates: Option<String>,
    pub lde: u32,
    pub distance: f64,
    /// Raw 8D lattice vector `(u, t)` selected by the successful search.
    pub x: Option<[i64; 8]>,
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
        Self {
            epsilon,
            max_lde,
            min_lde,
            // Keep the D&C inner problem on the validated ZOmicron lattice
            // path.  The old value 6 landed below LATTICE_K_MIN, so every
            // prefix paid the brute-force inner search cost.
            direct_limit: default_dc_inner_k(epsilon),
        }
    }

    pub fn with_max_lde(mut self, v: u32) -> Self {
        self.max_lde = v;
        self
    }
    pub fn with_min_lde(mut self, v: u32) -> Self {
        self.min_lde = v;
        self
    }
    pub fn with_direct_limit(mut self, v: u32) -> Self {
        self.direct_limit = v;
        self
    }

    /// Synthesize a Clifford+R_z(π/6) circuit approximating `target`.
    pub fn synthesize(&self, target: Mat2) -> Option<SynthResultPi6> {
        let raw = unitary_to_uv_n6(&target);
        let v = normalize4(raw).unwrap_or([1.0, 0.0, 0.0, 0.0]);
        let trace = crate::synthesis::diag::trace_enabled();

        let start = std::time::Instant::now();
        for k in self.min_lde..=self.max_lde {
            // Label reflects actual inner path: dc_search always uses k_inner=direct_limit,
            // so the lattice threshold is evaluated against direct_limit, not k.
            let inner_k = if k <= self.direct_limit {
                k
            } else {
                self.direct_limit
            };
            let path = if inner_k < LATTICE_K_MIN {
                "brute"
            } else {
                "lattice"
            };
            let result = if k <= self.direct_limit {
                self.direct_search(&target, v, k)
            } else {
                let k_prefix = k - self.direct_limit;
                self.dc_search(&target, v, k, k_prefix)
            };
            if trace {
                eprintln!(
                    "attempting k={k} via {path} (inner_k={inner_k}, {} ms elapsed)",
                    start.elapsed().as_millis()
                );
            }
            if result.is_some() {
                return result;
            }
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
                    let inner = decompose_pi6(&u_r_ring);
                    for gates in dc_gate_candidates(prefix_gates, &inner, "") {
                        let actual =
                            diamond_distance_float(&eval_gate_string_n6_float(&gates), target);
                        if actual < eps {
                            return Some(SynthResultPi6 {
                                gates: Some(gates),
                                lde: k,
                                distance: actual,
                                x: Some(sol),
                            });
                        }
                    }
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
                        let inner = decompose_pi6(&u_r_ring);
                        for gates in dc_gate_candidates(prefix_gates, &inner, "R") {
                            let actual =
                                diamond_distance_float(&eval_gate_string_n6_float(&gates), target);
                            if actual < eps {
                                return Some(SynthResultPi6 {
                                    gates: Some(gates),
                                    lde: k,
                                    distance: actual,
                                    x: Some(sol),
                                });
                            }
                        }
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

        let clif_vs: Vec<[f64; 4]> = CLIFFORD_TABLE_T
            .iter()
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
                let u_ring = solution_to_u2_omicron(&sol, k);

                let full_mat = match tag {
                    Branch::Even => u_mat,
                    Branch::Rz => mat_mul(u_mat, rz_pi6_mat()),
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
                    let inner_gates = decompose_pi6(&u_ring);
                    for gates in direct_branch_gate_candidates(&inner_gates, tag) {
                        let actual =
                            diamond_distance_float(&eval_gate_string_n6_float(&gates), target);
                        if actual < eps {
                            return Some(SynthResultPi6 {
                                gates: Some(gates),
                                lde: k,
                                distance: actual,
                                x: Some(sol),
                            });
                        }
                    }
                }
            }
            None
        })
    }
}

fn dc_gate_candidates(prefix_gates: &str, inner_gates: &str, suffix: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut push = |s: String| {
        let s = simplify_gate_string_n6(&s);
        if !out.iter().any(|existing| existing == &s) {
            out.push(s);
        }
    };
    push(format!("{prefix_gates}{inner_gates}{suffix}"));
    push(format!("{prefix_gates}{suffix}{inner_gates}"));
    push(format!("{inner_gates}{suffix}{prefix_gates}"));
    push(format!("{suffix}{inner_gates}{prefix_gates}"));
    out
}

fn direct_branch_gate_candidates(inner_gates: &str, tag: &Branch) -> Vec<String> {
    let mut out = Vec::new();
    let mut push = |s: String| {
        let s = simplify_gate_string_n6(&s);
        if !out.iter().any(|existing| existing == &s) {
            out.push(s);
        }
    };
    let mut push_magic = |magic: &str| {
        push(format!("{inner_gates}{magic}"));
        push(format!("{magic}{inner_gates}"));
    };

    match tag {
        Branch::Even => push(inner_gates.to_string()),
        Branch::Rz => push_magic("R"),
        Branch::RzDag => push_magic("RRRRR"),
        Branch::Clif(i) => {
            let c = CLIFFORD_TABLE_T[*i].0;
            let c_rev: String = c.chars().rev().collect();
            for c_str in [c, c_rev.as_str()] {
                push(format!("{inner_gates}{c_str}"));
                push(format!("{c_str}{inner_gates}"));
            }
        }
        Branch::ClifRz(i) => {
            let c = CLIFFORD_TABLE_T[*i].0;
            let c_rev: String = c.chars().rev().collect();
            push_clifford_magic_permutations(&mut push, inner_gates, "R", c);
            push_clifford_magic_permutations(&mut push, inner_gates, "R", &c_rev);
        }
        Branch::ClifRzDag(i) => {
            let c = CLIFFORD_TABLE_T[*i].0;
            let c_rev: String = c.chars().rev().collect();
            push_clifford_magic_permutations(&mut push, inner_gates, "RRRRR", c);
            push_clifford_magic_permutations(&mut push, inner_gates, "RRRRR", &c_rev);
        }
    }
    out
}

fn push_clifford_magic_permutations(
    push: &mut impl FnMut(String),
    inner_gates: &str,
    magic: &str,
    c: &str,
) {
    push(format!("{c}{inner_gates}{magic}"));
    push(format!("{c}{magic}{inner_gates}"));
    push(format!("{inner_gates}{c}{magic}"));
    push(format!("{inner_gates}{magic}{c}"));
    push(format!("{magic}{c}{inner_gates}"));
    push(format!("{magic}{inner_gates}{c}"));
}

fn eval_gate_string_n6_float(gates: &str) -> Mat2 {
    let mut m = eye_mat();
    for ch in gates.chars() {
        let g = match ch {
            'H' => CLIFFORD_TABLE_T
                .iter()
                .find(|(n, _)| *n == "H")
                .unwrap()
                .1
                .to_float(),
            'S' => CLIFFORD_TABLE_T
                .iter()
                .find(|(n, _)| *n == "S")
                .unwrap()
                .1
                .to_float(),
            'X' => CLIFFORD_TABLE_T
                .iter()
                .find(|(n, _)| *n == "X")
                .unwrap()
                .1
                .to_float(),
            'Y' => CLIFFORD_TABLE_T
                .iter()
                .find(|(n, _)| *n == "Y")
                .unwrap()
                .1
                .to_float(),
            'Z' => CLIFFORD_TABLE_T
                .iter()
                .find(|(n, _)| *n == "Z")
                .unwrap()
                .1
                .to_float(),
            'R' => rz_pi6_mat(),
            'I' => eye_mat(),
            other => panic!("unknown gate '{other}'"),
        };
        m = mat_mul(m, g);
    }
    m
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
            a[0][0] * b[0][0] + a[0][1] * b[1][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1],
        ],
        [
            a[1][0] * b[0][0] + a[1][1] * b[1][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1],
        ],
    ]
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::{U2, U2Q, U2T};
    use crate::rings::zomicron::SIGMA_GRAM_U;
    use crate::rings::ZOmicron;
    use crate::synthesis::clifford_sqrt_t::SynthesizerQ;
    use crate::synthesis::clifford_t::SynthesizerT;
    use crate::synthesis::decomposer::GateRing;
    use crate::synthesis::distance::diamond_distance_u2q_float;
    use std::f64::consts::PI;

    fn near(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn rz(theta: f64) -> Mat2 {
        [
            [
                Complex64::from_polar(1.0, -theta / 2.0),
                Complex64::new(0.0, 0.0),
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::from_polar(1.0, theta / 2.0),
            ],
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
                    result[i],
                    expected
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
                assert!(
                    near(dot, expected, 1e-12),
                    "G[{i}][{j}] = {dot}, expected {expected}"
                );
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
            for j in 0..8 {
                embed[i] += sigma[i][j] * x[j] as f64;
            }
        }
        assert!(near(embed[0], 0.5, 1e-12), "Re(u)={}", embed[0]);
        assert!(near(embed[1], SQRT3_HALF, 1e-12), "Im(u)={}", embed[1]);
        assert!(near(embed[2], 0.5, 1e-12), "Re(u•)={}", embed[2]);
        assert!(near(embed[3], -SQRT3_HALF, 1e-12), "Im(u•)={}", embed[3]);

        // For u = ξ = (0,1,0,0): Re(u) = √3/2, Im(u) = 1/2.
        // bullet(ξ) = ξ⁵ = ξ³−ξ: Re(u•) = −√3/2, Im(u•) = 1/2.
        let x2 = [0i64, 1, 0, 0, 0, 0, 0, 0];
        let mut embed2 = [0.0f64; 8];
        for i in 0..8 {
            for j in 0..8 {
                embed2[i] += sigma[i][j] * x2[j] as f64;
            }
        }
        assert!(near(embed2[0], SQRT3_HALF, 1e-12), "Re(u)={}", embed2[0]);
        assert!(near(embed2[1], 0.5, 1e-12), "Im(u)={}", embed2[1]);
        assert!(near(embed2[2], -SQRT3_HALF, 1e-12), "Re(u•)={}", embed2[2]);
        assert!(near(embed2[3], 0.5, 1e-12), "Im(u•)={}", embed2[3]);
    }

    #[test]
    fn check_norm_and_bilinear_on_known_point() {
        // u=1 (a0=1), t=0: rational(|u|²)=1=2^0 → k=0.
        let x = [1i64, 0, 0, 0, 0, 0, 0, 0];
        assert!(check_norm_eq(&x, 0), "identity should have k=0 norm");
        assert!(check_bilinear(&x), "identity bilinear");

        // u=1, t=1: rational(|u|²)+rational(|t|²)=1+1=2=2^1 → k=1.
        let x1 = [1i64, 0, 0, 0, 1, 0, 0, 0];
        assert!(check_norm_eq(&x1, 1), "u=1,t=1 should have k=1 norm");
        assert!(check_bilinear(&x1), "u=1,t=1 bilinear");

        // u=ξ (a1=1): rational(|ξ|²)=0+1+0+0+0+0=1=2^0 → k=0.
        let x2 = [0i64, 1, 0, 0, 0, 0, 0, 0];
        assert!(check_norm_eq(&x2, 0), "xi should have k=0 norm");
        assert!(check_bilinear(&x2), "xi bilinear");
    }

    #[test]
    fn solution_to_mat2_identity() {
        // u=1, t=0, k=0 → identity.
        let x = [1i64, 0, 0, 0, 0, 0, 0, 0];
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
        let x = [1i64, 0, 0, 0, 0, 0, 0, 0];
        let m = solution_to_mat2(&x, 0);
        let (u00, u01, u10, u11) = (m[0][0], m[0][1], m[1][0], m[1][1]);
        let d00 = u00.norm_sqr() + u10.norm_sqr();
        let d11 = u01.norm_sqr() + u11.norm_sqr();
        let off = u00.conj() * u01 + u10.conj() * u11;
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
            [
                (2.0 / 3.0) * a - (1.0 / 3.0) * c,
                (2.0 / 3.0) * b - (1.0 / 3.0) * d,
                -(1.0 / 3.0) * a + (2.0 / 3.0) * c,
                -(1.0 / 3.0) * b + (2.0 / 3.0) * d,
            ]
        };
        // Use v1=(0.6+0.8i), v2=0; |v1|=1 so total norm = 1.
        let y = compute_y(0.6, 0.8, 0.0, 0.0);
        let yu = [y[0], y[1], y[2], y[3]];
        let yt = [y[4], y[5], y[6], y[7]];
        let gu_yu = g_inv_u(yu);
        let gu_yt = g_inv_u(yt);
        let norm: f64 = yu.iter().zip(gu_yu.iter()).map(|(a, b)| a * b).sum::<f64>()
            + yt.iter().zip(gu_yt.iter()).map(|(a, b)| a * b).sum::<f64>();
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
        let expected = re_u * v1_re + im_u * v1_im + re_t * v2_re + im_t * v2_im;
        let dot: f64 = x
            .iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| xi as f64 * yi)
            .sum();
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
        let found_id = sols
            .iter()
            .any(|s| *s == [1, 0, 0, 0, 0, 0, 0, 0] || *s == [-1, 0, 0, 0, 0, 0, 0, 0]);
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
            let dot: f64 = sol.iter().zip(y.iter()).map(|(&x, &y)| x as f64 * y).sum();
            let thresh = pow2k * (1.0 - 0.25);
            assert!(dot * dot >= thresh - 1e-9, "alignment: {sol:?} dot={dot}");
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
        let dot: f64 = x
            .iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| (xi as f64) * yi)
            .sum();
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
        let synth = SynthesizerPi6::new(0.05).with_max_lde(8);
        let result = synth.synthesize(target).expect("should synthesize Rz(0.3)");
        assert!(result.distance < 0.05, "distance={:.4e}", result.distance);
        eprintln!(
            "Rz(0.3) @ eps=0.05: lde={} dist={:.4e}",
            result.lde, result.distance
        );
    }

    // ── SO3 and decomposer ────────────────────────────────────────────────────

    #[test]
    fn so3_of_identity_is_identity() {
        let id = eye_mat();
        let m = mat_to_so3(&id);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    near(m[i][j], expected, 1e-12),
                    "SO3(I)[{i}][{j}]={}",
                    m[i][j]
                );
            }
        }
    }

    #[test]
    fn so3_of_rz_pi6_matches_formula() {
        let rz = rz_pi6_mat();
        let m = mat_to_so3(&rz);
        let (c, s) = ((PI / 6.).cos(), (PI / 6.).sin());
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
            eprintln!(
                "Rz(π/3): lde={} gates=\"{g}\" dist={:.4e}",
                result.lde, result.distance
            );
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
        eprintln!(
            "DC Rz(0.3) @ eps=0.1: lde={} dist={:.4e} gates={:?}",
            result.lde, result.distance, result.gates
        );
    }

    #[test]
    fn build_l_pi6_sizes() {
        for k_prime in 0u32..=4 {
            let l = build_l_pi6(k_prime);
            eprintln!("|L_{{{}}}| = {}", k_prime, l.len());
            assert!(!l.is_empty());
            if k_prime == 0 {
                assert_eq!(l.len(), 1);
            } else {
                assert!(l.len() >= 24);
            }
        }
    }

    // ── Helper: evaluate a gate string to a float Mat2 ────────────────────────

    fn eval_gate_string(gates: &str) -> Mat2 {
        let mut m = eye_mat();
        for ch in gates.chars() {
            let g: Mat2 = match ch {
                'H' => CLIFFORD_TABLE_T
                    .iter()
                    .find(|(n, _)| *n == "H")
                    .unwrap()
                    .1
                    .to_float(),
                'S' => CLIFFORD_TABLE_T
                    .iter()
                    .find(|(n, _)| *n == "S")
                    .unwrap()
                    .1
                    .to_float(),
                'X' => CLIFFORD_TABLE_T
                    .iter()
                    .find(|(n, _)| *n == "X")
                    .unwrap()
                    .1
                    .to_float(),
                'Y' => CLIFFORD_TABLE_T
                    .iter()
                    .find(|(n, _)| *n == "Y")
                    .unwrap()
                    .1
                    .to_float(),
                'Z' => CLIFFORD_TABLE_T
                    .iter()
                    .find(|(n, _)| *n == "Z")
                    .unwrap()
                    .1
                    .to_float(),
                'R' => rz_pi6_mat(),
                'I' => eye_mat(),
                other => panic!("unknown gate '{other}'"),
            };
            m = mat_mul(m, g);
        }
        m
    }

    fn eval_gate_string_t(gates: &str) -> U2T {
        let mut m = U2T::eye();
        for ch in gates.chars() {
            let g = match ch {
                'H' => U2T::h(),
                'S' => U2T::s(),
                'T' => U2T::t(),
                'X' => U2T::x(),
                'Y' => U2T::y(),
                'Z' => U2T::z(),
                'I' => U2T::eye(),
                other => panic!("unknown Clifford+T gate '{other}'"),
            };
            m = m * g;
        }
        m
    }

    fn eval_gate_string_q(gates: &str) -> U2Q {
        let mut m = U2Q::eye();
        for ch in gates.chars() {
            let g = match ch {
                'H' => U2Q::h(),
                'S' => U2Q::s(),
                'T' => U2Q::t(),
                'Q' => U2Q::q(),
                'X' => U2Q::x(),
                'Y' => U2Q::y(),
                'Z' => U2Q::z(),
                'I' => U2Q::eye(),
                other => panic!("unknown Clifford+sqrt(T) gate '{other}'"),
            };
            m = m * g;
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
            let off = m[0][0].conj() * m[0][1] + m[1][0].conj() * m[1][1];
            assert!(near(d00, 1.0, 1e-9), "d00={d00} for {sol:?}");
            assert!(near(d11, 1.0, 1e-9), "d11={d11} for {sol:?}");
            assert!(near(off.norm(), 0.0, 1e-9), "off={off} for {sol:?}");
        }
    }

    #[test]
    fn synthesize_h_clifford_zero_rz_count() {
        // H is a Clifford gate — should synthesize with 0 R-gates (R = R_z(π/6)).
        let h: Mat2 = CLIFFORD_TABLE_T
            .iter()
            .find(|(n, _)| *n == "H")
            .unwrap()
            .1
            .to_float();
        let synth = SynthesizerPi6::new(1e-6).with_min_lde(0).with_max_lde(3);
        let result = synth.synthesize(h).expect("H should synthesize");

        // Verify the reconstructed unitary matches
        if let Some(ref gates) = result.gates {
            let r_count = gates.chars().filter(|&c| c == 'R').count();
            assert_eq!(
                r_count, 0,
                "H is Clifford; expected 0 R-gates, got {} in \"{}\"",
                r_count, gates
            );

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
        let result = synth
            .synthesize(target)
            .expect("R_z(π/6) should synthesize");

        if let Some(ref gates) = result.gates {
            let r_count = gates.chars().filter(|&c| c == 'R').count();
            assert_eq!(
                r_count, 1,
                "R_z(π/6) should have 1 R-gate, got {} in \"{}\"",
                r_count, gates
            );

            let m = eval_gate_string(gates);
            let d = diamond_distance_float(&m, &target);
            assert!(
                d < 1e-5,
                "R_z(π/6) round-trip dist={:.3e}, gates=\"{}\"",
                d,
                gates
            );
        } else {
            panic!("R_z(π/6) synthesized with no gate string");
        }
    }

    #[test]
    fn synthesize_small_angle_within_eps() {
        let theta = 0.3_f64;
        let target = rz(theta);
        let eps = 0.001; // or whatever you actually used
        let synth = SynthesizerPi6::new(eps).with_max_lde(20);
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
            eprintln!(
                "target = \n  [{:?}, {:?}]\n  [{:?}, {:?}]",
                target[0][0], target[0][1], target[1][0], target[1][1]
            );
            eprintln!(
                "from gates = \n  [{:?}, {:?}]\n  [{:?}, {:?}]",
                m[0][0], m[0][1], m[1][0], m[1][1]
            );
        }
    }

    /// Synthesize a deterministic Haar-style random SU(2) unitary at ε=1e-3.
    /// The fixed seed keeps the regression reproducible while still exercising
    /// a non-axis-aligned target.
    #[test]
    #[ignore = "slow n=6 random-unitary synthesis regression at eps=1e-3 - 1e-5"]
    fn synthesize_random_unitary() {
        use rand::{rngs::StdRng, Rng, SeedableRng};

        let mut rng = StdRng::seed_from_u64(0xC0DEC0DE);
        let eps = 1e-6_f64;

        let theta = rng.random::<f64>() * (2.0 * PI);
        let phi = rng.random::<f64>() * (2.0 * PI);
        let lambda = rng.random::<f64>() * (2.0 * PI);
        let ct = (theta / 2.0).cos();
        let st = (theta / 2.0).sin();

        // U3(θ, φ, λ), normalized by a global phase so det(target)=1.
        let global_phase = Complex64::from_polar(1.0, -(phi + lambda) / 2.0);
        let target: Mat2 = [
            [
                global_phase * Complex64::new(ct, 0.0),
                global_phase * (-Complex64::from_polar(st, lambda)),
            ],
            [
                global_phase * Complex64::from_polar(st, phi),
                global_phase * Complex64::from_polar(ct, phi + lambda),
            ],
        ];

        let t0 = std::time::Instant::now();
        let result_t = SynthesizerT::new(eps)
            .synthesize(target)
            .expect("n=4 should synthesize deterministic random unitary at eps=1e-3");
        let elapsed_t = t0.elapsed();
        let gates_t = result_t
            .gates
            .as_ref()
            .expect("n=4 synthesized without a gate string");
        let recovered_t = eval_gate_string_t(gates_t);
        let roundtrip_t = diamond_distance_float(&recovered_t.to_float(), &target);
        eprintln!(
            "n=4 eps={eps:.1e}: elapsed={}ms lde={} claimed={:.6e} actual={:.6e} gates_len={} x8={:?}",
            elapsed_t.as_millis(),
            result_t.lde,
            result_t.distance,
            roundtrip_t,
            gates_t.len(),
            result_t.x
        );
        assert!(
            result_t.distance < eps && roundtrip_t < eps,
            "n=4 failed: claimed={:.6e}, actual={:.6e}, epsilon={:.6e}, lde={}, gates={}",
            result_t.distance,
            roundtrip_t,
            eps,
            result_t.lde,
            gates_t
        );

        let t0 = std::time::Instant::now();
        let result_pi6 = SynthesizerPi6::new(eps)
            .synthesize(target)
            .expect("n=6 should synthesize deterministic random unitary at eps=1e-3");
        let elapsed_pi6 = t0.elapsed();
        let gates_pi6 = result_pi6
            .gates
            .as_ref()
            .expect("n=6 synthesized without a gate string");
        let recovered_pi6 = eval_gate_string(gates_pi6);
        let roundtrip_pi6 = diamond_distance_float(&recovered_pi6, &target);
        eprintln!(
            "n=6 eps={eps:.1e}: elapsed={}ms lde={} claimed={:.6e} actual={:.6e} gates_len={} x8={:?}",
            elapsed_pi6.as_millis(),
            result_pi6.lde,
            result_pi6.distance,
            roundtrip_pi6,
            gates_pi6.len(),
            result_pi6.x
        );
        assert!(
            result_pi6.distance < eps && roundtrip_pi6 < eps,
            "n=6 failed: claimed={:.6e}, actual={:.6e}, epsilon={:.6e}, lde={}, gates={}",
            result_pi6.distance,
            roundtrip_pi6,
            eps,
            result_pi6.lde,
            gates_pi6
        );

        let t0 = std::time::Instant::now();
        let result_q = SynthesizerQ::new(eps)
            .synthesize(target)
            .expect("n=8 should synthesize deterministic random unitary at eps=1e-3");
        let elapsed_q = t0.elapsed();
        let gates_q = result_q
            .gates
            .as_ref()
            .expect("n=8 synthesized without a gate string");
        let recovered_q = eval_gate_string_q(gates_q);
        let roundtrip_q = diamond_distance_u2q_float(&recovered_q, &target);
        eprintln!(
            "n=8 eps={eps:.1e}: elapsed={}ms lde={} claimed={:.6e} actual={:.6e} gates_len={} x16={:?}",
            elapsed_q.as_millis(),
            result_q.lde,
            result_q.distance,
            roundtrip_q,
            gates_q.len(),
            result_q.x
        );
        assert!(
            result_q.distance < eps && roundtrip_q < eps,
            "n=8 failed: claimed={:.6e}, actual={:.6e}, epsilon={:.6e}, lde={}, gates={}",
            result_q.distance,
            roundtrip_q,
            eps,
            result_q.lde,
            gates_q
        );
    }

    #[test]
    fn check_h_r_h_decomposition() {
        let h = eval_gate_string("H"); // adjust to your codebase
        let r_pi6 = rz_pi6_mat();
        let hrh = mat_mul(mat_mul(h, r_pi6), h);

        let synth = SynthesizerPi6::new(0.001);
        let result = synth.synthesize(hrh).expect("should synthesize HRH");

        eprintln!(
            "HRH: lde={} gates={:?} dist={:.3e}",
            result.lde, result.gates, result.distance
        );

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
        let synth = SynthesizerPi6::new(0.01).with_max_lde(15);
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
                    'H' => CLIFFORD_TABLE_T
                        .iter()
                        .find(|(n, _)| *n == "H")
                        .unwrap()
                        .1
                        .to_float(),
                    'S' => CLIFFORD_TABLE_T
                        .iter()
                        .find(|(n, _)| *n == "S")
                        .unwrap()
                        .1
                        .to_float(),
                    'X' => CLIFFORD_TABLE_T
                        .iter()
                        .find(|(n, _)| *n == "X")
                        .unwrap()
                        .1
                        .to_float(),
                    'Y' => CLIFFORD_TABLE_T
                        .iter()
                        .find(|(n, _)| *n == "Y")
                        .unwrap()
                        .1
                        .to_float(),
                    'Z' => CLIFFORD_TABLE_T
                        .iter()
                        .find(|(n, _)| *n == "Z")
                        .unwrap()
                        .1
                        .to_float(),
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
    #[ignore] // slow; run with `cargo test -- --ignored`
    fn stress_random_angles_n6() {
        use std::f64::consts::PI;
        let synth = SynthesizerPi6::new(1e-4);

        let mut state: u64 = 0xDEADBEEF;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 32) as f64 / (1u64 << 32) as f64
        };

        let mut max_dist: f64 = 0.0;
        let mut max_r = 0;
        let mut total_r = 0;
        let n = 20;

        for i in 0..n {
            let theta = (next() - 0.5) * PI;
            let target = rz(theta);
            let result = synth
                .synthesize(target)
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
        eprintln!(
            "max_dist={max_dist:.3e} max_r={max_r} mean_r={:.1}",
            total_r as f64 / n as f64
        );
    }

    #[test]
    #[ignore] // specific angles; run with `cargo test -- --ignored`
    fn synthesize_at_various_small_angles() {
        // Test a range of angles. Each angle should synthesize correctly.
        let eps = 0.01;
        let synth = SynthesizerPi6::new(eps).with_max_lde(20);

        // Angles spanning from "near identity" to "near R" to general.
        let angles = [
            0.001,   // basically identity
            0.01,    // at-the-edge of "Clifford suffices"
            0.05,    // forces real synthesis
            0.0699,  // the original failing case
            -0.0699, // also a known failure
            0.1,
            0.3, // the original "small" test
            0.5,
            PI / 6.0 - 0.001, // just shy of R
            PI / 6.0 + 0.001, // just past R
            PI / 4.0,
            1.0,
            2.0,
        ];

        for &theta in &angles {
            let target = rz(theta);
            let result = match synth.synthesize(target) {
                Some(r) => r,
                None => {
                    eprintln!("θ={theta:+.4}: NO RESULT");
                    continue;
                }
            };

            if let Some(ref gates) = result.gates {
                let m = eval_gate_string(gates);
                let d = diamond_distance_float(&m, &target);
                let r_count = gates.chars().filter(|&c| c == 'R').count();
                let status = if d < eps { "ok " } else { "BAD" };
                eprintln!(
                    "{status} θ={theta:+.4} lde={} r={r_count} claim={:.3e} actual={:.3e} gates={}",
                    result.lde,
                    result.distance,
                    d,
                    if gates.len() > 30 {
                        format!("{}...", &gates[..30])
                    } else {
                        gates.clone()
                    }
                );
            }
        }
    }

    #[test]
    fn lattice_omicron_at_k0() {
        use crate::synthesis::lattice_omicron;
        use std::sync::atomic::AtomicBool;

        // Identity target: y = compute_y(1, 0, 0, 0).
        let y = compute_y(1.0, 0.0, 0.0, 0.0);
        eprintln!("y for identity: {y:?}");

        let mut scratch = lattice_omicron::LatticeScratch::new(0.01);
        let budget_hit = AtomicBool::new(false);
        let sols = lattice_omicron::phase1(&mut scratch, &y, 0, 0.01, 1000, &budget_hit);
        eprintln!("k=0 returned {} solutions", sols.len());
        for x in &sols {
            eprintln!("  x = {x:?}");
        }

        // Now try with scaled y (factor 2^(0/2) = 1, so no actual change at k=0).
        let scale = 1.0;
        let y_scaled: [f64; 8] = std::array::from_fn(|i| y[i] * scale);
        let mut scratch2 = lattice_omicron::LatticeScratch::new(0.01);
        let sols2 = lattice_omicron::phase1(&mut scratch2, &y_scaled, 0, 0.01, 1000, &budget_hit);
        eprintln!("k=0 with scaled y: {} solutions", sols2.len());

        // Try k=1
        let mut scratch3 = lattice_omicron::LatticeScratch::new(0.01);
        let sols3 = lattice_omicron::phase1(&mut scratch3, &y, 1, 0.01, 1000, &budget_hit);
        eprintln!("k=1 returned {} solutions", sols3.len());
        for x in &sols3 {
            eprintln!("  x = {x:?}");
        }

        // Try k=1 with scaled y (factor √2 ≈ 1.414)
        let scale = 2.0_f64.sqrt();
        let y_scaled: [f64; 8] = std::array::from_fn(|i| y[i] * scale);
        let mut scratch4 = lattice_omicron::LatticeScratch::new(0.01);
        let sols4 = lattice_omicron::phase1(&mut scratch4, &y_scaled, 1, 0.01, 1000, &budget_hit);
        eprintln!("k=1 with scaled y: {} solutions", sols4.len());
    }

    #[test]
    #[ignore]
    fn lattice_omicron_with_scaling_at_various_k() {
        use crate::synthesis::lattice_omicron;
        use std::sync::atomic::AtomicBool;

        // R_z(0.3) — a non-trivial target with known good synthesis at moderate k.
        let target = rz(0.3_f64);
        let v = unitary_to_uv_n6(&target);
        let y_unit = compute_y(v[0], v[1], v[2], v[3]);

        for k in [5u32, 8, 10, 12, 15, 18, 20] {
            let target_k = 1_i64 << k;
            let scale = (target_k as f64).sqrt();
            let y_scaled: [f64; 8] = std::array::from_fn(|i| y_unit[i] * scale);

            // Try with both unit-magnitude and scaled y.
            let mut s1 = lattice_omicron::LatticeScratch::new(0.01);
            let bh1 = AtomicBool::new(false);
            let sols_unit = lattice_omicron::phase1(&mut s1, &y_unit, k, 0.01, 100_000, &bh1);

            let mut s2 = lattice_omicron::LatticeScratch::new(0.01);
            let bh2 = AtomicBool::new(false);
            let sols_scaled = lattice_omicron::phase1(&mut s2, &y_scaled, k, 0.01, 100_000, &bh2);

            eprintln!(
                "k={k}: |y_unit|={:.2} → {} sols ;  |y_scaled|={:.2} → {} sols",
                (y_unit.iter().map(|v| v * v).sum::<f64>()).sqrt(),
                sols_unit.len(),
                (y_scaled.iter().map(|v| v * v).sum::<f64>()).sqrt(),
                sols_scaled.len(),
            );

            for x in sols_unit.iter().chain(sols_scaled.iter()).take(2) {
                let valid = check_norm_eq(x, k) && check_bilinear(x);
                eprintln!("  x={x:?} valid={valid}");
            }
        }
    }
    #[test]
    #[ignore]
    fn release_deep_eps_check() {
        let target = rz(0.3);
        for &eps in &[1e-3_f64, 1e-4, 1e-5] {
            let synth = SynthesizerPi6::new(eps);
            let t = std::time::Instant::now();
            let result = synth.synthesize(target);
            let elapsed = t.elapsed();
            match result {
                Some(r) => {
                    let actual = diamond_distance_float(
                        &eval_gate_string(r.gates.as_deref().unwrap_or("")),
                        &target,
                    );
                    eprintln!(
                        "ε={eps:.0e}: lde={} claimed={:.3e} actual={:.3e} t={:.2}s",
                        r.lde,
                        r.distance,
                        actual,
                        elapsed.as_secs_f64()
                    );
                }
                None => eprintln!("ε={eps:.0e}: FAILED in {:.2}s", elapsed.as_secs_f64()),
            }
        }
    }

    #[test]
    fn lattice_with_proper_y_scaling() {
        use crate::synthesis::lattice_omicron::{self, LatticeScratch};
        use std::f64::consts::PI;
        use std::sync::atomic::AtomicBool;

        fn compute_align_vec_n6(v: [f64; 4]) -> [f64; 8] {
            let mut y = [0.0f64; 8];
            for j in 0..4 {
                let theta = (j as f64) * PI / 6.0;
                let c = theta.cos();
                let s = theta.sin();
                y[j] = c * v[0] + s * v[1]; // u block
                y[4 + j] = c * v[2] + s * v[3]; // t block
            }
            y
        }

        fn uv_to_xy_n6(v: [f64; 4], k: u32) -> [f64; 8] {
            let scale = 2.0_f64.powf(k as f64 / 2.0) / 2.0;
            let raw = compute_align_vec_n6(v);
            std::array::from_fn(|i| raw[i] * scale)
        }

        let theta = 0.3_f64;
        let target = rz(theta);
        let v = unitary_to_uv_n6(&target);

        for k in [8u32, 10, 12, 14] {
            let y = uv_to_xy_n6(v, k);
            let mut scratch = LatticeScratch::new(1e-3);
            let budget_hit = AtomicBool::new(false);
            let t = std::time::Instant::now();
            let sols = lattice_omicron::phase1(&mut scratch, &y, k, 1e-3, 100_000, &budget_hit);
            let elapsed = t.elapsed();
            eprintln!(
                "k={k}: {} solutions in {:.2}s",
                sols.len(),
                elapsed.as_secs_f64()
            );
            for sol in &sols {
                let u_ring = solution_to_u2_omicron(sol, k);
                let u_float = u_ring.to_float();
                let dist = diamond_distance_float(&u_float, &target);
                eprintln!("  sol={sol:?} dist_to_target={dist:.3e}");
            }
        }
    }

    #[test]
    fn lattice_per_call_microbench() {
        use crate::synthesis::lattice_omicron::{self, LatticeScratch};
        use std::sync::atomic::AtomicBool;
        use std::time::Instant;

        let v = [0.99, 0.05, 0.0, 0.0]; // close to identity
        let y = compute_y(v[0], v[1], v[2], v[3]);
        let eps = 1e-3;

        for k in [8u32, 10, 12, 15] {
            let mut scratch = LatticeScratch::new(eps);
            let budget_hit = AtomicBool::new(false);

            // Warm up (first call allocates)
            let _ = lattice_omicron::phase1(&mut scratch, &y, k, eps, 100_000, &budget_hit);

            // Time second call (no allocation)
            let t = Instant::now();
            let sols = lattice_omicron::phase1(&mut scratch, &y, k, eps, 100_000, &budget_hit);
            let elapsed = t.elapsed();

            eprintln!(
                "k={k}: {} sols, second call: {:.3}s",
                sols.len(),
                elapsed.as_secs_f64()
            );
        }
    }

    #[test]
    fn time_single_brute_force_call() {
        // Time one sequential brute_force_direct_search_n6(64, ...) call in isolation.
        let target = rz(0.3_f64);
        let v = unitary_to_uv_n6(&target);
        let y = compute_y(v[0], v[1], v[2], v[3]);
        let target_k: i64 = 1 << 6; // k=6, the inner k used by dc_search
        let eps = 0.01_f64;
        let n_calls = 20;
        let t = std::time::Instant::now();
        for _ in 0..n_calls {
            let _ = direct_search_n6(target_k, &y, eps, 1);
        }
        let total_ms = t.elapsed().as_millis();
        eprintln!(
            "{n_calls} calls to direct_search_n6(64,...): {}ms total, {:.1}ms/call",
            total_ms,
            total_ms as f64 / n_calls as f64
        );
    }

    #[test]
    #[ignore]
    fn direct_search_only_at_high_k() {
        let target = rz(0.3);
        let v = unitary_to_uv_n6(&target);
        let eps = 1e-3;
        let synth = SynthesizerPi6::new(eps);

        for k in [13u32, 14, 15] {
            let t = std::time::Instant::now();
            let result = synth.direct_search(&target, v, k);
            let elapsed = t.elapsed();
            match result {
                Some(r) => eprintln!(
                    "k={k}: SUCCESS dist={:.3e} in {:.1}s",
                    r.distance,
                    elapsed.as_secs_f64()
                ),
                None => eprintln!("k={k}: NONE in {:.1}s", elapsed.as_secs_f64()),
            }
        }
    }
    #[test]
    #[ignore]
    fn lattice_at_k7_with_rotated_target() {
        let target = rz(0.3);
        let eps = 1e-3;
        let k_inner = 7u32;
        let target_k = 1_i64 << k_inner;

        let prefixes = build_l_pi6(8); // k_prefix=8, total k=15
        let scale_factors = [
            1.0,
            2.0_f64.powf(k_inner as f64 / 2.0),
            2.0_f64.powf(k_inner as f64 / 2.0 - 1.0),
            2.0_f64.powf(k_inner as f64 / 2.0) / 4.0, // n=8 style
            2.0_f64.powf(k_inner as f64 / 2.0) / 2.0,
        ];

        use crate::synthesis::lattice_omicron;
        use std::sync::atomic::AtomicBool;

        for (idx, scale_factor) in scale_factors.iter().enumerate() {
            let mut total_sols = 0;
            for (_, u_l) in prefixes.iter().take(20) {
                let u_l_dag = mat_dag(u_l);
                let m_inner = mat_mul(u_l_dag, target);
                let v_inner = unitary_to_uv_n6(&m_inner);
                let y = compute_y(v_inner[0], v_inner[1], v_inner[2], v_inner[3]);
                let y_scaled: [f64; 8] = std::array::from_fn(|i| y[i] * scale_factor);

                let mut scratch = lattice_omicron::LatticeScratch::new(eps);
                let budget_hit = AtomicBool::new(false);
                let sols = lattice_omicron::phase1(
                    &mut scratch,
                    &y_scaled,
                    k_inner,
                    eps,
                    100_000,
                    &budget_hit,
                );

                // Verify solutions are actually valid
                for sol in &sols {
                    if check_norm_eq(sol, k_inner) && check_bilinear(sol) {
                        let dot: f64 = sol
                            .iter()
                            .zip(y.iter())
                            .map(|(&xi, &yi)| xi as f64 * yi)
                            .sum();
                        let thresh = (target_k as f64 * (1.0 - eps * eps)).sqrt();
                        if dot.abs() >= thresh {
                            total_sols += 1;
                        }
                    }
                }
            }
            eprintln!("scale_factor[{idx}]={scale_factor:.3}: total valid sols across 20 prefixes = {total_sols}");
        }
    }
    #[test]
    #[ignore]
    fn inspect_dc_inner_search_at_k15() {
        let target = rz(0.3);
        let eps = 1e-3;

        let k = 15u32;
        let k_prefix = k - 6; // = 9
        let target_k_inner = 64_i64;

        let prefixes = build_l_pi6(k_prefix);
        eprintln!("k_prefix={k_prefix}, |L|={}", prefixes.len());

        // Inspect first 5 prefixes
        for (idx, (prefix_gates, u_l)) in prefixes.iter().take(5).enumerate() {
            let u_l_dag = mat_dag(u_l);
            let m_inner = mat_mul(u_l_dag, target);
            let v_inner = unitary_to_uv_n6(&m_inner);
            let v_norm: f64 = v_inner.iter().map(|x| x * x).sum::<f64>().sqrt();

            let y = compute_y(v_inner[0], v_inner[1], v_inner[2], v_inner[3]);
            let y_norm: f64 = y.iter().map(|x| x * x).sum::<f64>().sqrt();
            let thresh = (target_k_inner as f64 * (1.0 - eps * eps)).sqrt();

            // What's the max |x·y| achievable for x with norm constraint at k_inner?
            // By Cauchy-Schwarz, max |x·y| ≤ |x|·|y|.
            // |x|² + cross ≤ 2·2^k_inner, so |x| ≤ √(2·64) ≈ 11.3
            let max_xy_possible = (2.0 * target_k_inner as f64).sqrt() * y_norm;

            eprintln!(
                "prefix #{idx}: gates={} (len {})",
                prefix_gates,
                prefix_gates.len()
            );
            eprintln!("  v_inner={v_inner:?}");
            eprintln!("  |v_inner|={v_norm:.3} |y|={y_norm:.3}");
            eprintln!("  threshold |x·y| ≥ {thresh:.3}");
            eprintln!("  max possible |x·y| ≤ {max_xy_possible:.3}");
            eprintln!("  feasible? {}", max_xy_possible >= thresh);
        }
    }
}
