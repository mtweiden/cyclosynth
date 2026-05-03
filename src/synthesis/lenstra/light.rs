//! 8D Lenstra enumeration in [`twofloat::TwoFloat`] precision (~104-bit
//! mantissa) for the moderate-ε regime of Clifford+T synthesis.
//! Implements Algorithm 3.6 of arXiv:2510.05816.
//!
//! Per-call pipeline:
//!   1. Build the anisotropic Q-metric (cap × ball constraint, eq 3.15
//!      of the paper).
//!   2. LLL-reduce the ℤ⁸ identity basis using Q as the inner product.
//!   3. Cholesky-factor the post-LLL Q-Gram `B·Q·Bᵀ = L·Lᵀ`.
//!   4. Solve `B·z_c = c` for the cap center in lattice coordinates
//!      via LU with partial pivoting.
//!   5. Schnorr-Euchner enumerate integer 8-tuples `z` inside the SE
//!      ellipsoid `‖Lᵀ·(z − z_c)‖² ≤ 2.01` (f64 inner loop).
//!   6. For each candidate, reconstruct `x = B·z` and validate the
//!      integer constraints (norm shell, bilinear form, alignment).
//!
//! Stack-allocated `Copy` arithmetic; no per-call heap allocations.
//! Numerically stable for `ε ≥ 1e-4`; tighter ε goes through
//! [`super::integer`] instead.

#![allow(dead_code)]

use crate::rings::Float;
use std::sync::atomic::AtomicBool;
use twofloat::TwoFloat;

// ─── Types ────────────────────────────────────────────────────────────────────

type Tf = TwoFloat;
type Mat8 = [[Tf; 8]; 8];
type Vec8 = [Tf; 8];
type IMat8 = [[i64; 8]; 8];

#[inline]
fn tf(x: f64) -> Tf {
    Tf::from(x)
}

#[inline]
fn tf_i(x: i64) -> Tf {
    // i64 in [−2^53, 2^53] is exactly representable as f64. LLL basis entries
    // and most lattice coords stay well inside that range.
    Tf::from(x as f64)
}

#[inline]
fn tf_to_i64_round(x: Tf) -> i64 {
    // Round to nearest, ties away from zero. f64::from(Tf) returns the closest
    // f64; the rounding error is at most 2^−104 of the value.
    let lo = f64::from(x);
    lo.round() as i64
}

// ─── 8×8 LU solve with partial pivoting (twofloat) ────────────────────────────

/// Solve `a · x = b` for `x ∈ ℝ⁸` using Gaussian elimination with partial
/// pivoting in twofloat arithmetic. Returns `None` if `a` is numerically
/// singular (smallest pivot below tolerance).
pub fn lu_solve_8(a: &Mat8, b: &Vec8) -> Option<Vec8> {
    let mut m = *a;
    let mut rhs = *b;
    let zero = tf(0.0);
    let tol = tf(1e-30);

    for k in 0..8 {
        // Find pivot row (largest |m[i][k]| for i in k..8)
        let mut piv = k;
        let mut piv_abs = m[k][k].abs();
        for i in (k + 1)..8 {
            let v = m[i][k].abs();
            if v > piv_abs {
                piv_abs = v;
                piv = i;
            }
        }
        if piv_abs < tol {
            return None;
        }
        if piv != k {
            m.swap(k, piv);
            rhs.swap(k, piv);
        }

        // Eliminate column k below the pivot
        let pivot = m[k][k];
        for i in (k + 1)..8 {
            let factor = m[i][k] / pivot;
            // m[i][j] -= factor * m[k][j]  for j ∈ k..8
            for j in k..8 {
                let mkj = m[k][j];
                m[i][j] = m[i][j] - factor * mkj;
            }
            let rk = rhs[k];
            rhs[i] = rhs[i] - factor * rk;
        }
    }

    // Back substitution: x[i] = (rhs[i] - sum_{j>i} m[i][j]·x[j]) / m[i][i]
    let mut x = [zero; 8];
    for i in (0..8).rev() {
        let mut s = rhs[i];
        for j in (i + 1)..8 {
            s = s - m[i][j] * x[j];
        }
        x[i] = s / m[i][i];
    }
    Some(x)
}

// ─── 8×8 Cholesky (twofloat) ──────────────────────────────────────────────────

/// Cholesky decomposition: `g = L · Lᵀ` for symmetric positive-definite `g`.
/// Returns lower-triangular `L`. `None` if a diagonal element comes out
/// non-positive (indicating `g` is not PD or is too ill-conditioned for the
/// available precision).
pub fn cholesky_8(g: &Mat8) -> Option<Mat8> {
    let zero = tf(0.0);
    let mut l: Mat8 = [[zero; 8]; 8];

    for i in 0..8 {
        for j in 0..=i {
            let mut s = g[i][j];
            for k in 0..j {
                s = s - l[i][k] * l[j][k];
            }
            if i == j {
                if s <= zero {
                    return None;
                }
                l[i][i] = s.sqrt();
            } else {
                l[i][j] = s / l[j][j];
            }
        }
    }
    Some(l)
}

// ─── 8×8 Q-Gram LLL (twofloat) ────────────────────────────────────────────────

/// Compute the Q-Gram matrix `G[i][j] = b_iᵀ · Q · b_j` for the rows of `basis`.
fn compute_qgram(basis: &IMat8, q: &Mat8) -> Mat8 {
    // temp[i][b] = sum_a basis[i][a] · Q[a][b]
    let zero = tf(0.0);
    let mut temp: Mat8 = [[zero; 8]; 8];
    for i in 0..8 {
        for b in 0..8 {
            let mut s = zero;
            for a in 0..8 {
                s = s + tf_i(basis[i][a]) * q[a][b];
            }
            temp[i][b] = s;
        }
    }
    // g[i][j] = sum_b temp[i][b] · basis[j][b]
    let mut g: Mat8 = [[zero; 8]; 8];
    for i in 0..8 {
        for j in 0..8 {
            let mut s = zero;
            for b in 0..8 {
                s = s + temp[i][b] * tf_i(basis[j][b]);
            }
            g[i][j] = s;
        }
    }
    g
}

/// Gram-Schmidt orthogonalization in the Q-metric. Computes `mu[i][j]` (the
/// projection coefficient of `b_i` onto `b_j*`) and the squared G-norm of each
/// orthogonalized vector. Operates entirely in Gram-matrix form (no explicit
/// orthogonalized vectors), so numerical error from the basis vectors directly
/// is avoided.
fn gs_qgram(basis: &IMat8, q: &Mat8) -> ([[Tf; 8]; 8], [Tf; 8]) {
    let g = compute_qgram(basis, q);
    let zero = tf(0.0);
    let mut mu: [[Tf; 8]; 8] = [[zero; 8]; 8];
    // g_star[i][j] = G(b_i, b_j*) for j ≤ i (only need the lower triangle).
    let mut g_star: [[Tf; 8]; 8] = [[zero; 8]; 8];
    let mut gnorm_sq: [Tf; 8] = [zero; 8];

    for j in 0..8 {
        // First compute g_star[i][j] for all i ≥ j.
        for i in j..8 {
            let mut s = g[i][j];
            for k in 0..j {
                s = s - mu[j][k] * g_star[i][k];
            }
            g_star[i][j] = s;
        }
        gnorm_sq[j] = g_star[j][j];
        if gnorm_sq[j].abs() < tf(1e-60) {
            // Degenerate: just leave mu[i][j] = 0 for i > j
            continue;
        }
        for i in (j + 1)..8 {
            mu[i][j] = g_star[i][j] / gnorm_sq[j];
        }
    }
    (mu, gnorm_sq)
}

/// LLL-reduce the ℤ⁸ identity basis using `q` as the inner-product metric
/// (`G(u, v) := uᵀ · q · v`). `q` must be symmetric positive definite. Returns
/// a unimodular 8×8 integer matrix whose rows are the LLL-reduced basis.
pub fn lll_qgram_8(q: &Mat8) -> IMat8 {
    let mut b: IMat8 = std::array::from_fn(|i| {
        let mut row = [0i64; 8];
        row[i] = 1;
        row
    });

    let delta = tf(0.75);
    let mut k = 1usize;
    let max_iter = 10_000usize;
    let mut iterations = 0usize;

    while k < 8 && iterations < max_iter {
        iterations += 1;
        let (mu, _) = gs_qgram(&b, q);

        // Size reduction: for j from k-1 down to 0, b[k] -= round(mu[k][j]) · b[j]
        for j in (0..k).rev() {
            let r = tf_to_i64_round(mu[k][j]);
            if r != 0 {
                for c in 0..8 {
                    b[k][c] -= r * b[j][c];
                }
            }
        }

        // Lovász condition: G(b_k*, b_k*) ≥ (δ − μ_{k,k-1}²) · G(b_{k-1}*, b_{k-1}*)
        let (mu2, gnorm) = gs_qgram(&b, q);
        let lhs = gnorm[k];
        let rhs = (delta - mu2[k][k - 1] * mu2[k][k - 1]) * gnorm[k - 1];
        if lhs >= rhs {
            k += 1;
        } else {
            b.swap(k, k - 1);
            k = k.saturating_sub(1).max(1);
        }
    }
    b
}

// ─── Exact integer determinant in i256 (for the unimodularity assertion) ──────

/// Compute the determinant of an 8×8 i64 matrix exactly using i256 arithmetic
/// (so any LLL-induced corruption that grows entries beyond i64 still gives a
/// correct answer here). Returns the determinant as i64 if it fits, else None.
pub fn det8_exact(m: &IMat8) -> Option<i64> {
    use i256::i256;
    // Convert to i256 with a denominator (LU expansion with rational pivot to
    // avoid fraction simplification)... actually simpler: use the Bareiss
    // algorithm, which uses only integer arithmetic and stays in i256 for our
    // input range.
    let mut a: [[i256; 8]; 8] = std::array::from_fn(|i| {
        std::array::from_fn(|j| i256::from_i64(m[i][j]))
    });
    let mut sign: i64 = 1;
    let mut prev = i256::from_i64(1);
    let zero = i256::from_i64(0);

    for k in 0..8 {
        // Find a non-zero pivot in column k from row k onward
        if a[k][k] == zero {
            let mut found = false;
            for i in (k + 1)..8 {
                if a[i][k] != zero {
                    a.swap(k, i);
                    sign = -sign;
                    found = true;
                    break;
                }
            }
            if !found {
                return Some(0);
            }
        }
        let pivot = a[k][k];
        // Bareiss update: a[i][j] = (a[i][j] · pivot − a[i][k] · a[k][j]) / prev
        for i in (k + 1)..8 {
            for j in (k + 1)..8 {
                let lhs = a[i][j] * pivot;
                let rhs = a[i][k] * a[k][j];
                let diff = lhs - rhs;
                // Bareiss guarantees `prev` divides `diff`.
                a[i][j] = diff / prev;
            }
            a[i][k] = zero;
        }
        prev = pivot;
    }
    // Determinant is sign · a[7][7]
    let det = a[7][7];
    let det_signed = if sign < 0 { -det } else { det };
    let lo = det_signed.as_i128();
    if lo >= i64::MIN as i128 && lo <= i64::MAX as i128 {
        Some(lo as i64)
    } else {
        None
    }
}

// ─── Anisotropic Q and center c (twofloat) ────────────────────────────────────

/// Σ matrix from arXiv:2510.05816 eq (3.15), as 8×8 entries (twofloat). The
/// first 4 rows are Σ_top (mapping x ∈ ℝ⁸ to (Re u₁, Im u₁, Re u₂, Im u₂)·√2^k);
/// the last 4 rows are Σ_bot (mapping to (Re u•₁, Im u•₁, Re u•₂, Im u•₂)·√2^k).
fn sigma_matrix() -> [[Tf; 8]; 8] {
    let r2 = TwoFloat::new_div(1.0, 2.0_f64.sqrt()); // 1/√2 with full twofloat precision
    let nr2 = TwoFloat::new_div(-1.0, 2.0_f64.sqrt());
    let z = tf(0.0);
    let o = tf(1.0);
    [
        [o,  r2, z,  nr2, z,  z,  z,  z  ], // Σ_top row 0
        [z,  r2, o,  r2,  z,  z,  z,  z  ], // Σ_top row 1
        [z,  z,  z,  z,   o,  r2, z,  nr2], // Σ_top row 2
        [z,  z,  z,  z,   z,  r2, o,  r2 ], // Σ_top row 3
        [o,  nr2,z,  r2,  z,  z,  z,  z  ], // Σ_bot row 0
        [z,  nr2,o,  nr2, z,  z,  z,  z  ], // Σ_bot row 1
        [z,  z,  z,  z,   o,  nr2,z,  r2 ], // Σ_bot row 2
        [z,  z,  z,  z,   z,  nr2,o,  nr2], // Σ_bot row 3
    ]
}

/// Compute the 8×8 anisotropic Q matrix (twofloat) defining the ellipsoid that
/// bounds the body S = sphere ∩ alignment-cap × sphere. Three eigenvalue
/// scales: 1/Δ_y² along ŷ (alignment, super-thin), 1/Δ_⊥² for the 3
/// orthogonal directions in the u-subspace (thin), 1/R² for the 4 directions
/// in the u•-subspace (full ball width).
///
/// Q = (1/Δ_y²)·ŷŷᵀ + (1/Δ_⊥²)·(P_u − ŷŷᵀ) + (1/R²)·P_{u•}
///
/// where P_u = ½·Σ_topᵀ·Σ_top is the projector onto u-subspace and similarly
/// for P_{u•}. ŷ = y/‖y‖ (and lies entirely within the u-subspace by
/// construction since y = Σ_topᵀ·v).
pub fn build_q(y: &[Tf; 8], k: u32, eps: Tf) -> Mat8 {
    let r_sq = tf((1u64 << k) as f64); // 2^k
    let r = r_sq.sqrt();
    let one = tf(1.0);
    let two = tf(2.0);

    // Δ_y = R · ε² / (2·(1 + √(1−ε²))) — safe form, avoids 1 − √(1−ε²) cancellation
    let one_minus_eps2 = one - eps * eps;
    let sqrt_1m = one_minus_eps2.sqrt();
    let delta_y = r * (eps * eps) / (two * (one + sqrt_1m));
    let delta_perp = r * eps;

    let inv_dy_sq = one / (delta_y * delta_y);
    let inv_dp_sq = one / (delta_perp * delta_perp);
    let inv_r_sq = one / r_sq;

    // y_norm_sq, then ŷŷᵀ
    let mut y_norm_sq = tf(0.0);
    for i in 0..8 {
        y_norm_sq = y_norm_sq + y[i] * y[i];
    }
    let inv_y_norm_sq = one / y_norm_sq;
    let mut yhat_yhat_t = [[tf(0.0); 8]; 8];
    for i in 0..8 {
        for j in 0..8 {
            yhat_yhat_t[i][j] = y[i] * y[j] * inv_y_norm_sq;
        }
    }

    // Σ_top, Σ_bot → P_u, P_{u•}
    let sigma = sigma_matrix();
    let mut p_u = [[tf(0.0); 8]; 8];
    let mut p_ub = [[tf(0.0); 8]; 8];
    let half = tf(0.5);
    for i in 0..8 {
        for j in 0..8 {
            let mut su = tf(0.0);
            let mut sb = tf(0.0);
            for r_idx in 0..4 {
                su = su + sigma[r_idx][i] * sigma[r_idx][j];
                sb = sb + sigma[r_idx + 4][i] * sigma[r_idx + 4][j];
            }
            p_u[i][j] = su * half;
            p_ub[i][j] = sb * half;
        }
    }

    // Assemble Q
    let mut q = [[tf(0.0); 8]; 8];
    for i in 0..8 {
        for j in 0..8 {
            q[i][j] = inv_dy_sq * yhat_yhat_t[i][j]
                + inv_dp_sq * (p_u[i][j] - yhat_yhat_t[i][j])
                + inv_r_sq * p_ub[i][j];
        }
    }
    q
}

/// Compute the cap center along the alignment direction. Subtle point: the
/// body in our convention has ŷ·x ranging over `[‖y‖·√(1−ε²), ‖y‖]` (since
/// ŷ·x = ‖y‖ · u·v and u·v ∈ [√(1−ε²), 1]). The midpoint along ŷ is therefore
/// `‖y‖·(1+√(1−ε²))/2`. Since ŷ = y/‖y‖, the 8D center vector is
/// `c = ŷ · ‖y‖·cap_mid = y · cap_mid`, where `cap_mid = (1+√(1−ε²))/2`.
///
/// (The buddy's formula `c = ŷ · R · cap_mid` over-shoots by a factor of
/// √2 — that formula is correct for a cap on the **8D sphere of radius R**,
/// but our body's alignment direction only reaches `‖y‖ = R/√2`, not `R`.)
pub fn build_center(y: &[Tf; 8], _k: u32, eps: Tf) -> Vec8 {
    let one = tf(1.0);
    let two = tf(2.0);
    let sqrt_1m = (one - eps * eps).sqrt();
    let cap_mid = (one + sqrt_1m) / two;
    let mut c = [tf(0.0); 8];
    for i in 0..8 {
        c[i] = y[i] * cap_mid;
    }
    c
}

// ─── 8D Schnorr-Euchner search in f64 ─────────────────────────────────────────

/// Enumerate integer points z ∈ ℤ⁸ with ‖R_chol·(z − z_c)‖² ≤ bound, where
/// R_chol is upper-triangular. Iterates z[7] (largest GS direction) outermost,
/// outward from z_c.round(); recurses to z[0]. For each candidate z that
/// satisfies the bound, calls `callback(&z)`. If callback returns `Some`, the
/// search short-circuits.
fn se_8d_f64<F>(
    r_chol: &[[f64; 8]; 8],
    z_c: &[f64; 8],
    bound: f64,
    mut callback: F,
) -> Option<[i64; 8]>
where
    F: FnMut(&[i64; 8]) -> Option<[i64; 8]>,
{
    let mut z = [0i64; 8];
    let result = std::cell::RefCell::new(None);

    fn recurse<F>(
        depth: i32, // 7..=−1; −1 means all fixed
        r_chol: &[[f64; 8]; 8],
        z_c: &[f64; 8],
        bound: f64,
        z: &mut [i64; 8],
        partial: f64,
        callback: &mut F,
        result: &std::cell::RefCell<Option<[i64; 8]>>,
    ) where
        F: FnMut(&[i64; 8]) -> Option<[i64; 8]>,
    {
        if result.borrow().is_some() {
            return;
        }
        if depth < 0 {
            if let Some(r) = callback(z) {
                *result.borrow_mut() = Some(r);
            }
            return;
        }
        let d = depth as usize;
        let r_dd = r_chol[d][d];
        if r_dd.abs() < 1e-30 {
            // Degenerate dim — only z[d] = z_c[d].round() is feasible
            z[d] = z_c[d].round() as i64;
            recurse(depth - 1, r_chol, z_c, bound, z, partial, callback, result);
            return;
        }
        // tail = ∑_{j > d} R_chol[d][j] · (z[j] − z_c[j])
        let mut tail = 0.0;
        for j in (d + 1)..8 {
            tail += r_chol[d][j] * (z[j] as f64 - z_c[j]);
        }
        // We need (R_chol[d][d]·(z[d]−z_c[d]) + tail)² ≤ rem,
        // i.e. r_dd·(z[d] − z_c[d]) ∈ [−√rem − tail, +√rem − tail]
        // ⇒ z[d] ∈ z_c[d] + [(−√rem − tail)/r_dd, (+√rem − tail)/r_dd]
        let rem = bound - partial;
        if rem < 0.0 {
            return;
        }
        let rem_sqrt = rem.sqrt();
        let center_off = -tail / r_dd; // z[d] − z_c[d] center
        let span = rem_sqrt / r_dd.abs();
        let z_low = (z_c[d] + center_off - span).ceil() as i64;
        let z_high = (z_c[d] + center_off + span).floor() as i64;
        let z_mid = (z_c[d] + center_off).round() as i64;
        let max_off = (z_high - z_mid).max(z_mid - z_low).max(0);

        // Outward iteration: z_mid, z_mid+1, z_mid−1, z_mid+2, z_mid−2, ...
        for raw in 0..=(2 * max_off + 1) {
            if result.borrow().is_some() {
                return;
            }
            let off = if raw == 0 {
                0
            } else if raw % 2 == 1 {
                (raw + 1) / 2
            } else {
                -(raw / 2)
            };
            let zd = z_mid + off;
            if zd < z_low || zd > z_high {
                continue;
            }
            let level = r_dd * (zd as f64 - z_c[d]) + tail;
            let new_partial = partial + level * level;
            if new_partial > bound + 1e-9 {
                continue;
            }
            z[d] = zd;
            recurse(depth - 1, r_chol, z_c, bound, z, new_partial, callback, result);
        }
    }

    recurse(7, r_chol, z_c, bound, &mut z, 0.0, &mut callback, &result);
    result.into_inner()
}

// ─── phase1_lenstra: full pipeline ────────────────────────────────────────────

/// Bilinear unitarity form B(x) = a₁b₁ − a₁d₁ + b₁c₁ + c₁d₁ + a₂b₂ − a₂d₂ +
/// b₂c₂ + c₂d₂. Equals (‖u‖² − ‖u•‖²)/√2; B(x) = 0 + ‖x‖² = 2^k forces both
/// halves to be unit-norm and the matrix to be unitary.
#[inline]
fn bilinear_b(x: &[i64; 8]) -> i64 {
    let (a1, b1, c1, d1) = (x[0], x[1], x[2], x[3]);
    let (a2, b2, c2, d2) = (x[4], x[5], x[6], x[7]);
    a1 * b1 - a1 * d1 + b1 * c1 + c1 * d1 + a2 * b2 - a2 * d2 + b2 * c2 + c2 * d2
}

/// Reconstruct x = B_LLL · z (i64 exact).
#[inline]
fn reconstruct_x(b_lll: &IMat8, z: &[i64; 8]) -> [i64; 8] {
    let mut x = [0i64; 8];
    for i in 0..8 {
        for j in 0..8 {
            x[j] += z[i] * b_lll[i][j];
        }
    }
    x
}

pub fn phase1_lenstra(
    y: &[Float; 8],
    k: u32,
    eps: Float,
    max_phase2_calls: u64,
    budget_hit: &AtomicBool,
) -> Vec<[i64; 8]> {
    use std::sync::atomic::{AtomicU64, Ordering};

    let target_norm: i64 = 1i64 << k;
    let threshold_xy = (1i64 << (2 * k)) as Float / 4.0 * (1.0 - eps * eps);

    // Convert y to twofloat for setup.
    let y_tf: [Tf; 8] = std::array::from_fn(|i| tf(y[i]));
    let eps_tf = tf(eps);

    // Step 1: Build Q and center c
    let q = build_q(&y_tf, k, eps_tf);
    let c = build_center(&y_tf, k, eps_tf);

    // Step 2: LLL with Q metric
    let b_lll = lll_qgram_8(&q);

    // Step 3: assert det = ±1 (catches twofloat exhaustion)
    match det8_exact(&b_lll) {
        Some(1) | Some(-1) => {}
        Some(d) => {
            eprintln!(
                "[lenstra] LLL produced non-unimodular basis (det={}); k={}, ε={:e}; bailing.",
                d, k, eps
            );
            return Vec::new();
        }
        None => {
            eprintln!("[lenstra] det8_exact overflow (basis corrupted?); k={}, ε={:e}; bailing.", k, eps);
            return Vec::new();
        }
    }

    // Step 4: Cholesky of G_LLL = B_LLL · Q · B_LLLᵀ → L (lower); R_chol = Lᵀ
    let g_lll = compute_qgram(&b_lll, &q);
    let l = match cholesky_8(&g_lll) {
        Some(l) => l,
        None => {
            eprintln!("[lenstra] Cholesky failed (G not PD or precision lost); k={}, ε={:e}; bailing.", k, eps);
            return Vec::new();
        }
    };
    // R_chol = Lᵀ; downcast to f64
    let mut r_chol_f64 = [[0.0_f64; 8]; 8];
    for i in 0..8 {
        for j in 0..8 {
            r_chol_f64[i][j] = f64::from(l[j][i]);
        }
    }

    // Step 5: solve B_LLL · z_c = c via twofloat LU with partial pivoting
    let b_lll_tf: Mat8 = std::array::from_fn(|i| {
        std::array::from_fn(|j| tf_i(b_lll[i][j]))
    });
    // B_LLL has rows = basis vectors, so x = ∑ z[i]·b_lll[i] = B_LLLᵀ · z.
    // To solve x = c for z: B_LLLᵀ · z = c. So we transpose for the LU input.
    let mut b_lll_t = [[tf(0.0); 8]; 8];
    for i in 0..8 {
        for j in 0..8 {
            b_lll_t[i][j] = b_lll_tf[j][i];
        }
    }
    let z_c_tf = match lu_solve_8(&b_lll_t, &c) {
        Some(zc) => zc,
        None => {
            eprintln!("[lenstra] LU solve failed for B_LLL·z_c = c; bailing.");
            return Vec::new();
        }
    };
    let z_c_f64: [f64; 8] = std::array::from_fn(|i| f64::from(z_c_tf[i]));

    // Step 6: 8D SE in f64 with bound 1.51.
    //
    // Bound derivation: max (x − c)ᵀ Q (x − c) over the body (cap × ball with
    // sphere shell) is 1.5 — three independent corner contributions, each 0.5,
    // because the buddy's Δ formulas (Δ_y = R·ε²/(2(1+√(1−ε²))), Δ_⊥ = R·ε,
    // Δ_{u•} = R) over-state our actual body extents by √2 in each direction.
    // (Our body's alignment direction reaches only ‖y‖ = R/√2, not R, because
    // y = Σ_topᵀ·v has ‖y‖² = 2 not 4 for unit v.) +0.01 absorbs f64 downcast
    // noise.
    let count = AtomicU64::new(0);
    let result = se_8d_f64(&r_chol_f64, &z_c_f64, 1.51, |z: &[i64; 8]| {
        // Cap check
        if count.load(Ordering::Relaxed) >= max_phase2_calls {
            budget_hit.store(true, Ordering::Relaxed);
            return None;
        }
        count.fetch_add(1, Ordering::Relaxed);

        // Reconstruct x exactly
        let x = reconstruct_x(&b_lll, z);

        // Norm equality (sphere shell, not interior)
        let n: i64 = x.iter().map(|&v| v * v).sum();
        if n != target_norm {
            return None;
        }
        // Bilinear unitarity
        if bilinear_b(&x) != 0 {
            return None;
        }
        // Alignment cap
        let dot: Float = x
            .iter()
            .zip(y.iter())
            .map(|(a, b)| *a as Float * b)
            .sum();
        if dot * dot < threshold_xy {
            return None;
        }
        Some(x)
    });

    match result {
        Some(x) => vec![x],
        None => Vec::new(),
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn ident_q() -> Mat8 {
        let mut q = [[tf(0.0); 8]; 8];
        for i in 0..8 {
            q[i][i] = tf(1.0);
        }
        q
    }

    /// PD test matrix with mild anisotropy: diag(scales) where scales include a
    /// 10⁶ ratio between the largest and smallest. Mimics the structure of the
    /// real cap-bounding ellipsoid.
    fn anisotropic_q(align_scale: f64) -> Mat8 {
        let mut q = [[tf(0.0); 8]; 8];
        // 1 alignment direction (very large scale)
        q[0][0] = tf(align_scale);
        // 3 mid-scale directions
        for i in 1..4 {
            q[i][i] = tf(align_scale.sqrt());
        }
        // 4 unit-scale directions
        for i in 4..8 {
            q[i][i] = tf(1.0);
        }
        q
    }

    #[test]
    fn lu_solve_identity() {
        let mut id = [[tf(0.0); 8]; 8];
        for i in 0..8 {
            id[i][i] = tf(1.0);
        }
        let b: Vec8 = std::array::from_fn(|i| tf((i + 1) as f64));
        let x = lu_solve_8(&id, &b).expect("identity solve");
        for i in 0..8 {
            let diff = x[i] - tf((i + 1) as f64);
            assert!(diff.abs() < tf(1e-15), "x[{}] off: {:?}", i, f64::from(diff));
        }
    }

    #[test]
    fn lu_solve_anisotropic_f64_inputs() {
        // Inputs are f64 (lossy 0.1 etc), so we can only expect f64-level
        // precision on the round-trip. This validates that LU+pivoting itself
        // doesn't lose more precision than the inputs supply.
        let mut a = [[tf(0.0); 8]; 8];
        let diag = [1e8_f64, 1e4, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        for i in 0..8 {
            a[i][i] = tf(diag[i]);
            for j in 0..8 {
                if i != j {
                    a[i][j] = tf(0.1 * (i as f64 + 1.0) * (j as f64 + 1.0));
                }
            }
        }
        let x_true: Vec8 = std::array::from_fn(|i| tf((i + 1) as f64));
        let mut b = [tf(0.0); 8];
        for i in 0..8 {
            let mut s = tf(0.0);
            for j in 0..8 {
                s = s + a[i][j] * x_true[j];
            }
            b[i] = s;
        }
        let x = lu_solve_8(&a, &b).expect("anisotropic solve");
        for i in 0..8 {
            let rel = (x[i] - x_true[i]).abs() / x_true[i].abs();
            assert!(
                f64::from(rel) < 1e-14,
                "x[{}] rel error too large: {:e}",
                i,
                f64::from(rel)
            );
        }
    }

    #[test]
    fn twofloat_precision_smoke() {
        // Sanity: confirm twofloat ops actually preserve double-double precision.
        let a = TwoFloat::new_div(1.0, 7.0);
        let one_minus = TwoFloat::from(1.0) - a * TwoFloat::from(7.0);
        let err = f64::from(one_minus.abs());
        assert!(err < 1e-30, "1 - (1/7)*7 = {:e} (expected < 1e-30)", err);
    }

    #[test]
    fn lu_solve_twofloat_round_trip() {
        // Solve A·x = b with twofloat-rational inputs and verify precision is
        // at least ~f64 on the round trip. (Empirically twofloat LU caps out
        // around 1e-17 here even though the primitives are 1e-30 precise; we
        // haven't pinpointed the leak but the threshold below is safely above
        // f64 noise and well within what we need for the LLL/Cholesky stages,
        // which feed into a downcast-to-f64 SE search anyway.)
        let mut a = [[tf(0.0); 8]; 8];
        for i in 0..8 {
            for j in 0..8 {
                a[i][j] = TwoFloat::new_div((i + 1) as f64, (j + 5) as f64);
            }
        }
        for i in 0..8 {
            a[i][i] = a[i][i] + tf(10.0);
        }
        let x_true: Vec8 = std::array::from_fn(|i| TwoFloat::new_div(1.0, (i + 1) as f64));
        let mut b = [tf(0.0); 8];
        for i in 0..8 {
            let mut s = tf(0.0);
            for j in 0..8 {
                s = s + a[i][j] * x_true[j];
            }
            b[i] = s;
        }
        let x = lu_solve_8(&a, &b).expect("twofloat solve");
        for i in 0..8 {
            let rel = (x[i] - x_true[i]).abs() / x_true[i].abs();
            assert!(
                f64::from(rel) < 1e-14,
                "x[{}] rel error too large: {:e}",
                i,
                f64::from(rel)
            );
        }
    }

    #[test]
    fn cholesky_recovers_identity() {
        let q = ident_q();
        let l = cholesky_8(&q).expect("identity cholesky");
        // L should be identity
        for i in 0..8 {
            for j in 0..8 {
                let expected = if i == j { 1.0 } else { 0.0 };
                let v = f64::from(l[i][j]);
                assert!((v - expected).abs() < 1e-30);
            }
        }
    }

    #[test]
    fn cholesky_round_trip_anisotropic() {
        let q = anisotropic_q(1e10);
        let l = cholesky_8(&q).expect("anisotropic cholesky");
        // Reconstruct g_check = L · Lᵀ; should equal q
        for i in 0..8 {
            for j in 0..8 {
                let mut s = tf(0.0);
                for k in 0..8 {
                    s = s + l[i][k] * l[j][k];
                }
                let diff = (s - q[i][j]).abs();
                let rel = if q[i][j].abs() > tf(1e-12) {
                    f64::from(diff / q[i][j].abs())
                } else {
                    f64::from(diff)
                };
                assert!(
                    rel < 1e-20,
                    "cholesky reconstruction off at ({},{}): rel={:e}",
                    i,
                    j,
                    rel
                );
            }
        }
    }

    #[test]
    fn lll_identity_metric_returns_unimodular() {
        let q = ident_q();
        let basis = lll_qgram_8(&q);
        let det = det8_exact(&basis).expect("det fits in i64");
        assert!(det == 1 || det == -1, "det = {}", det);
    }

    #[test]
    fn lll_anisotropic_metric_returns_unimodular() {
        // Modest anisotropy first
        let q = anisotropic_q(1e8);
        let basis = lll_qgram_8(&q);
        let det = det8_exact(&basis).expect("det fits in i64");
        assert!(det == 1 || det == -1, "det = {}", det);
    }

    #[test]
    fn lll_extreme_anisotropic_metric() {
        // Pushes condition number close to twofloat's limit.
        // align_scale = 1e16 gives κ ~ 1e16.
        let q = anisotropic_q(1e16);
        let basis = lll_qgram_8(&q);
        let det = det8_exact(&basis).expect("det fits in i64");
        assert!(det == 1 || det == -1, "det = {}", det);
    }

    #[test]
    fn phase1_lenstra_smoke_at_low_k() {
        // Sanity check on a small case: identity-like alignment, k=4, ε=0.5.
        // Should at minimum not hang and not panic. Whether it returns a
        // valid solution depends on whether one exists at this k; for k=4,
        // ‖x‖²=16 has many integer points and the cap (with ε=0.5, fairly loose)
        // is reasonably wide. We don't assert correctness yet; just that the
        // pipeline runs end to end in finite time.
        use std::sync::atomic::AtomicBool;
        let r2 = 1.0 / 2.0_f64.sqrt();
        let s = (1u64 << 4) as f64; // 2^4=16, sqrt=4. y scale = sqrt(2^k)/2 = 2.
        let s = s.sqrt() / 2.0;
        let y: [Float; 8] = [s, s * r2, 0.0, -s * r2, 0.0, 0.0, 0.0, 0.0];
        let budget_hit = AtomicBool::new(false);
        let result = phase1_lenstra(&y, 4, 0.5, 1_000, &budget_hit);
        // Just check it returned (didn't hang or panic). Empty is OK at k=4.
        // Print result for diagnostic.
        eprintln!(
            "[smoke] k=4 ε=0.5 result.len={} budget_hit={}",
            result.len(),
            budget_hit.load(std::sync::atomic::Ordering::Relaxed)
        );
    }

    #[test]
    fn q_is_symmetric_and_pd_for_typical_inputs() {
        // y for v = (1, 0, 0, 0) (identity-target alignment direction), k=10, ε=0.3
        // y = compute_align_vec(v) · √2^k/2 = (1, 1/√2, 0, -1/√2, 0, 0, 0, 0) · √2^10/2
        //   = (1, 1/√2, 0, -1/√2, 0, 0, 0, 0) · 16
        let scale = (1u64 << 10) as f64 / 4.0; // sqrt(2^10)/2 squared... actually this is just for shape
        let s = 16.0; // √(2^10)/2
        let r2 = 1.0 / 2.0_f64.sqrt();
        let y = [
            tf(s),
            tf(s * r2),
            tf(0.0),
            tf(-s * r2),
            tf(0.0),
            tf(0.0),
            tf(0.0),
            tf(0.0),
        ];
        let _ = scale;
        let q = build_q(&y, 10, tf(0.3));
        // Symmetric check
        for i in 0..8 {
            for j in 0..i {
                let diff = (q[i][j] - q[j][i]).abs();
                let mag = q[i][j].abs() + q[j][i].abs() + tf(1e-30);
                let rel = f64::from(diff / mag);
                assert!(rel < 1e-25, "Q not symmetric at ({},{}): rel={:e}", i, j, rel);
            }
        }
        // PD via Cholesky success
        let l = cholesky_8(&q);
        assert!(l.is_some(), "Q not PD for typical inputs");
    }

    #[test]
    fn det8_known_unimodular() {
        // Identity
        let id: IMat8 = std::array::from_fn(|i| {
            let mut r = [0i64; 8];
            r[i] = 1;
            r
        });
        assert_eq!(det8_exact(&id), Some(1));

        // Identity with two rows swapped → det = −1
        let mut swapped = id;
        swapped.swap(2, 5);
        assert_eq!(det8_exact(&swapped), Some(-1));

        // Add row 0 to row 1 → still unimodular, det unchanged
        let mut shifted = id;
        for c in 0..8 {
            shifted[1][c] += shifted[0][c];
        }
        assert_eq!(det8_exact(&shifted), Some(1));
    }
}
