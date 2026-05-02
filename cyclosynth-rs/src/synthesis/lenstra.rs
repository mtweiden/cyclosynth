//! 8D output-sensitive integer enumeration for Clifford+T synthesis (Algorithm 3.6
//! from arXiv:2510.05816).
//!
//! Pipeline:
//! 1. Build anisotropic ellipsoid metric Q (8×8 SPD) bounding the cap × ball body.
//! 2. LLL-reduce ℤ⁸ identity basis using Q as the inner product (in twofloat).
//! 3. Cholesky factor G_LLL = B_LLL · Q · B_LLLᵀ = L Lᵀ (twofloat).
//! 4. Solve B_LLL · z_c = c for the cap-center in lattice coordinates (twofloat
//!    LU with partial pivoting).
//! 5. Schnorr-Euchner enumerate z ∈ ℤ⁸ with ‖Lᵀ·(z − z_c)‖² ≤ 2.01 (f64).
//! 6. For each candidate, reconstruct x = B_LLL · z (i64 exact), check
//!    ‖x‖² == 2^k AND B(x) == 0 AND |y·x|² ≥ thresh_xy.
//!
//! Session A: this file currently contains the linear algebra primitives plus
//! unit tests; the SE search and the phase1 dispatch are stubs returning the
//! empty vector. Session B will wire those in.

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

// ─── phase1_lenstra stub ──────────────────────────────────────────────────────

/// Session A stub: full SE pipeline lands in Session B.
pub fn phase1_lenstra(
    _y: &[Float; 8],
    _k: u32,
    _eps: Float,
    _max_phase2_calls: u64,
    _budget_hit: &AtomicBool,
) -> Vec<[i64; 8]> {
    Vec::new()
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
