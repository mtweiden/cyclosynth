//! 8D output-sensitive integer enumeration for Clifford+T synthesis (Algorithm 3.6
//! from arXiv:2510.05816). Replaces the 4D-outer + 3D-inner cuboid-with-CS-pruning
//! enumeration in `phase1_enumerate` with a Lenstra-style enumeration on the 8D
//! convex body S_{ε,k}(v⃗) = √2^k · Σ⁻¹(R_ε(v⃗) × D).
//!
//! Strategy:
//! 1. LLL-reduce ℤ⁸ with metric G(u, v) = u·v + λ²·(y·u)(y·v) so the basis is
//!    aligned with the alignment direction y. Vectors with smaller G-norm (more
//!    orthogonal to y) sort first; the alignment-carrier vector ends up last.
//! 2. Schnorr-Euchner enumerate (z₇, z₆, ..., z₀) ∈ ℤ⁸ outward, with sphere bound
//!    ‖x‖² ≤ 2^k where x = Σᵢ zᵢ·bᵢ.
//! 3. At each level prune via partial alignment + sphere remainder. The alignment
//!    constraint |y·x|² ≥ thresh_xy concentrates feasible z₇ near a particular
//!    integer (cap-narrow direction), so enumeration is approximately
//!    output-sensitive.
//!
//! This is a "Lenstra-light" — full Lenstra '83 recursion with flatness-theorem
//! d→(d-1) reduction is the gold standard, but the alignment-weighted basis +
//! tight per-level pruning already captures most of the asymptotic gain in
//! practice. If this doesn't deliver the expected speedup, full recursion is the
//! next step.

use crate::rings::Float;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Bilinear unitarity form B(x) for x = (a₁, b₁, c₁, d₁, a₂, b₂, c₂, d₂) ∈ ℤ⁸.
/// B(x) = 0 iff ‖u‖² = ‖u•‖² (when ‖x‖² = 2^k, this forces unitarity).
///
/// From paper eq 3.13 + the • automorphism (eq 3.8 / 3.11):
///   ‖u‖² + ‖u•‖² = ‖x‖²
///   ‖u‖² − ‖u•‖² = √2 · B(x)
/// where B(x) = a₁b₁ − a₁d₁ + b₁c₁ + c₁d₁ + a₂b₂ − a₂d₂ + b₂c₂ + c₂d₂.
#[inline]
fn bilinear_b(x: &[i64; 8]) -> i64 {
    let (a1, b1, c1, d1) = (x[0], x[1], x[2], x[3]);
    let (a2, b2, c2, d2) = (x[4], x[5], x[6], x[7]);
    a1 * b1 - a1 * d1 + b1 * c1 + c1 * d1
        + a2 * b2 - a2 * d2 + b2 * c2 + c2 * d2
}

/// Top-level entry: enumerate 8D integer points on the sphere ‖x‖² = 2^k that
/// satisfy the bilinear unitarity constraint B(x) = 0 and the alignment
/// |y·x|² ≥ threshold_xy. Returns the first such x via `callback` (which can
/// further validate via phase2-style tests). Honors the per-phase1 budget cap.
///
/// `y` is the 8D alignment vector (compute_align_vec(v) · √2^k/2 in the
/// existing convention); `k` defines target_norm = 2^k; `threshold_xy` is
/// 2^(2k-2)·(1−ε²) (the squared alignment threshold).
pub fn phase1_lenstra(
    y: &[Float; 8],
    k: u32,
    eps: Float,
    max_phase2_calls: u64,
    budget_hit: &AtomicBool,
) -> Vec<[i64; 8]> {
    let target_norm: i64 = 1i64 << k;
    let threshold_xy = (1i64 << (2 * k)) as Float / 4.0 * (1.0 - eps * eps);

    // λ²: balance Euclidean vs alignment in the Gram metric. The alignment term
    // contributes λ²·(y·x)² which can be up to λ²·threshold_xy ≈ λ²·2^(2k-2).
    // Setting λ² ≈ 1 / 2^(2k-2) keeps the alignment term comparable to the
    // Euclidean term for typical x (norm² ~ 2^k). We tune empirically; start
    // with a moderate value.
    let lambda_sq: Float = 1.0;
    let basis = lll_aligned_8d(*y, lambda_sq);
    let r_mat = qr_upper_8(&basis);

    let count = AtomicU64::new(0);
    let cb = |x: [i64; 8]| -> Option<[i64; 8]> {
        // Cap check
        if count.load(Ordering::Relaxed) >= max_phase2_calls {
            budget_hit.store(true, Ordering::Relaxed);
            return None;
        }
        count.fetch_add(1, Ordering::Relaxed);
        // Norm equality (sphere shell, not interior)
        let n: i64 = x.iter().map(|&v| v * v).sum();
        if n != target_norm {
            return None;
        }
        // Bilinear unitarity
        if bilinear_b(&x) != 0 {
            return None;
        }
        Some(x)
    };

    match se_aligned_8d(&basis, &r_mat, *y, target_norm, threshold_xy, cb) {
        Some(x) => vec![x],
        None => Vec::new(),
    }
}

/// Recursive 8D Schnorr-Euchner enumeration of ℤ⁸ lattice points within sphere
/// bound + alignment threshold. The basis is expected to be LLL-reduced with the
/// alignment-weighted Gram metric (see `lll_aligned_8d`).
///
/// The basis matrix has rows = basis vectors. The QR factor `r_mat` is the upper-
/// triangular factor of QR(B^T) where B is `basis` — i.e., for x = ∑ zᵢ·bᵢ we
/// have ‖x‖² = ∑_i (∑_{j≥i} R[i][j]·zⱼ)².
pub fn se_aligned_8d<F>(
    basis: &[[i64; 8]; 8],
    r_mat: &[[Float; 8]; 8],
    y: [Float; 8],
    target_norm: i64,
    thresh_xy: Float,
    mut callback: F,
) -> Option<[i64; 8]>
where
    F: FnMut([i64; 8]) -> Option<[i64; 8]>,
{
    // Per-basis-vector y dot — used to bound partial alignment at each level.
    let y_dot_b: [Float; 8] = std::array::from_fn(|i| {
        (0..8).map(|c| basis[i][c] as Float * y[c]).sum::<Float>()
    });
    // L2-norm² of y, for CS bound on remaining alignment given remaining sphere
    // budget rem: max |y·(remaining contribution)| ≤ √(rem · ‖y‖²).
    let y_norm_sq: Float = y.iter().map(|x| x * x).sum();

    let radius = (target_norm as Float).sqrt();
    let mut z = [0i64; 8];
    let result = std::cell::RefCell::new(None);

    // The recursive search. d counts down: 7, 6, …, 0; at d=usize::MAX (after 0)
    // we have a fully determined z and can test/dispatch.
    fn recurse<F>(
        depth: usize,
        basis: &[[i64; 8]; 8],
        r_mat: &[[Float; 8]; 8],
        y: [Float; 8],
        y_dot_b: &[Float; 8],
        y_norm_sq: Float,
        target_norm: i64,
        thresh_xy: Float,
        radius: Float,
        z: &mut [i64; 8],
        partial_norm_sq: Float,
        partial_align: Float,
        callback: &mut F,
        result: &std::cell::RefCell<Option<[i64; 8]>>,
    ) where
        F: FnMut([i64; 8]) -> Option<[i64; 8]>,
    {
        if result.borrow().is_some() {
            return;
        }
        // Base case: all coordinates fixed; reconstruct x and test.
        if depth == usize::MAX {
            let mut x = [0i64; 8];
            for i in 0..8 {
                if z[i] == 0 {
                    continue;
                }
                for c in 0..8 {
                    x[c] += z[i] * basis[i][c];
                }
            }
            // Sphere check (exact integer)
            let n: i64 = x.iter().map(|&v| v * v).sum();
            if n > target_norm {
                return;
            }
            // Alignment check (exact float)
            let dot: Float = x.iter().zip(y.iter()).map(|(a, b)| *a as Float * b).sum();
            if dot * dot < thresh_xy {
                return;
            }
            if let Some(r) = callback(x) {
                *result.borrow_mut() = Some(r);
            }
            return;
        }

        // Sphere-bound on z[depth]: |R[depth][depth]·z[depth] + tail| ≤ √rem
        // where tail = ∑_{j>depth} R[depth][j]·z[j]. Rem = target_norm - partial_norm_sq.
        let r_dd = r_mat[depth][depth];
        if r_dd.abs() < 1e-12 {
            // Degenerate dimension; just try z[depth] = 0
            z[depth] = 0;
            recurse(
                depth.checked_sub(1).map(|d| d).unwrap_or(usize::MAX),
                basis, r_mat, y, y_dot_b, y_norm_sq, target_norm, thresh_xy, radius,
                z, partial_norm_sq, partial_align, callback, result,
            );
            return;
        }
        let mut tail: Float = 0.0;
        for j in (depth + 1)..8 {
            tail += r_mat[depth][j] * z[j] as Float;
        }
        let rem = (target_norm as Float - partial_norm_sq).max(0.0);
        let rem_sqrt = rem.sqrt();
        let z_center = -tail / r_dd;
        let z_low = ((z_center * r_dd - rem_sqrt) / r_dd).ceil() as i64;
        let z_high = ((z_center * r_dd + rem_sqrt) / r_dd).floor() as i64;
        let z_ci = z_center.round() as i64;
        let z_max_off = (z_high - z_ci).max(z_ci - z_low).max(0);

        // Outward iteration: 0, +1, -1, +2, -2, …
        for raw in 0..=(2 * z_max_off + 1) {
            if result.borrow().is_some() {
                return;
            }
            let off: i64 = if raw == 0 {
                0
            } else if raw % 2 == 1 {
                (raw + 1) / 2
            } else {
                -(raw / 2)
            };
            let zd = z_ci + off;
            if zd < z_low || zd > z_high {
                continue;
            }
            // Update partial sphere norm contribution from this level.
            let level_term = r_dd * zd as Float + tail;
            let new_partial_norm = partial_norm_sq + level_term * level_term;
            if new_partial_norm > target_norm as Float + 1e-6 {
                continue;
            }
            // Partial alignment + CS upper bound on remaining.
            let new_partial_align = partial_align + zd as Float * y_dot_b[depth];
            let rem_after = (target_norm as Float - new_partial_norm).max(0.0);
            let max_remaining_align = (rem_after * y_norm_sq).sqrt();
            let bound = new_partial_align.abs() + max_remaining_align;
            if bound * bound < thresh_xy {
                continue;
            }
            z[depth] = zd;
            let next_depth = depth.checked_sub(1).unwrap_or(usize::MAX);
            recurse(
                next_depth,
                basis, r_mat, y, y_dot_b, y_norm_sq, target_norm, thresh_xy, radius,
                z, new_partial_norm, new_partial_align, callback, result,
            );
        }
    }

    recurse(
        7, basis, r_mat, y, &y_dot_b, y_norm_sq, target_norm, thresh_xy, radius,
        &mut z, 0.0, 0.0, &mut callback, &result,
    );
    result.into_inner()
}

/// Gram-Schmidt orthogonalization of a 8×8 float basis (rows = basis vectors)
/// using the alignment-weighted inner product G(u, v) = u·v + λ²·(y·u)(y·v).
///
/// Returns (mu, gnorm_sq) where mu[i][j] = G(b_i, b_j*) / G(b_j*, b_j*) and
/// gnorm_sq[i] = G(b_i*, b_i*) is the squared G-norm of the i-th GS vector.
fn gs_aligned_8(
    bf: &[[Float; 8]; 8],
    y: [Float; 8],
    lambda_sq: Float,
) -> ([[Float; 8]; 8], [Float; 8]) {
    let mut bs = *bf;
    let mut mu = [[0.0_f64; 8]; 8];
    let mut ydot: [Float; 8] = std::array::from_fn(|i| {
        bf[i].iter().zip(y.iter()).map(|(a, b)| a * b).sum()
    });
    let mut gnorm_sq: [Float; 8] = [0.0; 8];
    for i in 0..8 {
        for j in 0..i {
            let dot_ij: Float = bs[j].iter().zip(bf[i].iter()).map(|(a, b)| a * b).sum::<Float>()
                + lambda_sq * ydot[i] * ydot[j];
            if gnorm_sq[j].abs() < 1e-14 {
                continue;
            }
            mu[i][j] = dot_ij / gnorm_sq[j];
            for k in 0..8 {
                bs[i][k] -= mu[i][j] * bs[j][k];
            }
            ydot[i] -= mu[i][j] * ydot[j];
        }
        let dot_ii: Float = bs[i].iter().map(|x| x * x).sum::<Float>()
            + lambda_sq * ydot[i] * ydot[i];
        gnorm_sq[i] = dot_ii;
    }
    (mu, gnorm_sq)
}

/// LLL basis reduction for ℤ⁸ using the alignment-weighted inner product
/// G(u, v) = u·v + λ²·(y·u)(y·v). After reduction, basis vectors with smaller
/// G-norm sort first; vectors near-orthogonal to y come early, the
/// "alignment-carrier" vector comes last.
pub fn lll_aligned_8d(y: [Float; 8], lambda_sq: Float) -> [[i64; 8]; 8] {
    let mut b: [[i64; 8]; 8] = std::array::from_fn(|i| {
        let mut row = [0i64; 8];
        row[i] = 1;
        row
    });
    let mut bf: [[Float; 8]; 8] = b.map(|r| r.map(|x| x as Float));

    let mut k = 1usize;
    let mut iterations = 0usize;
    let max_iter = 10_000usize; // safety bound for numerical issues
    while k < 8 && iterations < max_iter {
        iterations += 1;
        let (mu, _) = gs_aligned_8(&bf, y, lambda_sq);
        // Size reduction
        for j in (0..k).rev() {
            let r = mu[k][j].round() as i64;
            if r != 0 {
                for c in 0..8 {
                    b[k][c] -= r * b[j][c];
                    bf[k][c] -= r as Float * bf[j][c];
                }
            }
        }
        let (mu2, gnorm) = gs_aligned_8(&bf, y, lambda_sq);
        let delta = 0.75_f64;
        if gnorm[k] >= (delta - mu2[k][k - 1].powi(2)) * gnorm[k - 1] {
            k += 1;
        } else {
            b.swap(k, k - 1);
            bf.swap(k, k - 1);
            k = k.saturating_sub(1).max(1);
        }
    }
    b
}

/// QR decomposition R-factor for an 8×8 integer matrix (rows = basis vectors).
/// Modified Gram-Schmidt on columns of B^T. Returns 8×8 upper-triangular R with
/// nonnegative diagonal.
pub fn qr_upper_8(basis: &[[i64; 8]; 8]) -> [[Float; 8]; 8] {
    let mut cols: [[Float; 8]; 8] = std::array::from_fn(|i| {
        let mut c = [0.0_f64; 8];
        for j in 0..8 {
            c[j] = basis[i][j] as Float;
        }
        c
    });
    let mut r = [[0.0_f64; 8]; 8];
    for i in 0..8 {
        let norm_i: Float = cols[i].iter().map(|x| x * x).sum::<Float>().sqrt();
        r[i][i] = norm_i;
        if norm_i < 1e-14 {
            continue;
        }
        let q_i: [Float; 8] = std::array::from_fn(|j| cols[i][j] / norm_i);
        for j in (i + 1)..8 {
            let dot: Float = q_i.iter().zip(cols[j].iter()).map(|(a, b)| a * b).sum();
            r[i][j] = dot;
            for row in 0..8 {
                cols[j][row] -= dot * q_i[row];
            }
        }
        cols[i] = q_i;
    }
    for i in 0..8 {
        if r[i][i] < 0.0 {
            for j in 0..8 {
                r[i][j] = -r[i][j];
            }
        }
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Compute determinant of an 8×8 i64 matrix (via cofactor expansion / LU).
    fn det8(m: &[[i64; 8]; 8]) -> i128 {
        // Convert to f64 for LU determinant, then round. Adequate for unimodular check.
        let mut a: [[Float; 8]; 8] = m.map(|r| r.map(|x| x as Float));
        let mut sign = 1.0_f64;
        for i in 0..8 {
            // Find pivot
            let mut max_row = i;
            for r in (i + 1)..8 {
                if a[r][i].abs() > a[max_row][i].abs() {
                    max_row = r;
                }
            }
            if max_row != i {
                a.swap(i, max_row);
                sign = -sign;
            }
            if a[i][i].abs() < 1e-14 {
                return 0;
            }
            for r in (i + 1)..8 {
                let factor = a[r][i] / a[i][i];
                for c in i..8 {
                    a[r][c] -= factor * a[i][c];
                }
            }
        }
        let mut det = sign;
        for i in 0..8 {
            det *= a[i][i];
        }
        det.round() as i128
    }

    #[test]
    fn lll_aligned_8d_returns_unimodular_basis() {
        // Standard alignment direction
        let y: [Float; 8] = [0.5, 0.3, -0.2, 0.4, 0.1, -0.5, 0.6, -0.1];
        let lambda_sq = 1000.0_f64;
        let basis = lll_aligned_8d(y, lambda_sq);
        let d = det8(&basis);
        assert!(d == 1 || d == -1, "expected det ±1, got {}", d);
    }

    #[test]
    fn lll_aligned_8d_identity_when_y_zero() {
        let y: [Float; 8] = [0.0; 8];
        let lambda_sq = 1.0;
        let basis = lll_aligned_8d(y, lambda_sq);
        let d = det8(&basis);
        assert!(d == 1 || d == -1);
    }

    #[test]
    fn qr_upper_8_correct_norm_for_identity() {
        let id = std::array::from_fn(|i| {
            let mut row = [0i64; 8];
            row[i] = 1;
            row
        });
        let r = qr_upper_8(&id);
        for i in 0..8 {
            assert!((r[i][i] - 1.0).abs() < 1e-10);
            for j in 0..8 {
                if i != j {
                    assert!(r[i][j].abs() < 1e-10);
                }
            }
        }
    }
}
