//! Schnorr-Euchner enumeration over the 8-dimensional integer lattice, plus
//! the candidate-validation helpers that go with it.
//!
//! Inputs (produced by the L²-LLL pipeline):
//!   - The LLL-reduced basis B (`[[i64; 8]; 8]`).
//!   - The Cholesky factor R of the Q-metric Gram matrix on the LLL basis,
//!     in [`twofloat::TwoFloat`] precision (~104 bits).
//!   - The target's projection onto the lattice basis (cap center) in
//!     TwoFloat coordinates.
//!   - The Euclidean Cholesky of B·Bᵀ used for an additional norm-shell
//!     prune (optional).
//!
//! The walk visits each integer 8-tuple `z` whose ‖R·(z − z_c)‖² ≤ bound,
//! invoking a caller-supplied callback for each visit. The callback typically
//! reconstructs the lattice point `x = B·z`, validates it against the
//! synthesis constraints (norm shell, bilinear form, alignment cap), and
//! returns the first candidate that passes.

use std::sync::atomic::{AtomicBool, Ordering};

use rug::{Assign, Float as RFloat};

use i256::i256;

type IMat8 = [[i64; 8]; 8];

/// MPFR precision used by SE. 128 bits ≈ TwoFloat's prior 104-bit budget
/// with margin. f64-only SE is known to break at ε ≤ 1e-5 due to
/// "ghost-node" bound-check noise from squared-norm cancellation; this
/// precision restores the safety margin TwoFloat had.
pub const SE_PREC: u32 = 128;

/// Convert an arbitrary-precision `RFloat` (built at scratch.prec_q for
/// post-LLL Cholesky) to the SE working precision (128 bits). Single
/// allocation, single MPFR conversion.
pub fn rfloat_to_se(r: &RFloat) -> RFloat {
    RFloat::with_val(SE_PREC, r)
}

// ─── Exact 8×8 determinant in i256 (Bareiss) ──────────────────────────────────

/// Compute the determinant of an 8×8 i64 matrix exactly via the Bareiss
/// fraction-free elimination algorithm, working in `i256` to absorb any
/// transient growth from a corrupted-LLL output. Returns `None` if the result
/// doesn't fit in i64; otherwise returns the exact determinant.
///
/// Used after LLL to validate that the output basis is unimodular (det = ±1).
/// A non-unimodular result indicates the GS lost orthogonalization — for the
/// L² pipeline this should never happen at our dimension (d=8), but the check
/// is cheap and catches algorithm bugs early.
pub fn det8_exact(m: &IMat8) -> Option<i64> {
    let mut a: [[i256; 8]; 8] =
        std::array::from_fn(|i| std::array::from_fn(|j| i256::from_i64(m[i][j])));
    let mut sign: i64 = 1;
    let mut prev = i256::from_i64(1);
    let zero = i256::from_i64(0);

    for k in 0..8 {
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
        for i in (k + 1)..8 {
            for j in (k + 1)..8 {
                let lhs = a[i][j] * pivot;
                let rhs = a[i][k] * a[k][j];
                a[i][j] = (lhs - rhs) / prev;
            }
            a[i][k] = zero;
        }
        prev = pivot;
    }
    let det = a[7][7];
    let det_signed = if sign < 0 { -det } else { det };
    let lo = det_signed.as_i128();
    if lo >= i64::MIN as i128 && lo <= i64::MAX as i128 {
        Some(lo as i64)
    } else {
        None
    }
}

// ─── Euclidean Cholesky for the SE norm-shell prune ──────────────────────────

/// Compute the upper-triangular Cholesky factor R of `B·Bᵀ` (Euclidean Gram
/// of the LLL basis) in f64. Used by the SE walk as a partial-prune lower
/// bound: at depth d in the recursion, `Σ_{i ≥ d} (R·z)_i²` is a strict
/// lower bound on the Euclidean ‖x‖² regardless of the remaining `z[< d]`,
/// because each level contributes a non-negative squared term in the GS
/// decomposition. Branches whose lower bound already exceeds `2^k` (the
/// target norm shell) can be cut.
///
/// Returns `None` if the Gram is not numerically positive-definite in f64
/// (extremely rare for an LLL-output basis; would indicate a bug upstream).
pub fn euclidean_cholesky(basis: &IMat8) -> Option<[[f64; 8]; 8]> {
    // Exact integer Gram = B·Bᵀ.
    let mut gram = [[0_i64; 8]; 8];
    for i in 0..8 {
        for j in 0..8 {
            let mut s = 0_i64;
            for k in 0..8 {
                s += basis[i][k] * basis[j][k];
            }
            gram[i][j] = s;
        }
    }
    // f64 Cholesky. For a typical LLL-output basis (entries up to ~2^15),
    // gram entries reach ~2^33 — within f64's 15-digit margin.
    let mut l = [[0.0_f64; 8]; 8];
    for i in 0..8 {
        for j in 0..=i {
            let mut s = gram[i][j] as f64;
            for k in 0..j {
                s -= l[i][k] * l[j][k];
            }
            if i == j {
                if s <= 0.0 {
                    return None;
                }
                l[i][i] = s.sqrt();
            } else {
                l[i][j] = s / l[j][j];
            }
        }
    }
    // Transpose to upper-triangular R = Lᵀ.
    let mut r = [[0.0_f64; 8]; 8];
    for i in 0..8 {
        for j in 0..8 {
            r[i][j] = l[j][i];
        }
    }
    Some(r)
}

// ─── 8D Schnorr-Euchner enumeration ──────────────────────────────────────────

/// Enumerate integer 8-tuples z ∈ ℤ⁸ satisfying ‖R·(z − z_c)‖² ≤ bound, in
/// distance-from-center order, invoking `callback(&z)` at each leaf. Returns
/// the first non-`None` callback result, or `None` if the search exhausts.
///
/// All distance arithmetic uses MPFR `RFloat` at 128-bit precision — the
/// f64-only version was insufficient at extreme ε (Cholesky-diagonal
/// ratios > 10¹⁰ caused "ghost-node" SE blowup from squared-norm noise).
///
/// `r_chol_eucl` is an optional Euclidean-Cholesky factor for an additional
/// norm-shell prune; pass `None` to disable it. With it, branches whose
/// partial Euclidean norm already exceeds `target_norm_eucl` are cut.
///
/// `abort` is checked at every recursion entry — when set, the enumeration
/// returns `None` immediately.
pub fn schnorr_euchner_8d<F>(
    r_chol: &[[RFloat; 8]; 8],
    z_c: &[RFloat; 8],
    bound: &RFloat,
    r_chol_eucl: Option<&[[f64; 8]; 8]>,
    target_norm_eucl: f64,
    abort: &AtomicBool,
    mut callback: F,
) -> Option<[i64; 8]>
where
    F: FnMut(&[i64; 8]) -> Option<[i64; 8]>,
{
    let mut z = [0i64; 8];
    let result = std::cell::RefCell::new(None);
    let zero = RFloat::with_val(SE_PREC, 0.0_f64);

    recurse(
        7,
        r_chol,
        z_c,
        bound,
        r_chol_eucl,
        target_norm_eucl,
        0.0,
        &mut z,
        &zero,
        abort,
        &mut callback,
        &result,
    );
    result.into_inner()
}

#[allow(clippy::too_many_arguments)]
fn recurse<F>(
    depth: i32,
    r_chol: &[[RFloat; 8]; 8],
    z_c: &[RFloat; 8],
    bound: &RFloat,
    r_chol_eucl: Option<&[[f64; 8]; 8]>,
    target_norm_eucl: f64,
    partial_eucl: f64,
    z: &mut [i64; 8],
    partial: &RFloat,
    abort: &AtomicBool,
    callback: &mut F,
    result: &std::cell::RefCell<Option<[i64; 8]>>,
) where
    F: FnMut(&[i64; 8]) -> Option<[i64; 8]>,
{
    if result.borrow().is_some() || abort.load(Ordering::Relaxed) {
        return;
    }
    if depth < 0 {
        if let Some(r) = callback(z) {
            *result.borrow_mut() = Some(r);
        }
        return;
    }
    let d = depth as usize;
    let r_dd = &r_chol[d][d];

    // Per-call scratch pre-allocated once, reused inside the inner loop via
    // assign() patterns. ~10 allocations per recurse call instead of per
    // inner iteration.
    let mut tail = RFloat::with_val(SE_PREC, 0.0_f64);
    let mut tmp = RFloat::with_val(SE_PREC, 0.0_f64);
    let mut diff = RFloat::with_val(SE_PREC, 0.0_f64);
    let mut prod = RFloat::with_val(SE_PREC, 0.0_f64);
    let mut zd_rf = RFloat::with_val(SE_PREC, 0.0_f64);
    let mut level = RFloat::with_val(SE_PREC, 0.0_f64);
    let mut level_sq = RFloat::with_val(SE_PREC, 0.0_f64);
    let mut new_partial = RFloat::with_val(SE_PREC, 0.0_f64);

    // Structural guard against a degenerate diagonal (r_chol PD-ness should
    // exclude this, but tolerate it gracefully).
    tmp.assign(r_dd.clone().abs());
    if tmp.to_f64() < 1e-30 {
        z[d] = z_c[d].to_f64().round() as i64;
        recurse(
            depth - 1, r_chol, z_c, bound, r_chol_eucl, target_norm_eucl,
            partial_eucl, z, partial, abort, callback, result,
        );
        return;
    }

    // tail = Σ_{j > d} R[d][j] · (z[j] − z_c[j])
    for j in (d + 1)..8 {
        diff.assign(z[j] as f64);
        diff -= &z_c[j];
        prod.assign(&r_chol[d][j] * &diff);
        tail += &prod;
    }

    // rem = bound - partial; check >= 0 then sqrt.
    tmp.assign(bound - partial);
    if tmp.to_f64() < 0.0 {
        return;
    }
    let rem_sqrt_f = tmp.to_f64().sqrt();

    // Iteration bounds in f64.
    let r_dd_f = r_dd.to_f64();
    let z_c_d_f = z_c[d].to_f64();
    let center_off = -tail.to_f64() / r_dd_f;
    let span = rem_sqrt_f / r_dd_f.abs();
    let z_low = (z_c_d_f + center_off - span).ceil() as i64;
    let z_high = (z_c_d_f + center_off + span).floor() as i64;
    let z_mid = (z_c_d_f + center_off).round() as i64;
    let max_off = (z_high - z_mid).max(z_mid - z_low).max(0);

    // Pre-compute the Euclidean tail at level d (uses fixed levels j > d).
    let tail_eucl = if let Some(re) = r_chol_eucl {
        let mut t = 0.0_f64;
        for j in (d + 1)..8 {
            t += re[d][j] * (z[j] as f64);
        }
        t
    } else {
        0.0
    };

    // Iterate offsets in distance-from-center order: 0, +1, -1, +2, -2, ...
    for raw in 0..=(2 * max_off + 1) {
        if result.borrow().is_some() || abort.load(Ordering::Relaxed) {
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

        // level = r_dd · (zd − z_c[d]) + tail; squared.
        zd_rf.assign(zd as f64);
        diff.assign(&zd_rf - &z_c[d]);
        level.assign(r_dd * &diff);
        level += &tail;
        level_sq.assign(&level * &level);
        new_partial.assign(partial + &level_sq);
        tmp.assign(&new_partial - bound);
        if tmp.to_f64() > 1e-9 {
            continue;
        }

        // Optional Euclidean-norm prune.
        let new_partial_eucl = if let Some(re) = r_chol_eucl {
            let level_eucl = re[d][d] * (zd as f64) + tail_eucl;
            let p = partial_eucl + level_eucl * level_eucl;
            if p > target_norm_eucl + 1.0 {
                continue;
            }
            p
        } else {
            partial_eucl
        };

        z[d] = zd;
        recurse(
            depth - 1, r_chol, z_c, bound, r_chol_eucl, target_norm_eucl,
            new_partial_eucl, z, &new_partial, abort, callback, result,
        );
    }
}

// ─── Lattice-point reconstruction + bilinear-form check ──────────────────────

/// Reconstruct the lattice point `x = B·z` where `B` is the LLL-reduced
/// basis (rows are basis vectors) and `z` are the SE-output coordinates.
/// Done in i64; for our problem the components stay within i64 by Theorem 2's
/// L³-reduced-basis bound combined with the SE bound.
#[inline]
pub fn reconstruct_x(b_lll: &IMat8, z: &[i64; 8]) -> [i64; 8] {
    let mut x = [0i64; 8];
    for i in 0..8 {
        for j in 0..8 {
            x[j] += z[i] * b_lll[i][j];
        }
    }
    x
}

/// Evaluate the bilinear form `B(x) = a₁b₁ − a₁d₁ + b₁c₁ + c₁d₁ + a₂b₂ −
/// a₂d₂ + b₂c₂ + c₂d₂` where `x = (a₁,b₁,c₁,d₁,a₂,b₂,c₂,d₂)`. This is the
/// unitarity constraint from arXiv:2510.05816 eq (3.10): a candidate is a
/// valid (u₁, u₂) pair iff `B(x) = 0`.
///
/// Returns `i128` to avoid silent overflow at deep ε where x_i can reach
/// ~2^41 and pairwise products hit ~2^82.
#[inline]
pub fn bilinear_b(x: &[i64; 8]) -> i128 {
    let (a1, b1, c1, d1) = (x[0] as i128, x[1] as i128, x[2] as i128, x[3] as i128);
    let (a2, b2, c2, d2) = (x[4] as i128, x[5] as i128, x[6] as i128, x[7] as i128);
    a1 * b1 - a1 * d1 + b1 * c1 + c1 * d1 + a2 * b2 - a2 * d2 + b2 * c2 + c2 * d2
}
