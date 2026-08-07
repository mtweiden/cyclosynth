//! Post-LLL linear algebra: Cholesky and LU on the reduced Gram + basis.
//!
//! `cholesky_f64` is the production path — f64 Cholesky on the natural-
//! scale post-LLL Gram. Justified by the LLL invariant κ(G) ≤ 16 at d=8
//! (one bit of conditioning loss per κ doubling, four bits total) — f64's
//! 53-bit mantissa yields ~10⁻¹⁵ absolute error at the SE unit-scale bound
//! check, six orders below SE's 10⁻⁹ tolerance.
//!
//! `cholesky_int` and `snapshot_gram_to_mpfr` are the MPFR oracle path,
//! kept so the test suite can validate `cholesky_f64` across ε regimes.
//!
//! `lu_solve_int_inplace` solves `Bᵀ · z_c = c` for the cap-center in
//! lattice coordinates, in MPFR at `lu_prec` (≈ 6·log₂(1/ε) bits) — enough
//! precision for SE's 10⁻⁹ tolerance even at ε=1e-8 where post-LLL basis
//! entries reach ~2^41 and pivot ratios run to ~10²⁰.

#![allow(clippy::needless_range_loop)]

use i256::i256;

use super::lll::i256_to_f64;
use super::scratch::IMat8;
use super::scratch::{rfv, rfz, IntScratch};
use crate::rings::MpFloat;
use crate::synthesis::lattice::common::i256_to_rfloat;

/// Convert the post-LLL i256 Gram (`scratch.gram`) into a fresh MPFR matrix at
/// `scratch.prec_q` bits, dividing out `2^scale_bits` to recover the natural
/// Q-metric scale `G`. Returned matrix lives on the stack; the caller passes
/// it to `cholesky_int`. Test/diagnostic oracle only — kept out of
/// `IntScratch` so production doesn't carry a second 8x8 MPFR matrix.
#[cfg_attr(not(test), allow(dead_code))] // MPFR oracle path: tests + python+trace diag_inner_cap
pub(crate) fn snapshot_gram_to_mpfr(scratch: &IntScratch) -> [[MpFloat; 8]; 8] {
    let prec = scratch.prec_q;
    let shift = scratch.scale_bits;
    let mut tmp = rfz(prec);
    let mut g_post: [[MpFloat; 8]; 8] =
        std::array::from_fn(|_| std::array::from_fn(|_| rfz(prec)));
    for i in 0..8 {
        for j in 0..8 {
            i256_to_rfloat(scratch.gram[i][j], &mut tmp);
            // Recover natural-scale G: ÷ 2^scale_bits.
            if shift > 0 {
                tmp >>= shift as u32;
            } else if shift < 0 {
                tmp <<= (-shift) as u32;
            }
            g_post[i][j].assign(&tmp);
        }
    }
    g_post
}

// ─── MPFR Cholesky (oracle) ──────────────────────────────────────────────────

/// MPFR Cholesky on `g_post` (natural-scale post-LLL Gram): lower-triangular
/// factor at `scratch.prec_q` bits, or `None` on a non-positive-definite pivot
/// (extremely rare for LLL-output bases). Reference oracle for `cholesky_f64`;
/// not used in production. The fresh stack matrices are kept out of
/// `IntScratch` so production doesn't carry a second pair of 8x8 MPFR matrices.
#[cfg_attr(not(test), allow(dead_code))] // MPFR oracle path: tests + python+trace diag_inner_cap
pub(crate) fn cholesky_int(
    scratch: &IntScratch,
    g_post: &[[MpFloat; 8]; 8],
) -> Option<[[MpFloat; 8]; 8]> {
    let prec = scratch.prec_q;
    let mut l: [[MpFloat; 8]; 8] =
        std::array::from_fn(|_| std::array::from_fn(|_| rfz(prec)));
    let zero = rfz(prec);
    let mut acc = rfz(prec);
    let mut tmp = rfz(prec);
    for i in 0..8 {
        for j in 0..=i {
            acc.assign(&g_post[i][j]);
            for k in 0..j {
                tmp.assign(&l[i][k] * &l[j][k]);
                let acc_clone = acc.clone();
                acc.assign(&acc_clone - &tmp);
            }
            if i == j {
                if acc <= zero {
                    return None;
                }
                let acc_clone = acc.clone();
                l[i][i].assign(acc_clone.sqrt());
            } else {
                let denom = l[j][j].clone();
                l[i][j].assign(&acc / &denom);
            }
        }
    }
    Some(l)
}

use rug::Assign;

// ─── MPFR LU solve (production: cap-center → lattice coords) ────────────────

/// Partial-pivoting LU solve of `Bᵀ · z_c = c` in MPFR at `lu_prec`.
/// Reads `scratch.lu_a` (the matrix), `scratch.lu_rhs` (the RHS),
/// writes `scratch.lu_x` (the solution). Returns `false` if the matrix is
/// numerically singular (pivot below 1e-30).
pub(crate) fn lu_solve_int_inplace(scratch: &mut IntScratch) -> bool {
    let tol = rfv(scratch.lu_prec, 1e-30);

    for k in 0..8 {
        let mut piv = k;
        let mut piv_abs = scratch.lu_a[k][k].clone().abs();
        for i in (k + 1)..8 {
            let v = scratch.lu_a[i][k].clone().abs();
            if v > piv_abs {
                piv_abs = v;
                piv = i;
            }
        }
        if piv_abs < tol {
            return false;
        }
        if piv != k {
            scratch.lu_a.swap(k, piv);
            scratch.lu_rhs.swap(k, piv);
        }
        for i in (k + 1)..8 {
            scratch.lu_tmp.assign(&scratch.lu_a[i][k] / &scratch.lu_a[k][k]);
            let factor = scratch.lu_tmp.clone();
            // a[i][j] -= factor · a[k][j] for j in k..8.
            // Avoid simultaneous &mut borrows on rows i and k.
            let (row_i, row_k) = if i < k {
                let (head, tail) = scratch.lu_a.split_at_mut(k);
                (&mut head[i], &mut tail[0])
            } else {
                let (head, tail) = scratch.lu_a.split_at_mut(i);
                (&mut tail[0], &mut head[k])
            };
            for j in k..8 {
                scratch.lu_tmp.assign(&factor * &row_k[j]);
                let cur = row_i[j].clone();
                row_i[j].assign(&cur - &scratch.lu_tmp);
            }
            scratch.lu_tmp.assign(&factor * &scratch.lu_rhs[k]);
            let rhs_i_cur = scratch.lu_rhs[i].clone();
            scratch.lu_rhs[i].assign(&rhs_i_cur - &scratch.lu_tmp);
        }
    }
    for i in (0..8).rev() {
        scratch.lu_acc.assign(&scratch.lu_rhs[i]);
        for j in (i + 1)..8 {
            scratch.lu_tmp.assign(&scratch.lu_a[i][j] * &scratch.lu_x[j]);
            let cur = scratch.lu_acc.clone();
            scratch.lu_acc.assign(&cur - &scratch.lu_tmp);
        }
        let acc_clone = scratch.lu_acc.clone();
        scratch.lu_x[i].assign(&acc_clone / &scratch.lu_a[i][i]);
    }
    true
}

// ─── f64 Cholesky (production) ───────────────────────────────────────────────

/// Run f64 Cholesky on the natural-scale post-LLL Gram, reading the i256
/// Gram via `i256_to_f64` with `2^-scale_bits` (an exponent shift, no
/// precision cost) folded into the conversion. Output: lower-triangular
/// `scratch.l_f64`. Returns `false` on a non-positive-definite pivot
/// (extremely rare for LLL-output bases — would indicate an upstream bug).
///
/// f64 is sufficient because the L³-reduction invariant after L²-LLL bounds
/// `κ(G) ≤ (4/3)^(d-1) ≤ 16` at d=8 (paper Theorem 3 corollary). The
/// reduced Gram is well-conditioned even when the input Q has κ ≈ 2^137 at
/// ε=1e-10, and the SE walk's MPFR-128 bound check tolerance (~10⁻⁹) is six
/// orders above the f64 Cholesky's ~10⁻¹⁵ absolute error at unit scale.
pub(crate) fn cholesky_f64(scratch: &mut IntScratch) -> bool {
    let scale = 2.0_f64.powi(-scratch.scale_bits);
    let mut g = [[0.0_f64; 8]; 8];
    for i in 0..8 {
        for j in 0..=i {
            g[i][j] = i256_to_f64(scratch.gram[i][j]) * scale;
        }
    }
    for i in 0..8 {
        for j in 0..8 {
            scratch.l_f64[i][j] = 0.0;
        }
    }
    for i in 0..8 {
        for j in 0..=i {
            let mut s = g[i][j];
            for k in 0..j {
                s -= scratch.l_f64[i][k] * scratch.l_f64[j][k];
            }
            if i == j {
                if s <= 0.0 {
                    return false;
                }
                scratch.l_f64[i][i] = s.sqrt();
            } else {
                scratch.l_f64[i][j] = s / scratch.l_f64[j][j];
            }
        }
    }
    true
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
pub(crate) fn det_exact(m: &IMat8) -> Option<i64> {
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
    if lo >= i128::from(i64::MIN) && lo <= i128::from(i64::MAX) {
        #[allow(clippy::cast_possible_truncation)] // range-checked above
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
/// Returns `None` — DISABLING the (optional) prune — when the factor cannot
/// be trusted at the prune's `target + 1.0` absolute slack:
///
/// - The Gram is not numerically positive-definite in f64.
/// - A Gram diagonal exceeds 2^53 (f64 integer-exactness limit).
/// - The Cholesky diagonal ratio exceeds 1e6: the basis is LLL-reduced
///   in the Q metric, not the Euclidean one, so Q-short vectors can be
///   Euclid-long — there the f64 partial sums cancel at 1e18 scale and
///   the prune cuts branches containing TRUE solutions (masked pre-
///   dedup by the 8× coset-mate redundancy).
pub(crate) fn euclidean_cholesky(basis: &IMat8) -> Option<[[f64; 8]; 8]> {
    // Exact integer Gram = B·Bᵀ in i128 (basis entries can reach ~2^33 in
    // Euclid-pathological frames, where i64 products would overflow).
    let mut gram = [[0_i128; 8]; 8];
    for i in 0..8 {
        for j in 0..8 {
            let mut s = 0_i128;
            for k in 0..8 {
                s += i128::from(basis[i][k]) * i128::from(basis[j][k]);
            }
            gram[i][j] = s;
        }
    }
    // Trust guard 1: every Gram entry must be exactly representable in f64.
    for row in &gram {
        for &v in row {
            if v.unsigned_abs() > (1u128 << 53) {
                return None;
            }
        }
    }
    let mut l = [[0.0_f64; 8]; 8];
    // f64 GS is the designed approximate channel: SE brackets carry 1e-9
    // relative slack and the exact leaf filter arbitrates (diag_inner_cap probe).
    #[allow(clippy::cast_precision_loss)]
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
    // Trust guard 2: diagonal-ratio condition estimate. Beyond ~1e6 the
    // f64 partial sums are no longer accurate to the prune's O(1) slack.
    let mut dmin = f64::INFINITY;
    let mut dmax = 0.0_f64;
    for (i, row) in l.iter().enumerate() {
        dmin = dmin.min(row[i]);
        dmax = dmax.max(row[i]);
    }
    if dmax > 1e6 * dmin {
        return None;
    }
    let mut r = [[0.0_f64; 8]; 8]; // R = Lᵀ
    for i in 0..8 {
        for j in 0..8 {
            r[i][j] = l[j][i];
        }
    }
    Some(r)
}

