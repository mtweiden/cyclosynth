//! Post-LLL linear algebra: Cholesky and LU on the reduced Gram + basis.
//!
//! `cholesky_f64_8` is the production path — f64 Cholesky on the natural-
//! scale post-LLL Gram. Justified by the LLL invariant κ(G) ≤ 16 at d=8
//! (one bit of conditioning loss per κ doubling, four bits total) — f64's
//! 53-bit mantissa yields ~10⁻¹⁵ absolute error at the SE unit-scale bound
//! check, six orders below SE's 10⁻⁹ tolerance.
//!
//! `cholesky_int_8` and `snapshot_gram_to_mpfr` are the MPFR oracle path,
//! kept so the test suite can validate `cholesky_f64_8` across ε regimes.
//!
//! `lu_solve_int_inplace` solves `Bᵀ · z_c = c` for the cap-center in
//! lattice coordinates, in MPFR at `lu_prec` (≈ 6·log₂(1/ε) bits) — enough
//! precision for SE's 10⁻⁹ tolerance even at ε=1e-8 where post-LLL basis
//! entries reach ~2^41 and pivot ratios run to ~10²⁰.

#![allow(clippy::needless_range_loop)]

use gmp_mpfr_sys::{gmp, mpfr};
use i256::i256;
use rug::Float as RFloat;
use std::ptr::NonNull;

use super::lll::i256_to_f64;
use super::scratch::IMat8;
use super::scratch::{rfv, rfz, IntScratch};

// ─── i256 → MPFR conversion ──────────────────────────────────────────────────

/// Set `dst` (an MPFR variable) to the value of i256 `v`. Zero allocation.
/// Constructs a stack-allocated read-only mpz_t view of the i256 limbs and
/// passes it to `mpfr::set_z`. Safe for all i256 values including 0 and
/// negatives (caller's `dst` must be initialized with a precision adequate
/// to represent the value exactly — 256 bits suffices for any i256). All
/// unsafe code uses only the documented public mpfr/gmp API.
#[inline]
pub fn i256_to_rfloat(v: i256, dst: &mut RFloat) {
    let zero = i256::from_i64(0);
    if v == zero {
        unsafe { mpfr::set_zero(dst.as_raw_mut(), 0) };
        return;
    }
    let neg = v < zero;
    let abs = if neg { -v } else { v };
    let bytes = abs.to_le_bytes();
    let mut limbs: [gmp::limb_t; 4] = std::array::from_fn(|i| {
        let mut buf = [0u8; 8];
        buf.copy_from_slice(&bytes[i * 8..(i + 1) * 8]);
        u64::from_le_bytes(buf) as gmp::limb_t
    });
    // Trim trailing-zero limbs to determine `_mp_size`.
    let mut size: i32 = 4;
    while size > 0 && limbs[(size - 1) as usize] == 0 {
        size -= 1;
    }
    let signed_size = if neg { -size } else { size };
    // Stack mpz_t view: `alloc=0` means "non-owned"; mpfr::set_z only reads
    // from it.
    let mpz = gmp::mpz_t {
        alloc: 0,
        size: signed_size,
        d: unsafe { NonNull::new_unchecked(limbs.as_mut_ptr()) },
    };
    unsafe {
        mpfr::set_z(dst.as_raw_mut(), &mpz as *const _, mpfr::rnd_t::RNDN);
    }
    // limbs goes out of scope; mpfr::set_z has already copied the bits into dst.
}

/// Convert the post-LLL i256 Gram into MPFR `g_post_lll` so the MPFR
/// Cholesky oracle can run on it. The integer Gram is divided by
/// `2^scale_bits` during conversion to recover the natural Q-metric scale.
pub fn snapshot_gram_to_mpfr(scratch: &mut IntScratch) {
    let prec = scratch.prec_q;
    let shift = scratch.scale_bits;
    let mut tmp = rfz(prec);
    for i in 0..8 {
        for j in 0..8 {
            i256_to_rfloat(scratch.gram[i][j], &mut tmp);
            // Recover natural-scale G: ÷ 2^scale_bits.
            if shift > 0 {
                tmp >>= shift as u32;
            } else if shift < 0 {
                tmp <<= (-shift) as u32;
            }
            scratch.g_post_lll[i][j].assign(&tmp);
        }
    }
}

// ─── MPFR Cholesky (oracle) ──────────────────────────────────────────────────

/// MPFR Cholesky on the natural-scale post-LLL Gram. Reference oracle for
/// `cholesky_f64_8`; not used in production. Returns `false` on a
/// non-positive-definite pivot (extremely rare for LLL-output bases).
pub fn cholesky_int_8(scratch: &mut IntScratch) -> bool {
    let prec = scratch.prec_q;
    for i in 0..8 {
        for j in 0..8 {
            scratch.l[i][j].assign(0.0_f64);
        }
    }
    let zero = rfz(prec);
    for i in 0..8 {
        for j in 0..=i {
            scratch.acc.assign(&scratch.g_post_lll[i][j]);
            for k in 0..j {
                scratch.tmp.assign(&scratch.l[i][k] * &scratch.l[j][k]);
                let acc_clone = scratch.acc.clone();
                scratch.acc.assign(&acc_clone - &scratch.tmp);
            }
            if i == j {
                if scratch.acc <= zero {
                    return false;
                }
                let acc_clone = scratch.acc.clone();
                scratch.l[i][i].assign(acc_clone.sqrt());
            } else {
                scratch.tmp2.assign(&scratch.l[j][j]);
                scratch.l[i][j].assign(&scratch.acc / &scratch.tmp2);
            }
        }
    }
    true
}

use rug::Assign;

// ─── MPFR LU solve (production: cap-center → lattice coords) ────────────────

/// Partial-pivoting LU solve of `Bᵀ · z_c = c` in MPFR at `lu_prec`.
/// Reads `scratch.lu_a` (the matrix), `scratch.lu_rhs` (the RHS),
/// writes `scratch.lu_x` (the solution). Returns `false` if the matrix is
/// numerically singular (pivot below 1e-30).
pub fn lu_solve_int_inplace(scratch: &mut IntScratch) -> bool {
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
pub fn cholesky_f64_8(scratch: &mut IntScratch) -> bool {
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
pub fn euclidean_cholesky(basis: &IMat8) -> Option<[[f64; 8]; 8]> {
    // Exact integer Gram = B·Bᵀ in i128 (basis entries can reach ~2^33 in
    // Euclid-pathological frames; i64 products overflowed there).
    let mut gram = [[0_i128; 8]; 8];
    for i in 0..8 {
        for j in 0..8 {
            let mut s = 0_i128;
            for k in 0..8 {
                s += (basis[i][k] as i128) * (basis[j][k] as i128);
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

