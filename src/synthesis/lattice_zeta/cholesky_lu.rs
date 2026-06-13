//! Post-LLL linear algebra at d=16: Cholesky and LU on the reduced Gram +
//! basis. Mirrors `super::super::lattice::cholesky_lu` (the d=8 path) with
//! buffers and loops dimension-bumped to 16.
//!
//! Production path:
//!   - `cholesky_f64_16` — f64 Cholesky on the natural-scale post-LLL Gram,
//!     reading `scratch.gram` (i256) via `i256_to_f64` with `2^-scale_bits`
//!     folded into the conversion. Output: lower-triangular `scratch.l_f64`.
//!   - `lu_solve_int_inplace_16` — partial-pivoting LU at `scratch.lu_prec`
//!     bits, solving `Bᵀ · z = c` for the cap-center in lattice coords,
//!     where `B = scratch.basis` (i64) and `c = scratch.c` (MPFR). The basis
//!     and RHS are loaded into `scratch.lu_a` / `scratch.lu_rhs` first, then
//!     the standard partial-pivoting algorithm runs in place. Solution lands
//!     in `scratch.lu_x`.
//!
//! Test-oracle path:
//!   - `snapshot_gram_to_mpfr_16` — convert `scratch.gram` (i256) to a fresh
//!     MPFR matrix. Used by the integer-Cholesky oracle below.
//!   - `cholesky_int_16` — MPFR Cholesky on the natural-scale post-LLL Gram.
//!     Reference oracle for `cholesky_f64_16`. Returns `None` on a
//!     non-positive-definite pivot.

#![allow(clippy::needless_range_loop)]

use i256::i256;
use rug::{Assign, Float as RFloat};

use super::scratch::{rfv, rfz, IntScratch16};
use crate::synthesis::lattice::omega::cholesky_lu::i256_to_rfloat;
use crate::synthesis::lattice_common::i256_to_f64;


/// Convert the post-LLL i256 Gram (`scratch.gram`) into a fresh MPFR matrix
/// at `scratch.prec_q` bits, dividing out `2^scale_bits` so the result is the
/// natural-scale `G`. Returned matrix lives on the stack; the caller passes
/// it to `cholesky_int_16`.
pub fn snapshot_gram_to_mpfr_16(scratch: &mut IntScratch16) -> [[RFloat; 16]; 16] {
    let prec = scratch.prec_q;
    let shift = scratch.scale_bits;
    let mut tmp = rfz(prec);
    let mut g_post: [[RFloat; 16]; 16] =
        std::array::from_fn(|_| std::array::from_fn(|_| rfz(prec)));
    for i in 0..16 {
        for j in 0..16 {
            i256_to_rfloat(scratch.gram[i][j], &mut tmp);
            // Divide by 2^scale_bits to recover natural-scale G.
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


/// MPFR Cholesky on `g_post` (natural-scale post-LLL Gram): lower-triangular
/// factor at `scratch.prec_q` bits, or `None` on a non-positive-definite
/// pivot. Test-only oracle for `cholesky_f64_16`; the fresh stack matrix is
/// kept out of `IntScratch16` so production doesn't carry a second 16x16.
pub fn cholesky_int_16(
    scratch: &mut IntScratch16,
    g_post: &[[RFloat; 16]; 16],
) -> Option<[[RFloat; 16]; 16]> {
    let prec = scratch.prec_q;
    let mut l: [[RFloat; 16]; 16] =
        std::array::from_fn(|_| std::array::from_fn(|_| rfz(prec)));
    let zero = rfz(prec);
    let mut acc = rfz(prec);
    let mut tmp = rfz(prec);
    for i in 0..16 {
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


/// Solve `Bᵀ · z = c` in MPFR at `scratch.lu_prec` bits, where
/// `B = scratch.basis` (i64) and `c = scratch.c` (MPFR). The matrix and RHS
/// are first loaded into `scratch.lu_a` (with `lu_a[i][j] = basis[j][i]`,
/// i.e. `Bᵀ`) and `scratch.lu_rhs`, then partial-pivoting LU runs in place.
/// Solution lands in `scratch.lu_x`. Returns `false` if the matrix is
/// numerically singular (pivot below 1e-30).
pub fn lu_solve_int_inplace_16(scratch: &mut IntScratch16) -> bool {
    // Load lu_a = Bᵀ and lu_rhs = c. Convert i64 → MPFR via f64 (every i64
    // basis entry post-LLL is well within f64's 53-bit exact range — the
    // basis stays under 2^41 even at deep ε, and at moderate ε ≤ 2^15).
    let prec = scratch.lu_prec;
    for i in 0..16 {
        for j in 0..16 {
            scratch.lu_a[i][j].assign(rfv(prec, scratch.basis[j][i] as f64));
        }
        scratch.lu_rhs[i].assign(&scratch.c[i]);
    }

    let tol = rfv(prec, 1e-30);

    for k in 0..16 {
        let mut piv = k;
        let mut piv_abs = scratch.lu_a[k][k].clone().abs();
        for i in (k + 1)..16 {
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
        for i in (k + 1)..16 {
            scratch
                .lu_tmp
                .assign(&scratch.lu_a[i][k] / &scratch.lu_a[k][k]);
            let factor = scratch.lu_tmp.clone();
            // a[i][j] -= factor · a[k][j] for j in k..16.
            // Avoid simultaneous &mut borrows on rows i and k.
            let (row_i, row_k) = if i < k {
                let (head, tail) = scratch.lu_a.split_at_mut(k);
                (&mut head[i], &mut tail[0])
            } else {
                let (head, tail) = scratch.lu_a.split_at_mut(i);
                (&mut tail[0], &mut head[k])
            };
            for j in k..16 {
                scratch.lu_tmp.assign(&factor * &row_k[j]);
                let cur = row_i[j].clone();
                row_i[j].assign(&cur - &scratch.lu_tmp);
            }
            scratch.lu_tmp.assign(&factor * &scratch.lu_rhs[k]);
            let rhs_i_cur = scratch.lu_rhs[i].clone();
            scratch.lu_rhs[i].assign(&rhs_i_cur - &scratch.lu_tmp);
        }
    }
    for i in (0..16).rev() {
        scratch.lu_acc.assign(&scratch.lu_rhs[i]);
        for j in (i + 1)..16 {
            scratch
                .lu_tmp
                .assign(&scratch.lu_a[i][j] * &scratch.lu_x[j]);
            let cur = scratch.lu_acc.clone();
            scratch.lu_acc.assign(&cur - &scratch.lu_tmp);
        }
        let acc_clone = scratch.lu_acc.clone();
        scratch.lu_x[i].assign(&acc_clone / &scratch.lu_a[i][i]);
    }
    true
}


/// Run f64 Cholesky on the natural-scale post-LLL Gram, reading the i256
/// Gram via `i256_to_f64` with `2^-scale_bits` (an exponent shift, no
/// precision cost) folded into the conversion. Output: lower-triangular
/// `scratch.l_f64`. Returns `false` on a non-positive-definite pivot
/// (extremely rare for LLL-output bases — would indicate an upstream bug).
///
/// f64 is sufficient because the L³-reduction invariant after L²-LLL bounds
/// `κ(G) ≤ (4/3)^15 ≈ 75` at d=16 (~6 bits of conditioning loss). f64's
/// 53-bit mantissa absorbs that with wide margin, yielding ~10⁻¹⁴ error at
/// the SE bound check — five orders below SE's 10⁻⁹ tolerance.
pub fn cholesky_f64_16(scratch: &mut IntScratch16) -> bool {
    let scale = 2.0_f64.powi(-scratch.scale_bits);
    let mut g = [[0.0_f64; 16]; 16];
    for i in 0..16 {
        for j in 0..=i {
            g[i][j] = i256_to_f64(scratch.gram[i][j]) * scale;
        }
    }
    for i in 0..16 {
        for j in 0..16 {
            scratch.l_f64[i][j] = 0.0;
        }
    }
    for i in 0..16 {
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

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: install a rational integer Gram on the scratch from an f64
    /// matrix `g_nat` at `scale_bits = B`, i.e. `gram[i][j] = round(2^B · g_nat[i][j])`.
    fn install_gram_from_f64(
        scratch: &mut IntScratch16,
        g_nat: &[[f64; 16]; 16],
        scale_bits: i32,
    ) {
        let scale = 2.0_f64.powi(scale_bits);
        for i in 0..16 {
            for j in 0..16 {
                let scaled = (g_nat[i][j] * scale).round();
                // Use i64 → i256 (suffices for our small test matrices).
                scratch.gram[i][j] = i256::from_i64(scaled as i64);
            }
        }
        scratch.scale_bits = scale_bits;
    }

    fn identity_16() -> [[f64; 16]; 16] {
        let mut m = [[0.0_f64; 16]; 16];
        for i in 0..16 {
            m[i][i] = 1.0;
        }
        m
    }

    #[test]
    fn cholesky_f64_16_round_trip() {
        // Construct a known PSD 16x16 matrix: G = 4·I_16. (Matches the
        // structure of Σᵀ·Σ for the Z[ζ_16] embedding.) Cholesky factor
        // should be 2·I_16.
        let mut g = [[0.0_f64; 16]; 16];
        for i in 0..16 {
            g[i][i] = 4.0;
        }
        let mut s = IntScratch16::new(1e-3);
        install_gram_from_f64(&mut s, &g, 8);
        assert!(cholesky_f64_16(&mut s));
        // Verify Lᵀ L = G to f64 precision.
        for i in 0..16 {
            for j in 0..16 {
                let mut sum = 0.0_f64;
                for k in 0..16 {
                    sum += s.l_f64[i][k] * s.l_f64[j][k];
                }
                let diff = (sum - g[i][j]).abs();
                assert!(
                    diff < 1e-12,
                    "L Lᵀ mismatch at ({i},{j}): got {sum}, expected {}",
                    g[i][j]
                );
            }
        }
        // Sanity: L should be 2·I.
        for i in 0..16 {
            assert!((s.l_f64[i][i] - 2.0).abs() < 1e-12);
        }
    }

    #[test]
    fn cholesky_int_16_on_identity() {
        // Identity Gram → identity Cholesky factor.
        let g = identity_16();
        let mut s = IntScratch16::new(1e-3);
        install_gram_from_f64(&mut s, &g, 8);
        let g_post = snapshot_gram_to_mpfr_16(&mut s);
        let l = cholesky_int_16(&mut s, &g_post).expect("identity should be PSD");
        for i in 0..16 {
            for j in 0..16 {
                let v = l[i][j].to_f64();
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (v - expected).abs() < 1e-12,
                    "L[{i}][{j}] = {v}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn cholesky_int_16_rejects_non_psd() {
        // Diagonal with one negative entry: clearly non-PSD.
        let mut g = identity_16();
        g[3][3] = -1.0;
        let mut s = IntScratch16::new(1e-3);
        install_gram_from_f64(&mut s, &g, 8);
        let g_post = snapshot_gram_to_mpfr_16(&mut s);
        let res = cholesky_int_16(&mut s, &g_post);
        assert!(res.is_none(), "non-PSD Gram should reject");
    }

    #[test]
    fn lu_solve_int_inplace_16_identity_basis() {
        // B = I_16, c chosen, expect z = c (as an f64 vector).
        let mut s = IntScratch16::new(1e-3);
        s.basis = super::super::scratch::identity_basis_16();
        let c_vals: [f64; 16] = std::array::from_fn(|i| (i as f64) - 7.5);
        for i in 0..16 {
            s.c[i] = rfv(s.prec_q, c_vals[i]);
        }
        assert!(lu_solve_int_inplace_16(&mut s));
        for i in 0..16 {
            let z = s.lu_x[i].to_f64();
            assert!(
                (z - c_vals[i]).abs() < 1e-12,
                "z[{i}] = {z}, expected {}",
                c_vals[i]
            );
        }
    }

    #[test]
    fn lu_solve_int_inplace_16_against_known_basis() {
        // B is a permutation matrix (cyclic shift): row i has a single 1 at
        // column (i + 1) mod 16. Then Bᵀ has row i with a single 1 at
        // column (i - 1) mod 16. Solving Bᵀ · z = c gives z[i] = c[(i+1) % 16]
        // (since Bᵀ z[i] = z[(i-1) % 16] = c[i] ⇒ z[i] = c[(i+1) % 16]).
        let mut s = IntScratch16::new(1e-3);
        let mut b = [[0i64; 16]; 16];
        for i in 0..16 {
            b[i][(i + 1) % 16] = 1;
        }
        s.basis = b;
        let c_vals: [f64; 16] = std::array::from_fn(|i| (i as f64 + 1.0) * 0.25);
        for i in 0..16 {
            s.c[i] = rfv(s.prec_q, c_vals[i]);
        }
        assert!(lu_solve_int_inplace_16(&mut s));
        for i in 0..16 {
            let expected = c_vals[(i + 1) % 16];
            let z = s.lu_x[i].to_f64();
            assert!(
                (z - expected).abs() < 1e-12,
                "z[{i}] = {z}, expected {expected}"
            );
        }
    }

    #[test]
    fn snapshot_gram_to_mpfr_16_round_trip() {
        // i256 → MPFR → f64 round-trip preserves values to f64 precision for
        // moderate-sized inputs. Build a Gram with values in [-2^20, 2^20]
        // and `scale_bits = 0` so MPFR ÷ 1 is a no-op; check round-trip.
        let mut s = IntScratch16::new(1e-3);
        for i in 0..16 {
            for j in 0..16 {
                let val: i64 = ((i as i64) * 31 + (j as i64) * 17 - 200) << 10;
                s.gram[i][j] = i256::from_i64(val);
            }
        }
        s.scale_bits = 0;
        let g_post = snapshot_gram_to_mpfr_16(&mut s);
        for i in 0..16 {
            for j in 0..16 {
                let expected = (((i as i64) * 31 + (j as i64) * 17 - 200) << 10) as f64;
                let got = g_post[i][j].to_f64();
                assert!(
                    (got - expected).abs() < 1e-9,
                    "snapshot[{i}][{j}] = {got}, expected {expected}"
                );
            }
        }
    }
}


/// Exact integer determinant of a 16×16 i64 matrix via Bareiss
/// fraction-free elimination, working in i128. Returns `None` if the result
/// (or any intermediate) doesn't fit in i64; otherwise returns the exact
/// determinant.
///
/// Used after LLL to validate that the output basis is unimodular (det = ±1).
/// A non-unimodular result indicates the GS lost orthogonalization — for the
/// L²-LLL pipeline this should never happen at d=16, but the check is cheap
/// and catches algorithm bugs early.
///
/// **Overflow note**: At d=16 with post-LLL basis entries up to ~2^41 (deep
/// ε), Bareiss intermediates can transiently exceed i64. We use i128
/// throughout; if any intermediate value exceeds i128 range the result is
/// `None` (saturation). For unimodular bases the *final* det is ±1 so there
/// is no issue, but spurious overflow during elimination is possible at
/// pathological inputs.
pub fn det16_exact(m: &[[i64; 16]; 16]) -> Option<i64> {
    let mut a: [[i128; 16]; 16] =
        std::array::from_fn(|i| std::array::from_fn(|j| m[i][j] as i128));
    let mut sign: i128 = 1;
    let mut prev: i128 = 1;

    for k in 0..16 {
        if a[k][k] == 0 {
            let mut found = false;
            for i in (k + 1)..16 {
                if a[i][k] != 0 {
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
        for i in (k + 1)..16 {
            for j in (k + 1)..16 {
                let lhs = a[i][j].checked_mul(pivot)?;
                let rhs = a[i][k].checked_mul(a[k][j])?;
                let diff = lhs.checked_sub(rhs)?;
                // diff is divisible by prev exactly (Bareiss invariant).
                a[i][j] = diff / prev;
            }
            a[i][k] = 0;
        }
        prev = pivot;
    }
    let det = a[15][15].checked_mul(sign)?;
    if det >= i64::MIN as i128 && det <= i64::MAX as i128 {
        Some(det as i64)
    } else {
        None
    }
}


/// Upper-triangular Cholesky factor `R` of the Euclidean Gram, as an f64
/// snapshot plus a double-double `(hi, lo)` projection of the same factor.
pub type CholeskyDual16 = ([[f64; 16]; 16], [[(f64, f64); 16]; 16]);

/// MPFR-128 Cholesky of the Euclidean Gram `B·Bᵀ`, returning the
/// upper-triangular factor R (`Rᵀ·R = B·Bᵀ`) as both an f64 snapshot (the SE
/// walk's primary f64 prune) and a double-double projection (the verify path
/// gated by [`set_verify_prune_mpfr`]). The factorization runs at MPFR-128 so
/// the per-leaf `‖R·z‖²` accumulator drifts only by f64 round-off, not by
/// f64-Cholesky error — which at deep k (Gram ~2^34+) reaches 0.1%+ and
/// corrupts the prune threshold. 106-bit was tried but gave rank-deficient
/// false alarms at small lde where `s -= l[i][k]*l[j][k]` cancellation is
/// tight. `None` if the Gram is not PD (rank-deficient basis = upstream bug).
pub fn euclidean_cholesky_16_mpfr_dual(basis: &[[i64; 16]; 16]) -> Option<CholeskyDual16> {
    const PREC: u32 = 128;
    let mut gram = [[0_i128; 16]; 16];
    for i in 0..16 {
        for j in 0..16 {
            let mut s = 0_i128;
            for k in 0..16 {
                s += (basis[i][k] as i128) * (basis[j][k] as i128);
            }
            gram[i][j] = s;
        }
    }
    let mut g: [[rug::Float; 16]; 16] = std::array::from_fn(|_| {
        std::array::from_fn(|_| rug::Float::with_val(PREC, 0.0))
    });
    for i in 0..16 {
        for j in 0..16 {
            g[i][j] = i128_to_mpfr(gram[i][j], PREC);
        }
    }
    mpfr_cholesky_dual_16(&g)
}

/// MPFR-128 Cholesky of the post-LLL **Q-metric** Gram (`scratch.gram` as
/// i256, scaled by `2^scale_bits`), returning the upper-triangular factor
/// R (`Rᵀ·R = G`) as an f64 snapshot plus its double-double projection —
/// the Q-side mirror of [`euclidean_cholesky_16_mpfr_dual`].
///
/// Consumed by the deep-ε dd-verified Q bracket: the f64 snapshot replaces
/// the `cholesky_f64_16` factor as the SE walk's `l_upper` (strictly more
/// accurate — the f64 Cholesky factorization error was one of the channels
/// behind the ε=1.5e-8 partial-Q overshoot), and the dd projection drives
/// the incremental dd partial-Q that makes the bound-1.5 prune decisions
/// sound (docs/bound_sq_soundness.md, docs/w_q_bracket_notes.md).
///
/// The i256 Gram is exact through both LLL and BKZ (gram-update
/// invariant), so this factors the same matrix `cholesky_f64_16` reads —
/// at 128-bit precision instead of f64. Returns `None` if the Gram is not
/// positive-definite (rank-deficient basis — upstream-bug territory).
pub fn q_cholesky_16_mpfr_dual(
    gram: &[[i256; 16]; 16],
    scale_bits: i32,
) -> Option<CholeskyDual16> {
    const PREC: u32 = 128;
    let mut tmp = rug::Float::with_val(PREC, 0.0);
    let mut g: [[rug::Float; 16]; 16] = std::array::from_fn(|_| {
        std::array::from_fn(|_| rug::Float::with_val(PREC, 0.0))
    });
    for i in 0..16 {
        for j in 0..16 {
            crate::synthesis::lattice::omega::cholesky_lu::i256_to_rfloat(gram[i][j], &mut tmp);
            // Divide by 2^scale_bits (exponent shift — no precision cost)
            // to recover the natural-scale Q-metric Gram.
            if scale_bits > 0 {
                tmp >>= scale_bits as u32;
            } else if scale_bits < 0 {
                tmp <<= (-scale_bits) as u32;
            }
            g[i][j] = tmp.clone();
        }
    }
    mpfr_cholesky_dual_16(&g)
}

/// Shared MPFR-128 Cholesky + dual projection: factor `g` (must be at
/// 128-bit precision) into lower-triangular L (`L·Lᵀ = g`), transpose to
/// upper-triangular R, and emit (f64 snapshot, dd projection). Op order
/// and precision are identical for the Euclidean and Q-metric callers so
/// the two dd factors carry the same (validated) error model.
fn mpfr_cholesky_dual_16(g: &[[rug::Float; 16]; 16]) -> Option<CholeskyDual16> {
    use rug::Float;
    const PREC: u32 = 128;
    let mut l: [[Float; 16]; 16] = std::array::from_fn(|_| {
        std::array::from_fn(|_| Float::with_val(PREC, 0.0))
    });
    for i in 0..16 {
        for j in 0..=i {
            let mut s = g[i][j].clone();
            for k in 0..j {
                let prod = Float::with_val(PREC, &l[i][k] * &l[j][k]);
                s -= &prod;
            }
            if i == j {
                if s.is_zero() || s.is_sign_negative() {
                    return None;
                }
                l[i][i] = s.sqrt();
            } else {
                let q = Float::with_val(PREC, &s / &l[j][j]);
                l[i][j] = q;
            }
        }
    }
    // R = L^T (upper-triangular). Snapshot to f64 (used by f64 prune) and
    // project to dd (used by verify_partial_dd_exceeds / the dd Q bracket).
    // The MPFR factor itself is consumed here; the dd projection is the
    // kept output.
    let mut r_f64 = [[0.0_f64; 16]; 16];
    let mut r_dd = [[(0.0_f64, 0.0_f64); 16]; 16];
    for i in 0..16 {
        for j in 0..16 {
            let rij = &l[j][i];
            let hi = rij.to_f64();
            let mut lo_f = Float::with_val(PREC, rij);
            lo_f -= hi;
            let lo = lo_f.to_f64();
            r_f64[i][j] = hi;
            r_dd[i][j] = (hi, lo);
        }
    }
    Some((r_f64, r_dd))
}

/// Convert i128 → MPFR Float, lossless. rug doesn't accept i128 directly.
fn i128_to_mpfr(v: i128, prec: u32) -> rug::Float {
    use rug::Float;
    let neg = v < 0;
    let abs = if neg { -v } else { v } as u128;
    let hi = (abs >> 64) as u64;
    let lo = abs as u64;
    let mut f = Float::with_val(prec, hi);
    f <<= 64u32;
    f += Float::with_val(prec, lo);
    if neg { -f } else { f }
}

/// Upper-triangular Cholesky factor R of the Euclidean Gram `B·Bᵀ` in f64,
/// or `None` if not numerically PD. Test-only oracle (production's
/// partial-prune lower bound is `euclidean_cholesky_16_mpfr_dual`). The Gram
/// is accumulated in i128 first — at deep ε an inflated basis (~2^25) pushes
/// entries to ~2^54, the edge of f64's mantissa — then converted to f64.
pub fn euclidean_cholesky_16(basis: &[[i64; 16]; 16]) -> Option<[[f64; 16]; 16]> {
    // Exact integer Gram = B·Bᵀ in i128 to absorb deep-ε basis growth.
    let mut gram = [[0_i128; 16]; 16];
    for i in 0..16 {
        for j in 0..16 {
            let mut s = 0_i128;
            for k in 0..16 {
                s += (basis[i][k] as i128) * (basis[j][k] as i128);
            }
            gram[i][j] = s;
        }
    }
    // f64 Cholesky on the (lower) triangular factor L such that L·Lᵀ = G.
    let mut l = [[0.0_f64; 16]; 16];
    for i in 0..16 {
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
    // Transpose to upper-triangular R = Lᵀ (caller convention).
    let mut r = [[0.0_f64; 16]; 16];
    for i in 0..16 {
        for j in 0..16 {
            r[i][j] = l[j][i];
        }
    }
    Some(r)
}

