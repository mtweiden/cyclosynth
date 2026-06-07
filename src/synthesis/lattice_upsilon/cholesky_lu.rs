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
//!     Reference oracle for `cholesky_f64_16`. Returns `false` on a
//!     non-positive-definite pivot.

#![allow(clippy::needless_range_loop)]

use rug::{Assign, Float as RFloat};

use super::scratch::{rfv, rfz, IntScratch16};
use crate::synthesis::lattice::cholesky_lu::i256_to_rfloat;
use crate::synthesis::lattice::lll::i256_to_f64;

// ─── snapshot Gram to MPFR (test-oracle) ─────────────────────────────────────

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

// ─── MPFR Cholesky (test-oracle) ─────────────────────────────────────────────

/// MPFR Cholesky on `g_post` (natural-scale post-LLL Gram). Output: lower-
/// triangular `l_post` at `scratch.prec_q` bits. Returns `false` on a
/// non-positive-definite pivot (extremely rare for valid LLL-output bases).
///
/// Reference oracle for `cholesky_f64_16`; not used in production. Allocates
/// a fresh `l_post` matrix on the stack — kept out of `IntScratch16` so this
/// path doesn't pay for a second 16x16 MPFR matrix in production scratch.
pub fn cholesky_int_16(
    scratch: &mut IntScratch16,
    g_post: &[[RFloat; 16]; 16],
) -> Option<[[RFloat; 16]; 16]> {
    let prec = scratch.prec_q;
    let mut l: [[RFloat; 16]; 16] = std::array::from_fn(|_| std::array::from_fn(|_| rfz(prec)));
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

// ─── MPFR LU solve (production: cap-center → lattice coords) ────────────────

/// Solve `Bᵀ · z = c` in MPFR at `scratch.lu_prec` bits, where
/// `B = scratch.basis` (i64) and `c = scratch.c` (MPFR). The matrix and RHS
/// are first loaded into `scratch.lu_a` (with `lu_a[i][j] = basis[j][i]`,
/// i.e. `Bᵀ`) and `scratch.lu_rhs`, then partial-pivoting LU runs in place.
/// Solution lands in `scratch.lu_x`. Returns `false` if the matrix is
/// numerically singular (pivot below 1e-30).
pub fn lu_solve_int_inplace_16(scratch: &mut IntScratch16) -> bool {
    if std::env::var_os("CYCLOSYNTH_TRACE_DEEP_EPS").is_some() {
        eprintln!("[trace stage 4 lu_solve_int_inplace_16] REACHED");
    }
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
            if std::env::var_os("CYCLOSYNTH_TRACE_DEEP_EPS").is_some() {
                eprintln!(
                    "[trace stage 4 lu_solve_int_inplace_16] PIVOT GUARD TRIPPED at k={k}: |piv|={:.3e} < tol → returning false",
                    piv_abs.to_f64()
                );
            }
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
    if std::env::var_os("CYCLOSYNTH_TRACE_DEEP_EPS").is_some() {
        eprintln!("[trace stage 4 lu_solve_int_inplace_16] SUCCESS");
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
/// `κ(G) ≤ (4/3)^(d-1) ≤ (4/3)^15 ≈ 240` at d=16 (paper Theorem 3
/// corollary). That's 4× the conditioning of the d=8 path (κ ≤ 16) but still
/// log₂(240) ≈ 8 bits of conditioning loss. f64's 53-bit mantissa absorbs it
/// with ~45 bits of margin and yields ~10⁻¹⁴ absolute error at the SE
/// unit-scale bound check, five orders below SE's 10⁻⁹ tolerance.
pub fn cholesky_f64_16(scratch: &mut IntScratch16) -> bool {
    let scale = 2.0_f64.powi(-scratch.scale_bits);
    let mut g = [[0.0_f64; 16]; 16];
    for i in 0..16 {
        for j in 0..=i {
            g[i][j] = i256_to_f64(scratch.gram[i][j]) * scale;
        }
    }
    let trace = std::env::var_os("CYCLOSYNTH_TRACE_DEEP_EPS").is_some();
    if trace {
        let mut max_g: f64 = 0.0;
        for i in 0..16 {
            for j in 0..=i {
                let a = g[i][j].abs();
                if a > max_g {
                    max_g = a;
                }
            }
        }
        let f64_ceil = (53_f64).exp2();
        let cross = if max_g > f64_ceil { "EXCEEDS" } else { "under" };
        eprintln!(
            "[trace stage 3 cholesky_f64_16] max|G_ij| (post-LLL, natural-scale) = {max_g:.3e}  F64_EXACT_CEIL = 2^53 = {f64_ceil:.3e}  ({cross} the f64 exact ceiling)"
        );
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
                    if trace {
                        eprintln!(
                            "[trace stage 3 cholesky_f64_16] PIVOT GUARD TRIPPED at i=j={i}: s={s:.3e} ≤ 0 → returning false"
                        );
                    }
                    return false;
                }
                scratch.l_f64[i][i] = s.sqrt();
            } else {
                scratch.l_f64[i][j] = s / scratch.l_f64[j][j];
            }
        }
    }
    if trace {
        eprintln!("[trace stage 3 cholesky_f64_16] SUCCESS (factorization complete)");
    }
    true
}

/// Run MPFR Cholesky on the natural-scale post-LLL Gram, then copy the
/// factor into the f64 buffer used by the SE walker.
///
/// The n=12 Q metric is very thin in the radial cap direction at deep ε.
/// After LLL, a mathematically PSD Gram can have pivots small enough that
/// f64 Cholesky trips a false non-PSD pivot. MPFR factorization avoids that
/// false rejection while preserving the existing f64 Schnorr-Euchner walker.
pub fn cholesky_mpfr_to_f64_16(scratch: &mut IntScratch16) -> bool {
    let trace = std::env::var_os("CYCLOSYNTH_TRACE_DEEP_EPS").is_some();
    let g_post = snapshot_gram_to_mpfr_16(scratch);
    let Some(l_post) = cholesky_int_16(scratch, &g_post) else {
        if trace {
            eprintln!("[trace stage 3 cholesky_mpfr_to_f64_16] MPFR pivot guard tripped");
        }
        return false;
    };
    for i in 0..16 {
        for j in 0..16 {
            scratch.l_f64[i][j] = if j <= i { l_post[i][j].to_f64() } else { 0.0 };
        }
    }
    if trace {
        let mut max_l: f64 = 0.0;
        for i in 0..16 {
            for j in 0..=i {
                max_l = max_l.max(scratch.l_f64[i][j].abs());
            }
        }
        eprintln!(
            "[trace stage 3 cholesky_mpfr_to_f64_16] SUCCESS (copied MPFR factor to f64, max|L_ij|={max_l:.3e})"
        );
    }
    true
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use i256::i256;

    /// Helper: install a rational integer Gram on the scratch from an f64
    /// matrix `g_nat` at `scale_bits = B`, i.e. `gram[i][j] = round(2^B · g_nat[i][j])`.
    fn install_gram_from_f64(scratch: &mut IntScratch16, g_nat: &[[f64; 16]; 16], scale_bits: i32) {
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
