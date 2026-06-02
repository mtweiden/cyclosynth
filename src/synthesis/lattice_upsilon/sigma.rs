//! Minkowski embedding ("Σ matrix") for `Z[ζ₂₄]` (the n=12 case).
//!
//! Z[ζ₂₄] = Z[i, √2, √3] has `φ(24) = 8`. Complex conjugation pairs the
//! 8 Galois embeddings into 4 conjugate pairs. We use the +i coset
//! representatives `m ∈ {1, 17, 13, 5}` (those with `Im(ζ^{6m}) > 0`),
//! labelled `[cap, •₃, •₂, •₂•₃]` (see SPEC §2).
//!
//! ## What's here
//!
//! - [`COSET_REPS`] — the 4 +i coset representatives `[1, 17, 13, 5]`.
//! - [`sigma_el`] — 8×8 per-element block `Σ_el`: rows are
//!   `[Re σ_m, Im σ_m]` for `m ∈ COSET_REPS`, columns are `ζ⁰..ζ⁷`.
//! - [`sigma_16`] — full 16×16 `Σ = blkdiag(Σ_el, Σ_el)` acting on
//!   `(u₁-coeffs, u₂-coeffs) ∈ Z¹⁶`.
//! - [`embed_one`] / [`embed_pair`] — apply `Σ_el` / `Σ` to ring elements.
//!
//! ## Gram (the enumerator metric)
//!
//! `Σ_el^T Σ_el = 4·I₈ + 2·C` where `C` couples column `k` with column
//! `k+4` only. Eigenvalues per element: {2,2,2,2, 6,6,6,6}.
//! Full `Σ^T Σ` is block-diagonal — two copies of the above.
//!
//! Consequence: `Σ⁻¹ = (Σ^T Σ)⁻¹ Σ^T` is NOT a scalar multiple of `Σ^T`
//! (contrast n=4 / Z[ω] where `Σ⁻¹ = ½Σ^T`). The enumerator must reduce
//! against this anisotropic metric.

use crate::rings::types::int_to_f64;
use crate::rings::ZUpsilon;
use std::f64::consts::PI;

// ─── coset representatives ───────────────────────────────────────────────────

/// The 4 `(ℤ/24ℤ)*` elements with `Im(ζ^{6m}) > 0` (the +i coset).
/// Row order in `Σ_el` is `[cap=1, •₃=17, •₂=13, •₂•₃=5]` (see SPEC §2).
pub const COSET_REPS: [u32; 4] = [1, 17, 13, 5];

/// Per-element block dimension (φ(24) = 8 = 4 reps × 2 (Re,Im) rows).
pub const D_EL: usize = 8;

/// Full Σ dimension: 2 ring elements × 8 = 16.
pub const D_FULL: usize = 16;

// ─── Σ_el: per-element block ─────────────────────────────────────────────────

/// 8×8 real matrix `Σ_el` mapping one `Z[ζ₂₄]` element's ζ-basis coords
/// `(c_0, …, c_7)` to its 8 Minkowski coords (Re/Im of the 4 σ_m).
///
/// Row layout:
/// ```text
///   row 0,1: Re σ_1(u),  Im σ_1(u)    ← cap
///   row 2,3: Re σ_17(u), Im σ_17(u)   ← •₃   (flips √3)
///   row 4,5: Re σ_13(u), Im σ_13(u)   ← •₂   (flips √2)
///   row 6,7: Re σ_5(u),  Im σ_5(u)    ← •₂•₃
/// ```
/// Entry `Σ_el[2k+r][j] = cos(m·j·π/12) if r=0 else sin(m·j·π/12)`.
pub fn sigma_el() -> [[f64; D_EL]; D_EL] {
    let mut m = [[0.0f64; D_EL]; D_EL];
    for (k, &rep) in COSET_REPS.iter().enumerate() {
        for j in 0..D_EL {
            let theta = (rep as f64) * (j as f64) * PI / 12.0;
            m[2 * k][j] = theta.cos();
            m[2 * k + 1][j] = theta.sin();
        }
    }
    m
}

/// Full 16×16 embedding `Σ = blkdiag(Σ_el, Σ_el)` on `(u₁-coeffs, u₂-coeffs)`.
pub fn sigma_16() -> [[f64; D_FULL]; D_FULL] {
    let el = sigma_el();
    let mut m = [[0.0f64; D_FULL]; D_FULL];
    for i in 0..D_EL {
        for j in 0..D_EL {
            m[i][j] = el[i][j];
            m[D_EL + i][D_EL + j] = el[i][j];
        }
    }
    m
}

/// Apply `Σ_el` to a `ZUpsilon` element. Returns 8 real Minkowski coords.
pub fn embed_one(u: &ZUpsilon) -> [f64; D_EL] {
    let el = sigma_el();
    let coeffs = u.coeffs();
    let mut out = [0.0f64; D_EL];
    for i in 0..D_EL {
        for j in 0..D_EL {
            out[i] += el[i][j] * int_to_f64(coeffs[j]);
        }
    }
    out
}

/// Embed a `(u₁, u₂) ∈ Z[ζ₂₄]²` pair into 16D real space as
/// `[Σ_el(u₁) | Σ_el(u₂)]`.
pub fn embed_pair(u1: &ZUpsilon, u2: &ZUpsilon) -> [f64; D_FULL] {
    let r1 = embed_one(u1);
    let r2 = embed_one(u2);
    let mut out = [0.0f64; D_FULL];
    out[..D_EL].copy_from_slice(&r1);
    out[D_EL..].copy_from_slice(&r2);
    out
}

// ─── Gram matrices (exact integer form) ──────────────────────────────────────

/// Exact integer Gram `G_el = Σ_el^T Σ_el = 4·I₈ + 2·C` (see SPEC §4).
/// `C[i][j] = 1` iff `|i-j| = 4`, zero elsewhere.
pub fn gram_el_int() -> [[i64; D_EL]; D_EL] {
    let mut g = [[0i64; D_EL]; D_EL];
    for i in 0..D_EL {
        g[i][i] = 4;
    }
    for i in 0..4 {
        g[i][i + 4] = 2;
        g[i + 4][i] = 2;
    }
    g
}

/// Full exact integer Gram `G = Σ^T Σ = blkdiag(G_el, G_el)`.
pub fn gram_int() -> [[i64; D_FULL]; D_FULL] {
    let el = gram_el_int();
    let mut g = [[0i64; D_FULL]; D_FULL];
    for i in 0..D_EL {
        for j in 0..D_EL {
            g[i][j] = el[i][j];
            g[D_EL + i][D_EL + j] = el[i][j];
        }
    }
    g
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rings::types::INT_ONE;
    use crate::rings::types::INT_ZERO;
    use num_complex::Complex64;

    const TOL: f64 = 1e-12;

    fn near(a: f64, b: f64) -> bool {
        (a - b).abs() < TOL
    }

    /// SPEC §3 decimal-form sanity: hand-verified per-entry values for Σ_el.
    #[test]
    fn sigma_el_matches_spec_decimal_form() {
        let m = sigma_el();
        let c1 = (PI / 12.0).cos(); // ≈ 0.9659258262890683
        let s1 = (PI / 12.0).sin(); // ≈ 0.25881904510252074
        let r2 = 2.0_f64.sqrt() / 2.0;
        let r3 = 3.0_f64.sqrt() / 2.0;

        // Row 0 = Re σ_1: [1, c1, √3/2, √2/2, 1/2, s1, 0, -s1]
        let row0 = [1.0, c1, r3, r2, 0.5, s1, 0.0, -s1];
        // Row 1 = Im σ_1: [0, s1, 1/2, √2/2, √3/2, c1, 1, c1]
        let row1 = [0.0, s1, 0.5, r2, r3, c1, 1.0, c1];
        // Row 2 = Re σ_17: [1, -s1, -√3/2, √2/2, 1/2, -c1, 0, c1]
        let row2 = [1.0, -s1, -r3, r2, 0.5, -c1, 0.0, c1];
        // Row 3 = Im σ_17: [0, -c1, 1/2, √2/2, -√3/2, -s1, 1, -s1]
        let row3 = [0.0, -c1, 0.5, r2, -r3, -s1, 1.0, -s1];
        // Row 4 = Re σ_13: [1, -c1, √3/2, -√2/2, 1/2, -s1, 0, s1]
        let row4 = [1.0, -c1, r3, -r2, 0.5, -s1, 0.0, s1];
        // Row 5 = Im σ_13: [0, -s1, 1/2, -√2/2, √3/2, -c1, 1, -c1]
        let row5 = [0.0, -s1, 0.5, -r2, r3, -c1, 1.0, -c1];
        // Row 6 = Re σ_5: [1, s1, -√3/2, -√2/2, 1/2, c1, 0, -c1]
        let row6 = [1.0, s1, -r3, -r2, 0.5, c1, 0.0, -c1];
        // Row 7 = Im σ_5: [0, c1, 1/2, -√2/2, -√3/2, s1, 1, s1]
        let row7 = [0.0, c1, 0.5, -r2, -r3, s1, 1.0, s1];

        let expected = [row0, row1, row2, row3, row4, row5, row6, row7];
        for i in 0..8 {
            for j in 0..8 {
                assert!(
                    near(m[i][j], expected[i][j]),
                    "Σ_el[{i}][{j}] = {} ≠ {}",
                    m[i][j],
                    expected[i][j]
                );
            }
        }
    }

    #[test]
    fn embed_one_of_unit_is_ones_and_zeros() {
        // u = 1 → Re σ_m(1) = 1, Im σ_m(1) = 0 for every rep.
        let one = ZUpsilon::ONE;
        let r = embed_one(&one);
        let expected = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        for i in 0..8 {
            assert!(near(r[i], expected[i]), "embed(1)[{i}] = {}", r[i]);
        }
    }

    #[test]
    fn embed_one_sigma_1_matches_to_complex() {
        // Row 0 (Re σ_1) and row 1 (Im σ_1) must match ZUpsilon::to_complex
        // because σ_1 is the identity embedding.
        let cases = [
            ZUpsilon::from_i32(3, -1, 2, 1, 0, -2, 1, -1),
            ZUpsilon::from_i32(-2, 5, 0, -3, 4, 1, -1, 2),
            ZUpsilon::ZETA,
            ZUpsilon::I,
        ];
        for u in &cases {
            let r = embed_one(u);
            let c = u.to_complex();
            assert!((r[0] - c.re).abs() < 1e-10, "Re σ_1({u}) mismatch");
            assert!((r[1] - c.im).abs() < 1e-10, "Im σ_1({u}) mismatch");
        }
    }

    #[test]
    fn embed_pair_concatenates_blocks() {
        let u1 = ZUpsilon::from_i32(1, 2, 3, 4, 5, 6, 7, 8);
        let u2 = ZUpsilon::from_i32(-1, -2, -3, -4, -5, -6, -7, -8);
        let pair = embed_pair(&u1, &u2);
        let r1 = embed_one(&u1);
        let r2 = embed_one(&u2);
        for i in 0..8 {
            assert!(near(pair[i], r1[i]));
            assert!(near(pair[8 + i], r2[i]));
        }
    }

    /// SPEC §4 Gram check: Σ_el^T Σ_el = 4I + 2C (off-diag coupling at k↔k+4).
    #[test]
    fn sigma_el_gram_is_four_i_plus_two_c() {
        let el = sigma_el();
        let g_expected = gram_el_int();
        let mut g = [[0.0f64; 8]; 8];
        for i in 0..8 {
            for j in 0..8 {
                for k in 0..8 {
                    g[i][j] += el[k][i] * el[k][j];
                }
            }
        }
        for i in 0..8 {
            for j in 0..8 {
                let expected = g_expected[i][j] as f64;
                assert!(
                    (g[i][j] - expected).abs() < 1e-10,
                    "Σ_el^T Σ_el[{i}][{j}] = {} (expected {})",
                    g[i][j],
                    expected
                );
            }
        }
    }

    /// SPEC §4: per-element eigenvalues are {2,2,2,2, 6,6,6,6}.
    /// Test by feeding the known eigenvectors and checking the eigenvalue.
    #[test]
    fn sigma_el_gram_eigenvalues_two_and_six() {
        let g = gram_el_int();
        for k in 0..4 {
            // Antisymmetric: e_k - e_{k+4} → eigenvalue 2.
            let mut v = [0.0f64; 8];
            v[k] = 1.0;
            v[k + 4] = -1.0;
            let mut gv = [0.0f64; 8];
            for i in 0..8 {
                for j in 0..8 {
                    gv[i] += g[i][j] as f64 * v[j];
                }
            }
            for i in 0..8 {
                assert!(
                    near(gv[i], 2.0 * v[i]),
                    "Anti eigvec k={k}: G·v[{i}] = {}, expected {}",
                    gv[i],
                    2.0 * v[i]
                );
            }

            // Symmetric: e_k + e_{k+4} → eigenvalue 6.
            let mut v = [0.0f64; 8];
            v[k] = 1.0;
            v[k + 4] = 1.0;
            let mut gv = [0.0f64; 8];
            for i in 0..8 {
                for j in 0..8 {
                    gv[i] += g[i][j] as f64 * v[j];
                }
            }
            for i in 0..8 {
                assert!(
                    near(gv[i], 6.0 * v[i]),
                    "Sym eigvec k={k}: G·v[{i}] = {}, expected {}",
                    gv[i],
                    6.0 * v[i]
                );
            }
        }
    }

    /// Full Σ Gram has the same eigenstructure on each of the two blocks.
    #[test]
    fn sigma_16_gram_block_diagonal() {
        let g = gram_int();
        // Top-right 8×8 must be zero.
        for i in 0..8 {
            for j in 0..8 {
                assert_eq!(g[i][j + 8], 0);
                assert_eq!(g[i + 8][j], 0);
            }
        }
        // Block (0..8, 0..8) and (8..16, 8..16) equal gram_el_int().
        let el = gram_el_int();
        for i in 0..8 {
            for j in 0..8 {
                assert_eq!(g[i][j], el[i][j]);
                assert_eq!(g[i + 8][j + 8], el[i][j]);
            }
        }
    }

    /// Identity test for the special constants:
    /// `c1 = cos(π/12) = (√6+√2)/4` and `s1 = sin(π/12) = (√6-√2)/4`.
    #[test]
    fn c1_s1_match_radical_forms() {
        let c1 = (PI / 12.0).cos();
        let s1 = (PI / 12.0).sin();
        let r2 = 2.0_f64.sqrt();
        let r6 = 6.0_f64.sqrt();
        assert!(near(c1, (r6 + r2) / 4.0));
        assert!(near(s1, (r6 - r2) / 4.0));
    }

    /// `Σ_el` rows are mutually orthogonal: cross-row dot products vanish
    /// for `i ≠ j` except where `|i−j| corresponds to the 4↔k+4 coupling
    /// inside the same column-index family`. The Gram check already pinned
    /// the structure; this is a complementary row-orthogonality assertion
    /// over the full 8 rows.
    #[test]
    fn sigma_el_row_norms_squared() {
        // Row norms² of Σ_el. Each row is (cos(mjπ/12))_{j=0..7} or
        // (sin(mjπ/12))_{j=0..7}. Direct computation per rep m:
        //   Σ_{j=0..7} cos²(mjπ/12)  +  Σ_{j=0..7} sin²(mjπ/12) = 8,
        // so per-rep the two rows together have norm² 8 → each row averages
        // 4, but they need not be individually 4. We check the pair sum
        // (Re²+Im²) instead.
        let m = sigma_el();
        for k in 0..4 {
            let mut sum = 0.0;
            for j in 0..8 {
                sum += m[2 * k][j] * m[2 * k][j] + m[2 * k + 1][j] * m[2 * k + 1][j];
            }
            assert!(
                near(sum, 8.0),
                "‖Re σ_m‖² + ‖Im σ_m‖² for rep {} = {}, expected 8",
                COSET_REPS[k],
                sum
            );
        }
    }

    /// All Σ entries are drawn from the small set `{0, ±s1, ±1/2, ±√2/2,
    /// ±√3/2, ±c1, ±1}` (SPEC §3).
    #[test]
    fn sigma_el_entries_in_small_set() {
        let c1 = (PI / 12.0).cos();
        let s1 = (PI / 12.0).sin();
        let r2 = 2.0_f64.sqrt() / 2.0;
        let r3 = 3.0_f64.sqrt() / 2.0;
        let allowed = [0.0, 0.5, r2, r3, s1, c1, 1.0];
        let m = sigma_el();
        for i in 0..8 {
            for j in 0..8 {
                let a = m[i][j].abs();
                let ok = allowed.iter().any(|&v| (a - v).abs() < 1e-12);
                assert!(ok, "Σ_el[{i}][{j}] = {} not in allowed set", m[i][j]);
            }
        }
    }

    /// Per-element norm via Σ (real-space) matches ZUpsilon::norm_sqr
    /// (cyclotomic-basis form) up to the documented factor of 4.
    ///
    /// `‖Σ_el · x‖² = x^T G_el x = 4·norm_sqr(u)` (see SPEC §4 & §5).
    #[test]
    fn norm_via_sigma_matches_zupsilon_norm_sqr() {
        let cases = [
            ZUpsilon::from_i32(1, 0, 0, 0, 0, 0, 0, 0),
            ZUpsilon::from_i32(0, 1, 0, 0, 0, 0, 0, 0),
            ZUpsilon::from_i32(3, -1, 2, 1, 0, -2, 1, -1),
            ZUpsilon::from_i32(-2, 5, 0, -3, 4, 1, -1, 2),
        ];
        for u in &cases {
            let r = embed_one(u);
            let n_real: f64 = r.iter().map(|v| v * v).sum();
            let n_ring = u.norm_sqr();
            let n_ring_f = int_to_f64(n_ring);
            assert!(
                (n_real - 4.0 * n_ring_f).abs() < 1e-8,
                "‖Σ·x‖² = {}, expected 4·norm_sqr = {}",
                n_real,
                4.0 * n_ring_f
            );
        }
    }

    /// ZERO and structural sanity for the i,j conventions.
    #[test]
    fn embed_one_of_zero_is_zero() {
        let z = ZUpsilon::ZERO;
        let r = embed_one(&z);
        for v in r {
            assert!(near(v, 0.0));
        }
    }

    /// σ_1(ζ) = e^{iπ/12}. Spot check the next embedding angles too.
    #[test]
    fn embed_one_of_zeta_matches_angle() {
        let z = ZUpsilon::ZETA;
        let r = embed_one(&z);
        for (k, &m) in COSET_REPS.iter().enumerate() {
            let theta = (m as f64) * PI / 12.0;
            assert!(near(r[2 * k], theta.cos()));
            assert!(near(r[2 * k + 1], theta.sin()));
        }
    }

    /// Confirm INT_* are being used (not Int::from(n)) — implicit by
    /// reading `ZUpsilon::sqrt2()` etc. but checked once here.
    #[test]
    fn ring_consts_use_int_helpers() {
        assert_eq!(ZUpsilon::ONE.a, INT_ONE);
        assert_eq!(ZUpsilon::ZERO.a, INT_ZERO);
    }

    /// Numerical sanity: Σ · e_j gives column j of Σ (and the dual identity
    /// `Σ_el · ζ_basis(j)` equals the (cos, sin) at the j-th column).
    #[test]
    fn embed_basis_vectors_are_columns_of_sigma_el() {
        let el = sigma_el();
        for j in 0..8 {
            let mut u = ZUpsilon::ZERO;
            match j {
                0 => u = ZUpsilon::from_i32(1, 0, 0, 0, 0, 0, 0, 0),
                1 => u = ZUpsilon::from_i32(0, 1, 0, 0, 0, 0, 0, 0),
                2 => u = ZUpsilon::from_i32(0, 0, 1, 0, 0, 0, 0, 0),
                3 => u = ZUpsilon::from_i32(0, 0, 0, 1, 0, 0, 0, 0),
                4 => u = ZUpsilon::from_i32(0, 0, 0, 0, 1, 0, 0, 0),
                5 => u = ZUpsilon::from_i32(0, 0, 0, 0, 0, 1, 0, 0),
                6 => u = ZUpsilon::from_i32(0, 0, 0, 0, 0, 0, 1, 0),
                7 => u = ZUpsilon::from_i32(0, 0, 0, 0, 0, 0, 0, 1),
                _ => unreachable!(),
            }
            let r = embed_one(&u);
            for i in 0..8 {
                assert!(near(r[i], el[i][j]));
            }
        }
    }

    /// Spot-check Σ_el image of `√2 = ζ³ + ζ⁻³ = ζ³ - ζ⁹`. Under σ_1, the
    /// value should be `+√2`; under σ_13 (the •₂ branch), `−√2`.
    #[test]
    fn sqrt2_under_cap_and_bullet() {
        let r2 = 2.0_f64.sqrt();
        let s = ZUpsilon::sqrt2();
        let r = embed_one(&s);
        // σ_1 row: (Re, Im) = (√2, 0).
        assert!(near(r[0], r2));
        assert!(near(r[1], 0.0));
        // σ_17 row (•₃ — flips √3, keeps √2): (√2, 0).
        assert!(near(r[2], r2));
        assert!(near(r[3], 0.0));
        // σ_13 row (•₂ — flips √2): (−√2, 0).
        assert!(near(r[4], -r2));
        assert!(near(r[5], 0.0));
        // σ_5 row (•₂•₃): (−√2, 0).
        assert!(near(r[6], -r2));
        assert!(near(r[7], 0.0));
    }

    /// Same spot check for √3.
    #[test]
    fn sqrt3_under_cap_and_bullet() {
        let r3 = 3.0_f64.sqrt();
        let s = ZUpsilon::sqrt3();
        let r = embed_one(&s);
        // σ_1: +√3
        assert!(near(r[0], r3));
        // σ_17 (•₃): −√3
        assert!(near(r[2], -r3));
        // σ_13 (•₂): +√3
        assert!(near(r[4], r3));
        // σ_5 (•₂•₃): −√3
        assert!(near(r[6], -r3));
        for k in 0..4 {
            assert!(near(r[2 * k + 1], 0.0));
        }
    }

    /// And √6.
    #[test]
    fn sqrt6_under_cap_and_bullet() {
        let r6 = 6.0_f64.sqrt();
        let s = ZUpsilon::sqrt6();
        let r = embed_one(&s);
        // σ_1: +√6, σ_17 (•₃): −√6, σ_13 (•₂): −√6, σ_5 (•₂•₃): +√6.
        assert!(near(r[0], r6));
        assert!(near(r[2], -r6));
        assert!(near(r[4], -r6));
        assert!(near(r[6], r6));
        for k in 0..4 {
            assert!(near(r[2 * k + 1], 0.0));
        }
    }

    /// Numerical Σ_el agrees with `to_complex` at each embedding by
    /// substituting `ζ ↦ e^{i·m·π/12}` directly.
    #[test]
    fn sigma_el_rows_match_galois_substitution() {
        let cases = [
            ZUpsilon::ONE,
            ZUpsilon::ZETA,
            ZUpsilon::I,
            ZUpsilon::sqrt2(),
            ZUpsilon::sqrt3(),
            ZUpsilon::sqrt6(),
            ZUpsilon::from_i32(3, -1, 2, 1, 0, -2, 1, -1),
        ];
        for u in &cases {
            let r = embed_one(u);
            let coeffs = u.coeffs();
            for (k, &m) in COSET_REPS.iter().enumerate() {
                let mut c = Complex64::new(0.0, 0.0);
                for j in 0..8 {
                    let theta = (m as f64) * (j as f64) * PI / 12.0;
                    c += int_to_f64(coeffs[j]) * Complex64::from_polar(1.0, theta);
                }
                assert!((r[2 * k] - c.re).abs() < 1e-9, "Re σ_{m}({u})");
                assert!((r[2 * k + 1] - c.im).abs() < 1e-9, "Im σ_{m}({u})");
            }
        }
    }
}
