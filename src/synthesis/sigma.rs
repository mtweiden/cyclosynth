//! Minkowski embedding ("Σ matrix") for `Z[ω]` and `Z[ζ_16]`.
//!
//! The Minkowski (canonical) embedding maps a degree-n cyclotomic ring of
//! integers into ℝⁿ via the n distinct field embeddings, splitting each
//! complex pair into (Re, Im). For our totally complex cyclotomics:
//!
//!   - `Z[ω] = Z[ζ_8]` → ℝ⁴ (4 embeddings = 2 conjugate pairs)
//!   - `Z[ζ_16]` → ℝ⁸ (8 embeddings = 4 conjugate pairs)
//!
//! See `clifford_sqrt_t_research.md` § Σ matrix and the
//! `project_minkowski_embedding.md` memory for the math derivation.
//!
//! ## What's here
//!
//! - [`sigma_8`] — 8×8 real matrix mapping a single Z[ζ_16] element's 8
//!   integer ζ-basis coords to its 8 real Minkowski coords (Re/Im of the
//!   four σ_a embeddings for `a ∈ {1, 5, 9, 13}`, the (Z/16Z)*-elements
//!   that fix `i = ζ⁴`).
//!
//! - [`embed_one`] — apply [`sigma_8`] to a `ZZeta` element.
//!
//! - [`embed_pair`] — embed a `(u_1, u_2) ∈ Z[ζ_16]²` pair as a 16D real
//!   vector, ordered `[u_1's 8 coords, u_2's 8 coords]`.
//!
//! - [`sigma_eta`] — 4×4 real matrix for the totally-real subring
//!   `Z[η] = Z[2cos(π/8)]`, the Minkowski embedding via the 4 real
//!   embeddings `τ_a` for `a ∈ {1, 5, 9, 13}`.
//!
//! - [`zzeta_to_double_re_im_eta_coords`] — given a `ZZeta` element u,
//!   return the η-basis coords of `(2·Re(u), 2·Im(u))` as integer
//!   8-tuples. (Direct `Re(u)` may have half-integer η-coords; doubling
//!   keeps everything integer.)

use crate::rings::types::{int_to_f64, Int, INT_THREE, INT_TWO};
use crate::rings::ZZeta;
use std::f64::consts::PI;

// ─── coset representatives ───────────────────────────────────────────────────

/// The four (ℤ/16ℤ)* elements ≡ 1 (mod 4) — the kernel of the projection
/// `(ℤ/16ℤ)* → (ℤ/4ℤ)*` corresponds to elements that fix `i = ζ⁴`.
///
/// These are our canonical coset representatives for the 4 distinct real
/// embeddings of the totally-real subring `Z[η]`. Picking i-fixing reps
/// makes the Σ for u = α + β·i cleanly block-decompose into Σ_η acting
/// on α and β separately.
pub const COSET_REPS: [u32; 4] = [1, 5, 9, 13];

// ─── Σ_8: full Minkowski embedding for one Z[ζ_16] element ──────────────────

/// 8×8 real matrix mapping one Z[ζ_16] element's ζ-basis integer coords
/// `(u_0, …, u_7)` to its 8 Minkowski real coords:
///
/// ```text
///   row 0,1: Re σ_1(u),  Im σ_1(u)
///   row 2,3: Re σ_5(u),  Im σ_5(u)
///   row 4,5: Re σ_9(u),  Im σ_9(u)
///   row 6,7: Re σ_13(u), Im σ_13(u)
/// ```
///
/// where `σ_a(ζ^j) = e^{i·a·j·π/8}`. Entry `Σ_8[i][j]` is the
/// coefficient of `u_j` in real coord `i`.
pub fn sigma_8() -> [[f64; 8]; 8] {
    let mut m = [[0.0f64; 8]; 8];
    for (k, &a) in COSET_REPS.iter().enumerate() {
        for j in 0..8 {
            let theta = (a as f64) * (j as f64) * PI / 8.0;
            m[2 * k][j] = theta.cos();
            m[2 * k + 1][j] = theta.sin();
        }
    }
    m
}

/// Apply `Σ_8` to a `ZZeta` element. Returns 8 real Minkowski coords.
pub fn embed_one(u: &ZZeta) -> [f64; 8] {
    let m = sigma_8();
    let coeffs = [u.a, u.b, u.c, u.d, u.e, u.f, u.g, u.h];
    let mut out = [0.0f64; 8];
    for i in 0..8 {
        for j in 0..8 {
            out[i] += m[i][j] * int_to_f64(coeffs[j]);
        }
    }
    out
}

/// Embed a `(u_1, u_2) ∈ Z[ζ_16]²` pair into 16D real space as
/// `[Σ_8(u_1) | Σ_8(u_2)]`.
pub fn embed_pair(u1: &ZZeta, u2: &ZZeta) -> [f64; 16] {
    let r1 = embed_one(u1);
    let r2 = embed_one(u2);
    let mut out = [0.0f64; 16];
    out[..8].copy_from_slice(&r1);
    out[8..].copy_from_slice(&r2);
    out
}

// ─── Σ_η: Minkowski embedding of the totally-real subring Z[η] ───────────────

/// 4×4 real matrix mapping a `Z[η]` element's η-basis coords
/// `(a_0, a_1, a_2, a_3)` (representing `a_0 + a_1·η + a_2·η² + a_3·η³`)
/// to its 4 real Minkowski coords `(τ_1(α), τ_5(α), τ_9(α), τ_13(α))`,
/// where `τ_a(η) = ζ_16^a + ζ_16^{−a} = 2 cos(a·π/8)`.
///
/// Numerical values (for sanity):
/// ```text
///   τ_1(η)  = +γ      ≈ +1.848
///   τ_5(η)  = -η_c    ≈ -0.765
///   τ_9(η)  = -γ      ≈ -1.848
///   τ_13(η) = +η_c    ≈ +0.765
/// ```
/// where `γ = √(2+√2)` and `η_c = √(2-√2)`.
pub fn sigma_eta() -> [[f64; 4]; 4] {
    let mut m = [[0.0f64; 4]; 4];
    for (i, &a) in COSET_REPS.iter().enumerate() {
        let tau_a_eta = 2.0 * ((a as f64) * PI / 8.0).cos();
        let mut p = 1.0;
        for j in 0..4 {
            m[i][j] = p;
            p *= tau_a_eta;
        }
    }
    m
}

// ─── ZZeta ↔ (Re, Im) η-coords conversion ────────────────────────────────────

/// Given a `ZZeta` element u, return η-basis integer coords of
/// `2·Re(u)` and `2·Im(u)` (each a 4-tuple in `Z[η]`).
///
/// Used because `Re(u)` and `Im(u)` for general `u ∈ Z[ζ_16]` may have
/// half-integer η-coords (the index-2 issue between `Z[i, √(2+√2)]`
/// and `Z[ζ_16]`); doubling keeps everything integer.
///
/// Returns `(double_re_eta_coords, double_im_eta_coords)`.
///
/// Derivation (see `clifford_sqrt_t_research.md`):
///
///   2·Re(u) = 2·u_0·1 + (u_1 - u_7)·η + (u_2 - u_6)·√2 + (u_3 - u_5)·η_c
/// where √2 = η²−2 and η_c = η³−3η (both in `Z[η]`).
///
///   2·Im(u) = 2·u_4·1 + (u_1 + u_7)·η_c + (u_2 + u_6)·√2 + (u_3 + u_5)·η
pub fn zzeta_to_double_re_im_eta_coords(u: &ZZeta) -> ([Int; 4], [Int; 4]) {
    // Substitute √2 = η²−2, η_c = η³−3η, then collect like η-powers.
    //   Let p1 = u_1 - u_7, p2 = u_2 - u_6, p3 = u_3 - u_5.
    //   2·Re(u) = 2·u_0 + p1·η + p2·(η²−2) + p3·(η³−3η)
    //          = (2·u_0 − 2·p2) + (p1 − 3·p3)·η + p2·η² + p3·η³
    let p1 = u.b - u.h;
    let p2 = u.c - u.g;
    let p3 = u.d - u.f;
    let double_re = [INT_TWO * u.a - INT_TWO * p2, p1 - INT_THREE * p3, p2, p3];

    //   Let s1 = u_1 + u_7, s2 = u_2 + u_6, s3 = u_3 + u_5.
    //   2·Im(u) = 2·u_4 + s1·(η³−3η) + s2·(η²−2) + s3·η
    //          = (2·u_4 − 2·s2) + (s3 − 3·s1)·η + s2·η² + s1·η³
    let s1 = u.b + u.h;
    let s2 = u.c + u.g;
    let s3 = u.d + u.f;
    let double_im = [INT_TWO * u.e - INT_TWO * s2, s3 - INT_THREE * s1, s2, s1];

    (double_re, double_im)
}

// ─── y-vector construction (Phase 2) ─────────────────────────────────────────

use num_complex::Complex64;

/// Build the 16D alignment vector for a target single-qubit unitary at
/// log-denominator-exponent `k`.
///
/// The lattice search wants to find `(u_1, u_2) ∈ Z[ζ_16]²` such that:
///   - `σ_1(u_1) / √(2^k) ≈ V_{11}` and `σ_1(u_2) / √(2^k) ≈ V_{21}`
///     (target alignment on the identity-embedding block);
///   - `|σ_a(u_1)|², |σ_a(u_2)|² ≤ 2^k` for `a ∈ {5, 9, 13}` (the
///     "bullet" balls — boundedness, not target-aligned).
///
/// So y has the target × √(2^k) on the σ_1 entries and zero elsewhere.
/// Layout matches [`embed_pair`]:
///
/// ```text
///   y[0..2]   = (Re V_{11}, Im V_{11}) · √(2^k)   ← σ_1 target for u_1
///   y[2..8]   = 0                                  ← σ_5/9/13 bullets, u_1
///   y[8..10]  = (Re V_{21}, Im V_{21}) · √(2^k)   ← σ_1 target for u_2
///   y[10..16] = 0                                  ← σ_5/9/13 bullets, u_2
/// ```
///
/// Only the first column of `target` is used (the second column is
/// determined up to phase by the first via unitarity).
pub fn build_y_vector(target: &[[Complex64; 2]; 2], k: u32) -> [f64; 16] {
    let scale = (2.0f64).powi(k as i32).sqrt();
    let mut y = [0.0f64; 16];
    // u_1 ↔ V_{11} (first column, first row).
    y[0] = target[0][0].re * scale;
    y[1] = target[0][0].im * scale;
    // u_2 ↔ V_{21} (first column, second row).
    y[8] = target[1][0].re * scale;
    y[9] = target[1][0].im * scale;
    y
}

// ─── Q-metric construction (Phase 4) ─────────────────────────────────────────

/// Build the 16D anisotropic Q-metric matrix for Z[ζ_16] synthesis at lde
/// `k` and precision `eps`. Q is symmetric positive-definite; the lattice
/// search finds integer 16-vectors x minimising `xᵀ Q x` (subject to the
/// norm-shell + 3 bilinear constraints, which are leaf checks).
///
/// Q is constructed in the *real-coordinate* (Σ-image) frame, which is
/// the natural geometric frame. The lattice-coord Q used by the LLL+SE
/// pipeline is `Q_lat = Σ_full^T · Q_real · Σ_full` (standard
/// transformation rule for quadratic forms under linear change of
/// variable). The factor differing between rings (1/2 for Z[ω], 1/4 for
/// Z[ζ_16]) appears in the *block-projector* definitions
/// `P_σa = (1/row_norm²) · Σ_σa^T Σ_σa`, where `row_norm² = 2` for Z[ω]
/// and `4` for Z[ζ_16] (each Σ_8 row in Z[ζ_16] has norm² = 4 because
/// of the 8-term DFT over `cos(jπ/8)`/`sin(jπ/8)`).
///
/// Three principal-axis terms (analog of `q_metric.rs` for Z[ω], with 3
/// "ball" terms instead of 1):
///
///   Q = (1/Δ_y²) · ŷ ŷᵀ                                      ← cap radial
///     + (1/Δ_⊥²) · (Π_σ1 − ŷ ŷᵀ)                              ← cap tangential
///     + (1/R²)   · (Π_σ5 + Π_σ9 + Π_σ13)                      ← bullet balls
///
/// where:
///   - `R² = 2^k`, `R = √(2^k)`.
///   - `Δ_y = R·ε² / (2(1+√(1−ε²)))` (cap radial thickness, ≈ R·ε²/4).
///   - `Δ_⊥ = R·ε` (cap tangential thickness).
///   - `ŷ` is the unit-norm alignment direction (target × √(2^k) on σ_1
///     block, zero elsewhere).
///   - `Π_σa` is the diagonal projector onto the σ_a-block coordinates
///     in the per-element [`embed_pair`] layout:
///     ```text
///     σ_1:  indices {0, 1, 8, 9}      (Re/Im of u_1, u_2 under σ_1)
///     σ_5:  indices {2, 3, 10, 11}
///     σ_9:  indices {4, 5, 12, 13}
///     σ_13: indices {6, 7, 14, 15}
///     ```
///
/// Eigenstructure (sanity):
///   - 1 eigenvalue `1/Δ_y²` (along ŷ) — strongest penalty (cap radial)
///   - 3 eigenvalues `1/Δ_⊥²` (σ_1 block ⊥ ŷ) — moderate (cap tangential)
///   - 12 eigenvalues `1/R²` (σ_5/9/13 blocks) — weakest (bullet balls)
pub fn build_q_zzeta_real(target: &[[Complex64; 2]; 2], k: u32, eps: f64) -> [[f64; 16]; 16] {
    let r_sq = 2.0f64.powi(k as i32);
    let r = r_sq.sqrt();
    let delta_y = r * eps * eps / (2.0 * (1.0 + (1.0 - eps * eps).sqrt()));
    let delta_perp = r * eps;
    let inv_dy_sq = 1.0 / (delta_y * delta_y);
    let inv_dp_sq = 1.0 / (delta_perp * delta_perp);
    let inv_r_sq = 1.0 / r_sq;

    let y_real = build_y_vector(target, k);
    let y_norm_sq: f64 = y_real.iter().map(|v| v * v).sum();
    let y_norm = y_norm_sq.sqrt();
    let yhat: [f64; 16] = if y_norm > 0.0 {
        std::array::from_fn(|i| y_real[i] / y_norm)
    } else {
        [0.0; 16]
    };

    // σ_1 block in per-element layout: indices {0, 1, 8, 9}.
    let in_sigma1 = |i: usize| matches!(i, 0 | 1 | 8 | 9);

    let mut q = [[0.0f64; 16]; 16];
    for i in 0..16 {
        for j in 0..16 {
            let yhat_outer = yhat[i] * yhat[j];
            let pi_sigma1 = if i == j && in_sigma1(i) { 1.0 } else { 0.0 };
            let pi_bullet = if i == j && !in_sigma1(i) { 1.0 } else { 0.0 };
            q[i][j] = inv_dy_sq * yhat_outer
                + inv_dp_sq * (pi_sigma1 - yhat_outer)
                + inv_r_sq * pi_bullet;
        }
    }
    q
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rings::types::INT_ONE;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn approx_eq_arr(a: &[f64], b: &[f64], tol: f64) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| approx_eq(*x, *y, tol))
    }

    #[test]
    fn sigma_8_dimensions_and_one() {
        let m = sigma_8();
        // Σ_8 acting on ZZeta(1) = (1,0,0,0,0,0,0,0): only column 0 contributes.
        // Column 0 has cos(0) and sin(0) for each Galois rep — i.e. 1 and 0.
        for (k, _) in COSET_REPS.iter().enumerate() {
            assert!(approx_eq(m[2 * k][0], 1.0, 1e-12), "Re σ_a(1) ≠ 1");
            assert!(approx_eq(m[2 * k + 1][0], 0.0, 1e-12), "Im σ_a(1) ≠ 0");
        }
    }

    #[test]
    fn embed_one_of_unit() {
        let one = ZZeta::ONE;
        let r = embed_one(&one);
        let expected = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        assert!(approx_eq_arr(&r, &expected, 1e-12), "embed(1) = {r:?}");
    }

    #[test]
    fn embed_one_of_zeta() {
        // ζ = (0,1,0,0,0,0,0,0). σ_a(ζ) = ζ^a = e^{i·a·π/8}.
        let zeta = ZZeta::ZETA;
        let r = embed_one(&zeta);
        let expected: [f64; 8] = [
            (PI / 8.0).cos(),
            (PI / 8.0).sin(), // σ_1
            (5.0 * PI / 8.0).cos(),
            (5.0 * PI / 8.0).sin(), // σ_5
            (9.0 * PI / 8.0).cos(),
            (9.0 * PI / 8.0).sin(), // σ_9
            (13.0 * PI / 8.0).cos(),
            (13.0 * PI / 8.0).sin(), // σ_13
        ];
        assert!(
            approx_eq_arr(&r, &expected, 1e-12),
            "embed(ζ) = {r:?}, expected {expected:?}"
        );
    }

    #[test]
    fn embed_one_sigma_1_matches_to_complex() {
        // The first 2 entries of embed_one(u) should be (Re σ_1(u), Im σ_1(u))
        // and σ_1 is the identity embedding, so they should equal Re/Im of
        // ZZeta::to_complex().
        let test_elems = [
            ZZeta::from_i32(1, 0, 0, 0, 0, 0, 0, 0),
            ZZeta::from_i32(0, 1, 0, 0, 0, 0, 0, 0),
            ZZeta::from_i32(0, 0, 1, 0, 0, 0, 0, 0),
            ZZeta::from_i32(0, 0, 0, 0, 1, 0, 0, 0),
            ZZeta::from_i32(3, -1, 2, 1, 0, -2, 1, -1),
            ZZeta::from_i32(-2, 5, 0, -3, 4, 1, -1, 2),
        ];
        for u in &test_elems {
            let r = embed_one(u);
            let c = u.to_complex();
            assert!(
                approx_eq(r[0], c.re, 1e-10),
                "Re σ_1({u:?}) = {} but to_complex.re = {}",
                r[0],
                c.re
            );
            assert!(
                approx_eq(r[1], c.im, 1e-10),
                "Im σ_1({u:?}) = {} but to_complex.im = {}",
                r[1],
                c.im
            );
        }
    }

    #[test]
    fn sigma_eta_first_column_is_one() {
        // η^0 = 1, embedded under any τ_a is 1.
        let m = sigma_eta();
        for i in 0..4 {
            assert!(approx_eq(m[i][0], 1.0, 1e-12), "τ_a(1) ≠ 1");
        }
    }

    #[test]
    fn sigma_eta_known_values() {
        // τ_1(η) = γ = √(2+√2), τ_5(η) = -η_c = -√(2-√2),
        // τ_9(η) = -γ, τ_13(η) = +η_c.
        let sqrt2 = 2.0f64.sqrt();
        let gamma = (2.0 + sqrt2).sqrt();
        let eta_c = (2.0 - sqrt2).sqrt();
        let m = sigma_eta();
        let expected_col_1 = [gamma, -eta_c, -gamma, eta_c];
        for i in 0..4 {
            assert!(
                approx_eq(m[i][1], expected_col_1[i], 1e-12),
                "Σ_η[{i}][1] = {} but expected {}",
                m[i][1],
                expected_col_1[i]
            );
        }
        // Column 2: η² should be (γ², η_c², γ², η_c²) = (2+√2, 2-√2, 2+√2, 2-√2).
        let g2 = 2.0 + sqrt2;
        let ec2 = 2.0 - sqrt2;
        let expected_col_2 = [g2, ec2, g2, ec2];
        for i in 0..4 {
            assert!(
                approx_eq(m[i][2], expected_col_2[i], 1e-12),
                "Σ_η[{i}][2] = {} but expected {}",
                m[i][2],
                expected_col_2[i]
            );
        }
    }

    #[test]
    fn double_re_im_of_unit() {
        // u = 1 → 2·Re(u) = 2, 2·Im(u) = 0.
        let one = ZZeta::ONE;
        let (re2, im2) = zzeta_to_double_re_im_eta_coords(&one);
        assert_eq!(
            re2,
            [
                INT_TWO,
                Int::from_i32(0),
                Int::from_i32(0),
                Int::from_i32(0)
            ]
        );
        assert_eq!(im2, [Int::from_i32(0); 4]);
    }

    #[test]
    fn double_re_im_of_i() {
        // u = i = ζ⁴ → 2·Re(u) = 0, 2·Im(u) = 2.
        let i_elem = ZZeta::I;
        let (re2, im2) = zzeta_to_double_re_im_eta_coords(&i_elem);
        assert_eq!(re2, [Int::from_i32(0); 4]);
        assert_eq!(
            im2,
            [
                INT_TWO,
                Int::from_i32(0),
                Int::from_i32(0),
                Int::from_i32(0)
            ]
        );
    }

    #[test]
    fn double_re_im_of_zeta() {
        // u = ζ = e^{iπ/8} = γ/2 + i·η_c/2.
        // So 2·Re(u) = γ = (0, 1, 0, 0)·η-basis,
        //    2·Im(u) = η_c = -3·η + η³ = (0, -3, 0, 1)·η-basis.
        let zeta = ZZeta::ZETA;
        let (re2, im2) = zzeta_to_double_re_im_eta_coords(&zeta);
        assert_eq!(
            re2,
            [
                Int::from_i32(0),
                INT_ONE,
                Int::from_i32(0),
                Int::from_i32(0)
            ]
        );
        assert_eq!(
            im2,
            [
                Int::from_i32(0),
                Int::from_i32(-3),
                Int::from_i32(0),
                INT_ONE
            ]
        );
    }

    #[test]
    fn block_decomp_consistent_with_full() {
        // For any u ∈ Z[ζ_16]:
        //   Re σ_a(u) = τ_a(Re(u))
        //   Im σ_a(u) = τ_a(Im(u))
        // Equivalent to: 2·Re σ_a(u) = τ_a(2·Re(u))
        //                2·Im σ_a(u) = τ_a(2·Im(u)).
        // Verifies the block-diagonal structure under (Re, Im) split.
        let test_elems = [
            ZZeta::from_i32(1, 0, 0, 0, 0, 0, 0, 0),
            ZZeta::from_i32(0, 1, 0, 0, 0, 0, 0, 0),
            ZZeta::from_i32(3, -1, 2, 1, 0, -2, 1, -1),
            ZZeta::from_i32(-2, 5, 0, -3, 4, 1, -1, 2),
        ];
        let m_eta = sigma_eta();
        for u in &test_elems {
            let r = embed_one(u);
            let (re2, im2) = zzeta_to_double_re_im_eta_coords(u);
            let re2_f: [f64; 4] = std::array::from_fn(|i| int_to_f64(re2[i]));
            let im2_f: [f64; 4] = std::array::from_fn(|i| int_to_f64(im2[i]));
            for k in 0..4 {
                let mut expected_2_re_sigma_a = 0.0;
                let mut expected_2_im_sigma_a = 0.0;
                for j in 0..4 {
                    expected_2_re_sigma_a += m_eta[k][j] * re2_f[j];
                    expected_2_im_sigma_a += m_eta[k][j] * im2_f[j];
                }
                let actual_2_re = 2.0 * r[2 * k];
                let actual_2_im = 2.0 * r[2 * k + 1];
                assert!(
                    approx_eq(actual_2_re, expected_2_re_sigma_a, 1e-9),
                    "block-decomp Re mismatch at k={k} for u={u:?}: \
                    direct={actual_2_re}, via Σ_η={expected_2_re_sigma_a}"
                );
                assert!(
                    approx_eq(actual_2_im, expected_2_im_sigma_a, 1e-9),
                    "block-decomp Im mismatch at k={k} for u={u:?}: \
                    direct={actual_2_im}, via Σ_η={expected_2_im_sigma_a}"
                );
            }
        }
    }

    #[test]
    fn y_vector_identity_target_at_k_0() {
        // Identity target V_{11} = 1, V_{21} = 0, k=0 ⇒ scale = 1.
        let id = [
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        ];
        let y = build_y_vector(&id, 0);
        let mut expected = [0.0f64; 16];
        expected[0] = 1.0; // Re V_{11} · 1
        for i in 0..16 {
            assert!(
                approx_eq(y[i], expected[i], 1e-12),
                "y[{i}] = {}, expected {}",
                y[i],
                expected[i]
            );
        }
    }

    #[test]
    fn y_vector_h_target_at_k_5() {
        // Hadamard's first column: V_{11} = V_{21} = 1/√2. Scale = √(2^5) = √32.
        let inv_sqrt2 = 1.0 / 2.0f64.sqrt();
        let h = [
            [
                Complex64::new(inv_sqrt2, 0.0),
                Complex64::new(inv_sqrt2, 0.0),
            ],
            [
                Complex64::new(inv_sqrt2, 0.0),
                Complex64::new(-inv_sqrt2, 0.0),
            ],
        ];
        let y = build_y_vector(&h, 5);
        let scale = (2.0f64).powi(5).sqrt();
        // y[0] = (1/√2)·√32 = √16 = 4
        assert!(approx_eq(y[0], inv_sqrt2 * scale, 1e-10));
        // y[1] = 0 (V_{11} is real)
        assert!(approx_eq(y[1], 0.0, 1e-12));
        // y[8] = (1/√2)·√32 = 4 too
        assert!(approx_eq(y[8], inv_sqrt2 * scale, 1e-10));
        assert!(approx_eq(y[9], 0.0, 1e-12));
        // All other entries 0.
        for &i in &[2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15] {
            assert!(approx_eq(y[i], 0.0, 1e-12), "y[{i}] should be 0");
        }
    }

    fn matvec_16(m: &[[f64; 16]; 16], v: &[f64; 16]) -> [f64; 16] {
        let mut out = [0.0; 16];
        for i in 0..16 {
            for j in 0..16 {
                out[i] += m[i][j] * v[j];
            }
        }
        out
    }

    fn norm_16(v: &[f64; 16]) -> f64 {
        v.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    fn unit_target(k: u32) -> [[Complex64; 2]; 2] {
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        // Hadamard column for u_1 = u_2 = 1/√2 keeps things symmetric.
        let _ = k;
        [
            [
                Complex64::new(inv_sqrt2, 0.0),
                Complex64::new(inv_sqrt2, 0.0),
            ],
            [
                Complex64::new(inv_sqrt2, 0.0),
                Complex64::new(-inv_sqrt2, 0.0),
            ],
        ]
    }

    #[test]
    fn q_is_symmetric() {
        let target = unit_target(8);
        let q = build_q_zzeta_real(&target, 8, 1e-3);
        for i in 0..16 {
            for j in 0..16 {
                assert!(
                    approx_eq(q[i][j], q[j][i], 1e-12),
                    "Q non-symmetric at ({i},{j}): {} vs {}",
                    q[i][j],
                    q[j][i]
                );
            }
        }
    }

    #[test]
    fn q_eigvec_along_yhat() {
        let target = unit_target(8);
        let k = 8;
        let eps = 1e-3;
        let q = build_q_zzeta_real(&target, k, eps);
        let y = build_y_vector(&target, k);
        let yn = norm_16(&y);
        let yhat: [f64; 16] = std::array::from_fn(|i| y[i] / yn);
        let qy = matvec_16(&q, &yhat);
        // Expected eigenvalue: 1/Δ_y².
        let r = (k as f64).exp2().sqrt();
        let delta_y = r * eps * eps / (2.0 * (1.0 + (1.0 - eps * eps).sqrt()));
        let lambda_y = 1.0 / (delta_y * delta_y);
        for i in 0..16 {
            let expected = lambda_y * yhat[i];
            assert!(
                approx_eq(qy[i], expected, 1e-3 * lambda_y.abs().max(1.0)),
                "Q·ŷ != (1/Δ_y²)·ŷ at i={i}: got {}, expected {}",
                qy[i],
                expected
            );
        }
    }

    #[test]
    fn q_eigvec_in_sigma5_block() {
        // Any unit vector in the σ_5 block (indices {2, 3, 10, 11}) should
        // be an eigenvector of Q with eigenvalue 1/R² = 1/2^k.
        let target = unit_target(6);
        let k = 6;
        let eps = 1e-3;
        let q = build_q_zzeta_real(&target, k, eps);
        let mut v = [0.0; 16];
        v[2] = 1.0; // pure Re σ_5(u_1) direction
        let qv = matvec_16(&q, &v);
        let lambda_r = 1.0 / (1u64 << k) as f64;
        for i in 0..16 {
            let expected = if i == 2 { lambda_r } else { 0.0 };
            assert!(
                approx_eq(qv[i], expected, 1e-12),
                "Q·v in σ_5 block: at i={i}, got {}, expected {}",
                qv[i],
                expected
            );
        }
    }

    #[test]
    fn q_eigvec_in_sigma9_and_sigma13_blocks() {
        let target = unit_target(6);
        let k = 6;
        let eps = 1e-3;
        let q = build_q_zzeta_real(&target, k, eps);
        let lambda_r = 1.0 / (1u64 << k) as f64;
        for &block_idx in &[4, 6, 12, 14] {
            let mut v = [0.0; 16];
            v[block_idx] = 1.0;
            let qv = matvec_16(&q, &v);
            for i in 0..16 {
                let expected = if i == block_idx { lambda_r } else { 0.0 };
                assert!(
                    approx_eq(qv[i], expected, 1e-12),
                    "Q·v at idx={block_idx}, i={i}: got {}, expected {}",
                    qv[i],
                    expected
                );
            }
        }
    }

    #[test]
    fn q_eigenvalue_count() {
        // Q should have eigenvalue spectrum:
        //   1× 1/Δ_y²  (along ŷ)
        //   3× 1/Δ_⊥²  (σ_1 block ⊥ ŷ)
        //   12× 1/R²    (σ_5, σ_9, σ_13 blocks)
        // We don't compute eigenvalues directly here (no LAPACK in lib);
        // instead we verify the trace and the count of distinct eigenvalues
        // via known eigenvectors.
        let target = unit_target(6);
        let k = 6;
        let eps = 1e-3;
        let q = build_q_zzeta_real(&target, k, eps);
        let r_sq = (1u64 << k) as f64;
        let r = r_sq.sqrt();
        let delta_y = r * eps * eps / (2.0 * (1.0 + (1.0 - eps * eps).sqrt()));
        let delta_perp = r * eps;
        let lambda_y = 1.0 / (delta_y * delta_y);
        let lambda_perp = 1.0 / (delta_perp * delta_perp);
        let lambda_r = 1.0 / r_sq;
        let expected_trace = lambda_y + 3.0 * lambda_perp + 12.0 * lambda_r;
        let actual_trace: f64 = (0..16).map(|i| q[i][i]).sum();
        let rel_err = (actual_trace - expected_trace).abs() / expected_trace.abs();
        assert!(
            rel_err < 1e-6,
            "trace(Q) = {actual_trace}, expected {expected_trace}, rel err {rel_err:.3e}"
        );
    }

    #[test]
    fn sigma_8_rows_orthogonal_with_norm_sq_4() {
        // Load-bearing: Σ_8 rows are mutually orthogonal with row_norm² = 4.
        // This is what makes the block projectors `(1/4)·Σ_σa^T Σ_σa` proper
        // rank-4 projectors that sum to the identity.
        let m = sigma_8();
        for r in 0..8 {
            let norm_sq: f64 = (0..8).map(|j| m[r][j] * m[r][j]).sum();
            assert!(
                approx_eq(norm_sq, 4.0, 1e-12),
                "row {r} norm² = {norm_sq}, expected 4"
            );
            for r2 in (r + 1)..8 {
                let dot: f64 = (0..8).map(|j| m[r][j] * m[r2][j]).sum();
                assert!(
                    dot.abs() < 1e-12,
                    "rows {r}, {r2} dot product = {dot}, expected 0"
                );
            }
        }
    }

    #[test]
    fn q_eigvec_cap_tangential() {
        // A vector in σ_1 block (indices {0, 1, 8, 9}) that's perpendicular
        // to ŷ should be an eigenvector with eigenvalue 1/Δ_⊥².
        // Build target where ŷ has equal components at {0, 8} (Hadamard col).
        let target = unit_target(6);
        let k = 6;
        let eps = 1e-3;
        let q = build_q_zzeta_real(&target, k, eps);
        let y = build_y_vector(&target, k);
        let yn = norm_16(&y);
        let yhat: [f64; 16] = std::array::from_fn(|i| y[i] / yn);
        // For Hadamard target, ŷ has nonzero entries only at {0, 8} (real
        // V_{11}, V_{21}; imaginary parts are zero). A perpendicular vector
        // in σ_1 block: pick (1, 0, ..., -1 at 8, 0...) normalized.
        // Verify: ŷ ∝ (1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0)/√2.
        // Perpendicular: (1, 0, ..., -1 at 8, 0...)/√2.
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        let mut v = [0.0; 16];
        v[0] = inv_sqrt2;
        v[8] = -inv_sqrt2;
        // Sanity: v ⊥ ŷ.
        let dot: f64 = (0..16).map(|i| v[i] * yhat[i]).sum();
        assert!(
            dot.abs() < 1e-12,
            "test vector not perpendicular to ŷ: {dot}"
        );
        // Q · v should equal (1/Δ_⊥²) · v.
        let qv = matvec_16(&q, &v);
        let r = (k as f64).exp2().sqrt();
        let delta_perp = r * eps;
        let lambda_perp = 1.0 / (delta_perp * delta_perp);
        for i in 0..16 {
            let expected = lambda_perp * v[i];
            // tolerance scales with the eigenvalue magnitude
            let tol = 1e-6 * lambda_perp.abs().max(1.0);
            assert!(
                approx_eq(qv[i], expected, tol),
                "Q·v cap-tangential at i={i}: got {}, expected {}, diff {}",
                qv[i],
                expected,
                qv[i] - expected
            );
        }
    }

    #[test]
    fn q_for_random_unitary_target() {
        // Same eigenstructure check but with a generic unitary target
        // (not aligned with axes).
        // V = Rz(0.3) · Ry(0.7) · Rz(1.1) — a generic SU(2) element.
        use std::f64::consts::E;
        let _ = E;
        let alpha = 0.3_f64;
        let beta = 0.7_f64;
        let gamma = 1.1_f64;
        // Rz(α) · Ry(β) · Rz(γ), first column entries:
        //   V_{11} = e^{-i(α+γ)/2} · cos(β/2)
        //   V_{21} = e^{-i(γ-α)/2} · sin(β/2)
        let v11 = Complex64::from_polar(1.0, -(alpha + gamma) / 2.0) * (beta / 2.0).cos();
        let v21 = Complex64::from_polar(1.0, -(gamma - alpha) / 2.0) * (beta / 2.0).sin();
        let target = [
            [v11, Complex64::new(0.0, 0.0)],
            [v21, Complex64::new(0.0, 0.0)],
        ];
        let k = 6;
        let eps = 1e-3;
        let q = build_q_zzeta_real(&target, k, eps);
        // Symmetry
        for i in 0..16 {
            for j in 0..16 {
                assert!(
                    approx_eq(q[i][j], q[j][i], 1e-12),
                    "Q non-symmetric for random target at ({i},{j})"
                );
            }
        }
        // ŷ-direction eigenvector check
        let y = build_y_vector(&target, k);
        let yn = norm_16(&y);
        let yhat: [f64; 16] = std::array::from_fn(|i| y[i] / yn);
        let qy = matvec_16(&q, &yhat);
        let r = (k as f64).exp2().sqrt();
        let delta_y = r * eps * eps / (2.0 * (1.0 + (1.0 - eps * eps).sqrt()));
        let lambda_y = 1.0 / (delta_y * delta_y);
        for i in 0..16 {
            let expected = lambda_y * yhat[i];
            assert!(
                approx_eq(qy[i], expected, 1e-3 * lambda_y.abs().max(1.0)),
                "random target Q·ŷ at i={i}: got {}, expected {}",
                qy[i],
                expected
            );
        }
    }

    #[test]
    fn q_psd_via_spectral_basis_test() {
        // Sample a handful of vectors and check vᵀ Q v ≥ 0.
        let target = unit_target(8);
        let q = build_q_zzeta_real(&target, 8, 1e-4);
        let test_vecs = [
            [
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            [
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            ],
            [
                1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0, 0.5, -0.5, 0.25, -0.25, 0.1, 0.0, 0.7,
                -0.3,
            ],
        ];
        for v in &test_vecs {
            let qv = matvec_16(&q, v);
            let vqv: f64 = v.iter().zip(qv.iter()).map(|(a, b)| a * b).sum();
            assert!(vqv > 0.0, "vᵀQv = {} should be > 0 for v = {:?}", vqv, v);
        }
    }

    #[test]
    fn embed_pair_is_concatenation() {
        let u1 = ZZeta::from_i32(1, 2, 3, 4, 5, 6, 7, 8);
        let u2 = ZZeta::from_i32(-1, -2, -3, -4, -5, -6, -7, -8);
        let pair = embed_pair(&u1, &u2);
        let r1 = embed_one(&u1);
        let r2 = embed_one(&u2);
        for i in 0..8 {
            assert!(approx_eq(pair[i], r1[i], 1e-12), "pair[{i}] mismatch");
            assert!(
                approx_eq(pair[i + 8], r2[i], 1e-12),
                "pair[{}] mismatch",
                i + 8
            );
        }
    }
}
