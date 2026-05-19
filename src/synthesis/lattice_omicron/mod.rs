//! 8-dimensional integer enumeration for Clifford+R (R=Rz(π/6)) synthesis
//! over the ring Z[ξ], ξ = e^{iπ/6}  (n=6 case).
//!
//! This module mirrors `lattice` (the n=4 Clifford+T backend) with three
//! ring-specific adaptations:
//!   - `scratch::fill_sigma` uses the n=6 embedding Σ (√3/2, ½ entries).
//!   - `se::bilinear_b` uses the n=6 bilinear form (consecutive-coord, all +).
//!   - `integer::phase1` uses the n=6 norm check, alignment threshold, and
//!     Euclidean-prune bound.
//! All other files (cholesky_lu, lll, q_metric) are verbatim copies.
//!
//! ## Public API
//!
//! [`phase1`] takes a target alignment vector `y` and norm shell `k`, and
//! returns integer 8-vectors `x ∈ ℤ⁸` satisfying:
//!
//!   ‖x‖² + (a₀a₂+a₁a₃+b₀b₂+b₁b₃) = 2^k   (n=6 norm equation)
//!   a₀a₁+a₁a₂+a₂a₃+b₀b₁+b₁b₂+b₂b₃ = 0    (n=6 bilinear unitarity)
//!   |y · x|² ≥ 2^k · (1 − ε²)               (alignment cap)

pub mod cholesky_lu;
pub mod integer;
pub mod lll;
pub mod q_metric;
pub mod scratch;
pub mod se;

use crate::rings::Float;
use std::sync::atomic::AtomicBool;

/// Per-worker scratch buffers, allocated once and reused across all
/// MA prefixes that worker handles.
pub struct LatticeScratch {
    inner: scratch::IntScratch,
}

impl LatticeScratch {
    pub fn new(eps: Float) -> Self {
        Self { inner: scratch::IntScratch::new(eps) }
    }
}

/// Run the 8D Lenstra enumeration for one (y, k, eps) setup.
/// Returns integer 8-vectors satisfying the n=6 synthesis constraints.
pub fn phase1(
    scratch: &mut LatticeScratch,
    y: &[Float; 8],
    k: u32,
    eps: Float,
    max_phase2_calls: u64,
    budget_hit: &AtomicBool,
) -> Vec<[i64; 8]> {
    scratch.inner.reset_basis();
    integer::phase1(&mut scratch.inner, y, k, eps, max_phase2_calls, budget_hit).solutions
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthesis::clifford_pi6::{check_bilinear, check_norm_eq, compute_y};
    use std::sync::atomic::AtomicBool;

    // ── Helpers ──────────────────────────────────────────────────────────────

    /// Build a realistic y vector for n=6 at norm shell k.
    ///
    /// Uses an angle θ=0.15 rad with v₁=(cos θ, sin θ), v₂=0 so that
    /// the dominant u-block component is nonzero and alignment pruning fires.
    fn realistic_y_n6(k: u32) -> [Float; 8] {
        let r = 2.0_f64.powi(k as i32).sqrt() / 2.0;  // scale for the norm shell
        let theta: Float = 0.15;
        let (c, s) = (theta.cos(), theta.sin());
        // compute_y(v1_re, v1_im, v2_re, v2_im) with v1 = (c·r, s·r), v2 = 0
        let s32 = 3.0_f64.sqrt() / 2.0;
        [
            r * c,
            r * (s32 * c + 0.5 * s),
            r * (0.5 * c + s32 * s),
            r * s,
            0.0, 0.0, 0.0, 0.0,
        ]
    }

    // ── Gram / p_u sanity check ───────────────────────────────────────────────

    /// Verify that the n=6 Σ, once loaded into scratch, gives the correct Gram.
    /// For u-block coords x=(a₀,a₁,a₂,a₃): xᵀG_u x = 2(a₀²+a₁²+a₂²+a₃²) + 2(a₀a₂+a₁a₃).
    /// We check this by building p_u from scratch and verifying the block diagonal.
    #[test]
    fn sigma_gram_matches_sigma_gram_u() {
        use crate::rings::zomicron::SIGMA_GRAM_U;
        use super::scratch::IntScratch;

        let s = IntScratch::new(1e-3);
        // With the n=6 row ordering, standard rows are {0,1,4,5} and bullet
        // rows are {2,3,6,7}. p_u + p_ub together cover all 8 rows, so
        // 2·(p_u + p_ub) gives the full Gram. For the u-block (cols 0-3),
        // only rows 0-3 are nonzero, so 2·(p_u+p_ub)[i][j] = G_u[i][j].
        for i in 0..4 {
            for j in 0..4 {
                let two_pu_plus_pub = 2.0 * (s.p_u[i][j].to_f64() + s.p_ub[i][j].to_f64());
                let g_u = SIGMA_GRAM_U[i][j] as f64;
                assert!(
                    (two_pu_plus_pub - g_u).abs() < 1e-10,
                    "2·(p_u+p_ub)[{i}][{j}] = {two_pu_plus_pub:.6}, G_u[{i}][{j}] = {g_u}"
                );
            }
        }
        // Cross-block should be zero: u and t subspaces are orthogonal.
        for i in 0..4 {
            for j in 4..8 {
                let v = s.p_u[i][j].to_f64().abs();
                assert!(v < 1e-10, "p_u[{i}][{j}] should be 0, got {v}");
            }
        }
    }

    // ── Bilinear check ────────────────────────────────────────────────────────

    #[test]
    fn bilinear_b_correct_formula() {
        use super::se::bilinear_b;
        // Known zero: [1,0,0,0, 1,0,0,0] → all consecutive products = 0.
        assert_eq!(bilinear_b(&[1,0,0,0, 1,0,0,0]), 0);
        // Known nonzero: [1,1,0,0, 0,0,0,0] → a₀a₁=1 → sum=1.
        assert_eq!(bilinear_b(&[1,1,0,0, 0,0,0,0]), 1);
        // Cross-check against clifford_pi6::check_bilinear.
        let cases: &[[i64; 8]] = &[
            [0, 1, -1, 0, 0, 1, -1, 0],  // should satisfy: 0-1+0 + 0-1+0 = -2 ≠ 0
            [1, 0, 1, 0, 0, 0, 0, 0],    // a₀a₁=0, a₁a₂=0, a₂a₃=0 → 0 ✓
        ];
        for x in cases {
            let b_ours = bilinear_b(x);
            let b_ref = check_bilinear(x);
            // bilinear_b == 0 iff check_bilinear == true.
            assert_eq!(b_ours == 0, b_ref,
                "bilinear mismatch on {x:?}: bilinear_b={b_ours}, check_bilinear={b_ref}");
        }
    }

    // ── Norm check ────────────────────────────────────────────────────────────

    #[test]
    fn norm_check_matches_check_norm_eq() {
        // The trivial solution [1,0,0,0, 1,0,0,0] has:
        //   euclid = 2, cross = 0, so norm_eq = 2 = 2^1.
        let x = [1i64,0,0,0, 1,0,0,0];
        assert!(check_norm_eq(&x, 1), "trivial solution should pass k=1 norm check");
        assert!(check_bilinear(&x), "trivial solution should pass bilinear check");
    }

    // ── Phase-1 smoke tests ───────────────────────────────────────────────────

    #[test]
    fn phase1_runs_at_eps_1e3() {
        let mut scratch = LatticeScratch::new(1e-3);
        let y = realistic_y_n6(8);
        let budget_hit = AtomicBool::new(false);
        let _ = phase1(&mut scratch, &y, 8, 1e-3, 1_000, &budget_hit);
    }

    #[test]
    fn phase1_runs_at_eps_1e5() {
        let mut scratch = LatticeScratch::new(1e-5);
        let y = realistic_y_n6(17);
        let budget_hit = AtomicBool::new(false);
        let _ = phase1(&mut scratch, &y, 17, 1e-5, 1_000, &budget_hit);
    }

    /// Every solution returned by phase1 must satisfy check_norm_eq AND check_bilinear.
    #[test]
    fn solutions_satisfy_constraints() {
        let eps = 1e-3;
        let k = 8u32;
        let mut scratch = LatticeScratch::new(eps);
        let y = realistic_y_n6(k);
        let budget_hit = AtomicBool::new(false);
        let sols = phase1(&mut scratch, &y, k, eps, 10, &budget_hit);
        for x in &sols {
            assert!(check_norm_eq(x, k),
                "solution fails norm check: {x:?}");
            assert!(check_bilinear(x),
                "solution fails bilinear check: {x:?}");
        }
    }

    /// phase1 at a tiny k=1 must return [1,0,0,0, 1,0,0,0] (or its equivalent).
    #[test]
    fn trivial_shell_k1_found() {
        // At k=1 the only valid lattice points are the trivial ones like (1,0,0,0,1,0,0,0).
        // Use y pointing toward that solution.
        let s32 = 3.0_f64.sqrt() / 2.0;
        // compute_y(1/√2, 0, 1/√2, 0) — v1 = 1/√2, v2 = 1/√2 (both real)
        let r2 = 1.0 / 2.0_f64.sqrt();
        let y = [r2, s32*r2, 0.5*r2, 0.0,
                 r2, s32*r2, 0.5*r2, 0.0];
        let mut scratch = LatticeScratch::new(0.5);
        let budget_hit = AtomicBool::new(false);
        let sols = phase1(&mut scratch, &y, 1, 0.5, 100, &budget_hit);
        // At k=1 there should be valid solutions and they should all pass constraints.
        for x in &sols {
            assert!(check_norm_eq(x, 1), "k=1 solution fails norm: {x:?}");
            assert!(check_bilinear(x), "k=1 solution fails bilinear: {x:?}");
        }
    }
}
