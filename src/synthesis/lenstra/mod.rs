//! Lenstra-style 8-dimensional integer enumeration for Clifford+T synthesis
//! (Algorithm 3.6 of arXiv:2510.05816). The whole subsystem is a single
//! function — [`phase1`] — that takes a target alignment vector `y` and a
//! norm shell `k`, and returns the integer 8-vectors `x ∈ ℤ⁸` satisfying:
//!
//!   ‖x‖² = 2^k   (norm shell — the cyclotomic norm constraint)
//!   B(x) = 0     (bilinear unitarity constraint, eq (3.10) of the paper)
//!   |y · x|² ≥ thresh_xy(k, ε)   (alignment to target within ε)
//!
//! ## Pipeline
//!
//! [`integer`] is the `phase1` driver that orchestrates:
//! [`q_metric`] (anisotropic Q-metric construction), [`lll`] (L²-LLL of
//! Nguyen-Stehlé 2009 with exact i256 Gram + f64 GS coefficients),
//! [`cholesky_lu`] (post-LLL Cholesky + cap-center LU solve), and [`se`]
//! (Schnorr-Euchner walk over the candidate ellipsoid). [`scratch`] holds
//! the pre-allocated MPFR/i256 buffers reused across calls.

pub mod cholesky_lu;
pub mod integer;
pub mod lll;
pub mod q_metric;
pub mod scratch;
pub mod se;

use crate::rings::Float;
use std::sync::atomic::AtomicBool;

/// Per-worker scratch buffers, allocated once via rayon's `map_init` and
/// reused across all MA prefixes that worker handles.
pub struct LenstraScratch {
    inner: scratch::IntScratch,
}

impl LenstraScratch {
    pub fn new(eps: Float) -> Self {
        Self { inner: scratch::IntScratch::new(eps) }
    }
}

/// Run the 8D Lenstra enumeration for one MA-prefix's `(y, k, eps)` setup.
/// Returns integer 8-vectors satisfying the synthesis constraints (norm
/// shell, bilinear, alignment).
pub fn phase1(
    scratch: &mut LenstraScratch,
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
    use std::sync::atomic::AtomicBool;

    fn realistic_y(k: u32) -> [Float; 8] {
        let r2 = 1.0 / 2.0_f64.sqrt();
        let s = ((1u64 << k) as Float).sqrt() / 2.0;
        let c = 0.15_f64.cos();
        let ns = -0.15_f64.sin();
        [
            s * c, s * (c + ns) * r2, s * ns, s * (-c + ns) * r2,
            0.0, 0.0, 0.0, 0.0,
        ]
    }

    #[test]
    fn integer_path_at_eps_1e_3_runs() {
        let mut scratch = LenstraScratch::new(1e-3);
        let y = realistic_y(14);
        let budget_hit = AtomicBool::new(false);
        let _ = phase1(&mut scratch, &y, 14, 1e-3, 1_000, &budget_hit);
    }

    #[test]
    fn integer_path_at_eps_1e_5_runs() {
        let mut scratch = LenstraScratch::new(1e-5);
        let y = realistic_y(21);
        let budget_hit = AtomicBool::new(false);
        let _ = phase1(&mut scratch, &y, 21, 1e-5, 1_000, &budget_hit);
    }
}
