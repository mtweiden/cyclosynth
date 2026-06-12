//! Lenstra-style 8-dimensional integer enumeration for Clifford+T synthesis
//! (Algorithm 3.6 of arXiv:2510.05816). The whole subsystem is a single
//! function — [`find_aligned_lattice_points`] — that takes a target alignment vector `y` and a
//! norm shell `k`, and returns the integer 8-vectors `x ∈ ℤ⁸` satisfying:
//!
//!   ‖x‖² = 2^k   (norm shell — the cyclotomic norm constraint)
//!   B(x) = 0     (bilinear unitarity constraint, eq (3.10) of the paper)
//!   |y · x|² ≥ thresh_xy(k, ε)   (alignment to target within ε)
//!
//! ## Pipeline
//!
//! [`integer`] is the `find_aligned_lattice_points` driver that orchestrates:
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


/// Per-worker scratch buffers, allocated once via rayon's `map_init` and
/// reused across all MA prefixes that worker handles.
pub use integer::find_aligned_lattice_points;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rings::Float;
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
        let mut scratch = scratch::IntScratch::new(1e-3);
        let y = realistic_y(14);
        let budget_hit = AtomicBool::new(false);
        let _ = find_aligned_lattice_points(
            &mut scratch, &y, 14, 1e-3, 1, 1_000, 1_000_000, &budget_hit, None,
        );
    }

    #[test]
    fn integer_path_at_eps_1e_5_runs() {
        let mut scratch = scratch::IntScratch::new(1e-5);
        let y = realistic_y(21);
        let budget_hit = AtomicBool::new(false);
        let _ = find_aligned_lattice_points(
            &mut scratch, &y, 21, 1e-5, 1, 1_000, 1_000_000, &budget_hit, None,
        );
    }

    /// The node budget must terminate the walk and report the truncation
    /// via `budget_hit`. A budget of 1 binds at the very first
    /// recurse-entry, independent of how small the enumeration region is
    /// (at this config the full walk completes in < 50 nodes).
    #[test]
    fn node_budget_terminates_and_reports() {
        let mut scratch = scratch::IntScratch::new(1e-3);
        let y = realistic_y(14);
        let budget_hit = AtomicBool::new(false);
        let sols = find_aligned_lattice_points(
            &mut scratch, &y, 14, 1e-3, usize::MAX, u64::MAX, 1, &budget_hit, None,
        );
        assert!(
            budget_hit.load(std::sync::atomic::Ordering::Relaxed),
            "a 1-node budget must be reported as hit"
        );
        assert!(sols.is_empty(), "no leaf is reachable within 1 node");
    }

    /// A pre-set external abort must return immediately with no solutions
    /// and must NOT report a budget hit.
    #[test]
    fn external_abort_returns_immediately() {
        let mut scratch = scratch::IntScratch::new(1e-3);
        let y = realistic_y(14);
        let budget_hit = AtomicBool::new(false);
        let abort = AtomicBool::new(true);
        let sols = find_aligned_lattice_points(
            &mut scratch, &y, 14, 1e-3, usize::MAX, u64::MAX, u64::MAX,
            &budget_hit, Some(&abort),
        );
        assert!(sols.is_empty(), "aborted walk must return no solutions");
        assert!(
            !budget_hit.load(std::sync::atomic::Ordering::Relaxed),
            "external abort is not a budget hit"
        );
    }
}
