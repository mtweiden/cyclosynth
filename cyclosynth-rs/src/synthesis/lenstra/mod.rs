//! Lenstra-style 8-dimensional integer enumeration for Clifford+T synthesis
//! (Algorithm 3.6 of arXiv:2510.05816). The whole subsystem is a single
//! function — [`phase1`] — that takes a target alignment vector `y` and a
//! norm shell `k`, and returns the integer 8-vectors `x ∈ ℤ⁸` satisfying:
//!
//!   ‖x‖² = 2^k   (norm shell — the cyclotomic norm constraint)
//!   B(x) = 0     (bilinear unitarity constraint, eq (3.10) of the paper)
//!   |y · x|² ≥ thresh_xy(k, ε)   (alignment to target within ε)
//!
//! ## Two implementation paths
//!
//! For ε ≥ 1e-4 we use the [`light`] path: stack-allocated [`twofloat`] (~104
//! bits) for the LLL+Cholesky setup. Cheap because TwoFloat is `Copy`,
//! arithmetic has zero per-op overhead, and at moderate ε the κ(Q) condition
//! number stays within TwoFloat's margin.
//!
//! For ε < 1e-4 we use the [`integer`] path: the L²-LLL algorithm
//! (Nguyen-Stehlé 2009) with exact-integer Gram in `i256` and pure-f64
//! Gram-Schmidt coefficients. Both paths share the [`se`] module's
//! Schnorr-Euchner enumeration to walk candidate `z` values, then validate
//! against the shell + bilinear + alignment constraints.
//!
//! ## Why the split?
//!
//! TwoFloat is faster than f64 + integer Gram for moderate κ but loses
//! orthogonalization at deep ε (κ(Q) ≈ 16/ε⁴ exceeds f128-class margins).
//! L²-LLL is provably stable for ε down to 1e-10 at d=8 (Theorem 2 +
//! Figure 7 of the paper) but has slightly higher per-iteration overhead
//! at moderate ε. The [`LIGHT_EPS_FLOOR`] threshold picks the cheaper path.

pub mod integer;
pub mod light;
pub mod se;

use crate::rings::Float;
use std::sync::atomic::AtomicBool;

/// ε threshold separating the [`light`] path (twofloat) from the [`integer`]
/// path (L²-LLL). Below this, twofloat's ~104-bit precision becomes
/// insufficient for the LLL Gram-Schmidt to maintain a unimodular basis —
/// κ(Q) ≈ 16/ε⁴ exceeds f128-class margins after GS cancellation.
const LIGHT_EPS_FLOOR: Float = 1e-4;

/// Per-worker scratch buffers, allocated once via rayon's `map_init` and
/// reused across all MA prefixes that worker handles. The variants are
/// disjoint: which one is active is decided at construction time by ε.
pub enum LenstraScratch {
    /// twofloat path — no scratch state (TwoFloat is `Copy`, stack-allocated).
    Light,
    /// L²-LLL path — pre-allocated `IntScratch` with all i256/f64/RFloat
    /// working buffers needed by the L²-LLL pipeline.
    Integer(integer::IntScratch),
}

impl LenstraScratch {
    /// Pick the appropriate scratch variant for `eps` based on
    /// [`LIGHT_EPS_FLOOR`].
    pub fn new(eps: Float) -> Self {
        if eps >= LIGHT_EPS_FLOOR {
            LenstraScratch::Light
        } else {
            LenstraScratch::Integer(integer::IntScratch::new(eps))
        }
    }
}

/// Run the 8D Lenstra enumeration for one MA-prefix's `(y, k, eps)` setup.
/// Dispatches to the [`light`] or [`integer`] backend based on the variant of
/// `scratch`. Both paths return integer 8-vectors satisfying the synthesis
/// constraints (norm shell, bilinear, alignment).
pub fn phase1(
    scratch: &mut LenstraScratch,
    y: &[Float; 8],
    k: u32,
    eps: Float,
    max_phase2_calls: u64,
    budget_hit: &AtomicBool,
) -> Vec<[i64; 8]> {
    match scratch {
        LenstraScratch::Light => light::phase1_lenstra(y, k, eps, max_phase2_calls, budget_hit),
        LenstraScratch::Integer(s) => {
            s.reset_basis();
            integer::phase1(s, y, k, eps, max_phase2_calls, budget_hit).solutions
        }
    }
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
    fn light_path_at_eps_1e_3_runs() {
        let mut scratch = LenstraScratch::new(1e-3);
        assert!(matches!(scratch, LenstraScratch::Light));
        let y = realistic_y(14);
        let budget_hit = AtomicBool::new(false);
        let _ = phase1(&mut scratch, &y, 14, 1e-3, 1_000, &budget_hit);
    }

    #[test]
    fn light_path_at_eps_1e_4_runs() {
        let mut scratch = LenstraScratch::new(1e-4);
        assert!(matches!(scratch, LenstraScratch::Light));
        let y = realistic_y(17);
        let budget_hit = AtomicBool::new(false);
        let _ = phase1(&mut scratch, &y, 17, 1e-4, 1_000, &budget_hit);
    }

    #[test]
    fn integer_path_at_eps_1e_5_runs() {
        let mut scratch = LenstraScratch::new(1e-5);
        assert!(matches!(scratch, LenstraScratch::Integer(_)));
        let y = realistic_y(21);
        let budget_hit = AtomicBool::new(false);
        let _ = phase1(&mut scratch, &y, 21, 1e-5, 1_000, &budget_hit);
    }
}
