//! Dispatch layer for the 8D Clifford+T synthesis Lenstra pipeline.
//!
//! Two parallel implementations live in sibling modules:
//!
//! - **`lenstra_light`** uses `twofloat` (double-double, ~104 bits) for the
//!   LLL+Cholesky setup. Stack-allocated `Copy` arithmetic with near-zero
//!   per-op overhead. Numerically stable for ε ≥ ~1e-4 (κ(Q) up to ~10¹⁶).
//!
//! - **`lenstra_integer`** uses `i256` for the LLL Gram matrix, with MPFR
//!   only for μ-values and the post-LLL Cholesky/LU. Per-iteration LLL cost
//!   drops because i256 multiplies are ~10× cheaper than MPFR at the same
//!   precision. Validated for ε ∈ [1e-10, 1e-3]; replaces the older
//!   `lenstra_heavy` MPFR-throughout pipeline (still in tree for fallback).
//!
//! Dispatch happens in `LenstraScratch::new(eps)` and `phase1_lenstra(scratch, …)`:
//! ε ≥ 1e-4 → `Light` (fast common path), ε < 1e-4 → `Integer` (precision-
//! correct universal path).

use crate::rings::Float;
use std::sync::atomic::AtomicBool;

/// ε threshold separating the Light (twofloat) path from the Integer (i256+MPFR)
/// path. Below this, twofloat's ~104-bit precision becomes insufficient for the
/// LLL Gram-Schmidt to maintain a unimodular basis (κ(Q) ≈ 16/ε⁴ exceeds
/// f128-class margins after Gram-Schmidt cancellation).
const LIGHT_EPS_FLOOR: Float = 1e-4;

/// Per-worker scratch holding pre-allocated buffers for whichever precision
/// path is active. Allocate once via rayon's `map_init`, reuse across all MA
/// prefixes that worker handles.
pub enum LenstraScratch {
    /// twofloat path — no scratch needed (TwoFloat is `Copy` and stack-allocated).
    Light,
    /// i256 + MPFR path — pre-allocated `IntScratch` with all working buffers.
    Integer(crate::synthesis::lenstra_integer::IntScratch),
}

impl LenstraScratch {
    /// Construct the appropriate scratch for the given ε.
    pub fn new(eps: Float) -> Self {
        if eps >= LIGHT_EPS_FLOOR {
            LenstraScratch::Light
        } else {
            LenstraScratch::Integer(
                crate::synthesis::lenstra_integer::IntScratch::new(eps),
            )
        }
    }
}

/// Run the full 8D Lenstra pipeline. Dispatches to Light or Integer based on
/// the variant of `scratch`. Both paths return integer 8-vectors satisfying
/// the same constraints (‖x‖² = 2^k, B(x) = 0, alignment ≥ thresh).
pub fn phase1_lenstra(
    scratch: &mut LenstraScratch,
    y: &[Float; 8],
    k: u32,
    eps: Float,
    max_phase2_calls: u64,
    budget_hit: &AtomicBool,
) -> Vec<[i64; 8]> {
    match scratch {
        LenstraScratch::Light => crate::synthesis::lenstra_light::phase1_lenstra(
            y, k, eps, max_phase2_calls, budget_hit,
        ),
        LenstraScratch::Integer(s) => {
            s.reset_basis();
            crate::synthesis::lenstra_integer::phase1_lenstra_int(
                s, y, k, eps, max_phase2_calls, budget_hit,
            )
            .solutions
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
        let _ = phase1_lenstra(&mut scratch, &y, 14, 1e-3, 1_000, &budget_hit);
    }

    #[test]
    fn light_path_at_eps_1e_4_runs() {
        let mut scratch = LenstraScratch::new(1e-4);
        assert!(matches!(scratch, LenstraScratch::Light));
        let y = realistic_y(17);
        let budget_hit = AtomicBool::new(false);
        let _ = phase1_lenstra(&mut scratch, &y, 17, 1e-4, 1_000, &budget_hit);
    }

    #[test]
    fn integer_path_at_eps_1e_5_runs() {
        let mut scratch = LenstraScratch::new(1e-5);
        assert!(matches!(scratch, LenstraScratch::Integer(_)));
        let y = realistic_y(21);
        let budget_hit = AtomicBool::new(false);
        let _ = phase1_lenstra(&mut scratch, &y, 21, 1e-5, 1_000, &budget_hit);
    }
}
