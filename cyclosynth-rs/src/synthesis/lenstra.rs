//! Dispatch layer for the 8D Clifford+T synthesis Lenstra pipeline.
//!
//! Two parallel implementations live in sibling modules:
//!
//! - **`lenstra_light`** uses `twofloat` (double-double, ~104 bits) for the
//!   LLL+Cholesky setup. Stack-allocated `Copy` arithmetic with near-zero
//!   per-op overhead. Numerically stable for ε ≥ ~1e-4 (κ(Q) up to ~10¹⁶).
//!
//! - **`lenstra_heavy`** uses `rug::Float` (MPFR) at adaptive precision. Heap-
//!   allocated arithmetic with disciplined in-place mutation via macros and a
//!   pre-allocated `HeavyScratch` per rayon worker. Stable at any ε but ~25–30×
//!   slower at moderate ε due to MPFR fixed-overhead vs dual-double.
//!
//! Dispatch happens in `LenstraScratch::new(eps)` and `phase1_lenstra(scratch, …)`:
//! ε ≥ 1e-4 → `Light` (fast common path), ε < 1e-4 → `Heavy` (precision-
//! correct universal path). The Heavy path also handles the f64 SE precision
//! issue that emerges at extreme ε (the SE itself is in f64 in both cases; this
//! becomes a real concern around ε ≤ 1e-5 — see the SE-node-count diagnostic
//! in `lenstra_heavy::phase1_lenstra` if added later).

use crate::rings::Float;
use std::sync::atomic::AtomicBool;

/// ε threshold separating the Light (twofloat) path from the Heavy (rug) path.
/// Below this, twofloat's ~104-bit precision becomes insufficient for the LLL
/// Gram-Schmidt to maintain a unimodular basis (κ(Q) ≈ 4/ε⁴ exceeds f128-class
/// margins after Gram-Schmidt cancellation).
const LIGHT_EPS_FLOOR: Float = 1e-4;

/// Per-worker scratch holding pre-allocated buffers for whichever precision
/// path is active. Allocate once via rayon's `map_init`, reuse across all MA
/// prefixes that worker handles.
///
/// The `Heavy` variant carries TWO HeavyScratch buffers — a low-precision
/// scratch that handles the typical prefix and a full-precision scratch used
/// only when the low-precision attempt signals escalation (det/Cholesky/LU
/// failure or SE-node circuit breaker tripped). Pre-allocating both eliminates
/// per-prefix allocation cost on escalation; memory cost is ~2× per worker
/// (~10 KB) which is negligible.
pub enum LenstraScratch {
    /// twofloat path — no scratch needed (TwoFloat is `Copy` and stack-allocated).
    Light,
    /// rug path — adaptive: try `low` first, escalate to `high` if needed.
    Heavy {
        low: crate::synthesis::lenstra_heavy::HeavyScratch,
        high: crate::synthesis::lenstra_heavy::HeavyScratch,
    },
}

impl LenstraScratch {
    /// Construct the appropriate scratch for the given ε.
    pub fn new(eps: Float) -> Self {
        if eps >= LIGHT_EPS_FLOOR {
            LenstraScratch::Light
        } else {
            let prec_low = crate::synthesis::lenstra_heavy::compute_prec_low(eps);
            let prec_high = crate::synthesis::lenstra_heavy::compute_prec(eps);
            LenstraScratch::Heavy {
                low: crate::synthesis::lenstra_heavy::HeavyScratch::new(prec_low),
                high: crate::synthesis::lenstra_heavy::HeavyScratch::new(prec_high),
            }
        }
    }
}

/// Run the full 8D Lenstra pipeline. Dispatches to Light or Heavy based on
/// the variant of `scratch`. Both paths return integer 8-vectors satisfying
/// the same constraints (‖x‖² = 2^k, B(x) = 0, alignment ≥ thresh).
///
/// In the Heavy path, the low-precision scratch is tried first. If it returns
/// `should_escalate` (det/Cholesky/LU failure or SE-node count > threshold
/// without finding a solution), the high-precision scratch retries the same
/// prefix. The basis must be reset between attempts since the low-prec attempt
/// already mutated it.
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
        LenstraScratch::Heavy { low, high } => {
            low.reset_basis();
            let outcome = crate::synthesis::lenstra_heavy::phase1_lenstra_attempt(
                low, y, k, eps, max_phase2_calls, budget_hit, true,
            );
            if !outcome.should_escalate {
                return outcome.solutions;
            }
            high.reset_basis();
            crate::synthesis::lenstra_heavy::phase1_lenstra_attempt(
                high, y, k, eps, max_phase2_calls, budget_hit, false,
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
    fn heavy_path_at_eps_1e_5_runs() {
        let mut scratch = LenstraScratch::new(1e-5);
        assert!(matches!(scratch, LenstraScratch::Heavy { .. }));
        let y = realistic_y(21);
        let budget_hit = AtomicBool::new(false);
        let _ = phase1_lenstra(&mut scratch, &y, 21, 1e-5, 1_000, &budget_hit);
    }
}
