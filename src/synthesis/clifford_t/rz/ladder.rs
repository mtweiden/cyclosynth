//! Native Ross–Selinger route for diagonal (β = 0) targets.
//!
//! Routes z-rotations to [`super`] (`clifford_t::rz`) — the crate's own
//! gridsynth implementation — instead of the 8D lattice pipeline: the 1D
//! grid method is T-optimal for Rz and runs in milliseconds at any ε,
//! including targets a distance δ with ε < δ ≲ 10³ε from a Clifford,
//! whose required T-count exceeds the ε-tuned lde band and whose
//! prefix-split cost grows as 2^t' trying to reach it (issue #2).
//!
//! ## Correctness gotchas
//! - The Ross–Selinger internal ε criterion (half-angle metric) is ~2×
//!   tighter than our diamond-distance spec, so we probe a loose→tight ε
//!   ladder and accept the first (loosest) candidate that passes OUR
//!   acceptance check ([`diamond_distance_u2t_float`], the same check the
//!   8D path uses). This recovers the true optimal T-count; calling at
//!   1× ε costs ~3–5 extra T.

use crate::rings::types::MpFloat;
use crate::synthesis::distance::{diamond_distance_u2t_float, Mat2};
use crate::synthesis::clifford_t::rz::gridsynth_gates_native;
use crate::synthesis::near_clifford::eval_gates_t;
use crate::synthesis::synthesizer::SynthResult;

/// ε multipliers probed loose→tight; the first verified candidate (fewest T)
/// wins. Calibrated against the Haskell gridsynth oracle: 2.0–2.4× typically
/// verifies and matches its T-count.
const EPS_LADDER: [f64; 5] = [2.4, 2.0, 1.6, 1.3, 1.0];

/// Support floor: exact-verified through 1e-48 (bench_gridsynth_depth_limit);
/// below, the decomposer's I256 SO(3) numerators overflow at lde ≈ 254, so
/// the native path declines outright rather than probing a doomed ladder.
pub(crate) const MIN_EPSILON: f64 = 1e-48;

/// Working precision: newsynth's 15 + 2.5 digits/decade, in bits.
pub(crate) fn prec_for_epsilon(eps: f64) -> u32 {
    let decades = (1.0 / eps).log10().ceil().max(1.0);
    // decades ≤ ~330 for any positive f64 ε — fits u32 comfortably.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let dps = (15.0 + 2.5 * decades) as u32;
    // dps ≤ ~840 for any positive f64 ε — the bit count fits u32.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let bits = (f64::from(dps) * 3.322).ceil() as u32;
    bits + 16
}

/// Synthesize `Rz(theta)` over Clifford+T to diamond distance < `epsilon`.
/// `target` is the f64 SU(2) acceptance target built from the same angles.
/// Returns `None` on any internal failure — callers fall back to the 8D path.
pub(crate) fn synthesize_rz(theta: &MpFloat, target: &Mat2, epsilon: f64) -> Option<SynthResult> {
    if !(MIN_EPSILON..1.0).contains(&epsilon) {
        return None;
    }
    let prec = prec_for_epsilon(epsilon);
    // The rungs are independent and deterministic (per-call RNG): compute
    // them in parallel, then select exactly as the sequential loop would —
    // the loosest rung whose verified diamond distance meets ε.
    use rayon::prelude::*;
    let mut results: Vec<Option<SynthResult>> = Vec::new();
    EPS_LADDER
        .par_iter()
        .map(|mult| {
            let gates = gridsynth_gates_native(theta, epsilon * mult, prec)?;
            let u = eval_gates_t(&gates)?; // unexpected alphabet → skip rung
            let dist = diamond_distance_u2t_float(&u, target);
            if dist < epsilon {
                Some(SynthResult { lde: u.k, gates: Some(gates), distance: dist })
            } else {
                None
            }
        })
        .collect_into_vec(&mut results);
    results.into_iter().flatten().next()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use crate::synthesis::angle::Angle;
    use crate::synthesis::synthesizer::Synthesizer;

    /// Issue #2 repro: near-identity angle in the pathological ring
    /// (ε < δ ≲ 10³ε) must return quickly instead of hanging/None.
    #[test]
    fn test_near_clifford_rz_issue2() {
        let synth = Synthesizer::new(1e-8, false);
        for lam in [3.74507e-7, std::f64::consts::FRAC_PI_2 + 3.74507e-7] {
            let r = synth
                .synthesize_u1(Angle::Rad(lam))
                .unwrap_or_else(|| panic!("no result for lam={lam}"));
            assert!(r.distance < 1e-8, "distance {} ≥ ε for lam={lam}", r.distance);
            assert!(r.gates.is_some());
        }
    }

    /// Below-ε: the Clifford itself is returned (T-count 0).
    #[test]
    fn test_below_eps_returns_clifford() {
        let synth = Synthesizer::new(1e-8, false);
        let r = synth.synthesize_u1(Angle::Rad(5e-9)).expect("no result");
        let gates = r.gates.expect("gates");
        let t_count = gates.matches(['T', 't']).count();
        assert_eq!(t_count, 0, "expected T-free result, got {gates}");
        assert!(r.distance < 1e-8);
    }

    /// Deterministic: identical calls give identical gate strings.
    #[test]
    fn test_deterministic() {
        let synth = Synthesizer::new(1e-6, false);
        let a = synth.synthesize_u1(Angle::PiRatio(1, 8)).expect("a").gates;
        let b = synth.synthesize_u1(Angle::PiRatio(1, 8)).expect("b").gates;
        assert_eq!(a, b);
    }

    /// The native route matches the 8D lattice path's optimal T-count
    /// (both are T-optimal for z-rotations; allow ±1 for tie-breaks).
    #[test]
    fn test_matches_lattice_t_count() {
        for lam in [Angle::Rad(std::f64::consts::FRAC_PI_3), Angle::PiRatio(1, 8), Angle::PiRatio(3, 16)] {
            let native = Synthesizer::new(1e-3, false)
                .synthesize_u1(lam)
                .expect("native");
            let lattice = Synthesizer::new(1e-3, false)
                .with_native_rz(false)
                .synthesize_u1(lam)
                .expect("lattice");
            let t = |r: &crate::synthesis::synthesizer::SynthResult| {
                r.gates.as_ref().map_or(0, |g| {
                    g.chars().filter(|&c| c == 'T' || c == 't').count()
                })
            };
            let (tn, tl) = (t(&native), t(&lattice));
            assert!(
                tn <= tl + 1,
                "native T-count {tn} worse than lattice {tl} for {lam:?}"
            );
            assert!(native.distance < 1e-3);
        }
    }

    /// Deep-ε native route stays within spec (validated range boundary).
    #[test]
    fn test_deep_eps_1e10() {
        let synth = Synthesizer::new(1e-10, false);
        let r = synth.synthesize_u1(Angle::Rad(3.74507e-7)).expect("no result");
        assert!(r.distance < 1e-10, "distance {}", r.distance);
    }

    /// Below the 1e-48 floor the native route declines immediately (no
    /// doomed ladder probing; the front door then falls through and the
    /// Python policy layer rejects such ε up front).
    #[test]
    fn test_below_floor_declines_fast() {
        let target = [[num_complex::Complex::new(1.0, 0.0), num_complex::Complex::new(0.0, 0.0)],
                      [num_complex::Complex::new(0.0, 0.0), num_complex::Complex::new(1.0, 0.0)]];
        let theta = crate::rings::types::MpFloat::with_val(64, std::f64::consts::FRAC_PI_3);
        let t0 = std::time::Instant::now();
        assert!(super::synthesize_rz(&theta, &target, 1e-49).is_none());
        assert!(t0.elapsed().as_millis() < 50, "floor decline must be immediate");
    }
}
