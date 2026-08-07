//! Native Ross–Selinger-style route for diagonal targets over
//! Clifford+√T — the ζ₁₆ counterpart of [`crate::synthesis::clifford_t::rz::ladder`].
//!
//! Same wrapper contract: probe a loose→tight ε ladder against the
//! crate's own acceptance check, then return the CHEAPEST verified
//! candidate (cost-first policy — every rung runs in parallel anyway).
//! Below 1e-12 verification switches to the exact MPFR trace formula
//! (`exact_rz_distance`) — f64 diamond distance saturates near 1e-15.

use crate::rings::types::MpFloat;
use crate::synthesis::distance::{diamond_distance_float, Mat2};
use crate::synthesis::clifford_sqrt_t::rz::gridsynth_q_gates_native;
use crate::synthesis::near_clifford::eval_gates_q;
use crate::synthesis::clifford_t::rz::ladder::prec_for_epsilon;
use crate::synthesis::synthesizer::SynthResult;

const EPS_LADDER: [f64; 5] = [2.4, 2.0, 1.6, 1.3, 1.0];

/// Support floor: exact-verified through 1e-48 (gridsynth_q_depth),
/// matching the Clifford+T route's policy floor. The next wall is the
/// i64 candidate coordinates (~2^{k/2}, overflowing near 56 decades) —
/// candidates would need Integer coords to go deeper.
const MIN_EPSILON: f64 = 1e-48;

/// Exact Rz-distance at MPFR precision from the exact unitary — the f64
/// diamond check saturates near 1e-15 (target quantization).
pub(crate) fn exact_rz_distance(u: &crate::matrix::U2Q, theta: &MpFloat, prec: u32) -> f64 {
    use rug::Integer;
    let vp = prec + 128;
    let half = MpFloat::with_val(vp, theta / 2u32);
    let (t_re, t_im) = (half.clone().cos(), half.sin());
    let mut sc = MpFloat::with_val(vp, 1.0);
    sc <<= u.k / 2;
    if u.k % 2 == 1 {
        sc *= MpFloat::with_val(vp, 2.0).sqrt();
    }
    let pi = MpFloat::with_val(vp, rug::float::Constant::Pi);
    let exact = |x: crate::rings::types::Int| -> MpFloat {
        let i = Integer::from_str_radix(&format!("{x}"), 10).expect("decimal I256");
        MpFloat::with_val(vp, &i)
    };
    let entry = |z: &crate::rings::ZZeta| -> (MpFloat, MpFloat) {
        let mut re = MpFloat::with_val(vp, 0.0);
        let mut im = MpFloat::with_val(vp, 0.0);
        for i in 0..8u32 {
            let ang = MpFloat::with_val(vp, &pi * i) / 8u32;
            let c = exact(z.coeff(i as usize));
            re += MpFloat::with_val(vp, &c * &ang.clone().cos());
            im += MpFloat::with_val(vp, &c * &ang.sin());
        }
        (
            MpFloat::with_val(vp, &re / &sc),
            MpFloat::with_val(vp, &im / &sc),
        )
    };
    let (u00r, u00i) = entry(&u.u11);
    let (u11r, u11i) = entry(&u.u22);
    // tr(U·Rz(θ)†) = u00·(cos+i·sin) + u11·(cos−i·sin); D² = q(8−q)/16.
    let tr_re = MpFloat::with_val(vp, &u00r * &t_re) - MpFloat::with_val(vp, &u00i * &t_im)
        + MpFloat::with_val(vp, &u11r * &t_re)
        + MpFloat::with_val(vp, &u11i * &t_im);
    let tr_im = MpFloat::with_val(vp, &u00r * &t_im) + MpFloat::with_val(vp, &u00i * &t_re)
        - MpFloat::with_val(vp, &u11r * &t_im)
        + MpFloat::with_val(vp, &u11i * &t_re);
    let tr_abs = (MpFloat::with_val(vp, &tr_re * &tr_re)
        + MpFloat::with_val(vp, &tr_im * &tr_im))
    .sqrt();
    let q = MpFloat::with_val(vp, 4.0) - MpFloat::with_val(vp, &tr_abs * 2u32);
    let q8 = MpFloat::with_val(vp, 8.0) - &q;
    (MpFloat::with_val(vp, &q * &q8) / 16u32)
        .max(&MpFloat::with_val(vp, 0.0))
        .sqrt()
        .to_f64()
}

/// Synthesize `Rz(theta)` over Clifford+√T to diamond distance < `epsilon`.
/// Returns `None` on any internal failure — callers fall back.
pub(crate) fn synthesize_rz_q(
    theta: &MpFloat,
    target: &Mat2,
    epsilon: f64,
    q_cost_x2: usize,
) -> Option<SynthResult> {
    if !(MIN_EPSILON..1.0).contains(&epsilon) {
        return None;
    }
    let prec = prec_for_epsilon(epsilon);
    use crate::synthesis::clifford_sqrt_t::gates_cost;
    use rayon::prelude::*;
    let mut results: Vec<Option<SynthResult>> = Vec::new();
    EPS_LADDER
        .par_iter()
        .map(|mult| {
            let gates = gridsynth_q_gates_native(theta, epsilon * mult, prec, q_cost_x2)?;
            let u = eval_gates_q(&gates)?;
            // f64 verification saturates near 1e-15 (target quantization);
            // below 1e-12 verify exactly against the MPFR angle.
            let dist = if epsilon < 1e-12 {
                exact_rz_distance(&u, theta, prec)
            } else {
                diamond_distance_float(&u.to_float(), target)
            };
            if dist < epsilon {
                Some(SynthResult { lde: u.k, gates: Some(gates), distance: dist })
            } else {
                None
            }
        })
        .collect_into_vec(&mut results);
    // Cost-first policy: every rung already ran in parallel, so pick the
    // cheapest verified circuit rather than the loosest (ladder order
    // breaks ties, keeping the result deterministic).
    results
        .into_iter()
        .flatten()
        .min_by_key(|r| r.gates.as_deref().map_or(usize::MAX, |g| gates_cost(g, q_cost_x2)))
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn rz_target(theta: f64) -> Mat2 {
        [
            [
                num_complex::Complex::from_polar(1.0, -theta / 2.0),
                num_complex::Complex::new(0.0, 0.0),
            ],
            [
                num_complex::Complex::new(0.0, 0.0),
                num_complex::Complex::from_polar(1.0, theta / 2.0),
            ],
        ]
    }

    /// Coarse-ε first light: solves, verifies, and uses √T gates.
    #[test]
    fn test_first_light_coarse() {
        for (theta, eps) in [(0.7_f64, 1e-2_f64), (1.9, 1e-2), (0.3, 1e-3)] {
            let t = rz_target(theta);
            let th = MpFloat::with_val(128, theta);
            let r = synthesize_rz_q(&th, &t, eps, 6)
                .unwrap_or_else(|| panic!("no result for theta={theta} eps={eps}"));
            assert!(r.distance < eps, "dist {} ≥ {eps}", r.distance);
        }
    }

    /// Deterministic across calls.
    #[test]
    fn test_deterministic() {
        let t = rz_target(0.7);
        let th = MpFloat::with_val(128, 0.7_f64);
        let a = synthesize_rz_q(&th, &t, 1e-3, 6).expect("a").gates;
        let b = synthesize_rz_q(&th, &t, 1e-3, 6).expect("b").gates;
        assert_eq!(a, b);
    }
}
