//! Run synthesize_mpfr with TRUE MPFR-precision target (constructed from
//! MPFR theta, not f64-lifted), at ε values bracketing the cliff.

use cyclosynth::synthesis::clifford_sqrt_t::{SynthesizerQ, Mat2Mpfr};
use rug::Float as RFloat;
use std::time::Instant;

fn rz_mpfr(theta_mpfr: &RFloat, prec: u32) -> Mat2Mpfr {
    let half = RFloat::with_val(prec, theta_mpfr / 2);
    let cos_half = half.clone().cos();
    let sin_half = half.clone().sin();
    let zero = RFloat::with_val(prec, 0.0);
    [
        [(cos_half.clone(), RFloat::with_val(prec, -&sin_half)), (zero.clone(), zero.clone())],
        [(zero.clone(), zero.clone()), (cos_half, sin_half)],
    ]
}

fn main() {
    let theta = 0.3_f64;
    let prec: u32 = 192;
    let theta_mpfr = RFloat::with_val(prec, theta);
    let target_mpfr = rz_mpfr(&theta_mpfr, prec);

    for &eps in &[1.5e-8_f64] {
        eprintln!("─── ε={:e} (MPFR-theta target, prec={prec}) ───", eps);
        let synth = SynthesizerQ::new(eps).with_max_lde(35);
        let t = Instant::now();
        let r = synth.synthesize_mpfr(&target_mpfr);
        let dt = t.elapsed().as_secs_f64();
        match r {
            Some(r) => eprintln!("  RESULT: lde={}, dist={:.3e}, took {:.2} s", r.lde, r.distance, dt),
            None => eprintln!("  RESULT: None after {:.2} s", dt),
        }
    }
}
