//! Validate ε=1e-8 via the full MPFR D&C pipeline.

use cyclosynth::synthesis::clifford_sqrt_t::{SynthesizerQ, Mat2Mpfr};
use rug::Float as RFloat;
use std::time::Instant;

fn main() {
    std::env::set_var("CYCLOSYNTH_TRACE", "1");
    let theta = 0.3_f64;
    let eps = 1e-8_f64;
    let prec: u32 = 192;

    // Construct Rz(θ) at MPFR precision.
    // Rz(θ) = [[e^(-iθ/2), 0], [0, e^(iθ/2)]]
    let theta_mpfr = RFloat::with_val(prec, theta);
    let half = RFloat::with_val(prec, &theta_mpfr / 2);
    let cos_half = half.clone().cos();
    let sin_half = half.clone().sin();
    let zero = RFloat::with_val(prec, 0.0);
    let target_mpfr: Mat2Mpfr = [
        [
            (cos_half.clone(), RFloat::with_val(prec, -&sin_half)), // e^(-iθ/2) = (cos, -sin)
            (zero.clone(), zero.clone()),
        ],
        [
            (zero.clone(), zero.clone()),
            (cos_half, sin_half),                                    // e^(+iθ/2) = (cos, +sin)
        ],
    ];

    eprintln!("─── Q at ε=1e-8 via synthesize_mpfr (Rz, MPFR-{prec} D&C) ───");
    let synth = SynthesizerQ::new(eps);
    eprintln!(
        "  config: max_lde={}, min_lde={}, dc_split={:?}, dr_filter={:?}, bkz={}",
        synth.max_lde, synth.min_lde, synth.dc_split, synth.dc_dr_filter,
        synth.bkz_block_size,
    );
    let t = Instant::now();
    let r = synth.synthesize_mpfr(&target_mpfr);
    let dt = t.elapsed().as_secs_f64();
    match r {
        Some(r) => eprintln!("  RESULT: lde={}, dist={:.3e}, took {:.2} s", r.lde, r.distance, dt),
        None => eprintln!("  RESULT: None after {:.2} s", dt),
    }
}
