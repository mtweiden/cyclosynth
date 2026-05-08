//! ε=1e-8 with relaxed d_R filter [0, 1, 15] instead of strict [0].

use cyclosynth::synthesis::clifford_sqrt_t::{SynthesizerQ, Mat2Mpfr};
use rug::Float as RFloat;
use std::time::Instant;

fn main() {
    std::env::set_var("CYCLOSYNTH_TRACE", "1");
    let theta = 0.3_f64;
    let eps = 1e-8_f64;
    let prec: u32 = 192;

    let theta_mpfr = RFloat::with_val(prec, theta);
    let half = RFloat::with_val(prec, &theta_mpfr / 2);
    let cos_half = half.clone().cos();
    let sin_half = half.clone().sin();
    let zero = RFloat::with_val(prec, 0.0);
    let target_mpfr: Mat2Mpfr = [
        [(cos_half.clone(), RFloat::with_val(prec, -&sin_half)), (zero.clone(), zero.clone())],
        [(zero.clone(), zero.clone()), (cos_half, sin_half)],
    ];

    eprintln!("─── Q at ε=1e-8 MPFR D&C, dc_split=2, dr=[0,1,15] (relaxed) ───");
    let mut synth = SynthesizerQ::new(eps);
    synth.dc_dr_filter = vec![0u32, 1, 15];
    eprintln!(
        "  config: max_lde={}, min_lde={}, dc_split={:?}, dr_filter={:?}",
        synth.max_lde, synth.min_lde, synth.dc_split, synth.dc_dr_filter,
    );
    let t = Instant::now();
    let r = synth.synthesize_mpfr(&target_mpfr);
    let dt = t.elapsed().as_secs_f64();
    match r {
        Some(r) => eprintln!("  RESULT: lde={}, dist={:.3e}, took {:.2} s", r.lde, r.distance, dt),
        None => eprintln!("  RESULT: None after {:.2} s", dt),
    }
}
