//! Probe ε=1e-8 via the MPFR-precision entry point (`synthesize_v_mpfr`).
//! Constructs Rz(0.3) at MPFR-128 precision, bypassing the f64 floor.

use cyclosynth::synthesis::clifford_sqrt_t::SynthesizerQ;
use num_complex::Complex;
use rug::Float as RFloat;
use std::time::Instant;

type C64 = Complex<f64>;
type Mat2 = [[C64; 2]; 2];

fn rz_f64(t: f64) -> Mat2 {
    [[C64::from_polar(1.0, -t / 2.0), C64::new(0.0, 0.0)],
     [C64::new(0.0, 0.0), C64::from_polar(1.0, t / 2.0)]]
}

fn main() {
    std::env::set_var("CYCLOSYNTH_TRACE", "1");
    let theta = 0.3_f64;
    let eps = 1e-8_f64;
    let prec: u32 = 192;

    // f64 target for distance check (fine — diamond_distance_u2q_float is MPFR-direct).
    let target = rz_f64(theta);

    // MPFR v from MPFR theta. Rz(θ) = [[e^(-iθ/2), 0], [0, e^(iθ/2)]];
    // column 1 is (e^(-iθ/2), 0). v = (Re, Im, 0, 0) = (cos(θ/2), -sin(θ/2), 0, 0).
    let theta_mpfr = RFloat::with_val(prec, theta);
    let half = RFloat::with_val(prec, &theta_mpfr / 2);
    let cos_half = half.clone().cos();
    let sin_half = half.clone().sin();
    let zero = RFloat::with_val(prec, 0.0);
    let v_mpfr: [RFloat; 4] = [
        cos_half,
        RFloat::with_val(prec, -&sin_half),
        zero.clone(),
        zero,
    ];

    eprintln!("─── Q at ε=1e-8 via synthesize_v_mpfr (Rz, MPFR-{prec}) ───");
    let synth = SynthesizerQ::new(eps);
    eprintln!(
        "  config: max_lde={}, min_lde={}, bkz={}",
        synth.max_lde, synth.min_lde, synth.bkz_block_size,
    );
    let t = Instant::now();
    let r = synth.synthesize_v_mpfr(&v_mpfr, target);
    let dt = t.elapsed().as_secs_f64();
    match r {
        Some(r) => eprintln!("  RESULT: lde={}, dist={:.3e}, took {:.2} s", r.lde, r.distance, dt),
        None => eprintln!("  RESULT: None after {:.2} s", dt),
    }
}
