//! Compare f64 single-search vs MPFR single-search at ε=2e-8.

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
    let eps = 2e-8_f64;
    let prec: u32 = 192;
    let target = rz_f64(theta);

    eprintln!("─── f64 single-search at ε=2e-8 ───");
    let mut synth_f64 = SynthesizerQ::new(eps);
    synth_f64.dc_split = None;
    synth_f64.dc_dr_filter = Vec::new();
    synth_f64.max_lde = 35;
    let t = Instant::now();
    let r = synth_f64.synthesize(target);
    let dt = t.elapsed().as_secs_f64();
    match r {
        Some(r) => eprintln!("  lde={}, dist={:.3e}, took {:.2} s", r.lde, r.distance, dt),
        None => eprintln!("  None after {:.2} s", dt),
    }

    eprintln!("─── MPFR single-search at ε=2e-8 ───");
    let theta_mpfr = RFloat::with_val(prec, theta);
    let half = RFloat::with_val(prec, &theta_mpfr / 2);
    let cos_half = half.clone().cos();
    let sin_half = half.clone().sin();
    let zero = RFloat::with_val(prec, 0.0);
    let v_mpfr: [RFloat; 4] = [
        cos_half, RFloat::with_val(prec, -&sin_half), zero.clone(), zero,
    ];
    let synth_mpfr = SynthesizerQ::new(eps);
    let t = Instant::now();
    let r2 = synth_mpfr.synthesize_v_mpfr(&v_mpfr, target);
    let dt = t.elapsed().as_secs_f64();
    match r2 {
        Some(r) => eprintln!("  lde={}, dist={:.3e}, took {:.2} s", r.lde, r.distance, dt),
        None => eprintln!("  None after {:.2} s", dt),
    }
}
