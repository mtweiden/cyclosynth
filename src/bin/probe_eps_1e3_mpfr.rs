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
    let theta = 0.3_f64;
    let eps = 1e-3_f64;
    let prec: u32 = 128;

    let target = rz_f64(theta);
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

    println!("MPFR path:");
    let synth = SynthesizerQ::new(eps);
    let t = Instant::now();
    let r = synth.synthesize_v_mpfr(&v_mpfr, target);
    let dt = t.elapsed().as_secs_f64();
    match r {
        Some(r) => println!("  lde={}, dist={:.3e}, took {:.3} s", r.lde, r.distance, dt),
        None => println!("  None after {:.3} s", dt),
    }

    println!("f64 path:");
    let synth2 = SynthesizerQ::new(eps);
    let t = Instant::now();
    let r2 = synth2.synthesize(target);
    let dt = t.elapsed().as_secs_f64();
    match r2 {
        Some(r) => println!("  lde={}, dist={:.3e}, took {:.3} s", r.lde, r.distance, dt),
        None => println!("  None after {:.3} s", dt),
    }
}
