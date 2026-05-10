//! Compare MPFR D&C path vs f64 D&C path at ε=1e-7 / Rz(0.3).
//! If MPFR finds at higher lde than f64, the MPFR D&C has a bug.

use cyclosynth::synthesis::clifford_sqrt_t::{SynthesizerQ, Mat2Mpfr, mat2_to_mat2_mpfr};
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
    let target = rz_f64(theta);

    for &eps in &[2e-8_f64, 1.5e-8, 1.2e-8, 1.1e-8, 1.05e-8, 1.0e-8] {
        eprintln!("─── ε={:e} ───", eps);

        let synth_f64 = SynthesizerQ::new(eps);
        let t = Instant::now();
        let r_f64 = synth_f64.synthesize(target);
        let dt_f64 = t.elapsed().as_secs_f64();
        match &r_f64 {
            Some(r) => eprintln!("  f64  : lde={}, dist={:.3e}, took {:.2} s", r.lde, r.distance, dt_f64),
            None => eprintln!("  f64  : None after {:.2} s", dt_f64),
        }

        // MPFR-precision Mat2 (lifted from f64 — no precision gain in target,
        // but uses MPFR pipeline internally).
        let prec: u32 = 192;
        let target_mpfr: Mat2Mpfr = mat2_to_mat2_mpfr(&target, prec);
        let synth_mpfr = SynthesizerQ::new(eps);
        let t = Instant::now();
        let r_mpfr = synth_mpfr.synthesize_mpfr(&target_mpfr);
        let dt_mpfr = t.elapsed().as_secs_f64();
        match &r_mpfr {
            Some(r) => eprintln!("  MPFR : lde={}, dist={:.3e}, took {:.2} s", r.lde, r.distance, dt_mpfr),
            None => eprintln!("  MPFR : None after {:.2} s", dt_mpfr),
        }

        let _ = (r_f64, r_mpfr);
        eprintln!();
    }
}
