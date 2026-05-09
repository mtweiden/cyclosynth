//! Verify SynthesizerQ::synthesize_with_ct_fallback works at ε=1e-8.

use cyclosynth::synthesis::clifford_sqrt_t::SynthesizerQ;
use num_complex::Complex;
use std::time::Instant;

type C64 = Complex<f64>;
type Mat2 = [[C64; 2]; 2];

fn rz(t: f64) -> Mat2 {
    [[C64::from_polar(1.0, -t / 2.0), C64::new(0.0, 0.0)],
     [C64::new(0.0, 0.0), C64::from_polar(1.0, t / 2.0)]]
}

fn main() {
    // At ε=1e-7 this should hit the Q-path normally and return quickly.
    // At ε=1e-8 it falls through to CT and embeds.
    let target = rz(0.3);
    for &eps in &[1e-7_f64, 1e-8_f64] {
        eprintln!("─── ε={:e} ───", eps);
        let synth = SynthesizerQ::new(eps).with_max_lde(35);
        let t = Instant::now();
        let r = synth.synthesize_with_ct_fallback(target);
        let dt = t.elapsed().as_secs_f64();
        match r {
            Some(r) => eprintln!("  RESULT: lde={}, dist={:.3e}, took {:.2} s", r.lde, r.distance, dt),
            None => eprintln!("  RESULT: None after {:.2} s", dt),
        }
    }
}
