//! Probe ε=1e-8 with single-search (dc_split=None). Tests whether the
//! D&C m=2-strict prefix structure is hiding solutions.

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
    std::env::set_var("CYCLOSYNTH_TRACE", "1");
    let target = rz(0.3);
    let eps = 1e-8_f64;

    eprintln!("─── Q at ε=1e-8, single search (dc_split=None) ───");
    let mut synth = SynthesizerQ::new(eps);
    synth.dc_split = None;
    synth.dc_dr_filter = Vec::new();
    eprintln!(
        "  config: max_lde={}, min_lde={}, dc_split={:?}, bkz={}",
        synth.max_lde, synth.min_lde, synth.dc_split, synth.bkz_block_size,
    );
    let t = Instant::now();
    let r = synth.synthesize(target);
    let dt = t.elapsed().as_secs_f64();
    match r {
        Some(r) => eprintln!("  RESULT: lde={}, dist={:.3e}, took {:.2} s", r.lde, r.distance, dt),
        None => eprintln!("  RESULT: None after {:.2} s", dt),
    }
}
