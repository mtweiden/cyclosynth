//! Probe ε=1e-8 starting at lde=24 (we know lde=22-24 hit budget without
//! finding; ε=2e-8 found at lde=23 with dist=1.71e-8, so ε=1e-8 needs
//! lde≥24 to halve the distance below ε).

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

    eprintln!("─── Q at ε=1e-8, min_lde=24 ───");
    let synth = SynthesizerQ::new(eps).with_min_lde(24).with_max_lde(40);
    eprintln!(
        "  config: max_lde={}, min_lde={}, dc_split={:?}, dr_filter={:?}, bkz={}",
        synth.max_lde, synth.min_lde, synth.dc_split, synth.dc_dr_filter,
        synth.bkz_block_size,
    );
    let t = Instant::now();
    let r = synth.synthesize(target);
    let dt = t.elapsed().as_secs_f64();
    match r {
        Some(r) => eprintln!("  RESULT: lde={}, dist={:.3e}, took {:.2} s", r.lde, r.distance, dt),
        None => eprintln!("  RESULT: None after {:.2} s", dt),
    }
}
