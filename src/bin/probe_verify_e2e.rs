//! End-to-end synthesis check at a specific (ε, theta). Args:
//!   cargo run --release --bin probe_verify_e2e -- <theta> <eps>
//! Defaults: theta=1.1 eps=1.5e-8. Always uses verify_prune_mpfr=on.

use cyclosynth::synthesis::clifford_sqrt_t::SynthesizerQ;
use cyclosynth::synthesis::lenstra_zeta::set_verify_prune_mpfr;
use num_complex::Complex;
use std::time::Instant;

type C64 = Complex<f64>;
type Mat2 = [[C64; 2]; 2];

fn rz_f64(t: f64) -> Mat2 {
    [
        [C64::from_polar(1.0, -t / 2.0), C64::new(0.0, 0.0)],
        [C64::new(0.0, 0.0), C64::from_polar(1.0, t / 2.0)],
    ]
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let theta: f64 = args.first().and_then(|s| s.parse().ok()).unwrap_or(1.1);
    let eps: f64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1.5e-8);
    // Auto-enable verify happens inside synthesize() for ε < 2e-8.
    // Don't force it here, so we exercise the production path.
    let _ = set_verify_prune_mpfr;
    let bkz_override = args.get(2).and_then(|s| s.parse::<u32>().ok());
    // Optional parallel-LDE window (4th arg). >=2 enables parallel speculation.
    let plde_window = args.get(3).and_then(|s| s.parse::<u32>().ok()).unwrap_or(1);
    let target = rz_f64(theta);
    let mut synth = SynthesizerQ::new(eps).with_max_lde(35);
    if let Some(bs) = bkz_override {
        synth = synth.with_bkz(bs);
        eprintln!("  (BKZ block_size override: {bs})");
    }
    if plde_window > 1 {
        synth = synth.with_parallel_lde_window(plde_window);
        eprintln!("  (parallel-LDE window: {plde_window})");
    }
    let t0 = Instant::now();
    let result = synth.synthesize(target);
    let dt = t0.elapsed().as_secs_f64();
    match result {
        Some(r) => println!(
            "theta={} eps={:e} verify=on → FOUND lde={} dist={:.2e} time={:.2}s",
            theta, eps, r.lde, r.distance, dt
        ),
        None => println!(
            "theta={} eps={:e} verify=on → NOT FOUND time={:.2}s",
            theta, eps, dt
        ),
    }
}
