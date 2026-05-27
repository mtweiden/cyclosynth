//! Minimal driver for tracing n=6 (Clifford+R_z(π/6)) synthesis instrumentation.
//! Synthesizes one deterministic random target at configurable eps.
//! Stderr carries the [SE], [DC], and [OPT] eprintln lines.
//!
//!   cargo run --release --bin trace_pi6 2> /tmp/n6_trace.txt

use cyclosynth::synthesis::clifford_pi6::SynthesizerPi6;
use num_complex::Complex64;

fn main() {
    let eps = std::env::var("EPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1e-8_f64);

    let theta = 1.234_567_89_f64;
    let phi = 2.345_678_91_f64;
    let lambda = 3.456_789_12_f64;
    let ct = (theta / 2.0).cos();
    let st = (theta / 2.0).sin();
    let global_phase = Complex64::from_polar(1.0, -(phi + lambda) / 2.0);
    let target = [
        [
            global_phase * Complex64::new(ct, 0.0),
            global_phase * (-Complex64::from_polar(st, lambda)),
        ],
        [
            global_phase * Complex64::from_polar(st, phi),
            global_phase * Complex64::from_polar(ct, phi + lambda),
        ],
    ];

    eprintln!("trace_pi6: fixed U3 target eps={eps:.0e}");

    let synth = SynthesizerPi6::new(eps);

    match synth.synthesize(target) {
        Some(r) => {
            eprintln!(
                "trace_pi6: SUCCESS lde={} dist={:.3e} gates={:?}",
                r.lde, r.distance, r.gates
            );
        }
        None => {
            eprintln!("trace_pi6: FAILED");
        }
    }
}
