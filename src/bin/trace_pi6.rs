//! Minimal driver for tracing n=6 (Clifford+R_z(π/6)) synthesis instrumentation.
//! Synthesizes one fixed target at eps=1e-3 with a deterministic seed.
//! Stderr carries the [SE], [DC], and [OPT] eprintln lines.
//!
//!   cargo run --release --bin trace_pi6 2> /tmp/n6_trace.txt

use cyclosynth::synthesis::clifford_pi6::SynthesizerPi6;
use num_complex::Complex64;

fn rz(theta: f64) -> [[Complex64; 2]; 2] {
    [
        [
            Complex64::from_polar(1.0, -theta / 2.0),
            Complex64::new(0.0, 0.0),
        ],
        [
            Complex64::new(0.0, 0.0),
            Complex64::from_polar(1.0, theta / 2.0),
        ],
    ]
}

fn main() {
    // Deterministic target: Rz(0.3) — the canonical "small angle" test case.
    // Fix k=9 region by using eps=1e-3 (default min_lde ≈ 13, so this exercises
    // k=13+ where the lattice path fires via direct_search_n6).
    let theta = 0.3_f64;
    let target = rz(theta);
    let eps = 1e-3_f64;

    eprintln!("trace_pi6: theta={theta} eps={eps:.0e}");

    let synth = SynthesizerPi6::new(eps).with_max_lde(20);

    match synth.synthesize(target) {
        Some(r) => {
            eprintln!(
                "trace_pi6: SUCCESS lde={} dist={:.3e} gates={:?}",
                r.lde, r.distance, r.gates
            );
        }
        None => {
            eprintln!("trace_pi6: FAILED (no solution within max_lde=20)");
        }
    }
}
