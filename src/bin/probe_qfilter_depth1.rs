//! Diagnostic: measure depth-1 shell-discriminant filter rejection rate.
//!
//! For each z[1] candidate that survives the existing partial_eucl prune
//! (i.e., that would recurse into depth 0), classify by:
//!   D < 0                    — analytical filter would prune (no real z[0])
//!   D ≥ 0, mod-16 says "no"  — D is not a perfect square (no integer z[0])
//!   D ≥ 0, mod-16 says "yes" — could be a perfect square (filter passes)
//!
//! High rejection rate (≥ 80%) means the path-2 filter is structurally
//! powerful and worth the phase-1 budget refactor.
//!
//! Usage: cargo run --release --bin probe_qfilter_depth1 -- <theta> <eps>

use cyclosynth::synthesis::clifford_sqrt_t::SynthesizerQ;
use cyclosynth::synthesis::diag;
use num_complex::Complex;
use std::sync::atomic::Ordering;
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
    std::env::set_var("CYCLOSYNTH_TRACE", "1");
    let theta: f64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(1.1);
    let eps: f64 = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(1.5e-8);

    diag::reset_all();
    let target = rz_f64(theta);
    let synth = SynthesizerQ::new(eps).with_max_lde(35);
    let t0 = Instant::now();
    let r = synth.synthesize(target);
    let dt = t0.elapsed().as_secs_f64();

    let total = diag::N_QFILTER_TOTAL.load(Ordering::Relaxed);
    let d_neg = diag::N_QFILTER_D_NEG.load(Ordering::Relaxed);
    let mod16_bad = diag::N_QFILTER_D_GE0_MOD16_BAD.load(Ordering::Relaxed);
    let not_sq = diag::N_QFILTER_D_GE0_NOT_SQUARE.load(Ordering::Relaxed);
    let perfect = diag::N_QFILTER_PERFECT_SQUARE.load(Ordering::Relaxed);

    println!("=== depth-1 Q-filter: theta={} eps={:e} ===", theta, eps);
    match r {
        Some(r) => println!("  FOUND lde={} dist={:.2e} time={:.2}s", r.lde, r.distance, dt),
        None => println!("  NOT FOUND time={:.2}s", dt),
    }
    println!("  z[1] candidates measured (= depth-1 recurses to depth 0):");
    println!("    total                              {total:>12}");
    if total > 0 {
        let pct = |x: u64| 100.0 * x as f64 / total as f64;
        println!("    [reject] D < 0                     {d_neg:>12} ({:>5.1}%)", pct(d_neg));
        println!("    [reject] D ≥ 0, mod-16 BAD         {mod16_bad:>12} ({:>5.1}%)", pct(mod16_bad));
        println!("    [reject] D ≥ 0, isqrt²≠D           {not_sq:>12} ({:>5.1}%)", pct(not_sq));
        println!("    [PASS]   D is a perfect square     {perfect:>12} ({:>5.1}%)", pct(perfect));
        let reject = d_neg + mod16_bad + not_sq;
        println!("    -- total filter rejections        {reject:>12} ({:>5.1}%)", pct(reject));
    }
}
