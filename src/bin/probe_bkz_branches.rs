//! Diagnostic: count how often each BKZ insertion branch fires during
//! a cliff-regime synthesis. Hypothesis: if Branch 1/2 dominate, the
//! prior unimplemented Branch 3 was rarely hit — implementing it adds
//! little perf value.

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
    let theta: f64 = std::env::args().nth(1)
        .and_then(|s| s.parse().ok()).unwrap_or(1.1);
    let eps: f64 = std::env::args().nth(2)
        .and_then(|s| s.parse().ok()).unwrap_or(1.5e-8);

    diag::reset_all();
    let target = rz_f64(theta);
    let synth = SynthesizerQ::new(eps).with_max_lde(35);
    let t0 = Instant::now();
    let r = synth.synthesize(target);
    let dt = t0.elapsed().as_secs_f64();

    let b1 = diag::N_BKZ_BRANCH1.load(Ordering::Relaxed);
    let b2 = diag::N_BKZ_BRANCH2.load(Ordering::Relaxed);
    let b3 = diag::N_BKZ_BRANCH3_SUCCESS.load(Ordering::Relaxed);
    let b3np = diag::N_BKZ_BRANCH3_NONPRIMITIVE.load(Ordering::Relaxed);
    let total = b1 + b2 + b3 + b3np;

    println!("=== BKZ branch frequency: theta={} eps={:e} ===", theta, eps);
    match r {
        Some(r) => println!("  FOUND lde={} dist={:.2e} time={:.1}s", r.lde, r.distance, dt),
        None => println!("  NOT FOUND time={:.1}s", dt),
    }
    println!("  total bkz_insert calls:        {total:>10}");
    if total > 0 {
        let pct = |x: u64| 100.0 * x as f64 / total as f64;
        println!("    branch 1 (single ±1):          {b1:>10} ({:>5.1}%)", pct(b1));
        println!("    branch 2 (some ±1, pivot):     {b2:>10} ({:>5.1}%)", pct(b2));
        println!("    branch 3 success (gcd=1):      {b3:>10} ({:>5.1}%)", pct(b3));
        println!("    branch 3 non-primitive (gcd>1): {b3np:>9} ({:>5.1}%)", pct(b3np));
    }
}
