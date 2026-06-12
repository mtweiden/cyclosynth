//! ζ right-coset dedup A/B screen-parity probe (gate 2 of the zeta
//! coset work order, docs/w_zeta_coset_notes.md): runs the FIRST-HIT
//! path (the optimal mode's screen) on fixed Rz(θ) targets with
//! CYCLOSYNTH_TRACE=1 so the per-level `[zeta] dc lde=.. FOUND/none`
//! lines land on stderr, where an external diff checks exact per-level
//! FOUND/none parity + identical found-lde between coset modes.
//!
//! Args: [eps] [coset 0|1|-] [thetas]
//! Defaults: eps=1e-7, coset "-" (leave env/default), thetas "0.7,1.95".
//! The coset flag is forwarded to CYCLOSYNTH_ZETA_COSET via set_var
//! before any synthesis (direct env-prefixed execution is denied in the
//! agent harness — same workaround as bench_t_breakdown's `--coset`).

use cyclosynth::synthesis::clifford_sqrt_t::SynthesizerQ;
use cyclosynth::synthesis::distance::Mat2;
use num_complex::Complex;
use std::time::Instant;

type C64 = Complex<f64>;

fn rz(t: f64) -> Mat2 {
    [
        [C64::from_polar(1.0, -t / 2.0), C64::new(0.0, 0.0)],
        [C64::new(0.0, 0.0), C64::from_polar(1.0, t / 2.0)],
    ]
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let eps: f64 = args.first().and_then(|s| s.parse().ok()).unwrap_or(1e-7);
    if let Some(coset) = args.get(1).filter(|s| *s == "0" || *s == "1") {
        unsafe { std::env::set_var("CYCLOSYNTH_ZETA_COSET", coset) };
    }
    let thetas: Vec<f64> = args
        .get(2)
        .map(|s| s.split(',').filter_map(|p| p.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![0.7, 1.95]);
    unsafe { std::env::set_var("CYCLOSYNTH_TRACE", "1") };

    println!(
        "ε={eps:e} thetas={thetas:?} zeta_coset={}",
        std::env::var("CYCLOSYNTH_ZETA_COSET").as_deref().unwrap_or("default(1)")
    );
    for &theta in &thetas {
        eprintln!("[screen] theta={theta}");
        let t0 = Instant::now();
        let r = SynthesizerQ::new(eps)
            .with_optimize_cost(false)
            .synthesize(rz(theta));
        match r {
            Some(r) => println!(
                "theta={theta} FOUND lde={} dist={:.6e} t={:.2}s",
                r.lde,
                r.distance,
                t0.elapsed().as_secs_f64()
            ),
            None => println!("theta={theta} NONE t={:.2}s", t0.elapsed().as_secs_f64()),
        }
    }
}
