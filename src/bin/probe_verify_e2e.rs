//! End-to-end synthesis check at a specific (ε, theta). Args:
//!   <theta> <eps> [<bkz> [<plde_window> [<plde_trigger> [<dc_m> [<dc_filter>]]]]]
//! Defaults: theta=1.1 eps=1.5e-8. Always uses verify_prune_mpfr=on.
//!
//! dc_m: -1 = use auto-default, 0 = disable D&C (single search), N≥1 = force m=N.
//! dc_filter: "auto" (default), "strict" = [0], "relaxed" = [0,1,15], "open" = [].

use cyclosynth::synthesis::clifford_sqrt_t::SynthesizerQ;
use cyclosynth::synthesis::lattice_zeta::set_verify_prune_mpfr;
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
    // Optional budget-trigger threshold (5th arg, node count).
    let plde_trigger = args.get(4).and_then(|s| s.parse::<u64>().ok()).unwrap_or(0);
    if plde_trigger > 0 {
        synth = synth.with_parallel_lde_trigger_nodes(plde_trigger);
        eprintln!("  (parallel-LDE budget trigger: {plde_trigger} nodes)");
    }
    // Optional D&C split override (6th arg). -1 = auto, 0 = disable, N≥1 = force m=N.
    let dc_m: i32 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(-1);
    if dc_m == 0 {
        synth.dc_split = None;
        synth.dc_dr_filter = Vec::new();
        eprintln!("  (D&C disabled — single search)");
    } else if dc_m > 0 {
        synth.dc_split = Some(dc_m as u32);
        eprintln!("  (D&C split m={dc_m})");
    }
    // Optional dc_dr_filter override (7th arg). "auto"/"strict"/"relaxed"/"open".
    if let Some(f) = args.get(6).map(|s| s.as_str()) {
        match f {
            "strict" => { synth.dc_dr_filter = vec![0u32]; }
            "relaxed" => { synth.dc_dr_filter = vec![0u32, 1, 15]; }
            "open" => { synth.dc_dr_filter = Vec::new(); }
            "auto" => {}
            _ => eprintln!("  (unknown filter '{f}'; using auto)"),
        }
        if f != "auto" {
            eprintln!("  (dc_dr_filter={:?})", synth.dc_dr_filter);
        }
    }
    let t0 = Instant::now();
    let result = synth.synthesize(target);
    let dt = t0.elapsed().as_secs_f64();
    match result {
        Some(r) => {
            let (t, q, n) = match &r.gates {
                Some(g) => (
                    g.chars().filter(|&c| c == 'T').count(),
                    g.chars().filter(|&c| c == 'Q').count(),
                    g.chars().count(),
                ),
                None => (0, 0, 0),
            };
            let cost = t + 3 * q;
            println!(
                "theta={} eps={:e} → FOUND lde={} dist={:.2e} time={:.2}s  T={} Q={} len={} cost(T+3Q)={}",
                theta, eps, r.lde, r.distance, dt, t, q, n, cost
            );
        }
        None => println!(
            "theta={} eps={:e} → NOT FOUND time={:.2}s",
            theta, eps, dt
        ),
    }
}
