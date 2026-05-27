//! Timing harness for Clifford+√T synthesis.
//!
//! Mirrors `time_synthesis.rs` (the Z[ω]/Clifford+T harness) but
//! synthesizes via `Synthesizer (sqrt_t=true)`. Generates N random SU(2) targets
//! via Rz(α)·Ry(β)·Rz(γ) once (deterministic given the seed), then
//! synthesizes each at every ε level. Same targets reused across ε so
//! timings are directly comparable.
//!
//! ## Backend status
//!
//! The current `Synthesizer (sqrt_t=true)` uses a brute-force backend
//! (`phase1_brute`) which is exponential in k. Practical operating
//! range:
//!   - max_lde = 0..2: very fast (milliseconds), tight-ε targets
//!     unreachable.
//!   - max_lde = 3:   ~ms-s, ε ≥ 0.05 typically reachable.
//!   - max_lde = 4:   ~10 s/call, ε ≥ 0.02 typically reachable.
//!   - max_lde ≥ 5:   intractable.
//!
//! When Phase 5b M4-M5 land (16D LLL+SE), max_lde can be lifted to
//! ~30+ and ε down to 1e-7 becomes feasible.
//!
//! Usage:
//!   time_zeta_synthesis [--max-lde N] [--trials N] [--n-targets N]
//!                       [--seed HEX] [--filter SUBSTRING]
//!                       [--eps EPS_LIST]
//!
//! Defaults: max-lde 3, 3 trials, 5 targets, seed 0xC0FFEEBAADD0E,
//! ε ∈ {0.5, 0.2, 0.1, 0.05}.
//!
//! `--eps` accepts a comma-separated list, e.g. `--eps 0.5,0.1,0.05`.

use cyclosynth::synthesis::Synthesizer;
use num_complex::Complex;
use std::f64::consts::PI;
use std::time::Instant;

type C64 = Complex<f64>;
type Mat2 = [[C64; 2]; 2];

fn mat_mul(a: Mat2, b: Mat2) -> Mat2 {
    [
        [
            a[0][0] * b[0][0] + a[0][1] * b[1][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1],
        ],
        [
            a[1][0] * b[0][0] + a[1][1] * b[1][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1],
        ],
    ]
}

fn rz(theta: f64) -> Mat2 {
    [
        [C64::from_polar(1.0, -theta / 2.0), C64::new(0.0, 0.0)],
        [C64::new(0.0, 0.0), C64::from_polar(1.0, theta / 2.0)],
    ]
}

fn ry(theta: f64) -> Mat2 {
    let c = (theta / 2.0).cos();
    let s = (theta / 2.0).sin();
    [
        [C64::new(c, 0.0), C64::new(-s, 0.0)],
        [C64::new(s, 0.0), C64::new(c, 0.0)],
    ]
}

/// SU(2) via Euler decomposition: Rz(a) · Ry(b) · Rz(c).
fn u3(a: f64, b: f64, c: f64) -> Mat2 {
    mat_mul(mat_mul(rz(a), ry(b)), rz(c))
}

/// xorshift64 — small deterministic PRNG.
fn xorshift64(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn rand_angle(state: &mut u64) -> f64 {
    let bits = xorshift64(state) >> 11;
    (bits as f64) / ((1u64 << 53) as f64) * 2.0 * PI
}

fn parse_seed(s: &str) -> u64 {
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u64::from_str_radix(hex, 16).expect("--seed hex parse")
    } else {
        s.parse().expect("--seed N (decimal or 0x-prefixed hex)")
    }
}

fn parse_eps_list(s: &str) -> Vec<f64> {
    s.split(',')
        .map(|t| t.trim().parse::<f64>().expect("--eps value parse"))
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut max_lde: u32 = 3;
    let mut n_trials: usize = 3;
    let mut n_targets: usize = 5;
    let mut seed: u64 = 0xC0FFEE_BAADD0E_u64;
    let mut filter: Option<String> = None;
    let mut eps_list: Vec<f64> = vec![0.5, 0.2, 0.1, 0.05];

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--max-lde" => {
                i += 1;
                max_lde = args[i].parse().expect("--max-lde N");
            }
            "--trials" => {
                i += 1;
                n_trials = args[i].parse().expect("--trials N");
            }
            "--n-targets" => {
                i += 1;
                n_targets = args[i].parse().expect("--n-targets N");
            }
            "--seed" => {
                i += 1;
                seed = parse_seed(&args[i]);
            }
            "--filter" => {
                i += 1;
                filter = Some(args[i].clone());
            }
            "--eps" => {
                i += 1;
                eps_list = parse_eps_list(&args[i]);
            }
            _ => {}
        }
        i += 1;
    }

    // Generate target set once.
    let mut rng_state = seed | 1;
    let targets: Vec<(String, Mat2, [f64; 3])> = (0..n_targets)
        .map(|idx| {
            let a = rand_angle(&mut rng_state);
            let b = rand_angle(&mut rng_state);
            let c = rand_angle(&mut rng_state);
            (format!("target_{:02}", idx), u3(a, b, c), [a, b, c])
        })
        .collect();

    println!(
        "max_lde: {max_lde}  trials: {n_trials}  targets: {n_targets}  \
         seed: {:#018x}  backend: hybrid (brute k≤3, LLL+SE k≥4)",
        seed
    );
    println!(
        "{:<12}  {:>6}  {:>4}  {:>11}  {:>10}  {:>10}",
        "target", "eps", "lde", "dist", "min_ms", "avg_ms"
    );
    println!("{}", "─".repeat(64));

    let mut grand_total_min = 0.0_f64;
    let mut grand_failures = 0_usize;

    for &eps in &eps_list {
        let synth = Synthesizer::new(eps, true).with_max_lde(max_lde);

        let mut eps_total_min = 0.0_f64;
        let mut eps_failures = 0_usize;
        let mut eps_ran = 0_usize;

        for (name, target, _angles) in &targets {
            if let Some(f) = &filter {
                if !name.contains(f.as_str()) {
                    continue;
                }
            }
            eps_ran += 1;

            let mut times = Vec::with_capacity(n_trials);
            let mut last_result = None;

            for trial in 0..n_trials {
                let t0 = Instant::now();
                let result = synth.synthesize(*target);
                times.push(t0.elapsed().as_secs_f64() * 1000.0);
                if trial == n_trials - 1 {
                    last_result = result;
                }
            }

            let min_ms = times.iter().cloned().fold(f64::INFINITY, f64::min);
            let avg_ms = times.iter().sum::<f64>() / times.len() as f64;

            match &last_result {
                Some(r) => {
                    println!(
                        "{:<12}  {:>6.0e}  {:>4}  {:>11.3e}  {:>10.1}  {:>10.1}",
                        name, eps, r.lde, r.distance, min_ms, avg_ms
                    );
                    eps_total_min += min_ms;
                }
                None => {
                    println!(
                        "{:<12}  {:>6.0e}    --      --   no solution within max_lde={}",
                        name, eps, max_lde
                    );
                    eps_failures += 1;
                }
            }
        }

        let n_ok = eps_ran.saturating_sub(eps_failures);
        if n_ok > 0 {
            let avg_per_target = eps_total_min / n_ok as f64;
            println!(
                "{:<12}  {:>6.0e}  ── subtotal: min sum {:.1} ms across {} ok ({:.1} ms/target avg){}",
                "", eps, eps_total_min, n_ok, avg_per_target,
                if eps_failures > 0 { format!(", {} failures", eps_failures) } else { String::new() }
            );
        } else {
            println!(
                "{:<12}  {:>6.0e}  ── all {} targets failed (likely beyond brute-force range)",
                "", eps, eps_failures
            );
        }
        println!("{}", "─".repeat(64));

        grand_total_min += eps_total_min;
        grand_failures += eps_failures;
    }

    println!(
        "total (min): {:.1} ms across {} ε levels × {} targets = {} runs",
        grand_total_min,
        eps_list.len(),
        n_targets,
        eps_list.len() * n_targets
    );
    if grand_failures > 0 {
        println!(
            "{} runs returned no solution within max_lde — expected for ε beyond \
             the brute-force backend's range. Will be addressed when M4-M5 lifts \
             max_lde with the LLL+SE backend.",
            grand_failures
        );
    }
}
