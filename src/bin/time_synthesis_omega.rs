//! Timing harness for Clifford+T synthesis.
//!
//! Generates N random SU(2) targets via Rz(α)·Ry(β)·Rz(γ) once (deterministic
//! given the seed), then synthesizes each target at every ε level. The same
//! targets are reused across ε so timings are directly comparable.
//!
//! Usage:
//!   time_synthesis_omega [--threads N] [--max-lde N] [--trials N]
//!                  [--n-targets N] [--seed HEX] [--skip-tight]
//!                  [--filter SUBSTRING]
//!
//! Defaults: 8 threads, max-lde auto, 3 trials, 10 targets,
//!           seed 0xC0FFEEBAADD0E, all ε levels.
//! Set `CYCLOSYNTH_TRACE=1` to print per-phase breakdowns (build_q, lll,
//! cholesky, lu, se) for each lde level via the diag subsystem.
//! --skip-tight omits ε ≤ 1e-4.

use cyclosynth::synthesis::Synthesizer;
use num_complex::Complex;
use std::f64::consts::PI;
use std::time::Instant;

type C64 = Complex<f64>;
type Mat2 = [[C64; 2]; 2];

fn mat_mul(a: Mat2, b: Mat2) -> Mat2 {
    [[
        a[0][0]*b[0][0] + a[0][1]*b[1][0],
        a[0][0]*b[0][1] + a[0][1]*b[1][1],
    ],[
        a[1][0]*b[0][0] + a[1][1]*b[1][0],
        a[1][0]*b[0][1] + a[1][1]*b[1][1],
    ]]
}

fn rz(theta: f64) -> Mat2 {
    [[C64::from_polar(1.0, -theta / 2.0), C64::new(0.0, 0.0)],
     [C64::new(0.0, 0.0),                 C64::from_polar(1.0, theta / 2.0)]]
}

fn ry(theta: f64) -> Mat2 {
    let c = (theta / 2.0).cos();
    let s = (theta / 2.0).sin();
    [[C64::new(c, 0.0), C64::new(-s, 0.0)],
     [C64::new(s, 0.0), C64::new(c, 0.0)]]
}

/// SU(2) via Euler decomposition: Rz(a) · Ry(b) · Rz(c).
fn u3(a: f64, b: f64, c: f64) -> Mat2 {
    mat_mul(mat_mul(rz(a), ry(b)), rz(c))
}

/// xorshift64 — small deterministic PRNG with full uniform period.
/// Same seed ⇒ identical sequence across runs and platforms.
fn xorshift64(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// Random angle uniformly in [0, 2π) at f64 precision.
fn rand_angle(state: &mut u64) -> f64 {
    // Take the upper 53 bits to avoid bias when casting to f64.
    let bits = xorshift64(state) >> 11; // 53 bits
    (bits as f64) / ((1u64 << 53) as f64) * 2.0 * PI
}

fn parse_seed(s: &str) -> u64 {
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u64::from_str_radix(hex, 16).expect("--seed hex parse")
    } else {
        s.parse().expect("--seed N (decimal or 0x-prefixed hex)")
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut num_threads: Option<usize> = Some(8);
    let mut max_lde: Option<u32> = None;
    let mut n_trials: usize = 3;
    let mut skip_tight = false;
    let mut n_targets: usize = 10;
    let mut seed: u64 = 0xC0FFEE_BAADD0E_u64;
    let mut filter: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--threads"    => { i += 1; num_threads = Some(args[i].parse().expect("--threads N")); }
            "--max-lde"    => { i += 1; max_lde = Some(args[i].parse().expect("--max-lde N")); }
            "--trials"     => { i += 1; n_trials = args[i].parse().expect("--trials N"); }
            "--n-targets"  => { i += 1; n_targets = args[i].parse().expect("--n-targets N"); }
            "--seed"       => { i += 1; seed = parse_seed(&args[i]); }
            "--skip-tight" => { skip_tight = true; }
            "--filter"     => { i += 1; filter = Some(args[i].clone()); }
            _ => {}
        }
        i += 1;
    }

    if let Some(n) = num_threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
            .expect("failed to build rayon thread pool");
    }
    let n_threads = rayon::current_num_threads();

    // Generate the target set once. Using `seed | 1` guards against the
    // xorshift degenerate state (all-zero), which produces only zeros.
    let mut rng_state = seed | 1;
    let targets: Vec<(String, Mat2, [f64; 3])> = (0..n_targets)
        .map(|idx| {
            let a = rand_angle(&mut rng_state);
            let b = rand_angle(&mut rng_state);
            let c = rand_angle(&mut rng_state);
            (format!("target_{:02}", idx), u3(a, b, c), [a, b, c])
        })
        .collect();

    // ε levels. Coarse runs in milliseconds; tight regimes can take seconds
    // (1e-7) to tens of seconds (1e-8) per target.
    let coarse_eps: Vec<f64> = vec![1e-2, 1e-3];
    let tight_eps: Vec<f64> = vec![1e-4, 1e-5, 1e-6, 1e-7, 1e-8];
    let eps_list: Vec<f64> = if skip_tight {
        coarse_eps
    } else {
        coarse_eps.into_iter().chain(tight_eps).collect()
    };

    let max_lde_label = max_lde
        .map(|v| v.to_string())
        .unwrap_or_else(|| "auto".to_string());
    println!(
        "threads: {n_threads}  max_lde: {max_lde_label}  trials: {n_trials}  \
         targets: {n_targets}  seed: {:#018x}",
        seed
    );

    // Header. Columns: target name | ε | T-count | lde | dist | min_ms | avg_ms.
    println!(
        "{:<12}  {:>6}  {:>4}  {:>4}  {:>11}  {:>10}  {:>10}",
        "target", "eps", "tcnt", "lde", "dist", "min_ms", "avg_ms"
    );
    println!("{}", "─".repeat(64));

    let mut grand_total_min = 0.0_f64;
    let mut grand_failures = 0_usize;

    for &eps in &eps_list {
        let synth = match max_lde {
            Some(v) => Synthesizer::new(eps, false).with_max_lde(v),
            None => Synthesizer::new(eps, false),
        };

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
                    let tcnt = r
                        .gates
                        .as_deref()
                        .map(|g| g.chars().filter(|&c| c == 'T' || c == 't').count())
                        .unwrap_or(0);
                    println!(
                        "{:<12}  {:>6.0e}  {:>4}  {:>4}  {:>11.3e}  {:>10.1}  {:>10.1}",
                        name, eps, tcnt, r.lde, r.distance, min_ms, avg_ms
                    );
                    eps_total_min += min_ms;
                }
                None => {
                    println!(
                        "{:<12}  {:>6.0e}    --    --      --   FAILED (no solution within max_lde={})",
                        name, eps, synth.max_lde()
                    );
                    eps_failures += 1;
                }
            }
        }

        // Per-ε subtotal line. `eps_ran` excludes filter-skipped targets.
        let n_ok = eps_ran.saturating_sub(eps_failures);
        if n_ok > 0 {
            let avg_per_target = eps_total_min / n_ok as f64;
            println!(
                "{:<12}  {:>6.0e}  ── subtotal: min sum {:.1} ms across {} ok ({:.1} ms / target avg){}",
                "", eps, eps_total_min, n_ok, avg_per_target,
                if eps_failures > 0 { format!(", {} failures", eps_failures) } else { String::new() }
            );
        } else {
            println!(
                "{:<12}  {:>6.0e}  ── all {} targets failed",
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
        println!("FAILURES: {grand_failures} runs returned no solution within max_lde");
    }
}
