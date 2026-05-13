//! Timing harness for Clifford+T synthesis.
//!
<<<<<<< HEAD
//! Usage:
//!   time_synthesis [--threads N] [--max-lde N] [--trials N] [--skip-tight]
//!
//! Defaults: 8 threads, max-lde 50, 3 trials.
//! Build with --features profiling to see per-phase breakdowns.
//! Use --skip-tight to omit the slow 1e-4 cases.

use cyclosynth::synthesis::synthesizer::Synthesizer;
=======
//! Generates N random SU(2) targets via Rz(α)·Ry(β)·Rz(γ) once (deterministic
//! given the seed), then synthesizes each target at every ε level. The same
//! targets are reused across ε so timings are directly comparable.
//!
//! Usage:
//!   time_synthesis [--threads N] [--max-lde N] [--trials N]
//!                  [--n-targets N] [--seed HEX] [--skip-tight]
//!                  [--filter SUBSTRING]
//!
//! Defaults: 8 threads, max-lde auto, 3 trials, 10 targets,
//!           seed 0xC0FFEEBAADD0E, all ε levels.
//! Set `CYCLOSYNTH_TRACE=1` to print per-phase breakdowns (build_q, lll,
//! cholesky, lu, se) for each lde level via the diag subsystem.
//! --skip-tight omits ε ≤ 1e-4.

use cyclosynth::synthesis::Synthesizer;
>>>>>>> 1e7acb8c776129424a7185a30afdea183b0fc2be
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

<<<<<<< HEAD
// General SU(2) via Euler decomposition: Rz(a) · Ry(b) · Rz(c)
=======
/// SU(2) via Euler decomposition: Rz(a) · Ry(b) · Rz(c).
>>>>>>> 1e7acb8c776129424a7185a30afdea183b0fc2be
fn u3(a: f64, b: f64, c: f64) -> Mat2 {
    mat_mul(mat_mul(rz(a), ry(b)), rz(c))
}

<<<<<<< HEAD
=======
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

>>>>>>> 1e7acb8c776129424a7185a30afdea183b0fc2be
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut num_threads: Option<usize> = Some(8);
    let mut max_lde: Option<u32> = None;
    let mut n_trials: usize = 3;
    let mut skip_tight = false;
<<<<<<< HEAD

=======
    let mut n_targets: usize = 10;
    let mut seed: u64 = 0xC0FFEE_BAADD0E_u64;
>>>>>>> 1e7acb8c776129424a7185a30afdea183b0fc2be
    let mut filter: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
<<<<<<< HEAD
            "--threads" => { i += 1; num_threads = Some(args[i].parse().expect("--threads N")); }
            "--max-lde" => { i += 1; max_lde = Some(args[i].parse().expect("--max-lde N")); }
            "--trials"  => { i += 1; n_trials = args[i].parse().expect("--trials N"); }
            "--skip-tight" => { skip_tight = true; }
            "--filter"  => { i += 1; filter = Some(args[i].clone()); }
=======
            "--threads"    => { i += 1; num_threads = Some(args[i].parse().expect("--threads N")); }
            "--max-lde"    => { i += 1; max_lde = Some(args[i].parse().expect("--max-lde N")); }
            "--trials"     => { i += 1; n_trials = args[i].parse().expect("--trials N"); }
            "--n-targets"  => { i += 1; n_targets = args[i].parse().expect("--n-targets N"); }
            "--seed"       => { i += 1; seed = parse_seed(&args[i]); }
            "--skip-tight" => { skip_tight = true; }
            "--filter"     => { i += 1; filter = Some(args[i].clone()); }
>>>>>>> 1e7acb8c776129424a7185a30afdea183b0fc2be
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
<<<<<<< HEAD

    let n_threads = rayon::current_num_threads();

    let r = std::f64::consts::FRAC_1_SQRT_2;
    let h: Mat2 = [[C64::new(r, 0.0), C64::new(r, 0.0)],
                   [C64::new(r, 0.0), C64::new(-r, 0.0)]];
    let id: Mat2 = [[C64::new(1.0, 0.0), C64::new(0.0, 0.0)],
                    [C64::new(0.0, 0.0), C64::new(1.0, 0.0)]];

    // cases: (name, matrix, epsilon)
    // Rz-only, eps=1e-2
    // Rz-only, eps=1e-3
    // General SU(2) (Ry and Rz*Ry*Rz), eps=1e-2 and 1e-3
    // Tight eps=1e-4
    let cases: Vec<(&str, Mat2, f64)> = vec![
        // ── eps = 1e-2 ─────────────────────────────────────────────────────────
        ("identity",             id,                  1e-2),
        ("H",                    h,                   1e-2),
        ("T",                    rz(PI / 4.0),        1e-2),
        ("Rz(0.30)_1e-2",        rz(0.3),             1e-2),
        ("Rz(1.34)_1e-2",        rz(1.34),            1e-2),
        ("Rz(pi/7)_1e-2",        rz(PI / 7.0),        1e-2),
        ("Ry(0.50)_1e-2",        ry(0.5),             1e-2),
        ("U3(0.3,0.7,1.2)_1e-2", u3(0.3, 0.7, 1.2),   1e-2),
        ("U3(1.1,0.4,2.3)_1e-2", u3(1.1, 0.4, 2.3),   1e-2),
        // ── eps = 1e-3 ─────────────────────────────────────────────────────────
        ("Rz(0.30)_1e-3",        rz(0.3),             1e-3),
        ("Rz(1.34)_1e-3",        rz(1.34),            1e-3),
        ("Rz(pi/7)_1e-3",        rz(PI / 7.0),        1e-3),
        ("Ry(0.50)_1e-3",        ry(0.5),             1e-3),
        ("U3(0.3,0.7,1.2)_1e-3", u3(0.3, 0.7, 1.2),   1e-3),
    ];

    let tight_cases: Vec<(&str, Mat2, f64)> = vec![
        // ── eps = 1e-4 (slow, skip with --skip-tight) ───────────────────────────
        ("Rz(0.30)_1e-4",        rz(0.3),             1e-4),
        ("Ry(pi/7)_1e-4",        ry(PI / 7.0),        1e-4),
        ("U3(0.3,0.7,1.2)_1e-4", u3(0.3, 0.7, 1.2),   1e-4),
        ("U3(4.3,1.8,0.2)_1e-4", u3(4.3, 1.8, 0.2),   1e-4),
        ("U3(6.1,3.4,3.3)_1e-4", u3(6.1, 3.4, 3.3),   1e-4),
        // ── eps = 1e-5 ──────────────────────────────────────────────────────────
        ("Rz(0.30)_1e-5",        rz(0.3),             1e-5),
        ("Ry(pi/7)_1e-5",        ry(PI / 7.0),        1e-5),
        ("U3(0.3,0.7,1.2)_1e-5", u3(0.3, 0.7, 1.2),   1e-5),
        ("U3(4.3,1.8,0.2)_1e-5", u3(4.3, 1.8, 0.2),   1e-5),
        ("U3(6.1,3.4,3.3)_1e-5", u3(6.1, 3.4, 3.3),   1e-5),
        // ── eps = 1e-6 ──────────────────────────────────────────────────────────
        ("Rz(0.30)_1e-6",        rz(0.3),             1e-6),
        ("Ry(pi/7)_1e-6",        ry(PI / 7.0),        1e-6),
        ("U3(0.3,0.7,1.2)_1e-6", u3(0.3, 0.7, 1.2),   1e-6),
        ("U3(4.3,1.8,0.2)_1e-6", u3(4.3, 1.8, 0.2),   1e-6),
        ("U3(6.1,3.4,3.3)_1e-6", u3(6.1, 3.4, 3.3),   1e-6),
        // ── eps = 1e-7 ──────────────────────────────────────────────────────────
        ("Rz(0.30)_1e-7",        rz(0.3),             1e-7),
        ("Ry(pi/7)_1e-7",        ry(PI / 7.0),        1e-7),
        ("U3(0.3,0.7,1.2)_1e-7", u3(0.3, 0.7, 1.2),   1e-7),
        ("U3(4.3,1.8,0.2)_1e-7", u3(4.3, 1.8, 0.2),   1e-7),
        ("U3(6.1,3.4,3.3)_1e-7", u3(6.1, 3.4, 3.3),   1e-7),
        // ── eps = 1e-8 (stretch goal) ──────────────────────────────────────────
        ("Rz(0.30)_1e-8",        rz(0.3),             1e-8),
    ];

    let cases: Vec<(&str, Mat2, f64)> = if skip_tight {
        cases
    } else {
        cases.into_iter().chain(tight_cases.into_iter()).collect()
    };

    let cases: Vec<(&str, Mat2, f64)> = if let Some(f) = &filter {
        cases.into_iter().filter(|(n, _, _)| n.contains(f.as_str())).collect()
    } else {
        cases
    };

    let max_lde_label = max_lde.map(|v| v.to_string()).unwrap_or_else(|| "auto".to_string());
    println!("threads: {n_threads}  max_lde: {max_lde_label}  trials: {n_trials}");
    println!("{:<26} {:>6}  {:>4}  {:>10}  {:>10}  {:>10}",
             "name", "eps", "lde", "dist", "min_ms", "avg_ms");
    println!("{}", "-".repeat(76));

    let mut total_min_ms = 0.0_f64;

    for (name, target, eps) in &cases {
        let synth = match max_lde {
            Some(v) => Synthesizer::new(*eps).with_max_lde(v),
            None => Synthesizer::new(*eps),
        };

        let mut times = Vec::with_capacity(n_trials);
        let mut last_result = None;

        for trial in 0..n_trials {
            #[cfg(feature = "profiling")]
            if trial == n_trials - 1 {
                // Only profile the last trial to avoid cross-trial contamination
                cyclosynth::synthesis::synthesizer::reset_profiling();
            }

            let t0 = Instant::now();
            let result = synth.synthesize(*target);
            times.push(t0.elapsed().as_secs_f64() * 1000.0);
            if trial == n_trials - 1 { last_result = result; }
        }

        let min_ms = times.iter().cloned().fold(f64::INFINITY, f64::min);
        let avg_ms = times.iter().sum::<f64>() / times.len() as f64;
        total_min_ms += min_ms;

        match &last_result {
            Some(r) => println!(
                "{:<26} {:>6.0e}  {:>4}  {:>10.3e}  {:>10.1}  {:>10.1}",
                name, eps, r.lde, r.distance, min_ms, avg_ms
            ),
            None => println!(
                "{:<26} {:>6.0e}  FAILED (no solution within max_lde={})",
                name, eps, synth.max_lde
            ),
        }

        #[cfg(feature = "profiling")]
        cyclosynth::synthesis::synthesizer::report_profiling();
    }

    println!("{}", "-".repeat(76));
    println!("total (min): {total_min_ms:.1} ms");
=======
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

    // Header. Columns: target name | ε | lde | dist | min_ms | avg_ms.
    println!(
        "{:<12}  {:>6}  {:>4}  {:>11}  {:>10}  {:>10}",
        "target", "eps", "lde", "dist", "min_ms", "avg_ms"
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
                    println!(
                        "{:<12}  {:>6.0e}  {:>4}  {:>11.3e}  {:>10.1}  {:>10.1}",
                        name, eps, r.lde, r.distance, min_ms, avg_ms
                    );
                    eps_total_min += min_ms;
                }
                None => {
                    println!(
                        "{:<12}  {:>6.0e}    --      --   FAILED (no solution within max_lde={})",
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
>>>>>>> 1e7acb8c776129424a7185a30afdea183b0fc2be
}
