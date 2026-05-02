//! Timing harness for Clifford+T synthesis.
//!
//! Usage:
//!   time_synthesis [--threads N] [--max-lde N] [--trials N] [--skip-tight]
//!
//! Defaults: 8 threads, max-lde 50, 3 trials.
//! Build with --features profiling to see per-phase breakdowns.
//! Use --skip-tight to omit the slow 1e-4 cases.

use cyclosynth::synthesis::synthesizer::Synthesizer;
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

// General SU(2) via Euler decomposition: Rz(a) · Ry(b) · Rz(c)
fn u3(a: f64, b: f64, c: f64) -> Mat2 {
    mat_mul(mat_mul(rz(a), ry(b)), rz(c))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut num_threads: Option<usize> = Some(8);
    let mut max_lde: Option<u32> = None;
    let mut n_trials: usize = 3;
    let mut skip_tight = false;

    let mut filter: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--threads" => { i += 1; num_threads = Some(args[i].parse().expect("--threads N")); }
            "--max-lde" => { i += 1; max_lde = Some(args[i].parse().expect("--max-lde N")); }
            "--trials"  => { i += 1; n_trials = args[i].parse().expect("--trials N"); }
            "--skip-tight" => { skip_tight = true; }
            "--filter"  => { i += 1; filter = Some(args[i].clone()); }
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
        ("Rz(pi/7)_1e-4",        rz(PI / 7.0),        1e-4),
        ("U3(0.3,0.7,1.2)_1e-4", u3(0.3, 0.7, 1.2),   1e-4),
        // ── eps = 1e-5 ──────────────────────────────────────────────────────────
        ("Rz(0.30)_1e-5",        rz(0.3),             1e-5),
        ("Rz(pi/7)_1e-5",        rz(PI / 7.0),        1e-5),
        ("U3(0.3,0.7,1.2)_1e-5", u3(0.3, 0.7, 1.2),   1e-5),
        // ── eps = 1e-6 ──────────────────────────────────────────────────────────
        ("Rz(0.30)_1e-6",        rz(0.3),             1e-6),
        ("Rz(pi/7)_1e-6",        rz(PI / 7.0),        1e-6),
        ("U3(0.3,0.7,1.2)_1e-6", u3(0.3, 0.7, 1.2),   1e-6),
        // ── eps = 1e-7 ──────────────────────────────────────────────────────────
        ("Rz(0.30)_1e-7",        rz(0.3),             1e-7),
        ("Rz(pi/7)_1e-7",        rz(PI / 7.0),        1e-7),
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
}
