//! Timing harness for Clifford+T synthesis.
//!
//! Usage:
//!   time_synthesis [--threads N] [--max-lde N]
//!
//! RAYON_NUM_THREADS env var also controls thread count.

use cyclosynth::synthesis::synthesizer::Synthesizer;
use num_complex::Complex;
use std::f64::consts::PI;
use std::time::Instant;

type C64 = Complex<f64>;
type Mat2 = [[C64; 2]; 2];

fn rz(theta: f64) -> Mat2 {
    [
        [C64::from_polar(1.0, -theta / 2.0), C64::new(0.0, 0.0)],
        [C64::new(0.0, 0.0), C64::from_polar(1.0, theta / 2.0)],
    ]
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut num_threads: Option<usize> = None;
    let mut max_lde: u32 = 50;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--threads" => {
                i += 1;
                num_threads = Some(args[i].parse().expect("--threads requires a number"));
            }
            "--max-lde" => {
                i += 1;
                max_lde = args[i].parse().expect("--max-lde requires a number");
            }
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
    let h: Mat2 = [
        [C64::new(r, 0.0), C64::new(r, 0.0)],
        [C64::new(r, 0.0), C64::new(-r, 0.0)],
    ];
    let id: Mat2 = [
        [C64::new(1.0, 0.0), C64::new(0.0, 0.0)],
        [C64::new(0.0, 0.0), C64::new(1.0, 0.0)],
    ];

    let cases: Vec<(&str, Mat2, f64)> = vec![
        ("identity",       id,           0.01),
        ("H_gate",         h,            0.01),
        ("T_gate",         rz(PI / 4.0), 0.01),
        ("Rz_0.3",         rz(0.3),      0.01),
        ("Rz_1.34",        rz(1.34),     0.01),
        ("Rz_pi/7",        rz(PI / 7.0), 0.01),
        ("Rz_0.3_tight",   rz(0.3),      0.001),
        ("Rz_1.34_tight",  rz(1.34),     0.001),
    ];

    println!("threads: {n_threads}  max_lde: {max_lde}");
    println!("{:<18} {:>7}  {:>4}  {:>10}  {:>10}", "name", "eps", "lde", "dist", "time_ms");
    println!("{}", "-".repeat(60));

    let mut total_ms = 0.0_f64;

    for (name, target, eps) in &cases {
        #[cfg(feature = "profiling")]
        cyclosynth::synthesis::synthesizer::reset_profiling();

        let synth = Synthesizer::new(*eps).with_max_lde(max_lde);
        let t0 = Instant::now();
        let result = synth.synthesize(*target);
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        total_ms += elapsed_ms;

        match result {
            Some(r) => println!(
                "{:<18} {:>7.4}  {:>4}  {:>10.3e}  {:>10.1}",
                name, eps, r.lde, r.distance, elapsed_ms
            ),
            None => println!(
                "{:<18} {:>7.4}  FAILED (no solution within max_lde={max_lde})",
                name, eps
            ),
        }

        #[cfg(feature = "profiling")]
        cyclosynth::synthesis::synthesizer::report_profiling();
    }

    println!("{}", "-".repeat(60));
    println!("total: {total_ms:.1} ms");
}
