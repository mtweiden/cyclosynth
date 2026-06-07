//! Read-only side-by-side trace for PROMPT_confirm_n12_cholesky.md.
//!
//! Runs `synthesize_circuit_at_k` (no wrapper) at the same seed twice:
//!   - WORKING control: ε=1e-3, k=8.
//!   - FAILING:         ε=1e-4, k=12.
//!   - Extra:           ε=1e-4, k=16.
//!
//! With env var `CYCLOSYNTH_TRACE_DEEP_EPS=1` set, the source-side
//! read-only instrumentation prints labeled lines at each of the five
//! stages (Q metric → LLL → Cholesky → LU → SE). The point is to name
//! the FIRST stage that bails, side by side with the control.
//!
//! Run via:
//!   cargo test --release --test trace_n12_deep_eps -- --ignored --nocapture

use cyclosynth::synthesis::clifford_pi12::synthesize_circuit_at_k;
use num_complex::Complex64;
use rand::{rngs::StdRng, Rng, SeedableRng};

fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

fn haar_target(seed: u64) -> [[Complex64; 2]; 2] {
    let mut rng = StdRng::seed_from_u64(seed);
    loop {
        let raw: [f64; 4] = std::array::from_fn(|_| {
            let mut s = 0.0;
            for _ in 0..12 {
                s += rng.random::<f64>();
            }
            s - 6.0
        });
        let v00 = c(raw[0], raw[1]);
        let v10 = c(raw[2], raw[3]);
        let n = (v00.norm_sqr() + v10.norm_sqr()).sqrt();
        if n < 1e-6 {
            continue;
        }
        let v00 = v00 / n;
        let v10 = v10 / n;
        return [[v00, -v10.conj()], [v10, v00.conj()]];
    }
}

fn run_one(label: &str, seed: u64, k: u32, eps: f64) {
    eprintln!("\n────────────────────────────────────────────────────────");
    eprintln!("▶ {label}  seed={seed}  k={k}  ε={eps:.0e}");
    eprintln!("────────────────────────────────────────────────────────");
    let target = haar_target(seed);
    // SAFETY: serial test execution; env var read by the lib.
    unsafe {
        std::env::set_var("CYCLOSYNTH_TRACE_DEEP_EPS", "1");
    }
    let t0 = std::time::Instant::now();
    let r = synthesize_circuit_at_k(&target, k, eps);
    let dt = t0.elapsed();
    unsafe {
        std::env::remove_var("CYCLOSYNTH_TRACE_DEEP_EPS");
    }
    eprintln!(
        "▶ {label}  result = {}  wall = {:.1} ms\n",
        if r.is_some() { "Some(...)" } else { "None" },
        dt.as_secs_f64() * 1000.0
    );
}

#[test]
#[ignore = "diagnostic-only; run via cargo test --release --test trace_n12_deep_eps \
            -- --ignored --nocapture"]
fn trace_failing_vs_working() {
    let seed = 0;
    // WORKING control first so the magnitudes are visible above the
    // failing run.
    run_one("WORKING (ε=1e-3, k=8)", seed, 8, 1e-3);
    run_one("FAILING (ε=1e-4, k=12)", seed, 12, 1e-4);
    run_one("FAILING (ε=1e-4, k=16)", seed, 16, 1e-4);
}
