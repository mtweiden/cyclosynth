//! Emit a CSV in the structure of `comparison_sqrtt_data.csv`, using
//! Clifford+T defaults vs Clifford+√T with Stage-4 optimisation
//! (`with_optimize_cost(true).with_optimal_lde_window(1)`).
//!
//! Target generator: `rz(α) · ry(β) · rz(γ)` with α, β, γ ∈ [0, 2π),
//! matching `comparison_sqrtt.py`. Costs reported:
//!   * `cost` = T + 3·Q   (matches the synthesizer's internal optimisation
//!     target in `gates_cost`)
//!   * `non_clifford_depth` = T + 1.5·Q  (NCD with H/S/X/Y/Z free)
//!
//! Args:
//!   <n_trials> [<seed> [<eps1> <eps2> ...]]
//! Defaults: n=20, seed=42, eps_list=[1e-3, 1e-4, 1e-5, 1e-6].
//!
//! Writes CSV to stdout; redirect to a file. Progress prints go to
//! stderr so they don't pollute the CSV stream.
//!
//! Example:
//!   cargo run --release --bin comparison_sqrtt_stage4 -- \
//!       20 42 1e-3 1e-4 1e-5 1e-6 > comparison_sqrtt_data_stage4.csv

use cyclosynth::synthesis::clifford_sqrt_t::SynthesizerQ;
use cyclosynth::synthesis::clifford_t::SynthesizerT;
use cyclosynth::synthesis::distance::Mat2;
use num_complex::Complex;
use std::f64::consts::PI;
use std::io::Write;
use std::time::Instant;

type C64 = Complex<f64>;

struct Xs(u64);
impl Xs {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

fn rz(t: f64) -> Mat2 {
    [
        [C64::from_polar(1.0, -t / 2.0), C64::new(0.0, 0.0)],
        [C64::new(0.0, 0.0), C64::from_polar(1.0, t / 2.0)],
    ]
}

fn ry(t: f64) -> Mat2 {
    let (c, s) = ((t / 2.0).cos(), (t / 2.0).sin());
    [
        [C64::new(c, 0.0), C64::new(-s, 0.0)],
        [C64::new(s, 0.0), C64::new(c, 0.0)],
    ]
}

fn matmul(a: Mat2, b: Mat2) -> Mat2 {
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

fn count_gates(g: &str) -> (usize, usize) {
    let t = g.chars().filter(|&c| c == 'T').count();
    let q = g.chars().filter(|&c| c == 'Q').count();
    (t, q)
}

const T_COST: f64 = 1.0;
const Q_COST: f64 = 3.0;

fn cost_of(t: usize, q: usize) -> f64 {
    T_COST * t as f64 + Q_COST * q as f64
}

/// Non-Clifford depth: T-equivalent with H/S/X/Y/Z free.
/// T weighs 1, Q (= √T) weighs 1.5.
fn non_clifford_depth_of(t: usize, q: usize) -> f64 {
    t as f64 + 1.5 * q as f64
}

fn run_t(target: Mat2, eps: f64) -> (Option<String>, f64, f64) {
    let t0 = Instant::now();
    let r = SynthesizerT::new(eps).synthesize(target);
    let dt_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let dist = r.as_ref().map(|r| r.distance).unwrap_or(f64::INFINITY);
    (r.and_then(|r| r.gates), dist, dt_ms)
}

fn run_q_stage4(target: Mat2, eps: f64) -> (Option<String>, f64, f64) {
    let t0 = Instant::now();
    let r = SynthesizerQ::new(eps)
        .with_optimize_cost(true)
        .with_optimal_lde_window(1)
        .synthesize(target);
    let dt_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let dist = r.as_ref().map(|r| r.distance).unwrap_or(f64::INFINITY);
    (r.and_then(|r| r.gates), dist, dt_ms)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let n: usize = args.first().and_then(|s| s.parse().ok()).unwrap_or(20);
    let seed: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(42);
    let eps_list: Vec<f64> = if args.len() > 2 {
        args[2..].iter().filter_map(|s| s.parse().ok()).collect()
    } else {
        vec![1e-3, 1e-4, 1e-5, 1e-6]
    };

    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    writeln!(
        out,
        "epsilon,method,trial,alpha,beta,gamma,t_count,q_count,cost,non_clifford_depth,distance,duration_ms,success,gates"
    ).unwrap();
    out.flush().ok();

    let stderr = std::io::stderr();
    let mut err = stderr.lock();

    for &eps in &eps_list {
        writeln!(err, "\n=== epsilon = {eps:.0e} ===").ok();
        let mut rng = Xs(seed);
        for trial in 0..n {
            let alpha = 2.0 * PI * rng.unit();
            let beta = 2.0 * PI * rng.unit();
            let gamma = 2.0 * PI * rng.unit();
            let target = matmul(matmul(rz(alpha), ry(beta)), rz(gamma));

            for method in &["clifford_t", "clifford_sqrt_t"] {
                let (gates_opt, dist, dur_ms) = if *method == "clifford_t" {
                    run_t(target, eps)
                } else {
                    run_q_stage4(target, eps)
                };
                let gates_upper = gates_opt
                    .as_deref()
                    .map(|g| g.to_uppercase())
                    .unwrap_or_default();
                let (t_count, q_count) = count_gates(&gates_upper);
                let (cost, ncd) = if gates_upper.is_empty() {
                    (f64::NAN, f64::NAN)
                } else {
                    (cost_of(t_count, q_count), non_clifford_depth_of(t_count, q_count))
                };
                let success = dist <= eps;
                writeln!(
                    out,
                    "{:.0e},{},{},{},{},{},{},{},{:.1},{:.1},{:.6e},{:.3},{},{}",
                    eps,
                    method,
                    trial,
                    alpha,
                    beta,
                    gamma,
                    t_count,
                    q_count,
                    cost,
                    ncd,
                    dist,
                    dur_ms,
                    if success { "True" } else { "False" },
                    gates_upper,
                ).unwrap();
                out.flush().ok();
                let tag = if success { "OK  " } else { "FAIL" };
                writeln!(
                    err,
                    "  trial {:>3}  {:<16}  T={t_count:>3} Q={q_count:>3}  cost={cost:>6.1}  ncd={ncd:>6.1}  d={dist:.3e}  {dur_ms:>9.1} ms  {tag}",
                    trial + 1,
                    method,
                ).ok();
            }
        }
    }
}
