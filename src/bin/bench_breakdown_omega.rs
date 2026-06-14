//! M0 attribution driver for the 8D Z[ω] prefix-sweep rework
//! (docs/plan_8d_prefix_rework.md, stage 0).
//!
//! Self-sets `CYCLOSYNTH_TRACE=1` (env-prefixed invocations are not always
//! possible in the sandboxed harness), then synthesizes the SAME targets as
//! `time_synthesis_omega` (identical xorshift64 generator, identical `seed | 1`
//! guard, default seed 0xC0FFEEBAADD0E) at each requested ε.
//!
//! Per-pass `[trace]` blocks land on stderr — `try_at_lde` calls
//! `diag::reset_all()` per pass, so end-of-run snapshots only see the LAST
//! pass; the per-pass stderr stream is the ground truth and is summed by
//! `bench_logs/prefix_rework_2026_06_11/parse_trace.py`. Stdout carries the
//! per-target wall table (one `[m0] target=... eps=...` line per run, used
//! as the run delimiter by the parser).
//!
//! Usage:
//!   bench_breakdown_omega [--threads N] [--n-targets N] [--seed HEX]
//!                     [--eps CSV] [--trials N]
//!                     [--coset 0|1] [--two-sweep 0|1] [--sweep1 even|odd]
//!                     [--tp-offset N]
//! Defaults: 8 threads, 3 targets, seed 0xC0FFEEBAADD0E,
//!           eps 1e-4,1e-5,1e-6,1e-7,1e-8, 1 trial.
//! `--coset/--two-sweep/--sweep1/--tp-offset` forward to the
//! CYCLOSYNTH_L_COSET / CYCLOSYNTH_TWO_SWEEP / CYCLOSYNTH_SWEEP1 /
//! CYCLOSYNTH_TP_OFFSET env gates via set_var (the sandboxed harness
//! cannot always pass env prefixes); they must therefore be trusted only
//! for THIS process (the gates are LazyLock-once).

use cyclosynth::synthesis::clifford_t::SynthesizerT;
use num_complex::Complex;
use std::f64::consts::PI;
use std::time::Instant;

type C64 = Complex<f64>;
type Mat2 = [[C64; 2]; 2];

fn mat_mul(a: Mat2, b: Mat2) -> Mat2 {
    [[a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]],
     [a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]]]
}
fn rz(t: f64) -> Mat2 {
    [[C64::from_polar(1.0, -t / 2.0), C64::new(0.0, 0.0)],
     [C64::new(0.0, 0.0), C64::from_polar(1.0, t / 2.0)]]
}
fn ry(t: f64) -> Mat2 {
    let c = (t / 2.0).cos(); let s = (t / 2.0).sin();
    [[C64::new(c, 0.0), C64::new(-s, 0.0)],
     [C64::new(s, 0.0), C64::new(c, 0.0)]]
}
fn u3(a: f64, b: f64, c: f64) -> Mat2 { mat_mul(mat_mul(rz(a), ry(b)), rz(c)) }
fn xorshift64(s: &mut u64) -> u64 { *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; *s }
fn rand_angle(s: &mut u64) -> f64 {
    let b = xorshift64(s) >> 11; (b as f64) / ((1u64 << 53) as f64) * 2.0 * PI
}
fn parse_seed(s: &str) -> u64 {
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u64::from_str_radix(hex, 16).expect("--seed hex parse")
    } else {
        s.parse().expect("--seed N (decimal or 0x-prefixed hex)")
    }
}

fn main() {
    std::env::set_var("CYCLOSYNTH_TRACE", "1");
    assert!(
        cyclosynth::synthesis::diag::trace_enabled(),
        "build with `--features trace` for the per-phase breakdown"
    );

    let args: Vec<String> = std::env::args().collect();
    let mut n_threads: usize = 8;
    let mut n_targets: usize = 3;
    let mut seed: u64 = 0xC0FFEE_BAADD0E_u64;
    let mut n_trials: usize = 1;
    let mut eps_list: Vec<f64> = vec![1e-4, 1e-5, 1e-6, 1e-7, 1e-8];
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--threads"   => { i += 1; n_threads = args[i].parse().expect("--threads N"); }
            "--n-targets" => { i += 1; n_targets = args[i].parse().expect("--n-targets N"); }
            "--seed"      => { i += 1; seed = parse_seed(&args[i]); }
            "--trials"    => { i += 1; n_trials = args[i].parse().expect("--trials N"); }
            "--eps"       => {
                i += 1;
                eps_list = args[i].split(',').map(|s| s.parse().expect("--eps CSV")).collect();
            }
            "--coset"     => { i += 1; std::env::set_var("CYCLOSYNTH_L_COSET", &args[i]); }
            "--two-sweep" => { i += 1; std::env::set_var("CYCLOSYNTH_TWO_SWEEP", &args[i]); }
            "--sweep1"    => { i += 1; std::env::set_var("CYCLOSYNTH_SWEEP1", &args[i]); }
            "--tp-offset" => { i += 1; std::env::set_var("CYCLOSYNTH_TP_OFFSET", &args[i]); }
            _ => {}
        }
        i += 1;
    }

    rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build_global()
        .expect("failed to build rayon thread pool");

    // Identical target stream to time_synthesis_omega (seed | 1 guard included).
    let mut rng_state = seed | 1;
    let targets: Vec<(String, Mat2)> = (0..n_targets)
        .map(|idx| {
            let a = rand_angle(&mut rng_state);
            let b = rand_angle(&mut rng_state);
            let c = rand_angle(&mut rng_state);
            (format!("target_{idx:02}"), u3(a, b, c))
        })
        .collect();

    println!(
        "[m0] threads={n_threads} targets={n_targets} seed={seed:#018x} trials={n_trials} \
         eps={eps_list:?} L_COSET={} TWO_SWEEP={} SWEEP1={} TP_OFFSET={}",
        std::env::var("CYCLOSYNTH_L_COSET").unwrap_or_else(|_| "default".into()),
        std::env::var("CYCLOSYNTH_TWO_SWEEP").unwrap_or_else(|_| "default".into()),
        std::env::var("CYCLOSYNTH_SWEEP1").unwrap_or_else(|_| "default".into()),
        std::env::var("CYCLOSYNTH_TP_OFFSET").unwrap_or_else(|_| "default".into()),
    );

    for &eps in &eps_list {
        let synth = SynthesizerT::new(eps);
        let mut eps_total_min = 0.0_f64;
        for (name, target) in &targets {
            let mut min_ms = f64::INFINITY;
            let mut last = None;
            for _ in 0..n_trials {
                // Run delimiter for the stderr parser: emitted BEFORE the
                // run's trace blocks.
                eprintln!("[m0] run target={name} eps={eps:e}");
                let t0 = Instant::now();
                last = synth.synthesize(*target);
                min_ms = min_ms.min(t0.elapsed().as_secs_f64() * 1000.0);
            }
            match &last {
                Some(r) => println!(
                    "[m0] target={name} eps={eps:e} lde={} dist={:.3e} min_ms={min_ms:.1}",
                    r.lde, r.distance
                ),
                None => println!("[m0] target={name} eps={eps:e} FAILED"),
            }
            eps_total_min += min_ms;
        }
        println!("[m0] eps={eps:e} total_min_ms={eps_total_min:.1}");
    }

    // M2 summary: cumulative branch-win counters (never reset).
    use cyclosynth::synthesis::diag;
    use std::sync::atomic::Ordering;
    let even = diag::N_BRANCH_WIN_EVEN.load(Ordering::Relaxed);
    let odd = diag::N_BRANCH_WIN_ODD.load(Ordering::Relaxed);
    let idx_sum = diag::N_WIN_PREFIX_IDX_SUM.load(Ordering::Relaxed);
    let len_sum = diag::N_WIN_PREFIX_LEN_SUM.load(Ordering::Relaxed);
    println!(
        "[m2] branch wins: even={even} odd={odd}  mean win sweep-pos={:.0}  mean win fraction={:.3}",
        idx_sum as f64 / (even + odd).max(1) as f64,
        idx_sum as f64 / (len_sum.max(1)) as f64,
    );
}
