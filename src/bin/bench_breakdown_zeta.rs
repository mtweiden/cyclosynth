//! Print the per-phase time breakdown (build/LLL/cholesky/LU/SE) for
//! Clifford+√T optimal-mode synthesis. Decides whether per-prefix setup
//! amortization (warm-LLL) is worth building: if t_lll + t_build ≪ t_se,
//! it is not (prediction: t_se ≥ 90% of the critical path).
//!
//! Args: [eps] [n] [seed]   (defaults 1e-5, 4, 0xC0FFEEBAADD0E)
//! Note: CYCLOSYNTH_TRACE inflates SE walls (per-leaf counters), which
//! BIASES the SE share upward — but if LLL+build still lose by 20×
//! under trace, they lose untraced too (the falsification direction is
//! robust).

use cyclosynth::synthesis::clifford_sqrt_t::SynthesizerQ;
use cyclosynth::synthesis::diag;
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

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let eps: f64 = args.first().and_then(|s| s.parse().ok()).unwrap_or(1e-5);
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(4);
    let mut state: u64 = args.get(2)
        .and_then(|s| u64::from_str_radix(s.trim_start_matches("0x"), 16).ok())
        .unwrap_or(0xC0FFEEBAADD0E);

    unsafe { std::env::set_var("CYCLOSYNTH_TRACE", "1") };
    assert!(
        diag::trace_enabled(),
        "build with `--features trace` for the per-phase breakdown"
    );
    let targets: Vec<Mat2> = (0..n).map(|_| {
        u3(rand_angle(&mut state), rand_angle(&mut state), rand_angle(&mut state))
    }).collect();

    diag::reset_all();
    let t = Instant::now();
    // Defaults: optimize_cost on, m_sweep per default, window 2 — the
    // production optimal path whose critical path we are attributing.
    let synth = SynthesizerQ::new(eps);
    for target in &targets {
        let _ = synth.synthesize(*target);
    }
    let wall_ms = t.elapsed().as_secs_f64() * 1000.0;
    let s = diag::snapshot();
    let stage = s.t_build_ms + s.t_lll_ms + s.t_cholesky_ms + s.t_lu_ms + s.t_se_ms;
    let pct = |x: f64| if stage > 0.0 { 100.0 * x / stage } else { 0.0 };
    println!(
        "ε={:e} n={n}  wall {:>9.1} ms (trace-on)  lattice_search_calls={}",
        eps, wall_ms, s.lattice_search_calls,
    );
    println!(
        "  build {:>8.1} ms ({:>4.1}%)  lll {:>8.1} ms ({:>4.1}%)  chol+lu {:>7.1} ms ({:>4.1}%)  se {:>9.1} ms ({:>4.1}%)",
        s.t_build_ms, pct(s.t_build_ms),
        s.t_lll_ms, pct(s.t_lll_ms),
        s.t_cholesky_ms + s.t_lu_ms, pct(s.t_cholesky_ms + s.t_lu_ms),
        s.t_se_ms, pct(s.t_se_ms),
    );
    if s.lattice_search_calls > 0 {
        println!(
            "  per-prefix: lll {:.3} ms  build {:.3} ms  ({} find_aligned_lattice_points calls; wall {:.1} ms)",
            s.t_lll_ms / s.lattice_search_calls as f64,
            s.t_build_ms / s.lattice_search_calls as f64,
            s.lattice_search_calls, wall_ms,
        );
    }
}
