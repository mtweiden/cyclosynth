//! One-shot per-stage timing report for n=4, n=6, and n=8 synthesis phases.
//!
//! Uses a fixed deterministic target Rz(0.3) and eps=1e-3. Stage timings come
//! from the existing diagnostic counters and are CPU-summed across Rayon tasks.

use cyclosynth::synthesis::clifford_pi6::{compute_y, unitary_to_uv_n6};
use cyclosynth::synthesis::clifford_sqrt_t::unitary_to_uv_zeta;
use cyclosynth::synthesis::diag;
use cyclosynth::synthesis::lattice;
use cyclosynth::synthesis::lattice_omicron;
use cyclosynth::synthesis::lattice_zeta::{phase1 as phase1_zeta, IntScratch16};
use cyclosynth::synthesis::search::compute_align_vec;
use cyclosynth::synthesis::search_zeta::uv_to_xy_zeta;
use num_complex::Complex64;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

type Mat2 = [[Complex64; 2]; 2];

fn rz(theta: f64) -> Mat2 {
    [
        [
            Complex64::from_polar(1.0, -theta / 2.0),
            Complex64::new(0.0, 0.0),
        ],
        [
            Complex64::new(0.0, 0.0),
            Complex64::from_polar(1.0, theta / 2.0),
        ],
    ]
}

fn stage_total(s: &diag::Snapshot) -> f64 {
    s.t_build_ms + s.t_lll_ms + s.t_cholesky_ms + s.t_lu_ms + s.t_se_ms
}

fn uv_to_xy_n4(v: [f64; 4], k: u32) -> [f64; 8] {
    let scale = 2.0_f64.powf(k as f64 / 2.0 - 1.0);
    compute_align_vec(v).map(|x| x * scale)
}

fn print_row(
    label: &str,
    k: u32,
    wall_ms: f64,
    n_solutions: usize,
    budget_hit: bool,
    s: &diag::Snapshot,
) {
    let stage_ms = stage_total(s);
    let overhead_ms = (wall_ms - stage_ms).max(0.0);
    println!(
        "{label:<18} k={k:<2} wall={wall_ms:>9.3} ms  sols={n_solutions:<3} budget_hit={budget_hit:<5} stage_sum={stage_ms:>9.3} ms  overhead={overhead_ms:>9.3} ms",
    );
    println!(
        "  build_q={:>9.3}  lll={:>9.3}  cholesky={:>9.3}  lu_solve={:>9.3}  se_walk={:>9.3}",
        s.t_build_ms, s.t_lll_ms, s.t_cholesky_ms, s.t_lu_ms, s.t_se_ms,
    );
    println!(
        "  phase1_calls={} se_callbacks={} lll_iters_total={} lll_iters_max={} lll_at_cap={}",
        s.phase1_calls, s.se_callbacks, s.lll_iters_total, s.lll_iters_max, s.lll_at_cap,
    );
}

fn main() {
    std::env::set_var("CYCLOSYNTH_TRACE", "1");

    let eps = 1e-3_f64;
    let target = rz(0.3);

    println!("target=Rz(0.3) eps={eps:.0e}");
    println!("one representative lattice phase per n; stage timings are CPU-summed counters");
    println!();

    let v_n4 = [(0.3_f64 / 2.0).cos(), -(0.3_f64 / 2.0).sin(), 0.0, 0.0];
    let k_n4 = 13;
    let y_n4 = uv_to_xy_n4(v_n4, k_n4);
    let budget_hit = AtomicBool::new(false);
    let mut scratch = lattice::LatticeScratch::new(eps);
    diag::reset_all();
    let t0 = Instant::now();
    let sols = lattice::phase1(&mut scratch, &y_n4, k_n4, eps, 100_000, &budget_hit);
    let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let s = diag::snapshot();
    print_row(
        "n=4 Z[omega]",
        k_n4,
        wall_ms,
        sols.len(),
        budget_hit.load(std::sync::atomic::Ordering::Relaxed),
        &s,
    );
    println!();

    let v_n6 = unitary_to_uv_n6(&target);
    let k_n6 = 7;
    let y_n6 = compute_y(v_n6[0], v_n6[1], v_n6[2], v_n6[3]);
    let budget_hit = AtomicBool::new(false);
    let mut scratch = lattice_omicron::LatticeScratch::new(eps);
    diag::reset_all();
    let t0 = Instant::now();
    let sols = lattice_omicron::phase1(&mut scratch, &y_n6, k_n6, eps, 100_000, &budget_hit);
    let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let s = diag::snapshot();
    print_row(
        "n=6 Z[xi]",
        k_n6,
        wall_ms,
        sols.len(),
        budget_hit.load(std::sync::atomic::Ordering::Relaxed),
        &s,
    );
    println!();

    let v_n8 = unitary_to_uv_zeta(&target);
    let k_n8 = 7;
    let y_n8 = uv_to_xy_zeta(v_n8, k_n8);
    let budget_hit = AtomicBool::new(false);
    let mut scratch = IntScratch16::new(eps);
    diag::reset_all();
    let t0 = Instant::now();
    let sols = phase1_zeta(&mut scratch, &y_n8, k_n8, eps, 100_000, &budget_hit);
    let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let s = diag::snapshot();
    print_row(
        "n=8 Z[zeta16]",
        k_n8,
        wall_ms,
        sols.len(),
        budget_hit.load(std::sync::atomic::Ordering::Relaxed),
        &s,
    );
}
