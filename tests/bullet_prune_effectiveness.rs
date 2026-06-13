//! Effectiveness measurement for bullet-aware SE pruning.
//!
//! For each (seed, k, ε), measure leaves-enumerated and wall-time with
//! pruning OFF vs ON and print a small table. The soundness gate
//! (`bullet_prune_soundness.rs`) is the prerequisite — only here do we
//! claim a speedup.

use cyclosynth::synthesis::lattice_upsilon::{phase1_with_stop_stats, LatticeScratch};
use num_complex::Complex64;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::f64::consts::PI;
use std::sync::atomic::AtomicBool;

fn haar_target(seed: u64) -> [[Complex64; 2]; 2] {
    let mut rng = StdRng::seed_from_u64(seed);
    let theta = rng.random::<f64>() * (2.0 * PI);
    let phi = rng.random::<f64>() * (2.0 * PI);
    let lambda = rng.random::<f64>() * (2.0 * PI);
    let ct = (theta / 2.0).cos();
    let st = (theta / 2.0).sin();
    let global = Complex64::from_polar(1.0, -(phi + lambda) / 2.0);
    [
        [
            global * Complex64::new(ct, 0.0),
            global * (-Complex64::from_polar(st, lambda)),
        ],
        [
            global * Complex64::from_polar(st, phi),
            global * Complex64::from_polar(ct, phi + lambda),
        ],
    ]
}

fn run(
    target: &[[Complex64; 2]; 2],
    k: u32,
    eps: f64,
    prune_on: bool,
    budget: u64,
) -> (usize, usize, usize, bool, f64) {
    unsafe {
        std::env::set_var(
            "CYCLOSYNTH_BULLET_PRUNE_N12",
            if prune_on { "1" } else { "0" },
        );
    }
    let v = [
        target[0][0].re,
        target[0][0].im,
        target[1][0].re,
        target[1][0].im,
    ];
    let mut scratch = LatticeScratch::new(eps);
    let budget_hit = AtomicBool::new(false);
    let t0 = std::time::Instant::now();
    let (_sols, stats) = phase1_with_stop_stats(
        &mut scratch,
        v,
        k,
        eps,
        budget,
        &budget_hit,
        |_| false,
    );
    let wall_s = t0.elapsed().as_secs_f64();
    (
        stats.se_leaves,
        stats.pass_norm,
        stats.pass_bullets,
        stats.budget_hit,
        wall_s,
    )
}

#[test]
#[ignore = "diagnostic — prints the effectiveness table; run with --ignored --nocapture"]
fn bullet_prune_effectiveness_table() {
    eprintln!(
        "ε      | seed | k  | mode | leaves     | norm-pass  | bullet-pass | budget_hit | wall_s"
    );
    eprintln!("{}", "-".repeat(90));
    let budget: u64 = 50_000_000;
    for &eps in &[1e-3_f64, 1e-4_f64] {
        for seed in [1_u64, 3_u64] {
            let target = haar_target(seed);
            for k in [10_u32, 12, 14] {
                for &on in &[false, true] {
                    let (leaves, pn, pb, bh, ws) = run(&target, k, eps, on, budget);
                    eprintln!(
                        "{eps:>6.0e} | {seed:>4} | {k:>2} | {:>3} | {leaves:>10} | {pn:>10} | {pb:>11} | {bh:>10} | {ws:>6.2}",
                        if on { "ON" } else { "OFF" }
                    );
                }
            }
        }
    }
}
