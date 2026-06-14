//! Part 4 payoff measurement for PROMPT_lattice_upsilon_mitm_8d_se.md.
//!
//! For seeds 0..5 at ε ∈ {1e-5, 1e-6}, sweep `k ∈ k_min..k_max` and report
//! whether MITM (8D LLL+SE half-enumerator backend) finds an oracle-confirmed
//! distance ≤ ε solution, with k, pool sizes, table size, wall time. The
//! n=8 envelope from Step 0 of PROMPT_n8_vs_n12_funnel was 0.5–2.3 s at
//! 1e-5/1e-6 — that's the bar for "the anisotropy penalty is paid off."
//!
//! Oracle: reconstruct via `best_phase` → `solution_to_unitary`, then
//! `diamond_distance_float`. Independent of MITM's internal accept logic.

use cyclosynth::synthesis::distance::diamond_distance_float;
use cyclosynth::synthesis::lattice_upsilon::mitm::{HalfSide, PerHalfRegion};
use cyclosynth::synthesis::lattice_upsilon::mitm_half_se::{
    lll_se_enumerate_half, lll_se_mitm_norm_bullet_set,
};
use cyclosynth::synthesis::lattice_upsilon::synthesize::best_phase;
use num_complex::Complex64;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::f64::consts::PI;
use std::time::Instant;

type Mat2 = [[Complex64; 2]; 2];

fn haar_target(seed: u64) -> Mat2 {
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

#[derive(Debug, Clone)]
struct PayoffRow {
    eps: f64,
    seed: u64,
    k: u32,
    pool1: usize,
    pool2: usize,
    table_pairs: usize,
    found: bool,
    actual_d: f64,
    wall_s: f64,
}

/// Run MITM at (eps, k, seed). Returns one row with pool sizes, # joint
/// hits before joint-alignment, # leaves with oracle d ≤ eps, and wall.
fn run_mitm_at_k(target: &Mat2, eps: f64, k: u32) -> PayoffRow {
    let r1 = PerHalfRegion::new(HalfSide::U1, target[0][0], k, eps);
    let r2 = PerHalfRegion::new(HalfSide::U2, target[1][0], k, eps);
    let t0 = Instant::now();
    let pool1 = lll_se_enumerate_half(&r1);
    let pool2 = lll_se_enumerate_half(&r2);
    let joined = lll_se_mitm_norm_bullet_set(target, k, eps);
    let mut actual_d = f64::INFINITY;
    let mut found = false;
    for x in &joined {
        let (u_ring, _phase, d) = best_phase(x, k, target);
        let actual = diamond_distance_float(&u_ring.to_float(), target);
        if actual < actual_d {
            actual_d = actual;
        }
        if actual <= eps {
            found = true;
        }
    }
    let wall = t0.elapsed().as_secs_f64();
    PayoffRow {
        eps,
        seed: 0, // filled by caller
        k,
        pool1: pool1.len(),
        pool2: pool2.len(),
        table_pairs: joined.len(),
        found,
        actual_d,
        wall_s: wall,
    }
}

/// Lever-2 frontier-k: the smallest exhaustible k whose per-half region
/// is expected to be non-empty for `ε`. The pool size estimate is
/// `~2^{4k}·ε²`; we pick `k` so this is ~ε⁻¹·8 (modest but ≥ 1). For
/// ε=1e-5 → k ≈ 12; for ε=1e-6 → k ≈ 14. We then step up by one if k_start
/// has an empty region (only the unlucky-target case).
fn frontier_k(eps: f64) -> u32 {
    // Solve 2^(4k)·ε² ≥ 8 → 4k ≥ log2(8/ε²) → k ≥ ⌈(log2(8) − 2·log2(ε))/4⌉.
    let target_pool_log2 = 3.0_f64; // log2(8)
    let k_est = (target_pool_log2 - 2.0 * eps.log2()) / 4.0;
    k_est.ceil() as u32
}

fn try_seed(eps: f64, seed: u64, _k_min: u32, k_max: u32) -> PayoffRow {
    let target = haar_target(seed);
    let mut total_wall = 0.0_f64;
    let mut last_row: Option<PayoffRow> = None;
    let k_start = frontier_k(eps);
    // From the frontier k upward; in practice the first hit is at
    // k_start or k_start+1 with the BKZ-reduced 8D walker.
    for k in k_start..=k_max {
        let mut row = run_mitm_at_k(&target, eps, k);
        row.seed = seed;
        total_wall += row.wall_s;
        if row.found {
            row.wall_s = total_wall;
            return row;
        }
        last_row = Some(row);
    }
    let mut last = last_row.unwrap();
    last.seed = seed;
    last.wall_s = total_wall;
    last
}

#[test]
#[ignore = "Part 4 payoff: MITM (8D LLL+SE backend) at ε=1e-5/1e-6 on Haar seeds 0..5"]
fn part4_payoff_table() {
    eprintln!(
        "\n[Part 4] MITM (8D LLL+SE) payoff vs n=8 envelope (0.5-2.3 s @ 1e-5/1e-6)\n\
         oracle distance = diamond(reconstruct(best_phase, x, k), target)\n"
    );
    eprintln!(
        "    ε    | seed |   k | pool1 | pool2 | table | found |   d (oracle) |   wall_s"
    );
    eprintln!("{}", "-".repeat(85));

    // Probe a tight k window where the prompt's count sanity suggests
    // tractability: per-half pool ~ R⁸·ε² ≈ 2^(4k)·ε² should be at
    // least 1. At ε=1e-5 that's k ≥ ~5 (R⁸ε² ≈ 10). At ε=1e-6, k ≥ ~6.
    // With the 8D LLL+SE backend the per-region cost no longer scales as
    // box⁸ — start `k` lower since SE prunes most of the cube.
    let mut rows: Vec<PayoffRow> = Vec::new();
    for &eps in &[1e-5_f64, 1e-6_f64] {
        for seed in 0..6_u64 {
            let row = try_seed(eps, seed, 5, 18);
            eprintln!(
                "{:>8.0e} | {:>4} | {:>3} | {:>5} | {:>5} | {:>5} | {:>5} | {:>12.3e} | {:>8.2}",
                row.eps,
                row.seed,
                row.k,
                row.pool1,
                row.pool2,
                row.table_pairs,
                if row.found { "yes" } else { "no" },
                row.actual_d,
                row.wall_s,
            );
            rows.push(row);
        }
    }

    eprintln!("\n[Part 4] Summary:");
    for &eps in &[1e-5_f64, 1e-6_f64] {
        let bucket: Vec<&PayoffRow> = rows.iter().filter(|r| r.eps == eps).collect();
        let found = bucket.iter().filter(|r| r.found).count();
        let total = bucket.len();
        let avg_wall = bucket.iter().map(|r| r.wall_s).sum::<f64>() / total as f64;
        eprintln!(
            "  ε={eps:.0e}: {found}/{total} seeds met oracle ≤ ε, avg wall {avg_wall:.1} s"
        );
    }
}
