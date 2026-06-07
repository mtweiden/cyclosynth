//! Measurement harness for PROMPT_lattice_upsilon_bkz.md.

use cyclosynth::synthesis::lattice_upsilon::integer::phase1_with_stop_stats;
use cyclosynth::synthesis::lattice_upsilon::scratch::IntScratch16;
use cyclosynth::synthesis::lattice_upsilon::synthesize::best_phase;
use num_complex::Complex64;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::sync::atomic::AtomicBool;
use std::time::Instant;

type Mat2 = [[Complex64; 2]; 2];

fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

fn haar_target(seed: u64) -> Mat2 {
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

fn dist_oracle(a: &Mat2, b: &Mat2) -> f64 {
    let mut tr = c(0.0, 0.0);
    for i in 0..2 {
        for j in 0..2 {
            tr += a[i][j] * b[i][j].conj();
        }
    }
    let tr_abs = tr.norm();
    let phi = if tr_abs > 1e-300 {
        tr / tr_abs
    } else {
        c(1.0, 0.0)
    };
    let mut fro_sq = 0.0_f64;
    for i in 0..2 {
        for j in 0..2 {
            let diff = a[i][j] - phi * b[i][j];
            fro_sq += diff.norm_sqr();
        }
    }
    let d_sq = fro_sq * (8.0 - fro_sq) / 16.0;
    d_sq.max(0.0).sqrt()
}

fn v_of(target: &Mat2) -> [f64; 4] {
    [
        target[0][0].re,
        target[0][0].im,
        target[1][0].re,
        target[1][0].im,
    ]
}

#[test]
#[ignore = "diagnostic measurement; run with --ignored --nocapture"]
fn bkz_vs_lll_part2_table() {
    let eps = 1e-4_f64;
    let max_leaves: u64 = std::env::var("CYCLOSYNTH_N12_MEASURE_MAX_LEAVES")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(10_000_000);
    let seeds: Vec<u64> = (0..6).collect();
    let ks = [12u32, 14, 16, 18];
    let reductions = [("LLL", 0u32), ("BKZ", 4u32)];

    eprintln!("n=12 BKZ measurement eps={eps:.0e}, max_leaves={max_leaves}");
    eprintln!(
        "seed | k  | reduction | se leaves | shell leaves | pass-bullets | pass-align | solution | oracle d | wall ms"
    );

    for seed in seeds {
        let target = haar_target(seed);
        let v = v_of(&target);
        for k in ks {
            for (label, block) in reductions {
                unsafe {
                    std::env::set_var("CYCLOSYNTH_BKZ_BLOCK_N12", block.to_string());
                }
                let mut scratch = IntScratch16::new(eps);
                let budget_hit = AtomicBool::new(false);
                let t0 = Instant::now();
                let (sols, stats) = phase1_with_stop_stats(
                    &mut scratch,
                    v,
                    k,
                    eps,
                    max_leaves,
                    &budget_hit,
                    |_| false,
                );
                let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
                let mut best_d = f64::NAN;
                let mut solution = false;
                for sol in &sols {
                    let (u, _phase, _claimed_d) = best_phase(sol, k, &target);
                    let d = dist_oracle(&u.to_float(), &target);
                    if best_d.is_nan() || d < best_d {
                        best_d = d;
                    }
                    if d <= eps {
                        solution = true;
                    }
                }
                eprintln!(
                    "{seed:>4} | {k:>2} | {label:>9} | {:>9} | {:>12} | {:>12} | {:>10} | {:>8} | {:>8.3e} | {:>7.1}",
                    stats.se_leaves,
                    stats.pass_norm,
                    stats.pass_bullets,
                    stats.pass_align,
                    if solution { "yes" } else { "no" },
                    best_d,
                    wall_ms
                );
            }
        }
    }
    unsafe {
        std::env::remove_var("CYCLOSYNTH_BKZ_BLOCK_N12");
    }
}
