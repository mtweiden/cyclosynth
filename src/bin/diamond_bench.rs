//! Microbenchmark for `diamond_distance_float` (f64 algebraic) vs
//! `diamond_distance_float_mpfr` (MPFR-128). Picks representative inputs:
//! random U(2) targets paired with their cyclosynth-synthesized approximants
//! across a range of ε, then times ~10⁶ calls per variant.

use cyclosynth::synthesis::distance::{diamond_distance_float, diamond_distance_float_mpfr};
use cyclosynth::synthesis::Synthesizer;
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

fn u3(a: f64, b: f64, c: f64) -> Mat2 {
    mat_mul(mat_mul(rz(a), ry(b)), rz(c))
}

/// Apply f64 gate string to identity → final unitary.
fn rebuild_f64(gates: &str) -> Mat2 {
    let h_inv2 = 1.0 / 2.0_f64.sqrt();
    let h: Mat2 = [
        [C64::new(h_inv2, 0.0), C64::new(h_inv2, 0.0)],
        [C64::new(h_inv2, 0.0), C64::new(-h_inv2, 0.0)],
    ];
    let s: Mat2 = [
        [C64::new(1.0, 0.0), C64::new(0.0, 0.0)],
        [C64::new(0.0, 0.0), C64::new(0.0, 1.0)],
    ];
    let t: Mat2 = [
        [C64::new(1.0, 0.0), C64::new(0.0, 0.0)],
        [C64::new(0.0, 0.0), C64::from_polar(1.0, PI / 4.0)],
    ];
    let x: Mat2 = [
        [C64::new(0.0, 0.0), C64::new(1.0, 0.0)],
        [C64::new(1.0, 0.0), C64::new(0.0, 0.0)],
    ];
    let y: Mat2 = [
        [C64::new(0.0, 0.0), C64::new(0.0, -1.0)],
        [C64::new(0.0, 1.0), C64::new(0.0,  0.0)],
    ];
    let z: Mat2 = [
        [C64::new(1.0, 0.0), C64::new( 0.0, 0.0)],
        [C64::new(0.0, 0.0), C64::new(-1.0, 0.0)],
    ];
    let mut u: Mat2 = [
        [C64::new(1.0, 0.0), C64::new(0.0, 0.0)],
        [C64::new(0.0, 0.0), C64::new(1.0, 0.0)],
    ];
    for c in gates.chars() {
        let g = match c {
            'H' => h, 'S' => s, 'T' => t, 'X' => x, 'Y' => y, 'Z' => z,
            'I' => continue,
            _ => panic!("unknown gate {c}"),
        };
        u = mat_mul(u, g);
    }
    u
}

fn main() {
    println!("Generating benchmark inputs (target, rebuilt) pairs at varying ε...");
    let epsilons = [1e-3_f64, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8];
    let mut pairs: Vec<(Mat2, Mat2, f64)> = Vec::new(); // (target, rebuild, eps)

    // Deterministic-ish angle sequence so runs are reproducible.
    let trial_angles: Vec<(f64, f64, f64)> = (0..6)
        .map(|i| {
            let t = i as f64;
            (
                2.0 * PI * (0.111 + 0.083 * t).fract(),
                2.0 * PI * (0.317 + 0.149 * t).fract(),
                2.0 * PI * (0.512 + 0.073 * t).fract(),
            )
        })
        .collect();

    for &eps in &epsilons {
        let synth = Synthesizer::new(eps, false);
        for &(a, b, c) in &trial_angles {
            let target = u3(a, b, c);
            let result = match synth.synthesize(target) {
                Some(r) => r,
                None => continue,
            };
            let gates = match &result.gates {
                Some(g) => g.as_str(),
                None => continue,
            };
            let rebuilt = rebuild_f64(gates);
            pairs.push((target, rebuilt, eps));
        }
        println!("  ε={:.0e}: {} pairs collected", eps, pairs.len());
    }

    println!("\nCollected {} (target, rebuilt) pairs total.", pairs.len());
    println!("Each call below operates on a randomly-cycled subset.\n");

    // Reference correctness check at high precision (MPFR-256). Compute
    // each pair's distance under three methods and report agreement.
    println!("Correctness check (sampled from each ε bucket):");
    println!("  ε         d_f64           d_mpfr_128       d_mpfr_256       agree?");
    for &eps in &epsilons {
        let sample = pairs.iter().find(|p| (p.2 - eps).abs() < eps * 0.01);
        if let Some((target, rebuilt, _)) = sample {
            let d_f64 = diamond_distance_float(rebuilt, target);
            let d_128 = diamond_distance_float_mpfr(rebuilt, target, 128);
            let d_256 = diamond_distance_float_mpfr(rebuilt, target, 256);
            let max_d = d_f64.max(d_128).max(d_256);
            let min_d = d_f64.min(d_128).min(d_256);
            let agree = if max_d > 0.0 {
                (max_d - min_d) / max_d < 1e-3
            } else {
                true
            };
            println!(
                "  {:<10}{:<16.6e}{:<17.6e}{:<17.6e}{}",
                format!("{:.0e}", eps),
                d_f64,
                d_128,
                d_256,
                if agree { "yes" } else { "NO ★" }
            );
        }
    }

    // Timing: cycle through pairs N times.
    let n_iters = 1_000_000;
    println!("\nTiming ({} calls each):", n_iters);

    let mut sink = 0.0_f64;
    let t0 = Instant::now();
    for i in 0..n_iters {
        let (a, b, _) = &pairs[i % pairs.len()];
        sink += diamond_distance_float(a, b);
    }
    let dt_f64 = t0.elapsed();
    println!("  diamond_distance_float           : {:>9.2?}  ({:.1} ns/call)",
             dt_f64, dt_f64.as_nanos() as f64 / n_iters as f64);

    let n_iters_mpfr = 100_000; // MPFR is much slower per call
    let t0 = Instant::now();
    for i in 0..n_iters_mpfr {
        let (a, b, _) = &pairs[i % pairs.len()];
        sink += diamond_distance_float_mpfr(a, b, 128);
    }
    let dt_mpfr_128 = t0.elapsed();
    println!("  diamond_distance_float_mpfr(128) : {:>9.2?}  ({:.1} ns/call)",
             dt_mpfr_128, dt_mpfr_128.as_nanos() as f64 / n_iters_mpfr as f64);

    let t0 = Instant::now();
    for i in 0..n_iters_mpfr {
        let (a, b, _) = &pairs[i % pairs.len()];
        sink += diamond_distance_float_mpfr(a, b, 256);
    }
    let dt_mpfr_256 = t0.elapsed();
    println!("  diamond_distance_float_mpfr(256) : {:>9.2?}  ({:.1} ns/call)",
             dt_mpfr_256, dt_mpfr_256.as_nanos() as f64 / n_iters_mpfr as f64);

    println!("\nRatio (MPFR-128 / f64): {:.1}×",
             (dt_mpfr_128.as_nanos() as f64 / n_iters_mpfr as f64) /
             (dt_f64.as_nanos() as f64 / n_iters as f64));

    println!("\n(sink={:.3e} — printed to prevent dead-code elimination)", sink);
}
