//! Quick-and-dirty A/B bench: BKZ-β post-pass vs no-BKZ on Q synthesis.
//!
//! Generates a small fixed set of random SU(2) targets and synthesizes each
//! at one ε with `SynthesizerQ::with_bkz(β)` and again without. Reports
//! per-target time + lde for both. Run with:
//!
//!   cargo run --release --bin bench_bkz_zeta -- --eps 1e-7 --beta 4 --n 4

use cyclosynth::synthesis::clifford_sqrt_t::SynthesizerQ;
use num_complex::Complex;
use std::f64::consts::PI;
use std::time::Instant;

type C64 = Complex<f64>;
type Mat2 = [[C64; 2]; 2];

fn mat_mul(a: Mat2, b: Mat2) -> Mat2 {
    [[
        a[0][0] * b[0][0] + a[0][1] * b[1][0],
        a[0][0] * b[0][1] + a[0][1] * b[1][1],
    ], [
        a[1][0] * b[0][0] + a[1][1] * b[1][0],
        a[1][0] * b[0][1] + a[1][1] * b[1][1],
    ]]
}

fn rz(t: f64) -> Mat2 {
    [[C64::from_polar(1.0, -t / 2.0), C64::new(0.0, 0.0)],
     [C64::new(0.0, 0.0), C64::from_polar(1.0, t / 2.0)]]
}
fn ry(t: f64) -> Mat2 {
    let c = (t / 2.0).cos();
    let s = (t / 2.0).sin();
    [[C64::new(c, 0.0), C64::new(-s, 0.0)],
     [C64::new(s, 0.0), C64::new(c, 0.0)]]
}
fn u3(a: f64, b: f64, c: f64) -> Mat2 { mat_mul(mat_mul(rz(a), ry(b)), rz(c)) }

fn xorshift64(s: &mut u64) -> u64 {
    *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; *s
}
fn rand_angle(s: &mut u64) -> f64 {
    let b = xorshift64(s) >> 11;
    (b as f64) / ((1u64 << 53) as f64) * 2.0 * PI
}

fn main() {
    let mut eps = 1e-7_f64;
    let mut beta: u32 = 4;
    let mut n: usize = 4;
    let mut max_lde: u32 = 35;
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--eps" => { eps = args[i+1].parse().unwrap(); i += 2; }
            "--beta" => { beta = args[i+1].parse().unwrap(); i += 2; }
            "--n" => { n = args[i+1].parse().unwrap(); i += 2; }
            "--max-lde" => { max_lde = args[i+1].parse().unwrap(); i += 2; }
            _ => { eprintln!("unknown arg: {}", args[i]); std::process::exit(2); }
        }
    }
    let mut state: u64 = 0xC0FFEEBAADD0E;
    let targets: Vec<Mat2> = (0..n).map(|_| {
        u3(rand_angle(&mut state), rand_angle(&mut state), rand_angle(&mut state))
    }).collect();

    println!("# bench_bkz_zeta: ε={:e}, β={}, n={}, max_lde={}", eps, beta, n, max_lde);
    println!("# legend: pure = no BKZ, bkz{} = BKZ-{} post-pass", beta, beta);

    let mut total_pure = 0.0_f64;
    let mut total_bkz = 0.0_f64;
    for (idx, target) in targets.iter().enumerate() {
        let t = Instant::now();
        let synth_pure = SynthesizerQ::new(eps).with_optimize_cost(false).with_max_lde(max_lde);
        let r_pure = synth_pure.synthesize(*target);
        let dt_pure = t.elapsed().as_secs_f64();
        total_pure += dt_pure;

        let t = Instant::now();
        let synth_bkz = SynthesizerQ::new(eps).with_optimize_cost(false).with_max_lde(max_lde).with_bkz(beta);
        let r_bkz = synth_bkz.synthesize(*target);
        let dt_bkz = t.elapsed().as_secs_f64();
        total_bkz += dt_bkz;

        let lde_pure = r_pure.as_ref().map(|r| r.lde as i32).unwrap_or(-1);
        let lde_bkz = r_bkz.as_ref().map(|r| r.lde as i32).unwrap_or(-1);
        println!(
            "target_{:02}: pure {:7.3} s (lde {:>2}) | bkz{} {:7.3} s (lde {:>2}) | ratio {:.2}×",
            idx, dt_pure, lde_pure, beta, dt_bkz, lde_bkz, dt_pure / dt_bkz.max(1e-9)
        );
    }
    println!(
        "# total pure {:.3} s | total bkz{} {:.3} s | speedup {:.2}×",
        total_pure, beta, total_bkz, total_pure / total_bkz.max(1e-9)
    );
}
