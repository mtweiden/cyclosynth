//! Test ε=1e-8 at lde=20, 21 — below the lattice_start estimate.
//! If any solutions are found, the lattice_start heuristic is too aggressive.

use cyclosynth::matrix::U2Q;
use cyclosynth::synthesis::clifford_sqrt_t::{
    Mat2Mpfr, det_phase_of, solution_to_u2q_d,
};
use cyclosynth::synthesis::lenstra_zeta::{phase1_with_stop_mpfr, IntScratch16};
use cyclosynth::synthesis::search_zeta::uv_to_xy_zeta_mpfr;
use cyclosynth::synthesis::distance::diamond_distance_u2q_float;
use num_complex::Complex;
use rug::Float as RFloat;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

type C64 = Complex<f64>;
type Mat2 = [[C64; 2]; 2];

fn rz_f64(t: f64) -> Mat2 {
    [[C64::from_polar(1.0, -t / 2.0), C64::new(0.0, 0.0)],
     [C64::new(0.0, 0.0), C64::from_polar(1.0, t / 2.0)]]
}

fn main() {
    let theta = 0.3_f64;
    let eps = 1e-8_f64;
    let prec: u32 = 192;
    let target = rz_f64(theta);
    let d = det_phase_of(&target);

    // MPFR v from MPFR theta.
    let theta_mpfr = RFloat::with_val(prec, theta);
    let half = RFloat::with_val(prec, &theta_mpfr / 2);
    let cos_half = half.clone().cos();
    let sin_half = half.clone().sin();
    let zero = RFloat::with_val(prec, 0.0);
    let v_mpfr: [RFloat; 4] = [
        cos_half, RFloat::with_val(prec, -&sin_half), zero.clone(), zero,
    ];

    for k in &[22u32, 23, 24, 25, 26] {
        let k = *k;
        eprintln!("─── lde={k} ───");
        let mut scratch = IntScratch16::new(eps);
        scratch.use_f64_gs = false; // Force MPFR LLL at this depth.
        scratch.bkz_block_size = 4;
        let y_mpfr = uv_to_xy_zeta_mpfr(&v_mpfr, k, prec);
        let budget_hit = AtomicBool::new(false);
        let target_local = target;
        let should_stop = |x: &[i64; 16]| -> bool {
            let cand = solution_to_u2q_d(x, k, d);
            diamond_distance_u2q_float(&cand, &target_local) < eps
        };
        let t = Instant::now();
        let sols = phase1_with_stop_mpfr(
            &mut scratch, &y_mpfr, &v_mpfr, k, eps,
            500_000_000, &budget_hit, should_stop,
        );
        let dt = t.elapsed().as_secs_f64();

        let mut best: Option<f64> = None;
        for sol in &sols {
            let cand: U2Q = solution_to_u2q_d(sol, k, d);
            let dist = diamond_distance_u2q_float(&cand, &target);
            if dist < eps {
                if best.map_or(true, |b| dist < b) {
                    best = Some(dist);
                }
            }
        }
        match best {
            Some(d) => eprintln!("  FOUND lde={k}: dist={:.3e}, {} sols, took {:.2} s",
                                 d, sols.len(), dt),
            None => eprintln!("  NONE  lde={k}: {} sols, took {:.2} s",
                              sols.len(), dt),
        }
    }
}
