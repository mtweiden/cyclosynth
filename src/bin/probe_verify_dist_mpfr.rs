//! Verify that synthesis results at moderate ε actually satisfy
//! `dist(U_synth, target) < ε` when computed with full MPFR (target
//! constructed from MPFR theta, U2Q evaluated via MPFR cos/sin basis).
//!
//! If the f64-reported dist matches MPFR dist, the f64 reporting is fine.
//! If MPFR shows dist >= ε, the f64 path is silently accepting bad results.

use cyclosynth::matrix::U2Q;
use cyclosynth::synthesis::clifford_sqrt_t::{SynthesizerQ, Mat2Mpfr};
use cyclosynth::synthesis::distance::{diamond_distance_u2q_float, diamond_distance_u2q_mpfr_target};
use num_complex::Complex;
use rug::Float as RFloat;

type C64 = Complex<f64>;
type Mat2 = [[C64; 2]; 2];

fn rz_f64(t: f64) -> Mat2 {
    [[C64::from_polar(1.0, -t / 2.0), C64::new(0.0, 0.0)],
     [C64::new(0.0, 0.0), C64::from_polar(1.0, t / 2.0)]]
}

fn rz_mpfr(theta_mpfr: &RFloat, prec: u32) -> Mat2Mpfr {
    let half = RFloat::with_val(prec, theta_mpfr / 2);
    let cos_half = half.clone().cos();
    let sin_half = half.clone().sin();
    let zero = RFloat::with_val(prec, 0.0);
    [
        [(cos_half.clone(), RFloat::with_val(prec, -&sin_half)), (zero.clone(), zero.clone())],
        [(zero.clone(), zero.clone()), (cos_half, sin_half)],
    ]
}

/// Rebuild U2Q from the gate string returned by the synthesizer.
fn rebuild_u2q(gates: &str) -> U2Q {
    let mut u = U2Q::eye();
    for ch in gates.chars() {
        u = match ch {
            'T' => u * U2Q::t(),
            'H' => u * U2Q::h(),
            'S' => u * U2Q::s(),
            'X' => u * U2Q::x(),
            'Y' => u * U2Q::y(),
            'Z' => u * U2Q::z(),
            'Q' => u * U2Q::q(),
            _ => panic!("unexpected gate {ch}"),
        };
    }
    u
}

fn main() {
    let theta = 0.3_f64;
    let target_f64 = rz_f64(theta);
    let prec: u32 = 256;
    let theta_mpfr = RFloat::with_val(prec, theta);
    let target_mpfr = rz_mpfr(&theta_mpfr, prec);

    for &eps in &[5e-8_f64, 2e-8, 1.5e-8] {
        eprintln!("─── ε={:e} ───", eps);
        let synth = SynthesizerQ::new(eps).with_max_lde(35);
        let r = match synth.synthesize(target_f64) {
            Some(r) => r,
            None => { eprintln!("  None"); continue; }
        };
        let gates = r.gates.clone().expect("gates");
        let dist_reported = r.distance;

        // Rebuild U2Q from gates and recompute distance via MPFR target.
        let u2q = rebuild_u2q(&gates);
        let dist_f64 = diamond_distance_u2q_float(&u2q, &target_f64);
        let dist_mpfr = diamond_distance_u2q_mpfr_target(&u2q, &target_mpfr);

        eprintln!("  gates len: {}", gates.len());
        eprintln!("  reported lde={}, dist={:.6e}", r.lde, dist_reported);
        eprintln!("  rebuilt U2Q.k = {}", u2q.k);
        eprintln!("  dist (f64 target):   {:.6e}", dist_f64);
        eprintln!("  dist (MPFR target):  {:.6e}", dist_mpfr);
        eprintln!("  ε:                   {:.6e}", eps);
        if dist_mpfr < eps {
            eprintln!("  ✓ VERIFIED — MPFR-target dist < ε");
        } else {
            eprintln!("  ✗ FAIL — MPFR-target dist >= ε; f64 was lying!");
        }
    }
}
