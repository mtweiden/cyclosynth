//! ε=1e-8 with a random SU(2) target (Rz·Ry·Rz) at MPFR precision.
//! Tests whether Rz(0.3) is anomalously hard or whether the failure
//! generalises.

use cyclosynth::synthesis::clifford_sqrt_t::{SynthesizerQ, Mat2Mpfr};
use rug::Float as RFloat;
use std::time::Instant;

fn xorshift64(s: &mut u64) -> u64 { *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; *s }
fn rand_angle(s: &mut u64) -> f64 {
    use std::f64::consts::PI;
    let b = xorshift64(s) >> 11;
    (b as f64) / ((1u64 << 53) as f64) * 2.0 * PI
}

/// Build U(2) = Rz(α)·Ry(β)·Rz(γ) at MPFR precision.
fn u3_mpfr(prec: u32, alpha: f64, beta: f64, gamma: f64) -> Mat2Mpfr {
    let alpha_m = RFloat::with_val(prec, alpha);
    let beta_m = RFloat::with_val(prec, beta);
    let gamma_m = RFloat::with_val(prec, gamma);
    // Rz(θ) = diag(e^(-iθ/2), e^(+iθ/2))
    let rz = |theta: &RFloat| -> Mat2Mpfr {
        let half = RFloat::with_val(prec, theta / 2);
        let cos_h = half.clone().cos();
        let sin_h = half.clone().sin();
        let zero = RFloat::with_val(prec, 0.0);
        [
            [(cos_h.clone(), RFloat::with_val(prec, -&sin_h)), (zero.clone(), zero.clone())],
            [(zero.clone(), zero.clone()), (cos_h, sin_h)],
        ]
    };
    // Ry(θ) = [[cos(θ/2), -sin(θ/2)], [sin(θ/2), cos(θ/2)]] (real entries)
    let ry = |theta: &RFloat| -> Mat2Mpfr {
        let half = RFloat::with_val(prec, theta / 2);
        let c = half.clone().cos();
        let s = half.clone().sin();
        let zero = RFloat::with_val(prec, 0.0);
        [
            [(c.clone(), zero.clone()), (RFloat::with_val(prec, -&s), zero.clone())],
            [(s, zero.clone()), (c, zero)],
        ]
    };
    // Matrix mult
    let mul = |a: &Mat2Mpfr, b: &Mat2Mpfr| -> Mat2Mpfr {
        let cm = |x: &(RFloat, RFloat), y: &(RFloat, RFloat)| -> (RFloat, RFloat) {
            (
                RFloat::with_val(prec, &x.0 * &y.0) - RFloat::with_val(prec, &x.1 * &y.1),
                RFloat::with_val(prec, &x.0 * &y.1) + RFloat::with_val(prec, &x.1 * &y.0),
            )
        };
        let cadd = |x: (RFloat, RFloat), y: (RFloat, RFloat)| -> (RFloat, RFloat) {
            (RFloat::with_val(prec, &x.0 + &y.0), RFloat::with_val(prec, &x.1 + &y.1))
        };
        [
            [cadd(cm(&a[0][0], &b[0][0]), cm(&a[0][1], &b[1][0])),
             cadd(cm(&a[0][0], &b[0][1]), cm(&a[0][1], &b[1][1]))],
            [cadd(cm(&a[1][0], &b[0][0]), cm(&a[1][1], &b[1][0])),
             cadd(cm(&a[1][0], &b[0][1]), cm(&a[1][1], &b[1][1]))],
        ]
    };
    let m1 = rz(&alpha_m);
    let m2 = ry(&beta_m);
    let m3 = rz(&gamma_m);
    mul(&mul(&m1, &m2), &m3)
}

fn main() {
    std::env::set_var("CYCLOSYNTH_TRACE", "1");
    let eps = 1e-8_f64;
    let prec: u32 = 192;

    // Use the same seed as the bench harness so the target is reproducible.
    let mut state: u64 = 0xC0FFEEBAADD0E;
    // First random target (target_00)
    let alpha = rand_angle(&mut state);
    let beta = rand_angle(&mut state);
    let gamma = rand_angle(&mut state);
    eprintln!("─── Random U(2) target_00 at ε=1e-8 (MPFR D&C) ───");
    eprintln!("  α={alpha:.4} β={beta:.4} γ={gamma:.4}");

    let target_mpfr = u3_mpfr(prec, alpha, beta, gamma);

    let synth = SynthesizerQ::new(eps);
    eprintln!(
        "  config: max_lde={}, min_lde={}, dc_split={:?}, dr_filter={:?}, bkz={}",
        synth.max_lde, synth.min_lde, synth.dc_split, synth.dc_dr_filter,
        synth.bkz_block_size,
    );
    let t = Instant::now();
    let r = synth.synthesize_mpfr(&target_mpfr);
    let dt = t.elapsed().as_secs_f64();
    match r {
        Some(r) => eprintln!("  RESULT: lde={}, dist={:.3e}, took {:.2} s", r.lde, r.distance, dt),
        None => eprintln!("  RESULT: None after {:.2} s", dt),
    }
}
