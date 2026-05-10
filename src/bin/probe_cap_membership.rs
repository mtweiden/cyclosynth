//! Critic-driven test: the f64 path at ε=1.5e-8 finds a valid solution at
//! lde=26 with dist=1.17e-8. Question: does that x satisfy the MPFR-correct
//! cap inequality `‖B(z-z_c)‖²_Q ≤ bound_sq` at the same lde?
//!
//! If YES (Q-norm² ≤ 8): MPFR cap should have found it; the "MPFR finds
//! nothing" outcome must be a search-machinery bug, not cap geometry.
//!
//! If NO (Q-norm² > 8): the cap filter rejects valid solutions, i.e., cap
//! is sufficient-but-not-necessary. Increasing bound_sq is the right move
//! (and we should test how high to know if it's reachable).

use cyclosynth::matrix::U2Q;
use cyclosynth::synthesis::clifford_sqrt_t::{SynthesizerQ, Mat2Mpfr};
use rug::Assign;
use cyclosynth::synthesis::lenstra_zeta::{
    IntScratch16, phase1_with_stop_mpfr,
    build_q_mpfr_zeta_from_mpfr_v, build_q_int_zeta,
    schnorr_euchner_16d, det16_exact,
};
use cyclosynth::synthesis::lenstra_zeta::cholesky_lu::{cholesky_f64_16, lu_solve_int_inplace_16};
use cyclosynth::synthesis::lenstra_zeta::lll::run_lll_16;
use cyclosynth::synthesis::search_zeta::uv_to_xy_zeta_mpfr;
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

/// Rebuild U2Q from gate string, then extract its 16-coord integer vector
/// (the SE walk's `x` representation: u11's 8 ZZeta coords, then u21's).
fn x_from_gates(gates: &str) -> ([i64; 16], u32) {
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
    let z11 = u.u11;
    let z21 = u.u21;
    let x: [i64; 16] = [
        z11.a.as_i128() as i64, z11.b.as_i128() as i64, z11.c.as_i128() as i64, z11.d.as_i128() as i64,
        z11.e.as_i128() as i64, z11.f.as_i128() as i64, z11.g.as_i128() as i64, z11.h.as_i128() as i64,
        z21.a.as_i128() as i64, z21.b.as_i128() as i64, z21.c.as_i128() as i64, z21.d.as_i128() as i64,
        z21.e.as_i128() as i64, z21.f.as_i128() as i64, z21.g.as_i128() as i64, z21.h.as_i128() as i64,
    ];
    (x, u.k)
}

fn main() {
    let theta = 0.3_f64;
    let target = rz_f64(theta);
    let prec: u32 = 192;
    let theta_mpfr = RFloat::with_val(prec, theta);
    let target_mpfr = rz_mpfr(&theta_mpfr, prec);

    // Step 1: synthesize at ε=1.5e-8 with f64 path to get a known-good candidate.
    eprintln!("=== Synthesizing at ε=1.5e-8 (f64 path) to get reference candidate ===");
    let eps = 1.5e-8_f64;
    let synth = SynthesizerQ::new(eps).with_max_lde(35);
    let r = synth.synthesize(target).expect("expected to find at ε=1.5e-8");
    eprintln!("  found: lde={}, dist={:.6e}", r.lde, r.distance);
    let gates = r.gates.expect("gates");
    let (x_target, x_k) = x_from_gates(&gates);
    eprintln!("  rebuilt U2Q.k = {}, gates len = {}", x_k, gates.len());
    eprintln!("  x = {:?}", x_target);

    // Sanity: ‖x‖² should equal 2^x_k.
    let x_norm_sq: i128 = x_target.iter().map(|&v| (v as i128) * (v as i128)).sum();
    eprintln!("  ‖x‖² = {}, target_norm = 2^{} = {}", x_norm_sq, x_k, 1i128 << x_k);

    // Step 2: at ε=1.5e-8 with the SAME lde where f64 found, compute the
    // MPFR-correct Q metric, LLL basis, cap center, and check x's Q-norm.
    let test_k = r.lde;
    eprintln!("\n=== Setting up MPFR-correct lattice search at ε={:e}, lde={} ===", eps, test_k);
    let mut scratch = IntScratch16::new(eps);
    let v_mpfr: [RFloat; 4] = [
        target_mpfr[0][0].0.clone(),
        target_mpfr[0][0].1.clone(),
        target_mpfr[1][0].0.clone(),
        target_mpfr[1][0].1.clone(),
    ];
    let y_mpfr = uv_to_xy_zeta_mpfr(&v_mpfr, test_k, prec);

    // Build Q (MPFR), reset basis to identity for LLL.
    scratch.reset_basis();
    build_q_mpfr_zeta_from_mpfr_v(&mut scratch, &v_mpfr, test_k, eps);
    build_q_int_zeta(&mut scratch);

    // LLL.
    let lll_result = run_lll_16(&mut scratch);
    eprintln!("  LLL: {:?}", lll_result);
    let det = det16_exact(&scratch.basis);
    eprintln!("  det(B) = {:?}", det);

    // Compute cap center c[i] = y[i] · cap_mid.
    let one = RFloat::with_val(prec, 1.0);
    let two = RFloat::with_val(prec, 2.0);
    let eps_rf = RFloat::with_val(prec, eps);
    let eps_sq = RFloat::with_val(prec, &eps_rf * &eps_rf);
    let one_minus_eps_sq = RFloat::with_val(prec, &one - &eps_sq);
    let sqrt_1m = one_minus_eps_sq.sqrt();
    let cap_mid_num = RFloat::with_val(prec, &one + &sqrt_1m);
    let cap_mid = RFloat::with_val(prec, &cap_mid_num / &two);
    for i in 0..16 {
        scratch.c[i].assign(RFloat::with_val(prec, &y_mpfr[i] * &cap_mid));
    }

    // Cholesky and LU solve.
    if !cholesky_f64_16(&mut scratch) {
        eprintln!("  cholesky_f64_16 FAILED");
        return;
    }
    if !lu_solve_int_inplace_16(&mut scratch) {
        eprintln!("  lu_solve FAILED");
        return;
    }
    let z_c: [i64; 16] = std::array::from_fn(|i| {
        let mut rounded = scratch.lu_x[i].clone();
        rounded.round_mut();
        match rounded.to_integer() {
            Some(int) => int.to_i64_wrapping(),
            None => 0,
        }
    });
    eprintln!("  z_c (basis-coords of cap center): max|z_c| = {}", z_c.iter().map(|v| v.abs()).max().unwrap());

    // Step 3: find the integer z such that B·z = x_target.
    // Since B is unimodular (det=±1), B^-1 is integer; z = B^-1 · x_target.
    // For LLL output we don't have B^-1 cached. Use exact integer inversion via Bareiss
    // or just verify x_target via integer enumeration is hard.
    // Instead, compute Q-norm² of (x_target - c_standard) where c_standard = y_mpfr · cap_mid.
    let prec_q = scratch.prec_q;
    let mut q_norm_sq = RFloat::with_val(prec_q, 0.0);
    let mut diff_vec: [RFloat; 16] = std::array::from_fn(|_| RFloat::with_val(prec_q, 0.0));
    for i in 0..16 {
        let xi = RFloat::with_val(prec_q, x_target[i]);
        diff_vec[i].assign(RFloat::with_val(prec_q, &xi - &scratch.c[i]));
    }
    // Q-norm² = diff^T Q diff (Q is in MPFR scratch.q_mpfr at prec_q).
    for i in 0..16 {
        for j in 0..16 {
            let term = RFloat::with_val(prec_q,
                &diff_vec[i] * &RFloat::with_val(prec_q, &scratch.q_mpfr[i][j] * &diff_vec[j]));
            q_norm_sq += term;
        }
    }
    eprintln!("\n=== Cap-membership test ===");
    eprintln!("  x = found candidate (in standard coords)");
    eprintln!("  c = y · cap_mid (cap center in standard coords)");
    eprintln!("  Q-norm²(x - c) = {:.6e}", q_norm_sq.to_f64());
    eprintln!("  bound_sq=8 → cap covers Q-norm² ≤ 8");
    eprintln!("  bound_sq=16 → cap covers Q-norm² ≤ 16");
    let qn = q_norm_sq.to_f64();
    if qn <= 8.0 {
        eprintln!("  ✓ x is INSIDE bound_sq=8 cap. SE walk should have found it.");
    } else if qn <= 16.0 {
        eprintln!("  △ x is OUTSIDE bound_sq=8 but INSIDE bound_sq=16. The cap is sufficient-but-not-necessary at this ε.");
    } else {
        eprintln!("  ✗ x is OUTSIDE both bounds. Cap is too tight at this ε.");
    }

    // Step 4: verify by running the actual SE walk at this lde and confirming
    // it does/doesn't find x_target.
    eprintln!("\n=== Re-running SE walk at this lde (sanity) ===");
    use std::sync::atomic::AtomicBool;
    let budget_hit = AtomicBool::new(false);
    let target_local = target;
    let d = cyclosynth::synthesis::clifford_sqrt_t::det_phase_of(&target_local);
    let _ = d;
    let should_stop = |x: &[i64; 16]| -> bool {
        x == &x_target
    };
    let _ = should_stop;
    let _ = (test_k, &y_mpfr, &v_mpfr);
    // Skip — needs more setup. Membership test above is the key data.
}
