//! Phase 1-4 diagnostic per critic feedback.
//!
//! Phase 1: capture raw x_inner from SE walk via CYCLOSYNTH_CAPTURE=1.
//! Phase 2: verify x_inner→U_full is MPFR-valid (dist < ε in MPFR).
//! Phase 3: test cap membership for x_inner under MPFR-correct setup.
//! Phase 4: not yet — would require SE walk path-trace instrumentation.

use cyclosynth::matrix::U2Q;
use cyclosynth::synthesis::clifford_sqrt_t::{
    SynthesizerQ, Mat2Mpfr, mat2_to_mat2_mpfr, u2q_dag_times_mat2_mpfr,
    unitary_to_uv_zeta_mpfr, det_phase_of_mat2_mpfr, solution_to_u2q_d,
};
use cyclosynth::synthesis::diag;
use cyclosynth::synthesis::distance::{
    diamond_distance_u2q_float, diamond_distance_u2q_mpfr_target,
};
use cyclosynth::synthesis::lenstra_zeta::{
    IntScratch16, build_q_int_zeta, build_q_mpfr_zeta_from_mpfr_v,
    bilinear_forms, det16_exact,
};
use cyclosynth::synthesis::lenstra_zeta::cholesky_lu::{
    cholesky_f64_16, lu_solve_int_inplace_16,
};
use cyclosynth::synthesis::lenstra_zeta::lll::run_lll_16;
use cyclosynth::synthesis::search_zeta::uv_to_xy_zeta_mpfr;
use num_complex::Complex;
use rug::{Assign, Float as RFloat};

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

fn main() {
    std::env::set_var("CYCLOSYNTH_CAPTURE", "1");
    let theta = 0.3_f64;
    let target = rz_f64(theta);
    let prec: u32 = 192;
    let theta_mpfr = RFloat::with_val(prec, theta);
    let target_mpfr = rz_mpfr(&theta_mpfr, prec);

    let eps = 1.5e-8_f64;

    eprintln!("=== Phase 1: synthesize at ε={:e} with capture ===", eps);
    let synth = SynthesizerQ::new(eps).with_max_lde(35);
    let r = match synth.synthesize(target) {
        Some(r) => r,
        None => { eprintln!("  None — synthesize didn't find"); return; }
    };
    eprintln!("  found: lde={}, dist (f64-target)={:.6e}", r.lde, r.distance);

    let cap = match diag::CAPTURED_FIND.lock().unwrap().clone() {
        Some(c) => c,
        None => { eprintln!("  ERROR: no capture"); return; }
    };
    eprintln!("  CAPTURED: k_total={}, k_inner={}, d_r={}, d_l={}",
        cap.k_total, cap.k_inner, cap.d_r, cap.d_l);
    eprintln!("  x_inner = {:?}", cap.x_inner);
    let x_norm_sq: i128 = cap.x_inner.iter().map(|&v| (v as i128) * (v as i128)).sum();
    eprintln!("  ‖x_inner‖² = {}, target_norm = 2^{} = {}",
        x_norm_sq, cap.k_inner, 1i128 << cap.k_inner);

    let (b1, b2, b3) = bilinear_forms(&cap.x_inner);
    eprintln!("  bilinear: B1={}, B2={}, B3={} (must all be 0)", b1, b2, b3);

    eprintln!("\n=== Phase 2: MPFR distance check on captured x ===");
    let u_r = solution_to_u2q_d(&cap.x_inner, cap.k_inner, cap.d_r);
    // Reconstruct U_L from gates? We need it. Easier: try every prefix in build_l_q
    // and find the one with matching d_l and producing a valid U_full.
    // For a quick check, just compute dist on U_R alone (assuming U_L identity)
    // is wrong; we need the actual U_L. Let me reconstruct via gate string.
    let gates = r.gates.clone().expect("gates");
    let mut u_full_from_gates = U2Q::eye();
    for ch in gates.chars() {
        u_full_from_gates = match ch {
            'T' => u_full_from_gates * U2Q::t(),
            'H' => u_full_from_gates * U2Q::h(),
            'S' => u_full_from_gates * U2Q::s(),
            'X' => u_full_from_gates * U2Q::x(),
            'Y' => u_full_from_gates * U2Q::y(),
            'Z' => u_full_from_gates * U2Q::z(),
            'Q' => u_full_from_gates * U2Q::q(),
            _ => panic!("unexpected gate {ch}"),
        };
    }
    let dist_full_f64 = diamond_distance_u2q_float(&u_full_from_gates, &target);
    let dist_full_mpfr = diamond_distance_u2q_mpfr_target(&u_full_from_gates, &target_mpfr);
    eprintln!("  U_full from gates: k={}", u_full_from_gates.k);
    eprintln!("  dist (f64 target):  {:.6e}", dist_full_f64);
    eprintln!("  dist (MPFR target): {:.6e}", dist_full_mpfr);
    eprintln!("  ε = {:.6e}", eps);
    if dist_full_mpfr < eps {
        eprintln!("  ✓ MPFR-valid: f64-found candidate satisfies dist < ε under MPFR semantics");
    } else {
        eprintln!("  ✗ MPFR-INVALID: f64 path returned a false positive!");
        eprintln!("    This means precision IS the bug, in the f64 distance check.");
        return;
    }

    eprintln!("\n=== Phase 3: cap membership for x_inner under MPFR-correct setup ===");
    eprintln!("  Building MPFR Q at k_inner={}, deriving inner v_mpfr from U_L†·target...",
        cap.k_inner);

    // Need U_L. Find it among build_l_q prefixes by matching d_l.
    use cyclosynth::synthesis::clifford_sqrt_t::build_l_q;
    let m_split = 2u32; // assume m=2 (default at deep ε)
    let prefixes = build_l_q(m_split);
    let target_d_target_mpfr = det_phase_of_mat2_mpfr(&target_mpfr);
    let _ = target_d_target_mpfr;
    let mut u_l_candidate: Option<U2Q> = None;
    // Check that U_L · U_R == U_full reconstructed from gates.
    // The k of U_L · U_R should be k_inner + k_prefix; gates rebuild gives k=41.
    // We just need ANY u_l in the prefix set such that u_l * u_r matches the
    // matrix we'd expect. Easier: check via float comparison.
    let mut tried = 0usize;
    let mut by_d_l = 0usize;
    let mut k_dist: std::collections::BTreeMap<u32, usize> = std::collections::BTreeMap::new();
    for u_l in prefixes.iter() {
        *k_dist.entry(u_l.k).or_insert(0) += 1;
    }
    eprintln!("  prefix k distribution: {:?}", k_dist);
    for u_l in prefixes.iter() {
        tried += 1;
        let d_l = cyclosynth::synthesis::clifford_sqrt_t::det_phase_of(&u_l.to_float());
        if d_l != cap.d_l { continue; }
        by_d_l += 1;
        let u_full_test = *u_l * u_r;
        // Compare via float matrices (exact comparison may fail because k differs)
        // Diamond distance is global-phase-invariant; compare via that.
        let f_test = u_full_test.to_float();
        let diff = cyclosynth::synthesis::diamond_distance_float(&f_test, &target);
        if (diff - r.distance).abs() < 1e-9 {
            u_l_candidate = Some(*u_l);
            eprintln!("  matched U_L: u_l.k={}, diff vs u_full_from_gates: {:.3e}", u_l.k, diff);
            break;
        }
    }
    if u_l_candidate.is_none() {
        // Show the 5 closest prefixes for diagnosis.
        let mut diffs: Vec<(f64, U2Q)> = prefixes.iter().filter_map(|u_l| {
            let d_l = cyclosynth::synthesis::clifford_sqrt_t::det_phase_of(&u_l.to_float());
            if d_l != cap.d_l { return None; }
            let u_full_test = *u_l * u_r;
            let f_test = u_full_test.to_float();
            let diff = cyclosynth::synthesis::diamond_distance_float(&f_test, &target);
            Some((diff, *u_l))
        }).collect();
        diffs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        eprintln!("  closest 5 d_l=0 prefixes by diff:");
        for (d, u_l) in diffs.iter().take(5) {
            eprintln!("    diff={:.3e}, u_l.k={}", d, u_l.k);
        }
    }
    eprintln!("  search: tried {} prefixes, {} matched d_l={}", tried, by_d_l, cap.d_l);
    let u_l = match u_l_candidate {
        Some(u) => u,
        None => {
            eprintln!("  WARNING: couldn't identify U_L. Skipping cap test.");
            return;
        }
    };
    eprintln!("  identified U_L (k={}, d_l={})", u_l.k, cap.d_l);

    // Compute MPFR v_inner = unitary_to_uv_zeta(U_L† · target_mpfr)
    let m_inner_mpfr = u2q_dag_times_mat2_mpfr(&u_l, &target_mpfr, prec);
    let v_inner_mpfr = unitary_to_uv_zeta_mpfr(&m_inner_mpfr);
    let y_inner_mpfr = uv_to_xy_zeta_mpfr(&v_inner_mpfr, cap.k_inner, prec);

    let mut scratch = IntScratch16::new(eps);
    scratch.reset_basis();
    build_q_mpfr_zeta_from_mpfr_v(&mut scratch, &v_inner_mpfr, cap.k_inner, eps);
    build_q_int_zeta(&mut scratch);
    let lll_result = run_lll_16(&mut scratch);
    eprintln!("  LLL (MPFR-correct): {:?}", lll_result);
    let det = det16_exact(&scratch.basis);
    eprintln!("  det(B_MPFR) = {:?}", det);

    // Compute cap_mid in MPFR and c[i].
    let prec_q = scratch.prec_q;
    let one = RFloat::with_val(prec_q, 1.0);
    let two = RFloat::with_val(prec_q, 2.0);
    let eps_rf = RFloat::with_val(prec_q, eps);
    let eps_sq = RFloat::with_val(prec_q, &eps_rf * &eps_rf);
    let one_minus_eps_sq = RFloat::with_val(prec_q, &one - &eps_sq);
    let sqrt_1m = one_minus_eps_sq.sqrt();
    let cap_mid_num = RFloat::with_val(prec_q, &one + &sqrt_1m);
    let cap_mid = RFloat::with_val(prec_q, &cap_mid_num / &two);
    for i in 0..16 {
        scratch.c[i].assign(RFloat::with_val(prec_q, &y_inner_mpfr[i] * &cap_mid));
    }
    if !cholesky_f64_16(&mut scratch) {
        eprintln!("  cholesky FAILED"); return;
    }
    if !lu_solve_int_inplace_16(&mut scratch) {
        eprintln!("  lu_solve FAILED"); return;
    }
    let z_c: [i64; 16] = std::array::from_fn(|i| {
        let mut rounded = scratch.lu_x[i].clone();
        rounded.round_mut();
        match rounded.to_integer() {
            Some(int) => int.to_i64_wrapping(),
            None => 0,
        }
    });
    eprintln!("  z_c (MPFR-correct): max|z_c| = {}",
        z_c.iter().map(|v| v.abs()).max().unwrap());

    // Compute Q-norm² of (x_inner - c_inner) in standard coords using MPFR Q.
    let mut q_norm_sq = RFloat::with_val(prec_q, 0.0);
    let mut diff_vec: [RFloat; 16] = std::array::from_fn(|_| RFloat::with_val(prec_q, 0.0));
    for i in 0..16 {
        let xi = RFloat::with_val(prec_q, cap.x_inner[i]);
        diff_vec[i].assign(RFloat::with_val(prec_q, &xi - &scratch.c[i]));
    }
    for i in 0..16 {
        for j in 0..16 {
            let term = RFloat::with_val(prec_q,
                &diff_vec[i] * &RFloat::with_val(prec_q, &scratch.q_mpfr[i][j] * &diff_vec[j]));
            q_norm_sq += term;
        }
    }
    let qn = q_norm_sq.to_f64();
    eprintln!("\n=== Cap-membership result ===");
    eprintln!("  Q-norm²(x_inner - c) under MPFR-correct Q = {:.6e}", qn);
    eprintln!("  bound_sq=8 → covered if Q-norm² ≤ 8");
    eprintln!("  bound_sq=16 → covered if Q-norm² ≤ 16");
    if qn <= 8.0 {
        eprintln!("  ✓ INSIDE bound_sq=8 cap. SE walk should have enumerated it.");
        eprintln!("    => MPFR no-find at ε=1.5e-8 is a SEARCH MACHINERY bug, not cap geometry.");
        eprintln!("    Phase 4 needed: trace which prune rejects this candidate.");
    } else if qn <= 16.0 {
        eprintln!("  △ OUTSIDE bound_sq=8, INSIDE bound_sq=16. CAP IS SUFFICIENT-NOT-NECESSARY.");
        eprintln!("    => bound_sq scaling is the right answer; the f64 noise expanded the cap correctly.");
    } else {
        eprintln!("  ✗ OUTSIDE both bounds (Q-norm² > 16).");
        eprintln!("    => Cap is far too tight at deep ε, or my Q construction has a remaining bug.");
    }
}
