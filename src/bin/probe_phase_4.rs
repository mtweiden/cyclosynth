//! Phase 4: Trace SE walk path. Given the captured x_target known to be
//! inside the MPFR-correct cap (Q-norm²=0.91, Phase 3 verified), walk
//! through the SE recursion levels d=15..0 and report at each level:
//!   - z_target[d]: where z_target's path enters the bracket
//!   - z_low, z_high: SE walk's bracket bounds
//!   - in_bracket: is z_target[d] ∈ [z_low, z_high]?
//!   - partial_q, partial_eucl: cumulative bound checks

use cyclosynth::matrix::U2Q;
use cyclosynth::synthesis::clifford_sqrt_t::{
    SynthesizerQ, Mat2Mpfr, build_l_q, det_phase_of, solution_to_u2q_d,
    u2q_dag_times_mat2_mpfr, unitary_to_uv_zeta_mpfr,
};
use cyclosynth::synthesis::diag;
use cyclosynth::synthesis::lenstra_zeta::{
    IntScratch16, build_q_int_zeta, build_q_mpfr_zeta_from_mpfr_v,
    det16_exact, euclidean_cholesky_16,
};
use cyclosynth::synthesis::lenstra_zeta::se::euclidean_cholesky_16_mpfr;
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

fn find_u_l(prefixes: &[U2Q], d_l: u32, u_r: U2Q, target: &Mat2, expected_dist: f64) -> Option<U2Q> {
    use cyclosynth::synthesis::diamond_distance_float;
    for u_l in prefixes.iter() {
        if det_phase_of(&u_l.to_float()) != d_l { continue; }
        let u_full_test = *u_l * u_r;
        let f_test = u_full_test.to_float();
        let diff = diamond_distance_float(&f_test, target);
        if (diff - expected_dist).abs() < 1e-9 {
            return Some(*u_l);
        }
    }
    None
}

fn main() {
    std::env::set_var("CYCLOSYNTH_CAPTURE", "1");
    let theta = 0.3_f64;
    let target = rz_f64(theta);
    let prec: u32 = 192;
    let theta_mpfr = RFloat::with_val(prec, theta);
    let target_mpfr = rz_mpfr(&theta_mpfr, prec);
    let eps = 1.5e-8_f64;

    eprintln!("=== Phase 1: capture x_target ===");
    let synth = SynthesizerQ::new(eps).with_max_lde(35);
    let r = synth.synthesize(target).expect("expected to find");
    let cap = diag::CAPTURED_FIND.lock().unwrap().clone()
        .expect("capture must fire");
    eprintln!("  k_total={}, k_inner={}, d_r={}, d_l={}",
        cap.k_total, cap.k_inner, cap.d_r, cap.d_l);

    let u_r = solution_to_u2q_d(&cap.x_inner, cap.k_inner, cap.d_r);
    let prefixes = build_l_q(2);
    let u_l = find_u_l(&prefixes, cap.d_l, u_r, &target, r.distance)
        .expect("U_L must be found");
    eprintln!("  identified U_L (k={})", u_l.k);

    eprintln!("\n=== Phase 4: SE walk path-trace ===");
    let m_inner_mpfr = u2q_dag_times_mat2_mpfr(&u_l, &target_mpfr, prec);
    let v_inner_mpfr = unitary_to_uv_zeta_mpfr(&m_inner_mpfr);
    let y_inner_mpfr = uv_to_xy_zeta_mpfr(&v_inner_mpfr, cap.k_inner, prec);

    let mut scratch = IntScratch16::new(eps);
    scratch.reset_basis();
    build_q_mpfr_zeta_from_mpfr_v(&mut scratch, &v_inner_mpfr, cap.k_inner, eps);
    build_q_int_zeta(&mut scratch);
    let lll_result = run_lll_16(&mut scratch);
    eprintln!("  LLL: {:?}", lll_result);
    let det = det16_exact(&scratch.basis);
    eprintln!("  det(B) = {:?}", det);

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
    if !cholesky_f64_16(&mut scratch) { eprintln!("cholesky FAIL"); return; }
    if !lu_solve_int_inplace_16(&mut scratch) { eprintln!("lu_solve FAIL"); return; }
    let z_c: [i64; 16] = std::array::from_fn(|i| {
        let mut rounded = scratch.lu_x[i].clone();
        rounded.round_mut();
        rounded.to_integer().map(|n| n.to_i64_wrapping()).unwrap_or(0)
    });
    eprintln!("  z_c (basis-coords cap center): max|z_c| = {}",
        z_c.iter().map(|v| v.abs()).max().unwrap());

    // Compute z_target = (B^T)^-1 · x_target. lu_solve loads RHS from
    // scratch.c, so put x_target there.
    let lu_prec = scratch.lu_prec;
    for i in 0..16 {
        scratch.c[i].assign(rug::Float::with_val(lu_prec, cap.x_inner[i] as f64));
    }
    if !lu_solve_int_inplace_16(&mut scratch) { eprintln!("lu_solve(x_target) FAIL"); return; }
    let z_target: [i64; 16] = std::array::from_fn(|i| {
        let mut rounded = scratch.lu_x[i].clone();
        rounded.round_mut();
        rounded.to_integer().map(|n| n.to_i64_wrapping()).unwrap_or(0)
    });
    eprintln!("  z_target (B^T)^-1 · x_target: max|z_target| = {}",
        z_target.iter().map(|v| v.abs()).max().unwrap());

    // Verify B · z_target ≈ x_target (sanity check).
    let mut x_check = [0i64; 16];
    for i in 0..16 {
        for j in 0..16 {
            x_check[i] = x_check[i].wrapping_add(scratch.basis[j][i].wrapping_mul(z_target[j]));
        }
    }
    let matches: bool = x_check == cap.x_inner;
    eprintln!("  B · z_target = x_target? {}", if matches { "✓" } else {
        let diffs: Vec<i64> = x_check.iter().zip(cap.x_inner.iter()).map(|(a,b)| a - b).collect();
        eprintln!("    x_check    = {:?}", x_check);
        eprintln!("    x_target   = {:?}", cap.x_inner);
        eprintln!("    diff       = {:?}", diffs);
        "✗ — z_target rounding lost precision"
    });

    if !matches {
        eprintln!("  Cannot proceed with path trace; z_target isn't a valid lattice basis representation.");
        return;
    }

    // Recreate l (Q-Cholesky) and r_eucl (Euclidean Cholesky).
    let l_lower = scratch.l_f64;
    let l: [[f64; 16]; 16] = std::array::from_fn(|i| std::array::from_fn(|j| l_lower[j][i]));  // upper-triangular
    let r_eucl = euclidean_cholesky_16_mpfr(&scratch.basis)
        .or_else(|| euclidean_cholesky_16(&scratch.basis))
        .expect("eucl cholesky");

    let bound_sq = 8.0_f64;
    let target_norm_sq = 2.0_f64.powi(cap.k_inner as i32);

    eprintln!("\n  bound_sq = {}", bound_sq);
    eprintln!("  target_norm² = 2^{} = {}", cap.k_inner, target_norm_sq);

    // Walk through depths 15 → 0, replicating SE bracket arithmetic. At each
    // depth, check whether z_target[d] is within [z_low, z_high].
    eprintln!("\n  depth | z_target[d] | z_c[d]   | center_off  | span      | z_low      | z_high     | in? | partial_q | partial_eucl");
    eprintln!("  ──────┼──────────────┼──────────┼─────────────┼───────────┼────────────┼────────────┼─────┼───────────┼─────────────");
    let mut z = [0i64; 16];
    let mut partial_q = 0.0_f64;
    let mut partial_eucl = 0.0_f64;
    let mut w = [0f64; 16];
    let mut all_in = true;
    let mut prune_at: Option<i32> = None;

    for d in (0..16).rev() {
        let l_dd = l[d][d];
        let mut tail = 0.0_f64;
        for j in (d + 1)..16 {
            tail += l[d][j] * ((z[j] - z_c[j]) as f64);
        }
        let rem = bound_sq - partial_q;
        let (z_low, z_high, center_off, span) = if rem < 0.0 {
            (i64::MAX, i64::MIN, f64::NAN, f64::NAN)
        } else if l_dd.abs() < 1e-30 {
            (z_c[d], z_c[d], 0.0, 0.0)
        } else {
            let rs = rem.sqrt();
            let co = -tail / l_dd;
            let sp = rs / l_dd.abs();
            (
                z_c[d].saturating_add((co - sp).ceil() as i64),
                z_c[d].saturating_add((co + sp).floor() as i64),
                co, sp,
            )
        };
        let zd = z_target[d];
        let in_bracket = z_low <= zd && zd <= z_high;
        // Now set z[d] and update partial_q + partial_eucl as if we descended.
        let delta = zd - z[d];
        z[d] = zd;
        if delta != 0 {
            let delta_f = delta as f64;
            for i in 0..=d {
                w[i] += delta_f * r_eucl[i][d];
            }
        }
        let level = l_dd * ((zd - z_c[d]) as f64) + tail;
        let new_partial_q = partial_q + level * level;
        let level_eucl = w[d];
        let new_partial_eucl = partial_eucl + level_eucl * level_eucl;
        let q_cut = new_partial_q > bound_sq + 1e-9 * bound_sq.abs();
        let eucl_cut = d > 0 && new_partial_eucl > target_norm_sq * (1.0 + 1e-9);
        eprintln!("  {:>5} | {:>12} | {:>8} | {:>11.4e} | {:>9.3} | {:>10} | {:>10} | {:>3} | {:>9.3} | {:>11.3}",
            d, zd, z_c[d], center_off, span, z_low, z_high,
            if in_bracket { "✓" } else { "✗" },
            new_partial_q, new_partial_eucl);
        if !in_bracket && all_in {
            eprintln!("    ⮕ z_target[{}] OUTSIDE bracket [{}, {}]; offset from z_c = {}",
                d, z_low, z_high, zd - z_c[d]);
            all_in = false;
            prune_at = Some(d as i32);
        }
        if q_cut && all_in {
            eprintln!("    ⮕ partial_q={:.3} EXCEEDS bound_sq={} at depth {}",
                new_partial_q, bound_sq, d);
            all_in = false;
            prune_at = Some(d as i32);
        }
        if eucl_cut && all_in {
            eprintln!("    ⮕ partial_eucl={:.3} EXCEEDS target_norm²={} at depth {}",
                new_partial_eucl, target_norm_sq, d);
            all_in = false;
            prune_at = Some(d as i32);
        }
        partial_q = new_partial_q;
        partial_eucl = new_partial_eucl;
    }

    eprintln!("\n  Result:");
    if all_in {
        eprintln!("  ✓ z_target's path lies fully within the SE walk's enumerated region.");
        eprintln!("    The candidate WAS reachable. If MPFR doesn't find, it's budget exhaustion.");
    } else {
        eprintln!("  ✗ z_target's path is PRUNED at depth {} — that's the bug location.",
            prune_at.unwrap());
    }
}
