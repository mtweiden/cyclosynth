//! Critic-driven oracle test:
//!   (1) Pin failure point: catch prune firing on the path through z_target.
//!   (2) Oracle compare: recompute partial_eucl in MPFR-128 from the same z.
//!   (3) Classify: numerical false-negative vs logical unsoundness vs scale mismatch.

use cyclosynth::matrix::U2Q;
use cyclosynth::synthesis::clifford_sqrt_t::{
    SynthesizerQ, build_l_q, det_phase_of, solution_to_u2q_d,
};
use cyclosynth::synthesis::diag;
use cyclosynth::synthesis::distance::diamond_distance_float;
use cyclosynth::synthesis::lattice_zeta::{
    IntScratch16, build_q_int_zeta, build_q_mpfr_zeta_from_mpfr_v,
    det16_exact, phase1_with_stop_mpfr, set_bypass_norm_prune,
};
use cyclosynth::synthesis::lattice_zeta::cholesky_lu::{
    cholesky_f64_16, lu_solve_int_inplace_16,
};
use cyclosynth::synthesis::lattice_zeta::lll::run_lll_16;
use cyclosynth::synthesis::search_zeta::uv_to_xy_zeta_mpfr;
use num_complex::Complex;
use rug::{Assign, Float as RFloat};
use std::sync::atomic::AtomicBool;

type C64 = Complex<f64>;
type Mat2 = [[C64; 2]; 2];

// ─── MPFR helpers (moved from synthesis::clifford_sqrt_t — used only here) ───

/// MPFR-precision 2×2 complex matrix.
type Mat2Mpfr = [[(rug::Float, rug::Float); 2]; 2];

/// Convert `U2Q` to `Mat2Mpfr`. Lifts ZZeta integer coefficients to MPFR
/// via the basis `(cos(kπ/8), sin(kπ/8))`, k=0..7, then divides by `√2^k`.
fn u2q_to_mat2_mpfr(u: &U2Q, prec: u32) -> Mat2Mpfr {
    use std::f64::consts::PI;
    use cyclosynth::rings::types::int_to_f64;

    let two = RFloat::with_val(prec, 2.0);
    let inv_sqrt2 = RFloat::with_val(prec, 1.0) / two.clone().sqrt();
    let half_k = u.k / 2;
    let mut inv_scale = RFloat::with_val(prec, 1.0);
    inv_scale >>= half_k;
    if u.k % 2 == 1 {
        inv_scale *= &inv_sqrt2;
    }
    let basis: [(RFloat, RFloat); 8] = std::array::from_fn(|k| {
        let theta = (k as f64) * PI / 8.0;
        (RFloat::with_val(prec, theta.cos()), RFloat::with_val(prec, theta.sin()))
    });
    let zzeta_to_re_im = |z: &cyclosynth::rings::ZZeta| -> (RFloat, RFloat) {
        let coeffs = [
            int_to_f64(z.a), int_to_f64(z.b), int_to_f64(z.c), int_to_f64(z.d),
            int_to_f64(z.e), int_to_f64(z.f), int_to_f64(z.g), int_to_f64(z.h),
        ];
        let mut re = RFloat::with_val(prec, 0.0);
        let mut im = RFloat::with_val(prec, 0.0);
        for k in 0..8 {
            let c = RFloat::with_val(prec, coeffs[k]);
            re += RFloat::with_val(prec, &c * &basis[k].0);
            im += RFloat::with_val(prec, &c * &basis[k].1);
        }
        (re * &inv_scale, im * &inv_scale)
    };
    [
        [zzeta_to_re_im(&u.u11), zzeta_to_re_im(&u.u12)],
        [zzeta_to_re_im(&u.u21), zzeta_to_re_im(&u.u22)],
    ]
}

/// MPFR `U_L† · target`.
fn u2q_dag_times_mat2_mpfr(u_l: &U2Q, target: &Mat2Mpfr, prec: u32) -> Mat2Mpfr {
    let u = u2q_to_mat2_mpfr(u_l, prec);
    let ud00 = (u[0][0].0.clone(), RFloat::with_val(prec, -&u[0][0].1));
    let ud01 = (u[1][0].0.clone(), RFloat::with_val(prec, -&u[1][0].1));
    let ud10 = (u[0][1].0.clone(), RFloat::with_val(prec, -&u[0][1].1));
    let ud11 = (u[1][1].0.clone(), RFloat::with_val(prec, -&u[1][1].1));
    let mul = |a: &(RFloat, RFloat), b: &(RFloat, RFloat)| -> (RFloat, RFloat) {
        let re = RFloat::with_val(prec, &a.0 * &b.0)
            - RFloat::with_val(prec, &a.1 * &b.1);
        let im = RFloat::with_val(prec, &a.0 * &b.1)
            + RFloat::with_val(prec, &a.1 * &b.0);
        (RFloat::with_val(prec, re), RFloat::with_val(prec, im))
    };
    let add = |a: (RFloat, RFloat), b: (RFloat, RFloat)| -> (RFloat, RFloat) {
        (RFloat::with_val(prec, &a.0 + &b.0), RFloat::with_val(prec, &a.1 + &b.1))
    };
    [
        [
            add(mul(&ud00, &target[0][0]), mul(&ud01, &target[1][0])),
            add(mul(&ud00, &target[0][1]), mul(&ud01, &target[1][1])),
        ],
        [
            add(mul(&ud10, &target[0][0]), mul(&ud11, &target[1][0])),
            add(mul(&ud10, &target[0][1]), mul(&ud11, &target[1][1])),
        ],
    ]
}

/// Column-1 of an MPFR target as `(Re V₀₀, Im V₀₀, Re V₁₀, Im V₁₀)`.
fn unitary_to_uv_zeta_mpfr(target: &Mat2Mpfr) -> [rug::Float; 4] {
    [
        target[0][0].0.clone(), target[0][0].1.clone(),
        target[1][0].0.clone(), target[1][0].1.clone(),
    ]
}

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

/// Compute partial_eucl_oracle: the SE walk's partial Euclidean norm at depth d,
/// computed in MPFR-128 from the integer Gram, full Cholesky, and integer z.
///
/// partial_eucl(d) = sum_{i ≥ d} (R · z)[i]² where R is the upper-triangular
/// Cholesky factor of B B^T.
fn partial_eucl_mpfr(basis: &[[i64; 16]; 16], z: &[i64; 16], depth_set: usize) -> RFloat {
    use rug::Float;
    const PREC: u32 = 192;

    // Compute G = B B^T in i128.
    let mut gram = [[0_i128; 16]; 16];
    for i in 0..16 {
        for j in 0..16 {
            let mut s = 0_i128;
            for k in 0..16 {
                s += (basis[i][k] as i128) * (basis[j][k] as i128);
            }
            gram[i][j] = s;
        }
    }

    // Lift to MPFR.
    let mut g: [[Float; 16]; 16] = std::array::from_fn(|_| std::array::from_fn(|_| Float::with_val(PREC, 0.0)));
    for i in 0..16 {
        for j in 0..16 {
            // i128 → MPFR via two-limb: hi*2^64 + lo.
            let v = gram[i][j];
            let neg = v < 0;
            let abs = if neg { -v } else { v } as u128;
            let hi = (abs >> 64) as u64;
            let lo = abs as u64;
            let mut f = Float::with_val(PREC, hi);
            f <<= 64u32;
            f += Float::with_val(PREC, lo);
            g[i][j] = if neg { -f } else { f };
        }
    }

    // Cholesky G = L L^T (L lower triangular, MPFR).
    let mut l: [[Float; 16]; 16] = std::array::from_fn(|_| std::array::from_fn(|_| Float::with_val(PREC, 0.0)));
    for i in 0..16 {
        for j in 0..=i {
            let mut s = g[i][j].clone();
            for k in 0..j {
                let prod = Float::with_val(PREC, &l[i][k] * &l[j][k]);
                s -= &prod;
            }
            if i == j {
                if s.is_zero() || s.is_sign_negative() {
                    panic!("MPFR Cholesky failed: G not PD at i={i}");
                }
                l[i][i] = s.sqrt();
            } else {
                let q = Float::with_val(PREC, &s / &l[j][j]);
                l[i][j] = q;
            }
        }
    }

    // R = L^T (upper-triangular). Compute (R · z)[i] = sum_{j>=i} R[i][j] · z[j]
    //                                                = sum_{j>=i} L[j][i] · z[j].
    // partial_eucl_oracle = sum_{i = depth_set..15} (R z)[i]²
    // (depth_set is the depth at which prune fires; z[depth_set..15] are set).
    let mut total = Float::with_val(PREC, 0.0);
    for i in depth_set..16 {
        let mut row = Float::with_val(PREC, 0.0);
        for j in i..16 {
            // R[i][j] = L[j][i]
            let zj = Float::with_val(PREC, z[j]);
            row += Float::with_val(PREC, &l[j][i] * &zj);
        }
        total += Float::with_val(PREC, &row * &row);
    }
    total
}

// ─── Inline double-double primitives (~106-bit precision) ────────────────────
//
// Used by `partial_eucl_dd_scratch` to evaluate the same partial-eucl quantity
// at ~32-decimal-digit precision without the rug allocation cost. If dd is
// sufficient on the failure instance, Option (B) qd-everywhere can be built
// on these primitives.

#[inline]
fn quick_two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let err = b - (s - a);
    (s, err)
}

#[inline]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    let err = (a - (s - bb)) + (b - bb);
    (s, err)
}

#[inline]
fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    let err = a.mul_add(b, -p);
    (p, err)
}

type DD = (f64, f64);

#[inline]
fn dd_add(a: DD, b: DD) -> DD {
    let (s, e) = two_sum(a.0, b.0);
    let e = e + a.1 + b.1;
    quick_two_sum(s, e)
}

#[inline]
fn dd_sub(a: DD, b: DD) -> DD { dd_add(a, (-b.0, -b.1)) }

#[inline]
fn dd_mul(a: DD, b: DD) -> DD {
    let (p, e) = two_prod(a.0, b.0);
    let e = e + a.0 * b.1 + a.1 * b.0;
    quick_two_sum(p, e)
}

#[inline]
fn dd_from_f64(a: f64) -> DD { (a, 0.0) }

#[inline]
fn dd_to_f64(a: DD) -> f64 { a.0 + a.1 }

/// Reciprocal in dd: one Newton step from a f64 initial guess gives
/// ~106-bit accuracy. r' = r · (2 − b·r).
#[inline]
fn dd_recip(b: DD) -> DD {
    let r0 = 1.0 / b.0;
    let r0_dd = dd_from_f64(r0);
    let bp = dd_mul(b, r0_dd);
    let two = (2.0_f64, 0.0_f64);
    let two_minus_bp = dd_sub(two, bp);
    dd_mul(r0_dd, two_minus_bp)
}

#[inline]
fn dd_div(a: DD, b: DD) -> DD { dd_mul(a, dd_recip(b)) }

/// Square root in dd. One Newton step from f64 guess: x_new = x + (s − x²)/(2x).
/// The (s − x²) is computed in dd; the divide uses dd_recip for ~106-bit accuracy.
#[inline]
fn dd_sqrt(s: DD) -> DD {
    if s.0 <= 0.0 { return (0.0, 0.0); }
    let x = s.0.sqrt();
    let x_dd = dd_from_f64(x);
    let x_sq = dd_mul(x_dd, x_dd);
    let resid = dd_sub(s, x_sq);
    let two_x = dd_add(x_dd, x_dd);
    let corr = dd_div(resid, two_x);
    dd_add(x_dd, corr)
}

/// Same partial_eucl quantity as MPFR oracle but computed in inline dd:
/// integer Gram → dd Cholesky → dd dot product → dd squared sum.
fn partial_eucl_dd_scratch(basis: &[[i64; 16]; 16], z: &[i64; 16], depth_set: usize) -> DD {
    // Integer Gram in i128 (exact for these magnitudes).
    let mut gram_i = [[0_i128; 16]; 16];
    for i in 0..16 {
        for j in 0..16 {
            let mut s: i128 = 0;
            for k in 0..16 {
                s += (basis[i][k] as i128) * (basis[j][k] as i128);
            }
            gram_i[i][j] = s;
        }
    }
    // Lift to dd: i128 = hi·2^64 + lo. Both halves fit in f64 exactly when
    // ≤ 2^53; we exploit dd's 106-bit range via split.
    let i128_to_dd = |v: i128| -> DD {
        if v == 0 { return (0.0, 0.0); }
        let neg = v < 0;
        let abs = if neg { -v } else { v } as u128;
        let hi = (abs >> 64) as u64;
        let lo = abs as u64;
        let hi_f = hi as f64;
        let lo_f = lo as f64;
        // value = hi_f · 2^64 + lo_f
        let two64 = (1u128 << 63) as f64 * 2.0;
        let p = dd_mul(dd_from_f64(hi_f), dd_from_f64(two64));
        let r = dd_add(p, dd_from_f64(lo_f));
        if neg { (-r.0, -r.1) } else { r }
    };

    let mut g: [[DD; 16]; 16] = [[(0.0, 0.0); 16]; 16];
    for i in 0..16 {
        for j in 0..16 {
            g[i][j] = i128_to_dd(gram_i[i][j]);
        }
    }
    // dd Cholesky: G = L L^T, L lower-triangular.
    let mut l: [[DD; 16]; 16] = [[(0.0, 0.0); 16]; 16];
    for i in 0..16 {
        for j in 0..=i {
            let mut s = g[i][j];
            for k in 0..j {
                let prod = dd_mul(l[i][k], l[j][k]);
                s = dd_sub(s, prod);
            }
            if i == j {
                if s.0 <= 0.0 { return (f64::INFINITY, 0.0); }
                l[i][i] = dd_sqrt(s);
            } else {
                l[i][j] = dd_div(s, l[j][j]);
            }
        }
    }
    // R = L^T (upper-triangular). (R z)[i] = Σ_{j ≥ i} L[j][i] · z[j].
    // partial = Σ_{i ≥ depth_set} (R z)[i]².
    let mut total: DD = (0.0, 0.0);
    for i in depth_set..16 {
        let mut row: DD = (0.0, 0.0);
        for j in i..16 {
            // z[j] · L[j][i], lifted to dd. z[j] is i64; convert to dd.
            let zj_dd = {
                let v = z[j];
                if v.unsigned_abs() <= (1u64 << 53) {
                    dd_from_f64(v as f64)
                } else {
                    // Two-piece: z = hi·2^32 + lo, both fit in f64.
                    let neg = v < 0;
                    let abs = v.unsigned_abs();
                    let hi = (abs >> 32) as u32 as f64;
                    let lo = (abs & 0xFFFFFFFF) as u32 as f64;
                    let two32 = (1u64 << 32) as f64;
                    let p = dd_mul(dd_from_f64(hi), dd_from_f64(two32));
                    let r = dd_add(p, dd_from_f64(lo));
                    if neg { (-r.0, -r.1) } else { r }
                }
            };
            let term = dd_mul(l[j][i], zj_dd);
            row = dd_add(row, term);
        }
        let sq = dd_mul(row, row);
        total = dd_add(total, sq);
    }
    total
}

/// Sanity check: identical algorithm to partial_eucl_dd_scratch but using
/// rug::Float at 106-bit precision. If this matches the MPFR-192 oracle,
/// dd's 106-bit precision is theoretically sufficient and any disagreement
/// with MPFR-192 from `partial_eucl_dd_scratch` is an implementation bug
/// in our inline dd primitives, not a fundamental precision deficit.
fn partial_eucl_rug106(basis: &[[i64; 16]; 16], z: &[i64; 16], depth_set: usize) -> rug::Float {
    use rug::Float;
    const PREC: u32 = 106;
    let mut gram = [[0_i128; 16]; 16];
    for i in 0..16 {
        for j in 0..16 {
            let mut s = 0_i128;
            for k in 0..16 {
                s += (basis[i][k] as i128) * (basis[j][k] as i128);
            }
            gram[i][j] = s;
        }
    }
    let lift = |v: i128| -> Float {
        let neg = v < 0;
        let abs = if neg { -v } else { v } as u128;
        let hi = (abs >> 64) as u64;
        let lo = abs as u64;
        let mut f = Float::with_val(PREC, hi);
        f <<= 64u32;
        f += Float::with_val(PREC, lo);
        if neg { -f } else { f }
    };
    let mut g: [[Float; 16]; 16] = std::array::from_fn(|_| std::array::from_fn(|_| Float::with_val(PREC, 0.0)));
    for i in 0..16 {
        for j in 0..16 { g[i][j] = lift(gram[i][j]); }
    }
    let mut l: [[Float; 16]; 16] = std::array::from_fn(|_| std::array::from_fn(|_| Float::with_val(PREC, 0.0)));
    for i in 0..16 {
        for j in 0..=i {
            let mut s = g[i][j].clone();
            for k in 0..j {
                let prod = Float::with_val(PREC, &l[i][k] * &l[j][k]);
                s -= &prod;
            }
            if i == j {
                l[i][i] = s.sqrt();
            } else {
                l[i][j] = Float::with_val(PREC, &s / &l[j][j]);
            }
        }
    }
    let mut total = Float::with_val(PREC, 0.0);
    for i in depth_set..16 {
        let mut row = Float::with_val(PREC, 0.0);
        for j in i..16 {
            let zj = Float::with_val(PREC, z[j]);
            row += Float::with_val(PREC, &l[j][i] * &zj);
        }
        total += Float::with_val(PREC, &row * &row);
    }
    total
}

/// Scratch f64 recompute: compute the same partial_eucl quantity in f64 from
/// scratch — fresh integer Gram, f64 Cholesky, f64 dot products. This is what
/// Option (1) “recompute on prune-fire” would produce. Used to test whether
/// a from-scratch f64 partial avoids the false-negative.
fn partial_eucl_f64_scratch(basis: &[[i64; 16]; 16], z: &[i64; 16], depth_set: usize) -> f64 {
    let mut gram = [[0.0_f64; 16]; 16];
    for i in 0..16 {
        for j in 0..16 {
            let mut s = 0_i128;
            for k in 0..16 {
                s += (basis[i][k] as i128) * (basis[j][k] as i128);
            }
            gram[i][j] = s as f64;
        }
    }
    let mut l = [[0.0_f64; 16]; 16];
    for i in 0..16 {
        for j in 0..=i {
            let mut s = gram[i][j];
            for k in 0..j {
                s -= l[i][k] * l[j][k];
            }
            if i == j {
                if s <= 0.0 { return f64::INFINITY; }
                l[i][i] = s.sqrt();
            } else {
                l[i][j] = s / l[j][j];
            }
        }
    }
    // R[i][j] = L[j][i]; (R·z)[i] = sum_{j ≥ i} L[j][i] · z[j].
    let mut total = 0.0_f64;
    for i in depth_set..16 {
        let mut row = 0.0_f64;
        for j in i..16 {
            row += l[j][i] * (z[j] as f64);
        }
        total += row * row;
    }
    total
}

fn main() {
    std::env::set_var("CYCLOSYNTH_CAPTURE", "1");
    let theta = 0.3_f64;
    let target = rz_f64(theta);
    let prec: u32 = 192;
    let theta_mpfr = RFloat::with_val(prec, theta);
    let target_mpfr = rz_mpfr(&theta_mpfr, prec);
    let eps = 1.5e-8_f64;

    eprintln!("=== Phase 1: capture x_target via f64 synthesize (bypass=on) ===");
    set_bypass_norm_prune(true); // capture must succeed despite the f64 prune false-negative
    let synth = SynthesizerQ::new(eps).with_optimize_cost(false).with_max_lde(35);
    let r = synth.synthesize(target).expect("expected to find");
    let cap = diag::CAPTURED_FIND.lock().unwrap().clone().expect("capture must fire");
    eprintln!("  k_total={}, k_inner={}, d_r={}, d_l={}",
        cap.k_total, cap.k_inner, cap.d_r, cap.d_l);

    let u_r = solution_to_u2q_d(&cap.x_inner, cap.k_inner, cap.d_r);
    let prefixes = build_l_q(2);
    let u_l = find_u_l(&prefixes, cap.d_l, u_r, &target, r.distance)
        .expect("U_L must be found");

    eprintln!("\n=== Phase 1.5: build MPFR pipeline at SAME (k_inner, v_inner_mpfr) ===");
    let m_inner_mpfr = u2q_dag_times_mat2_mpfr(&u_l, &target_mpfr, prec);
    let v_inner_mpfr = unitary_to_uv_zeta_mpfr(&m_inner_mpfr);
    let y_inner_mpfr = uv_to_xy_zeta_mpfr(&v_inner_mpfr, cap.k_inner, prec);

    let mut scratch = IntScratch16::new(eps);
    scratch.reset_basis();
    build_q_mpfr_zeta_from_mpfr_v(&mut scratch, &v_inner_mpfr, cap.k_inner, eps);
    build_q_int_zeta(&mut scratch);
    let lll_result = run_lll_16(&mut scratch);
    eprintln!("  LLL: {:?}, det(B)={:?}", lll_result, det16_exact(&scratch.basis));

    // Step 1 (critic Q1): measure i256 Gram G = B B^T magnitude vs qd::Double's
    // 2^106 integer-exact range. We use i128 here as a proxy (max basis entry
    // ~2^50 → max term ~2^100 → 16-fold sum ~2^104). If max bitlength fits in
    // i128 and ≤ 106, qd conversion is exact for this regime.
    {
        let mut max_abs: i128 = 0;
        let mut overflowed_i128 = false;
        for i in 0..16 {
            for j in 0..16 {
                let mut s: i128 = 0;
                for k in 0..16 {
                    let prod = (scratch.basis[i][k] as i128).checked_mul(scratch.basis[j][k] as i128);
                    match prod.and_then(|p| s.checked_add(p)) {
                        Some(v) => s = v,
                        None => { overflowed_i128 = true; break; }
                    }
                }
                let abs = s.unsigned_abs();
                if abs as i128 > max_abs { max_abs = abs as i128; }
            }
            if overflowed_i128 { break; }
        }
        let bitlen = if max_abs == 0 { 0 } else { 128 - (max_abs as u128).leading_zeros() };
        eprintln!("  G_ij magnitude (cliff lde={}): max |G_ij| ≈ 2^{}, i128 overflow: {}",
            cap.k_inner, bitlen, overflowed_i128);
        if overflowed_i128 {
            eprintln!("  → G exceeds i128. Need i256 path for exact representation.");
        } else if bitlen <= 106 {
            eprintln!("  → G_ij fits in qd::Double's 2^106 integer-exact range. Conversion is exact.");
        } else {
            eprintln!("  → G_ij exceeds qd's 2^106 exact range by {} bits. qd Cholesky has rounded input; audit decides safety.",
                bitlen - 106);
        }
    }

    // Cap mid + c[i].
    let prec_q = scratch.prec_q;
    let one = RFloat::with_val(prec_q, 1.0);
    let two = RFloat::with_val(prec_q, 2.0);
    let eps_rf = RFloat::with_val(prec_q, eps);
    let eps_sq = RFloat::with_val(prec_q, &eps_rf * &eps_rf);
    let one_minus_eps_sq = RFloat::with_val(prec_q, &one - &eps_sq);
    let sqrt_1m = one_minus_eps_sq.sqrt();
    let cap_mid = RFloat::with_val(prec_q, RFloat::with_val(prec_q, &one + &sqrt_1m) / &two);
    for i in 0..16 {
        scratch.c[i].assign(RFloat::with_val(prec_q, &y_inner_mpfr[i] * &cap_mid));
    }
    if !cholesky_f64_16(&mut scratch) || !lu_solve_int_inplace_16(&mut scratch) {
        eprintln!("setup failed"); return;
    }

    // z_target_mpfr = (B^T_mpfr)^-1 · x_target.
    for i in 0..16 {
        scratch.c[i].assign(rug::Float::with_val(scratch.lu_prec, cap.x_inner[i] as f64));
    }
    if !lu_solve_int_inplace_16(&mut scratch) { eprintln!("lu_solve(x) FAIL"); return; }
    let z_target_mpfr: [i64; 16] = std::array::from_fn(|i| {
        let mut rounded = scratch.lu_x[i].clone();
        rounded.round_mut();
        rounded.to_integer().map(|n| n.to_i64_wrapping()).unwrap_or(0)
    });

    // Sanity: B · z_target_mpfr = x_target?
    let mut x_check = [0i64; 16];
    for i in 0..16 {
        for j in 0..16 {
            x_check[i] = x_check[i].wrapping_add(scratch.basis[j][i].wrapping_mul(z_target_mpfr[j]));
        }
    }
    if x_check != cap.x_inner {
        eprintln!("  WARNING: B · z_target_mpfr ≠ x_target — z_target_mpfr is wrong");
        return;
    }
    eprintln!("  z_target_mpfr verified (B · z = x_target)");

    eprintln!("\n=== Phase 2-3: turn bypass OFF, arm watchdog + sampling, run MPFR phase1 ===");
    set_bypass_norm_prune(false);
    diag::WATCH_HITS.lock().unwrap().clear();
    diag::watch_arm(z_target_mpfr);
    diag::arm_sampling();

    // Re-build clean scratch and run phase1_with_stop_mpfr.
    let mut scratch2 = IntScratch16::new(eps);
    scratch2.reset_basis();
    let budget_hit = AtomicBool::new(false);
    let target_for_check = target;
    let d_r = cap.d_r;
    let k_inner = cap.k_inner;
    let _u_l_for_check = u_l;
    let should_stop = |x: &[i64; 16]| -> bool {
        let cand = solution_to_u2q_d(x, k_inner, d_r);
        let u_full = u_l * cand;
        cyclosynth::synthesis::distance::diamond_distance_u2q_float(&u_full, &target_for_check) < eps
    };
    let _sols = phase1_with_stop_mpfr(
        &mut scratch2, &y_inner_mpfr, &v_inner_mpfr, k_inner, eps,
        100_000_000, &budget_hit, should_stop, None, None,
    );
    eprintln!("  phase1_with_stop_mpfr done. budget_hit={}", budget_hit.load(std::sync::atomic::Ordering::Relaxed));
    eprintln!("  sols.len() = {}", _sols.len());

    let hits = diag::WATCH_HITS.lock().unwrap().clone();
    eprintln!("\n=== Watchdog firings on z_target_mpfr's path: {} ===", hits.len());
    if hits.is_empty() {
        eprintln!("  No prune fired on z_target_mpfr's path.");
        eprintln!("  Either: (a) the SE walk found a candidate before reaching this depth, or");
        eprintln!("          (b) the path was pruned for a different reason (Q-bound, bracket).");
        eprintln!("  Need wider instrumentation (Q-bracket watchdog) to fully classify.");
        return;
    }

    eprintln!("\n=== Oracle comparison: f64-incr vs f64-scratch vs DD vs rug-106 vs MPFR-192 ===");
    eprintln!("  Threshold T = 2^k_inner · (1 + 1e-9) = {:.6e}", hits[0].threshold);
    eprintln!();
    eprintln!("  depth | f64 incr       | f64 scratch    | dd scratch     | rug-106        | MPFR-192       | T              | dd>T | r106>T | mpfr>T");
    eprintln!("  ──────┼────────────────┼────────────────┼────────────────┼────────────────┼────────────────┼────────────────┼──────┼────────┼───────");
    for (i, hit) in hits.iter().enumerate().take(8) {
        let mpfr_partial = partial_eucl_mpfr(&scratch2.basis, &hit.z_at_prune, hit.depth as usize);
        let mpfr_partial_f = mpfr_partial.to_f64();
        let scratch_f64 = partial_eucl_f64_scratch(&scratch2.basis, &hit.z_at_prune, hit.depth as usize);
        let dd_p = partial_eucl_dd_scratch(&scratch2.basis, &hit.z_at_prune, hit.depth as usize);
        let dd_f = dd_to_f64(dd_p);
        let r106 = partial_eucl_rug106(&scratch2.basis, &hit.z_at_prune, hit.depth as usize).to_f64();
        let dd_over = dd_f > hit.threshold;
        let r106_over = r106 > hit.threshold;
        let mpfr_over = mpfr_partial_f > hit.threshold;
        eprintln!(
            "  {:>5} | {:>14.6e} | {:>14.6e} | {:>14.6e} | {:>14.6e} | {:>14.6e} | {:>14.6e} | {:>4} | {:>6} | {:>6}",
            hit.depth, hit.partial_eucl_f64, scratch_f64, dd_f, r106, mpfr_partial_f, hit.threshold,
            dd_over, r106_over, mpfr_over,
        );
        if i == 0 {
            eprintln!("    z[d..16] (the path): {:?}", &hit.z_at_prune[hit.depth as usize..]);
            // Decompose: print per-element |R[i][:] · z| and |R z|² for context.
            eprintln!("    per-row decomposition (scratch f64):");
            // Recompute Gram + Cholesky locally to get individual rows.
            let mut gram = [[0.0_f64; 16]; 16];
            for ii in 0..16 {
                for jj in 0..16 {
                    let mut s = 0_i128;
                    for k in 0..16 {
                        s += (scratch2.basis[ii][k] as i128) * (scratch2.basis[jj][k] as i128);
                    }
                    gram[ii][jj] = s as f64;
                }
            }
            let mut ll = [[0.0_f64; 16]; 16];
            for ii in 0..16 {
                for jj in 0..=ii {
                    let mut s = gram[ii][jj];
                    for k in 0..jj { s -= ll[ii][k] * ll[jj][k]; }
                    if ii == jj { ll[ii][ii] = s.sqrt(); } else { ll[ii][jj] = s / ll[jj][jj]; }
                }
            }
            for ii in (hit.depth as usize)..(hit.depth as usize + 4).min(16) {
                let mut row = 0.0_f64;
                let mut max_term = 0.0_f64;
                for jj in ii..16 {
                    let term = ll[jj][ii] * (hit.z_at_prune[jj] as f64);
                    row += term;
                    if term.abs() > max_term { max_term = term.abs(); }
                }
                eprintln!("      row[{ii}]: (Rz)[i] = {:>12.4e}, |max term| = {:>12.4e}, cancel ratio = {:>5.2}× ULP",
                    row, max_term, max_term.abs() / row.abs() / f64::EPSILON,
                );
            }
        }
    }
    if hits.len() > 8 {
        eprintln!("  ... ({} more firings)", hits.len() - 8);
    }

    // Q4: prune-firing frequency stats.
    use std::sync::atomic::Ordering as AOrdering;
    let total_fires = diag::N_PRUNE_FIRES.load(AOrdering::Relaxed);
    let near_fires = diag::N_PRUNE_FIRES_NEAR.load(AOrdering::Relaxed);
    let very_near = diag::N_PRUNE_FIRES_VERY_NEAR.load(AOrdering::Relaxed);
    eprintln!("\n=== Q4: prune-firing frequency at ε=1.5e-8 ===");
    eprintln!("  total prune firings:           {:>12}", total_fires);
    if total_fires > 0 {
        eprintln!("  within 10% of threshold (≤1.10): {:>10} ({:>5.1}%)",
            near_fires, 100.0 * near_fires as f64 / total_fires as f64);
        eprintln!("  within  1% of threshold (≤1.01): {:>10} ({:>5.1}%)",
            very_near, 100.0 * very_near as f64 / total_fires as f64);
    }

    // Classify based on the table.
    eprintln!("\n=== Classification ===");
    let scratch_fixes = hits.iter().all(|h| {
        let s = partial_eucl_f64_scratch(&scratch2.basis, &h.z_at_prune, h.depth as usize);
        s <= h.threshold
    });
    let mpfr_consistent = hits.iter().all(|h| {
        let m = partial_eucl_mpfr(&scratch2.basis, &h.z_at_prune, h.depth as usize).to_f64();
        m <= h.threshold
    });
    eprintln!("  MPFR consistently says these paths should NOT prune: {}", mpfr_consistent);
    eprintln!("  Scratch-f64 consistently says these paths should NOT prune: {}", scratch_fixes);
    if mpfr_consistent && scratch_fixes {
        eprintln!("  → Option (1) recompute-in-f64 IS sufficient for these instances.");
        eprintln!("  → Dominant error is accumulation drift in incremental w[d], NOT intrinsic cancellation.");
    } else if mpfr_consistent && !scratch_fixes {
        eprintln!("  → Option (1) is INSUFFICIENT — scratch-f64 still false-negatives.");
        eprintln!("  → Intrinsic cancellation in the f64 dot product. Use Option (3) qd::Double.");
    } else {
        eprintln!("  → MPFR also disagrees with itself, or threshold mismatch. Investigate.");
    }

    // ─── Critic Step 1: oracle audit of false-negative ratio tail ─────────────
    let samples = diag::collect_all_samples();
    eprintln!("\n=== Step 1: oracle audit of false-negative tail ===");
    eprintln!("  Stratified samples collected (1000 per bin, 5 bins): {}", samples.len());
    if samples.is_empty() {
        eprintln!("  No samples — nothing to audit.");
        return;
    }

    // For each sample, recompute the oracle MPFR partial. Classify:
    //   FN (false-negative): f64 says prune (already ratio > 1) AND MPFR ≤ T
    //   TN (true-positive prune): f64 says prune AND MPFR > T
    let mut fn_ratios: Vec<f64> = Vec::new();   // ratio = f64_partial / T (false negs)
    let mut tn_ratios: Vec<f64> = Vec::new();   // ratio for true-positive prunes
    let mut fn_per_bin: [u64; 5] = [0; 5];
    let mut tn_per_bin: [u64; 5] = [0; 5];

    let bin_label = |r: f64| -> usize {
        if r < 1.05 { 0 }
        else if r < 1.5 { 1 }
        else if r < 2.0 { 2 }
        else if r < 5.0 { 3 }
        else { 4 }
    };

    for s in &samples {
        let mpfr_p = partial_eucl_mpfr(&scratch2.basis, &s.z, s.depth as usize).to_f64();
        let r = s.f64_partial / s.threshold;
        let bin = bin_label(r);
        if mpfr_p <= s.threshold {
            fn_ratios.push(r);
            fn_per_bin[bin] += 1;
        } else {
            tn_ratios.push(r);
            tn_per_bin[bin] += 1;
        }
    }

    eprintln!("  Audit results (n={} samples):", samples.len());
    eprintln!("    True-positive prunes (MPFR>T):     {:>6} ({:>5.1}%)",
        tn_ratios.len(), 100.0 * tn_ratios.len() as f64 / samples.len() as f64);
    eprintln!("    False-negative prunes (MPFR≤T):    {:>6} ({:>5.1}%)",
        fn_ratios.len(), 100.0 * fn_ratios.len() as f64 / samples.len() as f64);
    eprintln!();
    eprintln!("  Distribution of f64_partial/T across the 5 bins:");
    eprintln!("    bin            range          | FN count | TN count");
    eprintln!("    ──────────────────────────────┼──────────┼─────────");
    let bin_ranges = ["[1.00, 1.05)", "[1.05, 1.50)", "[1.50, 2.00)", "[2.00, 5.00)", "[5.00,   ∞)"];
    for i in 0..5 {
        eprintln!("    bin {} {:14}    | {:>8} | {:>8}", i, bin_ranges[i], fn_per_bin[i], tn_per_bin[i]);
    }

    if !fn_ratios.is_empty() {
        fn_ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let max_fn = *fn_ratios.last().unwrap();
        let p99 = fn_ratios[fn_ratios.len() * 99 / 100];
        let p90 = fn_ratios[fn_ratios.len() * 9 / 10];
        let p50 = fn_ratios[fn_ratios.len() / 2];
        eprintln!();
        eprintln!("  False-negative tail of f64_partial/T:");
        eprintln!("    median:       {:.4}", p50);
        eprintln!("    p90:          {:.4}", p90);
        eprintln!("    p99:          {:.4}", p99);
        eprintln!("    MAX observed: {:.4}", max_fn);
        eprintln!();
        if max_fn < 2.0 {
            eprintln!("  → Guard `f64_partial ≤ 2T` would catch ALL observed false negatives.");
            eprintln!("  → The 2T guard is empirically validated for this target.");
        } else if max_fn < 5.0 {
            eprintln!("  → Guard `f64_partial ≤ 5T` needed (max observed = {:.2}).", max_fn);
            eprintln!("  → 2T is INSUFFICIENT — would leave false negatives at ratio in [2, {:.2}].", max_fn);
        } else {
            eprintln!("  → MAX observed f64/T = {:.2} ≥ 5. Bounded guards are not safe.", max_fn);
            eprintln!("  → Must always-recompute in qd::Double on prune-fire (no f64 guard).");
        }
    } else {
        eprintln!("  → No false negatives in this sample. The cliff failure may be concentrated in");
        eprintln!("    a small region of the search tree not hit by stratified random sampling.");
    }
}
