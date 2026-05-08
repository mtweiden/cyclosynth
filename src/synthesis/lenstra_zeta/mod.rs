//! Native 16D Lenstra-style search for Clifford+√T (Z[ζ_16]) synthesis.
//!
//! This module is the Z[ζ_16] analog of [`super::lenstra`] (which targets
//! Z[ω] / Clifford+T). The two modules are deliberately kept separate to
//! isolate the precision and integer-width choices: f64 Gram-Schmidt is
//! provably sufficient at d=8 (Theorem 2 of Nguyen-Stehlé 2009) but not at
//! d=16, so the 16D path uses MPFR throughout.
//!
//! ## Pipeline (per phase1 call)
//!
//!   1. **Build Q** in MPFR ([`q_metric::build_q_mpfr_zeta`]) — anisotropic
//!      ellipsoid metric in lattice coordinates. Cap radial (1 dim) +
//!      cap tangential (3 dims) + bullet balls (12 dims).
//!   2. **Snapshot Q to i256** ([`q_metric::build_q_int_zeta`]) with adaptive
//!      scale `S = 2^B` chosen so `max(|S·Q|) ≈ 2^TARGET_BITS`.
//!   3. **L²-LLL** ([`lll::run_lll_16`]): MPFR Gram-Schmidt on the exact
//!      i256 Gram (mandatory at d=16; f64 GS is only proved sufficient at
//!      d ≤ 11). INSERT semantics + lazy size-reduction maintain the
//!      L³-reduced invariant.
//!   4. **Schnorr-Euchner** ([`se::schnorr_euchner_16d`]): walk candidate
//!      `z` values within the SE ellipsoid; reconstruct `x = B·z` and
//!      validate against the synthesis constraints (norm shell, 3 bilinear
//!      forms `B_1, B_2, B_3 = 0`, alignment cap).
//!
//! ## Module layout (mirrors [`super::lenstra`])
//!
//! - [`cholesky_lu`] — Cholesky / LU helpers for d=16.
//! - [`integer`] — phase1 driver wiring all stages.
//! - [`lll`] — 16D L²-LLL with MPFR Gram-Schmidt.
//! - [`q_metric`] — MPFR Q-metric + i256 snapshot.
//! - [`scratch`] — `IntScratch16` per-thread buffers.
//! - [`se`] — Schnorr-Euchner walk + bilinear leaf checks.
//!
//! Brute-force enumeration (`phase1_brute`) and y-vector helpers
//! (`compute_align_vec_zeta`, `uv_to_xy_zeta`) live in
//! [`super::search_zeta`], parallel to [`super::search`] for Z[ω].
//! `U2Q` reconstruction (`solution_to_u2q*`, `unitary_to_uv_zeta`,
//! `det_phase_of`) lives in [`super::clifford_sqrt_t`], parallel to
//! [`super::clifford_t`] for Z[ω].
//!
//! ## Solution layout
//!
//!   `sol = [u_1.a, u_1.b, …, u_1.h, u_2.a, …, u_2.h]`
//!     i.e. `sol[0..8]` = u_1's ζ-basis coefficients,
//!          `sol[8..16]` = u_2's ζ-basis coefficients.
//!
//! Reconstruction follows the SU(2) convention used by Z[ω]'s
//! `solution_to_u2t`:
//!
//!   `U = [[u_1, −u_2*], [u_2, u_1*]] / √(2^k)`

pub mod bkz;
pub mod cholesky_lu;
pub mod integer;
pub mod lll;
pub mod lll_f64;
pub mod q_metric;
pub mod scratch;
pub mod se;

pub use cholesky_lu::{
    cholesky_f64_16, cholesky_int_16, lu_solve_int_inplace_16,
    snapshot_gram_to_mpfr_16,
};
pub use integer::{phase1, phase1_with_stop, phase1_with_stop_mpfr};
pub use lll::{lll_l2_16, run_lll_16, LllResult};
pub use q_metric::{build_q_int_zeta, build_q_mpfr_zeta, build_q_mpfr_zeta_from_mpfr_v};
pub use scratch::IntScratch16;
pub use se::{
    beta_1, beta_2, beta_3, bilinear_forms, det16_exact, euclidean_cholesky_16,
    reconstruct_x, schnorr_euchner_16d,
};

// ─── Tests preserving the previous flat-module test suite ────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthesis::decomposer::BlochDecomposer;
    use crate::synthesis::distance::Mat2;
    use crate::synthesis::search_zeta::{
        compute_align_vec_zeta, phase1_brute, uv_to_xy_zeta,
    };
    use crate::synthesis::clifford_sqrt_t::{
        det_phase_of, solution_to_u2q, solution_to_u2q_d,
    };
    use super::q_metric::build_q_zzeta_lattice;
    use crate::rings::ZZeta;
    use num_complex::Complex64;
    use std::f64::consts::PI;

    #[test]
    fn phase1_brute_at_k_2_finds_solutions() {
        // At k=2 (norm² = 4), there should be 2848 solutions per Phase 3 data.
        let sols = phase1_brute(2);
        assert_eq!(sols.len(), 2848, "expected 2848 valid solutions at k=2");
    }

    #[test]
    fn phase1_brute_at_k_3_finds_solutions() {
        let sols = phase1_brute(3);
        assert_eq!(sols.len(), 54048, "expected 54048 valid solutions at k=3");
    }

    #[test]
    fn solution_to_u2q_well_formed() {
        let sol = [1, 0, 0, 0, 0, 0, 0, 0,    // u_1 = 1
                   0, 1, 0, 0, 0, 0, 0, 0];   // u_2 = ζ
        let u = solution_to_u2q(&sol, 1);
        assert_eq!(u.u11, ZZeta::ONE);
        assert_eq!(u.u21, ZZeta::ZETA);
        assert_eq!(u.u22, ZZeta::ONE);
        assert_eq!(u.k, 1);
    }

    #[test]
    fn brute_finds_t_at_k_0() {
        use crate::synthesis::distance::diamond_distance_float;

        let one = Complex64::new(1.0, 0.0);
        let omega = Complex64::from_polar(1.0, PI / 4.0);
        let zero = Complex64::new(0.0, 0.0);
        let target = [[one, zero], [zero, omega]];

        let sols = phase1_brute(0);
        let mut min_dist = f64::INFINITY;
        for sol in &sols {
            let u = solution_to_u2q(sol, 0);
            let mat = u.to_float();
            let d = diamond_distance_float(&mat, &target);
            if d < min_dist { min_dist = d; }
        }
        assert!(min_dist < 1e-9, "min distance to T at k=0: {min_dist}");
    }

    fn det_phase_of_circuit(circuit: &str) -> i32 {
        let mut p = 0i32;
        for c in circuit.chars() {
            p += match c {
                'H' | 'X' | 'Z' => 8,
                'S' => 4,
                'T' => 2,
                'Q' => 1,
                'Y' => 0,
                _ => 0,
            };
        }
        p
    }

    #[test]
    fn brute_finds_qhq_at_k_1() {
        use crate::matrix::u2::U2Q;
        use crate::synthesis::distance::diamond_distance_float;
        let qhq: U2Q = U2Q::q() * U2Q::h() * U2Q::q();
        let target = qhq.to_float();
        assert_eq!(qhq.k, 1, "QHQ should have k=1");
        let sols = phase1_brute(1);
        let mut min_dist = f64::INFINITY;
        for sol in &sols {
            let u = solution_to_u2q(sol, 1);
            let d = diamond_distance_float(&u.to_float(), &target);
            if d < min_dist { min_dist = d; }
        }
        assert!(min_dist < 1e-9, "min dist to QHQ at k=1: {min_dist:.3e}");
    }

    #[test]
    fn brute_finds_random_circuits_with_even_q_count() {
        use rand::Rng;
        use crate::matrix::u2::U2Q;
        use crate::synthesis::distance::diamond_distance_float;
        let mut rng = rand::rng();
        let mut tested = 0;
        for _ in 0..20 {
            let n = rng.random_range(2..=5);
            let circuit: String = (0..n).map(|_| {
                ['H', 'S', 'Q'][rng.random_range(0..3)]
            }).collect();
            if det_phase_of_circuit(&circuit) % 2 != 0 { continue; }
            let mut u = U2Q::eye();
            for c in circuit.chars() {
                u = u * match c {
                    'H' => U2Q::h(),
                    'S' => U2Q::s(),
                    'Q' => U2Q::q(),
                    _ => unreachable!(),
                };
            }
            let k = u.k;
            if k > 3 { continue; }
            let target = u.to_float();
            let sols = phase1_brute(k);
            let min_dist = sols.iter().map(|sol| {
                let cand = solution_to_u2q(sol, k);
                diamond_distance_float(&cand.to_float(), &target)
            }).fold(f64::INFINITY, f64::min);
            assert!(min_dist < 1e-9,
                "circuit \"{circuit}\" k={k}: min dist = {min_dist:.3e}");
            tested += 1;
        }
        assert!(tested > 0, "test sampled zero circuits — increase loop range");
    }

    // ── Phase 5a regression tests ─────────────────────────────────────────────

    #[test]
    fn zomega_subset_at_k_2() {
        let zomega_sols: Vec<[i64; 8]> = {
            let target = 1i64 << 2;
            let mut x = [0i64; 8];
            let mut sols = Vec::new();
            fn enum8<F: FnMut(&[i64; 8])>(x: &mut [i64; 8], pos: usize, rem: i64, cb: &mut F) {
                if pos == 8 { if rem == 0 { cb(x); } return; }
                let bound = (rem as f64).sqrt().floor() as i64;
                for v in -bound..=bound {
                    if v * v > rem { continue; }
                    x[pos] = v;
                    enum8(x, pos + 1, rem - v * v, cb);
                }
            }
            enum8(&mut x, 0, target, &mut |x| {
                let b = x[0]*x[1] - x[0]*x[3] + x[1]*x[2] + x[2]*x[3]
                      + x[4]*x[5] - x[4]*x[7] + x[5]*x[6] + x[6]*x[7];
                if b == 0 { sols.push(*x); }
            });
            sols
        };
        for sol_w in &zomega_sols {
            let mut sol_z = [0i64; 16];
            for (i, &v) in sol_w.iter().enumerate() {
                let block = i / 4;
                let off = i % 4;
                sol_z[block * 8 + 2 * off] = v;
            }
            let (b1, b2, b3) = bilinear_forms(&sol_z);
            assert_eq!(b1, 0, "B_1 should be 0 for Z[ω]-embedded solution");
            assert_eq!(b2, 0, "B_2 should be 0 for Z[ω]-embedded solution");
            assert_eq!(b3, 0, "B_3 should be 0 for Z[ω]-embedded solution");
        }
    }

    #[test]
    fn negative_bilinear_cases_excluded() {
        let bad_x = [1i64, 1, 0, 0, 0, 0, 0, 0,    0, 0, 0, 0, 0, 0, 0, 0];
        let norm_sq: i64 = bad_x.iter().map(|v| v * v).sum();
        assert_eq!(norm_sq, 2);
        let (b1, _, _) = bilinear_forms(&bad_x);
        assert_ne!(b1, 0, "constructed example should fail B_1");
        let sols = phase1_brute(1);
        assert!(!sols.contains(&bad_x), "phase1_brute(1) must exclude bad_x");
    }

    #[test]
    fn det_phase_of_known_circuits() {
        use crate::matrix::u2::U2Q;
        assert_eq!(det_phase_of(&U2Q::eye().to_float()), 0);
        assert_eq!(det_phase_of(&U2Q::q().to_float()), 1);
        assert_eq!(det_phase_of(&U2Q::t().to_float()), 2);
        assert_eq!(det_phase_of(&U2Q::s().to_float()), 4);
        assert_eq!(det_phase_of(&U2Q::h().to_float()), 8);
        assert_eq!(det_phase_of(&U2Q::x().to_float()), 8);
        assert_eq!(det_phase_of(&U2Q::y().to_float()), 8);
        assert_eq!(det_phase_of(&U2Q::z().to_float()), 8);
        let hqh = U2Q::h() * U2Q::q() * U2Q::h();
        assert_eq!(det_phase_of(&hqh.to_float()), 1);
        let qhq = U2Q::q() * U2Q::h() * U2Q::q();
        assert_eq!(det_phase_of(&qhq.to_float()), 10);
    }

    #[test]
    fn extended_reconstruction_full_pipeline() {
        use crate::matrix::u2::U2Q;
        use crate::synthesis::distance::diamond_distance_float;
        let circuits: Vec<&str> = vec!["Q", "T", "QQ", "HQH", "QHQ", "HQHQ"];
        for circuit in circuits {
            let mut u = U2Q::eye();
            for c in circuit.chars() {
                u = u * match c {
                    'H' => U2Q::h(), 'S' => U2Q::s(), 'Q' => U2Q::q(),
                    'T' => U2Q::t(), _ => unreachable!(),
                };
            }
            let k = u.k;
            if k > 3 { continue; }
            let target = u.to_float();
            let d_target = det_phase_of(&target);
            let sols = phase1_brute(k);
            let min_dist = sols.iter().map(|sol| {
                let cand = solution_to_u2q_d(sol, k, d_target);
                diamond_distance_float(&cand.to_float(), &target)
            }).fold(f64::INFINITY, f64::min);
            assert!(min_dist < 1e-9,
                "circuit \"{circuit}\" k={k} d={d_target}: min dist = {min_dist:.3e}");
        }
    }

    #[test]
    fn y_lattice_image_matches_y_real() {
        use crate::synthesis::sigma::{build_y_vector, sigma_8};
        let v = [0.5, 0.3, 0.7, -0.4];
        let target: Mat2 = [
            [Complex64::new(v[0], v[1]), Complex64::new(0.0, 0.0)],
            [Complex64::new(v[2], v[3]), Complex64::new(0.0, 0.0)],
        ];
        let k = 6;
        let y_real = build_y_vector(&target, k);
        let y_lattice = uv_to_xy_zeta(v, k);
        let s8 = sigma_8();
        let mut y_real_from_lattice = [0.0f64; 16];
        for i in 0..8 {
            for j in 0..8 {
                y_real_from_lattice[i] += s8[i][j] * y_lattice[j];
                y_real_from_lattice[i + 8] += s8[i][j] * y_lattice[j + 8];
            }
        }
        for i in 0..16 {
            assert!((y_real_from_lattice[i] - y_real[i]).abs() < 1e-10,
                "mismatch at i={i}: lattice→real = {}, build_y_vector = {}",
                y_real_from_lattice[i], y_real[i]);
        }
    }

    #[test]
    fn brute_finds_hqh_via_extended_reconstruction() {
        use crate::matrix::u2::U2Q;
        use crate::synthesis::distance::diamond_distance_float;
        let hqh: U2Q = U2Q::h() * U2Q::q() * U2Q::h();
        let target = hqh.to_float();
        let sols = phase1_brute(2);
        let mut min_dist = f64::INFINITY;
        for sol in &sols {
            let cand = solution_to_u2q_d(sol, 2, 1);
            let d = diamond_distance_float(&cand.to_float(), &target);
            if d < min_dist { min_dist = d; }
        }
        assert!(min_dist < 1e-9, "min dist to HQH at k=2, d=1: {min_dist}");
    }

    #[test]
    fn solution_to_u2q_d_0_matches_su2() {
        let sol = [1, 2, -1, 0, 0, 1, 0, -1,    -2, 1, 0, 1, 1, 0, -1, 0];
        let u_default = solution_to_u2q(&sol, 4);
        let u_d0 = solution_to_u2q_d(&sol, 4, 0);
        assert_eq!(u_default.u11, u_d0.u11);
        assert_eq!(u_default.u12, u_d0.u12);
        assert_eq!(u_default.u21, u_d0.u21);
        assert_eq!(u_default.u22, u_d0.u22);
        assert_eq!(u_default.k, u_d0.k);
    }

    #[test]
    fn extended_reconstruction_finds_q_alone() {
        use crate::matrix::u2::U2Q;
        use crate::synthesis::distance::diamond_distance_float;
        let q = U2Q::q();
        let target = q.to_float();
        let sols = phase1_brute(0);
        let mut min_dist = f64::INFINITY;
        for sol in &sols {
            for d in 0..16 {
                let cand = solution_to_u2q_d(sol, 0, d);
                let dd = diamond_distance_float(&cand.to_float(), &target);
                if dd < min_dist { min_dist = dd; }
            }
        }
        assert!(min_dist < 1e-9, "extended reconstruction should find Q exactly: {min_dist}");
    }

    #[test]
    fn extended_reconstruction_finds_random_clifford_sqrt_t() {
        use rand::Rng;
        use crate::matrix::u2::U2Q;
        use crate::synthesis::distance::diamond_distance_float;
        let mut rng = rand::rng();
        for _ in 0..30 {
            let n = rng.random_range(1..=5);
            let circuit: String = (0..n).map(|_|
                ['H', 'S', 'Q'][rng.random_range(0..3)]
            ).collect();
            let mut u = U2Q::eye();
            for c in circuit.chars() {
                u = u * match c {
                    'H' => U2Q::h(), 'S' => U2Q::s(), 'Q' => U2Q::q(), _ => unreachable!(),
                };
            }
            let k = u.k;
            if k > 3 { continue; }
            let target = u.to_float();
            let sols = phase1_brute(k);
            let mut min_dist = f64::INFINITY;
            for sol in &sols {
                for d in 0..16 {
                    let cand = solution_to_u2q_d(sol, k, d);
                    let dd = diamond_distance_float(&cand.to_float(), &target);
                    if dd < min_dist { min_dist = dd; }
                }
            }
            assert!(min_dist < 1e-9,
                "circuit \"{circuit}\" k={k}: extended-recon min dist = {min_dist:.3e}");
        }
    }

    #[test]
    fn hqh_su2_limitation_distance() {
        use crate::matrix::u2::U2Q;
        use crate::synthesis::distance::diamond_distance_float;
        let hqh: U2Q = U2Q::h() * U2Q::q() * U2Q::h();
        let target = hqh.to_float();
        let sols = phase1_brute(2);
        let min_dist = sols.iter().map(|sol| {
            let cand = solution_to_u2q(sol, 2);
            diamond_distance_float(&cand.to_float(), &target)
        }).fold(f64::INFINITY, f64::min);
        let expected = (PI / 16.0).sin();
        let rel_err = (min_dist - expected).abs() / expected;
        assert!(rel_err < 0.01,
            "HQH SU(2) limitation: min dist = {min_dist:.6}, expected ≈ {expected:.6} (sin(π/16))");
    }

    // ── M1 (lattice-coord Q-metric) tests ─────────────────────────────────────

    fn matvec_16(m: &[[f64; 16]; 16], v: &[f64; 16]) -> [f64; 16] {
        let mut out = [0.0; 16];
        for i in 0..16 {
            for j in 0..16 {
                out[i] += m[i][j] * v[j];
            }
        }
        out
    }

    #[test]
    fn q_lattice_is_symmetric() {
        let v = [1.0 / 2.0_f64.sqrt(), 0.0, 1.0 / 2.0_f64.sqrt(), 0.0];
        let q = build_q_zzeta_lattice(v, 6, 1e-3);
        for i in 0..16 {
            for j in 0..16 {
                let diff = (q[i][j] - q[j][i]).abs();
                let scale = q[i][j].abs().max(q[j][i].abs()).max(1.0);
                assert!(diff < 1e-12 * scale,
                    "Q_lat non-symmetric at ({i},{j}): {} vs {} (diff {})",
                    q[i][j], q[j][i], diff);
            }
        }
    }

    #[test]
    fn q_lattice_yhat_is_eigenvector() {
        let v = [0.5, 0.3, 0.7, -0.4];
        let k = 6;
        let eps = 1e-3;
        let q = build_q_zzeta_lattice(v, k, eps);
        let y = compute_align_vec_zeta(v);
        let y_norm: f64 = y.iter().map(|x| x * x).sum::<f64>().sqrt();
        let yhat: [f64; 16] = std::array::from_fn(|i| y[i] / y_norm);
        let qy = matvec_16(&q, &yhat);
        let r = (k as f64).exp2().sqrt();
        let delta_y = r * eps * eps / (2.0 * (1.0 + (1.0 - eps * eps).sqrt()));
        let lambda_y = 1.0 / (delta_y * delta_y);
        for i in 0..16 {
            let expected = lambda_y * yhat[i];
            let tol = 1e-5 * lambda_y.abs().max(1.0);
            assert!((qy[i] - expected).abs() < tol,
                "Q_lat·ŷ at i={i}: got {}, expected {}", qy[i], expected);
        }
    }

    #[test]
    fn q_lattice_trace_matches_spectrum() {
        let v = [0.7, 0.2, 0.1, 0.5];
        let k = 6;
        let eps = 1e-3;
        let q = build_q_zzeta_lattice(v, k, eps);
        let r_sq = (k as f64).exp2();
        let r = r_sq.sqrt();
        let delta_y = r * eps * eps / (2.0 * (1.0 + (1.0 - eps * eps).sqrt()));
        let delta_perp = r * eps;
        let lambda_y = 1.0 / (delta_y * delta_y);
        let lambda_perp = 1.0 / (delta_perp * delta_perp);
        let lambda_r = 1.0 / r_sq;
        let expected_trace = lambda_y + 3.0 * lambda_perp + 12.0 * lambda_r;
        let actual_trace: f64 = (0..16).map(|i| q[i][i]).sum();
        let rel_err = (actual_trace - expected_trace).abs() / expected_trace.abs();
        assert!(rel_err < 1e-6,
            "trace(Q_lat) = {actual_trace}, expected {expected_trace}, rel err {rel_err:.3e}");
    }

    #[test]
    fn q_lattice_psd() {
        let v = [0.5, 0.5, 0.5, 0.5];
        let q = build_q_zzeta_lattice(v, 6, 1e-3);
        let test_vecs = [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0, 0.5, -0.5, 0.25, -0.25, 0.1, 0.0, 0.7, -0.3],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        ];
        for w in &test_vecs {
            let qw = matvec_16(&q, w);
            let wqw: f64 = w.iter().zip(qw.iter()).map(|(a, b)| a * b).sum();
            assert!(wqw > 0.0, "vᵀ Q_lat v = {wqw} should be > 0");
        }
    }

    #[test]
    fn brute_solutions_roundtrip_via_decomposer() {
        let sols = phase1_brute(2);
        let mut tested = 0;
        let stride = sols.len().max(1) / 50;
        for sol in sols.iter().step_by(stride.max(1)) {
            let u = solution_to_u2q(sol, 2);
            let gates = BlochDecomposer.decompose(&u);
            let _ = gates;
            tested += 1;
        }
        assert!(tested > 0);
    }

    // ── M3 (Q-metric MPFR / i256 snapshot) tests ─────────────────────────────

    #[test]
    fn q_int_zeta_matches_q_mpfr_zeta() {
        // After build_q_int_zeta, q_int / 2^scale_bits ≈ q_mpfr.
        use rug::Float as RFloat;
        let v = [0.5, 0.3, 0.7, -0.4];
        let mut s = IntScratch16::new(1e-3);
        build_q_mpfr_zeta(&mut s, v, 6, 1e-3);
        build_q_int_zeta(&mut s);
        let mut max_rel = 0.0f64;
        let mut tmp = RFloat::with_val(s.prec_q, 0.0);
        for i in 0..16 {
            for j in 0..16 {
                q_metric::i256_to_rfloat(s.q_int[i][j], &mut tmp);
                let recovered = if s.scale_bits >= 0 {
                    tmp.clone() >> s.scale_bits as u32
                } else {
                    tmp.clone() << (-s.scale_bits) as u32
                };
                let q_true = s.q_mpfr[i][j].to_f64();
                let q_rec = recovered.to_f64();
                let abs = q_true.abs().max(q_rec.abs()).max(1e-300);
                let rel = (q_true - q_rec).abs() / abs;
                if rel > max_rel {
                    max_rel = rel;
                }
            }
        }
        // 2^-(TARGET_BITS-1) ≈ 1.5e-54 worst-case — be very forgiving.
        assert!(max_rel < 1e-25, "q_int round-trip rel err = {}", max_rel);
    }

    #[test]
    fn q_mpfr_matches_q_lattice_f64() {
        // build_q_mpfr_zeta(prec=high) ≈ build_q_zzeta_lattice (f64) within
        // f64 precision of the f64 inputs.
        let v = [0.5, 0.3, 0.7, -0.4];
        let q_f64 = build_q_zzeta_lattice(v, 6, 1e-3);
        let mut s = IntScratch16::new(1e-3);
        build_q_mpfr_zeta(&mut s, v, 6, 1e-3);
        for i in 0..16 {
            for j in 0..16 {
                let diff = (q_f64[i][j] - s.q_mpfr[i][j].to_f64()).abs();
                let scale = q_f64[i][j].abs().max(s.q_mpfr[i][j].to_f64().abs()).max(1.0);
                assert!(diff < 1e-9 * scale,
                    "q_mpfr vs q_f64 mismatch at ({i},{j}): {} vs {} (diff {})",
                    q_f64[i][j], s.q_mpfr[i][j].to_f64(), diff);
            }
        }
    }
}
