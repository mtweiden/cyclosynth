//! Research probes (all #[ignore]): backend-level diagnostic harnesses kept
//! runnable but out of the unit-test file. Run individually, e.g.
//! `cargo test --release --lib probe_f64_entry_radial_error -- --ignored --nocapture`.
//!
//! File-internal probes (LLL/SE/Q-metric telemetry coupled to a specific
//! module's private helpers) live inline in that module's `#[cfg(test)]`
//! block instead.

#![allow(unused_imports)]
use crate::rings::MpFloat;
use super::*; // the tests module
use super::super::*; // lattice::zeta internals
use crate::synthesis::lattice::zeta::integer::find_aligned_lattice_points;
use crate::synthesis::lattice::zeta::scratch::IntScratch16;
use crate::synthesis::lattice::zeta::se::SeCenter16;
use crate::synthesis::lattice::zeta::brute::uv_to_lattice_y_zeta;
use crate::synthesis::clifford_sqrt_t::{det_phase_of, solution_to_u2q_with_det_phase, unitary_to_uv_zeta};
use std::sync::atomic::{AtomicBool, Ordering};

    /// Measures the radial (norm) error the f64 target→lattice-vector chain
    /// introduces, to check the f64-input entry path doesn't push the cap
    /// center far enough to drop exact solutions. At MPFR-300 (far above
    /// production's 213-bit precision, so only the f64 steps' error survives),
    /// for 12 random targets at ε=1e-8, it reports two error sources:
    ///   ν      = |col1(target)| − 1            (the target's own f64 rounding)
    ///   η_tot  = ‖y‖ / ρ − 1, ρ = 2^(k/2)/2    (ν + the f64 cos/sin embedding)
    /// as a fraction D = η / (ε²/2) of the full cap window. |D| ≳ 0.1 loses
    /// the exact (best-distance) solutions outright.
    /// Run: cargo test --release --lib probe_f64_entry_radial_error -- \
    ///      --ignored --nocapture
    #[test]
    #[ignore = "census probe, print-only; see doc comment"]
    fn probe_f64_entry_radial_error() {
        use crate::synthesis::lattice::zeta::brute::uv_to_lattice_y_zeta_mpfr;

        // SplitMix64 + u3, replicated from src/bin/probe_omega_vs_zeta.rs.
        struct Xs(u64);
        impl Xs {
            fn next(&mut self) -> u64 {
                self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
                let mut z = self.0;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
                z ^ (z >> 31)
            }
            fn unit(&mut self) -> f64 {
                (self.next() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
            }
            fn range(&mut self, lo: f64, hi: f64) -> f64 {
                lo + (hi - lo) * self.unit()
            }
        }
        fn u3(theta: f64, phi: f64, lam: f64) -> Mat2 {
            let (c, s) = ((theta / 2.0).cos(), (theta / 2.0).sin());
            let eilam = Complex64::from_polar(1.0, lam);
            let eiphi = Complex64::from_polar(1.0, phi);
            let m = [
                [Complex64::new(c, 0.0), -eilam * s],
                [eiphi * s, eiphi * eilam * c],
            ];
            let g = Complex64::from_polar(1.0, -(phi + lam) / 2.0);
            [
                [m[0][0] * g, m[0][1] * g],
                [m[1][0] * g, m[1][1] * g],
            ]
        }

        const PREC: u32 = 300;
        let eps = 1e-8_f64;
        let k = 22_u32; // the deepest lde the √T search reaches; ρ = 2^11
        let window_rel = eps * eps / 2.0; // full radial window, relative

        let mut rng = Xs(12648430);
        let targets: Vec<(f64, f64, f64)> = (0..12)
            .map(|_| {
                (
                    rng.range(0.2, PI - 0.2),
                    rng.range(0.1, 2.0 * PI - 0.1),
                    rng.range(0.1, 2.0 * PI - 0.1),
                )
            })
            .collect();

        eprintln!(
            "ε={eps:.0e}: radial window (rel) = {window_rel:.3e}; \
             apex tolerance ≈ 0.21·window (bound 1.5)\n\
             tgt |   ν = |col1|−1 |  η_emb (cos/sin) |   η_total | D = η_tot/window"
        );
        for (i, &(th, ph, la)) in targets.iter().enumerate() {
            // The production deep path projects det first; replicate.
            let t = crate::synthesis::clifford_sqrt_t::project_det_to_zeta_coset(
                &u3(th, ph, la),
            );
            let v: [MpFloat; 4] = [
                MpFloat::with_val(PREC, t[0][0].re),
                MpFloat::with_val(PREC, t[0][0].im),
                MpFloat::with_val(PREC, t[1][0].re),
                MpFloat::with_val(PREC, t[1][0].im),
            ];
            let mut v_norm_sq = MpFloat::with_val(PREC, 0.0);
            for c in &v {
                v_norm_sq += MpFloat::with_val(PREC, c * c);
            }
            let v_norm = v_norm_sq.sqrt();
            let nu = MpFloat::with_val(PREC, &v_norm - 1.0_f64).to_f64();

            let y = uv_to_lattice_y_zeta_mpfr(&v, k, PREC);
            let mut y_norm_sq = MpFloat::with_val(PREC, 0.0);
            for c in &y {
                y_norm_sq += MpFloat::with_val(PREC, c * c);
            }
            let y_norm = y_norm_sq.sqrt();
            // ρ = 2^(k/2)/2 at PREC.
            let mut rho = MpFloat::with_val(PREC, 1.0);
            rho <<= k / 2;
            if k % 2 == 1 {
                rho *= MpFloat::with_val(PREC, 2.0).sqrt();
            }
            rho /= 2u32;
            let ratio_tot = MpFloat::with_val(PREC, &y_norm / &rho);
            let eta_tot = MpFloat::with_val(PREC, &ratio_tot - 1.0_f64).to_f64();
            // Embedding-only part: ‖y‖/(ρ·|v|) − 1.
            let rho_v = MpFloat::with_val(PREC, &rho * &v_norm);
            let ratio_emb = MpFloat::with_val(PREC, &y_norm / &rho_v);
            let eta_emb = MpFloat::with_val(PREC, &ratio_emb - 1.0_f64).to_f64();
            eprintln!(
                " {i:>2} | {nu:>+13.3e} | {eta_emb:>+13.3e} | {eta_tot:>+10.3e} | {:>+8.2}",
                eta_tot / window_rel
            );
        }
    }

    // ---- relocated from integer.rs (diagnostic probes, not unit tests) ----

    /// Diagnostic (ignored): decompose the QHQ@k=1 solution's Q-norm into
    /// geometric Q (true fractional cap center), Q from the i64-rounded
    /// center, and Q from the fractional SeCenter16 the walk uses (should
    /// match geometric to ~1e-6). The rounded center inflates Q and forces a
    /// generous bound; geometric Q ≤ 1.25. Run with --ignored --nocapture.
    #[test]
    #[ignore = "forensic probe; run individually (env-var setup)"]
    fn q_norm_center_source_breakdown() {
        use crate::matrix::u2::U2Q;

        // SAFETY: single-threaded at this point; racy if another --ignored
        // test in the same process reads CYCLOSYNTH_* concurrently — run
        // these probes individually.
        unsafe { std::env::set_var("CYCLOSYNTH_BOUND_SQ", "8") };
        let qhq: U2Q = U2Q::q() * U2Q::h() * U2Q::q();
        let target = qhq.to_float();
        let v = unitary_to_uv_zeta(&target);
        let k = qhq.k;
        let y = uv_to_lattice_y_zeta(v, k);
        let eps = 0.1_f64;

        let mut s = IntScratch16::new(eps);
        let abort = AtomicBool::new(false);
        let sols = find_aligned_lattice_points(&mut s, &y, k, eps, 100_000_000, &abort);
        // SAFETY: single-threaded at this point; racy if another --ignored
        // test in the same process reads CYCLOSYNTH_* concurrently — run
        // these probes individually.
        unsafe { std::env::remove_var("CYCLOSYNTH_BOUND_SQ") };
        assert!(!sols.is_empty(), "find_aligned_lattice_points@bound8 must find QHQ");

        let q = build_q_zzeta_lattice(v, k, eps);
        // True cap center (ambient), rounded-z_c effective center, and the
        // fractional SE center (int + frac pair) the walk uses.
        let c_true: [f64; 16] = std::array::from_fn(|i| s.c[i].to_f64());
        let mut c_rounded = [0.0f64; 16];
        for i in 0..16 {
            let zi = s.lu_x[i].to_f64().round();
            for j in 0..16 {
                c_rounded[j] += zi * s.basis[i][j] as f64;
            }
        }
        let se_center = SeCenter16::from_lu_x(&s.lu_x);
        let mut c_se = [0.0f64; 16];
        for i in 0..16 {
            let zi = se_center.int[i] as f64 + se_center.frac[i];
            for j in 0..16 {
                c_se[j] += zi * s.basis[i][j] as f64;
            }
        }
        let q_norm = |x: &[i64; 16], c: &[f64; 16]| -> f64 {
            let d: [f64; 16] = std::array::from_fn(|i| x[i] as f64 - c[i]);
            let mut acc = 0.0;
            for i in 0..16 {
                for j in 0..16 {
                    acc += d[i] * q[i][j] * d[j];
                }
            }
            acc
        };
        for (n, sol) in sols.iter().enumerate() {
            eprintln!(
                "sol {n}: Q_geometric={:.6}  Q_se_rounded_center={:.4}  Q_se_effective={:.6}",
                q_norm(sol, &c_true),
                q_norm(sol, &c_rounded),
                q_norm(sol, &c_se)
            );
        }
        let frac_err: f64 = (0..16)
            .map(|i| (s.lu_x[i].to_f64() - s.lu_x[i].to_f64().round()).abs())
            .fold(0.0, f64::max);
        eprintln!("max |frac(lu_x)| = {frac_err:.4}");
    }

    /// Telemetry (ignored): geometric Q-norm² distribution of ε-close
    /// solutions across a θ × ε × k grid, enumerated at a wide bound to
    /// observe the full distribution. The geometric band is [0.875, 1.25];
    /// this sweep is how that was measured.
    /// Run: `cargo test --release --lib q_norm_distribution_sweep_16d -- --ignored --nocapture`
    #[test]
    #[ignore = "forensic probe; run individually (env-var setup)"]
    fn q_norm_distribution_sweep_16d() {
        use crate::synthesis::distance::diamond_distance_float;
        use num_complex::Complex;

        // SAFETY: single-threaded at this point; racy if another --ignored
        // test in the same process reads CYCLOSYNTH_* concurrently — run
        // these probes individually.
        unsafe { std::env::set_var("CYCLOSYNTH_BOUND_SQ", "4") };
        let mut global_max_close = 0.0f64;
        let mut global_max_all = 0.0f64;
        let mut total_close = 0usize;

        for &theta in &[0.3f64, 0.55, 0.8, 1.05, 1.3] {
            let target: Mat2 = [
                [Complex::from_polar(1.0, -theta / 2.0), Complex::new(0.0, 0.0)],
                [Complex::new(0.0, 0.0), Complex::from_polar(1.0, theta / 2.0)],
            ];
            let v = unitary_to_uv_zeta(&target);
            let d = det_phase_of(&target);
            for &(eps, k_lo, k_hi) in &[(3e-2f64, 5u32, 7u32), (1e-3, 9, 10)] {
                for k in k_lo..=k_hi {
                    let y = uv_to_lattice_y_zeta(v, k);
                    let mut s = IntScratch16::new(eps);
                    let abort = AtomicBool::new(false);
                    let sols = find_aligned_lattice_points(&mut s, &y, k, eps, 100_000_000, &abort);
                    if sols.is_empty() {
                        continue;
                    }
                    let q = build_q_zzeta_lattice(
                        v, k, eps,
                    );
                    let c: [f64; 16] = std::array::from_fn(|i| s.c[i].to_f64());
                    let mut max_close = 0.0f64;
                    let mut max_all = 0.0f64;
                    let mut n_close = 0usize;
                    for sol in &sols {
                        let dvec: [f64; 16] =
                            std::array::from_fn(|i| sol[i] as f64 - c[i]);
                        let mut qn = 0.0;
                        for i in 0..16 {
                            for j in 0..16 {
                                qn += dvec[i] * q[i][j] * dvec[j];
                            }
                        }
                        max_all = max_all.max(qn);
                        let cand = solution_to_u2q_with_det_phase(sol, k, d);
                        if diamond_distance_float(&cand.to_float(), &target) <= eps {
                            max_close = max_close.max(qn);
                            n_close += 1;
                        }
                    }
                    if n_close > 0 {
                        eprintln!(
                            "θ={theta:<4} ε={eps:.0e} k={k:<2} sols={:<5} close={n_close:<4} maxQ_close={max_close:.4} maxQ_all={max_all:.4}",
                            sols.len()
                        );
                    }
                    global_max_close = global_max_close.max(max_close);
                    global_max_all = global_max_all.max(max_all);
                    total_close += n_close;
                }
            }
        }
        // SAFETY: single-threaded at this point; racy if another --ignored
        // test in the same process reads CYCLOSYNTH_* concurrently — run
        // these probes individually.
        unsafe { std::env::remove_var("CYCLOSYNTH_BOUND_SQ") };
        eprintln!(
            "GLOBAL: eps-close sols={total_close}  maxQ_close={global_max_close:.4}  maxQ_all={global_max_all:.4}"
        );
    }

    /// Diagnostic: for Rz(0.3) at ε=1e-3, first establish the lde the 8D
    /// Clifford+T synthesizer reaches (upper bound for Clifford+√T since
    /// `T = QQ` as gates and lde counts √2 denominators identically). Then
    /// verify the Z[ζ_16] / Clifford+√T flow hits it at ≤ that lde.
    /// Behind `#[ignore]`: `cargo test --release --lib sqrt_t_depth_vs_clifford_t_baseline --
    /// --ignored --nocapture`.
    #[test]
    #[ignore = "slow diagnostic probe; see doc comment"]
    fn sqrt_t_depth_vs_clifford_t_baseline() {
        use crate::synthesis::distance::diamond_distance_float;
        use crate::synthesis::clifford_t::SynthesizerT;
        let theta = 0.3_f64;
        let target: Mat2 = [
            [Complex64::from_polar(1.0, -theta / 2.0),
             Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0),
             Complex64::from_polar(1.0, theta / 2.0)],
        ];
        let eps = 1e-3_f64;

        // 1. Upper bound from 8D Clifford+T.
        let synth_t = SynthesizerT::new(eps);
        let t0 = std::time::Instant::now();
        let r_t = synth_t.synthesize(target).expect("8D should land Rz(0.3) at ε=1e-3");
        eprintln!(
            "8D Clifford+T:  lde={}  dist={:.3e}  t={:?}",
            r_t.lde, r_t.distance, t0.elapsed()
        );
        let upper_bound = r_t.lde;

        // 2. Sweep Clifford+√T at increasing budget at each k up to upper_bound.
        let v = unitary_to_uv_zeta(&target);
        let d = det_phase_of(&target);
        eprintln!("upper bound k = {upper_bound}; v={v:?}, d={d}");
        for k in 5u32..=(upper_bound + 2).min(20) {
            let y = uv_to_lattice_y_zeta(v, k);
            let budget = 1_000_000_000_u64;
            let mut s = IntScratch16::new(eps);
            let abort = AtomicBool::new(false);
            let t0 = std::time::Instant::now();
            let sols = find_aligned_lattice_points(&mut s, &y, k, eps, budget, &abort);
            let dt = t0.elapsed();
            let abort_v = abort.load(Ordering::Relaxed);
            let min_dist = sols.iter().map(|sol| {
                let cand = solution_to_u2q_with_det_phase(sol, k, d);
                diamond_distance_float(&cand.to_float(), &target)
            }).fold(f64::INFINITY, f64::min);
            let hit = min_dist < eps;
            eprintln!(
                "k={k:>2}  sols={:>4}  budget_hit={abort_v:>5}  \
                 min_dist={min_dist:.3e}  hit_eps={hit:>5}  t={:?}",
                sols.len(), dt
            );
            if hit { break; }
        }
    }
