//! Research probes (all #[ignore]): backend-level diagnostic harnesses kept
//! runnable but out of the unit-test file. Run individually, e.g.
//! `cargo test --release --lib audit_radial_displacement_probe -- --ignored --nocapture`.
//!
//! File-internal probes (LLL/SE/Q-metric telemetry coupled to a specific
//! module's private helpers) live inline in that module's `#[cfg(test)]`
//! block instead.

#![allow(unused_imports)]
use crate::rings::MpFloat;
use super::*; // the tests module
use super::super::*; // lattice::zeta internals

    /// Precision-audit probe E1 (ignored): per-target radial cap
    /// displacement at ε = 1e-8 for the probe_omega_vs_zeta seed-12648430
    /// targets. Computes, at MPFR-300 (≫ production prec_q = 213, so
    /// only the f64 entry points under audit survive):
    ///
    ///   ν      = |col1(target)| − 1   (target's own f64 quantization
    ///            defect; prefix_residual_uv_mpfr never normalizes it away)
    ///   η_tot  = ‖uv_to_lattice_y_zeta_mpfr(v)‖ / ρ − 1, ρ = 2^(k/2)/2
    ///            (total radial norm error of the production y chain:
    ///            ν + the f64 cos/sin embedding error)
    ///
    /// and reports the cap displacement D = η / (ε²/2) in units of the
    /// FULL radial window width. |D| ≳ 0.1 (≈ the bound-1.5 tolerance
    /// 0.21·Δ_y on a half-width offset) loses apex solutions outright.
    /// Run: cargo test --release --lib audit_radial_displacement -- \
    ///      --ignored --nocapture
    #[test]
    #[ignore]
    fn audit_radial_displacement_probe() {
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
        let k = 22_u32; // the jackpot lde; ρ = 2^11
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