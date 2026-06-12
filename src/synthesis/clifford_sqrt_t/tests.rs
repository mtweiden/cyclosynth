    use super::*;
    use crate::synthesis::distance::diamond_distance_float;
    use num_complex::Complex64;
    use std::f64::consts::PI;

    fn complex_target(matrix: [[Complex64; 2]; 2]) -> Mat2 {
        matrix
    }

    /// Print raw vs deduped size of `L_m^Q` for m ∈ [0, 5]. Behind
    /// `--nocapture` in normal runs; the assertions are minimal — this is
    /// a measurement, not a correctness contract.
    #[test]
    fn build_l_q_size_growth() {
        for m in 0..=5 {
            let raw = if m == 0 {
                1
            } else {
                9 * 6u64.pow(m - 1) * 24
            };
            let l = build_fgkm_prefix_set(m);
            let dedup = l.len();
            let factor = raw as f64 / dedup as f64;
            eprintln!(
                "m={m}  raw={raw:>8}  dedup={dedup:>8}  factor={factor:.2}x"
            );
            // Sanity: dedup never grows the set.
            assert!((dedup as u64) <= raw,
                "dedup ({dedup}) > raw ({raw}) at m={m}");
            // m=0 is just identity.
            if m == 0 {
                assert_eq!(dedup, 1);
            }
        }
    }

    /// Back-of-envelope: under cost model C(k) = c·α^k, the D&C cost
    /// ratio (vs single search at k_total) is
    ///   S(m, α) = Σ_k count(m, k) / α^k
    /// and is independent of k_total (the c·α^{k_total} term cancels).
    /// D&C wins at m when S(m, α) < 1.
    #[test]
    fn build_l_q_dc_cost_ratio() {
        // Coarse k → count map per m, then evaluate S(m, α) for several α.
        for m in 1..=5 {
            let l = build_fgkm_prefix_set(m);
            let mut counts: Vec<u64> = vec![0; 64];
            for u in l.iter() {
                let k = u.k as usize;
                if k < counts.len() {
                    counts[k] += 1;
                }
            }
            eprint!("m={m:>2}  total={:>7}", l.len());
            for &alpha in &[2.0_f64, 2.5, 3.0, 3.5, 4.0] {
                let s: f64 = counts
                    .iter()
                    .enumerate()
                    .map(|(k, &c)| (c as f64) / alpha.powi(k as i32))
                    .sum();
                eprint!("   S(α={alpha:.1})={s:>10.2}");
            }
            eprintln!();
        }
        // Also show what threshold-filtering buys: keep only
        // prefixes with k_prefix ≥ τ, recompute S(m, α=2.0).
        eprintln!("\nThreshold filter τ on k_prefix, S(m, α=2):");
        eprintln!("{:>3}  {:>8}  τ=0    τ=4    τ=8    τ=12   τ=16   τ=20",
                  "m", "|L_m^Q|");
        for m in 1..=5 {
            let l = build_fgkm_prefix_set(m);
            let mut counts: Vec<u64> = vec![0; 64];
            for u in l.iter() {
                let k = u.k as usize;
                if k < counts.len() {
                    counts[k] += 1;
                }
            }
            eprint!("{m:>3}  {:>8}", l.len());
            for &tau in &[0usize, 4, 8, 12, 16, 20] {
                let s: f64 = counts
                    .iter()
                    .enumerate()
                    .skip(tau)
                    .map(|(k, &c)| (c as f64) / (2.0_f64).powi(k as i32))
                    .sum();
                let n_kept: u64 = counts.iter().skip(tau).sum();
                eprint!("  {s:>5.2} ({n_kept:>5})");
            }
            eprintln!();
        }
    }

    /// Histogram of `k_prefix` across `L_m^Q` for m ∈ [1, 5]. We expect
    /// k_prefix ≤ m by FGKM Theorem 4.1(b) (each syllable peels max_exp
    /// by ≥ 1, so the word's denominator exponent grows by at most m).
    /// The shape of the distribution determines how we bin prefixes by
    /// k for the inner LLL+SE search.
    #[test]
    fn build_l_q_k_distribution() {
        for m in 1..=5 {
            let l = build_fgkm_prefix_set(m);
            // Bins 0..=m+a few extra for safety in case the bound is
            // looser than expected.
            let max_bin: usize = (m as usize) + 4;
            let mut hist: Vec<u64> = vec![0; max_bin + 1];
            let mut k_min: u32 = u32::MAX;
            let mut k_max: u32 = 0;
            for u in l.iter() {
                let k = u.k as usize;
                k_min = k_min.min(u.k);
                k_max = k_max.max(u.k);
                if k <= max_bin {
                    hist[k] += 1;
                } else {
                    // Out-of-bound: extend the histogram (cheap, we'll
                    // see in the print).
                    while hist.len() <= k {
                        hist.push(0);
                    }
                    hist[k] += 1;
                }
            }
            let total: u64 = hist.iter().sum();
            eprintln!(
                "m={m}  total={total}  k range [{k_min}, {k_max}]"
            );
            for (k, count) in hist.iter().enumerate() {
                if *count == 0 { continue; }
                let pct = 100.0 * (*count as f64) / (total as f64);
                eprintln!("    k={k:>2}: {count:>7}  ({pct:>5.1}%)");
            }
        }
    }

    /// Multi-target benchmark: average D&C-with-filter vs single across
    /// random U3 targets at fixed ε.
    /// The optimize-cost hybrid runs a Clifford+T baseline and returns
    /// the min, so its weighted cost can never exceed the Clifford+T
    /// result on the same target. Guard that invariant.
    #[test]
    fn optimal_cost_never_exceeds_clifford_t() {
        fn rz(t: f64) -> Mat2 {
            [
                [Complex64::from_polar(1.0, -t/2.0), Complex64::new(0.0, 0.0)],
                [Complex64::new(0.0, 0.0), Complex64::from_polar(1.0, t/2.0)],
            ]
        }
        for &(theta, eps) in &[(0.3_f64, 1e-3_f64), (1.1, 1e-3), (2.37, 1e-4)] {
            let target = rz(theta);
            let rt = crate::synthesis::clifford_t::SynthesizerT::new(eps)
                .synthesize(target)
                .expect("clifford_t baseline should synthesize");
            let t_cost = gates_cost(rt.gates.as_deref().unwrap_or(""), 7);
            let rq = SynthesizerQ::new(eps)
                .with_optimize_cost(true)
                .with_optimal_lde_window(2)
                .synthesize(target)
                .expect("hybrid optimal should synthesize");
            assert!(rq.distance < eps);
            let q_cost = gates_cost(rq.gates.as_deref().unwrap_or(""), 7);
            assert!(
                q_cost <= t_cost,
                "hybrid cost {q_cost} > clifford_t cost {t_cost} at θ={theta}, ε={eps:e}"
            );
        }
    }

    /// Screen-truncation out-param plumbing: on an easy coarse-ε target
    /// no level hits a budget cap, so `synthesize_with_unverified_levels` must
    /// report zero unclear levels and agree with the public entry point.
    /// (ε = 1e-2: near-z-axis diagonal targets at 1e-3 burn every
    /// level's budget for ~5 min — known sparse-region hardness, not a
    /// plumbing concern.)
    #[test]
    fn screen_unclear_empty_on_easy_target() {
        let target: Mat2 = [
            [Complex64::from_polar(1.0, -0.35), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::from_polar(1.0, 0.35)],
        ];
        let synth = SynthesizerQ::new(1e-2).with_optimize_cost(false);
        let mut unclear = Vec::new();
        let r1 = synth
            .synthesize_with_unverified_levels(target, Some(&mut unclear))
            .expect("should synthesize");
        let r2 = synth.synthesize(target).expect("should synthesize");
        assert!(unclear.is_empty(), "unexpected unclear levels: {unclear:?}");
        assert_eq!(r1.lde, r2.lde);
        assert!(r1.distance < 1e-2);
    }

    /// Production-path certificate (items 1+2): the hybrid search with
    /// `certify` on must return a well-formed interval, and at coarse ε
    /// the floor-driven extension should CLOSE it on a generic target.
    #[test]
    fn production_certificate_well_formed_and_closes_at_coarse_eps() {
        fn rzm(t: f64) -> Mat2 {
            [
                [Complex64::from_polar(1.0, -t/2.0), Complex64::new(0.0, 0.0)],
                [Complex64::new(0.0, 0.0), Complex64::from_polar(1.0, t/2.0)],
            ]
        }
        let (r, cert) = SynthesizerQ::new(3e-2)
            .with_certify_extra_ms(20_000)
            .synthesize_with_certificate(rzm(0.7))
            .expect("should synthesize");
        assert!(r.distance < 3e-2);
        assert!(cert.lower_half_units <= cert.upper_half_units);
        assert_eq!(
            cert.upper_half_units,
            gates_cost(r.gates.as_deref().unwrap(), 7)
        );
        // At 3e-2 the optimum costs ~19 HU; the extension reaches the
        // closing horizon (k ≈ 6) within the budget.
        assert!(cert.certified_optimal,
            "expected closure at coarse ε: upper {} lower {} k {}",
            cert.upper_half_units, cert.lower_half_units, cert.k_searched);
    }

    /// Tier-1 closing certificate at the cheapest scale: a T gate costs
    /// 2 half-units and the beyond-horizon floor L(3) = 2 matches, so
    /// k_max = 2 must CLOSE the certificate. (Unbudgeted shell walks
    /// grow fast with k — a k=8 closure test ran minutes; keep tests at
    /// the smallest k that exercises the logic.)
    #[test]
    fn certificate_closes_on_t_target() {
        let t_f = U2Q::t().to_float();
        let g = Complex64::from_polar(1.0, -PI / 8.0); // det(T)=ζ₁₆² → g²=ζ₁₆⁻²
        let target: Mat2 = [
            [t_f[0][0] * g, t_f[0][1] * g],
            [t_f[1][0] * g, t_f[1][1] * g],
        ];
        let (r, cert) = SynthesizerQ::new(1e-3)
            .synthesize_exhaustive_certified(target, 2)
            .expect("certified synthesis should succeed");
        assert!(r.distance < 1e-3);
        assert_eq!(cert.upper_half_units, 2, "T circuit costs 2 HU");
        assert!(cert.certified_optimal,
            "upper {} vs floor {} at k=2",
            cert.upper_half_units, cert.lower_half_units);
        assert_eq!(cert.lower_half_units, cert.upper_half_units);
    }

    /// Tier-1 gap certificate on a generic target at a small horizon:
    /// interval well-formed, does not close.
    #[test]
    fn certificate_gap_on_generic_target() {
        fn rzm(t: f64) -> Mat2 {
            [
                [Complex64::from_polar(1.0, -t/2.0), Complex64::new(0.0, 0.0)],
                [Complex64::new(0.0, 0.0), Complex64::from_polar(1.0, t/2.0)],
            ]
        }
        let (r, cert) = SynthesizerQ::new(1e-2)
            .synthesize_exhaustive_certified(rzm(0.7), 4)
            .expect("certified synthesis should succeed");
        assert!(r.distance < 1e-2);
        assert!(cert.lower_half_units <= cert.upper_half_units);
        assert_eq!(cert.k_searched, 4);
        // A 1e-2 approximation of a generic angle costs well over
        // L(5) = 4 HU, so the interval stays open.
        assert!(!cert.certified_optimal,
            "unexpected closure: upper {} lower {}",
            cert.upper_half_units, cert.lower_half_units);
    }

    /// k = 8 closure on the single-Q target (cost 7 HU needs the L(9)=8
    /// floor). Minutes-scale unbudgeted walk — milestone runs only.
    #[test]
    #[ignore = "unbudgeted k=8 shell walk; run with --ignored"]
    fn certificate_closes_on_single_q_target_slow() {
        let g = Complex64::from_polar(1.0, -PI / 16.0);
        let hqh = (U2Q::h() * U2Q::q() * U2Q::h()).reduced().to_float();
        let target: Mat2 = [
            [hqh[0][0] * g, hqh[0][1] * g],
            [hqh[1][0] * g, hqh[1][1] * g],
        ];
        let (_, cert) = SynthesizerQ::new(1e-3)
            .synthesize_exhaustive_certified(target, 8)
            .expect("certified synthesis should succeed");
        assert_eq!(cert.upper_half_units, 7);
        assert!(cert.certified_optimal);
    }

    /// The odd-parity branch must reach circuits the single-target
    /// pipeline cannot: V = e^{-iπ/16}·(H·Q·H) has det 1 (even class),
    /// but its physical optimum is the single-Q circuit (odd class,
    /// cost 3.5). Without the branch the search can only offer even-Q
    /// approximations.
    #[test]
    fn odd_parity_branch_finds_single_q() {
        let g = Complex64::from_polar(1.0, -PI / 16.0);
        let hqh = {
            let u = (U2Q::h() * U2Q::q() * U2Q::h()).reduced();
            u.to_float()
        };
        let target: Mat2 = [
            [hqh[0][0] * g, hqh[0][1] * g],
            [hqh[1][0] * g, hqh[1][1] * g],
        ];
        let r = SynthesizerQ::new(1e-3)
            .synthesize(target)
            .expect("should synthesize");
        let gates = r.gates.expect("gates");
        let q = gates.chars().filter(|&c| c == 'Q').count();
        let t = gates.chars().filter(|&c| c == 'T').count();
        assert!(r.distance < 1e-3);
        assert_eq!((t, q), (0, 1),
            "odd branch should find the exact single-Q circuit, got {gates}");
    }

    #[test]
    #[ignore]
    fn z1_dc_dr_filter_random_targets() {
        use rand::{Rng, SeedableRng};
        use rand::rngs::StdRng;

        fn rz(t: f64) -> Mat2 {
            [
                [Complex64::from_polar(1.0, -t/2.0), Complex64::new(0.0, 0.0)],
                [Complex64::new(0.0, 0.0), Complex64::from_polar(1.0, t/2.0)],
            ]
        }
        fn ry(t: f64) -> Mat2 {
            let c = (t/2.0).cos();
            let s = (t/2.0).sin();
            [
                [Complex64::new(c, 0.0), Complex64::new(-s, 0.0)],
                [Complex64::new(s, 0.0), Complex64::new(c, 0.0)],
            ]
        }
        fn matmul(a: Mat2, b: Mat2) -> Mat2 {
            [
                [a[0][0]*b[0][0] + a[0][1]*b[1][0], a[0][0]*b[0][1] + a[0][1]*b[1][1]],
                [a[1][0]*b[0][0] + a[1][1]*b[1][0], a[1][0]*b[0][1] + a[1][1]*b[1][1]],
            ]
        }

        let mut rng = StdRng::seed_from_u64(0xBEEF);
        let n = 4;
        let eps: f64 = std::env::var("Z1_EPS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1e-4);

        eprintln!("\n=== ε={eps:.0e}, {n} random U3 targets ===");
        let mut total_single = 0.0_f64;
        let mut total_m1_relaxed = 0.0_f64;
        let mut total_m2_strict = 0.0_f64;
        let mut wins_m1 = 0;
        let mut wins_m2 = 0;

        for i in 0..n {
            let alpha = 2.0 * std::f64::consts::PI * rng.random::<f64>();
            let beta = 2.0 * std::f64::consts::PI * rng.random::<f64>();
            let gamma = 2.0 * std::f64::consts::PI * rng.random::<f64>();
            let target = matmul(matmul(rz(alpha), ry(beta)), rz(gamma));

            let synth_s = SynthesizerQ::new(eps).with_max_lde(20);
            let t0 = std::time::Instant::now();
            let r_s = synth_s.synthesize(target);
            let ts = t0.elapsed().as_secs_f64() * 1000.0;
            assert!(r_s.is_some());

            let synth_m1 = SynthesizerQ::new(eps).with_max_lde(20)
                .with_prefix_split_m(1).with_inner_det_phase_filter(vec![0, 1, 15]);
            let t0 = std::time::Instant::now();
            let r_m1 = synth_m1.synthesize(target);
            let tm1 = t0.elapsed().as_secs_f64() * 1000.0;

            let synth_m2 = SynthesizerQ::new(eps).with_max_lde(20)
                .with_prefix_split_m(2).with_inner_det_phase_filter(vec![0]);
            let t0 = std::time::Instant::now();
            let r_m2 = synth_m2.synthesize(target);
            let tm2 = t0.elapsed().as_secs_f64() * 1000.0;

            total_single += ts;
            total_m1_relaxed += tm1;
            total_m2_strict += tm2;
            if tm1 < ts { wins_m1 += 1; }
            if tm2 < ts { wins_m2 += 1; }
            eprintln!(
                "  trial {i}  single={ts:>6.0}ms  m1_relaxed={tm1:>6.0}ms ({:.2}×)  m2_strict={tm2:>6.0}ms ({:.2}×)",
                ts/tm1, ts/tm2
            );
            // Sanity: dc found a valid result.
            if let Some(r) = r_m1 {
                assert!(r.distance < eps, "m1 trial {i} dist={:.3e}", r.distance);
            }
            if let Some(r) = r_m2 {
                assert!(r.distance < eps, "m2 trial {i} dist={:.3e}", r.distance);
            }
        }
        eprintln!("\n  TOTAL  single={total_single:.0}ms  m1_relaxed={total_m1_relaxed:.0}ms ({:.2}×)  m2_strict={total_m2_strict:.0}ms ({:.2}×)",
            total_single/total_m1_relaxed, total_single/total_m2_strict);
        eprintln!("  wins:  m1_relaxed {wins_m1}/{n}   m2_strict {wins_m2}/{n}");
    }

    /// Z1 det-phase filter test: with various allowed-d_R sets, see how
    /// many prefixes pass the filter and how the dispatcher does.
    #[test]
    #[ignore = "slow diagnostic; run with --ignored"]
    fn z1_dc_dr_filter() {
        let theta = 0.3_f64;
        let target: Mat2 = [
            [Complex64::from_polar(1.0, -theta / 2.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::from_polar(1.0, theta / 2.0)],
        ];
        let eps = 1e-3_f64;

        // d_R distribution on L_m^Q for this target.
        let d_target = det_phase_of(&target);
        for m in [1u32, 2, 3] {
            let prefixes = build_fgkm_prefix_set(m);
            let mut hist = [0u64; 16];
            for u_l in prefixes.iter() {
                let d_l = det_phase_of(&u_l.to_float());
                let d_r = ((d_target as i32 - d_l as i32).rem_euclid(16)) as u32;
                hist[d_r as usize] += 1;
            }
            let mut s = format!("m={m} d_R hist (target d={d_target}):");
            for (d, c) in hist.iter().enumerate() {
                if *c > 0 {
                    s.push_str(&format!("  d_R={d}:{c}"));
                }
            }
            eprintln!("{s}");
        }
        eprintln!();

        // Try several filter configurations.
        let configs: &[(u32, &[u32], &str)] = &[
            (1, &[], "no filter"),
            (1, &[0], "strict d_R=0"),
            (1, &[0, 1, 15], "relaxed |d_R|≤1"),
            (1, &[0, 1, 2, 14, 15], "relaxed |d_R|≤2"),
            (2, &[], "no filter"),
            (2, &[0], "strict d_R=0"),
            (2, &[0, 1, 15], "relaxed |d_R|≤1"),
            (2, &[0, 1, 2, 14, 15], "relaxed |d_R|≤2"),
            (3, &[], "no filter"),
            (3, &[0], "strict d_R=0"),
            (3, &[0, 1, 15], "relaxed |d_R|≤1"),
            (3, &[0, 1, 2, 14, 15], "relaxed |d_R|≤2"),
        ];
        for (m, filter, label) in configs {
            let synth = SynthesizerQ::new(eps)
                .with_max_lde(15)
                .with_prefix_split_m(*m)
                .with_inner_det_phase_filter(filter.to_vec());
            let t0 = std::time::Instant::now();
            let r = synth.synthesize(target);
            let dt = t0.elapsed();
            let l_size = build_fgkm_prefix_set(*m).len();
            let n_pass = if filter.is_empty() {
                l_size as u64
            } else {
                let prefixes = build_fgkm_prefix_set(*m);
                prefixes.iter().filter(|u| {
                    let d_l = det_phase_of(&u.to_float());
                    let d_r = ((d_target as i32 - d_l as i32).rem_euclid(16)) as u32;
                    filter.contains(&d_r)
                }).count() as u64
            };
            eprintln!(
                "  m={m} {label:<22} pass={n_pass:>5}/{l_size:<6}  lde={:?}  t={:>7.0}ms",
                r.as_ref().map(|r| r.lde),
                dt.as_secs_f64() * 1000.0
            );
        }
    }

    #[test]
    #[ignore = "slow diagnostic; run with --ignored"]
    fn z1_dc_smoke_rz_eps_1e_3() {
        let theta = 0.3_f64;
        let target: Mat2 = [
            [Complex64::from_polar(1.0, -theta / 2.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::from_polar(1.0, theta / 2.0)],
        ];
        let eps = 1e-3_f64;

        // Single-search baseline.
        let synth_single = SynthesizerQ::new(eps).with_max_lde(15);
        let t0 = std::time::Instant::now();
        let r_single = synth_single.synthesize(target);
        let t_single = t0.elapsed();
        eprintln!(
            "single: lde={:?} dist={:?} t={:.1}ms",
            r_single.as_ref().map(|r| r.lde),
            r_single.as_ref().map(|r| r.distance),
            t_single.as_secs_f64() * 1000.0
        );
        assert!(r_single.is_some());

        // D&C across several m values to characterize per-prefix cost.
        for m in [1u32, 2, 3] {
            let synth_dc = SynthesizerQ::new(eps).with_max_lde(15).with_prefix_split_m(m);
            let t1 = std::time::Instant::now();
            let r_dc = synth_dc.synthesize(target);
            let t_dc = t1.elapsed();
            let l_size = build_fgkm_prefix_set(m).len();
            let per_prefix_us = t_dc.as_secs_f64() * 1e6 / (l_size as f64);
            eprintln!(
                "  d&c m={m}: |L|={l_size:>6}  lde={:?}  t={:.1}ms  per-prefix={per_prefix_us:.0}μs",
                r_dc.as_ref().map(|r| r.lde),
                t_dc.as_secs_f64() * 1000.0
            );
            assert!(r_dc.is_some(), "D&C m={m} should also find a solution");
        }
    }

    #[test]
    fn auto_defaults_at_various_eps() {
        // Default at ε=1e-6: D&C with m=1, |d_R|≤1 (relaxed filter).
        let s = SynthesizerQ::new(1e-6);
        assert_eq!(s.prefix_split_m, Some(1));
        assert_eq!(s.inner_det_phase_filter, vec![0u32, 1, 15]);

        // Default at ε ≤ 1e-7: D&C with m=2, d_R=0 (strict filter) —
        // empirically faster + better lde quality at this depth.
        let s7 = SynthesizerQ::new(1e-7);
        assert_eq!(s7.prefix_split_m, Some(2));
        assert_eq!(s7.inner_det_phase_filter, vec![0u32]);
        assert_eq!(s7.max_lde, 35, "max_lde should auto-bump at ε ≤ 1e-7");

        let s8 = SynthesizerQ::new(1e-8);
        assert_eq!(s8.prefix_split_m, Some(2));
        assert_eq!(s8.inner_det_phase_filter, vec![0u32]);

        // Default at moderate ε: single search.
        let s3 = SynthesizerQ::new(1e-3);
        assert_eq!(s3.prefix_split_m, None);
        assert_eq!(s3.inner_det_phase_filter, Vec::<u32>::new());
        assert_eq!(s3.max_lde, 30);

        // f64 GS is on at moderate ε but auto-disabled at ε ≤ 1e-8
        // (where f64's 2-bit precision margin causes ladder thrashing).
        for &eps in &[1e-3, 1e-4, 1e-5, 1e-6, 1e-7_f64] {
            assert!(SynthesizerQ::new(eps).use_f64_gs, "f64 default should be on at ε={eps:.0e}");
        }
        let eps = 1e-8_f64;
        assert!(!SynthesizerQ::new(eps).use_f64_gs, "f64 default should be OFF at ε={eps:.0e}");

        // Manual override still works.
        let s_override = SynthesizerQ::new(1e-7).with_prefix_split_m(1).with_inner_det_phase_filter(vec![0, 1, 15]);
        assert_eq!(s_override.prefix_split_m, Some(1));
        assert_eq!(s_override.inner_det_phase_filter, vec![0u32, 1, 15]);
        let s_no_f64 = SynthesizerQ::new(1e-3).with_f64_gs(false);
        assert!(!s_no_f64.use_f64_gs);

        // BKZ-4 default: on at ε ≤ 1e-7, off above.
        for &eps in &[1e-3, 1e-4, 1e-5, 1e-6_f64] {
            assert_eq!(SynthesizerQ::new(eps).bkz_block_size, 0,
                "BKZ default should be 0 at ε={eps:.0e}");
        }
        for &eps in &[1e-7, 1e-8_f64] {
            assert_eq!(SynthesizerQ::new(eps).bkz_block_size, 4,
                "BKZ default should be 4 at ε={eps:.0e}");
        }
    }

    #[test]
    fn synthesize_identity_at_k_0() {
        let one = Complex64::new(1.0, 0.0);
        let zero = Complex64::new(0.0, 0.0);
        let target = complex_target([[one, zero], [zero, one]]);
        let synth = SynthesizerQ::new(1e-7);
        let result = synth.synthesize(target).expect("identity should synthesize");
        assert_eq!(result.lde, 0, "identity should be at k=0");
        assert!(result.distance < 1e-7);
    }

    #[test]
    fn synthesize_q_gate() {
        let q = U2Q::q();
        let target = q.to_float();
        let synth = SynthesizerQ::new(1e-7);
        let result = synth.synthesize(target).expect("Q should synthesize");
        assert_eq!(result.lde, 0, "Q should be found at k=0");
        assert!(result.distance < 1e-7);
        // The synthesized gate string, when applied, should give back Q.
        assert!(result.gates.is_some());
    }

    #[test]
    fn synthesize_t_gate() {
        let t = U2Q::t();
        let target = t.to_float();
        let synth = SynthesizerQ::new(1e-7);
        let result = synth.synthesize(target).expect("T should synthesize");
        assert_eq!(result.lde, 0, "T should be found at k=0");
        assert!(result.distance < 1e-7);
    }

    #[test]
    fn synthesize_hqh() {
        let hqh: U2Q = U2Q::h() * U2Q::q() * U2Q::h();
        let target = hqh.to_float();
        let synth = SynthesizerQ::new(1e-7);
        let result = synth.synthesize(target).expect("HQH should synthesize");
        // HQH has k=2 (1 from each H).
        assert_eq!(result.lde, 2);
        assert!(result.distance < 1e-7);
    }

    #[test]
    fn synthesize_qhq() {
        let qhq: U2Q = U2Q::q() * U2Q::h() * U2Q::q();
        let target = qhq.to_float();
        let synth = SynthesizerQ::new(1e-7);
        let result = synth.synthesize(target).expect("QHQ should synthesize");
        assert_eq!(result.lde, 1);
        assert!(result.distance < 1e-7);
    }

    #[test]
    fn synthesize_h_gate() {
        // H has k=1 (one H gate). Should be found at k=1.
        let h = U2Q::h();
        let target = h.to_float();
        let synth = SynthesizerQ::new(1e-7);
        let result = synth.synthesize(target).expect("H should synthesize");
        assert_eq!(result.lde, 1);
        assert!(result.distance < 1e-7);
    }

    #[test]
    fn synthesize_returns_none_when_unreachable() {
        // Target Rx(π/16) — angle isn't a multiple of π/8, so the closest
        // Clifford+√T circuit at any small k is bounded away from it. With
        // ε=1e-7 (tight) and max_lde=2 (so the test stays under a second),
        // should return None.
        let theta = PI / 16.0;
        let c = (theta / 2.0).cos();
        let s = (theta / 2.0).sin();
        let i = Complex64::new(0.0, 1.0);
        let target: Mat2 = [
            [Complex64::new(c, 0.0), -i * s],
            [-i * s, Complex64::new(c, 0.0)],
        ];
        let synth = SynthesizerQ::new(1e-8).with_optimize_cost(false).with_max_lde(2);
        let result = synth.synthesize(target);
        assert!(result.is_none(),
            "Rx(π/16) should not be reachable in Clifford+√T at k≤2 with ε=1e-8");
    }

    #[test]
    fn synthesize_approximation_with_loose_epsilon() {
        // For Rx(π/16) at LOOSE ε, the synthesizer should find a closeby
        // approximation at small k. Tests the "approximate synthesis" path.
        let theta = PI / 16.0;
        let c = (theta / 2.0).cos();
        let s = (theta / 2.0).sin();
        let i = Complex64::new(0.0, 1.0);
        let target: Mat2 = [
            [Complex64::new(c, 0.0), -i * s],
            [-i * s, Complex64::new(c, 0.0)],
        ];
        let synth = SynthesizerQ::new(0.3).with_max_lde(2);  // very loose
        let result = synth.synthesize(target);
        assert!(result.is_some(), "loose ε should find an approximation");
        let r = result.unwrap();
        assert!(r.distance < 0.3);
    }

    #[test]
    fn synthesized_gate_string_roundtrip() {
        // For each of several Clifford+√T targets, the gate string from
        // the synthesizer should reconstruct (via gates_to_u2q) to a
        // U2Q close to the target.
        use crate::matrix::u2::U2Q;
        let targets: Vec<U2Q> = vec![
            U2Q::q(),
            U2Q::t(),
            U2Q::q() * U2Q::q(),  // = T
            U2Q::h() * U2Q::q() * U2Q::h(),
            U2Q::q() * U2Q::h() * U2Q::q(),
        ];
        // First-hit: this tests gate-string reconstruction, not the
        // cost-optimal pipeline.
        let synth = SynthesizerQ::new(1e-7).with_optimize_cost(false);
        for u in targets {
            let target = u.to_float();
            let result = synth.synthesize(target).expect("should synthesize");
            let gates = result.gates.expect("should have gate string");
            // Reconstruct via gates_to_u2q.
            let mut rebuilt = U2Q::eye();
            for c in gates.chars() {
                rebuilt = rebuilt * match c {
                    'H' => U2Q::h(),
                    'S' => U2Q::s(),
                    'T' => U2Q::t(),
                    'Q' => U2Q::q(),
                    'X' => U2Q::x(),
                    'Y' => U2Q::y(),
                    'Z' => U2Q::z(),
                    _ => panic!("unexpected gate {c}"),
                };
            }
            let dist = diamond_distance_float(&rebuilt.to_float(), &target);
            assert!(dist < 1e-7,
                "round-trip dist for gate string \"{gates}\" = {dist:.3e}");
        }
    }

    /// End-to-end deep-ε test: Rz(0.3) at ε=1e-3. Behind `#[ignore]` because
    /// it can take minutes — the lattice search at k=10 needs ~1G SE leaves.
    /// Run with `cargo test --release --lib synthesize_rz_eps_1e_3 --
    /// --ignored --nocapture`.
    #[test]
    #[ignore]
    fn synthesize_rz_eps_1e_3() {
        let theta = 0.3_f64;
        let target: Mat2 = [
            [Complex64::from_polar(1.0, -theta / 2.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::from_polar(1.0, theta / 2.0)],
        ];
        let synth = SynthesizerQ::new(1e-3).with_optimize_cost(false).with_max_lde(15);
        let t0 = std::time::Instant::now();
        let result = synth.synthesize(target).expect("Rz(0.3) at ε=1e-3 should land");
        eprintln!(
            "Rz(0.3) at ε=1e-3: lde={} dist={:.3e} t={:?}",
            result.lde, result.distance, t0.elapsed()
        );
        assert!(result.distance < 1e-3);
        // Upper bound from 8D Clifford+T: lde=28. Z[ζ_16] should land much
        // smaller (~10) since `T = QQ` doubles the effective denominator
        // factor in the 8D path.
        assert!(result.lde <= 14,
            "expected lde ≤ 14 (8D Clifford+T is 28); got {}", result.lde);
    }

    #[test]
    fn synthesize_rz_via_lattice_backend() {
        // Rz(0.3) at ε=0.05 is unreachable at k ≤ 4 (brute regime), so
        // forcing min_lde > BRUTE_LIMIT exercises the LLL+SE lattice path.
        let theta = 0.3_f64;
        let target: Mat2 = [
            [Complex64::from_polar(1.0, -theta / 2.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::from_polar(1.0, theta / 2.0)],
        ];
        let synth = SynthesizerQ::new(0.05)
            .with_min_lde(BRUTE_LIMIT + 1)
            .with_max_lde(12);
        let result = synth.synthesize(target).expect("Rz(0.3) at ε=0.05 should land");
        assert!(result.lde > BRUTE_LIMIT,
            "expected lattice backend (k > {BRUTE_LIMIT}), got k={}", result.lde);
        assert!(result.distance < 0.05,
            "diamond distance {:.3e} exceeds ε=0.05", result.distance);
        assert!(result.gates.is_some());
    }

    /// Census: how much of `build_fgkm_prefix_set(m)` is right-coset duplicate work
    /// under ⟨S,X⟩, and how much survives the d_R filters. Soundness
    /// premise: (U_L·C)·U_R = U_L·(C·U_R) on the same shell — the rep's
    /// search covers every mate's solutions with identical totals. The
    /// list member matched to u·C is ζ^p·(u·C), so mates' d_R differ by
    /// arbitrary EVEN offsets (the argument is d_R-agnostic).
    /// Run: `cargo test --release --lib zeta_coset_census -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn zeta_coset_census() {
        use std::collections::{HashMap, HashSet};

        // lde-0 Clifford subgroup as U2Q (rebuilt from table names, the
        // same route build_fgkm_prefix_set_inner uses for its Clifford suffixes —
        // shared with the production orbit table).
        let lde0 = lde0_cliffords_q();
        for c in &lde0 {
            assert_eq!(c.k, 0, "lde-0 Clifford has k != 0 as U2Q");
        }

        for m in 1..=3u32 {
            let prefixes = build_fgkm_prefix_set(m);
            let n = prefixes.len();
            let key_of: Vec<[i64; 8]> = prefixes.iter().map(canonical_key_q).collect();
            let idx_of: HashMap<[i64; 8], usize> =
                key_of.iter().enumerate().map(|(i, k)| (*k, i)).collect();

            // Right-coset orbits: orbit(u) = {u·c} is exactly the coset
            // u·⟨S,X⟩, so one multiplication sweep finds the whole orbit.
            // Orbit id = min member index. `missing` counts mates whose
            // canonical key is absent from L (float-key rounding or a
            // genuine coverage hole — must stay ~0 for the dedup claim).
            let mut orbit_id: Vec<usize> = (0..n).collect();
            let mut missing = 0usize;
            for i in 0..n {
                let mut mn = i;
                for c in &lde0 {
                    let key = canonical_key_q(&(prefixes[i] * *c));
                    match idx_of.get(&key) {
                        Some(&j) => mn = mn.min(j),
                        None => missing += 1,
                    }
                }
                orbit_id[i] = mn;
            }
            let orbits: HashSet<usize> = orbit_id.iter().copied().collect();
            eprintln!(
                "\nm={m}: |L|={n}  orbits={}  full-orbit ratio={:.2}x  (missing mate keys: {missing})",
                orbits.len(),
                n as f64 / orbits.len() as f64
            );

            // Self-consistency with the production dedup: the cached
            // orbit table the searches use must be IDENTICAL to the
            // census's locally computed linking (gate 5).
            assert_eq!(
                orbit_id,
                *build_fgkm_prefix_orbits(m).as_ref(),
                "production build_fgkm_prefix_orbits({m}) diverges from census linking"
            );

            // d_R-respecting census per filter. For each d_target the
            // usable set is {u : (d_target − d_L) mod 16 ∈ filter}; the
            // dedup that survives = |usable| / |orbits among usable|.
            // `classes` additionally splits orbits by the unreduced k —
            // the PRODUCTION dedup grouping (`build_fgkm_prefix_coset_keys`;
            // cross-k orbit links are float-real but their coverage is
            // asymmetric, so the implementation keeps one rep per
            // (orbit, k) ∩ usable): the classes column is the actual
            // achieved reduction.
            let d_l: Vec<u32> = prefixes
                .iter()
                .map(|u| det_phase_of(&u.to_float()))
                .collect();
            let coset_keys = build_fgkm_prefix_coset_keys(m);
            for (fname, filter) in [
                ("strict [0]   (m=2 1st-hit default)", vec![0u32]),
                ("relaxed [0,1,15] (m=1 default)", vec![0u32, 1, 15]),
                ("OPEN (optimal_open_dr_filter, prod at eps<=1e-5)", vec![]),
            ] {
                let mut tot_usable = 0usize;
                let mut tot_orbits = 0usize;
                let mut tot_classes = 0usize;
                let mut per_d: Vec<(u32, usize, usize)> = Vec::new();
                for d_target in 0..16u32 {
                    let usable: Vec<usize> = (0..n)
                        .filter(|&i| {
                            if filter.is_empty() {
                                return true;
                            }
                            let d_r = ((d_target as i32 - d_l[i] as i32)
                                .rem_euclid(16)) as u32;
                            filter.contains(&d_r)
                        })
                        .collect();
                    let uorb: HashSet<usize> =
                        usable.iter().map(|&i| orbit_id[i]).collect();
                    let uclass: HashSet<(usize, u32)> =
                        usable.iter().map(|&i| coset_keys[i]).collect();
                    tot_usable += usable.len();
                    tot_orbits += uorb.len();
                    tot_classes += uclass.len();
                    per_d.push((d_target, usable.len(), uorb.len()));
                }
                eprintln!(
                    "  filter {fname}: avg usable {:.1} -> orbits {:.1} (dedup {:.2}x) | (orbit,k) classes {:.1} (PROD dedup {:.2}x)",
                    tot_usable as f64 / 16.0,
                    tot_orbits as f64 / 16.0,
                    tot_usable as f64 / tot_orbits.max(1) as f64,
                    tot_classes as f64 / 16.0,
                    tot_usable as f64 / tot_classes.max(1) as f64
                );
                if m == 2 {
                    let row: Vec<String> = per_d
                        .iter()
                        .map(|(d, u, o)| format!("d{d}:{u}/{o}"))
                        .collect();
                    eprintln!("    per-d usable/orbits: {}", row.join(" "));
                }
            }
        }
    }

    /// Ring-exact soundness pin: every pair sharing a dedup class
    /// (orbit, k) satisfies `u_i = ζ^p · u_rep · C` for some lde-0 C —
    /// exactly the relation the dedup's coverage argument consumes.
    #[test]
    fn zeta_coset_orbits_sound() {
        let lde0 = lde0_cliffords_q();
        for c in &lde0 {
            assert_eq!(c.k, 0, "lde-0 Clifford has k != 0 as U2Q");
        }
        let scale = |u: &U2Q, z: ZZeta| -> U2Q {
            U2Q::new(z * u.u11, z * u.u12, z * u.u21, z * u.u22, u.k)
        };
        for m in 1..=2u32 {
            let prefixes = build_fgkm_prefix_set(m);
            let keys = build_fgkm_prefix_coset_keys(m);
            assert_eq!(prefixes.len(), keys.len());
            // First member per (orbit, k) class = the class rep ties
            // resolve to in production when costs tie.
            let mut rep_of: HashMap<(usize, u32), usize> = HashMap::new();
            let mut classes = 0usize;
            for (i, u) in prefixes.iter().enumerate() {
                assert!(keys[i].0 <= i, "orbit id must be a min index (m={m}, i={i})");
                assert_eq!(keys[i].1, u.k, "class k must be the prefix k");
                let rep = *rep_of.entry(keys[i]).or_insert_with(|| {
                    classes += 1;
                    i
                });
                if rep == i {
                    continue;
                }
                let r = &prefixes[rep];
                let mate = lde0.iter().any(|c| {
                    let rc = *r * *c;
                    (0..16u32).any(|p| scale(&rc, zeta_16_pow(p)) == *u)
                });
                assert!(
                    mate,
                    "class-mates not ring-exact coset mates (m={m}, i={i}, rep={rep})"
                );
            }
            assert!(
                classes < prefixes.len(),
                "coset dedup must merge something at m={m}"
            );
        }
    }

    /// Coset-regression probe (ignored): probe_t_vs_qt target 0
    /// (θ=2.37 φ=5.73 λ=3.33, seed 12648430) at ε=1e-6 optimal w2 —
    /// coset-off finds cost 52.5, coset-on falls to the T baseline 53.
    /// Runs ONE mode per process (env LazyLock): set the mode via the
    /// test name. Prints the enum trace for diffing.
    /// Run: cargo test --release --lib probe_zeta_coset_t0_off -- --ignored --nocapture
    #[test]
    #[ignore]
    fn probe_zeta_coset_t0_off() {
        probe_zeta_coset_target(0, 1e-6, "0");
    }
    #[test]
    #[ignore]
    fn probe_zeta_coset_t0_on() {
        probe_zeta_coset_target(0, 1e-6, "1");
    }
    /// 1e-8 flip probe: probe_t_vs_qt target 6 (θ=1.80 φ=0.59 λ=1.62)
    /// — coset-off screen finds lde=24 (cost 73.5), coset-on drifts to
    /// the lde-78 fallback (cost 78).
    #[test]
    #[ignore]
    fn probe_zeta_coset_t6_1e8_off() {
        probe_zeta_coset_target(6, 1e-8, "0");
    }
    #[test]
    #[ignore]
    fn probe_zeta_coset_t6_1e8_on() {
        probe_zeta_coset_target(6, 1e-8, "1");
    }
    fn probe_zeta_coset_target(index: usize, eps: f64, coset: &str) {
        unsafe {
            std::env::set_var("CYCLOSYNTH_ZETA_COSET", coset);
            std::env::set_var("CYCLOSYNTH_TRACE", "1");
        }
        // SplitMix64 target gen, first triple of seed 12648430
        // (probe_t_vs_qt's Xs) — replicated from tests/qt_guard_1e5.rs.
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
        let mut rng = Xs(12648430);
        let mut tpl = (0.0, 0.0, 0.0);
        for _ in 0..=index {
            tpl = (
                rng.range(0.2, PI - 0.2),
                rng.range(0.1, 2.0 * PI - 0.1),
                rng.range(0.1, 2.0 * PI - 0.1),
            );
        }
        let (th, ph, la) = tpl;
        eprintln!("[t{index}] θ={th:.3} φ={ph:.3} λ={la:.3} ε={eps:e} coset={coset}");
        let (c, s) = ((th / 2.0).cos(), (th / 2.0).sin());
        let eilam = Complex64::from_polar(1.0, la);
        let eiphi = Complex64::from_polar(1.0, ph);
        let g = Complex64::from_polar(1.0, -(ph + la) / 2.0);
        let target: Mat2 = [
            [Complex64::new(c, 0.0) * g, -eilam * s * g],
            [eiphi * s * g, eiphi * eilam * Complex64::new(c, 0.0) * g],
        ];
        let r = SynthesizerQ::new(eps)
            .with_optimize_cost(true)
            .with_optimal_lde_window(2)
            .synthesize(target);
        match r {
            Some(r) => {
                let g = r.gates.as_deref().unwrap_or("");
                let (t, q) = gates_tq(g);
                eprintln!(
                    "[t{index}] RESULT lde={} T={t} Q={q} cost={} dist={:.3e}",
                    r.lde,
                    t as f64 + 3.5 * q as f64,
                    r.distance
                );
            }
            None => eprintln!("[t{index}] RESULT NONE"),
        }
    }

    /// Is the strict-filter ([0]) deep-ε screen blind to non-class-0
    /// solutions? Re-running tie targets with the relaxed filter should
    /// collapse first-hit lde if so.
    /// Run: cargo test --release --lib h1_dr_filter_blindness -- --ignored --nocapture
    #[test]
    #[ignore]
    fn h1_dr_filter_blindness() {
        fn xorshift64(s: &mut u64) -> u64 { *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; *s }
        fn rand_angle(s: &mut u64) -> f64 {
            let b = xorshift64(s) >> 11;
            (b as f64) / ((1u64 << 53) as f64) * 2.0 * std::f64::consts::PI
        }
        let mut state: u64 = 12648430 | 1;
        // probe_t_vs_qt target gen: theta in (0.2, PI-0.2), phi/lambda in (0.1, 2PI-0.1)
        let mut angles = Vec::new();
        for _ in 0..2 {
            let t = 0.2 + rand_angle(&mut state) / (2.0 * std::f64::consts::PI)
                * (std::f64::consts::PI - 0.4);
            let p = 0.1 + rand_angle(&mut state) / (2.0 * std::f64::consts::PI)
                * (2.0 * std::f64::consts::PI - 0.2);
            let l = 0.1 + rand_angle(&mut state) / (2.0 * std::f64::consts::PI)
                * (2.0 * std::f64::consts::PI - 0.2);
            angles.push((t, p, l));
        }
        eprintln!("targets (must match probe rows 0,1: θ=2.37/1.17): {angles:?}");
        for (i, &(t, p, l)) in angles.iter().enumerate() {
            let ct = (t / 2.0).cos();
            let st = (t / 2.0).sin();
            let gp = Complex64::from_polar(1.0, -(p + l) / 2.0);
            let target: Mat2 = [
                [gp * Complex64::new(ct, 0.0), gp * (-Complex64::from_polar(st, l))],
                [gp * Complex64::from_polar(st, p), gp * Complex64::from_polar(ct, p + l)],
            ];
            for (label, filt) in [("strict[0]", vec![0u32]), ("relaxed[0,1,15]", vec![0u32, 1, 15])] {
                let synth = SynthesizerQ::new(1e-8).with_inner_det_phase_filter(filt);
                let t0 = std::time::Instant::now();
                let r = synth.synthesize(target);
                match r {
                    Some(r) => {
                        let g = r.gates.as_deref().unwrap_or("");
                        let (tc, qc) = gates_tq(g);
                        eprintln!(
                            "target {i} {label}: lde={} T={tc} Q={qc} cost={} dist={:.2e} t={:.1}s",
                            r.lde,
                            gates_cost(g, 7) as f64 / 2.0,
                            r.distance,
                            t0.elapsed().as_secs_f64()
                        );
                    }
                    None => eprintln!("target {i} {label}: NONE t={:.1}s", t0.elapsed().as_secs_f64()),
                }
            }
        }
    }
