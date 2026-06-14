//! Research probes (all #[ignore]): census/diagnostic harnesses kept
//! runnable but out of the unit-test file. Run individually, e.g.
//! `cargo test --release --lib fgkm_prefix_split_cost_ratio -- --ignored --nocapture`.

#![allow(unused_imports)]
use super::*; // the tests module
use super::super::*; // clifford_sqrt_t internals
use crate::synthesis::distance::{diamond_distance_float, Mat2};
use num_complex::Complex64;
use std::f64::consts::PI;

    /// Back-of-envelope: under cost model C(k) = c·α^k, the D&C cost
    /// ratio (vs single search at lde_total) is
    ///   S(m, α) = Σ_k count(m, k) / α^k
    /// and is independent of lde_total (the c·α^{lde_total} term cancels).
    /// D&C wins at m when S(m, α) < 1.
    /// Census probe (no assertions): run with `-- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn fgkm_prefix_split_cost_ratio() {
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
    /// Census probe (no assertions): run with `-- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn fgkm_prefix_k_distribution() {
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

    /// Z1 det-phase filter test: with various allowed-d_R sets, see how
    /// many prefixes pass the filter and how the dispatcher does.
    #[test]
    #[ignore = "slow diagnostic; run with --ignored"]
    fn det_phase_filter_coverage() {
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

    /// Coset-regression probe (ignored): probe_omega_vs_zeta target 0
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

    /// Coset-ON counterpart of [`probe_zeta_coset_t0_off`] (same target 0,
    /// ε=1e-6): expected to drift from cost 52.5 up to the T baseline 53.
    /// Run: cargo test --release --lib probe_zeta_coset_t0_on -- --ignored --nocapture
    #[test]
    #[ignore]
    fn probe_zeta_coset_t0_on() {
        probe_zeta_coset_target(0, 1e-6, "1");
    }

    /// 1e-8 flip probe: probe_omega_vs_zeta target 6 (θ=1.80 φ=0.59 λ=1.62)
    /// — coset-off screen finds lde=24 (cost 73.5), coset-on drifts to
    /// the lde-78 fallback (cost 78).
    #[test]
    #[ignore]
    fn probe_zeta_coset_t6_1e8_off() {
        probe_zeta_coset_target(6, 1e-8, "0");
    }

    /// Coset-ON counterpart of [`probe_zeta_coset_t6_1e8_off`] (same target 6,
    /// ε=1e-8): expected to drift from the lde=24 hit (cost 73.5) to the
    /// lde-78 fallback (cost 78).
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
        // (probe_omega_vs_zeta's Xs) — replicated from tests/qt_guard_1e5.rs.
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
        // probe_omega_vs_zeta target gen: theta in (0.2, PI-0.2), phi/lambda in (0.1, 2PI-0.1)
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