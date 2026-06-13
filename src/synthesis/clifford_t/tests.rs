#[path = "probes.rs"]
mod probes;

    use super::*;
    use crate::synthesis::distance::diamond_distance_float;
    use std::{f64::consts::{FRAC_1_SQRT_2, PI}};

    fn rz(theta: Float) -> Mat2 {
        [
            [Complex::from_polar(1., -theta / 2.), Complex::new(0., 0.)],
            [Complex::new(0., 0.), Complex::from_polar(1., theta / 2.)],
        ]
    }

    fn ry(theta: Float) -> Mat2 {
        let c = (theta / 2.).cos();
        let s = (theta / 2.).sin();
        [
            [Complex::new(c, 0.), Complex::new(-s, 0.)],
            [Complex::new(s, 0.), Complex::new(c, 0.)],
        ]
    }

    fn mat_mul(a: Mat2, b: Mat2) -> Mat2 {
        [
            [a[0][0]*b[0][0] + a[0][1]*b[1][0], a[0][0]*b[0][1] + a[0][1]*b[1][1]],
            [a[1][0]*b[0][0] + a[1][1]*b[1][0], a[1][0]*b[0][1] + a[1][1]*b[1][1]],
        ]
    }

    /// Same convention as bin/time_synthesis: U3(a,b,c) = Rz(a)·Ry(b)·Rz(c).
    fn u3(a: Float, b: Float, c: Float) -> Mat2 {
        mat_mul(mat_mul(rz(a), ry(b)), rz(c))
    }

    fn check_result(result: &SynthResultT, _target: &Mat2, eps: Float) {
        assert!(
            result.distance < eps,
            "distance={:.6e} ≥ epsilon={:.6e}",
            result.distance, eps
        );
    }

    /// Re-build a U2T from the synthesized gate string by parsing left-to-right.
    fn gates_to_u2t_verify(gate_str: &str) -> crate::matrix::U2T {
        use crate::matrix::U2T;
        let mut u = U2T::eye();
        for ch in gate_str.chars() {
            let g = match ch {
                'H' => U2T::h(),
                'S' => U2T::s(),
                'T' => U2T::t(),
                'Z' => U2T::z(),
                'X' => U2T::x(),
                'Y' => U2T::y(),
                'I' => U2T::eye(),
                _ => panic!("unexpected gate char: {ch}"),
            };
            u = u * g;
        }
        u
    }

    /// End-to-end correctness verification: synthesize, then independently
    /// re-evaluate the gate string and confirm the result still satisfies the
    /// approximation bound. Validates that:
    ///   1. result.distance < eps (reported distance is below threshold)
    ///   2. The gate string parses to a U2T whose lde matches result.lde
    ///   3. Re-evaluated diamond distance to target matches result.distance
    ///   4. T-count of the gate string is consistent with the lde
    fn verify_synthesis_round_trip(target: &Mat2, eps: Float, label: &str) {
        // max_lde generously oversized so very tight ε (1e-5+) has room.
        let synth = SynthesizerT::new(eps).with_max_lde(80);
        let result = synth
            .synthesize(*target)
            .unwrap_or_else(|| panic!("{label}: synthesis returned None"));

        // Check 1: reported distance under threshold
        assert!(
            result.distance < eps,
            "{label}: result.distance={:.6e} ≥ eps={:.6e}",
            result.distance,
            eps
        );

        // Check 2: gate string round-trips. Re-build the U2T from the gate
        // string and verify the diamond distance is the same as reported.
        let gates = result
            .gates
            .as_ref()
            .unwrap_or_else(|| panic!("{label}: result.gates is None"));
        let rebuilt = gates_to_u2t_verify(gates);
        let rebuilt_float = rebuilt.to_float();
        let recomputed_dist = diamond_distance_float(&rebuilt_float, target);
        assert!(
            recomputed_dist < eps,
            "{label}: re-evaluated distance={:.6e} ≥ eps={:.6e} (gate string does not approximate target)",
            recomputed_dist,
            eps
        );
        // Reported and rebuilt distances should agree to FP precision (the
        // synth doesn't round-trip through the gate string internally, so
        // small rounding from to_float()/diamond_distance_float() is expected,
        // but they should agree to ~1e-12).
        let dist_consistency = (recomputed_dist - result.distance).abs();
        // Tolerance: diamond distance involves catastrophic cancellation in
        // `1 − |tr(U·V†)|²/4` when U is close to V. Plus the rebuilt path
        // accumulates f64 error through ~n_gates U2T products. Empirically
        // ~n_gates · 1e-12 covers the round-trip noise even for 200+ gate
        // sequences at ε=1e-7. Floor at 1e-10 for short sequences; the
        // tolerance must remain << ε so the "within ε" guarantee isn't
        // compromised.
        let n_gates = result.gates.as_ref().map(|s| s.len()).unwrap_or(0) as f64;
        // Per-gate bound + floor. The `dist < ε` check above is the real
        // correctness gate; this consistency check is a self-sanity ratchet
        // against silent algorithmic divergence between synth.synthesize's
        // reported distance and the gate-replay distance.
        let tol = (n_gates * 5e-11).max(1e-9);
        assert!(
            dist_consistency < tol,
            "{label}: rebuilt distance ({:.6e}) differs from reported ({:.6e}) by {:e} (tol={:e}, gates_len={})",
            recomputed_dist,
            result.distance,
            dist_consistency,
            tol,
            n_gates as usize
        );

        // Check 3: T-count of the gate string. result.lde holds the
        // synthesizer's t-loop value (the *target* T-count for the search).
        // The actual gate string can have at most that many T gates.
        let t_count = gates.chars().filter(|&c| c == 'T').count() as u32;
        // We accept up to lde + a few (the BlochDecomposer can introduce
        // small constant overhead from final Clifford fixup).
        assert!(
            t_count <= result.lde + 8,
            "{label}: T-count={} far exceeds reported lde={}",
            t_count,
            result.lde
        );

        eprintln!(
            "[verify] {label}: lde={} dist={:.4e} (rebuilt: {:.4e}) T-count={} gates_len={} U2T_k={}",
            result.lde,
            result.distance,
            recomputed_dist,
            t_count,
            gates.len(),
            rebuilt.k
        );
    }

    #[test]
    fn verify_correctness_at_1e_3_rz_03() {
        verify_synthesis_round_trip(&rz(0.30), 1e-3, "Rz(0.30) @ 1e-3");
    }

    #[test]
    fn verify_correctness_at_1e_4_rz_03() {
        verify_synthesis_round_trip(&rz(0.30), 1e-4, "Rz(0.30) @ 1e-4");
    }

    #[test]
    fn verify_correctness_at_1e_4_rz_pi7() {
        verify_synthesis_round_trip(&rz(PI / 7.0), 1e-4, "Rz(π/7) @ 1e-4");
    }

    #[test]
    fn verify_correctness_at_1e_4_u3() {
        verify_synthesis_round_trip(&u3(0.3, 0.7, 1.2), 1e-4, "U3(0.3,0.7,1.2) @ 1e-4");
    }

    #[test]
    fn verify_correctness_at_1e_5_rz_03() {
        verify_synthesis_round_trip(&rz(0.30), 1e-5, "Rz(0.30) @ 1e-5");
    }

    /// Round-trip at ε=1e-7. Validates the L²-LLL backend at deeper ε.
    /// Fast (~40 ms) on `Rz(0.30)` after the post-Frobenius perf fixes.
    #[test]
    fn verify_correctness_at_1e_7_rz_03() {
        verify_synthesis_round_trip(&rz(0.30), 1e-7, "Rz(0.30) @ 1e-7");
    }

    /// Round-trip at ε=1e-7 on `Rz(π/7)` — the worst-case 1e-7 target in
    /// the bench (lde=70 vs typical 66). Slowest test in the suite (~2 s),
    /// kept in the default run because it's the only direct guard for the
    /// "outlier-target at deep ε" failure mode that motivated the
    /// MPFR-alignment / Frobenius-distance fixes.
    #[test]
    fn verify_correctness_at_1e_7_rz_pi7() {
        verify_synthesis_round_trip(&rz(PI / 7.0), 1e-7, "Rz(π/7) @ 1e-7");
    }

    #[test]
    fn test_synthesize_identity() {
        let id: Mat2 = [[Complex::new(1., 0.), Complex::new(0., 0.)], [Complex::new(0., 0.), Complex::new(1., 0.)]];
        // with_min_lde(0): identity is a Clifford with exact solution at lde=0.
        let synth = SynthesizerT::new(0.01).with_min_lde(0);
        let result = synth.synthesize(id).expect("Should synthesize identity");
        check_result(&result, &id, 0.01);
        assert_eq!(result.lde, 0, "Identity should have lde=0");
    }

    #[test]
    fn test_synthesize_s_gate() {
        let s: Mat2 = [
            [Complex::new(1., 0.), Complex::new(0., 0.)],
            [Complex::new(0., 0.), Complex::new(0., 1.)],
        ];
        // with_min_lde(0): S is a Clifford with exact solution at lde=0.
        let synth = SynthesizerT::new(0.01).with_min_lde(0);
        let result = synth.synthesize(s).expect("Should synthesize S");
        println!("{:?}", result.gates);
        check_result(&result, &s, 0.01);
        assert_eq!(result.lde, 0, "S is a Clifford, should need lde=0");
    }

    #[test]
    fn test_synthesize_h_gate() {
        let r = FRAC_1_SQRT_2 as Float;
        let h: Mat2 = [
            [Complex::new(r, 0.), Complex::new(r, 0.)],
            [Complex::new(r, 0.), Complex::new(-r, 0.)],
        ];
        let synth = SynthesizerT::new(0.01);
        let result = synth.synthesize(h).expect("Should synthesize H");
        check_result(&result, &h, 0.01);
    }

    #[test]
    fn test_synthesize_rz_small() {
        // Rz(π/4) = T gate, should need lde=1.
        let target = rz(PI as Float / 4.);
        let synth = SynthesizerT::new(0.01);
        let result = synth.synthesize(target).expect("Should synthesize Rz(π/4)");
        println!("{:?}", result.gates);

        check_result(&result, &target, 0.01);
    }

    #[test]
    fn test_synthesize_rz_moderate_1() {
        let target = rz(0.3);
        let synth = SynthesizerT::new(0.01);
        let result = synth.synthesize(target).expect("Should synthesize Rz(0.3)");
        println!("{:?}", result.gates);
        check_result(&result, &target, 0.01);
    }

    #[test]
    fn test_synthesize_rz_moderate_2() {
        let target = rz(1.34);
        let synth = SynthesizerT::new(0.01);
        let result = synth.synthesize(target).expect("Should synthesize Rz(1.34)");
        println!("{:?}", result.gates);
        check_result(&result, &target, 0.01);
    }

    #[test]
    fn test_synthesize_rz_hard_1() {
        // eps=0.01: needs t~26, DC kicks in at t>=17 (t'=t-17, t_inner=17).
        // Much faster than eps=0.001 which needs t~40.
        let target = rz(0.3);
        let synth = SynthesizerT::new(0.01);
        let result = synth.synthesize(target).expect("Should synthesize Rz(0.3) at eps=0.01");
        println!("{:?}", result.gates);
        check_result(&result, &target, 0.01);
    }

    #[test]
    fn test_synthesize_rz_hard_2() {
        let target = rz(1.34);
        let synth = SynthesizerT::new(0.001);
        let result = synth.synthesize(target).expect("Should synthesize Rz(1.34) at eps=0.01");
        println!("{:?}", result.gates);

        check_result(&result, &target, 0.01);
    }

    /// Stage-2 contract: `prefix_split_search`'s `budget_hit` is shared across the
    /// branch sweeps (OR semantics) and surfaces to the caller — the 2-pass
    /// requeue depends on it. With a 1-node SE budget every walk in BOTH
    /// sweeps trips immediately on an empty level → `(None, true)`; the
    /// same level at the production pass-1 caps completes exhaustively →
    /// `(None, false)`. Together these pin the budget-driven requeue
    /// signal through the two-sweep restructure.
    #[test]
    fn budget_hit_ors_across_sweeps() {
        // Rz(π/7) @ 1e-5 first hits around lde 51 — the DC band (t'=1 at
        // t=42) below it is a wide stretch of cheap empty levels.
        let target = rz(PI / 7.0);
        let eps = 1e-5_f64;
        let synth = SynthesizerT::new(eps);
        let raw_uv = unitary_to_uv(&target);
        let v = normalize4(raw_uv).unwrap();
        // Scan UP from the DC threshold for an EMPTY level (production
        // caps → (None, false), i.e. exhaustive) with surviving prefixes,
        // then verify that a 1-node SE budget on that level — which trips
        // every walk in BOTH sweeps at its first recurse-entry — surfaces
        // through the shared budget_hit as (None, true). The scan STOPS at
        // the first FOUND level: empty levels live below first-hit, and
        // climbing past it would build exponentially larger L_{t'} sets
        // for nothing.
        let mut verified = false;
        for t in 42..=52u32 {
            if optimal_t_prime(t, eps) == 0 {
                continue;
            }
            let (res, hit) = synth.prefix_split_search(&target, v, t, PASS1_CAP, PASS1_NODE_CAP);
            if res.is_some() {
                break; // first-hit reached; no empty levels above
            }
            assert!(!hit, "production caps should be exhaustive at lde={t}");
            let (res1, hit1) = synth.prefix_split_search(&target, v, t, u64::MAX, 1);
            assert!(res1.is_none(), "no solution reachable on a 1-node budget (lde={t})");
            if hit1 {
                verified = true;
                break;
            }
            // else: level had no surviving prefixes (odd-t' parity
            // wipeout) — no walk ran, keep scanning.
        }
        assert!(
            verified,
            "no empty dc level with surviving prefixes found below first-hit — \
             budget_hit OR-across-sweeps could not be exercised"
        );
    }

    /// Structural soundness of the right-coset dedup (stage 1, lever B1):
    /// every plain-dedup prefix must be reachable as `rep · c` for some
    /// kept representative `rep` and lde-0 Clifford `c` — i.e. the coset
    /// orbits of the kept reps COVER the full prefix set, so no subproblem
    /// is lost. Checked exactly (canonical-key equality, the same
    /// equivalence the production dedup uses) for t' = 1..6.
    #[test]
    fn coset_dedup_covers_all_prefixes() {
        for tp in 1..=6 {
            let plain = build_l_inner_with(tp, false);
            let coset = build_l_inner_with(tp, true);
            assert!(coset.len() < plain.len(), "t'={tp}: coset dedup removed nothing");
            let mut covered: std::collections::HashSet<[i64; 8]> =
                std::collections::HashSet::new();
            for u in &coset {
                for &ci in CLIFFORD_LDE0_IDX.iter() {
                    covered.insert(canonical_key(&(*u * CLIFFORD_TABLE_T[ci].1)));
                }
            }
            for (i, u) in plain.iter().enumerate() {
                assert!(
                    covered.contains(&canonical_key(u)),
                    "t'={tp}: plain prefix {i} not covered by any kept coset orbit"
                );
            }
        }
    }

    /// Test that DC (Algorithm 3.11) fires and finds a solution.
    /// Uses a tight eps where direct_search would hang but DC with MA prefixes
    /// and LLL/CVP inner search should terminate quickly.
    #[test]
    fn test_dc_fires_and_finds_solution() {
        // eps=0.01, Rz(0.3): DC fires at t>=17.  We go straight to t=20 to
        // ensure prefix_split_search is exercised (t'=4, t_inner=16, |L|~16).
        let target = rz(0.3);
        let eps = 0.01_f64;
        let synth = SynthesizerT::new(eps).with_max_lde(35);
        assert!(optimal_t_prime(20, eps) > 0, "DC should fire at t=20 for eps=0.01");
        let result = synth.synthesize(target).expect("Should find a solution");
        check_result(&result, &target, eps);
        // Verify that a solution was found via DC (lde > direct_limit)
        //println!("lde={}, dist={:.4e}", result.lde, result.distance);
    }

    /// Test that optimal_t_prime gives correct thresholds (Proposition 3.13).
    #[test]
    fn test_optimal_t_prime_thresholds() {
        // ε=0.1: threshold ≈ 8.3, so t'=0 for t<=8, t'>=1 for t>=9.
        assert_eq!(optimal_t_prime(8, 0.1), 0);
        assert!(optimal_t_prime(9, 0.1) >= 1);
        // ε=0.01: threshold ≈ 16.6, so t'=0 for t<=16, t'>=1 for t>=17.
        assert_eq!(optimal_t_prime(16, 0.01), 0);
        assert!(optimal_t_prime(17, 0.01) >= 1);
        // t_inner = t - t' should satisfy: t_inner <= threshold (i.e. t' >= t - threshold).
        for &eps in &[0.1_f64, 0.01, 0.001] {
            for t in 0u32..30 {
                let tp = optimal_t_prime(t, eps);
                let t_inner = t - tp;
                let threshold = (5.0 / 2.0) * (1.0 / eps).log2();
                // t_inner should be <= threshold (direct_search is cheap enough).
                assert!(
                    t_inner as Float <= threshold + 1.0,
                    "t={t}, eps={eps}: t_inner={t_inner} > threshold={threshold:.1}"
                );
            }
        }
    }

    /// Test that DC is never triggered at t=0 (no prefix possible).
    #[test]
    fn test_dc_not_triggered_at_t0() {
        for &eps in &[0.1_f64, 0.01, 0.001] {
            assert_eq!(optimal_t_prime(0, eps), 0, "t'=0 always for t=0");
        }
    }

    /// Synthesize a Haar-random SU(2) unitary at ε=1e-3. Exercises the
    /// prefix_split_search path on a non-trivial target (not just Rz/Ry); the named
    /// tests above mostly cover axis-aligned rotations.
    #[test]
    fn test_synthesize_random_unitary() {
        use rand::{SeedableRng, rngs::StdRng, Rng};

        let mut rng = StdRng::seed_from_u64(42);
        let eps = 0.001_f64;

        let theta: Float = rng.random::<Float>() * (2.0 * std::f64::consts::PI);
        let phi: Float = rng.random::<Float>() * (2.0 * std::f64::consts::PI);
        let lambda: Float = rng.random::<Float>() * (2.0 * std::f64::consts::PI);
        
        let ct = (theta / 2.0).cos();
        let st = (theta / 2.0).sin();

        // U3(θ,φ,λ) has det = e^{i(φ+λ)}, which is SU(2) only if φ+λ=0.
        // Normalize to SU(2) by multiplying by e^{-i(φ+λ)/2}.
        let global_phase = Complex::from_polar(1.0, -(phi + lambda) / 2.0);
        let target: Mat2 = [
            [global_phase * Complex::new(ct, 0.0), global_phase * (-Complex::from_polar(st, lambda))],
            [global_phase * Complex::from_polar(st, phi), global_phase * Complex::from_polar(ct, phi + lambda)],
        ];
        println!("Target unitary:\n{:?}", target);

        let synth = SynthesizerT::new(eps);
        let result = synth.synthesize(target).expect("Should synthesize random unitary");
        println!("Random unitary synthesis result: gates={:?}, lde={}, distance={:.6e}",
            result.gates, result.lde, result.distance);
        assert!(result.distance < eps,
            "distance={:.6e} >= epsilon={:.6e}", result.distance, eps);
    }
