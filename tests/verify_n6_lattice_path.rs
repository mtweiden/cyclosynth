//! verify_n6_lattice_path.rs — exercise the LATTICE path of n=6 search at k=7.
//!
//! Context.  We have two passing tests upstream:
//!   - verify_n6_lattice  proves Σ and G are correct
//!   - verify_n6_search   proves brute_force_direct_search_n6 returns valid
//!                        solutions, but only at k ∈ {1, 2} — below the
//!                        LATTICE_K_MIN = 7 threshold, so it never invoked
//!                        the lattice path (schnorr_euchner_8d → phase1).
//!
//! Symptom we're diagnosing.  The lattice path at k=7 returns zero solutions
//! across multiple y-scalings on synthetic targets.  Possible causes:
//!   (i)  the synthetic target is genuinely unreachable at k=7
//!   (ii) the lattice path is broken — wrong basis, bad pruning, or wrong
//!        y-scaling, and would fail even on a reachable target
//!
//! This test uses a TARGET WE KNOW IS REACHABLE AT EXACTLY k=7.  Specifically:
//!
//!     U = H·R·H·R·H·R·H·R·H·R·H·R·H
//!         (7 H gates alternating with R = diag(1, ξ), ξ = e^{iπ/6})
//!
//!     u = (−1, 8, 8, −6)    in basis {1, ξ, ξ², ξ³}
//!     t = ( 1, 0, −2,  4)
//!     k = 7,  l = 0,  η = ξ^0
//!
//! Three outcomes possible when we run this against the production search:
//!
//!   PASS    — search returns this x (or an equivalent +/-/Galois variant)
//!             that decodes within ε of V[:,0].  Lattice path works.
//!
//!   FAIL with empty result —
//!             the lattice path is broken in a way that affects reachable
//!             targets.  This is the n=6-specific bug to chase.
//!
//!   FAIL with non-empty result that fails bounds or decode —
//!             search is returning malformed solutions (wrong Σ, wrong scale).

use cyclosynth::synthesis::clifford_pi6::{direct_search_n6, sigma_matrix};
use num_complex::Complex64;
use std::f64::consts::PI;

const SQRT2: f64 = std::f64::consts::SQRT_2;
const TOL: f64 = 1e-9;
const EPS: f64 = 1e-2;

// ── wire-up 1/2: production search ─────────────────────────────────────────
//
// At k=7, direct_search_n6 should dispatch into the lattice path
// (phase1 / schnorr_euchner_8d) rather than brute force.

fn crate_search(y: &[f64; 8], k: u32, eps: f64, max_sol: usize) -> Vec<[i64; 8]> {
    direct_search_n6(1_i64 << k, y, eps, max_sol)
}

// ── wire-up 2/2: sigma and gram ─────────────────────────────────────────────
//
// Crate row order: [σ₁u, σ₅u, σ₁t, σ₅t]  (rows 0,1 = σ₁u; 2,3 = σ₅u; 4,5 = σ₁t; 6,7 = σ₅t)
// Test row order:  [σ₁u, σ₁t, σ₅u, σ₅t]  (Σ_top = rows 0..3 = σ₁u + σ₁t)
// Permutation:     test[i] = crate[PERM[i]]

const PERM: [usize; 8] = [0, 1, 4, 5, 2, 3, 6, 7];

fn crate_sigma() -> [[f64; 8]; 8] {
    let s = sigma_matrix();
    let mut out = [[0.0_f64; 8]; 8];
    for (new_row, &old_row) in PERM.iter().enumerate() {
        out[new_row] = s[old_row];
    }
    out
}

fn crate_gram() -> [[f64; 8]; 8] {
    let s = crate_sigma();
    let st = transpose_8(&s);
    mat_mat_8(&st, &s)
}

// ── ℤ[ξ] algebra ─────────────────────────────────────────────────────────────

type ZXi = [i64; 4];

fn xi() -> Complex64 {
    Complex64::from_polar(1.0, PI / 6.0)
}

fn zxi_to_c(p: ZXi) -> Complex64 {
    let x = xi();
    Complex64::new(p[0] as f64, 0.0)
        + Complex64::new(p[1] as f64, 0.0) * x
        + Complex64::new(p[2] as f64, 0.0) * x * x
        + Complex64::new(p[3] as f64, 0.0) * x * x * x
}

fn alpha(p: ZXi) -> i64 {
    p[0] * p[0] + p[1] * p[1] + p[2] * p[2] + p[3] * p[3] + p[0] * p[2] + p[1] * p[3]
}
fn beta(p: ZXi) -> i64 {
    p[0] * p[1] + p[1] * p[2] + p[2] * p[3]
}

// ── matrix helpers ────────────────────────────────────────────────────────────

fn mat_vec_8(m: &[[f64; 8]; 8], v: &[f64; 8]) -> [f64; 8] {
    let mut out = [0.0_f64; 8];
    for i in 0..8 {
        for j in 0..8 {
            out[i] += m[i][j] * v[j];
        }
    }
    out
}
fn mat_mat_8(a: &[[f64; 8]; 8], b: &[[f64; 8]; 8]) -> [[f64; 8]; 8] {
    let mut out = [[0.0_f64; 8]; 8];
    for i in 0..8 {
        for j in 0..8 {
            for k in 0..8 {
                out[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    out
}
fn transpose_8(a: &[[f64; 8]; 8]) -> [[f64; 8]; 8] {
    let mut out = [[0.0_f64; 8]; 8];
    for i in 0..8 {
        for j in 0..8 {
            out[i][j] = a[j][i];
        }
    }
    out
}

// ── alignment vector y ────────────────────────────────────────────────────────

fn compute_y(v_col0: [Complex64; 2], sigma: &[[f64; 8]; 8]) -> [f64; 8] {
    let v = [v_col0[0].re, v_col0[0].im, v_col0[1].re, v_col0[1].im];
    let nrm = (v.iter().map(|x| x * x).sum::<f64>()).sqrt();
    assert!((nrm - 1.0).abs() < TOL, "V[:,0] not unit norm: {}", nrm);
    let mut y = [0.0_f64; 8];
    for i in 0..8 {
        for j in 0..4 {
            y[i] += sigma[j][i] * v[j];
        }
    }
    y
}

// ── decode x → first column of U ─────────────────────────────────────────────

fn x_to_col0(x: &[i64; 8], k: u32) -> [Complex64; 2] {
    let u: ZXi = [x[0], x[1], x[2], x[3]];
    let t: ZXi = [x[4], x[5], x[6], x[7]];
    let scale = SQRT2.powi(k as i32);
    [
        zxi_to_c(u) / Complex64::new(scale, 0.0),
        zxi_to_c(t) / Complex64::new(scale, 0.0),
    ]
}

fn col0_distance(x: &[i64; 8], k: u32, v_col0: [Complex64; 2]) -> f64 {
    let cand = x_to_col0(x, k);
    let xv = xi();
    let mut best = f64::INFINITY;
    for eta_idx in 0..12 {
        let eta = xv.powi(eta_idx);
        let d0 = eta * v_col0[0] - cand[0];
        let d1 = eta * v_col0[1] - cand[1];
        let d = (d0.norm_sqr() + d1.norm_sqr()).sqrt();
        if d < best {
            best = d;
        }
    }
    best
}

// ── bounds verification ───────────────────────────────────────────────────────

fn verify_x_bounds(
    x: &[i64; 8],
    y: &[f64; 8],
    k: u32,
    eps: f64,
    gram: &[[f64; 8]; 8],
) -> Result<(), String> {
    let u: ZXi = [x[0], x[1], x[2], x[3]];
    let t: ZXi = [x[4], x[5], x[6], x[7]];
    let asum = alpha(u) + alpha(t);
    let expected_alpha: i64 = 1 << k;
    if asum != expected_alpha {
        return Err(format!("α-sum {} ≠ 2^{} = {}", asum, k, expected_alpha));
    }
    if beta(u) + beta(t) != 0 {
        return Err(format!("β-sum {} ≠ 0", beta(u) + beta(t)));
    }
    let xf: [f64; 8] = std::array::from_fn(|i| x[i] as f64);
    let gx = mat_vec_8(gram, &xf);
    let nsq: f64 = xf.iter().zip(gx.iter()).map(|(a, b)| a * b).sum();
    if (nsq - (1 << (k + 1)) as f64).abs() > 1e-9 {
        return Err(format!("‖x‖²_G = {} ≠ 2^{} = {}", nsq, k + 1, 1 << (k + 1)));
    }
    let dot: f64 = xf.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
    let threshold = (1 << k) as f64 * (1.0 - eps * eps);
    if dot * dot < threshold {
        return Err(format!(
            "(x·y)² = {} < 2^{}·(1−ε²) = {}",
            dot * dot,
            k,
            threshold
        ));
    }
    Ok(())
}

// ── the k=7 target ────────────────────────────────────────────────────────────

/// V = H·R·H·R·H·R·H·R·H·R·H·R·H, with R = diag(1, ξ).
fn target_k7() -> [[Complex64; 2]; 2] {
    let s = 1.0 / SQRT2;
    let h: [[Complex64; 2]; 2] = [
        [Complex64::new(s, 0.0), Complex64::new(s, 0.0)],
        [Complex64::new(s, 0.0), Complex64::new(-s, 0.0)],
    ];
    let r: [[Complex64; 2]; 2] = [
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), xi()],
    ];
    fn mul(a: [[Complex64; 2]; 2], b: [[Complex64; 2]; 2]) -> [[Complex64; 2]; 2] {
        let mut o = [[Complex64::new(0.0, 0.0); 2]; 2];
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    o[i][j] = o[i][j] + a[i][k] * b[k][j];
                }
            }
        }
        o
    }
    let hr = mul(h, r);
    let hrhr = mul(hr, hr);
    let hr4 = mul(hrhr, hrhr);
    let hr6 = mul(hr4, hrhr);
    mul(hr6, h)
}

/// The expected (u, t, k) for the k=7 target — pre-computed.
fn expected_x_k7() -> ([i64; 8], u32) {
    ([-1, 8, 8, -6, 1, 0, -2, 4], 7)
}

// ── the test ──────────────────────────────────────────────────────────────────

fn run_lattice_path_test() -> bool {
    let sigma = crate_sigma();
    let gram = crate_gram();

    let v = target_k7();
    let v_col0 = [v[0][0], v[1][0]];
    let k: u32 = 7;
    let y = compute_y(v_col0, &sigma);
    let (expected_x, expected_k) = expected_x_k7();
    assert_eq!(expected_k, k);

    println!("Target: H·R·H·R·H·R·H·R·H·R·H·R·H");
    println!(
        "  V[:,0]      = [{:+.6} + {:+.6}i,  {:+.6} + {:+.6}i]",
        v_col0[0].re, v_col0[0].im, v_col0[1].re, v_col0[1].im
    );
    println!("  expected x  = {:?}", expected_x);
    println!(
        "  α(u)+α(t)   = {} (must = 2^7 = 128)",
        alpha([expected_x[0], expected_x[1], expected_x[2], expected_x[3]])
            + alpha([expected_x[4], expected_x[5], expected_x[6], expected_x[7]])
    );
    println!(
        "  β(u)+β(t)   = {} (must = 0)",
        beta([expected_x[0], expected_x[1], expected_x[2], expected_x[3]])
            + beta([expected_x[4], expected_x[5], expected_x[6], expected_x[7]])
    );

    // Sanity: does expected_x satisfy all four bounds?
    match verify_x_bounds(&expected_x, &y, k, EPS, &gram) {
        Ok(()) => println!("  expected_x passes all four bounds ✓"),
        Err(e) => {
            println!("  ✗ expected_x violates bounds: {}", e);
            println!("    This is a TEST setup bug, not a search bug.  Stop here.");
            return false;
        }
    }
    let d = col0_distance(&expected_x, k, v_col0);
    println!("  expected_x decodes to col-0 distance {:.3e}", d);
    if d > EPS {
        println!("  ✗ expected_x doesn't decode to V[:,0]; test setup is broken");
        return false;
    }

    println!("\nCalling production search at k=7...");
    let sols = crate_search(&y, k, EPS, usize::MAX);
    println!("  search returned {} solutions", sols.len());

    if sols.is_empty() {
        println!("\n  ✗ LATTICE PATH RETURNED NOTHING on a reachable target.");
        println!("    This rules out 'target unreachable' — there IS a solution");
        println!("    (verified above by expected_x).  The lattice path is broken.");
        return false;
    }

    let mut all_bounds_ok = true;
    let mut all_decode_ok = true;
    let mut best_dist = f64::INFINITY;
    let mut found_expected = false;

    for x in &sols {
        if let Err(e) = verify_x_bounds(x, &y, k, EPS, &gram) {
            println!("  ✗ returned x = {:?} violates bounds: {}", x, e);
            all_bounds_ok = false;
        }
        let d = col0_distance(x, k, v_col0);
        if d > EPS {
            println!(
                "  ✗ returned x = {:?} decodes to col-0 dist {:.3e} > ε",
                x, d
            );
            all_decode_ok = false;
        }
        if d < best_dist {
            best_dist = d;
        }
        if x == &expected_x {
            found_expected = true;
        }
    }

    println!("  best decoded distance: {:.3e}", best_dist);
    println!("  expected_x found in results: {}", found_expected);
    println!("  all returned bounds ok:      {}", all_bounds_ok);
    println!("  all returned decode within ε: {}", all_decode_ok);

    all_bounds_ok && all_decode_ok && best_dist <= EPS
}

#[test]
fn n6_lattice_path_k7() {
    assert!(
        run_lattice_path_test(),
        "lattice path failed on a reachable k=7 target"
    );
}

// ────────────────────────────────────────────────────────────────────────────
// Generic gate-word builder
// ────────────────────────────────────────────────────────────────────────────

type Mat2c = [[Complex64; 2]; 2];

fn mat2_mul(a: Mat2c, b: Mat2c) -> Mat2c {
    let mut o = [[Complex64::new(0.0, 0.0); 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            for kk in 0..2 {
                o[i][j] = o[i][j] + a[i][kk] * b[kk][j];
            }
        }
    }
    o
}

/// Build U = G_1 · G_2 · ... · G_n where each char in `word` is one gate:
/// 'H' = Hadamard, 'R' = diag(1, ξ), 'S' = diag(1, i).
fn build_unitary_from_word(word: &str) -> Mat2c {
    let s = 1.0 / SQRT2;
    let h: Mat2c = [
        [Complex64::new(s, 0.0), Complex64::new(s, 0.0)],
        [Complex64::new(s, 0.0), Complex64::new(-s, 0.0)],
    ];
    let r: Mat2c = [
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), xi()],
    ];
    let sg: Mat2c = [
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), Complex64::new(0.0, 1.0)],
    ];
    let id: Mat2c = [
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
    ];
    let mut acc = id;
    for g in word.chars() {
        let m = match g {
            'H' => h,
            'R' => r,
            'S' => sg,
            _ => continue,
        };
        acc = mat2_mul(acc, m);
    }
    acc
}

// ────────────────────────────────────────────────────────────────────────────
// Parameterized lattice-path runner.  Galois-aware on the expected_x sanity
// check (only α-sum, β-sum, norm, and col0_distance are required — the
// alignment check is intentionally skipped since the user's expected x may
// represent ξ^l · V[:,0] for nonzero l, which doesn't dot-align with
// y = compute_y(V[:,0]) but still decodes to V[:,0] under one of the 12
// η-rotations checked by col0_distance).
//
// Returned solutions, on the other hand, are required to pass ALL four
// invariants — the production search returns x's in the l=0 form by
// construction.
// ────────────────────────────────────────────────────────────────────────────

fn run_lattice_path_for_target(
    target_name: &str,
    v_mat: Mat2c,
    expected_x: Option<[i64; 8]>,
    k: u32,
) -> bool {
    let sigma = crate_sigma();
    let gram = crate_gram();
    let v_col0 = [v_mat[0][0], v_mat[1][0]];
    let y = compute_y(v_col0, &sigma);

    println!("\n=== Target: {}  (k = {}) ===", target_name, k);
    println!(
        "  V[:,0]      = [{:+.6} + {:+.6}i,  {:+.6} + {:+.6}i]",
        v_col0[0].re, v_col0[0].im, v_col0[1].re, v_col0[1].im
    );

    if let Some(exp) = expected_x {
        println!("  expected x  = {:?}", exp);
        // Galois-invariant checks: α, β, norm equation.
        let a_sum =
            alpha([exp[0], exp[1], exp[2], exp[3]]) + alpha([exp[4], exp[5], exp[6], exp[7]]);
        let b_sum = beta([exp[0], exp[1], exp[2], exp[3]]) + beta([exp[4], exp[5], exp[6], exp[7]]);
        println!("  α(u)+α(t)   = {}  (must = {})", a_sum, 1i64 << k);
        println!("  β(u)+β(t)   = {}  (must = 0)", b_sum);
        if a_sum != (1i64 << k) || b_sum != 0 {
            println!("  ✗ expected_x violates α/β invariants — TEST SETUP BUG");
            return false;
        }
        let d = col0_distance(&exp, k, v_col0);
        println!(
            "  expected_x decodes to col-0 distance {:.3e}  (some η ∈ {{ξ^0..ξ^11}})",
            d
        );
        if d > EPS {
            println!("  ✗ expected_x doesn't decode to V[:,0] under any η — TEST SETUP BUG");
            return false;
        }
    }

    println!("Calling production search at k={}...", k);
    let sols = crate_search(&y, k, EPS, usize::MAX);
    println!("  search returned {} solutions", sols.len());

    if sols.is_empty() {
        println!("  ✗ lattice path returned nothing");
        return false;
    }

    let mut all_bounds_ok = true;
    let mut all_decode_ok = true;
    let mut best_dist = f64::INFINITY;

    for x in &sols {
        if let Err(e) = verify_x_bounds(x, &y, k, EPS, &gram) {
            println!("  ✗ returned x = {:?} violates bounds: {}", x, e);
            all_bounds_ok = false;
        }
        let d = col0_distance(x, k, v_col0);
        if d > EPS {
            println!(
                "  ✗ returned x = {:?} decodes to col-0 dist {:.3e} > ε",
                x, d
            );
            all_decode_ok = false;
        }
        if d < best_dist {
            best_dist = d;
        }
    }

    println!("  best decoded distance: {:.3e}", best_dist);
    println!("  all returned bounds ok:       {}", all_bounds_ok);
    println!("  all returned decode within ε: {}", all_decode_ok);

    all_bounds_ok && all_decode_ok && best_dist <= EPS
}

// ────────────────────────────────────────────────────────────────────────────
// ITEM A: second k=7 target — V = HRHRHRHRHRHRRH (14 gates).
//
// Expected solution (from user):
//   u = (5, 3, 2, 4), t = (1, 1, -6, 4), k = 7, l = 1, η = ξ^0
//
// The l=1 factor means (u, t)/√128 = ξ · V[:,0].  The production search
// returns the l=0 form (some Galois rotate of this x), so col0_distance
// handles the rotation; verify_x_bounds is intentionally skipped for the
// expected_x sanity check (see runner).
//
// This target has β(u)=29, β(t)=−29 (nonzero individually) — so its
// cap-image lies OFF the y-axis perpendicular direction.  This exercises
// the Δ_p side of Q, which the first k=7 target (with β(u)=8, β(t)=−8,
// also nonzero, but symmetric in a different way) did not stress as much.
// ────────────────────────────────────────────────────────────────────────────

#[test]
fn n6_lattice_path_k7_word_b() {
    let v = build_unitary_from_word("HRHRHRHRHRHRRH");
    let expected = [5, 3, 2, 4, 1, 1, -6, 4];
    assert!(
        run_lattice_path_for_target("HRHRHRHRHRHRRH", v, Some(expected), 7),
        "lattice path failed on second k=7 target (HRHRHRHRHRHRRH)"
    );
}

// ────────────────────────────────────────────────────────────────────────────
// ITEM B: k=8 target — V = HRHRHRHRHRHRHRH (15 gates, 8 H's).
//
// We do NOT have a pre-computed expected x.  Instead:
//   1. Confirm k=7 is INSUFFICIENT (search returns nothing OR returns
//      solutions that don't decode to V[:,0] within ε).
//   2. Confirm k=8 returns ≥1 valid solution that decodes within ε.
//
// This both tests that the lattice path scales beyond k=7 AND verifies
// V actually lands at k=8 (not a smaller k that we missed).
// ────────────────────────────────────────────────────────────────────────────

#[test]
fn n6_lattice_path_k8() {
    let v = build_unitary_from_word("HRHRHRHRHRHRHRH");
    let sigma = crate_sigma();
    let v_col0 = [v[0][0], v[1][0]];
    let y = compute_y(v_col0, &sigma);

    println!("\n=== Target: HRHRHRHRHRHRHRH (8 H gates) ===");
    println!(
        "  V[:,0]      = [{:+.6} + {:+.6}i,  {:+.6} + {:+.6}i]",
        v_col0[0].re, v_col0[0].im, v_col0[1].re, v_col0[1].im
    );

    // Probe k=7: expect either zero solutions OR solutions that don't decode
    // to V[:,0] within ε.  Any solution that DID decode would mean V lands
    // at k≤7, contradicting the "k=8 target" intent.
    println!("\nProbe at k=7 (expect: no decoding solution)...");
    let sols_k7 = crate_search(&y, 7, EPS, usize::MAX);
    println!("  k=7 returned {} solutions", sols_k7.len());
    let any_k7_decodes = sols_k7.iter().any(|x| col0_distance(x, 7, v_col0) <= EPS);
    if any_k7_decodes {
        println!("  ✗ Target IS reachable at k=7 — not a true k=8 target.");
        println!("    Pick a longer or different gate word for the k=8 test.");
        panic!("k=8 target was actually reachable at k=7");
    }
    println!("  ✓ k=7 has no decoding solution — V is genuinely ≥ k=8");

    // Now the real test: k=8 must return ≥1 valid decoding solution.
    println!("\nProbe at k=8...");
    assert!(
        run_lattice_path_for_target("HRHRHRHRHRHRHRH", v, None, 8),
        "lattice path failed at k=8"
    );
}
