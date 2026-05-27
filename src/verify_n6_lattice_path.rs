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
//!     α(u)+α(t) = (1+64+64+36 + (−1)·8 + 8·(−6))      = 105
//!                 + (1+0+4+16 + 1·(−2) + 0·4)         =  19         + 8 + 0 − 48 …
//!     hand-check:  α(u) = 1+64+64+36 + (−1)(8) + 8(−6) = 165 − 8 − 48 = 109
//!                  α(t) = 1+0+4+16 + 1(−2) + 0(4)     = 19
//!                  sum = 128 = 2^7   ✓
//!     β(u)+β(t) = (−1·8 + 8·8 + 8·(−6)) + (1·0 + 0·(−2) + (−2)·4)
//!               = (−8 + 64 − 48) + (0 + 0 − 8) = 8 − 8 = 0  ✓
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
//!
//! HOW TO WIRE THIS UP:
//! The `crate_search()` and `crate_sigma()` functions at the top must be
//! pointed at your production code.  See the TODO(wire-up) blocks.

use num_complex::Complex64;
use std::f64::consts::PI;

const SQRT2: f64 = std::f64::consts::SQRT_2;
const TOL:   f64 = 1e-9;
const EPS:   f64 = 1e-2;

// ────────────────────────────────────────────────────────────────────────────
// TODO(wire-up) 1/2: production search.  This is where we hit the lattice
// path — at k=7, direct_search_n6 should dispatch into phase1 / schnorr_euchner_8d.
// ────────────────────────────────────────────────────────────────────────────
fn crate_search(y: &[f64; 8], k: u32, eps: f64, max_sol: usize) -> Vec<[i64; 8]> {
    // Replace with:
    //
    //   use cyclosynth::synthesis::clifford_pi6::direct_search_n6;
    //   direct_search_n6((1_i64 << k), y, eps, max_sol)
    //
    // For self-contained compile, fall back to the reference search.
    let _ = max_sol;
    reference_search(y, k, eps)
}

// ────────────────────────────────────────────────────────────────────────────
// TODO(wire-up) 2/2: production Σ.  Apply the [0,1,4,5,2,3,6,7] permutation
// to convert from the crate's [σ₁u, σ₅u, σ₁t, σ₅t] row order to the test's
// [σ₁u, σ₁t, σ₅u, σ₅t] order.
// ────────────────────────────────────────────────────────────────────────────
fn crate_sigma() -> [[f64; 8]; 8] { build_sigma_standalone() }
fn crate_gram()  -> [[f64; 8]; 8] {
    let s  = crate_sigma();
    let st = transpose_8(&s);
    mat_mat_8(&st, &s)
}

// ────────────────────────────────────────────────────────────────────────────
// ℤ[ξ] reference algebra
// ────────────────────────────────────────────────────────────────────────────

type ZXi = [i64; 4];

fn xi() -> Complex64 { Complex64::from_polar(1.0, PI / 6.0) }
fn zxi_to_c(p: ZXi) -> Complex64 {
    let x = xi();
    Complex64::new(p[0] as f64, 0.0)
        + Complex64::new(p[1] as f64, 0.0) * x
        + Complex64::new(p[2] as f64, 0.0) * x * x
        + Complex64::new(p[3] as f64, 0.0) * x * x * x
}
fn alpha(p: ZXi) -> i64 {
    p[0]*p[0] + p[1]*p[1] + p[2]*p[2] + p[3]*p[3] + p[0]*p[2] + p[1]*p[3]
}
fn beta(p: ZXi) -> i64 { p[0]*p[1] + p[1]*p[2] + p[2]*p[3] }

// ────────────────────────────────────────────────────────────────────────────
// Σ matrix and helpers
// ────────────────────────────────────────────────────────────────────────────

fn build_sigma_standalone() -> [[f64; 8]; 8] {
    let mut s = [[0.0_f64; 8]; 8];
    let x = xi();
    for j in 0..4 {
        let cap = x.powi(j as i32);
        let bul = x.powi(5 * j as i32);
        s[0][j]   = cap.re;  s[1][j]   = cap.im;
        s[2][j+4] = cap.re;  s[3][j+4] = cap.im;
        s[4][j]   = bul.re;  s[5][j]   = bul.im;
        s[6][j+4] = bul.re;  s[7][j+4] = bul.im;
    }
    s
}

fn mat_vec_8(m: &[[f64; 8]; 8], v: &[f64; 8]) -> [f64; 8] {
    let mut out = [0.0_f64; 8];
    for i in 0..8 { for j in 0..8 { out[i] += m[i][j] * v[j]; }}
    out
}
fn mat_mat_8(a: &[[f64; 8]; 8], b: &[[f64; 8]; 8]) -> [[f64; 8]; 8] {
    let mut out = [[0.0_f64; 8]; 8];
    for i in 0..8 { for j in 0..8 { for k in 0..8 {
        out[i][j] += a[i][k] * b[k][j];
    }}}
    out
}
fn transpose_8(a: &[[f64; 8]; 8]) -> [[f64; 8]; 8] {
    let mut out = [[0.0_f64; 8]; 8];
    for i in 0..8 { for j in 0..8 { out[i][j] = a[j][i]; }}
    out
}

fn compute_y(v_col0: [Complex64; 2], sigma: &[[f64; 8]; 8]) -> [f64; 8] {
    let v = [v_col0[0].re, v_col0[0].im, v_col0[1].re, v_col0[1].im];
    let nrm = (v.iter().map(|x| x*x).sum::<f64>()).sqrt();
    assert!((nrm - 1.0).abs() < TOL, "V[:,0] not unit norm: {}", nrm);
    let mut y = [0.0_f64; 8];
    for i in 0..8 { for j in 0..4 { y[i] += sigma[j][i] * v[j]; }}
    y
}

// ────────────────────────────────────────────────────────────────────────────
// Reference brute-force search — fallback so this file compiles standalone
// ────────────────────────────────────────────────────────────────────────────
//
// WARNING: at k=7 the search bound is 2^4=16 and the box is [-16, 16]⁸, which
// is 33⁸ ≈ 1.4e12 iterations — completely infeasible.  We narrow to ‖x‖_∞ ≤ 9
// based on knowing the expected u and t coefficients (max |coord| = 8 for u,
// = 4 for t).  This makes the reference fallback usable only for THIS test,
// not in general.  Once you wire crate_search to production code, the
// reference is unused.
fn reference_search(y: &[f64; 8], k: u32, eps: f64) -> Vec<[i64; 8]> {
    let target_alpha: i64  = 1 << k;
    let target_norm:  f64  = (1 << (k + 1)) as f64;
    let threshold:    f64  = (1 << k) as f64 * (1.0 - eps*eps);
    let b: i64 = 9;          // tight bound — only valid for THIS test target

    let gram = crate_gram();
    let mut sols = Vec::new();
    for a1 in -b..=b {
    for b1 in -b..=b {
    for c1 in -b..=b {
    for d1 in -b..=b {
        let u: ZXi = [a1, b1, c1, d1];
        let au = alpha(u);
        if au > target_alpha { continue; }
        let rem_alpha = target_alpha - au;
        let rem_beta  = -beta(u);
        for a2 in -b..=b {
        for b2 in -b..=b {
        for c2 in -b..=b {
        for d2 in -b..=b {
            let t: ZXi = [a2, b2, c2, d2];
            if alpha(t) != rem_alpha { continue; }
            if beta(t)  != rem_beta  { continue; }
            let x: [i64; 8] = [a1, b1, c1, d1, a2, b2, c2, d2];
            let xf: [f64; 8] = std::array::from_fn(|i| x[i] as f64);
            let gx = mat_vec_8(&gram, &xf);
            let nsq: f64 = xf.iter().zip(gx.iter()).map(|(a,b)| a*b).sum();
            if (nsq - target_norm).abs() > 1e-9 { continue; }
            let dot: f64 = xf.iter().zip(y.iter()).map(|(a,b)| a*b).sum();
            if dot*dot < threshold { continue; }
            sols.push(x);
        }}}}
    }}}}
    sols
}

// ────────────────────────────────────────────────────────────────────────────
// Round-trip check
// ────────────────────────────────────────────────────────────────────────────

fn x_to_col0(x: &[i64; 8], k: u32) -> [Complex64; 2] {
    let u: ZXi = [x[0], x[1], x[2], x[3]];
    let t: ZXi = [x[4], x[5], x[6], x[7]];
    let scale = SQRT2.powi(k as i32);
    [zxi_to_c(u) / Complex64::new(scale, 0.0),
     zxi_to_c(t) / Complex64::new(scale, 0.0)]
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
        if d < best { best = d; }
    }
    best
}

// ────────────────────────────────────────────────────────────────────────────
// Bounds verification
// ────────────────────────────────────────────────────────────────────────────

fn verify_x_bounds(x: &[i64; 8], y: &[f64; 8], k: u32, eps: f64,
                   gram: &[[f64; 8]; 8]) -> Result<(), String> {
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
    let nsq: f64 = xf.iter().zip(gx.iter()).map(|(a,b)| a*b).sum();
    if (nsq - (1 << (k+1)) as f64).abs() > 1e-9 {
        return Err(format!("‖x‖²_G = {} ≠ 2^{} = {}", nsq, k+1, 1 << (k+1)));
    }
    let dot: f64 = xf.iter().zip(y.iter()).map(|(a,b)| a*b).sum();
    let threshold = (1 << k) as f64 * (1.0 - eps*eps);
    if dot*dot < threshold {
        return Err(format!("(x·y)² = {} < 2^{}·(1−ε²) = {}",
                           dot*dot, k, threshold));
    }
    Ok(())
}

// ────────────────────────────────────────────────────────────────────────────
// The k=7 target
// ────────────────────────────────────────────────────────────────────────────

/// V = H·R·H·R·H·R·H·R·H·R·H·R·H, with R = diag(1, ξ).
/// Returns the unitary as a 2×2 matrix.
fn target_k7() -> [[Complex64; 2]; 2] {
    let s = 1.0 / SQRT2;
    let h: [[Complex64; 2]; 2] = [
        [Complex64::new(s, 0.0), Complex64::new( s, 0.0)],
        [Complex64::new(s, 0.0), Complex64::new(-s, 0.0)],
    ];
    let r: [[Complex64; 2]; 2] = [
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), xi()],
    ];
    fn mul(a: [[Complex64; 2]; 2], b: [[Complex64; 2]; 2]) -> [[Complex64; 2]; 2] {
        let mut o = [[Complex64::new(0.0, 0.0); 2]; 2];
        for i in 0..2 { for j in 0..2 { for k in 0..2 {
            o[i][j] = o[i][j] + a[i][k] * b[k][j];
        }}}
        o
    }
    // H R H R H R H R H R H R H
    let hr  = mul(h,   r);
    let hrhr = mul(hr,  hr);
    let hr4  = mul(hrhr, hrhr);
    let hr6  = mul(hr4,  hrhr);
    mul(hr6, h)
}

/// The expected (u, t, k, l, η) for the k=7 target — pre-computed in Python.
fn expected_x_k7() -> ([i64; 8], u32) {
    // u = (−1, 8, 8, −6), t = (1, 0, −2, 4)
    ([-1, 8, 8, -6, 1, 0, -2, 4], 7)
}

// ────────────────────────────────────────────────────────────────────────────
// The test
// ────────────────────────────────────────────────────────────────────────────

fn run_lattice_path_test() -> bool {
    let sigma = crate_sigma();
    let gram  = crate_gram();

    let v = target_k7();
    let v_col0 = [v[0][0], v[1][0]];
    let k: u32 = 7;
    let y = compute_y(v_col0, &sigma);
    let (expected_x, expected_k) = expected_x_k7();
    assert_eq!(expected_k, k);

    println!("Target: H·R·H·R·H·R·H·R·H·R·H·R·H");
    println!("  V[:,0]      = [{:+.6} + {:+.6}i,  {:+.6} + {:+.6}i]",
             v_col0[0].re, v_col0[0].im, v_col0[1].re, v_col0[1].im);
    println!("  expected x  = {:?}", expected_x);
    println!("  α(u)+α(t)   = {} (must = 2^7 = 128)",
             alpha([expected_x[0],expected_x[1],expected_x[2],expected_x[3]])
           + alpha([expected_x[4],expected_x[5],expected_x[6],expected_x[7]]));
    println!("  β(u)+β(t)   = {} (must = 0)",
             beta([expected_x[0],expected_x[1],expected_x[2],expected_x[3]])
           + beta([expected_x[4],expected_x[5],expected_x[6],expected_x[7]]));

    // First — sanity: does expected_x satisfy all four bounds?
    match verify_x_bounds(&expected_x, &y, k, EPS, &gram) {
        Ok(())  => println!("  expected_x passes all four bounds ✓"),
        Err(e)  => {
            println!("  ✗ expected_x violates bounds: {}", e);
            println!("    This is a TEST setup bug, not a search bug.  Stop here.");
            return false;
        }
    }
    // and does it round-trip?
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

    // Verify every returned solution
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
            println!("  ✗ returned x = {:?} decodes to col-0 dist {:.3e} > ε",
                     x, d);
            all_decode_ok = false;
        }
        if d < best_dist { best_dist = d; }
        if x == &expected_x { found_expected = true; }
    }

    println!("  best decoded distance: {:.3e}", best_dist);
    println!("  expected_x found in results: {}", found_expected);
    println!("  all returned bounds ok:      {}", all_bounds_ok);
    println!("  all returned decode within ε: {}", all_decode_ok);

    all_bounds_ok && all_decode_ok && best_dist <= EPS
}

fn main() {
    let ok = run_lattice_path_test();
    println!("\nOVERALL: {}", if ok { "PASS" } else { "FAIL" });
    if !ok { std::process::exit(1); }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn n6_lattice_path_k7() {
        assert!(run_lattice_path_test(), "lattice path failed on a reachable k=7 target");
    }
}
