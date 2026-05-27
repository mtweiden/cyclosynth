//! verify_n6_search.rs — search-layer test for n=6 (Clifford+R_z(π/6)).
//!
//! Companion to verify_n6_lattice.rs.  The lattice test verified Σ and G.
//! This test verifies the SEARCH: given a target unitary V, ε, and k, the
//! production search must return at least one x ∈ ℤ⁸ that
//!
//!   (a) α(u)+α(t) = 2^k                  (rational unitarity)
//!   (b) β(u)+β(t) = 0                    (√3 unitarity)
//!   (c) ‖Σx‖²_E   = 2^{k+1}              (norm shell — anisotropic in n=6)
//!   (d) (x·y)²    ≥ 2^k·(1 − ε²)         (alignment)
//!   (e) the resulting (u/√2^k, t/√2^k) is within ε of V[:,0] (up to phase)
//!
//! (a)-(d) are the four "inner" bounds your search must enforce.  (e) is the
//! external contract: the bounds must compose into a real approximation.  A
//! search can satisfy (a)-(d) and still fail (e) only if Σ/G is wrong (which
//! verify_n6_lattice already rules out) or y was built from V incorrectly.
//!
//! HOW TO WIRE THIS UP TO YOUR CRATE:
//! Search for `TODO(wire-up)` below.  There are exactly two places to edit:
//!  1. `crate_search()` — call your production search function
//!  2. `crate_sigma()` and `crate_gram()` — point to your matrices
//!
//! Drop into `tests/verify_n6_search.rs` and run:
//!   cargo test --test verify_n6_search -- --nocapture

use num_complex::Complex64;
use std::f64::consts::PI;

const SQRT2: f64 = std::f64::consts::SQRT_2;
const TOL:   f64 = 1e-9;
const EPS:   f64 = 1e-2;     // approximation tolerance for the search

// ────────────────────────────────────────────────────────────────────────────
// TODO(wire-up) 1/2: replace `crate_search` with your production search.
// ────────────────────────────────────────────────────────────────────────────
//
// Your function should have a signature compatible with:
//   fn search(y: &[f64; 8], k: u32, eps: f64) -> Vec<[i64; 8]>
//
// (or returns a single Option<[i64; 8]> — adjust the test logic below).
//
// Replace the body with a call like:
//
//   use cyclosynth::synthesis::clifford_pi6::direct_search_n6;
//   direct_search_n6(k, y, eps, /* max_solutions */ usize::MAX)
//
// The reference implementation below is a brute-force enumerator over
// [−B, B]⁸ — used as a fallback so this file compiles standalone.
// Remove or `#[cfg(test)] feature-gate` it once your real search is wired up.
//
fn crate_search(y: &[f64; 8], k: u32, eps: f64) -> Vec<[i64; 8]> {
    reference_search(y, k, eps)
}

// ────────────────────────────────────────────────────────────────────────────
// TODO(wire-up) 2/2: replace these with your crate's matrices.
// ────────────────────────────────────────────────────────────────────────────
//
//   use cyclosynth::synthesis::clifford_pi6::sigma_matrix;
//   use cyclosynth::rings::zomicron::SIGMA_GRAM_U;
//
// IMPORTANT: row order.  The standalone build_sigma below uses
//   [σ₁(u), σ₁(t), σ₅(u), σ₅(t)]
// but the crate's sigma_matrix() uses
//   [σ₁(u), σ₅(u), σ₁(t), σ₅(t)]
// Apply the permutation [0,1,4,5,2,3,6,7] when adapting, or change the test
// to match your row order.
//
// (For Gram: the per-block formula is identical in both orderings, so
//  SIGMA_GRAM_U works directly.)
//
fn crate_sigma() -> [[f64; 8]; 8] { build_sigma_standalone() }
fn crate_gram()  -> [[f64; 8]; 8] {
    let s   = crate_sigma();
    let st  = transpose_8(&s);
    mat_mat_8(&st, &s)
}

// ────────────────────────────────────────────────────────────────────────────
// ℤ[ξ] algebra — reference truth (identical to verify_n6_lattice)
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
// Σ matrix — standalone fallback (same row order as verify_n6_lattice)
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

// ────────────────────────────────────────────────────────────────────────────
// y from V — alignment direction.
// ────────────────────────────────────────────────────────────────────────────
//
// y = Σ_topᵀ · v,  where v = (Re V[0,0], Im V[0,0], Re V[1,0], Im V[1,0])
// and ‖v‖ = 1.
//
// Σ_top here = first 4 rows of build_sigma_standalone() =
// (Re σ₁u, Im σ₁u, Re σ₁t, Im σ₁t).
//
// If your production code uses the crate's [σ₁u, σ₅u, σ₁t, σ₅t] row order,
// "Σ_top" = rows {0, 1, 4, 5} of that matrix.  Adjust accordingly.

fn compute_y(v_col0: [Complex64; 2], sigma: &[[f64; 8]; 8]) -> [f64; 8] {
    let v = [v_col0[0].re, v_col0[0].im, v_col0[1].re, v_col0[1].im];
    let nrm = (v.iter().map(|x| x*x).sum::<f64>()).sqrt();
    assert!((nrm - 1.0).abs() < TOL, "V[:,0] not unit norm: {}", nrm);
    // y_i = Σ_j Σ_top[j, i] · v[j]  (Σ_topᵀ · v),  Σ_top = sigma[0..4][..]
    let mut y = [0.0_f64; 8];
    for i in 0..8 { for j in 0..4 { y[i] += sigma[j][i] * v[j]; }}
    y
}

// ────────────────────────────────────────────────────────────────────────────
// Reference brute-force search — used as a fallback for crate_search().
// Replace the call in crate_search() with your production search.
// ────────────────────────────────────────────────────────────────────────────

fn reference_search(y: &[f64; 8], k: u32, eps: f64) -> Vec<[i64; 8]> {
    let target_alpha: i64  = 1 << k;
    let target_norm:  f64  = (1 << (k + 1)) as f64;
    let threshold:    f64  = (1 << k) as f64 * (1.0 - eps*eps);

    // Each |x_i| ≤ √(2^{k+1}) because G has minimum eigenvalue 1.
    let b = (target_norm.sqrt().ceil() as i64) + 1;
    let gram = crate_gram();

    let mut sols = Vec::new();
    for a1 in -b..=b {
    for b1 in -b..=b {
    for c1 in -b..=b {
    for d1 in -b..=b {
        let u: ZXi = [a1, b1, c1, d1];
        let au = alpha(u);
        if au > target_alpha { continue; }
        let bu = beta(u);
        let rem_alpha = target_alpha - au;
        let rem_beta  = -bu;
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
// Round-trip: does x decode to V[:,0] within ε?
// ────────────────────────────────────────────────────────────────────────────

fn x_to_col0(x: &[i64; 8], k: u32) -> [Complex64; 2] {
    let u: ZXi = [x[0], x[1], x[2], x[3]];
    let t: ZXi = [x[4], x[5], x[6], x[7]];
    let scale = SQRT2.powi(k as i32);
    [zxi_to_c(u) / Complex64::new(scale, 0.0),
     zxi_to_c(t) / Complex64::new(scale, 0.0)]
}

/// L² distance ‖ηV[:,0] − col₀(x)‖, minimised over η ∈ {ξ⁰,…,ξ¹¹}.
fn col0_distance(x: &[i64; 8], k: u32, v_col0: [Complex64; 2]) -> f64 {
    let cand = x_to_col0(x, k);
    let x_xi = xi();
    let mut best = f64::INFINITY;
    for eta_idx in 0..12 {
        let eta = x_xi.powi(eta_idx);
        let d0 = eta * v_col0[0] - cand[0];
        let d1 = eta * v_col0[1] - cand[1];
        let d = (d0.norm_sqr() + d1.norm_sqr()).sqrt();
        if d < best { best = d; }
    }
    best
}

// ────────────────────────────────────────────────────────────────────────────
// Verify a single x against the four search invariants explicitly
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
    let bsum = beta(u) + beta(t);
    if bsum != 0 {
        return Err(format!("β-sum {} ≠ 0", bsum));
    }
    let xf: [f64; 8] = std::array::from_fn(|i| x[i] as f64);
    let gx = mat_vec_8(gram, &xf);
    let nsq: f64 = xf.iter().zip(gx.iter()).map(|(a,b)| a*b).sum();
    let expected_norm = (1 << (k+1)) as f64;
    if (nsq - expected_norm).abs() > 1e-9 {
        return Err(format!("‖x‖²_G = {} ≠ 2^{} = {}", nsq, k+1, expected_norm));
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
// Gate zoo
// ────────────────────────────────────────────────────────────────────────────

type Mat2 = [[Complex64; 2]; 2];

fn c(re: f64, im: f64) -> Complex64 { Complex64::new(re, im) }
fn mat2_mul(a: Mat2, b: Mat2) -> Mat2 {
    let mut out = [[c(0.0, 0.0); 2]; 2];
    for i in 0..2 { for j in 0..2 { for k in 0..2 {
        out[i][j] = out[i][j] + a[i][k] * b[k][j];
    }}}
    out
}
fn mat2_prod(ms: &[Mat2]) -> Mat2 {
    let mut out = [[c(1.0, 0.0), c(0.0, 0.0)], [c(0.0, 0.0), c(1.0, 0.0)]];
    for m in ms { out = mat2_mul(out, *m); }
    out
}
fn gate_h() -> Mat2 {
    let s = 1.0 / SQRT2;
    [[c(s, 0.0), c( s, 0.0)], [c(s, 0.0), c(-s, 0.0)]]
}
fn gate_s() -> Mat2 { [[c(1.0, 0.0), c(0.0, 0.0)], [c(0.0, 0.0), c(0.0, 1.0)]] }
fn gate_r() -> Mat2 { [[c(1.0, 0.0), c(0.0, 0.0)], [c(0.0, 0.0), xi()]] }

// ────────────────────────────────────────────────────────────────────────────
// Per-gate check
// ────────────────────────────────────────────────────────────────────────────

struct SearchResult {
    name:        &'static str,
    found:       usize,        // number returned by search
    best_dist:   f64,          // smallest col-0 distance among returned
    bounds_ok:   bool,         // every returned x passed verify_x_bounds
    decoded_ok:  bool,         // every returned x decoded within ε
    ok:          bool,         // overall: found ≥ 1, bounds_ok, decoded_ok
}

fn check_search_for_gate(name: &'static str, v_mat: Mat2, k: u32,
                         sigma: &[[f64; 8]; 8], gram: &[[f64; 8]; 8],
                         eps: f64) -> SearchResult {
    let v_col0 = [v_mat[0][0], v_mat[1][0]];
    let y      = compute_y(v_col0, sigma);
    let sols   = crate_search(&y, k, eps);

    let mut bounds_ok  = true;
    let mut decoded_ok = true;
    let mut best_dist  = f64::INFINITY;

    for x in &sols {
        if let Err(msg) = verify_x_bounds(x, &y, k, eps, gram) {
            println!("    ✗ {} returned x = {:?} violating bounds: {}", name, x, msg);
            bounds_ok = false;
        }
        let d = col0_distance(x, k, v_col0);
        if d > eps {
            println!("    ✗ {} returned x = {:?} decoding to col-0 distance {:.3e} > ε",
                     name, x, d);
            decoded_ok = false;
        }
        if d < best_dist { best_dist = d; }
    }

    let ok = !sols.is_empty() && bounds_ok && decoded_ok && best_dist <= eps;
    SearchResult {
        name, found: sols.len(), best_dist, bounds_ok, decoded_ok, ok,
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Entry point — runs both as `cargo run` and `cargo test`
// ────────────────────────────────────────────────────────────────────────────

fn run_all() -> bool {
    let sigma = crate_sigma();
    let gram  = crate_gram();

    let h = gate_h();
    let s = gate_s();
    let r = gate_r();

    let zoo: Vec<(&'static str, Mat2, u32)> = vec![
        ("H",       h,                            1),
        ("H·R",     mat2_prod(&[h, r]),           1),
        ("H·S·R",   mat2_prod(&[h, s, r]),        1),
        ("H·R^2",   mat2_prod(&[h, r, r]),        1),
        ("R·H·R",   mat2_prod(&[r, h, r]),        1),
        ("H·R·H",   mat2_prod(&[h, r, h]),        2),
    ];

    let mut results = Vec::new();
    for (name, v_mat, k) in zoo {
        println!("\n--- {}  (k = {}) ---", name, k);
        let r = check_search_for_gate(name, v_mat, k, &sigma, &gram, EPS);
        println!("  search returned {} solution(s); best col-0 distance = {:.3e}",
                 r.found, r.best_dist);
        if r.ok {
            println!("  ✓ all returned x pass the four bounds AND decode within ε");
        }
        results.push(r);
    }

    println!("\n========================================================================");
    println!("SEARCH-LAYER SUMMARY");
    println!("========================================================================");
    let n_pass = results.iter().filter(|r| r.ok).count();
    for r in &results {
        let tag = if r.ok { "✓" } else { "✗" };
        println!("  {}  {:12} found={:3}  best_dist={:.2e}  bounds_ok={}  decoded_ok={}",
                 tag, r.name, r.found, r.best_dist, r.bounds_ok, r.decoded_ok);
    }
    let all_ok = n_pass == results.len();
    println!("\n  OVERALL: {} ({}/{})",
             if all_ok { "PASS" } else { "FAIL" }, n_pass, results.len());
    all_ok
}

fn main() {
    let ok = run_all();
    if !ok { std::process::exit(1); }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn n6_search_contract() {
        assert!(run_all(), "n=6 search-layer contract failed");
    }
}
