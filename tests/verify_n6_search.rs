//! verify_n6_search.rs — search-layer integration test for n=6.
//!
//! Wired to production crate functions:
//!   crate_search  → direct_search_n6(1_i64 << k, y, eps, usize::MAX)
//!   crate_sigma   → sigma_matrix() with row permutation [0,1,4,5,2,3,6,7]
//!   crate_gram    → rebuilt as ΣᵀΣ from the permuted sigma
//!
//! All logic, invariants, and gate zoo are unchanged from the standalone.

use cyclosynth::synthesis::clifford_pi6::{direct_search_n6, sigma_matrix};
use num_complex::Complex64;
use std::f64::consts::PI;

const SQRT2: f64 = std::f64::consts::SQRT_2;
const TOL: f64 = 1e-9;
const EPS: f64 = 1e-2;

// ── wire-up 1/2: production search ─────────────────────────────────────────

fn crate_search(y: &[f64; 8], k: u32, eps: f64) -> Vec<[i64; 8]> {
    direct_search_n6(1_i64 << k, y, eps, usize::MAX)
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
//
// y = Σ_topᵀ · v, where Σ_top = sigma[0..4] = [σ₁u, σ₁t] rows.

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
    let x_xi = xi();
    let mut best = f64::INFINITY;
    for eta_idx in 0..12 {
        let eta = x_xi.powi(eta_idx);
        let d0 = eta * v_col0[0] - cand[0];
        let d1 = eta * v_col0[1] - cand[1];
        let d = (d0.norm_sqr() + d1.norm_sqr()).sqrt();
        if d < best {
            best = d;
        }
    }
    best
}

// ── verify a single x against all four search invariants ─────────────────────

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
    let bsum = beta(u) + beta(t);
    if bsum != 0 {
        return Err(format!("β-sum {} ≠ 0", bsum));
    }
    let xf: [f64; 8] = std::array::from_fn(|i| x[i] as f64);
    let gx = mat_vec_8(gram, &xf);
    let nsq: f64 = xf.iter().zip(gx.iter()).map(|(a, b)| a * b).sum();
    let expected_norm = (1 << (k + 1)) as f64;
    if (nsq - expected_norm).abs() > 1e-9 {
        return Err(format!(
            "‖x‖²_G = {} ≠ 2^{} = {}",
            nsq,
            k + 1,
            expected_norm
        ));
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

// ── gate zoo ─────────────────────────────────────────────────────────────────

type Mat2 = [[Complex64; 2]; 2];

fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}
fn mat2_mul(a: Mat2, b: Mat2) -> Mat2 {
    let mut out = [[c(0.0, 0.0); 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                out[i][j] = out[i][j] + a[i][k] * b[k][j];
            }
        }
    }
    out
}
fn mat2_prod(ms: &[Mat2]) -> Mat2 {
    let mut out = [[c(1.0, 0.0), c(0.0, 0.0)], [c(0.0, 0.0), c(1.0, 0.0)]];
    for m in ms {
        out = mat2_mul(out, *m);
    }
    out
}
fn gate_h() -> Mat2 {
    let s = 1.0 / SQRT2;
    [[c(s, 0.0), c(s, 0.0)], [c(s, 0.0), c(-s, 0.0)]]
}
fn gate_s() -> Mat2 {
    [[c(1.0, 0.0), c(0.0, 0.0)], [c(0.0, 0.0), c(0.0, 1.0)]]
}
fn gate_r() -> Mat2 {
    [[c(1.0, 0.0), c(0.0, 0.0)], [c(0.0, 0.0), xi()]]
}

// ── per-gate check ────────────────────────────────────────────────────────────

struct SearchResult {
    name: &'static str,
    found: usize,
    best_dist: f64,
    bounds_ok: bool,
    decoded_ok: bool,
    ok: bool,
}

fn check_search_for_gate(
    name: &'static str,
    v_mat: Mat2,
    k: u32,
    sigma: &[[f64; 8]; 8],
    gram: &[[f64; 8]; 8],
    eps: f64,
) -> SearchResult {
    let v_col0 = [v_mat[0][0], v_mat[1][0]];
    let y = compute_y(v_col0, sigma);
    let sols = crate_search(&y, k, eps);

    let mut bounds_ok = true;
    let mut decoded_ok = true;
    let mut best_dist = f64::INFINITY;

    for x in &sols {
        if let Err(msg) = verify_x_bounds(x, &y, k, eps, gram) {
            println!(
                "    ✗ {} returned x = {:?} violating bounds: {}",
                name, x, msg
            );
            bounds_ok = false;
        }
        let d = col0_distance(x, k, v_col0);
        if d > eps {
            println!(
                "    ✗ {} returned x = {:?} decoding to col-0 distance {:.3e} > ε",
                name, x, d
            );
            decoded_ok = false;
        }
        if d < best_dist {
            best_dist = d;
        }
    }

    let ok = !sols.is_empty() && bounds_ok && decoded_ok && best_dist <= eps;
    SearchResult {
        name,
        found: sols.len(),
        best_dist,
        bounds_ok,
        decoded_ok,
        ok,
    }
}

// ── main loop ─────────────────────────────────────────────────────────────────

fn run_all() -> bool {
    let sigma = crate_sigma();
    let gram = crate_gram();

    let h = gate_h();
    let s = gate_s();
    let r = gate_r();

    let zoo: &[(&'static str, Mat2, u32)] = &[
        ("H", h, 1),
        ("H·R", mat2_prod(&[h, r]), 1),
        ("H·S·R", mat2_prod(&[h, s, r]), 1),
        ("H·R^2", mat2_prod(&[h, r, r]), 1),
        ("R·H·R", mat2_prod(&[r, h, r]), 1),
        ("H·R·H", mat2_prod(&[h, r, h]), 2),
    ];

    let mut results = Vec::new();
    for &(name, v_mat, k) in zoo {
        println!("\n--- {}  (k = {}) ---", name, k);
        let res = check_search_for_gate(name, v_mat, k, &sigma, &gram, EPS);
        println!(
            "  search returned {} solution(s); best col-0 distance = {:.3e}",
            res.found, res.best_dist
        );
        if res.ok {
            println!("  ✓ all returned x pass the four bounds AND decode within ε");
        }
        results.push(res);
    }

    println!("\n========================================================================");
    println!("SEARCH-LAYER SUMMARY");
    println!("========================================================================");
    let n_pass = results.iter().filter(|r| r.ok).count();
    for r in &results {
        let tag = if r.ok { "✓" } else { "✗" };
        println!(
            "  {}  {:12} found={:3}  best_dist={:.2e}  bounds_ok={}  decoded_ok={}",
            tag, r.name, r.found, r.best_dist, r.bounds_ok, r.decoded_ok
        );
    }
    let all_ok = n_pass == results.len();
    println!(
        "\n  OVERALL: {} ({}/{})",
        if all_ok { "PASS" } else { "FAIL" },
        n_pass,
        results.len()
    );
    all_ok
}

#[test]
fn n6_search_contract() {
    assert!(run_all(), "n=6 search-layer contract failed");
}
