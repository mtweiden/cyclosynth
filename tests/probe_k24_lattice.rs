//! probe_k24_lattice.rs — single-shot probe of the lattice path at k=24.
//!
//! Built by lattice-side construction (no gate-word required):
//!   u = (2048, 0, 2048, 0)   ⇒ α(u) = 3·2048² = 12,582,912
//!   t = (2048, 0, 0,    0)   ⇒ α(t) = 2048²   =  4,194,304
//!   Σ α      = 16,777,216 = 2^24   ✓
//!   Σ β      = 0                   ✓
//!   V[:,0]   = ((1+ξ²)/2, 1/2) = (0.75 + i·√3/4, 0.5)
//!   |V[:,0]|² = 0.75 + 0.25 = 1   ✓
//!
//! Tests that direct_search_n6(2^24, y, eps=1e-3, ...) returns a valid
//! decoding solution in reasonable time.  This probes whether the new
//! n=4-style dispatch (which would feed k_inner = 24 at outer k=25..30)
//! is viable at all.

use cyclosynth::synthesis::clifford_pi6::{direct_search_n6, sigma_matrix};
use num_complex::Complex64;
use std::f64::consts::PI;
use std::time::Instant;

const SQRT2: f64 = std::f64::consts::SQRT_2;
const EPS: f64 = 1e-3;

const PERM: [usize; 8] = [0, 1, 4, 5, 2, 3, 6, 7];

fn crate_sigma() -> [[f64; 8]; 8] {
    let s = sigma_matrix();
    let mut out = [[0.0_f64; 8]; 8];
    for (new_row, &old_row) in PERM.iter().enumerate() {
        out[new_row] = s[old_row];
    }
    out
}

fn xi() -> Complex64 {
    Complex64::from_polar(1.0, PI / 6.0)
}

fn alpha(p: [i64; 4]) -> i64 {
    p[0] * p[0] + p[1] * p[1] + p[2] * p[2] + p[3] * p[3] + p[0] * p[2] + p[1] * p[3]
}
fn beta(p: [i64; 4]) -> i64 {
    p[0] * p[1] + p[1] * p[2] + p[2] * p[3]
}

fn zxi_to_c(p: [i64; 4]) -> Complex64 {
    let x = xi();
    Complex64::new(p[0] as f64, 0.0)
        + Complex64::new(p[1] as f64, 0.0) * x
        + Complex64::new(p[2] as f64, 0.0) * x * x
        + Complex64::new(p[3] as f64, 0.0) * x * x * x
}

fn col0_distance(x: &[i64; 8], k: u32, v_col0: [Complex64; 2]) -> f64 {
    let u: [i64; 4] = [x[0], x[1], x[2], x[3]];
    let t: [i64; 4] = [x[4], x[5], x[6], x[7]];
    let scale = SQRT2.powi(k as i32);
    let cand = [
        zxi_to_c(u) / Complex64::new(scale, 0.0),
        zxi_to_c(t) / Complex64::new(scale, 0.0),
    ];
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

fn compute_y(v_col0: [Complex64; 2], sigma: &[[f64; 8]; 8]) -> [f64; 8] {
    let v = [v_col0[0].re, v_col0[0].im, v_col0[1].re, v_col0[1].im];
    let mut y = [0.0_f64; 8];
    for i in 0..8 {
        for j in 0..4 {
            y[i] += sigma[j][i] * v[j];
        }
    }
    y
}

#[test]
fn probe_k24_lattice() {
    let k: u32 = 24;
    let target_k: i64 = 1_i64 << k;

    let expected_x: [i64; 8] = [2048, 0, 2048, 0, 2048, 0, 0, 0];
    let u: [i64; 4] = [expected_x[0], expected_x[1], expected_x[2], expected_x[3]];
    let t: [i64; 4] = [expected_x[4], expected_x[5], expected_x[6], expected_x[7]];

    println!("Probe target: u=(2048,0,2048,0), t=(2048,0,0,0)  (constructed at k=24)");
    println!(
        "  α(u)={}, α(t)={}, sum={}  (must = 2^24 = {})",
        alpha(u),
        alpha(t),
        alpha(u) + alpha(t),
        target_k
    );
    println!(
        "  β(u)={}, β(t)={}, sum={}  (must = 0)",
        beta(u),
        beta(t),
        beta(u) + beta(t)
    );
    assert_eq!(alpha(u) + alpha(t), target_k);
    assert_eq!(beta(u) + beta(t), 0);

    let scale = SQRT2.powi(k as i32);
    let v_col0 = [
        zxi_to_c(u) / Complex64::new(scale, 0.0),
        zxi_to_c(t) / Complex64::new(scale, 0.0),
    ];
    let unit_err = (v_col0[0].norm_sqr() + v_col0[1].norm_sqr() - 1.0).abs();
    println!(
        "  V[:,0] = [{:+.6} + {:+.6}i, {:+.6} + {:+.6}i]   ‖V[:,0]‖² − 1 = {:.3e}",
        v_col0[0].re, v_col0[0].im, v_col0[1].re, v_col0[1].im, unit_err
    );
    assert!(unit_err < 1e-12, "V[:,0] must be unit-norm");

    let d_expected = col0_distance(&expected_x, k, v_col0);
    println!("  expected_x decodes at distance {:.3e}", d_expected);
    assert!(
        d_expected < 1e-12,
        "expected_x should decode to V[:,0] at machine precision"
    );

    let sigma = crate_sigma();
    let y = compute_y(v_col0, &sigma);

    println!(
        "\nCalling direct_search_n6(2^{}, y, eps={:.0e}, max_sol=usize::MAX)...",
        k, EPS
    );
    let t0 = Instant::now();
    let sols = direct_search_n6(target_k, &y, EPS, usize::MAX);
    let elapsed = t0.elapsed();
    println!(
        "  → {} solutions in {:.3}s",
        sols.len(),
        elapsed.as_secs_f64()
    );

    if sols.is_empty() {
        panic!("lattice path returned 0 solutions at k=24");
    }

    let mut best_dist = f64::INFINITY;
    let mut all_bounds_ok = true;
    let mut all_decode_ok = true;
    for x in &sols {
        let xu: [i64; 4] = [x[0], x[1], x[2], x[3]];
        let xt: [i64; 4] = [x[4], x[5], x[6], x[7]];
        let a_sum = alpha(xu) + alpha(xt);
        let b_sum = beta(xu) + beta(xt);
        if a_sum != target_k || b_sum != 0 {
            println!(
                "  ✗ returned x = {:?} has α-sum={}, β-sum={}",
                x, a_sum, b_sum
            );
            all_bounds_ok = false;
        }
        let d = col0_distance(x, k, v_col0);
        if d > EPS {
            println!("  ✗ returned x decodes at distance {:.3e} > ε", d);
            all_decode_ok = false;
        }
        if d < best_dist {
            best_dist = d;
        }
    }
    println!("  best decoded distance: {:.3e}", best_dist);
    println!("  all returned bounds ok:       {}", all_bounds_ok);
    println!("  all returned decode within ε: {}", all_decode_ok);
    println!("  TOTAL wall-clock: {:.3}s", elapsed.as_secs_f64());

    assert!(all_bounds_ok);
    assert!(all_decode_ok);
    assert!(best_dist <= EPS);
}
