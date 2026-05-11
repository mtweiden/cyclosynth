//! Unit tests for inline double-double primitives, validating against rug-128.
//! Run this to find dd implementation bugs in isolation, faster than re-running
//! the cliff probe.

use rug::Float;

// ─── Inline DD primitives (the version under test) ────────────────────────────

type DD = (f64, f64);

#[inline]
fn quick_two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let err = b - (s - a);
    (s, err)
}

#[inline]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    let err = (a - (s - bb)) + (b - bb);
    (s, err)
}

#[inline]
fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    let err = a.mul_add(b, -p);
    (p, err)
}

#[inline]
fn dd_add(a: DD, b: DD) -> DD {
    // "ieee"/robust dd_add: separately accumulate lo parts via two_sum,
    // then merge. Tolerates cancellation in a.0 + b.0.
    let (s1, e1) = two_sum(a.0, b.0);
    let (s2, e2) = two_sum(a.1, b.1);
    let e1 = e1 + s2;
    let (s, e1) = quick_two_sum(s1, e1);
    let e1 = e1 + e2;
    quick_two_sum(s, e1)
}

#[inline]
fn dd_sub(a: DD, b: DD) -> DD { dd_add(a, (-b.0, -b.1)) }

#[inline]
fn dd_mul(a: DD, b: DD) -> DD {
    let (p, e) = two_prod(a.0, b.0);
    let e = e + a.0 * b.1 + a.1 * b.0;
    quick_two_sum(p, e)
}

#[inline]
fn dd_from_f64(a: f64) -> DD { (a, 0.0) }

#[inline]
fn dd_to_f64(a: DD) -> f64 { a.0 + a.1 }

#[inline]
fn dd_recip(b: DD) -> DD {
    let r0 = 1.0 / b.0;
    let r0_dd = dd_from_f64(r0);
    let bp = dd_mul(b, r0_dd);
    let two = (2.0_f64, 0.0_f64);
    let two_minus_bp = dd_sub(two, bp);
    dd_mul(r0_dd, two_minus_bp)
}

#[inline]
fn dd_div(a: DD, b: DD) -> DD { dd_mul(a, dd_recip(b)) }

#[inline]
fn dd_sqrt(s: DD) -> DD {
    if s.0 <= 0.0 { return (0.0, 0.0); }
    let x = s.0.sqrt();
    let x_dd = dd_from_f64(x);
    let x_sq = dd_mul(x_dd, x_dd);
    let resid = dd_sub(s, x_sq);
    let two_x = dd_add(x_dd, x_dd);
    let corr = dd_div(resid, two_x);
    dd_add(x_dd, corr)
}

// ─── Test scaffolding ────────────────────────────────────────────────────────

fn ulp_diff_at_dd_scale(dd_val: DD, rug_val: &Float) -> f64 {
    // Convert dd to rug-128 to compare lossless.
    let dd_as_f128 = Float::with_val(128, dd_val.0) + Float::with_val(128, dd_val.1);
    let diff: Float = Float::with_val(128, &dd_as_f128 - rug_val);
    let rug_abs = rug_val.clone().abs();
    if rug_abs.is_zero() {
        return diff.to_f64().abs();
    }
    let rel = Float::with_val(128, &diff / &rug_abs).to_f64();
    rel.abs()
}

fn test(name: &str, dd_val: DD, rug_val: Float, max_rel_err: f64) {
    let rel = ulp_diff_at_dd_scale(dd_val, &rug_val);
    let dd_f = dd_to_f64(dd_val);
    let rug_f = rug_val.to_f64();
    let status = if rel < max_rel_err { "PASS" } else { "FAIL" };
    println!(
        "  [{status}] {name:30}  rel_err = {rel:>12.4e}  dd→f64 = {dd_f:>22.15e}  rug→f64 = {rug_f:>22.15e}"
    );
}

fn main() {
    println!("=== Inline dd unit tests vs rug-128 ===\n");

    // Test 1: sqrt(2)
    let s = (2.0_f64, 0.0_f64);
    let dd_r = dd_sqrt(s);
    let rug_r = Float::with_val(128, 2.0).sqrt();
    test("sqrt(2)", dd_r, rug_r, 1e-30);

    // Test 2: sqrt(3)
    let s = (3.0_f64, 0.0_f64);
    let dd_r = dd_sqrt(s);
    let rug_r = Float::with_val(128, 3.0).sqrt();
    test("sqrt(3)", dd_r, rug_r, 1e-30);

    // Test 3: 1/3
    let b = (3.0_f64, 0.0_f64);
    let dd_r = dd_recip(b);
    let rug_r = Float::with_val(128, 1.0) / Float::with_val(128, 3.0);
    test("1/3", dd_r, rug_r, 1e-30);

    // Test 4: 1/7
    let b = (7.0_f64, 0.0_f64);
    let dd_r = dd_recip(b);
    let rug_r = Float::with_val(128, 1.0) / Float::with_val(128, 7.0);
    test("1/7", dd_r, rug_r, 1e-30);

    // Test 5: dd_add catastrophic cancellation: (1e16 + 1) - 1e16
    let a = dd_from_f64(1e16);
    let b = dd_add(a, dd_from_f64(1.0));
    let c = dd_sub(b, dd_from_f64(1e16));
    // Expected: exactly 1.0
    let rug_r = Float::with_val(128, 1.0);
    test("(1e16+1) - 1e16", c, rug_r, 1e-30);

    // Test 6: simulating SE walk dot product magnitude.
    // Compute Σ R[k] · z[k] for R values ~1 and z values ~1e15, with cancellation.
    let r_vals: [f64; 4] = [1.0, -2.0, 1.5, -0.5];
    let z_vals: [f64; 4] = [1e15, 1e15, 2e15, -1e15];
    let mut total_dd: DD = (0.0, 0.0);
    let mut total_rug = Float::with_val(128, 0.0);
    for k in 0..4 {
        let prod_dd = dd_mul(dd_from_f64(r_vals[k]), dd_from_f64(z_vals[k]));
        total_dd = dd_add(total_dd, prod_dd);
        let mut tmp = Float::with_val(128, r_vals[k]);
        tmp *= Float::with_val(128, z_vals[k]);
        total_rug += tmp;
    }
    // Expected: 1·1e15 + (-2)·1e15 + 1.5·2e15 + (-0.5)·(-1e15) = 1e15 - 2e15 + 3e15 + 0.5e15 = 2.5e15
    test("dot product (R·z, R~1 z~1e15)", total_dd, total_rug, 1e-30);

    // Test 7: simulate Cholesky-like decomposition.
    // G = [[5, 2], [2, 3]], expected L = [[sqrt(5), 0], [2/sqrt(5), sqrt(3 - 4/5)]]
    let g_dd: [[DD; 2]; 2] = [
        [dd_from_f64(5.0), dd_from_f64(2.0)],
        [dd_from_f64(2.0), dd_from_f64(3.0)],
    ];
    let mut l_dd: [[DD; 2]; 2] = [[(0.0, 0.0); 2]; 2];
    // L[0][0] = sqrt(G[0][0])
    l_dd[0][0] = dd_sqrt(g_dd[0][0]);
    // L[1][0] = G[1][0] / L[0][0]
    l_dd[1][0] = dd_div(g_dd[1][0], l_dd[0][0]);
    // L[1][1] = sqrt(G[1][1] - L[1][0]²)
    let l10_sq = dd_mul(l_dd[1][0], l_dd[1][0]);
    let resid = dd_sub(g_dd[1][1], l10_sq);
    l_dd[1][1] = dd_sqrt(resid);

    let rug_l00 = Float::with_val(128, 5.0).sqrt();
    let rug_l10 = Float::with_val(128, 2.0) / rug_l00.clone();
    let rug_l10_sq = rug_l10.clone() * &rug_l10;
    let rug_l11 = (Float::with_val(128, 3.0) - rug_l10_sq).sqrt();

    test("chol[0][0]=sqrt(5)", l_dd[0][0], rug_l00, 1e-30);
    test("chol[1][0]=2/sqrt(5)", l_dd[1][0], rug_l10, 1e-30);
    test("chol[1][1]=sqrt(11/5)", l_dd[1][1], rug_l11, 1e-30);
}
