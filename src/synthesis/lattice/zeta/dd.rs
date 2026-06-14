//! Inline double-double primitives (~106 bits, ~32 decimal digits).
//!
//! ~1e-33 relative error on sqrt, recip, div, Cholesky, and dot-product
//! cases. Used by `se::verify_partial_dd_exceeds` for fast prune
//! verification — ~10× cheaper than rug-128 in the hot loop because no heap
//! allocation and no mpfr_t init/clear per op. Values are bare `(f64, f64)`
//! hi/lo pairs (no wrapper struct) so the compiler keeps them in registers.

#[inline]
fn dd_quick_two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let err = b - (s - a);
    (s, err)
}

#[inline]
fn dd_two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    let err = (a - (s - bb)) + (b - bb);
    (s, err)
}

#[inline]
fn dd_two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    let err = a.mul_add(b, -p);
    (p, err)
}

/// Robust ("ieee") dd_add: separately captures lo-part sum via two_sum.
/// Handles cancellation in a.0 + b.0 correctly.
#[inline]
pub fn dd_add(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    let (s1, e1) = dd_two_sum(a.0, b.0);
    let (s2, e2) = dd_two_sum(a.1, b.1);
    let e1 = e1 + s2;
    let (s, e1) = dd_quick_two_sum(s1, e1);
    let e1 = e1 + e2;
    dd_quick_two_sum(s, e1)
}

#[inline]
pub fn dd_sub(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    dd_add(a, (-b.0, -b.1))
}

#[inline]
pub fn dd_mul(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    let (p, e) = dd_two_prod(a.0, b.0);
    let e = e + a.0 * b.1 + a.1 * b.0;
    dd_quick_two_sum(p, e)
}

#[inline]
pub fn dd_recip(b: (f64, f64)) -> (f64, f64) {
    let r0 = 1.0 / b.0;
    let r0_dd = (r0, 0.0);
    let bp = dd_mul(b, r0_dd);
    let two_minus_bp = dd_sub((2.0, 0.0), bp);
    dd_mul(r0_dd, two_minus_bp)
}

#[inline]
pub fn dd_div(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    dd_mul(a, dd_recip(b))
}

#[inline]
pub fn dd_sqrt(s: (f64, f64)) -> (f64, f64) {
    if s.0 <= 0.0 { return (0.0, 0.0); }
    let x = s.0.sqrt();
    let x_dd = (x, 0.0);
    let x_sq = dd_mul(x_dd, x_dd);
    let resid = dd_sub(s, x_sq);
    let two_x = dd_add(x_dd, x_dd);
    let corr = dd_div(resid, two_x);
    dd_add(x_dd, corr)
}

/// Convert i64 → dd. Exact for any i64 (since |z| ≤ 2^63 fits in dd's
/// 2^106 range; two-piece split if |z| > 2^53).
#[inline]
pub fn dd_from_i64(z: i64) -> (f64, f64) {
    if z.unsigned_abs() <= (1u64 << 53) {
        (z as f64, 0.0)
    } else {
        let neg = z < 0;
        let abs = z.unsigned_abs();
        let hi = (abs >> 32) as u32 as f64;
        let lo = (abs & 0xFFFFFFFF) as u32 as f64;
        let two32 = (1u64 << 32) as f64;
        let p = dd_mul((hi, 0.0), (two32, 0.0));
        let r = dd_add(p, (lo, 0.0));
        if neg { (-r.0, -r.1) } else { r }
    }
}
