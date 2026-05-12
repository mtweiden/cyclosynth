//! Schnorr-Euchner enumeration over the 16D integer lattice (Z[ζ_16] flow).
//!
//! ## Status (M4 chunk 2)
//!
//! Full Schnorr-Euchner walk with Q-bound pruning, in f64 arithmetic. Mirrors
//! the 8D path in `super::super::lenstra::se`, dimension-bumped to d=16 and
//! switched from MPFR to f64 for the inner walk: at d=16 with the L³-reduced
//! invariant after L²-LLL, the conditioning bound `κ(G) ≤ (4/3)^15 ≈ 240`
//! gives ~8 bits of conditioning loss in f64, well within the 53-bit mantissa
//! and four orders below SE's 10⁻⁹ tolerance.
//!
//! Helpers ported alongside the walk:
//!   - [`det16_exact`] — exact i64 determinant for unimodularity sanity checks
//!     after LLL. Returns `None` on overflow.
//!   - [`euclidean_cholesky_16`] — alternative f64 Cholesky path used as a
//!     numerical sanity-check oracle for `cholesky_f64_16`. Not exercised in
//!     production but ported for completeness so chunk 3 can wire either path.
//!
//! The walk's signature uses an **upper-triangular** Cholesky factor `l` such
//! that `lᵀ l = G` (post-LLL Gram in basis coords). The chunk-1 `l_f64` is
//! lower-triangular (`l_f64 · l_f64ᵀ = G`); chunk 3's call site transposes
//! before invoking SE.

#![allow(clippy::needless_range_loop)]

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::OnceLock;

use i256::i256;

/// Diagnostic-only: skip the partial Euclidean norm prune in the SE walk.
/// Initialized from `CYCLOSYNTH_BYPASS_NORM_PRUNE=1`, but **mutable at
/// runtime** via `set_bypass_norm_prune` so a single process can toggle
/// the prune between phases (e.g., probe Phase 1 captures with bypass on
/// and Phase 4 watches with bypass off). The leaf integer-exact
/// `‖x‖² == 2^k` check still arbitrates correctness.
static BYPASS_NORM_PRUNE: AtomicBool = AtomicBool::new(false);
static BYPASS_INIT: OnceLock<()> = OnceLock::new();

fn bypass_norm_prune() -> bool {
    BYPASS_INIT.get_or_init(|| {
        let v = std::env::var("CYCLOSYNTH_BYPASS_NORM_PRUNE").ok().as_deref() == Some("1");
        BYPASS_NORM_PRUNE.store(v, Ordering::Relaxed);
    });
    BYPASS_NORM_PRUNE.load(Ordering::Relaxed)
}

/// Diagnostic-only: override the bypass flag at runtime.
pub fn set_bypass_norm_prune(value: bool) {
    BYPASS_INIT.get_or_init(|| {});
    BYPASS_NORM_PRUNE.store(value, Ordering::Relaxed);
}

// Depth-1 Q-filter (phase 3). DEFAULT OFF.
//
// Sound: rejects only z[1] candidates with no integer z[0] making
// ‖x‖² = T exactly (the leaf filter's hard requirement). 99.98%
// rejection rate at the cliff per the qfilter measurement.
//
// Why default off: the filter interacts badly with the multi-lde
// search at deep ε. With filter active, depth-1 nodes are much
// cheaper than depth-0 recursion, so the per-node budget lets the
// walker explore ~9× wider at depth 1. At lde levels with no
// solution, the walker burns the entire budget at filter overhead
// (~200 ns/node × 9× wider × num_threads) instead of bailing
// quickly. Cliff probe (theta=1.1, ε=1.5e-8): lde=21 (no sol)
// took 631 s with filter vs ~5 s baseline.
//
// Set `CYCLOSYNTH_QFILTER=1` to enable, or call
// `set_qfilter_enabled(true)`. Unlocking this in production
// requires either: (a) bailing on "lde has no solution" before
// exhausting budget; (b) a separate per-lde budget; or (c) a
// cheaper depth-1 precompute.
static QFILTER_ENABLED: AtomicBool = AtomicBool::new(false);
static QFILTER_INIT: OnceLock<()> = OnceLock::new();

fn qfilter_enabled() -> bool {
    QFILTER_INIT.get_or_init(|| {
        let v = std::env::var("CYCLOSYNTH_QFILTER").ok().as_deref() == Some("1");
        QFILTER_ENABLED.store(v, Ordering::Relaxed);
    });
    QFILTER_ENABLED.load(Ordering::Relaxed)
}

/// Diagnostic-only: enable the depth-1 Q-filter at runtime.
pub fn set_qfilter_enabled(value: bool) {
    QFILTER_INIT.get_or_init(|| {});
    QFILTER_ENABLED.store(value, Ordering::Relaxed);
}

/// MPFR-128 verification of the f64 norm-shell prune predicate. When ON,
/// every f64 prune-fire event is re-checked at 128-bit precision using the
/// MPFR Cholesky factor; if MPFR says "keep" (true partial < threshold),
/// the prune does NOT actually fire. Necessary at ε ≤ 1.5e-8 where the f64
/// dot product suffers catastrophic cancellation (oracle-measured FN ratio
/// up to 3.8×, p99 = 3.5×). At shallower ε, leave OFF — f64 is precise enough
/// and the MPFR recompute would be pure overhead.
static VERIFY_PRUNE_MPFR: AtomicBool = AtomicBool::new(false);

#[inline]
pub fn verify_prune_mpfr() -> bool {
    VERIFY_PRUNE_MPFR.load(Ordering::Relaxed)
}

pub fn set_verify_prune_mpfr(value: bool) {
    VERIFY_PRUNE_MPFR.store(value, Ordering::Release);
}

// ─── Inline double-double primitives (~106 bits, ~32 decimal digits) ─────────
//
// Validated against rug-128 via `probe_dd_unit` to ~1e-33 relative error on
// sqrt, recip, div, Cholesky, and dot-product cases. Used by
// `verify_partial_dd_exceeds` for fast prune verification — ~10× cheaper
// than rug-128 in the hot loop because no heap allocation and no mpfr_t
// init/clear per op.

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
fn dd_from_i64(z: i64) -> (f64, f64) {
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

// ─── Analytical depth-0 z[0] selection ───────────────────────────────────────
//
// At depth 0 with z[1..16] fixed and x = B·z computed (with the current z[0]
// = z0_curr), the future ‖x_new‖² for any candidate z[0]_new = z0_curr + δ is
//   ‖x_new‖² = A + 2·δ·B + δ²·C
// where:
//   A = ‖x − z0_curr·basis[0]‖²   (≡ ‖x‖² with z[0] set to 0)
//   B = (x − z0_curr·basis[0]) · basis[0]
//   C = ‖basis[0]‖²
// All three are exact integers in i128. To hit the shell ‖x_new‖² = T = 2^k:
//   C·δ² + 2B·δ + (A − T) = 0
//   δ = (−B ± √(B² − C·(A − T))) / C
//
// We return up to 6 integer z[0] candidates: floor/ceil of each of the 2 roots
// plus ±1 nudges, filtered to the SE bracket [z_low, z_high]. Conservative
// (over-covers) so the leaf-filter's exact `‖x‖² == T` check arbitrates final
// correctness — we cannot miss a shell hit.
//
// Replaces the full depth-0 bracket enumeration (up to ~10 z[0] values per
// node), addressing the survivorship-data finding that ~75% of leaves come
// from depth-1-near-threshold nodes where the brute z[0] sweep produces ~99%
// far-above-shell candidates.

#[inline]
fn isqrt_i128(n: i128) -> i128 {
    if n < 0 { return -1; }
    if n < 2 { return n; }
    let mut x = n;
    let mut y = (n + 1) / 2;
    while y < x {
        x = y;
        y = (x + n / x) / 2;
    }
    x
}

#[inline]
fn floor_div_i128(a: i128, b: i128) -> i128 {
    // b > 0 in our use; assert here keeps the codegen tight.
    debug_assert!(b > 0);
    let q = a / b;
    let r = a % b;
    if r < 0 { q - 1 } else { q }
}

/// Find integer z[0] candidates that could yield `‖x_new‖² == target_norm`,
/// where `x_new = x − z0_curr·basis_0 + z0_new·basis_0`. Returns the count
/// of candidates written into `out` (at most 6); each is unique and inside
/// `[z_low, z_high]`. Returns 0 if no integer solution exists (discriminant
/// < 0 or all candidates fall outside the bracket).
#[inline]
pub fn analytical_depth0_z0_candidates(
    x: &[i64; 16],
    z0_curr: i64,
    basis_0: &[i64; 16],
    target_norm: i128,
    z_low: i64,
    z_high: i64,
    out: &mut [i64; 6],
) -> usize {
    let mut a: i128 = 0;
    let mut b: i128 = 0;
    let mut c: i128 = 0;
    for i in 0..16 {
        let b0 = basis_0[i] as i128;
        let xz = (x[i] as i128) - (z0_curr as i128) * b0;
        a += xz * xz;
        b += xz * b0;
        c += b0 * b0;
    }
    if c == 0 {
        // basis[0] = 0: degenerate row. Bail (let caller use enumeration).
        // In practice this doesn't happen with LLL-reduced bases.
        return 0;
    }
    let d = a - target_norm;
    // disc = B² − C·(A − T)
    let disc = b * b - c * d;
    if disc < 0 {
        return 0;
    }
    let sqrt_disc = isqrt_i128(disc);
    let mut n: usize = 0;
    // Two roots: (−B ± sqrt_disc) / C. Compute floor; nudge by {−1, 0, +1}
    // to cover rounding both for non-perfect-square disc and for integer-div
    // rounding directionality.
    for &sign in &[1_i128, -1_i128] {
        let numerator = sign * sqrt_disc - b;
        let q = floor_div_i128(numerator, c);
        for nudge in -1_i64..=1 {
            let cand_i128 = q + nudge as i128;
            // Range check: must fit in i64 and within [z_low, z_high].
            if cand_i128 < i64::MIN as i128 || cand_i128 > i64::MAX as i128 {
                continue;
            }
            let cand = cand_i128 as i64;
            if cand < z_low || cand > z_high {
                continue;
            }
            let mut already = false;
            for k in 0..n {
                if out[k] == cand {
                    already = true;
                    break;
                }
            }
            if !already && n < 6 {
                out[n] = cand;
                n += 1;
            }
        }
    }
    n
}

/// Depth-1 shell-discriminant state. At depth 1 with `z[2..15]` fixed, the
/// shell equation `‖x‖² = T` (T = `target_norm_sq_i64`) is the quadratic
/// `G_00·z[0]² + 2(G_01·z[1] + v_0)·z[0] + (G_11·z[1]² + 2·v_1·z[1] + A − T) = 0`
/// in z[0], parametrized by z[1]. For an integer solution to exist for a
/// given z[1], the discriminant must be ≥ 0 and a perfect square.
///
/// Returns `(G_00, G_01, G_11, A, v_0, v_1)` as i256. Magnitudes can exceed
/// i128 at cliff conditions (basis ~ 2^41, z_c ~ 2^43, so y_i ~ 2^88 and
/// y_i² sum ~ 2^180). i256 carries everything safely.
#[inline]
fn qfilter_depth1_state(
    basis: &[[i64; 16]; 16],
    x: &[i64; 16],
    z0_curr: i64,
    z1_curr: i64,
) -> (i256, i256, i256, i256, i256, i256) {
    let mut g_00 = i256::from_i64(0);
    let mut g_01 = i256::from_i64(0);
    let mut g_11 = i256::from_i64(0);
    for i in 0..16 {
        let b0 = i256::from_i64(basis[0][i]);
        let b1 = i256::from_i64(basis[1][i]);
        g_00 = g_00.wrapping_add(b0.wrapping_mul(b0));
        g_01 = g_01.wrapping_add(b0.wrapping_mul(b1));
        g_11 = g_11.wrapping_add(b1.wrapping_mul(b1));
    }
    let z0 = i256::from_i64(z0_curr);
    let z1 = i256::from_i64(z1_curr);
    let mut a = i256::from_i64(0);
    let mut v_0 = i256::from_i64(0);
    let mut v_1 = i256::from_i64(0);
    for i in 0..16 {
        let b0 = i256::from_i64(basis[0][i]);
        let b1 = i256::from_i64(basis[1][i]);
        let y_i = i256::from_i64(x[i])
            .wrapping_sub(z0.wrapping_mul(b0))
            .wrapping_sub(z1.wrapping_mul(b1));
        a = a.wrapping_add(y_i.wrapping_mul(y_i));
        v_0 = v_0.wrapping_add(y_i.wrapping_mul(b0));
        v_1 = v_1.wrapping_add(y_i.wrapping_mul(b1));
    }
    (g_00, g_01, g_11, a, v_0, v_1)
}

/// Newton's-method floor-isqrt for non-negative i256. Returns ⌊√n⌋. Caller
/// must ensure n ≥ 0 (returns garbage for n < 0).
///
/// Convergence: with a 2^⌈bits/2⌉ seed, Newton's quadratic convergence
/// reaches the fixed point in O(log(bits)) ≈ 7-8 iterations for full-i256.
#[inline]
fn isqrt_i256(n: i256) -> i256 {
    let zero = i256::from_i64(0);
    if n <= zero {
        return zero;
    }
    if n < i256::from_i64(4) {
        return i256::from_i64(1);
    }
    let bits = 256 - n.leading_zeros();
    let seed_shift = (bits + 1) / 2;
    let mut x = i256::from_i64(1).wrapping_shl(seed_shift);
    loop {
        let q = n.wrapping_div(x);
        if q >= x {
            break;
        }
        x = x.wrapping_add(q).wrapping_shr(1);
    }
    x
}

/// For a depth-1 z[1] candidate `zd`, classify the shell discriminant into
/// four buckets:
///   `0` — D < 0 (no real z[0] solution)
///   `1` — D ≥ 0 but mod-16 says not a perfect square (no integer z[0])
///   `2` — D ≥ 0, mod-16 OK, but isqrt²≠D (not a perfect square — no int z[0])
///   `3` — D ≥ 0 and D is a perfect square (filter passes — recurse)
///
/// The mod-16 test is a cheap necessary condition; the isqrt test is the
/// definitive integer-z[0]-existence check.
///
/// D = 4·D_per_4 where D_per_4 = `b_lin² − G_00·c`. D is a perfect square
/// iff D_per_4 is (since 4 = 2²). So we operate on D_per_4 throughout.
#[inline]
fn qfilter_discriminant_class(
    g_00: i256,
    g_01: i256,
    g_11: i256,
    a: i256,
    v_0: i256,
    v_1: i256,
    target_norm_sq_i64: i64,
    zd: i64,
) -> u8 {
    let zd_i = i256::from_i64(zd);
    let two = i256::from_i64(2);
    let b_lin = g_01.wrapping_mul(zd_i).wrapping_add(v_0);
    let c = g_11
        .wrapping_mul(zd_i)
        .wrapping_mul(zd_i)
        .wrapping_add(two.wrapping_mul(v_1).wrapping_mul(zd_i))
        .wrapping_add(a.wrapping_sub(i256::from_i64(target_norm_sq_i64)));
    let d_per_4 = b_lin
        .wrapping_mul(b_lin)
        .wrapping_sub(g_00.wrapping_mul(c));
    if d_per_4 < i256::from_i64(0) {
        return 0;
    }
    let rem = d_per_4.wrapping_rem_i128(4);
    let rem_pos = if rem < 0 { rem + 4 } else { rem };
    if rem_pos != 0 && rem_pos != 1 {
        return 1;
    }
    let s = isqrt_i256(d_per_4);
    if s.wrapping_mul(s) == d_per_4 { 3 } else { 2 }
}

/// Compute `Σ_{i ≥ depth} (R · z)[i]²` in inline double-double (~106 bits)
/// and return true iff the result exceeds `threshold`. No heap allocation,
/// no thread-local state — fully stack-resident. About 10× faster than the
/// rug-128 verify in the hot SE prune-firing path.
#[inline]
pub fn verify_partial_dd_exceeds(
    r_eucl_dd: &[[(f64, f64); 16]; 16],
    z: &[i64; 16],
    depth: usize,
    threshold: f64,
) -> bool {
    let mut total: (f64, f64) = (0.0, 0.0);
    for i in depth..16 {
        let mut row: (f64, f64) = (0.0, 0.0);
        for j in i..16 {
            let z_dd = dd_from_i64(z[j]);
            let term = dd_mul(r_eucl_dd[i][j], z_dd);
            row = dd_add(row, term);
        }
        let sq = dd_mul(row, row);
        total = dd_add(total, sq);
    }
    // total > threshold (compare hi + lo to threshold)
    total.0 + total.1 > threshold
}

// ─── Bilinear leaf checks ────────────────────────────────────────────────────

/// Per-element β_1: see `clifford_sqrt_t_research.md` for derivation.
/// Returns i128 to avoid silent overflow on pairwise products at deep k.
///
/// Mirror of [`super::super::lenstra::se::bilinear_b`] for the Z[ζ_16] /
/// Clifford+√T flow. Three forms here vs one in 8D because the
/// totally-real-subring decomposition of unitarity over Z[ζ_16] yields
/// three independent constraints (one per non-σ_1 Galois embedding).
#[inline]
pub fn beta_1(u: &[i64; 8]) -> i128 {
    let u: [i128; 8] = std::array::from_fn(|i| u[i] as i128);
    u[0]*u[1] + u[1]*u[2] + u[2]*u[3] + u[3]*u[4]
        + u[4]*u[5] + u[5]*u[6] + u[6]*u[7]
        - u[0]*u[7]
}

#[inline]
pub fn beta_2(u: &[i64; 8]) -> i128 {
    let u: [i128; 8] = std::array::from_fn(|i| u[i] as i128);
    u[0]*u[2] + u[1]*u[3] + u[2]*u[4] + u[3]*u[5]
        + u[4]*u[6] + u[5]*u[7]
        - u[0]*u[6] - u[1]*u[7]
}

#[inline]
pub fn beta_3(u: &[i64; 8]) -> i128 {
    let u: [i128; 8] = std::array::from_fn(|i| u[i] as i128);
    u[0]*u[3] + u[1]*u[4] + u[2]*u[5] + u[3]*u[6] + u[4]*u[7]
        - u[0]*u[5] - u[1]*u[6] - u[2]*u[7]
}

/// Joint bilinear forms on the 16-vector `x = (u_1's 8 coords, u_2's 8 coords)`.
/// Returns `(B_1, B_2, B_3)`. All three must equal 0 for a valid Clifford+√T
/// candidate (the totally-real-subring decomposition of unitarity).
#[inline]
pub fn bilinear_forms(x: &[i64; 16]) -> (i128, i128, i128) {
    let u1: [i64; 8] = x[0..8].try_into().unwrap();
    let u2: [i64; 8] = x[8..16].try_into().unwrap();
    (
        beta_1(&u1) + beta_1(&u2),
        beta_2(&u1) + beta_2(&u2),
        beta_3(&u1) + beta_3(&u2),
    )
}

// ─── Lattice-point reconstruction ────────────────────────────────────────────

/// Reconstruct the lattice point `x = B·z` where `B` is the LLL-reduced
/// basis (rows are basis vectors) and `z` are the SE-output coordinates.
///
/// Convention: `B[i]` is the i-th basis vector (a row), so
/// `x[j] = Σ_i z[i] · B[i][j]`.
#[inline]
pub fn reconstruct_x(b_lll: &[[i64; 16]; 16], z: &[i64; 16]) -> [i64; 16] {
    let mut x = [0i64; 16];
    for i in 0..16 {
        for j in 0..16 {
            x[j] += z[i] * b_lll[i][j];
        }
    }
    x
}

// ─── Exact 16×16 determinant via Gaussian elimination ────────────────────────

/// Exact integer determinant of a 16×16 i64 matrix via Bareiss
/// fraction-free elimination, working in i128. Returns `None` if the result
/// (or any intermediate) doesn't fit in i64; otherwise returns the exact
/// determinant.
///
/// Used after LLL to validate that the output basis is unimodular (det = ±1).
/// A non-unimodular result indicates the GS lost orthogonalization — for the
/// L²-LLL pipeline this should never happen at d=16, but the check is cheap
/// and catches algorithm bugs early.
///
/// **Overflow note**: At d=16 with post-LLL basis entries up to ~2^41 (deep
/// ε), Bareiss intermediates can transiently exceed i64. We use i128
/// throughout; if any intermediate value exceeds i128 range the result is
/// `None` (saturation). For unimodular bases the *final* det is ±1 so there
/// is no issue, but spurious overflow during elimination is possible at
/// pathological inputs.
pub fn det16_exact(m: &[[i64; 16]; 16]) -> Option<i64> {
    let mut a: [[i128; 16]; 16] =
        std::array::from_fn(|i| std::array::from_fn(|j| m[i][j] as i128));
    let mut sign: i128 = 1;
    let mut prev: i128 = 1;

    for k in 0..16 {
        if a[k][k] == 0 {
            let mut found = false;
            for i in (k + 1)..16 {
                if a[i][k] != 0 {
                    a.swap(k, i);
                    sign = -sign;
                    found = true;
                    break;
                }
            }
            if !found {
                return Some(0);
            }
        }
        let pivot = a[k][k];
        for i in (k + 1)..16 {
            for j in (k + 1)..16 {
                let lhs = a[i][j].checked_mul(pivot)?;
                let rhs = a[i][k].checked_mul(a[k][j])?;
                let diff = lhs.checked_sub(rhs)?;
                // diff is divisible by prev exactly (Bareiss invariant).
                a[i][j] = diff / prev;
            }
            a[i][k] = 0;
        }
        prev = pivot;
    }
    let det = a[15][15].checked_mul(sign)?;
    if det >= i64::MIN as i128 && det <= i64::MAX as i128 {
        Some(det as i64)
    } else {
        None
    }
}

// ─── Euclidean Cholesky (test-oracle / sanity check) ─────────────────────────

/// MPFR-precision Euclidean Cholesky. Compute `R · Rᵀ = B · Bᵀ` with the
/// factorization done at 128-bit precision, then snapshot R to f64. Used by
/// the norm-shell-pruned SE walk so the per-leaf `‖R·z‖²` accumulator drifts
/// only by f64 round-off (not the much larger error from doing Cholesky
/// itself in f64). This matters at deep k where post-LLL basis entries can
/// be up to ~2^15+: the Gram reaches ~2^34+, and f64 Cholesky on those
/// values drifts by 0.1 % or more, corrupting the prune threshold.
///
/// Returns `None` if the Gram is not positive-definite (only happens if the
/// basis is rank-deficient — a bug upstream).
/// MPFR-128 Cholesky of `B·Bᵀ` (Euclidean Gram of an LLL-reduced lattice
/// basis). Returns the upper-triangular factor R (`Rᵀ·R = B·Bᵀ`) as both an
/// f64 snapshot (consumed by the SE walk's primary f64 prune) AND a
/// double-double (~106-bit) projection (consumed by the verify path set via
/// [`set_verify_prune_mpfr`]). The Cholesky itself runs at MPFR-128
/// internally: 106-bit Cholesky was tried and produced rank-deficient
/// false alarms at small lde where the intermediate
/// `s -= l[i][k]*l[j][k]` cancellation is tight; 128-bit is safe. The
/// dd projection of the final factor is probe-confirmed to match
/// MPFR-192 oracle on the cliff failure instance.
pub fn euclidean_cholesky_16_mpfr_dual(
    basis: &[[i64; 16]; 16],
) -> Option<([[f64; 16]; 16], [[(f64, f64); 16]; 16])> {
    use rug::Float;
    const PREC: u32 = 128;
    let mut gram = [[0_i128; 16]; 16];
    for i in 0..16 {
        for j in 0..16 {
            let mut s = 0_i128;
            for k in 0..16 {
                s += (basis[i][k] as i128) * (basis[j][k] as i128);
            }
            gram[i][j] = s;
        }
    }
    let mut g: [[Float; 16]; 16] = std::array::from_fn(|_| {
        std::array::from_fn(|_| Float::with_val(PREC, 0.0))
    });
    for i in 0..16 {
        for j in 0..16 {
            g[i][j] = i128_to_mpfr(gram[i][j], PREC);
        }
    }
    let mut l: [[Float; 16]; 16] = std::array::from_fn(|_| {
        std::array::from_fn(|_| Float::with_val(PREC, 0.0))
    });
    for i in 0..16 {
        for j in 0..=i {
            let mut s = g[i][j].clone();
            for k in 0..j {
                let prod = Float::with_val(PREC, &l[i][k] * &l[j][k]);
                s -= &prod;
            }
            if i == j {
                if s.is_zero() || s.is_sign_negative() {
                    return None;
                }
                l[i][i] = s.sqrt();
            } else {
                let q = Float::with_val(PREC, &s / &l[j][j]);
                l[i][j] = q;
            }
        }
    }
    // R = L^T (upper-triangular). Snapshot to f64 (used by f64 prune) and
    // project to dd (used by verify_partial_dd_exceeds). The MPFR factor
    // itself is consumed here; the dd projection is the kept output.
    let mut r_f64 = [[0.0_f64; 16]; 16];
    let mut r_dd = [[(0.0_f64, 0.0_f64); 16]; 16];
    for i in 0..16 {
        for j in 0..16 {
            let rij = &l[j][i];
            let hi = rij.to_f64();
            let mut lo_f = Float::with_val(PREC, rij);
            lo_f -= hi;
            let lo = lo_f.to_f64();
            r_f64[i][j] = hi;
            r_dd[i][j] = (hi, lo);
        }
    }
    Some((r_f64, r_dd))
}

/// Convert i128 → MPFR Float, lossless. rug doesn't accept i128 directly.
fn i128_to_mpfr(v: i128, prec: u32) -> rug::Float {
    use rug::Float;
    let neg = v < 0;
    let abs = if neg { -v } else { v } as u128;
    let hi = (abs >> 64) as u64;
    let lo = abs as u64;
    let mut f = Float::with_val(prec, hi);
    f <<= 64u32;
    f += Float::with_val(prec, lo);
    if neg { -f } else { f }
}

/// Compute the upper-triangular Cholesky factor R of `B·Bᵀ` (Euclidean Gram
/// of the LLL basis) in f64. Used as a partial-prune lower bound and as a
/// numerical sanity oracle alongside `cholesky_f64_16` (which factors the
/// post-LLL **Q-metric** Gram, not the Euclidean one).
///
/// Returns `None` if the Gram is not numerically positive-definite in f64
/// (extremely rare for an LLL-output basis; would indicate a bug upstream).
///
/// **Overflow note**: For d=16 with post-LLL basis entries up to ~2^15 at
/// moderate ε, gram entries reach ~2^34 (well inside f64). At deep ε with
/// inflated basis (~2^25), gram can hit ~2^54 — at the edge of f64's 53-bit
/// mantissa. We accumulate in i128 first, then convert to f64 for the
/// Cholesky factorization itself.
pub fn euclidean_cholesky_16(basis: &[[i64; 16]; 16]) -> Option<[[f64; 16]; 16]> {
    // Exact integer Gram = B·Bᵀ in i128 to absorb deep-ε basis growth.
    let mut gram = [[0_i128; 16]; 16];
    for i in 0..16 {
        for j in 0..16 {
            let mut s = 0_i128;
            for k in 0..16 {
                s += (basis[i][k] as i128) * (basis[j][k] as i128);
            }
            gram[i][j] = s;
        }
    }
    // f64 Cholesky on the (lower) triangular factor L such that L·Lᵀ = G.
    let mut l = [[0.0_f64; 16]; 16];
    for i in 0..16 {
        for j in 0..=i {
            let mut s = gram[i][j] as f64;
            for k in 0..j {
                s -= l[i][k] * l[j][k];
            }
            if i == j {
                if s <= 0.0 {
                    return None;
                }
                l[i][i] = s.sqrt();
            } else {
                l[i][j] = s / l[j][j];
            }
        }
    }
    // Transpose to upper-triangular R = Lᵀ (caller convention).
    let mut r = [[0.0_f64; 16]; 16];
    for i in 0..16 {
        for j in 0..16 {
            r[i][j] = l[j][i];
        }
    }
    Some(r)
}

// ─── 16D Schnorr-Euchner enumeration ─────────────────────────────────────────

/// Run the Schnorr-Euchner walk over ℤ¹⁶, visiting every integer point `z`
/// with `‖l·(z − z_c)‖² ≤ bound_sq`, in distance-from-center order at each
/// recursion level. Calls `callback(&z)` at every leaf; the callback returns
/// `true` to continue or `false` to abort.
///
/// `l` is the **upper-triangular** Cholesky factor of the post-LLL Q-metric
/// Gram on the basis coordinates: `lᵀ · l = G`. For each level i, the walk
/// computes `level_i = l[i][i] · (z[i] − z_c[i]) + Σ_{j > i} l[i][j] · (z[j]
/// − z_c[j])` and prunes branches whose partial sum-of-squares exceeds
/// `bound_sq`. Visiting closest-to-center first (`z_c[i]` rounded to i64)
/// allows early termination.
///
/// `budget` is decremented once per leaf callback. When it reaches zero the
/// walk aborts and returns the leaf count visited so far.
///
/// Returns the total number of leaf callbacks made.
pub fn schnorr_euchner_16d<F>(
    l: &[[f64; 16]; 16],
    z_c: &[i64; 16],
    bound_sq: f64,
    mut callback: F,
    budget: &AtomicU64,
) -> usize
where
    F: FnMut(&[i64; 16]) -> bool,
{
    let mut z = [0i64; 16];
    let mut leaves: usize = 0;
    let mut aborted = false;
    recurse_16(
        15,
        l,
        z_c,
        bound_sq,
        0.0,
        &mut z,
        &mut callback,
        budget,
        &mut leaves,
        &mut aborted,
    );
    leaves
}

#[allow(clippy::too_many_arguments)]
fn recurse_16<F>(
    depth: i32,
    l: &[[f64; 16]; 16],
    z_c: &[i64; 16],
    bound_sq: f64,
    partial: f64,
    z: &mut [i64; 16],
    callback: &mut F,
    budget: &AtomicU64,
    leaves: &mut usize,
    aborted: &mut bool,
) where
    F: FnMut(&[i64; 16]) -> bool,
{
    if *aborted {
        return;
    }
    if depth < 0 {
        // Leaf: invoke callback and decrement budget.
        *leaves += 1;
        if budget.fetch_sub(1, Ordering::Relaxed) <= 1 {
            *aborted = true;
        }
        if !callback(z) {
            *aborted = true;
        }
        return;
    }
    let d = depth as usize;
    let l_dd = l[d][d];

    // Degenerate diagonal guard (positive-definiteness should exclude this,
    // but tolerate gracefully).
    if l_dd.abs() < 1e-30 {
        z[d] = z_c[d];
        recurse_16(
            depth - 1, l, z_c, bound_sq, partial, z, callback, budget, leaves,
            aborted,
        );
        return;
    }

    // tail = Σ_{j > d} l[d][j] · (z[j] − z_c[j])
    let mut tail = 0.0_f64;
    for j in (d + 1)..16 {
        tail += l[d][j] * ((z[j] - z_c[j]) as f64);
    }

    // Remaining budget for this level.
    let rem = bound_sq - partial;
    if rem < 0.0 {
        return;
    }
    let rem_sqrt = rem.sqrt();

    // The level value at offset Δ from z_c[d] is l_dd · Δ + tail; minimized at
    // Δ = -tail / l_dd. Bound: |l_dd · Δ + tail| ≤ rem_sqrt → Δ ∈
    // [(-tail − rem_sqrt)/l_dd, (-tail + rem_sqrt)/l_dd].
    //
    // **Precision**: at deep ε (1e-8) `z_c[d]` can exceed 2^53 (the f64
    // exact-integer ceiling). Casting `z_c[d] as f64` and adding a small
    // continuous offset would lose 1-2 ULP, mis-bracketing the integer
    // search range. Compute the ranged offsets in f64 then add to `z_c[d]`
    // as i64 — exact whenever |center_off ± span| < 2^53 (always for our
    // bound_sq).
    let center_off = -tail / l_dd;
    let span = rem_sqrt / l_dd.abs();
    let z_low = z_c[d].saturating_add((center_off - span).ceil() as i64);
    let z_high = z_c[d].saturating_add((center_off + span).floor() as i64);
    let z_mid = z_c[d].saturating_add(center_off.round() as i64);
    let max_off = (z_high - z_mid).max(z_mid - z_low).max(0);

    // Walk offsets in distance-from-center order: 0, +1, -1, +2, -2, …
    for raw in 0..=(2 * max_off + 1) {
        if *aborted {
            return;
        }
        let off = if raw == 0 {
            0
        } else if raw % 2 == 1 {
            (raw + 1) / 2
        } else {
            -(raw / 2)
        };
        let zd = z_mid + off;
        if zd < z_low || zd > z_high {
            continue;
        }
        let level = l_dd * ((zd - z_c[d]) as f64) + tail;
        let new_partial = partial + level * level;
        // Slack for f64 round-off at the bound check: 1e-9 * bound_sq matches
        // the 8D "10⁻⁹ tolerance" semantics.
        if new_partial > bound_sq + 1e-9 * bound_sq.abs() {
            continue;
        }
        z[d] = zd;
        recurse_16(
            depth - 1, l, z_c, bound_sq, new_partial, z, callback, budget,
            leaves, aborted,
        );
    }
}

// ─── Norm-shell-pruned Schnorr-Euchner ───────────────────────────────────────

/// SE walk with a SECOND pruning criterion: the Euclidean norm of `x = B·z`
/// must equal `2^k` (the lattice synthesis norm-shell constraint). At every
/// depth we track partial `‖R_eucl · z‖²` (where `R_eucl·R_euclᵀ = B·Bᵀ`,
/// upper-triangular) and prune branches whose partial Euclidean lower bound
/// already exceeds the target norm + slack.
///
/// `r_eucl` is the upper-triangular Euclidean Cholesky factor of the
/// post-LLL basis (compute via [`euclidean_cholesky_16`]). `target_norm_sq`
/// is `2^k` as f64. Pruning is exact in real arithmetic; an additive slack
/// `1e-9 * target_norm_sq` absorbs f64 round-off.
///
/// Mirrors [`schnorr_euchner_16d`]'s interface; same callback semantics.
pub fn schnorr_euchner_16d_norm_pruned<F>(
    l: &[[f64; 16]; 16],
    z_c: &[i64; 16],
    bound_sq: f64,
    r_eucl: &[[f64; 16]; 16],
    target_norm_sq: f64,
    basis: &[[i64; 16]; 16],
    mut callback: F,
    budget: &AtomicU64,
) -> usize
where
    F: FnMut(&[i64; 16]) -> bool,
{
    let mut z = [0i64; 16];
    let mut x = [0i64; 16];
    // **Incremental orthogonalized projection** w = R_eucl · z. Maintained
    // throughout the SE walk by delta updates when z[d] changes:
    //
    //   w[i] += delta · R[i][d]   for i ≤ d  (R is upper-triangular)
    //
    // Replaces the per-call `tail_eucl = Σ R[d][j]·(z[j] as f64)` which
    // suffered catastrophic cancellation at deep ε (z[j] > 2^53). The
    // incremental delta is bounded by the SE bracket span (~few lattice
    // units), so `delta · R[i][d]` stays in f64-precise range. Drift over
    // many iterations is bounded by ULP per update × tree depth, well
    // below the 1e-9 prune slack.
    //
    // Crucial invariant: w[d] depends only on z[d..15] (upper-tri R).
    // So recursion to lower depths cannot corrupt w[d]; no save/restore
    // needed across recursion.
    let mut w = [0f64; 16];
    let mut leaves: usize = 0;
    let mut aborted = false;
    // Convert target_norm_sq (f64) to i64 for exact pruning. f64 represents
    // 2^k exactly for k ≤ 1023 so no precision loss.
    let target_norm_sq_i64 = target_norm_sq as i64;
    recurse_16_norm_pruned(
        15, l, z_c, bound_sq, r_eucl, target_norm_sq, target_norm_sq_i64,
        0.0, 0.0, &mut z, &mut x, &mut w, basis, &mut callback, budget,
        &mut leaves, &mut aborted,
    );
    leaves
}

#[allow(clippy::too_many_arguments)]
fn recurse_16_norm_pruned<F>(
    depth: i32,
    l: &[[f64; 16]; 16],
    z_c: &[i64; 16],
    bound_sq: f64,
    r_eucl: &[[f64; 16]; 16],
    target_norm_sq: f64,
    target_norm_sq_i64: i64,
    partial_q: f64,
    partial_eucl: f64,
    z: &mut [i64; 16],
    x: &mut [i64; 16],
    w: &mut [f64; 16],
    basis: &[[i64; 16]; 16],
    callback: &mut F,
    budget: &AtomicU64,
    leaves: &mut usize,
    aborted: &mut bool,
) where
    F: FnMut(&[i64; 16]) -> bool,
{
    if *aborted {
        return;
    }
    if depth < 0 {
        // x is maintained incrementally — pass it directly to the callback.
        *leaves += 1;
        if budget.fetch_sub(1, Ordering::Relaxed) <= 1 {
            *aborted = true;
        }
        if !callback(x) {
            *aborted = true;
        }
        return;
    }
    let d = depth as usize;
    let l_dd = l[d][d];

    if l_dd.abs() < 1e-30 {
        let new_zd = z_c[d];
        let delta = new_zd - z[d];
        if delta != 0 {
            update_x_for_z_change(x, basis, d, delta);
            // Update w incrementally: w[i] += delta · R[i][d] for i ≤ d.
            let delta_f = delta as f64;
            for i in 0..=d {
                w[i] += delta_f * r_eucl[i][d];
            }
        }
        z[d] = new_zd;
        let level_eucl = w[d];
        let new_partial_eucl = partial_eucl + level_eucl * level_eucl;
        if bypass_norm_prune() || new_partial_eucl <= target_norm_sq * (1.0 + 1e-9) {
            recurse_16_norm_pruned(
                depth - 1, l, z_c, bound_sq, r_eucl, target_norm_sq, target_norm_sq_i64,
                partial_q, new_partial_eucl, z, x, w, basis, callback, budget,
                leaves, aborted,
            );
        }
        return;
    }

    // Q-bound: tail and span as in recurse_16.
    let mut tail = 0.0_f64;
    for j in (d + 1)..16 {
        tail += l[d][j] * ((z[j] - z_c[j]) as f64);
    }
    let rem = bound_sq - partial_q;
    if rem < 0.0 {
        return;
    }
    let rem_sqrt = rem.sqrt();
    let center_off = -tail / l_dd;
    let span = rem_sqrt / l_dd.abs();
    // See recurse_16 above for the deep-ε precision rationale.
    let z_low = z_c[d].saturating_add((center_off - span).ceil() as i64);
    let z_high = z_c[d].saturating_add((center_off + span).floor() as i64);
    let z_mid = z_c[d].saturating_add(center_off.round() as i64);
    let max_off = (z_high - z_mid).max(z_mid - z_low).max(0);

    for raw in 0..=(2 * max_off + 1) {
        if *aborted {
            return;
        }
        let off = if raw == 0 {
            0
        } else if raw % 2 == 1 {
            (raw + 1) / 2
        } else {
            -(raw / 2)
        };
        let zd = z_mid + off;
        if zd < z_low || zd > z_high {
            continue;
        }
        let level = l_dd * ((zd - z_c[d]) as f64) + tail;
        let new_partial_q = partial_q + level * level;
        if new_partial_q > bound_sq + 1e-9 * bound_sq.abs() {
            continue;
        }
        // Update z[d], x = B·z, and w = R·z incrementally. delta is small
        // (within the SE bracket span), so the f64 update of w is precise.
        let delta = zd - z[d];
        if delta != 0 {
            update_x_for_z_change(x, basis, d, delta);
            let delta_f = delta as f64;
            for i in 0..=d {
                w[i] += delta_f * r_eucl[i][d];
            }
        }
        z[d] = zd;

        // Norm-shell pruning: orthogonalized partial Σ_{j ≥ d} w[j]² is
        // monotone-increasing as depth decreases (each w[j]² added). Use
        // w[d] (incrementally maintained, no cancellation) instead of
        // recomputing from scratch each call.
        let level_eucl = w[d];
        let new_partial_eucl = partial_eucl + level_eucl * level_eucl;
        let threshold = target_norm_sq * (1.0 + 1e-9);
        let prune_fires = depth > 0 && new_partial_eucl > threshold;
        let bypass = bypass_norm_prune();
        if prune_fires && !bypass {
            crate::synthesis::diag::N_PRUNE_FIRES.fetch_add(1, Ordering::Relaxed);
            if (depth as usize) < 16 {
                crate::synthesis::diag::N_PRUNE_FIRES_AT_DEPTH[depth as usize]
                    .fetch_add(1, Ordering::Relaxed);
            }
            let ratio = new_partial_eucl / threshold;
            if ratio <= 1.10 {
                crate::synthesis::diag::N_PRUNE_FIRES_NEAR.fetch_add(1, Ordering::Relaxed);
            }
            if ratio <= 1.01 {
                crate::synthesis::diag::N_PRUNE_FIRES_VERY_NEAR.fetch_add(1, Ordering::Relaxed);
            }
            crate::synthesis::diag::sample_prune_event(depth, z, new_partial_eucl, threshold);
            if crate::synthesis::diag::watch_path_match_at_depth(z, depth) {
                crate::synthesis::diag::watch_record(crate::synthesis::diag::WatchHit {
                    depth, z_at_prune: *z,
                    partial_eucl_f64: new_partial_eucl,
                    threshold,
                    partial_q_f64: new_partial_q,
                    r_eucl_diag_d: r_eucl[d as usize][d as usize],
                    w_d: w[d as usize],
                });
            }
        }
        if !bypass && prune_fires {
            continue;
        }
        recurse_16_norm_pruned(
            depth - 1, l, z_c, bound_sq, r_eucl, target_norm_sq, target_norm_sq_i64,
            new_partial_q, new_partial_eucl, z, x, w, basis, callback, budget,
            leaves, aborted,
        );
    }
}

struct Bracket {
    tail: f64,
    z_low: i64,
    z_high: i64,
    z_mid: i64,
    max_off: i64,
}

/// Compute the SE bracket [z_low, z_high] for the current depth's z[d]
/// enumeration. Returns None if the Q-bound is already exhausted.
#[inline]
fn compute_bracket_at_depth(
    d: usize,
    l: &[[f64; 16]; 16],
    z_c: &[i64; 16],
    z: &[i64; 16],
    partial_q: f64,
    bound_sq: f64,
    l_dd: f64,
) -> Option<Bracket> {
    let mut tail = 0.0_f64;
    for j in (d + 1)..16 {
        tail += l[d][j] * ((z[j] - z_c[j]) as f64);
    }
    let rem = bound_sq - partial_q;
    if rem < 0.0 {
        return None;
    }
    let rem_sqrt = rem.sqrt();
    let center_off = -tail / l_dd;
    let span = rem_sqrt / l_dd.abs();
    // Same i64 bracket fix as recurse_16: keep z_c[d] as i64 to avoid
    // f64 quantization at deep ε where z_c can exceed 2^53.
    let z_low = z_c[d].saturating_add((center_off - span).ceil() as i64);
    let z_high = z_c[d].saturating_add((center_off + span).floor() as i64);
    let z_mid = z_c[d].saturating_add(center_off.round() as i64);
    let max_off = (z_high - z_mid).max(z_mid - z_low).max(0);
    Some(Bracket { tail, z_low, z_high, z_mid, max_off })
}

/// Apply `x[c] += delta · basis[d][c]` for c=0..16. Fast tight loop —
/// LLVM auto-vectorizes this on Apple Silicon (4 i64s per NEON op).
#[inline]
fn update_x_for_z_change(
    x: &mut [i64; 16],
    basis: &[[i64; 16]; 16],
    d: usize,
    delta: i64,
) {
    let row = &basis[d];
    for c in 0..16 {
        x[c] += delta * row[c];
    }
}

// ─── Parallel Schnorr-Euchner ────────────────────────────────────────────────

/// Parallel SE walker. Partitions the outermost coordinate `z[15]` into
/// independent subtrees and walks them in parallel via rayon. Each subtree
/// is sequential; cross-subtree state is shared via atomics:
///   - `budget`: leaf-callback budget (decremented atomically).
///   - `aborted`: set when budget is exhausted; checked at every level.
///
/// The leaf filter must be `Fn + Sync` (no per-leaf mutable state). Returns
/// `(solutions, budget_hit)` where `solutions` are the leaves passing the
/// filter and `budget_hit` is true iff the walk terminated early.
///
/// Mirror of [`schnorr_euchner_16d`] in interface, but bypasses the
/// `FnMut` callback in favour of a pure filter that returns whether each
/// leaf should be collected. Used by [`super::phase1`].
pub fn schnorr_euchner_16d_par<F>(
    l: &[[f64; 16]; 16],
    z_c: &[i64; 16],
    bound_sq: f64,
    leaf_filter: F,
    budget: &AtomicU64,
) -> (Vec<[i64; 16]>, bool)
where
    F: Fn(&[i64; 16]) -> bool + Sync,
{
    use rayon::prelude::*;
    use std::sync::atomic::AtomicBool;

    let aborted = AtomicBool::new(false);
    let l_15 = l[15][15];
    if l_15.abs() < 1e-30 {
        return (Vec::new(), false);
    }
    let span = bound_sq.sqrt() / l_15.abs();
    let z_low = ((z_c[15] as f64) - span).ceil() as i64;
    let z_high = ((z_c[15] as f64) + span).floor() as i64;
    let z_mid = z_c[15];

    // Schnorr-Euchner ordering at the outermost level: closest-to-center
    // first. Doesn't change correctness (same SET visited) but lets early
    // budget-exhaust prefer the most promising subtrees.
    let mut prefixes: Vec<i64> = (z_low..=z_high).collect();
    prefixes.sort_by_key(|&z| (z - z_mid).abs());

    let solutions: Vec<[i64; 16]> = prefixes
        .into_par_iter()
        .flat_map_iter(|z_15| {
            if aborted.load(Ordering::Relaxed) {
                return Vec::new().into_iter();
            }
            // Contribution of z[15] to the partial accumulator.
            let level = l_15 * ((z_15 - z_c[15]) as f64);
            let partial = level * level;
            if partial > bound_sq + 1e-9 * bound_sq.abs() {
                return Vec::new().into_iter();
            }
            let mut z = [0i64; 16];
            z[15] = z_15;
            let mut local: Vec<[i64; 16]> = Vec::new();
            recurse_collect(
                14,
                l,
                z_c,
                bound_sq,
                partial,
                &mut z,
                &leaf_filter,
                budget,
                &aborted,
                &mut local,
            );
            local.into_iter()
        })
        .collect();

    let budget_hit = aborted.load(Ordering::Relaxed);
    (solutions, budget_hit)
}

#[allow(clippy::too_many_arguments)]
fn recurse_collect<F>(
    depth: i32,
    l: &[[f64; 16]; 16],
    z_c: &[i64; 16],
    bound_sq: f64,
    partial: f64,
    z: &mut [i64; 16],
    leaf_filter: &F,
    budget: &AtomicU64,
    aborted: &std::sync::atomic::AtomicBool,
    results: &mut Vec<[i64; 16]>,
) where
    F: Fn(&[i64; 16]) -> bool,
{
    if aborted.load(Ordering::Relaxed) {
        return;
    }
    if depth < 0 {
        // Leaf: decrement budget and apply filter.
        let prev = budget.fetch_sub(1, Ordering::Relaxed);
        if prev <= 1 {
            aborted.store(true, Ordering::Relaxed);
        }
        if leaf_filter(z) {
            results.push(*z);
        }
        return;
    }
    let d = depth as usize;
    let l_dd = l[d][d];
    if l_dd.abs() < 1e-30 {
        z[d] = z_c[d];
        recurse_collect(
            depth - 1, l, z_c, bound_sq, partial, z, leaf_filter, budget,
            aborted, results,
        );
        return;
    }
    let mut tail = 0.0_f64;
    for j in (d + 1)..16 {
        tail += l[d][j] * ((z[j] - z_c[j]) as f64);
    }
    let rem = bound_sq - partial;
    if rem < 0.0 {
        return;
    }
    let rem_sqrt = rem.sqrt();
    let center_off = -tail / l_dd;
    let span = rem_sqrt / l_dd.abs();
    let z_low = ((z_c[d] as f64) + center_off - span).ceil() as i64;
    let z_high = ((z_c[d] as f64) + center_off + span).floor() as i64;
    let z_mid = ((z_c[d] as f64) + center_off).round() as i64;
    let max_off = (z_high - z_mid).max(z_mid - z_low).max(0);
    for raw in 0..=(2 * max_off + 1) {
        if aborted.load(Ordering::Relaxed) {
            return;
        }
        let off = if raw == 0 {
            0
        } else if raw % 2 == 1 {
            (raw + 1) / 2
        } else {
            -(raw / 2)
        };
        let zd = z_mid + off;
        if zd < z_low || zd > z_high {
            continue;
        }
        let level = l_dd * ((zd - z_c[d]) as f64) + tail;
        let new_partial = partial + level * level;
        if new_partial > bound_sq + 1e-9 * bound_sq.abs() {
            continue;
        }
        z[d] = zd;
        recurse_collect(
            depth - 1, l, z_c, bound_sq, new_partial, z, leaf_filter, budget,
            aborted, results,
        );
    }
}

// ─── Parallel norm-pruned Schnorr-Euchner ────────────────────────────────────

/// What the SE walker should do with a leaf candidate.
#[derive(Clone, Copy, Debug)]
pub enum LeafAction {
    /// Discard this leaf, keep walking.
    Skip,
    /// Collect this leaf, keep walking.
    Take,
    /// Collect this leaf, then abort the walk.
    TakeAndStop,
}

/// Parallel + norm-shell-pruned + incremental-x SE walker. This is the
/// production workhorse used by [`super::phase1`].
///
/// Combines all three accelerations:
///   1. **Norm-shell pruning** via the upper-triangular Euclidean Cholesky
///      `r_eucl` (`R·Rᵀ = B·Bᵀ`). Branches whose partial `‖R·z‖²` exceeds
///      `target_norm_sq + slack` get cut early.
///   2. **Per-z[15] parallelism** via rayon. Each outermost-coordinate
///      subtree runs in its own thread with its own (z, x) state.
///   3. **Incremental `x = B·z`** maintenance. When z[d] changes by δ the
///      x buffer is updated by δ·basis[d] (16 ops vs 256 for full
///      reconstruct).
///
/// `leaf_filter` receives the reconstructed `x` (NOT z) and must be
/// `Fn + Sync`. Returns a [`LeafAction`]: `Take`/`Skip`/`TakeAndStop`.
///
/// Returns `(solutions, budget_hit)`.
pub fn schnorr_euchner_16d_par_norm_pruned<F>(
    l: &[[f64; 16]; 16],
    z_c: &[i64; 16],
    bound_sq: f64,
    r_eucl: &[[f64; 16]; 16],
    r_eucl_dd: &[[(f64, f64); 16]; 16],
    target_norm_sq: f64,
    basis: &[[i64; 16]; 16],
    leaf_filter: F,
    budget: &AtomicU64,
) -> (Vec<[i64; 16]>, bool)
where
    F: Fn(&[i64; 16]) -> LeafAction + Sync,
{
    use rayon::prelude::*;
    use std::sync::atomic::AtomicBool;

    let aborted = AtomicBool::new(false);
    let l_15 = l[15][15];
    let target_norm_sq_i64 = target_norm_sq as i64;
    if l_15.abs() < 1e-30 {
        return (Vec::new(), false);
    }

    // z[15] range from the Q-bound. Keep z_c[15] as i64 to avoid the
    // deep-ε f64 quantization issue (same fix as recurse_16).
    let span_q = bound_sq.sqrt() / l_15.abs();
    let z_low = z_c[15].saturating_add((-span_q).ceil() as i64);
    let z_high = z_c[15].saturating_add(span_q.floor() as i64);
    let z_mid = z_c[15];

    // Closest-to-center first ordering at the outermost level. Tried
    // sharding at (z[15], z[14]) for finer-grained parallelism, but
    // rayon-scheduling overhead and a sort key that doesn't track the
    // true SE "closest-to-center" (z[14]'s center depends on z[15] via
    // tail) made it 2-4× slower at deep ε. Single-level z[15] sharding
    // wins: each worker walks one z[15] subtree in its native SE order.
    let mut prefixes: Vec<i64> = (z_low..=z_high).collect();
    prefixes.sort_by_key(|&z| (z - z_mid).abs());

    let solutions: Vec<[i64; 16]> = prefixes
        .into_par_iter()
        .flat_map_iter(|z_15| {
            if aborted.load(Ordering::Relaxed) {
                return Vec::new().into_iter();
            }
            // Q-bound contribution at depth 15.
            let level_q = l_15 * ((z_15 - z_c[15]) as f64);
            let partial_q = level_q * level_q;
            if partial_q > bound_sq + 1e-9 * bound_sq.abs() {
                return Vec::new().into_iter();
            }
            // Per-thread state.
            let mut z = [0i64; 16];
            z[15] = z_15;
            let mut x = [0i64; 16];
            if z_15 != 0 {
                let row = &basis[15];
                for c in 0..16 {
                    x[c] = z_15 * row[c];
                }
            }
            // Incremental w = R_eucl · z. Initialize from z[15] only:
            //   w[i] = z_15 · R[i][15]   for i ≤ 15
            // (z[0..15] are zero on entry).
            let mut w = [0f64; 16];
            let z_15_f = z_15 as f64;
            for i in 0..=15 {
                w[i] = z_15_f * r_eucl[i][15];
            }
            let level_eucl = w[15];
            let partial_eucl = level_eucl * level_eucl;
            if !bypass_norm_prune() && partial_eucl > target_norm_sq * (1.0 + 1e-9) {
                return Vec::new().into_iter();
            }
            let mut local: Vec<[i64; 16]> = Vec::new();
            recurse_collect_norm_pruned(
                14, l, z_c, bound_sq, r_eucl, r_eucl_dd, target_norm_sq, target_norm_sq_i64,
                partial_q, partial_eucl, &mut z, &mut x, &mut w, basis,
                &leaf_filter, budget, &aborted, &mut local,
            );
            local.into_iter()
        })
        .collect();

    let budget_hit = aborted.load(Ordering::Relaxed);
    (solutions, budget_hit)
}

#[allow(clippy::too_many_arguments)]
fn recurse_collect_norm_pruned<F>(
    depth: i32,
    l: &[[f64; 16]; 16],
    z_c: &[i64; 16],
    bound_sq: f64,
    r_eucl: &[[f64; 16]; 16],
    r_eucl_dd: &[[(f64, f64); 16]; 16],
    target_norm_sq: f64,
    target_norm_sq_i64: i64,
    partial_q: f64,
    partial_eucl: f64,
    z: &mut [i64; 16],
    x: &mut [i64; 16],
    w: &mut [f64; 16],
    basis: &[[i64; 16]; 16],
    leaf_filter: &F,
    budget: &AtomicU64,
    aborted: &std::sync::atomic::AtomicBool,
    results: &mut Vec<[i64; 16]>,
) where
    F: Fn(&[i64; 16]) -> LeafAction,
{
    if aborted.load(Ordering::Relaxed) {
        return;
    }
    // Per-node budget (phase 1): decrement on every recurse-enter so the
    // budget bounds total tree-traversal work, not just leaf checks. This
    // is the prerequisite for depth-1 / depth-0 analytical filters whose
    // gain is "skip subtrees" — under a per-leaf budget those filters
    // regressed because cheaper leaves let the walker enter more depth-0
    // nodes within the same budget (full recursion-from-depth-15 each
    // time). Bounding nodes makes the budget proportional to traversal
    // cost. PASS{1,2}_CAP are calibrated empirically (see
    // clifford_sqrt_t.rs); the new units are nodes, not leaves.
    if budget.fetch_sub(1, Ordering::Relaxed) <= 1 {
        aborted.store(true, Ordering::Relaxed);
        return;
    }
    let trace = crate::synthesis::diag::trace_enabled();
    if trace && depth >= 0 && (depth as usize) < 16 {
        crate::synthesis::diag::N_RECURSE_ENTER_AT_DEPTH[depth as usize]
            .fetch_add(1, Ordering::Relaxed);
    }
    // Capture partial_eucl at depth-0 entry — this is the outgoing depth-1
    // partial. Read at leaf time to condition the shell-ratio histogram.
    if trace && depth == 0 {
        crate::synthesis::diag::D1_PARTIAL_TLS.with(|c| c.set(partial_eucl));
    }
    // Depth-1 shell-discriminant filter (phase 3) — see `qfilter_enabled`
    // for status. Default off; opt-in via `CYCLOSYNTH_QFILTER=1`.
    let qfilter_state: Option<(i256, i256, i256, i256, i256, i256)> =
        if depth == 1 && qfilter_enabled() {
            Some(qfilter_depth1_state(basis, x, z[0], z[1]))
        } else {
            None
        };
    if depth < 0 {
        // Shell-ratio histogram: record where x lands relative to the
        // target shell, regardless of leaf_filter outcome. Reveals whether
        // the SE walk is delivering near-shell or far-interior leaves.
        if trace {
            let n: i64 = x.iter().map(|&v| v * v).sum();
            crate::synthesis::diag::record_leaf_shell_ratio(n, target_norm_sq_i64);
        }
        match leaf_filter(x) {
            LeafAction::Skip => {}
            LeafAction::Take => results.push(*x),
            LeafAction::TakeAndStop => {
                results.push(*x);
                aborted.store(true, Ordering::Relaxed);
            }
        }
        return;
    }
    let d = depth as usize;
    let l_dd = l[d][d];
    if l_dd.abs() < 1e-30 {
        let new_zd = z_c[d];
        let delta = new_zd - z[d];
        if delta != 0 {
            update_x_for_z_change(x, basis, d, delta);
            let delta_f = delta as f64;
            for i in 0..=d {
                w[i] += delta_f * r_eucl[i][d];
            }
        }
        z[d] = new_zd;
        let level_eucl = w[d];
        let new_partial_eucl = partial_eucl + level_eucl * level_eucl;
        if bypass_norm_prune() || new_partial_eucl <= target_norm_sq * (1.0 + 1e-9) {
            recurse_collect_norm_pruned(
                depth - 1, l, z_c, bound_sq, r_eucl, r_eucl_dd, target_norm_sq, target_norm_sq_i64,
                partial_q, new_partial_eucl, z, x, w, basis, leaf_filter,
                budget, aborted, results,
            );
        }
        return;
    }
    let bracket = match compute_bracket_at_depth(d, l, z_c, z, partial_q, bound_sq, l_dd) {
        Some(b) => b,
        None => return,
    };
    let tail = bracket.tail;
    let z_low = bracket.z_low;
    let z_high = bracket.z_high;
    let z_mid = bracket.z_mid;
    let max_off = bracket.max_off;

    // NOTE: a depth-0 analytical shell-equation elimination is available via
    // [`analytical_depth0_z0_candidates`] (i128-exact integer roots of
    // `‖x‖² = 2^k`). Plugging it into the SE walk here was tried multiple
    // times and consistently regresses cliff wall-time (up to 5×) under
    // the current per-leaf budget: fewer leaves per depth-0 enter just
    // makes the walker exhaust budget after more depth-0 enters, each
    // costing full recursion from depth 15 down. Unlocking it requires
    // switching to a per-recurse-enter (or per-depth-0-enter) budget +
    // recalibrating `dc_pass1_cap_for(eps)` / `PASS{1,2}_CAP` constants
    // across the supported ε range. The helper is preserved for that
    // future refactor.

    for raw in 0..=(2 * max_off + 1) {
        if aborted.load(Ordering::Relaxed) {
            return;
        }
        let off = if raw == 0 {
            0
        } else if raw % 2 == 1 {
            (raw + 1) / 2
        } else {
            -(raw / 2)
        };
        let zd = z_mid + off;
        if zd < z_low || zd > z_high {
            continue;
        }
        let level = l_dd * ((zd - z_c[d]) as f64) + tail;
        let new_partial_q = partial_q + level * level;
        if new_partial_q > bound_sq + 1e-9 * bound_sq.abs() {
            continue;
        }
        // Update z[d], x = B·z, and w = R·z incrementally.
        let delta = zd - z[d];
        if delta != 0 {
            update_x_for_z_change(x, basis, d, delta);
            let delta_f = delta as f64;
            for i in 0..=d {
                w[i] += delta_f * r_eucl[i][d];
            }
        }
        z[d] = zd;

        // Norm-shell pruning using incremental w (no f64 cancellation).
        let level_eucl = w[d];
        let new_partial_eucl = partial_eucl + level_eucl * level_eucl;
        let threshold = target_norm_sq * (1.0 + 1e-9);
        let prune_fires = depth > 0 && new_partial_eucl > threshold;
        let bypass = bypass_norm_prune();
        if trace && prune_fires && !bypass {
            crate::synthesis::diag::N_PRUNE_FIRES.fetch_add(1, Ordering::Relaxed);
            if (depth as usize) < 16 {
                crate::synthesis::diag::N_PRUNE_FIRES_AT_DEPTH[depth as usize]
                    .fetch_add(1, Ordering::Relaxed);
            }
            let ratio = new_partial_eucl / threshold;
            if ratio <= 1.10 {
                crate::synthesis::diag::N_PRUNE_FIRES_NEAR.fetch_add(1, Ordering::Relaxed);
            }
            if ratio <= 1.01 {
                crate::synthesis::diag::N_PRUNE_FIRES_VERY_NEAR.fetch_add(1, Ordering::Relaxed);
            }
            crate::synthesis::diag::sample_prune_event(depth, z, new_partial_eucl, threshold);
            if crate::synthesis::diag::watch_path_match_at_depth(z, depth) {
                crate::synthesis::diag::watch_record(crate::synthesis::diag::WatchHit {
                    depth, z_at_prune: *z,
                    partial_eucl_f64: new_partial_eucl,
                    threshold,
                    partial_q_f64: new_partial_q,
                    r_eucl_diag_d: r_eucl[d][d],
                    w_d: w[d],
                });
            }
        }
        // Extended-precision verification of the prune-fire decision via
        // inline double-double (~106 bits). Necessary at ε ≤ 1.5e-8 where
        // the f64 partial-eucl accumulator suffers catastrophic cancellation
        // in the dot product. Guard: only verify when ratio ≤ VERIFY_RATIO_CAP.
        // Empirically 0 FNs in 1000 samples at ratio ≥ 5×T.
        //
        // Integer-exact fast-path: at depth d with z[0..d]=0, the relation
        // ‖x‖² = z^T G z = prefix_d + partial_eucl_d (prefix_d ≥ 0) means
        //   ‖x‖² ≤ T_int  ⟹  partial_eucl_d ≤ T_int  ⟹  do not prune.
        // This is cheap (16 i64 muls; ~30 ns) and catches the FN subset
        // where integer-exact norm proves the prune wrong, BEFORE running
        // dd verify (~450 ns). Net win iff a non-negligible fraction of
        // prune-fires have ‖x‖² ≤ T_int.
        const VERIFY_RATIO_CAP: f64 = 5.0;
        let actually_prune = if !bypass && prune_fires {
            // Integer-exact short-circuit (no false negatives, may miss some
            // true keeps where prefix_d > ‖x‖² − T).
            let x_norm_sq: i64 = x.iter().map(|&v| v.wrapping_mul(v)).sum();
            if x_norm_sq <= target_norm_sq_i64 {
                false  // confirmed keep, skip dd verify
            } else if verify_prune_mpfr() && new_partial_eucl <= threshold * VERIFY_RATIO_CAP {
                let t_v = if trace { Some(std::time::Instant::now()) } else { None };
                let dd_prune = verify_partial_dd_exceeds(r_eucl_dd, z, depth as usize, threshold);
                if let Some(t) = t_v {
                    crate::synthesis::diag::T_VERIFY_DD_NS
                        .fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    if !dd_prune {
                        crate::synthesis::diag::N_VERIFY_PRUNE_CORRECTED.fetch_add(1, Ordering::Relaxed);
                    }
                    crate::synthesis::diag::N_VERIFY_PRUNE_FIRES.fetch_add(1, Ordering::Relaxed);
                }
                dd_prune
            } else {
                true
            }
        } else {
            false
        };
        if actually_prune {
            if trace && (depth as usize) < 16 {
                crate::synthesis::diag::N_PRUNE_ACTUAL_AT_DEPTH[depth as usize]
                    .fetch_add(1, Ordering::Relaxed);
            }
            continue;
        }
        // NOTE: a depth-0 integer-exact early-out (`if d==0 && Σx[i]² ≠ 2^k:
        // continue;`) was tried here and regressed cliff wall-time **20×**
        // (41.6s → 840.7s). Same root cause as the analytical depth-0
        // candidate filter: under per-leaf budget the walk just consumes
        // budget through more depth-0 enters when individual leaves are
        // cheaper, multiplying tree-traversal cost. The integer check is
        // already inside `leaf_filter`'s first stage; replicating it here
        // is pure overhead without budget-model changes.

        // Depth-1 Q-filter: at z[1] = zd, decide if any integer z[0] makes
        // ‖x‖² = T exactly. Skip recursion when no perfect-square solution
        // exists. Sound: leaf_filter requires ‖x‖² == T strictly, so a
        // non-perfect-square discriminant guarantees no leaf survives.
        if let Some((g_00, g_01, g_11, a_q, v_0, v_1)) = qfilter_state {
            let class = qfilter_discriminant_class(
                g_00, g_01, g_11, a_q, v_0, v_1, target_norm_sq_i64, zd,
            );
            if trace {
                crate::synthesis::diag::N_QFILTER_TOTAL.fetch_add(1, Ordering::Relaxed);
                match class {
                    0 => crate::synthesis::diag::N_QFILTER_D_NEG.fetch_add(1, Ordering::Relaxed),
                    1 => crate::synthesis::diag::N_QFILTER_D_GE0_MOD16_BAD
                        .fetch_add(1, Ordering::Relaxed),
                    2 => crate::synthesis::diag::N_QFILTER_D_GE0_NOT_SQUARE
                        .fetch_add(1, Ordering::Relaxed),
                    _ => crate::synthesis::diag::N_QFILTER_PERFECT_SQUARE
                        .fetch_add(1, Ordering::Relaxed),
                };
            }
            if class != 3 {
                continue;
            }
        }

        recurse_collect_norm_pruned(
            depth - 1, l, z_c, bound_sq, r_eucl, r_eucl_dd, target_norm_sq, target_norm_sq_i64,
            new_partial_q, new_partial_eucl, z, x, w, basis, leaf_filter,
            budget, aborted, results,
        );
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod par_tests {
    use super::*;
    use std::collections::HashSet;

    /// Round-trip check: ‖R·z‖² (sum of squared rows R[d] · z, accumulated
    /// top-down as in the SE pruning) must equal ‖B·z‖² exactly (modulo
    /// f64 round-off). If this fails, the Euclidean Cholesky / pruning
    /// math is wrong.
    #[test]
    fn euclidean_cholesky_partial_matches_xnorm() {
        // Random-looking integer basis (well-conditioned).
        let mut b = [[0_i64; 16]; 16];
        for i in 0..16 {
            for j in 0..16 {
                b[i][j] = (((i as i64) * 7 + (j as i64) * 13 + 11) % 19) - 9;
            }
            b[i][i] += 7; // boost diagonal for PSD
        }
        let r = euclidean_cholesky_16(&b).expect("PSD");

        // Pick a z, compute ‖B·z‖² directly.
        let z: [i64; 16] = [1, -2, 3, 0, -1, 2, 1, -3, 4, 0, -1, 2, 1, -2, 3, -1];
        let x = reconstruct_x(&b, &z);
        let xnorm_sq: i128 = x.iter().map(|&v| (v as i128) * (v as i128)).sum();
        let xnorm_sq_f = xnorm_sq as f64;

        // Compute ‖R·z‖² as my pruning would: top-down summed squared
        // (R-row d · z) for d = 15, 14, ..., 0.
        let mut partial_eucl = 0.0_f64;
        for d in (0..16).rev() {
            let mut level = r[d][d] * (z[d] as f64);
            for j in (d + 1)..16 {
                level += r[d][j] * (z[j] as f64);
            }
            partial_eucl += level * level;
        }
        let rel_err = (partial_eucl - xnorm_sq_f).abs() / xnorm_sq_f.max(1.0);
        assert!(rel_err < 1e-10,
            "‖R·z‖² ({}) != ‖B·z‖² ({}); rel_err {:.3e}",
            partial_eucl, xnorm_sq_f, rel_err);
    }

    /// Sanity: parallel and serial SE walks on the same setup produce the
    /// same SET of leaves, when the budget is large enough that neither
    /// aborts. If this fails, there's a real bug in the parallel walker.
    #[test]
    fn parallel_and_serial_produce_same_set() {
        // Identity Cholesky factor + non-trivial z_c + small bound_sq for
        // a manageable region.
        let mut l = [[0.0_f64; 16]; 16];
        for i in 0..16 {
            l[i][i] = 1.0;
        }
        let z_c = [0_i64; 16];
        let bound_sq = 4.0;

        // Serial walk: collect all leaves.
        let serial_budget = AtomicU64::new(1_000_000);
        let mut serial_set: HashSet<[i64; 16]> = HashSet::new();
        schnorr_euchner_16d(
            &l, &z_c, bound_sq,
            |z| { serial_set.insert(*z); true },
            &serial_budget,
        );

        // Parallel walk: collect all leaves passing a tautological filter.
        let par_budget = AtomicU64::new(1_000_000);
        let (par_zs, _hit) = schnorr_euchner_16d_par(
            &l, &z_c, bound_sq, |_z| true, &par_budget,
        );
        let par_set: HashSet<[i64; 16]> = par_zs.into_iter().collect();

        assert_eq!(serial_set, par_set,
            "serial leaves ({}) != parallel leaves ({})",
            serial_set.len(), par_set.len());
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::cholesky_lu::{cholesky_f64_16, lu_solve_int_inplace_16};
    use super::super::lll::run_lll_16;
    use super::super::q_metric::{build_q_int_zeta, build_q_mpfr_zeta};
    use super::super::scratch::IntScratch16;
    use crate::synthesis::search_zeta::phase1_brute;
    use std::collections::HashSet;

    fn realistic_v() -> [f64; 4] {
        let v = [0.5, 0.3, 0.7, -0.4];
        let n: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        std::array::from_fn(|i| v[i] / n)
    }

    /// Verify the LLL basis is unimodular (det=±1, so spans full ℤ¹⁶ lattice)
    /// and that each basis row, taken as a candidate `x = b_i`, has norm²
    /// matching the LLL "shortest in this direction" expectation. This is a
    /// structural test that doesn't require enumerating the lattice.
    #[test]
    fn lll_basis_first_row_is_short_lattice_vector() {
        let v = realistic_v();
        let mut s = IntScratch16::new(1e-3);
        build_q_mpfr_zeta(&mut s, v, 6, 1e-3);
        build_q_int_zeta(&mut s);
        let r = run_lll_16(&mut s);
        assert!(matches!(r, super::super::lll::LllResult::Converged));
        let nz_count = s.basis.iter().filter(|row| row.iter().any(|&v| v != 0)).count();
        assert_eq!(nz_count, 16, "basis should have 16 non-zero rows");
    }

    /// Verify a brute-force solution is in the lattice spanned by the LLL
    /// basis. Since the LLL basis is unimodular (det = ±1), every integer
    /// 16-vector is expressible as `B·z` for some integer z; the question is
    /// whether z is small. For *Euclidean*-short brute solutions, the
    /// LLL basis (reduced under the *Q-metric*) may yield large z because
    /// the Q-metric and Euclidean metric differ vastly in the cap-radial
    /// direction.
    #[test]
    fn lll_basis_inverse_recovers_integer_z_for_brute_solution() {
        let v = realistic_v();
        let mut s = IntScratch16::new(1e-3);
        build_q_mpfr_zeta(&mut s, v, 6, 1e-3);
        build_q_int_zeta(&mut s);
        let r = run_lll_16(&mut s);
        assert!(matches!(r, super::super::lll::LllResult::Converged));

        let brute = phase1_brute(1);
        assert!(!brute.is_empty());
        let target = brute[0];

        use rug::{Assign, Float as RFloat};
        let prec = 256_u32;
        let mut a: [[RFloat; 17]; 16] = std::array::from_fn(|_| {
            std::array::from_fn(|_| RFloat::with_val(prec, 0.0))
        });
        for i in 0..16 {
            for j in 0..16 {
                a[i][j].assign(s.basis[j][i]);
            }
            a[i][16].assign(target[i]);
        }
        for k in 0..16 {
            let mut piv = k;
            let mut piv_abs = a[k][k].clone().abs();
            for i in (k + 1)..16 {
                let v = a[i][k].clone().abs();
                if v > piv_abs {
                    piv_abs = v;
                    piv = i;
                }
            }
            if piv != k {
                a.swap(k, piv);
            }
            assert!(a[k][k].to_f64().abs() > 1e-30,
                "B is singular at column {} — not unimodular?", k);
            for i in (k + 1)..16 {
                let factor = RFloat::with_val(prec, &a[i][k] / &a[k][k]);
                for j in k..17 {
                    let new_val = RFloat::with_val(prec, &a[i][j] - &factor * &a[k][j]);
                    a[i][j].assign(&new_val);
                }
            }
        }
        let mut z = [0_i64; 16];
        for i in (0..16).rev() {
            let mut s_acc = a[i][16].clone();
            for j in (i + 1)..16 {
                let term = RFloat::with_val(prec, &a[i][j] * z[j]);
                s_acc -= term;
            }
            let zi = RFloat::with_val(prec, &s_acc / &a[i][i]);
            let zi_round = zi.to_f64().round() as i64;
            let residual = (zi.to_f64() - zi_round as f64).abs();
            assert!(residual < 1e-6,
                "z[{}] = {} is not an integer (residual {}); basis non-unimodular?",
                i, zi.to_f64(), residual);
            z[i] = zi_round;
        }
        let recovered = reconstruct_x(&s.basis, &z);
        assert_eq!(recovered, target,
            "round-trip failed: z = {:?} gives x = {:?}, want {:?}",
            z, recovered, target);
    }

    // ── det16_exact tests ────────────────────────────────────────────────────

    #[test]
    fn det16_exact_on_identity() {
        let mut id = [[0i64; 16]; 16];
        for i in 0..16 {
            id[i][i] = 1;
        }
        assert_eq!(det16_exact(&id), Some(1));
    }

    #[test]
    fn det16_exact_on_swap() {
        // Identity with rows 0 and 1 swapped: det = -1.
        let mut m = [[0i64; 16]; 16];
        for i in 0..16 {
            m[i][i] = 1;
        }
        m[0][0] = 0;
        m[1][1] = 0;
        m[0][1] = 1;
        m[1][0] = 1;
        assert_eq!(det16_exact(&m), Some(-1));
    }

    #[test]
    fn det16_exact_on_lll_basis() {
        // A real LLL output basis must be unimodular.
        let v = realistic_v();
        let mut s = IntScratch16::new(1e-3);
        build_q_mpfr_zeta(&mut s, v, 6, 1e-3);
        build_q_int_zeta(&mut s);
        let r = run_lll_16(&mut s);
        assert!(matches!(r, super::super::lll::LllResult::Converged));
        let det = det16_exact(&s.basis).expect("LLL basis det must fit in i64");
        assert!(det == 1 || det == -1,
            "LLL output basis must be unimodular; got det = {}", det);
    }

    // ── euclidean_cholesky_16 tests ──────────────────────────────────────────

    #[test]
    fn euclidean_cholesky_16_round_trip() {
        // Identity basis: B·Bᵀ = I, R = I.
        let mut id = [[0i64; 16]; 16];
        for i in 0..16 {
            id[i][i] = 1;
        }
        let r = euclidean_cholesky_16(&id).expect("identity should be PD");
        for i in 0..16 {
            for j in 0..16 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((r[i][j] - expected).abs() < 1e-12,
                    "R[{i}][{j}] = {}, expected {expected}", r[i][j]);
            }
        }
        // Diagonal basis with entries 2: B·Bᵀ = 4·I, R = 2·I.
        let mut diag2 = [[0i64; 16]; 16];
        for i in 0..16 {
            diag2[i][i] = 2;
        }
        let r = euclidean_cholesky_16(&diag2).expect("2·I should be PD");
        for i in 0..16 {
            for j in 0..16 {
                let expected = if i == j { 2.0 } else { 0.0 };
                assert!((r[i][j] - expected).abs() < 1e-12,
                    "R[{i}][{j}] = {}, expected {expected}", r[i][j]);
            }
        }
        // Verify Rᵀ·R = B·Bᵀ for a slightly less trivial basis.
        let mut tri = [[0i64; 16]; 16];
        for i in 0..16 {
            for j in 0..=i {
                tri[i][j] = if i == j { 3 } else { 1 };
            }
        }
        let r = euclidean_cholesky_16(&tri).expect("lower-triangular full-rank should be PD");
        // Check Rᵀ·R = B·Bᵀ.
        let mut bbt = [[0.0_f64; 16]; 16];
        for i in 0..16 {
            for j in 0..16 {
                let mut s = 0.0_f64;
                for k in 0..16 {
                    s += (tri[i][k] as f64) * (tri[j][k] as f64);
                }
                bbt[i][j] = s;
            }
        }
        for i in 0..16 {
            for j in 0..16 {
                let mut s = 0.0_f64;
                for k in 0..16 {
                    s += r[k][i] * r[k][j];
                }
                assert!((s - bbt[i][j]).abs() < 1e-9,
                    "Rᵀ·R != B·Bᵀ at ({i},{j}): {} vs {}", s, bbt[i][j]);
            }
        }
    }

    // ── schnorr_euchner_16d tests ────────────────────────────────────────────

    /// SE walk on the identity basis with z_c = 0 should enumerate exactly
    /// the integer 16-vectors with ‖z‖² ≤ bound_sq. At bound_sq = 1, that's
    /// the origin + 32 nearest neighbours = 33 leaves.
    #[test]
    fn schnorr_euchner_16d_identity_basis_small_bound() {
        let mut l = [[0.0_f64; 16]; 16];
        for i in 0..16 {
            l[i][i] = 1.0;
        }
        let z_c = [0_i64; 16];
        let budget = AtomicU64::new(10_000);
        let mut visited: HashSet<[i64; 16]> = HashSet::new();
        let leaves = schnorr_euchner_16d(&l, &z_c, 1.0, |z| {
            visited.insert(*z);
            true
        }, &budget);
        assert_eq!(leaves, 33, "expected 33 leaves at bound_sq=1, got {leaves}");
        assert_eq!(visited.len(), 33);
        // Verify origin is present.
        assert!(visited.contains(&[0i64; 16]));
        // Verify each ±e_i unit vector is present.
        for i in 0..16 {
            let mut e = [0i64; 16];
            e[i] = 1;
            assert!(visited.contains(&e), "missing +e_{i}");
            e[i] = -1;
            assert!(visited.contains(&e), "missing -e_{i}");
        }
    }

    /// SE walk respects its budget: with budget=10, we get exactly 10 leaves.
    #[test]
    fn schnorr_euchner_16d_respects_budget() {
        let mut l = [[0.0_f64; 16]; 16];
        for i in 0..16 {
            l[i][i] = 1.0;
        }
        let z_c = [0_i64; 16];
        let budget = AtomicU64::new(10);
        let leaves = schnorr_euchner_16d(&l, &z_c, 4.0, |_z| true, &budget);
        assert_eq!(leaves, 10, "budget should cap leaves at 10");
    }

    /// At k=2 (norm² = 4), `phase1_brute(2)` returns 2848 valid solutions.
    /// Build the LLL+SE pipeline, run SE with a generous bound, verify any
    /// solution returned by SE that passes the leaf checks is in the brute
    /// set (no spurious solutions from SE's enumeration).
    #[test]
    fn schnorr_euchner_16d_returns_subset_of_brute_at_k_2() {
        let v = realistic_v();
        let mut s = IntScratch16::new(1e-3);
        build_q_mpfr_zeta(&mut s, v, 6, 1e-3);
        build_q_int_zeta(&mut s);
        let r = run_lll_16(&mut s);
        assert!(matches!(r, super::super::lll::LllResult::Converged));

        // f64 Cholesky on the post-LLL Gram (lower-triangular L).
        assert!(cholesky_f64_16(&mut s));
        // Transpose to upper-triangular for SE.
        let mut l_upper = [[0.0_f64; 16]; 16];
        for i in 0..16 {
            for j in 0..16 {
                l_upper[i][j] = s.l_f64[j][i];
            }
        }
        // LU solve: cap-center in basis coords. Round to i64 (SE's z_c
        // convention is integer).
        assert!(lu_solve_int_inplace_16(&mut s));
        let mut z_c = [0_i64; 16];
        for i in 0..16 {
            z_c[i] = s.lu_x[i].to_f64().round() as i64;
        }

        // Brute solutions at k=2.
        let brute_set: HashSet<[i64; 16]> = phase1_brute(2).into_iter().collect();

        // Generous bound: large enough that *some* candidates land in the
        // ellipsoid. We don't claim coverage of all 2848; the assertion is
        // that any SE candidate that ALSO passes the leaf checks is in brute.
        let budget = AtomicU64::new(1_000_000);
        let bound_sq = 1.0e6_f64;
        let mut se_set: HashSet<[i64; 16]> = HashSet::new();
        schnorr_euchner_16d(&l_upper, &z_c, bound_sq, |z| {
            // Reconstruct x = B·z and apply leaf checks.
            let x = reconstruct_x(&s.basis, z);
            let norm_sq: i64 = x.iter().map(|v| v * v).sum();
            if norm_sq != 4 {
                return true;
            }
            let (b1, b2, b3) = bilinear_forms(&x);
            if b1 == 0 && b2 == 0 && b3 == 0 {
                se_set.insert(x);
            }
            true
        }, &budget);
        // Every leaf-check-passing SE result must be in the brute set.
        for x in &se_set {
            assert!(brute_set.contains(x),
                "SE returned x={:?} not in brute set (lattice consistency bug)", x);
        }
        eprintln!("SE at k=2: found {}/{} brute solutions within bound_sq={}",
                  se_set.len(), brute_set.len(), bound_sq);
    }

    /// SE pruning: at a moderate bound the leaf count is finite and the walk
    /// terminates within budget. Uses a tight bound (radius 1.5 on the LLL-
    /// reduced first-basis-vector norm) on a moderate (k=2, ε=1e-3) target.
    #[test]
    fn schnorr_euchner_16d_pruning_is_real() {
        let v = realistic_v();
        let mut s = IntScratch16::new(1e-3);
        build_q_mpfr_zeta(&mut s, v, 6, 1e-3);
        build_q_int_zeta(&mut s);
        let r = run_lll_16(&mut s);
        assert!(matches!(r, super::super::lll::LllResult::Converged));

        assert!(cholesky_f64_16(&mut s));
        let mut l_upper = [[0.0_f64; 16]; 16];
        for i in 0..16 {
            for j in 0..16 {
                l_upper[i][j] = s.l_f64[j][i];
            }
        }
        assert!(lu_solve_int_inplace_16(&mut s));
        let mut z_c = [0_i64; 16];
        for i in 0..16 {
            z_c[i] = s.lu_x[i].to_f64().round() as i64;
        }

        // Pick a bound a few times the smallest diagonal² of the upper
        // factor: this scales the ellipsoid to cover ~1-10 leaves in the
        // tightest direction. Concretely, the smallest diagonal² of L is the
        // Q-norm of the *last* GS-orthogonalised basis row, which post-LLL
        // is the longest direction in the lattice; using 4× that as the
        // bound covers a small but non-trivial number of integer points.
        let mut min_diag_sq = f64::INFINITY;
        for i in 0..16 {
            let v = l_upper[i][i] * l_upper[i][i];
            if v > 0.0 && v < min_diag_sq {
                min_diag_sq = v;
            }
        }
        let bound_sq = 4.0 * min_diag_sq;
        let budget = AtomicU64::new(1_000_000);
        let leaves = schnorr_euchner_16d(&l_upper, &z_c, bound_sq, |_z| true, &budget);
        // Walk completes within budget (no abort).
        assert!(leaves < 1_000_000,
            "SE walk did not terminate within budget: visited {leaves} leaves");
        // Pruning is real: leaf count is far below the unrestricted box of
        // even radius 1 in 16D (3^16 ≈ 4.3×10⁷).
        assert!(leaves < 1_000_000,
            "SE pruning failed: visited {leaves} leaves (budget = 1M)");
        eprintln!(
            "SE pruning at k=6, ε=1e-3: visited {leaves} leaves (bound_sq={bound_sq:.3e}, \
             min_diag_sq={min_diag_sq:.3e})"
        );
    }
}
