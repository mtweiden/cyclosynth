//! Lattice enumeration constraints + brute-force phase1 for Z[ζ₂₄] / n=12.
//!
//! ## Constraints (SPEC §5)
//!
//! Per ring element u with coeffs `c_0..c_7`, `u·conj(u)` lives in the real
//! subfield `Q(√2, √3)` and decomposes as `r + s₂·√2 + s₃·√3 + s₆·√6`.
//! Synthesis requires:
//!
//!   1. **Norm shell** — `N(u₁) + N(u₂) = 2^k` (default √2 denominator).
//!      `N(u) = (c₀² + … + c₇²) + (c₀c₄ + c₁c₅ + c₂c₆ + c₃c₇)` in the
//!      cyclotomic basis. Equivalently, the radical-basis form
//!      `a₀² + a₁² + 2a₂² + 2a₃² + 3a₄² + 3a₅² + 6a₆² + 6a₇²` (these agree;
//!      tested in [`norm_via_sigma_matches_zupsilon_norm_sqr`] of
//!      `sigma.rs`).
//!
//!   2. **Three bullet vanishings** — `s₂(u₁) + s₂(u₂) = 0`,
//!      `s₃(u₁) + s₃(u₂) = 0`, `s₆(u₁) + s₆(u₂) = 0`.
//!      In the radical basis these read
//!        √2:  `a₀a₂ + a₁a₃ + 3a₄a₆ + 3a₅a₇ = 0`
//!        √3:  `a₀a₄ + a₁a₅ + 2a₂a₆ + 2a₃a₇ = 0`
//!        √6:  `a₀a₆ + a₁a₇ + a₂a₄ + a₃a₅ = 0`
//!      (sum over u₁,u₂). See [`bullets_total_twice`] for the
//!      cyclotomic-basis version actually computed by the enumerator.
//!
//!   3. **Alignment** — `|x · ỹ|² ≥ target · (1 − ε²)` where
//!      `ỹ = Σ_{σ_1}^T v` is the cap-block pullback of the SU(2) target's
//!      first column `v ∈ R⁴`. Implemented in `synthesize.rs`.
//!
//! ## Enumerator structure
//!
//! The √T module's LLL + Schnorr-Euchner pipeline (anisotropic Gram `G =
//! Σ^T Σ`, `Σ⁻¹ = (Σ^T Σ)⁻¹ Σ^T`, see SPEC §4) ports to this case
//! verbatim — the only ring-specific pieces are the Gram (now `4I+2C`
//! per element instead of `4I` for Z[ω]) and the bullet leaf check (three
//! forms instead of one). That port is gated on §7's gate-set decision
//! and tracked at the top of [`mod`].
//!
//! For now we expose [`phase1_brute`]: a depth-first enumeration of
//! integer 16-vectors with the exact norm and bullet checks, suitable as
//! a correctness oracle for the LLL pipeline and for small Clifford
//! recovery tests. Matches the role of `search_zeta::phase1_brute` for
//! the n=16 / √T case. Carries the two performance/correctness levers
//! from `bandb5/7.py`:
//!
//! - `max_solutions = 1` short-circuit ([`phase1_brute_first`]) — return
//!   on the first valid hit.
//! - Bullet leaf check uses the **cyclotomic-basis formula** derived
//!   directly from `u·conj(u)` so we avoid the parity / index-2 issue
//!   that a naive radical-basis (a-coord) DFS would hit (SPEC §8).

use crate::rings::types::{Int, INT_ONE, INT_TWO, INT_ZERO};
use crate::rings::ZUpsilon;

/// Per-element cyclotomic-basis norm: `(c₀² + … + c₇²) + Σ_{i=0..3} c_i c_{i+4}`.
///
/// Equals `(1/4)·x^T G_el x` where `G_el = Σ_el^T Σ_el = 4I + 2C` (SPEC §4)
/// and is integer-valued on integer cyclotomic coords. Sum over u₁,u₂
/// equals the synthesis radius `2^k`.
#[inline]
pub fn norm_sqr_per_element(c: &[i64; 8]) -> i64 {
    let mut s: i64 = 0;
    for v in c {
        s += v * v;
    }
    for i in 0..4 {
        s += c[i] * c[i + 4];
    }
    s
}

/// Total norm `N(u₁) + N(u₂)` for a 16-element solution.
#[inline]
pub fn norm_sqr_total(x: &[i64; 16]) -> i64 {
    let (a, b) = x.split_at(8);
    let ca: [i64; 8] = a.try_into().expect("len 8");
    let cb: [i64; 8] = b.try_into().expect("len 8");
    norm_sqr_per_element(&ca) + norm_sqr_per_element(&cb)
}

/// Return `(2·s₂, 2·s₃, 2·s₆)` of `u·conj(u)` for a single ring element
/// from its cyclotomic-basis coefficients `c_0..c_7`.
///
/// Derived by multiplying `u · conj(u)` symbolically (matching
/// [`ZUpsilon::Mul`] reduction `ζ⁸ = ζ⁴ − 1`) and extracting the
/// (ζ¹, ζ², ζ⁶, ζ⁷) coefficients of the resulting ZUpsilon `p`:
///
///   `2·s₂ = 2·p.b + p.h`,  `2·s₃ = p.c`,  `2·s₆ = −p.h`.
///
/// The function returns integers (no division), making it safe to use as
/// a leaf check in integer-coefficient enumerators.
#[inline]
pub fn bullets_per_element_twice(c: &[i64; 8]) -> (i64, i64, i64) {
    // conj(u) coefficients in cyclotomic basis (see `ZUpsilon::conj`):
    //   d_0 = c_0 + c_4,           d_1 = c_3,
    //   d_2 = c_2,                 d_3 = c_1,
    //   d_4 = -c_4,                d_5 = -c_3 - c_7,
    //   d_6 = -c_2 - c_6,          d_7 = -c_1 - c_5.
    let d0 = c[0] + c[4];
    let d1 = c[3];
    let d2 = c[2];
    let d3 = c[1];
    let d4 = -c[4];
    let d5 = -c[3] - c[7];
    let d6 = -c[2] - c[6];
    let d7 = -c[1] - c[5];
    let d = [d0, d1, d2, d3, d4, d5, d6, d7];

    // Polynomial product t_k = Σ_{i+j=k} c_i d_j, k = 0..14.
    let mut t = [0i64; 15];
    for i in 0..8 {
        for j in 0..8 {
            t[i + j] += c[i] * d[j];
        }
    }
    // Reduce with ζ⁸ = ζ⁴ − 1 (iterative, high-to-low).
    // result[0] = t0 − (t8 + t12)
    // result[1] = t1 − (t9 + t13)
    // result[2] = t2 − (t10 + t14)
    // result[3] = t3 − t11
    // result[4] = t4 + (t8 + t12)
    // result[5] = t5 + (t9 + t13)
    // result[6] = t6 + (t10 + t14)
    // result[7] = t7 + t11
    let p_b = t[1] - (t[9] + t[13]); // coeff of ζ¹ in u·conj(u)
    let p_c = t[2] - (t[10] + t[14]); // coeff of ζ²
    let p_h = t[7] + t[11]; // coeff of ζ⁷

    let two_s2 = 2 * p_b + p_h;
    let two_s3 = p_c;
    let two_s6 = -p_h;
    (two_s2, two_s3, two_s6)
}

/// Sum-of-elements version. Returns `(2(s₂(u₁)+s₂(u₂)), 2(s₃...), 2(s₆...))`.
///
/// All three components must be zero for `x` to encode a unitary pair
/// `(u₁, u₂)` after dividing by the synthesis denominator.
#[inline]
pub fn bullets_total_twice(x: &[i64; 16]) -> (i64, i64, i64) {
    let (a, b) = x.split_at(8);
    let ca: [i64; 8] = a.try_into().expect("len 8");
    let cb: [i64; 8] = b.try_into().expect("len 8");
    let (s2a, s3a, s6a) = bullets_per_element_twice(&ca);
    let (s2b, s3b, s6b) = bullets_per_element_twice(&cb);
    (s2a + s2b, s3a + s3b, s6a + s6b)
}

/// True iff `x` satisfies all three bullet vanishings (sum over u₁,u₂).
#[inline]
pub fn bullets_zero(x: &[i64; 16]) -> bool {
    let (b2, b3, b6) = bullets_total_twice(x);
    b2 == 0 && b3 == 0 && b6 == 0
}

// ─── Int-typed variants (overflow-safe for the LLL+SE path) ──────────────────
//
// The i64 helpers above are safe for the brute-force range (`k ≤ ~5`). For
// higher k — and for any caller that's already in `Int` (i256) territory,
// like `clifford_pi12`'s ring algebra — use the variants below. Same formulas,
// no widening conversion cost since `Int` ops are already i256.

/// Per-element cyclotomic-basis norm in `Int`. Same formula as
/// [`norm_sqr_per_element`] but in `Int` to avoid overflow at large `k`.
pub fn norm_sqr_per_element_int(c: &[Int; 8]) -> Int {
    let mut s = INT_ZERO;
    for v in c {
        s = s + *v * *v;
    }
    for i in 0..4 {
        s = s + c[i] * c[i + 4];
    }
    s
}

/// Total norm `N(u₁) + N(u₂)` (Int).
pub fn norm_sqr_total_int(x: &[Int; 16]) -> Int {
    let mut ca = [INT_ZERO; 8];
    let mut cb = [INT_ZERO; 8];
    ca.copy_from_slice(&x[..8]);
    cb.copy_from_slice(&x[8..]);
    norm_sqr_per_element_int(&ca) + norm_sqr_per_element_int(&cb)
}

/// `(2·s₂, 2·s₃, 2·s₆)` of `u·conj(u)` in `Int`. See
/// [`bullets_per_element_twice`] for the derivation.
pub fn bullets_per_element_twice_int(c: &[Int; 8]) -> (Int, Int, Int) {
    let d0 = c[0] + c[4];
    let d1 = c[3];
    let d2 = c[2];
    let d3 = c[1];
    let d4 = -c[4];
    let d5 = -c[3] - c[7];
    let d6 = -c[2] - c[6];
    let d7 = -c[1] - c[5];
    let d = [d0, d1, d2, d3, d4, d5, d6, d7];

    let mut t = [INT_ZERO; 15];
    for i in 0..8 {
        for j in 0..8 {
            t[i + j] = t[i + j] + c[i] * d[j];
        }
    }
    let p_b = t[1] - (t[9] + t[13]);
    let p_c = t[2] - (t[10] + t[14]);
    let p_h = t[7] + t[11];

    let two_s2 = INT_TWO * p_b + p_h;
    let two_s3 = p_c;
    let two_s6 = -p_h;
    (two_s2, two_s3, two_s6)
}

/// Sum over u₁,u₂ in `Int`.
pub fn bullets_total_twice_int(x: &[Int; 16]) -> (Int, Int, Int) {
    let mut ca = [INT_ZERO; 8];
    let mut cb = [INT_ZERO; 8];
    ca.copy_from_slice(&x[..8]);
    cb.copy_from_slice(&x[8..]);
    let (s2a, s3a, s6a) = bullets_per_element_twice_int(&ca);
    let (s2b, s3b, s6b) = bullets_per_element_twice_int(&cb);
    (s2a + s2b, s3a + s3b, s6a + s6b)
}

/// True iff all three bullet vanishings hold (Int).
#[inline]
pub fn bullets_zero_int(x: &[Int; 16]) -> bool {
    let (b2, b3, b6) = bullets_total_twice_int(x);
    b2 == INT_ZERO && b3 == INT_ZERO && b6 == INT_ZERO
}

/// `2^k` as an `Int`, for setting the synthesis radius without an
/// intermediate `i64` (which overflows for `k ≥ 63`).
#[inline]
pub fn target_norm_int(k: u32) -> Int {
    let mut t = INT_ONE;
    for _ in 0..k {
        t = t * INT_TWO;
    }
    t
}

// ─── Brute-force phase1 ──────────────────────────────────────────────────────

/// Depth-first enumeration of integer 16-vectors with bounded Euclidean
/// norm, calling `cb` at each leaf with `‖x‖² = target_sum_sq`.
///
/// The cyclotomic-basis norm `N(u₁)+N(u₂)` differs from `Σxᵢ²` by cross
/// terms `xᵢxᵢ₊₄`. Eigenvalue bounds on `G_el = 4I+2C` give
/// `Σxᵢ² ∈ [(2/3)·N_tot, 2·N_tot]`, so DFS by Σxᵢ² up to `2·2^k` and
/// then doing the precise norm check at the leaf is correct and fast.
fn enumerate_by_sum_sq<F: FnMut(&[i64; 16])>(
    x: &mut [i64; 16],
    pos: usize,
    remaining: i64,
    cb: &mut F,
) {
    if pos == 16 {
        if remaining >= 0 {
            cb(x);
        }
        return;
    }
    let bound = (remaining as f64).sqrt().floor() as i64;
    for v in -bound..=bound {
        let v2 = v * v;
        if v2 > remaining {
            continue;
        }
        x[pos] = v;
        enumerate_by_sum_sq(x, pos + 1, remaining - v2, cb);
    }
}

/// Brute-force phase1 for n=12 / Z[ζ₂₄]: returns all integer 16-vectors
/// `(u₁-coeffs, u₂-coeffs)` with
///   * `N(u₁) + N(u₂) = 2^k`  (√2 denominator — forced by ζ₂₄ being in
///     the golden set, see module docs), and
///   * all three bullets zero.
///
/// Cost is exponential in `k` (per the Euclidean bound `Σxᵢ² ≤ 2^(k+1)`);
/// usable for `k ≤ 3` as a correctness oracle. Mirrors the role of
/// `search_zeta::phase1_brute` for the n=16 √T case.
pub fn phase1_brute(k: u32) -> Vec<[i64; 16]> {
    let target = 1i64 << k;
    let euclid_bound = 2 * target;
    let mut x = [0i64; 16];
    let mut out = Vec::new();
    enumerate_by_sum_sq(&mut x, 0, euclid_bound, &mut |x| {
        if norm_sqr_total(x) != target {
            return;
        }
        if !bullets_zero(x) {
            return;
        }
        out.push(*x);
    });
    out
}

/// `max_solutions = 1` short-circuit version: returns `Some(x)` on the
/// first valid hit (matches `bandb5/7.py` semantics, SPEC §8).
pub fn phase1_brute_first(k: u32) -> Option<[i64; 16]> {
    let target = 1i64 << k;
    let euclid_bound = 2 * target;
    let mut x = [0i64; 16];
    let mut found: Option<[i64; 16]> = None;
    // Early-exit via a closure that flips `found`.
    fn walk<F: FnMut(&[i64; 16]) -> bool>(
        x: &mut [i64; 16],
        pos: usize,
        remaining: i64,
        cb: &mut F,
    ) -> bool {
        if pos == 16 {
            if remaining >= 0 {
                return cb(x);
            }
            return false;
        }
        let bound = (remaining as f64).sqrt().floor() as i64;
        for v in -bound..=bound {
            let v2 = v * v;
            if v2 > remaining {
                continue;
            }
            x[pos] = v;
            if walk(x, pos + 1, remaining - v2, cb) {
                return true;
            }
        }
        false
    }
    let _ = walk(&mut x, 0, euclid_bound, &mut |x| {
        if norm_sqr_total(x) != target {
            return false;
        }
        if !bullets_zero(x) {
            return false;
        }
        found = Some(*x);
        true
    });
    found
}

// ─── Alignment helpers (cap pullback) ────────────────────────────────────────

use crate::synthesis::lattice_upsilon::sigma::{sigma_el, D_EL, D_FULL};

/// Cap-block alignment vector `ỹ` in lattice coordinates.
///
/// The synthesizer wants `|x · ỹ|² ≥ target · (1 − ε²)` where `ỹ` is the
/// pullback of the σ_1 (cap) image of `(u₁,u₂) ≈ √(2^k) · target` into
/// lattice space. The cap rows of `Σ_el` are rows 0 (Re σ_1) and 1
/// (Im σ_1); the pullback is `ỹ_block = Σ_{σ_1}^T · v` per ring element.
///
/// Layout of the input `v`:
///   `v = (Re V_{11}, Im V_{11}, Re V_{21}, Im V_{21})`.
///
/// Output: 16-element lattice-coord y vector (no scale; multiply by
/// `√(2^k)` to set the alignment radius — see [`uv_to_xy`]).
pub fn compute_align_vec(v: [f64; 4]) -> [f64; D_FULL] {
    let el = sigma_el();
    let mut y = [0.0f64; D_FULL];
    for j in 0..D_EL {
        // row 0 = Re σ_1, row 1 = Im σ_1 of Σ_el
        y[j] = el[0][j] * v[0] + el[1][j] * v[1];
        y[D_EL + j] = el[0][j] * v[2] + el[1][j] * v[3];
    }
    y
}

/// Lattice-coord alignment vector: just `compute_align_vec(v)` — NO `√(2^k)`
/// scaling.
///
/// Matches the lattice_omicron / n=6 convention where `y = Σ_topᵀ · v_pad`
/// (raw, no R factor). With this normalization, target `(y · x_target)² =
/// 2^k` (not `2^(2k)`), which keeps the Q-metric's rank-1 term k-independent
/// in Q-norm units. The earlier `√(2^k) · compute_align_vec(v)` form leaked
/// a hidden `2^k` factor into the rank-1 Q² contribution, blowing the
/// SE bound out at production depth.
///
/// `‖ỹ‖²` is NOT constant in `v` (the cap-block Gram is `4I+2C` — not a
/// scalar identity — SPEC §6); compute it per call rather than caching.
pub fn uv_to_xy(v: [f64; 4], _k: u32) -> [f64; D_FULL] {
    compute_align_vec(v)
}

/// `|x · ỹ|²` for a candidate `x` and target `ỹ` at the same scale.
#[inline]
pub fn alignment_sq(x: &[i64; D_FULL], y: &[f64; D_FULL]) -> f64 {
    let mut s = 0.0f64;
    for i in 0..D_FULL {
        s += x[i] as f64 * y[i];
    }
    s * s
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rings::types::{Int, INT_ZERO};

    /// `bullets_per_element_twice` agrees with `ZUpsilon::complex_norm_sqr_components_twice`.
    /// Establishes that our fast i64 derivation matches the symbolic ring Mul.
    #[test]
    fn bullets_match_ring_mul() {
        let cases: [[i64; 8]; 6] = [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [3, -1, 2, 1, 0, -2, 1, -1],
            [-2, 5, 0, -3, 4, 1, -1, 2],
            [1, -1, 1, -1, 1, -1, 1, -1],
        ];
        for c in &cases {
            let u = ZUpsilon::from_i32(
                c[0] as i32,
                c[1] as i32,
                c[2] as i32,
                c[3] as i32,
                c[4] as i32,
                c[5] as i32,
                c[6] as i32,
                c[7] as i32,
            );
            let (r_ring, s2_ring, s3_ring, s6_ring) = u.complex_norm_sqr_components_twice();
            let (s2_fast, s3_fast, s6_fast) = bullets_per_element_twice(c);
            // The rational component is u.norm_sqr (returns r·1). We just compare irrational parts.
            let _ = r_ring;
            assert_eq!(s2_ring, Int::from_i64(s2_fast), "2·s₂ mismatch for {c:?}");
            assert_eq!(s3_ring, Int::from_i64(s3_fast), "2·s₃ mismatch for {c:?}");
            assert_eq!(s6_ring, Int::from_i64(s6_fast), "2·s₆ mismatch for {c:?}");
        }
    }

    /// `bullets_per_element_twice` matches the radical-basis SPEC formula
    /// when `u` is expressed via its radical-basis a-coords. We pick
    /// integer-radical inputs whose cyclotomic equivalents are integer:
    /// `u = a_0 + a_1·i + a_2·√2 + a_3·i√2 + a_4·√3 + a_5·i√3 + a_6·√6 + a_7·i√6`.
    /// Cyclotomic-basis: build via ring identities ζ⁶ = i, sqrt2/sqrt3/sqrt6.
    #[test]
    fn bullets_radical_formula() {
        let cases: [[i64; 8]; 4] = [
            [1, 0, 1, 0, 0, 0, 0, 0],
            [2, -1, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ];
        for a in &cases {
            // Build u = a0·1 + a1·i + a2·√2 + a3·i√2 + a4·√3 + a5·i√3 + a6·√6 + a7·i√6
            let to_zu = |n: i64| Int::from_i64(n);
            let u = ZUpsilon::ONE.scale(to_zu(a[0]))
                + ZUpsilon::I.scale(to_zu(a[1]))
                + ZUpsilon::sqrt2().scale(to_zu(a[2]))
                + (ZUpsilon::I * ZUpsilon::sqrt2()).scale(to_zu(a[3]))
                + ZUpsilon::sqrt3().scale(to_zu(a[4]))
                + (ZUpsilon::I * ZUpsilon::sqrt3()).scale(to_zu(a[5]))
                + ZUpsilon::sqrt6().scale(to_zu(a[6]))
                + (ZUpsilon::I * ZUpsilon::sqrt6()).scale(to_zu(a[7]));
            // Cyclotomic coeffs as i64
            let c: [i64; 8] = std::array::from_fn(|i| u.coeff(i).as_i128() as i64);
            let (two_s2, two_s3, two_s6) = bullets_per_element_twice(&c);

            // SPEC §5 radical-basis formulas (the "minimal" integer form
            // — what you actually want to vanish). Expanding
            // `(A + B·i)·(A − B·i)` with `A = a₀ + a₂√2 + a₄√3 + a₆√6`
            // (and `B` similarly with the odd-index a-coords) gives
            // each irrational component as **2×** the SPEC formula
            // (e.g. `2a₀a₂√2 + 6a₄a₆√2 = 2·(a₀a₂ + 3a₄a₆)√2`); since
            // `bullets_per_element_twice` returns `2·s_k`, the relating
            // factor is 4×.
            let expect_s2 = 4 * (a[0] * a[2] + a[1] * a[3] + 3 * a[4] * a[6] + 3 * a[5] * a[7]);
            let expect_s3 = 4 * (a[0] * a[4] + a[1] * a[5] + 2 * a[2] * a[6] + 2 * a[3] * a[7]);
            let expect_s6 = 4 * (a[0] * a[6] + a[1] * a[7] + a[2] * a[4] + a[3] * a[5]);

            assert_eq!(two_s2, expect_s2, "√2 bullet mismatch for a={a:?}");
            assert_eq!(two_s3, expect_s3, "√3 bullet mismatch for a={a:?}");
            assert_eq!(two_s6, expect_s6, "√6 bullet mismatch for a={a:?}");

            // Equivalent vanishing-test form: SPEC formula is zero iff
            // `bullets_per_element_twice` returns zero (load-bearing for
            // the enumerator's leaf check correctness).
            let spec_s2 = a[0] * a[2] + a[1] * a[3] + 3 * a[4] * a[6] + 3 * a[5] * a[7];
            assert_eq!(two_s2 == 0, spec_s2 == 0, "√2 vanishing form for a={a:?}");
        }
    }

    /// `norm_sqr_per_element` matches `ZUpsilon::norm_sqr`.
    #[test]
    fn norm_matches_ring_norm() {
        let cases: [[i64; 8]; 4] = [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [1, -1, 1, -1, 1, -1, 1, -1],
            [3, -1, 2, 1, 0, -2, 1, -1],
        ];
        for c in &cases {
            let u = ZUpsilon::from_i32(
                c[0] as i32,
                c[1] as i32,
                c[2] as i32,
                c[3] as i32,
                c[4] as i32,
                c[5] as i32,
                c[6] as i32,
                c[7] as i32,
            );
            let n_fast = norm_sqr_per_element(c);
            let n_ring = u.norm_sqr();
            assert_eq!(n_ring, Int::from_i64(n_fast));
            // Sanity: norm is non-negative.
            assert!(n_fast >= 0, "norm_sqr_per_element negative on {c:?}");
            let _ = INT_ZERO;
        }
    }

    /// `phase1_brute(0)` must include the trivial unit solutions (those with
    /// total norm 1: u₁ = ±1·ζ^j with u₂ = 0, and vice versa). Each such
    /// element has integer cyclotomic coords ±e_j for j∈0..7 plus the
    /// "lifted" ones c = ±(1,0,0,0,−1,0,0,0) etc.? Actually e_j alone has
    /// norm = 1 always (Σxᵢ² = 1, no cross-term active). So 16 trivial
    /// units * 2 = 32 placements times 2 signs = 64 single-coord-only sols.
    #[test]
    fn phase1_brute_k0_finds_trivial_units() {
        let sols = phase1_brute(0);
        // Every solution must pass both checks.
        for x in &sols {
            assert_eq!(norm_sqr_total(x), 1, "norm not 1 on {x:?}");
            assert!(bullets_zero(x), "bullets not zero on {x:?}");
        }
        // Must contain trivial e_j (u₁ = ζ^j, u₂ = 0).
        for j in 0..8 {
            let mut x = [0i64; 16];
            x[j] = 1;
            assert!(sols.contains(&x), "missing trivial unit u₁=ζ^{j}");
        }
    }

    /// `phase1_brute_first` short-circuits and matches the first hit
    /// (modulo enumeration ordering this is the first feasible 16-tuple).
    #[test]
    fn phase1_brute_first_returns_valid_solution() {
        let sol = phase1_brute_first(0).expect("k=0 should have solutions");
        assert_eq!(norm_sqr_total(&sol), 1);
        assert!(bullets_zero(&sol));
    }

    /// `phase1_brute` at small k produces only solutions satisfying both
    /// constraints (sanity that the leaf checks aren't accidentally
    /// short-circuiting on partial state).
    #[test]
    fn phase1_brute_k1_constraints_hold() {
        let sols = phase1_brute(1);
        for x in &sols {
            assert_eq!(norm_sqr_total(x), 2);
            assert!(bullets_zero(x));
        }
        // At k=1 (total norm 2), explicit example: u₁=1, u₂=1 (e_0 + e_8).
        let mut trivial = [0i64; 16];
        trivial[0] = 1;
        trivial[8] = 1;
        assert!(sols.contains(&trivial), "missing (1, 1) at k=1");
    }

    /// Int-typed helpers agree with i64 helpers on values that fit in i64.
    #[test]
    fn int_helpers_match_i64_at_small_values() {
        let cases: [[i64; 16]; 4] = [
            [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, -1, 2, 0, 1, 1, 0, -1, 0, 1, -1, 2, 0, 0, 1, 1],
            [3, 2, -1, 0, 4, -2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [-1, 0, 1, 0, 1, 0, -1, 0, 1, 0, -1, 0, -1, 0, 1, 0],
        ];
        for x in &cases {
            let x_int: [Int; 16] = std::array::from_fn(|i| Int::from_i64(x[i]));
            assert_eq!(norm_sqr_total_int(&x_int), Int::from_i64(norm_sqr_total(x)));
            let (b2_i64, b3_i64, b6_i64) = bullets_total_twice(x);
            let (b2_int, b3_int, b6_int) = bullets_total_twice_int(&x_int);
            assert_eq!(b2_int, Int::from_i64(b2_i64));
            assert_eq!(b3_int, Int::from_i64(b3_i64));
            assert_eq!(b6_int, Int::from_i64(b6_i64));
        }
    }

    /// **Overflow regression**: at `k = 60`, the synthesis target `2^60`
    /// and per-coord bound `|c_i| ≈ 2^30` together generate `c_i² ≈ 2^60`
    /// and 16-fold sums that overflow `i64`. The `Int` helpers must
    /// absorb this cleanly. Construct a degenerate but valid candidate
    /// `u₁ = 2^30, u₂ = 0` (norm = `2^60`, bullets zero), and check the
    /// constraint identities hold in `Int`. The matching `i64` call
    /// would overflow on `c_0²`.
    #[test]
    fn int_helpers_no_overflow_at_k60() {
        let mut x_int = [INT_ZERO; 16];
        x_int[0] = target_norm_int(30); // u₁ = 2^30 ∈ Z[ζ₂₄]
        let target = target_norm_int(60);
        let norm = norm_sqr_total_int(&x_int);
        assert_eq!(norm, target, "norm should equal 2^60 in Int");
        assert!(
            bullets_zero_int(&x_int),
            "rational u₁=2^30, u₂=0 must satisfy all bullets"
        );
    }

    /// `alignment_sq` is invariant under x · sign(y).
    #[test]
    fn alignment_sq_is_squared() {
        let mut x = [0i64; 16];
        x[0] = 1;
        let y = compute_align_vec([1.0, 0.0, 0.0, 0.0]);
        let v_pos = alignment_sq(&x, &y);
        let y_neg: [f64; 16] = std::array::from_fn(|i| -y[i]);
        let v_neg = alignment_sq(&x, &y_neg);
        assert!((v_pos - v_neg).abs() < 1e-12);
    }

    /// `uv_to_xy(v, k)` returns `compute_align_vec(v)` unscaled (k-independent).
    /// Post-y-rescaling fix: dropped the spurious `√(2^k)` factor so the
    /// Q-metric's rank-1 term is k-independent. See `uv_to_xy` docstring.
    #[test]
    fn uv_to_xy_is_raw_align_vec() {
        let v = [0.5, 0.3, 0.7, -0.4];
        for &k in &[0u32, 3, 6, 12] {
            let raw = compute_align_vec(v);
            let scaled = uv_to_xy(v, k);
            for i in 0..16 {
                assert!(
                    (scaled[i] - raw[i]).abs() < 1e-14,
                    "uv_to_xy at k={k} differs from compute_align_vec",
                );
            }
        }
    }

    /// `x · ỹ` measures `v · σ_1(u)` in R² — the cap alignment. Verifies
    /// the inner-product semantics that the synthesizer uses.
    ///
    /// Unlike the ζ₈ case, Σ_el rows are NOT pairwise orthogonal (only
    /// columns are — `Σ_el^T Σ_el = 4I+2C`), so `Σ_el · ỹ` is NOT
    /// `(4v, 0, …, 0)`. Instead the load-bearing fact is `x · ỹ = v_u₁ ·
    /// σ_1(u₁) + v_u₂ · σ_1(u₂)`. See SPEC §6.
    #[test]
    fn align_vec_inner_product_equals_v_dot_sigma1() {
        use crate::synthesis::lattice_upsilon::sigma::embed_one;
        let v = [0.5, 0.3, 0.7, -0.4];
        let y = compute_align_vec(v);
        let u1 = ZUpsilon::from_i32(3, -1, 2, 1, 0, -2, 1, -1);
        let u2 = ZUpsilon::from_i32(-2, 5, 0, -3, 4, 1, -1, 2);
        let s1_u1 = embed_one(&u1);
        let s1_u2 = embed_one(&u2);
        // Build x ∈ Z¹⁶.
        let mut x = [0i64; 16];
        for i in 0..8 {
            x[i] = u1.coeff(i).as_i128() as i64;
            x[8 + i] = u2.coeff(i).as_i128() as i64;
        }
        // x · ỹ in f64
        let mut dot = 0.0f64;
        for i in 0..16 {
            dot += x[i] as f64 * y[i];
        }
        // Expected: v · σ_1(u₁) + v · σ_1(u₂) (the cap rows are 0,1 of Σ_el).
        let expected = v[0] * s1_u1[0] + v[1] * s1_u1[1] + v[2] * s1_u2[0] + v[3] * s1_u2[1];
        assert!(
            (dot - expected).abs() < 1e-9,
            "x·ỹ = {dot}, expected v·σ_1(u₁) + v·σ_1(u₂) = {expected}"
        );
    }
}
