//! Meet-in-the-middle (MITM) phase1 backend for n=12 (`Z[ζ₂₄]`).
//!
//! Implements PROMPT_lattice_upsilon_mitm.md. The 16D joint problem
//! `x = [u₁(8) | u₂(8)]` with leaf checks {`rat-norm = 2^k`, `b₂=b₃=b₆=0`,
//! alignment} is split into two 8D half-problems. The three leaf
//! invariants are additive across the split:
//!
//!   rat(|u₁|²) + rat(|u₂|²) = 2^k,
//!   b_j(u₁) + b_j(u₂) = 0   for j ∈ {2, 3, 6},
//!   y · x = y₁ · u₁ + y₂ · u₂.
//!
//! Hence a half-pair is norm + bullet valid iff
//!
//!   key(u₂) = (2^k − r₁, −β₂_1, −β₃_1, −β₆_1)
//!
//! where `key(u) = (rat(|u|²), 2s₂, 2s₃, 2s₆)` reuses
//! [`super::enumerate::norm_sqr_per_element`] and
//! [`super::enumerate::bullets_per_element_twice`] (so the doubled-bullet
//! convention here is the SAME the joint leaf check uses — keys agree by
//! construction). All four key components are exact integers, the hash-join
//! is exact, and alignment + diamond distance are checked at the joint
//! level after the join.
//!
//! # Per-half soundness region
//!
//! Re-derived against the project's exact threshold `(y·x)² ≥ 2^k (1−ε²)`.
//! With `R² = 2^k`, target column `V = (V₁₁, V₂₁)` and `‖V‖² = 1`,
//! defining `δ_i = σ₁(u_i) − R·V_i` and using that the leaf check forces
//!   • rational norm shell: `|σ₁(u₁)|² + |σ₁(u₂)|² = R²` (one of the
//!     algebraic shells implied by `r₁+r₂=2^k` ∧ bullets vanish),
//!   • alignment: `y·x = R + Re(δ·V*) ≥ R√(1−ε²)`,
//! Pythagoras gives
//!   `‖δ‖² = 2R² − 2(y·x) ≤ 2R²(1 − √(1−ε²)) ≤ 2R² ε²`,
//! and `‖δ‖² = ‖δ_1‖² + ‖δ_2‖²` makes each half satisfy
//!   `‖σ₁(u_i) − R·V_i‖² ≤ 2R² ε²`           (σ₁-cap)
//! while bullet vanishing at the conjugate embeddings `m ∈ {17,13,5}` plus
//! the norm shell give
//!   `|σ_m(u_i)|² ≤ R²`                       (conjugate-norm balls)
//! as the sound per-half outer cover. Anything outside this cannot be a
//! valid half of any leaf-passing pair.
//!
//! (Threshold differs from the prompt's literal form by less than ε² (the
//! 1 − √(1−ε²) ≤ ε²/2 + O(ε⁴) tightening); we use the looser `2R²ε²` cap
//! exactly as the prompt prescribes — it's still a sound outer cover.)
//!
//! # Backends
//!
//! - [`brute_enumerate_half`] — box-enumerate the coefficient cube with
//!   `|c_j| ≤ ⌈R√2⌉` and filter by the per-half region. Sound for any k;
//!   tractable for small k (k ≤ 3 trivial, k = 4 ~10 s). Used for the
//!   Part-3 soundness gates and as the fallback when the smart 8D
//!   enumerator is not available.
//! - (`SmartEnumerateHalf` — 8D LLL/SE enumerator: TODO for Part 4.)

#![allow(clippy::needless_range_loop)]

use super::enumerate::{bullets_per_element_twice, norm_sqr_per_element};
use super::sigma::sigma_el;
use num_complex::Complex64;
use std::collections::HashMap;

/// The four exact integer invariants used as a hash key for the join.
/// `(r, β₂, β₃, β₆)` where `r = rat(|u|²)` and `(β₂, β₃, β₆) = 2(s₂, s₃, s₆)`
/// (the doubled-bullet convention; matches
/// [`super::enumerate::bullets_per_element_twice`]).
pub type HalfKey = (i64, i64, i64, i64);

#[inline]
pub fn key_of(u: &[i64; 8]) -> HalfKey {
    let r = norm_sqr_per_element(u);
    let (b2, b3, b6) = bullets_per_element_twice(u);
    // i64 key guard. Earlier work (pre-MITM) had silent overflow on
    // doubled-bullet accumulators when `|u|` exceeded ~2^31; the per-half
    // soundness region caps `|u|` by ~R = √(2^k), so rat-norm ≤ 2^k and
    // bullet magnitudes stay within ~k·2^k. Trip at the safe-k boundary.
    debug_assert!(
        r >= 0 && r <= 1_i64 << 40,
        "HalfKey rat-norm out of safe range: {r} (k too large for i64 keys)"
    );
    debug_assert!(
        b2.unsigned_abs() < 1_u64 << 60
            && b3.unsigned_abs() < 1_u64 << 60
            && b6.unsigned_abs() < 1_u64 << 60,
        "HalfKey bullet magnitudes exceed i64 safety margin: ({b2}, {b3}, {b6})"
    );
    (r, b2, b3, b6)
}

/// Complement-key needed of `u₂` given `u₁`'s key, for a leaf pair at lde `k`.
#[inline]
pub fn complement(target_norm: i64, key: HalfKey) -> HalfKey {
    (target_norm - key.0, -key.1, -key.2, -key.3)
}

/// Per-half soundness region. Sound outer cover for all valid halves of
/// any leaf-passing pair.
#[derive(Debug, Clone)]
pub struct PerHalfRegion {
    /// Whether this region is for `u_1` (paired with `V_{11}`) or `u_2`
    /// (paired with `V_{21}`). Used only for sanity messages.
    pub side: HalfSide,
    /// `R = √(2^k)`.
    pub r: f64,
    /// `R² = 2^k`.
    pub r_sq: f64,
    /// σ₁ cap center: `R · V_i` as a complex number in (Re, Im) form.
    pub sigma1_center: [f64; 2],
    /// σ₁ cap radius squared: `2 R² ε²` (the sound outer cover from the
    /// prompt's derivation; covers the slack from `1 − √(1−ε²)`).
    pub sigma1_cap_radius_sq: f64,
    /// |σ_m(u_i)|² bound for `m ∈ {17, 13, 5}` (the three conjugate
    /// embeddings). Equal to `R²`.
    pub conj_norm_bound: f64,
    /// Box bound for brute enumeration: `|c_j| ≤ ⌈R⌉ + 1`. Derived from
    /// `|σ_m(c_j · e_j)| = |c_j|` (each ring-basis vector has unit-modulus
    /// embeddings) and the per-half bound `|σ_m(u)|² ≤ R²` ⇒ `|c_j| ≤ R`.
    /// The `+1` is integer-quantization slack; soundness is preserved
    /// because any candidate exceeding it fails the `contains` check
    /// anyway, but the box loop never visits it.
    pub box_bound: i64,
    /// `eps` value used when constructing the region (re-derivation sanity).
    pub eps: f64,
    /// `k` value used.
    pub k: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HalfSide {
    U1,
    U2,
}

impl PerHalfRegion {
    /// Build the region for half `side` with target column entry
    /// `v_i ∈ {V_{11}, V_{21}}` from the SU(2) target.
    pub fn new(side: HalfSide, v_i: Complex64, k: u32, eps: f64) -> Self {
        let r_sq = 2.0_f64.powi(k as i32);
        let r = r_sq.sqrt();
        let sigma1_center = [v_i.re * r, v_i.im * r];
        let sigma1_cap_radius_sq = 2.0 * r_sq * eps * eps;
        // |c_j · ζ^(jm)| ≤ R for each embedding m ⇒ |c_j| ≤ R.
        let box_bound = r.ceil() as i64 + 1;
        Self {
            side,
            r,
            r_sq,
            sigma1_center,
            sigma1_cap_radius_sq,
            conj_norm_bound: r_sq,
            box_bound,
            eps,
            k,
        }
    }

    /// True iff the candidate 8D `u` lies inside the sound per-half region.
    /// A small numerical slack (`1e-9`) is added on the upper bounds to
    /// avoid f64-round-off rejecting boundary integer points (the σ_m
    /// values are typically O(R) ~ O(2^(k/2)) so this slack is tiny in
    /// relative terms and dominated by the integer-coefficient quantization).
    pub fn contains(&self, u: &[i64; 8]) -> bool {
        let sigma = sigma_el();
        const SLACK_RATIO: f64 = 1e-9;
        let slack_cap = self.sigma1_cap_radius_sq * SLACK_RATIO + 1e-12;
        let slack_conj = self.conj_norm_bound * SLACK_RATIO + 1e-12;
        for m_idx in 0..4 {
            // Rows of Σ_el: row 2m_idx = Re σ_{rep(m_idx)}, 2m_idx+1 = Im.
            let mut re = 0.0_f64;
            let mut im = 0.0_f64;
            for j in 0..8 {
                let uj = u[j] as f64;
                re += sigma[2 * m_idx][j] * uj;
                im += sigma[2 * m_idx + 1][j] * uj;
            }
            let mag_sq = re * re + im * im;
            if m_idx == 0 {
                // σ_1 cap
                let dx = re - self.sigma1_center[0];
                let dy = im - self.sigma1_center[1];
                if dx * dx + dy * dy > self.sigma1_cap_radius_sq + slack_cap {
                    return false;
                }
                // Also enforce |σ_1|² ≤ R² (cap implies ≈ |V|²·R² ≤ R²
                // up to the cap slack; not strictly required for soundness
                // but cheap and reduces emit rate).
                if mag_sq > self.conj_norm_bound + slack_conj + 2.0 * self.r * self.r * self.eps {
                    return false;
                }
            } else if mag_sq > self.conj_norm_bound + slack_conj {
                return false;
            }
        }
        true
    }

    /// Read-only counter of box-iterations attempted; useful for sanity
    /// (counts the brute coefficient cube size, not the # emitted).
    pub fn box_cube_size(&self) -> u128 {
        (2 * self.box_bound as u128 + 1).pow(8)
    }
}

/// Brute box-enumerate all integer 8-tuples in the per-half region.
/// `|c_j| ≤ region.box_bound`. Filtered by `region.contains` at every leaf.
/// Returns the sorted list of valid `u` (deterministic order).
pub fn brute_enumerate_half(region: &PerHalfRegion) -> Vec<[i64; 8]> {
    let mut out: Vec<[i64; 8]> = Vec::new();
    let mut u = [0i64; 8];
    enum_recurse(0, region.box_bound, &mut u, region, &mut out);
    out.sort();
    out.dedup();
    out
}

fn enum_recurse(
    pos: usize,
    bound: i64,
    u: &mut [i64; 8],
    region: &PerHalfRegion,
    out: &mut Vec<[i64; 8]>,
) {
    if pos == 8 {
        if region.contains(u) {
            out.push(*u);
        }
        return;
    }
    for c in -bound..=bound {
        u[pos] = c;
        enum_recurse(pos + 1, bound, u, region, out);
    }
}

/// Smart 8D enumerator with **σ₁-cap pruning** at every recursion node.
///
/// Walks the coefficient cube in order `c_0, c_1, …, c_7` and maintains
/// partial sums for the embeddings `σ_m` for `m ∈ {1, 17, 13, 5}`. At
/// every node, we have a partial complex value for each embedding plus a
/// known upper bound on the magnitude any remaining coordinates can add
/// (the per-row Σ_el entries are bounded by 1 since `|ζ^j| = 1`, so the
/// extra magnitude is at most `box_bound · √(8 − pos)` summed over
/// remaining slots — we use the looser `Σ_j |c_j| ≤ box_bound·(8−pos)`).
///
/// Pruning rules (sound — only kills branches whose closest reachable
/// completion of all `σ_m` lies outside the region):
///   - σ₁: distance from partial-σ₁ to center is `d`, max-remainder is
///     `m_rem`. If `d > m_rem + cap_radius`, prune.
///   - σ_m (m ∈ {17,13,5}): partial magnitude `p`, max remainder `m_rem`.
///     If `p > R + m_rem`, no completion stays in the `R`-ball, prune.
///
/// Returns the same set as `brute_enumerate_half` (verified by the
/// `smart_enumerate_matches_brute` test); much faster for moderate `k`.
pub fn smart_enumerate_half(region: &PerHalfRegion) -> Vec<[i64; 8]> {
    let sigma = sigma_el();
    let mut out: Vec<[i64; 8]> = Vec::new();
    let mut u = [0i64; 8];
    // Per-embedding partial (Re, Im). 4 embeddings × 2 components = 8 values.
    let mut partial = [[0.0_f64; 2]; 4];
    let center = region.sigma1_center;
    let cap_radius = region.sigma1_cap_radius_sq.sqrt();
    let conj_bound = region.conj_norm_bound.sqrt();
    smart_recurse(
        0,
        region.box_bound,
        &mut u,
        &sigma,
        &mut partial,
        center,
        cap_radius,
        conj_bound,
        region,
        &mut out,
    );
    out.sort();
    out.dedup();
    out
}

#[allow(clippy::too_many_arguments)]
fn smart_recurse(
    pos: usize,
    bound: i64,
    u: &mut [i64; 8],
    sigma: &[[f64; 8]; 8],
    partial: &mut [[f64; 2]; 4],
    center: [f64; 2],
    cap_radius: f64,
    conj_bound: f64,
    region: &PerHalfRegion,
    out: &mut Vec<[i64; 8]>,
) {
    if pos == 8 {
        // Final exact integer check via the existing contains().
        if region.contains(u) {
            out.push(*u);
        }
        return;
    }

    // Remaining-slot magnitude upper bound for each embedding row.
    // |Σ_{j > pos} c_j · ζ^(jm)| ≤ Σ_{j > pos} |c_j| ≤ bound · (8 − pos − 1).
    // (Used after we account for c[pos] itself which we will set in the loop.)
    let remaining_slots = (8 - pos - 1) as f64;
    let m_rem = bound as f64 * remaining_slots;

    // Slight numerical slack to avoid f64 round-off killing borderline
    // integer candidates.
    const PRUNE_SLACK: f64 = 1e-6;

    for c in -bound..=bound {
        u[pos] = c;
        let cf = c as f64;

        // Add this coordinate's contribution to each embedding row.
        let mut new_partial = *partial;
        for m_idx in 0..4 {
            new_partial[m_idx][0] += sigma[2 * m_idx][pos] * cf;
            new_partial[m_idx][1] += sigma[2 * m_idx + 1][pos] * cf;
        }

        // ── σ₁ cap prune ──────────────────────────────────────────────
        let dx = new_partial[0][0] - center[0];
        let dy = new_partial[0][1] - center[1];
        let d = (dx * dx + dy * dy).sqrt();
        // After this c, can the remaining slots reach within cap_radius?
        if d > m_rem + cap_radius + PRUNE_SLACK {
            continue;
        }

        // ── conjugate σ_m bound prune (m ∈ {17, 13, 5}) ────────────────
        let mut prune = false;
        for m_idx in 1..4 {
            let pre = new_partial[m_idx][0];
            let pim = new_partial[m_idx][1];
            let pmag = (pre * pre + pim * pim).sqrt();
            if pmag > conj_bound + m_rem + PRUNE_SLACK {
                prune = true;
                break;
            }
        }
        if prune {
            continue;
        }

        // Recurse with updated partial state.
        let mut nxt = new_partial;
        smart_recurse(
            pos + 1,
            bound,
            u,
            sigma,
            &mut nxt,
            center,
            cap_radius,
            conj_bound,
            region,
            out,
        );
    }
}

/// Drop-in smart variant of [`brute_mitm_norm_bullet_set`]: enumerates the
/// per-half regions with the SE-style σ₁-cap pruner and joins on the exact
/// integer key.
pub fn smart_mitm_norm_bullet_set(
    target: &[[Complex64; 2]; 2],
    k: u32,
    eps: f64,
) -> Vec<[i64; 16]> {
    let v_11 = target[0][0];
    let v_21 = target[1][0];
    let r1 = PerHalfRegion::new(HalfSide::U1, v_11, k, eps);
    let r2 = PerHalfRegion::new(HalfSide::U2, v_21, k, eps);
    let pool1 = smart_enumerate_half(&r1);
    let pool2 = smart_enumerate_half(&r2);
    mitm_join(&pool1, &pool2, k)
}

/// MITM hash-join. Returns all 16-tuples `x = [u₁ | u₂]` where
/// `key(u₂) = complement(2^k, key(u₁))`. Every emitted `x` is exact
/// norm-shell + bullet-zero (by key match); alignment + reconstruction
/// must be checked by the caller.
pub fn mitm_join(u1_set: &[[i64; 8]], u2_set: &[[i64; 8]], k: u32) -> Vec<[i64; 16]> {
    let target_norm: i64 = 1i64 << k;
    let mut by_key: HashMap<HalfKey, Vec<[i64; 8]>> = HashMap::new();
    for u2 in u2_set {
        by_key.entry(key_of(u2)).or_default().push(*u2);
    }
    let mut out: Vec<[i64; 16]> = Vec::new();
    for u1 in u1_set {
        let need = complement(target_norm, key_of(u1));
        if let Some(u2s) = by_key.get(&need) {
            for u2 in u2s {
                let mut x = [0i64; 16];
                x[..8].copy_from_slice(u1);
                x[8..].copy_from_slice(u2);
                out.push(x);
            }
        }
    }
    out
}

/// Convenience: build both halves brute-style and join. Top-level entry
/// for the Part-3 soundness gates. Returns all `x` passing norm + bullets;
/// alignment + ε-distance is the caller's job (e.g. via the same
/// `best_phase` path the existing `synthesize` uses).
pub fn brute_mitm_norm_bullet_set(
    target: &[[Complex64; 2]; 2],
    k: u32,
    eps: f64,
) -> Vec<[i64; 16]> {
    let v_11 = target[0][0];
    let v_21 = target[1][0];
    let r1 = PerHalfRegion::new(HalfSide::U1, v_11, k, eps);
    let r2 = PerHalfRegion::new(HalfSide::U2, v_21, k, eps);
    let pool1 = brute_enumerate_half(&r1);
    let pool2 = brute_enumerate_half(&r2);
    mitm_join(&pool1, &pool2, k)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::U2;
    use crate::rings::ZUpsilon;

    /// Sanity: the key tuple really is additive under the join.
    #[test]
    fn key_additivity() {
        let u1: [i64; 8] = [1, 2, -1, 0, 0, 0, 0, 0];
        let u2: [i64; 8] = [1, 0, 1, 0, 0, 0, 0, 0];
        let k1 = key_of(&u1);
        let k2 = key_of(&u2);
        // Total norm + bullets of (u1, u2) computed jointly should match
        // (k1.0 + k2.0, k1.1 + k2.1, k1.2 + k2.2, k1.3 + k2.3).
        let total_norm = norm_sqr_per_element(&u1) + norm_sqr_per_element(&u2);
        assert_eq!(total_norm, k1.0 + k2.0);
        let (b2_1, b3_1, b6_1) = bullets_per_element_twice(&u1);
        let (b2_2, b3_2, b6_2) = bullets_per_element_twice(&u2);
        assert_eq!(b2_1 + b2_2, k1.1 + k2.1);
        assert_eq!(b3_1 + b3_2, k1.2 + k2.2);
        assert_eq!(b6_1 + b6_2, k1.3 + k2.3);
    }

    /// Soundness: for the Gate-A fixture H·P·H (k=2, x = [1,1,0..|1,-1,0..]),
    /// both halves must lie INSIDE their per-half region for the SAME
    /// target U used to derive the region.
    #[test]
    fn fixture_halves_inside_per_half_region_h_p_h() {
        let p: U2<ZUpsilon> = U2::p();
        let h: U2<ZUpsilon> = U2::h();
        let u: U2<ZUpsilon> = h * p * h;
        let target = u.to_float();
        let u1: [i64; 8] = [1, 1, 0, 0, 0, 0, 0, 0];
        let u2: [i64; 8] = [1, -1, 0, 0, 0, 0, 0, 0];
        // ε large enough that the cap definitely contains the exact halves.
        // The exact x lies on the leaf, so δ = 0 and any ε > 0 covers it.
        let eps = 1e-2_f64;
        let k = 2;
        let r1 = PerHalfRegion::new(HalfSide::U1, target[0][0], k, eps);
        let r2 = PerHalfRegion::new(HalfSide::U2, target[1][0], k, eps);
        assert!(r1.contains(&u1), "H·P·H: u1 not in per-half region");
        assert!(r2.contains(&u2), "H·P·H: u2 not in per-half region");
        // Key complement check too.
        let target_norm: i64 = 1 << k;
        assert_eq!(complement(target_norm, key_of(&u1)), key_of(&u2));
    }

    /// Same containment check for P·H (k=1) and H·P·S·H (k=2).
    #[test]
    fn fixture_halves_inside_per_half_region_p_h() {
        let p: U2<ZUpsilon> = U2::p();
        let h: U2<ZUpsilon> = U2::h();
        let u: U2<ZUpsilon> = p * h;
        let target = u.to_float();
        let u1: [i64; 8] = [1, 0, 0, 0, 0, 0, 0, 0];
        let u2: [i64; 8] = [0, 1, 0, 0, 0, 0, 0, 0];
        let eps = 1e-2_f64;
        let k = 1;
        let r1 = PerHalfRegion::new(HalfSide::U1, target[0][0], k, eps);
        let r2 = PerHalfRegion::new(HalfSide::U2, target[1][0], k, eps);
        assert!(r1.contains(&u1), "P·H: u1 not in region");
        assert!(r2.contains(&u2), "P·H: u2 not in region");
    }

    #[test]
    fn fixture_halves_inside_per_half_region_h_p_s_h() {
        let p: U2<ZUpsilon> = U2::p();
        let s: U2<ZUpsilon> = U2::s();
        let h: U2<ZUpsilon> = U2::h();
        let u: U2<ZUpsilon> = h * p * s * h;
        let target = u.to_float();
        let u1: [i64; 8] = [1, 0, 0, 0, 0, 0, 0, 1];
        let u2: [i64; 8] = [1, 0, 0, 0, 0, 0, 0, -1];
        let eps = 1e-2_f64;
        let k = 2;
        let r1 = PerHalfRegion::new(HalfSide::U1, target[0][0], k, eps);
        let r2 = PerHalfRegion::new(HalfSide::U2, target[1][0], k, eps);
        assert!(r1.contains(&u1), "H·P·S·H: u1 not in region");
        assert!(r2.contains(&u2), "H·P·S·H: u2 not in region");
    }

    #[test]
    fn fixture_halves_inside_per_half_region_h_p_h_p_h() {
        let p: U2<ZUpsilon> = U2::p();
        let h: U2<ZUpsilon> = U2::h();
        let u: U2<ZUpsilon> = h * p * h * p * h;
        let target = u.to_float();
        let u1: [i64; 8] = [1, 2, -1, 0, 0, 0, 0, 0];
        let u2: [i64; 8] = [1, 0, 1, 0, 0, 0, 0, 0];
        let eps = 1e-2_f64;
        let k = 3;
        let r1 = PerHalfRegion::new(HalfSide::U1, target[0][0], k, eps);
        let r2 = PerHalfRegion::new(HalfSide::U2, target[1][0], k, eps);
        assert!(r1.contains(&u1), "H·P·H·P·H: u1 not in region");
        assert!(r2.contains(&u2), "H·P·H·P·H: u2 not in region");
    }

    /// Soundness: every brute-MITM-emitted candidate at k=2 for the H·P·H
    /// fixture passes the joint norm + bullet checks.
    #[test]
    fn brute_mitm_emits_only_norm_bullet_valid() {
        use crate::synthesis::lattice_upsilon::enumerate::{bullets_zero, norm_sqr_total};
        let p: U2<ZUpsilon> = U2::p();
        let h: U2<ZUpsilon> = U2::h();
        let u: U2<ZUpsilon> = h * p * h;
        let target = u.to_float();
        let k = 2;
        let eps = 1e-2_f64;
        let cands = brute_mitm_norm_bullet_set(&target, k, eps);
        assert!(!cands.is_empty(), "H·P·H: brute MITM returned no candidates");
        let target_norm: i64 = 1 << k;
        for x in &cands {
            assert_eq!(
                norm_sqr_total(x),
                target_norm,
                "MITM candidate {x:?} has wrong norm"
            );
            assert!(
                bullets_zero(x),
                "MITM candidate {x:?} fails bullets-zero check"
            );
        }
    }

    /// Smart enumerator must emit the SAME set as brute on a feasible
    /// (small k) configuration. Drop = unsound pruner.
    #[test]
    fn smart_enumerate_matches_brute_h_p_h() {
        let p: U2<ZUpsilon> = U2::p();
        let h: U2<ZUpsilon> = U2::h();
        let u: U2<ZUpsilon> = h * p * h;
        let target = u.to_float();
        let eps = 1e-1_f64;
        let k = 2;
        let r1 = PerHalfRegion::new(HalfSide::U1, target[0][0], k, eps);
        let r2 = PerHalfRegion::new(HalfSide::U2, target[1][0], k, eps);
        let brute1 = brute_enumerate_half(&r1);
        let brute2 = brute_enumerate_half(&r2);
        let smart1 = smart_enumerate_half(&r1);
        let smart2 = smart_enumerate_half(&r2);
        assert_eq!(
            brute1, smart1,
            "smart u1 set ≠ brute u1 set (smart dropped {} valid pts)",
            brute1.len() as i64 - smart1.len() as i64
        );
        assert_eq!(
            brute2, smart2,
            "smart u2 set ≠ brute u2 set (smart dropped {} valid pts)",
            brute2.len() as i64 - smart2.len() as i64
        );
    }

    /// Same on the heavier H·P·H·P·H fixture (k=3) where both pools have
    /// a couple of elements.
    #[test]
    fn smart_enumerate_matches_brute_h_p_h_p_h() {
        let p: U2<ZUpsilon> = U2::p();
        let h: U2<ZUpsilon> = U2::h();
        let u: U2<ZUpsilon> = h * p * h * p * h;
        let target = u.to_float();
        let eps = 1e-1_f64;
        let k = 3;
        let r1 = PerHalfRegion::new(HalfSide::U1, target[0][0], k, eps);
        let r2 = PerHalfRegion::new(HalfSide::U2, target[1][0], k, eps);
        let brute1 = brute_enumerate_half(&r1);
        let brute2 = brute_enumerate_half(&r2);
        let smart1 = smart_enumerate_half(&r1);
        let smart2 = smart_enumerate_half(&r2);
        assert_eq!(brute1, smart1);
        assert_eq!(brute2, smart2);
    }

    /// Soundness: the known good (u1, u2) for H·P·H must be in the emitted set.
    #[test]
    fn brute_mitm_finds_h_p_h_fixture() {
        let p: U2<ZUpsilon> = U2::p();
        let h: U2<ZUpsilon> = U2::h();
        let u: U2<ZUpsilon> = h * p * h;
        let target = u.to_float();
        let k = 2;
        let eps = 1e-2_f64;
        let cands = brute_mitm_norm_bullet_set(&target, k, eps);
        let expected = [
            1, 1, 0, 0, 0, 0, 0, 0, //
            1, -1, 0, 0, 0, 0, 0, 0,
        ];
        assert!(
            cands.iter().any(|x| *x == expected),
            "brute MITM at k=2 did NOT emit the H·P·H fixture x16; emitted {} candidates",
            cands.len()
        );
    }
}
