//! Schnorr-Euchner CVP enumeration for the 16D Z[ζ₂₄] lattice.
//!
//! Minimal port from `lattice_zeta::se` — keeps the dim-generic SE walker
//! (`recurse_16`, `schnorr_euchner_16d`, `reconstruct_x`, `det16_exact`)
//! verbatim and replaces the n=16-specific bilinear leaf check with the
//! n=12 three-bullet check from `enumerate.rs`.
//!
//! Carried over from lattice_zeta with NO changes (all dim-generic at d=16):
//!   - `reconstruct_x` (x = B·z)
//!   - `det16_exact` (basis unimodularity check)
//!   - `recurse_16` / `schnorr_euchner_16d` (the SE walker)
//!
//! Replaced for n=12:
//!   - `bilinear_forms` (was `(β_1, β_2, β_3)` specific to Z[ζ₁₆]) →
//!     `bullet_forms` returning the n=12 (√2, √3, √6) triple summed
//!     over u₁,u₂.

#![allow(clippy::needless_range_loop)]

use std::sync::atomic::{AtomicU64, Ordering};

// ─── n=12 bullet forms (the leaf check) ──────────────────────────────────────

/// Three n=12 bullet forms summed over `(u₁, u₂)`. Returns
/// `(2·s₂, 2·s₃, 2·s₆)`; all three must equal zero for a valid lattice
/// solution (SPEC §5, derived from `u·conj(u)` ∈ Q(√2, √3)).
///
/// Uses `i128` accumulators for safety at deep ε (sum of `x_i·x_j`
/// products at large basis growth can transiently exceed i64).
pub fn bullet_forms(x: &[i64; 16]) -> (i128, i128, i128) {
    // Per-element bullets, then sum. Inline to keep i128 widening tight.
    fn per_element(c: &[i128; 8]) -> (i128, i128, i128) {
        let d0 = c[0] + c[4];
        let d1 = c[3];
        let d2 = c[2];
        let d3 = c[1];
        let d4 = -c[4];
        let d5 = -c[3] - c[7];
        let d6 = -c[2] - c[6];
        let d7 = -c[1] - c[5];
        let d = [d0, d1, d2, d3, d4, d5, d6, d7];

        let mut t = [0i128; 15];
        for i in 0..8 {
            for j in 0..8 {
                t[i + j] += c[i] * d[j];
            }
        }
        let p_b = t[1] - (t[9] + t[13]);
        let p_c = t[2] - (t[10] + t[14]);
        let p_h = t[7] + t[11];
        (2 * p_b + p_h, p_c, -p_h)
    }
    let ca: [i128; 8] = std::array::from_fn(|i| x[i] as i128);
    let cb: [i128; 8] = std::array::from_fn(|i| x[8 + i] as i128);
    let (s2a, s3a, s6a) = per_element(&ca);
    let (s2b, s3b, s6b) = per_element(&cb);
    (s2a + s2b, s3a + s3b, s6a + s6b)
}

/// True iff all three bullets vanish.
#[inline]
pub fn bullets_zero_i128(x: &[i64; 16]) -> bool {
    let (b2, b3, b6) = bullet_forms(x);
    b2 == 0 && b3 == 0 && b6 == 0
}

/// Total cyclotomic-basis norm at `i128` precision.
pub fn norm_sqr_i128(x: &[i64; 16]) -> i128 {
    let mut s: i128 = 0;
    for v in x {
        let vi = *v as i128;
        s += vi * vi;
    }
    for i in 0..4 {
        s += (x[i] as i128) * (x[i + 4] as i128);
    }
    for i in 0..4 {
        s += (x[8 + i] as i128) * (x[8 + i + 4] as i128);
    }
    s
}

// ─── x = B · z reconstruction (verbatim from lattice_zeta) ───────────────────

/// Reconstruct `x = B·z` where `B[i]` is the i-th LLL basis vector (row).
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

// ─── Bareiss det (verbatim from lattice_zeta) ────────────────────────────────

/// Exact 16×16 i64 determinant via Bareiss in i128. Returns `None` on
/// overflow. Used to validate basis unimodularity after LLL.
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

// ─── Schnorr-Euchner walker (verbatim from lattice_zeta::recurse_16) ────────

/// SE walk over integer 16-tuples within the Q-ellipsoid centered at
/// `z_c` with bound `bound_sq`. Calls `callback(z)` at each leaf; returns
/// the leaf count. If `callback` returns `false`, the walk aborts (used
/// for `max_solutions = 1` short-circuit).
///
/// `l` is the f64 lower-triangular Cholesky factor `L · Lᵀ = G_post_LLL`.
/// The Q-norm of `z − z_c` is `Σ_d (L[d][d]·(z[d]−z_c[d]) + tail[d])²`.
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

    if l_dd.abs() < 1e-30 {
        z[d] = z_c[d];
        recurse_16(
            depth - 1,
            l,
            z_c,
            bound_sq,
            partial,
            z,
            callback,
            budget,
            leaves,
            aborted,
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
        let new_partial = partial + level * level;
        if new_partial > bound_sq + 1e-9 * bound_sq.abs() {
            continue;
        }
        z[d] = zd;
        recurse_16(
            depth - 1,
            l,
            z_c,
            bound_sq,
            new_partial,
            z,
            callback,
            budget,
            leaves,
            aborted,
        );
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthesis::lattice_upsilon::enumerate::{bullets_total_twice, norm_sqr_total};

    /// `bullet_forms` (i128) matches `bullets_total_twice` (i64) on values
    /// that fit in i64.
    #[test]
    fn bullet_forms_matches_enumerate_i64() {
        let cases: [[i64; 16]; 4] = [
            [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, -1, 2, 0, 1, 1, 0, -1, 0, 1, -1, 2, 0, 0, 1, 1],
            [3, 2, -1, 0, 4, -2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [-1, 0, 1, 0, 1, 0, -1, 0, 1, 0, -1, 0, -1, 0, 1, 0],
        ];
        for x in &cases {
            let (b2_i128, b3_i128, b6_i128) = bullet_forms(x);
            let (b2_i64, b3_i64, b6_i64) = bullets_total_twice(x);
            assert_eq!(b2_i128, b2_i64 as i128);
            assert_eq!(b3_i128, b3_i64 as i128);
            assert_eq!(b6_i128, b6_i64 as i128);
        }
    }

    #[test]
    fn norm_sqr_i128_matches_i64() {
        let cases: [[i64; 16]; 3] = [
            [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, -1, 2, 0, 1, 1, 0, -1, 0, 1, -1, 2, 0, 0, 1, 1],
            [3, -2, 1, 0, 1, 0, -1, 2, -1, 1, 0, 1, -1, 0, 0, 1],
        ];
        for x in &cases {
            assert_eq!(norm_sqr_i128(x), norm_sqr_total(x) as i128);
        }
    }

    /// `det16_exact` returns `1` on the identity 16×16.
    #[test]
    fn det16_exact_identity() {
        let mut id = [[0i64; 16]; 16];
        for i in 0..16 {
            id[i][i] = 1;
        }
        assert_eq!(det16_exact(&id), Some(1));
    }

    /// `reconstruct_x` is the linear map `z ↦ B · z` (row convention).
    #[test]
    fn reconstruct_x_identity() {
        let mut b = [[0i64; 16]; 16];
        for i in 0..16 {
            b[i][i] = 1;
        }
        let z: [i64; 16] = [1, 2, 3, 0, -1, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0];
        let x = reconstruct_x(&b, &z);
        assert_eq!(x, z);
    }

    /// Trivial SE walk: identity Cholesky factor, bound 0, z_c = 0 →
    /// exactly one leaf at z = 0.
    #[test]
    fn schnorr_euchner_bound_zero_finds_origin() {
        let mut l = [[0.0f64; 16]; 16];
        for i in 0..16 {
            l[i][i] = 1.0;
        }
        let z_c = [0i64; 16];
        let mut found = Vec::new();
        let budget = AtomicU64::new(1000);
        let leaves = schnorr_euchner_16d(
            &l,
            &z_c,
            0.0,
            |z| {
                found.push(*z);
                true
            },
            &budget,
        );
        assert_eq!(leaves, 1);
        assert_eq!(found.len(), 1);
        assert!(found[0].iter().all(|&v| v == 0));
    }
}
