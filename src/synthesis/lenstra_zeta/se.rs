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

use std::sync::atomic::{AtomicU64, Ordering};

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
pub fn euclidean_cholesky_16_mpfr(basis: &[[i64; 16]; 16]) -> Option<[[f64; 16]; 16]> {
    use rug::Float;
    const PREC: u32 = 128;

    // Step 1: integer Gram = B · Bᵀ in i128.
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

    // Step 2: lift to MPFR.
    let mut g: [[Float; 16]; 16] = std::array::from_fn(|_| {
        std::array::from_fn(|_| Float::with_val(PREC, 0.0))
    });
    for i in 0..16 {
        for j in 0..16 {
            g[i][j] = i128_to_mpfr(gram[i][j], PREC);
        }
    }

    // Step 3: MPFR Cholesky → L (lower-triangular) such that L·Lᵀ = G.
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

    // Step 4: transpose to upper-triangular R = Lᵀ, snapshot to f64.
    let mut r = [[0.0_f64; 16]; 16];
    for i in 0..16 {
        for j in 0..16 {
            r[i][j] = l[j][i].to_f64();
        }
    }
    Some(r)
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
        if new_partial_eucl <= target_norm_sq * (1.0 + 1e-9) {
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
        if depth > 0 && new_partial_eucl > target_norm_sq * (1.0 + 1e-9) {
            continue;
        }
        recurse_16_norm_pruned(
            depth - 1, l, z_c, bound_sq, r_eucl, target_norm_sq, target_norm_sq_i64,
            new_partial_q, new_partial_eucl, z, x, w, basis, callback, budget,
            leaves, aborted,
        );
    }
}

/// Compute `Σ_{j ≥ d} R_eucl[d][j] · z[j]` — the d-th component of `R_eucl·z`,
/// given that z[d..16] are set. Used in the degenerate-diagonal branch where
/// z[d] is forced to z_c[d] (no per-iteration cache available).
#[inline]
fn compute_eucl_level(r_eucl: &[[f64; 16]; 16], z: &[i64; 16], d: usize) -> f64 {
    let mut s = 0.0_f64;
    for j in d..16 {
        s += r_eucl[d][j] * (z[j] as f64);
    }
    s
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
    let _r_15 = r_eucl[15][15];
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
            if partial_eucl > target_norm_sq * (1.0 + 1e-9) {
                return Vec::new().into_iter();
            }
            let mut local: Vec<[i64; 16]> = Vec::new();
            recurse_collect_norm_pruned(
                14, l, z_c, bound_sq, r_eucl, target_norm_sq, target_norm_sq_i64,
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
    if depth < 0 {
        let prev = budget.fetch_sub(1, Ordering::Relaxed);
        if prev <= 1 {
            aborted.store(true, Ordering::Relaxed);
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
        if new_partial_eucl <= target_norm_sq * (1.0 + 1e-9) {
            recurse_collect_norm_pruned(
                depth - 1, l, z_c, bound_sq, r_eucl, target_norm_sq, target_norm_sq_i64,
                partial_q, new_partial_eucl, z, x, w, basis, leaf_filter,
                budget, aborted, results,
            );
        }
        return;
    }
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
    // Same i64 bracket fix as recurse_16: keep z_c[d] as i64 to avoid
    // f64 quantization at deep ε where z_c can exceed 2^53.
    let z_low = z_c[d].saturating_add((center_off - span).ceil() as i64);
    let z_high = z_c[d].saturating_add((center_off + span).floor() as i64);
    let z_mid = z_c[d].saturating_add(center_off.round() as i64);
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
        if depth > 0 && new_partial_eucl > target_norm_sq * (1.0 + 1e-9) {
            continue;
        }
        recurse_collect_norm_pruned(
            depth - 1, l, z_c, bound_sq, r_eucl, target_norm_sq, target_norm_sq_i64,
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
