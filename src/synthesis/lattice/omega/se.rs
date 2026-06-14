//! Schnorr-Euchner enumeration over the 8-dimensional integer lattice, plus
//! the candidate-validation helpers that go with it.
//!
//! Inputs (produced by the L²-LLL pipeline):
//!   - The LLL-reduced basis B (`[[i64; 8]; 8]`).
//!   - The Cholesky factor R of the Q-metric Gram matrix on the LLL basis,
//!     in MPFR `RFloat` at [`SE_PREC`] = 128 bits.
//!   - The target's projection onto the lattice basis (cap center) at the
//!     same MPFR precision.
//!   - The Euclidean Cholesky of B·Bᵀ used for an additional norm-shell
//!     prune (optional).
//!
//! The walk visits each integer 8-tuple `z` whose ‖R·(z − z_c)‖² ≤ bound,
//! invoking a caller-supplied callback for each visit. The callback typically
//! reconstructs the lattice point `x = B·z`, validates it against the
//! synthesis constraints (norm shell, bilinear form, alignment cap), and
//! returns the first candidate that passes.

// 8×8 matrix code reads more clearly with explicit (i, j) indexing than with
// iterator combinators that thread multiple arrays in lockstep.
#![allow(clippy::needless_range_loop)]

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use rug::{Assign, Float as RFloat};


type IMat8 = [[i64; 8]; 8];

/// MPFR precision used by SE. 128 bits gives enough margin for SE's
/// 10⁻⁹ bound-check tolerance at all supported ε; f64-only SE breaks at
/// ε ≤ 1e-5 from squared-norm cancellation noise.
pub const SE_PREC: u32 = 128;

/// Convert an arbitrary-precision `RFloat` (built at scratch.prec_q for
/// post-LLL Cholesky) to the SE working precision (128 bits). Single
/// allocation, single MPFR conversion.
pub fn rfloat_to_se(r: &RFloat) -> RFloat {
    RFloat::with_val(SE_PREC, r)
}

// ─── 8D Schnorr-Euchner enumeration ──────────────────────────────────────────

/// Enumerate integer 8-tuples z ∈ ℤ⁸ satisfying ‖R·(z − z_c)‖² ≤ bound, in
/// distance-from-center order, invoking `callback(&z)` at each leaf. Returns
/// the first non-`None` callback result, or `None` if the search exhausts.
///
/// All distance arithmetic uses MPFR `RFloat` at 128-bit precision — the
/// f64-only version was insufficient at extreme ε (Cholesky-diagonal
/// ratios > 10¹⁰ caused "ghost-node" SE blowup from squared-norm noise).
///
/// `r_chol_eucl` is an optional Euclidean-Cholesky factor for an additional
/// norm-shell prune; pass `None` to disable it. With it, branches whose
/// partial Euclidean norm already exceeds `target_norm_eucl` are cut.
///
/// `abort` is checked at every recursion entry — when set, the enumeration
/// returns `None` immediately. It is read-only here: the caller (or a peer
/// branch under cross-branch racing) sets it.
///
/// `node_budget` is a TRUE node budget: decremented once per recurse-entry
/// (interior nodes AND leaves — the 16D walker's semantics). When it runs
/// out, `budget_exhausted` is set and the walk unwinds. This is the fix for
/// the "empty level walks unbudgeted to region exhaustion" failure mode:
/// the leaf-callback budget never binds on a no-solution level because
/// almost nothing reaches a leaf. The walk is single-threaded, so a plain
/// decrementing atomic (no chunked reservation à la 16D `BudgetCache`) is
/// contention-free; the per-entry `fetch_sub` is noise against the ~10 MPFR
/// allocations each recurse-entry already performs.
///
/// `budget_exhausted` may also be set from inside `callback` (the leaf-cap
/// path) to abort the walk without reporting a solution.
#[allow(clippy::too_many_arguments)]
pub fn schnorr_euchner_8d<F>(
    r_chol: &[[RFloat; 8]; 8],
    z_c: &[RFloat; 8],
    bound: &RFloat,
    r_chol_eucl: Option<&[[f64; 8]; 8]>,
    target_norm_eucl: f64,
    abort: &AtomicBool,
    node_budget: &AtomicU64,
    budget_exhausted: &AtomicBool,
    mut callback: F,
) -> Option<[i64; 8]>
where
    F: FnMut(&[i64; 8]) -> Option<[i64; 8]>,
{
    let mut z = [0i64; 8];
    let result = std::cell::RefCell::new(None);
    let zero = RFloat::with_val(SE_PREC, 0.0_f64);

    recurse_8(
        7,
        r_chol,
        z_c,
        bound,
        r_chol_eucl,
        target_norm_eucl,
        0.0,
        &mut z,
        &zero,
        abort,
        node_budget,
        budget_exhausted,
        &mut callback,
        &result,
    );
    result.into_inner()
}

#[allow(clippy::too_many_arguments)]
fn recurse_8<F>(
    depth: i32,
    r_chol: &[[RFloat; 8]; 8],
    z_c: &[RFloat; 8],
    bound: &RFloat,
    r_chol_eucl: Option<&[[f64; 8]; 8]>,
    target_norm_eucl: f64,
    partial_eucl: f64,
    z: &mut [i64; 8],
    partial: &RFloat,
    abort: &AtomicBool,
    node_budget: &AtomicU64,
    budget_exhausted: &AtomicBool,
    callback: &mut F,
    result: &std::cell::RefCell<Option<[i64; 8]>>,
) where
    F: FnMut(&[i64; 8]) -> Option<[i64; 8]>,
{
    if result.borrow().is_some()
        || abort.load(Ordering::Relaxed)
        || budget_exhausted.load(Ordering::Relaxed)
    {
        return;
    }
    // True node budget: one unit per recurse-entry (interior + leaf), the
    // 16D walker's accounting. On exhaustion, flag and unwind — the flag
    // (not the wrapped counter) is what stops the remaining stack levels.
    if node_budget.fetch_sub(1, Ordering::Relaxed) <= 1 {
        budget_exhausted.store(true, Ordering::Relaxed);
        return;
    }
    if depth < 0 {
        if let Some(r) = callback(z) {
            *result.borrow_mut() = Some(r);
        }
        return;
    }
    let d = depth as usize;
    let r_dd = &r_chol[d][d];

    // Per-call scratch pre-allocated once, reused inside the inner loop via
    // assign() patterns. ~10 allocations per recurse call instead of per
    // inner iteration.
    let mut tail = RFloat::with_val(SE_PREC, 0.0_f64);
    let mut tmp = RFloat::with_val(SE_PREC, 0.0_f64);
    let mut diff = RFloat::with_val(SE_PREC, 0.0_f64);
    let mut prod = RFloat::with_val(SE_PREC, 0.0_f64);
    let mut zd_rf = RFloat::with_val(SE_PREC, 0.0_f64);
    let mut level = RFloat::with_val(SE_PREC, 0.0_f64);
    let mut level_sq = RFloat::with_val(SE_PREC, 0.0_f64);
    let mut new_partial = RFloat::with_val(SE_PREC, 0.0_f64);

    // Structural guard against a degenerate diagonal (r_chol PD-ness should
    // exclude this, but tolerate it gracefully).
    tmp.assign(r_dd.clone().abs());
    if tmp.to_f64() < 1e-30 {
        z[d] = z_c[d]
            .clone()
            .round()
            .to_integer()
            .and_then(|n| n.to_i64())
            .unwrap_or(0);
        recurse_8(
            depth - 1, r_chol, z_c, bound, r_chol_eucl, target_norm_eucl,
            partial_eucl, z, partial, abort, node_budget, budget_exhausted,
            callback, result,
        );
        return;
    }

    // tail = Σ_{j > d} R[d][j] · (z[j] − z_c[j])
    for j in (d + 1)..8 {
        // Exact i64 → MPFR lift. `z[j] as f64` loses low bits once
        // |z| > 2^53 — at deep ε the lattice coordinates reach ~1.6e16
        // (ε=1e-8, lde_inner=34) in Euclid-pathological frames, and a ±2-ulp
        // error here times R[d][j] is an O(1) error in `level` against an
        // O(1) span.
        diff.assign(z[j]);
        diff -= &z_c[j];
        prod.assign(&r_chol[d][j] * &diff);
        tail += &prod;
    }

    tmp.assign(bound - partial);
    if tmp.to_f64() < 0.0 {
        return;
    }
    let rem_sqrt_f = tmp.to_f64().sqrt();

    // Iteration bounds. The CENTER must be computed and rounded in MPFR:
    // with |z| beyond f64's exact-integer range the old f64 center
    // (`z_c[d].to_f64() − tail/r_dd`) was off by ±2 ulps ≈ ±4 units while
    // the per-level span is O(1), so the branch holding a TRUE solution
    // could fall outside [z_low, z_high] — observed live at ε=1e-8
    // (docs/w_8d_rework_notes.md; frame-dependent FOUND→none flips that
    // build_ma_prefix_set's coset-mate redundancy used to mask). The span itself is
    // O(1) and stays f64.
    let r_dd_f = r_dd.to_f64();
    let span = rem_sqrt_f / r_dd_f.abs();
    let center = {
        let mut c = RFloat::with_val(SE_PREC, &tail / r_dd);
        c = RFloat::with_val(SE_PREC, &z_c[d] - &c);
        c
    };
    let to_i64 = |v: RFloat| -> Option<i64> { v.to_integer().and_then(|n| n.to_i64()) };
    let (Some(z_low), Some(z_high), Some(z_mid)) = (
        to_i64(RFloat::with_val(SE_PREC, &center - span).ceil()),
        to_i64(RFloat::with_val(SE_PREC, &center + span).floor()),
        to_i64(center.clone().round()),
    ) else {
        return;
    };
    let max_off = (z_high - z_mid).max(z_mid - z_low).max(0);

    let tail_eucl = if let Some(re) = r_chol_eucl {
        let mut t = 0.0_f64;
        for j in (d + 1)..8 {
            t += re[d][j] * (z[j] as f64);
        }
        t
    } else {
        0.0
    };

    // Iterate offsets in distance-from-center order: 0, +1, -1, +2, -2, ...
    for raw in 0..=(2 * max_off + 1) {
        if result.borrow().is_some()
            || abort.load(Ordering::Relaxed)
            || budget_exhausted.load(Ordering::Relaxed)
        {
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

        // level = r_dd·(zd − z_c[d]) + tail; exact i64 lift (see tail loop).
        zd_rf.assign(zd);
        diff.assign(&zd_rf - &z_c[d]);
        level.assign(r_dd * &diff);
        level += &tail;
        level_sq.assign(&level * &level);
        new_partial.assign(partial + &level_sq);
        tmp.assign(&new_partial - bound);
        if tmp.to_f64() > 1e-9 {
            continue;
        }

        let new_partial_eucl = if let Some(re) = r_chol_eucl {
            let level_eucl = re[d][d] * (zd as f64) + tail_eucl;
            let p = partial_eucl + level_eucl * level_eucl;
            // Relative slack: target_norm_eucl = 2^k, so the bare `+ 1.0`
            // vanishes once k ≥ 53 and would cut a true solution whose
            // f64-accumulated `p` sits a few ULP above 2^k. 1e-9 relative
            // dwarfs the ~2^-49 accumulation error; the exact leaf filter
            // arbitrates the over-retained nodes.
            if p > target_norm_eucl * (1.0 + 1e-9) + 1.0 {
                continue;
            }
            p
        } else {
            partial_eucl
        };

        z[d] = zd;
        recurse_8(
            depth - 1, r_chol, z_c, bound, r_chol_eucl, target_norm_eucl,
            new_partial_eucl, z, &new_partial, abort, node_budget,
            budget_exhausted, callback, result,
        );
    }
}

// ─── Lattice-point reconstruction + bilinear-form check ──────────────────────

/// Reconstruct the lattice point `x = B·z` where `B` is the LLL-reduced
/// basis (rows are basis vectors) and `z` are the SE-output coordinates.
///
/// The FINAL components fit i64 (Theorem 2's L³-reduced-basis bound plus
/// the SE bound), but in Euclid-pathological frames (basis entries ~2^33,
/// `z` ~ 1e10 — see `euclidean_cholesky`) the INTERMEDIATE products and
/// sums can exceed i64. Two's-complement wrapping arithmetic is exact mod
/// 2^64 and the true value fits, so explicit wrapping ops give the correct
/// result in every build profile (plain `+`/`*` would panic in debug).
#[inline]
pub fn reconstruct_x(b_lll: &IMat8, z: &[i64; 8]) -> [i64; 8] {
    let mut x = [0i64; 8];
    for i in 0..8 {
        for j in 0..8 {
            x[j] = x[j].wrapping_add(z[i].wrapping_mul(b_lll[i][j]));
        }
    }
    x
}

/// Evaluate the bilinear form `B(x) = a₁b₁ − a₁d₁ + b₁c₁ + c₁d₁ + a₂b₂ −
/// a₂d₂ + b₂c₂ + c₂d₂` where `x = (a₁,b₁,c₁,d₁,a₂,b₂,c₂,d₂)`. This is the
/// unitarity constraint from arXiv:2510.05816 eq (3.10): a candidate is a
/// valid (u₁, u₂) pair iff `B(x) = 0`.
///
/// Returns `i128` to avoid silent overflow at deep ε where x_i can reach
/// ~2^41 and pairwise products hit ~2^82.
#[inline]
pub fn bilinear_b(x: &[i64; 8]) -> i128 {
    let (a1, b1, c1, d1) = (x[0] as i128, x[1] as i128, x[2] as i128, x[3] as i128);
    let (a2, b2, c2, d2) = (x[4] as i128, x[5] as i128, x[6] as i128, x[7] as i128);
    a1 * b1 - a1 * d1 + b1 * c1 + c1 * d1 + a2 * b2 - a2 * d2 + b2 * c2 + c2 * d2
}
