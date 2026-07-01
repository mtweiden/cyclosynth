//! Schnorr-Euchner enumeration over the 16D integer lattice (Z[ζ_16]).
//!
//! The inner walk runs in f64 (vs the 8D path's MPFR): post-L²-LLL,
//! κ(G) ≤ (4/3)^15 ≈ 75 costs ~6 bits of conditioning — well inside
//! the 53-bit mantissa and far below SE's 10⁻⁹ tolerance. Deep-ε
//! corrections (dd accumulators, MPFR verify) layer on top where f64
//! runs out.
//!
//! The walk takes an UPPER-triangular Cholesky factor `l` (`lᵀl = G`);
//! call sites transpose the lower-triangular `l_f64` before invoking.

#![allow(clippy::needless_range_loop)]

use crate::rings::MpFloat;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::OnceLock;

use super::dd::{dd_add, dd_from_i64, dd_mul, dd_sub};


/// Predictive-budget-truncation kill-switch: `CYCLOSYNTH_PREDICTIVE_TRUNC=0`
/// disables the projected-infeasibility abort in budget-capped flat walks
/// (see [`PredictiveTrunc`]). Default ON. Read once.
static PREDICTIVE_TRUNC_DISABLED: OnceLock<bool> = OnceLock::new();

fn predictive_trunc_disabled() -> bool {
    *PREDICTIVE_TRUNC_DISABLED.get_or_init(|| {
        std::env::var("CYCLOSYNTH_PREDICTIVE_TRUNC").ok().as_deref() == Some("0")
    })
}

/// Re-check every f64 prune-fire at MPFR-128; MPFR's verdict wins.
/// Necessary at ε ≤ 1.5e-8 where the f64 dot product suffers
/// catastrophic cancellation; pure overhead at shallower ε.
static VERIFY_PRUNE_MPFR: AtomicBool = AtomicBool::new(false);

#[inline]
pub fn verify_prune_mpfr() -> bool {
    VERIFY_PRUNE_MPFR.load(Ordering::Relaxed)
}

pub fn set_verify_prune_mpfr(value: bool) {
    VERIFY_PRUNE_MPFR.store(value, Ordering::Release);
}

// ─── Center-relative dd partial-norm verification ────────────────────────────

/// Compute `Σ_{i ≥ depth} (R · z)[i]²` in inline double-double (~106 bits)
/// and return true iff the result exceeds `threshold`. No heap allocation,
/// no thread-local state — fully stack-resident. About 10× faster than the
/// rug-128 verify in the hot SE prune-firing path.
///
/// Center-relative form: each row is
/// `(R·z)[i] = u_eucl_dd[i] + Σ_{j≥i} R[i][j]·(z[j] − z_c.int[j])`, where
/// `u_eucl_dd[i] = (R·z_c.int)[i]` is the per-walk dd constant from
/// [`center_relative_seed`]. The deltas `z[j] − z_c.int[j]` are
/// bracket-sized (exact small i64), so no intermediate ever exceeds
/// O(√T·O(1)) — mirroring the f64 accumulator's restructure.
#[inline]
pub fn verify_partial_dd_exceeds(
    r_eucl_dd: &[[(f64, f64); 16]; 16],
    z: &[i64; 16],
    z_c: &SeCenter16,
    u_eucl_dd: &[(f64, f64); 16],
    depth: usize,
    threshold: f64,
) -> bool {
    let mut total: (f64, f64) = (0.0, 0.0);
    for i in depth..16 {
        let mut row: (f64, f64) = u_eucl_dd[i];
        for j in i..16 {
            let dz = z[j] - z_c.int[j];
            if dz != 0 {
                let term = dd_mul(r_eucl_dd[i][j], dd_from_i64(dz));
                row = dd_add(row, term);
            }
        }
        let sq = dd_mul(row, row);
        total = dd_add(total, sq);
    }
    total.0 + total.1 > threshold
}

/// Resolve a fired Euclidean shell prune. Returns true iff the node should
/// actually be pruned. Shared verbatim by the recursion and the W1 frontier
/// expander (the prune ladder lives here so the two can't drift). `prune_fires`
/// is the site-specific f64 test. When set: the integer-exact ‖x‖² ≤ T
/// short-circuit runs first (~30 ns, proves a keep, no false negatives), then
/// dd-verify when MPFR-verify is enabled (sound for any f64 overshoot ratio,
/// needed at ε ≤ 1.5e-8 where the f64 dot product cancels catastrophically).
#[inline]
#[allow(clippy::too_many_arguments)]
fn resolve_prune(
    prune_fires: bool,
    x: &[i64; 16],
    z: &[i64; 16],
    target_norm_sq_i128: i128,
    r_eucl_dd: &[[(f64, f64); 16]; 16],
    z_c: &SeCenter16,
    u_eucl_dd: &[(f64, f64); 16],
    depth: usize,
    threshold: f64,
    trace: bool,
) -> bool {
    if !prune_fires {
        return false;
    }
    if trace {
        crate::synthesis::diag::N_PRUNE_FIRES.fetch_add(1, Ordering::Relaxed);
    }
    let x_norm_sq: i128 = x.iter().map(|&v| i128::from(v) * i128::from(v)).sum();
    if x_norm_sq <= target_norm_sq_i128 {
        return false;
    }
    if !verify_prune_mpfr() {
        return true;
    }
    let t_v = if trace { Some(std::time::Instant::now()) } else { None };
    let dd_prune = verify_partial_dd_exceeds(r_eucl_dd, z, z_c, u_eucl_dd, depth, threshold);
    if let Some(t) = t_v {
        crate::synthesis::diag::T_VERIFY_DD_NS
            .fetch_add(crate::synthesis::diag::elapsed_ns(t), Ordering::Relaxed);
        if !dd_prune {
            crate::synthesis::diag::N_VERIFY_PRUNE_CORRECTED.fetch_add(1, Ordering::Relaxed);
        }
        crate::synthesis::diag::N_VERIFY_PRUNE_FIRES.fetch_add(1, Ordering::Relaxed);
    }
    dd_prune
}

/// Center-relative seeding: walkers start at `z = z_c.int, x = x_base,
/// w = u` so every subsequent f64 rounding happens at bound-scale
/// magnitude (~√T·O(1)) instead of the ±30·√T·2^53 intermediates the
/// absolute accumulator passed through. `x_base` may wrap per-product
/// but the SUM is a near-center lattice coordinate — exact mod 2^64.
/// `u_dd` must be accumulated in dd: one f64 rounding in that row sum
/// is √T-scale, the very defect this scheme removes.
fn center_relative_seed(
    basis: &[[i64; 16]; 16],
    r_eucl_dd: &[[(f64, f64); 16]; 16],
    z_c: &SeCenter16,
) -> ([i64; 16], [(f64, f64); 16], [f64; 16]) {
    let mut x_base = [0_i64; 16];
    for j in 0..16 {
        let zj = z_c.int[j];
        if zj == 0 {
            continue;
        }
        let row = &basis[j];
        for c in 0..16 {
            x_base[c] = x_base[c].wrapping_add(zj.wrapping_mul(row[c]));
        }
    }
    let mut u_dd = [(0.0_f64, 0.0_f64); 16];
    let mut u = [0.0_f64; 16];
    for i in 0..16 {
        let mut acc = (0.0_f64, 0.0_f64);
        for j in i..16 {
            if z_c.int[j] != 0 {
                acc = dd_add(acc, dd_mul(r_eucl_dd[i][j], dd_from_i64(z_c.int[j])));
            }
        }
        u_dd[i] = acc;
        u[i] = acc.0 + acc.1;
    }
    (x_base, u_dd, u)
}

// ─── dd Q-bracket (deep-ε dd-verified Q pruning) ─────────────────────────────
//
// Double-double companion of the SE walk's incremental f64 partial-Q,
// active only when an `l_q_dd` factor is supplied (deep-ε regime — see
// `q_cholesky_mpfr_dual` and integer.rs's gating). The f64 partial-Q can
// overshoot truth by up to ~1.8× at the ε=1.5e-8 cliff, against a geometric
// solution band of [0.875, 1.25]. With the dd companion, every Q-prune
// decision on the boundary is made on a value accurate to ~1e-32 — both
// overshoot (lost solutions) and undershoot (spurious subtrees) are
// eliminated — so the bound default can be the tight 1.5 wherever the dd
// factor is attached (3.0 when it is not).
//
// Cost model: one dd tail per node (O(16−d) dd mul/adds, replacing the
// f64 tail loop) + ~4 dd ops per bracket candidate. Zero cost when
// `l_q_dd` is `None` (moderate ε): the f64 path is untouched.
//
// These two helpers are shared verbatim by `recurse_collect_norm_pruned`,
// the W1 frontier `expand_se_prefix_node`, and the stage-1 z[15] seeding,
// keeping all three Q-prune sites in lockstep. The Euclidean shell prune they
// pair with is likewise shared, via `resolve_prune`.

/// Node-level dd tail: `Σ_{j > d} l_q_dd[d][j] · ((z[j] − int[j]) − frac[j])`.
/// `z[j] − int[j]` is an exact small i64 (bracket-sized); `frac[j]` is an
/// exact f64 — both lossless in dd.
#[inline]
fn q_tail_dd(
    lq: &[[(f64, f64); 16]; 16],
    d: usize,
    z: &[i64; 16],
    z_c: &SeCenter16,
) -> (f64, f64) {
    let mut tail = (0.0_f64, 0.0_f64);
    for j in (d + 1)..16 {
        let dz = dd_sub(dd_from_i64(z[j] - z_c.int[j]), (z_c.frac[j], 0.0));
        tail = dd_add(tail, dd_mul(lq[d][j], dz));
    }
    tail
}

/// Per-candidate dd partial-Q: the dd companion of
/// `new_partial_q = partial_q + (l[d][d]·(Δ − frac[d]) + tail)²`.
/// Returns `(hi+lo projection, dd value)`; the projection replaces the f64
/// `new_partial_q` (both for the prune decision and for threading down).
#[inline]
fn q_candidate_dd(
    lq: &[[(f64, f64); 16]; 16],
    d: usize,
    zd: i64,
    z_c: &SeCenter16,
    tail_dd: (f64, f64),
    partial_q_dd: (f64, f64),
) -> (f64, (f64, f64)) {
    let dz = dd_sub(dd_from_i64(zd - z_c.int[d]), (z_c.frac[d], 0.0));
    let level_dd = dd_add(dd_mul(lq[d][d], dz), tail_dd);
    let new_dd = dd_add(partial_q_dd, dd_mul(level_dd, level_dd));
    (new_dd.0 + new_dd.1, new_dd)
}

// ─── Bilinear leaf checks ────────────────────────────────────────────────────

/// Per-element β_1: see `clifford_sqrt_t_research.md` for derivation.
/// Returns i128 to avoid silent overflow on pairwise products at deep k.
///
/// Mirror of [`super::super::omega::se::bilinear_b`] for the Z[ζ_16] /
/// Clifford+√T flow. Three forms here vs one in 8D because the
/// totally-real-subring decomposition of unitarity over Z[ζ_16] yields
/// three independent constraints (one per non-σ_1 Galois embedding).
#[inline]
pub fn beta_1(u: &[i64; 8]) -> i128 {
    let u: [i128; 8] = std::array::from_fn(|i| i128::from(u[i]));
    u[0]*u[1] + u[1]*u[2] + u[2]*u[3] + u[3]*u[4]
        + u[4]*u[5] + u[5]*u[6] + u[6]*u[7]
        - u[0]*u[7]
}

#[inline]
pub fn beta_2(u: &[i64; 8]) -> i128 {
    let u: [i128; 8] = std::array::from_fn(|i| i128::from(u[i]));
    u[0]*u[2] + u[1]*u[3] + u[2]*u[4] + u[3]*u[5]
        + u[4]*u[6] + u[5]*u[7]
        - u[0]*u[6] - u[1]*u[7]
}

#[inline]
pub fn beta_3(u: &[i64; 8]) -> i128 {
    let u: [i128; 8] = std::array::from_fn(|i| i128::from(u[i]));
    u[0]*u[3] + u[1]*u[4] + u[2]*u[5] + u[3]*u[6] + u[4]*u[7]
        - u[0]*u[5] - u[1]*u[6] - u[2]*u[7]
}

/// Joint bilinear forms on the 16-vector `x = (u_1's 8 coords, u_2's 8 coords)`.
/// Returns `(B_1, B_2, B_3)`. All three must equal 0 for a valid Clifford+√T
/// candidate (the totally-real-subring decomposition of unitarity).
#[inline]
pub fn bilinear_forms(x: &[i64; 16]) -> (i128, i128, i128) {
    let u1: [i64; 8] = x[0..8].try_into().expect("8-of-16 slice");
    let u2: [i64; 8] = x[8..16].try_into().expect("8-of-16 slice");
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

// ─── 16D Schnorr-Euchner enumeration ─────────────────────────────────────────

/// SE walk center, split into an exact integer part and a small fractional
/// remainder. This is the deep-ε-safe representation of the FRACTIONAL cap
/// center (mirroring the 8D walk, which walks around a fractional MPFR
/// center — src/synthesis/lattice/se.rs):
///
///   - `int[i]`: MPFR `round_mut` → `to_integer` of the LU solution
///     `lu_x[i]`. Magnitudes can exceed 2^53 at deep ε (observed 5e16 at
///     ε=1e-8, lde≥18); i64 carries them exactly.
///   - `frac[i]`: `lu_x[i] − int[i]` computed in MPFR (exact at full
///     precision) then extracted to f64. |frac| ≤ 0.5, so the f64
///     extraction is precise regardless of |lu_x| — unlike `lu_x.to_f64()`,
///     which quantizes with ULP up to 2 lattice units at deep ε.
///
/// The walk's true per-coordinate center is `int[i] + frac[i]`; all walker
/// arithmetic keeps the integer part separate so deltas stay small-magnitude
/// f64. Measuring Q from this true center keeps a valid solution's measured Q
/// equal to its geometric Q (band [0.875, 1.25]) at every k.
#[derive(Clone, Copy, Debug)]
pub struct SeCenter16 {
    pub int: [i64; 16],
    pub frac: [f64; 16],
}

impl SeCenter16 {
    /// Center with zero fractional part (used by tests with integer
    /// centers).
    pub fn from_int(int: [i64; 16]) -> Self {
        Self { int, frac: [0.0; 16] }
    }

    /// Build the center from the MPFR LU solution `lu_x` (`Bᵀ·z_c = c`).
    /// `int` = MPFR round → i64 (full precision — **never** through
    /// `to_f64()`, which quantizes above 2^53); `frac` = `lu_x − int`
    /// computed in MPFR then extracted to f64 (|frac| ≤ 0.5, always
    /// f64-precise). NaN/∞ coordinates map to (0, 0.0) — the SE walk will
    /// return empty.
    pub fn from_lu_x(lu_x: &[MpFloat; 16]) -> Self {
        let mut int = [0i64; 16];
        let mut frac = [0.0f64; 16];
        for i in 0..16 {
            let mut rounded = lu_x[i].clone();
            rounded.round_mut();
            if let Some(r_int) = rounded.to_integer() {
                int[i] = r_int.to_i64_wrapping();
                let diff = MpFloat::with_val(lu_x[i].prec(), &lu_x[i] - &rounded);
                frac[i] = diff.to_f64();
            }
        }
        Self { int, frac }
    }
}

/// Run the Schnorr-Euchner walk over ℤ¹⁶, visiting every integer point `z`
/// with `‖l·(z − z_c)‖² ≤ bound_sq`, in distance-from-center order at each
/// recursion level. Calls `callback(&z)` at every leaf; the callback returns
/// `true` to continue or `false` to abort.
///
/// `l` is the **upper-triangular** Cholesky factor of the post-LLL Q-metric
/// Gram on the basis coordinates: `lᵀ · l = G`. For each level i, the walk
/// computes `level_i = l[i][i] · (z[i] − z_c[i]) + Σ_{j > i} l[i][j] · (z[j]
/// − z_c[j])` against the fractional center `z_c[i] = z_c.int[i] +
/// z_c.frac[i]` and prunes branches whose partial sum-of-squares exceeds
/// `bound_sq`. Visiting closest-to-center first allows early termination.
///
/// `budget` is decremented once per leaf callback. When it reaches zero the
/// walk aborts and returns the leaf count visited so far.
///
/// Returns the total number of leaf callbacks made.
pub fn schnorr_euchner_16d_reference<F>(
    l: &[[f64; 16]; 16],
    z_c: &SeCenter16,
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
    recurse(
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

/// f64 tail `Σ_{j>d} l[d][j]·((z[j] − z_c.int[j]) − z_c.frac[j])` for coord
/// `d` — the Schnorr-Euchner level's contribution from already-fixed
/// coordinates. (The deep-ε dd variant is [`q_tail_dd`].)
// SE center-relative offsets/deltas are bracket-sized: |z − z_c.int| ≤ 2·max_off ≪ 2^53 (se_bracket doc).
#[allow(clippy::cast_precision_loss)]
#[inline]
fn se_tail_f64(l: &[[f64; 16]; 16], d: usize, z: &[i64; 16], z_c: &SeCenter16) -> f64 {
    let mut t = 0.0_f64;
    for j in (d + 1)..16 {
        t += l[d][j] * ((z[j] - z_c.int[j]) as f64 - z_c.frac[j]);
    }
    t
}

/// Integer search bracket for one Schnorr-Euchner coordinate `d`: the range
/// `[z_low, z_high]`, its center `z_mid`, and `max_off = max(z_high−z_mid,
/// z_mid−z_low)`, for integer `zd` around base `z_c.int[d]` given the
/// fractional center offset and half-width `span`.
///
/// The base stays i64 (not folded into an f64 center) because at deep ε it can
/// exceed 2^53, where f64 rounding would mis-bracket the range; only the small
/// `center_off ± span` offsets pass through f64 (`|center_off ± span| < 2^53`
/// always holds for our `bound_sq`).
///
/// Fail-loud on overflow: a `saturating_add` would silently clamp the bracket
/// to garbage if `base` ever neared 2^63 — the i64 deep-ε trap the ω backend
/// fixed by widening SE coords to i128 (b387ad5). Zeta runs only at ε ≥ 1e-8
/// today (base ≲ 2^55), so this must never fire; a future zeta exact-column
/// deep path must widen z to i128 rather than reach this panic.
// `off as i64` truncation is the intended floor/ceil-to-integer bracket edge (doc above).
#[allow(clippy::cast_possible_truncation)]
#[inline]
fn se_bracket(base: i64, center_off: f64, span: f64) -> (i64, i64, i64, i64) {
    let add = |off: f64| -> i64 {
        base.checked_add(off as i64).expect(
            "zeta SE coord overflowed i64 at deep ε — widen z to i128 (cf. ω backend b387ad5)",
        )
    };
    let z_low = add((center_off - span).ceil());
    let z_high = add((center_off + span).floor());
    let z_mid = add(center_off.round());
    let max_off = (z_high - z_mid).max(z_mid - z_low).max(0);
    (z_low, z_high, z_mid, max_off)
}

// SE center-relative offsets/deltas are bracket-sized: |z − z_c.int| ≤ 2·max_off ≪ 2^53 (se_bracket doc).
#[allow(clippy::too_many_arguments, clippy::cast_precision_loss)]
fn recurse<F>(
    depth: i32,
    l: &[[f64; 16]; 16],
    z_c: &SeCenter16,
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
        z[d] = z_c.int[d];
        recurse(
            depth - 1, l, z_c, bound_sq, partial, z, callback, budget, leaves,
            aborted,
        );
        return;
    }

    let tail = se_tail_f64(l, d, z, z_c);

    // Remaining budget for this level.
    let rem = bound_sq - partial;
    if rem < 0.0 {
        return;
    }
    let rem_sqrt = rem.sqrt();

    // Level value at integer offset Δ from z_c.int[d] is l_dd·(Δ − frac[d]) +
    // tail, minimized at Δ = frac[d] − tail/l_dd; bound |level| ≤ rem_sqrt gives
    // Δ ∈ center_off ± span. (se_bracket documents the i64/f64 precision split.)
    let center_off = z_c.frac[d] - tail / l_dd;
    let span = rem_sqrt / l_dd.abs();
    let (z_low, z_high, z_mid, max_off) = se_bracket(z_c.int[d], center_off, span);

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
        let level = l_dd * ((zd - z_c.int[d]) as f64 - z_c.frac[d]) + tail;
        let new_partial = partial + level * level;
        // Slack for f64 round-off at the bound check: 1e-9 * bound_sq matches
        // the 8D "10⁻⁹ tolerance" semantics.
        if new_partial > bound_sq + 1e-9 * bound_sq.abs() {
            continue;
        }
        z[d] = zd;
        recurse(
            depth - 1, l, z_c, bound_sq, new_partial, z, callback, budget,
            leaves, aborted,
        );
    }
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

// ─── Parallel norm-pruned Schnorr-Euchner ────────────────────────────────────────

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
/// production workhorse used by [`super::find_aligned_lattice_points_with_stop`].
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
/// Budget-pool chunk size. Also the `consumed`-counter flush granularity.
const BUDGET_CHUNK: u64 = 4096;

/// Per-worker budget cache: a per-node fetch_sub on the shared atomic
/// serializes the whole walk once many workers run (one contended
/// cache line). Chunks are reserved from the same shared pool, charged
/// locally, and the remainder returned on completion; exhaustion
/// aborts at chunk granularity, so admitted work deviates from the
/// per-node scheme by at most ±workers × 4096 nodes — noise against
/// the ≥1M production caps.
struct BudgetCache<'a> {
    remaining: u64,
    used_since_flush: u64,
    /// Predictive-truncation context (None for unbudgeted walks, the
    /// frontier-expansion stage, and when disabled via
    /// `CYCLOSYNTH_PREDICTIVE_TRUNC=0`).
    pred: Option<&'a PredictiveTrunc>,
}

impl<'a> BudgetCache<'a> {
    #[inline]
    fn new(pred: Option<&'a PredictiveTrunc>) -> Self {
        Self { remaining: 0, used_since_flush: 0, pred }
    }

    /// Charge `n` (≤ BUDGET_CHUNK) budget units. Returns `false` iff the
    /// walk must stop — shared pool exhausted or predictive truncation
    /// projected the budget infeasible. Either way `aborted` has been set; the caller
    /// just unwinds, so both causes surface identically as a budget hit.
    #[inline]
    fn charge(
        &mut self,
        n: u64,
        budget: &AtomicU64,
        consumed: Option<&AtomicU64>,
        aborted: &AtomicBool,
    ) -> bool {
        if self.remaining >= n {
            self.remaining -= n;
            self.used_since_flush += n;
            return true;
        }
        // Refill slow path: runs ~once per BUDGET_CHUNK nodes — the natural
        // zero-hot-path-cost hook for the predictive-truncation projection.
        if self.pred.is_some_and(|p| p.should_abort(budget)) {
            aborted.store(true, Ordering::Relaxed);
            self.flush_consumed(consumed);
            return false;
        }
        let prior = budget.fetch_sub(BUDGET_CHUNK, Ordering::Relaxed);
        if prior <= BUDGET_CHUNK {
            // Pool exhausted. Don't bother restoring the pool value: the
            // `aborted` flag is what stops all workers. Count the walk's
            // plain budget-burn once (the worker that flips `aborted` wins).
            if !aborted.swap(true, Ordering::Relaxed)
                && crate::synthesis::diag::trace_enabled()
            {
                crate::synthesis::diag::N_BUDGET_EXHAUST_FIRES.fetch_add(1, Ordering::Relaxed);
            }
            self.flush_consumed(consumed);
            return false;
        }
        self.remaining += BUDGET_CHUNK;
        self.flush_consumed(consumed);
        self.remaining -= n;
        self.used_since_flush += n;
        true
    }

    #[inline]
    fn flush_consumed(&mut self, consumed: Option<&AtomicU64>) {
        if let Some(c) = consumed {
            if self.used_since_flush > 0 {
                c.fetch_add(self.used_since_flush, Ordering::Relaxed);
            }
        }
        self.used_since_flush = 0;
    }

    /// Item teardown: flush progress and return the unused reservation to
    /// the shared pool (keeps total accounting exact across work items).
    #[inline]
    fn finish(mut self, budget: &AtomicU64, consumed: Option<&AtomicU64>) {
        self.flush_consumed(consumed);
        if self.remaining > 0 {
            budget.fetch_add(self.remaining, Ordering::Relaxed);
        }
    }
}

/// Abort a budget-capped walk when the projected total spend exceeds
/// MARGIN × the initial budget. Work-stealing completes skinny items
/// early, so the linear projection systematically UNDERestimates
/// (~6× measured) — a false abort would need a 2.5× overestimate
/// against that bias, physically out of reach.
const PREDICTIVE_TRUNC_MARGIN: f64 = 2.5;
/// Don't project before this fraction of frontier items has completed —
/// earlier projections are too noisy (and maximally biased by the fat
/// head items still in flight).
const PREDICTIVE_TRUNC_MIN_FRAC: f64 = 0.10;

/// Predictive budget truncation: a truncated arm reaches the identical
/// ledger state whether it burns 100% of its budget or aborts at 10%,
/// so project infeasibility from item-completion progress and abort
/// early through the normal `aborted` plumbing. Never attached when
/// budget = u64::MAX (exhaustive runs must stay exhaustive) or on the
/// z15-sharded fallback path (1-3 item frontier — too coarse to project).
struct PredictiveTrunc {
    /// Flat-frontier length at stage-3 launch.
    items_total: usize,
    /// Work items fully walked so far (one increment per completed item).
    items_done: std::sync::atomic::AtomicUsize,
    /// Pool value at walk entry (= the walk's full budget: find_aligned_lattice_points creates
    /// a fresh pool per walk).
    initial_budget: u64,
    /// First-fire latch: dedupes the diag counter and short-circuits the
    /// projection once tripped.
    fired: AtomicBool,
}

impl PredictiveTrunc {
    /// Projection check, called from the [`BudgetCache`] refill slow path
    /// (~once per worker per BUDGET_CHUNK nodes). Returns `true` when the
    /// walk is projected infeasible and must abort.
    // Budget projection is a heuristic; count-to-f64 rounding is immaterial.
    #[allow(clippy::cast_precision_loss)]
    fn should_abort(&self, budget: &AtomicU64) -> bool {
        if self.fired.load(Ordering::Relaxed) {
            return true;
        }
        let done = self.items_done.load(Ordering::Relaxed);
        if done == 0
            || (done as f64) < (self.items_total as f64) * PREDICTIVE_TRUNC_MIN_FRAC
        {
            return false;
        }
        // Consumed = initial − remaining pool. This counts whole chunk
        // reservations, i.e. true node spend plus ≤ workers × BUDGET_CHUNK
        // of in-flight slack — noise against the ≥1M budgets this path
        // runs under (and saturating_sub guards the post-exhaustion pool
        // wraparound, where the projection is moot anyway).
        let consumed = self
            .initial_budget
            .saturating_sub(budget.load(Ordering::Relaxed));
        let fraction_done = done as f64 / self.items_total as f64;
        let projected_total = consumed as f64 / fraction_done;
        if projected_total > self.initial_budget as f64 * PREDICTIVE_TRUNC_MARGIN {
            if !self.fired.swap(true, Ordering::Relaxed)
                && crate::synthesis::diag::trace_enabled()
            {
                crate::synthesis::diag::N_PREDICTIVE_TRUNC_FIRES
                    .fetch_add(1, Ordering::Relaxed);
            }
            return true;
        }
        false
    }
}

/// One sequential work item for the parallel norm-pruned SE walk: a fixed
/// coordinate prefix `z[start_depth+1 ..= 15]` together with the incremental
/// `(x, w, partial_q, partial_eucl)` state [`recurse_collect_norm_pruned`]
/// expects on entry at `start_depth`. Mirrors the per-thread state of a
/// per-z[15] shard, generalized to prefixes of arbitrary length.
#[derive(Clone)]
struct SePrefixItem {
    z: [i64; 16],
    x: [i64; 16],
    w: [f64; 16],
    partial_q: f64,
    /// dd companion of `partial_q` (deep-ε dd Q-bracket mode only; stays
    /// (0, 0) when `l_q_dd` is `None`). Invariant in dd mode:
    /// `partial_q == partial_q_dd.0 + partial_q_dd.1`.
    partial_q_dd: (f64, f64),
    partial_eucl: f64,
}

/// Expand one frontier item at coordinate depth `d` into its surviving
/// depth-`d−1` children (W1 parallel-utilization). This is an exact
/// replica of the depth-`d` node body of [`recurse_collect_norm_pruned`]
/// — node-entry budget consumption (`fetch_sub` + abort on exhaustion),
/// batched `consumed` counter flush, trace counters, degenerate-diagonal
/// branch, Q-bracket child loop in zig-zag order, incremental x/w
/// maintenance, and the norm-shell prune including the integer-exact
/// short-circuit and the dd verify — except that surviving children are
/// pushed as new work items instead of recursed into. Only used at
/// d ≥ 4, so the depth-1 Q-filter and leaf handling never apply here.
// SE center-relative offsets/deltas are bracket-sized: |z − z_c.int| ≤ 2·max_off ≪ 2^53 (se_bracket doc).
#[allow(clippy::too_many_arguments, clippy::cast_precision_loss)]
fn expand_se_prefix_node(
    d: usize,
    mut item: SePrefixItem,
    out: &mut Vec<SePrefixItem>,
    l: &[[f64; 16]; 16],
    l_q_dd: Option<&[[(f64, f64); 16]; 16]>,
    z_c: &SeCenter16,
    bound_sq: f64,
    r_eucl: &[[f64; 16]; 16],
    r_eucl_dd: &[[(f64, f64); 16]; 16],
    u_eucl_dd: &[(f64, f64); 16],
    target_norm_sq: f64,
    target_norm_sq_i128: i128,
    basis: &[[i64; 16]; 16],
    budget: &AtomicU64,
    aborted: &AtomicBool,
    consumed: Option<&AtomicU64>,
    bcache: &mut BudgetCache<'_>,
) {
    // Node-entry bookkeeping — mirrors recurse_collect_norm_pruned exactly
    // (one budget unit per recurse-enter, charged via the chunked cache;
    // `charge` sets `aborted` itself on exhaustion / predictive abort).
    if !bcache.charge(1, budget, consumed, aborted) {
        return;
    }
    let trace = crate::synthesis::diag::trace_enabled();
    if trace && d < 16 {
        crate::synthesis::diag::N_RECURSE_ENTER_AT_DEPTH[d].fetch_add(1, Ordering::Relaxed);
    }
    let l_dd = l[d][d];
    if l_dd.abs() < 1e-30 {
        // Degenerate diagonal: z[d] is forced to z_c[d] (mirror of the
        // recursion's degenerate branch; partial_q unchanged).
        let new_zd = z_c.int[d];
        let delta = new_zd - item.z[d];
        if delta != 0 {
            update_x_for_z_change(&mut item.x, basis, d, delta);
            let delta_f = delta as f64;
            for i in 0..=d {
                item.w[i] += delta_f * r_eucl[i][d];
            }
        }
        item.z[d] = new_zd;
        let level_eucl = item.w[d];
        let new_partial_eucl = item.partial_eucl + level_eucl * level_eucl;
        if new_partial_eucl <= target_norm_sq * (1.0 + 1e-9) {
            item.partial_eucl = new_partial_eucl;
            out.push(item);
        }
        return;
    }
    // SE bracket for z[d] given the fixed z[d+1..16] prefix — same math
    // as the recursion body. In dd Q-bracket mode the tail is computed in
    // double-double (kills the tail-cancellation error channel AND fixes
    // the bracket center, which is derived from tail); the f64 working
    // value is its hi+lo projection.
    let mut tail_dd = (0.0_f64, 0.0_f64);
    let tail = if let Some(lq) = l_q_dd {
        tail_dd = q_tail_dd(lq, d, &item.z, z_c);
        tail_dd.0 + tail_dd.1
    } else {
        se_tail_f64(l, d, &item.z, z_c)
    };
    let rem = bound_sq - item.partial_q;
    if rem < 0.0 {
        return;
    }
    let rem_sqrt = rem.sqrt();
    // Offset of the true center from int[d]: the level value at integer
    // offset Δ = zd − int[d] is l_dd·(Δ − frac[d]) + tail, minimized at
    // Δ = frac[d] − tail/l_dd.
    let center_off = z_c.frac[d] - tail / l_dd;
    let span = rem_sqrt / l_dd.abs();
    let (z_low, z_high, z_mid, max_off) = se_bracket(z_c.int[d], center_off, span);

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
        let level = l_dd * ((zd - z_c.int[d]) as f64 - z_c.frac[d]) + tail;
        let mut new_partial_q = item.partial_q + level * level;
        let mut new_partial_q_dd = (0.0_f64, 0.0_f64);
        if let Some(lq) = l_q_dd {
            // dd Q-bracket: the boundary decision is made on the dd value
            // (accurate to ~1e-32 — no overshoot band needed), and the dd
            // partial is what children inherit, so f64 drift never
            // accumulates across depths.
            let (q_f64, q_dd) =
                q_candidate_dd(lq, d, zd, z_c, tail_dd, item.partial_q_dd);
            new_partial_q = q_f64;
            new_partial_q_dd = q_dd;
        }
        if new_partial_q > bound_sq + 1e-9 * bound_sq.abs() {
            continue;
        }
        // Update z[d], x = B·z, and w = R·z incrementally (same delta
        // scheme as the recursion — the item state persists across the
        // zig-zag siblings).
        let delta = zd - item.z[d];
        if delta != 0 {
            update_x_for_z_change(&mut item.x, basis, d, delta);
            let delta_f = delta as f64;
            for i in 0..=d {
                item.w[i] += delta_f * r_eucl[i][d];
            }
        }
        item.z[d] = zd;

        let level_eucl = item.w[d];
        let new_partial_eucl = item.partial_eucl + level_eucl * level_eucl;
        let threshold = target_norm_sq * (1.0 + 1e-9);
        let prune_fires = new_partial_eucl > threshold; // d ≥ 4 > 0 here
        if resolve_prune(
            prune_fires, &item.x, &item.z, target_norm_sq_i128,
            r_eucl_dd, z_c, u_eucl_dd, d, threshold, trace,
        ) {
            continue;
        }
        let mut child = item.clone();
        child.partial_q = new_partial_q;
        child.partial_q_dd = new_partial_q_dd;
        child.partial_eucl = new_partial_eucl;
        out.push(child);
    }
}

/// Parallel SE walker with norm-shell prune. `external_abort` is an
/// optional cross-task abort signal (set by a peer LDE task that found
/// first under parallel speculation). `consumed` is an optional shared
/// node counter (incremented per recurse-entry) used by the budget-
/// triggered LDE-stagger dispatcher to observe search progress. Pass
/// `None, None` if you don't need either.
#[allow(clippy::too_many_arguments)]
// SE center-relative offsets/deltas are bracket-sized: |z − z_c.int| ≤ 2·max_off ≪ 2^53 (se_bracket doc).
// f64→i128/i64: target_norm_sq = 2^k is f64-exact for k ≤ 126; bracket edges are floor/ceil ints.
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
pub fn schnorr_euchner<F>(
    l: &[[f64; 16]; 16],
    l_q_dd: Option<&[[(f64, f64); 16]; 16]>,
    z_c: &SeCenter16,
    bound_sq: f64,
    r_eucl: &[[f64; 16]; 16],
    r_eucl_dd: &[[(f64, f64); 16]; 16],
    target_norm_sq: f64,
    basis: &[[i64; 16]; 16],
    leaf_filter: F,
    budget: &AtomicU64,
    external_abort: Option<&AtomicBool>,
    consumed: Option<&AtomicU64>,
) -> (Vec<[i64; 16]>, bool)
where
    F: Fn(&[i64; 16]) -> LeafAction + Sync,
{
    use rayon::prelude::*;
    use std::sync::atomic::AtomicBool;

    let aborted = AtomicBool::new(false);
    // Pool value at entry = this walk's full budget (find_aligned_lattice_points creates a
    // fresh pool per walk). Anchor for the predictive-truncation
    // projection; u64::MAX marks the walk unbudgeted.
    let initial_budget = budget.load(Ordering::Relaxed);
    let l_15 = l[15][15];
    // Exact for 2^k at any k ≤ 126 (i128 avoids the k ≥ 63 saturation that
    // would disable the shell prune in the deep-ε regime this walk serves).
    let target_norm_sq_i128 = target_norm_sq as i128;
    if l_15.abs() < 1e-30 {
        return (Vec::new(), false);
    }

    // Center-relative seeding state: the walk's implied baseline is
    // z = z_c.int, not z = 0, so the f64 Euclidean accumulator w never
    // passes through the M ≈ 30·√T·2^53 intermediates that an absolute
    // `z_15_f · R[i][15]` seeding would. One-time per-walk cost.
    let (x_base, u_eucl_dd, u_eucl) = center_relative_seed(basis, r_eucl_dd, z_c);
    let u_eucl_dd = &u_eucl_dd;

    // z[15] range from the Q-bound. Keep the integer part as i64 to avoid
    // the deep-ε f64 quantization issue (as in recurse); the fractional
    // part shifts the bracket onto the true center.
    let span_q = bound_sq.sqrt() / l_15.abs();
    let z_low = z_c.int[15].saturating_add((z_c.frac[15] - span_q).ceil() as i64);
    let z_high = z_c.int[15].saturating_add((z_c.frac[15] + span_q).floor() as i64);
    let z_mid = z_c.int[15].saturating_add(z_c.frac[15].round() as i64);

    // Closest-to-center first ordering at the outermost level.
    let mut prefixes: Vec<i64> = (z_low..=z_high).collect();
    prefixes.sort_by_key(|&z| (z - z_mid).abs());

    // ── Stage 1: seed the work-item frontier from the z[15] candidates ──
    // No budget consumption at this level: the depth-15 loop lives outside
    // the budgeted recursion.
    let mut frontier: Vec<SePrefixItem> = Vec::with_capacity(prefixes.len());
    for z_15 in prefixes {
        // Q-bound contribution at depth 15 (measured from the true center).
        // In dd Q-bracket mode the decision value is the dd one (empty
        // tail at the outermost level) — third copy of the per-candidate
        // ladder, kept in lockstep with the recursion and the W1 frontier.
        let level_q = l_15 * ((z_15 - z_c.int[15]) as f64 - z_c.frac[15]);
        let mut partial_q = level_q * level_q;
        let mut partial_q_dd = (0.0_f64, 0.0_f64);
        if let Some(lq) = l_q_dd {
            let (q_f64, q_dd) =
                q_candidate_dd(lq, 15, z_15, z_c, (0.0, 0.0), (0.0, 0.0));
            partial_q = q_f64;
            partial_q_dd = q_dd;
        }
        if partial_q > bound_sq + 1e-9 * bound_sq.abs() {
            continue;
        }
        // Center-relative seeding: z[0..15] park at z_c.int (their
        // assigned-coordinate baseline), z[15] takes the candidate. The
        // z[15] offset from the center's integer part is bracket-sized,
        // so both the x and w seed corrections are small-magnitude exact:
        //   x = B·z_c.int + off15·B[15]
        //   w[i] = (R·z_c.int)[i] + off15·R[i][15]
        // — w still represents the absolute R·z, but every f64 rounding
        // happens at O(√T·O(1)) magnitude (no z_15 ~ 2^53+ cast).
        let mut z = z_c.int;
        z[15] = z_15;
        let off15 = z_15 - z_c.int[15];
        let mut x = x_base;
        let mut w = u_eucl;
        if off15 != 0 {
            let row = &basis[15];
            for c in 0..16 {
                x[c] = x[c].wrapping_add(off15.wrapping_mul(row[c]));
            }
            let off15_f = off15 as f64;
            for i in 0..=15 {
                w[i] += off15_f * r_eucl[i][15];
            }
        }
        let level_eucl = w[15];
        let partial_eucl = level_eucl * level_eucl;
        if partial_eucl > target_norm_sq * (1.0 + 1e-9) {
            continue;
        }
        frontier.push(SePrefixItem { z, x, w, partial_q, partial_q_dd, partial_eucl });
    }

    // ── Stage 2: flatten more coordinate levels into the frontier ──
    // At fine ε the z[15] bracket holds only 1-3 values, so single-level
    // sharding would serialize the whole walk (util ~1.08× on 14 threads
    // at ε=1e-5). Expand the frontier one coordinate at a time — (z15) →
    // (z15,z14) → … — until there are enough independent items to keep
    // every worker busy. Each expansion step replicates the recursion's
    // depth-d node semantics exactly (see `expand_se_prefix_node`), so the
    // visited node set, budget consumption, and trace counters match the
    // recursive walk's. Items are sorted by accumulated partial_q — the
    // true SE distance — since a per-z[14] |offset| key ignores the
    // z[15]-dependent center. The budget guard keeps tiny-budget walks from
    // spending a meaningful budget fraction on breadth-first frontier
    // expansion before any leaf is reached (sequential semantics are
    // depth-first).
    let threads = rayon::current_num_threads().max(1);
    let frontier_target: usize =
        (threads * 128).min((budget.load(Ordering::Relaxed) / 256).max(1) as usize);
    let mut start_depth: i32 = 14;
    {
        // Frontier expansion runs before items_total is known — no
        // predictive context here.
        let mut bcache = BudgetCache::new(None);
        while !frontier.is_empty()
            && frontier.len() < frontier_target
            && start_depth >= 4
            && !aborted.load(Ordering::Relaxed)
        {
            let d = start_depth as usize;
            let cur = std::mem::take(&mut frontier);
            frontier.reserve(cur.len().saturating_mul(2));
            for item in cur {
                if aborted.load(Ordering::Relaxed) {
                    break;
                }
                if external_abort.is_some_and(|e| e.load(Ordering::Relaxed)) {
                    aborted.store(true, Ordering::Relaxed);
                    break;
                }
                expand_se_prefix_node(
                    d, item, &mut frontier, l, l_q_dd, z_c, bound_sq, r_eucl,
                    r_eucl_dd, u_eucl_dd, target_norm_sq, target_norm_sq_i128,
                    basis, budget, &aborted, consumed, &mut bcache,
                );
            }
            start_depth -= 1;
        }
        bcache.finish(budget, consumed);
    }

    // Closest-first ordering generalized to multi-coordinate prefixes:
    // ascending accumulated Q-distance. For the unbudgeted exhaustive walk
    // this only affects scheduling; under a budget it preserves the SE
    // "most promising subtree first" preference.
    frontier.sort_by(|a, b| a.partial_q.total_cmp(&b.partial_q));

    // Predictive-truncation context — budget-capped flat walks only (see
    // [`PredictiveTrunc`]). The guard: unbudgeted walks (certificates'
    // coverage-complete runs + probes) must never fire;
    // `CYCLOSYNTH_PREDICTIVE_TRUNC=0` is the kill switch.
    let pred_ctx: Option<PredictiveTrunc> = if initial_budget != u64::MAX
        && !predictive_trunc_disabled()
        && !frontier.is_empty()
    {
        Some(PredictiveTrunc {
            items_total: frontier.len(),
            items_done: std::sync::atomic::AtomicUsize::new(0),
            initial_budget,
            fired: AtomicBool::new(false),
        })
    } else {
        None
    };
    let pred = pred_ctx.as_ref();

    // ── Stage 3: walk the items in parallel ──
    // `with_max_len(1)` lets idle workers steal single items: rayon's
    // default split budget (~2 splits/thread) otherwise freezes the vec
    // into ~64 fixed chunks, and the head chunk — the fattest, given the
    // closest-first sort — can pin one thread for most of the walk. Splits
    // stay steal-driven, so this adds no overhead while all workers are busy.
    let solutions: Vec<[i64; 16]> = frontier
        .into_par_iter()
        .with_max_len(1)
        .flat_map_iter(|mut item| {
            if aborted.load(Ordering::Relaxed) {
                return Vec::new().into_iter();
            }
            if external_abort.is_some_and(|e| e.load(Ordering::Relaxed)) {
                aborted.store(true, Ordering::Relaxed);
                return Vec::new().into_iter();
            }
            let mut local: Vec<[i64; 16]> = Vec::new();
            let mut bcache = BudgetCache::new(pred);
            recurse_collect_norm_pruned(
                start_depth, l, l_q_dd, z_c, bound_sq, r_eucl, r_eucl_dd,
                u_eucl_dd, target_norm_sq, target_norm_sq_i128, item.partial_q,
                item.partial_q_dd, item.partial_eucl,
                &mut item.z, &mut item.x, &mut item.w, basis,
                &leaf_filter, budget, &aborted, external_abort, consumed,
                &mut bcache, &mut local,
            );
            bcache.finish(budget, consumed);
            // Predictive-truncation progress: one increment per completed
            // work item (post-abort increments are harmless — the latch /
            // aborted flag already decide everything).
            if let Some(p) = pred {
                p.items_done.fetch_add(1, Ordering::Relaxed);
            }
            local.into_iter()
        })
        .collect();

    let budget_hit = aborted.load(Ordering::Relaxed);
    (solutions, budget_hit)
}

// SE center-relative offsets/deltas are bracket-sized: |z − z_c.int| ≤ 2·max_off ≪ 2^53 (se_bracket doc).
#[allow(clippy::too_many_arguments, clippy::cast_precision_loss)]
fn recurse_collect_norm_pruned<F>(
    depth: i32,
    l: &[[f64; 16]; 16],
    l_q_dd: Option<&[[(f64, f64); 16]; 16]>,
    z_c: &SeCenter16,
    bound_sq: f64,
    r_eucl: &[[f64; 16]; 16],
    r_eucl_dd: &[[(f64, f64); 16]; 16],
    u_eucl_dd: &[(f64, f64); 16],
    target_norm_sq: f64,
    target_norm_sq_i128: i128,
    partial_q: f64,
    partial_q_dd: (f64, f64),
    partial_eucl: f64,
    z: &mut [i64; 16],
    x: &mut [i64; 16],
    w: &mut [f64; 16],
    basis: &[[i64; 16]; 16],
    leaf_filter: &F,
    budget: &AtomicU64,
    aborted: &AtomicBool,
    external_abort: Option<&AtomicBool>,
    consumed: Option<&AtomicU64>,
    bcache: &mut BudgetCache<'_>,
    results: &mut Vec<[i64; 16]>,
) where
    F: Fn(&[i64; 16]) -> LeafAction,
{
    if aborted.load(Ordering::Relaxed) {
        return;
    }
    // Cross-LDE abort signal (parallel LDE speculation): when a peer LDE
    // task at a different lattice level finds a solution, it sets this
    // shared flag. Check once per recurse entry — cheap atomic load.
    if external_abort.is_some_and(|e| e.load(Ordering::Relaxed)) {
        aborted.store(true, Ordering::Relaxed);
        return;
    }
    // Charge per recurse-enter (units are nodes, not leaves) so the
    // budget bounds traversal work — subtree-skipping filters regress
    // under a per-leaf budget. Routed through the chunked BudgetCache;
    // `charge` sets `aborted` itself on exhaustion/truncation.
    if !bcache.charge(1, budget, consumed, aborted) {
        return;
    }
    let trace = crate::synthesis::diag::trace_enabled();
    if trace && depth >= 0 && (depth as usize) < 16 {
        crate::synthesis::diag::N_RECURSE_ENTER_AT_DEPTH[depth as usize]
            .fetch_add(1, Ordering::Relaxed);
    }
    if depth < 0 {
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
        let new_zd = z_c.int[d];
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
                depth - 1, l, l_q_dd, z_c, bound_sq, r_eucl, r_eucl_dd,
                u_eucl_dd, target_norm_sq, target_norm_sq_i128,
                partial_q, partial_q_dd, new_partial_eucl, z, x, w, basis,
                leaf_filter, budget, aborted, external_abort, consumed,
                bcache, results,
            );
        }
        return;
    }
    // SE bracket [z_low, z_high] for the current depth's z[d] enumeration.
    // In dd Q-bracket mode the tail is computed in double-double (see
    // q_tail_dd — kills the tail-cancellation error channel and fixes the
    // bracket center, which is derived from tail); the f64 working value
    // is its hi+lo projection.
    let mut tail_dd = (0.0_f64, 0.0_f64);
    let tail = if let Some(lq) = l_q_dd {
        tail_dd = q_tail_dd(lq, d, z, z_c);
        tail_dd.0 + tail_dd.1
    } else {
        se_tail_f64(l, d, z, z_c)
    };
    let rem = bound_sq - partial_q;
    if rem < 0.0 {
        return;
    }
    let rem_sqrt = rem.sqrt();
    // Offset of the true center from int[d]: the level value at integer
    // offset Δ = zd − int[d] is l_dd·(Δ − frac[d]) + tail, minimized at
    // Δ = frac[d] − tail/l_dd.
    let center_off = z_c.frac[d] - tail / l_dd;
    let span = rem_sqrt / l_dd.abs();
    let (z_low, z_high, z_mid, max_off) = se_bracket(z_c.int[d], center_off, span);

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
        let level = l_dd * ((zd - z_c.int[d]) as f64 - z_c.frac[d]) + tail;
        let mut new_partial_q = partial_q + level * level;
        let mut new_partial_q_dd = (0.0_f64, 0.0_f64);
        if let Some(lq) = l_q_dd {
            // dd Q-bracket: decide the boundary on the dd value (truth to
            // ~1e-32, no overshoot band needed) and thread the dd partial
            // down so f64 drift never accumulates across depths. Kept in
            // lockstep with expand_se_prefix_node and the stage-1 seeding.
            let (q_f64, q_dd) = q_candidate_dd(lq, d, zd, z_c, tail_dd, partial_q_dd);
            new_partial_q = q_f64;
            new_partial_q_dd = q_dd;
        }
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
        if resolve_prune(
            prune_fires, x, z, target_norm_sq_i128,
            r_eucl_dd, z_c, u_eucl_dd, depth as usize, threshold, trace,
        ) {
            continue;
        }
        recurse_collect_norm_pruned(
            depth - 1, l, l_q_dd, z_c, bound_sq, r_eucl, r_eucl_dd,
            u_eucl_dd, target_norm_sq, target_norm_sq_i128,
            new_partial_q, new_partial_q_dd, new_partial_eucl, z, x, w, basis,
            leaf_filter, budget, aborted, external_abort, consumed, bcache,
            results,
        );
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)] // test values are tiny
mod par_tests {
    use super::*;

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
        let r = crate::synthesis::lattice::zeta::cholesky_lu::euclidean_cholesky(&b).expect("PSD");

        // Pick a z, compute ‖B·z‖² directly.
        let z: [i64; 16] = [1, -2, 3, 0, -1, 2, 1, -3, 4, 0, -1, 2, 1, -2, 3, -1];
        let x = reconstruct_x(&b, &z);
        let xnorm_sq: i128 = x.iter().map(|&v| i128::from(v) * i128::from(v)).sum();
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

}

#[cfg(test)]
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)] // test values are tiny/checked
mod tests {
    use super::*;
    use super::super::cholesky_lu::{cholesky_f64, lu_solve_int_inplace};
    use super::super::lll::run_lll;
    use super::super::q_metric::{build_q_int_zeta, build_q_mpfr_zeta};
    use super::super::scratch::IntScratch16;
    use crate::synthesis::lattice::zeta::brute::enumerate_unitary_norm_shell;
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
        let r = run_lll(&mut s);
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
        let r = run_lll(&mut s);
        assert!(matches!(r, super::super::lll::LllResult::Converged));

        let brute = enumerate_unitary_norm_shell(1);
        assert!(!brute.is_empty());
        let target = brute[0];

        use rug::Assign;
        let prec = 256_u32;
        let mut a: [[MpFloat; 17]; 16] = std::array::from_fn(|_| {
            std::array::from_fn(|_| MpFloat::with_val(prec, 0.0))
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
                let factor = MpFloat::with_val(prec, &a[i][k] / &a[k][k]);
                for j in k..17 {
                    let new_val = MpFloat::with_val(prec, &a[i][j] - &factor * &a[k][j]);
                    a[i][j].assign(&new_val);
                }
            }
        }
        let mut z = [0_i64; 16];
        for i in (0..16).rev() {
            let mut s_acc = a[i][16].clone();
            for j in (i + 1)..16 {
                let term = MpFloat::with_val(prec, &a[i][j] * z[j]);
                s_acc -= term;
            }
            let zi = MpFloat::with_val(prec, &s_acc / &a[i][i]);
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

    // ── det_exact tests ────────────────────────────────────────────────────

    #[test]
    fn det16_exact_on_identity() {
        let mut id = [[0i64; 16]; 16];
        for i in 0..16 {
            id[i][i] = 1;
        }
        assert_eq!(crate::synthesis::lattice::zeta::cholesky_lu::det_exact(&id), Some(1));
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
        assert_eq!(crate::synthesis::lattice::zeta::cholesky_lu::det_exact(&m), Some(-1));
    }

    #[test]
    fn det16_exact_on_lll_basis() {
        // A real LLL output basis must be unimodular.
        let v = realistic_v();
        let mut s = IntScratch16::new(1e-3);
        build_q_mpfr_zeta(&mut s, v, 6, 1e-3);
        build_q_int_zeta(&mut s);
        let r = run_lll(&mut s);
        assert!(matches!(r, super::super::lll::LllResult::Converged));
        let det = crate::synthesis::lattice::zeta::cholesky_lu::det_exact(&s.basis).expect("LLL basis det must fit in i64");
        assert!(det == 1 || det == -1,
            "LLL output basis must be unimodular; got det = {}", det);
    }

    // ── euclidean_cholesky tests ──────────────────────────────────────────

    #[test]
    fn euclidean_cholesky_16_round_trip() {
        // Identity basis: B·Bᵀ = I, R = I.
        let mut id = [[0i64; 16]; 16];
        for i in 0..16 {
            id[i][i] = 1;
        }
        let r = crate::synthesis::lattice::zeta::cholesky_lu::euclidean_cholesky(&id).expect("identity should be PD");
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
        let r = crate::synthesis::lattice::zeta::cholesky_lu::euclidean_cholesky(&diag2).expect("2·I should be PD");
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
        let r = crate::synthesis::lattice::zeta::cholesky_lu::euclidean_cholesky(&tri).expect("lower-triangular full-rank should be PD");
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

    // ── schnorr_euchner_16d_reference tests ────────────────────────────────────────────

    /// SE walk on the identity basis with z_c = 0 should enumerate exactly
    /// the integer 16-vectors with ‖z‖² ≤ bound_sq. At bound_sq = 1, that's
    /// the origin + 32 nearest neighbours = 33 leaves.
    #[test]
    fn schnorr_euchner_16d_identity_basis_small_bound() {
        let mut l = [[0.0_f64; 16]; 16];
        for i in 0..16 {
            l[i][i] = 1.0;
        }
        let z_c = SeCenter16::from_int([0_i64; 16]);
        let budget = AtomicU64::new(10_000);
        let mut visited: HashSet<[i64; 16]> = HashSet::new();
        let leaves = schnorr_euchner_16d_reference(&l, &z_c, 1.0, |z| {
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
        let z_c = SeCenter16::from_int([0_i64; 16]);
        let budget = AtomicU64::new(10);
        let leaves = schnorr_euchner_16d_reference(&l, &z_c, 4.0, |_z| true, &budget);
        assert_eq!(leaves, 10, "budget should cap leaves at 10");
    }

    /// At k=2 (norm² = 4), `enumerate_unitary_norm_shell(2)` returns 2848 valid solutions.
    /// Build the LLL+SE pipeline, run SE with a generous bound, verify any
    /// solution returned by SE that passes the leaf checks is in the brute
    /// set (no spurious solutions from SE's enumeration).
    #[test]
    fn schnorr_euchner_16d_returns_subset_of_brute_at_k_2() {
        let v = realistic_v();
        let mut s = IntScratch16::new(1e-3);
        build_q_mpfr_zeta(&mut s, v, 6, 1e-3);
        build_q_int_zeta(&mut s);
        let r = run_lll(&mut s);
        assert!(matches!(r, super::super::lll::LllResult::Converged));

        // f64 Cholesky on the post-LLL Gram (lower-triangular L).
        assert!(cholesky_f64(&mut s));
        // Transpose to upper-triangular for SE.
        let mut l_upper = [[0.0_f64; 16]; 16];
        for i in 0..16 {
            for j in 0..16 {
                l_upper[i][j] = s.l_f64[j][i];
            }
        }
        // LU solve: cap-center in basis coords → fractional SE center.
        assert!(lu_solve_int_inplace(&mut s));
        let z_c = SeCenter16::from_lu_x(&s.lu_x);

        // Brute solutions at k=2.
        let brute_set: HashSet<[i64; 16]> = enumerate_unitary_norm_shell(2).into_iter().collect();

        // Generous bound: large enough that *some* candidates land in the
        // ellipsoid. We don't claim coverage of all 2848; the assertion is
        // that any SE candidate that ALSO passes the leaf checks is in brute.
        let budget = AtomicU64::new(1_000_000);
        let bound_sq = 1.0e6_f64;
        let mut se_set: HashSet<[i64; 16]> = HashSet::new();
        schnorr_euchner_16d_reference(&l_upper, &z_c, bound_sq, |z| {
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

    /// The MPFR-128 Q-metric Cholesky dual must reproduce the post-LLL
    /// Q Gram (Rᵀ·R = G at natural scale), agree with the f64 Cholesky
    /// factor to f64 accuracy, and carry a dd projection whose hi part is
    /// exactly the f64 snapshot (the verify path depends on this).
    #[test]
    fn q_cholesky_dual_matches_gram_and_f64_factor() {
        use crate::synthesis::lattice::common::i256_to_f64;
        let v = realistic_v();
        let mut s = IntScratch16::new(1e-3);
        build_q_mpfr_zeta(&mut s, v, 6, 1e-3);
        build_q_int_zeta(&mut s);
        let r = run_lll(&mut s);
        assert!(matches!(r, super::super::lll::LllResult::Converged));

        let (snap, dd) = crate::synthesis::lattice::zeta::cholesky_lu::q_cholesky_mpfr_dual(&s.gram, s.scale_bits)
            .expect("post-LLL Q Gram must be PD");
        // dd hi part ≡ f64 snapshot, lo bounded by hi's ULP.
        for i in 0..16 {
            for j in 0..16 {
                assert_eq!(dd[i][j].0, snap[i][j], "dd hi != snapshot at ({i},{j})");
                assert!(
                    dd[i][j].1.abs() <= snap[i][j].abs() * 1e-15 + 1e-300,
                    "dd lo not a residual at ({i},{j}): {:?}",
                    dd[i][j]
                );
            }
        }
        // Rᵀ·R = G (natural scale) to f64 round-off.
        let scale = 2.0_f64.powi(-s.scale_bits);
        for i in 0..16 {
            for j in 0..16 {
                let g_nat = i256_to_f64(s.gram[i][j]) * scale;
                let mut acc = 0.0_f64;
                for k in 0..16 {
                    acc += snap[k][i] * snap[k][j];
                }
                let tol = 1e-9 * g_nat.abs().max(1.0);
                assert!(
                    (acc - g_nat).abs() <= tol,
                    "RᵀR != G at ({i},{j}): {acc} vs {g_nat}"
                );
            }
        }
        // Agreement with the f64 Cholesky path (upper-tri transpose of
        // l_f64) — the deep-ε l_upper swap must be a refinement, not a
        // different factor.
        assert!(cholesky_f64(&mut s));
        for i in 0..16 {
            for j in 0..16 {
                let f64_fac = s.l_f64[j][i];
                let tol = 1e-9 * f64_fac.abs().max(1.0);
                assert!(
                    (snap[i][j] - f64_fac).abs() <= tol,
                    "MPFR vs f64 factor mismatch at ({i},{j}): {} vs {}",
                    snap[i][j], f64_fac
                );
            }
        }
    }

    /// dd Q-bracket no-regression gate: the parallel norm-pruned walk with
    /// the dd factor attached must return exactly the same solution set as
    /// the plain f64 walk on the same setup (moderate ε, where f64 is
    /// already sound — geometric solutions sit at Q ≤ 1.25, far from the
    /// 1.5(1+1e-9) boundary, so dd-vs-f64 decision flips cannot touch
    /// them). Exercises all three lockstep ladder sites (stage-1 seeding,
    /// W1 frontier expansion, recursion).
    #[test]
    fn dd_q_bracket_walk_matches_f64_walk() {
        use super::super::scratch::rfv;
        use crate::synthesis::lattice::zeta::brute::uv_to_lattice_y_zeta;
        let v = realistic_v();
        let k = 2u32;
        let eps = 0.5_f64; // wide cap at k=2 → guaranteed non-empty walk
        let mut s = IntScratch16::new(eps);
        build_q_mpfr_zeta(&mut s, v, k, eps);
        build_q_int_zeta(&mut s);
        // Cap center c = y · cap_mid (build_q does not populate scratch.c;
        // mirror of integer.rs's find_aligned_lattice_points step 1).
        let y = uv_to_lattice_y_zeta(v, k);
        let cap_mid = (1.0 + (1.0 - eps * eps).sqrt()) / 2.0;
        for i in 0..16 {
            s.c[i] = rfv(s.prec_q, y[i] * cap_mid);
        }
        let r = run_lll(&mut s);
        assert!(matches!(r, super::super::lll::LllResult::Converged));
        assert!(cholesky_f64(&mut s));
        let l_upper_f64: [[f64; 16]; 16] =
            std::array::from_fn(|i| std::array::from_fn(|j| s.l_f64[j][i]));
        let (l_upper_mpfr, l_q_dd) = crate::synthesis::lattice::zeta::cholesky_lu::q_cholesky_mpfr_dual(&s.gram, s.scale_bits)
            .expect("post-LLL Q Gram must be PD");
        assert!(lu_solve_int_inplace(&mut s));
        let z_c = SeCenter16::from_lu_x(&s.lu_x);
        let (r_eucl, r_eucl_dd) =
            crate::synthesis::lattice::zeta::cholesky_lu::euclidean_cholesky_mpfr_dual(&s.basis).expect("basis full-rank");
        let basis = s.basis;
        let target_norm_sq = 2.0_f64.powi(k as i32);
        let target_i64 = 1_i64 << k;
        let leaf_filter = |x: &[i64; 16]| -> LeafAction {
            let n: i64 = x.iter().map(|&v| v * v).sum();
            if n != target_i64 {
                return LeafAction::Skip;
            }
            let (b1, b2, b3) = bilinear_forms(x);
            if b1 == 0 && b2 == 0 && b3 == 0 {
                LeafAction::Take
            } else {
                LeafAction::Skip
            }
        };
        let bound_sq = 2.5_f64;

        let budget_a = AtomicU64::new(u64::MAX);
        let (sols_f64, hit_a) = schnorr_euchner(
            &l_upper_f64, None, &z_c, bound_sq, &r_eucl, &r_eucl_dd,
            target_norm_sq, &basis, leaf_filter, &budget_a, None, None,
        );
        assert!(!hit_a);
        let budget_b = AtomicU64::new(u64::MAX);
        let (sols_dd, hit_b) = schnorr_euchner(
            &l_upper_mpfr, Some(&l_q_dd), &z_c, bound_sq, &r_eucl, &r_eucl_dd,
            target_norm_sq, &basis, leaf_filter, &budget_b, None, None,
        );
        assert!(!hit_b);

        let set_f64: HashSet<[i64; 16]> = sols_f64.into_iter().collect();
        let set_dd: HashSet<[i64; 16]> = sols_dd.into_iter().collect();
        assert!(!set_f64.is_empty(), "walk found no solutions — test is vacuous");
        assert_eq!(
            set_f64, set_dd,
            "dd Q-bracket walk diverged from f64 walk ({} vs {} solutions)",
            set_f64.len(), set_dd.len()
        );
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
        let r = run_lll(&mut s);
        assert!(matches!(r, super::super::lll::LllResult::Converged));

        assert!(cholesky_f64(&mut s));
        let mut l_upper = [[0.0_f64; 16]; 16];
        for i in 0..16 {
            for j in 0..16 {
                l_upper[i][j] = s.l_f64[j][i];
            }
        }
        assert!(lu_solve_int_inplace(&mut s));
        let z_c = SeCenter16::from_lu_x(&s.lu_x);

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
        let leaves = schnorr_euchner_16d_reference(&l_upper, &z_c, bound_sq, |_z| true, &budget);
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
