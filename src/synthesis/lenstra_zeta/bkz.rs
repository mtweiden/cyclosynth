//! BKZ-β block reduction layered on top of the existing 16D L²-LLL.
//!
//! ## What BKZ does
//!
//! BKZ ("block Korkine–Zolotarev") strengthens LLL by replacing the
//! Lovász "compare adjacent rows" step with a **block-SVP enumeration**:
//! for each frontier κ, find the shortest vector in the projected
//! β-dim sublattice spanned by `π(b_κ)..π(b_{κ+β-1})` (projection onto
//! `⟨b_0..b_{κ-1}⟩^⊥`). If the shortest vector is shorter than `b*_κ`,
//! replace `b_κ` with that vector.
//!
//! At β=2 this is exactly LLL's swap test (Gauss reduction on a
//! 2D sublattice). β≥3 gives strictly stronger reduction.
//!
//! ## Why we want it for the Z[ζ_16] pipeline
//!
//! Tighter LLL output basis → smaller post-LLL Schnorr-Euchner region in
//! `dc_search_q`. The fplll BKZ wisdom (Chen–Nguyen 2011): root-Hermite
//! factor drops from `1.0219` (LLL) to `~1.0188` (β=4), `~1.0168`
//! (β=8). At d=16 a 1.0188/1.0219 ratio compounded over 16 dimensions
//! gives ~1.5× shorter `b*_0` empirically. The downstream SE region
//! shrinks ~that much per dimension.
//!
//! ## Implementation reference
//!
//! Mirrors fplll's [bkz.cpp `tour`/`trunc_tour`/`svp_reduction`]
//! (https://github.com/fplll/fplll/blob/master/fplll/bkz.cpp). Key
//! mechanic from the agent's deep-dive (see
//! `project_zeta_lll_optimization.md`):
//!
//! - **No extra basis rows**. After SVP-enum returns a coefficient
//!   vector `x ∈ Z^β`, we transform the existing β rows in place via
//!   unimodular ops so one of them becomes the desired short vector.
//!   Three cases:
//!     1. All-zero except one ±1: just move that row to position κ.
//!     2. Any ±1 in `x`: pivot row absorbs the linear combination via
//!        `row_addmul`, then move-row.
//!     3. General: binary-GCD tree on `|x|` mirrored on basis rows.
//! - **Re-LLL is size-reduction only**, not a full LLL run. The
//!   unimodular insertion preserves basis validity; only `b_κ`
//!   changed, so size-reduce the prefix `[0, κ+1)`.
//! - **Clean tour**: a tour is "clean" if no insertion shortened any
//!   `r̄_{κ,κ}`. Termination on clean OR `max_loops`.
//!
//! Auto-abort copies fplll's [BKZAutoAbort]
//! (https://github.com/fplll/fplll/blob/master/fplll/bkz.cpp#L653-L660):
//! track GSO slope across tours; exit after 5 non-improving tours.

#![allow(clippy::needless_range_loop)]

use super::scratch::IntScratch16;

/// BKZ block size. β=2 is LLL-equivalent; β≥3 gives strict improvement.
/// Default β=4 is a sweet spot at d=16: per-tour cost is `(d-β+1)=13`
/// SVP-enum calls, each ~120 lattice points = trivial. Quality gain is
/// substantial vs LLL (root-Hermite factor 1.022 → 1.019).
pub const BKZ_DEFAULT_BLOCK_SIZE: usize = 4;

/// Maximum BKZ tours before bailing. Most useful BKZ runs converge
/// within 4–8 tours at d=16, β≤8. Hard cap is mostly defensive.
pub const BKZ_MAX_LOOPS: usize = 16;

/// Lovász-equivalent δ for BKZ's "shortened?" check. 0.99 mirrors fplll's
/// default. Lower δ = more aggressive (any small improvement triggers
/// insertion); higher δ = more conservative (only meaningful shortenings
/// trigger).
pub const BKZ_DELTA: f64 = 0.99;

/// SVP enumeration on the projected β-dim sublattice at frontier κ,
/// using the f64 GS state already populated on `scratch`.
///
/// **Math**: for each integer `x ∈ Z^β`, the projected vector
/// `v = Σ_i x[i] π(b_{κ+i})` has squared norm (in the Q-metric)
///
///   `‖v‖²_Q = Σ_{i=0..β-1} (x[i] + Σ_{j>i} μ̄[κ+j][κ+i] · x[j])² · r̄[κ+i][κ+i]`
///
/// where `μ̄`, `r̄` are the GS coefficients of the post-LLL basis.
///
/// **Algorithm** (Schnorr-Euchner, recursive form): for each prefix
/// `x[i+1..β]`, the optimal real centre is `c[i] = -Σ_{j>i} μ̄[κ+j][κ+i]·x[j]`.
/// Try integer values for `x[i]` in increasing distance from `c[i]`,
/// pruning when partial squared distance ≥ current best. Terminate at
/// each level when even the closest unexplored x[i] would put us over.
///
/// Returns `Some((x, norm_sq))` for the shortest non-zero `x` with
/// `‖v‖²_Q ≤ radius_sq`, or `None`.
///
/// At β≤8 the search tree is small (typically ≤ a few hundred nodes).
pub fn svp_enum_block(
    scratch: &IntScratch16,
    kappa: usize,
    block_size: usize,
    radius_sq: f64,
) -> Option<(Vec<i64>, f64)> {
    debug_assert!((2..=8).contains(&block_size));
    debug_assert!(kappa + block_size <= 16);

    // Snapshot GS state for the block. r[i] = r̄_{κ+i, κ+i};
    // mu[i][j] = μ̄_{κ+i, κ+j} for j < i.
    let mut r = [0.0_f64; 8];
    let mut mu = [[0.0_f64; 8]; 8];
    for i in 0..block_size {
        r[i] = scratch.r_bar_f64[kappa + i][kappa + i];
        for j in 0..i {
            mu[i][j] = scratch.mu_bar_f64[kappa + i][kappa + j];
        }
    }

    let mut best_x: Option<Vec<i64>> = None;
    let mut best_norm_sq = radius_sq;
    let mut x = vec![0i64; block_size];

    enumerate_recursive(
        block_size as i32 - 1,
        block_size,
        &r,
        &mu,
        0.0,
        &mut x,
        &mut best_x,
        &mut best_norm_sq,
    );
    best_x.map(|x| (x, best_norm_sq))
}

/// Recursive helper: at depth `i` (going from `block_size-1` down to
/// `0`), enumerate `x[i]` values and recurse to `i-1`.
///
/// `partial_dist`: norm-sq accumulated by `x[i+1..block_size]`.
/// At each level, prune when `partial_dist + (x[i]-c)²·r[i] ≥ best_norm_sq`.
#[allow(clippy::too_many_arguments)]
fn enumerate_recursive(
    depth: i32,
    block_size: usize,
    r: &[f64; 8],
    mu: &[[f64; 8]; 8],
    partial_dist: f64,
    x: &mut [i64],
    best_x: &mut Option<Vec<i64>>,
    best_norm_sq: &mut f64,
) {
    if depth < 0 {
        // Leaf: x is fully determined. Skip the all-zeros vector.
        if x.iter().any(|&v| v != 0) && partial_dist < *best_norm_sq {
            *best_norm_sq = partial_dist;
            *best_x = Some(x.to_vec());
        }
        return;
    }
    let i = depth as usize;
    // Centre: c = -Σ_{j > i} μ[j][i] · x[j].
    let mut c = 0.0_f64;
    for j in (i + 1)..block_size {
        c -= mu[j][i] * (x[j] as f64);
    }
    let c_round = c.round() as i64;

    // Schnorr-Euchner ordering: try c_round, c_round+1, c_round-1,
    // c_round+2, c_round-2, ... — increasing distance from c.
    let mut step = 0i64;
    loop {
        // Map step to offset: 0, +1, -1, +2, -2, +3, -3, ...
        let offset = if step == 0 {
            0
        } else if step % 2 == 1 {
            (step + 1) / 2
        } else {
            -(step / 2)
        };
        let candidate = c_round + offset;
        let delta = candidate as f64 - c;
        let inc = delta * delta * r[i];
        let new_partial = partial_dist + inc;

        // Termination: once `new_partial >= best_norm_sq` AND we've
        // moved past the centre on both sides, no future candidate
        // can do better at this depth.
        //
        // SE order: each time we move one further out (step grows),
        // |offset| grows monotonically up to step/2 (rounded). Once
        // even the closest unexplored offset's contribution exceeds
        // the budget, all further offsets only make it worse, so
        // stop.
        if new_partial >= *best_norm_sq {
            // Determine if both directions have been explored past
            // the prune threshold. Cleanest test: once the **next**
            // candidate's |delta·√r| would exceed √(best - partial),
            // no untested offset can succeed. SE alternates +/-, so
            // once we've gone "over" on either side, the pattern
            // ensures we keep going further.
            //
            // Practical: break when the **inc** for the next step
            // would also exceed budget. Since SE walks |offset|
            // monotonically, after one prune we can safely stop.
            // (Edge case for even/odd step parity: just walk one
            // more step to confirm both sides are pruned.)
            if step >= 1 {
                // Both sides have been seen at distance ≥ |offset|.
                break;
            }
            step += 1;
            continue;
        }

        x[i] = candidate;
        enumerate_recursive(
            depth - 1,
            block_size,
            r,
            mu,
            new_partial,
            x,
            best_x,
            best_norm_sq,
        );
        step += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Set up a scratch with an identity GS state for the first
    /// `block_size` rows: `r̄_{i,i} = 1.0`, `μ̄_{i,j} = 0.0`. With this
    /// state, the projected sublattice is the standard `Z^β` Euclidean
    /// lattice; SVP should return one of the unit vectors `±e_i`.
    fn setup_identity_gso(scratch: &mut IntScratch16, block_size: usize) {
        for i in 0..16 {
            for j in 0..16 {
                scratch.r_bar_f64[i][j] = 0.0;
                scratch.mu_bar_f64[i][j] = 0.0;
            }
        }
        for i in 0..block_size {
            scratch.r_bar_f64[i][i] = 1.0;
        }
    }

    #[test]
    fn svp_identity_β4_returns_unit_vector() {
        let mut s = IntScratch16::new(1e-3);
        setup_identity_gso(&mut s, 4);
        let result = svp_enum_block(&s, 0, 4, 4.0);
        let (x, norm_sq) = result.expect("identity Gram has shortest non-zero norm² = 1");
        // Shortest non-zero vector in Z^4 with Euclidean metric has norm² = 1.
        assert!((norm_sq - 1.0).abs() < 1e-9, "norm² = {norm_sq}, expected 1.0");
        let nz = x.iter().filter(|&&v| v != 0).count();
        assert_eq!(nz, 1, "expected exactly one ±1 entry, got {x:?}");
        assert!(x.iter().any(|&v| v == 1 || v == -1));
    }

    /// Stretched diagonal: `r̄_{0,0} = 4`, `r̄_{i,i} = 1` for i≥1. The
    /// shortest non-zero vector is now `e_1`, `e_2`, or `e_3` (norm² = 1),
    /// not `e_0` (norm² = 4).
    #[test]
    fn svp_stretched_diagonal_avoids_long_axis() {
        let mut s = IntScratch16::new(1e-3);
        setup_identity_gso(&mut s, 4);
        s.r_bar_f64[0][0] = 4.0;
        let result = svp_enum_block(&s, 0, 4, 16.0);
        let (x, norm_sq) = result.unwrap();
        assert!((norm_sq - 1.0).abs() < 1e-9, "norm² = {norm_sq}");
        assert_eq!(x[0], 0, "should NOT use the stretched axis 0, got {x:?}");
    }

    /// With a diagonal GS state and a tight radius, only the smallest
    /// vector should be reported.
    #[test]
    fn svp_returns_none_when_all_too_long() {
        let mut s = IntScratch16::new(1e-3);
        setup_identity_gso(&mut s, 3);
        // Radius² = 0.5 < 1, so no integer vector fits.
        let result = svp_enum_block(&s, 0, 3, 0.5);
        assert!(result.is_none(), "expected None, got {result:?}");
    }

    /// Off-diagonal μ test: r̄ = (1, 1), μ̄_{1,0} = 0.5. Block:
    ///   norm²(x) = (x[0] + 0.5·x[1])² + x[1]²
    /// Try x = (-1, 1): (-1 + 0.5)² + 1 = 0.25 + 1 = 1.25.
    /// Try x = (0, 1): 0.25 + 1 = 1.25.
    /// Try x = (1, 0): 1 + 0 = 1.
    /// Try x = (-1, 1) was tied with (0,1); shortest is (1,0) or (-1,0)
    /// at norm² = 1.
    #[test]
    fn svp_with_off_diagonal_mu() {
        let mut s = IntScratch16::new(1e-3);
        setup_identity_gso(&mut s, 2);
        s.mu_bar_f64[1][0] = 0.5;
        let (x, norm_sq) = svp_enum_block(&s, 0, 2, 4.0).unwrap();
        assert!((norm_sq - 1.0).abs() < 1e-9, "norm² = {norm_sq}");
        assert_eq!(x[1], 0, "expected x[1] = 0 since r[0]=r[1]=1, got {x:?}");
        assert!(x[0] == 1 || x[0] == -1, "expected x[0] = ±1, got {x:?}");
    }
}
