//! BKZ-β block reduction on top of the 16D L²-LLL, mirroring fplll's
//! bkz.cpp. Each frontier κ replaces LLL's Lovász swap with block-SVP
//! enumeration over the projected β-dim sublattice; if it finds a vector
//! shorter than b*_κ, that vector is inserted. A tighter basis means a
//! smaller post-LLL Schnorr-Euchner region downstream.
//!
//! Insertion adds no rows: the existing β rows are transformed in place by
//! unimodular ops so one becomes the short vector — three cases by the
//! shape of the SVP coefficient vector `x`: a lone ±1 is a row move; any
//! ±1 absorbs the combination via `row_addmul` then moves; otherwise a
//! binary-GCD tree on `|x|`. Only `b_κ` changes, so the re-reduction is
//! size-reduction over `[0, κ+1)`, not a full LLL. A tour is clean when no
//! insertion shortened any `r̄_{κ,κ}`; stop on a clean tour or `max_loops`.

#![allow(clippy::needless_range_loop)]

use super::scratch::IntScratch16;

/// Maximum BKZ tours before bailing. Most useful BKZ runs converge
/// within 4–8 tours at d=16, β≤8. Hard cap is mostly defensive.
pub const BKZ_MAX_LOOPS: usize = 16;

/// Lovász-equivalent δ for BKZ's "shortened?" check. 0.99 mirrors fplll's
/// default. Lower δ = more aggressive (any small improvement triggers
/// insertion); higher δ = more conservative (only meaningful shortenings
/// trigger).
pub const BKZ_DELTA: f64 = 0.99;

/// SVP enumeration on the projected β-dim sublattice at frontier κ, reading
/// the (MPFR) GS state already populated on `scratch` as f64.
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
        r[i] = scratch.r_bar[kappa + i][kappa + i].to_f64();
        for j in 0..i {
            mu[i][j] = scratch.mu_bar[kappa + i][kappa + j].to_f64();
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

        // SE walks |offset| monotonically outward, alternating ±, so once
        // the closest unexplored offset already exceeds the budget every
        // further one does too. step >= 1 confirms both sides were pruned.
        if new_partial >= *best_norm_sq {
            if step >= 1 {
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

/// Apply the basis update `b_κ ← Σ x[i] · b_{κ+i}` via unimodular row
/// operations on the basis (and the i256 Gram), preserving the
/// lattice. Mirrors fplll's `svp_postprocessing` /
/// `svp_postprocessing_generic` (bkz.cpp:232–379).
///
/// **Strategy**: never add an extra basis row. Instead transform the
/// existing β rows in place so one of them *becomes* the desired
/// short vector, then move it into position κ. Three branches:
///
/// 1. **All-zero except one ±1**: the desired vector already exists
///    as a basis row — just swap it to position κ.
/// 2. **Some ±1 in `x`**: pivot row absorbs the linear combination
///    via `gram_update_size_reduce`-style ops, then swap to position
///    κ.
/// 3. **General**: binary-GCD tree on `|x|` mirrored on basis rows.
///    Each subtraction is an elementary unimodular op; after the
///    tree finishes, one row is exactly `Σ x_i · b_{κ+i}` (assuming
///    `gcd(x) = 1`, generically true for SE output).
///
/// The Gram matrix is updated in lockstep using
/// [`super::lll::gram_update_size_reduce`] for `add` ops and
/// [`super::lll::gram_update_swap`] for swap ops.
/// Returns `Ok(())` on successful insertion; `Err(())` when the SVP
/// coordinate vector is non-primitive (`gcd(x) > 1`), which insertion
/// can't realize with a single unimodular row op. The caller skips the
/// BKZ tour step then — the basis stays LLL-reduced (prior state preserved).
// `Err(())` is a documented "skip this tour step" signal, not an error
// value the caller inspects; a dedicated error type adds nothing.
#[allow(clippy::result_unit_err)]
pub fn bkz_insert(
    scratch: &mut IntScratch16,
    kappa: usize,
    block_size: usize,
    x: &[i64],
) -> Result<(), ()> {
    debug_assert_eq!(x.len(), block_size);
    debug_assert!(kappa + block_size <= 16);

    // Branch 1: all-zero except one ±1 — just move it to κ.
    let nonzero: Vec<usize> = (0..block_size).filter(|&i| x[i] != 0).collect();
    if nonzero.len() == 1 {
        let i = nonzero[0];
        let sign = x[i];
        if i != 0 {
            // Move row κ+i to position κ via adjacent swaps.
            for k in (0..i).rev() {
                let from = kappa + k + 1;
                let to = kappa + k;
                scratch.basis.swap(from, to);
                super::lll::gram_update_swap(scratch, from, to);
            }
        }
        if sign < 0 {
            negate_row(scratch, kappa);
        }
        return Ok(());
    }

    // Branch 2: some |x[i]| = 1 — use it as a pivot.
    if let Some(piv_idx) = (0..block_size).find(|&i| x[i].abs() == 1) {
        let piv_sign = x[piv_idx];
        // For every other non-zero coord j, do
        //   b_{κ+piv_idx} ← b_{κ+piv_idx} + sign · |x[j]| · b_{κ+j}
        // where the sign comes from x[j]·piv_sign so the resulting
        // row is `Σ x[i] · b_{κ+i}`. Then negate-if-needed and move.
        for j in 0..block_size {
            if j == piv_idx || x[j] == 0 {
                continue;
            }
            // We want the final row to equal Σ_i x[i] b_{κ+i}.
            // Currently: row κ+piv_idx is just b_{κ+piv_idx} (= ±1
            // contribution). After this op, we want it to absorb
            // x[j] b_{κ+j}.
            //
            // gram_update_size_reduce(scratch, k, j, r) computes
            //   b_k ← b_k − r · b_j
            // So with k = κ+piv_idx, j = κ+j, r = -piv_sign · x[j]:
            //   b_{κ+piv_idx} ← b_{κ+piv_idx} - (-piv_sign · x[j]) · b_{κ+j}
            //                 = b_{κ+piv_idx} + piv_sign · x[j] · b_{κ+j}.
            // Doing this over all j != piv_idx accumulates
            //   b_{κ+piv_idx} ← piv_sign · b_{κ+piv_idx}
            //                 + piv_sign · Σ_{j≠piv_idx} x[j] · b_{κ+j}
            //                 = piv_sign · Σ_i x[i] · b_{κ+i}.
            // Final negate (if piv_sign < 0) gives the desired row.
            let r = -piv_sign * x[j];
            // Update basis row.
            for c in 0..16 {
                scratch.basis[kappa + piv_idx][c] -= r * scratch.basis[kappa + j][c];
            }
            super::lll::gram_update_size_reduce(scratch, kappa + piv_idx, kappa + j, r);
        }
        // Now row κ+piv_idx = piv_sign · (target). Negate if
        // piv_sign = -1 to get exactly the target.
        if piv_sign < 0 {
            negate_row(scratch, kappa + piv_idx);
        }
        // Move row κ+piv_idx to position κ via adjacent swaps.
        if piv_idx != 0 {
            for k in (0..piv_idx).rev() {
                let from = kappa + k + 1;
                let to = kappa + k;
                scratch.basis.swap(from, to);
                super::lll::gram_update_swap(scratch, from, to);
            }
        }
        return Ok(());
    }

    // Branch 3: general case — binary-GCD tree on |x|, mirrored on
    // basis rows. After the tree, one row equals the gcd-multiple of
    // the target; if gcd = 1 (generic) it equals the target.
    //
    // **Cofactor ↔ basis relationship**: the SVP vector is v = Σᵢ xᵢ bᵢ.
    // To preserve v, the cofactor op `x[k] ← x[k] − x[k_off]` must be
    // mirrored by the basis op `b[k_off] ← b[k_off] + b[k]`. Proof: if
    // we set b[k_off]' = b[k_off] + b[k] and substitute into v, then
    // (x[k] − x[k_off])·b[k] + x[k_off]·b[k_off]' = x[k]·b[k] − x[k_off]·b[k]
    // + x[k_off]·b[k_off] + x[k_off]·b[k] = x[k]·b[k] + x[k_off]·b[k_off]. ✓
    //
    // Equivalent for swap: `x.swap(k, k_off)` mirrors `b.swap(k, k_off)`.
    // Equivalent for negate: `x[i] ← -x[i]` mirrors `b[i] ← -b[i]`.
    //
    // **gcd > 1 case** (rare): the SVP returned a non-primitive vector
    // — v = g·v_prim where v_prim is the truly shortest. We can't pull
    // out the g factor with pure-integer ops on the basis. Bail (no
    // mutation) so the caller can skip κ with the LLL-reduced basis
    // intact. Detect early by computing gcd(x) up front.

    // Compute gcd(|x[0..block_size]|). If > 1, return Err without
    // mutating basis (caller skips, basis stays LLL-reduced).
    fn gcd_i64(a: i64, b: i64) -> i64 {
        let (mut a, mut b) = (a.unsigned_abs(), b.unsigned_abs());
        while b != 0 {
            let t = a % b;
            a = b;
            b = t;
        }
        a as i64
    }
    let mut g: i64 = 0;
    for &xi in x {
        g = gcd_i64(g, xi);
        if g == 1 { break; }
    }
    if g != 1 {
        return Err(());
    }

    // From here on, gcd = 1 guaranteed → the algorithm will succeed.
    let mut x: Vec<i64> = x.to_vec();

    // Sign-normalise: ensure x[i] ≥ 0 by negating both the cofactor
    // and the corresponding basis row.
    for i in 0..block_size {
        if x[i] < 0 {
            x[i] = -x[i];
            negate_row(scratch, kappa + i);
        }
    }

    // Binary-GCD-like tree: at off=1 reduce adjacent pairs, then off=2,
    // off=4, … doubling each round. After each pair-reduction the
    // higher index of the pair holds the partial gcd; the lower is 0.
    //
    // Uses integer division (x[k] mod x[k_off]) for O(log) inner-loop
    // instead of fplll's repeated subtraction. The corresponding
    // basis op b[k_off] += q·b[k] is one row addition with multiplier q.
    let mut off = 1usize;
    while off < block_size {
        let step = 2 * off;
        let mut k = block_size - 1;
        loop {
            if k < off { break; }
            let k_off = k - off;
            // Ensure x[k] ≥ x[k_off] entering the Euclidean reduction.
            if x[k] < x[k_off] {
                x.swap(k, k_off);
                scratch.basis.swap(kappa + k, kappa + k_off);
                super::lll::gram_update_swap(scratch, kappa + k, kappa + k_off);
            }
            // Euclidean reduction: reduce x[k] mod x[k_off] until 0.
            while x[k_off] != 0 {
                // q = x[k] / x[k_off] ≥ 1 here (since x[k] ≥ x[k_off] > 0).
                let q = x[k] / x[k_off];
                x[k] -= q * x[k_off];
                // Basis op: b[k_off] ← b[k_off] + q · b[k].
                // Realize as: b[k_off] ← b[k_off] − (−q)·b[k].
                for c in 0..16 {
                    scratch.basis[kappa + k_off][c] +=
                        q * scratch.basis[kappa + k][c];
                }
                super::lll::gram_update_size_reduce(
                    scratch, kappa + k_off, kappa + k, -q,
                );
                // Now x[k] < x[k_off]. Swap to restore invariant.
                x.swap(k, k_off);
                scratch.basis.swap(kappa + k, kappa + k_off);
                super::lll::gram_update_swap(scratch, kappa + k, kappa + k_off);
            }
            // x[k_off] = 0 here; x[k] holds gcd of the pair's originals.
            if k < step { break; }
            k -= step;
        }
        off *= 2;
    }

    // After all rounds: x[block_size - 1] = 1 (since we pre-checked gcd = 1).
    // All other x[i] = 0. Row κ+block_size-1 now equals Σ x_orig[i]·b[κ+i].
    let final_idx = block_size - 1;
    debug_assert_eq!(x[final_idx], 1,
        "branch-3 invariant: gcd-bearing position should equal 1 (gcd was checked = 1)");

    // Move row κ+final_idx to position κ.
    if final_idx != 0 {
        for k in (0..final_idx).rev() {
            let from = kappa + k + 1;
            let to = kappa + k;
            scratch.basis.swap(from, to);
            super::lll::gram_update_swap(scratch, from, to);
        }
    }
    Ok(())
}

/// Negate row `i` of the basis (and the corresponding rows/columns
/// of the i256 Gram). Negating a basis vector is unimodular and
/// preserves the lattice.
///
/// Math: with `b_i ← -b_i`,
///   - `G_{i,j} = ⟨b_i, b_j⟩` for `j ≠ i` flips sign (b_i flipped, b_j not).
///   - `G_{j,i}` symmetric, also flips.
///   - `G_{i,i} = ⟨b_i, b_i⟩` is unchanged (both factors flip).
fn negate_row(scratch: &mut IntScratch16, i: usize) {
    for c in 0..16 {
        scratch.basis[i][c] = -scratch.basis[i][c];
    }
    for j in 0..16 {
        if j != i {
            scratch.gram[i][j] = -scratch.gram[i][j];
            scratch.gram[j][i] = -scratch.gram[j][i];
        }
    }
    // G[i][i] unchanged.
}

/// Run BKZ-β tours on the basis already in `scratch.basis` (assumed
/// LLL-reduced). Returns `true` if anything changed (any short vector
/// inserted), `false` if all tours were "clean" (no insertions). The
/// caller can re-run LLL after this to clean up any size-reduction
/// staleness in rows above the highest-touched κ.
///
/// **Outer loop**: up to `max_loops` tours. Each tour iterates κ from
/// 0 to `16 - β`. For each κ:
///   1. Recompute the GS state (lll::cfa_row) for rows 0..κ+β so the
///      block's `r̄`, `μ̄` reflect the current basis. (Lazy: we just
///      run cfa for rows 0..16 once at the start of each tour.)
///   2. SVP-enum on the projected β-block at κ.
///   3. If the found vector's norm² is shorter than `δ · r̄_{κ,κ}`,
///      insert via [`bkz_insert`] and mark the tour dirty.
///
/// **Termination**: clean tour OR `max_loops`.
///
/// A non-primitive SVP vector (`gcd(x) > 1`) can't be inserted with one
/// unimodular op; [`bkz_insert`] returns `Err(())` and that tour step is
/// skipped, leaving the basis LLL-reduced. This is rare on LLL-reduced
/// bases (the SE walk usually returns an x with a ±1 coord).
pub fn bkz_tours(
    scratch: &mut IntScratch16,
    block_size: usize,
    max_loops: usize,
) -> bool {
    debug_assert!((3..=8).contains(&block_size));
    let mut any_change = false;

    for _tour in 0..max_loops {
        // Refresh GS state for the whole basis. We need the projected
        // r̄, μ̄ to be current at every κ; doing one full pass at the
        // start of each tour is cheaper than incremental updates and
        // is what fplll does in its tour loop.
        for i in 0..16 {
            super::lll::cfa_row(scratch, i);
        }

        let mut clean = true;
        for kappa in 0..(16 - block_size + 1) {
            let r_kk = scratch.r_bar[kappa][kappa].to_f64();
            // Search radius: δ · r̄_{κ,κ}. fplll uses δ=0.99.
            let radius_sq = BKZ_DELTA * r_kk;
            let svp = svp_enum_block(scratch, kappa, block_size, radius_sq);
            let Some((x, found_norm_sq)) = svp else {
                continue;
            };

            // SVP returned a non-trivially-shorter vector at radius_sq.
            // If it's the trivial "x = ±e_0" (already-in-place), skip.
            // Otherwise insert.
            //
            // Detection: x = ±e_0 has only one nonzero entry at index 0
            // with value ±1. After insertion this is a no-op (negate
            // then move-row, but row 0 is already at position 0).
            let nonzero = x.iter().filter(|&&v| v != 0).count();
            if nonzero == 1 && x[0].abs() == 1 && x[0] == 1 {
                // x = e_0 — basis row κ is already the shortest, skip.
                continue;
            }
            // Check we're shorter than just the existing b_κ. r̄_{κ,κ}
            // is exactly the squared length of the projected b_κ, so
            // found_norm_sq < r̄_{κ,κ} means we found something
            // strictly better (within δ-tolerance handled by radius_sq).
            if found_norm_sq < r_kk {
                // Try the insertion. On a non-primitive vector (gcd > 1)
                // bkz_insert returns Err; skip this κ and keep going —
                // basis remains LLL-reduced.
                if bkz_insert(scratch, kappa, block_size, &x).is_err() {
                    continue;
                }
                clean = false;
                any_change = true;

                // Insertion changed `b_κ`; the GS state for κ and
                // beyond is stale. Refresh it before the next κ.
                for i in kappa..16 {
                    super::lll::cfa_row(scratch, i);
                }
            }
        }

        if clean {
            break;
        }
    }

    any_change
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Set up a scratch with an identity GS state for the first
    /// `block_size` rows: `r̄_{i,i} = 1.0`, `μ̄_{i,j} = 0.0`. With this
    /// state, the projected sublattice is the standard `Z^β` Euclidean
    /// lattice; SVP should return one of the unit vectors `±e_i`.
    fn setup_identity_gso(scratch: &mut IntScratch16, block_size: usize) {
        use rug::Assign;
        for i in 0..16 {
            for j in 0..16 {
                scratch.r_bar[i][j].assign(0.0);
                scratch.mu_bar[i][j].assign(0.0);
            }
        }
        for i in 0..block_size {
            scratch.r_bar[i][i].assign(1.0);
        }
    }

    #[test]
    fn svp_identity_beta4_returns_unit_vector() {
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
        rug::Assign::assign(&mut s.r_bar[0][0], 4.0);
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

    /// Helper: set up a non-trivial 16×16 basis + matching Gram for
    /// insertion tests. We'll use basis = identity (b_i = e_i), and
    /// Gram = identity (so ⟨b_i, b_j⟩ = δ_ij). This is the simplest
    /// case where we can easily verify the post-insertion Gram against
    /// the new basis.
    fn setup_identity_basis(scratch: &mut IntScratch16) {
        scratch.reset_basis();
        for i in 0..16 {
            for j in 0..16 {
                scratch.gram[i][j] = if i == j {
                    i256::i256::from_i64(1)
                } else {
                    i256::i256::from_i64(0)
                };
            }
        }
    }

    /// Recompute Gram from current basis (identity-Q to keep it simple)
    /// and check it matches scratch.gram. This catches any incorrect
    /// Gram update during the insertion.
    fn assert_gram_matches_basis(scratch: &IntScratch16) {
        // For Q = identity (assumed in these tests), Gram[i][j] = b_i · b_j (Euclidean).
        for i in 0..16 {
            for j in 0..16 {
                let mut acc: i128 = 0;
                for c in 0..16 {
                    acc += (scratch.basis[i][c] as i128) * (scratch.basis[j][c] as i128);
                }
                let expected = i256::i256::from_i64(acc as i64);
                assert_eq!(
                    scratch.gram[i][j], expected,
                    "Gram[{i}][{j}] mismatch: actual={:?}, expected (b·b)={:?}",
                    scratch.gram[i][j], expected
                );
            }
        }
    }

    #[test]
    fn bkz_insert_branch1_unit_swap() {
        let mut s = IntScratch16::new(1e-3);
        setup_identity_basis(&mut s);
        // x = [0, 1, 0, 0] at kappa=0: swap row 1 → row 0.
        bkz_insert(&mut s, 0, 4, &[0, 1, 0, 0]).expect("test case should succeed");
        // After swap: basis row 0 = e_1 = [0,1,0,...], row 1 = e_0.
        assert_eq!(s.basis[0][1], 1);
        assert_eq!(s.basis[0][0], 0);
        assert_eq!(s.basis[1][0], 1);
        assert_eq!(s.basis[1][1], 0);
        assert_gram_matches_basis(&s);
    }

    #[test]
    fn bkz_insert_branch1_negate() {
        let mut s = IntScratch16::new(1e-3);
        setup_identity_basis(&mut s);
        // x = [-1, 0, 0, 0]: just negate row 0.
        bkz_insert(&mut s, 0, 4, &[-1, 0, 0, 0]).expect("test case should succeed");
        assert_eq!(s.basis[0][0], -1);
        for j in 1..16 {
            assert_eq!(s.basis[0][j], 0);
        }
        assert_gram_matches_basis(&s);
    }

    #[test]
    fn bkz_insert_branch2_pivot_sum() {
        let mut s = IntScratch16::new(1e-3);
        setup_identity_basis(&mut s);
        // x = [1, 1, 0, 0]: row 0 ← e_0 + e_1 = [1, 1, 0, ...].
        bkz_insert(&mut s, 0, 4, &[1, 1, 0, 0]).expect("test case should succeed");
        assert_eq!(s.basis[0][0], 1);
        assert_eq!(s.basis[0][1], 1);
        for j in 2..16 {
            assert_eq!(s.basis[0][j], 0);
        }
        assert_gram_matches_basis(&s);
    }

    #[test]
    fn bkz_insert_branch2_pivot_difference() {
        let mut s = IntScratch16::new(1e-3);
        setup_identity_basis(&mut s);
        // x = [-1, 2, 1, 0]: pivot at idx=0 (sign=-1), absorbs 2·e_1 + 1·e_2.
        // Result: row 0 should equal -1·e_0 + 2·e_1 + 1·e_2 = [-1, 2, 1, 0,...].
        bkz_insert(&mut s, 0, 4, &[-1, 2, 1, 0]).expect("test case should succeed");
        assert_eq!(s.basis[0][0], -1);
        assert_eq!(s.basis[0][1], 2);
        assert_eq!(s.basis[0][2], 1);
        for j in 3..16 {
            assert_eq!(s.basis[0][j], 0);
        }
        assert_gram_matches_basis(&s);
    }

    #[test]
    fn bkz_insert_branch3_general_case() {
        let mut s = IntScratch16::new(1e-3);
        setup_identity_basis(&mut s);
        // x = [2, 3, 0, 0]: no ±1 entries. gcd(2,3) = 1.
        // Row 0 should end up = 2·e_0 + 3·e_1 = [2, 3, 0, …].
        bkz_insert(&mut s, 0, 4, &[2, 3, 0, 0]).expect("gcd=1 case should succeed");
        assert_eq!(s.basis[0][0], 2);
        assert_eq!(s.basis[0][1], 3);
        for j in 2..16 {
            assert_eq!(s.basis[0][j], 0);
        }
        assert_gram_matches_basis(&s);
    }

    #[test]
    fn bkz_insert_branch3_negative_coords() {
        let mut s = IntScratch16::new(1e-3);
        setup_identity_basis(&mut s);
        // x = [-3, 2, -5, 0]: negative entries, gcd(3,2,5) = 1.
        // Row 0 should equal -3·e_0 + 2·e_1 + (-5)·e_2 = [-3, 2, -5, 0, ...].
        bkz_insert(&mut s, 0, 4, &[-3, 2, -5, 0])
            .expect("gcd=1 case should succeed");
        assert_eq!(s.basis[0][0], -3);
        assert_eq!(s.basis[0][1], 2);
        assert_eq!(s.basis[0][2], -5);
        for j in 3..16 {
            assert_eq!(s.basis[0][j], 0);
        }
        assert_gram_matches_basis(&s);
    }

    #[test]
    fn bkz_insert_branch3_nonprimitive_bails() {
        let mut s = IntScratch16::new(1e-3);
        setup_identity_basis(&mut s);
        let basis_before = s.basis;
        // x = [2, 4, 6, 0]: gcd = 2 > 1. Should return Err without
        // mutating the basis.
        let res = bkz_insert(&mut s, 0, 4, &[2, 4, 6, 0]);
        assert!(res.is_err(), "non-primitive cofactor should return Err");
        assert_eq!(s.basis, basis_before,
            "basis must be unchanged when Branch 3 bails on gcd > 1");
    }

    /// End-to-end: run our LLL on a real Q-matrix (from a synthesis
    /// target), then run BKZ-4 tours, check (a) basis remains
    /// unimodular and (b) ||b*_0||² doesn't increase.
    #[test]
    fn bkz_4_smoke_on_lll_basis() {
        use crate::synthesis::lattice::zeta::{
            integer::find_aligned_lattice_points_with_stop,
            lll::cfa_row,
            cholesky_lu::det16_exact,
        };
        use crate::synthesis::lattice::zeta::brute::uv_to_lattice_y_zeta;
        use std::sync::atomic::AtomicBool;

        // Use Rz(0.3) target as in the rest of the bench suite.
        let v: [f64; 4] = [(0.3_f64 / 2.0).cos(), -(0.3_f64 / 2.0).sin(), 0.0, 0.0];
        let eps = 1e-3_f64;
        let k = 12u32;
        let y = uv_to_lattice_y_zeta(v, k);

        let mut s = IntScratch16::new(eps);

        // First run regular LLL+SE to get an LLL-reduced basis.
        let budget_hit = AtomicBool::new(false);
        let _sols = find_aligned_lattice_points_with_stop(&mut s, &y, k, eps, 100_000, &budget_hit, |_| false, None, None);

        // Verify basis is unimodular pre-BKZ.
        let det_pre = det16_exact(&s.basis);
        assert!(
            matches!(det_pre, Some(1) | Some(-1) | None),
            "pre-BKZ basis should be unimodular (or det-overflow), got det = {det_pre:?}"
        );

        // Refresh GS state and snapshot ||b*_0||² before BKZ.
        for i in 0..16 {
            cfa_row(&mut s, i);
        }
        let r_00_pre = s.r_bar[0][0].to_f64();

        // Run BKZ-4 tours and just observe (non-primitive SVP vectors are
        // skipped, not inserted).
        let _changed = bkz_tours(&mut s, 4, 4);

        // Post-BKZ: still unimodular?
        let det_post = det16_exact(&s.basis);
        assert!(
            matches!(det_post, Some(1) | Some(-1) | None),
            "post-BKZ basis should be unimodular, got det = {det_post:?}"
        );

        // Refresh GS and check ||b*_0||² didn't increase.
        for i in 0..16 {
            cfa_row(&mut s, i);
        }
        let r_00_post = s.r_bar[0][0].to_f64();
        assert!(
            r_00_post <= r_00_pre + 1e-9,
            "BKZ should not lengthen b*_0: pre={r_00_pre}, post={r_00_post}"
        );
        eprintln!(
            "BKZ-4 smoke: r̄_00 pre={r_00_pre:.4e} post={r_00_post:.4e} \
             (ratio={:.4})",
            r_00_post / r_00_pre
        );
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
        rug::Assign::assign(&mut s.mu_bar[1][0], 0.5);
        let (x, norm_sq) = svp_enum_block(&s, 0, 2, 4.0).unwrap();
        assert!((norm_sq - 1.0).abs() < 1e-9, "norm² = {norm_sq}");
        assert_eq!(x[1], 0, "expected x[1] = 0 since r[0]=r[1]=1, got {x:?}");
        assert!(x[0] == 1 || x[0] == -1, "expected x[0] = ±1, got {x:?}");
    }
}
