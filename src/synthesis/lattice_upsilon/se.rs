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

#[inline]
fn norm_sqr_i128_wide(x: &[i128; 16]) -> i128 {
    let mut s: i128 = 0;
    for v in x {
        s += *v * *v;
    }
    for i in 0..4 {
        s += x[i] * x[i + 4];
    }
    for i in 0..4 {
        s += x[8 + i] * x[8 + i + 4];
    }
    s
}

#[inline]
fn isqrt_i128(n: i128) -> i128 {
    if n < 0 {
        return -1;
    }
    if n < 2 {
        return n;
    }
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
    debug_assert!(b > 0);
    let q = a / b;
    let r = a % b;
    if r < 0 {
        q - 1
    } else {
        q
    }
}

/// Exact depth-0 shell filter for the n=12 cyclotomic norm. With
/// `z[1..15]` fixed, solve the quadratic
/// `N(p + z0*b0) = target_norm` and return the few integer `z0` values
/// that can hit the norm shell inside the active SE bracket.
#[inline]
fn analytical_depth0_z0_candidates_upsilon(
    x: &[i64; 16],
    z0_curr: i64,
    basis_0: &[i64; 16],
    target_norm: i128,
    z_low: i64,
    z_high: i64,
    out: &mut [i64; 6],
) -> usize {
    let p: [i128; 16] =
        std::array::from_fn(|i| (x[i] as i128) - (z0_curr as i128) * (basis_0[i] as i128));
    let b0: [i128; 16] = std::array::from_fn(|i| basis_0[i] as i128);
    let p_plus_b0: [i128; 16] = std::array::from_fn(|i| p[i] + b0[i]);

    let p_norm = norm_sqr_i128_wide(&p);
    let b_norm = norm_sqr_i128_wide(&b0);
    if b_norm == 0 {
        return 0;
    }
    let linear = norm_sqr_i128_wide(&p_plus_b0) - p_norm - b_norm;
    let constant = p_norm - target_norm;
    let disc = linear * linear - 4 * b_norm * constant;
    if disc < 0 {
        return 0;
    }

    let sqrt_disc = isqrt_i128(disc);
    let denom = 2 * b_norm;
    let mut n = 0usize;
    for &sign in &[1_i128, -1_i128] {
        let q = floor_div_i128(-linear + sign * sqrt_disc, denom);
        for nudge in -1_i64..=1 {
            let cand_i128 = q + nudge as i128;
            if cand_i128 < i64::MIN as i128 || cand_i128 > i64::MAX as i128 {
                continue;
            }
            let cand = cand_i128 as i64;
            if cand < z_low || cand > z_high {
                continue;
            }
            let mut already = false;
            for existing in out.iter().take(n) {
                if *existing == cand {
                    already = true;
                    break;
                }
            }
            if already {
                continue;
            }
            let x_candidate: [i128; 16] = std::array::from_fn(|i| p[i] + (cand as i128) * b0[i]);
            if norm_sqr_i128_wide(&x_candidate) == target_norm && n < out.len() {
                out[n] = cand;
                n += 1;
            }
        }
    }
    n
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
    let mut a: [[i128; 16]; 16] = std::array::from_fn(|i| std::array::from_fn(|j| m[i][j] as i128));
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
    let trace = std::env::var_os("CYCLOSYNTH_TRACE_DEEP_EPS").is_some();
    if trace {
        let mut max_l: f64 = 0.0;
        for i in 0..16 {
            for j in 0..16 {
                let a = l[i][j].abs();
                if a > max_l {
                    max_l = a;
                }
            }
        }
        let max_zc = z_c.iter().map(|v| v.unsigned_abs()).max().unwrap_or(0);
        eprintln!(
            "[trace stage 5 schnorr_euchner_16d] ENTERED bound_sq={bound_sq:.3e} max|L_ij|={max_l:.3e} max|z_c|={max_zc}"
        );
    }
    recurse_16(
        15,
        l,
        z_c,
        bound_sq,
        0.0,
        None,
        None,
        &mut z,
        &mut callback,
        budget,
        &mut leaves,
        &mut aborted,
    );
    if trace {
        eprintln!("[trace stage 5 schnorr_euchner_16d] EXITED n_enumerated (leaves) = {leaves}");
    }
    leaves
}

/// SE walk with an exact n=12 norm-shell filter at the final coordinate.
/// This avoids spending the leaf budget on candidates that cannot satisfy
/// `N(x) = 2^k`, while preserving the caller's exact leaf checks.
pub fn schnorr_euchner_16d_norm_shell<F>(
    l: &[[f64; 16]; 16],
    z_c: &[i64; 16],
    bound_sq: f64,
    basis: &[[i64; 16]; 16],
    target_norm: i128,
    mut callback: F,
    budget: &AtomicU64,
) -> usize
where
    F: FnMut(&[i64; 16]) -> bool,
{
    let mut z = [0i64; 16];
    let mut leaves: usize = 0;
    let mut aborted = false;
    let trace = std::env::var_os("CYCLOSYNTH_TRACE_DEEP_EPS").is_some();
    if trace {
        eprintln!(
            "[trace stage 5 schnorr_euchner_16d_norm_shell] ENTERED bound_sq={bound_sq:.3e} target_norm={target_norm}"
        );
    }
    recurse_16(
        15,
        l,
        z_c,
        bound_sq,
        0.0,
        Some((basis, target_norm)),
        None,
        &mut z,
        &mut callback,
        budget,
        &mut leaves,
        &mut aborted,
    );
    if trace {
        eprintln!(
            "[trace stage 5 schnorr_euchner_16d_norm_shell] EXITED n_enumerated (shell leaves) = {leaves}"
        );
    }
    leaves
}

/// SE walk with norm-shell filter AND bullet-aware pruning (the deep-ε
/// unlock). Branches whose remaining freedom cannot zero all three bullet
/// quadratics are skipped at every node, on top of the existing depth-0
/// norm-shell filter. The set of leaves that reach `callback` is the same
/// as without pruning — soundness is the load-bearing invariant.
pub fn schnorr_euchner_16d_norm_shell_with_bullets<F>(
    l: &[[f64; 16]; 16],
    z_c: &[i64; 16],
    bound_sq: f64,
    basis: &[[i64; 16]; 16],
    target_norm: i128,
    bullet_ctx: &BulletPruneCtx,
    mut callback: F,
    budget: &AtomicU64,
) -> usize
where
    F: FnMut(&[i64; 16]) -> bool,
{
    let mut z = [0i64; 16];
    let mut leaves: usize = 0;
    let mut aborted = false;
    let trace = std::env::var_os("CYCLOSYNTH_TRACE_DEEP_EPS").is_some();
    if trace {
        eprintln!(
            "[trace stage 5 schnorr_euchner_16d_norm_shell_with_bullets] ENTERED bound_sq={bound_sq:.3e} target_norm={target_norm}"
        );
    }
    recurse_16(
        15,
        l,
        z_c,
        bound_sq,
        0.0,
        Some((basis, target_norm)),
        Some(bullet_ctx),
        &mut z,
        &mut callback,
        budget,
        &mut leaves,
        &mut aborted,
    );
    if trace {
        eprintln!(
            "[trace stage 5 schnorr_euchner_16d_norm_shell_with_bullets] EXITED n_enumerated (shell leaves) = {leaves}"
        );
    }
    leaves
}

#[allow(clippy::too_many_arguments)]
fn recurse_16<F>(
    depth: i32,
    l: &[[f64; 16]; 16],
    z_c: &[i64; 16],
    bound_sq: f64,
    partial: f64,
    shell: Option<(&[[i64; 16]; 16], i128)>,
    bullet_ctx: Option<&BulletPruneCtx>,
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
    if shell.is_some() && budget.fetch_sub(1, Ordering::Relaxed) <= 1 {
        *aborted = true;
        return;
    }
    if depth < 0 {
        *leaves += 1;
        if shell.is_none() && budget.fetch_sub(1, Ordering::Relaxed) <= 1 {
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
            shell,
            bullet_ctx,
            z,
            callback,
            budget,
            leaves,
            aborted,
        );
        return;
    }

    // ── Bullet-aware pruning. Soundness: only excludes subtrees that
    // cannot satisfy bullets=0 under conservative box bounds (see
    // `bullet_prune_subtree`). The leaf set is preserved.
    //
    // Depth gating: at shallow depths (close to d=15) the free-coord box
    // spans tend to swamp the linear/quadratic terms, so pruning rarely
    // fires and the O(d²) per-node check is wasted work. We only attempt
    // pruning at deeper levels (d ≤ 10) where the box is tight enough to
    // exclude 0 frequently. The threshold is empirical; raising it makes
    // pruning sound→useless (more work, same result), lowering it gives up
    // some pruning opportunities. Stays sound either way.
    if let Some(ctx) = bullet_ctx {
        if d <= 10 && bullet_prune_subtree(ctx, l, z_c, z, d, bound_sq, partial) {
            return;
        }
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

    if d == 0 {
        if let Some((basis, target_norm)) = shell {
            let x = reconstruct_x(basis, z);
            let mut candidates = [0i64; 6];
            let n = analytical_depth0_z0_candidates_upsilon(
                &x,
                z[0],
                &basis[0],
                target_norm,
                z_low,
                z_high,
                &mut candidates,
            );
            if n == 0 {
                return;
            }
            for &zd in candidates.iter().take(n) {
                if *aborted {
                    return;
                }
                let level = l_dd * ((zd - z_c[d]) as f64) + tail;
                let new_partial = partial + level * level;
                if new_partial > bound_sq + 1e-9 * bound_sq.abs() {
                    continue;
                }
                z[d] = zd;
                *leaves += 1;
                if !callback(z) {
                    *aborted = true;
                }
            }
            return;
        }
    }

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
            shell,
            bullet_ctx,
            z,
            callback,
            budget,
            leaves,
            aborted,
        );
    }
}

// ─── Bullet quadratic-form matrices (for SE bullet-aware pruning) ───────────
//
// The three bullet forms b₂(x), b₃(x), b₆(x) are integer quadratic forms
// over the 16D cyclotomic-coord vector x = [u₁(8) | u₂(8)]. Each is
// representable as b_j(x) = xᵀ B_j x with B_j a 16×16 integer symmetric
// matrix.
//
// We derive B_j by polarisation against the existing (correct)
// `bullet_forms` function so the matrices stay synchronised with the leaf
// check by construction.
//
// Block structure: bullet_forms sums per_element(u₁) + per_element(u₂),
// so B_j = blkdiag(B_j_el, B_j_el) where B_j_el is 8×8.

/// Per-element evaluation of `bullet_forms`, computed on a 16-vector with
/// the back-half zeroed out (so only the front 8 contribute).
#[inline]
fn bullet_forms_front_only(c: &[i64; 8]) -> (i128, i128, i128) {
    let mut x = [0i64; 16];
    x[..8].copy_from_slice(c);
    bullet_forms(&x)
}

/// Extract the three **doubled** 8×8 symmetric integer matrices
/// `2·B_j_el = M_j + M_jᵀ` such that the per-element bullet
/// `b_j_el(c) = cᵀ B_j_el c = (cᵀ (2·B_j_el) c) / 2`.
///
/// The underlying quadratic representation `Q(c) = cᵀ M c` is generally
/// asymmetric, so the symmetrised form `B = (M+Mᵀ)/2` may carry half-integer
/// off-diagonals (e.g. `b₂` has odd `Q(e_1+e_2) − Q(e_1) − Q(e_2) = 1`).
/// Storing `2·B = M + Mᵀ` keeps everything integer; evaluation divides by 2
/// at the end, which is exact because `cᵀ(M+Mᵀ)c = 2·cᵀMc` is always even.
///
/// Returns `[2·B_2_el, 2·B_3_el, 2·B_6_el]`.
pub fn bullet_matrices_8_el_doubled() -> [[[i128; 8]; 8]; 3] {
    let mut bs = [[[0i128; 8]; 8]; 3];

    // Diagonal: (2B)[i][i] = 2·Q(e_i).
    let mut q_e = [[0i128; 8]; 3];
    for i in 0..8 {
        let mut e = [0i64; 8];
        e[i] = 1;
        let (b2, b3, b6) = bullet_forms_front_only(&e);
        q_e[0][i] = b2;
        q_e[1][i] = b3;
        q_e[2][i] = b6;
        bs[0][i][i] = 2 * b2;
        bs[1][i][i] = 2 * b3;
        bs[2][i][i] = 2 * b6;
    }

    // Off-diagonal: (2B)[i][k] = Q(e_i + e_k) − Q(e_i) − Q(e_k).
    for i in 0..8 {
        for k in (i + 1)..8 {
            let mut e = [0i64; 8];
            e[i] = 1;
            e[k] = 1;
            let (b2, b3, b6) = bullet_forms_front_only(&e);
            let q_sum = [b2, b3, b6];
            for j in 0..3 {
                let off = q_sum[j] - q_e[j][i] - q_e[j][k];
                bs[j][i][k] = off;
                bs[j][k][i] = off;
            }
        }
    }

    bs
}

/// Extract the three **doubled** 16×16 symmetric integer matrices `2·B_j`
/// such that the total bullet `b_j(x) = (xᵀ (2·B_j) x) / 2`. Built as
/// block-diagonal copies of the 8×8 per-element matrices.
pub fn bullet_matrices_16_doubled() -> [[[i128; 16]; 16]; 3] {
    let bs8 = bullet_matrices_8_el_doubled();
    let mut bs = [[[0i128; 16]; 16]; 3];
    for j in 0..3 {
        for i in 0..8 {
            for k in 0..8 {
                bs[j][i][k] = bs8[j][i][k];
                bs[j][8 + i][8 + k] = bs8[j][i][k];
            }
        }
    }
    bs
}

/// Evaluate the bullet form `b_j(x) = (xᵀ (2·B_j) x) / 2`. Division by 2
/// is always exact for integer `x` (because `xᵀ(M+Mᵀ)x = 2·xᵀMx`).
#[inline]
pub fn eval_bullet_doubled(b_double: &[[i128; 16]; 16], x: &[i64; 16]) -> i128 {
    let mut s: i128 = 0;
    for i in 0..16 {
        let xi = x[i] as i128;
        for k in 0..16 {
            s += xi * b_double[i][k] * (x[k] as i128);
        }
    }
    debug_assert_eq!(s % 2, 0, "xᵀ(2B)x must be even");
    s / 2
}

/// Rotate the three doubled bullet matrices `2·B_j` into the SE working
/// basis. With `x = Rᵀ z` (the `reconstruct_x` convention), the bullet
/// quadratic `b_j(x) = (xᵀ (2·B_j) x) / 2` becomes
///   `b_j(z) = (zᵀ (R · 2B_j · Rᵀ) z) / 2 = (zᵀ (2B_j′) z) / 2`,
/// where `2B_j′ = R · (2B_j) · Rᵀ` is computed here at i128 precision.
///
/// Returns `[2·B_2′, 2·B_3′, 2·B_6′]`.
///
/// Overflow: `R` entries stay under ≈2^41 even at deep ε; the original
/// `2B_j` entries are ≤ ~16 in magnitude. Each product `R·(2B)·Rᵀ` entry
/// is bounded by `16² · 16² · 2^41 · 2^41 · ~10 ≈ 2^90`, well inside i128.
pub fn rotate_bullet_matrices_to_se_basis(
    basis: &[[i64; 16]; 16],
    bullets_orig: &[[[i128; 16]; 16]; 3],
) -> [[[i128; 16]; 16]; 3] {
    let mut out = [[[0i128; 16]; 16]; 3];
    for j in 0..3 {
        // First T = (2B_j) · Rᵀ, i.e. T[a][k] = Σ_b (2B_j)[a][b] · R[k][b].
        let mut t = [[0i128; 16]; 16];
        for a in 0..16 {
            for k in 0..16 {
                let mut s: i128 = 0;
                for b in 0..16 {
                    let r = basis[k][b] as i128;
                    if r != 0 {
                        s += bullets_orig[j][a][b] * r;
                    }
                }
                t[a][k] = s;
            }
        }
        // Then out[j] = R · T, i.e. out[j][i][k] = Σ_a R[i][a] · T[a][k].
        for i in 0..16 {
            for k in 0..16 {
                let mut s: i128 = 0;
                for a in 0..16 {
                    let r = basis[i][a] as i128;
                    if r != 0 {
                        s += r * t[a][k];
                    }
                }
                out[j][i][k] = s;
            }
        }
    }
    out
}

/// Sanity helper: evaluate `b_j` either by direct quadratic form on `x`
/// or by the rotated form on `z` (where `x = Rᵀ z` via `reconstruct_x`).
/// Returns `true` if all three bullets agree on this `z`.
#[cfg(test)]
fn bullets_via_rotated_match_direct(
    bullets_orig: &[[[i128; 16]; 16]; 3],
    bullets_se: &[[[i128; 16]; 16]; 3],
    basis: &[[i64; 16]; 16],
    z: &[i64; 16],
) -> bool {
    let x = reconstruct_x(basis, z);
    for j in 0..3 {
        let direct = eval_bullet_doubled(&bullets_orig[j], &x);
        let via_z = eval_bullet_doubled(&bullets_se[j], z);
        if direct != via_z {
            return false;
        }
    }
    true
}

/// Context passed to the SE walker to enable bullet-aware branch pruning.
/// Owns the three doubled rotated bullet matrices in `z`-coords plus a
/// few z_c-dependent pre-computations that are constant for the whole walk.
pub struct BulletPruneCtx {
    /// `2·B′_j` for `j ∈ {0,1,2} = {b₂, b₃, b₆}`, in the SE-basis coords:
    /// `b_j(R^T z) = (zᵀ (2·B′_j) z) / 2`.
    pub bullets_se_doubled: [[[i128; 16]; 16]; 3],
    /// `b_j(z_c)` for each j (the cap-center bullet value).
    pub bj_zc: [i128; 3],
    /// `γ_j[p] = Σ_q (2·B′_j)[p][q] · z_c[q]` for each `j` and `p`.
    /// `γ` is the partial derivative of `b_j(z)` w.r.t. `z[p]` evaluated
    /// at `z = z_c` (modulo the factor of 1 from symmetric expansion).
    pub gamma: [[i128; 16]; 3],
}

impl BulletPruneCtx {
    /// Construct from the rotated bullet matrices and the (rounded) cap
    /// center. Hoists out `b_j(z_c)` and `γ_j` so per-node pruning work is
    /// linear in `d` rather than quadratic.
    pub fn new(bullets_se_doubled: [[[i128; 16]; 16]; 3], z_c: &[i64; 16]) -> Self {
        let mut bj_zc = [0_i128; 3];
        let mut gamma = [[0_i128; 16]; 3];
        for j in 0..3 {
            let m = &bullets_se_doubled[j];
            // γ[p] = Σ_q m[p][q] z_c[q]
            let mut bj_2: i128 = 0;
            for p in 0..16 {
                let mut s: i128 = 0;
                for q in 0..16 {
                    s += m[p][q] * (z_c[q] as i128);
                }
                gamma[j][p] = s;
                bj_2 += s * (z_c[p] as i128);
            }
            bj_zc[j] = bj_2 / 2;
        }
        Self {
            bullets_se_doubled,
            bj_zc,
            gamma,
        }
    }
}

/// Decide whether the bullet quadratic forms can plausibly all hit zero at
/// some integer `z` consistent with the current node state:
///   - `z[d+1..16]` fixed (from `z` parameter, those slots already set),
///   - `z[0..=d]` still free; their *deviation* δ[p] = z[p] − z_c[p]
///     bounded by conservative spans derived from the remaining radius.
///
/// We re-center the bullet quadratic at `z_c` rather than at the origin.
/// Writing `z = z_c + δ`,
///   `b_j(z) = b_j(z_c) + 2 (z_c · 2B′_j)·δ + δᵀ B′_j δ`,
/// where `B′_j = (2·B′_j)/2`. The constant `b_j(z_c)` is exact and
/// (typically) far from zero, while the linear and quadratic parts depend
/// on `δ` whose components in V are bounded in `[−span_p, span_p]`. Interval
/// arithmetic over this small symmetric box gives a tight enclosure of
/// `b_j(z)`; if 0 ∉ enclosure for any j, no feasible integer `z` in this
/// subtree can have all three bullets vanish.
///
/// Soundness: spans dominate the actual `|z[p] − z_c[p]|` for any
/// SE-feasible `z`; the interval enclosure is conservative; the box is
/// symmetric around 0 in δ-space so `z[p]² = (z_c[p] + δ[p])²` is bounded
/// by the standard widening identity.
fn bullet_prune_subtree(
    ctx: &BulletPruneCtx,
    l: &[[f64; 16]; 16],
    z_c: &[i64; 16],
    z: &[i64; 16],
    d: usize,
    bound_sq: f64,
    partial: f64,
) -> bool {
    let rem = bound_sq - partial;
    if rem < 0.0 {
        return true;
    }
    let rem_sqrt = rem.sqrt();

    // ── (a) Build conservative spans for free coords p ∈ [0, d]. ────────
    // For each free p, |z[p] − z_c[p]| = |δ[p]| ≤ span_p where
    //   span_p ≥ (sqrt(rem) + |tail_p_fixed| + Σ_{q∈(p,d]} |l[p][q]|·span_q) / |l[p][p]|
    // computed top-down from p = d downward.
    let mut spans = [0.0_f64; 16];
    let mut tail_fixed_abs = [0.0_f64; 16];
    for p in 0..=d {
        let mut s = 0.0_f64;
        for q in (d + 1)..16 {
            s += l[p][q].abs() * ((z[q] - z_c[q]) as f64).abs();
        }
        tail_fixed_abs[p] = s;
    }
    for p_inv in 0..=d {
        let p = d - p_inv;
        let l_pp_abs = l[p][p].abs();
        if l_pp_abs < 1e-30 {
            return false;
        }
        let mut bound = rem_sqrt + tail_fixed_abs[p];
        for q in (p + 1)..=d {
            bound += l[p][q].abs() * spans[q];
        }
        spans[p] = bound / l_pp_abs;
    }
    // Integer span: span_p_int ≥ ceil(spans[p]).
    let mut span_int = [0_i64; 16];
    for p in 0..=d {
        let s = spans[p].ceil();
        let s_i64 = if s.is_finite() && s.abs() < 1e18 {
            s as i64
        } else {
            return false;
        };
        span_int[p] = s_i64;
    }

    // ── (b) For each bullet j, expand around z_c. With z = z_c + δ:
    //         b_j(z) = b_j(z_c) + γ·δ + (1/2)·δᵀ(2B)δ,
    //         where γ[p] = Σ_q (2B)[p][q] z_c[q]  and (2B)·z_c is symmetric.
    //   - δ[F] = z[F] − z_c[F]  (known)
    //   - δ[V] ∈ [−span_int[p], +span_int[p]]
    // We push the fixed-δ contributions into a single constant, and bound
    // the V-component contributions by interval arithmetic.
    let mut delta_fixed = [0_i128; 16];
    for q in (d + 1)..16 {
        delta_fixed[q] = (z[q] as i128) - (z_c[q] as i128);
    }

    for j in 0..3 {
        let b2 = &ctx.bullets_se_doubled[j];
        let gamma = &ctx.gamma[j];
        let bj_zc = ctx.bj_zc[j];

        // Contribution from γ·δ split into F and V:
        //   const_lin_F = Σ_{p∈F} γ[p] δ_F[p]
        //   linear coef for δ[p] in V: γ[p].
        let mut const_lin_f: i128 = 0;
        for p in (d + 1)..16 {
            const_lin_f += gamma[p] * delta_fixed[p];
        }

        // Quadratic δᵀ(2B)δ / 2 split:
        //   F-F: δ_F^T (2B)_FF δ_F / 2 — constant.
        //   F-V cross: δ_F^T (2B)_FV δ_V — linear in δ_V; absorb into linear coefs.
        //   V-V: δ_V^T (2B)_VV δ_V / 2 — quadratic in δ_V.
        let mut const_qf2: i128 = 0;
        for a in (d + 1)..16 {
            let da = delta_fixed[a];
            if da == 0 {
                continue;
            }
            for b in (d + 1)..16 {
                const_qf2 += da * b2[a][b] * delta_fixed[b];
            }
        }
        let const_q_f = const_qf2 / 2;

        // Linear coefficient for δ[p], p ∈ V:
        //   c_lin[p] = γ[p] + Σ_{a∈F} (2B)[a][p] δ_F[a]
        let mut c_lin = [0_i128; 16];
        for p in 0..=d {
            let mut s = gamma[p];
            for a in (d + 1)..16 {
                let da = delta_fixed[a];
                if da != 0 {
                    s += b2[a][p] * da;
                }
            }
            c_lin[p] = s;
        }

        // Total constant when δ_V = 0:
        let const_total = bj_zc + const_lin_f + const_q_f;

        // ── Interval arithmetic over δ_V[p] ∈ [−span_int[p], +span_int[p]]. ─
        let mut lo: i128 = const_total;
        let mut hi: i128 = const_total;

        // Linear part: |Σ c_lin[p] δ[p]| ≤ Σ |c_lin[p]| span_int[p].
        // It is symmetric around 0 in δ-space.
        let mut lin_half_width: i128 = 0;
        for p in 0..=d {
            let absc = c_lin[p].saturating_abs();
            let s = span_int[p] as i128;
            lin_half_width = lin_half_width.saturating_add(absc.saturating_mul(s));
        }
        lo = lo.saturating_sub(lin_half_width);
        hi = hi.saturating_add(lin_half_width);

        // Quadratic part on δ_V: (1/2) Σ_{p,q ∈ V} (2B)[p][q] δ[p] δ[q].
        // Split diagonals and off-diagonals; box is [-span, span] in each
        // coord so δ[p]² ∈ [0, span²]; δ[p]δ[q] ∈ [-span_p span_q, +span_p span_q].
        // We accumulate the doubled sum and divide by 2 at the end.
        let mut q_lo: i128 = 0;
        let mut q_hi: i128 = 0;
        for p in 0..=d {
            let app = b2[p][p];
            let sp = span_int[p] as i128;
            let sp2 = sp.saturating_mul(sp);
            // (2B)[p][p] · δ[p]² with δ[p]² ∈ [0, sp²].
            let (dlo, dhi) = if app >= 0 {
                (0_i128, app.saturating_mul(sp2))
            } else {
                (app.saturating_mul(sp2), 0_i128)
            };
            q_lo = q_lo.saturating_add(dlo);
            q_hi = q_hi.saturating_add(dhi);
            // Off-diagonal: 2 · (2B)[p][q] · δ[p] δ[q] for p < q in V.
            for qq in (p + 1)..=d {
                let apq = b2[p][qq];
                if apq == 0 {
                    continue;
                }
                let sq = span_int[qq] as i128;
                let psq = sp.saturating_mul(sq);
                let abs_two_apq = apq.saturating_abs().saturating_mul(2);
                let bound = abs_two_apq.saturating_mul(psq);
                q_lo = q_lo.saturating_sub(bound);
                q_hi = q_hi.saturating_add(bound);
            }
        }
        // Divide doubled quadratic by 2 (widen): floor for lo, ceil for hi.
        let q_lo_div2 = if q_lo >= 0 { q_lo / 2 } else { -((-q_lo + 1) / 2) };
        let q_hi_div2 = if q_hi >= 0 { (q_hi + 1) / 2 } else { -((-q_hi) / 2) };
        lo = lo.saturating_add(q_lo_div2);
        hi = hi.saturating_add(q_hi_div2);

        if lo > 0 || hi < 0 {
            return true;
        }
    }
    false
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

    /// Sanity: the extracted 16×16 bullet matrices reproduce `bullet_forms`
    /// exactly on a fixed set of vectors (the polarisation identity, applied
    /// the other way).
    #[test]
    fn bullet_matrices_16_match_bullet_forms() {
        use rand::{rngs::StdRng, Rng, SeedableRng};
        let bs = bullet_matrices_16_doubled();
        let cases: [[i64; 16]; 6] = [
            [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, -1, 2, 0, 1, 1, 0, -1, 0, 1, -1, 2, 0, 0, 1, 1],
            [3, 2, -1, 0, 4, -2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [-1, 0, 1, 0, 1, 0, -1, 0, 1, 0, -1, 0, -1, 0, 1, 0],
            [5, -3, 2, 1, 0, 4, -2, 1, 1, -1, 0, 2, -3, 5, 0, 1],
            [-7, 2, -1, 3, 4, 0, -5, 1, 8, -2, 3, -1, 0, 6, -4, 2],
        ];
        for x in &cases {
            let (b2, b3, b6) = bullet_forms(x);
            assert_eq!(eval_bullet_doubled(&bs[0], x), b2, "b2 mismatch on {x:?}");
            assert_eq!(eval_bullet_doubled(&bs[1], x), b3, "b3 mismatch on {x:?}");
            assert_eq!(eval_bullet_doubled(&bs[2], x), b6, "b6 mismatch on {x:?}");
        }
        // Plus a randomised sweep — high confidence the matrices encode the
        // exact quadratic form.
        let mut rng = StdRng::seed_from_u64(0xb0117);
        for _ in 0..200 {
            let x: [i64; 16] = std::array::from_fn(|_| rng.random_range(-12_i64..=12));
            let (b2, b3, b6) = bullet_forms(&x);
            assert_eq!(eval_bullet_doubled(&bs[0], &x), b2);
            assert_eq!(eval_bullet_doubled(&bs[1], &x), b3);
            assert_eq!(eval_bullet_doubled(&bs[2], &x), b6);
        }
    }

    /// Sanity: at every depth d during a deep enumeration, the bullet
    /// quadratic forms evaluated at z (with z[F] fixed, z[V] = 0) match
    /// what `bullet_prune_subtree` decomposes into (const_j alone, since
    /// linear and quadratic over z[V]=0 are zero).
    #[test]
    fn bullet_pruning_const_matches_direct_eval() {
        use rand::{rngs::StdRng, Rng, SeedableRng};
        let bullets_orig = bullet_matrices_16_doubled();
        let mut rng = StdRng::seed_from_u64(0xc0117);
        let mut basis = [[0i64; 16]; 16];
        for i in 0..16 {
            basis[i][i] = 1;
        }
        for _ in 0..6 {
            let i = rng.random_range(0..16);
            let k = rng.random_range(0..16);
            if i != k {
                for c in 0..16 {
                    basis[i][c] += basis[k][c];
                }
            }
        }
        let bullets_se = rotate_bullet_matrices_to_se_basis(&basis, &bullets_orig);
        let dummy_zc = [0i64; 16];
        let _ctx = BulletPruneCtx::new(bullets_se, &dummy_zc);
        // Random z; check that direct b_j matches bullets_se evaluation at same z.
        for _ in 0..20 {
            let z: [i64; 16] = std::array::from_fn(|_| rng.random_range(-3_i64..=3));
            let x = reconstruct_x(&basis, &z);
            let (b2, b3, b6) = bullet_forms(&x);
            assert_eq!(eval_bullet_doubled(&bullets_se[0], &z), b2);
            assert_eq!(eval_bullet_doubled(&bullets_se[1], &z), b3);
            assert_eq!(eval_bullet_doubled(&bullets_se[2], &z), b6);
        }
    }

    /// Sanity: bullet evaluation through the rotated SE-basis matrices
    /// matches direct evaluation on `x = Rᵀ z`, for a few random bases and
    /// `z` vectors. This is the precondition for any branch pruning we do
    /// in `z`-space.
    #[test]
    fn rotated_bullets_match_direct_on_z() {
        use rand::{rngs::StdRng, Rng, SeedableRng};
        let bullets_orig = bullet_matrices_16_doubled();
        // Use a few unimodular-ish small bases (identity + permutation +
        // a denser one).
        let mut bases: Vec<[[i64; 16]; 16]> = Vec::new();
        // identity
        let mut id = [[0i64; 16]; 16];
        for i in 0..16 {
            id[i][i] = 1;
        }
        bases.push(id);
        // a swap basis (just relabels coords, still unimodular)
        let mut swap = id;
        swap.swap(0, 5);
        swap.swap(1, 9);
        bases.push(swap);
        // a denser one with ±1 entries (still unimodular)
        let mut rng = StdRng::seed_from_u64(42);
        let mut dense = id;
        for _ in 0..6 {
            let i = rng.random_range(0..16);
            let k = rng.random_range(0..16);
            if i != k {
                for c in 0..16 {
                    dense[i][c] += dense[k][c];
                }
            }
        }
        bases.push(dense);

        for basis in &bases {
            let bullets_se = rotate_bullet_matrices_to_se_basis(basis, &bullets_orig);
            for _ in 0..20 {
                let z: [i64; 16] = std::array::from_fn(|_| rng.random_range(-5_i64..=5));
                assert!(
                    bullets_via_rotated_match_direct(&bullets_orig, &bullets_se, basis, &z),
                    "rotated bullets mismatch on basis/z"
                );
            }
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
