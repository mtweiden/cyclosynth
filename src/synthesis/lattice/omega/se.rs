//! Schnorr-Euchner enumeration over the 8-dimensional integer lattice, plus
//! the candidate-validation helpers that go with it.
//!
//! Inputs (produced by the L²-LLL pipeline):
//!   - The LLL-reduced basis B (`[[i64; 8]; 8]`).
//!   - The Cholesky factor R of the Q-metric Gram matrix on the LLL basis,
//!     in MPFR `MpFloat` at [`SE_PREC`] = 128 bits.
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
use std::sync::LazyLock;

use i256::i256;
use rug::Assign;
use crate::rings::MpFloat;


type IMat8 = [[i64; 8]; 8];

/// Norm-shell discriminant prune at the innermost SE coordinate. On by
/// default (neutral at shallow ε, ~1.4× at ε=1e-10); `CYCLOSYNTH_SHELL_FILTER=0`
/// is the kill-switch. Cached once per process.
static SHELL_FILTER: LazyLock<bool> =
    LazyLock::new(|| std::env::var("CYCLOSYNTH_SHELL_FILTER").as_deref() != Ok("0"));

/// Whether the depth-0 norm-shell prune is enabled (default on; `=0` disables).
pub fn shell_filter_enabled() -> bool {
    *SHELL_FILTER
}

/// Depth-0 norm-shell reachability test. With `z[1..8]` fixed, the shell
/// equation `‖x‖² = 2^k` (x = B·z) is quadratic in `z[0]`:
///   a·z[0]² + b·z[0] + c = 0,  a = ‖b₀‖²,  b = 2⟨x_rest, b₀⟩,  c = ‖x_rest‖² − 2^k
/// where `x_rest = Σ_{i≥1} z[i]·bᵢ`. An integer z[0] on the shell needs
/// `D = b² − 4ac ≥ 0` and a perfect square, so a non-square `D` proves the
/// whole z[0] range misses the shell — every leaf below would fail the norm
/// check. Discriminant math is checked i256 (partial norms reach ~2^205); any
/// overflow yields `None` and the caller falls back to full enumeration.
pub struct ShellFilter {
    basis: IMat8,
    a: i256,
    target_norm: i256,
}

impl ShellFilter {
    pub fn new(basis: &IMat8, target_norm: i128) -> Self {
        let mut a = i256::from_i64(0);
        for j in 0..8 {
            let b0 = i256::from_i64(basis[0][j]);
            // ‖b₀‖² ≤ 8·(2^33)² = 2^69, cannot overflow i256.
            a = a.checked_add(b0.checked_mul(b0).expect("‖b₀‖² ≤ 2^69, cannot overflow i256")).expect("‖b₀‖² ≤ 2^69, cannot overflow i256");
        }
        ShellFilter { basis: *basis, a, target_norm: i256::from_i128(target_norm) }
    }

    /// `Some(false)`: no integer z[0] lands on the shell → skip the whole
    /// z[0] enumeration. `Some(true)`: a shell-hitting z[0] may exist →
    /// enumerate normally. `None`: an i256 step overflowed → enumerate
    /// normally (never skips a real candidate).
    #[inline]
    fn shell_reachable(&self, z: &[i128; 8]) -> Option<bool> {
        let mut x_rest = [0i128; 8];
        for i in 1..8 {
            for j in 0..8 {
                let t = z[i].checked_mul(self.basis[i][j] as i128)?;
                x_rest[j] = x_rest[j].checked_add(t)?;
            }
        }
        let mut dot = i256::from_i64(0);
        let mut xr_sq = i256::from_i64(0);
        for j in 0..8 {
            let xr = i256::from_i128(x_rest[j]);
            let b0 = i256::from_i64(self.basis[0][j]);
            dot = dot.checked_add(xr.checked_mul(b0)?)?;
            xr_sq = xr_sq.checked_add(xr.checked_mul(xr)?)?;
        }
        let b = dot.checked_mul(i256::from_i64(2))?;
        let c = xr_sq.checked_sub(self.target_norm)?;
        let b2 = b.checked_mul(b)?;
        let four_ac = self.a.checked_mul(c)?.checked_mul(i256::from_i64(4))?;
        let d = b2.checked_sub(four_ac)?;
        if d.is_negative() {
            return Some(false);
        }
        let s = isqrt_i256(d);
        Some(s.checked_mul(s)? == d)
    }
}

/// `floor(√n)` for `n ≥ 0`, bit-by-bit (the i256 crate has no division/isqrt).
fn isqrt_i256(mut n: i256) -> i256 {
    let one = i256::from_i64(1);
    let mut x = i256::from_i64(0);
    let nbits = 256u32 - n.leading_zeros();
    if nbits == 0 {
        return x;
    }
    let mut shift = (nbits - 1) & !1u32; // largest even ≤ nbits−1
    let mut bit = one << shift;
    loop {
        let xb = x + bit;
        if n >= xb {
            n -= xb;
            x = (x >> 1u32) + bit;
        } else {
            x >>= 1u32;
        }
        if shift == 0 {
            break;
        }
        shift -= 2;
        bit >>= 2u32;
    }
    x
}

/// MPFR precision used by SE. 128 bits gives enough margin for SE's
/// 10⁻⁹ bound-check tolerance at all supported ε; f64-only SE breaks at
/// ε ≤ 1e-5 from squared-norm cancellation noise.
pub const SE_PREC: u32 = 128;

/// Convert an arbitrary-precision `MpFloat` (built at scratch.prec_q for
/// post-LLL Cholesky) to the SE working precision (128 bits). Single
/// allocation, single MPFR conversion.
pub fn rfloat_to_se(r: &MpFloat) -> MpFloat {
    MpFloat::with_val(SE_PREC, r)
}

/// Per-walk reusable MPFR scratch for [`recurse`]. These eight temporaries are
/// recomputed (via `assign`) at every node and never read across the recursive
/// call, so one shared set passed by `&mut` down the recursion replaces ~10
/// `MpFloat::with_val` allocations per node. (`tail` and `new_partial` are NOT
/// here: they must persist across the child call, so they stay per-frame.)
struct SharedTemps {
    tmp: MpFloat,
    diff: MpFloat,
    prod: MpFloat,
    zd_rf: MpFloat,
    level: MpFloat,
    level_sq: MpFloat,
    center: MpFloat,
    scratch_c: MpFloat,
}

impl SharedTemps {
    fn new() -> Self {
        let z = || MpFloat::with_val(SE_PREC, 0.0_f64);
        SharedTemps {
            tmp: z(), diff: z(), prod: z(), zd_rf: z(),
            level: z(), level_sq: z(), center: z(), scratch_c: z(),
        }
    }
}

// ─── 8D Schnorr-Euchner enumeration ──────────────────────────────────────────

/// Enumerate integer 8-tuples z ∈ ℤ⁸ satisfying ‖R·(z − z_c)‖² ≤ bound, in
/// distance-from-center order, invoking `callback(&z)` at each leaf. Returns
/// the first non-`None` callback result, or `None` if the search exhausts.
///
/// All distance arithmetic uses MPFR `MpFloat` at 128-bit precision: at
/// extreme ε (Cholesky-diagonal ratios > 10¹⁰) f64 squared-norm noise
/// causes "ghost-node" SE blowup.
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
/// out, `budget_exhausted` is set and the walk unwinds. It bounds
/// no-solution levels, where the leaf-callback budget never binds because
/// almost nothing reaches a leaf. The walk is single-threaded, so a plain
/// decrementing atomic (no chunked reservation à la 16D `BudgetCache`) is
/// contention-free; the per-entry `fetch_sub` is noise against the ~10 MPFR
/// allocations each recurse-entry already performs.
///
/// `budget_exhausted` may also be set from inside `callback` (the leaf-cap
/// path) to abort the walk without reporting a solution.
#[allow(clippy::too_many_arguments)]
pub fn schnorr_euchner<F>(
    r_chol: &[[MpFloat; 8]; 8],
    z_c: &[MpFloat; 8],
    bound: &MpFloat,
    r_chol_eucl: Option<&[[f64; 8]; 8]>,
    target_norm_eucl: f64,
    abort: &AtomicBool,
    node_budget: &AtomicU64,
    budget_exhausted: &AtomicBool,
    shell: Option<&ShellFilter>,
    mut callback: F,
) -> Option<[i64; 8]>
where
    F: FnMut(&[i128; 8]) -> Option<[i64; 8]>,
{
    // Coordinates in the LLL-reduced basis span ~√κ(Q); with κ(Q) ≈ 16/ε⁴ the
    // reduced coordinate crosses 2^63 near ε=1e-10 (inner shell k≈43) even
    // though the reconstructed point x stays ~2^21. i128 gives 2^127 of
    // headroom (good past ε≈1e-14); the reconstructed x still fits i64.
    let mut z = [0i128; 8];
    let result = std::cell::RefCell::new(None);
    let zero = MpFloat::with_val(SE_PREC, 0.0_f64);
    let mut shared = SharedTemps::new();

    recurse(
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
        shell,
        &mut callback,
        &result,
        &mut shared,
    );
    result.into_inner()
}

#[allow(clippy::too_many_arguments)]
fn recurse<F>(
    depth: i32,
    r_chol: &[[MpFloat; 8]; 8],
    z_c: &[MpFloat; 8],
    bound: &MpFloat,
    r_chol_eucl: Option<&[[f64; 8]; 8]>,
    target_norm_eucl: f64,
    partial_eucl: f64,
    z: &mut [i128; 8],
    partial: &MpFloat,
    abort: &AtomicBool,
    node_budget: &AtomicU64,
    budget_exhausted: &AtomicBool,
    shell: Option<&ShellFilter>,
    callback: &mut F,
    result: &std::cell::RefCell<Option<[i64; 8]>>,
    shared: &mut SharedTemps,
) where
    F: FnMut(&[i128; 8]) -> Option<[i64; 8]>,
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

    // `tail` and `new_partial` must survive the recursive call (tail is reused
    // across this frame's offset loop; new_partial is read by the child as its
    // `partial`), so they stay per-frame. Every other temporary lives in
    // `shared` (reused via assign, recomputed each node — never read across the
    // child call), replacing ~10 allocations per node with two.
    let mut tail = MpFloat::with_val(SE_PREC, 0.0_f64);
    let mut new_partial = MpFloat::with_val(SE_PREC, 0.0_f64);

    // Structural guard against a degenerate diagonal (r_chol PD-ness should
    // exclude this, but tolerate it gracefully).
    shared.tmp.assign(r_dd);
    shared.tmp.abs_mut();
    if shared.tmp.to_f64() < 1e-30 {
        shared.scratch_c.assign(&z_c[d]);
        shared.scratch_c.round_mut();
        z[d] = shared.scratch_c.to_integer().and_then(|n| n.to_i128()).unwrap_or(0);
        recurse(
            depth - 1, r_chol, z_c, bound, r_chol_eucl, target_norm_eucl,
            partial_eucl, z, partial, abort, node_budget, budget_exhausted,
            shell, callback, result, shared,
        );
        return;
    }

    // Depth-0 norm-shell prune: if no integer z[0] can land on ‖x‖²=2^k, every
    // leaf below fails the norm check — skip the whole z[0] enumeration.
    if depth == 0 {
        if let Some(f) = shell {
            if f.shell_reachable(z) == Some(false) {
                return;
            }
        }
    }

    // tail = Σ_{j > d} R[d][j] · (z[j] − z_c[j])
    for j in (d + 1)..8 {
        // Exact i64 → MPFR lift. `z[j] as f64` loses low bits once
        // |z| > 2^53 — at deep ε the lattice coordinates reach ~1.6e16
        // (ε=1e-8, lde_inner=34) in Euclid-pathological frames, and a ±2-ulp
        // error here times R[d][j] is an O(1) error in `level` against an
        // O(1) span.
        shared.diff.assign(z[j]);
        shared.diff -= &z_c[j];
        shared.prod.assign(&r_chol[d][j] * &shared.diff);
        tail += &shared.prod;
    }

    shared.tmp.assign(bound - partial);
    if shared.tmp.to_f64() < 0.0 {
        return;
    }
    let rem_sqrt_f = shared.tmp.to_f64().sqrt();

    // Iteration bounds. The CENTER must be computed and rounded in MPFR:
    // with |z| beyond f64's exact-integer range an f64 center
    // (`z_c[d].to_f64() − tail/r_dd`) is off by ±2 ulps ≈ ±4 units while
    // the per-level span is O(1), so the branch holding a TRUE solution
    // could fall outside [z_low, z_high] at ε=1e-8. The span itself is
    // O(1) and stays f64.
    let r_dd_f = r_dd.to_f64();
    let span = rem_sqrt_f / r_dd_f.abs();
    // center = z_c[d] − tail/r_dd, held in shared.center (scratch_c is the
    // intermediate tail/r_dd term, then reused for the round below).
    shared.scratch_c.assign(&tail / r_dd);
    shared.center.assign(&z_c[d] - &shared.scratch_c);
    let to_i128 = |v: &MpFloat| -> Option<i128> { v.to_integer().and_then(|n| n.to_i128()) };
    shared.tmp.assign(&shared.center - span);
    shared.tmp.ceil_mut();
    let z_low = to_i128(&shared.tmp);
    shared.tmp.assign(&shared.center + span);
    shared.tmp.floor_mut();
    let z_high = to_i128(&shared.tmp);
    shared.scratch_c.assign(&shared.center);
    shared.scratch_c.round_mut();
    let z_mid = to_i128(&shared.scratch_c);
    let (Some(z_low), Some(z_high), Some(z_mid)) = (z_low, z_high, z_mid) else {
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
        shared.zd_rf.assign(zd);
        shared.diff.assign(&shared.zd_rf - &z_c[d]);
        shared.level.assign(r_dd * &shared.diff);
        shared.level += &tail;
        shared.level_sq.assign(&shared.level * &shared.level);
        new_partial.assign(partial + &shared.level_sq);
        shared.tmp.assign(&new_partial - bound);
        if shared.tmp.to_f64() > 1e-9 {
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
        recurse(
            depth - 1, r_chol, z_c, bound, r_chol_eucl, target_norm_eucl,
            new_partial_eucl, z, &new_partial, abort, node_budget,
            budget_exhausted, shell, callback, result, shared,
        );
    }
}

// ─── Lattice-point reconstruction + bilinear-form check ──────────────────────

/// Reconstruct the lattice point `x = B·z` where `B` is the LLL-reduced
/// basis (rows are basis vectors) and `z` are the SE-output coordinates.
///
/// `z` reaches ~2^69 (reduced-basis coordinates span √κ(Q)) and basis entries
/// ~2^33, so intermediate products reach ~2^102; i128 accumulation is exact
/// and the final `x` fits i64 (Theorem 2's L³-reduced-basis bound + SE bound).
#[inline]
pub fn reconstruct_x(b_lll: &IMat8, z: &[i128; 8]) -> [i64; 8] {
    let mut x = [0i128; 8];
    for i in 0..8 {
        for j in 0..8 {
            x[j] += z[i] * b_lll[i][j] as i128;
        }
    }
    std::array::from_fn(|j| x[j] as i64)
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
