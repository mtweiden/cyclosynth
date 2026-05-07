//! Items shared between the two Lenstra-style LLL+SE backends:
//! [`super::lenstra`] (8D, Z[ω], Clifford+T) and
//! [`super::lenstra_zeta`] (16D, Z[ζ_16], Clifford+√T).
//!
//! Most of each backend is dim-specialized hot-path code (using
//! `[[T; 8]; 8]` vs `[[T; 16]; 16]`), and the inner-loop performance
//! depends on those constants being known at compile time. We don't try
//! to unify those.
//!
//! What we DO unify:
//!
//! - **L²-LLL parameters** (`L2_ETA`, `L2_DELTA`, etc.) — pure algorithmic
//!   constants from Nguyen-Stehlé 2009, identical for any dimension.
//! - **Iteration / overflow caps** — `MAX_LAZY_PASSES`, `TARGET_BITS`,
//!   `GRAM_OVERFLOW_THRESHOLD_BITS`. Not strictly dim-independent
//!   numerically (the GS proof has dim-dependent constants) but the
//!   *values* we pick happen to be the same.
//! - **`LllResult`** — return-type enum of every LLL entry-point.
//! - **`compute_scale_bits`** — tiny helper, identical formula.
//!
//! What we DON'T unify:
//!
//! - Dim-specific `IntScratch` / `IntScratch16` structs.
//! - Per-iteration LLL inner loops (`lll_l2_8` vs `lll_l2_16`).
//! - Bilinear forms (1 form for Z[ω], 3 for Z[ζ_16] — different
//!   structure, not just dim).
//! - Solution reconstruction (`solution_to_u2t` vs `solution_to_u2q_d`).
//! - Q-metric construction (different ring embeddings).
//!
//! When future fplll-inspired optimizations land (adaptive precision
//! ladder, prefix-sum Lovász cascade), they go in subordinate files of
//! this module so both backends pick them up.

// ─── L²-LLL parameters (Nguyen-Stehlé 2009, Figures 5-7) ─────────────────────

/// L² parameter η: relaxed size-reduction factor. Must satisfy 1/2 < η < √δ.
/// Per Figure 7 of NS09, (δ=0.75, η=0.55) supports d ≤ 11 in f64.
pub const L2_ETA: f64 = 0.55;

/// L² parameter δ: Lovász factor. (δ=0.75 is the classical LLL value.)
pub const L2_DELTA: f64 = 0.75;

/// δ̄ = (δ + 1) / 2 (used by the main loop's Lovász test, Figure 6 step 2).
pub const L2_DELTA_BAR: f64 = (L2_DELTA + 1.0) / 2.0;

/// η̄ = (η + 1/2) / 2 (used by lazy size-reduction, Figure 5 step 1).
pub const L2_ETA_BAR: f64 = (L2_ETA + 0.5) / 2.0;

/// Hard cap on lazy-size-reduce iterations per κ. Empirically converges in
/// 1-3 passes; the cap is a safety net against pathological inputs.
pub const MAX_LAZY_PASSES: usize = 32;

// ─── Numerical limits ────────────────────────────────────────────────────────

/// i256 magnitude target for the integer Gram. We pick a scale factor `B`
/// such that `round(2^B · Q[i][j])` lands at ≈ `2^TARGET_BITS`, leaving
/// headroom under `GRAM_OVERFLOW_THRESHOLD_BITS`.
pub const TARGET_BITS: u32 = 180;

/// Threshold for Gram-entry overflow detection: 2^240, leaving 16-bit
/// margin to i256::MAX. The safe operating range is roughly
/// `max(|B|)² · max(|Q_int|) · d ≤ 2^240`.
pub const GRAM_OVERFLOW_THRESHOLD_BITS: u32 = 240;

/// Compute the bit-shift `B` such that `round(2^B · Q[i][j])` lands in i256
/// with max entry ≈ `2^TARGET_BITS`. Same formula for both backends.
#[inline]
pub fn compute_scale_bits(max_q_log2: i32) -> i32 {
    TARGET_BITS as i32 - max_q_log2
}

// ─── Result type ─────────────────────────────────────────────────────────────

/// Outcome of an LLL run. Convergence with a unimodular basis on success;
/// overflow or iteration-cap on failure.
///
/// `GramOverflow`: the caller should reject this prefix and let the
/// dispatcher advance.
///
/// `IterCap`: indicates cycling or near-boundary precision noise. The
/// basis is still a valid lattice basis (just possibly under-reduced),
/// so most callers treat this as "convergence with reduced quality" and
/// proceed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LllResult {
    /// LLL converged within the iteration cap and no overflow.
    Converged,
    /// A Gram entry's magnitude exceeded `GRAM_OVERFLOW_THRESHOLD_BITS`.
    GramOverflow,
    /// Reached the iteration cap without convergence.
    IterCap,
}
