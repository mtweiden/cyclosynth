//! Dimension-independent items shared between the two Lenstra-style
//! LLL+SE backends: [`super::lattice`] (8D, Z[ω], Clifford+T) and
//! [`super::lattice::zeta`] (16D, Z[ζ_16], Clifford+√T). The hot-path code
//! (dim-specialized `[[T; 8]]` vs `[[T; 16]]` loops, bilinear forms,
//! ring-specific Q-metric and reconstruction) stays separate per backend
//! so the dimension is a compile-time constant. Only the L²-LLL
//! parameters, iteration/overflow caps, `LllResult`, and the i256/scale
//! helpers are unified here.

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

/// Outcome of an LLL run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LllResult {
    Converged,
    /// A Gram entry exceeded `GRAM_OVERFLOW_THRESHOLD_BITS`; the caller
    /// should reject this prefix and let the dispatcher advance.
    GramOverflow,
    /// Iteration cap hit (cycling or near-boundary noise). The basis is
    /// still valid, just possibly under-reduced — most callers proceed.
    IterCap,
}

use i256::i256;

/// Convert i256 to f64, summing limbs low-to-high so low bits round, not
/// high. Take abs() in i256 FIRST: a two's-complement conversion of a
/// small negative (high limb `0xFF…FF`) subtracts two near-equal large
/// f64s and loses all precision below ~2^140.
#[inline]
pub fn i256_to_f64(v: i256) -> f64 {
    const SCALE_64: f64 = 18446744073709551616.0; // 2^64
    const SCALE_128: f64 = SCALE_64 * SCALE_64;
    const SCALE_192: f64 = SCALE_128 * SCALE_64;
    let neg = v.is_negative();
    let abs = if neg { -v } else { v };
    let limbs = abs.to_ne_limbs();
    let r = (limbs[0] as f64)
        + (limbs[1] as f64) * SCALE_64
        + (limbs[2] as f64) * SCALE_128
        + (limbs[3] as f64) * SCALE_192;
    if neg { -r } else { r }
}
