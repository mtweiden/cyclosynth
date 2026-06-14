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

// ─── Adaptive precision + i256 ↔ MPFR scalar helpers (shared, dim-free) ──────

use crate::rings::Float;
use gmp_mpfr_sys::{gmp, mpfr};
use rug::{integer::Order, Float as RFloat};
use std::ptr::NonNull;

/// MPFR precision in bits used to construct the anisotropic Q metric.
/// `8·log₂(1/ε)` covers κ(Q) ≈ 16/ε⁴ with safety margin; floor at 100 bits
/// for moderate ε where the formula otherwise underflows.
pub fn compute_prec_q(eps: Float) -> u32 {
    let log_recip = (1.0 / eps).log2().max(1.0);
    let bits = (8.0 * log_recip).ceil() as u32;
    bits.max(100)
}

/// MPFR precision used by the cap-center LU solve, scaled with ε.
///
/// The basis `B` has det=±1 but its entries grow with ε (up to ~2¹⁵ at
/// ε=1e-5, ~2⁴¹ at ε=1e-8). Partial-pivoting LU on this basis can develop
/// pivot ratios up to ~max(|B|)^(d-1) in pathological cases — usually
/// much tighter, but enough to consume meaningful precision at deep ε.
/// Empirically at ε=1e-8 a 96-bit LU loses enough precision in z_c that SE
/// misses the canonical-lde solution; 6·log₂(1/ε) bits leaves margin (75% of
/// `prec_q`, so each MPFR op in the LU is ~1.3× cheaper).
pub fn compute_lu_prec(eps: Float) -> u32 {
    let log_recip = (1.0 / eps).log2().max(1.0);
    let bits = (6.0 * log_recip).ceil() as u32;
    bits.max(96)
}

/// A zero `RFloat` at the given precision.
#[inline]
pub fn rfz(prec: u32) -> RFloat {
    RFloat::with_val(prec, 0.0_f64)
}

/// An `RFloat` holding `x` at the given precision.
#[inline]
pub fn rfv(prec: u32, x: f64) -> RFloat {
    RFloat::with_val(prec, x)
}

/// `⌈log₂|v|⌉` for a nonzero i256 (returns -1 for v = 0). Used to pick the
/// integer-Gram scale `B` via [`compute_scale_bits`].
pub fn i256_log2_ceil(v: &i256) -> i32 {
    let zero = i256::from_i64(0);
    if *v == zero {
        return -1;
    }
    let abs = if *v < zero { -*v } else { *v };
    let bytes = abs.to_le_bytes();
    let mut leading_zeros: u32 = 0;
    for byte in bytes.iter().rev() {
        if *byte == 0 {
            leading_zeros += 8;
        } else {
            leading_zeros += byte.leading_zeros();
            break;
        }
    }
    (256 - leading_zeros as i32) - 1
}

/// Round `2^shift_bits · x` to i256 (negative `shift_bits` scales down).
/// Saturates to i256 bounds — callers pick `shift_bits` to avoid that.
pub fn rug_to_i256_scaled(x: &RFloat, shift_bits: i32) -> i256 {
    if x.is_zero() {
        return i256::from_i64(0);
    }
    let mut scaled = x.clone();
    if shift_bits >= 0 {
        scaled <<= shift_bits as u32;
    } else {
        scaled >>= (-shift_bits) as u32;
    }
    scaled.round_mut();
    rfloat_to_i256(&scaled)
}

/// Convert an integer-valued `RFloat` to i256. Saturates on overflow.
fn rfloat_to_i256(x: &RFloat) -> i256 {
    let sign_neg = x.is_sign_negative();
    let abs = x.clone().abs();
    // Fast path: fits in i64.
    if abs <= rug::Float::with_val(64, i64::MAX as f64) {
        let v = abs.to_f64() as i64;
        let res = i256::from_i64(v);
        return if sign_neg { -res } else { res };
    }
    let int = match abs.to_integer() {
        Some(i) => i,
        None => return i256::from_i64(0),
    };
    if int.significant_bits() > 254 {
        return if sign_neg { i256::MIN } else { i256::MAX };
    }
    let mut limbs = [0u64; 4];
    int.write_digits(&mut limbs, Order::Lsf);
    let mut bytes = [0u8; 32];
    for (idx, limb) in limbs.iter().enumerate() {
        bytes[idx * 8..(idx + 1) * 8].copy_from_slice(&limb.to_le_bytes());
    }
    let val = i256::from_le_bytes(bytes);
    if sign_neg { -val } else { val }
}

/// Write i256 `v` into a pre-allocated `RFloat` `dst` (zero-allocation:
/// a non-owned `mpz_t` stack view that `mpfr::set_z` reads from).
pub fn i256_to_rfloat(v: i256, dst: &mut RFloat) {
    let zero = i256::from_i64(0);
    if v == zero {
        unsafe { mpfr::set_zero(dst.as_raw_mut(), 0) };
        return;
    }
    let neg = v < zero;
    let abs = if neg { -v } else { v };
    let bytes = abs.to_le_bytes();
    let mut limbs: [gmp::limb_t; 4] = std::array::from_fn(|i| {
        let mut buf = [0u8; 8];
        buf.copy_from_slice(&bytes[i * 8..(i + 1) * 8]);
        u64::from_le_bytes(buf) as gmp::limb_t
    });
    // Trim trailing-zero limbs to determine `_mp_size`.
    let mut size: i32 = 4;
    while size > 0 && limbs[(size - 1) as usize] == 0 {
        size -= 1;
    }
    let signed_size = if neg { -size } else { size };
    let mpz = gmp::mpz_t {
        alloc: 0,
        size: signed_size,
        d: unsafe { NonNull::new_unchecked(limbs.as_mut_ptr()) },
    };
    unsafe {
        mpfr::set_z(dst.as_raw_mut(), &mpz as *const _, mpfr::rnd_t::RNDN);
    }
    // limbs goes out of scope; mpfr::set_z has already copied the bits.
}
