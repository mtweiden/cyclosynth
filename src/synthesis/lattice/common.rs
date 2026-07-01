//! Dimension-independent items shared between the two Lenstra-style
//! LLL+SE backends: [`super::omega`] (8D, Z[ω], Clifford+T) and
//! [`super::zeta`] (16D, Z[ζ_16], Clifford+√T). The hot-path code
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

/// Outer L²-LLL iteration caps (safety nets, never hit in regime). 16D is 5×
/// 8D because it runs far more swaps before converging (~230 vs a handful).
pub const MAX_LLL_ITERS_8D: usize = 10_000;
pub const MAX_LLL_ITERS_16D: usize = 50_000;

// ─── Numerical limits ────────────────────────────────────────────────────────

/// i256 magnitude target for the integer Gram. We pick a scale factor `B`
/// such that `round(2^B · Q[i][j])` lands at ≈ `2^TARGET_BITS`, leaving
/// headroom under `GRAM_OVERFLOW_THRESHOLD_BITS`.
pub const TARGET_BITS: u32 = 180;

/// Gram-entry overflow threshold: 2^240, 15 bits under i256::MAX. Detects
/// before wrap rather than preventing it — the check reads the entry after the
/// i256 multiply, so it's only sound because the basis grows ~1 bit/swap and
/// thus crosses 2^240 (caught, abort to fallback) before reaching 2^255. A
/// ring/dimension that could jump an entry >15 bits per update would need the
/// guard moved ahead of the multiply.
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
    // Least-significant limb first, via little-endian bytes — matches the
    // explicit endianness of the i256↔rug helpers below (no native-endian
    // assumption).
    let bytes = abs.to_le_bytes();
    let limb = |i: usize| u64::from_le_bytes(bytes[i * 8..i * 8 + 8].try_into().expect("8-byte little-endian limb"));
    // Per-limb rounding is ≤ 2^-53 relative to the total; f64 output is approximate by contract.
    #[allow(clippy::cast_precision_loss)]
    let r = (limb(0) as f64)
        + (limb(1) as f64) * SCALE_64
        + (limb(2) as f64) * SCALE_128
        + (limb(3) as f64) * SCALE_192;
    if neg { -r } else { r }
}

// ─── Adaptive precision + i256 ↔ MPFR scalar helpers (shared, dim-free) ──────

use gmp_mpfr_sys::{gmp, mpfr};
use rug::integer::Order;
use crate::rings::MpFloat;
use std::ptr::NonNull;

/// MPFR precision in bits used to construct the anisotropic Q metric.
/// `8·log₂(1/ε)` covers κ(Q) ≈ 16/ε⁴ with safety margin; floor at 100 bits
/// for moderate ε where the formula otherwise underflows.
pub fn compute_prec_q(eps: f64) -> u32 {
    let log_recip = (1.0 / eps).log2().max(1.0);
    // log₂(1/ε) ≤ 1074 for any positive f64 ε — bit counts fit u32.
    #[allow(clippy::cast_possible_truncation)]
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
pub fn compute_lu_prec(eps: f64) -> u32 {
    let log_recip = (1.0 / eps).log2().max(1.0);
    #[allow(clippy::cast_possible_truncation)] // same u32 bound as compute_prec_q
    let bits = (6.0 * log_recip).ceil() as u32;
    bits.max(96)
}

/// A zero `MpFloat` at the given precision.
#[inline]
pub fn rfz(prec: u32) -> MpFloat {
    MpFloat::with_val(prec, 0.0_f64)
}

/// An `MpFloat` holding `x` at the given precision.
#[inline]
pub fn rfv(prec: u32, x: f64) -> MpFloat {
    MpFloat::with_val(prec, x)
}

// ─── Dimension-generic zero-fill / identity constructors ─────────────────────
//
// Shared by both backends' scratch buffers; `const D` monomorphizes to the
// per-dimension shapes (8 for Z[ω], 16 for Z[ζ_16]). Callers usually let `D`
// infer from the target field/binding type.

/// A `D×D` matrix of zero `MpFloat`s at the given precision.
#[inline]
pub fn rmat_zero<const D: usize>(prec: u32) -> [[MpFloat; D]; D] {
    std::array::from_fn(|_| std::array::from_fn(|_| rfz(prec)))
}

/// A length-`D` vector of zero `MpFloat`s at the given precision.
#[inline]
pub fn rvec_zero<const D: usize>(prec: u32) -> [MpFloat; D] {
    std::array::from_fn(|_| rfz(prec))
}

/// A `D×D` matrix of i256 zeros.
#[inline]
pub fn imat_zero<const D: usize>() -> [[i256; D]; D] {
    let z = i256::from_i64(0);
    std::array::from_fn(|_| std::array::from_fn(|_| z))
}

/// The `D×D` identity basis (i64).
#[inline]
pub fn identity_basis<const D: usize>() -> [[i64; D]; D] {
    std::array::from_fn(|i| {
        let mut row = [0i64; D];
        row[i] = 1;
        row
    })
}

/// `⌈log₂|v|⌉` for a nonzero i256 (returns -1 for v = 0). Used to pick the
/// integer-Gram scale `B` via [`compute_scale_bits`].
fn i256_log2_ceil(v: &i256) -> i32 {
    let zero = i256::from_i64(0);
    if *v == zero {
        return -1;
    }
    let abs = if *v < zero { -*v } else { *v };
    // Index of the highest set bit (= ⌊log₂|v|⌋). `leading_zeros` is a
    // limb-wise count-leading-zeros, avoiding a 32-byte materialization.
    255 - abs.leading_zeros() as i32
}

// ─── Dimension-generic integer-Gram kernels ─────────────────────────────────
//
// These operate only on the exact i256 Gram / i64 basis (no Gram-Schmidt
// floats), so they are identical for d=8 (Z[ω]) and d=16 (Z[ζ_16]) modulo
// the dimension. Each backend's `lll` module keeps a thin wrapper that pulls
// the relevant scratch fields and calls these; `const D` monomorphizes to
// per-dimension code. The
// Cholesky/size-reduce routines are NOT here — they diverge (f64 GS at d=8,
// MPFR GS at d=16).

/// `true` if any Gram entry exceeds `2^GRAM_OVERFLOW_THRESHOLD_BITS`.
///
/// Runs once per LLL iteration over all D² entries, so it sidesteps
/// `i256_log2_ceil` entirely: `⌊log₂|g|⌋ > 240` is exactly `|g| ≥ 2^241`, a
/// pair of i256 comparisons against precomputed `±2^241` bounds (the first
/// short-circuits for the common in-range entry).
#[inline]
pub fn gram_overflow_check<const D: usize>(gram: &[[i256; D]; D]) -> bool {
    let pos = i256::from_i64(1).wrapping_shl(GRAM_OVERFLOW_THRESHOLD_BITS + 1);
    let neg = -pos;
    for i in 0..D {
        for j in 0..D {
            let g = gram[i][j];
            if g >= pos || g <= neg {
                return true;
            }
        }
    }
    false
}

/// Apply the basis transform `b_k -= r·b_j` to the i256 Gram in O(D) ops.
/// Math: `B_new = M·B` with `M = I − r·E_kj`, hence `G_new = M·G·Mᵀ`.
/// Two-step recurrence (row-k update, then column-k update); idempotent for
/// r=0. Caller must update the i64 basis row k separately.
#[inline]
pub fn gram_update_size_reduce<const D: usize>(
    gram: &mut [[i256; D]; D],
    k: usize,
    j: usize,
    r: i64,
) {
    if r == 0 {
        return;
    }
    let r256 = i256::from_i64(r);
    // Step 1: row k. Snapshot row j BEFORE mutating row k (new G[k][k]
    // depends on G[j][k]).
    let row_j_snapshot: [i256; D] = gram[j];
    for m in 0..D {
        gram[k][m] -= r256 * row_j_snapshot[m];
    }
    // Step 2: column k. For i = k we use the post-step-1 value of G[k][j],
    // which yields the correct G_new[k][k].
    let mut col_j_snapshot = [i256::from_i64(0); D];
    for i in 0..D {
        col_j_snapshot[i] = gram[i][j];
    }
    for i in 0..D {
        gram[i][k] -= r256 * col_j_snapshot[i];
    }
}

/// Apply the basis swap of rows a and b to the symmetric Gram: swap rows AND
/// columns. O(D) work.
#[inline]
pub fn gram_update_swap<const D: usize>(gram: &mut [[i256; D]; D], a: usize, b: usize) {
    if a == b {
        return;
    }
    gram.swap(a, b);
    for i in 0..D {
        gram[i].swap(a, b);
    }
}

/// L² INSERT (Figure 6 step 6 of Nguyen-Stehlé 2009): move basis row
/// `kappa_orig` to position `kappa_insert ≤ kappa_orig`, shifting the
/// intervening rows down. A chain of adjacent swaps keeps the i256 Gram
/// consistent. After this the GS state for rows kappa_insert..kappa_orig is
/// stale; the caller must refresh row kappa_insert via its CFA.
#[inline]
pub fn basis_insert<const D: usize>(
    gram: &mut [[i256; D]; D],
    basis: &mut [[i64; D]; D],
    kappa_orig: usize,
    kappa_insert: usize,
) {
    debug_assert!(kappa_insert <= kappa_orig);
    let mut current = kappa_orig;
    while current > kappa_insert {
        basis.swap(current, current - 1);
        gram_update_swap(gram, current, current - 1);
        current -= 1;
    }
}

/// Compute `G = B · Q_int · Bᵀ` entirely in i256, into `gram`, using
/// `temp_bq` (= B · Q_int) as intermediate. Returns `false` if any Gram
/// entry exceeds `2^GRAM_OVERFLOW_THRESHOLD_BITS` (caller aborts to
/// fallback).
#[inline]
pub fn compute_gram_full<const D: usize>(
    gram: &mut [[i256; D]; D],
    basis: &[[i64; D]; D],
    q_int: &[[i256; D]; D],
    temp_bq: &mut [[i256; D]; D],
) -> bool {
    let zero = i256::from_i64(0);

    // temp_bq[i][b] = Σ_a B[i][a] · Q_int[a][b]
    for i in 0..D {
        for b in 0..D {
            let mut acc = zero;
            for a in 0..D {
                let bi_a = i256::from_i64(basis[i][a]);
                acc += bi_a * q_int[a][b];
            }
            temp_bq[i][b] = acc;
        }
    }

    // gram[i][j] = Σ_b temp_bq[i][b] · B[j][b]
    let mut max_abs_log2: i32 = -1;
    for i in 0..D {
        for j in 0..D {
            let mut acc = zero;
            for b in 0..D {
                let bj_b = i256::from_i64(basis[j][b]);
                acc += temp_bq[i][b] * bj_b;
            }
            gram[i][j] = acc;
            let bits = i256_log2_ceil(&acc);
            if bits > max_abs_log2 {
                max_abs_log2 = bits;
            }
        }
    }
    max_abs_log2 <= GRAM_OVERFLOW_THRESHOLD_BITS as i32
}

/// Round `2^shift_bits · x` to i256 (negative `shift_bits` scales down).
/// Saturates to i256 bounds — callers pick `shift_bits` to avoid that.
pub fn rug_to_i256_scaled(x: &MpFloat, shift_bits: i32) -> i256 {
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

/// Convert an integer-valued `MpFloat` to i256. Saturates on overflow.
fn rfloat_to_i256(x: &MpFloat) -> i256 {
    let sign_neg = x.is_sign_negative();
    let abs = x.clone().abs();
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

/// Write i256 `v` into a pre-allocated `MpFloat` `dst` (zero-allocation:
/// a non-owned `mpz_t` stack view that `mpfr::set_z` reads from).
pub fn i256_to_rfloat(v: i256, dst: &mut MpFloat) {
    let zero = i256::from_i64(0);
    if v == zero {
        // SAFETY: `dst.as_raw_mut()` is a valid initialized `mpfr_t`; set_zero
        // only writes through it.
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
        // SAFETY: `limbs` is a live stack array, so its pointer is non-null.
        d: unsafe { NonNull::new_unchecked(limbs.as_mut_ptr()) },
    };
    // SAFETY: `mpz` is a well-formed read-only mpz view over `limbs` (which
    // outlives this call); set_z copies the bits out before we return.
    unsafe {
        mpfr::set_z(dst.as_raw_mut(), &mpz as *const _, mpfr::rnd_t::RNDN);
    }
    // limbs goes out of scope; mpfr::set_z has already copied the bits.
}

#[cfg(test)]
mod overflow_tests {
    use super::*;

    /// Byte-loop `i256_log2_ceil` as a reference oracle for the
    /// `leading_zeros` and direct-comparison implementations.
    fn ref_log2(v: &i256) -> i32 {
        let zero = i256::from_i64(0);
        if *v == zero {
            return -1;
        }
        let abs = if *v < zero { -*v } else { *v };
        let bytes = abs.to_le_bytes();
        let mut lz: u32 = 0;
        for byte in bytes.iter().rev() {
            if *byte == 0 {
                lz += 8;
            } else {
                lz += byte.leading_zeros();
                break;
            }
        }
        (256 - lz as i32) - 1
    }

    #[test]
    fn log2_ceil_matches_byte_loop() {
        for e in 0..255u32 {
            let p = i256::from_i64(1).wrapping_shl(e);
            assert_eq!(i256_log2_ceil(&p), e as i32, "2^{e}");
            assert_eq!(i256_log2_ceil(&p), ref_log2(&p), "2^{e}");
            assert_eq!(i256_log2_ceil(&(-p)), ref_log2(&(-p)), "-2^{e}");
        }
        assert_eq!(i256_log2_ceil(&i256::from_i64(0)), -1);

        let mut s = 0x1234_5678_9abc_def1u64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        for _ in 0..5000 {
            let mut bytes = [0u8; 32];
            for b in bytes.iter_mut() {
                *b = (next() & 0xff) as u8;
            }
            let v = i256::from_le_bytes(bytes);
            assert_eq!(i256_log2_ceil(&v), ref_log2(&v));
        }
    }

    #[test]
    fn overflow_check_matches_log2_threshold() {
        let thresh = GRAM_OVERFLOW_THRESHOLD_BITS as i32;

        for &e in &[200u32, 239, 240, 241, 242, 255] {
            let p = i256::from_i64(1).wrapping_shl(e);
            let want = ref_log2(&p) > thresh;
            let mut g = [[i256::from_i64(0); 8]; 8];
            g[3][5] = p;
            assert_eq!(gram_overflow_check(&g), want, "2^{e}");
            g[3][5] = -p;
            assert_eq!(gram_overflow_check(&g), want, "-2^{e}");
        }

        // Exact boundary: 2^241 - 1 is in range, 2^241 overflows.
        let mut g = [[i256::from_i64(0); 8]; 8];
        g[0][0] = i256::from_i64(1).wrapping_shl(241) - i256::from_i64(1);
        assert!(!gram_overflow_check(&g));
        g[0][0] = i256::from_i64(1).wrapping_shl(241);
        assert!(gram_overflow_check(&g));

        let mut s = 0xdead_beef_cafe_1234u64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        for _ in 0..3000 {
            let mut g = [[i256::from_i64(0); 8]; 8];
            for i in 0..8 {
                for j in 0..8 {
                    let shift = (next() % 250) as u32;
                    g[i][j] = i256::from_i64((next() as i64) | 1).wrapping_shl(shift);
                }
            }
            let want = (0..8).any(|i| (0..8).any(|j| ref_log2(&g[i][j]) > thresh));
            assert_eq!(gram_overflow_check(&g), want);
        }
    }
}
