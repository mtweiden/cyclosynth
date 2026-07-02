//! Central definitions for the scalar types used by the ring implementations.
//!
//! Change `Int` here to affect all ring-integer arithmetic uniformly.
//! Floats are two-tier: primitive `f64` for the fast path, `MpFloat` (below)
//! where f64 runs out of headroom.
use i256::i256;

/// Integer coefficient type for ring elements (ZOmega, ZZeta) and SO3 scalars (R2, R4).
pub(crate) type Int = i256;


/// Arbitrary-precision (MPFR) float, used wherever f64 runs out of headroom
/// (the lattice Q-metric, Gram-Schmidt, deep-ε verification). Precision is set
/// per use site (see `GS_PREC`, `compute_prec_q`), not fixed.
pub(crate) type MpFloat = rug::Float;


// Int constants — use these instead of Int::from(n) at call sites
pub(crate) const INT_ZERO:    Int = Int::from_i8(0);
pub(crate) const INT_ONE:     Int = Int::from_i8(1);
pub(crate) const INT_TWO:     Int = Int::from_i8(2);
pub(crate) const INT_FOUR:    Int = Int::from_i8(4);
pub(crate) const INT_NEG_ONE: Int = Int::from_i8(-1);


/// Convert an `Int` to `f64`.
// Per-limb rounding is ≤ 2^-53 relative to the total; f64 output is approximate by contract.
#[allow(clippy::cast_precision_loss)]
#[inline]
pub(crate) fn int_to_f64(x: Int) -> f64 {
    const SCALE_64: f64 = 18446744073709551616.0; // 2^64
    const SCALE_128: f64 = SCALE_64 * SCALE_64;
    const SCALE_192: f64 = SCALE_128 * SCALE_64;
    let neg = x.is_negative();
    let limbs = if neg { -x } else { x }.to_ne_limbs();
    let r = limbs[0] as f64
        + limbs[1] as f64 * SCALE_64
        + limbs[2] as f64 * SCALE_128
        + limbs[3] as f64 * SCALE_192;
    if neg { -r } else { r }
}
