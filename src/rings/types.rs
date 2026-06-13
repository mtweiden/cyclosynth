//! Central definitions for the scalar types used by the ring implementations.
//!
//! Change `Int` or `Float` here to affect all ring arithmetic uniformly.
use i256::i256;

/// Integer coefficient type for ring elements (ZOmega, ZZeta) and SO3 scalars (R2, R4).
pub type Int = i256;

/// Float type for element-wise conversion in `to_complex()`.
/// Note that for precision epsilon <= 1e-7, mpfr is used (see lattice files).
pub type Float = f64;


// Int constants — use these instead of Int::from(n) at call sites
pub const INT_ZERO:    Int = Int::from_i8(0);
pub const INT_ONE:     Int = Int::from_i8(1);
pub const INT_TWO:     Int = Int::from_i8(2);
pub const INT_THREE:   Int = Int::from_i8(3);
pub const INT_FOUR:    Int = Int::from_i8(4);
pub const INT_NEG_ONE: Int = Int::from_i8(-1);


/// Convert an `Int` to `f64`.
#[inline]
pub fn int_to_f64(x: Int) -> Float {
    const SCALE_64: Float = 18446744073709551616.0; // 2^64
    const SCALE_128: Float = SCALE_64 * SCALE_64;
    const SCALE_192: Float = SCALE_128 * SCALE_64;
    let neg = x.is_negative();
    let limbs = if neg { -x } else { x }.to_ne_limbs();
    let r = limbs[0] as Float
        + limbs[1] as Float * SCALE_64
        + limbs[2] as Float * SCALE_128
        + limbs[3] as Float * SCALE_192;
    if neg { -r } else { r }
}
