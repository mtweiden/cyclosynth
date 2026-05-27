//! Central definitions for the scalar types used by the ring implementations.
//!
//! Change `Int` or `Float` here to affect all ring arithmetic uniformly.
use i256::i256;

/// Integer coefficient type for ring elements (ZOmega, ZZeta) and SO3 scalars (R2, R4).
///
/// `i64`  — sufficient for circuits with fewer than ~32 T gates in any product path.
/// `i128` — sufficient for circuits with fewer than ~64 T gates in any product path.
///           Use this when working with circuits up to ~140 random gates from {H, S, T}.
/// `i256` — TODO
//pub type Int = i128;
pub type Int = i256;

/// Float type for element-wise conversion in `to_complex()`.
/// Changing to `f32` also requires updating the `RingElem::to_complex` return type.
/// TODO: add support for much higher precisions
pub type Float = f64;

// Int constants — use these instead of Int::from(n) at call sites
pub const INT_ZERO: Int = Int::from_i8(0);
pub const INT_ONE: Int = Int::from_i8(1);
pub const INT_TWO: Int = Int::from_i8(2);
pub const INT_THREE: Int = Int::from_i8(3);
pub const INT_FOUR: Int = Int::from_i8(4);
pub const INT_NEG_ONE: Int = Int::from_i8(-1);

/// Convert an `Int` to `f64`. Values are assumed small enough to fit in i128.
#[inline]
pub fn int_to_f64(x: Int) -> Float {
    x.as_i128() as Float
}
