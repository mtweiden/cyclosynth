//! Central definitions for the scalar types used by the ring implementations.
//!
//! Change `Int` here to affect all ring-integer arithmetic uniformly.
//! Floats are two-tier: primitive `f64` for the fast path, `MpFloat` (below)
//! where f64 runs out of headroom.
// Scaffolding: consumers land incrementally in later commits.
#![cfg_attr(not(test), allow(dead_code))]

use i256::i256;
use rug::ops::DivRounding;

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


/// Coefficient scalar for the generic ring elements ([`crate::rings::ZOmega`]
/// family): implemented by the fixed-width hot-path `Int` (i256) and by
/// arbitrary-precision `rug::Integer`. One ring, two widths — the fixed
/// width serves the bounded-coefficient lattice pipeline (Copy, no
/// allocation), the unbounded width serves the native z-rotation routes
/// whose norm-equation intermediates reach ~2^1000. All operations are
/// by-reference so the unbounded type never clones behind the caller's
/// back; the `Int` impl is fully `#[inline]` and monomorphizes to the
/// same code as direct arithmetic.
pub trait RingScalar: Clone + Default + PartialEq + std::fmt::Debug {
    fn from_i64(v: i64) -> Self;
    fn add(&self, o: &Self) -> Self;
    fn sub(&self, o: &Self) -> Self;
    fn mul(&self, o: &Self) -> Self;
    fn neg(&self) -> Self;
    fn is_zero(&self) -> bool;
    fn is_odd(&self) -> bool;
    /// Approximate f64 value (rounding per contract of the consumer).
    fn to_f64_approx(&self) -> f64;
    /// Trailing zero bits, or `if_zero` for the zero element.
    fn trailing_zeros_or(&self, if_zero: u32) -> u32;
    /// Exact right shift by `k` bits (caller ensures divisibility).
    fn shr_exact(&self, k: u32) -> Self;
    /// Round-to-nearest division with floor-division tie semantics.
    fn div_round(&self, den: &Self) -> Self;
    /// In-place accumulation (the unbounded width avoids a temporary).
    fn add_assign(&mut self, o: &Self);
    fn sub_assign(&mut self, o: &Self);
}

impl RingScalar for Int {
    #[inline(always)]
    fn from_i64(v: i64) -> Self {
        Int::from_i64(v)
    }
    #[inline(always)]
    fn add(&self, o: &Self) -> Self {
        *self + *o
    }
    #[inline(always)]
    fn sub(&self, o: &Self) -> Self {
        *self - *o
    }
    #[inline(always)]
    fn mul(&self, o: &Self) -> Self {
        *self * *o
    }
    #[inline(always)]
    fn neg(&self) -> Self {
        -*self
    }
    #[inline(always)]
    fn is_zero(&self) -> bool {
        *self == INT_ZERO
    }
    #[inline(always)]
    fn is_odd(&self) -> bool {
        (*self & INT_ONE) == INT_ONE
    }
    #[inline(always)]
    fn to_f64_approx(&self) -> f64 {
        int_to_f64(*self)
    }
    #[inline(always)]
    fn trailing_zeros_or(&self, if_zero: u32) -> u32 {
        if *self == INT_ZERO { if_zero } else { self.trailing_zeros() }
    }
    #[inline(always)]
    fn shr_exact(&self, k: u32) -> Self {
        *self >> k
    }
    #[inline(always)]
    fn add_assign(&mut self, o: &Self) {
        *self += *o;
    }
    #[inline(always)]
    fn sub_assign(&mut self, o: &Self) {
        *self -= *o;
    }
    #[inline(always)]
    fn div_round(&self, den: &Self) -> Self {
        // floor((2x + y) / (2y)) via truncating division corrected to floor.
        let two = INT_TWO;
        let num = *self * two + *den;
        let den2 = *den * two;
        let q = num / den2;
        let r = num % den2;
        if r != INT_ZERO && (r < INT_ZERO) != (den2 < INT_ZERO) { q - INT_ONE } else { q }
    }
}

impl RingScalar for rug::Integer {
    #[inline]
    fn from_i64(v: i64) -> Self {
        rug::Integer::from(v)
    }
    #[inline]
    fn add(&self, o: &Self) -> Self {
        rug::Integer::from(self + o)
    }
    #[inline]
    fn sub(&self, o: &Self) -> Self {
        rug::Integer::from(self - o)
    }
    #[inline]
    fn mul(&self, o: &Self) -> Self {
        rug::Integer::from(self * o)
    }
    #[inline]
    fn neg(&self) -> Self {
        rug::Integer::from(-self)
    }
    #[inline]
    fn is_zero(&self) -> bool {
        self.cmp0() == std::cmp::Ordering::Equal
    }
    #[inline]
    fn is_odd(&self) -> bool {
        rug::Integer::is_odd(self)
    }
    #[inline]
    fn to_f64_approx(&self) -> f64 {
        self.to_f64()
    }
    #[inline]
    fn trailing_zeros_or(&self, if_zero: u32) -> u32 {
        self.find_one(0).map_or(if_zero, |b| b)
    }
    #[inline]
    fn shr_exact(&self, k: u32) -> Self {
        rug::Integer::from(self >> k)
    }
    #[inline]
    fn add_assign(&mut self, o: &Self) {
        *self += o;
    }
    #[inline]
    fn sub_assign(&mut self, o: &Self) {
        *self -= o;
    }
    #[inline]
    fn div_round(&self, den: &Self) -> Self {
        let num = rug::Integer::from(self * 2u32) + den;
        let den2 = rug::Integer::from(den * 2u32);
        num.div_floor(den2)
    }
}


thread_local! {
    static SQRT2_CACHE: std::cell::RefCell<Option<(u32, MpFloat)>> =
        const { std::cell::RefCell::new(None) };
    static INV_SQRT2_CACHE: std::cell::RefCell<Option<(u32, MpFloat)>> =
        const { std::cell::RefCell::new(None) };
}

/// √2 at `prec`, cached per thread (the native routes request it per candidate).
pub(crate) fn cached_sqrt2(prec: u32) -> MpFloat {
    SQRT2_CACHE.with_borrow_mut(|c| {
        if c.as_ref().is_none_or(|(p, _)| *p != prec) {
            *c = Some((prec, MpFloat::with_val(prec, 2.0).sqrt()));
        }
        c.as_ref().expect("set above").1.clone()
    })
}

/// 1/√2 at `prec`, cached per thread.
pub(crate) fn cached_inv_sqrt2(prec: u32) -> MpFloat {
    INV_SQRT2_CACHE.with_borrow_mut(|c| {
        if c.as_ref().is_none_or(|(p, _)| *p != prec) {
            *c = Some((prec, MpFloat::with_val(prec, 0.5).sqrt()));
        }
        c.as_ref().expect("set above").1.clone()
    })
}


/// Identity shim so mixed owned/incomplete rug chains read uniformly.
pub(crate) trait CompleteId {
    fn complete_id(self) -> rug::Integer;
}
impl CompleteId for rug::Integer {
    fn complete_id(self) -> rug::Integer {
        self
    }
}

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
