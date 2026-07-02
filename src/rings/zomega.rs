//! Z[ω] — the ring of integers extended by ω = e^{iπ/4}.
//!
//! Every element has the form  a + b·ω + c·ω² + d·ω³
//! with a,b,c,d ∈ ℤ and the relation ω⁴ = −1.
//!
//! This is the coefficient ring for exactly-implementable Clifford+T unitaries
//! (entries of the SU(2) matrix live in Z[ω] / √2^k).

use num_complex::Complex64;
use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};
use super::types::{Int, INT_ZERO, INT_ONE, INT_NEG_ONE, int_to_f64};

/// An element of Z[ω], ω = e^{iπ/4}, ω⁴ = −1.
///
/// Represented as integer coefficients of the basis {1, ω, ω², ω³}.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct ZOmega {
    pub(crate) a: Int,
    pub(crate) b: Int,
    pub(crate) c: Int, // ω² = i
    pub(crate) d: Int,
}

impl ZOmega {
    pub(crate) const ZERO: Self = Self { a: INT_ZERO, b: INT_ZERO, c: INT_ZERO, d: INT_ZERO };
    pub(crate) const ONE: Self = Self { a: INT_ONE, b: INT_ZERO, c: INT_ZERO, d: INT_ZERO };
    pub(crate) const OMEGA: Self = Self { a: INT_ZERO, b: INT_ONE, c: INT_ZERO, d: INT_ZERO };
    /// i = ω²
    pub(crate) const I: Self = Self { a: INT_ZERO, b: INT_ZERO, c: INT_ONE, d: INT_ZERO };
    #[cfg_attr(not(test), allow(dead_code))] // test-only since the PyZOmega surface was removed
    pub(crate) const NEG_ONE: Self = Self { a: INT_NEG_ONE, b: INT_ZERO, c: INT_ZERO, d: INT_ZERO };

    #[inline]
    pub(crate) const fn new(a: Int, b: Int, c: Int, d: Int) -> Self {
        Self { a, b, c, d }
    }

    /// Construct from small integer coefficients, converting each via `Int::from_i32`.
    #[inline]
    pub(crate) const fn from_i32(a: i32, b: i32, c: i32, d: i32) -> Self {
        Self::new(Int::from_i32(a), Int::from_i32(b), Int::from_i32(c), Int::from_i32(d))
    }

    /// Complex conjugate: ω̄ = e^{−iπ/4} = ω⁷ = ω⁴·ω³ = −ω³.
    /// So conj(a + bω + cω² + dω³) = a − dω − cω² − bω³.
    #[inline]
    pub(crate) fn conj(self) -> Self {
        Self { a: self.a, b: -self.d, c: -self.c, d: -self.b }
    }

    /// Convert to a floating-point complex number.
    /// ω = e^{iπ/4} = (1+i)/√2, ω² = i, ω³ = (−1+i)/√2.
    pub(crate) fn to_complex(self) -> Complex64 {
        use std::f64::consts::FRAC_1_SQRT_2;
        let re = int_to_f64(self.a)
            + int_to_f64(self.b) * FRAC_1_SQRT_2
            + int_to_f64(self.c) * 0.0
            - int_to_f64(self.d) * FRAC_1_SQRT_2;
        let im = int_to_f64(self.b) * FRAC_1_SQRT_2
            + int_to_f64(self.c) * 1.0
            + int_to_f64(self.d) * FRAC_1_SQRT_2;
        Complex64::new(re, im)
    }

    /// Largest power of 2 that divides all four coefficients.
    pub(crate) fn gcd_power_of_2(self) -> u32 {
        let bits = self.a | self.b | self.c | self.d;
        if bits == INT_ZERO {
            return Int::BITS - 1;
        }
        bits.trailing_zeros()
    }

    /// Divide all coefficients by 2^shift (caller must ensure divisibility).
    #[inline]
    pub(crate) fn div2(self, shift: u32) -> Self {
        Self {
            a: self.a >> shift,
            b: self.b >> shift,
            c: self.c >> shift,
            d: self.d >> shift,
        }
    }

    /// Multiply by √2 in Z[ω], using √2 = ω − ω³
    /// (= (1+i)/√2 − (−1+i)/√2 = 2/√2 = √2).
    #[cfg_attr(not(test), allow(dead_code))] // test-only since the PyZOmega surface was removed
    pub(crate) fn mul_sqrt2(self) -> Self {
        let rhs = Self { a: INT_ZERO, b: INT_ONE, c: INT_ZERO, d: INT_NEG_ONE };
        self * rhs
    }
}

// ─── Arithmetic ───────────────────────────────────────────────────────────────

impl Add for ZOmega {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            a: self.a + rhs.a,
            b: self.b + rhs.b,
            c: self.c + rhs.c,
            d: self.d + rhs.d,
        }
    }
}

impl Sub for ZOmega {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            a: self.a - rhs.a,
            b: self.b - rhs.b,
            c: self.c - rhs.c,
            d: self.d - rhs.d,
        }
    }
}

impl Neg for ZOmega {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self { a: -self.a, b: -self.b, c: -self.c, d: -self.d }
    }
}

/// Multiplication in Z[ω] modulo ω⁴ = −1.
///
/// (a+bω+cω²+dω³)(e+fω+gω²+hω³) reduces to:
///   [1 ]: ae − bh − cg − df
///   [ω ]: af + be − ch − dg
///   [ω²]: ag + bf + ce − dh
///   [ω³]: ah + bg + cf + de
impl Mul for ZOmega {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let (a, b, c, d) = (self.a, self.b, self.c, self.d);
        let (e, f, g, h) = (rhs.a, rhs.b, rhs.c, rhs.d);
        Self {
            a: a * e - b * h - c * g - d * f,
            b: a * f + b * e - c * h - d * g,
            c: a * g + b * f + c * e - d * h,
            d: a * h + b * g + c * f + d * e,
        }
    }
}

// ─── Display ──────────────────────────────────────────────────────────────────

/// Format a list of `(coefficient, basis_symbol)` pairs as a polynomial.
/// Omits zero terms; elides coefficient ±1 when a non-empty symbol is present.
fn fmt_poly(terms: &[(Int, &str)], f: &mut fmt::Formatter<'_>) -> fmt::Result {
    let mut first = true;
    for &(coeff, sym) in terms {
        if coeff == INT_ZERO { continue; }
        let neg = coeff < INT_ZERO;
        let abs = if neg { -coeff } else { coeff };
        if first {
            if sym.is_empty() {
                write!(f, "{coeff}")?;
            } else if abs == INT_ONE {
                write!(f, "{}{sym}", if neg { "-" } else { "" })?;
            } else {
                write!(f, "{coeff}{sym}")?;
            }
            first = false;
        } else {
            let sign = if neg { " - " } else { " + " };
            if sym.is_empty() {
                write!(f, "{sign}{abs}")?;
            } else if abs == INT_ONE {
                write!(f, "{sign}{sym}")?;
            } else {
                write!(f, "{sign}{abs}{sym}")?;
            }
        }
    }
    if first { write!(f, "0")?; }
    Ok(())
}

impl fmt::Display for ZOmega {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_poly(&[
            (self.a, ""),
            (self.b, "ω"),
            (self.c, "ω²"),
            (self.d, "ω³"),
        ], f)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn near(a: Complex64, b: Complex64) -> bool {
        (a - b).norm() < 1e-12
    }

    #[test]
    fn test_to_complex_basis() {
        // 1 → 1
        assert!(near(ZOmega::ONE.to_complex(), Complex64::new(1.0, 0.0)));
        // ω → e^{iπ/4}
        let expected_omega = Complex64::from_polar(1.0, PI / 4.0);
        assert!(near(ZOmega::OMEGA.to_complex(), expected_omega));
        // ω² → i
        assert!(near(ZOmega::I.to_complex(), Complex64::new(0.0, 1.0)));
        // −1 → −1
        assert!(near(ZOmega::NEG_ONE.to_complex(), Complex64::new(-1.0, 0.0)));
    }

    #[test]
    fn test_omega4_eq_neg1() {
        // ω⁴ should equal −1
        let o = ZOmega::OMEGA;
        let o4 = o * o * o * o;
        assert_eq!(o4, ZOmega::NEG_ONE);
    }

    #[test]
    fn test_conj() {
        // conj(ω) = ω̄ = e^{−iπ/4}
        let c = ZOmega::OMEGA.conj().to_complex();
        let expected = Complex64::from_polar(1.0, -PI / 4.0);
        assert!(near(c, expected), "conj(ω) = {c}, expected {expected}");
    }

    #[test]
    fn test_mul_commutativity() {
        let x = ZOmega::from_i32(1, 2, -1, 3);
        let y = ZOmega::from_i32(-2, 1, 3, 0);
        assert_eq!(x * y, y * x);
    }

    #[test]
    fn test_mul_complex_consistent() {
        let x = ZOmega::from_i32(1, 2, -1, 3);
        let y = ZOmega::from_i32(-2, 1, 3, 0);
        let prod_ring = (x * y).to_complex();
        let prod_float = x.to_complex() * y.to_complex();
        assert!(near(prod_ring, prod_float), "ring {prod_ring} vs float {prod_float}");
    }

    #[test]
    fn test_mul_sqrt2() {
        // (ω − ω³) in float should be √2
        let sqrt2_ring = ZOmega::from_i32(1, 0, 0, 0).mul_sqrt2().to_complex();
        assert!(
            (sqrt2_ring.re - std::f64::consts::SQRT_2).abs() < 1e-12
                && sqrt2_ring.im.abs() < 1e-12,
            "mul_sqrt2 gives {sqrt2_ring}"
        );
    }

    #[test]
    fn test_add_sub() {
        let x = ZOmega::from_i32(1, 2, 3, 4);
        let y = ZOmega::from_i32(-1, 0, 1, -2);
        assert!(near((x + y).to_complex(), x.to_complex() + y.to_complex()));
        assert!(near((x - y).to_complex(), x.to_complex() - y.to_complex()));
    }

    #[test]
    fn test_gcd_power_of_2() {
        assert_eq!(ZOmega::from_i32(4, 8, 0, 12).gcd_power_of_2(), 2);
        assert_eq!(ZOmega::from_i32(1, 2, 3, 4).gcd_power_of_2(), 0);
        assert_eq!(ZOmega::from_i32(0, 0, 8, 0).gcd_power_of_2(), 3);
    }
}
