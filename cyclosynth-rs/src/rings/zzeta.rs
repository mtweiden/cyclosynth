//! Z[ζ] — the ring of integers extended by ζ = e^{iπ/8}.
//!
//! Every element has the form  a + b·ζ + c·ζ² + d·ζ³ + e·ζ⁴ + f·ζ⁵ + g·ζ⁶ + h·ζ⁷
//! with a,b,c,d,e,f,g,h ∈ ℤ and the relation ζ^8 = −1.
//!
//! This is the coefficient ring for exactly-implementable Clifford+√T unitaries.
//! Note that ZOmega embeds into ZZeta via ω = ζ² (odd-index coefficients are 0).

use num_complex::Complex64;
use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};
use super::types::{Int, Float, INT_ZERO, INT_ONE, INT_NEG_ONE, int_to_f64};

/// An element of Z[ζ], ζ = e^{iπ/8}, ζ^8 = −1.
///
/// Represented as integer coefficients of the basis {1, ζ, ζ², ζ³, ζ⁴, ζ⁵, ζ⁶, ζ⁷}.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct ZZeta {
    pub a: Int, // coefficient of 1   = ζ⁰
    pub b: Int, // coefficient of ζ
    pub c: Int, // coefficient of ζ²  = ω
    pub d: Int, // coefficient of ζ³
    pub e: Int, // coefficient of ζ⁴  = i
    pub f: Int, // coefficient of ζ⁵
    pub g: Int, // coefficient of ζ⁶
    pub h: Int, // coefficient of ζ⁷
}

impl ZZeta {
    pub const ZERO: Self = Self { a: INT_ZERO, b: INT_ZERO, c: INT_ZERO, d: INT_ZERO, e: INT_ZERO, f: INT_ZERO, g: INT_ZERO, h: INT_ZERO };
    pub const ONE:  Self = Self { a: INT_ONE,  b: INT_ZERO, c: INT_ZERO, d: INT_ZERO, e: INT_ZERO, f: INT_ZERO, g: INT_ZERO, h: INT_ZERO };
    /// ζ itself
    pub const ZETA: Self = Self { a: INT_ZERO, b: INT_ONE,  c: INT_ZERO, d: INT_ZERO, e: INT_ZERO, f: INT_ZERO, g: INT_ZERO, h: INT_ZERO };
    /// ζ² = ω (the Clifford+T generator)
    pub const OMEGA: Self = Self { a: INT_ZERO, b: INT_ZERO, c: INT_ONE,  d: INT_ZERO, e: INT_ZERO, f: INT_ZERO, g: INT_ZERO, h: INT_ZERO };
    /// i = ζ⁴
    pub const I: Self = Self { a: INT_ZERO, b: INT_ZERO, c: INT_ZERO, d: INT_ZERO, e: INT_ONE,  f: INT_ZERO, g: INT_ZERO, h: INT_ZERO };
    /// −1
    pub const NEG_ONE: Self = Self { a: INT_NEG_ONE, b: INT_ZERO, c: INT_ZERO, d: INT_ZERO, e: INT_ZERO, f: INT_ZERO, g: INT_ZERO, h: INT_ZERO };
    /// -i = -ζ⁴
    pub const NEG_I: Self = Self { a: INT_ZERO, b: INT_ZERO, c: INT_ZERO, d: INT_ZERO, e: INT_NEG_ONE, f: INT_ZERO, g: INT_ZERO, h: INT_ZERO };

    #[inline]
    pub const fn new(a: Int, b: Int, c: Int, d: Int, e: Int, f: Int, g: Int, h: Int) -> Self {
        Self { a, b, c, d, e, f, g, h }
    }

    /// Construct from small integer coefficients, converting each via `Int::from_i32`.
    #[inline]
    pub const fn from_i32(a: i32, b: i32, c: i32, d: i32, e: i32, f: i32, g: i32, h: i32) -> Self {
        Self::new(
            Int::from_i32(a), Int::from_i32(b), Int::from_i32(c), Int::from_i32(d),
            Int::from_i32(e), Int::from_i32(f), Int::from_i32(g), Int::from_i32(h),
        )
    }

    /// The squared norm of the complex number (computed as Int).
    #[inline]
    pub fn norm_sqr(self) -> Int {
        self.a * self.a + self.b * self.b + self.c * self.c + self.d * self.d
            + self.e * self.e + self.f * self.f + self.g * self.g + self.h * self.h
    }

    /// Coefficient of ζ^k, k = 0..7.
    #[inline]
    pub fn coeff(self, k: usize) -> Int {
        match k {
            0 => self.a, 1 => self.b, 2 => self.c, 3 => self.d,
            4 => self.e, 5 => self.f, 6 => self.g, 7 => self.h,
            _ => panic!("ZZeta::coeff: index {k} out of range"),
        }
    }

    /// Complex conjugate: ζ̄ = e^{−iπ/8} = ζ^{−1} = ζ^{15} = −ζ^7.
    ///
    /// conj_coeffs[0] = coeffs[0],
    /// conj_coeffs[k] = −coeffs[8−k]  for k = 1..7.
    pub fn conj(self) -> Self {
        Self {
            a:  self.a,
            b: -self.h,
            c: -self.g,
            d: -self.f,
            e: -self.e,
            f: -self.d,
            g: -self.c,
            h: -self.b,
        }
    }

    /// Scalar multiply by an integer.
    #[inline]
    pub fn scale(self, s: Int) -> Self {
        Self {
            a: self.a * s, b: self.b * s, c: self.c * s, d: self.d * s,
            e: self.e * s, f: self.f * s, g: self.g * s, h: self.h * s,
        }
    }

    /// Convert to a floating-point complex number.
    pub fn to_complex(self) -> Complex64 {
        use std::f64::consts::PI;
        let zeta = |k: u32| Complex64::from_polar(1.0, PI * k as Float / 8.0);
        int_to_f64(self.a) * zeta(0)
            + int_to_f64(self.b) * zeta(1)
            + int_to_f64(self.c) * zeta(2)
            + int_to_f64(self.d) * zeta(3)
            + int_to_f64(self.e) * zeta(4)
            + int_to_f64(self.f) * zeta(5)
            + int_to_f64(self.g) * zeta(6)
            + int_to_f64(self.h) * zeta(7)
    }

    /// Largest power of 2 dividing all coefficients (for normalization).
    pub fn gcd_power_of_2(self) -> u32 {
        let bits = self.a | self.b | self.c | self.d | self.e | self.f | self.g | self.h;
        if bits == INT_ZERO { Int::BITS - 1 } else { bits.trailing_zeros() }
    }

    /// Divide all coefficients by 2^shift.
    #[inline]
    pub fn div2(self, shift: u32) -> Self {
        Self {
            a: self.a >> shift, b: self.b >> shift, c: self.c >> shift, d: self.d >> shift,
            e: self.e >> shift, f: self.f >> shift, g: self.g >> shift, h: self.h >> shift,
        }
    }

    /// Multiply by √2 = ζ² − ζ⁶  (since e^{iπ/4} − e^{6iπ/8} = (1+i)/√2 − (−1+i)/√2 = √2).
    pub fn mul_sqrt2(self) -> Self {
        // self * (0 + 0·ζ + 1·ζ² + 0·ζ³ + 0·ζ⁴ + 0·ζ⁵ + (−1)·ζ⁶ + 0·ζ⁷)
        let rhs = Self::from_i32(0, 0, 1, 0, 0, 0, -1, 0);
        self * rhs
    }

    /// Embed a ZOmega element: ω = ζ², so (a + bω + cω² + dω³) → (a + bζ² + cζ⁴ + dζ⁶).
    pub fn from_zomega(a: Int, b: Int, c: Int, d: Int) -> Self {
        Self::new(a, INT_ZERO, b, INT_ZERO, c, INT_ZERO, d, INT_ZERO)
    }
}

// ─── Arithmetic ───────────────────────────────────────────────────────────────

impl Add for ZZeta {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            a: self.a + rhs.a, b: self.b + rhs.b, c: self.c + rhs.c, d: self.d + rhs.d,
            e: self.e + rhs.e, f: self.f + rhs.f, g: self.g + rhs.g, h: self.h + rhs.h,
        }
    }
}

impl Sub for ZZeta {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            a: self.a - rhs.a, b: self.b - rhs.b, c: self.c - rhs.c, d: self.d - rhs.d,
            e: self.e - rhs.e, f: self.f - rhs.f, g: self.g - rhs.g, h: self.h - rhs.h,
        }
    }
}

impl Neg for ZZeta {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self {
            a: -self.a, b: -self.b, c: -self.c, d: -self.d,
            e: -self.e, f: -self.f, g: -self.g, h: -self.h,
        }
    }
}

/// Multiplication in Z[ζ] modulo ζ^8 = −1.
///
/// ζ^i · ζ^j = ζ^{i+j}; if i+j ≥ 8: ζ^{i+j} = −ζ^{i+j−8}.
///
/// Written out explicitly (p = self, q = rhs):
///   result[k] = Σ_{i+j≡k (mod 8), i+j<8} p_i·q_j  −  Σ_{i+j≡k (mod 8), i+j≥8} p_i·q_j
impl Mul for ZZeta {
    type Output = Self;
    #[inline]
    fn mul(self, q: Self) -> Self {
        // Inline all 64 terms (p = self, q = rhs) for maximum clarity and speed.
        // Convolution mod ζ^8=−1: out[k] = Σ_{i+j=k} p_i·q_j − Σ_{i+j=k+8} p_i·q_j
        let (p0,p1,p2,p3,p4,p5,p6,p7) = (self.a,self.b,self.c,self.d,self.e,self.f,self.g,self.h);
        let (q0,q1,q2,q3,q4,q5,q6,q7) = (q.a, q.b, q.c, q.d, q.e, q.f, q.g, q.h);

        // Contributions to each output degree, with sign flip for deg ≥ 8:
        Self {
            a: p0*q0 - p1*q7 - p2*q6 - p3*q5 - p4*q4 - p5*q3 - p6*q2 - p7*q1,
            b: p0*q1 + p1*q0 - p2*q7 - p3*q6 - p4*q5 - p5*q4 - p6*q3 - p7*q2,
            c: p0*q2 + p1*q1 + p2*q0 - p3*q7 - p4*q6 - p5*q5 - p6*q4 - p7*q3,
            d: p0*q3 + p1*q2 + p2*q1 + p3*q0 - p4*q7 - p5*q6 - p6*q5 - p7*q4,
            e: p0*q4 + p1*q3 + p2*q2 + p3*q1 + p4*q0 - p5*q7 - p6*q6 - p7*q5,
            f: p0*q5 + p1*q4 + p2*q3 + p3*q2 + p4*q1 + p5*q0 - p6*q7 - p7*q6,
            g: p0*q6 + p1*q5 + p2*q4 + p3*q3 + p4*q2 + p5*q1 + p6*q0 - p7*q7,
            h: p0*q7 + p1*q6 + p2*q5 + p3*q4 + p4*q3 + p5*q2 + p6*q1 + p7*q0,
        }
    }
}

// ─── Display ──────────────────────────────────────────────────────────────────

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

impl fmt::Display for ZZeta {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_poly(&[
            (self.a, ""),
            (self.b, "ζ"),
            (self.c, "ζ²"),
            (self.d, "ζ³"),
            (self.e, "ζ⁴"),
            (self.f, "ζ⁵"),
            (self.g, "ζ⁶"),
            (self.h, "ζ⁷"),
        ], f)
    }
}

// ─── PyO3 ─────────────────────────────────────────────────────────────────────

use pyo3::prelude::*;

/// Python-facing ZZeta.
#[pyclass(name = "ZZeta", frozen)]
pub struct PyZZeta {
    pub inner: ZZeta,
}

impl PyZZeta {
    pub fn to_inner(&self) -> ZZeta { self.inner }

    pub fn from_inner(inner: ZZeta) -> Self { Self { inner } }
}

#[pymethods]
impl PyZZeta {
    /// Python always passes 64-bit integers; cast to `Int` (may be wider than i64).
    #[new]
    fn new(a: i64, b: i64, c: i64, d: i64, e: i64, f: i64, g: i64, h: i64) -> Self {
        Self { inner: ZZeta::new(
            Int::from_i64(a), Int::from_i64(b), Int::from_i64(c), Int::from_i64(d),
            Int::from_i64(e), Int::from_i64(f), Int::from_i64(g), Int::from_i64(h),
        )}
    }

    fn __add__(&self, other: &PyZZeta) -> Self { Self { inner: self.inner + other.inner } }
    fn __sub__(&self, other: &PyZZeta) -> Self { Self { inner: self.inner - other.inner } }
    fn __mul__(&self, other: &PyZZeta) -> Self { Self { inner: self.inner * other.inner } }
    fn __neg__(&self) -> Self { Self { inner: -self.inner } }

    fn to_complex(&self) -> (f64, f64) { let c = self.inner.to_complex(); (c.re, c.im) }
    fn mul_sqrt2(&self) -> Self { Self { inner: self.inner.mul_sqrt2() } }
    fn gcd_power_of_2(&self) -> u32 { self.inner.gcd_power_of_2() }

    #[staticmethod] fn one()     -> Self { Self { inner: ZZeta::ONE } }
    #[staticmethod] fn i()       -> Self { Self { inner: ZZeta::I } }
    #[staticmethod] fn zero()    -> Self { Self { inner: ZZeta::ZERO } }
    #[staticmethod] fn neg_one() -> Self { Self { inner: ZZeta::NEG_ONE } }

    fn __repr__(&self) -> String {
        let z = &self.inner;
        format!("ZZeta({},{},{},{},{},{},{},{})",
                z.a, z.b, z.c, z.d, z.e, z.f, z.g, z.h)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rings::zomega::ZOmega;
    use std::f64::consts::PI;

    fn near(a: Complex64, b: Complex64) -> bool {
        (a - b).norm() < 1e-12
    }

    #[test]
    fn test_to_complex_basis() {
        assert!(near(ZZeta::ONE.to_complex(), Complex64::new(1.0, 0.0)));
        let expected_zeta = Complex64::from_polar(1.0, PI / 8.0);
        assert!(near(ZZeta::ZETA.to_complex(), expected_zeta));
        assert!(near(ZZeta::I.to_complex(), Complex64::new(0.0, 1.0)));
        assert!(near(ZZeta::NEG_ONE.to_complex(), Complex64::new(-1.0, 0.0)));
    }

    #[test]
    fn test_zeta8_eq_neg1() {
        let z  = ZZeta::ZETA;
        let z8 = z * z * z * z * z * z * z * z;
        assert_eq!(z8, ZZeta::NEG_ONE, "ζ⁸ should equal −1");
    }

    #[test]
    fn test_conj() {
        let c        = ZZeta::ZETA.conj().to_complex();
        let expected = Complex64::from_polar(1.0, -PI / 8.0);
        assert!(near(c, expected), "conj(ζ) = {c}, expected {expected}");
    }

    #[test]
    fn test_mul_complex_consistent() {
        let x = ZZeta::from_i32(1, 2, -1, 3, 0, -2, 1, 0);
        let y = ZZeta::from_i32(-2, 1, 3, 0, 1, 0, -1, 2);
        let prod_ring  = (x * y).to_complex();
        let prod_float = x.to_complex() * y.to_complex();
        assert!(near(prod_ring, prod_float), "ring {prod_ring} vs float {prod_float}");
    }

    #[test]
    fn test_mul_commutative() {
        let x = ZZeta::from_i32(1, 2, -1, 3, 0, -2, 1, 0);
        let y = ZZeta::from_i32(-2, 1, 3, 0, 1, 0, -1, 2);
        assert_eq!(x * y, y * x);
    }

    #[test]
    fn test_zomega_embedding_consistent() {
        let z_omega = ZOmega::from_i32(1, 2, -1, 3);
        let z_zeta  = ZZeta::from_zomega(Int::from_i32(1), Int::from_i32(2), Int::from_i32(-1), Int::from_i32(3));
        assert!(
            near(z_omega.to_complex(), z_zeta.to_complex()),
            "omega={}, zeta={}",
            z_omega.to_complex(), z_zeta.to_complex()
        );
    }

    #[test]
    fn test_mul_sqrt2() {
        let sqrt2 = ZZeta::ONE.mul_sqrt2().to_complex();
        assert!(
            (sqrt2.re - std::f64::consts::SQRT_2).abs() < 1e-12 && sqrt2.im.abs() < 1e-12,
            "mul_sqrt2(1) = {sqrt2}"
        );
    }
}
