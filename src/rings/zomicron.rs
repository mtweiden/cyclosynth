//! Z[ξ] — the ring of integers extended by ξ = e^{iπ/6}.
//!
//! Every element has the form  a + b·ξ + c·ξ² + d·ξ³
//! with a,b,c,d ∈ ℤ and the relation ξ⁴ = ξ² − 1  (Φ₁₂(ξ) = 0).
//!
//! Useful facts: ξ³ = i, ξ⁶ = −1, √3 = 2ξ − ξ³.
//! The Gaussian integers Z[i] embed via i ↦ ξ³.

use num_complex::Complex64;
use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};
use super::types::{Int, Float, INT_ZERO, INT_ONE, INT_NEG_ONE, int_to_f64};

// ─── Constants for σ-matrix and Gram construction ──────────────────────────────

/// The 4×4 Gram matrix of the bullet-pair embedding ℤ[ξ] → ℝ⁴ in the
/// cyclotomic basis {1, ξ, ξ², ξ³}. Computed as ΣᵤᵀΣᵤ where the rows of Σᵤ are
/// (Re(z), Im(z), Re(z•), Im(z•)) viewed as ℝ-linear functions of (a, b, c, d).
///
/// This is the quadratic form xᵀGx that the Phase-2 CVP lattice uses for n=6.
/// Its off-diagonal 1s couple basis elements at "complementary" angles
/// (1 ↔ ξ², ξ ↔ ξ³). Determinant 9.
///
/// Concretely, for x = (a, b, c, d):
///   xᵀ G x = 2(a² + b² + c² + d²) + 2(ac + bd).
pub const SIGMA_GRAM_U: [[i64; 4]; 4] = [
    [2, 0, 1, 0],
    [0, 2, 0, 1],
    [1, 0, 2, 0],
    [0, 1, 0, 2],
];

/// 4×4 integer matrix representation of the bullet map on cyclotomic
/// coordinates (a, b, c, d) ↦ (a + c, −b, −c, b + d).
///
/// Provided for callers that need an explicit matrix form for batch operations.
/// For single elements, [`ZOmicron::bullet`] is faster.
pub const BULLET_MATRIX: [[i64; 4]; 4] = [
    [ 1,  0,  1,  0],
    [ 0, -1,  0,  0],
    [ 0,  0, -1,  0],
    [ 0,  1,  0,  1],
];

/// An element of Z[ξ], ξ = e^{iπ/6}, ξ⁴ = ξ² − 1.
///
/// Represented as integer coefficients of the basis {1, ξ, ξ², ξ³}.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct ZOmicron {
    pub a: Int, // coefficient of 1   = ξ⁰
    pub b: Int, // coefficient of ξ
    pub c: Int, // coefficient of ξ²
    pub d: Int, // coefficient of ξ³  = i
}

impl ZOmicron {
    pub const ZERO: Self = Self { a: INT_ZERO, b: INT_ZERO, c: INT_ZERO, d: INT_ZERO };
    pub const ONE:  Self = Self { a: INT_ONE,  b: INT_ZERO, c: INT_ZERO, d: INT_ZERO };
    /// ξ itself
    pub const XI: Self = Self { a: INT_ZERO, b: INT_ONE,  c: INT_ZERO, d: INT_ZERO };
    /// i = ξ³
    pub const I: Self = Self { a: INT_ZERO, b: INT_ZERO, c: INT_ZERO, d: INT_ONE };
    /// −1
    pub const NEG_ONE: Self = Self { a: INT_NEG_ONE, b: INT_ZERO, c: INT_ZERO, d: INT_ZERO };
    /// −i = −ξ³
    pub const NEG_I: Self = Self { a: INT_ZERO, b: INT_ZERO, c: INT_ZERO, d: INT_NEG_ONE };

    #[inline]
    pub const fn new(a: Int, b: Int, c: Int, d: Int) -> Self {
        Self { a, b, c, d }
    }

    /// Construct from small integer coefficients, converting each via `Int::from_i32`.
    #[inline]
    pub const fn from_i32(a: i32, b: i32, c: i32, d: i32) -> Self {
        Self::new(
            Int::from_i32(a), Int::from_i32(b), Int::from_i32(c), Int::from_i32(d),
        )
    }

    /// The squared Euclidean norm on coefficients (sum of squares).
    #[inline]
    pub fn coeff_l2_squared(self) -> Int {
        self.a * self.a + self.b * self.b + self.c * self.c + self.d * self.d
    }

    /// Coefficient of ξ^k, k = 0..3.
    #[inline]
    pub fn coeff(self, k: usize) -> Int {
        match k {
            0 => self.a, 1 => self.b, 2 => self.c, 3 => self.d,
            _ => panic!("ZOmicron::coeff: index {k} out of range"),
        }
    }

    /// Complex conjugate: ξ̄ = e^{−iπ/6} = ξ^{11} = ξ − ξ³.
    ///
    /// (ξ̄)² = 1 − ξ², (ξ̄)³ = −ξ³.
    /// conj(a + bξ + cξ² + dξ³) = (a+c) + bξ + (−c)ξ² + (−b−d)ξ³.
    pub fn conj(self) -> Self {
        Self {
            a:  self.a + self.c,
            b:  self.b,
            c: -self.c,
            d: -self.b - self.d,
        }
    }

    /// Scalar multiply by an integer.
    #[inline]
    pub fn scale(self, s: Int) -> Self {
        Self {
            a: self.a * s, b: self.b * s, c: self.c * s, d: self.d * s,
        }
    }

    /// Convert to a floating-point complex number.
    pub fn to_complex(self) -> Complex64 {
        use std::f64::consts::PI;
        let xi = |k: u32| Complex64::from_polar(1.0, PI * k as Float / 6.0);
        int_to_f64(self.a) * xi(0)
            + int_to_f64(self.b) * xi(1)
            + int_to_f64(self.c) * xi(2)
            + int_to_f64(self.d) * xi(3)
    }

    /// Largest power of 2 dividing all coefficients (for normalization).
    pub fn gcd_power_of_2(self) -> u32 {
        let bits = self.a | self.b | self.c | self.d;
        if bits == INT_ZERO { Int::BITS - 1 } else { bits.trailing_zeros() }
    }

    /// Divide all coefficients by 2^shift.
    #[inline]
    pub fn div2(self, shift: u32) -> Self {
        Self {
            a: self.a >> shift, b: self.b >> shift,
            c: self.c >> shift, d: self.d >> shift,
        }
    }

    /// Multiply by √3 = 2ξ − ξ³  (since 2e^{iπ/6} − e^{iπ/2} = √3 + i − i = √3).
    pub fn mul_sqrt3(self) -> Self {
        // self * (0 + 2·ξ + 0·ξ² + (−1)·ξ³)
        let rhs = Self::from_i32(0, 2, 0, -1);
        self * rhs
    }

    /// Embed a Gaussian integer: i = ξ³, so (a + bi) → a + b·ξ³.
    pub fn from_zi(a: Int, b: Int) -> Self {
        Self::new(a, INT_ZERO, INT_ZERO, b)
    }

    /// Bullet conjugate: the Galois automorphism √3 ↦ −√3, fixing i.
    ///
    /// This is one of the three nontrivial Galois automorphisms of ℚ(ζ₁₂)/ℚ,
    /// distinct from complex conjugation. Used for FGKM-style pruning bounds:
    /// for a synthesizable unitary, ‖u•‖² ≤ 1 must hold.
    ///
    /// On the cyclotomic basis:
    ///   1   ↦ 1
    ///   ξ   ↦ ξ⁵ = ξ³ − ξ        (coefficients (0, -1, 0, 1))
    ///   ξ²  ↦ ξ¹⁰ = 1 − ξ²       (coefficients (1, 0, -1, 0))
    ///   ξ³  ↦ i = ξ³             (i is fixed by bullet)
    ///
    /// In coordinates: (a, b, c, d) ↦ (a + c, −b, −c, b + d).
    pub fn bullet(self) -> Self {
        Self {
            a:  self.a + self.c,
            b: -self.b,
            c: -self.c,
            d:  self.b + self.d,
        }
    }

    /// Algebraic norm |z|² as an element of the totally real subring ℤ[√3].
    ///
    /// Returns (rational, sqrt3) where |z|² = rational + sqrt3·√3.
    ///
    /// For z = a + bξ + cξ² + dξ³:
    ///   |z|² = (a² + b² + c² + d² + ac + bd) + √3·(ab + bc + cd).
    ///
    /// This is the |z|² used in the FGKM unitarity equation |u|² + |t|² = 2^k,
    /// where the rational part equals 2^k and the √3 part must vanish.
    pub fn complex_norm_sqr(self) -> (Int, Int) {
        let (a, b, c, d) = (self.a, self.b, self.c, self.d);
        let rational = a*a + b*b + c*c + d*d + a*c + b*d;
        let sqrt3    = a*b + b*c + c*d;
        (rational, sqrt3)
    }

    /// Field norm N(z) = ∏_σ σ(z) over the full Galois group of ℚ(ζ₁₂)/ℚ.
    ///
    /// Equals z · z̄ · z• · (z•)̄. Always a rational integer (since the product
    /// is Galois-invariant). Useful for divisibility tests and smoothness checks.
    ///
    /// In terms of the algebraic norm components (r, s) where |z|² = r + s√3:
    ///   N(z) = (r + s√3)(r − s√3) = r² − 3s².
    ///
    /// This is *positive* for nonzero z (since |z|² > 0 in every embedding) and
    /// can grow large; consider returning a wider integer type in production.
    pub fn field_norm(self) -> Int {
        let (r, s) = self.complex_norm_sqr();
        r * r - Int::from_i32(3) * s * s
    }

    /// Multiplicative inverse of ξ within ℤ[ξ].
    ///
    /// ξ⁻¹ = ξ¹¹ = ξ − ξ³ (since ξ¹² = 1 and reducing ξ¹¹ via Φ₁₂ gives this).
    /// Useful for left-multiplying by R_z(−π/6) during _T_dag_on_uv-style
    /// preprocessing of odd-k unitaries.
    pub const INV_XI: Self = Self {
        a: INT_ZERO,
        b: INT_ONE,
        c: INT_ZERO,
        d: INT_NEG_ONE,
    };
}

// ─── Arithmetic ───────────────────────────────────────────────────────────────

impl Add for ZOmicron {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            a: self.a + rhs.a, b: self.b + rhs.b,
            c: self.c + rhs.c, d: self.d + rhs.d,
        }
    }
}

impl Sub for ZOmicron {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            a: self.a - rhs.a, b: self.b - rhs.b,
            c: self.c - rhs.c, d: self.d - rhs.d,
        }
    }
}

impl Neg for ZOmicron {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self { a: -self.a, b: -self.b, c: -self.c, d: -self.d }
    }
}

/// Multiplication in Z[ξ] modulo ξ⁴ = ξ² − 1.
///
/// Higher powers reduce as: ξ⁴ = ξ² − 1, ξ⁵ = ξ³ − ξ, ξ⁶ = −1.
///
/// (p₀ + p₁ξ + p₂ξ² + p₃ξ³)(q₀ + q₁ξ + q₂ξ² + q₃ξ³) gives:
///   [1 ]: p₀q₀ − (p₁q₃ + p₂q₂ + p₃q₁) − p₃q₃
///   [ξ ]: p₀q₁ + p₁q₀ − (p₂q₃ + p₃q₂)
///   [ξ²]: p₀q₂ + p₁q₁ + p₂q₀ + (p₁q₃ + p₂q₂ + p₃q₁)
///   [ξ³]: p₀q₃ + p₁q₂ + p₂q₁ + p₃q₀ + (p₂q₃ + p₃q₂)
impl Mul for ZOmicron {
    type Output = Self;
    #[inline]
    fn mul(self, q: Self) -> Self {
        let (p0, p1, p2, p3) = (self.a, self.b, self.c, self.d);
        let (q0, q1, q2, q3) = (q.a,    q.b,    q.c,    q.d);
        // Degree-4 wrap: p1q3 + p2q2 + p3q1  (ξ⁴ = ξ²−1 → subtracts from [1], adds to [ξ²])
        // Degree-5 wrap: p2q3 + p3q2          (ξ⁵ = ξ³−ξ → subtracts from [ξ],  adds to [ξ³])
        // Degree-6 wrap: p3q3                 (ξ⁶ = −1   → subtracts from [1])
        Self {
            a: p0*q0 - p1*q3 - p2*q2 - p3*q1 - p3*q3,
            b: p0*q1 + p1*q0 - p2*q3 - p3*q2,
            c: p0*q2 + p1*q1 + p2*q0 + p1*q3 + p2*q2 + p3*q1,
            d: p0*q3 + p1*q2 + p2*q1 + p3*q0 + p2*q3 + p3*q2,
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

impl fmt::Display for ZOmicron {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_poly(&[
            (self.a, ""),
            (self.b, "ξ"),
            (self.c, "ξ²"),
            (self.d, "ξ³"),
        ], f)
    }
}

// ─── PyO3 ─────────────────────────────────────────────────────────────────────

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Python-facing ZOmicron.
#[cfg(feature = "python")]
#[pyclass(name = "ZOmicron", frozen)]
pub struct PyZOmicron {
    pub inner: ZOmicron,
}

#[cfg(feature = "python")]
impl PyZOmicron {
    pub fn to_inner(&self) -> ZOmicron { self.inner }

    pub fn from_inner(inner: ZOmicron) -> Self { Self { inner } }
}

#[cfg(feature = "python")]
#[pymethods]
impl PyZOmicron {
    /// Python always passes 64-bit integers; cast to `Int` (may be wider than i64).
    #[new]
    fn new(a: i64, b: i64, c: i64, d: i64) -> Self {
        Self { inner: ZOmicron::new(
            Int::from_i64(a), Int::from_i64(b), Int::from_i64(c), Int::from_i64(d),
        )}
    }

    fn __add__(&self, other: &PyZOmicron) -> Self { Self { inner: self.inner + other.inner } }
    fn __sub__(&self, other: &PyZOmicron) -> Self { Self { inner: self.inner - other.inner } }
    fn __mul__(&self, other: &PyZOmicron) -> Self { Self { inner: self.inner * other.inner } }
    fn __neg__(&self) -> Self { Self { inner: -self.inner } }

    fn to_complex(&self) -> (f64, f64) { let c = self.inner.to_complex(); (c.re, c.im) }
    fn mul_sqrt3(&self) -> Self { Self { inner: self.inner.mul_sqrt3() } }
    fn gcd_power_of_2(&self) -> u32 { self.inner.gcd_power_of_2() }

    #[staticmethod] fn one()     -> Self { Self { inner: ZOmicron::ONE } }
    #[staticmethod] fn xi()      -> Self { Self { inner: ZOmicron::XI } }
    #[staticmethod] fn i()       -> Self { Self { inner: ZOmicron::I } }
    #[staticmethod] fn zero()    -> Self { Self { inner: ZOmicron::ZERO } }
    #[staticmethod] fn neg_one() -> Self { Self { inner: ZOmicron::NEG_ONE } }

    fn __repr__(&self) -> String {
        let z = &self.inner;
        format!("ZOmicron({},{},{},{})", z.a, z.b, z.c, z.d)
    }
    fn bullet(&self) -> Self {
        Self { inner: self.inner.bullet() }
    }

    fn complex_norm_sqr(&self) -> (i64, i64) {
        let (r, s) = self.inner.complex_norm_sqr();
        // Adjust the conversion if Int is wider than i64
        (r as i64, s as i64)
    }

    fn field_norm(&self) -> i64 {
        self.inner.field_norm() as i64
    }

    #[staticmethod]
    fn inv_xi() -> Self {
        Self { inner: ZOmicron::INV_XI }
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
        assert!(near(ZOmicron::ONE.to_complex(), Complex64::new(1.0, 0.0)));
        let expected_xi = Complex64::from_polar(1.0, PI / 6.0);
        assert!(near(ZOmicron::XI.to_complex(), expected_xi));
        assert!(near(ZOmicron::I.to_complex(), Complex64::new(0.0, 1.0)));
        assert!(near(ZOmicron::NEG_ONE.to_complex(), Complex64::new(-1.0, 0.0)));
    }

    #[test]
    fn test_xi12_eq_1() {
        let x = ZOmicron::XI;
        let x12 = x*x * x*x * x*x * x*x * x*x * x*x;
        assert_eq!(x12, ZOmicron::ONE, "ξ¹² should equal 1");
    }

    #[test]
    fn test_xi4_eq_xi2_minus_1() {
        let x = ZOmicron::XI;
        let x4 = x * x * x * x;
        let xi2_minus_1 = ZOmicron::from_i32(-1, 0, 1, 0);
        assert_eq!(x4, xi2_minus_1, "ξ⁴ should equal ξ² − 1");
    }

    #[test]
    fn test_xi3_eq_i() {
        let x = ZOmicron::XI;
        let x3 = x * x * x;
        assert_eq!(x3, ZOmicron::I, "ξ³ should equal i");
    }

    #[test]
    fn test_xi6_eq_neg1() {
        let x = ZOmicron::XI;
        let x6 = x*x * x*x * x*x;
        assert_eq!(x6, ZOmicron::NEG_ONE, "ξ⁶ should equal −1");
    }

    #[test]
    fn test_conj() {
        let c        = ZOmicron::XI.conj().to_complex();
        let expected = Complex64::from_polar(1.0, -PI / 6.0);
        assert!(near(c, expected), "conj(ξ) = {c}, expected {expected}");
    }

    #[test]
    fn test_conj_formula() {
        // conj(a + bξ + cξ² + dξ³) = (a+c) + bξ − cξ² + (−b−d)ξ³
        let z = ZOmicron::from_i32(1, 2, 3, 4);
        let got      = z.conj().to_complex();
        let expected = z.to_complex().conj();
        assert!(near(got, expected), "conj: got {got}, expected {expected}");
    }

    #[test]
    fn test_mul_complex_consistent() {
        let x = ZOmicron::from_i32(1, 2, -1, 3);
        let y = ZOmicron::from_i32(-2, 1, 3, 0);
        let prod_ring  = (x * y).to_complex();
        let prod_float = x.to_complex() * y.to_complex();
        assert!(near(prod_ring, prod_float), "ring {prod_ring} vs float {prod_float}");
    }

    #[test]
    fn test_mul_commutative() {
        let x = ZOmicron::from_i32(1, 2, -1, 3);
        let y = ZOmicron::from_i32(-2, 1, 3, 0);
        assert_eq!(x * y, y * x);
    }

    #[test]
    fn test_mul_associative() {
        let x = ZOmicron::from_i32(1, -1, 2, 3);
        let y = ZOmicron::from_i32(2, 1, -3, 0);
        let z = ZOmicron::from_i32(-1, 3, 1, -2);
        assert_eq!((x * y) * z, x * (y * z));
    }

    #[test]
    fn test_mul_sqrt3() {
        let sqrt3 = ZOmicron::ONE.mul_sqrt3().to_complex();
        assert!(
            (sqrt3.re - 3.0_f64.sqrt()).abs() < 1e-12 && sqrt3.im.abs() < 1e-12,
            "mul_sqrt3(1) = {sqrt3}"
        );
    }

    #[test]
    fn test_zi_embedding_consistent() {
        // Gaussian integer 3 + 2i embeds as (3, 0, 0, 2) in Z[ξ]
        let z_zi  = Complex64::new(3.0, 2.0);
        let z_omc = ZOmicron::from_zi(Int::from_i32(3), Int::from_i32(2));
        assert!(
            near(z_zi, z_omc.to_complex()),
            "zi={z_zi}, omc={}",
            z_omc.to_complex()
        );
    }

    #[test]
    fn test_add_sub() {
        let x = ZOmicron::from_i32(1, 2, 3, 4);
        let y = ZOmicron::from_i32(-1, 0, 1, -2);
        assert!(near((x + y).to_complex(), x.to_complex() + y.to_complex()));
        assert!(near((x - y).to_complex(), x.to_complex() - y.to_complex()));
    }

    #[test]
    fn test_gcd_power_of_2() {
        assert_eq!(ZOmicron::from_i32(4, 8, 0, 12).gcd_power_of_2(), 2);
        assert_eq!(ZOmicron::from_i32(1, 2, 3, 4).gcd_power_of_2(), 0);
        assert_eq!(ZOmicron::from_i32(0, 0, 8, 0).gcd_power_of_2(), 3);
    }

    #[test]
    fn test_bullet_basis_elements() {
        // 1 is fixed by bullet
        assert_eq!(ZOmicron::ONE.bullet(), ZOmicron::ONE);

        // ξ ↦ ξ⁵ = ξ³ − ξ, coefficients (0, -1, 0, 1)
        assert_eq!(
            ZOmicron::XI.bullet(),
            ZOmicron::from_i32(0, -1, 0, 1),
            "bullet(ξ) should equal ξ⁵ = ξ³ − ξ"
        );

        // ξ² ↦ 1 − ξ², coefficients (1, 0, -1, 0)
        let xi2 = ZOmicron::XI * ZOmicron::XI;
        assert_eq!(
            xi2.bullet(),
            ZOmicron::from_i32(1, 0, -1, 0),
            "bullet(ξ²) should equal 1 − ξ²"
        );

        // ξ³ = i is fixed by bullet
        assert_eq!(ZOmicron::I.bullet(), ZOmicron::I);
    }

    #[test]
    fn test_bullet_involution() {
        // Applying bullet twice should give back the original element.
        let z = ZOmicron::from_i32(3, -2, 5, 7);
        assert_eq!(z.bullet().bullet(), z);

        // Try a few more random ones
        for &(a, b, c, d) in &[(0, 0, 0, 0), (1, 1, 1, 1), (-1, 2, -3, 4), (100, -50, 25, 13)] {
            let z = ZOmicron::from_i32(a, b, c, d);
            assert_eq!(z.bullet().bullet(), z, "bullet² != id on ({a},{b},{c},{d})");
        }
    }

    #[test]
    fn test_bullet_vs_complex_value() {
        // bullet(z) in ℂ should be the value of z with √3 replaced by -√3.
        // The cleanest check: bullet(ξ) = ξ⁵ has complex value e^{i·5π/6}.
        let bullet_xi = ZOmicron::XI.bullet().to_complex();
        let expected = Complex64::from_polar(1.0, 5.0 * PI / 6.0);
        assert!(near(bullet_xi, expected), "bullet(ξ) = {bullet_xi}, expected {expected}");

        // bullet(ξ²) = ξ¹⁰ has complex value e^{i·10π/6} = e^{-i·π/3}.
        let xi2 = ZOmicron::XI * ZOmicron::XI;
        let bullet_xi2 = xi2.bullet().to_complex();
        let expected = Complex64::from_polar(1.0, -PI / 3.0);
        assert!(near(bullet_xi2, expected), "bullet(ξ²) = {bullet_xi2}, expected {expected}");
    }

    #[test]
    fn test_bullet_distinct_from_conj() {
        // bullet and conj are different Galois automorphisms.
        // bullet fixes i, conj sends i → -i, so they should disagree on ξ³ = i.
        // Both happen to fix i actually — wait, conj sends i to -i.
        let bull_i = ZOmicron::I.bullet();
        let conj_i = ZOmicron::I.conj();
        assert_ne!(bull_i, conj_i, "bullet and conj should differ on i");

        // Concrete check:
        assert_eq!(bull_i, ZOmicron::I, "bullet should fix i (since √3 ↦ -√3 fixes i)");
        assert_eq!(conj_i, ZOmicron::NEG_I, "conj should send i ↦ -i");
    }

    #[test]
    fn test_bullet_is_ring_homomorphism() {
        // bullet(xy) = bullet(x) · bullet(y)  (Galois automorphisms are ring homs)
        let x = ZOmicron::from_i32(2, -1, 3, 1);
        let y = ZOmicron::from_i32(1, 4, -2, 5);
        assert_eq!(
            (x * y).bullet(),
            x.bullet() * y.bullet(),
            "bullet should be a ring homomorphism"
        );

        // bullet(x + y) = bullet(x) + bullet(y)
        assert_eq!(
            (x + y).bullet(),
            x.bullet() + y.bullet(),
        );
    }

    #[test]
    fn test_complex_norm_sqr_basis_elements() {
        // |1|² = 1
        let (r, s) = ZOmicron::ONE.complex_norm_sqr();
        assert_eq!((r, s), (Int::from_i32(1), Int::from_i32(0)));

        // |ξ|² = 1 (since |e^{iπ/6}| = 1)
        let (r, s) = ZOmicron::XI.complex_norm_sqr();
        assert_eq!((r, s), (Int::from_i32(1), Int::from_i32(0)),
            "|ξ|² should be 1, got {r} + {s}√3");

        // |ξ²|² = 1
        let xi2 = ZOmicron::XI * ZOmicron::XI;
        let (r, s) = xi2.complex_norm_sqr();
        assert_eq!((r, s), (Int::from_i32(1), Int::from_i32(0)));

        // |i|² = |ξ³|² = 1
        let (r, s) = ZOmicron::I.complex_norm_sqr();
        assert_eq!((r, s), (Int::from_i32(1), Int::from_i32(0)));
    }

    #[test]
    fn test_complex_norm_sqr_compound() {
        // Verify against to_complex() for a non-basis element.
        // z = 1 + 2ξ has |z|² = (1 + √3)² + 1² = 1 + 2√3 + 3 + 1 = 5 + 2√3.
        // So rational = 5, sqrt3 = 2.
        let z = ZOmicron::from_i32(1, 2, 0, 0);
        let (r, s) = z.complex_norm_sqr();
        assert_eq!(r, Int::from_i32(5));
        assert_eq!(s, Int::from_i32(2));

        // Cross-check numerically
        let c = z.to_complex();
        let expected_norm_sq = c.norm_sqr();  // Complex64::norm_sqr
        let computed_norm_sq = int_to_f64(r) + int_to_f64(s) * 3.0_f64.sqrt();
        assert!(
            (expected_norm_sq - computed_norm_sq).abs() < 1e-12,
            "|z|² mismatch: complex {expected_norm_sq} vs ring {computed_norm_sq}"
        );
    }

    #[test]
    fn test_complex_norm_sqr_matches_to_complex_general() {
        // Random element: verify rational + sqrt3·√3 matches |to_complex()|².
        let test_cases = [
            ( 1,  2,  3,  4),
            (-1,  0,  2, -3),
            ( 5, -2, -7,  1),
            ( 0,  1,  0,  1),
        ];
        for &(a, b, c, d) in &test_cases {
            let z = ZOmicron::from_i32(a, b, c, d);
            let (r, s) = z.complex_norm_sqr();
            let expected = z.to_complex().norm_sqr();
            let computed = int_to_f64(r) + int_to_f64(s) * 3.0_f64.sqrt();
            assert!(
                (expected - computed).abs() < 1e-9,
                "|z|² mismatch on ({a},{b},{c},{d}): expected {expected}, got {computed} (r={r}, s={s})"
            );
        }
    }

    #[test]
    fn test_field_norm_basis_elements() {
        // N(1) = 1, N(ξ) = 1, etc. — all roots of unity have field norm 1.
        assert_eq!(ZOmicron::ONE.field_norm(), Int::from_i32(1));
        assert_eq!(ZOmicron::XI.field_norm(), Int::from_i32(1));
        assert_eq!(ZOmicron::I.field_norm(), Int::from_i32(1));
        let xi2 = ZOmicron::XI * ZOmicron::XI;
        assert_eq!(xi2.field_norm(), Int::from_i32(1));
    }

    #[test]
    fn test_field_norm_multiplicative() {
        // N(xy) = N(x) · N(y)
        let x = ZOmicron::from_i32(1, 2, -1, 3);
        let y = ZOmicron::from_i32(-2, 1, 3, 0);
        let n_x = x.field_norm();
        let n_y = y.field_norm();
        let n_xy = (x * y).field_norm();
        assert_eq!(n_xy, n_x * n_y, "field_norm should be multiplicative");
    }

    #[test]
    fn test_field_norm_consistency_with_complex_norm_sqr() {
        // N(z) = (r + s√3)(r - s√3) = r² - 3s² where |z|² = r + s√3.
        let test_cases = [
            ( 1,  2,  3,  4),
            (-1,  0,  2, -3),
            ( 5, -2, -7,  1),
            ( 1,  1,  1,  1),
        ];
        for &(a, b, c, d) in &test_cases {
            let z = ZOmicron::from_i32(a, b, c, d);
            let (r, s) = z.complex_norm_sqr();
            let expected = r * r - Int::from_i32(3) * s * s;
            assert_eq!(z.field_norm(), expected, "field_norm inconsistency on ({a},{b},{c},{d})");
        }
    }

    #[test]
    fn test_inv_xi() {
        // INV_XI · XI = 1
        assert_eq!(ZOmicron::INV_XI * ZOmicron::XI, ZOmicron::ONE);
        assert_eq!(ZOmicron::XI * ZOmicron::INV_XI, ZOmicron::ONE);

        // Numerically, INV_XI = e^{-iπ/6}
        let expected = Complex64::from_polar(1.0, -PI / 6.0);
        assert!(near(ZOmicron::INV_XI.to_complex(), expected));
    }

    #[test]
    fn test_sigma_gram_consistency() {
        // The xᵀGx computed via SIGMA_GRAM_U should equal 2·(complex_norm_sqr.0)
        // because xᵀGx for n=6 = 2 · (a² + b² + c² + d² + ac + bd)
        //                     = 2 · (rational part of |u|²)
        let test_cases = [
            ( 1,  2,  3,  4),
            ( 0,  1,  0,  0),  // ξ
            ( 1,  0,  0,  1),  // 1 + i
        ];
        for &(a, b, c, d) in &test_cases {
            let z = ZOmicron::from_i32(a, b, c, d);
            let (r, _) = z.complex_norm_sqr();

            // xᵀGx by the Gram formula
            let coords = [a as i64, b as i64, c as i64, d as i64];
            let mut gram_form = 0i64;
            for i in 0..4 {
                for j in 0..4 {
                    gram_form += SIGMA_GRAM_U[i][j] * coords[i] * coords[j];
                }
            }
            assert_eq!(
                gram_form,
                2 * (r as i64),
                "Gram form should equal 2·rational(|z|²) for ({a},{b},{c},{d})"
            );
        }
    }
}
