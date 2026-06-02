//! Z[υ] — the ring of integers extended by υ = e^{iπ/12}.
//!
//! Every element has the form
//! `a + bυ + cυ² + dυ³ + eυ⁴ + fυ⁵ + gυ⁶ + hυ⁷`
//! with integer coefficients and relation `υ⁸ = υ⁴ - 1`
//! (`Φ₂₄(x) = x⁸ - x⁴ + 1`).
//!
//! Useful facts: `υ⁶ = i`, `υ¹² = -1`,
//! `√2 = υ + υ³ - υ⁵`, `√3 = 2υ² - υ⁶`, and
//! `√6 = υ + υ³ + υ⁵ - 2υ⁷`.
//!
//! TODO(gate-set): the n=12 denominator generator is not fixed yet. This file
//! defaults valuation helpers to √2 because the current target shell is `2^k`.
//! If the gate set selects √3 or √6, replace `mul_sqrt2` / `sqrt2_valuation`
//! callers with the corresponding generator and change synthesis targets to
//! `3^k` or `6^k`.

use super::types::{int_to_f64, Float, Int, INT_NEG_ONE, INT_ONE, INT_TWO, INT_ZERO};
use num_complex::Complex64;
use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};

/// An element of Z[υ], υ = e^{iπ/12}, υ⁸ = υ⁴ - 1.
///
/// Represented as integer coefficients of the basis
/// `{1, υ, υ², υ³, υ⁴, υ⁵, υ⁶, υ⁷}`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct ZUpsilon {
    pub a: Int, // coefficient of 1   = υ⁰
    pub b: Int, // coefficient of υ
    pub c: Int, // coefficient of υ²
    pub d: Int, // coefficient of υ³
    pub e: Int, // coefficient of υ⁴
    pub f: Int, // coefficient of υ⁵
    pub g: Int, // coefficient of υ⁶ = i
    pub h: Int, // coefficient of υ⁷
}

impl ZUpsilon {
    pub const ZERO: Self = Self {
        a: INT_ZERO,
        b: INT_ZERO,
        c: INT_ZERO,
        d: INT_ZERO,
        e: INT_ZERO,
        f: INT_ZERO,
        g: INT_ZERO,
        h: INT_ZERO,
    };
    pub const ONE: Self = Self {
        a: INT_ONE,
        b: INT_ZERO,
        c: INT_ZERO,
        d: INT_ZERO,
        e: INT_ZERO,
        f: INT_ZERO,
        g: INT_ZERO,
        h: INT_ZERO,
    };
    /// υ itself.
    pub const UPSILON: Self = Self {
        a: INT_ZERO,
        b: INT_ONE,
        c: INT_ZERO,
        d: INT_ZERO,
        e: INT_ZERO,
        f: INT_ZERO,
        g: INT_ZERO,
        h: INT_ZERO,
    };
    /// Alias for callers that name the root ζ₂₄.
    pub const ZETA: Self = Self::UPSILON;
    /// i = υ⁶.
    pub const I: Self = Self {
        a: INT_ZERO,
        b: INT_ZERO,
        c: INT_ZERO,
        d: INT_ZERO,
        e: INT_ZERO,
        f: INT_ZERO,
        g: INT_ONE,
        h: INT_ZERO,
    };
    /// -1.
    pub const NEG_ONE: Self = Self {
        a: INT_NEG_ONE,
        b: INT_ZERO,
        c: INT_ZERO,
        d: INT_ZERO,
        e: INT_ZERO,
        f: INT_ZERO,
        g: INT_ZERO,
        h: INT_ZERO,
    };
    /// -i = -υ⁶.
    pub const NEG_I: Self = Self {
        a: INT_ZERO,
        b: INT_ZERO,
        c: INT_ZERO,
        d: INT_ZERO,
        e: INT_ZERO,
        f: INT_ZERO,
        g: INT_NEG_ONE,
        h: INT_ZERO,
    };

    #[inline]
    #[allow(clippy::too_many_arguments)] // 8 ring coefficients are intrinsic to Z[υ].
    pub const fn new(a: Int, b: Int, c: Int, d: Int, e: Int, f: Int, g: Int, h: Int) -> Self {
        Self {
            a,
            b,
            c,
            d,
            e,
            f,
            g,
            h,
        }
    }

    /// Construct from small integer coefficients, converting via `Int::from_i32`.
    #[inline]
    #[allow(clippy::too_many_arguments)] // 8 ring coefficients are intrinsic to Z[υ].
    pub const fn from_i32(a: i32, b: i32, c: i32, d: i32, e: i32, f: i32, g: i32, h: i32) -> Self {
        Self::new(
            Int::from_i32(a),
            Int::from_i32(b),
            Int::from_i32(c),
            Int::from_i32(d),
            Int::from_i32(e),
            Int::from_i32(f),
            Int::from_i32(g),
            Int::from_i32(h),
        )
    }

    /// Coefficient of υ^k, k = 0..7.
    #[inline]
    pub fn coeff(self, k: usize) -> Int {
        match k {
            0 => self.a,
            1 => self.b,
            2 => self.c,
            3 => self.d,
            4 => self.e,
            5 => self.f,
            6 => self.g,
            7 => self.h,
            _ => panic!("ZUpsilon::coeff: index {k} out of range"),
        }
    }

    /// Complex conjugate: υ ↦ υ⁻¹ = υ²³ = υ³ - υ⁷.
    ///
    /// Reducing `υ^(24-k)` modulo `Φ₂₄` gives:
    /// `1 ↦ 1`, `υ ↦ υ³-υ⁷`, `υ² ↦ υ²-υ⁶`,
    /// `υ³ ↦ υ-υ⁵`, `υ⁴ ↦ 1-υ⁴`, `υ⁵ ↦ -υ⁷`,
    /// `υ⁶ ↦ -υ⁶`, `υ⁷ ↦ -υ⁵`.
    pub fn conj(self) -> Self {
        Self {
            a: self.a + self.e,
            b: self.d,
            c: self.c,
            d: self.b,
            e: -self.e,
            f: -self.d - self.h,
            g: -self.c - self.g,
            h: -self.b - self.f,
        }
    }

    /// Scalar multiply by an integer.
    #[inline]
    pub fn scale(self, s: Int) -> Self {
        Self {
            a: self.a * s,
            b: self.b * s,
            c: self.c * s,
            d: self.d * s,
            e: self.e * s,
            f: self.f * s,
            g: self.g * s,
            h: self.h * s,
        }
    }

    /// Divide all coefficients by 2^shift.
    #[inline]
    pub fn div2(self, shift: u32) -> Self {
        Self {
            a: self.a >> shift,
            b: self.b >> shift,
            c: self.c >> shift,
            d: self.d >> shift,
            e: self.e >> shift,
            f: self.f >> shift,
            g: self.g >> shift,
            h: self.h >> shift,
        }
    }

    /// Convert to a floating-point complex number.
    pub fn to_complex(self) -> Complex64 {
        use std::f64::consts::PI;
        let upsilon = |k: u32| Complex64::from_polar(1.0, PI * k as Float / 12.0);
        int_to_f64(self.a) * upsilon(0)
            + int_to_f64(self.b) * upsilon(1)
            + int_to_f64(self.c) * upsilon(2)
            + int_to_f64(self.d) * upsilon(3)
            + int_to_f64(self.e) * upsilon(4)
            + int_to_f64(self.f) * upsilon(5)
            + int_to_f64(self.g) * upsilon(6)
            + int_to_f64(self.h) * upsilon(7)
    }

    /// Multiply by √2 = υ + υ³ - υ⁵.
    pub fn mul_sqrt2(self) -> Self {
        self * Self::sqrt2()
    }

    /// Multiply by √3 = 2υ² - υ⁶.
    pub fn mul_sqrt3(self) -> Self {
        self * Self::sqrt3()
    }

    /// Multiply by √6 = υ + υ³ + υ⁵ - 2υ⁷.
    pub fn mul_sqrt6(self) -> Self {
        self * Self::sqrt6()
    }

    /// √2 = υ³ + υ⁻³ = υ + υ³ - υ⁵.
    pub const fn sqrt2() -> Self {
        Self {
            a: INT_ZERO,
            b: INT_ONE,
            c: INT_ZERO,
            d: INT_ONE,
            e: INT_ZERO,
            f: INT_NEG_ONE,
            g: INT_ZERO,
            h: INT_ZERO,
        }
    }

    /// √3 = υ² + υ⁻² = 2υ² - υ⁶.
    pub const fn sqrt3() -> Self {
        Self {
            a: INT_ZERO,
            b: INT_ZERO,
            c: INT_TWO,
            d: INT_ZERO,
            e: INT_ZERO,
            f: INT_ZERO,
            g: INT_NEG_ONE,
            h: INT_ZERO,
        }
    }

    /// √6 = √2·√3 = υ + υ³ + υ⁵ - 2υ⁷.
    pub const fn sqrt6() -> Self {
        Self {
            a: INT_ZERO,
            b: INT_ONE,
            c: INT_ZERO,
            d: INT_ONE,
            e: INT_ZERO,
            f: INT_ONE,
            g: INT_ZERO,
            h: Int::from_i32(-2),
        }
    }

    /// Embed a ZOmicron element: ξ = υ², so
    /// `(a + bξ + cξ² + dξ³) → a + bυ² + cυ⁴ + dυ⁶`.
    pub fn from_zomicron(a: Int, b: Int, c: Int, d: Int) -> Self {
        Self::new(a, INT_ZERO, b, INT_ZERO, c, INT_ZERO, d, INT_ZERO)
    }

    /// Largest k such that `(√2)^k` divides this element in Z[υ].
    ///
    /// This uses exact algebraic division: `u / √2 = u·√2 / 2`, and succeeds
    /// exactly when every resulting cyclotomic coefficient is even. It is not
    /// a coefficient gcd shortcut.
    pub fn sqrt2_valuation(self) -> u32 {
        if self == Self::ZERO {
            return Int::BITS - 1;
        }
        let mut value = self;
        let mut k = 0;
        loop {
            let doubled_quotient = value.mul_sqrt2();
            if !doubled_quotient.all_coeffs_even() {
                break;
            }
            value = doubled_quotient.div2(1);
            k += 1;
        }
        k
    }

    /// The rational component of `u·conj(u)`, computed from the n=12 Gram.
    #[inline]
    pub fn norm_sqr(self) -> Int {
        let x = self.coeffs();
        let mut sum = INT_ZERO;
        for xi in x {
            sum = sum + xi * xi;
        }
        for i in 0..4 {
            sum = sum + x[i] * x[i + 4];
        }
        sum
    }

    /// Decompose `u·conj(u)` as `r + s2√2 + s3√3 + s6√6`.
    ///
    /// For products `u·conj(u)`, the element is in the real subfield and the
    /// returned components are integral.
    pub fn complex_norm_sqr_components(self) -> (Int, Int, Int, Int) {
        (self * self.conj()).real_radical_components()
    }

    /// Decompose `u·conj(u)` as `(r, 2s2, 2s3, 2s6)`, where the irrational
    /// part is `s2√2 + s3√3 + s6√6`.
    ///
    /// The doubled form stays integral for every cyclotomic input. Use it for
    /// zero tests in the enumerator.
    pub fn complex_norm_sqr_components_twice(self) -> (Int, Int, Int, Int) {
        let p = self * self.conj();
        (p.a, INT_TWO * p.b + p.h, p.c, -p.h)
    }

    /// Coefficients in basis order.
    #[inline]
    pub fn coeffs(self) -> [Int; 8] {
        [
            self.a, self.b, self.c, self.d, self.e, self.f, self.g, self.h,
        ]
    }

    #[inline]
    fn all_coeffs_even(self) -> bool {
        (self.a & INT_ONE) == INT_ZERO
            && (self.b & INT_ONE) == INT_ZERO
            && (self.c & INT_ONE) == INT_ZERO
            && (self.d & INT_ONE) == INT_ZERO
            && (self.e & INT_ONE) == INT_ZERO
            && (self.f & INT_ONE) == INT_ZERO
            && (self.g & INT_ONE) == INT_ZERO
            && (self.h & INT_ONE) == INT_ZERO
    }

    fn real_radical_components(self) -> (Int, Int, Int, Int) {
        debug_assert_eq!(self.e, INT_ZERO, "real-subfield υ⁴ coefficient");
        debug_assert_eq!(self.b, self.d, "real-subfield υ/υ³ mismatch");
        debug_assert_eq!(self.c, -self.g * INT_TWO, "real-subfield √3 mismatch");
        debug_assert_eq!(self.f, -self.b - self.h, "real-subfield √2/√6 mismatch");
        debug_assert_eq!(
            self.h & INT_ONE,
            INT_ZERO,
            "real-subfield √6 component is half-integral"
        );

        let r = self.a;
        let s3 = -self.g;
        let s6 = -self.h / INT_TWO;
        let s2 = self.b - s6;
        (r, s2, s3, s6)
    }
}

// ─── Arithmetic ───────────────────────────────────────────────────────────────

impl Add for ZUpsilon {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            a: self.a + rhs.a,
            b: self.b + rhs.b,
            c: self.c + rhs.c,
            d: self.d + rhs.d,
            e: self.e + rhs.e,
            f: self.f + rhs.f,
            g: self.g + rhs.g,
            h: self.h + rhs.h,
        }
    }
}

impl Sub for ZUpsilon {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            a: self.a - rhs.a,
            b: self.b - rhs.b,
            c: self.c - rhs.c,
            d: self.d - rhs.d,
            e: self.e - rhs.e,
            f: self.f - rhs.f,
            g: self.g - rhs.g,
            h: self.h - rhs.h,
        }
    }
}

impl Neg for ZUpsilon {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self {
            a: -self.a,
            b: -self.b,
            c: -self.c,
            d: -self.d,
            e: -self.e,
            f: -self.f,
            g: -self.g,
            h: -self.h,
        }
    }
}

/// Multiplication in Z[υ] modulo `υ⁸ = υ⁴ - 1`.
impl Mul for ZUpsilon {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let p = self.coeffs();
        let q = rhs.coeffs();
        let mut tmp = [INT_ZERO; 15];
        for i in 0..8 {
            for j in 0..8 {
                tmp[i + j] = tmp[i + j] + p[i] * q[j];
            }
        }
        for d in (8..=14).rev() {
            let v = tmp[d];
            if v != INT_ZERO {
                tmp[d - 4] = tmp[d - 4] + v;
                tmp[d - 8] = tmp[d - 8] - v;
            }
        }
        Self::new(
            tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], tmp[6], tmp[7],
        )
    }
}

// ─── Display ──────────────────────────────────────────────────────────────────

fn fmt_poly(terms: &[(Int, &str)], f: &mut fmt::Formatter<'_>) -> fmt::Result {
    let mut first = true;
    for &(coeff, sym) in terms {
        if coeff == INT_ZERO {
            continue;
        }
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
    if first {
        write!(f, "0")?;
    }
    Ok(())
}

impl fmt::Display for ZUpsilon {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_poly(
            &[
                (self.a, ""),
                (self.b, "υ"),
                (self.c, "υ²"),
                (self.d, "υ³"),
                (self.e, "υ⁴"),
                (self.f, "υ⁵"),
                (self.g, "υ⁶"),
                (self.h, "υ⁷"),
            ],
            f,
        )
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use std::f64::consts::PI;

    fn near(a: Complex64, b: Complex64) -> bool {
        (a - b).norm() < 1e-9
    }

    fn pow(mut z: ZUpsilon, n: usize) -> ZUpsilon {
        let mut out = ZUpsilon::ONE;
        for _ in 0..n {
            out = out * z;
        }
        z = out;
        z
    }

    fn brute_mul(p: ZUpsilon, q: ZUpsilon) -> ZUpsilon {
        let pc = p.coeffs();
        let qc = q.coeffs();
        let mut tmp = [INT_ZERO; 15];
        for i in 0..8 {
            for j in 0..8 {
                tmp[i + j] = tmp[i + j] + pc[i] * qc[j];
            }
        }
        for d in (8..=14).rev() {
            let v = tmp[d];
            tmp[d] = INT_ZERO;
            tmp[d - 4] = tmp[d - 4] + v;
            tmp[d - 8] = tmp[d - 8] - v;
        }
        ZUpsilon::new(
            tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], tmp[6], tmp[7],
        )
    }

    #[test]
    fn to_complex_basis() {
        assert!(near(ZUpsilon::ONE.to_complex(), Complex64::new(1.0, 0.0)));
        assert!(near(
            ZUpsilon::ZETA.to_complex(),
            Complex64::from_polar(1.0, PI / 12.0)
        ));
        assert!(near(ZUpsilon::I.to_complex(), Complex64::new(0.0, 1.0)));
        assert!(near(
            ZUpsilon::NEG_ONE.to_complex(),
            Complex64::new(-1.0, 0.0)
        ));
    }

    #[test]
    fn powers_match_zeta24_identities() {
        let z = ZUpsilon::ZETA;
        assert_eq!(pow(z, 24), ZUpsilon::ONE);
        assert_eq!(pow(z, 12), ZUpsilon::NEG_ONE);
        assert_eq!(pow(z, 6), ZUpsilon::I);
        assert_eq!(pow(z, 8), ZUpsilon::from_i32(-1, 0, 0, 0, 1, 0, 0, 0));
    }

    #[test]
    fn radicals_square_correctly() {
        assert_eq!(
            ZUpsilon::sqrt2() * ZUpsilon::sqrt2(),
            ZUpsilon::from_i32(2, 0, 0, 0, 0, 0, 0, 0)
        );
        assert_eq!(
            ZUpsilon::sqrt3() * ZUpsilon::sqrt3(),
            ZUpsilon::from_i32(3, 0, 0, 0, 0, 0, 0, 0)
        );
        assert_eq!(
            ZUpsilon::sqrt6() * ZUpsilon::sqrt6(),
            ZUpsilon::from_i32(6, 0, 0, 0, 0, 0, 0, 0)
        );
    }

    #[test]
    fn one_is_mul_identity_and_mul_matches_bruteforce() {
        let mut rng = rand::rng();
        for _ in 0..200 {
            let p = ZUpsilon::from_i32(
                rng.random_range(-3..=3),
                rng.random_range(-3..=3),
                rng.random_range(-3..=3),
                rng.random_range(-3..=3),
                rng.random_range(-3..=3),
                rng.random_range(-3..=3),
                rng.random_range(-3..=3),
                rng.random_range(-3..=3),
            );
            let q = ZUpsilon::from_i32(
                rng.random_range(-3..=3),
                rng.random_range(-3..=3),
                rng.random_range(-3..=3),
                rng.random_range(-3..=3),
                rng.random_range(-3..=3),
                rng.random_range(-3..=3),
                rng.random_range(-3..=3),
                rng.random_range(-3..=3),
            );
            assert_eq!(p * ZUpsilon::ONE, p);
            assert_eq!(ZUpsilon::ONE * p, p);
            assert_eq!(p * q, brute_mul(p, q));
        }
    }

    #[test]
    fn conj_matches_numeric_conjugation() {
        let mut rng = rand::rng();
        for _ in 0..200 {
            let z = ZUpsilon::from_i32(
                rng.random_range(-5..=5),
                rng.random_range(-5..=5),
                rng.random_range(-5..=5),
                rng.random_range(-5..=5),
                rng.random_range(-5..=5),
                rng.random_range(-5..=5),
                rng.random_range(-5..=5),
                rng.random_range(-5..=5),
            );
            let got = z.conj().to_complex();
            let expected = z.to_complex().conj();
            assert!(
                near(got, expected),
                "z={z}, got={got:?}, expected={expected:?}"
            );
        }
    }

    #[test]
    fn sqrt2_valuation_uses_exact_division() {
        let base = ZUpsilon::from_i32(1, 2, -1, 0, 1, -2, 0, 3);
        let once = base.mul_sqrt2();
        let twice = once.mul_sqrt2();
        assert_eq!(once.sqrt2_valuation(), base.sqrt2_valuation() + 1);
        assert_eq!(twice.sqrt2_valuation(), base.sqrt2_valuation() + 2);
    }

    #[test]
    fn norm_components_match_radicals() {
        let z = ZUpsilon::ONE.scale(Int::from_i32(2))
            + ZUpsilon::I.scale(Int::from_i32(-1))
            + ZUpsilon::sqrt2().scale(Int::from_i32(3))
            + (ZUpsilon::I * ZUpsilon::sqrt3()).scale(Int::from_i32(1))
            + ZUpsilon::sqrt6().scale(Int::from_i32(-2));
        let (r, s2, s3, s6) = z.complex_norm_sqr_components();
        let reconstructed = ZUpsilon::ONE.scale(r)
            + ZUpsilon::sqrt2().scale(s2)
            + ZUpsilon::sqrt3().scale(s3)
            + ZUpsilon::sqrt6().scale(s6);
        assert_eq!(z * z.conj(), reconstructed);
        assert_eq!(r, z.norm_sqr());
    }
}
