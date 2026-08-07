//! Z[ζ] — the ring of integers extended by ζ = e^{iπ/8}.
//!
//! Every element has the form  a + b·ζ + c·ζ² + d·ζ³ + e·ζ⁴ + f·ζ⁵ + g·ζ⁶ + h·ζ⁷
//! with a,b,c,d,e,f,g,h ∈ ℤ and the relation ζ^8 = −1.
//!
//! This is the coefficient ring for exactly-implementable Clifford+√T unitaries.
//! Note that ZOmega embeds into ZZeta via ω = ζ² (odd-index coefficients are 0).
//!
//! One ring, two coefficient widths (see [`RingScalar`]), mirroring
//! [`super::zomega`]: [`ZZeta`] = `ZZetaG<Int>` is the fixed-width (i256)
//! hot-path type; [`ZZetaBig`] = `ZZetaG<rug::Integer>` is the unbounded
//! type the native √T route's grid/norm-equation stages need. The
//! real-subfield conversions (`from_real`/`rel_norm`/…) live beside the
//! big rings in `crate::rings::real` — they involve Z[g], which has no
//! fixed-width counterpart.

use num_complex::Complex64;
use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};
use super::types::{Int, RingScalar, INT_ZERO, INT_ONE, INT_NEG_ONE};

/// An element of Z[ζ], ζ = e^{iπ/8}, ζ^8 = −1, generic over the
/// coefficient width.
///
/// Represented as integer coefficients of the basis {1, ζ, ζ², ζ³, ζ⁴, ζ⁵, ζ⁶, ζ⁷}.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct ZZetaG<C: RingScalar> {
    pub(crate) a: C,
    pub(crate) b: C,
    pub(crate) c: C, // ζ² = ω
    pub(crate) d: C,
    pub(crate) e: C, // ζ⁴ = i
    pub(crate) f: C,
    pub(crate) g: C,
    pub(crate) h: C,
}

impl<C: RingScalar + Copy> Copy for ZZetaG<C> {}

/// Fixed-width (i256) instantiation — the hot-path type.
pub type ZZeta = ZZetaG<Int>;

/// Arbitrary-precision instantiation — the native-route type.
pub(crate) type ZZetaBig = ZZetaG<rug::Integer>;

// ─── Generic core (both widths) ───────────────────────────────────────────────

impl<C: RingScalar> ZZetaG<C> {
    #[inline]
    #[allow(clippy::too_many_arguments)] // 8 ring coefficients are intrinsic to Z[ζ].
    pub(crate) fn from_parts(a: C, b: C, c: C, d: C, e: C, f: C, g: C, h: C) -> Self {
        Self { a, b, c, d, e, f, g, h }
    }

    /// Construct from coefficients in ζ-power order.
    pub(crate) fn from_coeffs(v: [C; 8]) -> Self {
        let [a, b, c, d, e, f, g, h] = v;
        Self::from_parts(a, b, c, d, e, f, g, h)
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_i64(a: i64, b: i64, c: i64, d: i64, e: i64, f: i64, g: i64, h: i64) -> Self {
        Self::from_parts(
            C::from_i64(a), C::from_i64(b), C::from_i64(c), C::from_i64(d),
            C::from_i64(e), C::from_i64(f), C::from_i64(g), C::from_i64(h),
        )
    }

    pub(crate) fn zero() -> Self {
        Self::from_i64(0, 0, 0, 0, 0, 0, 0, 0)
    }

    pub(crate) fn one() -> Self {
        Self::from_i64(1, 0, 0, 0, 0, 0, 0, 0)
    }

    /// ζ itself.
    pub(crate) fn zeta() -> Self {
        Self::from_i64(0, 1, 0, 0, 0, 0, 0, 0)
    }

    /// Coefficients in ζ-power order, by reference.
    #[inline]
    pub(crate) fn coeffs(&self) -> [&C; 8] {
        [&self.a, &self.b, &self.c, &self.d, &self.e, &self.f, &self.g, &self.h]
    }

    #[inline]
    pub(crate) fn is_zero(&self) -> bool {
        self.coeffs().iter().all(|x| x.is_zero())
    }

    #[inline]
    pub(crate) fn add(&self, o: &Self) -> Self {
        let (s, t) = (self.coeffs(), o.coeffs());
        Self::from_coeffs(std::array::from_fn(|i| s[i].add(t[i])))
    }

    #[inline]
    pub(crate) fn sub(&self, o: &Self) -> Self {
        let (s, t) = (self.coeffs(), o.coeffs());
        Self::from_coeffs(std::array::from_fn(|i| s[i].sub(t[i])))
    }

    #[inline]
    pub(crate) fn neg(&self) -> Self {
        let s = self.coeffs();
        Self::from_coeffs(std::array::from_fn(|i| s[i].neg()))
    }

    /// Scale every coefficient by `x`.
    #[inline]
    pub(crate) fn scale(&self, x: &C) -> Self {
        let s = self.coeffs();
        Self::from_coeffs(std::array::from_fn(|i| s[i].mul(x)))
    }

    /// Multiplication in Z[ζ] modulo ζ^8 = −1.
    ///
    /// ζ^i · ζ^j = ζ^{i+j}; if i+j ≥ 8: ζ^{i+j} = −ζ^{i+j−8}, so
    /// result[k] = Σ_{i+j≡k (mod 8), i+j<8} p_i·q_j − Σ_{i+j≥8} p_i·q_j.
    #[inline]
    pub(crate) fn mul(&self, rhs: &Self) -> Self {
        let p = self.coeffs();
        let q = rhs.coeffs();
        // Unrolled with in-place accumulation: the fixed width inlines to
        // plain adds; the unbounded width avoids per-term temporaries.
        let mut out: [C; 8] = std::array::from_fn(|_| C::default());
        macro_rules! acc {
            ($k:expr, + $i:expr, $j:expr) => { out[$k].add_assign(&p[$i].mul(q[$j])); };
            ($k:expr, - $i:expr, $j:expr) => { out[$k].sub_assign(&p[$i].mul(q[$j])); };
        }
        for i in 0..8usize {
            for j in 0..8usize {
                if i + j < 8 {
                    acc!(i + j, + i, j);
                } else {
                    acc!(i + j - 8, - i, j);
                }
            }
        }
        Self::from_coeffs(out)
    }

    /// Multiply by ζ^m (coefficient rotation with ζ⁸ = −1 sign wrap).
    pub(crate) fn mul_by_zeta_power(&self, m: u32) -> Self {
        let m = (m % 16) as usize;
        let s = self.coeffs();
        Self::from_coeffs(std::array::from_fn(|j| {
            // Which source index lands on ζ^j, and with which sign?
            let i = (j + 16 - m) % 16;
            if i < 8 {
                s[i].clone()
            } else {
                s[i - 8].neg()
            }
        }))
    }

    /// Complex conjugate: ζ̄ = e^{−iπ/8} = ζ^{−1} = ζ^{15} = −ζ^7.
    ///
    /// conj_coeffs[0] = coeffs[0],
    /// conj_coeffs[k] = −coeffs[8−k]  for k = 1..7.
    pub(crate) fn conj(&self) -> Self {
        Self::from_parts(
            self.a.clone(),
            self.h.neg(),
            self.g.neg(),
            self.f.neg(),
            self.e.neg(),
            self.d.neg(),
            self.c.neg(),
            self.b.neg(),
        )
    }

    /// Convert to a floating-point complex number.
    pub(crate) fn to_complex(&self) -> Complex64 {
        use std::f64::consts::PI;
        let zeta = |k: u32| Complex64::from_polar(1.0, PI * f64::from(k) / 8.0);
        self.coeffs()
            .iter()
            .enumerate()
            .map(|(k, x)| x.to_f64_approx() * zeta(u32::try_from(k).expect("k < 8")))
            .sum()
    }

    /// Largest power of 2 dividing all coefficients (for normalization);
    /// large sentinel for the zero element.
    pub(crate) fn gcd_power_of_2(&self) -> u32 {
        let cap = Int::BITS - 1;
        self.coeffs()
            .iter()
            .map(|x| x.trailing_zeros_or(cap))
            .min()
            .expect("8 coefficients")
    }

    /// Divide all coefficients by 2^shift.
    #[inline]
    pub(crate) fn div2(&self, shift: u32) -> Self {
        let s = self.coeffs();
        Self::from_coeffs(std::array::from_fn(|i| s[i].shr_exact(shift)))
    }
}

// ─── Fixed-width surface (source compatibility for the hot path) ──────────────

impl ZZeta {
    pub(crate) const ZERO: Self = Self { a: INT_ZERO, b: INT_ZERO, c: INT_ZERO, d: INT_ZERO, e: INT_ZERO, f: INT_ZERO, g: INT_ZERO, h: INT_ZERO };
    pub(crate) const ONE:  Self = Self { a: INT_ONE,  b: INT_ZERO, c: INT_ZERO, d: INT_ZERO, e: INT_ZERO, f: INT_ZERO, g: INT_ZERO, h: INT_ZERO };
    pub(crate) const ZETA: Self = Self { a: INT_ZERO, b: INT_ONE,  c: INT_ZERO, d: INT_ZERO, e: INT_ZERO, f: INT_ZERO, g: INT_ZERO, h: INT_ZERO };
    /// ζ² = ω (the Clifford+T generator)
    pub(crate) const OMEGA: Self = Self { a: INT_ZERO, b: INT_ZERO, c: INT_ONE,  d: INT_ZERO, e: INT_ZERO, f: INT_ZERO, g: INT_ZERO, h: INT_ZERO };
    /// i = ζ⁴
    pub(crate) const I: Self = Self { a: INT_ZERO, b: INT_ZERO, c: INT_ZERO, d: INT_ZERO, e: INT_ONE,  f: INT_ZERO, g: INT_ZERO, h: INT_ZERO };
    #[cfg_attr(not(test), allow(dead_code))] // test-only since the PyZZeta surface was removed
    pub(crate) const NEG_ONE: Self = Self { a: INT_NEG_ONE, b: INT_ZERO, c: INT_ZERO, d: INT_ZERO, e: INT_ZERO, f: INT_ZERO, g: INT_ZERO, h: INT_ZERO };

    #[inline]
    #[allow(clippy::too_many_arguments)] // 8 ring coefficients are intrinsic to Z[ζ].
    pub(crate) const fn new(a: Int, b: Int, c: Int, d: Int, e: Int, f: Int, g: Int, h: Int) -> Self {
        Self { a, b, c, d, e, f, g, h }
    }

    /// Construct from small integer coefficients, converting each via `Int::from_i32`.
    #[inline]
    #[allow(clippy::too_many_arguments)] // 8 ring coefficients are intrinsic to Z[ζ].
    pub(crate) const fn from_i32(a: i32, b: i32, c: i32, d: i32, e: i32, f: i32, g: i32, h: i32) -> Self {
        Self::new(
            Int::from_i32(a), Int::from_i32(b), Int::from_i32(c), Int::from_i32(d),
            Int::from_i32(e), Int::from_i32(f), Int::from_i32(g), Int::from_i32(h),
        )
    }

    /// Coefficient of ζ^k, k = 0..7 (by value — `Copy` instantiation).
    ///
    /// # Panics
    /// Panics if `k >= 8` (programmer error, like an out-of-bounds index).
    #[inline]
    pub(crate) fn coeff(self, k: usize) -> Int {
        match k {
            0 => self.a, 1 => self.b, 2 => self.c, 3 => self.d,
            4 => self.e, 5 => self.f, 6 => self.g, 7 => self.h,
            _ => panic!("ZZeta::coeff: index {k} out of range"),
        }
    }

    /// Multiply by √2 = ζ² − ζ⁶  (since e^{iπ/4} − e^{6iπ/8} = (1+i)/√2 − (−1+i)/√2 = √2).
    #[cfg_attr(not(test), allow(dead_code))] // test-only since the PyZZeta surface was removed
    pub(crate) fn mul_sqrt2(self) -> Self {
        let rhs = Self::from_i32(0, 0, 1, 0, 0, 0, -1, 0);
        self * rhs
    }

    /// Embed a ZOmega element: ω = ζ², so (a + bω + cω² + dω³) → (a + bζ² + cζ⁴ + dζ⁶).
    #[cfg_attr(not(test), allow(dead_code))] // test-only since the PyZZeta surface was removed
    pub(crate) fn from_zomega(a: Int, b: Int, c: Int, d: Int) -> Self {
        Self::new(a, INT_ZERO, b, INT_ZERO, c, INT_ZERO, d, INT_ZERO)
    }
}

// ─── Big-width extras (native √T route) ───────────────────────────────────────

impl ZZetaBig {
    pub(crate) fn from_int(x: rug::Integer) -> Self {
        let mut z = Self::zero();
        z.a = x;
        z
    }
}

// ─── Arithmetic operators (fixed-width instantiation) ─────────────────────────

impl Add for ZZeta {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        ZZetaG::add(&self, &rhs)
    }
}

impl Sub for ZZeta {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        ZZetaG::sub(&self, &rhs)
    }
}

impl Neg for ZZeta {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        ZZetaG::neg(&self)
    }
}

impl Mul for ZZeta {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        ZZetaG::mul(&self, &rhs)
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

    /// Cross-width oracle: both instantiations implement the SAME ring.
    #[test]
    fn test_fixed_vs_big_agree() {
        let mut state = 0xfeed_beef_1234_5678_u64;
        let mut rnd = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 34) as i64) - (1 << 29)
        };
        for _ in 0..100 {
            let xs: [i64; 8] = std::array::from_fn(|_| rnd());
            let ys: [i64; 8] = std::array::from_fn(|_| rnd());
            let xf = ZZeta::from_i64(xs[0], xs[1], xs[2], xs[3], xs[4], xs[5], xs[6], xs[7]);
            let yf = ZZeta::from_i64(ys[0], ys[1], ys[2], ys[3], ys[4], ys[5], ys[6], ys[7]);
            let xb = ZZetaBig::from_i64(xs[0], xs[1], xs[2], xs[3], xs[4], xs[5], xs[6], xs[7]);
            let yb = ZZetaBig::from_i64(ys[0], ys[1], ys[2], ys[3], ys[4], ys[5], ys[6], ys[7]);
            let pairs: [(ZZeta, ZZetaBig); 5] = [
                (ZZetaG::mul(&xf, &yf), xb.mul(&yb)),
                (ZZetaG::add(&xf, &yf), xb.add(&yb)),
                (ZZetaG::sub(&xf, &yf), xb.sub(&yb)),
                (ZZetaG::conj(&xf), xb.conj()),
                (xf.mul_by_zeta_power(5), xb.mul_by_zeta_power(5)),
            ];
            for (ff, bb) in pairs {
                let (fc, bc) = (ff.coeffs(), bb.coeffs());
                for i in 0..8 {
                    assert_eq!(format!("{}", fc[i]), format!("{}", bc[i]));
                }
            }
        }
    }
}
