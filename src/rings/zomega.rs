//! Z[ω] — the ring of integers extended by ω = e^{iπ/4}.
//!
//! Every element has the form  a + b·ω + c·ω² + d·ω³
//! with a,b,c,d ∈ ℤ and the relation ω⁴ = −1.
//!
//! This is the coefficient ring for exactly-implementable Clifford+T unitaries
//! (entries of the SU(2) matrix live in Z[ω] / √2^k).
//!
//! One ring, two coefficient widths (see [`RingScalar`]):
//! [`ZOmega`] = `ZOmegaG<Int>` is the fixed-width (i256) hot-path type the
//! lattice pipeline and exact decomposer use; [`ZOmegaBig`] =
//! `ZOmegaG<rug::Integer>` is the unbounded type the native z-rotation
//! route's grid/norm-equation stages need (their intermediates square and
//! fourth-power the coefficients). Both share every algebraic operation
//! below; the cross-width oracle test at the bottom pins the equivalence.

// Scaffolding: consumers land incrementally in later commits.
#![cfg_attr(not(test), allow(dead_code))]

use num_complex::Complex64;
use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};
use super::types::{Int, RingScalar, INT_ZERO, INT_ONE, INT_NEG_ONE};

/// An element of Z[ω], ω = e^{iπ/4}, ω⁴ = −1, generic over the
/// coefficient width.
///
/// Represented as integer coefficients of the basis {1, ω, ω², ω³}.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct ZOmegaG<C: RingScalar> {
    pub(crate) a: C,
    pub(crate) b: C,
    pub(crate) c: C, // ω² = i
    pub(crate) d: C,
}

impl<C: RingScalar + Copy> Copy for ZOmegaG<C> {}

/// Fixed-width (i256) instantiation — the hot-path type.
pub type ZOmega = ZOmegaG<Int>;

/// Arbitrary-precision instantiation — the native-route type.
pub(crate) type ZOmegaBig = ZOmegaG<rug::Integer>;

// ─── Generic core (both widths) ───────────────────────────────────────────────

impl<C: RingScalar> ZOmegaG<C> {
    #[inline]
    pub(crate) fn from_parts(a: C, b: C, c: C, d: C) -> Self {
        Self { a, b, c, d }
    }

    /// Construct from coefficients in ω-power order.
    #[cfg_attr(not(test), allow(dead_code))] // parity with `ZZetaG::from_coeffs`
    pub(crate) fn from_coeffs(v: [C; 4]) -> Self {
        let [a, b, c, d] = v;
        Self::from_parts(a, b, c, d)
    }

    /// Coefficients in ω-power order, by reference.
    #[inline]
    #[cfg_attr(not(test), allow(dead_code))] // parity with `ZZetaG::coeffs`
    pub(crate) fn coeffs(&self) -> [&C; 4] {
        [&self.a, &self.b, &self.c, &self.d]
    }

    #[inline]
    pub(crate) fn from_i64(a: i64, b: i64, c: i64, d: i64) -> Self {
        Self::from_parts(C::from_i64(a), C::from_i64(b), C::from_i64(c), C::from_i64(d))
    }

    pub(crate) fn zero() -> Self {
        Self::from_i64(0, 0, 0, 0)
    }

    pub(crate) fn one() -> Self {
        Self::from_i64(1, 0, 0, 0)
    }

    #[inline]
    pub(crate) fn is_zero(&self) -> bool {
        self.a.is_zero() && self.b.is_zero() && self.c.is_zero() && self.d.is_zero()
    }

    #[inline]
    pub(crate) fn add(&self, o: &Self) -> Self {
        Self::from_parts(
            self.a.add(&o.a),
            self.b.add(&o.b),
            self.c.add(&o.c),
            self.d.add(&o.d),
        )
    }

    #[inline]
    pub(crate) fn sub(&self, o: &Self) -> Self {
        Self::from_parts(
            self.a.sub(&o.a),
            self.b.sub(&o.b),
            self.c.sub(&o.c),
            self.d.sub(&o.d),
        )
    }

    #[inline]
    pub(crate) fn neg(&self) -> Self {
        Self::from_parts(self.a.neg(), self.b.neg(), self.c.neg(), self.d.neg())
    }

    /// Multiplication in Z[ω] modulo ω⁴ = −1.
    ///
    /// (a+bω+cω²+dω³)(e+fω+gω²+hω³) reduces to:
    ///   [1 ]: ae − bh − cg − df
    ///   [ω ]: af + be − ch − dg
    ///   [ω²]: ag + bf + ce − dh
    ///   [ω³]: ah + bg + cf + de
    #[inline]
    pub(crate) fn mul(&self, o: &Self) -> Self {
        let (a, b, c, d) = (&self.a, &self.b, &self.c, &self.d);
        let (e, f, g, h) = (&o.a, &o.b, &o.c, &o.d);
        Self::from_parts(
            a.mul(e).sub(&b.mul(h)).sub(&c.mul(g)).sub(&d.mul(f)),
            a.mul(f).add(&b.mul(e)).sub(&c.mul(h)).sub(&d.mul(g)),
            a.mul(g).add(&b.mul(f)).add(&c.mul(e)).sub(&d.mul(h)),
            a.mul(h).add(&b.mul(g)).add(&c.mul(f)).add(&d.mul(e)),
        )
    }

    #[inline]
    pub(crate) fn scale(&self, x: &C) -> Self {
        Self::from_parts(self.a.mul(x), self.b.mul(x), self.c.mul(x), self.d.mul(x))
    }

    pub(crate) fn pow(&self, e: u64) -> Self {
        let mut out = Self::one();
        let mut base = self.clone();
        let mut e = e;
        while e > 0 {
            if e & 1 == 1 {
                out = out.mul(&base);
            }
            base = base.mul(&base);
            e >>= 1;
        }
        out
    }

    /// Complex conjugate: ω̄ = e^{−iπ/4} = ω⁷ = ω⁴·ω³ = −ω³.
    /// So conj(a + bω + cω² + dω³) = a − dω − cω² − bω³.
    #[inline]
    pub(crate) fn conj(&self) -> Self {
        Self::from_parts(self.a.clone(), self.d.neg(), self.c.neg(), self.b.neg())
    }

    /// √2-conjugate (ω ↦ −ω): negates the odd-power coefficients.
    #[inline]
    pub(crate) fn conj_sq2(&self) -> Self {
        Self::from_parts(self.a.clone(), self.b.neg(), self.c.clone(), self.d.neg())
    }

    /// ω·x: coefficient rotation with ω⁴ = −1 wraparound.
    #[inline]
    pub(crate) fn mul_by_omega(&self) -> Self {
        Self::from_parts(self.d.neg(), self.a.clone(), self.b.clone(), self.c.clone())
    }

    /// ω⁻¹·x = −ω³·x.
    #[inline]
    pub(crate) fn mul_by_omega_inv(&self) -> Self {
        Self::from_parts(self.b.clone(), self.c.clone(), self.d.clone(), self.a.neg())
    }

    pub(crate) fn mul_by_omega_power(&self, n: i32) -> Self {
        let mut out = self.clone();
        for _ in 0..n.rem_euclid(8) {
            out = out.mul_by_omega();
        }
        out
    }

    /// Low bits of the coefficients, high-to-low from the ω³ coefficient:
    /// `(d&1)<<3 | (c&1)<<2 | (b&1)<<1 | (a&1)`.
    pub(crate) fn residue(&self) -> u8 {
        (u8::from(self.d.is_odd()) << 3)
            | (u8::from(self.c.is_odd()) << 2)
            | (u8::from(self.b.is_odd()) << 1)
            | u8::from(self.a.is_odd())
    }

    /// Field norm N(x) ∈ Z (nonnegative for nonzero x up to sign convention).
    pub(crate) fn norm(&self) -> C {
        let sq = self
            .a
            .mul(&self.a)
            .add(&self.b.mul(&self.b))
            .add(&self.c.mul(&self.c))
            .add(&self.d.mul(&self.d));
        let cross = self
            .a
            .mul(&self.b)
            .add(&self.b.mul(&self.c))
            .add(&self.c.mul(&self.d))
            .sub(&self.d.mul(&self.a));
        sq.mul(&sq).sub(&C::from_i64(2).mul(&cross.mul(&cross)))
    }

    /// Euclidean divmod via norm rounding.
    pub(crate) fn divmod(&self, o: &Self) -> (Self, Self) {
        let p = self.mul(&o.conj()).mul(&o.conj().conj_sq2()).mul(&o.conj_sq2());
        let k = o.norm();
        let q = Self::from_parts(
            p.a.div_round(&k),
            p.b.div_round(&k),
            p.c.div_round(&k),
            p.d.div_round(&k),
        );
        let r = self.sub(&o.mul(&q));
        (q, r)
    }

    pub(crate) fn rem(&self, o: &Self) -> Self {
        self.divmod(o).1
    }

    /// Associates: each divides the other.
    pub(crate) fn sim(a: &Self, b: &Self) -> bool {
        !a.is_zero() && !b.is_zero() && a.rem(b).is_zero() && b.rem(a).is_zero()
    }

    pub(crate) fn gcd(a: &Self, b: &Self) -> Self {
        let mut a = a.clone();
        let mut b = b.clone();
        while !b.is_zero() {
            let r = a.rem(&b);
            a = b;
            b = r;
        }
        a
    }

    /// Convert to a floating-point complex number.
    /// ω = e^{iπ/4} = (1+i)/√2, ω² = i, ω³ = (−1+i)/√2.
    pub(crate) fn to_complex(&self) -> Complex64 {
        use std::f64::consts::FRAC_1_SQRT_2;
        let (a, b, c, d) = (
            self.a.to_f64_approx(),
            self.b.to_f64_approx(),
            self.c.to_f64_approx(),
            self.d.to_f64_approx(),
        );
        Complex64::new(a + (b - d) * FRAC_1_SQRT_2, c + (b + d) * FRAC_1_SQRT_2)
    }

    /// Largest power of 2 that divides all four coefficients (a large
    /// sentinel for the zero element, matching the fixed-width behavior).
    pub(crate) fn gcd_power_of_2(&self) -> u32 {
        let cap = Int::BITS - 1;
        self.a
            .trailing_zeros_or(cap)
            .min(self.b.trailing_zeros_or(cap))
            .min(self.c.trailing_zeros_or(cap))
            .min(self.d.trailing_zeros_or(cap))
    }

    /// Divide all coefficients by 2^shift (caller must ensure divisibility).
    #[inline]
    pub(crate) fn div2(&self, shift: u32) -> Self {
        Self::from_parts(
            self.a.shr_exact(shift),
            self.b.shr_exact(shift),
            self.c.shr_exact(shift),
            self.d.shr_exact(shift),
        )
    }
}

// ─── Fixed-width surface (source compatibility for the hot path) ──────────────

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

    /// Multiply by √2 in Z[ω], using √2 = ω − ω³
    /// (= (1+i)/√2 − (−1+i)/√2 = 2/√2 = √2).
    #[cfg_attr(not(test), allow(dead_code))] // test-only since the PyZOmega surface was removed
    pub(crate) fn mul_sqrt2(self) -> Self {
        let rhs = Self { a: INT_ZERO, b: INT_ONE, c: INT_ZERO, d: INT_NEG_ONE };
        self * rhs
    }
}

// ─── Arithmetic operators (fixed-width instantiation) ─────────────────────────

impl Add for ZOmega {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        ZOmegaG::add(&self, &rhs)
    }
}

impl Sub for ZOmega {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        ZOmegaG::sub(&self, &rhs)
    }
}

impl Neg for ZOmega {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        ZOmegaG::neg(&self)
    }
}

impl Mul for ZOmega {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        ZOmegaG::mul(&self, &rhs)
    }
}

// ─── Big-width extras (native z-rotation route) ───────────────────────────────

impl ZOmegaBig {
    pub(crate) fn from_int(x: rug::Integer) -> Self {
        Self::from_parts(x, rug::Integer::new(), rug::Integer::new(), rug::Integer::new())
    }

    /// Real part in the standard embedding: a + (b−d)/√2.
    pub(crate) fn real(&self, prec: u32) -> super::types::MpFloat {
        use super::types::MpFloat;
        MpFloat::with_val(prec, &self.a)
            + MpFloat::with_val(prec, rug::Integer::from(&self.b - &self.d))
                * super::types::cached_inv_sqrt2(prec)
    }

    /// Imag part: c + (b+d)/√2.
    pub(crate) fn imag(&self, prec: u32) -> super::types::MpFloat {
        use super::types::MpFloat;
        MpFloat::with_val(prec, &self.c)
            + MpFloat::with_val(prec, rug::Integer::from(&self.b + &self.d))
                * super::types::cached_inv_sqrt2(prec)
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

    /// Cross-width oracle: both instantiations implement the SAME ring.
    /// Random elements agree on every shared operation after conversion.
    #[test]
    fn test_fixed_vs_big_agree() {
        let mut state = 0x1234_5678_9abc_def0_u64;
        let mut rnd = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 33) as i64) - (1 << 30)
        };
        for _ in 0..200 {
            let (a, b, c, d) = (rnd(), rnd(), rnd(), rnd());
            let (e, f, g, h) = (rnd(), rnd(), rnd(), rnd());
            let xf = ZOmega::from_i64(a, b, c, d);
            let yf = ZOmega::from_i64(e, f, g, h);
            let xb = ZOmegaBig::from_i64(a, b, c, d);
            let yb = ZOmegaBig::from_i64(e, f, g, h);
            let pairs: [(ZOmega, ZOmegaBig); 6] = [
                (ZOmegaG::mul(&xf, &yf), xb.mul(&yb)),
                (ZOmegaG::add(&xf, &yf), xb.add(&yb)),
                (ZOmegaG::sub(&xf, &yf), xb.sub(&yb)),
                (ZOmegaG::conj(&xf), xb.conj()),
                (xf.conj_sq2(), xb.conj_sq2()),
                (xf.mul_by_omega(), xb.mul_by_omega()),
            ];
            for (ff, bb) in pairs {
                for (cf, cb) in [(&ff.a, &bb.a), (&ff.b, &bb.b), (&ff.c, &bb.c), (&ff.d, &bb.d)] {
                    assert_eq!(format!("{cf}"), format!("{cb}"));
                }
            }
            assert_eq!(format!("{}", xf.norm()), format!("{}", xb.norm()));
            assert_eq!(xf.residue(), xb.residue());
            if !yf.is_zero() {
                let (qf, rf) = xf.divmod(&yf);
                let (qb, rb) = xb.divmod(&yb);
                assert_eq!(format!("{}", qf.a), format!("{}", qb.a));
                assert_eq!(format!("{}", rf.a), format!("{}", rb.a));
            }
        }
    }
}
