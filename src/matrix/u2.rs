//! 2×2 unitary matrices over a cyclotomic ring.
//!
//! Every unitary matrix with entries in R/√2^k is represented as
//!
//!   U = (1/√2^k) · [[u11, u12], [u21, u22]]
//!
//! where u11, u12, u21, u22 ∈ R are arbitrary ring elements.
//!
//! This works for both Clifford+T (ZOmega) and Clifford+√T (ZZeta).

use num_complex::Complex64;
use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};
use crate::rings::zomega::ZOmega;
use crate::rings::zzeta::ZZeta;

// ─── Trait ────────────────────────────────────────────────────────────────────

/// Minimal interface needed to build a U2 matrix.
pub trait RingElem: Copy + Add<Output = Self> + Neg<Output = Self> {
    fn conj(self) -> Self;
    fn to_complex(self) -> Complex64;
    fn zero() -> Self;
    fn one() -> Self;
    fn i() -> Self;  // imaginary unit
    fn omega() -> Self;  // ω = e^{iπ/4}
}

impl RingElem for ZOmega {
    fn conj(self) -> Self { self.conj() }
    fn to_complex(self) -> Complex64 { self.to_complex() }
    fn zero() -> Self { Self::ZERO }
    fn one() -> Self { Self::ONE }
    fn i() -> Self { Self::I }
    fn omega() -> Self { Self::OMEGA }
}

impl RingElem for ZZeta {
    fn conj(self) -> Self { self.conj() }
    fn to_complex(self) -> Complex64 { self.to_complex() }
    fn zero() -> Self { Self::ZERO }
    fn one() -> Self { Self::ONE }
    fn i() -> Self { Self::I }
    fn omega() -> Self { Self::OMEGA }
}

// ─── U2<R> ────────────────────────────────────────────────────────────────────

/// Unitary matrix  U = [[u11, u12], [u21, u22]] / √2^k.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct U2<R: RingElem + Mul<Output = R> + Sub<Output = R>> {
    /// Numerator elements
    pub(crate) u11: R,
    pub(crate) u12: R,
    pub(crate) u21: R,
    pub(crate) u22: R,
    /// Denominator exponent: actual matrix entries are divided by √2^k.
    pub(crate) k: u32,
}

impl<R: RingElem + Mul<Output = R> + Sub<Output = R>> U2<R> {
    pub(crate) const fn new(u11: R, u12: R, u21: R, u22: R, k: u32) -> Self {
        Self { u11, u12, u21, u22, k }
    }

    /// Hermitian adjoint: U† = conj-transpose = [[ū11, ū21], [ū12, ū22]] / √2^k.
    pub(crate) fn dagger(&self) -> Self {
        Self {
            u11: self.u11.conj(),
            u12: self.u21.conj(),
            u21: self.u12.conj(),
            u22: self.u22.conj(),
            k:  self.k,
        }
    }

    /// Convert to 2×2 complex float matrix (row-major [[a,b],[c,d]]).
    pub fn to_float(self) -> [[Complex64; 2]; 2] {
        let scale = 1.0 / (f64::from(self.k) / 2.0).exp2();  // 1 / √2^k = 2^{-k/2}
        [
            [self.u11.to_complex() * scale, self.u12.to_complex() * scale],
            [self.u21.to_complex() * scale, self.u22.to_complex() * scale],
        ]
    }

    /// Diamond distance to another U2 matrix (both must be unitary up to global phase).
    ///
    /// dist = √(max(0, 1 − |tr(U·V†)|²/4))
    ///
    /// Each ring element is converted to Complex64 individually before multiplying.
    /// This avoids i64 overflow in ring arithmetic when k is large (≳50 for ZOmega),
    /// while still deferring the denominator scaling to the final float step.
    pub(crate) fn diamond_distance(&self, other: &Self) -> f64 {
        let p = self.u11.to_complex() * other.u11.to_complex().conj()
              + self.u12.to_complex() * other.u12.to_complex().conj()
              + self.u21.to_complex() * other.u21.to_complex().conj()
              + self.u22.to_complex() * other.u22.to_complex().conj();
        let denom = 4.0 * (2.0_f64).powi((self.k + other.k) as i32);
        let t = p.norm_sqr() / denom;
        (1.0_f64 - t).max(0.0).sqrt()
    }
}

// ─── Helpful constructors ────────────────────────────────────────────────────

impl <R: RingElem + Mul<Output = R> + Sub<Output = R>> U2<R> {
    /// Identity matrix: [[1,0],[0,1]] / √2^0
    pub(crate) fn eye() -> Self {
        Self::new(R::one(), R::zero(), R::zero(), R::one(), 0)
    }

    /// H gate: [[1,1],[1,−1]] / √2.
    pub(crate) fn h() -> Self {
        Self::new(R::one(), R::one(), R::one(), -R::one(), 1)
    }

    /// S gate: diag(1, i).
    pub(crate) fn s() -> Self {
        Self::new(R::one(), R::zero(), R::zero(), R::i(), 0)
    }

    /// T gate: [[1,0],[0,ω]] (ω = e^{iπ/4} in both rings).
    pub(crate) fn t() -> Self {
        Self::new(R::one(), R::zero(), R::zero(), R::omega(), 0)
    }

    pub(crate) fn x() -> Self {
        Self::new(R::zero(), R::one(), R::one(), R::zero(), 0)
    }

    pub(crate) fn y() -> Self {
        Self::new(R::zero(), -R::i(), R::i(), R::zero(), 0)
    }

    pub(crate) fn z() -> Self {
        Self::new(R::one(), R::zero(), R::zero(), -R::one(), 0)
    }
}

impl U2<ZOmega> {
    /// Fully reduce the denominator exponent (sde) for a Z[ω] unitary by
    /// repeatedly dividing every entry by √2 = ω − ω³ while all four stay in
    /// Z[ω]. `Mul` accumulates `k` without reducing, so this recovers the true
    /// lde of a product. Mirror of the ZZeta `reduced()`.
    #[allow(dead_code)] // consumed by the python+trace diag pyfunctions and oracle tests
    pub(crate) fn reduced(self) -> Self {
        let sqrt2 = ZOmega::OMEGA - ZOmega::OMEGA * ZOmega::OMEGA * ZOmega::OMEGA;
        let mut m = self;
        while m.k > 0 {
            let y11 = m.u11 * sqrt2;
            let y12 = m.u12 * sqrt2;
            let y21 = m.u21 * sqrt2;
            let y22 = m.u22 * sqrt2;
            let divisible = y11.gcd_power_of_2() >= 1
                && y12.gcd_power_of_2() >= 1
                && y21.gcd_power_of_2() >= 1
                && y22.gcd_power_of_2() >= 1;
            if !divisible {
                break;
            }
            m = Self {
                u11: y11.div2(1),
                u12: y12.div2(1),
                u21: y21.div2(1),
                u22: y22.div2(1),
                k: m.k - 1,
            };
        }
        m
    }
}

impl U2<ZZeta> {
    /// Q gate: [[1,0],[0,ζ]] / √2^0 (for U2Q)
    pub(crate) fn q() -> Self {
        Self::new(ZZeta::ONE, ZZeta::ZERO, ZZeta::ZERO, ZZeta::ZETA, 0)
    }

    /// Fully reduce the denominator exponent: repeatedly divide every
    /// entry by √2 = ζ² − ζ⁶ while all four stay in Z[ζ₁₆], decrementing
    /// `k` each time. `Mul` accumulates `k` without reducing, so `k` of
    /// a product is only an upper bound on the true lde until this is
    /// called. Uses: x is divisible by √2 ⟺ all coefficients of x·√2
    /// are even (then x/√2 = (x·√2)/2).
    pub fn reduced(self) -> Self {
        let z2 = ZZeta::ZETA * ZZeta::ZETA;
        let z6 = z2 * z2 * z2;
        let sqrt2 = z2 - z6;
        let mut m = self;
        while m.k > 0 {
            let y11 = m.u11 * sqrt2;
            let y12 = m.u12 * sqrt2;
            let y21 = m.u21 * sqrt2;
            let y22 = m.u22 * sqrt2;
            let divisible = y11.gcd_power_of_2() >= 1
                && y12.gcd_power_of_2() >= 1
                && y21.gcd_power_of_2() >= 1
                && y22.gcd_power_of_2() >= 1;
            if !divisible {
                break;
            }
            m = Self {
                u11: y11.div2(1),
                u12: y12.div2(1),
                u21: y21.div2(1),
                u22: y22.div2(1),
                k: m.k - 1,
            };
        }
        m
    }
}

// ─── Multiplication (matrix product) ─────────────────────────────────────────

/// Matrix product. `k` accumulates (`self.k + rhs.k`) without reduction;
/// call `reduced()` to recover the true lde.
impl<R: RingElem + Mul<Output = R> + Sub<Output = R>> Mul for U2<R> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let k = self.k + rhs.k;
        let u11 = self.u11 * rhs.u11 + self.u12 * rhs.u21;
        let u12 = self.u11 * rhs.u12 + self.u12 * rhs.u22;
        let u21 = self.u21 * rhs.u11 + self.u22 * rhs.u21;
        let u22 = self.u21 * rhs.u12 + self.u22 * rhs.u22;
        Self { u11, u12, u21, u22, k }
    }
}

// ─── Display ─────────────────────────────────────────────────────────────────

impl<R: RingElem + Mul<Output = R> + Sub<Output = R> + fmt::Display> fmt::Display for U2<R> {
    /// Formats as `[[u11, u12], [u21, u22]] / √2^k`, omitting `/ √2^0`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[[{}, {}], [{}, {}]]",
               self.u11, self.u12, self.u21, self.u22)?;
        if self.k > 0 { write!(f, " / √2^{}", self.k)?; }
        Ok(())
    }
}

// ─── Concrete type aliases ────────────────────────────────────────────────────

/// Clifford+T unitary matrix (denominator in Z[ω]).
pub type U2T = U2<ZOmega>;

/// Clifford+√T unitary matrix (denominator in Z[ζ]).
pub type U2Q = U2<ZZeta>;

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rings::ZOmega;
    use rand::Rng;

    /// Identity: [[1,0],[0,1]] / √2^0
    fn identity_t() -> U2T {
        U2T::new(ZOmega::ONE, ZOmega::ZERO, ZOmega::ZERO, ZOmega::ONE, 0)
    }

    fn random_zomega() -> ZOmega {
        let mut rng = rand::rng();
        ZOmega::from_i32(
            rng.random_range(-1000..1000),
            rng.random_range(-1000..1000),
            rng.random_range(-1000..1000),
            rng.random_range(-1000..1000),
        )
    }

    /// i·H = [[i,i],[i,−i]] / √2.
    fn h_gate() -> U2T {
        let i  = ZOmega::I;
        let ni = -ZOmega::I;
        U2T::new(i, i, i, ni, 1)
    }

    #[test]
    fn test_identity_diamond_distance() {
        let id = identity_t();
        assert!(id.diamond_distance(&id) < 1e-12, "d(I,I) should be 0");
    }

    #[test]
    fn test_dagger_is_inverse() {
        let h = h_gate();
        let hh = h * h.dagger();
        let id = identity_t();
        assert!(
            hh.diamond_distance(&id) < 1e-10,
            "H·H† should be identity, dist={}",
            hh.diamond_distance(&id)
        );
    }

    #[test]
    fn test_mul_associativity() {
        let h = h_gate();
        let lhs = (h * h) * h;
        let rhs = h * (h * h);
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn test_to_float_h_gate() {
        let h = h_gate();
        let m = h.to_float();
        let r2inv = 1.0 / std::f64::consts::SQRT_2;
        // i·H = [[i,i],[i,−i]] / √2 — entries are purely imaginary
        assert!((m[0][0].im - r2inv).abs() < 1e-10, "m[0][0].im={}", m[0][0].im);
        assert!((m[0][1].im - r2inv).abs() < 1e-10, "m[0][1].im={}", m[0][1].im);
        assert!((m[1][0].im - r2inv).abs() < 1e-10, "m[1][0].im={}", m[1][0].im);
        assert!((m[1][1].im + r2inv).abs() < 1e-10, "m[1][1].im={}", m[1][1].im);
    }

    #[test]
    fn test_u2q_identity() {
        let id = U2Q::new(ZZeta::ONE, ZZeta::ZERO, ZZeta::ZERO, ZZeta::ONE, 0);
        assert!(id.diamond_distance(&id) < 1e-12);
    }

    #[test]
    fn test_random_mul() {
        for _ in 0..100 {
            let a = U2T::new(random_zomega(), random_zomega(), random_zomega(), random_zomega(), 100);
            let b = U2T::new(random_zomega(), random_zomega(), random_zomega(), random_zomega(), 100);
            let c = U2T::new(random_zomega(), random_zomega(), random_zomega(), random_zomega(), 100);
            let lhs = (a * b) * c;
            let rhs = a * (b * c);
            assert_eq!(lhs, rhs, "Associativity failed for random U2T");
        }
    }

    #[test]
    fn test_h_gate_mul() {
        let eye = U2T::eye();
        let h = U2T::h();
        let hdg = h.dagger();
        let hh = h * hdg;
        assert!(hh.diamond_distance(&eye) < 1e-10, "H·H† should be identity");

        let eye_q = U2Q::eye();
        let h_q = U2Q::h();
        let hdg_q = h_q.dagger();
        let hh_q = h_q * hdg_q;
        assert!(hh_q.diamond_distance(&eye_q) < 1e-10, "H·H† should be identity");
    }

    #[test]
    fn test_s_gate_mul() {
        let eye = U2T::eye();
        let s = U2T::s();
        let sdg = s.dagger();
        let ss = s * sdg;
        assert!(ss.diamond_distance(&eye) < 1e-10, "S·S† should be identity");

        let eye_q = U2Q::eye();
        let s_q = U2Q::s();
        let sdg_q = s_q.dagger();
        let ss_q = s_q * sdg_q;
        assert!(ss_q.diamond_distance(&eye_q) < 1e-10, "S·S† should be identity");
    }

    #[test]
    fn test_t_gate_mul() {
        let eye = U2T::eye();
        let t = U2T::t();
        let tdg = t.dagger();
        let tt = t * tdg;
        assert!(tt.diamond_distance(&eye) < 1e-10, "T·T† should be identity");

        let eye_q = U2Q::eye();
        let t_q = U2Q::t();
        let tdg_q = t_q.dagger();
        let tt_q = t_q * tdg_q;
        assert!(tt_q.diamond_distance(&eye_q) < 1e-10, "T·T† should be identity");
    }

    #[test]
    fn test_q_gate_mul() {
        let eye_q = U2Q::eye();
        let q = U2Q::q();
        let qdg = q.dagger();
        let qq = q * qdg;
        assert!(qq.diamond_distance(&eye_q) < 1e-10, "Q·Q† should be identity");
    }
}
