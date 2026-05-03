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
#[cfg(feature = "python")]
use crate::rings::zomega::PyZOmega;
#[cfg(feature = "python")]
use crate::rings::zzeta::PyZZeta;

// ─── Trait ────────────────────────────────────────────────────────────────────

/// Minimal interface needed to build a U2 matrix.
pub trait RingElem: Copy + Add<Output = Self> + Neg<Output = Self> {
    fn conj(self) -> Self;
    fn to_complex(self) -> Complex64;
    fn zero() -> Self;
    fn one() -> Self;
    fn i() -> Self;  // imaginary unit
    fn omega() -> Self;  // ω = e^{iπ/4}
    fn root_of_unity() -> Self;  // ω = e^{iπ/4} or ζ = e^{iπ/8}
}

impl RingElem for ZOmega {
    fn conj(self) -> Self { self.conj() }
    fn to_complex(self) -> Complex64 { self.to_complex() }
    fn zero() -> Self { Self::ZERO }
    fn one() -> Self { Self::ONE }
    fn i() -> Self { Self::I }
    fn omega() -> Self { Self::OMEGA }
    fn root_of_unity() -> Self { Self::OMEGA }
}

impl RingElem for ZZeta {
    fn conj(self) -> Self { self.conj() }
    fn to_complex(self) -> Complex64 { self.to_complex() }
    fn zero() -> Self { Self::ZERO }
    fn one() -> Self { Self::ONE }
    fn i() -> Self { Self::I }
    fn omega() -> Self { Self::OMEGA }
    fn root_of_unity() -> Self { Self::ZETA }
}

// ─── U2<R> ────────────────────────────────────────────────────────────────────

/// Unitary matrix  U = [[u11, u12], [u21, u22]] / √2^k.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct U2<R: RingElem + Mul<Output = R> + Sub<Output = R>> {
    /// Numerator elements
    pub u11: R,
    pub u12: R,
    pub u21: R,
    pub u22: R,
    /// Denominator exponent: actual matrix entries are divided by √2^k.
    pub k: u32,
}

impl<R: RingElem + Mul<Output = R> + Sub<Output = R>> U2<R> {
    pub const fn new(u11: R, u12: R, u21: R, u22: R, k: u32) -> Self {
        Self { u11, u12, u21, u22, k }
    }

    /// Hermitian adjoint: U† = conj-transpose = [[ū11, ū21], [ū12, ū22]] / √2^k.
    pub fn dagger(&self) -> Self {
        Self {
            u11: self.u11.conj(),
            u12: self.u21.conj(),
            u21: self.u12.conj(),
            u22: self.u22.conj(),
            k:  self.k,
        }
    }

    /// Convert to 2×2 complex float matrix (row-major [[a,b],[c,d]]).
    pub fn to_float(&self) -> [[Complex64; 2]; 2] {
        let scale = 1.0 / (self.k as f64 / 2.0).exp2();  // 1 / √2^k = 2^{-k/2}
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
    pub fn diamond_distance(&self, other: &Self) -> f64 {
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
    pub fn eye() -> Self {
        Self::new(R::one(), R::zero(), R::zero(), R::one(), 0)
    }

    /// H gate: [[1,1],[1,−1]] / √2.
    pub fn h() -> Self {
        Self::new(R::one(), R::one(), R::one(), -R::one(), 1)
    }

    /// S gate: [[1,0],[0,i]] / √2^0.
    pub fn s() -> Self {
        Self::new(R::one(), R::zero(), R::zero(), R::i().conj(), 0)
    }

    /// T gate: [[1,0],[0,ω]] / √2^0 (for U2T)
    pub fn t() -> Self {
        Self::new(R::one(), R::zero(), R::zero(), R::omega(), 0)
    }

    pub fn x() -> Self {
        Self::new(R::zero(), R::one(), R::one(), R::zero(), 0)
    }

    pub fn y() -> Self {
        Self::new(R::zero(), -R::i(), R::i(), R::zero(), 0)
    }

    pub fn z() -> Self {
        Self::new(R::one(), R::zero(), R::zero(), -R::one(), 0)
    }
}

impl U2<ZZeta> {
    /// Q gate: [[1,0],[0,ζ]] / √2^0 (for U2Q)
    pub fn q() -> Self {
        Self::new(ZZeta::ONE, ZZeta::ZERO, ZZeta::ZERO, ZZeta::ZETA, 0)
    }
}

// ─── Multiplication (matrix product) ─────────────────────────────────────────

/// U2 matrix multiplication
/// TODO: There should be a "reduction" method that I can call that reduces the 
/// value of k as much as possible while keeping coefficients as integers.
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

// ─── PyO3 ─────────────────────────────────────────────────────────────────────

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Tagged-enum wrapper over the two concrete `U2` instantiations.
///
/// Used by `PyU2` to expose a single Python class regardless of the underlying
/// ring (`ZOmega` for Clifford+T, `ZZeta` for Clifford+√T).
#[cfg(feature = "python")]
#[derive(Clone)]
pub enum U2Variant {
    Omega(U2T),
    Zeta(U2Q),
}

/// Python-facing U2 classes
#[cfg(feature = "python")]
#[pyclass(name = "U2")]
pub struct PyU2 {
    inner: U2Variant,
}

#[cfg(feature = "python")]
impl PyU2 {
    pub fn to_inner(&self) -> &U2Variant { &self.inner }
}

#[cfg(feature = "python")]
#[pymethods]
impl PyU2 {
    #[new]
    fn new(
        u11: Bound<'_, PyAny>,
        u12: Bound<'_, PyAny>,
        u21: Bound<'_, PyAny>,
        u22: Bound<'_, PyAny>,
        k: u32,
    ) -> PyResult<Self> {
        let type_err = |name: &str| PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            format!("{name} must be the same ring type as u11 (ZOmega or ZZeta)")
        );

        // Try ZOmega
        if let Ok(a) = u11.downcast::<PyZOmega>() {
            let b = u12.downcast::<PyZOmega>().map_err(|_| type_err("u12"))?;
            let c = u21.downcast::<PyZOmega>().map_err(|_| type_err("u21"))?;
            let d = u22.downcast::<PyZOmega>().map_err(|_| type_err("u22"))?;
            return Ok(Self {
                inner: U2Variant::Omega(U2T::new(
                    a.get().to_inner(), b.get().to_inner(),
                    c.get().to_inner(), d.get().to_inner(),
                    k,
                )),
            });
        }

        // Try ZZeta
        if let Ok(a) = u11.downcast::<PyZZeta>() {
            let b = u12.downcast::<PyZZeta>().map_err(|_| type_err("u12"))?;
            let c = u21.downcast::<PyZZeta>().map_err(|_| type_err("u21"))?;
            let d = u22.downcast::<PyZZeta>().map_err(|_| type_err("u22"))?;
            return Ok(Self {
                inner: U2Variant::Zeta(U2Q::new(
                    a.get().to_inner(), b.get().to_inner(),
                    c.get().to_inner(), d.get().to_inner(),
                    k,
                )),
            });
        }

        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "u11 must be an instance of ZOmega or ZZeta"
        ))
    }

    fn dagger(&self) -> Self {
        let inner = match self.inner {
            U2Variant::Omega(u) => U2Variant::Omega(u.dagger()),
            U2Variant::Zeta(u) => U2Variant::Zeta(u.dagger()),
        };
        Self { inner }
    }

    fn u11(&self, py: Python<'_>) -> PyResult<PyObject> {
        match self.inner {
            U2Variant::Omega(u) => Ok(Bound::new(py, PyZOmega { inner: u.u11 })?.to_object(py)),
            U2Variant::Zeta(u)  => Ok(Bound::new(py, PyZZeta  { inner: u.u11 })?.to_object(py)),
        }
    }

    fn u12(&self, py: Python<'_>) -> PyResult<PyObject> {
        match self.inner {
            U2Variant::Omega(u) => Ok(Bound::new(py, PyZOmega { inner: u.u12 })?.to_object(py)),
            U2Variant::Zeta(u)  => Ok(Bound::new(py, PyZZeta  { inner: u.u12 })?.to_object(py)),
        }
    }

    fn u21(&self, py: Python<'_>) -> PyResult<PyObject> {
        match self.inner {
            U2Variant::Omega(u) => Ok(Bound::new(py, PyZOmega { inner: u.u21 })?.to_object(py)),
            U2Variant::Zeta(u)  => Ok(Bound::new(py, PyZZeta  { inner: u.u21 })?.to_object(py)),
        }
    }

    fn u22(&self, py: Python<'_>) -> PyResult<PyObject> {
        match self.inner {
            U2Variant::Omega(u) => Ok(Bound::new(py, PyZOmega { inner: u.u22 })?.to_object(py)),
            U2Variant::Zeta(u)  => Ok(Bound::new(py, PyZZeta  { inner: u.u22 })?.to_object(py)),
        }
    }

    fn to_float(&self) -> Vec<Vec<(f64, f64)>> {
        let mat = match self.inner {
            U2Variant::Omega(u) => u.to_float(),
            U2Variant::Zeta(u) => u.to_float(),
        };

        mat.iter()
            .map(|row| {
                row.iter()
                    .map(|c| (c.re, c.im))
                    .collect()
            })
            .collect()
    }

    fn __mul__(&self, other: &Self) -> PyResult<Self> {
        match (&self.inner, &other.inner) {
            (U2Variant::Omega(a), U2Variant::Omega(b)) => {
                Ok(Self { inner: U2Variant::Omega((*a) * (*b)) })
            }
            (U2Variant::Zeta(a), U2Variant::Zeta(b)) => {
                Ok(Self { inner: U2Variant::Zeta((*a) * (*b)) })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Cannot multiply U2 matrices of different ring types"
            )),
        }
    }
}

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
