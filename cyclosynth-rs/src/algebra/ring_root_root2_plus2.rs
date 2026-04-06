//! Z[√(√2+2)] — algebraic integers with 4 coefficients.
//!
//! Uses `num_bigint::BigInt` for the coefficient values to avoid i128 overflow
//! when composing many gates (e.g. 100 random Q-gates).

use num_bigint::BigInt;
use num_integer::Integer;
use num_traits::{One, ToPrimitive, Zero};
use pyo3::prelude::*;
use std::ops::{Add, Mul, Neg, Sub};

use super::ring_root2::RingRoot2;

/// An algebraic integer in Z[√(√2+2)].
/// Represented as a + b√2 + c√(√2+2) + d√2·√(√2+2).
#[pyclass]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RingRootRoot2Plus2 {
    pub values: [BigInt; 4],
}

impl RingRootRoot2Plus2 {
    pub fn new_big(values: [BigInt; 4]) -> Self {
        Self { values }
    }

    pub fn new(values: [i128; 4]) -> Self {
        Self {
            values: values.map(BigInt::from),
        }
    }

    pub fn zero() -> Self {
        Self {
            values: [BigInt::zero(), BigInt::zero(), BigInt::zero(), BigInt::zero()],
        }
    }

    pub fn one() -> Self {
        Self {
            values: [BigInt::one(), BigInt::zero(), BigInt::zero(), BigInt::zero()],
        }
    }

    pub fn to_f64(&self) -> f64 {
        let [a, b, c, d] = &self.values;
        let sqrt2 = std::f64::consts::SQRT_2;
        let sqrt_r2p2 = (sqrt2 + 2.0).sqrt();
        a.to_f64().unwrap_or(f64::NAN)
            + b.to_f64().unwrap_or(f64::NAN) * sqrt2
            + c.to_f64().unwrap_or(f64::NAN) * sqrt_r2p2
            + d.to_f64().unwrap_or(f64::NAN) * sqrt2 * sqrt_r2p2
    }

    pub fn from_ring_root2(r: &RingRoot2) -> Self {
        Self {
            values: [
                BigInt::from(r.values[0]),
                BigInt::from(r.values[1]),
                BigInt::zero(),
                BigInt::zero(),
            ],
        }
    }

    pub fn greatest_divisor(&self) -> BigInt {
        let mut g = BigInt::zero();
        for v in &self.values {
            g = g.gcd(v);
        }
        g
    }

    /// Return the values as i128 for PyO3 (lossy if BigInt is very large).
    pub fn to_i128_array(&self) -> [i128; 4] {
        self.values.each_ref().map(|v| v.to_i128().unwrap_or(i128::MAX))
    }
}

impl Add for &RingRootRoot2Plus2 {
    type Output = RingRootRoot2Plus2;
    fn add(self, rhs: &RingRootRoot2Plus2) -> RingRootRoot2Plus2 {
        RingRootRoot2Plus2 {
            values: [
                &self.values[0] + &rhs.values[0],
                &self.values[1] + &rhs.values[1],
                &self.values[2] + &rhs.values[2],
                &self.values[3] + &rhs.values[3],
            ],
        }
    }
}

impl Sub for &RingRootRoot2Plus2 {
    type Output = RingRootRoot2Plus2;
    fn sub(self, rhs: &RingRootRoot2Plus2) -> RingRootRoot2Plus2 {
        RingRootRoot2Plus2 {
            values: [
                &self.values[0] - &rhs.values[0],
                &self.values[1] - &rhs.values[1],
                &self.values[2] - &rhs.values[2],
                &self.values[3] - &rhs.values[3],
            ],
        }
    }
}

impl Mul for &RingRootRoot2Plus2 {
    type Output = RingRootRoot2Plus2;
    fn mul(self, rhs: &RingRootRoot2Plus2) -> RingRootRoot2Plus2 {
        let [a, b, c, d] = &self.values;
        let [w, x, y, z] = &rhs.values;
        // Multiplication table derived from:
        // √(√2+2)² = √2+2, (√2)² = 2, √2·√(√2+2)·√(√2+2) = √2(√2+2) = 2+2√2
        let two = BigInt::from(2i32);
        let four = BigInt::from(4i32);
        let aa = a * w + &two * b * x + &two * c * y + &two * c * z + &two * d * y + &four * d * z;
        let bb = a * x + b * w + c * y + &two * c * z + &two * d * y + &two * d * z;
        let cc = a * y + &two * b * z + c * w + &two * d * x;
        let dd = a * z + b * y + c * x + d * w;
        RingRootRoot2Plus2 {
            values: [aa, bb, cc, dd],
        }
    }
}

impl Mul<i128> for &RingRootRoot2Plus2 {
    type Output = RingRootRoot2Plus2;
    fn mul(self, rhs: i128) -> RingRootRoot2Plus2 {
        let r = BigInt::from(rhs);
        RingRootRoot2Plus2 {
            values: [
                &self.values[0] * &r,
                &self.values[1] * &r,
                &self.values[2] * &r,
                &self.values[3] * &r,
            ],
        }
    }
}

/// Multiply RingRootRoot2Plus2 by a RingRoot2 (embedding Root2 into RootRoot2Plus2).
impl Mul<&RingRoot2> for &RingRootRoot2Plus2 {
    type Output = RingRootRoot2Plus2;
    fn mul(self, rhs: &RingRoot2) -> RingRootRoot2Plus2 {
        let embedded = RingRootRoot2Plus2::from_ring_root2(rhs);
        self * &embedded
    }
}

impl Neg for &RingRootRoot2Plus2 {
    type Output = RingRootRoot2Plus2;
    fn neg(self) -> RingRootRoot2Plus2 {
        RingRootRoot2Plus2 {
            values: [
                -&self.values[0],
                -&self.values[1],
                -&self.values[2],
                -&self.values[3],
            ],
        }
    }
}

#[pymethods]
impl RingRootRoot2Plus2 {
    #[new]
    fn py_new(values: [i64; 4]) -> Self {
        Self::new(values.map(|v| v as i128))
    }

    fn __add__(&self, other: &RingRootRoot2Plus2) -> Self {
        self + other
    }

    fn __sub__(&self, other: &RingRootRoot2Plus2) -> Self {
        self - other
    }

    fn __mul__(&self, other: &RingRootRoot2Plus2) -> Self {
        self * other
    }

    fn __neg__(&self) -> Self {
        -self
    }

    fn __repr__(&self) -> String {
        let [a, b, c, d] = self.to_i128_array();
        format!(
            "{a} + {b}*sqrt(2) + {c}*sqrt(sqrt(2)+2) + {d}*sqrt(2)*sqrt(sqrt(2)+2)"
        )
    }

    #[getter]
    fn values(&self) -> [i128; 4] {
        self.to_i128_array()
    }

    #[getter]
    fn to_float(&self) -> f64 {
        self.to_f64()
    }

    fn copy(&self) -> Self {
        self.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    fn rand_values() -> [i128; 4] {
        let mut rng = rand::rng();
        [
            rng.random_range(-1_000_000_000_000_000i128..=1_000_000_000_000_000),
            rng.random_range(-1_000_000_000_000_000i128..=1_000_000_000_000_000),
            rng.random_range(-1_000_000_000_000_000i128..=1_000_000_000_000_000),
            rng.random_range(-1_000_000_000_000_000i128..=1_000_000_000_000_000),
        ]
    }

    fn to_float(values: &[i128; 4]) -> f64 {
        let sqrt2 = std::f64::consts::SQRT_2;
        let sqrt_r2p2 = (sqrt2 + 2.0).sqrt();
        values[0] as f64
            + values[1] as f64 * sqrt2
            + values[2] as f64 * sqrt_r2p2
            + values[3] as f64 * sqrt2 * sqrt_r2p2
    }

    #[test]
    fn test_construction() {
        for _ in 0..1000 {
            let _ = RingRootRoot2Plus2::new(rand_values());
        }
    }

    #[test]
    fn test_add() {
        for _ in 0..1000 {
            let va = rand_values();
            let vb = rand_values();
            let vc = [va[0] + vb[0], va[1] + vb[1], va[2] + vb[2], va[3] + vb[3]];
            let a = RingRootRoot2Plus2::new(va);
            let b = RingRootRoot2Plus2::new(vb);
            let c = &a + &b;
            let expected = to_float(&vc);
            let actual = c.to_f64();
            assert!(
                (expected - actual).abs() / expected.abs().max(1.0) < 1e-6,
                "add failed"
            );
        }
    }

    #[test]
    fn test_mul() {
        for _ in 0..1000 {
            let va = rand_values();
            let vb = rand_values();
            let a = RingRootRoot2Plus2::new(va);
            let b = RingRootRoot2Plus2::new(vb);
            let c = &a * &b;
            let expected = a.to_f64() * b.to_f64();
            let actual = c.to_f64();
            assert!(
                (expected - actual).abs() / expected.abs().max(1.0) < 1e-5,
                "mul: {expected} != {actual}"
            );
        }
    }
}
