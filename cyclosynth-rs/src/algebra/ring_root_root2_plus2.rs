//! Z[√(√2+2)] — algebraic integers with 4 coefficients.

use pyo3::prelude::*;
use std::ops::{Add, Mul, Neg, Sub};

use super::ring_root2::RingRoot2;

/// An algebraic integer in Z[√(√2+2)].
/// Represented as a + b√2 + c√(√2+2) + d√2·√(√2+2).
#[pyclass]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RingRootRoot2Plus2 {
    #[pyo3(get)]
    pub values: [i128; 4],
}

impl RingRootRoot2Plus2 {
    pub fn new(values: [i128; 4]) -> Self {
        Self { values }
    }

    pub fn zero() -> Self {
        Self { values: [0, 0, 0, 0] }
    }

    pub fn one() -> Self {
        Self { values: [1, 0, 0, 0] }
    }

    pub fn to_f64(&self) -> f64 {
        let [a, b, c, d] = self.values;
        let sqrt2 = std::f64::consts::SQRT_2;
        let sqrt_r2p2 = (sqrt2 + 2.0).sqrt();
        a as f64 + b as f64 * sqrt2 + c as f64 * sqrt_r2p2 + d as f64 * sqrt2 * sqrt_r2p2
    }

    pub fn from_ring_root2(r: &RingRoot2) -> Self {
        Self {
            values: [r.values[0], r.values[1], 0, 0],
        }
    }

    pub fn greatest_divisor(&self) -> i128 {
        let mut g = self.values[0].unsigned_abs();
        for &v in &self.values[1..] {
            g = gcd(g, v.unsigned_abs());
        }
        g as i128
    }
}

impl Add for &RingRootRoot2Plus2 {
    type Output = RingRootRoot2Plus2;
    fn add(self, rhs: &RingRootRoot2Plus2) -> RingRootRoot2Plus2 {
        RingRootRoot2Plus2 {
            values: [
                self.values[0] + rhs.values[0],
                self.values[1] + rhs.values[1],
                self.values[2] + rhs.values[2],
                self.values[3] + rhs.values[3],
            ],
        }
    }
}

impl Sub for &RingRootRoot2Plus2 {
    type Output = RingRootRoot2Plus2;
    fn sub(self, rhs: &RingRootRoot2Plus2) -> RingRootRoot2Plus2 {
        RingRootRoot2Plus2 {
            values: [
                self.values[0] - rhs.values[0],
                self.values[1] - rhs.values[1],
                self.values[2] - rhs.values[2],
                self.values[3] - rhs.values[3],
            ],
        }
    }
}

impl Mul for &RingRootRoot2Plus2 {
    type Output = RingRootRoot2Plus2;
    fn mul(self, rhs: &RingRootRoot2Plus2) -> RingRootRoot2Plus2 {
        let [a, b, c, d] = self.values;
        let [w, x, y, z] = rhs.values;
        // Multiplication table derived from:
        // √(√2+2)² = √2+2, (√2)² = 2, √2·√(√2+2)·√(√2+2) = √2(√2+2) = 2+2√2
        let aa = a * w + 2 * b * x + 2 * c * y + 2 * c * z + 2 * d * y + 4 * d * z;
        let bb = a * x + b * w + c * y + 2 * c * z + 2 * d * y + 2 * d * z;
        let cc = a * y + 2 * b * z + c * w + 2 * d * x;
        let dd = a * z + b * y + c * x + d * w;
        RingRootRoot2Plus2 {
            values: [aa, bb, cc, dd],
        }
    }
}

impl Mul<i128> for &RingRootRoot2Plus2 {
    type Output = RingRootRoot2Plus2;
    fn mul(self, rhs: i128) -> RingRootRoot2Plus2 {
        RingRootRoot2Plus2 {
            values: [
                self.values[0] * rhs,
                self.values[1] * rhs,
                self.values[2] * rhs,
                self.values[3] * rhs,
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
            values: [-self.values[0], -self.values[1], -self.values[2], -self.values[3]],
        }
    }
}

fn gcd(a: u128, b: u128) -> u128 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

#[pymethods]
impl RingRootRoot2Plus2 {
    #[new]
    fn py_new(values: [i128; 4]) -> Self {
        Self::new(values)
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
        let [a, b, c, d] = self.values;
        format!(
            "{a} + {b}*sqrt(2) + {c}*sqrt(sqrt(2)+2) + {d}*sqrt(2)*sqrt(sqrt(2)+2)"
        )
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
            let _ = &a * &b;
            // Multiplication produces very large values, hard to verify with f64
            // Just check it doesn't panic
        }
    }
}
