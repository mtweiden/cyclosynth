//! Z[√2] — algebraic integers of the form a + b√2.

use pyo3::prelude::*;
use std::ops::{Add, Mul, Neg, Sub};

/// An algebraic integer in Z[√2], represented as a + b√2.
#[pyclass]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RingRoot2 {
    #[pyo3(get)]
    pub values: [i128; 2],
}

impl RingRoot2 {
    pub fn new(values: [i128; 2]) -> Self {
        Self { values }
    }

    pub fn zero() -> Self {
        Self { values: [0, 0] }
    }

    pub fn one() -> Self {
        Self { values: [1, 0] }
    }

    pub fn sqrt2() -> Self {
        Self { values: [0, 1] }
    }

    pub fn to_f64(&self) -> f64 {
        let [a, b] = self.values;
        a as f64 + b as f64 * std::f64::consts::SQRT_2
    }

    pub fn conj(&self) -> Self {
        let [a, b] = self.values;
        Self { values: [a, -b] }
    }

    pub fn greatest_divisor(&self) -> i128 {
        gcd(self.values[0].unsigned_abs(), self.values[1].unsigned_abs()) as i128
    }

    pub fn pow(&self, n: u32) -> Self {
        if n == 0 {
            return Self::one();
        }
        let mut result = self.clone();
        for _ in 1..n {
            result = &result * self;
        }
        result
    }
}

impl Add for &RingRoot2 {
    type Output = RingRoot2;
    fn add(self, rhs: &RingRoot2) -> RingRoot2 {
        RingRoot2 {
            values: [
                self.values[0] + rhs.values[0],
                self.values[1] + rhs.values[1],
            ],
        }
    }
}

impl Sub for &RingRoot2 {
    type Output = RingRoot2;
    fn sub(self, rhs: &RingRoot2) -> RingRoot2 {
        RingRoot2 {
            values: [
                self.values[0] - rhs.values[0],
                self.values[1] - rhs.values[1],
            ],
        }
    }
}

impl Mul for &RingRoot2 {
    type Output = RingRoot2;
    fn mul(self, rhs: &RingRoot2) -> RingRoot2 {
        let [a, b] = self.values;
        let [x, y] = rhs.values;
        RingRoot2 {
            values: [a * x + 2 * b * y, a * y + b * x],
        }
    }
}

impl Mul<i128> for &RingRoot2 {
    type Output = RingRoot2;
    fn mul(self, rhs: i128) -> RingRoot2 {
        RingRoot2 {
            values: [self.values[0] * rhs, self.values[1] * rhs],
        }
    }
}

impl Neg for &RingRoot2 {
    type Output = RingRoot2;
    fn neg(self) -> RingRoot2 {
        RingRoot2 {
            values: [-self.values[0], -self.values[1]],
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
impl RingRoot2 {
    #[new]
    fn py_new(values: [i128; 2]) -> Self {
        Self::new(values)
    }

    fn __add__(&self, other: &RingRoot2) -> Self {
        self + other
    }

    fn __sub__(&self, other: &RingRoot2) -> Self {
        self - other
    }

    fn __mul__(&self, other: &RingRoot2) -> Self {
        self * other
    }

    fn __neg__(&self) -> Self {
        -self
    }

    pub fn __repr__(&self) -> String {
        let [a, b] = self.values;
        if a == 0 && b == 0 {
            "0".to_string()
        } else if b == 0 {
            format!("{a}")
        } else if a == 0 {
            if b == 1 {
                "sqrt(2)".to_string()
            } else if b == -1 {
                "-sqrt(2)".to_string()
            } else {
                format!("{b}*sqrt(2)")
            }
        } else {
            let sign = if b > 0 { "+" } else { "-" };
            let abs_b = b.unsigned_abs();
            if abs_b == 1 {
                format!("{a} {sign} sqrt(2)")
            } else {
                format!("{a} {sign} {abs_b}*sqrt(2)")
            }
        }
    }

    #[getter]
    fn to_float(&self) -> f64 {
        self.to_f64()
    }

    fn py_conj(&self) -> Self {
        self.conj()
    }

    fn py_pow(&self, n: u32) -> Self {
        self.pow(n)
    }

    fn copy(&self) -> Self {
        self.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    fn rand_values() -> [i128; 2] {
        let mut rng = rand::rng();
        [
            rng.random_range(-1_000_000_000_000_000i128..=1_000_000_000_000_000),
            rng.random_range(-1_000_000_000_000_000i128..=1_000_000_000_000_000),
        ]
    }

    fn to_float(values: &[i128; 2]) -> f64 {
        values[0] as f64 + values[1] as f64 * std::f64::consts::SQRT_2
    }

    #[test]
    fn test_construction() {
        for _ in 0..1000 {
            let v = rand_values();
            let _ = RingRoot2::new(v);
        }
    }

    #[test]
    fn test_add() {
        for _ in 0..1000 {
            let va = rand_values();
            let vb = rand_values();
            let vc = [va[0] + vb[0], va[1] + vb[1]];
            let a = RingRoot2::new(va);
            let b = RingRoot2::new(vb);
            let c = &a + &b;
            let expected = to_float(&vc);
            let actual = c.to_f64();
            assert!(
                (expected - actual).abs() / expected.abs().max(1.0) < 1e-6,
                "add failed: {expected} != {actual}"
            );
        }
    }

    #[test]
    fn test_mul() {
        for _ in 0..1000 {
            let va = rand_values();
            let vb = rand_values();
            let a = RingRoot2::new(va);
            let b = RingRoot2::new(vb);
            let c = &a * &b;
            let expected = to_float(&va) * to_float(&vb);
            let actual = c.to_f64();
            assert!(
                (expected - actual).abs() / expected.abs().max(1.0) < 1e-3,
                "mul failed: {expected} != {actual}"
            );
        }
    }

    #[test]
    fn test_conj() {
        for _ in 0..1000 {
            let v = rand_values();
            let a = RingRoot2::new(v);
            let c = a.conj();
            assert_eq!(c.values, [v[0], -v[1]]);
        }
    }

    #[test]
    fn test_pow() {
        let r = RingRoot2::new([1, 1]); // 1 + √2
        let r2 = r.pow(2);
        // (1+√2)² = 1 + 2√2 + 2 = 3 + 2√2
        assert_eq!(r2.values, [3, 2]);
    }
}
