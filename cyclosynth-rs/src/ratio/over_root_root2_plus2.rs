//! AlgebraicInteger / (√(√2+2))^power — ratios over sqrt(sqrt(2)+2).

use num_bigint::BigInt;
use num_traits::Zero;
use pyo3::prelude::*;

use crate::algebra::{DyadicComplexNumber, RingRootRoot2Plus2};

/// An algebraic integer divided by a power of √(√2+2).
#[pyclass]
#[derive(Clone, Debug)]
pub struct AlgebraicIntegerOverRootRoot2Plus2 {
    #[pyo3(get)]
    pub numerator: RingRootRoot2Plus2,
    #[pyo3(get)]
    pub denominator_power: u32,
}

impl AlgebraicIntegerOverRootRoot2Plus2 {
    pub fn new(numerator: RingRootRoot2Plus2, power: u32) -> Self {
        Self {
            numerator,
            denominator_power: power,
        }
    }

    pub fn zero() -> Self {
        Self::new(RingRootRoot2Plus2::zero(), 0)
    }

    pub fn to_f64(&self) -> f64 {
        let sqrt2 = std::f64::consts::SQRT_2;
        let sqrt_r2p2 = (sqrt2 + 2.0).sqrt();
        self.numerator.to_f64() / sqrt_r2p2.powi(self.denominator_power as i32)
    }

    pub fn simplify(&mut self) {
        if self.numerator.values.iter().all(|v| v.is_zero()) {
            self.denominator_power = 0;
            return;
        }
        // gamma = (2 - sqrt(2)) * sqrt(2 + sqrt(2)) in ring form: [0, 0, 2, -1]
        let gamma = RingRootRoot2Plus2::new([0, 0, 2, -1]);
        let two = BigInt::from(2i32);
        let mut result = self.numerator.clone();
        for _ in 0..self.denominator_power {
            let new_result = &result * &gamma;
            if new_result.values.iter().all(|v| v % &two == BigInt::zero()) {
                result = RingRootRoot2Plus2::new_big(
                    new_result.values.map(|v| v / &two),
                );
                self.denominator_power -= 1;
            } else {
                break;
            }
        }
        self.numerator = result;
    }

    pub fn add(
        &self,
        other: &AlgebraicIntegerOverRootRoot2Plus2,
    ) -> AlgebraicIntegerOverRootRoot2Plus2 {
        let rr2p2 = RingRootRoot2Plus2::new([0, 0, 1, 0]);
        let mut self_num = self.numerator.clone();
        let mut other_num = other.numerator.clone();
        let mut self_pow = self.denominator_power;
        let other_pow = other.denominator_power;

        if self_pow < other_pow {
            for _ in 0..(other_pow - self_pow) {
                self_num = &self_num * &rr2p2;
            }
            self_pow = other_pow;
        } else if other_pow < self_pow {
            for _ in 0..(self_pow - other_pow) {
                other_num = &other_num * &rr2p2;
            }
        }
        let new_num = &self_num + &other_num;
        let mut result = AlgebraicIntegerOverRootRoot2Plus2::new(new_num, self_pow);
        result.simplify();
        result
    }

    pub fn mul(
        &self,
        other: &AlgebraicIntegerOverRootRoot2Plus2,
    ) -> AlgebraicIntegerOverRootRoot2Plus2 {
        let new_num = &self.numerator * &other.numerator;
        let new_pow = self.denominator_power + other.denominator_power;
        let mut result = AlgebraicIntegerOverRootRoot2Plus2::new(new_num, new_pow);
        result.simplify();
        result
    }

    pub fn neg(&self) -> AlgebraicIntegerOverRootRoot2Plus2 {
        AlgebraicIntegerOverRootRoot2Plus2::new(-&self.numerator, self.denominator_power)
    }

    pub fn from_dyadic(
        dyadic: &DyadicComplexNumber,
    ) -> AlgebraicIntegerOverRootRoot2Plus2 {
        assert!(dyadic.values.len() == 16);
        let mut d = dyadic.clone();
        d.simplify();
        let k = d.denominator_exponent;
        // Extract coefficients as BigInt to avoid overflow for large k.
        let c0 = BigInt::from(2 * d.values[2]);
        let c1 = BigInt::from(d.values[2] + d.values[6]);
        let c2 = BigInt::from(d.values[0]);
        let c3 = BigInt::from(d.values[4]);
        let dyadic_int = RingRootRoot2Plus2::new_big([c0, c1, c2, c3]);
        let mut result = AlgebraicIntegerOverRootRoot2Plus2::new(dyadic_int, 1);
        // Apply (1+√2)/r^2 exactly 2k times instead of computing (1+√2)^{2k} all at once.
        // Mathematically: (1+√2)^{2k} / r^{4k} = 1/2^k, so each step keeps values bounded.
        let factor = AlgebraicIntegerOverRootRoot2Plus2::new(
            RingRootRoot2Plus2::new([1, 1, 0, 0]),
            2,
        );
        for _ in 0..(2 * k) {
            result = factor.mul(&result);
        }
        result
    }
}

#[pymethods]
impl AlgebraicIntegerOverRootRoot2Plus2 {
    #[new]
    #[pyo3(signature = (values, power=0))]
    fn py_new(values: [i64; 4], power: u32) -> Self {
        Self::new(RingRootRoot2Plus2::new(values.map(|v| v as i128)), power)
    }

    fn __add__(&self, other: &AlgebraicIntegerOverRootRoot2Plus2) -> Self {
        self.add(other)
    }

    fn __mul__(&self, other: &AlgebraicIntegerOverRootRoot2Plus2) -> Self {
        self.mul(other)
    }

    fn __neg__(&self) -> Self {
        self.neg()
    }

    fn __repr__(&self) -> String {
        let n = format!("{:?}", self.numerator.to_i128_array());
        let rr2p2 = "sqrt(2 + sqrt(2))";
        if self.denominator_power == 0 {
            n
        } else if self.denominator_power == 1 {
            format!("{n} / {rr2p2}")
        } else {
            format!("{n} / {rr2p2}^{}", self.denominator_power)
        }
    }

    fn py_to_float(&self) -> f64 {
        self.to_f64()
    }

    fn py_simplify(&mut self) {
        self.simplify();
    }

    fn copy(&self) -> Self {
        self.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    fn rand_rr2p2() -> RingRootRoot2Plus2 {
        let mut rng = rand::rng();
        RingRootRoot2Plus2::new([
            rng.random_range(-1_000_000_000_000i128..=1_000_000_000_000),
            rng.random_range(-1_000_000_000_000i128..=1_000_000_000_000),
            rng.random_range(-1_000_000_000_000i128..=1_000_000_000_000),
            rng.random_range(-1_000_000_000_000i128..=1_000_000_000_000),
        ])
    }

    fn rand_power() -> u32 {
        let mut rng = rand::rng();
        rng.random_range(0..=32)
    }

    #[test]
    fn test_simplify() {
        for _ in 0..1000 {
            let mut rng = rand::rng();
            let power: u32 = rng.random_range(0..=100);
            let mut x = RingRootRoot2Plus2::one();
            let rr2p2 = RingRootRoot2Plus2::new([0, 0, 1, 0]);
            for _ in 0..power {
                x = &x * &rr2p2;
            }
            let mut ratio = AlgebraicIntegerOverRootRoot2Plus2::new(x, power);
            ratio.simplify();
            let expected = RingRootRoot2Plus2::one();
            assert_eq!(ratio.numerator, expected, "simplify should reduce to 1");
            assert_eq!(ratio.denominator_power, 0);
        }
    }

    #[test]
    fn test_add() {
        for _ in 0..1000 {
            let x = rand_rr2p2();
            let y = rand_rr2p2();
            let xp = rand_power();
            let yp = rand_power();
            let xr = AlgebraicIntegerOverRootRoot2Plus2::new(x.clone(), xp);
            let yr = AlgebraicIntegerOverRootRoot2Plus2::new(y.clone(), yp);
            let sqrt2 = std::f64::consts::SQRT_2;
            let sqrt_r2p2 = (sqrt2 + 2.0).sqrt();
            let xf = x.to_f64() / sqrt_r2p2.powi(xp as i32);
            let yf = y.to_f64() / sqrt_r2p2.powi(yp as i32);
            let expected = xf + yf;
            let actual = xr.add(&yr).to_f64();
            assert!(
                (expected - actual).abs() / expected.abs().max(1.0) < 1e-6,
                "add: {expected} != {actual}"
            );
        }
    }

    #[test]
    fn test_mul() {
        for _ in 0..1000 {
            let x = rand_rr2p2();
            let y = rand_rr2p2();
            let xp = rand_power();
            let yp = rand_power();
            let xr = AlgebraicIntegerOverRootRoot2Plus2::new(x.clone(), xp);
            let yr = AlgebraicIntegerOverRootRoot2Plus2::new(y.clone(), yp);
            let sqrt2 = std::f64::consts::SQRT_2;
            let sqrt_r2p2 = (sqrt2 + 2.0).sqrt();
            let xf = x.to_f64() / sqrt_r2p2.powi(xp as i32);
            let yf = y.to_f64() / sqrt_r2p2.powi(yp as i32);
            let expected = xf * yf;
            let actual = xr.mul(&yr).to_f64();
            assert!(
                (expected - actual).abs() / expected.abs().max(1.0) < 1e-6,
                "mul: {expected} != {actual}"
            );
        }
    }
}
