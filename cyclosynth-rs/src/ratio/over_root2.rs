//! AlgebraicInteger / (√2)^power — ratios over sqrt(2).

use pyo3::prelude::*;

use crate::algebra::{DyadicComplexNumber, RingRoot2, RingRootRoot2Plus2};
use super::over_root_root2_plus2::AlgebraicIntegerOverRootRoot2Plus2;

/// An algebraic integer divided by a power of √2.
#[pyclass]
#[derive(Clone, Debug)]
pub struct AlgebraicIntegerOverRoot2 {
    #[pyo3(get)]
    pub numerator: RingRoot2,
    #[pyo3(get)]
    pub denominator_power: u32,
}

impl AlgebraicIntegerOverRoot2 {
    pub fn new(numerator: RingRoot2, power: u32) -> Self {
        Self {
            numerator,
            denominator_power: power,
        }
    }

    pub fn zero() -> Self {
        Self::new(RingRoot2::zero(), 0)
    }

    pub fn to_f64(&self) -> f64 {
        self.numerator.to_f64() / std::f64::consts::SQRT_2.powi(self.denominator_power as i32)
    }

    pub fn simplify(&mut self) {
        if self.numerator.values.iter().all(|&v| v == 0) {
            self.denominator_power = 0;
            return;
        }
        let gamma = RingRoot2::new([0, 1]); // √2
        let mut result = self.numerator.clone();
        for _ in 0..self.denominator_power {
            let new_result = &gamma * &result;
            if new_result.values.iter().all(|&v| v % 2 == 0) {
                result = RingRoot2::new([new_result.values[0] / 2, new_result.values[1] / 2]);
                self.denominator_power -= 1;
            } else {
                break;
            }
        }
        self.numerator = result;
    }

    pub fn add(&self, other: &AlgebraicIntegerOverRoot2) -> AlgebraicIntegerOverRoot2 {
        let r2 = RingRoot2::new([0, 1]);
        let mut self_num = self.numerator.clone();
        let mut other_num = other.numerator.clone();
        let mut self_pow = self.denominator_power;
        let other_pow = other.denominator_power;

        if self_pow < other_pow {
            for _ in 0..(other_pow - self_pow) {
                self_num = &self_num * &r2;
            }
            self_pow = other_pow;
        } else if other_pow < self_pow {
            for _ in 0..(self_pow - other_pow) {
                other_num = &other_num * &r2;
            }
        }
        let new_num = &self_num + &other_num;
        let mut result = AlgebraicIntegerOverRoot2::new(new_num, self_pow);
        result.simplify();
        result
    }

    pub fn mul(&self, other: &AlgebraicIntegerOverRoot2) -> AlgebraicIntegerOverRoot2 {
        let new_num = &self.numerator * &other.numerator;
        let new_pow = self.denominator_power + other.denominator_power;
        let mut result = AlgebraicIntegerOverRoot2::new(new_num, new_pow);
        result.simplify();
        result
    }

    pub fn neg(&self) -> AlgebraicIntegerOverRoot2 {
        AlgebraicIntegerOverRoot2::new(-&self.numerator, self.denominator_power)
    }

    pub fn conj(&self) -> AlgebraicIntegerOverRoot2 {
        let mut result =
            AlgebraicIntegerOverRoot2::new(self.numerator.conj(), self.denominator_power);
        if self.denominator_power % 2 == 1 {
            result = result.neg();
        }
        result
    }

    pub fn to_rr2p2(&self) -> AlgebraicIntegerOverRootRoot2Plus2 {
        if self.denominator_power == 0 {
            let rr = RingRootRoot2Plus2::from_ring_root2(&self.numerator);
            return AlgebraicIntegerOverRootRoot2Plus2::new(rr, 0);
        }
        let conversion_num = RingRootRoot2Plus2::new([1, 1, 0, 0]);
        let mut factor = conversion_num.clone();
        for _ in 1..self.denominator_power {
            factor = &factor * &conversion_num;
        }
        let num_rr = RingRootRoot2Plus2::from_ring_root2(&self.numerator);
        let new_num = &factor * &num_rr;
        let new_pow = self.denominator_power * 2;
        let mut ratio = AlgebraicIntegerOverRootRoot2Plus2::new(new_num, new_pow);
        ratio.simplify();
        ratio
    }

    pub fn from_dyadic(dyadic: &DyadicComplexNumber) -> AlgebraicIntegerOverRoot2 {
        let mut d = dyadic.clone();
        if d.values.len() < 8 {
            let ref_d = DyadicComplexNumber::new(vec![0; 8], 0);
            d.match_base_size(&ref_d);
        }
        assert!(d.values.len() == 8);
        d.simplify();
        let k = d.denominator_exponent;
        let c0 = d.values[0];
        let c1 = d.values[2];
        let num = RingRoot2::new([c0, c1]);
        AlgebraicIntegerOverRoot2::new(num, (2 * k) as u32)
    }
}

#[pymethods]
impl AlgebraicIntegerOverRoot2 {
    #[new]
    #[pyo3(signature = (values, power=0))]
    fn py_new(values: [i128; 2], power: u32) -> Self {
        Self::new(RingRoot2::new(values), power)
    }

    fn __add__(&self, other: &AlgebraicIntegerOverRoot2) -> Self {
        self.add(other)
    }

    fn __mul__(&self, other: &AlgebraicIntegerOverRoot2) -> Self {
        self.mul(other)
    }

    fn __neg__(&self) -> Self {
        self.neg()
    }

    fn __repr__(&self) -> String {
        let n = format!("{}", self.numerator.__repr__());
        if self.denominator_power == 0 {
            n
        } else if self.denominator_power == 1 {
            format!("{n} / sqrt(2)")
        } else {
            format!("{n} / sqrt(2)^{}", self.denominator_power)
        }
    }

    fn py_to_float(&self) -> f64 {
        self.to_f64()
    }

    fn py_simplify(&mut self) {
        self.simplify();
    }

    fn py_conj(&self) -> Self {
        self.conj()
    }

    fn copy(&self) -> Self {
        self.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    fn rand_r2() -> RingRoot2 {
        let mut rng = rand::rng();
        RingRoot2::new([
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
            let mut x = RingRoot2::new([1, 0]);
            let r2 = RingRoot2::new([0, 1]);
            for _ in 0..power {
                x = &x * &r2;
            }
            let mut ratio = AlgebraicIntegerOverRoot2::new(x, power);
            ratio.simplify();
            assert_eq!(ratio.numerator.values, [1, 0]);
            assert_eq!(ratio.denominator_power, 0);
        }
    }

    #[test]
    fn test_add() {
        for _ in 0..1000 {
            let x = rand_r2();
            let y = rand_r2();
            let xp = rand_power();
            let yp = rand_power();
            let xr = AlgebraicIntegerOverRoot2::new(x.clone(), xp);
            let yr = AlgebraicIntegerOverRoot2::new(y.clone(), yp);
            let r2 = std::f64::consts::SQRT_2;
            let xf = x.to_f64() / r2.powi(xp as i32);
            let yf = y.to_f64() / r2.powi(yp as i32);
            let expected = xf + yf;
            let actual = xr.add(&yr).to_f64();
            assert!(
                (expected - actual).abs() / expected.abs().max(1.0) < 1e-6,
                "add failed: {expected} != {actual}"
            );
        }
    }

    #[test]
    fn test_mul() {
        for _ in 0..1000 {
            let x = rand_r2();
            let y = rand_r2();
            let xp = rand_power();
            let yp = rand_power();
            let xr = AlgebraicIntegerOverRoot2::new(x.clone(), xp);
            let yr = AlgebraicIntegerOverRoot2::new(y.clone(), yp);
            let r2 = std::f64::consts::SQRT_2;
            let xf = x.to_f64() / r2.powi(xp as i32);
            let yf = y.to_f64() / r2.powi(yp as i32);
            let expected = xf * yf;
            let actual = xr.mul(&yr).to_f64();
            assert!(
                (expected - actual).abs() / expected.abs().max(1.0) < 1e-6,
                "mul failed: {expected} != {actual}"
            );
        }
    }
}
