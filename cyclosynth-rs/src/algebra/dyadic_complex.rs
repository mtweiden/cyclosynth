//! Z[e^{iπ/n}, 1/2] — dyadic complex numbers with variable base size.

use num_complex::Complex64;
use pyo3::prelude::*;
use std::f64::consts::PI;
use std::ops::{Add, Mul, Neg, Sub};

/// A number in Z[e^{iπ/n}, 1/2].
///
/// Represented as (Σ_k a_k · e^{ikπ/m}) / 2^l where m = len(values).
#[pyclass]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DyadicComplexNumber {
    #[pyo3(get)]
    pub values: Vec<i128>,
    #[pyo3(get)]
    pub denominator_exponent: i32,
}

impl DyadicComplexNumber {
    pub fn new(values: Vec<i128>, denominator_exponent: i32) -> Self {
        Self {
            values,
            denominator_exponent,
        }
    }

    pub fn zero(n: usize) -> Self {
        Self {
            values: vec![0; n],
            denominator_exponent: 0,
        }
    }

    pub fn match_base_size(&mut self, other: &DyadicComplexNumber) {
        let self_base = self.values.len();
        let other_base = other.values.len();
        if other_base <= self_base {
            return;
        }
        let gap = other_base / self_base;
        assert!(
            gap * self_base == other_base && gap.is_power_of_two(),
            "New base must be a power of 2 of the old base."
        );
        let mut new_values = vec![0i128; other_base];
        for (i, &v) in self.values.iter().enumerate() {
            new_values[i * gap] = v;
        }
        self.values = new_values;
    }

    pub fn simplify(&mut self) {
        for _ in 0..self.denominator_exponent {
            if self.values.iter().all(|&v| v % 2 == 0) {
                self.values.iter_mut().for_each(|v| *v >>= 1);
                self.denominator_exponent -= 1;
            } else {
                break;
            }
        }
    }

    pub fn conj(&self) -> Self {
        let mut new_values = self.values.clone();
        let n = new_values.len();
        if n > 1 {
            new_values[1..].reverse();
            for v in &mut new_values[1..] {
                *v = -*v;
            }
        }
        Self {
            values: new_values,
            denominator_exponent: self.denominator_exponent,
        }
    }

    pub fn to_complex(&self) -> Complex64 {
        let m = self.values.len();
        let mut total = Complex64::new(0.0, 0.0);
        for (i, &coeff) in self.values.iter().enumerate() {
            let phase = Complex64::from_polar(1.0, PI * i as f64 / m as f64);
            total += phase * coeff as f64;
        }
        total / 2.0f64.powi(self.denominator_exponent)
    }

    pub fn abs(&self) -> f64 {
        self.to_complex().norm()
    }
}

fn ensure_same_base(a: &mut DyadicComplexNumber, b: &mut DyadicComplexNumber) {
    if a.values.len() < b.values.len() {
        a.match_base_size(b);
    } else if a.values.len() > b.values.len() {
        b.match_base_size(a);
    }
}

impl Add for &DyadicComplexNumber {
    type Output = DyadicComplexNumber;
    fn add(self, rhs: &DyadicComplexNumber) -> DyadicComplexNumber {
        let mut lhs = self.clone();
        let mut rhs = rhs.clone();
        ensure_same_base(&mut lhs, &mut rhs);
        let offset = (lhs.denominator_exponent - rhs.denominator_exponent).unsigned_abs();
        let mut lhs_vals = lhs.values;
        let mut rhs_vals = rhs.values;
        if lhs.denominator_exponent < rhs.denominator_exponent {
            lhs_vals.iter_mut().for_each(|v| *v <<= offset);
        } else if lhs.denominator_exponent > rhs.denominator_exponent {
            rhs_vals.iter_mut().for_each(|v| *v <<= offset);
        }
        let new_power = lhs.denominator_exponent.max(rhs.denominator_exponent);
        let new_values: Vec<i128> = lhs_vals
            .iter()
            .zip(rhs_vals.iter())
            .map(|(&a, &b)| a + b)
            .collect();
        let mut result = DyadicComplexNumber::new(new_values, new_power);
        result.simplify();
        result
    }
}

impl Neg for &DyadicComplexNumber {
    type Output = DyadicComplexNumber;
    fn neg(self) -> DyadicComplexNumber {
        DyadicComplexNumber {
            values: self.values.iter().map(|&v| -v).collect(),
            denominator_exponent: self.denominator_exponent,
        }
    }
}

impl Sub for &DyadicComplexNumber {
    type Output = DyadicComplexNumber;
    fn sub(self, rhs: &DyadicComplexNumber) -> DyadicComplexNumber {
        self + &(-rhs)
    }
}

impl Mul for &DyadicComplexNumber {
    type Output = DyadicComplexNumber;
    fn mul(self, rhs: &DyadicComplexNumber) -> DyadicComplexNumber {
        let mut lhs = self.clone();
        let mut rhs = rhs.clone();
        ensure_same_base(&mut lhs, &mut rhs);
        let new_power = lhs.denominator_exponent + rhs.denominator_exponent;
        let m = lhs.values.len();
        let mut new_values = vec![0i128; m];
        for (i, &ca) in lhs.values.iter().enumerate() {
            for (j, &cb) in rhs.values.iter().enumerate() {
                let k = i + j;
                let coeff = ca * cb;
                let sign = if k >= m { -1i128 } else { 1i128 };
                new_values[k % m] += sign * coeff;
            }
        }
        let mut result = DyadicComplexNumber::new(new_values, new_power);
        result.simplify();
        result
    }
}

#[pymethods]
impl DyadicComplexNumber {
    #[new]
    fn py_new(values: Vec<i128>, denominator_exponent: i32) -> Self {
        Self::new(values, denominator_exponent)
    }

    fn __add__(&self, other: &DyadicComplexNumber) -> Self {
        self + other
    }

    fn __sub__(&self, other: &DyadicComplexNumber) -> Self {
        self - other
    }

    fn __mul__(&self, other: &DyadicComplexNumber) -> Self {
        self * other
    }

    fn __neg__(&self) -> Self {
        -self
    }

    fn __repr__(&self) -> String {
        format_omega_repr(&self.values, self.denominator_exponent, true)
    }

    fn py_conj(&self) -> Self {
        self.conj()
    }

    fn py_to_complex(&self) -> (f64, f64) {
        let c = self.to_complex();
        (c.re, c.im)
    }

    fn py_abs(&self) -> f64 {
        self.abs()
    }

    fn py_simplify(&mut self) {
        self.simplify();
    }

    fn copy(&self) -> Self {
        self.clone()
    }
}

/// Shared formatting for omega-based representations.
pub fn format_omega_repr(values: &[i128], denom_exp: i32, is_dyadic: bool) -> String {
    let mut terms = String::new();
    for (i, &coeff) in values.iter().enumerate() {
        if coeff == 0 {
            continue;
        }
        let phase = match i {
            0 => String::new(),
            1 => "ω".to_string(),
            n => format!("ω^{n}"),
        };
        if terms.is_empty() {
            match (coeff, phase.as_str()) {
                (1, "") => terms.push('1'),
                (-1, "") => terms.push_str("-1"),
                (1, p) => terms.push_str(p),
                (-1, p) => {
                    terms.push('-');
                    terms.push_str(p);
                }
                (c, p) => {
                    terms.push_str(&format!("{c}{p}"));
                }
            }
        } else {
            let link = if coeff > 0 { "+" } else { "-" };
            let abs_c = coeff.unsigned_abs();
            if abs_c == 1 {
                terms.push_str(&format!(" {link} {phase}"));
            } else {
                terms.push_str(&format!(" {link} {abs_c}{phase}"));
            }
        }
    }
    let numerator = if terms.is_empty() {
        "0".to_string()
    } else {
        terms
    };
    if denom_exp == 0 {
        numerator
    } else if is_dyadic {
        let de = 2 * denom_exp;
        if de == 1 {
            format!("({numerator}) / √2")
        } else {
            format!("({numerator}) / √2^{de}")
        }
    } else if denom_exp == 1 {
        format!("({numerator}) / √2")
    } else {
        format!("({numerator}) / √2^{denom_exp}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    fn rand_values(n: usize) -> Vec<i128> {
        let mut rng = rand::rng();
        (0..n)
            .map(|_| rng.random_range(-1_000_000_000_000_000i128..=1_000_000_000_000_000))
            .collect()
    }

    fn rand_power() -> i32 {
        let mut rng = rand::rng();
        rng.random_range(0..=50)
    }

    fn reference_complex(values: &[i128], power: i32) -> Complex64 {
        let m = values.len();
        let mut total = Complex64::new(0.0, 0.0);
        for (i, &c) in values.iter().enumerate() {
            let phase = Complex64::from_polar(1.0, PI * i as f64 / m as f64);
            total += phase * c as f64;
        }
        total / 2.0f64.powi(power)
    }

    #[test]
    fn test_add() {
        let n = 8;
        let m = 16;
        for _ in 0..1000 {
            // Same base
            let va = rand_values(n);
            let vb = rand_values(n);
            let pa = rand_power();
            let pb = rand_power();
            let ta = reference_complex(&va, pa);
            let tb = reference_complex(&vb, pb);
            let expected = ta + tb;
            let a = DyadicComplexNumber::new(va, pa);
            let b = DyadicComplexNumber::new(vb, pb);
            let c = &a + &b;
            let actual = c.to_complex();
            assert!(
                (expected - actual).norm() / expected.norm().max(1.0) < 1e-6,
                "same base add failed"
            );

            // Mixed base
            let vb2 = rand_values(m);
            let pb2 = rand_power();
            let tb2 = reference_complex(&vb2, pb2);
            let expected2 = ta + tb2;
            let b2 = DyadicComplexNumber::new(vb2, pb2);
            let c2 = &a + &b2;
            let actual2 = c2.to_complex();
            assert!(
                (expected2 - actual2).norm() / expected2.norm().max(1.0) < 1e-6,
                "mixed base add failed"
            );
        }
    }

    #[test]
    fn test_sub() {
        let n = 8;
        for _ in 0..1000 {
            let va = rand_values(n);
            let vb = rand_values(n);
            let pa = rand_power();
            let pb = rand_power();
            let ta = reference_complex(&va, pa);
            let tb = reference_complex(&vb, pb);
            let expected = ta - tb;
            let a = DyadicComplexNumber::new(va, pa);
            let b = DyadicComplexNumber::new(vb, pb);
            let c = &a - &b;
            let actual = c.to_complex();
            assert!(
                (expected - actual).norm() / expected.norm().max(1.0) < 1e-6,
                "sub failed"
            );
        }
    }

    #[test]
    fn test_mul() {
        let n = 8;
        let m = 16;
        for _ in 0..1000 {
            // Same base
            let va = rand_values(n);
            let vb = rand_values(n);
            let pa = rand_power();
            let pb = rand_power();
            let ta = reference_complex(&va, pa);
            let tb = reference_complex(&vb, pb);
            let expected = ta * tb;
            let a = DyadicComplexNumber::new(va.clone(), pa);
            let b = DyadicComplexNumber::new(vb, pb);
            let c = &a * &b;
            let actual = c.to_complex();
            assert!(
                (expected - actual).norm() / expected.norm().max(1.0) < 1e-6,
                "same base mul failed"
            );

            // Mixed base
            let vb2 = rand_values(m);
            let pb2 = rand_power();
            let tb2 = reference_complex(&vb2, pb2);
            let expected2 = ta * tb2;
            let b2 = DyadicComplexNumber::new(vb2, pb2);
            let c2 = &a * &b2;
            let actual2 = c2.to_complex();
            assert!(
                (expected2 - actual2).norm() / expected2.norm().max(1.0) < 1e-6,
                "mixed base mul failed"
            );
        }
    }

    #[test]
    fn test_conjugate() {
        for _ in 0..1000 {
            for n_pow in 2..10 {
                let n = 1 << n_pow;
                let mut rng = rand::rng();
                let values: Vec<i128> = (0..n).map(|_| rng.random_range(-100..=100)).collect();
                let power = rng.random_range(0..=8);
                let d = DyadicComplexNumber::new(values, power);
                let conj_expected = d.to_complex().conj();
                let conj_actual = d.conj().to_complex();
                assert!(
                    (conj_expected - conj_actual).norm() < 1e-6,
                    "conjugate failed"
                );
            }
        }
    }
}
