//! Z[e^{iπ/4}, 1/√2] — the DOmega ring used for Clifford+T synthesis.

use num_complex::Complex64;
use pyo3::prelude::*;
use std::f64::consts::PI;
use std::ops::{Add, Mul, Neg, Sub};

use super::dyadic_complex::format_omega_repr;

/// A number in Z[ω, 1/√2] where ω = e^{iπ/4}.
///
/// Represented as (a + bω + cω² + dω³) / (√2)^l.
#[pyclass]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DOmega {
    #[pyo3(get)]
    pub values: [i128; 4],
    #[pyo3(get)]
    pub denominator_exponent: i32,
}

impl DOmega {
    pub fn new(values: [i128; 4], denominator_exponent: i32) -> Self {
        Self {
            values,
            denominator_exponent,
        }
    }

    pub fn zero() -> Self {
        Self {
            values: [0, 0, 0, 0],
            denominator_exponent: 0,
        }
    }

    pub fn one() -> Self {
        Self {
            values: [1, 0, 0, 0],
            denominator_exponent: 0,
        }
    }

    pub fn simplify(&mut self) {
        for _ in 0..self.denominator_exponent / 2 {
            if self.values.iter().all(|&v| v % 2 == 0) && self.denominator_exponent >= 2 {
                self.values.iter_mut().for_each(|v| *v >>= 1);
                self.denominator_exponent -= 2;
            } else {
                break;
            }
        }
    }

    pub fn conj(&self) -> Self {
        let mut new_values = self.values;
        new_values[1..].reverse();
        for v in &mut new_values[1..] {
            *v = -*v;
        }
        Self {
            values: new_values,
            denominator_exponent: self.denominator_exponent,
        }
    }

    pub fn bullet(&self) -> Self {
        Self {
            values: [
                self.values[0],
                -self.values[1],
                -self.values[2],
                self.values[3],
            ],
            denominator_exponent: self.denominator_exponent,
        }
    }

    pub fn to_complex(&self) -> Complex64 {
        let m = 4;
        let mut total = Complex64::new(0.0, 0.0);
        for (i, &coeff) in self.values.iter().enumerate() {
            let phase = Complex64::from_polar(1.0, PI * i as f64 / m as f64);
            total += phase * coeff as f64;
        }
        total / std::f64::consts::SQRT_2.powi(self.denominator_exponent)
    }

    pub fn abs(&self) -> f64 {
        self.to_complex().norm()
    }

    pub fn magnitude_squared(&self) -> f64 {
        let a = self;
        let b = self.conj();
        let c = &*a * &b;
        c.to_complex().re
    }
}

impl Add for &DOmega {
    type Output = DOmega;
    fn add(self, rhs: &DOmega) -> DOmega {
        let mut lhs_vals = self.values;
        let mut rhs_vals = rhs.values;
        let offset = (self.denominator_exponent - rhs.denominator_exponent).unsigned_abs();
        if self.denominator_exponent < rhs.denominator_exponent {
            lhs_vals.iter_mut().for_each(|v| *v <<= offset);
        } else if self.denominator_exponent > rhs.denominator_exponent {
            rhs_vals.iter_mut().for_each(|v| *v <<= offset);
        }
        let new_power = self.denominator_exponent.max(rhs.denominator_exponent);
        let new_values = [
            lhs_vals[0] + rhs_vals[0],
            lhs_vals[1] + rhs_vals[1],
            lhs_vals[2] + rhs_vals[2],
            lhs_vals[3] + rhs_vals[3],
        ];
        let mut result = DOmega::new(new_values, new_power);
        result.simplify();
        result
    }
}

impl Neg for &DOmega {
    type Output = DOmega;
    fn neg(self) -> DOmega {
        DOmega {
            values: [-self.values[0], -self.values[1], -self.values[2], -self.values[3]],
            denominator_exponent: self.denominator_exponent,
        }
    }
}

impl Sub for &DOmega {
    type Output = DOmega;
    fn sub(self, rhs: &DOmega) -> DOmega {
        self + &(-rhs)
    }
}

impl Mul for &DOmega {
    type Output = DOmega;
    fn mul(self, rhs: &DOmega) -> DOmega {
        let new_power = self.denominator_exponent + rhs.denominator_exponent;
        let m = 4;
        let mut new_values = [0i128; 4];
        for (i, &ca) in self.values.iter().enumerate() {
            for (j, &cb) in rhs.values.iter().enumerate() {
                let k = i + j;
                let coeff = ca * cb;
                let sign = if k >= m { -1i128 } else { 1i128 };
                new_values[k % m] += sign * coeff;
            }
        }
        let mut result = DOmega::new(new_values, new_power);
        result.simplify();
        result
    }
}

#[pymethods]
impl DOmega {
    #[new]
    fn py_new(values: [i128; 4], denominator_exponent: i32) -> Self {
        Self::new(values, denominator_exponent)
    }

    fn __add__(&self, other: &DOmega) -> Self {
        self + other
    }

    fn __sub__(&self, other: &DOmega) -> Self {
        self - other
    }

    fn __mul__(&self, other: &DOmega) -> Self {
        self * other
    }

    fn __neg__(&self) -> Self {
        -self
    }

    fn __repr__(&self) -> String {
        format_omega_repr(&self.values, self.denominator_exponent, false)
    }

    fn py_conj(&self) -> Self {
        self.conj()
    }

    fn py_bullet(&self) -> Self {
        self.bullet()
    }

    fn py_to_complex(&self) -> (f64, f64) {
        let c = self.to_complex();
        (c.re, c.im)
    }

    fn py_abs(&self) -> f64 {
        self.abs()
    }

    fn py_magnitude_squared(&self) -> f64 {
        self.magnitude_squared()
    }

    fn py_simplify(&mut self) {
        self.simplify();
    }

    fn copy(&self) -> Self {
        self.clone()
    }
}
