//! Generic ratio of algebraic integers.

use pyo3::prelude::*;

use super::alg_int::{gcd_values, AlgInt};

/// A ratio of two algebraic integers: numerator / denominator.
#[pyclass]
#[derive(Clone, Debug)]
pub struct IntegerRatio {
    pub numerator: AlgInt,
    pub denominator: AlgInt,
}

impl IntegerRatio {
    pub fn new(numerator: AlgInt, denominator: AlgInt) -> Self {
        Self {
            numerator,
            denominator,
        }
    }

    pub fn to_f64(&self) -> f64 {
        self.numerator.to_f64() / self.denominator.to_f64()
    }

    pub fn simplify(&mut self) {
        if self.numerator.is_zero() {
            self.denominator = AlgInt::Int(1);
            return;
        }
        if matches!(&self.denominator, AlgInt::Int(1)) {
            return;
        }
        let num_vals = self.numerator.values();
        let den_vals = self.denominator.values();
        let num_gcd = gcd_values(&num_vals);
        let den_gcd = gcd_values(&den_vals);
        let full_gcd = gcd_u128(num_gcd, den_gcd);
        if full_gcd <= 1 {
            return;
        }
        let fg = full_gcd as i128;
        match &mut self.numerator {
            AlgInt::Int(n) => *n /= fg,
            AlgInt::Root2(r) => r.values.iter_mut().for_each(|v| *v /= fg),
            AlgInt::RootRoot2Plus2(r) => r.values.iter_mut().for_each(|v| *v /= fg),
        }
        match &mut self.denominator {
            AlgInt::Int(n) => *n /= fg,
            AlgInt::Root2(r) => r.values.iter_mut().for_each(|v| *v /= fg),
            AlgInt::RootRoot2Plus2(r) => r.values.iter_mut().for_each(|v| *v /= fg),
        }
    }

    pub fn add(&self, other: &IntegerRatio) -> IntegerRatio {
        // a/b + c/d = (a*d + c*b) / (b*d)
        let ad = self.numerator.mul(&other.denominator);
        let cb = other.numerator.mul(&self.denominator);
        let new_num = ad.add(&cb);
        let new_den = self.denominator.mul(&other.denominator);
        let mut r = IntegerRatio::new(new_num, new_den);
        r.simplify();
        r
    }

    pub fn sub(&self, other: &IntegerRatio) -> IntegerRatio {
        let neg_other = IntegerRatio::new(other.numerator.neg(), other.denominator.clone());
        self.add(&neg_other)
    }

    pub fn mul(&self, other: &IntegerRatio) -> IntegerRatio {
        let new_num = self.numerator.mul(&other.numerator);
        let new_den = self.denominator.mul(&other.denominator);
        let mut r = IntegerRatio::new(new_num, new_den);
        r.simplify();
        r
    }

    pub fn mul_int(&self, n: i128) -> IntegerRatio {
        let new_num = self.numerator.mul(&AlgInt::Int(n));
        let mut r = IntegerRatio::new(new_num, self.denominator.clone());
        r.simplify();
        r
    }

    pub fn neg(&self) -> IntegerRatio {
        IntegerRatio::new(self.numerator.neg(), self.denominator.clone())
    }

    pub fn inverse(&self) -> IntegerRatio {
        IntegerRatio::new(self.denominator.clone(), self.numerator.clone())
    }

    pub fn conj(&self) -> IntegerRatio {
        IntegerRatio::new(self.numerator.conj(), self.denominator.conj())
    }
}

fn gcd_u128(a: u128, b: u128) -> u128 {
    if b == 0 {
        a
    } else {
        gcd_u128(b, a % b)
    }
}

#[pymethods]
impl IntegerRatio {
    #[new]
    #[pyo3(signature = (numerator, denominator=1))]
    fn py_new(numerator: i128, denominator: i128) -> Self {
        Self::new(AlgInt::Int(numerator), AlgInt::Int(denominator))
    }

    fn __add__(&self, other: &IntegerRatio) -> Self {
        self.add(other)
    }

    fn __sub__(&self, other: &IntegerRatio) -> Self {
        self.sub(other)
    }

    fn __mul__(&self, other: &IntegerRatio) -> Self {
        self.mul(other)
    }

    fn __neg__(&self) -> Self {
        self.neg()
    }

    fn __repr__(&self) -> String {
        format!("({:?}) / ({:?})", self.numerator, self.denominator)
    }

    fn py_to_float(&self) -> f64 {
        self.to_f64()
    }

    fn py_inverse(&self) -> Self {
        self.inverse()
    }

    fn py_simplify(&mut self) {
        self.simplify();
    }
}
