//! 2×2 unitary matrices over DyadicComplexNumber.

use pyo3::prelude::*;

use crate::algebra::DyadicComplexNumber;

/// A 2×2 unitary matrix with DyadicComplexNumber entries (row-major).
#[pyclass]
#[derive(Clone, Debug)]
pub struct U2Matrix {
    pub values: [DyadicComplexNumber; 4],
}

impl U2Matrix {
    pub fn new(values: [DyadicComplexNumber; 4]) -> Self {
        Self { values }
    }

    pub fn get(&self, i: usize, j: usize) -> &DyadicComplexNumber {
        &self.values[i * 2 + j]
    }

    pub fn mul(&self, other: &U2Matrix) -> U2Matrix {
        let (a, b) = (self, other);
        let c11 = &(a.get(0, 0) * b.get(0, 0)) + &(a.get(0, 1) * b.get(1, 0));
        let c12 = &(a.get(0, 0) * b.get(0, 1)) + &(a.get(0, 1) * b.get(1, 1));
        let c21 = &(a.get(1, 0) * b.get(0, 0)) + &(a.get(1, 1) * b.get(1, 0));
        let c22 = &(a.get(1, 0) * b.get(0, 1)) + &(a.get(1, 1) * b.get(1, 1));
        U2Matrix::new([c11, c12, c21, c22])
    }

    pub fn dagger(&self) -> U2Matrix {
        let [a, b, c, d] = &self.values;
        U2Matrix::new([a.conj(), c.conj(), b.conj(), d.conj()])
    }

    pub fn hilbert_schmidt_distance(&self, other: &U2Matrix) -> f64 {
        let prod = other.dagger().mul(self);
        let trace = &(prod.get(0, 0) + prod.get(1, 1));
        let trace_abs = trace.abs();
        let val = (trace_abs / 2.0).min(1.0);
        let dist = (1.0 - val * val).sqrt();
        if dist > 0.0 { dist } else { 0.0 }
    }

    pub fn to_complex(&self) -> [(f64, f64); 4] {
        [
            {
                let c = self.values[0].to_complex();
                (c.re, c.im)
            },
            {
                let c = self.values[1].to_complex();
                (c.re, c.im)
            },
            {
                let c = self.values[2].to_complex();
                (c.re, c.im)
            },
            {
                let c = self.values[3].to_complex();
                (c.re, c.im)
            },
        ]
    }
}

#[pymethods]
impl U2Matrix {
    #[new]
    fn py_new(values: [DyadicComplexNumber; 4]) -> Self {
        Self::new(values)
    }

    fn __mul__(&self, other: &U2Matrix) -> Self {
        self.mul(other)
    }

    fn py_dagger(&self) -> Self {
        self.dagger()
    }

    fn py_hilbert_schmidt_distance(&self, other: &U2Matrix) -> f64 {
        self.hilbert_schmidt_distance(other)
    }

    fn py_to_complex(&self) -> [(f64, f64); 4] {
        self.to_complex()
    }

    fn copy(&self) -> Self {
        self.clone()
    }
}
