//! 3×3 SO(3) matrices over RatioEntry (AlgebraicIntegerOverRoot2 or RootRoot2Plus2).

use pyo3::prelude::*;

use crate::ratio::{AlgebraicIntegerOverRoot2, RatioEntry};

/// A 3×3 matrix in SO(3) with RatioEntry elements (row-major).
///
/// Entries are either AlgebraicIntegerOverRoot2 (n=4 gate sets) or
/// AlgebraicIntegerOverRootRoot2Plus2 (n=8 gate sets).
#[pyclass]
#[derive(Clone, Debug)]
pub struct SO3Matrix {
    pub values: Vec<RatioEntry>,
}

impl SO3Matrix {
    pub fn new(values: Vec<RatioEntry>) -> Self {
        assert_eq!(values.len(), 9, "SO3Matrix requires exactly 9 entries");
        Self { values }
    }

    pub fn from_r2(values: [AlgebraicIntegerOverRoot2; 9]) -> Self {
        Self {
            values: values.into_iter().map(RatioEntry::Root2).collect(),
        }
    }

    pub fn get(&self, i: usize, j: usize) -> &RatioEntry {
        &self.values[i * 3 + j]
    }

    pub fn mul(&self, other: &SO3Matrix) -> SO3Matrix {
        let mut result = Vec::with_capacity(9);
        for i in 0..3 {
            for j in 0..3 {
                let mut sum = self.get(i, 0).mul(other.get(0, j));
                sum = sum.add(&self.get(i, 1).mul(other.get(1, j)));
                sum = sum.add(&self.get(i, 2).mul(other.get(2, j)));
                result.push(sum);
            }
        }
        SO3Matrix::new(result)
    }

    pub fn to_float(&self) -> [f64; 9] {
        let mut out = [0.0f64; 9];
        for (i, v) in self.values.iter().enumerate() {
            out[i] = v.to_f64();
        }
        out
    }

    pub fn maximum_denominator_exponent(&self) -> u32 {
        self.values.iter().map(|v| v.denominator_power()).max().unwrap_or(0)
    }

    pub fn exponents(&self) -> [u32; 3] {
        let mut exp = [0u32; 3];
        for i in 0..3 {
            exp[i] = (0..3)
                .map(|j| self.get(i, j).denominator_power())
                .max()
                .unwrap_or(0);
        }
        exp
    }
}

#[pymethods]
impl SO3Matrix {
    #[new]
    fn py_new(values: Vec<AlgebraicIntegerOverRoot2>) -> PyResult<Self> {
        if values.len() != 9 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "SO3Matrix needs exactly 9 values",
            ));
        }
        Ok(Self::from_r2(values.try_into().unwrap()))
    }

    fn __mul__(&self, other: &SO3Matrix) -> Self {
        self.mul(other)
    }

    fn py_to_float(&self) -> Vec<f64> {
        self.to_float().to_vec()
    }

    fn py_maximum_denominator_exponent(&self) -> u32 {
        self.maximum_denominator_exponent()
    }

    fn py_exponents(&self) -> [u32; 3] {
        self.exponents()
    }

    fn copy(&self) -> Self {
        self.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::RingRoot2;
    use crate::matrix::factories::{bloch_identity, bloch_rx, bloch_ry, bloch_rz};

    fn one() -> AlgebraicIntegerOverRoot2 {
        AlgebraicIntegerOverRoot2::new(RingRoot2::new([1, 0]), 0)
    }
    fn zero() -> AlgebraicIntegerOverRoot2 {
        AlgebraicIntegerOverRoot2::new(RingRoot2::new([0, 0]), 0)
    }

    #[test]
    fn test_identity_mul() {
        let id = bloch_identity();
        let rx = bloch_rx(4, false);
        let result = id.mul(&rx);
        for (a, b) in result.to_float().iter().zip(rx.to_float().iter()) {
            assert!((a - b).abs() < 1e-10, "identity*rx should equal rx");
        }
    }

    #[test]
    fn test_inverse() {
        // rx * rx_dagger = identity for n=4
        let rx = bloch_rx(4, false);
        let rxdg = bloch_rx(4, true);
        let id = bloch_identity();
        let result = rx.mul(&rxdg);
        let id_f = id.to_float();
        let res_f = result.to_float();
        for (a, b) in id_f.iter().zip(res_f.iter()) {
            assert!((a - b).abs() < 1e-8, "rx*rxdg should be identity: {a} vs {b}");
        }
    }

    #[test]
    fn test_bloch_daggers_n4() {
        let id = bloch_identity();
        let id_f = id.to_float();
        for (m, md) in [
            (bloch_rx(4, false), bloch_rx(4, true)),
            (bloch_ry(4, false), bloch_ry(4, true)),
            (bloch_rz(4, false), bloch_rz(4, true)),
        ] {
            let r = m.mul(&md);
            for (a, b) in r.to_float().iter().zip(id_f.iter()) {
                assert!((a - b).abs() < 1e-8, "M*Mdg should be identity");
            }
            let r2 = md.mul(&m);
            for (a, b) in r2.to_float().iter().zip(id_f.iter()) {
                assert!((a - b).abs() < 1e-8, "Mdg*M should be identity");
            }
        }
    }

    #[test]
    fn test_bloch_daggers_n8() {
        let id = bloch_identity();
        let id_f = id.to_float();
        for (m, md) in [
            (bloch_rx(8, false), bloch_rx(8, true)),
            (bloch_ry(8, false), bloch_ry(8, true)),
            (bloch_rz(8, false), bloch_rz(8, true)),
        ] {
            let r = m.mul(&md);
            for (a, b) in r.to_float().iter().zip(id_f.iter()) {
                assert!((a - b).abs() < 1e-8, "M*Mdg n=8 should be identity, got {a} vs {b}");
            }
        }
    }
}
