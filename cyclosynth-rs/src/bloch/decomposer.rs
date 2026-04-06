//! Bloch sphere decomposition of exactly-implementable unitaries.

use pyo3::prelude::*;

use crate::algebra::DyadicComplexNumber;
use crate::matrix::{
    bloch_rx, bloch_ry, bloch_rz, unitary_rx, unitary_ry,
    unitary_rz, SO3Matrix, U2Matrix,
};
use crate::ratio::RatioEntry;

use super::cliffords::match_clifford;
use super::translation::translate_decomposition;

/// Decomposes exactly-implementable unitaries into discrete rotations.
#[pyclass]
#[derive(Clone)]
pub struct BlochDecomposer {
    target: U2Matrix,
    matrix: SO3Matrix,
    base: usize,
    translate_gates: bool,
    rx_so3: SO3Matrix,
    ry_so3: SO3Matrix,
    rz_so3: SO3Matrix,
    rx_u2: U2Matrix,
    ry_u2: U2Matrix,
    rz_u2: U2Matrix,
}

impl BlochDecomposer {
    pub fn new(target: U2Matrix, translate_gates: bool) -> Self {
        let matrix = from_unitary(&target);
        // base is determined by DyadicComplexNumber value length divided by 2
        let base = target.values[0].values.len() / 2;
        let rx_so3 = bloch_rx(base, true);
        let ry_so3 = bloch_ry(base, true);
        let rz_so3 = bloch_rz(base, true);
        let rx_u2 = unitary_rx(base, true);
        let ry_u2 = unitary_ry(base, true);
        let rz_u2 = unitary_rz(base, true);
        Self {
            target,
            matrix,
            base,
            translate_gates,
            rx_so3,
            ry_so3,
            rz_so3,
            rx_u2,
            ry_u2,
            rz_u2,
        }
    }

    pub fn try_rx(&self, residual: &SO3Matrix) -> Vec<u32> {
        let mut exponents = Vec::new();
        let mut mat = residual.clone();
        for _ in 0..(self.base / 2 - 1) {
            mat = self.rx_so3.mul(&mat);
            exponents.push(mat.maximum_denominator_exponent());
        }
        exponents
    }

    pub fn try_ry(&self, residual: &SO3Matrix) -> Vec<u32> {
        let mut exponents = Vec::new();
        let mut mat = residual.clone();
        for _ in 0..(self.base / 2 - 1) {
            mat = self.ry_so3.mul(&mat);
            exponents.push(mat.maximum_denominator_exponent());
        }
        exponents
    }

    pub fn try_rz(&self, residual: &SO3Matrix) -> Vec<u32> {
        let mut exponents = Vec::new();
        let mut mat = residual.clone();
        for _ in 0..(self.base / 2 - 1) {
            mat = self.rz_so3.mul(&mat);
            exponents.push(mat.maximum_denominator_exponent());
        }
        exponents
    }

    pub fn decompose(&self) -> String {
        let mut residual_so3 = self.matrix.clone();
        let mut residual_u2 = self.target.clone();
        let max_steps = residual_so3.maximum_denominator_exponent();
        let mut decomposition = String::new();

        for _ in 0..max_steps {
            let x = self.try_rx(&residual_so3);
            let y = self.try_ry(&residual_so3);
            let z = self.try_rz(&residual_so3);

            let min_x = x.iter().copied().min().unwrap_or(u32::MAX);
            let min_y = y.iter().copied().min().unwrap_or(u32::MAX);
            let min_z = z.iter().copied().min().unwrap_or(u32::MAX);
            let overall_min = min_x.min(min_y).min(min_z);

            if min_x == overall_min {
                let apps = x.iter().position(|&v| v == min_x).unwrap() + 1;
                for _ in 0..apps {
                    residual_so3 = self.rx_so3.mul(&residual_so3);
                    residual_u2 = residual_u2.mul(&self.rx_u2);
                }
                for _ in 0..apps {
                    decomposition.push('x');
                }
            } else if min_y == overall_min {
                let apps = y.iter().position(|&v| v == min_y).unwrap() + 1;
                for _ in 0..apps {
                    residual_so3 = self.ry_so3.mul(&residual_so3);
                    residual_u2 = residual_u2.mul(&self.ry_u2);
                }
                for _ in 0..apps {
                    decomposition.push('y');
                }
            } else {
                let apps = z.iter().position(|&v| v == min_z).unwrap() + 1;
                for _ in 0..apps {
                    residual_so3 = self.rz_so3.mul(&residual_so3);
                    residual_u2 = residual_u2.mul(&self.rz_u2);
                }
                for _ in 0..apps {
                    decomposition.push('z');
                }
            }

            if overall_min == 0 {
                break;
            }
        }

        if let Some(clifford_str) = match_clifford(&residual_u2) {
            decomposition.push_str(&clifford_str);
        }

        if self.translate_gates {
            let magic = if self.base == 4 { "T" } else { "Q" };
            translate_decomposition(&decomposition, magic)
        } else {
            decomposition
        }
    }
}

/// Convert a U2Matrix to SO3Matrix via the Bloch sphere map.
///
/// For DyadicComplexNumber values of length 8 (base=4), produces Root2 entries.
/// For length 16 (base=8), produces RootRoot2Plus2 entries.
fn from_unitary(unitary: &U2Matrix) -> SO3Matrix {
    let a = &unitary.values[0];
    let b = &unitary.values[1];
    let c = &unitary.values[2];
    let d = &unitary.values[3];
    let adg = a.conj();
    let bdg = b.conj();
    let cdg = c.conj();
    let ddg = d.conj();

    let n = a.values.len();
    let mut half_vals = vec![0i128; n];
    half_vals[0] = 1;
    let mut i_vals = vec![0i128; n];
    i_vals[n / 2] = 1;

    let half = DyadicComplexNumber::new(half_vals, 1);
    let half_i = DyadicComplexNumber::new(i_vals.clone(), 1);
    let dyadic_i = DyadicComplexNumber::new(i_vals, 0);
    let neg_i = -&dyadic_i;

    // ax = ½((c·b† + d·a†) + (a·d† + b·c†))
    let cb_dg = &(c * &bdg) + &(d * &adg);
    let ad_dg = &(a * &ddg) + &(b * &cdg);
    let ax = &half * &(&cb_dg + &ad_dg);

    // bx = -i(c·b† + d·a† - ax)
    let bx_inner = &cb_dg - &ax;
    let bx = &neg_i * &bx_inner;

    // cx = a·b† + b·a†
    let cx = &(a * &bdg) + &(b * &adg);

    // ay = ½i((-c·b† + d·a†) + (-a·d† + b·c†))
    let neg_cb_dg = &(d * &adg) - &(c * &bdg);
    let neg_ad_dg = &(b * &cdg) - &(a * &ddg);
    let ay = &half_i * &(&neg_cb_dg + &neg_ad_dg);

    // by = -i(i*(d·a† - c·b†) - ay)
    let by_inner = &(&dyadic_i * &neg_cb_dg) - &ay;
    let by = &neg_i * &by_inner;

    // cy = -i·a·b† + i·b·a†
    let cy_lhs = &neg_i * &(a * &bdg);
    let cy_rhs = &dyadic_i * &(b * &adg);
    let cy = &cy_lhs + &cy_rhs;

    // az = ½((c·a† - d·b†) + (a·c† - b·d†))
    let ca_dg = &(c * &adg) - &(d * &bdg);
    let ac_dg = &(a * &cdg) - &(b * &ddg);
    let az = &half * &(&ca_dg + &ac_dg);

    // bz = -i(c·a† - d·b† - az)
    let bz_inner = &ca_dg - &az;
    let bz = &neg_i * &bz_inner;

    // cz = a·a† - b·b†
    let cz = &(a * &adg) - &(b * &bdg);

    let values_dcn = [ax, bx, cx, ay, by, cy, az, bz, cz];
    let values: Vec<RatioEntry> = values_dcn
        .iter()
        .map(|v| RatioEntry::from_dyadic(v))
        .collect();
    SO3Matrix::new(values)
}

#[pymethods]
impl BlochDecomposer {
    #[new]
    #[pyo3(signature = (target, translate_gates=true))]
    fn py_new(target: U2Matrix, translate_gates: bool) -> Self {
        Self::new(target, translate_gates)
    }

    fn py_decompose(&self) -> String {
        self.decompose()
    }

    fn py_try_rx(&self, residual: &SO3Matrix) -> Vec<u32> {
        self.try_rx(residual)
    }

    fn py_try_ry(&self, residual: &SO3Matrix) -> Vec<u32> {
        self.try_ry(residual)
    }

    fn py_try_rz(&self, residual: &SO3Matrix) -> Vec<u32> {
        self.try_rz(residual)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::DyadicComplexNumber;
    use crate::matrix::factories::{unitary_identity, unitary_rx, unitary_ry, unitary_rz};
    use rand::Rng;

    fn random_u2(n: usize) -> U2Matrix {
        let mut rng = rand::rng();
        let rx = unitary_rx(n, false);
        let ry = unitary_ry(n, false);
        let rz = unitary_rz(n, false);
        let mut mat = unitary_rx(n, false);
        for _ in 0..rng.random_range(1..=100) {
            let gate = match rng.random_range(0..3) {
                0 => &rx,
                1 => &ry,
                _ => &rz,
            };
            mat = mat.mul(gate);
        }
        mat
    }

    /// Build Clifford single-char gate matrices (all use n=4 base size).
    fn clifford_gate(ch: char) -> U2Matrix {
        let dcn = |vals: [i128; 8], exp: i32| DyadicComplexNumber::new(vals.to_vec(), exp);
        let zero  = || dcn([0,0,0,0,0,0,0,0], 0);
        let one   = || dcn([1,0,0,0,0,0,0,0], 0);
        let neg1  = || dcn([-1,0,0,0,0,0,0,0], 0);
        let imag  = || dcn([0,0,0,0,1,0,0,0], 0);
        let nimag = || dcn([0,0,0,0,-1,0,0,0], 0);
        let osqrt2 = || dcn([0,0,1,0,0,0,-1,0], 1);  // 1/√2
        match ch {
            'H' => U2Matrix::new([osqrt2(), osqrt2(), osqrt2(), {let mut v=osqrt2(); v.values[2]*=-1; v.values[6]*=-1; v}]),
            'S' => U2Matrix::new([one(), zero(), zero(), imag()]),
            'X' => U2Matrix::new([zero(), one(), one(), zero()]),
            'Y' => U2Matrix::new([zero(), nimag(), imag(), zero()]),
            'Z' => U2Matrix::new([one(), zero(), zero(), neg1()]),
            _   => unitary_identity(4),
        }
    }

    /// Construct a U2 matrix from a gate string. Supports lowercase xyz (discrete rotations
    /// of any n), uppercase HSXYZ (Clifford gates, applied at n=4 base size), and 'I'.
    fn construct_u2(n: usize, gates: &[char]) -> U2Matrix {
        let mut mat = unitary_identity(n);
        let x = unitary_rx(n, false);
        let y = unitary_ry(n, false);
        let z = unitary_rz(n, false);
        for &g in gates {
            let gate = match g {
                'x' => x.clone(),
                'y' => y.clone(),
                'z' => z.clone(),
                c => clifford_gate(c),
            };
            mat = gate.mul(&mat);
        }
        mat
    }

    #[test]
    fn test_constructor() {
        BlochDecomposer::new(random_u2(4), true);
        BlochDecomposer::new(random_u2(8), true);
    }

    #[test]
    fn test_try_rz_prefers_z() {
        let mut rng = rand::rng();
        for _ in 0..20 {
            let n = 8;
            let mut target = unitary_identity(n);
            let mut num_rz = rng.random_range(1..=10);
            if num_rz % 2 == 0 {
                num_rz += 1;
            }
            let rz_gate = unitary_rz(n, false);
            for _ in 0..num_rz {
                target = rz_gate.mul(&target);
            }
            let bloch = BlochDecomposer::new(target, false);
            let min_z = *bloch.try_rz(&bloch.matrix).iter().min().unwrap();
            assert!(
                bloch.try_rx(&bloch.matrix).iter().all(|&v| v >= min_z),
                "try_rz should give minimum for pure Rz target"
            );
            assert!(
                bloch.try_ry(&bloch.matrix).iter().all(|&v| v >= min_z),
                "try_rz should give minimum for pure Rz target"
            );
        }
    }

    #[test]
    fn test_decompose_t() {
        let n = 4;
        let length = 100;
        let mut rng = rand::rng();
        for _ in 0..20 {
            let gates: Vec<char> = (0..length)
                .map(|_| match rng.random_range(0..3) {
                    0 => 'x',
                    1 => 'y',
                    _ => 'z',
                })
                .collect();
            let u2 = construct_u2(n, &gates);
            let bloch = BlochDecomposer::new(u2.clone(), false);
            let decomp: Vec<char> = bloch.decompose().chars().collect();

            if decomp != gates {
                let decomp_u = construct_u2(n, &decomp);
                let dist = decomp_u.hilbert_schmidt_distance(&u2);
                assert!(dist < 1e-8, "decompose_t: dist={dist}");
            }
        }
    }

    #[test]
    fn test_decompose_sqrtt() {
        let n = 8;
        let length = 100;
        let mut rng = rand::rng();
        for _ in 0..20 {
            let gates: Vec<char> = (0..length)
                .map(|_| match rng.random_range(0..3) {
                    0 => 'x',
                    1 => 'y',
                    _ => 'z',
                })
                .collect();
            let u2 = construct_u2(n, &gates);
            let bloch = BlochDecomposer::new(u2.clone(), false);
            let decomp: Vec<char> = bloch.decompose().chars().collect();

            if decomp != gates {
                let decomp_u = construct_u2(n, &decomp);
                let dist = decomp_u.hilbert_schmidt_distance(&u2);
                assert!(dist < 1e-8, "decompose_sqrtt: dist={dist}");
            }
        }
    }
}
