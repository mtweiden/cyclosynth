//! Factory functions for creating standard unitary and Bloch rotation matrices.

use crate::algebra::{DyadicComplexNumber, RingRoot2, RingRootRoot2Plus2};
use crate::ratio::{AlgebraicIntegerOverRoot2, AlgebraicIntegerOverRootRoot2Plus2, RatioEntry};
use super::u2::U2Matrix;
use super::so3::SO3Matrix;

// ─── Discrete trig (exact algebraic, for Bloch sphere matrices) ──────────────

/// cos(π/n) as a RatioEntry. Supports n=4 (Root2) and n=8 (RootRoot2Plus2).
pub fn discrete_cos(n: usize) -> RatioEntry {
    match n {
        4 => RatioEntry::Root2(AlgebraicIntegerOverRoot2::new(RingRoot2::new([0, 1]), 2)),
        8 => RatioEntry::RootRoot2Plus2(AlgebraicIntegerOverRootRoot2Plus2::new(
            RingRootRoot2Plus2::new([3, 2, 0, 0]),
            3,
        )),
        _ => panic!("discrete_cos: unsupported n={n}"),
    }
}

/// sin(π/n) as a RatioEntry. Supports n=4 (Root2) and n=8 (RootRoot2Plus2).
pub fn discrete_sin(n: usize) -> RatioEntry {
    match n {
        4 => RatioEntry::Root2(AlgebraicIntegerOverRoot2::new(RingRoot2::new([0, 1]), 2)),
        8 => RatioEntry::RootRoot2Plus2(AlgebraicIntegerOverRootRoot2Plus2::new(
            RingRootRoot2Plus2::new([1, 1, 0, 0]),
            3,
        )),
        _ => panic!("discrete_sin: unsupported n={n}"),
    }
}

// ─── Dyadic trig (complex, for U2 matrices) ──────────────────────────────────

/// cos(kπ/(2n)) as DyadicComplexNumber of length n (from the Python dyadic_cos).
pub fn dyadic_cos(k: i32, n: usize) -> DyadicComplexNumber {
    assert!(n.is_power_of_two());
    let k_mod = k.rem_euclid(2 * n as i32) as usize;
    if k_mod == 0 {
        let mut v = vec![0i128; n];
        v[0] = 1;
        return DyadicComplexNumber::new(v, 0);
    }
    if k_mod == n {
        let mut v = vec![0i128; n];
        v[0] = -1;
        return DyadicComplexNumber::new(v, 0);
    }
    let k1 = k_mod % n;
    let k2 = (n - k_mod) % n;
    let mut values = vec![0i128; n];
    let sign: i128 = if k_mod > n { -1 } else { 1 };
    values[k1] += sign;
    values[k2] -= sign;
    DyadicComplexNumber::new(values, 1)
}

/// sin(kπ/(2n)) as DyadicComplexNumber of length n (from the Python dyadic_sin).
pub fn dyadic_sin(k: i32, n: usize) -> DyadicComplexNumber {
    assert!(n.is_power_of_two());
    let k_mod = k.rem_euclid(2 * n as i32) as usize;
    let half_n = n / 2;
    if k_mod == half_n {
        let mut v = vec![0i128; n];
        v[0] = 1;
        return DyadicComplexNumber::new(v, 0);
    }
    if k_mod == 3 * half_n {
        let mut v = vec![0i128; n];
        v[0] = -1;
        return DyadicComplexNumber::new(v, 0);
    }
    let mut values = vec![0i128; n];
    let k1 = (half_n as i64 - k_mod as i64).rem_euclid(n as i64) as usize;
    let k2 = (half_n + k_mod) % n;
    // Match Python: sign_1 = (-1)**(k_1 < 0) * (-1)**((k - n//2) > n)
    // k_1 = (n//2 - k) → negative if k > n//2
    let raw_k1 = half_n as i64 - k_mod as i64;
    let sign1: i128 = if raw_k1 < 0 { -1 } else { 1 }
        * if (k_mod as i64 - half_n as i64) > n as i64 { -1 } else { 1 };
    // sign_2 = (-1)**(k_2 > n) * (-1)**((k - n//2) > n)
    let sign2: i128 = if k2 > n { -1 } else { 1 }
        * if (k_mod as i64 - half_n as i64) > n as i64 { -1 } else { 1 };
    values[k1] += sign1;
    values[k2] -= sign2;
    DyadicComplexNumber::new(values, 1)
}

// ─── U2 factories ─────────────────────────────────────────────────────────────

pub fn unitary_identity(n: usize) -> U2Matrix {
    let mut one_vals = vec![0i128; 2 * n];
    one_vals[0] = 1;
    let zero_vals = vec![0i128; 2 * n];
    let one = DyadicComplexNumber::new(one_vals, 0);
    let zero = DyadicComplexNumber::new(zero_vals, 0);
    U2Matrix::new([one.clone(), zero.clone(), zero, one])
}

pub fn unitary_rx(n: usize, dagger: bool) -> U2Matrix {
    let mut i_vals = vec![0i128; 2 * n];
    i_vals[n] = 1;
    let i_dcn = DyadicComplexNumber::new(i_vals, 0);
    let c = dyadic_cos(1, 2 * n);
    let s_raw = dyadic_sin(1, 2 * n);
    let neg_i = -&i_dcn;
    let s = &neg_i * &s_raw;
    let mut mat = U2Matrix::new([c.clone(), s.clone(), s, c]);
    if dagger {
        let rx = mat.clone();
        for _ in 0..(2 * n - 2) {
            mat = rx.mul(&mat);
        }
    }
    mat
}

pub fn unitary_ry(n: usize, dagger: bool) -> U2Matrix {
    let c = dyadic_cos(1, 2 * n);
    let s = dyadic_sin(1, 2 * n);
    let neg_s = -&s;
    let mut mat = U2Matrix::new([c.clone(), neg_s, s, c]);
    if dagger {
        let ry = mat.clone();
        for _ in 0..(2 * n - 2) {
            mat = ry.mul(&mat);
        }
    }
    mat
}

pub fn unitary_rz(n: usize, dagger: bool) -> U2Matrix {
    let len = 2 * n;
    let mut me_vals = vec![0i128; len];
    let mut pe_vals = vec![0i128; len];
    me_vals[len - 1] = -1;
    pe_vals[1] = 1;
    let zero_vals = vec![0i128; len];
    let me = DyadicComplexNumber::new(me_vals, 0);
    let pe = DyadicComplexNumber::new(pe_vals, 0);
    let zero = DyadicComplexNumber::new(zero_vals, 0);
    let mut mat = U2Matrix::new([me, zero.clone(), zero, pe]);
    if dagger {
        let rz = mat.clone();
        for _ in 0..(2 * n - 2) {
            mat = rz.mul(&mat);
        }
    }
    mat
}

// ─── SO3 / Bloch sphere factories ─────────────────────────────────────────────

fn one_r2() -> RatioEntry {
    RatioEntry::Root2(AlgebraicIntegerOverRoot2::new(RingRoot2::new([1, 0]), 0))
}

fn zero_r2() -> RatioEntry {
    RatioEntry::Root2(AlgebraicIntegerOverRoot2::zero())
}

pub fn bloch_identity() -> SO3Matrix {
    SO3Matrix::new(vec![
        one_r2(),  zero_r2(), zero_r2(),
        zero_r2(), one_r2(),  zero_r2(),
        zero_r2(), zero_r2(), one_r2(),
    ])
}

pub fn bloch_rx(n: usize, dagger: bool) -> SO3Matrix {
    let c = discrete_cos(n);
    let s = discrete_sin(n);
    let neg_s = s.neg();
    let mut mat = SO3Matrix::new(vec![
        one_r2(),  zero_r2(), zero_r2(),
        zero_r2(), c,         s,
        zero_r2(), neg_s,     discrete_cos(n),
    ]);
    if dagger {
        let rx = mat.clone();
        for _ in 0..(2 * n - 2) {
            mat = rx.mul(&mat);
        }
    }
    mat
}

pub fn bloch_ry(n: usize, dagger: bool) -> SO3Matrix {
    let c = discrete_cos(n);
    let s = discrete_sin(n);
    let neg_s = s.neg();
    let mut mat = SO3Matrix::new(vec![
        c,         zero_r2(), neg_s,
        zero_r2(), one_r2(),  zero_r2(),
        discrete_sin(n), zero_r2(), discrete_cos(n),
    ]);
    if dagger {
        let ry = mat.clone();
        for _ in 0..(2 * n - 2) {
            mat = ry.mul(&mat);
        }
    }
    mat
}

pub fn bloch_rz(n: usize, dagger: bool) -> SO3Matrix {
    let c = discrete_cos(n);
    let s = discrete_sin(n);
    let neg_s = s.neg();
    let mut mat = SO3Matrix::new(vec![
        c,         discrete_sin(n), zero_r2(),
        neg_s,     discrete_cos(n), zero_r2(),
        zero_r2(), zero_r2(),       one_r2(),
    ]);
    if dagger {
        let rz = mat.clone();
        for _ in 0..(2 * n - 2) {
            mat = rz.mul(&mat);
        }
    }
    mat
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn bloch_rx_f64(n: usize) -> [[f64; 3]; 3] {
        let c = (PI / n as f64).cos();
        let s = (PI / n as f64).sin();
        [[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]]
    }

    fn bloch_ry_f64(n: usize) -> [[f64; 3]; 3] {
        let c = (PI / n as f64).cos();
        let s = (PI / n as f64).sin();
        [[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]]
    }

    fn bloch_rz_f64(n: usize) -> [[f64; 3]; 3] {
        let c = (PI / n as f64).cos();
        let s = (PI / n as f64).sin();
        [[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]]
    }

    fn dyadic_rx_f64(n: usize) -> [[num_complex::Complex64; 2]; 2] {
        use num_complex::Complex64;
        let c = ((PI / (2 * n) as f64).cos()).into();
        let s = Complex64::new(0.0, -(PI / (2 * n) as f64).sin());
        [[c, s], [s, c]]
    }

    fn dyadic_ry_f64(n: usize) -> [[num_complex::Complex64; 2]; 2] {
        use num_complex::Complex64;
        let c: Complex64 = ((PI / (2 * n) as f64).cos()).into();
        let s: Complex64 = ((PI / (2 * n) as f64).sin()).into();
        [[c, -s], [s, c]]
    }

    fn dyadic_rz_f64(n: usize) -> [[num_complex::Complex64; 2]; 2] {
        use num_complex::Complex64;
        let pe = Complex64::from_polar(1.0, PI / (2 * n) as f64);
        let me = Complex64::from_polar(1.0, -PI / (2 * n) as f64);
        [[me, Complex64::ZERO], [Complex64::ZERO, pe]]
    }

    #[test]
    fn test_dyadic_rx_ry_rz() {
        for n_pow in 2..10usize {
            let n = 1 << n_pow;
            let rx = unitary_rx(n, false);
            let ry = unitary_ry(n, false);
            let rz = unitary_rz(n, false);
            let rx_n = dyadic_rx_f64(n);
            let ry_n = dyadic_ry_f64(n);
            let rz_n = dyadic_rz_f64(n);
            for i in 0..2 {
                for j in 0..2 {
                    let rx_c = rx.values[i * 2 + j].to_complex();
                    assert!((rx_c.re - rx_n[i][j].re).abs() < 1e-6 && (rx_c.im - rx_n[i][j].im).abs() < 1e-6,
                        "rx[{i}][{j}] failed for n={n}");
                    let ry_c = ry.values[i * 2 + j].to_complex();
                    assert!((ry_c.re - ry_n[i][j].re).abs() < 1e-6 && (ry_c.im - ry_n[i][j].im).abs() < 1e-6,
                        "ry[{i}][{j}] failed for n={n}");
                    let rz_c = rz.values[i * 2 + j].to_complex();
                    assert!((rz_c.re - rz_n[i][j].re).abs() < 1e-6 && (rz_c.im - rz_n[i][j].im).abs() < 1e-6,
                        "rz[{i}][{j}] failed for n={n}");
                }
            }
        }
    }

    #[test]
    fn test_bloch_values_n4() {
        for n in [4usize, 8] {
            let rx = bloch_rx(n, false);
            let ry = bloch_ry(n, false);
            let rz = bloch_rz(n, false);
            let rx_n = bloch_rx_f64(n);
            let ry_n = bloch_ry_f64(n);
            let rz_n = bloch_rz_f64(n);
            for i in 0..3 {
                for j in 0..3 {
                    assert!((rx.get(i, j).to_f64() - rx_n[i][j]).abs() < 1e-6,
                        "bloch_rx({n})[{i}][{j}]: {} != {}", rx.get(i,j).to_f64(), rx_n[i][j]);
                    assert!((ry.get(i, j).to_f64() - ry_n[i][j]).abs() < 1e-6,
                        "bloch_ry({n})[{i}][{j}]");
                    assert!((rz.get(i, j).to_f64() - rz_n[i][j]).abs() < 1e-6,
                        "bloch_rz({n})[{i}][{j}]");
                }
            }
        }
    }

    #[test]
    fn test_unitary_daggers() {
        let id8 = unitary_identity(8);
        for n in [4usize, 8] {
            for (rx, rxdg) in [
                (unitary_rx(n, false), unitary_rx(n, true)),
                (unitary_ry(n, false), unitary_ry(n, true)),
                (unitary_rz(n, false), unitary_rz(n, true)),
            ] {
                let d = rx.mul(&rxdg).hilbert_schmidt_distance(&id8);
                assert!(d < 1e-8, "U*U† not identity for n={n}: dist={d}");
            }
        }
    }

    #[test]
    fn test_hilbert_schmidt_distance() {
        let rx = unitary_rx(4, false);
        let ry = unitary_ry(4, false);
        assert_eq!(rx.hilbert_schmidt_distance(&rx), 0.0);
        assert!(rx.hilbert_schmidt_distance(&ry) > 0.0);
    }
}
