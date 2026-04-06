//! The 24 single-qubit Clifford group elements as U2 matrices, and Clifford matching.

use crate::algebra::DyadicComplexNumber;
use crate::matrix::U2Matrix;

fn dcn(vals: [i128; 8], exp: i32) -> DyadicComplexNumber {
    DyadicComplexNumber::new(vals.to_vec(), exp)
}

fn zero() -> DyadicComplexNumber { dcn([0,0,0,0,0,0,0,0], 0) }
fn one()  -> DyadicComplexNumber { dcn([1,0,0,0,0,0,0,0], 0) }
fn neg_one() -> DyadicComplexNumber { dcn([-1,0,0,0,0,0,0,0], 0) }
fn imag() -> DyadicComplexNumber { dcn([0,0,0,0,1,0,0,0], 0) }
fn neg_imag() -> DyadicComplexNumber { dcn([0,0,0,0,-1,0,0,0], 0) }
fn one_over_sqrt2() -> DyadicComplexNumber { dcn([0,0,1,0,0,0,-1,0], 1) }
fn neg_one_over_sqrt2() -> DyadicComplexNumber { dcn([0,0,-1,0,0,0,1,0], 1) }

/// Build the 24 single-qubit Clifford U2 matrices, each paired with its gate string.
///
/// Gate string convention matches the Python `clifford_gates_to_u2` dict.
pub fn clifford_table() -> Vec<(&'static str, U2Matrix)> {
    let i_mat  = U2Matrix::new([one(), zero(), zero(), one()]);
    let h      = U2Matrix::new([one_over_sqrt2(), one_over_sqrt2(),
                                one_over_sqrt2(), neg_one_over_sqrt2()]);
    let s      = U2Matrix::new([one(), zero(), zero(), imag()]);
    let x      = U2Matrix::new([zero(), one(), one(), zero()]);
    let y      = U2Matrix::new([zero(), neg_imag(), imag(), zero()]);
    let z      = U2Matrix::new([one(), zero(), zero(), neg_one()]);

    let hx  = h.mul(&x);
    let hy  = h.mul(&y);
    let hz  = h.mul(&z);
    let sx  = s.mul(&x);
    let sy  = s.mul(&y);
    let sz  = s.mul(&z);
    let hs  = h.mul(&s);
    let hsx = h.mul(&s).mul(&x);
    let hsy = h.mul(&s).mul(&y);
    let hsz = h.mul(&s).mul(&z);
    let sh  = s.mul(&h);
    let shx = s.mul(&h).mul(&x);
    let shy = s.mul(&h).mul(&y);
    let shz = s.mul(&h).mul(&z);
    let hsh  = h.mul(&s).mul(&h);
    let hshx = h.mul(&s).mul(&h).mul(&x);
    let hshy = h.mul(&s).mul(&h).mul(&y);
    let hshz = h.mul(&s).mul(&h).mul(&z);

    vec![
        ("I",    i_mat),
        ("H",    h),
        ("S",    s),
        ("X",    x),
        ("Y",    y),
        ("Z",    z),
        ("XH",   hx),
        ("YH",   hy),
        ("ZH",   hz),
        ("XS",   sx),
        ("YS",   sy),
        ("ZS",   sz),
        ("SH",   hs),
        ("XSH",  hsx),
        ("YSH",  hsy),
        ("ZSH",  hsz),
        ("HS",   sh),
        ("XHS",  shx),
        ("YHS",  shy),
        ("ZHS",  shz),
        ("HSH",  hsh),
        ("XHSH", hshx),
        ("YHSH", hshy),
        ("ZHSH", hshz),
    ]
}

/// Match `matrix` to a single-qubit Clifford gate, returning its gate string.
///
/// Returns `None` if the matrix is not within 1e-8 Hilbert-Schmidt distance of
/// any Clifford.  Returns `None` also for the identity ("I") since appending
/// nothing is the right behavior.
pub fn match_clifford(matrix: &U2Matrix) -> Option<String> {
    for (name, clifford) in clifford_table() {
        if matrix.hilbert_schmidt_distance(&clifford) <= 1e-8 {
            if name == "I" {
                return None;
            }
            return Some(name.to_string());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::factories::unitary_identity;

    #[test]
    fn test_clifford_table_size() {
        assert_eq!(clifford_table().len(), 24);
    }

    #[test]
    fn test_identity_matches_none() {
        let id = unitary_identity(4);
        assert!(match_clifford(&id).is_none());
    }

    #[test]
    fn test_h_matches() {
        let h = U2Matrix::new([
            one_over_sqrt2(), one_over_sqrt2(),
            one_over_sqrt2(), neg_one_over_sqrt2(),
        ]);
        assert_eq!(match_clifford(&h), Some("H".to_string()));
    }

    #[test]
    fn test_all_cliffords_self_match() {
        for (name, mat) in clifford_table() {
            let result = match_clifford(&mat);
            if name == "I" {
                assert!(result.is_none(), "identity should return None");
            } else {
                assert_eq!(result, Some(name.to_string()), "gate {name} should match itself");
            }
        }
    }
}
