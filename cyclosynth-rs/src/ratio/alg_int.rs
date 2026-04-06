//! Enum wrapper for algebraic integer types used in IntegerRatio.

use crate::algebra::{RingRoot2, RingRootRoot2Plus2};

/// A value that is either a plain integer or an algebraic integer.
#[derive(Clone, Debug)]
pub enum AlgInt {
    Int(i128),
    Root2(RingRoot2),
    RootRoot2Plus2(RingRootRoot2Plus2),
}

impl AlgInt {
    pub fn to_f64(&self) -> f64 {
        match self {
            AlgInt::Int(n) => *n as f64,
            AlgInt::Root2(r) => r.to_f64(),
            AlgInt::RootRoot2Plus2(r) => r.to_f64(),
        }
    }

    pub fn is_zero(&self) -> bool {
        match self {
            AlgInt::Int(n) => *n == 0,
            AlgInt::Root2(r) => r.values.iter().all(|&v| v == 0),
            AlgInt::RootRoot2Plus2(r) => r.values.iter().all(|&v| v == 0),
        }
    }

    pub fn values(&self) -> Vec<i128> {
        match self {
            AlgInt::Int(n) => vec![*n],
            AlgInt::Root2(r) => r.values.to_vec(),
            AlgInt::RootRoot2Plus2(r) => r.values.to_vec(),
        }
    }

    pub fn mul(&self, other: &AlgInt) -> AlgInt {
        match (self, other) {
            (AlgInt::Int(a), AlgInt::Int(b)) => AlgInt::Int(a * b),
            (AlgInt::Int(a), AlgInt::Root2(b)) | (AlgInt::Root2(b), AlgInt::Int(a)) => {
                AlgInt::Root2(b * *a)
            }
            (AlgInt::Int(a), AlgInt::RootRoot2Plus2(b))
            | (AlgInt::RootRoot2Plus2(b), AlgInt::Int(a)) => {
                AlgInt::RootRoot2Plus2(b * *a)
            }
            (AlgInt::Root2(a), AlgInt::Root2(b)) => AlgInt::Root2(a * b),
            (AlgInt::Root2(a), AlgInt::RootRoot2Plus2(b))
            | (AlgInt::RootRoot2Plus2(b), AlgInt::Root2(a)) => {
                AlgInt::RootRoot2Plus2(b * a)
            }
            (AlgInt::RootRoot2Plus2(a), AlgInt::RootRoot2Plus2(b)) => {
                AlgInt::RootRoot2Plus2(a * b)
            }
        }
    }

    pub fn add(&self, other: &AlgInt) -> AlgInt {
        match (self, other) {
            (AlgInt::Int(a), AlgInt::Int(b)) => AlgInt::Int(a + b),
            (AlgInt::Int(a), AlgInt::Root2(b)) | (AlgInt::Root2(b), AlgInt::Int(a)) => {
                let a_r2 = RingRoot2::new([*a, 0]);
                AlgInt::Root2(&a_r2 + b)
            }
            (AlgInt::Int(a), AlgInt::RootRoot2Plus2(b))
            | (AlgInt::RootRoot2Plus2(b), AlgInt::Int(a)) => {
                let a_rr = RingRootRoot2Plus2::new([*a, 0, 0, 0]);
                AlgInt::RootRoot2Plus2(&a_rr + b)
            }
            (AlgInt::Root2(a), AlgInt::Root2(b)) => AlgInt::Root2(a + b),
            (AlgInt::Root2(a), AlgInt::RootRoot2Plus2(b))
            | (AlgInt::RootRoot2Plus2(b), AlgInt::Root2(a)) => {
                let a_rr = RingRootRoot2Plus2::from_ring_root2(a);
                AlgInt::RootRoot2Plus2(&a_rr + b)
            }
            (AlgInt::RootRoot2Plus2(a), AlgInt::RootRoot2Plus2(b)) => {
                AlgInt::RootRoot2Plus2(a + b)
            }
        }
    }

    pub fn neg(&self) -> AlgInt {
        match self {
            AlgInt::Int(n) => AlgInt::Int(-n),
            AlgInt::Root2(r) => AlgInt::Root2(-r),
            AlgInt::RootRoot2Plus2(r) => AlgInt::RootRoot2Plus2(-r),
        }
    }

    pub fn conj(&self) -> AlgInt {
        match self {
            AlgInt::Int(n) => AlgInt::Int(*n),
            AlgInt::Root2(r) => AlgInt::Root2(r.conj()),
            AlgInt::RootRoot2Plus2(_r) => {
                // RingRootRoot2Plus2 doesn't have a standard conj in Python
                self.clone()
            }
        }
    }
}

fn gcd_u128(a: u128, b: u128) -> u128 {
    if b == 0 {
        a
    } else {
        gcd_u128(b, a % b)
    }
}

pub fn gcd_values(vals: &[i128]) -> u128 {
    let mut g = 0u128;
    for &v in vals {
        g = gcd_u128(g, v.unsigned_abs());
    }
    g
}
