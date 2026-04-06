//! Unified entry type for SO3 matrix cells — either over √2 or over √(√2+2).
//!
//! This mirrors the Python runtime polymorphism where `from_unitary` returns
//! either `AlgebraicIntegerOverRoot2` (for n=4 inputs) or
//! `AlgebraicIntegerOverRootRoot2Plus2` (for n=8 inputs).

use crate::algebra::DyadicComplexNumber;
use super::over_root2::AlgebraicIntegerOverRoot2;
use super::over_root_root2_plus2::AlgebraicIntegerOverRootRoot2Plus2;

/// A ratio entry that is either over √2 or over √(√2+2).
///
/// Cross-type operations always promote to RootRoot2Plus2 (the wider type).
#[derive(Clone, Debug)]
pub enum RatioEntry {
    Root2(AlgebraicIntegerOverRoot2),
    RootRoot2Plus2(AlgebraicIntegerOverRootRoot2Plus2),
}

impl RatioEntry {
    pub fn denominator_power(&self) -> u32 {
        match self {
            RatioEntry::Root2(r) => r.denominator_power,
            RatioEntry::RootRoot2Plus2(r) => r.denominator_power,
        }
    }

    pub fn to_f64(&self) -> f64 {
        match self {
            RatioEntry::Root2(r) => r.to_f64(),
            RatioEntry::RootRoot2Plus2(r) => r.to_f64(),
        }
    }

    pub fn neg(&self) -> RatioEntry {
        match self {
            RatioEntry::Root2(r) => RatioEntry::Root2(r.neg()),
            RatioEntry::RootRoot2Plus2(r) => RatioEntry::RootRoot2Plus2(r.neg()),
        }
    }

    pub fn to_rr2p2(&self) -> AlgebraicIntegerOverRootRoot2Plus2 {
        match self {
            RatioEntry::Root2(r) => r.to_rr2p2(),
            RatioEntry::RootRoot2Plus2(r) => r.clone(),
        }
    }

    pub fn add(&self, other: &RatioEntry) -> RatioEntry {
        match (self, other) {
            (RatioEntry::Root2(a), RatioEntry::Root2(b)) => {
                RatioEntry::Root2(a.add(b))
            }
            _ => {
                let a = self.to_rr2p2();
                let b = other.to_rr2p2();
                RatioEntry::RootRoot2Plus2(a.add(&b))
            }
        }
    }

    pub fn mul(&self, other: &RatioEntry) -> RatioEntry {
        match (self, other) {
            (RatioEntry::Root2(a), RatioEntry::Root2(b)) => {
                RatioEntry::Root2(a.mul(b))
            }
            _ => {
                let a = self.to_rr2p2();
                let b = other.to_rr2p2();
                RatioEntry::RootRoot2Plus2(a.mul(&b))
            }
        }
    }

    pub fn from_dyadic(d: &DyadicComplexNumber) -> RatioEntry {
        match d.values.len() {
            8 => RatioEntry::Root2(AlgebraicIntegerOverRoot2::from_dyadic(d)),
            16 => RatioEntry::RootRoot2Plus2(AlgebraicIntegerOverRootRoot2Plus2::from_dyadic(d)),
            n => panic!("Unsupported DyadicComplexNumber length {n} for RatioEntry"),
        }
    }

    pub fn zero_r2() -> RatioEntry {
        RatioEntry::Root2(AlgebraicIntegerOverRoot2::zero())
    }
}
