//! Exact number rings the synthesis works in: Z[ω] ([`ZOmega`], Clifford+T)
//! and Z[ζ₁₆] ([`ZZeta`], Clifford+√T), plus the shared scalar types
//! ([`Int`] = i256, `f64` = fast-path float, [`MpFloat`] = MPFR).

pub(crate) mod types;
pub(crate) mod zomega;
pub(crate) mod zzeta;

pub(crate) use types::{Int, MpFloat};
pub(crate) use zomega::ZOmega;
pub(crate) use zzeta::ZZeta;
