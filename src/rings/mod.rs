//! Exact number rings the synthesis works in: Z[ω] ([`ZOmega`], Clifford+T)
//! and Z[ζ₁₆] ([`ZZeta`], Clifford+√T), plus the shared scalar types
//! ([`Int`] = i256, [`Float`] = f64, [`MpFloat`] = MPFR).

pub mod types;
pub mod zomega;
pub mod zzeta;

pub use types::{Int, Float, MpFloat};
pub use zomega::ZOmega;
pub use zzeta::ZZeta;
