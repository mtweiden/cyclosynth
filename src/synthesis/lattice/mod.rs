//! Lenstra-style LLL+SE integer enumeration backends, one per gate set:
//! [`omega`] (8D, Z[ω], Clifford+T) and [`zeta`] (16D, Z[ζ_16], Clifford+√T).
//! [`common`] holds the dimension-independent L²-LLL parameters and helpers
//! shared by both.

#[allow(dead_code)] // owner-accepted LatticeBackend contract; not yet load-bearing (see memory)
pub(crate) mod backend;
pub(crate) mod common;
pub mod omega;
pub mod zeta;
