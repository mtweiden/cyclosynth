//! Dimension-independent kernel shared by the two Lenstra-style LLL+SE
//! backends, which live with their gate sets: `clifford_t::lattice` (8D,
//! Z[ω]) and `clifford_sqrt_t::lattice` (16D, Z[ζ]). [`common`] holds
//! the L²-LLL parameters and helpers; [`backend`] the shared contract.

#[allow(dead_code)] // owner-accepted LatticeBackend contract; not yet load-bearing (see memory)
pub(crate) mod backend;
pub(crate) mod common;
