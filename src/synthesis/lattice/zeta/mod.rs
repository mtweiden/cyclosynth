//! Native 16D Lenstra-style search for Clifford+√T (Z[ζ_16]) synthesis.
//!
//! This module is the Z[ζ_16] analog of [`super::lattice`] (which targets
//! Z[ω] / Clifford+T). The two modules are deliberately kept separate to
//! isolate the precision and integer-width choices: f64 Gram-Schmidt is
//! provably sufficient at d=8 (Theorem 2 of Nguyen-Stehlé 2009) but not at
//! d=16, so the 16D GS runs in f64 at moderate ε with escalation to MPFR
//! ([`lll_f64`] vs [`lll`]) and MPFR-only below ~1e-8.
//!
//! Pipeline and module layout mirror [`super::lattice`]; see
//! [`integer`] for the per-call stage breakdown. Brute force and
//! y-helpers live in [`super::brute_search_zeta`]; U2Q reconstruction in
//! [`super::clifford_sqrt_t`].
//!
//! ## Solution layout
//!
//!   `sol = [u_1.a, u_1.b, …, u_1.h, u_2.a, …, u_2.h]`
//!     i.e. `sol[0..8]` = u_1's ζ-basis coefficients,
//!          `sol[8..16]` = u_2's ζ-basis coefficients.
//!
//! Reconstruction follows the SU(2) convention used by Z[ω]'s
//! `solution_to_u2t`:
//!
//!   `U = [[u_1, −u_2*], [u_2, u_1*]] / √(2^k)`

pub mod brute;
pub mod bkz;
pub mod cholesky_lu;
pub mod integer;
pub mod lll;
pub mod lll_f64;
pub mod q_metric;
pub mod scratch;
pub mod se;

pub use integer::{find_aligned_lattice_points_with_stop, find_aligned_lattice_points_mpfr};
pub use scratch::IntScratch16;
pub use se::{set_verify_prune_mpfr, verify_prune_mpfr};

// ─── Tests preserving the previous flat-module test suite ────────────────────

#[cfg(test)]
mod tests;
