//! Native 16D Lenstra-style search for Clifford+√T (Z[ζ_16]) synthesis.
//!
//! This module is the Z[ζ_16] analog of [`super::omega`] (which targets
//! Z[ω] / Clifford+T). The two modules are deliberately kept separate to
//! isolate the precision and integer-width choices: f64 Gram-Schmidt is
//! provably sufficient at d=8 (Theorem 2 of Nguyen-Stehlé 2009) but not at
//! d=16, so the 16D GS runs in MPFR throughout.
//!
//! Pipeline and module layout mirror [`super::omega`]; see
//! [`integer`] for the per-call stage breakdown. Brute force and
//! y-helpers live in [`brute`]; U2Q reconstruction in
//! [`crate::synthesis::clifford_sqrt_t`].
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
pub mod dd;
pub mod integer;
pub mod lll;
pub mod q_metric;
pub mod scratch;
pub mod se;

pub use integer::{find_aligned_lattice_points_with_stop, find_aligned_lattice_points_mpfr};
pub use scratch::IntScratch16;
pub use se::{set_verify_prune_mpfr, verify_prune_mpfr};

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests;
