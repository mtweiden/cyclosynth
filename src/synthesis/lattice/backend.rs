//! Compiler-enforced contract shared by the per-ring lattice backends:
//! [`Omega`] (Z[ω], d=8, Clifford+T) and [`Zeta`] (Z[ζ], d=16, Clifford+√T).
//!
//! Captures the dimension-only-different core: the dimension and scratch type,
//! scratch build/reset, the exact-i256 Gram + L²-LLL reduction, the
//! unimodularity det check, and the f64 Cholesky + LU cap-center solve. The
//! ring-specific pieces (building the anisotropic Q-metric
//! from a target, and the prefix/enumeration strategy) stay in each backend:
//! their signatures genuinely differ — the target is a 4-vector for √T and an
//! 8-vector for T — and unifying them produces a leaky abstraction. This
//! trait's job is narrower: turn the "parallel function
//! names" convention into a contract the compiler checks, and give a third ring
//! a concrete checklist (implement this, then supply Q-build + enumeration).

use crate::synthesis::clifford_sqrt_t::lattice as zeta_lattice;
use crate::synthesis::clifford_t::lattice as omega_lattice;
use super::common::LllResult;

/// The lattice-enumeration core one ring must provide. See the module docs for
/// what deliberately lives outside it.
pub(crate) trait LatticeBackend {
    /// Lattice dimension (8 for Z[ω], 16 for Z[ζ]).
    const DIM: usize;

    /// Per-call working set: exact i256 Gram, i64 basis, and the Gram-Schmidt
    /// scratch (f64 at d=8, multi-precision at d=16).
    type Scratch;

    /// Recompute G = B·Q_int·Bᵀ in i256 from the current basis. Returns false
    /// on i256 overflow, signaling the caller to abort to its fallback.
    fn compute_gram_full(scratch: &mut Self::Scratch) -> bool;

    /// L²-LLL-reduce the basis from a clean start (reset → Gram → reduce) at
    /// the precision this ring requires.
    fn run_lll(scratch: &mut Self::Scratch) -> LllResult;

    /// Fresh per-call working set for tolerance `eps`.
    fn new_scratch(eps: f64) -> Self::Scratch;

    /// Reset the basis to the identity (to reuse a scratch across prefixes).
    fn reset_basis(scratch: &mut Self::Scratch);

    /// Determinant of the current basis, or `None` on integer overflow.
    fn det_exact(scratch: &Self::Scratch) -> Option<i64>;

    /// f64 Cholesky of the post-LLL Gram; `false` if not positive-definite.
    fn cholesky_f64(scratch: &mut Self::Scratch) -> bool;

    /// Solve for the cap-center in lattice coords; `false` on singular LU.
    fn lu_solve_int_inplace(scratch: &mut Self::Scratch) -> bool;
}

/// Z[ω] / Clifford+T, 8-dimensional.
pub(crate) struct Omega;

impl LatticeBackend for Omega {
    const DIM: usize = 8;
    type Scratch = omega_lattice::scratch::IntScratch;

    fn compute_gram_full(scratch: &mut Self::Scratch) -> bool {
        omega_lattice::lll::compute_gram_full(scratch)
    }

    fn run_lll(scratch: &mut Self::Scratch) -> LllResult {
        omega_lattice::lll::lll_l2(scratch)
    }

    fn new_scratch(eps: f64) -> Self::Scratch { omega_lattice::scratch::IntScratch::new(eps) }
    fn reset_basis(scratch: &mut Self::Scratch) { scratch.reset_basis() }
    fn det_exact(scratch: &Self::Scratch) -> Option<i64> {
        omega_lattice::cholesky_lu::det_exact(&scratch.basis)
    }
    fn cholesky_f64(scratch: &mut Self::Scratch) -> bool {
        omega_lattice::cholesky_lu::cholesky_f64(scratch)
    }
    fn lu_solve_int_inplace(scratch: &mut Self::Scratch) -> bool {
        omega_lattice::cholesky_lu::lu_solve_int_inplace(scratch)
    }
}

/// Z[ζ] / Clifford+√T, 16-dimensional.
pub(crate) struct Zeta;

impl LatticeBackend for Zeta {
    const DIM: usize = 16;
    type Scratch = zeta_lattice::scratch::IntScratch16;

    fn compute_gram_full(scratch: &mut Self::Scratch) -> bool {
        zeta_lattice::lll::compute_gram_full(scratch)
    }

    fn run_lll(scratch: &mut Self::Scratch) -> LllResult {
        zeta_lattice::lll::run_lll(scratch)
    }

    fn new_scratch(eps: f64) -> Self::Scratch { zeta_lattice::scratch::IntScratch16::new(eps) }
    fn reset_basis(scratch: &mut Self::Scratch) { scratch.reset_basis() }
    fn det_exact(scratch: &Self::Scratch) -> Option<i64> {
        zeta_lattice::cholesky_lu::det_exact(&scratch.basis)
    }
    fn cholesky_f64(scratch: &mut Self::Scratch) -> bool {
        zeta_lattice::cholesky_lu::cholesky_f64(scratch)
    }
    fn lu_solve_int_inplace(scratch: &mut Self::Scratch) -> bool {
        zeta_lattice::cholesky_lu::lu_solve_int_inplace(scratch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Generic consumer: only compiles if both rings satisfy the contract, so a
    // drift in either backend's primitive signatures breaks the build here.
    fn dim<B: LatticeBackend>() -> usize {
        B::DIM
    }

    #[test]
    fn backends_expose_their_dimension() {
        assert_eq!(dim::<Omega>(), 8);
        assert_eq!(dim::<Zeta>(), 16);
    }

    // Exercises the scratch primitives generically, so a signature drift in
    // either backend's new/reset breaks the build here.
    fn build_reset<B: LatticeBackend>(eps: f64) -> B::Scratch {
        let mut s = B::new_scratch(eps);
        B::reset_basis(&mut s);
        s
    }

    #[test]
    fn backends_build_and_reset_scratch() {
        let _o = build_reset::<Omega>(1e-5);
        let _z = build_reset::<Zeta>(1e-5);
    }
}
