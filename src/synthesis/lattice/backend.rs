//! Compiler-enforced contract shared by the per-ring lattice backends:
//! [`Omega`] (Z[ω], d=8, Clifford+T) and [`Zeta`] (Z[ζ_16], d=16, Clifford+√T).
//!
//! Only the structurally-identical lattice core is captured here — the
//! exact-i256 Gram recompute and the L²-LLL reduction, plus the dimension and
//! scratch type. The ring-specific pieces (building the anisotropic Q-metric
//! from a target, and the prefix/enumeration strategy) stay in each backend:
//! their signatures genuinely differ — the target is a 4-vector for √T and an
//! 8-vector for T — and an earlier attempt to unify them produced a leaky
//! abstraction. This trait's job is narrower: turn the "parallel function
//! names" convention into a contract the compiler checks, and give a third ring
//! a concrete checklist (implement this, then supply Q-build + enumeration).

use super::common::LllResult;

/// The lattice-enumeration core one ring must provide. See the module docs for
/// what deliberately lives outside it.
pub trait LatticeBackend {
    /// Lattice dimension (8 for Z[ω], 16 for Z[ζ_16]).
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
}

/// Z[ω] / Clifford+T, 8-dimensional.
pub struct Omega;

impl LatticeBackend for Omega {
    const DIM: usize = 8;
    type Scratch = super::omega::scratch::IntScratch;

    fn compute_gram_full(scratch: &mut Self::Scratch) -> bool {
        super::omega::lll::compute_gram_full(scratch)
    }

    fn run_lll(scratch: &mut Self::Scratch) -> LllResult {
        super::omega::lll::lll_l2(scratch)
    }
}

/// Z[ζ_16] / Clifford+√T, 16-dimensional.
pub struct Zeta;

impl LatticeBackend for Zeta {
    const DIM: usize = 16;
    type Scratch = super::zeta::scratch::IntScratch16;

    fn compute_gram_full(scratch: &mut Self::Scratch) -> bool {
        super::zeta::lll::compute_gram_full(scratch)
    }

    fn run_lll(scratch: &mut Self::Scratch) -> LllResult {
        super::zeta::lll::run_lll(scratch)
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
}
