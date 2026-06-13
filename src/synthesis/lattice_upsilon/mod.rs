//! Single-qubit exact synthesis pipeline for the n=12 case
//! (root ζ = e^{iπ/12} = ζ₂₄, ring Z[ζ₂₄]).
//!
//! Mirror of [`super::lattice_zeta`] (Clifford+√T, n=16, Z[ζ₁₆]) with two
//! ring-specific swaps:
//!
//! - **Σ matrix** ([`sigma`]): per-element block is 8×8 over the +i coset
//!   reps `{1, 17, 13, 5}` of `(Z/24)*` (instead of `{1, 5, 9, 13}` for
//!   the ζ₁₆ case). The Gram is anisotropic — `Σ_el^T Σ_el = 4I + 2C`
//!   coupling column `k` with `k+4` — so the enumerator must reduce
//!   against this metric, not against a scalar identity (SPEC §4).
//!
//! - **Bullet constraints** ([`enumerate`]): three vanishings (`√2`, `√3`,
//!   `√6` components of `u₁·conj(u₁) + u₂·conj(u₂)`) instead of the
//!   single bilinear of the n=4 case or the three different forms of
//!   n=16. The cyclotomic-basis derivation is in
//!   [`enumerate::bullets_per_element_twice`].
//!
//! - **Phase sweep** ([`synthesize`]): `ζ^ℓ` for `ℓ ∈ 0..24` (24 phases),
//!   generalizing the √T sweep.
//!
//! ## Reachability is determined by ring membership alone
//!
//! `ζ₂₄` is in the Forest–Gosset–Kliuchnikov–McKinnon "golden set"
//! `{2, 4, 6, 8, 12}` (J. Math. Phys. 56, 082201, 2015): `G₁₂ =
//! U₂(Z[ζ₂₄, 1/2])` is exactly the ancilla-free reachable group. So the
//! synthesis constraints are precisely **norm + three bullets + alignment**
//! — no `√2`-residue / F₄ parity / leading-unit check to bolt on. The local
//! 2-adic syndrome is implied by the rational-part equation `r = 2^k`.
//!
//! The denominator generator is `√2` (not `√3` or `√6`): in `Z[ζ₂₄]`,
//! `√2 = ζ³ + ζ⁻³`, so `Z[ζ₂₄, 1/2] = Z[ζ₂₄, 1/√2]`. The synthesis
//! radius is `2^k`. (n=12 is the *largest* golden-set n; n≥16 fails and
//! requires catalytic ancillas.)
//!
//! ## Module layout
//!
//! Mirrors [`super::lattice_zeta`] at coarse granularity (`sigma`,
//! `enumerate`, `synthesize`). The fine-grained √T split (`cholesky_lu`,
//! `integer`, `lll`, `q_metric`, `scratch`, `se`) is deferred until the
//! LLL+SE port lands — the algebra (Σ, Gram, bullets, norm, alignment)
//! here is verified independently and is the load-bearing input to that
//! port. The brute-force [`enumerate::phase1_brute`] is the same
//! correctness oracle role that `super::search_zeta::phase1_brute` plays
//! for n=16.

pub mod bkz;
pub mod cholesky_lu;
pub mod enumerate;
pub mod integer;
pub mod lll;
pub mod lll_f64;
pub mod mitm;
pub mod mitm_half_se;
pub mod q_metric;
pub mod scratch;
pub mod se;
pub mod sigma;
pub mod synthesize;

use crate::rings::Float;
use std::sync::atomic::AtomicBool;

/// Per-worker scratch for the LLL+SE pipeline. Allocated once per worker
/// (the underlying [`scratch::IntScratch16`] pre-allocates every MPFR /
/// i256 buffer up front).
pub struct LatticeScratch {
    inner: scratch::IntScratch16,
}

impl LatticeScratch {
    pub fn new(eps: Float) -> Self {
        Self {
            inner: scratch::IntScratch16::new(eps),
        }
    }
}

/// LLL+SE phase1 (production path for large k). Returns integer 16-vectors
/// `(u₁-coeffs, u₂-coeffs)` satisfying norm + 3 bullets + alignment.
pub fn phase1(
    scratch: &mut LatticeScratch,
    v: [Float; 4],
    k: u32,
    eps: Float,
    max_leaves: u64,
    budget_hit: &AtomicBool,
) -> Vec<[i64; 16]> {
    scratch.inner.reset_basis();
    integer::phase1(&mut scratch.inner, v, k, eps, max_leaves, budget_hit)
}

/// `max_solutions = 1` short-circuit variant.
pub fn phase1_first(
    scratch: &mut LatticeScratch,
    v: [Float; 4],
    k: u32,
    eps: Float,
    max_leaves: u64,
    budget_hit: &AtomicBool,
) -> Option<[i64; 16]> {
    scratch.inner.reset_basis();
    let sols = integer::phase1_with_stop(
        &mut scratch.inner,
        v,
        k,
        eps,
        max_leaves,
        budget_hit,
        |_| true,
    );
    sols.into_iter().next()
}

/// LLL+SE phase1 with an early-exit predicate.
pub fn phase1_with_stop<F>(
    scratch: &mut LatticeScratch,
    v: [Float; 4],
    k: u32,
    eps: Float,
    max_leaves: u64,
    budget_hit: &AtomicBool,
    should_stop: F,
) -> Vec<[i64; 16]>
where
    F: FnMut(&[i64; 16]) -> bool,
{
    scratch.inner.reset_basis();
    integer::phase1_with_stop(
        &mut scratch.inner,
        v,
        k,
        eps,
        max_leaves,
        budget_hit,
        should_stop,
    )
}

/// Same as [`phase1_with_stop`] but also returns the lightweight
/// `Phase1Stats` (leaves visited, norm/bullets/alignment-pass counts,
/// budget_hit flag). Useful for measurement / benchmarking.
pub fn phase1_with_stop_stats<F>(
    scratch: &mut LatticeScratch,
    v: [Float; 4],
    k: u32,
    eps: Float,
    max_leaves: u64,
    budget_hit: &AtomicBool,
    should_stop: F,
) -> (Vec<[i64; 16]>, integer::Phase1Stats)
where
    F: FnMut(&[i64; 16]) -> bool,
{
    scratch.inner.reset_basis();
    integer::phase1_with_stop_stats(
        &mut scratch.inner,
        v,
        k,
        eps,
        max_leaves,
        budget_hit,
        should_stop,
    )
}

pub use integer::Phase1Stats;

pub use enumerate::{
    alignment_sq, bullets_per_element_twice, bullets_per_element_twice_int, bullets_total_twice,
    bullets_total_twice_int, bullets_zero, bullets_zero_int, compute_align_vec,
    norm_sqr_per_element, norm_sqr_per_element_int, norm_sqr_total, norm_sqr_total_int,
    phase1_brute, phase1_brute_first, target_norm_int, uv_to_xy,
};
pub use sigma::{embed_one, embed_pair, gram_el_int, gram_int, sigma_16, sigma_el, COSET_REPS};
pub use synthesize::{
    best_phase, solution_to_unitary, synthesize, synthesize_first, zeta_pow, SynthResult,
    NUM_PHASES,
};
