//! Single-qubit gate synthesis: approximate a target unitary by a Clifford+T
//! or Clifford+√T circuit (arXiv:2510.05816).
//!
//! ## Domain glossary
//!
//! Recurring terms in the names and docs below — kept as-is because they're
//! the field's / this crate's standard vocabulary, defined here once:
//!
//! - **T / Q gates** — `T` is the π/8 phase gate; `Q` = √T is the Clifford+√T
//!   generator. The cost we minimize is `T_count + 3·Q_count`.
//! - **lde** — "least denominator exponent": the power `k` of √2 in a
//!   circuit's ring denominator, used as the search-depth parameter
//!   (`min_lde`/`max_lde`).
//! - **Z[ω] / Z[ζ₁₆]** — the rings the two backends search: Z[ω] (8-dim,
//!   `zomega`, Clifford+T) and Z[ζ₁₆] (16-dim, `zeta`/`zzeta`, Clifford+√T).
//! - **u2t / u2q** — a 2×2 unitary over the Clifford+T (`u2t`) or Clifford+√T
//!   (`u2q`) ring; **uv** is its (u, v) first-column encoding.
//! - **Matsumoto-Amano (MA) prefix** — the Clifford+T canonical-form left
//!   prefix `L_{t'}` (`ma_prefix`); **FGKM** — the analogous Clifford+√T
//!   canonical form (arXiv:1501.04944), enumerated by syllable count `m`.
//! - **det-phase / d_R** — the determinant's root-of-unity coset class;
//!   `dr`-filters prune candidate prefixes by it.
//! - Per-ring lattice search: **CFA** = Cholesky Factorization Algorithm
//!   (per-row, Fig. 4 of Nguyen-Stehlé 2009); **LLL / L²-LLL** = lattice basis
//!   reduction; **SE** = Schnorr-Euchner point enumeration; **BKZ** = block
//!   reduction; **SVP** = shortest-vector problem; **GSO** = Gram-Schmidt
//!   orthogonalization; **dd** = double-double (~106-bit) arithmetic.

pub mod clifford_sqrt_t;
pub mod clifford_t;
pub mod cliffords;
pub mod cost_bound;
pub mod decomposer;
pub mod diag;
pub mod distance;
pub mod lattice;
pub mod synthesizer;

/// Build the global rayon pool with 16 MiB worker stacks before its
/// lazy default init. The optimal-mode pipeline runs two parity
/// branches concurrently; their par_iters' stolen jobs nest per-prefix
/// `map_init` scratch frames on pool workers, overflowing rayon's 2 MiB
/// default stacks (the `OPTIMAL_PAR_MIN_LEN = 1` abort family, and the
/// intermittent full-suite SIGABRT flake). If the pool was already
/// initialised elsewhere (a binary setting num_threads, or a racing
/// par_iter), `build_global` errs and this is a no-op — callers get
/// whatever stacks that pool was built with.
pub(crate) fn ensure_rayon_stack() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let _ = rayon::ThreadPoolBuilder::new()
            .stack_size(16 * 1024 * 1024)
            .build_global();
    });
}

/// Transpose-interleave: deal `items` round-robin across `stride`
/// positions (position j gets ranks j, j+stride, j+2·stride, …).
/// Rayon's contiguous chunking would hand one worker all the
/// front-of-list items — exactly the cost-sorted cheapest prefixes
/// (16D) or the structurally-similar `build_ma_prefix_set` neighbours (8D) —
/// serializing the items most likely to finish first; dealing makes
/// every chunk's early items span the whole list.
pub(crate) fn stride_interleave<T: Copy>(items: &[T], stride: usize) -> Vec<T> {
    let stride = stride.max(1);
    let mut out = Vec::with_capacity(items.len());
    for j in 0..stride {
        let mut idx = j;
        while idx < items.len() {
            out.push(items[idx]);
            idx += stride;
        }
    }
    out
}

pub use cliffords::CLIFFORD_TABLE_T;
pub use decomposer::BlochDecomposer;
pub use distance::{
    diamond_distance_float, diamond_distance_float_mpfr,
    diamond_distance_u2q_float,
    Mat2,
};
pub use lattice::omega::brute::{brute_aligned_search, compute_align_vec, apply_u2t_dag_to_uv};
pub use synthesizer::{Synthesizer, SynthResult};
