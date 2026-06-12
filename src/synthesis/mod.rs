pub mod clifford_sqrt_t;
pub mod clifford_t;
pub mod cliffords;
pub mod cost_bound;
pub mod decomposer;
pub mod diag;
pub mod distance;
pub mod lattice;
pub mod lattice_common;
pub mod lattice_zeta;
pub mod search;
pub mod search_zeta;
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
pub use search::{brute_aligned_search, compute_align_vec, apply_u2t_dag_to_uv};
pub use synthesizer::{Synthesizer, SynthResult};
