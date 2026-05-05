//! Optional per-search diagnostic counters, gated by `CYCLOSYNTH_TRACE=1`.
//!
//! Usage:
//!   CYCLOSYNTH_TRACE=1 ./time_synthesis ...
//!
//! Output is printed to stderr (so it doesn't pollute timing tables on stdout).
//! The diagnostic boundary is one `dc_search` call: counters are reset at the
//! start and dumped at the end, showing per-lde where time and prefix count
//! went.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

static TRACE_ENABLED: OnceLock<bool> = OnceLock::new();

pub fn trace_enabled() -> bool {
    *TRACE_ENABLED.get_or_init(|| {
        std::env::var("CYCLOSYNTH_TRACE")
            .ok()
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

// ─── Per-lde counters (reset at the start of each dc_search) ─────────────────

/// MA prefixes considered at this lde.
pub static N_PREFIXES: AtomicU64 = AtomicU64::new(0);

/// Prefixes rejected by `mat_to_uv` (not in SU(2): wrong determinant or
/// odd-parity ζ̄ adjustment failed).
pub static N_MAT_TO_UV_REJECTED: AtomicU64 = AtomicU64::new(0);

/// Total Schnorr-Euchner leaf-callback invocations summed across all
/// prefixes in this dc_search. Useful for spotting individual prefixes
/// whose ellipsoid is "fat" relative to alignment requirements.
pub static N_SE_CALLBACKS: AtomicU64 = AtomicU64::new(0);

/// SE candidates that satisfied the integer constraints (norm shell,
/// bilinear form, alignment) but produced a unitary whose diamond
/// distance to the target exceeded ε. Should be 0 in steady state;
/// non-zero indicates a precision issue in the SE bound vs the actual
/// alignment threshold.
pub static N_DIST_REJECTED: AtomicU64 = AtomicU64::new(0);

// ─── Per-phase nanosecond accumulators ───────────────────────────────────────
//
// CPU-summed nanoseconds across all phase1 calls in the current dc_search.
// Total ≈ wall-time × n_threads in steady state (high parallel efficiency).

pub static T_BUILD_NS: AtomicU64 = AtomicU64::new(0);
pub static T_LLL_NS: AtomicU64 = AtomicU64::new(0);
pub static T_CHOLESKY_NS: AtomicU64 = AtomicU64::new(0);
pub static T_LU_NS: AtomicU64 = AtomicU64::new(0);
pub static T_SE_NS: AtomicU64 = AtomicU64::new(0);

// ─── LLL iteration telemetry ─────────────────────────────────────────────────

/// Sum of LLL inner-loop iterations across all phase1 calls in this dc_search.
pub static N_LLL_ITERS_TOTAL: AtomicU64 = AtomicU64::new(0);
/// Max iter count seen by any single LLL call in this dc_search.
pub static N_LLL_ITERS_MAX: AtomicU64 = AtomicU64::new(0);
/// LLL calls that hit the safety cap (typically 10_000 iters). Should be 0;
/// non-zero indicates LLL cycling at the active precision.
pub static N_LLL_AT_CAP: AtomicU64 = AtomicU64::new(0);

/// Cumulative passes across all `lazy_size_reduce` invocations. Diagnostic
/// signal for whether the size-reduction inner loop converges in 1-2 passes
/// (per Nguyen-Stehlé 2009 expectation) or runs hot.
pub static N_LAZY_PASSES_TOTAL: AtomicU64 = AtomicU64::new(0);
/// Number of `lazy_size_reduce` invocations.
pub static N_LAZY_CALLS_TOTAL: AtomicU64 = AtomicU64::new(0);
/// Max passes ever seen in a single invocation.
pub static N_LAZY_PASSES_MAX: AtomicU64 = AtomicU64::new(0);

/// Record one lazy_size_reduce invocation's pass count.
pub fn record_lazy_passes(passes: u64) {
    N_LAZY_PASSES_TOTAL.fetch_add(passes, Ordering::Relaxed);
    N_LAZY_CALLS_TOTAL.fetch_add(1, Ordering::Relaxed);
    let mut current = N_LAZY_PASSES_MAX.load(Ordering::Relaxed);
    while passes > current {
        match N_LAZY_PASSES_MAX.compare_exchange_weak(
            current,
            passes,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => return,
            Err(observed) => current = observed,
        }
    }
}

/// Record an LLL call's iteration count + atomically update
/// `N_LLL_ITERS_MAX` via a compare-exchange loop.
pub fn record_lll_iters(iters: u64, cap: u64) {
    N_LLL_ITERS_TOTAL.fetch_add(iters, Ordering::Relaxed);
    if iters >= cap {
        N_LLL_AT_CAP.fetch_add(1, Ordering::Relaxed);
    }
    let mut current = N_LLL_ITERS_MAX.load(Ordering::Relaxed);
    while iters > current {
        match N_LLL_ITERS_MAX.compare_exchange_weak(
            current,
            iters,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => return,
            Err(observed) => current = observed,
        }
    }
}

pub fn reset_all() {
    for c in [
        &N_PREFIXES,
        &N_MAT_TO_UV_REJECTED,
        &N_SE_CALLBACKS,
        &N_DIST_REJECTED,
        &T_BUILD_NS,
        &T_LLL_NS,
        &T_CHOLESKY_NS,
        &T_LU_NS,
        &T_SE_NS,
        &N_LLL_ITERS_TOTAL,
        &N_LLL_ITERS_MAX,
        &N_LLL_AT_CAP,
        &N_LAZY_PASSES_TOTAL,
        &N_LAZY_CALLS_TOTAL,
        &N_LAZY_PASSES_MAX,
    ] {
        c.store(0, Ordering::Relaxed);
    }
}

/// Snapshot of the current counter values for printing.
pub struct Snapshot {
    pub prefixes: u64,
    pub mat_to_uv_rejected: u64,
    pub se_callbacks: u64,
    pub dist_rejected: u64,
    pub t_build_ms: f64,
    pub t_lll_ms: f64,
    pub t_cholesky_ms: f64,
    pub t_lu_ms: f64,
    pub t_se_ms: f64,
    pub lll_iters_total: u64,
    pub lll_iters_max: u64,
    pub lll_at_cap: u64,
    pub lazy_passes_total: u64,
    pub lazy_calls_total: u64,
    pub lazy_passes_max: u64,
}

pub fn snapshot() -> Snapshot {
    Snapshot {
        prefixes: N_PREFIXES.load(Ordering::Relaxed),
        mat_to_uv_rejected: N_MAT_TO_UV_REJECTED.load(Ordering::Relaxed),
        se_callbacks: N_SE_CALLBACKS.load(Ordering::Relaxed),
        dist_rejected: N_DIST_REJECTED.load(Ordering::Relaxed),
        t_build_ms: T_BUILD_NS.load(Ordering::Relaxed) as f64 / 1.0e6,
        t_lll_ms: T_LLL_NS.load(Ordering::Relaxed) as f64 / 1.0e6,
        t_cholesky_ms: T_CHOLESKY_NS.load(Ordering::Relaxed) as f64 / 1.0e6,
        t_lu_ms: T_LU_NS.load(Ordering::Relaxed) as f64 / 1.0e6,
        t_se_ms: T_SE_NS.load(Ordering::Relaxed) as f64 / 1.0e6,
        lll_iters_total: N_LLL_ITERS_TOTAL.load(Ordering::Relaxed),
        lll_iters_max: N_LLL_ITERS_MAX.load(Ordering::Relaxed),
        lll_at_cap: N_LLL_AT_CAP.load(Ordering::Relaxed),
        lazy_passes_total: N_LAZY_PASSES_TOTAL.load(Ordering::Relaxed),
        lazy_calls_total: N_LAZY_CALLS_TOTAL.load(Ordering::Relaxed),
        lazy_passes_max: N_LAZY_PASSES_MAX.load(Ordering::Relaxed),
    }
}
