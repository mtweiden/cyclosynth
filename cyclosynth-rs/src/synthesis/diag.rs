//! Optional per-search diagnostic counters, gated by `CYCLOSYNTH_TRACE=1`.
//!
//! Usage from binaries:
//!   CYCLOSYNTH_TRACE=1 ./time_synthesis ...
//!
//! Output is printed via `eprintln!` so it doesn't pollute the timing tables on
//! stdout. The diagnostic boundary is one `dc_search` call: counters are reset
//! at the start and dumped at the end, showing per-lde where the time and
//! prefix count went, how often low-prec succeeded vs escalated, and whether
//! the search exhausted prefixes or hit the budget cap.

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

// Per-lde counters — reset at the start of each dc_search.
pub static N_PREFIXES: AtomicU64 = AtomicU64::new(0);
pub static N_MAT_TO_UV_REJECTED: AtomicU64 = AtomicU64::new(0);

/// Heavy-tier low-prec attempts (entered the `low` HeavyScratch).
pub static N_LOW_ATTEMPT: AtomicU64 = AtomicU64::new(0);
/// Low-prec attempts that returned a candidate (passed all SE filters).
pub static N_LOW_FOUND: AtomicU64 = AtomicU64::new(0);
/// Low-prec attempts that signalled escalation (det/Cholesky/LU fail or SE
/// circuit breaker tripped). These are followed by a high-prec retry.
pub static N_LOW_ESCALATE: AtomicU64 = AtomicU64::new(0);
/// High-prec attempts entered (= N_LOW_ESCALATE in steady state).
pub static N_HIGH_ATTEMPT: AtomicU64 = AtomicU64::new(0);
/// High-prec attempts that returned a candidate.
pub static N_HIGH_FOUND: AtomicU64 = AtomicU64::new(0);

/// Total SE leaf-callback invocations summed across all prefixes in the
/// current dc_search. Useful for detecting individual fat-ellipsoid prefixes
/// that swamp the per-call SE_ESCALATE_THRESHOLD trip count.
pub static N_SE_CALLBACKS: AtomicU64 = AtomicU64::new(0);

/// Number of valid 8-vector solutions found by SE that *failed* the
/// final diamond-distance check (correct integer constraints, but the
/// reconstructed unitary's distance to the target exceeded ε).
pub static N_DIST_REJECTED: AtomicU64 = AtomicU64::new(0);

pub fn reset_all() {
    for c in [
        &N_PREFIXES,
        &N_MAT_TO_UV_REJECTED,
        &N_LOW_ATTEMPT,
        &N_LOW_FOUND,
        &N_LOW_ESCALATE,
        &N_HIGH_ATTEMPT,
        &N_HIGH_FOUND,
        &N_SE_CALLBACKS,
        &N_DIST_REJECTED,
    ] {
        c.store(0, Ordering::Relaxed);
    }
}

/// Snapshot of the current counter values for printing.
pub struct Snapshot {
    pub prefixes: u64,
    pub mat_to_uv_rejected: u64,
    pub low_attempt: u64,
    pub low_found: u64,
    pub low_escalate: u64,
    pub high_attempt: u64,
    pub high_found: u64,
    pub se_callbacks: u64,
    pub dist_rejected: u64,
}

pub fn snapshot() -> Snapshot {
    Snapshot {
        prefixes: N_PREFIXES.load(Ordering::Relaxed),
        mat_to_uv_rejected: N_MAT_TO_UV_REJECTED.load(Ordering::Relaxed),
        low_attempt: N_LOW_ATTEMPT.load(Ordering::Relaxed),
        low_found: N_LOW_FOUND.load(Ordering::Relaxed),
        low_escalate: N_LOW_ESCALATE.load(Ordering::Relaxed),
        high_attempt: N_HIGH_ATTEMPT.load(Ordering::Relaxed),
        high_found: N_HIGH_FOUND.load(Ordering::Relaxed),
        se_callbacks: N_SE_CALLBACKS.load(Ordering::Relaxed),
        dist_rejected: N_DIST_REJECTED.load(Ordering::Relaxed),
    }
}
