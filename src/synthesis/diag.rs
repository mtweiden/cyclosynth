//! Optional per-search diagnostic counters. They require the `trace` cargo
//! feature: without it, `trace_enabled()` is a compile-time `const false` and
//! every counter write compiles out (zero cost). Built with `--features trace`,
//! the writes are then runtime-gated by `CYCLOSYNTH_TRACE=1`.
//!
//! Usage:
//!   cargo run --features trace --bin time_synthesis_omega   # then:
//!   CYCLOSYNTH_TRACE=1 ./time_synthesis_omega ...
//!
//! Output is printed to stderr (so it doesn't pollute timing tables on stdout).
//! The diagnostic boundary is one `prefix_split_search` call: counters are reset at the
//! start and dumped at the end, showing per-lde where time and prefix count
//! went.

// Telemetry values are approximate by nature; counter-to-f64 casts are display-only.
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(feature = "trace")]
use std::sync::OnceLock;

/// Elapsed nanoseconds as `u64` for the phase-timer counters (u64 ns wraps
/// after ~584 years; telemetry-only).
#[inline]
pub(crate) fn elapsed_ns(t: std::time::Instant) -> u64 {
    t.elapsed().as_nanos() as u64
}

#[cfg(all(feature = "python", feature = "trace"))]
use pyo3::prelude::*;

#[cfg(feature = "trace")]
static TRACE_ENABLED: OnceLock<bool> = OnceLock::new();

/// Whether telemetry collection is active. Compiles to a constant `false`
/// unless the `trace` feature is enabled, so every `if trace_enabled()`
/// block — the per-leaf hot path, the phase timers, the per-prefix
/// counters — is dead-code-eliminated in the default (no-telemetry) build.
#[cfg(feature = "trace")]
pub fn trace_enabled() -> bool {
    *TRACE_ENABLED.get_or_init(|| {
        std::env::var("CYCLOSYNTH_TRACE")
            .ok()
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

#[cfg(not(feature = "trace"))]
#[inline(always)]
pub const fn trace_enabled() -> bool {
    false
}

// ─── Per-lde counters (reset at the start of each prefix_split_search) ─────────────────

/// MA prefixes considered at this lde.
pub(crate) static N_PREFIXES: AtomicU64 = AtomicU64::new(0);

/// Prefixes rejected by `try_unitary_to_uv` (not in SU(2): wrong determinant or
/// odd-parity ζ̄ adjustment failed).
pub(crate) static N_UV_EXTRACT_REJECTED: AtomicU64 = AtomicU64::new(0);

/// Total Schnorr-Euchner leaf-callback invocations summed across all
/// prefixes in this prefix_split_search. Useful for spotting individual prefixes
/// whose ellipsoid is "fat" relative to alignment requirements.
pub(crate) static N_SE_CALLBACKS: AtomicU64 = AtomicU64::new(0);

/// Total 8D SE recurse-entries (true node count) summed across all find_aligned_lattice_points
/// calls since the last reset. Always accumulated (one fetch_add per
/// find_aligned_lattice_points call, not per node) — used to size the PASS1/PASS2 node caps.
pub(crate) static N_SE_NODES: AtomicU64 = AtomicU64::new(0);

/// Max 8D SE recurse-entries consumed by any single find_aligned_lattice_points call (one
/// prefix × one branch walk) since the last reset. The per-prefix node
/// caps must sit well above this on solution-bearing levels.
pub(crate) static N_SE_NODES_MAX: AtomicU64 = AtomicU64::new(0);

/// Record a per-walk node count into [`N_SE_NODES_MAX`] (relaxed cmpxchg
/// max loop; called once per find_aligned_lattice_points, not per node).
pub(crate) fn record_se_nodes_max(nodes: u64) {
    if trace_enabled() {
        let mut cur = N_SE_NODES_MAX.load(Ordering::Relaxed);
        while nodes > cur {
            match N_SE_NODES_MAX.compare_exchange_weak(
                cur, nodes, Ordering::Relaxed, Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(c) => cur = c,
            }
        }
    }
}

/// SE candidates that satisfied the integer constraints (norm shell,
/// bilinear form, alignment) but produced a unitary whose diamond
/// distance to the target exceeded ε. Should be 0 in steady state;
/// non-zero indicates a precision issue in the SE bound vs the actual
/// alignment threshold.
pub(crate) static N_DIST_REJECTED: AtomicU64 = AtomicU64::new(0);

// ─── Per-phase nanosecond accumulators ───────────────────────────────────────
//
// CPU-summed nanoseconds across all find_aligned_lattice_points calls in the current prefix_split_search.
// Total ≈ wall-time × n_threads in steady state (high parallel efficiency).

pub(crate) static T_BUILD_NS: AtomicU64 = AtomicU64::new(0);
pub(crate) static T_LLL_NS: AtomicU64 = AtomicU64::new(0);
pub(crate) static T_CHOLESKY_NS: AtomicU64 = AtomicU64::new(0);
pub(crate) static T_LU_NS: AtomicU64 = AtomicU64::new(0);
pub(crate) static T_SE_NS: AtomicU64 = AtomicU64::new(0);

// ─── LLL iteration telemetry ─────────────────────────────────────────────────

/// Sum of LLL inner-loop iterations across all find_aligned_lattice_points calls in this prefix_split_search.
pub(crate) static N_LLL_ITERS_TOTAL: AtomicU64 = AtomicU64::new(0);
/// Max iter count seen by any single LLL call in this prefix_split_search.
pub(crate) static N_LLL_ITERS_MAX: AtomicU64 = AtomicU64::new(0);
/// LLL calls that hit the safety cap (typically 10_000 iters). Should be 0;
/// non-zero indicates LLL cycling at the active precision.
pub(crate) static N_LLL_AT_CAP: AtomicU64 = AtomicU64::new(0);

/// Cumulative passes across all `lazy_size_reduce` invocations. Diagnostic
/// signal for whether the size-reduction inner loop converges in 1-2 passes
/// (per Nguyen-Stehlé 2009 expectation) or runs hot.
pub(crate) static N_LAZY_PASSES_TOTAL: AtomicU64 = AtomicU64::new(0);
/// Number of `lazy_size_reduce` invocations.
pub(crate) static N_LAZY_CALLS_TOTAL: AtomicU64 = AtomicU64::new(0);
/// Max passes ever seen in a single invocation.
pub(crate) static N_LAZY_PASSES_MAX: AtomicU64 = AtomicU64::new(0);

// ─── 16D Z[ζ_16] / Clifford+√T-specific counters ─────────────────────────────

/// Number of `find_aligned_lattice_points` invocations across this synthesize call (one per k).
pub(crate) static N_LATTICE_SEARCH_CALLS: AtomicU64 = AtomicU64::new(0);
/// SE leaves rejected by the norm-shell check (`‖x‖² == 2^k`).
pub(crate) static N_NORM_REJECTED: AtomicU64 = AtomicU64::new(0);
/// SE leaves rejected by the bilinear forms check (`B_1=B_2=B_3=0`).
pub(crate) static N_BILINEAR_REJECTED: AtomicU64 = AtomicU64::new(0);
/// SE leaves rejected by the alignment check (`(y·x)² ≥ threshold_xy`).
pub(crate) static N_ALIGN_REJECTED: AtomicU64 = AtomicU64::new(0);
/// SE leaves passing all filters (returned by `find_aligned_lattice_points`).
pub(crate) static N_SOLS_RETURNED: AtomicU64 = AtomicU64::new(0);
/// Time spent inside SE leaf-check closures (sum across all find_aligned_lattice_points calls).
pub(crate) static T_LEAF_CHECK_NS: AtomicU64 = AtomicU64::new(0);

/// Norm-shell prune firings in the SE walk (trace-gated).
pub(crate) static N_PRUNE_FIRES: AtomicU64 = AtomicU64::new(0);
/// Prune firings that triggered MPFR verification.
pub(crate) static N_VERIFY_PRUNE_FIRES: AtomicU64 = AtomicU64::new(0);
/// Prune firings where MPFR verification disagreed with f64 (MPFR said keep,
/// f64 said prune). These are the false negatives the verification rescues.
pub(crate) static N_VERIFY_PRUNE_CORRECTED: AtomicU64 = AtomicU64::new(0);

/// CPU-summed nanoseconds spent inside `verify_partial_dd_exceeds`.
pub(crate) static T_VERIFY_DD_NS: AtomicU64 = AtomicU64::new(0);



// ─── Stage walls (trace-gated; once per optimal synthesize call) ─────────────
//
// Wall-clock per pipeline stage, summed across parity branches and
// targets between resets. The screen wall overlaps the baseline thread
// (they share a scope); the baseline counter isolates its own span so
// the probe's [profile] line can attribute each.

pub(crate) static T_STAGE_SCREEN_NS: AtomicU64 = AtomicU64::new(0);
pub(crate) static T_STAGE_FRONTIER_NS: AtomicU64 = AtomicU64::new(0);
pub(crate) static T_STAGE_BASELINE_NS: AtomicU64 = AtomicU64::new(0);

/// One-line machine-parseable counter snapshot (`key=value` pairs).
/// The probe prints it per synthesis call as `[profile] ...`;
/// `scripts/profile_summary.py` parses it into the campaign scoreboard.
/// Timing keys are zero unless CYCLOSYNTH_TRACE=1 (phase timers and
/// stage walls are trace-gated, like the walk-outcome counters).
pub fn profile_line() -> String {
    let ms = |c: &AtomicU64| c.load(Ordering::Relaxed) as f64 / 1e6;
    let n = |c: &AtomicU64| c.load(Ordering::Relaxed);
    format!(
        "screen_ms={:.1} frontier_ms={:.1} baseline_ms={:.1} \
         build_ms={:.1} lll_ms={:.1} chol_ms={:.1} lu_ms={:.1} se_ms={:.1} \
         leaf_ms={:.1} verify_dd_ms={:.1} \
         lll_iters={} lll_iters_max={} lll_at_cap={} \
         se_nodes={} se_cb={} search_calls={} prefixes={} \
         uv_rej={} norm_rej={} bilin_rej={} align_rej={} dist_rej={} sols={} \
         prune_fires={} verify_fires={} verify_corrected={} \
         pred_trunc={} budget_exhaust={}",
        ms(&T_STAGE_SCREEN_NS), ms(&T_STAGE_FRONTIER_NS), ms(&T_STAGE_BASELINE_NS),
        ms(&T_BUILD_NS), ms(&T_LLL_NS), ms(&T_CHOLESKY_NS), ms(&T_LU_NS), ms(&T_SE_NS),
        ms(&T_LEAF_CHECK_NS), ms(&T_VERIFY_DD_NS),
        n(&N_LLL_ITERS_TOTAL), n(&N_LLL_ITERS_MAX), n(&N_LLL_AT_CAP),
        n(&N_SE_NODES), n(&N_SE_CALLBACKS), n(&N_LATTICE_SEARCH_CALLS), n(&N_PREFIXES),
        n(&N_UV_EXTRACT_REJECTED), n(&N_NORM_REJECTED), n(&N_BILINEAR_REJECTED),
        n(&N_ALIGN_REJECTED), n(&N_DIST_REJECTED), n(&N_SOLS_RETURNED),
        n(&N_PRUNE_FIRES), n(&N_VERIFY_PRUNE_FIRES), n(&N_VERIFY_PRUNE_CORRECTED),
        n(&N_PREDICTIVE_TRUNC_FIRES), n(&N_BUDGET_EXHAUST_FIRES),
    ) + &format!(" depth_enter=[{}]",
        N_RECURSE_ENTER_AT_DEPTH.iter()
            .map(|c| c.load(Ordering::Relaxed).to_string())
            .collect::<Vec<_>>().join(","))
}

// ─── Budget-truncation outcome counters (predictive trunc, se.rs) ────────────
//
// Count once per WALK (first-flipper dedupe in se.rs), and like every counter
// here are written only behind `trace_enabled()` — so tests reading them need
// the `trace` feature. They're off the hot path, but stay gated to keep the
// default build uniformly telemetry-free.

/// Walks aborted by predictive budget truncation: the projected total node
/// spend (consumed / fraction_of_frontier_items_done, checked at BudgetCache
/// refill granularity) exceeded the walk's initial budget × margin. These
/// surface upstream exactly as budget hits — same `budget_hit` flag, same
/// ledger truncation — just without burning the remaining budget.
pub(crate) static N_PREDICTIVE_TRUNC_FIRES: AtomicU64 = AtomicU64::new(0);

/// Walks that exhausted their budget pool the plain way (burned 100% of the
/// budget before completing). Compare against `N_PREDICTIVE_TRUNC_FIRES` to
/// see how many truncations the projection reclaimed.
pub(crate) static N_BUDGET_EXHAUST_FIRES: AtomicU64 = AtomicU64::new(0);

// ─── Per-depth survivorship ───────────────────────────────────────────────────
//
// Indexed by depth 0..16: recurse-enter count per level. Reveals where the
// SE tree fans out.

pub static N_RECURSE_ENTER_AT_DEPTH: [AtomicU64; 16] = [
    AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
    AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
    AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
    AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
];



/// Record one lazy_size_reduce invocation's pass count.
pub(crate) fn record_lazy_passes(passes: u64) {
    if !trace_enabled() {
        return;
    }
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
pub(crate) fn record_lll_iters(iters: u64, cap: u64) {
    if !trace_enabled() {
        return;
    }
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
        &N_UV_EXTRACT_REJECTED,
        &N_SE_CALLBACKS,
        &N_SE_NODES,
        &N_SE_NODES_MAX,
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
        &N_LATTICE_SEARCH_CALLS,
        &N_NORM_REJECTED,
        &N_BILINEAR_REJECTED,
        &N_ALIGN_REJECTED,
        &N_SOLS_RETURNED,
        &T_LEAF_CHECK_NS,
        &N_PRUNE_FIRES,
        &N_VERIFY_PRUNE_FIRES,
        &N_VERIFY_PRUNE_CORRECTED,
        &T_VERIFY_DD_NS,
        &N_PREDICTIVE_TRUNC_FIRES,
        &N_BUDGET_EXHAUST_FIRES,
        &T_STAGE_SCREEN_NS,
        &T_STAGE_FRONTIER_NS,
        &T_STAGE_BASELINE_NS,
    ] {
        c.store(0, Ordering::Relaxed);
    }
    for c in N_RECURSE_ENTER_AT_DEPTH.iter() {
        c.store(0, Ordering::Relaxed);
    }
}

/// Snapshot of the current counter values for printing.
pub struct Snapshot {
    pub(crate) prefixes: u64,
    pub(crate) uv_extract_rejected: u64,
    pub se_callbacks: u64,
    pub se_nodes: u64,
    pub(crate) se_nodes_max: u64,
    pub(crate) dist_rejected: u64,
    pub t_build_ms: f64,
    pub t_lll_ms: f64,
    pub t_cholesky_ms: f64,
    pub t_lu_ms: f64,
    pub t_se_ms: f64,
    pub(crate) lll_iters_total: u64,
    pub(crate) lll_iters_max: u64,
    pub(crate) lll_at_cap: u64,
    pub(crate) lazy_passes_total: u64,
    pub(crate) lazy_calls_total: u64,
    pub(crate) lazy_passes_max: u64,
    // 16D Z[ζ_16] fields.
    pub(crate) lattice_search_calls: u64,
    pub(crate) norm_rejected: u64,
    pub(crate) bilinear_rejected: u64,
    pub(crate) align_rejected: u64,
    pub(crate) sols_returned: u64,
    pub(crate) t_leaf_check_ms: f64,
    pub(crate) prune_fires: u64,
    pub(crate) verify_prune_fires: u64,
    pub(crate) verify_prune_corrected: u64,
    pub(crate) t_verify_dd_ms: f64,
}

pub fn snapshot() -> Snapshot {
    Snapshot {
        prefixes: N_PREFIXES.load(Ordering::Relaxed),
        uv_extract_rejected: N_UV_EXTRACT_REJECTED.load(Ordering::Relaxed),
        se_callbacks: N_SE_CALLBACKS.load(Ordering::Relaxed),
        se_nodes: N_SE_NODES.load(Ordering::Relaxed),
        se_nodes_max: N_SE_NODES_MAX.load(Ordering::Relaxed),
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
        lattice_search_calls: N_LATTICE_SEARCH_CALLS.load(Ordering::Relaxed),
        norm_rejected: N_NORM_REJECTED.load(Ordering::Relaxed),
        bilinear_rejected: N_BILINEAR_REJECTED.load(Ordering::Relaxed),
        align_rejected: N_ALIGN_REJECTED.load(Ordering::Relaxed),
        sols_returned: N_SOLS_RETURNED.load(Ordering::Relaxed),
        t_leaf_check_ms: T_LEAF_CHECK_NS.load(Ordering::Relaxed) as f64 / 1.0e6,
        prune_fires: N_PRUNE_FIRES.load(Ordering::Relaxed),
        verify_prune_fires: N_VERIFY_PRUNE_FIRES.load(Ordering::Relaxed),
        verify_prune_corrected: N_VERIFY_PRUNE_CORRECTED.load(Ordering::Relaxed),
        t_verify_dd_ms: T_VERIFY_DD_NS.load(Ordering::Relaxed) as f64 / 1.0e6,
    }
}

/// Emit one pass of the per-lde diagnostic block on stderr. Called at the
/// end of each `prefix_split_search` invocation when `CYCLOSYNTH_TRACE=1` is set.
#[allow(clippy::too_many_arguments)]
pub(crate) fn trace_dump_pass(
    t: u32,
    t_prime: u32,
    pass: u8,
    s: &Snapshot,
    budget_hit: bool,
    pass_ms: f64,
    found: bool,
) {
    eprintln!(
        "[trace] lde={:>2} pass{} t'={:>2} prefixes={:>6} mat_uv_rej={:>6} \
         se_cb={:>9} se_nodes={:>11} (max/walk {:>9}) dist_rej={} budget={} {:>9.1}ms result={}",
        t, pass, t_prime, s.prefixes, s.uv_extract_rejected, s.se_callbacks,
        s.se_nodes, s.se_nodes_max, s.dist_rejected, u8::from(budget_hit), pass_ms,
        if found { "FOUND" } else { "none" }
    );
    let phase_total = s.t_build_ms + s.t_lll_ms + s.t_cholesky_ms + s.t_lu_ms + s.t_se_ms;
    if phase_total > 0.0 {
        eprintln!(
            "[trace]            phase_ms (cpu-summed) build={:>7.1} lll={:>7.1} \
             chol={:>7.1} lu={:>7.1} se={:>7.1} sum={:>7.1}",
            s.t_build_ms, s.t_lll_ms, s.t_cholesky_ms, s.t_lu_ms, s.t_se_ms, phase_total
        );
        let n_lll_calls = s.prefixes.saturating_sub(s.uv_extract_rejected);
        let lll_avg = if n_lll_calls > 0 {
            s.lll_iters_total as f64 / n_lll_calls as f64
        } else {
            0.0
        };
        eprintln!(
            "[trace]            lll_iters total={} avg={:.0} max={} at_cap={} (cap=10000)",
            s.lll_iters_total, lll_avg, s.lll_iters_max, s.lll_at_cap
        );
        let lazy_avg = if s.lazy_calls_total > 0 {
            s.lazy_passes_total as f64 / s.lazy_calls_total as f64
        } else {
            0.0
        };
        eprintln!(
            "[trace]            lazy_passes total={} calls={} avg={:.2} max={}",
            s.lazy_passes_total, s.lazy_calls_total, lazy_avg, s.lazy_passes_max
        );
    }
}

// ─── prefix_split_search branch-win telemetry ───────────────────────────────
//
// Trace-gated; CUMULATIVE across the process — deliberately NOT in
// `reset_all`: they fire at most once per prefix_split_search find (cold
// path), and the per-pass reset_all would otherwise wipe them before a
// suite can aggregate. Read at end-of-run by probes; a per-win stderr
// line (trace-gated) carries the distribution.

/// prefix_split_search finds that landed in the EVEN inner branch (U_L·U_R).
pub static N_BRANCH_WIN_EVEN: AtomicU64 = AtomicU64::new(0);
/// prefix_split_search finds that landed in the ODD inner branch (U_L·U_R·T).
pub static N_BRANCH_WIN_ODD: AtomicU64 = AtomicU64::new(0);
/// Sum of winning-prefix sweep indices (mean position = sum / wins).
pub static N_WIN_PREFIX_IDX_SUM: AtomicU64 = AtomicU64::new(0);
/// Sum of sweep lengths at each win (mean fraction = idx_sum / len_sum).
pub static N_WIN_PREFIX_LEN_SUM: AtomicU64 = AtomicU64::new(0);

/// Record one prefix_split_search find. `idx` is the prefix's position in
/// the sweep order actually used, `len` the sweep length.
pub(crate) fn record_branch_win(odd: bool, idx: usize, len: usize, lde: u32) {
    if !trace_enabled() {
        return;
    }
    if odd {
        N_BRANCH_WIN_ODD.fetch_add(1, Ordering::Relaxed);
    } else {
        N_BRANCH_WIN_EVEN.fetch_add(1, Ordering::Relaxed);
    }
    N_WIN_PREFIX_IDX_SUM.fetch_add(idx as u64, Ordering::Relaxed);
    N_WIN_PREFIX_LEN_SUM.fetch_add(len as u64, Ordering::Relaxed);
    eprintln!(
        "[m2] lde={lde} branch={} prefix_idx={idx}/{len}",
        if odd { "odd" } else { "even" }
    );
}

/// Pretty-print 16D synthesis profile to stderr. Call after a full
/// `synthesize` run with `CYCLOSYNTH_TRACE=1`.
pub fn dump_zeta(s: &Snapshot, label: &str) {
    let total_se = s.norm_rejected + s.bilinear_rejected + s.align_rejected + s.sols_returned;
    let pct = |x: u64| -> f64 {
        if total_se > 0 { 100.0 * x as f64 / total_se as f64 } else { 0.0 }
    };
    let stage_total =
        s.t_build_ms + s.t_lll_ms + s.t_cholesky_ms + s.t_lu_ms + s.t_se_ms;
    let pct_t = |x: f64| -> f64 {
        if stage_total > 0.0 { 100.0 * x / stage_total } else { 0.0 }
    };
    eprintln!("─── [zeta diag {label}] ───────────────────────────────");
    eprintln!(
        "  find_aligned_lattice_points calls:    {}    LLL iters: total={} max={} at_cap={}",
        s.lattice_search_calls, s.lll_iters_total, s.lll_iters_max, s.lll_at_cap,
    );
    eprintln!(
        "  SE leaves:       {} ({:.1}M)    leaf-check time: {:.1} ms",
        s.se_callbacks,
        s.se_callbacks as f64 / 1.0e6,
        s.t_leaf_check_ms,
    );
    eprintln!(
        "    norm_reject:   {:>13}  ({:>5.1}%)",
        s.norm_rejected, pct(s.norm_rejected),
    );
    eprintln!(
        "    bilin_reject:  {:>13}  ({:>5.1}%)",
        s.bilinear_rejected, pct(s.bilinear_rejected),
    );
    eprintln!(
        "    align_reject:  {:>13}  ({:>5.1}%)",
        s.align_rejected, pct(s.align_rejected),
    );
    eprintln!(
        "    sols:          {:>13}  ({:>5.1}%)",
        s.sols_returned, pct(s.sols_returned),
    );
    eprintln!("  per-stage time (ms):");
    eprintln!("    build_q:   {:>10.1}  ({:>5.1}%)", s.t_build_ms, pct_t(s.t_build_ms));
    eprintln!("    lll:       {:>10.1}  ({:>5.1}%)", s.t_lll_ms, pct_t(s.t_lll_ms));
    eprintln!("    cholesky:  {:>10.1}  ({:>5.1}%)", s.t_cholesky_ms, pct_t(s.t_cholesky_ms));
    eprintln!("    lu_solve:  {:>10.1}  ({:>5.1}%)", s.t_lu_ms, pct_t(s.t_lu_ms));
    eprintln!("    se_walk:   {:>10.1}  ({:>5.1}%)  (incl. leaf checks)", s.t_se_ms, pct_t(s.t_se_ms));
    if s.prune_fires > 0 {
        eprintln!(
            "  norm-prune fires: {}  dd-verified: {}  corrected: {}  dd time: {:.1} ms",
            s.prune_fires, s.verify_prune_fires, s.verify_prune_corrected, s.t_verify_dd_ms,
        );
    }
    eprintln!("─────────────────────────────────────────────────────");
}

// ─── trace+python diagnostic pyfunctions ─────────────────────────────────────

/// Diagnostic (trace+python): decompose a Clifford+T gate word to the
/// BlochDecomposer canonical gate string; used to cross-check exact-synthesis
/// T-count.
#[cfg(all(feature = "python", feature = "trace"))]
#[pyfunction]
pub(crate) fn decompose_gates_t(gates: &str) -> String {
    use crate::matrix::U2T;
    let mut u = U2T::eye();
    for ch in gates.chars() {
        let g = match ch {
            'H' => U2T::h(),
            'S' => U2T::s(),
            'T' => U2T::t(),
            'X' => U2T::x(),
            'Y' => U2T::y(),
            'Z' => U2T::z(),
            _ => continue,
        };
        u = u * g;
    }
    crate::synthesis::decomposer::BlochDecomposer.decompose(&u)
}

/// Diagnostic (trace+python): trace the D&C decomposition of gate word `x*` at
/// split `t'` (U_L = BlochDecomposer-canonical t'-T prefix). Returns
/// (T(U_L), k(x*), T(U_R_even), k(U_R_even), T(U_R_odd), k(U_R_odd)).
#[cfg(all(feature = "python", feature = "trace"))]
#[pyfunction]
pub(crate) fn trace_inner(gates: &str, t_prime: u32) -> (u32, u32, u32, u32, u32, u32) {
    use crate::matrix::U2T;
    let g_of = |ch: char| -> Option<U2T> {
        match ch {
            'H' => Some(U2T::h()), 'S' => Some(U2T::s()), 'T' => Some(U2T::t()),
            'X' => Some(U2T::x()), 'Y' => Some(U2T::y()), 'Z' => Some(U2T::z()),
            's' => Some(U2T::s().dagger()), 't' => Some(U2T::t().dagger()),
            _ => None,
        }
    };
    let count_t = |s: &str| s.chars().filter(|&c| c == 'T' || c == 't').count() as u32;

    let mut x = U2T::eye();
    for ch in gates.chars() {
        if let Some(g) = g_of(ch) { x = x * g; }
    }
    let x = x.reduced();
    let canon = crate::synthesis::decomposer::BlochDecomposer.decompose(&x);

    // U_L = canonical prefix up to and including the t'-th T-gate.
    let mut u_l = U2T::eye();
    let mut tc = 0u32;
    for ch in canon.chars() {
        if let Some(g) = g_of(ch) {
            u_l = u_l * g;
            if ch == 'T' || ch == 't' {
                tc += 1;
                if tc >= t_prime { break; }
            }
        }
    }
    let u_r_even = (u_l.dagger() * x).reduced();
    let u_r_odd = (u_r_even * U2T::t().dagger()).reduced();
    let t_re = count_t(&crate::synthesis::decomposer::BlochDecomposer.decompose(&u_r_even));
    let t_ro = count_t(&crate::synthesis::decomposer::BlochDecomposer.decompose(&u_r_odd));
    (tc, x.k, t_re, u_r_even.k, t_ro, u_r_odd.k)
}

/// Diagnostic (trace+python): for a known D&C solution (gate word `x*`, split
/// `t_prime`, inner shell `k`, tolerance `eps`, target Rz((a_num/a_den)·π)),
/// rebuild the inner factor's lattice vector and cap as the production search
/// does and return a human-readable multi-line report of the decisive numbers.
#[cfg(all(feature = "python", feature = "trace"))]
#[pyfunction]
pub(crate) fn diag_inner_cap(
    gates: &str,
    t_prime: u32,
    k: u32,
    eps: f64,
    a_num: i64,
    a_den: i64,
) -> String {
    use crate::matrix::U2T;
    use crate::rings::MpFloat;
    use rug::Assign;
    use crate::synthesis::angle::{su2_col_mpfr, Angle};
    use crate::synthesis::clifford_t::solution_to_u2t;
    use crate::synthesis::decomposer::BlochDecomposer;
    use crate::synthesis::lattice::omega::brute::apply_u2t_dag_to_uv_mpfr;
    use crate::synthesis::lattice::omega::cholesky_lu::{
        cholesky_f64, cholesky_int, lu_solve_int_inplace, snapshot_gram_to_mpfr,
    };
    use crate::synthesis::lattice::omega::lll::lll_l2;
    use crate::synthesis::lattice::omega::q_metric::{build_q_int, build_q_mpfr_y, uv_to_lattice_y_mpfr};
    use crate::synthesis::lattice::omega::scratch::IntScratch;
    use crate::synthesis::lattice::omega::se::{bilinear_b, reconstruct_x, SE_PREC};

    let mut out = String::new();
    macro_rules! p { ($($t:tt)*) => {{ out.push_str(&format!($($t)*)); out.push('\n'); }} }

    let g_of = |ch: char| -> Option<U2T> {
        match ch {
            'H' => Some(U2T::h()), 'S' => Some(U2T::s()), 'T' => Some(U2T::t()),
            'X' => Some(U2T::x()), 'Y' => Some(U2T::y()), 'Z' => Some(U2T::z()),
            's' => Some(U2T::s().dagger()), 't' => Some(U2T::t().dagger()),
            _ => None,
        }
    };

    // ── x* and canonical t'-prefix U_L (mirror of trace_inner) ─────────────
    let mut x = U2T::eye();
    for ch in gates.chars() {
        if let Some(g) = g_of(ch) { x = x * g; }
    }
    let x = x.reduced();
    let canon = BlochDecomposer.decompose(&x);
    let mut u_l = U2T::eye();
    let mut tc = 0u32;
    for ch in canon.chars() {
        if let Some(g) = g_of(ch) {
            u_l = u_l * g;
            if ch == 'T' || ch == 't' {
                tc += 1;
                if tc >= t_prime { break; }
            }
        }
    }
    let u_r = (u_l.dagger() * x).reduced();
    p!("== diag_inner_cap  t'={t_prime} k={k} eps={eps:e}  target=Rz({a_num}/{a_den}·π) ==");
    p!("x*.k = {}   U_L.T = {}   U_R.k = {}", x.k, tc, u_r.k);

    // ── x_R = 8 integer coeffs (solution_to_u2t/reconstruct_x ordering) ──
    let x_r: [i64; 8] = [
        u_r.u11.a.as_i64(), u_r.u11.b.as_i64(), u_r.u11.c.as_i64(), u_r.u11.d.as_i64(),
        u_r.u21.a.as_i64(), u_r.u21.b.as_i64(), u_r.u21.c.as_i64(), u_r.u21.d.as_i64(),
    ];
    let roundtrip = solution_to_u2t(&x_r, k) == u_r;
    p!("(1) x_R = {x_r:?}");
    p!("    round-trip solution_to_u2t(x_R,{k}) == U_R : {roundtrip}   (U_R.k=={k}: {})", u_r.k == k);

    // ── (2) norm shell, (3) bilinear form ──────────────────────────────────
    let norm_sq: i128 = x_r.iter().map(|&v| (v as i128) * (v as i128)).sum();
    let target_norm: i128 = 1i128 << k;
    let bil = bilinear_b(&x_r);
    p!("(2) ‖x_R‖² = {norm_sq}   target 2^{k} = {target_norm}   on-shell: {}", norm_sq == target_norm);
    p!("(3) bilinear_b(x_R) = {bil}   (must be 0: {})", bil == 0);

    // ── alignment vector (even branch), exactly as production ───────────────
    let col = su2_col_mpfr(
        Angle::PiRatio(a_num, a_den), Angle::PiRatio(0, 1), Angle::PiRatio(0, 1), 384,
    );
    let mut s = IntScratch::new(eps);
    s.reset_basis();
    let prec_q = s.prec_q;
    let v_inner_mpfr = apply_u2t_dag_to_uv_mpfr(&u_l, &col, prec_q);
    let y_q = uv_to_lattice_y_mpfr(&v_inner_mpfr, k, prec_q);

    // ── (4) alignment acceptance test (MPFR-128, matches production) ────────
    let prec = SE_PREC;
    let y_mpfr: [MpFloat; 8] = std::array::from_fn(|i| MpFloat::with_val(prec, &y_q[i]));
    let two_to_2k = MpFloat::with_val(prec, 1.0) << (2 * k);
    let eps_rf = MpFloat::with_val(prec, eps);
    let one_minus_eps_sq = MpFloat::with_val(prec, 1.0) - eps_rf.clone() * &eps_rf;
    let threshold_xy = MpFloat::with_val(prec, &two_to_2k * &one_minus_eps_sq) / 4u32;
    let mut dot = MpFloat::with_val(prec, 0.0);
    for i in 0..8 {
        let mut t = MpFloat::with_val(prec, x_r[i]);
        t *= &y_mpfr[i];
        dot += &t;
    }
    let dot_sq = MpFloat::with_val(prec, &dot * &dot);
    let dot_ratio = MpFloat::with_val(prec, &dot_sq / &threshold_xy).to_f64();
    p!("(4) dot = Σx_R·y_q = {:.9e}   dot² = {:.9e}", dot.to_f64(), dot_sq.to_f64());
    p!("    threshold_xy = 2^(2k)(1−ε²)/4 = {:.9e}   dot²/thresh = {:.12}  (aligned: {})",
        threshold_xy.to_f64(), dot_ratio, dot_sq >= threshold_xy);

    // ── ALIGNED REPRESENTATIVE ─────────────────────────────────────────────
    // Alignment accepts ±x, but the cap center is +y; use −x_R when dot<0.
    let aligned_is_neg = dot < MpFloat::with_val(prec, 0.0);
    let x_a: [i64; 8] = if aligned_is_neg { x_r.map(|v| -v) } else { x_r };
    p!("    dot sign: {}  → SE-reachable representative x_a = {}x_R",
        if aligned_is_neg { "NEGATIVE (x_R anti-aligned with cap center +y)" } else { "positive" },
        if aligned_is_neg { "−" } else { "+" });

    // helper: TRUE Q-dist (x−c)ᵀ Q (x−c) via MPFR q_mpfr/c (built below).
    let true_q_of = |s: &IntScratch, x: &[i64; 8]| -> MpFloat {
        let mut acc = MpFloat::with_val(prec_q, 0.0);
        for a in 0..8 {
            for b in 0..8 {
                let da = MpFloat::with_val(prec_q, x[a]) - &s.c[a];
                let db = MpFloat::with_val(prec_q, x[b]) - &s.c[b];
                acc += MpFloat::with_val(prec_q, da * db) * &s.q_mpfr[a][b];
            }
        }
        acc
    };

    // ── (5) TRUE Q-distance for BOTH signs via MPFR q_mpfr/c ────────────────
    build_q_mpfr_y(&mut s, &y_q, k, eps);
    let true_q_xr = true_q_of(&s, &x_r);
    let true_q = true_q_of(&s, &x_a); // the reachable representative
    p!("(5) TRUE Q-dist (MPFR q_mpfr/c):  +x_R = {:.6e}   −x_R = {:.6e}",
        true_q_xr.to_f64(), true_q_of(&s, &x_r.map(|v| -v)).to_f64());
    p!("    → reachable representative x_a: TRUE Q-dist = {:.9}", true_q.to_f64());

    // ── production LLL + f64 Cholesky + LU z_c ─────────────────────────────
    build_q_int(&mut s);
    let lll_res = lll_l2(&mut s);
    let basis = s.basis;
    let chol_ok = cholesky_f64(&mut s);
    for i in 0..8 {
        for j in 0..8 {
            s.lu_a[i][j].assign(basis[j][i] as f64);
        }
        let ci = s.c[i].clone();
        s.lu_rhs[i].assign(&ci);
    }
    let lu_ok = lu_solve_int_inplace(&mut s);
    let z_c: [MpFloat; 8] = std::array::from_fn(|i| MpFloat::with_val(SE_PREC, &s.lu_x[i]));
    p!("    LLL={lll_res:?} scale_bits={} chol_f64={chol_ok} lu={lu_ok}", s.scale_bits);
    let zc_max_bits = z_c.iter()
        .map(|v| { let a = v.clone().abs().to_f64().max(1.0); a.log2() })
        .fold(0.0f64, f64::max);
    let zc_fits_i64 = z_c.iter().all(|v| v.clone().abs().to_f64() < 9.2e18);
    p!("    cap-center z_c log2|max| = {:.1} bits  (fits i64 {}, fits f64-exact-int {})",
        zc_max_bits, zc_fits_i64, zc_max_bits < 53.0);

    // z_a: exact solve Bᵀ z_a = x_a (fraction-free Bareiss, det ±1), in rug ints.
    use rug::Integer as RInt;
    let z_big: Vec<RInt> = {
        let aij = |i: usize, j: usize| RInt::from(basis[j][i]);
        let mut m: Vec<Vec<RInt>> = (0..8)
            .map(|i| {
                let mut row: Vec<RInt> = (0..8).map(|j| aij(i, j)).collect();
                row.push(RInt::from(x_a[i]));
                row
            })
            .collect();
        let mut sign = 1i32;
        let mut prev = RInt::from(1);
        for col in 0..8 {
            if m[col][col] == 0 {
                if let Some(pr) = (col + 1..8).find(|&r1| m[r1][col] != 0) {
                    m.swap(col, pr);
                    sign = -sign;
                }
            }
            for r2 in (col + 1)..8 {
                for cc in (col + 1)..9 {
                    let t1 = RInt::from(&m[col][col] * &m[r2][cc]);
                    let t2 = RInt::from(&m[r2][col] * &m[col][cc]);
                    let num = t1 - t2;
                    let (q, rem) = num.div_rem(prev.clone());
                    debug_assert!(rem == 0);
                    m[r2][cc] = q;
                }
                m[r2][col] = RInt::from(0);
            }
            prev = m[col][col].clone();
        }
        let det = RInt::from(&m[7][7] * sign);
        p!("    det(Bᵀ) = {det}");
        let mut z_big: Vec<RInt> = vec![RInt::from(0); 8];
        for r2 in (0..8).rev() {
            let mut v = m[r2][8].clone();
            for cc in (r2 + 1)..8 {
                v -= RInt::from(&m[r2][cc] * &z_big[cc]);
            }
            let (q, _rem) = v.div_rem(m[r2][r2].clone());
            z_big[r2] = q;
        }
        z_big
    };
    let zmax = z_big.iter().map(|z| z.significant_bits()).max().unwrap_or(0);
    p!("    z_a bit-width max = {zmax}");

    // diff = z_a − z_c in MPFR.
    let diff: [MpFloat; 8] = std::array::from_fn(|j| {
        let mut d = MpFloat::with_val(SE_PREC, &z_big[j]);
        d -= &z_c[j];
        d
    });

    // ── (6) f64-Cholesky ‖R(z_a − z_c)‖² (R = l_f64ᵀ, lifted to 128) ──
    let mut f64_q = MpFloat::with_val(SE_PREC, 0.0);
    for d in 0..8 {
        let mut lvl = MpFloat::with_val(SE_PREC, 0.0);
        for j in d..8 {
            let r = MpFloat::with_val(SE_PREC, s.l_f64[j][d]); // R[d][j] = l_f64[j][d]
            lvl += MpFloat::with_val(SE_PREC, &r * &diff[j]);
        }
        f64_q += MpFloat::with_val(SE_PREC, &lvl * &lvl);
    }
    p!("(6) f64-Cholesky ‖R(z_a−z_c)‖² = {:.9}   (production SE bound test uses this)", f64_q.to_f64());

    // MPFR-oracle cross-check: cholesky_int on the snapshotted post-LLL Gram.
    let g_post = snapshot_gram_to_mpfr(&s);
    let oracle_l_opt = cholesky_int(&s, &g_post);
    let oracle_ok = oracle_l_opt.is_some();
    let oracle_l = oracle_l_opt.unwrap_or_else(|| {
        std::array::from_fn(|_| std::array::from_fn(|_| MpFloat::with_val(prec_q, 0.0)))
    });
    let mut oracle_q = MpFloat::with_val(prec_q, 0.0);
    for d in 0..8 {
        let mut lvl = MpFloat::with_val(prec_q, 0.0);
        for j in d..8 {
            let r = MpFloat::with_val(prec_q, &oracle_l[j][d]);
            let mut dd = MpFloat::with_val(prec_q, &z_big[j]);
            dd -= &z_c[j];
            lvl += MpFloat::with_val(prec_q, &r * &dd);
        }
        oracle_q += MpFloat::with_val(prec_q, &lvl * &lvl);
    }
    p!("    MPFR-oracle ‖R(z_a−z_c)‖² (cholesky_int, ok={oracle_ok}) = {:.9}  (≈ TRUE cross-check)", oracle_q.to_f64());
    let _ = reconstruct_x; // (kept import; rug path used for z)

    // ── (7) se_bound + worst-diagonal f64-vs-MPFR R discrepancy ─────────────
    let bound = std::env::var("CYCLOSYNTH_SE_BOUND_8D").ok()
        .and_then(|v| v.parse::<f64>().ok()).unwrap_or(1.51);
    p!("(7) se_bound() = {bound}");
    let mut worst_rel = 0.0f64;
    let mut worst_at = (0usize, 0usize, 0.0f64, 0.0f64);
    for i in 0..8 {
        for j in 0..=i {
            let f = s.l_f64[i][j];
            let o = oracle_l[i][j].to_f64();
            let rel = (f - o).abs() / (1e-300 + o.abs());
            if rel > worst_rel { worst_rel = rel; worst_at = (i, j, f, o); }
        }
    }
    p!("    worst f64-vs-MPFR Cholesky L rel-err = {:.3e} at L[{}][{}] (f64={:.6e} mpfr={:.6e})",
        worst_rel, worst_at.0, worst_at.1, worst_at.2, worst_at.3);

    // ── verdict ────────────────────────────────────────────────────────────
    let on_shell = norm_sq == target_norm && bil == 0 && roundtrip;
    let tq = true_q.to_f64();
    let fq = f64_q.to_f64();
    let verdict = if !on_shell {
        "CONVENTION bug — x_R is not on the enumerated shell / does not round-trip"
    } else if tq > bound {
        "T2 — cap/bound miscalibrated: TRUE Q-dist exceeds se_bound at the inner shell"
    } else if fq > bound {
        "T1 — f64 Cholesky distorts the box: TRUE ≤ bound but f64-computed > bound"
    } else {
        "NEITHER — x_R is on-shell AND in-cap by both metrics (bug is elsewhere: alignment/threshold or upstream)"
    };
    p!("VERDICT: {verdict}");
    p!("  on_shell={on_shell}  TRUE_Q={tq:.6}  f64_Q={fq:.6}  bound={bound}  aligned={}",
        dot_sq >= threshold_xy);

    out
}
