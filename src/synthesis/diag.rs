//! Optional per-search diagnostic counters. Most are gated by
//! `CYCLOSYNTH_TRACE=1`; the budget-truncation outcome counters and the
//! prefix branch-win telemetry are always-on (at most once per
//! find_aligned_lattice_points call, so the hot path never sees them).
//!
//! Usage:
//!   CYCLOSYNTH_TRACE=1 ./time_synthesis_omega ...
//!
//! Output is printed to stderr (so it doesn't pollute timing tables on stdout).
//! The diagnostic boundary is one `prefix_split_search` call: counters are reset at the
//! start and dumped at the end, showing per-lde where time and prefix count
//! went.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
#[cfg(feature = "trace")]
use std::sync::OnceLock;

#[cfg(feature = "trace")]
static TRACE_ENABLED: OnceLock<bool> = OnceLock::new();

/// Diagnostic-only: capture the raw integer x at the moment a should_stop
/// check returns true inside the SE walk. Set when `CYCLOSYNTH_CAPTURE=1`
/// and `prefix_split_search_q`'s should_stop fires. Read by diagnostic probes
/// to do cap-membership / region-mismatch tests.
#[derive(Clone, Debug)]
pub struct CapturedFind {
    pub x_inner: [i64; 16],
    pub lde_inner: u32,
    pub lde_total: u32,
    pub d_r: u32,
    pub d_l: u32,
}

pub static CAPTURED_FIND: Mutex<Option<CapturedFind>> = Mutex::new(None);

pub fn capture_enabled() -> bool {
    std::env::var("CYCLOSYNTH_CAPTURE").ok().as_deref() == Some("1")
}

pub fn try_capture(c: CapturedFind) {
    if let Ok(mut guard) = CAPTURED_FIND.lock() {
        if guard.is_none() {
            *guard = Some(c);
        }
    }
}


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
pub static N_PREFIXES: AtomicU64 = AtomicU64::new(0);

/// Prefixes rejected by `try_unitary_to_uv` (not in SU(2): wrong determinant or
/// odd-parity ζ̄ adjustment failed).
pub static N_UV_EXTRACT_REJECTED: AtomicU64 = AtomicU64::new(0);

/// Total Schnorr-Euchner leaf-callback invocations summed across all
/// prefixes in this prefix_split_search. Useful for spotting individual prefixes
/// whose ellipsoid is "fat" relative to alignment requirements.
pub static N_SE_CALLBACKS: AtomicU64 = AtomicU64::new(0);

/// Total 8D SE recurse-entries (true node count) summed across all find_aligned_lattice_points
/// calls since the last reset. Always accumulated (one fetch_add per
/// find_aligned_lattice_points call, not per node) — used to size the PASS1/PASS2 node caps.
pub static N_SE_NODES: AtomicU64 = AtomicU64::new(0);

/// Max 8D SE recurse-entries consumed by any single find_aligned_lattice_points call (one
/// prefix × one branch walk) since the last reset. The per-prefix node
/// caps must sit well above this on solution-bearing levels.
pub static N_SE_NODES_MAX: AtomicU64 = AtomicU64::new(0);

/// Record a per-walk node count into [`N_SE_NODES_MAX`] (relaxed cmpxchg
/// max loop; called once per find_aligned_lattice_points, not per node).
pub fn record_se_nodes_max(nodes: u64) {
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
pub static N_DIST_REJECTED: AtomicU64 = AtomicU64::new(0);

// ─── Per-phase nanosecond accumulators ───────────────────────────────────────
//
// CPU-summed nanoseconds across all find_aligned_lattice_points calls in the current prefix_split_search.
// Total ≈ wall-time × n_threads in steady state (high parallel efficiency).

pub static T_BUILD_NS: AtomicU64 = AtomicU64::new(0);
pub static T_LLL_NS: AtomicU64 = AtomicU64::new(0);
pub static T_CHOLESKY_NS: AtomicU64 = AtomicU64::new(0);
pub static T_LU_NS: AtomicU64 = AtomicU64::new(0);
pub static T_SE_NS: AtomicU64 = AtomicU64::new(0);

// ─── LLL iteration telemetry ─────────────────────────────────────────────────

/// Sum of LLL inner-loop iterations across all find_aligned_lattice_points calls in this prefix_split_search.
pub static N_LLL_ITERS_TOTAL: AtomicU64 = AtomicU64::new(0);
/// Max iter count seen by any single LLL call in this prefix_split_search.
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

// ─── 16D Z[ζ_16] / Clifford+√T-specific counters ─────────────────────────────

/// Number of `find_aligned_lattice_points` invocations across this synthesize call (one per k).
pub static N_LATTICE_SEARCH_CALLS: AtomicU64 = AtomicU64::new(0);
/// Number of times the f64 GS path detected a failure (LLL not converged
/// or non-unimodular post-LLL basis) and the precision ladder escalated
/// to MPFR. Should be 0 in our regime (ε ≥ 1e-7); becomes non-zero at
/// deep ε when f64's 52-bit mantissa runs out of headroom.
pub static N_LLL_F64_ESCALATIONS: AtomicU64 = AtomicU64::new(0);
/// SE leaves rejected by the norm-shell check (`‖x‖² == 2^k`).
pub static N_NORM_REJECTED: AtomicU64 = AtomicU64::new(0);
/// SE leaves rejected by the bilinear forms check (`B_1=B_2=B_3=0`).
pub static N_BILINEAR_REJECTED: AtomicU64 = AtomicU64::new(0);
/// SE leaves rejected by the alignment check (`(y·x)² ≥ threshold_xy`).
pub static N_ALIGN_REJECTED: AtomicU64 = AtomicU64::new(0);
/// SE leaves passing all filters (returned by `find_aligned_lattice_points`).
pub static N_SOLS_RETURNED: AtomicU64 = AtomicU64::new(0);
/// Time spent inside SE leaf-check closures (sum across all find_aligned_lattice_points calls).
pub static T_LEAF_CHECK_NS: AtomicU64 = AtomicU64::new(0);

/// Norm-shell prune firings in the SE walk (trace-gated).
pub static N_PRUNE_FIRES: AtomicU64 = AtomicU64::new(0);
/// Prune firings that triggered MPFR verification.
pub static N_VERIFY_PRUNE_FIRES: AtomicU64 = AtomicU64::new(0);
/// Prune firings where MPFR verification disagreed with f64 (MPFR said keep,
/// f64 said prune). These are the false negatives the verification rescues.
pub static N_VERIFY_PRUNE_CORRECTED: AtomicU64 = AtomicU64::new(0);

/// CPU-summed nanoseconds spent inside `verify_partial_dd_exceeds`.
pub static T_VERIFY_DD_NS: AtomicU64 = AtomicU64::new(0);



// ─── Stage walls (always-on; once per optimal synthesize call) ───────────────
//
// Wall-clock per pipeline stage, summed across parity branches and
// targets between resets. The screen wall overlaps the baseline thread
// (they share a scope); the baseline counter isolates its own span so
// the probe's [profile] line can attribute each.

pub static T_STAGE_SCREEN_NS: AtomicU64 = AtomicU64::new(0);
pub static T_STAGE_FRONTIER_NS: AtomicU64 = AtomicU64::new(0);
pub static T_STAGE_BASELINE_NS: AtomicU64 = AtomicU64::new(0);

/// One-line machine-parseable counter snapshot (`key=value` pairs).
/// The probe prints it per synthesis call as `[profile] ...`;
/// `scripts/profile_summary.py` parses it into the campaign scoreboard.
/// Timing keys are zero unless CYCLOSYNTH_TRACE=1 (phase timers are
/// trace-gated); stage walls and walk-outcome counters are always-on.
pub fn profile_line() -> String {
    let ms = |c: &AtomicU64| c.load(Ordering::Relaxed) as f64 / 1e6;
    let n = |c: &AtomicU64| c.load(Ordering::Relaxed);
    format!(
        "screen_ms={:.1} frontier_ms={:.1} baseline_ms={:.1} \
         build_ms={:.1} lll_ms={:.1} chol_ms={:.1} lu_ms={:.1} se_ms={:.1} \
         leaf_ms={:.1} verify_dd_ms={:.1} \
         lll_iters={} lll_iters_max={} lll_at_cap={} f64_escal={} \
         se_nodes={} se_cb={} search_calls={} prefixes={} \
         uv_rej={} norm_rej={} bilin_rej={} align_rej={} dist_rej={} sols={} \
         prune_fires={} verify_fires={} verify_corrected={} \
         pred_trunc={} budget_exhaust={}",
        ms(&T_STAGE_SCREEN_NS), ms(&T_STAGE_FRONTIER_NS), ms(&T_STAGE_BASELINE_NS),
        ms(&T_BUILD_NS), ms(&T_LLL_NS), ms(&T_CHOLESKY_NS), ms(&T_LU_NS), ms(&T_SE_NS),
        ms(&T_LEAF_CHECK_NS), ms(&T_VERIFY_DD_NS),
        n(&N_LLL_ITERS_TOTAL), n(&N_LLL_ITERS_MAX), n(&N_LLL_AT_CAP), n(&N_LLL_F64_ESCALATIONS),
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
// Both count once per WALK (first-flipper dedupe inside se.rs) and are
// always-on (not trace-gated): they fire at most once per find_aligned_lattice_points call, so
// the hot path never sees them, and tests assert on them without needing
// CYCLOSYNTH_TRACE.

/// Walks aborted by predictive budget truncation: the projected total node
/// spend (consumed / fraction_of_frontier_items_done, checked at BudgetCache
/// refill granularity) exceeded the walk's initial budget × margin. These
/// surface upstream exactly as budget hits — same `budget_hit` flag, same
/// ledger truncation — just without burning the remaining budget.
pub static N_PREDICTIVE_TRUNC_FIRES: AtomicU64 = AtomicU64::new(0);

/// Walks that exhausted their budget pool the plain way (burned 100% of the
/// budget before completing). Compare against `N_PREDICTIVE_TRUNC_FIRES` to
/// see how many truncations the projection reclaimed.
pub static N_BUDGET_EXHAUST_FIRES: AtomicU64 = AtomicU64::new(0);

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
pub fn record_lazy_passes(passes: u64) {
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
pub fn record_lll_iters(iters: u64, cap: u64) {
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
        &N_LLL_F64_ESCALATIONS,
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
    pub prefixes: u64,
    pub uv_extract_rejected: u64,
    pub se_callbacks: u64,
    pub se_nodes: u64,
    pub se_nodes_max: u64,
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
    // 16D Z[ζ_16] fields.
    pub lattice_search_calls: u64,
    pub norm_rejected: u64,
    pub bilinear_rejected: u64,
    pub align_rejected: u64,
    pub sols_returned: u64,
    pub t_leaf_check_ms: f64,
    pub prune_fires: u64,
    pub verify_prune_fires: u64,
    pub verify_prune_corrected: u64,
    pub t_verify_dd_ms: f64,
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

// ─── prefix_split_search branch-win telemetry ───────────────────────────────
//
// Always-on and CUMULATIVE across the process — deliberately NOT in
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
pub fn record_branch_win(odd: bool, idx: usize, len: usize, lde: u32) {
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
