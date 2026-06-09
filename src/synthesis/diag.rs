//! Optional per-search diagnostic counters, gated by `CYCLOSYNTH_TRACE=1`.
//!
//! Usage:
//!   CYCLOSYNTH_TRACE=1 ./time_synthesis ...
//!
//! Output is printed to stderr (so it doesn't pollute timing tables on stdout).
//! The diagnostic boundary is one `dc_search` call: counters are reset at the
//! start and dumped at the end, showing per-lde where time and prefix count
//! went.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

static TRACE_ENABLED: OnceLock<bool> = OnceLock::new();

/// Diagnostic-only: capture the raw integer x at the moment a should_stop
/// check returns true inside the SE walk. Set when `CYCLOSYNTH_CAPTURE=1`
/// and `dc_search_q_mpfr`'s should_stop fires. Read by diagnostic probes
/// to do cap-membership / region-mismatch tests.
#[derive(Clone, Debug)]
pub struct CapturedFind {
    pub x_inner: [i64; 16],
    pub k_inner: u32,
    pub k_total: u32,
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

/// Watchdog: when set to a target z, the SE walk records every prune firing
/// where the SE path's z[d..16] matches the watched target[d..16]. Used by
/// the prune-mechanism diagnostic to find which prune (if any) rejects a
/// known-good lattice point's enumeration path.
pub static WATCH_Z_TARGET: Mutex<Option<[i64; 16]>> = Mutex::new(None);

/// Lock-free fast-path gate. Cleared (false) means `watch_path_match_at_depth`
/// returns immediately without touching the mutex, so the watchdog is free
/// for runs that don't arm it (e.g. the capture phase of
/// probe_prune_oracle that records x_target without watching).
pub static WATCH_ARMED: AtomicBool = AtomicBool::new(false);

pub fn watch_arm(target: [i64; 16]) {
    if let Ok(mut guard) = WATCH_Z_TARGET.lock() {
        *guard = Some(target);
    }
    WATCH_ARMED.store(true, Ordering::Release);
}

pub fn watch_disarm() {
    WATCH_ARMED.store(false, Ordering::Release);
    if let Ok(mut guard) = WATCH_Z_TARGET.lock() {
        *guard = None;
    }
}

#[derive(Clone, Debug)]
pub struct WatchHit {
    pub depth: i32,
    pub z_at_prune: [i64; 16],
    pub partial_eucl_f64: f64,
    pub threshold: f64,
    pub partial_q_f64: f64,
    pub r_eucl_diag_d: f64,
    pub w_d: f64,
}

pub static WATCH_HITS: Mutex<Vec<WatchHit>> = Mutex::new(Vec::new());

#[inline]
pub fn watch_path_match_at_depth(z: &[i64; 16], depth: i32) -> bool {
    if !WATCH_ARMED.load(Ordering::Acquire) {
        return false;
    }
    if let Ok(guard) = WATCH_Z_TARGET.lock() {
        if let Some(target) = guard.as_ref() {
            let d = depth.max(0) as usize;
            for j in d..16 {
                if z[j] != target[j] { return false; }
            }
            return true;
        }
    }
    false
}

pub fn watch_record(hit: WatchHit) {
    if let Ok(mut v) = WATCH_HITS.lock() {
        v.push(hit);
    }
}

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

// ─── 16D Z[ζ_16] / Clifford+√T-specific counters ─────────────────────────────

/// Number of `phase1` invocations across this synthesize call (one per k).
pub static N_PHASE1_CALLS: AtomicU64 = AtomicU64::new(0);
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
/// SE leaves passing all filters (returned by `phase1`).
pub static N_SOLS_RETURNED: AtomicU64 = AtomicU64::new(0);
/// Time spent inside SE leaf-check closures (sum across all phase1 calls).
pub static T_LEAF_CHECK_NS: AtomicU64 = AtomicU64::new(0);

/// Total norm-shell prune firings across the SE walk.
pub static N_PRUNE_FIRES: AtomicU64 = AtomicU64::new(0);
/// Prune firings whose `new_partial_eucl` is within 10% above the threshold
/// (i.e., `1.0 ≤ ratio ≤ 1.10`). These are the borderline cases where a
/// numerically-imprecise partial accumulator could be the difference
/// between firing and not-firing.
pub static N_PRUNE_FIRES_NEAR: AtomicU64 = AtomicU64::new(0);
/// Prune firings within 1% of the threshold (super-borderline).
pub static N_PRUNE_FIRES_VERY_NEAR: AtomicU64 = AtomicU64::new(0);
/// Prune firings that triggered MPFR verification (i.e., within the
/// `VERIFY_RATIO_CAP` band and verify_prune_mpfr was on).
pub static N_VERIFY_PRUNE_FIRES: AtomicU64 = AtomicU64::new(0);
/// Prune firings where MPFR verification disagreed with f64 (MPFR said keep,
/// f64 said prune). These are the false negatives the verification rescues.
pub static N_VERIFY_PRUNE_CORRECTED: AtomicU64 = AtomicU64::new(0);

/// CPU-summed nanoseconds spent inside `verify_partial_dd_exceeds`.
pub static T_VERIFY_DD_NS: AtomicU64 = AtomicU64::new(0);

// BKZ insertion branch counts (which case fires for each SVP coord vector).
pub static N_BKZ_BRANCH1: AtomicU64 = AtomicU64::new(0);
pub static N_BKZ_BRANCH2: AtomicU64 = AtomicU64::new(0);
pub static N_BKZ_BRANCH3_SUCCESS: AtomicU64 = AtomicU64::new(0);
pub static N_BKZ_BRANCH3_NONPRIMITIVE: AtomicU64 = AtomicU64::new(0);

// ─── Depth-1 shell-discriminant filter (measurement-only) ────────────────────
//
// At depth 1 with z[2..15] fixed, the shell equation ‖x‖² = T is a quadratic
// in z[0] with coefficients depending on z[1]:
//   a z[0]² + 2(G_01 z[1] + v_0) z[0] + (G_11 z[1]² + 2 v_1 z[1] + A − T) = 0
// where a = G_00 = ‖basis[0]‖², G_01 = basis[0]·basis[1], G_11 = ‖basis[1]‖²,
// v_j = y·basis[j], A = ‖y‖², y = x − z[0]_curr·basis[0] − z[1]_curr·basis[1].
//
// For an integer z[0] solution to exist, the discriminant D = b² − 4ac must
// be (a) ≥ 0 and (b) a perfect square. Counters measure how often each
// condition fires across z[1] candidates that survive the existing
// partial_eucl prune (i.e., that would otherwise recurse to depth 0).
//
// The mod-16 check is a cheap necessary condition for perfect-square: every
// perfect square is ≡ 0, 1, 4, or 9 mod 16, so D mod 16 ∉ {0,1,4,9} ⟹ not a
// square ⟹ no integer z[0]. Mod-16 rejects ~75% of random non-squares.
pub static N_QFILTER_TOTAL: AtomicU64 = AtomicU64::new(0);
pub static N_QFILTER_D_NEG: AtomicU64 = AtomicU64::new(0);
pub static N_QFILTER_D_GE0_MOD16_BAD: AtomicU64 = AtomicU64::new(0);
pub static N_QFILTER_D_GE0_NOT_SQUARE: AtomicU64 = AtomicU64::new(0);
pub static N_QFILTER_PERFECT_SQUARE: AtomicU64 = AtomicU64::new(0);
/// Wall-time accumulators for the depth-1 Q-filter (phase 3) — trace-only.
/// Used to validate microbench numbers against production cache state.
pub static T_QFILTER_PRECOMPUTE_NS: AtomicU64 = AtomicU64::new(0);
pub static T_QFILTER_CLASSIFY_NS: AtomicU64 = AtomicU64::new(0);
pub static N_QFILTER_PRECOMPUTE_CALLS: AtomicU64 = AtomicU64::new(0);
/// Nodes consumed in the first SE walk that returns a solution (= when
/// `should_stop` returns true on a leaf). Used to discriminate whether
/// filter-on regression is post-find drift, search-order disruption, or
/// per-node cost asymmetry. Recorded via compare_exchange so only the
/// first writer wins. Trace-only.
pub static N_NODES_AT_FIRST_SOLUTION: AtomicU64 = AtomicU64::new(0);

// ─── Per-depth survivorship (critic Step 1) ──────────────────────────────────
//
// Indexed by depth 0..16. recurse-call enter, prune-fire, and prune-actual
// counters at each level. Reveal where the SE tree fans out and where
// pruning actually trims.

pub static N_RECURSE_ENTER_AT_DEPTH: [AtomicU64; 16] = [
    AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
    AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
    AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
    AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
];
pub static N_PRUNE_FIRES_AT_DEPTH: [AtomicU64; 16] = [
    AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
    AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
    AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
    AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
];
pub static N_PRUNE_ACTUAL_AT_DEPTH: [AtomicU64; 16] = [
    AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
    AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
    AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
    AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
];

// ─── Distance-to-shell histogram at leaves ────────────────────────────────────
//
// For each leaf reached (i.e., passed all SE pruning), record where it lands
// in the ratio r = ‖x‖² / 2^k. Bins:
//   0: r ≤ 0.50      (far below shell)
//   1: 0.50 < r ≤ 0.90
//   2: 0.90 < r ≤ 0.99
//   3: 0.99 < r < 1.0  (just below shell)
//   4: r == 1.0        (exact shell — sols + bilin/align candidates)
//   5: 1.0 < r ≤ 1.01  (just above)
//   6: 1.01 < r ≤ 1.10
//   7: r > 1.10        (far above; should not happen if prune is right)
pub const N_SHELL_BINS: usize = 8;
pub static N_LEAF_BY_SHELL_RATIO: [AtomicU64; 8] = [
    AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
    AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
];

/// 2-D histogram: (depth-1 partial ratio p1/T, leaf shell ratio r).
/// Tells us whether r>1.10 leaves come from already-near-T at depth 1 (loose
/// at depth 1, depth-0 expansion adds little) or from arbitrary depth-1 mass
/// (depth-0 adds substantial mass). Rows = 4 depth-1 partial bins, cols = 8
/// shell-ratio bins.
pub const N_D1_BINS: usize = 4;
pub static N_LEAF_BY_D1_AND_SHELL: [[AtomicU64; 8]; 4] = [
    [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
     AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)],
    [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
     AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)],
    [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
     AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)],
    [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
     AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)],
];

thread_local! {
    /// f64 partial_eucl captured at depth-0 entry (= depth-1 outgoing partial).
    /// Read at leaf time to condition the shell-ratio histogram.
    pub static D1_PARTIAL_TLS: std::cell::Cell<f64> = const { std::cell::Cell::new(0.0) };
}

#[inline]
fn shell_bin(r: f64) -> usize {
    if r <= 0.50 { 0 }
    else if r <= 0.90 { 1 }
    else if r <= 0.99 { 2 }
    else if r < 1.00  { 3 }
    else if r == 1.00 { 4 }
    else if r <= 1.01 { 5 }
    else if r <= 1.10 { 6 }
    else { 7 }
}

#[inline]
fn d1_bin(p_over_t: f64) -> usize {
    if p_over_t < 0.5 { 0 }
    else if p_over_t < 0.9 { 1 }
    else if p_over_t < 0.99 { 2 }
    else { 3 } // 0.99..=1.0 (or above, but pruned at higher depths)
}

#[inline]
pub fn record_leaf_shell_ratio(norm_sq: i64, target: i64) {
    let r = norm_sq as f64 / target as f64;
    let s_bin = shell_bin(r);
    N_LEAF_BY_SHELL_RATIO[s_bin].fetch_add(1, Ordering::Relaxed);
    let p1 = D1_PARTIAL_TLS.with(|c| c.get());
    let p1_over_t = p1 / target as f64;
    let d_bin = d1_bin(p1_over_t);
    N_LEAF_BY_D1_AND_SHELL[d_bin][s_bin].fetch_add(1, Ordering::Relaxed);
}

/// One prune-event sample for the offline oracle audit. Captures z, depth,
/// and the f64 partial so we can recompute the MPFR oracle partial later
/// and classify true-positive vs false-negative.
#[derive(Clone, Copy, Debug)]
pub struct PruneSample {
    pub depth: i32,
    pub z: [i64; 16],
    pub f64_partial: f64,
    pub threshold: f64,
}

/// Stratified reservoir: 5 bins × 1000 samples by `f64_partial / threshold`.
/// Bin layout: [1.0,1.05), [1.05,1.5), [1.5,2.0), [2.0,5.0), [5.0,∞).
pub static SAMPLES_BIN_0: Mutex<Vec<PruneSample>> = Mutex::new(Vec::new());
pub static SAMPLES_BIN_1: Mutex<Vec<PruneSample>> = Mutex::new(Vec::new());
pub static SAMPLES_BIN_2: Mutex<Vec<PruneSample>> = Mutex::new(Vec::new());
pub static SAMPLES_BIN_3: Mutex<Vec<PruneSample>> = Mutex::new(Vec::new());
pub static SAMPLES_BIN_4: Mutex<Vec<PruneSample>> = Mutex::new(Vec::new());

pub static BIN_FULL_0: AtomicBool = AtomicBool::new(false);
pub static BIN_FULL_1: AtomicBool = AtomicBool::new(false);
pub static BIN_FULL_2: AtomicBool = AtomicBool::new(false);
pub static BIN_FULL_3: AtomicBool = AtomicBool::new(false);
pub static BIN_FULL_4: AtomicBool = AtomicBool::new(false);

pub static SAMPLE_ARMED: AtomicBool = AtomicBool::new(false);

const MAX_PER_BIN: usize = 1000;

pub fn arm_sampling() {
    for v in [&SAMPLES_BIN_0, &SAMPLES_BIN_1, &SAMPLES_BIN_2, &SAMPLES_BIN_3, &SAMPLES_BIN_4] {
        if let Ok(mut g) = v.lock() { g.clear(); }
    }
    for f in [&BIN_FULL_0, &BIN_FULL_1, &BIN_FULL_2, &BIN_FULL_3, &BIN_FULL_4] {
        f.store(false, Ordering::Relaxed);
    }
    SAMPLE_ARMED.store(true, Ordering::Release);
}

#[inline]
fn bin_of(ratio: f64) -> usize {
    if ratio < 1.05 { 0 }
    else if ratio < 1.5 { 1 }
    else if ratio < 2.0 { 2 }
    else if ratio < 5.0 { 3 }
    else { 4 }
}

#[inline]
pub fn sample_prune_event(depth: i32, z: &[i64; 16], f64_partial: f64, threshold: f64) {
    if !SAMPLE_ARMED.load(Ordering::Relaxed) { return; }
    let ratio = f64_partial / threshold;
    let bin = bin_of(ratio);
    let (full, samples) = match bin {
        0 => (&BIN_FULL_0, &SAMPLES_BIN_0),
        1 => (&BIN_FULL_1, &SAMPLES_BIN_1),
        2 => (&BIN_FULL_2, &SAMPLES_BIN_2),
        3 => (&BIN_FULL_3, &SAMPLES_BIN_3),
        _ => (&BIN_FULL_4, &SAMPLES_BIN_4),
    };
    if full.load(Ordering::Relaxed) { return; }
    if let Ok(mut v) = samples.lock() {
        if v.len() < MAX_PER_BIN {
            v.push(PruneSample { depth, z: *z, f64_partial, threshold });
            if v.len() >= MAX_PER_BIN {
                full.store(true, Ordering::Release);
            }
        }
    }
}

pub fn collect_all_samples() -> Vec<PruneSample> {
    let mut all = Vec::new();
    for v in [&SAMPLES_BIN_0, &SAMPLES_BIN_1, &SAMPLES_BIN_2, &SAMPLES_BIN_3, &SAMPLES_BIN_4] {
        if let Ok(g) = v.lock() {
            all.extend(g.iter().copied());
        }
    }
    all
}

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
        &N_PHASE1_CALLS,
        &N_LLL_F64_ESCALATIONS,
        &N_NORM_REJECTED,
        &N_BILINEAR_REJECTED,
        &N_ALIGN_REJECTED,
        &N_SOLS_RETURNED,
        &T_LEAF_CHECK_NS,
        &N_PRUNE_FIRES,
        &N_PRUNE_FIRES_NEAR,
        &N_PRUNE_FIRES_VERY_NEAR,
        &N_VERIFY_PRUNE_FIRES,
        &N_VERIFY_PRUNE_CORRECTED,
        &T_VERIFY_DD_NS,
        &N_QFILTER_TOTAL,
        &N_QFILTER_D_NEG,
        &N_QFILTER_D_GE0_MOD16_BAD,
        &N_QFILTER_D_GE0_NOT_SQUARE,
        &N_QFILTER_PERFECT_SQUARE,
        &T_QFILTER_PRECOMPUTE_NS,
        &T_QFILTER_CLASSIFY_NS,
        &N_QFILTER_PRECOMPUTE_CALLS,
        &N_NODES_AT_FIRST_SOLUTION,
    ] {
        c.store(0, Ordering::Relaxed);
    }
    for c in N_RECURSE_ENTER_AT_DEPTH.iter()
        .chain(N_PRUNE_FIRES_AT_DEPTH.iter())
        .chain(N_PRUNE_ACTUAL_AT_DEPTH.iter())
        .chain(N_LEAF_BY_SHELL_RATIO.iter())
    {
        c.store(0, Ordering::Relaxed);
    }
    for row in N_LEAF_BY_D1_AND_SHELL.iter() {
        for c in row.iter() {
            c.store(0, Ordering::Relaxed);
        }
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
    // 16D Z[ζ_16] fields.
    pub phase1_calls: u64,
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
        phase1_calls: N_PHASE1_CALLS.load(Ordering::Relaxed),
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
        "  phase1 calls:    {}    LLL iters: total={} max={} at_cap={}",
        s.phase1_calls, s.lll_iters_total, s.lll_iters_max, s.lll_at_cap,
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
