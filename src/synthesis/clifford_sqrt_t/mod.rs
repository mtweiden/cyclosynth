//! Clifford+√T synthesis backend over Z[ζ_16].
//!
//! [`SynthesizerQ`] is one of two backends behind the unified user-facing
//! [`crate::synthesis::Synthesizer`]; the other is
//! [`crate::synthesis::clifford_t::SynthesizerT`] (Clifford+T, Z[ω]). Code
//! shouldn't construct `SynthesizerQ` directly — use `Synthesizer` with
//! `sqrt_t = true`. The struct stays public so the test suite can poke at
//! it (`pub` instead of `pub(crate)`).
//!
//! ## Backend (hybrid, three modes)
//!
//! For `k ≤ BRUTE_LIMIT` (=3): brute-force enumeration via
//! [`crate::synthesis::search_zeta::enumerate_unitary_norm_shell`] — cheap exact-find
//! for small Clifford+√T targets (also the lattice pipeline's oracle).
//!
//! For larger `k`: single-shot 16D L²-LLL + Schnorr-Euchner via
//! [`crate::synthesis::lattice_zeta::find_aligned_lattice_points`] (with an optional BKZ-β
//! post-pass), plus an FGKM-prefix divide-and-conquer mode (`prefix_split_search_q`)
//! for deep `k`. Adaptive leaf budget scales exponentially in `k`;
//! reaches ε ≲ 1e-5 at k ≈ 30.
//!
//! ## Reconstruction
//!
//! Single det-phase reconstruction: `d = det_phase_of(target)` chosen
//! once, then `solution_to_u2q_with_det_phase(sol, k, d)` per candidate. Column-1
//! direction extracted directly from the target (no `/√det`
//! normalization — unlike 8D's `unitary_to_uv` — because our `d` parameter
//! in the reconstruction already absorbs the det-phase mismatch).

use crate::matrix::u2::U2Q;
use crate::rings::ZZeta;
use crate::rings::types::Int;
use crate::synthesis::cliffords::CLIFFORD_TABLE_T;
use crate::synthesis::decomposer::BlochDecomposer;
use crate::synthesis::distance::{diamond_distance_u2q_float, Mat2};
use crate::synthesis::lattice_zeta::{find_aligned_lattice_points_with_stop, find_aligned_lattice_points_mpfr, IntScratch16};
use crate::synthesis::search_zeta::{enumerate_unitary_norm_shell, uv_to_lattice_y_zeta, uv_to_lattice_y_zeta_mpfr};
use num_complex::Complex64;
use std::collections::HashMap;
use std::f64::consts::PI;
use std::sync::{Arc, LazyLock, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};

/// Result of a synthesis call: the gate string, its lde, and the diamond
/// distance achieved.
///
/// Field shape matches `crate::synthesis::clifford_t::SynthResultT` so
/// callers can swap implementations transparently after the merge.
#[derive(Debug, Clone)]
pub struct SynthResultQ {
    /// Clifford+√T gate string in the alphabet `{H, S, T, Q, X, Y, Z}`
    /// (leftmost gate = first applied; matching the rest of cyclosynth's
    /// composition convention). `None` if the gate string couldn't be
    /// extracted.
    pub gates: Option<String>,
    /// Denominator exponent of the synthesized unitary.
    pub lde: u32,
    /// Diamond distance to the target.
    pub distance: f64,
}

/// Optimality certificate from [`SynthesizerQ::synthesize_exhaustive_certified`].
///
/// `OPT ∈ [lower_half_units, upper_half_units]` is guaranteed, where
/// costs are in half-units (`2T + 7Q`). `certified_optimal` is true
/// when the interval closes: the returned circuit is provably the
/// cheapest ε-approximation over the whole gate set.
///
/// Soundness rests on: (1) one full (unbudgeted) enumeration at shell
/// `k_searched` covers every circuit with reduced lde ≤ k_searched —
/// lower-lde circuits appear as √2-scaled lattice points on the shell;
/// (2) both det-phase parity branches are searched (q ≡ d mod 2 and
/// the ζ₁₆-automorphism collapse mean two branches are complete);
/// (3) anything beyond the horizon costs ≥ `cost_lb_half_units(k+1)`
/// (verified staircase, cost_bound.rs). The certificate inherits the
/// pipeline's numeric trust boundary (f64+dd distance checks, cap
/// margin `bound_sq`), like every other result of this crate.
#[derive(Debug, Clone, Copy)]
pub struct CostCertificate {
    /// Cost of the returned circuit (half-units).
    pub upper_half_units: usize,
    /// Proven lower bound on the optimum (half-units).
    pub lower_half_units: usize,
    /// Horizon: every circuit with lde ≤ this was enumerated.
    pub k_searched: u32,
    /// `upper ≤ L(k_searched + 1)`: nothing beyond the horizon can win.
    pub certified_optimal: bool,
}

/// Clifford+√T synthesizer over `Z[ζ_16]`.
///
/// Field names match `crate::synthesis::clifford_t::SynthesizerT`'s for
/// the future merge. Defaults live in [`Self::new`].
#[derive(Clone)]
pub struct SynthesizerQ {
    /// Approximation precision in diamond distance.
    pub epsilon: f64,
    /// Maximum lde to search before giving up.
    pub max_lde: u32,
    /// Minimum lde to start searching from.
    pub min_lde: u32,
    /// FGKM-prefix divide-and-conquer split parameter; `None` = single
    /// search. Builder: [`Self::with_prefix_split_m`].
    pub prefix_split_m: Option<u32>,
    /// Allowed `(d_target − d_L) mod 16` offsets for a prefix to be
    /// processed; empty = no filter. Builder: [`Self::with_inner_det_phase_filter`].
    pub inner_det_phase_filter: Vec<u32>,
    /// f64 Gram-Schmidt state in LLL (vs MPFR). Builder: [`Self::with_f64_gs`].
    pub use_f64_gs: bool,
    /// BKZ-β post-pass block size (0 = off). Builder: [`Self::with_bkz`].
    pub bkz_block_size: u32,
    /// Number of lde levels the dc path dispatches concurrently, with a
    /// cross-LDE abort so the first finder cancels its peers. Builder:
    /// [`Self::with_parallel_lde_window`].
    pub parallel_lde_window: u32,
    /// Node count a predecessor LDE must burn without finding before the
    /// next speculative LDE launches (0 = off). Budget-triggered rather
    /// than time-based so easy targets never pay for speculation.
    /// Builder: [`Self::with_parallel_lde_trigger_nodes`].
    pub parallel_lde_trigger_nodes: u64,
    /// Enumerate all ε-close candidates and return the min-cost one
    /// (`cost = T + (q_cost_x2/2)·Q`) instead of the first hit.
    /// Builder: [`Self::with_optimize_cost`].
    pub optimize_cost: bool,
    /// m values the enum stage runs per lde (m=0 = single-search, m≥1 =
    /// D&C with that split); empty disables the sweep. Builder:
    /// [`Self::with_optimal_m_sweep`].
    pub optimal_m_sweep: Vec<u32>,
    /// Multiplier on every budget cap in optimize mode: first-hit gets an
    /// early-bail advantage that optimal-mode walkers must buy back with
    /// budget. Builder: [`Self::with_optimal_budget_multiplier`].
    pub optimal_budget_multiplier: u64,
    /// Cross-parity shared incumbent (half-units). Both branches' prefix
    /// prunes share it, and the screens poll it as a dynamic max_lde
    /// clamp (cost c̃ ⇒ lde ≤ c̃ + 1), which is what lets the parity
    /// branches run concurrently instead of serially capped.
    global_best_cost: Option<std::sync::Arc<std::sync::atomic::AtomicUsize>>,
    /// Unrotated target + ζ₃₂ power for the odd parity branch, consulted
    /// at ε ≤ 2e-8: the f64 rotated product carries ~1e-16 error — equal
    /// to the radial cap width ε² at 1e-8 — so the deep router must
    /// re-derive v in MPFR from the exact source and rotate exactly
    /// (the rotation commutes with the prefix product).
    deep_rot_src: Option<(Mat2, u32)>,
    /// Stage-2 handshake: a branch that finishes its screen first would
    /// flood the shared rayon pool with frontier tasks and starve the
    /// peer's still-running screen (~50×), so frontier dispatch waits
    /// (bounded at 4× the deadline) until both screens are done.
    my_screen_done: Option<std::sync::Arc<AtomicBool>>,
    peer_screen_done: Option<std::sync::Arc<AtomicBool>>,
    /// Extra lde levels enumerated above the first feasible one — the
    /// lde-vs-cost relationship is not monotone, so the cost minimum can
    /// sit above find-lde. Builder: [`Self::with_optimal_lde_window`].
    pub optimal_lde_window: u32,
    /// Divisor on the first-hit node caps. The optimal screen may set it
    /// > 1 ("screen-lite"): budget-truncated below-fl levels land in
    /// `screen_unclear` and are re-covered by the enum grid, so harsher
    /// screen caps risk no completeness. A screen that finds nothing
    /// anywhere retries at full budget.
    pub budget_div: u64,
    /// Per-parity-branch wall deadline (ms) for the merged enum frontier
    /// (one cost-floor-ordered stream of prefix units across all (k, m)
    /// arms); `None` = legacy per-(k, m) node-budget grid. Never applies
    /// in certify mode, which needs budget-truncation semantics.
    /// Builder: [`Self::with_optimal_deadline_ms`].
    pub optimal_deadline_ms: Option<u64>,
    /// Add m = 0 full-level tasks (the only variant that proves a level
    /// exhausted) and run the floor-driven extension. Builder:
    /// [`Self::with_certify`].
    pub certify: bool,
    /// Wall budget (ms) for the certify extension loop above the window.
    /// Builder: [`Self::with_certify_extra_ms`].
    pub certify_extra_ms: u64,
    /// Also search e^{iπ/16}·target: one parity class reaches only
    /// circuits with Q-count ≡ d(target) (mod 2) — half the pool.
    /// Builder: [`Self::with_odd_parity_branch`].
    pub odd_parity_branch: bool,
    /// Run enum tasks with an open det-phase filter (all 16 classes):
    /// the closed first-hit defaults exclude classes containing cost
    /// optima. Builder: [`Self::with_optimal_open_dr_filter`].
    pub optimal_open_dr_filter: bool,
    /// Q-gate cost weight in half-units of a T gate: cost is computed as
    /// integer `2·T + q_cost_x2·Q` so it stays exactly comparable (and
    /// CAS-able). Builder: [`Self::with_q_cost`].
    pub q_cost_x2: usize,
}

/// Smallest lde where a generic SU(2) target is reachable within ε,
/// per the Gaussian heuristic over the Minkowski-embedded Z[ζ_16]
/// lattice. We start the search 2 below this estimate so easy targets
/// land without an extra full-shell sweep.
fn lattice_lde_estimate(epsilon: f64) -> u32 {
    if !(epsilon > 0.0 && epsilon < 1.0) {
        return 0;
    }
    let raw = (-epsilon.log2()).ceil() as i32 - 3;
    raw.max(0) as u32
}

/// Default enum-stage m-sweep, A/B-tuned per ε band. m=0 was dropped
/// everywhere (6-7× slower for ≤2% cost); m=2 adds nothing above 1e-6
/// but earns its keep below. Below 1e-7 the sweep runs as SEQUENTIAL
/// per-m phases (see `run_optimal_search_certified`) — interleaved,
/// m=2's 6× prefix fan-out starves the deep m=1 units that hold the
/// decisive finds.
fn default_optimal_m_sweep(epsilon: f64) -> Vec<u32> {
    if epsilon >= 1e-6 {
        vec![1]
    } else {
        vec![1, 2]
    }
}

/// Default `inner_det_phase_filter` per m, mirroring the auto-defaults set in
/// [`SynthesizerQ::new`]: m=1 → relaxed `[0, 1, 15]`, m=2 → strict `[0]`,
/// anything else → open (no filter).
fn default_inner_det_phase_filter(m: u32) -> Vec<u32> {
    match m {
        1 => vec![0, 1, 15],
        2 => vec![0],
        _ => Vec::new(),
    }
}

/// Resource cost of a decomposed Clifford+√T gate string in half-units
/// of a T gate: `2·T + q_cost_x2·Q`. With the default `q_cost_x2 = 7`
/// this realises the `T + 3.5·Q` model from the plotting scripts while
/// staying integer (exact comparisons, atomic CAS in the prefix prune).
fn gates_cost(gates: &str, q_cost_x2: usize) -> usize {
    let (t, q) = gates_tq(gates);
    2 * t + q_cost_x2 * q
}

/// `(T_count, Q_count)` of a decomposed gate string.
fn gates_tq(gates: &str) -> (usize, usize) {
    let mut t = 0usize;
    let mut q = 0usize;
    for c in gates.chars() {
        match c {
            'T' => t += 1,
            'Q' => q += 1,
            _ => {}
        }
    }
    (t, q)
}

impl SynthesizerQ {
    /// Construct a synthesizer with ε-tuned defaults: Z1 D&C below 1e-6
    /// (single search becomes pathological at deeper ε) and BKZ-4 below
    /// 1e-7 (where the SE region is large enough to pay for the tighter
    /// Hermite factor).
    pub fn new(epsilon: f64) -> Self {
        // m=2 strict at deep ε (lde_inner coverage); m=1 relaxed at 1e-6
        // (m=2 has structural gaps at low lde); single search above.
        let (prefix_split_m, inner_det_phase_filter) = if epsilon <= 1e-7 {
            (Some(2u32), vec![0u32])
        } else if epsilon <= 1e-6 {
            (Some(1u32), vec![0u32, 1, 15])
        } else {
            (None, Vec::new())
        };
        let max_lde = if epsilon <= 1e-7 { 35 } else { 30 };
        // f64 GS needs ~46 bits at 1e-7 (fits the 52-bit mantissa); at
        // 1e-8 the requirement crosses 50 bits and LLL would mostly run
        // the f64 → MPFR-80 escalation.
        let use_f64_gs = epsilon > 1e-8;

        // At deep ε, scale [min_lde, max_lde] with log2(1/ε) to skip
        // guaranteed-empty levels and still reach the hard-target tail.
        let log2_recip = if epsilon > 0.0 && epsilon < 1.0 {
            (1.0 / epsilon).log2()
        } else { 0.0 };
        let min_lde = if epsilon <= 1e-8 {
            (0.7 * log2_recip).floor() as u32
        } else {
            0
        };
        let max_lde_override = if epsilon <= 1e-8 {
            (1.7 * log2_recip).ceil() as u32
        } else {
            max_lde
        };

        let bkz_block_size = std::env::var("CYCLOSYNTH_BKZ")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(if epsilon <= 1e-7 { 4 } else { 0 });

        // Below 2.5e-8 a no-solution lde level burns its full pass-1
        // budget before the search moves on; speculating the next lde
        // behind a consumed-nodes trigger overlaps that burn while
        // keeping likely-solution levels sequential. The 3× cap at
        // ≤1e-8: solution levels there can consume ~1× cap before
        // finding, so a 1× trigger spawns a spurious peer that dilutes
        // the find.
        let (parallel_lde_window, parallel_lde_trigger_nodes) = if epsilon < 2.5e-8 {
            let cap = pass1_prefix_leaf_cap_for(epsilon);
            let mult: u64 = if epsilon <= 1e-8 { 3 } else { 1 };
            (2, cap.saturating_mul(mult))
        } else {
            (1, 0)
        };

        Self {
            epsilon,
            min_lde,
            max_lde: max_lde_override,
            prefix_split_m,
            inner_det_phase_filter,
            use_f64_gs,
            bkz_block_size,
            parallel_lde_window,
            parallel_lde_trigger_nodes,
            // Cost-optimal by default: the user-facing objective is the
            // weighted cost, and the Clifford+T baseline floor inside
            // `synthesize_optimal` guarantees the result never costs
            // more than Clifford+T on the same target.
            optimize_cost: true,
            optimal_m_sweep: default_optimal_m_sweep(epsilon),
            optimal_budget_multiplier: 2,
            global_best_cost: None,
            deep_rot_src: None,
            my_screen_done: None,
            peer_screen_done: None,
            // Window 3 below 1e-7: the cost minimum often sits above
            // find-lde there; window 4 regresses (extra levels dilute
            // the deadline).
            optimal_lde_window: if epsilon < 1e-7 { 3 } else { 2 },
            budget_div: 1,
            // Open filters only where the cost they recover beats the
            // 3-6× enum wall they cost (audit: real optima excluded by
            // the closed first-hit filters at ε ≤ 1e-5; ~nothing above).
            optimal_open_dr_filter: epsilon <= 1e-5,
            odd_parity_branch: true,
            // ε-scaled anytime deadlines, each swept to the knee of its
            // cost/deadline curve (1e-7 cliffs at 3.0-3.5 s — the deep
            // arms' time-to-first-candidate; 1e-8 saturates near 10 s
            // under sequential phases). Certify mode ignores these by
            // construction.
            optimal_deadline_ms: if epsilon >= 1e-5 {
                Some(600)
            } else if epsilon >= 1e-6 {
                Some(1500)
            } else if epsilon >= 1e-7 {
                Some(3500)
            } else {
                Some(10_000)
            },
            certify: false,
            certify_extra_ms: 2_000,
            q_cost_x2: 7,
        }
    }

    pub fn with_max_lde(mut self, max_lde: u32) -> Self {
        self.max_lde = max_lde;
        self
    }

    pub fn with_min_lde(mut self, min_lde: u32) -> Self {
        self.min_lde = min_lde;
        self
    }

    /// Z1 prototype: enable FGKM-prefix divide-and-conquer at split parameter
    /// `m`. Splits each lattice search at lde lde_total into a length-m FGKM
    /// prefix `U_L` (enumerated from `L_m^Q`) plus an inner LLL+SE search at
    /// lde_inner = lde_total − k_prefix, then composes. Off by default.
    pub fn with_prefix_split_m(mut self, m: u32) -> Self {
        self.prefix_split_m = Some(m);
        self
    }

    /// Det-phase filter: only run the inner search for prefixes whose
    /// `d_R = (d_target − d_L) mod 16` is in the set (empty = no filter);
    /// the 16-valued analog of Clifford+T's `det_zeta_parity` check.
    /// Completeness caveat: a target's right factorization may not lie
    /// in any single d_R bucket — widening the set or iterating m covers
    /// more cases.
    pub fn with_inner_det_phase_filter(mut self, allowed_offsets: Vec<u32>) -> Self {
        self.inner_det_phase_filter = allowed_offsets;
        self
    }

    /// f64 GS state in LLL instead of MPFR. NS09's Theorem 2 doesn't
    /// cover d=16 in f64, but (per fplll's wrapper strategy) it
    /// converges and matches MPFR across our ε range, much faster.
    pub fn with_f64_gs(mut self, on: bool) -> Self {
        self.use_f64_gs = on;
        self
    }

    /// Run a BKZ-β post-pass after LLL inside `find_aligned_lattice_points_with_stop`. β=0
    /// disables (the default). β=2 is LLL-equivalent — use β≥3 to see
    /// any improvement. Empirically helpful at deep ε where the
    /// post-LLL SE region is large.
    pub fn with_bkz(mut self, block_size: u32) -> Self {
        debug_assert!(block_size == 0 || (3..=8).contains(&block_size));
        self.bkz_block_size = block_size;
        self
    }

    /// Toggle cost-optimal selection (vs first-hit). The enum-stage
    /// m-sweep stays owned by the constructor defaults; this only flips
    /// the flag.
    pub fn with_optimize_cost(mut self, on: bool) -> Self {
        self.optimize_cost = on;
        self
    }

    /// Override the Stage-2 m-sweep list (m=0 = single-search, m≥1 = D&C
    /// with that FGKM-prefix split). Empty Vec disables the m-sweep and
    /// falls back to Stage-1 behaviour (use the configured `prefix_split_m`).
    pub fn with_optimal_m_sweep(mut self, ms: Vec<u32>) -> Self {
        self.optimal_m_sweep = ms;
        self
    }

    /// Multiply every per-prefix and single-search budget cap by this
    /// when `optimize_cost` is on. Default 2. Higher values reduce the
    /// chance of budget-cap regressions but increase worst-case wall.
    pub fn with_optimal_budget_multiplier(mut self, mult: u64) -> Self {
        self.optimal_budget_multiplier = mult.max(1);
        self
    }

    /// Set the Stage-4 lde-window. 0 = strict min-lde-first (default,
    /// current behaviour). N>0 = after finding at lde `f`, also search
    /// lde `f+1..=f+N` and return the global min-cost candidate.
    pub fn with_optimal_lde_window(mut self, window: u32) -> Self {
        self.optimal_lde_window = window;
        self
    }

    /// Set (or clear) the anytime enum-stage deadline in milliseconds.
    /// See the `optimal_deadline_ms` field doc.
    pub fn with_optimal_deadline_ms(mut self, ms: Option<u64>) -> Self {
        self.optimal_deadline_ms = ms;
        self
    }

    /// Set the certify extension wall budget in milliseconds.
    pub fn with_certify_extra_ms(mut self, ms: u64) -> Self {
        self.certify_extra_ms = ms;
        self
    }

    /// Lift the enum-stage d_R det-phase filters (see the field doc).
    pub fn with_optimal_open_dr_filter(mut self, on: bool) -> Self {
        self.optimal_open_dr_filter = on;
        self
    }

    /// Set the Q-gate cost weight in T-gate units (e.g. `3.5` for the
    /// `T + 3.5·Q` model). Stored in exact half-units; weights are
    /// rounded to the nearest 0.5.
    pub fn with_q_cost(mut self, weight: f64) -> Self {
        debug_assert!(weight > 0.0 && weight.is_finite());
        self.q_cost_x2 = (2.0 * weight).round().max(1.0) as usize;
        self
    }


    /// Find a minimum-lde Clifford+√T circuit approximating `target`.
    ///
    /// Returns `None` if no circuit within `max_lde` achieves diamond
    /// distance < `epsilon`. Returns the FIRST candidate found at the
    /// smallest k that works (not necessarily √T-count optimal).
    ///
    /// **Backend**: hybrid — brute-force `enumerate_unitary_norm_shell` for `k ≤ BRUTE_LIMIT`
    /// (=3), then single-shot 16D L²-LLL + Schnorr-Euchner `find_aligned_lattice_points` (optionally
    /// BKZ-reduced) and an FGKM-prefix divide-and-conquer mode (`prefix_split_search_q`)
    /// for larger / deep k.
    pub fn synthesize(&self, target: Mat2) -> Option<SynthResultQ> {
        self.synthesize_with_unverified_levels(target, None)
    }

}

/// One in-flight walk the incumbent watcher may kill: a static cost
/// floor plus the abort flag its SE walk polls at recurse-entry.
pub(crate) struct PrefixWatch {
    pub(crate) abort: std::sync::atomic::AtomicBool,
    pub(crate) active: std::sync::atomic::AtomicBool,
    pub(crate) floor: usize,
}

/// Run `body` under a scoped incumbent watcher (shared by the two
/// cost-pruned search drivers). Every ~20 ms the watcher kills active
/// walks whose floor can no longer beat the incumbent — sound: only
/// walks whose every candidate costs ≥ the incumbent are cut — and
/// walks condemned by the driver-specific `extra_kill` condition
/// (cross-branch abort in the prefix search; the deadline in the
/// frontier, which also needs `on_extra_kill` to mark the unit's level
/// truncated). The RAII guard stops the watcher even on unwind, so
/// `thread::scope` can never join a watcher that won't exit.
pub(crate) fn with_incumbent_watcher<R: Send>(
    watches: &[PrefixWatch],
    best_cost: &std::sync::atomic::AtomicUsize,
    extra_kill: impl Fn() -> bool + Sync,
    on_extra_kill: impl Fn(usize) + Sync,
    body: impl FnOnce() -> R + Send,
) -> R {
    use std::sync::atomic::{AtomicBool, Ordering};
    let walks_done = AtomicBool::new(false);
    struct DoneGuard<'a>(&'a AtomicBool);
    impl Drop for DoneGuard<'_> {
        fn drop(&mut self) {
            self.0.store(true, Ordering::Relaxed);
        }
    }
    std::thread::scope(|wscope| {
        let _done_guard = DoneGuard(&walks_done);
        let walks_done_ref = &walks_done;
        let watches_ref = &watches;
        let extra_kill = &extra_kill;
        let on_extra_kill = &on_extra_kill;
        wscope.spawn(move || {
            while !walks_done_ref.load(Ordering::Relaxed) {
                std::thread::sleep(std::time::Duration::from_millis(20));
                let cur_best = best_cost.load(Ordering::Relaxed);
                let extra = extra_kill();
                for (i, w) in watches_ref.iter().enumerate() {
                    if !w.active.load(Ordering::Relaxed) {
                        continue;
                    }
                    if cur_best <= w.floor {
                        w.abort.store(true, Ordering::Relaxed);
                    } else if extra {
                        w.abort.store(true, Ordering::Relaxed);
                        on_extra_kill(i);
                    }
                }
            }
        });
        let r = body();
        walks_done.store(true, Ordering::Relaxed);
        r
    })
}

mod brute;
mod first_hit;
mod optimal;
mod prefix;
mod recon;

pub(crate) use brute::*;
pub(crate) use first_hit::*;
pub use prefix::{
    build_fgkm_prefix_coset_keys, build_fgkm_prefix_gate_counts,
    build_fgkm_prefix_orbits, build_fgkm_prefix_set,
};
pub(crate) use prefix::{coset_keep_mask, ZETA_COSET_DEDUP};
#[cfg(test)]
pub(crate) use prefix::{canonical_key_q, lde0_cliffords_q};
pub use recon::{
    det_phase_of, solution_to_u2q, solution_to_u2q_with_det_phase,
    unitary_to_uv_zeta,
};
pub use recon::project_det_to_zeta_coset;
#[cfg(test)]
pub(crate) use recon::zeta_16_pow;

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests;
