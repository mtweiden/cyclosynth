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
//! [`crate::synthesis::search_zeta::phase1_brute`] — cheap exact-find
//! for small Clifford+√T targets (also the lattice pipeline's oracle).
//!
//! For larger `k`: single-shot 16D L²-LLL + Schnorr-Euchner via
//! [`crate::synthesis::lattice_zeta::phase1`] (with an optional BKZ-β
//! post-pass), plus an FGKM-prefix divide-and-conquer mode (`dc_search_q`)
//! for deep `k`. Adaptive leaf budget scales exponentially in `k`;
//! reaches ε ≲ 1e-5 at k ≈ 30.
//!
//! ## Reconstruction
//!
//! Single det-phase reconstruction: `d = det_phase_of(target)` chosen
//! once, then `solution_to_u2q_d(sol, k, d)` per candidate. Column-1
//! direction extracted directly from the target (no `/√det`
//! normalization — unlike 8D's `unitary_to_uv` — because our `d` parameter
//! in the reconstruction already absorbs the det-phase mismatch).

use crate::matrix::u2::U2Q;
use crate::rings::ZZeta;
use crate::rings::types::Int;
use crate::synthesis::cliffords::CLIFFORD_TABLE_T;
use crate::synthesis::decomposer::BlochDecomposer;
use crate::synthesis::distance::{diamond_distance_u2q_float, Mat2};
use crate::synthesis::lattice_zeta::{phase1_with_stop, IntScratch16};
use crate::synthesis::search_zeta::{phase1_brute, uv_to_xy_zeta};
use num_complex::Complex64;
use std::collections::HashMap;
use std::f64::consts::PI;
use std::sync::{Arc, LazyLock, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};

// ─── Solution → U2Q reconstruction (Z[ζ_16] analog of solution_to_u2t) ───────

/// Build `U2Q` from a 16-element solution and denominator exponent.
///
/// Convention: `sol = [u_1.a, …, u_1.h, u_2.a, …, u_2.h]` with
/// `U = [[u_1, −u_2*], [u_2, u_1*]] / √(2^k)` (SU(2) form, det = 1).
pub fn solution_to_u2q(sol: &[i64; 16], k: u32) -> U2Q {
    solution_to_u2q_d(sol, k, 0)
}

/// `ζ_16^d` as a `ZZeta` element, for `d` in `0..16`. `ζ_16^8 = −1`, so
/// `ζ_16^(d+8) = −ζ_16^d`.
fn zeta_16_pow(d: u32) -> ZZeta {
    let d = d % 16;
    if d < 8 {
        let mut c = [0i32; 8];
        c[d as usize] = 1;
        ZZeta::from_i32(c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7])
    } else {
        -zeta_16_pow(d - 8)
    }
}

/// Build a Clifford+√T `U2Q` from a 16-element solution at lde `k` with
/// **det-phase parameter** `det_phase` in `0..16`.
///
/// The reconstructed `U2Q` has determinant `ζ_16^det_phase`. Convention:
///
/// ```text
/// U = [[u_1, ζ_16^d · (−u_2*)], [u_2, ζ_16^d · u_1*]] / √(2^k)
/// ```
///
/// For `d = 0` this matches [`solution_to_u2q`] (SU(2) form). For `d ≠ 0`
/// the second column is rotated by `ζ_16^d`, making `U` reach Clifford+√T
/// products with non-unit determinant (e.g. circuits containing an odd
/// number of Q gates).
pub fn solution_to_u2q_d(sol: &[i64; 16], k: u32, det_phase: u32) -> U2Q {
    let mk = |s: &[i64]| ZZeta::new(
        Int::from_i64(s[0]), Int::from_i64(s[1]), Int::from_i64(s[2]), Int::from_i64(s[3]),
        Int::from_i64(s[4]), Int::from_i64(s[5]), Int::from_i64(s[6]), Int::from_i64(s[7]),
    );
    let u1 = mk(&sol[0..8]);
    let u2 = mk(&sol[8..16]);
    let phase = zeta_16_pow(det_phase);
    U2Q::new(u1, phase * (-u2.conj()), u2, phase * u1.conj(), k)
}

/// Determine the det-phase `d ∈ {0..15}` of a target matrix V — the
/// integer such that `ζ_16^d` is closest to `det(V)` on the unit circle.
///
/// Z[ζ_16] analog of [`super::synthesizer`]'s `det_zeta_parity` (which
/// returns just a parity bit for Z[ω]).
pub fn det_phase_of(target: &Mat2) -> u32 {
    let det = target[0][0] * target[1][1] - target[0][1] * target[1][0];
    let arg = det.arg();
    let d_float = arg * 16.0 / (2.0 * PI);
    let d_int = d_float.round() as i32;
    (((d_int % 16) + 16) % 16) as u32
}

// ─── FGKM canonical-form prefix generation (Z1, syllable-count enumeration) ──
//
// Mirrors `clifford_t::build_l`. Where Clifford+T enumerates Matsumoto–Amano
// words `T^{a₀} · ∏ (HS^bᵢ T) · C` of T-count t', this enumerates
// Forest–Gosset–Kliuchnikov–McKinnon words `∏ R_{pᵢ}(aᵢπ/8) · C` of
// **syllable count** m. A "syllable" is one `R_p(a·π/8)` with
// `p ∈ {x,y,z}, a ∈ {1,2,3}`; consecutive syllables must have distinct
// axes (Lemma 3.1). Q-count = Σaᵢ ∈ [m, 3m] varies inside one m-bin —
// see `project_zeta_z1_split_coordinate.md` for why m is the right
// enumeration coordinate (each syllable peels √2-exp by ≥1, matching the
// inner-LLL+SE lde split, while Q-count does not).
//
// Raw count at m: 9 · 6^{m-1} · 24 (m ≥ 1). Post-dedup-up-to-global-phase
// the Clifford suffix mostly collapses; FGKM Theorem 4.1 says the body is
// otherwise canonical, so we expect roughly the body count `9 · 6^{m-1}`.

/// Global cache for `build_l_q` results, keyed by syllable count `m`.
static BUILD_L_Q_CACHE: LazyLock<Mutex<HashMap<u32, Arc<Vec<U2Q>>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Canonical float key for a `U2Q` matrix, invariant under global U(1)
/// phase. Mirrors `clifford_t::canonical_key`: rotates the flattened
/// matrix so the largest-magnitude entry is real-positive, then rounds to
/// 6 decimals. Used for O(n)-average dedup in `build_l_q_inner`.
fn canonical_key_q(u: &U2Q) -> [i64; 8] {
    let m = u.to_float();
    let flat = [m[0][0], m[0][1], m[1][0], m[1][1]];

    let (idx, _) = flat
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.norm_sqr().partial_cmp(&b.norm_sqr()).unwrap())
        .unwrap();
    let piv = flat[idx];

    let rot: Vec<f64> = if piv.norm() < 1e-12 {
        flat.iter().flat_map(|c| [c.re, c.im]).collect()
    } else {
        let phase = piv / piv.norm();
        flat.iter()
            .flat_map(|c| {
                let r = c / phase;
                [r.re, r.im]
            })
            .collect()
    };

    rot.iter()
        .map(|x| (x * 1_000_000.0).round() as i64)
        .collect::<Vec<_>>()
        .try_into()
        .unwrap()
}

/// Build `L_m^Q`: the FGKM canonical-form prefix set with Clifford suffix,
/// at syllable count `m`. Cached by `m` (Arc-cloned on hit).
#[allow(dead_code)]
pub fn build_l_q(m: u32) -> Arc<Vec<U2Q>> {
    {
        let cache = BUILD_L_Q_CACHE.lock().unwrap();
        if let Some(v) = cache.get(&m) {
            return Arc::clone(v);
        }
    }
    let result = Arc::new(build_l_q_inner(m));
    BUILD_L_Q_CACHE
        .lock()
        .unwrap()
        .insert(m, Arc::clone(&result));
    result
}

/// Cache for prefix gate costs (parallel to `BUILD_L_Q_CACHE`).
static BUILD_L_Q_COST_CACHE: LazyLock<Mutex<HashMap<u32, Arc<Vec<usize>>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Pre-computed `cost(U_L) = T + 3·Q` for each prefix in `build_l_q(m)`,
/// indexed parallel to that Vec. Cached forever per `m`.
///
/// Cost is the canonical [`BlochDecomposer`] decomposition's
/// `T + 3·Q`. NB: this is **not a lower bound** on `cost(U_L · U_R)` —
/// U_R can cancel parts of U_L. The number is used as a heuristic
/// ranking + prune, not a sound bound.
pub fn build_l_q_costs(m: u32) -> Arc<Vec<usize>> {
    {
        let cache = BUILD_L_Q_COST_CACHE.lock().unwrap();
        if let Some(v) = cache.get(&m) {
            return Arc::clone(v);
        }
    }
    let prefixes = build_l_q(m);
    let costs: Vec<usize> = prefixes
        .iter()
        .map(|u_l| {
            let gates = BlochDecomposer.decompose(u_l);
            gates_cost(&gates)
        })
        .collect();
    let arc = Arc::new(costs);
    BUILD_L_Q_COST_CACHE
        .lock()
        .unwrap()
        .insert(m, Arc::clone(&arc));
    arc
}

fn build_l_q_inner(m: u32) -> Vec<U2Q> {
    if m == 0 {
        return vec![U2Q::eye()];
    }

    // 9 base syllables: `R_p(a·π/8)` for p ∈ {x,y,z}, a ∈ {1,2,3}.
    // Convention matches `decomposer::canonical_candidates`:
    //   axis 0: R_x(π/8) = H · Q · H
    //   axis 1: R_y(π/8) = S · H · Q · H · S†
    //   axis 2: R_z(π/8) = Q
    let rx_base: U2Q = U2Q::h() * U2Q::q() * U2Q::h();
    let ry_base: U2Q = U2Q::s() * U2Q::h() * U2Q::q() * U2Q::h() * U2Q::s().dagger();
    let rz_base: U2Q = U2Q::q();
    let bases: [U2Q; 3] = [rx_base, ry_base, rz_base];

    // syllables[axis][a-1] = bases[axis]^a.
    let mut syllables: [[U2Q; 3]; 3] = [[U2Q::eye(); 3]; 3];
    for (axis, base) in bases.iter().enumerate() {
        let mut acc = U2Q::eye();
        for a in 0..3 {
            acc = acc * *base;
            syllables[axis][a] = acc;
        }
    }

    // Cliffords as U2Q, rebuilt from CLIFFORD_TABLE_T entry names. The
    // `(_, U2T)` field is the Z[ω] form; we discard it and re-evaluate in
    // U2Q so the embedding ZOmega → ZZeta is implicit and exact.
    let cliffords_q: Vec<U2Q> = CLIFFORD_TABLE_T
        .iter()
        .map(|(name, _)| {
            name.chars().fold(U2Q::eye(), |acc, ch| {
                acc * match ch {
                    'H' => U2Q::h(),
                    'S' => U2Q::s(),
                    'X' => U2Q::x(),
                    'Y' => U2Q::y(),
                    'Z' => U2Q::z(),
                    _ => U2Q::eye(),
                }
            })
        })
        .collect();

    // Enumerate all length-m FGKM bodies (axis-adjacency-distinct).
    let mut bodies: Vec<U2Q> = Vec::new();
    enumerate_bodies(m, 3, U2Q::eye(), &syllables, &mut bodies);

    // Append every Clifford suffix to every body.
    let mut candidates: Vec<U2Q> = Vec::with_capacity(bodies.len() * cliffords_q.len());
    for body in &bodies {
        for c in &cliffords_q {
            candidates.push(*body * *c);
        }
    }

    // Dedup up to global U(1) phase.
    let mut seen: std::collections::HashSet<[i64; 8]> = std::collections::HashSet::new();
    let mut unique: Vec<U2Q> = Vec::with_capacity(candidates.len());
    for u in candidates {
        let key = canonical_key_q(&u);
        if seen.insert(key) {
            unique.push(u);
        }
    }
    unique
}

/// Recursively enumerate length-m FGKM bodies under the
/// adjacent-axis-distinct constraint. `prev_axis = 3` is the sentinel
/// "no previous axis" — used at the first slot so all 3 axes are open.
fn enumerate_bodies(
    remaining: u32,
    prev_axis: usize,
    acc: U2Q,
    syllables: &[[U2Q; 3]; 3],
    out: &mut Vec<U2Q>,
) {
    if remaining == 0 {
        out.push(acc);
        return;
    }
    for axis in 0..3 {
        if axis == prev_axis {
            continue;
        }
        for a in 0..3 {
            let next = acc * syllables[axis][a];
            enumerate_bodies(remaining - 1, axis, next, syllables, out);
        }
    }
}


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

/// Clifford+√T synthesizer over `Z[ζ_16]`.
///
/// Field names match `crate::synthesis::clifford_t::SynthesizerT`'s
/// for the future merge. The brute-force backend caps `max_lde` at a
/// small value; LLL+SE (Phase 5b M3+) will lift this.
pub struct SynthesizerQ {
    /// Approximation precision in diamond distance.
    pub epsilon: f64,
    /// Maximum lde to search before giving up. Brute-force backend
    /// limits this to ~4 in practice.
    pub max_lde: u32,
    /// Minimum lde to start searching from.
    pub min_lde: u32,
    /// **Z1 D&C prototype**: when `Some(m)`, run the FGKM-prefix
    /// divide-and-conquer search with split parameter m at every k where
    /// k > m (so k_inner ≥ 1). Defaults to None (single-search path).
    /// Builder: [`Self::with_dc_split`].
    pub dc_split: Option<u32>,
    /// Z1 D&C det-phase filter — list of allowed `d_R` offsets (i.e., the
    /// values that `(d_target − d_L) mod 16` may take for a prefix to be
    /// processed). Empty = no filter. Builder: [`Self::with_dc_dr_filter`].
    pub dc_dr_filter: Vec<u32>,
    /// Use experimental f64 GS state in LLL. Builder: [`Self::with_f64_gs`].
    pub use_f64_gs: bool,
    /// Optional BKZ-β post-pass (0 = disable). Builder: [`Self::with_bkz`].
    pub bkz_block_size: u32,
    /// Parallel-LDE speculation window size. When ≥ 2, the dc_split path
    /// dispatches that many lde levels concurrently via rayon, with a
    /// cross-LDE abort signal so the first finder cancels its peers.
    /// Default 1 (sequential, no change). Builder:
    /// [`Self::with_parallel_lde_window`].
    pub parallel_lde_window: u32,
    /// **Budget-triggered speculation** (hardware-agnostic). Each LDE
    /// task at index i > 0 waits until the predecessor LDE has
    /// consumed at least this many search-tree nodes without finding
    /// a solution. Polled every 50ms; cross-LDE abort exits the wait
    /// immediately. Easy targets find before the predecessor reaches
    /// the threshold, so peer LDEs never launch (zero overhead).
    /// Hard targets exhaust the predicted LDE's threshold worth of
    /// search, peers spawn, rayon work-stealing balances them.
    ///
    /// Recommended starting value: 25% of estimated total cap. For
    /// ε=1e-8 with dc_pass1_cap_for=100M nodes × ~9 usable prefixes
    /// = ~900M total → threshold ≈ 225M. Default 0 (disabled).
    /// Builder: [`Self::with_parallel_lde_trigger_nodes`].
    pub parallel_lde_trigger_nodes: u64,
    /// When true, at the smallest lde that admits a solution the
    /// synthesizer enumerates **all** ε-close candidates (no early
    /// termination), decomposes each via [`BlochDecomposer`], and returns
    /// the one minimising cost `T + 3·Q`. Wall-time can grow 5-50× vs the
    /// default first-hit path. Default false. Builder:
    /// [`Self::with_optimize_cost`].
    pub optimize_cost: bool,
    /// Stage-2 m-sweep: when `optimize_cost` is on and this is non-empty,
    /// at each lde the synthesizer tries every m in this list (m=0 =
    /// single-search, m≥1 = D&C with that FGKM-prefix split). Candidates
    /// from every variant are collected and the min-cost one wins.
    /// Default `vec![]` (Stage-1 behaviour: just the configured
    /// `dc_split`). `with_optimize_cost(true)` auto-populates based on ε.
    /// Builder: [`Self::with_optimal_m_sweep`].
    pub optimal_m_sweep: Vec<u32>,
    /// Stage-2 budget multiplier: when `optimize_cost` is on, every
    /// per-prefix and single-search budget cap is multiplied by this.
    /// Counteracts the early-bail advantage first-hit gets — bigger
    /// budget means optimal-mode walkers can finish the SE region they
    /// need to find the same candidate (plus deeper enumeration).
    /// Default 4. Builder: [`Self::with_optimal_budget_multiplier`].
    pub optimal_budget_multiplier: u64,
    /// Stage-3 prefix-cost prune: in `optimize_cost` mode, sort prefixes
    /// by precomputed `cost(U_L) = T + 3·Q` ascending and skip any
    /// prefix whose own cost already exceeds the best total cost found
    /// so far. **Heuristic** — `cost(U_L · U_R)` can be lower than
    /// `cost(U_L)` when U_R cancels parts of U_L, so this can in
    /// principle miss the global minimum. Empirically it preserves the
    /// optimum on random SU(2) targets. Default true. Builder:
    /// [`Self::with_optimal_prefix_prune`].
    pub optimal_prefix_prune: bool,
    /// Stage-4 lde-window: after the m-sweep finds an ε-close candidate
    /// at lde `find_lde`, continue searching `find_lde + 1 ..= find_lde
    /// + window` and pick the global min-cost candidate across the
    /// whole window. 0 (default) = strict min-lde-first (current
    /// behaviour). Larger values can catch targets whose cost-min has
    /// a better T/Q split at lde + 1 (because Clifford+√T's
    /// lde-vs-cost relationship is not monotone). Builder:
    /// [`Self::with_optimal_lde_window`].
    pub optimal_lde_window: u32,
}

/// k cutoff: brute-force handles `k ≤ BRUTE_LIMIT`, lattice handles the rest.
/// At 3, brute tops out at ~10⁷ shell points (~100 ms).
const BRUTE_LIMIT: u32 = 3;

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

/// Default Stage-2 m-sweep for `optimize_cost` mode. Empirically m=0
/// (single-search) is **6-7× slower** than m=[1,2] for only ~0.5-2%
/// cheaper mean cost at ε ∈ [1e-5, 1e-3]; the trade is overwhelmingly
/// worth making, so we standardise on [1,2] everywhere except the
/// deepest regime.
/// * ε ≥ 1e-7: vec![1, 2]
/// * ε < 1e-7: vec![2] only (m=1 is too noisy at this depth).
fn default_optimal_m_sweep(epsilon: f64) -> Vec<u32> {
    if epsilon >= 1e-7 {
        vec![1, 2]
    } else {
        vec![2]
    }
}

/// Default `dc_dr_filter` per m, mirroring the auto-defaults set in
/// [`SynthesizerQ::new`]: m=1 → relaxed `[0, 1, 15]`, m=2 → strict `[0]`,
/// anything else → open (no filter).
fn default_dc_dr_filter(m: u32) -> Vec<u32> {
    match m {
        1 => vec![0, 1, 15],
        2 => vec![0],
        _ => Vec::new(),
    }
}

/// Resource cost of a decomposed Clifford+√T gate string: `T + 3·Q`.
/// Q (= √T) is weighted 3× under standard surface-code costing.
fn gates_cost(gates: &str) -> usize {
    let mut t = 0usize;
    let mut q = 0usize;
    for c in gates.chars() {
        match c {
            'T' => t += 1,
            'Q' => q += 1,
            _ => {}
        }
    }
    t + 3 * q
}

/// Two-pass leaf-budget strategy: pass 1 bails fast on doomed lde levels;
/// budget-hit lde levels are queued for pass 2 with a much larger cap.
/// Preserves completeness — a budget-hit lde is never skipped.
const PASS1_CAP: u64 = 100_000_000;
const PASS2_CAP: u64 = 4_000_000_000;

/// Per-prefix Z1 D&C pass-1 budget; scaled with ε since the post-LLL
/// SE region grows exponentially in k_inner.
fn dc_pass1_cap_for(epsilon: f64) -> u64 {
    if epsilon <= 1e-8 {
        100_000_000
    } else if epsilon <= 1e-7 {
        25_000_000
    } else {
        DC_PASS1_CAP
    }
}

fn dc_pass2_cap_for(epsilon: f64) -> u64 {
    if epsilon <= 1e-8 {
        500_000_000
    } else if epsilon <= 1e-7 {
        50_000_000
    } else {
        DC_PASS2_CAP
    }
}

const DC_PASS1_CAP: u64 = 5_000_000;
const DC_PASS2_CAP: u64 = 10_000_000;

/// Compute `U_L† · target` as a continuous Mat2.
/// `U_L` is exact (`U2Q`), `target` is float (`Mat2`). Mirrors the 8D
/// helper `clifford_t::u2t_dag_times_mat2` for use by the Z1 D&C path.
#[allow(dead_code)]
fn u2q_dag_times_mat2(u_l: &U2Q, target: &Mat2) -> Mat2 {
    let u_f = u_l.to_float();
    // (U_L†)[i][j] = conj(U_L[j][i])
    let ud00 = Complex64::new(u_f[0][0].re, -u_f[0][0].im);
    let ud01 = Complex64::new(u_f[1][0].re, -u_f[1][0].im);
    let ud10 = Complex64::new(u_f[0][1].re, -u_f[0][1].im);
    let ud11 = Complex64::new(u_f[1][1].re, -u_f[1][1].im);
    [
        [
            ud00 * target[0][0] + ud01 * target[1][0],
            ud00 * target[0][1] + ud01 * target[1][1],
        ],
        [
            ud10 * target[0][0] + ud11 * target[1][0],
            ud10 * target[0][1] + ud11 * target[1][1],
        ],
    ]
}

/// Column-1 of `target` as a 4-element real vector
/// `(Re V_{00}, Im V_{00}, Re V_{10}, Im V_{10})`. Used as the SU(2)-style
/// alignment direction `v` for the lattice search.
///
/// **Differs from 8D's `unitary_to_uv`**: that function divides by `√det`
/// to project to SU(2) because `solution_to_u2t` produces a fixed SU(2)
/// form. Here we leave the column unprojected and absorb the det-phase
/// mismatch via [`solution_to_u2q_d`]'s `d` parameter (set to
/// [`det_phase_of`]`(target)` at the call site). Column 1 of any 2×2
/// unitary is unit-norm by construction, so no further normalization is
/// needed.
pub fn unitary_to_uv_zeta(target: &Mat2) -> [f64; 4] {
    [target[0][0].re, target[0][0].im, target[1][0].re, target[1][0].im]
}

impl SynthesizerQ {
    /// Create a synthesizer with the given precision and sensible defaults.
    ///
    /// `min_lde = 0`: start from the trivial shell so exact small-T
    /// Clifford+√T targets (e.g. Q itself) are found immediately.
    /// Construct a synthesizer with sensible defaults. Auto-enables Z1
    /// D&C at ε ≤ 1e-6 (single search becomes pathological at deeper ε)
    /// and BKZ-4 at ε ≤ 1e-7 (where the SE region is large enough to
    /// pay for BKZ's tighter Hermite factor).
    pub fn new(epsilon: f64) -> Self {
        // ε ≤ 1e-7: m=2 strict (more k_inner coverage at deep lde).
        // ε ∈ (1e-7, 1e-6]: m=1 relaxed (avoids m=2 structural gaps at
        // low lde). ε > 1e-6: single search.
        let (dc_split, dc_dr_filter) = if epsilon <= 1e-7 {
            (Some(2u32), vec![0u32])
        } else if epsilon <= 1e-6 {
            (Some(1u32), vec![0u32, 1, 15])
        } else {
            (None, Vec::new())
        };
        let max_lde = if epsilon <= 1e-7 { 35 } else { 30 };
        // f64 GS is precision-sufficient through ε=1e-7 (~46-bit
        // requirement, 52-bit mantissa); at ε ≤ 1e-8 the requirement
        // crosses 50 bits and the LLL would spend most time in
        // f64 → MPFR-80 escalation. Skip f64 entirely there.
        let use_f64_gs = epsilon > 1e-8;

        // At ε ≤ 1e-8 typical lde lands ~22-24 with hard targets needing
        // ~28-32; scale min_lde / max_lde to skip guaranteed-empty levels
        // and reach the deep tail.
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

        let bkz_block_size = if epsilon <= 1e-7 { 4 } else { 0 };
        Self {
            epsilon,
            min_lde,
            max_lde: max_lde_override,
            dc_split,
            dc_dr_filter,
            use_f64_gs,
            bkz_block_size,
            parallel_lde_window: 1,
            parallel_lde_trigger_nodes: 0,
            optimize_cost: false,
            optimal_m_sweep: Vec::new(),
            optimal_budget_multiplier: 4,
            optimal_prefix_prune: true,
            optimal_lde_window: 0,
        }
    }

    /// Set the parallel-LDE speculation window (default 1 = sequential).
    /// See the field comment on `parallel_lde_window`.
    pub fn with_parallel_lde_window(mut self, window: u32) -> Self {
        debug_assert!(window >= 1);
        self.parallel_lde_window = window;
        self
    }

    /// Set the budget-triggered speculation threshold (default 0).
    /// See the field comment on `parallel_lde_trigger_nodes`.
    pub fn with_parallel_lde_trigger_nodes(mut self, nodes: u64) -> Self {
        self.parallel_lde_trigger_nodes = nodes;
        self
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
    /// `m`. Splits each lattice search at lde k_total into a length-m FGKM
    /// prefix `U_L` (enumerated from `L_m^Q`) plus an inner LLL+SE search at
    /// k_inner = k_total − k_prefix, then composes. Off by default.
    pub fn with_dc_split(mut self, m: u32) -> Self {
        self.dc_split = Some(m);
        self
    }

    /// Z1 det-phase filter: only run inner search for prefixes whose `d_R =
    /// (d_target − d_L) mod 16` is in the allowed-offsets set. The 16-valued
    /// analog of Clifford+T's `det_zeta_parity` check. Default
    /// (`Vec::new()` → no filter, every prefix runs). Strict
    /// SU(2)-only filter: pass `vec![0]`. Relaxed: e.g. `vec![0, 1, 15]`
    /// or `vec![0, 1, 2, 14, 15]`.
    ///
    /// **Completeness caveat:** filtering by `d_R` loses prefixes whose
    /// inner factor would have had a non-zero det phase. The right
    /// factorization for a given target may not be in any single d_R
    /// bucket; iterating m or widening the offset set covers more cases.
    pub fn with_dc_dr_filter(mut self, allowed_offsets: Vec<u32>) -> Self {
        self.dc_dr_filter = allowed_offsets;
        self
    }

    /// Use the experimental f64 GS state in LLL instead of MPFR. Theorem 2
    /// of NS09 doesn't cover d=16 in f64, but empirically (per fplll's
    /// `wrapper.cpp` strategy) it converges + matches the MPFR result
    /// across our ε range, with a per-LLL-call speedup of ~5× and
    /// end-to-end synthesis speedup of ~10× at moderate ε.
    pub fn with_f64_gs(mut self, on: bool) -> Self {
        self.use_f64_gs = on;
        self
    }

    /// Run a BKZ-β post-pass after LLL inside `phase1_with_stop`. β=0
    /// disables (the default). β=2 is LLL-equivalent — use β≥3 to see
    /// any improvement. Empirically helpful at deep ε where the
    /// post-LLL SE region is large.
    pub fn with_bkz(mut self, block_size: u32) -> Self {
        debug_assert!(block_size == 0 || (3..=8).contains(&block_size));
        self.bkz_block_size = block_size;
        self
    }

    /// Enable cost-optimal selection: enumerate every ε-close candidate
    /// at the smallest feasible lde and return the one with the lowest
    /// `T + 3·Q`. Off by default; see the `optimize_cost` field doc.
    ///
    /// When turning on, also auto-populates `optimal_m_sweep` based on
    /// ε if the user has not configured one. Shallower ε gets single-
    /// search (m=0) and m=1; deeper ε uses m=1 + m=2; very deep is
    /// m=2 only (m=0 single-search at deep lde would be far too slow).
    pub fn with_optimize_cost(mut self, on: bool) -> Self {
        self.optimize_cost = on;
        if on && self.optimal_m_sweep.is_empty() {
            self.optimal_m_sweep = default_optimal_m_sweep(self.epsilon);
        }
        self
    }

    /// Override the Stage-2 m-sweep list (m=0 = single-search, m≥1 = D&C
    /// with that FGKM-prefix split). Empty Vec disables the m-sweep and
    /// falls back to Stage-1 behaviour (use the configured `dc_split`).
    pub fn with_optimal_m_sweep(mut self, ms: Vec<u32>) -> Self {
        self.optimal_m_sweep = ms;
        self
    }

    /// Multiply every per-prefix and single-search budget cap by this
    /// when `optimize_cost` is on. Default 4. Higher values reduce the
    /// chance of budget-cap regressions but increase worst-case wall.
    pub fn with_optimal_budget_multiplier(mut self, mult: u64) -> Self {
        self.optimal_budget_multiplier = mult.max(1);
        self
    }

    /// Toggle the Stage-3 prefix-cost heuristic prune. Off → enumerate
    /// every (filtered) prefix; on → skip prefixes whose own decomposed
    /// cost already exceeds the best total found so far. See the
    /// `optimal_prefix_prune` field for the soundness caveat.
    pub fn with_optimal_prefix_prune(mut self, on: bool) -> Self {
        self.optimal_prefix_prune = on;
        self
    }

    /// Set the Stage-4 lde-window. 0 = strict min-lde-first (default,
    /// current behaviour). N>0 = after finding at lde `f`, also search
    /// lde `f+1..=f+N` and return the global min-cost candidate.
    pub fn with_optimal_lde_window(mut self, window: u32) -> Self {
        self.optimal_lde_window = window;
        self
    }


    /// Find a minimum-lde Clifford+√T circuit approximating `target`.
    ///
    /// Returns `None` if no circuit within `max_lde` achieves diamond
    /// distance < `epsilon`. Returns the FIRST candidate found at the
    /// smallest k that works (not necessarily √T-count optimal).
    ///
    /// **Backend**: hybrid — brute-force `phase1_brute` for `k ≤ BRUTE_LIMIT`
    /// (=3), then single-shot 16D L²-LLL + Schnorr-Euchner `phase1` (optionally
    /// BKZ-reduced) and an FGKM-prefix divide-and-conquer mode (`dc_search_q`)
    /// for larger / deep k.
    pub fn synthesize(&self, target: Mat2) -> Option<SynthResultQ> {
        use crate::synthesis::diag;
        use crate::synthesis::lattice_zeta::{set_verify_prune_mpfr, verify_prune_mpfr};
        let trace = diag::trace_enabled();
        if trace {
            diag::reset_all();
        }

        // Auto-enable MPFR prune verification in the cliff regime. At
        // ε ≤ ~1.5e-8 the f64 partial-Euclidean prune in the SE walk
        // suffers catastrophic cancellation (oracle-measured FN ratio up
        // to ~3.8×), causing silent false-negatives that drop valid
        // synthesis candidates. We turn verify on for ε < 2e-8 (a safe
        // margin above the audited cliff) and restore the prior global
        // flag value on exit so other paths aren't affected.
        let verify_was_on = verify_prune_mpfr();
        let need_verify = self.epsilon < 2e-8;
        if need_verify && !verify_was_on {
            set_verify_prune_mpfr(true);
        }
        // RAII guard so we restore even on early returns / panics.
        struct VerifyGuard {
            restore_to: bool,
            changed: bool,
        }
        impl Drop for VerifyGuard {
            fn drop(&mut self) {
                if self.changed {
                    crate::synthesis::lattice_zeta::set_verify_prune_mpfr(self.restore_to);
                }
            }
        }
        let _verify_guard = VerifyGuard {
            restore_to: verify_was_on,
            changed: need_verify && !verify_was_on,
        };

        let d = det_phase_of(&target);
        let v = unitary_to_uv_zeta(&target);

        // Lattice scratch is allocated lazily on first lattice call.
        let mut scratch: Option<Box<IntScratch16>> = None;

        let lattice_start = lattice_lde_estimate(self.epsilon)
            .saturating_sub(2)
            .max(BRUTE_LIMIT + 1)
            .max(self.min_lde);

        // `should_stop` runs only on leaves passing the integer-exact
        // filter (typically a handful per call) and short-circuits the
        // walker once a candidate's diamond distance is below ε.
        // In `optimize_cost` mode we return false unconditionally so the
        // walker enumerates *every* ε-close leaf at this k; check_sols
        // then picks the min-cost candidate.
        let epsilon = self.epsilon;
        let use_f64_gs = self.use_f64_gs;
        let bkz_block_size = self.bkz_block_size;
        let optimize_cost = self.optimize_cost;
        let try_lattice_k = |k: u32,
                             budget: u64,
                             scratch: &mut Option<Box<IntScratch16>>|
         -> (Vec<[i64; 16]>, bool) {
            let s = scratch
                .get_or_insert_with(|| {
                    let mut sb = Box::new(IntScratch16::new(epsilon));
                    sb.use_f64_gs = use_f64_gs;
                    sb.bkz_block_size = bkz_block_size;
                    sb
                });
            let y = uv_to_xy_zeta(v, k);
            let budget_hit = AtomicBool::new(false);
            let should_stop = |x: &[i64; 16]| -> bool {
                if optimize_cost { return false; }
                let cand = solution_to_u2q_d(x, k, d);
                diamond_distance_u2q_float(&cand, &target) < epsilon
            };
            let sols = phase1_with_stop(
                s.as_mut(), &y, k, epsilon, budget, &budget_hit, should_stop, None, None,
            );
            (sols, budget_hit.load(std::sync::atomic::Ordering::Relaxed))
        };

        let check_sols = |sols: &[[i64; 16]], k: u32| -> Option<SynthResultQ> {
            let mut best: Option<(usize, SynthResultQ)> = None;
            for sol in sols {
                let cand: U2Q = solution_to_u2q_d(sol, k, d);
                let dist = diamond_distance_u2q_float(&cand, &target);
                if dist < self.epsilon {
                    let gates = BlochDecomposer.decompose(&cand);
                    let cost = gates_cost(&gates);
                    let cand_result = SynthResultQ {
                        gates: Some(gates),
                        lde: k,
                        distance: dist,
                    };
                    if !optimize_cost {
                        return Some(cand_result);
                    }
                    match &best {
                        Some((bcost, _)) if *bcost <= cost => {}
                        _ => best = Some((cost, cand_result)),
                    }
                }
            }
            best.map(|(_, r)| r)
        };

        // Brute regime: iterate every k for exact small-T Clifford+√T finds.
        for k in self.min_lde..=BRUTE_LIMIT.min(self.max_lde) {
            let sols = phase1_brute(k);
            if let Some(r) = check_sols(&sols, k) {
                if trace {
                    diag::dump_zeta(&diag::snapshot(),
                        &format!("synthesize ε={:.0e} k={k}", self.epsilon));
                }
                return Some(r);
            }
        }

        // Stage-2 optimal m-sweep + Stage-4 lde-window with parallel
        // execution. Two phases:
        //
        // Phase 1 — first-hit *screen* (cost_min=false): sequentially
        // walk lde from lattice_start, with each `try_optimal_at_k`
        // using fast first-hit semantics (should_stop returns on first
        // ε-close leaf; dc_search_q uses find_map_any; no prefix-cost
        // prune). Purpose is *only* to determine find_lde. The
        // candidates returned here are discarded — phase 2 will
        // re-enumerate at find_lde for the actual cost-min selection.
        // Screen cost per failed lde is ~5-10× cheaper than full
        // enumeration, so phase 1 wall drops dramatically at deep ε
        // where lattice_start..find_lde spans 2-3 ldes.
        //
        // Phase 2 — full-enum cost-min (cost_min=true) over
        // `[find_lde, find_lde+window]`, parallel via thread::scope.
        // Each lde spawns one outer thread; each thread uses rayon
        // internally. Reduce by min cost across all spawned threads.
        //
        // Cost outcome is identical to the prior sequential-phase-1
        // implementation — same candidates explored, same min selection.
        // Wall at ε=1e-7 drops ~30-40% vs prior.
        if self.optimize_cost && !self.optimal_m_sweep.is_empty() {
            let lde_window = self.optimal_lde_window;
            let mut find_lde: Option<u32> = None;
            for k in lattice_start..=self.max_lde {
                let t_k = std::time::Instant::now();
                if self.try_optimal_at_k(target, d, v, k, /*cost_min=*/false).is_some() {
                    if trace {
                        eprintln!("[zeta] m-sweep lde={k:>2}  screen-hit  t={:.0}ms",
                            t_k.elapsed().as_secs_f64() * 1000.0);
                    }
                    find_lde = Some(k);
                    break;
                } else if trace {
                    eprintln!("[zeta] m-sweep lde={k:>2}  screen none   t={:.0}ms",
                        t_k.elapsed().as_secs_f64() * 1000.0);
                }
            }

            let mut best_overall: Option<(usize, SynthResultQ, u32)> = None;
            if let Some(fl) = find_lde {
                let window_kk: Vec<u32> = (0..=lde_window)
                    .map(|i| fl + i)
                    .filter(|&k| k <= self.max_lde)
                    .collect();
                let t_w = std::time::Instant::now();
                let window_results: Vec<Option<(usize, SynthResultQ, u32)>> =
                    std::thread::scope(|s| {
                        let handles: Vec<_> = window_kk
                            .iter()
                            .map(|&k| s.spawn(move || {
                                self.try_optimal_at_k(target, d, v, k, /*cost_min=*/true)
                            }))
                            .collect();
                        handles.into_iter().map(|h| h.join().unwrap()).collect()
                    });
                if trace {
                    eprintln!("[zeta] m-sweep enum {:?} parallel t={:.0}ms",
                        window_kk, t_w.elapsed().as_secs_f64() * 1000.0);
                }
                for r in window_results.into_iter().flatten() {
                    if trace {
                        eprintln!("[zeta]   enum  lde={:>2}  cost={} m={} dist={:.3e}",
                            r.1.lde, r.0, r.2, r.1.distance);
                    }
                    match &best_overall {
                        Some((bc, _, _)) if *bc <= r.0 => {}
                        _ => best_overall = Some(r),
                    }
                }
            }

            return best_overall.map(|(_, r, _)| r);
        }

        // Z1 D&C prototype: when `dc_split = Some(m)`, run the FGKM-prefix
        // dispatcher at each k instead of the single-search path.
        //
        // **2-pass dispatcher**: each lde first runs at `DC_PASS1_CAP=1M`
        // leaves per prefix. If found, return. If not found and at least
        // one prefix hit its budget, queue this lde for pass 2 (= the
        // search may have missed a solution beyond the budget). After
        // pass 1 sweeps all lde, retry the queued ones with
        // `DC_PASS2_CAP=10M`. This preserves minimum-lde correctness
        // (a budget-hit lde is never skipped) while letting easy targets
        // bail fast on NO-lde levels.
        if let Some(m_split) = self.dc_split {
            // Sequential small-k pass: dc_search_q cannot help for k <= m_split
            // (k_inner ≤ 0). These are typically few levels near lattice_start.
            for k in lattice_start..=m_split.min(self.max_lde) {
                let t_k = std::time::Instant::now();
                let (sols, _) = try_lattice_k(k, PASS1_CAP, &mut scratch);
                if let Some(r) = check_sols(&sols, k) {
                    if trace {
                        eprintln!("[zeta] dc lde={k:>2} (single fallback)  FOUND  dist={:.3e}  t={:.0}ms",
                            r.distance, t_k.elapsed().as_secs_f64() * 1000.0);
                    }
                    return Some(r);
                }
                if trace {
                    eprintln!("[zeta] dc lde={k:>2} (single fallback)  none   t={:.0}ms",
                        t_k.elapsed().as_secs_f64() * 1000.0);
                }
            }

            use std::sync::Mutex;
            let pass2_collector: Mutex<Vec<u32>> = Mutex::new(Vec::new());

            // ε > 1.6e-8: simple sequential loop, zero parallel-LDE
            // machinery (no thread::scope, no shared atomics, no
            // consumed-counter increments in the SE walker's hot path).
            // Atomic fetch_add on a 14-thread-shared counter costs ~25 ns
            // per recurse on contention; for million-node walks at
            // ε≥1e-7 that's a 30-50% wall regression for zero benefit
            // (parallel-LDE speculation only helps when hard targets
            // overshoot the predicted LDE, which doesn't happen at
            // shallow ε).
            if self.epsilon > 1.6e-8 {
                for k in (m_split + 1).max(lattice_start)..=self.max_lde {
                    let t_k = std::time::Instant::now();
                    let (result, budget_hit) = self.dc_search_q(
                        &target, k, m_split, None, dc_pass1_cap_for(self.epsilon),
                        None, None, None,
                    );
                    if let Some(r) = result {
                        if trace {
                            eprintln!("[zeta] dc lde={k:>2} m={m_split} pass1  FOUND  dist={:.3e}  t={:.0}ms",
                                r.distance, t_k.elapsed().as_secs_f64() * 1000.0);
                        }
                        return Some(r);
                    }
                    if trace {
                        eprintln!("[zeta] dc lde={k:>2} m={m_split} pass1  none{}  t={:.0}ms",
                            if budget_hit { " (budget hit)" } else { "" },
                            t_k.elapsed().as_secs_f64() * 1000.0);
                    }
                    if budget_hit { pass2_collector.lock().unwrap().push(k); }
                }
                let mut pass2_queue: Vec<u32> = pass2_collector.into_inner().unwrap();
                pass2_queue.sort_unstable();
                for k in pass2_queue {
                    let t_k = std::time::Instant::now();
                    if trace {
                        eprintln!("[zeta] dc lde={k:>2} m={m_split} pass2 dispatching ...");
                    }
                    let (result, _) = self.dc_search_q(
                        &target, k, m_split, None, dc_pass2_cap_for(self.epsilon), None, None, None,
                    );
                    if let Some(r) = result {
                        if trace {
                            eprintln!("[zeta] dc lde={k:>2} m={m_split} pass2  FOUND  dist={:.3e}  t={:.0}ms",
                                r.distance, t_k.elapsed().as_secs_f64() * 1000.0);
                        }
                        return Some(r);
                    }
                    if trace {
                        eprintln!("[zeta] dc lde={k:>2} m={m_split} pass2  none   t={:.0}ms",
                            t_k.elapsed().as_secs_f64() * 1000.0);
                    }
                }
                return None;
            }

            // ε ≤ 1.6e-8: parallel-LDE speculation. For k > m_split, run
            // a window of LDE levels concurrently. The first task to find
            // sets `cross_lde_abort`; in-flight peer walkers see it at
            // their next recurse-entry and abort. Hard-target wall drops
            // from "sum of no-sol burns + find" to "find at find-lde
            // alone" at the cost of thread-dilution overhead on easy
            // targets — only enabled in this regime because that's where
            // hard targets overshoot the predicted LDE.
            let cross_lde_abort = AtomicBool::new(false);
            let lde_window_size: u32 = self.parallel_lde_window.max(1);
            let mut k_cursor = (m_split + 1).max(lattice_start);

            let parallel_result: Option<SynthResultQ> = 'outer: loop {
                if k_cursor > self.max_lde { break 'outer None; }
                if cross_lde_abort.load(Ordering::Relaxed) { break 'outer None; }

                let window_end = (k_cursor + lde_window_size - 1).min(self.max_lde);
                let lde_window: Vec<u32> = (k_cursor..=window_end).collect();
                if trace {
                    eprintln!("[zeta] dc m={m_split} pass1 parallel-lde window={:?} dispatching ...", lde_window);
                }
                let t_window = std::time::Instant::now();

                // Asymmetric Staggered Speculation, budget-triggered.
                // Each LDE task at index i > 0 waits until the
                // predecessor (index i-1) has consumed `trigger_nodes`
                // search-tree nodes without finding, OR the cross-LDE
                // abort fires. When `trigger_nodes == 0` peers launch
                // immediately (window becomes naive parallel).
                let trigger_nodes = self.parallel_lde_trigger_nodes;
                let consumed_counters: Vec<std::sync::Arc<std::sync::atomic::AtomicU64>> =
                    (0..lde_window.len())
                        .map(|_| std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)))
                        .collect();
                let results: Mutex<Vec<(u32, Option<SynthResultQ>, bool)>> =
                    Mutex::new(Vec::new());
                std::thread::scope(|s| {
                    for (i, &k) in lde_window.iter().enumerate() {
                        let results_ref = &results;
                        let abort_ref = &cross_lde_abort;
                        let pass2_ref = &pass2_collector;
                        let my_consumed = consumed_counters[i].clone();
                        let predecessor_consumed: Option<std::sync::Arc<std::sync::atomic::AtomicU64>> =
                            if i > 0 { Some(consumed_counters[i - 1].clone()) } else { None };
                        s.spawn(move || {
                            // Wait for predecessor to consume `trigger_nodes`
                            // search-tree nodes (or for cross-LDE abort).
                            if i > 0 && trigger_nodes > 0 {
                                let pred = predecessor_consumed.as_ref().unwrap();
                                loop {
                                    if abort_ref.load(Ordering::Relaxed) { return; }
                                    if pred.load(Ordering::Relaxed) >= trigger_nodes { break; }
                                    std::thread::sleep(std::time::Duration::from_millis(50));
                                }
                                if abort_ref.load(Ordering::Relaxed) { return; }
                            }
                            let t_k = std::time::Instant::now();
                            // Pass shared signals only when they could
                            // actually fire: window=1 has no peer LDEs to
                            // abort us, and trigger_nodes=0 means no
                            // watcher reads the consumed counter. The
                            // SE walker pays an atomic load + fetch_add
                            // per recurse-enter on a contended cache
                            // line if either is Some — non-trivial wall
                            // overhead at deep ε.
                            let abort_opt = if lde_window_size > 1 { Some(abort_ref) } else { None };
                            let consumed_opt = if trigger_nodes > 0 {
                                Some(my_consumed.as_ref())
                            } else {
                                None
                            };
                            let (result, budget_hit) = self.dc_search_q(
                                &target, k, m_split, None, dc_pass1_cap_for(self.epsilon),
                                abort_opt,
                                consumed_opt,
                                None,
                            );
                            let dt = t_k.elapsed().as_secs_f64() * 1000.0;
                            if let Some(ref r) = result {
                                abort_ref.store(true, Ordering::Relaxed);
                                if trace {
                                    eprintln!("[zeta] dc lde={k:>2} m={m_split} pass1  FOUND  dist={:.3e}  t={:.0}ms  (consumed={})",
                                        r.distance, dt, my_consumed.load(Ordering::Relaxed));
                                }
                            } else if trace {
                                eprintln!("[zeta] dc lde={k:>2} m={m_split} pass1  none{}  t={:.0}ms  (consumed={})",
                                    if budget_hit { " (budget hit)" } else { "" }, dt,
                                    my_consumed.load(Ordering::Relaxed));
                            }
                            if result.is_none() && budget_hit {
                                pass2_ref.lock().unwrap().push(k);
                            }
                            results_ref.lock().unwrap().push((k, result, budget_hit));
                        });
                    }
                });
                // Pick the lowest-lde finder (minimum-circuit semantics).
                let mut found_results: Vec<(u32, SynthResultQ)> = results
                    .into_inner()
                    .unwrap()
                    .into_iter()
                    .filter_map(|(k, r, _)| r.map(|x| (k, x)))
                    .collect();
                found_results.sort_by_key(|(k, _)| *k);
                let res = found_results.into_iter().next().map(|(_, r)| r);

                if let Some(r) = res {
                    if trace {
                        eprintln!("[zeta] dc parallel-lde window wall  t={:.0}ms",
                            t_window.elapsed().as_secs_f64() * 1000.0);
                    }
                    break 'outer Some(r);
                }
                k_cursor = window_end + 1;
            };

            if let Some(r) = parallel_result { return Some(r); }

            let mut pass2_queue: Vec<u32> = pass2_collector.into_inner().unwrap();
            pass2_queue.sort_unstable();

            // Pass 2 retries: only the lde levels where pass 1's prefixes
            // hit budget without finding. Other lde levels were
            // exhausted at pass 1 (no solution exists at that lde).
            for k in pass2_queue {
                let t_k = std::time::Instant::now();
                if trace {
                    eprintln!("[zeta] dc lde={k:>2} m={m_split} pass2 dispatching ...");
                }
                let (result, _) = self.dc_search_q(&target, k, m_split, None, dc_pass2_cap_for(self.epsilon), None, None, None);
                if let Some(r) = result {
                    if trace {
                        eprintln!("[zeta] dc lde={k:>2} m={m_split} pass2  FOUND  dist={:.3e}  t={:.0}ms",
                            r.distance, t_k.elapsed().as_secs_f64() * 1000.0);
                    }
                    return Some(r);
                }
                if trace {
                    eprintln!("[zeta] dc lde={k:>2} m={m_split} pass2  none   t={:.0}ms",
                        t_k.elapsed().as_secs_f64() * 1000.0);
                }
            }
            return None;
        }

        // Lattice regime, Pass 1: aggressive budget cap. k's that hit the
        // budget without finding a sol get queued for Pass 2.
        let mut pass2_queue: Vec<u32> = Vec::new();
        for k in lattice_start..=self.max_lde {
            let t_k = std::time::Instant::now();
            let (sols, budget_was_hit) = try_lattice_k(k, PASS1_CAP, &mut scratch);
            if let Some(r) = check_sols(&sols, k) {
                if trace {
                    eprintln!("[zeta] pass1 lde={k:>2}  FOUND  dist={:.3e}  t={:.0}ms",
                        r.distance, t_k.elapsed().as_secs_f64() * 1000.0);
                    diag::dump_zeta(&diag::snapshot(),
                        &format!("synthesize ε={:.0e} k={k} (pass1)", self.epsilon));
                }
                return Some(r);
            }
            if trace {
                eprintln!("[zeta] pass1 lde={k:>2}  none{}  t={:.0}ms",
                    if budget_was_hit { " (budget hit)" } else { "" },
                    t_k.elapsed().as_secs_f64() * 1000.0);
            }
            if budget_was_hit {
                pass2_queue.push(k);
            }
        }

        // Lattice regime, Pass 2: only retry the k's that Pass 1
        // budget-hit. Guarantees no completeness loss vs single-pass-at-
        // PASS2_CAP, while skipping k's where Pass 1 was already
        // exhaustive.
        for k in pass2_queue {
            let t_k = std::time::Instant::now();
            let (sols, _) = try_lattice_k(k, PASS2_CAP, &mut scratch);
            if let Some(r) = check_sols(&sols, k) {
                if trace {
                    eprintln!("[zeta] pass2 lde={k:>2}  FOUND  dist={:.3e}  t={:.0}ms",
                        r.distance, t_k.elapsed().as_secs_f64() * 1000.0);
                    diag::dump_zeta(&diag::snapshot(),
                        &format!("synthesize ε={:.0e} k={k} (pass2)", self.epsilon));
                }
                return Some(r);
            }
            if trace {
                eprintln!("[zeta] pass2 lde={k:>2}  none   t={:.0}ms",
                    t_k.elapsed().as_secs_f64() * 1000.0);
            }
        }

        if trace {
            diag::dump_zeta(&diag::snapshot(),
                &format!("synthesize ε={:.0e} (no sol)", self.epsilon));
        }
        None
    }

    /// Z1 D&C inner-step at total lde `k_total` and split parameter
    /// `m_split`. Iterates every prefix `U_L ∈ L_{m_split}^Q`, computes
    /// the inner factor's alignment direction `v_inner = unitary_to_uv(
    /// U_L† · target)`, runs LLL+SE at `k_inner = k_total − k_prefix(U_L)`,
    /// reconstructs `U_full = U_L · U_R`, and returns the first candidate
    /// whose diamond distance to target is below ε.
    ///
    /// Per-prefix `d_R = (d_target − d_L) mod 16` parametrises the inner
    /// reconstruction so the assembled `U_full` matches target's det
    /// phase. This is the Z[ζ_16] analog of Clifford+T's `dc_search`.
    ///
    /// **Returns** `(Option<SynthResultQ>, bool)`: the result if found,
    /// and a flag indicating whether *any* prefix hit its per-prefix
    /// budget without finding. The 2-pass dispatcher in `synthesize` uses
    /// this flag to decide if a deeper-budget retry at this lde is
    /// warranted.
    ///
    /// **Parallelism**: prefixes are dispatched via rayon `par_iter` with
    /// per-thread `IntScratch16` allocated lazily through `map_init`. The
    /// `find_map_any` combinator aborts all workers as soon as any one
    /// returns `Some(_)`. Each per-prefix LLL+SE call is itself
    /// parallel inside (the SE walker forks at `z[15]`), so we
    /// over-subscribe rayon — empirically wins because rayon's
    /// work-stealing gracefully handles nested parallel work.
    #[allow(clippy::too_many_arguments)]
    fn dc_search_q(
        &self,
        target: &Mat2,
        k_total: u32,
        m_split: u32,
        dr_filter_override: Option<&[u32]>,
        per_prefix_cap: u64,
        external_abort: Option<&AtomicBool>,
        consumed: Option<&std::sync::atomic::AtomicU64>,
        cost_min_override: Option<bool>,
    ) -> (Option<SynthResultQ>, bool) {
        use rayon::prelude::*;
        use crate::synthesis::diag;

        let prefixes = build_l_q(m_split);
        let prefix_costs = build_l_q_costs(m_split);
        let d_target = det_phase_of(target);
        let epsilon = self.epsilon;
        let use_f64_gs = self.use_f64_gs;
        let bkz_block_size = self.bkz_block_size;

        // Shared across all prefix workers: any prefix that hits its
        // SE-leaf budget without finding sets this. The 2-pass dispatcher
        // uses it to decide if a pass2 retry is warranted.
        let any_budget_hit = Arc::new(AtomicBool::new(false));

        // Pre-filter the prefixes once: drop those whose lde already
        // exceeds k_total (k_inner would be ≤ 0), and drop those whose
        // required d_R isn't in the allowed-offsets set. Each entry
        // carries its precomputed decomposed cost for Stage-3 ranking
        // + heuristic pruning.
        let dc_dr_filter: &[u32] = dr_filter_override.unwrap_or(&self.dc_dr_filter);
        let mut usable: Vec<(&U2Q, usize)> = prefixes
            .iter()
            .zip(prefix_costs.iter().copied())
            .filter(|(u_l, _)| u_l.k < k_total)
            .filter(|(u_l, _)| {
                if dc_dr_filter.is_empty() {
                    return true;
                }
                let d_l = det_phase_of(&u_l.to_float());
                let d_r = ((d_target as i32 - d_l as i32).rem_euclid(16)) as u32;
                dc_dr_filter.contains(&d_r)
            })
            .collect();

        if usable.is_empty() {
            return (None, false);
        }

        let optimize_cost = cost_min_override.unwrap_or(self.optimize_cost);
        let prefix_prune = self.optimal_prefix_prune;

        // Sort order: optimal mode prefers cheap-prefix-cost first so
        // that the shared `best_cost` atomic drops quickly and later
        // prefixes can be heuristically skipped. First-hit mode keeps
        // the legacy k_prefix-desc heuristic (high k → small k_inner →
        // fast bail or hit, useful when |usable| > num_cores).
        if optimize_cost {
            usable.sort_by_key(|(_, c)| *c);
        } else {
            usable.sort_by(|(a, _), (b, _)| b.k.cmp(&a.k));
        }

        let n_threads = rayon::current_num_threads().max(1);
        let chunk = (usable.len() / n_threads).max(1);

        // Stage-3 shared best-cost tracker. Optimal-mode workers CAS this
        // when they find a candidate; later prefixes whose precomputed
        // cost(U_L) already exceeds the current best are skipped when
        // `optimal_prefix_prune` is on.
        let best_cost = Arc::new(std::sync::atomic::AtomicUsize::new(usize::MAX));

        let per_prefix = |scratch: &mut IntScratch16,
                          entry: &(&U2Q, usize)|
         -> Option<(usize, SynthResultQ)> {
            let (u_l, u_l_cost) = (entry.0, entry.1);
            if optimize_cost && prefix_prune {
                let cur_best = best_cost.load(std::sync::atomic::Ordering::Relaxed);
                if u_l_cost > cur_best {
                    return None;
                }
            }
            let k_prefix = u_l.k;
            let k_inner = k_total - k_prefix;

            // m_inner = U_L† · target as a continuous Mat2.
            let m_inner = u2q_dag_times_mat2(u_l, target);
            let v_inner = unitary_to_uv_zeta(&m_inner);

            // d_L from prefix's float det.
            let d_l = det_phase_of(&u_l.to_float());
            let d_r = ((d_target as i32 - d_l as i32).rem_euclid(16)) as u32;

            let y = uv_to_xy_zeta(v_inner, k_inner);
            let budget_hit = AtomicBool::new(false);
            let u_l_local = *u_l;
            let target_local = *target;
            let capture = diag::capture_enabled();
            let should_stop = |x: &[i64; 16]| -> bool {
                if optimize_cost { return false; }
                let u_r = solution_to_u2q_d(x, k_inner, d_r);
                let u_full = u_l_local * u_r;
                let hit = diamond_distance_u2q_float(&u_full, &target_local) < epsilon;
                if hit && capture {
                    diag::try_capture(diag::CapturedFind {
                        x_inner: *x, k_inner, k_total, d_r, d_l,
                    });
                }
                hit
            };

            let sols = phase1_with_stop(
                scratch, &y, k_inner, epsilon,
                per_prefix_cap, &budget_hit, should_stop,
                external_abort, consumed,
            );

            if budget_hit.load(std::sync::atomic::Ordering::Relaxed) {
                any_budget_hit.store(true, std::sync::atomic::Ordering::Relaxed);
            }

            // First-hit: return on first ε-close sol. Optimal: scan all
            // ε-close sols, decompose each, keep the min-cost one. In
            // optimal mode we also CAS-publish the per-prefix best into
            // the shared `best_cost` so subsequent prefixes can prune.
            let mut best: Option<(usize, SynthResultQ)> = None;
            for sol in &sols {
                let u_r = solution_to_u2q_d(sol, k_inner, d_r);
                let u_full = u_l_local * u_r;
                let dist = diamond_distance_u2q_float(&u_full, target);
                if dist < epsilon {
                    let gates = BlochDecomposer.decompose(&u_full);
                    let cost = gates_cost(&gates);
                    let result = SynthResultQ {
                        gates: Some(gates),
                        lde: k_total,
                        distance: dist,
                    };
                    if !optimize_cost {
                        return Some((cost, result));
                    }
                    match &best {
                        Some((bcost, _)) if *bcost <= cost => {}
                        _ => best = Some((cost, result)),
                    }
                }
            }
            if optimize_cost {
                if let Some((c, _)) = &best {
                    // CAS-publish: lower the shared best_cost if we
                    // improved it. Relaxed ordering is sufficient — the
                    // prune is a heuristic, not a correctness guarantee.
                    let mut cur = best_cost.load(std::sync::atomic::Ordering::Relaxed);
                    while *c < cur {
                        match best_cost.compare_exchange_weak(
                            cur, *c,
                            std::sync::atomic::Ordering::Relaxed,
                            std::sync::atomic::Ordering::Relaxed,
                        ) {
                            Ok(_) => break,
                            Err(actual) => cur = actual,
                        }
                    }
                }
            }
            best
        };

        let make_scratch = || {
            let mut s = IntScratch16::new(epsilon);
            s.use_f64_gs = use_f64_gs;
            s.bkz_block_size = bkz_block_size;
            s
        };

        let result_pair: Option<(usize, SynthResultQ)> = if optimize_cost {
            // Reduce across prefixes by min cost. No early-abort across
            // prefixes (every prefix runs to completion or its budget).
            usable
                .par_iter()
                .with_min_len(chunk)
                .map_init(make_scratch, per_prefix)
                .reduce(
                    || None,
                    |a, b| match (a, b) {
                        (None, x) | (x, None) => x,
                        (Some(a), Some(b)) => Some(if a.0 <= b.0 { a } else { b }),
                    },
                )
        } else {
            // First-hit: abort other prefixes as soon as one finds.
            usable
                .par_iter()
                .with_min_len(chunk)
                .map_init(make_scratch, per_prefix)
                .find_map_any(|x| x)
        };
        let result = result_pair.map(|(_, r)| r);

        let budget_hit = any_budget_hit.load(std::sync::atomic::Ordering::Relaxed);
        (result, budget_hit)
    }

    /// Single-search lattice probe at lde `k`, returning the best
    /// `(cost, SynthResultQ)` under the current `optimize_cost` mode.
    /// Mirrors the `try_lattice_k` + `check_sols` closures in
    /// [`Self::synthesize`] but as a method so it can be reused by the
    /// Stage-2/4 m-sweep and called concurrently from `thread::scope`.
    #[allow(clippy::too_many_arguments)]
    fn run_single_optimal(
        &self,
        target: &Mat2,
        d: u32,
        v: [f64; 4],
        k: u32,
        budget: u64,
        scratch: &mut Option<Box<IntScratch16>>,
        cost_min: bool,
    ) -> Option<(usize, SynthResultQ)> {
        let epsilon = self.epsilon;
        let s = scratch.get_or_insert_with(|| {
            let mut sb = Box::new(IntScratch16::new(epsilon));
            sb.use_f64_gs = self.use_f64_gs;
            sb.bkz_block_size = self.bkz_block_size;
            sb
        });
        let y = uv_to_xy_zeta(v, k);
        let budget_hit = AtomicBool::new(false);
        let should_stop = |x: &[i64; 16]| -> bool {
            if cost_min { return false; }
            let cand = solution_to_u2q_d(x, k, d);
            diamond_distance_u2q_float(&cand, target) < epsilon
        };
        let sols = phase1_with_stop(
            s.as_mut(), &y, k, epsilon, budget, &budget_hit, should_stop, None, None,
        );
        let mut best: Option<(usize, SynthResultQ)> = None;
        for sol in &sols {
            let cand: U2Q = solution_to_u2q_d(sol, k, d);
            let dist = diamond_distance_u2q_float(&cand, target);
            if dist < epsilon {
                let gates = BlochDecomposer.decompose(&cand);
                let cost = gates_cost(&gates);
                let result = SynthResultQ {
                    gates: Some(gates),
                    lde: k,
                    distance: dist,
                };
                if !cost_min {
                    return Some((cost, result));
                }
                match &best {
                    Some((bcost, _)) if *bcost <= cost => {}
                    _ => best = Some((cost, result)),
                }
            }
        }
        best
    }

    /// Stage-2/4 per-lde m-sweep: try every `m` in `optimal_m_sweep` and
    /// return the best `(cost, SynthResultQ, winning_m)` found at `k`.
    /// Used by both the sequential phase-1 search and the parallel-
    /// window phase-2 thread::scope spawns in [`Self::synthesize`].
    fn try_optimal_at_k(
        &self,
        target: Mat2,
        d: u32,
        v: [f64; 4],
        k: u32,
        cost_min: bool,
    ) -> Option<(usize, SynthResultQ, u32)> {
        let budget_mult = self.optimal_budget_multiplier.max(1);
        let mut local_scratch: Option<Box<IntScratch16>> = None;
        let mut best_at_k: Option<(usize, SynthResultQ, u32)> = None;
        for &m in &self.optimal_m_sweep {
            let cand_pair: Option<(usize, SynthResultQ)> = if m == 0 {
                let cap = PASS1_CAP.saturating_mul(budget_mult);
                self.run_single_optimal(&target, d, v, k, cap, &mut local_scratch, cost_min)
            } else if m < k {
                let filter = default_dc_dr_filter(m);
                let cap = dc_pass1_cap_for(self.epsilon).saturating_mul(budget_mult);
                let (r, _) = self.dc_search_q(
                    &target, k, m, Some(&filter), cap, None, None, Some(cost_min),
                );
                r.map(|res| {
                    let c = gates_cost(res.gates.as_deref().unwrap_or(""));
                    (c, res)
                })
            } else {
                None
            };
            if let Some((c, r)) = cand_pair {
                // Screen mode (first-hit): return on first find — we
                // only need find_lde, not the optimal cost at this k.
                if !cost_min {
                    return Some((c, r, m));
                }
                match &best_at_k {
                    Some((bc, _, _)) if *bc <= c => {}
                    _ => best_at_k = Some((c, r, m)),
                }
            }
        }
        best_at_k
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthesis::distance::diamond_distance_float;
    use num_complex::Complex64;
    use std::f64::consts::PI;

    fn complex_target(matrix: [[Complex64; 2]; 2]) -> Mat2 {
        matrix
    }

    /// Print raw vs deduped size of `L_m^Q` for m ∈ [0, 5]. Behind
    /// `--nocapture` in normal runs; the assertions are minimal — this is
    /// a measurement, not a correctness contract.
    #[test]
    fn build_l_q_size_growth() {
        for m in 0..=5 {
            let raw = if m == 0 {
                1
            } else {
                9 * 6u64.pow(m - 1) * 24
            };
            let l = build_l_q(m);
            let dedup = l.len();
            let factor = raw as f64 / dedup as f64;
            eprintln!(
                "m={m}  raw={raw:>8}  dedup={dedup:>8}  factor={factor:.2}x"
            );
            // Sanity: dedup never grows the set.
            assert!((dedup as u64) <= raw,
                "dedup ({dedup}) > raw ({raw}) at m={m}");
            // m=0 is just identity.
            if m == 0 {
                assert_eq!(dedup, 1);
            }
        }
    }

    /// Back-of-envelope: under cost model C(k) = c·α^k, the D&C cost
    /// ratio (vs single search at k_total) is
    ///   S(m, α) = Σ_k count(m, k) / α^k
    /// and is independent of k_total (the c·α^{k_total} term cancels).
    /// D&C wins at m when S(m, α) < 1.
    #[test]
    fn build_l_q_dc_cost_ratio() {
        // Coarse k → count map per m, then evaluate S(m, α) for several α.
        for m in 1..=5 {
            let l = build_l_q(m);
            let mut counts: Vec<u64> = vec![0; 64];
            for u in l.iter() {
                let k = u.k as usize;
                if k < counts.len() {
                    counts[k] += 1;
                }
            }
            eprint!("m={m:>2}  total={:>7}", l.len());
            for &alpha in &[2.0_f64, 2.5, 3.0, 3.5, 4.0] {
                let s: f64 = counts
                    .iter()
                    .enumerate()
                    .map(|(k, &c)| (c as f64) / alpha.powi(k as i32))
                    .sum();
                eprint!("   S(α={alpha:.1})={s:>10.2}");
            }
            eprintln!();
        }
        // Also show what threshold-filtering buys: keep only
        // prefixes with k_prefix ≥ τ, recompute S(m, α=2.0).
        eprintln!("\nThreshold filter τ on k_prefix, S(m, α=2):");
        eprintln!("{:>3}  {:>8}  τ=0    τ=4    τ=8    τ=12   τ=16   τ=20",
                  "m", "|L_m^Q|");
        for m in 1..=5 {
            let l = build_l_q(m);
            let mut counts: Vec<u64> = vec![0; 64];
            for u in l.iter() {
                let k = u.k as usize;
                if k < counts.len() {
                    counts[k] += 1;
                }
            }
            eprint!("{m:>3}  {:>8}", l.len());
            for &tau in &[0usize, 4, 8, 12, 16, 20] {
                let s: f64 = counts
                    .iter()
                    .enumerate()
                    .skip(tau)
                    .map(|(k, &c)| (c as f64) / (2.0_f64).powi(k as i32))
                    .sum();
                let n_kept: u64 = counts.iter().skip(tau).sum();
                eprint!("  {s:>5.2} ({n_kept:>5})");
            }
            eprintln!();
        }
    }

    /// Histogram of `k_prefix` across `L_m^Q` for m ∈ [1, 5]. We expect
    /// k_prefix ≤ m by FGKM Theorem 4.1(b) (each syllable peels max_exp
    /// by ≥ 1, so the word's denominator exponent grows by at most m).
    /// The shape of the distribution determines how we bin prefixes by
    /// k for the inner LLL+SE search.
    #[test]
    fn build_l_q_k_distribution() {
        for m in 1..=5 {
            let l = build_l_q(m);
            // Bins 0..=m+a few extra for safety in case the bound is
            // looser than expected.
            let max_bin: usize = (m as usize) + 4;
            let mut hist: Vec<u64> = vec![0; max_bin + 1];
            let mut k_min: u32 = u32::MAX;
            let mut k_max: u32 = 0;
            for u in l.iter() {
                let k = u.k as usize;
                k_min = k_min.min(u.k);
                k_max = k_max.max(u.k);
                if k <= max_bin {
                    hist[k] += 1;
                } else {
                    // Out-of-bound: extend the histogram (cheap, we'll
                    // see in the print).
                    while hist.len() <= k {
                        hist.push(0);
                    }
                    hist[k] += 1;
                }
            }
            let total: u64 = hist.iter().sum();
            eprintln!(
                "m={m}  total={total}  k range [{k_min}, {k_max}]"
            );
            for (k, count) in hist.iter().enumerate() {
                if *count == 0 { continue; }
                let pct = 100.0 * (*count as f64) / (total as f64);
                eprintln!("    k={k:>2}: {count:>7}  ({pct:>5.1}%)");
            }
        }
    }

    /// Multi-target benchmark: average D&C-with-filter vs single across
    /// random U3 targets at fixed ε.
    #[test]
    #[ignore]
    fn z1_dc_dr_filter_random_targets() {
        use rand::{Rng, SeedableRng};
        use rand::rngs::StdRng;

        fn rz(t: f64) -> Mat2 {
            [
                [Complex64::from_polar(1.0, -t/2.0), Complex64::new(0.0, 0.0)],
                [Complex64::new(0.0, 0.0), Complex64::from_polar(1.0, t/2.0)],
            ]
        }
        fn ry(t: f64) -> Mat2 {
            let c = (t/2.0).cos();
            let s = (t/2.0).sin();
            [
                [Complex64::new(c, 0.0), Complex64::new(-s, 0.0)],
                [Complex64::new(s, 0.0), Complex64::new(c, 0.0)],
            ]
        }
        fn matmul(a: Mat2, b: Mat2) -> Mat2 {
            [
                [a[0][0]*b[0][0] + a[0][1]*b[1][0], a[0][0]*b[0][1] + a[0][1]*b[1][1]],
                [a[1][0]*b[0][0] + a[1][1]*b[1][0], a[1][0]*b[0][1] + a[1][1]*b[1][1]],
            ]
        }

        let mut rng = StdRng::seed_from_u64(0xBEEF);
        let n = 4;
        let eps: f64 = std::env::var("Z1_EPS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1e-4);

        eprintln!("\n=== ε={eps:.0e}, {n} random U3 targets ===");
        let mut total_single = 0.0_f64;
        let mut total_m1_relaxed = 0.0_f64;
        let mut total_m2_strict = 0.0_f64;
        let mut wins_m1 = 0;
        let mut wins_m2 = 0;

        for i in 0..n {
            let alpha = 2.0 * std::f64::consts::PI * rng.random::<f64>();
            let beta = 2.0 * std::f64::consts::PI * rng.random::<f64>();
            let gamma = 2.0 * std::f64::consts::PI * rng.random::<f64>();
            let target = matmul(matmul(rz(alpha), ry(beta)), rz(gamma));

            let synth_s = SynthesizerQ::new(eps).with_max_lde(20);
            let t0 = std::time::Instant::now();
            let r_s = synth_s.synthesize(target);
            let ts = t0.elapsed().as_secs_f64() * 1000.0;
            assert!(r_s.is_some());

            let synth_m1 = SynthesizerQ::new(eps).with_max_lde(20)
                .with_dc_split(1).with_dc_dr_filter(vec![0, 1, 15]);
            let t0 = std::time::Instant::now();
            let r_m1 = synth_m1.synthesize(target);
            let tm1 = t0.elapsed().as_secs_f64() * 1000.0;

            let synth_m2 = SynthesizerQ::new(eps).with_max_lde(20)
                .with_dc_split(2).with_dc_dr_filter(vec![0]);
            let t0 = std::time::Instant::now();
            let r_m2 = synth_m2.synthesize(target);
            let tm2 = t0.elapsed().as_secs_f64() * 1000.0;

            total_single += ts;
            total_m1_relaxed += tm1;
            total_m2_strict += tm2;
            if tm1 < ts { wins_m1 += 1; }
            if tm2 < ts { wins_m2 += 1; }
            eprintln!(
                "  trial {i}  single={ts:>6.0}ms  m1_relaxed={tm1:>6.0}ms ({:.2}×)  m2_strict={tm2:>6.0}ms ({:.2}×)",
                ts/tm1, ts/tm2
            );
            // Sanity: dc found a valid result.
            if let Some(r) = r_m1 {
                assert!(r.distance < eps, "m1 trial {i} dist={:.3e}", r.distance);
            }
            if let Some(r) = r_m2 {
                assert!(r.distance < eps, "m2 trial {i} dist={:.3e}", r.distance);
            }
        }
        eprintln!("\n  TOTAL  single={total_single:.0}ms  m1_relaxed={total_m1_relaxed:.0}ms ({:.2}×)  m2_strict={total_m2_strict:.0}ms ({:.2}×)",
            total_single/total_m1_relaxed, total_single/total_m2_strict);
        eprintln!("  wins:  m1_relaxed {wins_m1}/{n}   m2_strict {wins_m2}/{n}");
    }

    /// Z1 det-phase filter test: with various allowed-d_R sets, see how
    /// many prefixes pass the filter and how the dispatcher does.
    #[test]
    #[ignore = "slow diagnostic; run with --ignored"]
    fn z1_dc_dr_filter() {
        let theta = 0.3_f64;
        let target: Mat2 = [
            [Complex64::from_polar(1.0, -theta / 2.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::from_polar(1.0, theta / 2.0)],
        ];
        let eps = 1e-3_f64;

        // d_R distribution on L_m^Q for this target.
        let d_target = det_phase_of(&target);
        for m in [1u32, 2, 3] {
            let prefixes = build_l_q(m);
            let mut hist = [0u64; 16];
            for u_l in prefixes.iter() {
                let d_l = det_phase_of(&u_l.to_float());
                let d_r = ((d_target as i32 - d_l as i32).rem_euclid(16)) as u32;
                hist[d_r as usize] += 1;
            }
            let mut s = format!("m={m} d_R hist (target d={d_target}):");
            for (d, c) in hist.iter().enumerate() {
                if *c > 0 {
                    s.push_str(&format!("  d_R={d}:{c}"));
                }
            }
            eprintln!("{s}");
        }
        eprintln!();

        // Try several filter configurations.
        let configs: &[(u32, &[u32], &str)] = &[
            (1, &[], "no filter"),
            (1, &[0], "strict d_R=0"),
            (1, &[0, 1, 15], "relaxed |d_R|≤1"),
            (1, &[0, 1, 2, 14, 15], "relaxed |d_R|≤2"),
            (2, &[], "no filter"),
            (2, &[0], "strict d_R=0"),
            (2, &[0, 1, 15], "relaxed |d_R|≤1"),
            (2, &[0, 1, 2, 14, 15], "relaxed |d_R|≤2"),
            (3, &[], "no filter"),
            (3, &[0], "strict d_R=0"),
            (3, &[0, 1, 15], "relaxed |d_R|≤1"),
            (3, &[0, 1, 2, 14, 15], "relaxed |d_R|≤2"),
        ];
        for (m, filter, label) in configs {
            let synth = SynthesizerQ::new(eps)
                .with_max_lde(15)
                .with_dc_split(*m)
                .with_dc_dr_filter(filter.to_vec());
            let t0 = std::time::Instant::now();
            let r = synth.synthesize(target);
            let dt = t0.elapsed();
            let l_size = build_l_q(*m).len();
            let n_pass = if filter.is_empty() {
                l_size as u64
            } else {
                let prefixes = build_l_q(*m);
                prefixes.iter().filter(|u| {
                    let d_l = det_phase_of(&u.to_float());
                    let d_r = ((d_target as i32 - d_l as i32).rem_euclid(16)) as u32;
                    filter.contains(&d_r)
                }).count() as u64
            };
            eprintln!(
                "  m={m} {label:<22} pass={n_pass:>5}/{l_size:<6}  lde={:?}  t={:>7.0}ms",
                r.as_ref().map(|r| r.lde),
                dt.as_secs_f64() * 1000.0
            );
        }
    }

    #[test]
    #[ignore = "slow diagnostic; run with --ignored"]
    fn z1_dc_smoke_rz_eps_1e_3() {
        let theta = 0.3_f64;
        let target: Mat2 = [
            [Complex64::from_polar(1.0, -theta / 2.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::from_polar(1.0, theta / 2.0)],
        ];
        let eps = 1e-3_f64;

        // Single-search baseline.
        let synth_single = SynthesizerQ::new(eps).with_max_lde(15);
        let t0 = std::time::Instant::now();
        let r_single = synth_single.synthesize(target);
        let t_single = t0.elapsed();
        eprintln!(
            "single: lde={:?} dist={:?} t={:.1}ms",
            r_single.as_ref().map(|r| r.lde),
            r_single.as_ref().map(|r| r.distance),
            t_single.as_secs_f64() * 1000.0
        );
        assert!(r_single.is_some());

        // D&C across several m values to characterize per-prefix cost.
        for m in [1u32, 2, 3] {
            let synth_dc = SynthesizerQ::new(eps).with_max_lde(15).with_dc_split(m);
            let t1 = std::time::Instant::now();
            let r_dc = synth_dc.synthesize(target);
            let t_dc = t1.elapsed();
            let l_size = build_l_q(m).len();
            let per_prefix_us = t_dc.as_secs_f64() * 1e6 / (l_size as f64);
            eprintln!(
                "  d&c m={m}: |L|={l_size:>6}  lde={:?}  t={:.1}ms  per-prefix={per_prefix_us:.0}μs",
                r_dc.as_ref().map(|r| r.lde),
                t_dc.as_secs_f64() * 1000.0
            );
            assert!(r_dc.is_some(), "D&C m={m} should also find a solution");
        }
    }

    #[test]
    fn auto_defaults_at_various_eps() {
        // Default at ε=1e-6: D&C with m=1, |d_R|≤1 (relaxed filter).
        let s = SynthesizerQ::new(1e-6);
        assert_eq!(s.dc_split, Some(1));
        assert_eq!(s.dc_dr_filter, vec![0u32, 1, 15]);

        // Default at ε ≤ 1e-7: D&C with m=2, d_R=0 (strict filter) —
        // empirically faster + better lde quality at this depth.
        let s7 = SynthesizerQ::new(1e-7);
        assert_eq!(s7.dc_split, Some(2));
        assert_eq!(s7.dc_dr_filter, vec![0u32]);
        assert_eq!(s7.max_lde, 35, "max_lde should auto-bump at ε ≤ 1e-7");

        let s8 = SynthesizerQ::new(1e-8);
        assert_eq!(s8.dc_split, Some(2));
        assert_eq!(s8.dc_dr_filter, vec![0u32]);

        // Default at moderate ε: single search.
        let s3 = SynthesizerQ::new(1e-3);
        assert_eq!(s3.dc_split, None);
        assert_eq!(s3.dc_dr_filter, Vec::<u32>::new());
        assert_eq!(s3.max_lde, 30);

        // f64 GS is on at moderate ε but auto-disabled at ε ≤ 1e-8
        // (where f64's 2-bit precision margin causes ladder thrashing).
        for &eps in &[1e-3, 1e-4, 1e-5, 1e-6, 1e-7_f64] {
            assert!(SynthesizerQ::new(eps).use_f64_gs, "f64 default should be on at ε={eps:.0e}");
        }
        let eps = 1e-8_f64;
        assert!(!SynthesizerQ::new(eps).use_f64_gs, "f64 default should be OFF at ε={eps:.0e}");

        // Manual override still works.
        let s_override = SynthesizerQ::new(1e-7).with_dc_split(1).with_dc_dr_filter(vec![0, 1, 15]);
        assert_eq!(s_override.dc_split, Some(1));
        assert_eq!(s_override.dc_dr_filter, vec![0u32, 1, 15]);
        let s_no_f64 = SynthesizerQ::new(1e-3).with_f64_gs(false);
        assert!(!s_no_f64.use_f64_gs);

        // BKZ-4 default: on at ε ≤ 1e-7, off above.
        for &eps in &[1e-3, 1e-4, 1e-5, 1e-6_f64] {
            assert_eq!(SynthesizerQ::new(eps).bkz_block_size, 0,
                "BKZ default should be 0 at ε={eps:.0e}");
        }
        for &eps in &[1e-7, 1e-8_f64] {
            assert_eq!(SynthesizerQ::new(eps).bkz_block_size, 4,
                "BKZ default should be 4 at ε={eps:.0e}");
        }
    }

    #[test]
    fn synthesize_identity_at_k_0() {
        let one = Complex64::new(1.0, 0.0);
        let zero = Complex64::new(0.0, 0.0);
        let target = complex_target([[one, zero], [zero, one]]);
        let synth = SynthesizerQ::new(1e-7);
        let result = synth.synthesize(target).expect("identity should synthesize");
        assert_eq!(result.lde, 0, "identity should be at k=0");
        assert!(result.distance < 1e-7);
    }

    #[test]
    fn synthesize_q_gate() {
        let q = U2Q::q();
        let target = q.to_float();
        let synth = SynthesizerQ::new(1e-7);
        let result = synth.synthesize(target).expect("Q should synthesize");
        assert_eq!(result.lde, 0, "Q should be found at k=0");
        assert!(result.distance < 1e-7);
        // The synthesized gate string, when applied, should give back Q.
        assert!(result.gates.is_some());
    }

    #[test]
    fn synthesize_t_gate() {
        let t = U2Q::t();
        let target = t.to_float();
        let synth = SynthesizerQ::new(1e-7);
        let result = synth.synthesize(target).expect("T should synthesize");
        assert_eq!(result.lde, 0, "T should be found at k=0");
        assert!(result.distance < 1e-7);
    }

    #[test]
    fn synthesize_hqh() {
        let hqh: U2Q = U2Q::h() * U2Q::q() * U2Q::h();
        let target = hqh.to_float();
        let synth = SynthesizerQ::new(1e-7);
        let result = synth.synthesize(target).expect("HQH should synthesize");
        // HQH has k=2 (1 from each H).
        assert_eq!(result.lde, 2);
        assert!(result.distance < 1e-7);
    }

    #[test]
    fn synthesize_qhq() {
        let qhq: U2Q = U2Q::q() * U2Q::h() * U2Q::q();
        let target = qhq.to_float();
        let synth = SynthesizerQ::new(1e-7);
        let result = synth.synthesize(target).expect("QHQ should synthesize");
        assert_eq!(result.lde, 1);
        assert!(result.distance < 1e-7);
    }

    #[test]
    fn synthesize_h_gate() {
        // H has k=1 (one H gate). Should be found at k=1.
        let h = U2Q::h();
        let target = h.to_float();
        let synth = SynthesizerQ::new(1e-7);
        let result = synth.synthesize(target).expect("H should synthesize");
        assert_eq!(result.lde, 1);
        assert!(result.distance < 1e-7);
    }

    #[test]
    fn synthesize_returns_none_when_unreachable() {
        // Target Rx(π/16) — angle isn't a multiple of π/8, so the closest
        // Clifford+√T circuit at any small k is bounded away from it. With
        // ε=1e-7 (tight) and max_lde=2 (so the test stays under a second),
        // should return None.
        let theta = PI / 16.0;
        let c = (theta / 2.0).cos();
        let s = (theta / 2.0).sin();
        let i = Complex64::new(0.0, 1.0);
        let target: Mat2 = [
            [Complex64::new(c, 0.0), -i * s],
            [-i * s, Complex64::new(c, 0.0)],
        ];
        let synth = SynthesizerQ::new(1e-8).with_max_lde(2);
        let result = synth.synthesize(target);
        assert!(result.is_none(),
            "Rx(π/16) should not be reachable in Clifford+√T at k≤2 with ε=1e-8");
    }

    #[test]
    fn synthesize_approximation_with_loose_epsilon() {
        // For Rx(π/16) at LOOSE ε, the synthesizer should find a closeby
        // approximation at small k. Tests the "approximate synthesis" path.
        let theta = PI / 16.0;
        let c = (theta / 2.0).cos();
        let s = (theta / 2.0).sin();
        let i = Complex64::new(0.0, 1.0);
        let target: Mat2 = [
            [Complex64::new(c, 0.0), -i * s],
            [-i * s, Complex64::new(c, 0.0)],
        ];
        let synth = SynthesizerQ::new(0.3).with_max_lde(2);  // very loose
        let result = synth.synthesize(target);
        assert!(result.is_some(), "loose ε should find an approximation");
        let r = result.unwrap();
        assert!(r.distance < 0.3);
    }

    #[test]
    fn synthesized_gate_string_roundtrip() {
        // For each of several Clifford+√T targets, the gate string from
        // the synthesizer should reconstruct (via gates_to_u2q) to a
        // U2Q close to the target.
        use crate::matrix::u2::U2Q;
        let targets: Vec<U2Q> = vec![
            U2Q::q(),
            U2Q::t(),
            U2Q::q() * U2Q::q(),  // = T
            U2Q::h() * U2Q::q() * U2Q::h(),
            U2Q::q() * U2Q::h() * U2Q::q(),
        ];
        let synth = SynthesizerQ::new(1e-7);
        for u in targets {
            let target = u.to_float();
            let result = synth.synthesize(target).expect("should synthesize");
            let gates = result.gates.expect("should have gate string");
            // Reconstruct via gates_to_u2q.
            let mut rebuilt = U2Q::eye();
            for c in gates.chars() {
                rebuilt = rebuilt * match c {
                    'H' => U2Q::h(),
                    'S' => U2Q::s(),
                    'T' => U2Q::t(),
                    'Q' => U2Q::q(),
                    'X' => U2Q::x(),
                    'Y' => U2Q::y(),
                    'Z' => U2Q::z(),
                    _ => panic!("unexpected gate {c}"),
                };
            }
            let dist = diamond_distance_float(&rebuilt.to_float(), &target);
            assert!(dist < 1e-7,
                "round-trip dist for gate string \"{gates}\" = {dist:.3e}");
        }
    }

    /// End-to-end deep-ε test: Rz(0.3) at ε=1e-3. Behind `#[ignore]` because
    /// it can take minutes — the lattice search at k=10 needs ~1G SE leaves.
    /// Run with `cargo test --release --lib synthesize_rz_eps_1e_3 --
    /// --ignored --nocapture`.
    #[test]
    #[ignore]
    fn synthesize_rz_eps_1e_3() {
        let theta = 0.3_f64;
        let target: Mat2 = [
            [Complex64::from_polar(1.0, -theta / 2.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::from_polar(1.0, theta / 2.0)],
        ];
        let synth = SynthesizerQ::new(1e-3).with_max_lde(15);
        let t0 = std::time::Instant::now();
        let result = synth.synthesize(target).expect("Rz(0.3) at ε=1e-3 should land");
        eprintln!(
            "Rz(0.3) at ε=1e-3: lde={} dist={:.3e} t={:?}",
            result.lde, result.distance, t0.elapsed()
        );
        assert!(result.distance < 1e-3);
        // Upper bound from 8D Clifford+T: lde=28. Z[ζ_16] should land much
        // smaller (~10) since `T = QQ` doubles the effective denominator
        // factor in the 8D path.
        assert!(result.lde <= 14,
            "expected lde ≤ 14 (8D Clifford+T is 28); got {}", result.lde);
    }

    #[test]
    fn synthesize_rz_via_lattice_backend() {
        // Rz(0.3) at ε=0.05 is unreachable at k ≤ 4 (brute regime), so
        // forcing min_lde > BRUTE_LIMIT exercises the LLL+SE lattice path.
        let theta = 0.3_f64;
        let target: Mat2 = [
            [Complex64::from_polar(1.0, -theta / 2.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::from_polar(1.0, theta / 2.0)],
        ];
        let synth = SynthesizerQ::new(0.05)
            .with_min_lde(BRUTE_LIMIT + 1)
            .with_max_lde(12);
        let result = synth.synthesize(target).expect("Rz(0.3) at ε=0.05 should land");
        assert!(result.lde > BRUTE_LIMIT,
            "expected lattice backend (k > {BRUTE_LIMIT}), got k={}", result.lde);
        assert!(result.distance < 0.05,
            "diamond distance {:.3e} exceeds ε=0.05", result.distance);
        assert!(result.gates.is_some());
    }
}
