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
use crate::synthesis::lattice_zeta::{phase1_with_stop, phase1_with_stop_mpfr, IntScratch16};
use crate::synthesis::search_zeta::{phase1_brute, uv_to_xy_zeta, uv_to_xy_zeta_mpfr};
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

/// Rotate `target` by a global phase so its det lands exactly on the
/// nearest ζ₁₆ power. Lossless (diamond distance is phase-invariant) —
/// but without it a U(2) input whose det is not a 16th root carries a
/// residual phase no completion can absorb, and the search burns to
/// max_lde finding nothing.
pub(crate) fn project_det_to_zeta_coset(target: &Mat2) -> Mat2 {
    let det = target[0][0] * target[1][1] - target[0][1] * target[1][0];
    let d = det_phase_of(target) as f64;
    let mut residual = det.arg() - d * PI / 8.0;
    while residual > PI {
        residual -= 2.0 * PI;
    }
    while residual <= -PI {
        residual += 2.0 * PI;
    }
    let g = Complex64::from_polar(1.0, -residual / 2.0);
    [
        [target[0][0] * g, target[0][1] * g],
        [target[1][0] * g, target[1][1] * g],
    ]
}

/// The det-phase `d ∈ {0..15}` of V: the integer with `ζ_16^d` closest
/// to `det(V)` on the unit circle (16-valued analog of Z[ω]'s
/// `det_zeta_parity`).
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
// axes (Lemma 3.1). m is the right enumeration coordinate because each
// syllable peels √2-exp by ≥1, matching the inner lde split; Q-count
// (Σaᵢ ∈ [m, 3m]) does not.
//

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

/// Cache for prefix `(T, Q)` gate counts (parallel to `BUILD_L_Q_CACHE`).
static BUILD_L_Q_TQ_CACHE: LazyLock<Mutex<HashMap<u32, Arc<Vec<(usize, usize)>>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Pre-computed `(T_count, Q_count)` of the canonical [`BlochDecomposer`]
/// decomposition for each prefix in `build_l_q(m)`, indexed parallel to
/// that Vec. Cached forever per `m`; the caller applies its own Q-cost
/// weight. NB: the weighted cost is **not a lower bound** on
/// `cost(U_L · U_R)` — U_R can cancel parts of U_L. It is used as a
/// heuristic ranking + prune, not a sound bound.
pub fn build_l_q_tq(m: u32) -> Arc<Vec<(usize, usize)>> {
    {
        let cache = BUILD_L_Q_TQ_CACHE.lock().unwrap();
        if let Some(v) = cache.get(&m) {
            return Arc::clone(v);
        }
    }
    let prefixes = build_l_q(m);
    let counts: Vec<(usize, usize)> = prefixes
        .iter()
        .map(|u_l| gates_tq(&BlochDecomposer.decompose(u_l)))
        .collect();
    let arc = Arc::new(counts);
    BUILD_L_Q_TQ_CACHE
        .lock()
        .unwrap()
        .insert(m, Arc::clone(&arc));
    arc
}

/// Right-coset dedup gate for the ζ prefix lists — the zeta mirror of
/// 8D's stage-1 `CYCLOSYNTH_L_COSET` (docs/w_8d_rework_notes.md):
/// `CYCLOSYNTH_ZETA_COSET=0` disables the dedup (no-dedup A/B escape),
/// anything else (or unset) enables it. Read once per process. Unlike
/// 8D there is no ε floor to start with: the zeta deep-ε pipeline
/// already computes `v_inner` in MPFR (`u2q_dag_v_inner_mpfr`), which
/// is exactly the per-frame-cap precision fix 8D's floor is waiting on.
static ZETA_COSET_DEDUP: LazyLock<bool> = LazyLock::new(|| {
    !matches!(std::env::var("CYCLOSYNTH_ZETA_COSET").as_deref(), Ok("0"))
});

/// The 8-element lde-0 Clifford subgroup ⟨S, X⟩ as U2Q, rebuilt from
/// [`CLIFFORD_TABLE_T`] entry names via [`CLIFFORD_LDE0_IDX`] — the same
/// name-folding route `build_l_q_inner` uses for its Clifford suffixes
/// (NOT the det-1 U2T table matrices, which differ by ζ-power phases;
/// orbit keys must match the list's own construction including float
/// tie-breaking, see `build_l_q_orbits`).
fn lde0_cliffords_q() -> [U2Q; 8] {
    use crate::synthesis::cliffords::CLIFFORD_LDE0_IDX;
    std::array::from_fn(|j| {
        let (name, _) = &CLIFFORD_TABLE_T[CLIFFORD_LDE0_IDX[j]];
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
}

/// Cache for per-prefix right-coset orbit ids (parallel to
/// [`BUILD_L_Q_CACHE`], keyed by syllable count `m`).
static BUILD_L_Q_ORBIT_CACHE: LazyLock<Mutex<HashMap<u32, Arc<Vec<usize>>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Orbit id per prefix under RIGHT multiplication by the lde-0
/// Clifford subgroup ⟨S, X⟩, mod global phase (id = min list index
/// among key-matched mates). Mates whose float key is absent from the
/// list stay unlinked — conservative: less dedup, never less coverage.
/// The linking is by float value and `build_l_q` stores the unreduced
/// peel-depth k, so an orbit can span several k; production dedup
/// groups by (orbit, k), within which mates are exact ring-unit coset
/// partners (pinned by `zeta_coset_orbits_sound`).
pub fn build_l_q_orbits(m: u32) -> Arc<Vec<usize>> {
    {
        let cache = BUILD_L_Q_ORBIT_CACHE.lock().unwrap();
        if let Some(v) = cache.get(&m) {
            return Arc::clone(v);
        }
    }
    let prefixes = build_l_q(m);
    let idx_of: HashMap<[i64; 8], usize> = prefixes
        .iter()
        .enumerate()
        .map(|(i, u)| (canonical_key_q(u), i))
        .collect();
    let lde0 = lde0_cliffords_q();
    let orbit: Vec<usize> = (0..prefixes.len())
        .map(|i| {
            let mut mn = i;
            for c in &lde0 {
                if let Some(&j) = idx_of.get(&canonical_key_q(&(prefixes[i] * *c))) {
                    mn = mn.min(j);
                }
            }
            mn
        })
        .collect();
    let arc = Arc::new(orbit);
    BUILD_L_Q_ORBIT_CACHE
        .lock()
        .unwrap()
        .insert(m, Arc::clone(&arc));
    arc
}

/// Keep the min-(weight, index) member of each (orbit, k) class of an
/// already-filtered candidate list. (orbit, k) and not raw orbit:
/// same-k mates are exact ring-unit isometries (identical inner
/// subproblems and totals) while cross-k coverage is asymmetric, so
/// cross-k members stay separate. Must run AFTER the usable filter —
/// a canonical rep can be filter-excluded while a usable mate
/// survives, and dropping the mate would flip FOUND→none. Min-weight
/// keeps the floor prune sound: the kept rep's floor never prunes a
/// class that still hides an improving total.
fn coset_keep_mask(cands: &[(usize, usize)], keys: &[(usize, u32)]) -> Vec<bool> {
    use std::collections::hash_map::Entry;
    let mut best: HashMap<(usize, u32), usize> = HashMap::new(); // class → pos
    for (pos, &(pi, w)) in cands.iter().enumerate() {
        match best.entry(keys[pi]) {
            Entry::Occupied(mut e) => {
                let (bpi, bw) = cands[*e.get()];
                if (w, pi) < (bw, bpi) {
                    e.insert(pos);
                }
            }
            Entry::Vacant(e) => {
                e.insert(pos);
            }
        }
    }
    let mut mask = vec![false; cands.len()];
    for pos in best.into_values() {
        mask[pos] = true;
    }
    mask
}

/// Cached per-m `(orbit id, k)` dedup keys, parallel to `build_l_q(m)`.
static BUILD_L_Q_COSET_KEY_CACHE: LazyLock<Mutex<HashMap<u32, Arc<Vec<(usize, u32)>>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// The `(orbit id, unreduced k)` dedup class per prefix of
/// `build_l_q(m)` — the key [`coset_keep_mask`] groups by.
pub fn build_l_q_coset_keys(m: u32) -> Arc<Vec<(usize, u32)>> {
    {
        let cache = BUILD_L_Q_COSET_KEY_CACHE.lock().unwrap();
        if let Some(v) = cache.get(&m) {
            return Arc::clone(v);
        }
    }
    let prefixes = build_l_q(m);
    let orbit = build_l_q_orbits(m);
    let keys: Vec<(usize, u32)> = prefixes
        .iter()
        .zip(orbit.iter())
        .map(|(u, &o)| (o, u.k))
        .collect();
    let arc = Arc::new(keys);
    BUILD_L_Q_COSET_KEY_CACHE
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
    //
    // NOTE on `k` semantics (2026-06-10): the stored `k` here is the
    // UNREDUCED accumulation — a *peel-depth* coordinate matching the
    // inner-LLL+SE shell split (`k_inner = k_total − u_l.k`), NOT the
    // prefix's reduced matrix lde. Reducing here (tried) makes z-axis
    // and Clifford-heavy prefixes drop to k ≈ 0-1, so their suffix
    // searches run at nearly full depth — a ~30× wall explosion — while
    // only partially fixing the coverage gap it was meant to address
    // (canonical routes clipped by heterogeneous inflation; see the
    // critic review in docs/design_certified_optimal_cost.md). The
    // right fix needs a dual-coordinate design: reduced lde for cost
    // accounting, peel depth for shell selection.
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

/// Optimality certificate from [`SynthesizerQ::synthesize_certified`].
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
    /// search. Builder: [`Self::with_dc_split`].
    pub dc_split: Option<u32>,
    /// Allowed `(d_target − d_L) mod 16` offsets for a prefix to be
    /// processed; empty = no filter. Builder: [`Self::with_dc_dr_filter`].
    pub dc_dr_filter: Vec<u32>,
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
    /// Skip prefixes whose own weighted cost already exceeds the
    /// incumbent. Heuristic: U_R can cancel parts of U_L, so this can in
    /// principle miss the optimum; empirically it never has on random
    /// SU(2) targets. Builder: [`Self::with_optimal_prefix_prune`].
    pub optimal_prefix_prune: bool,
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

/// k cutoff: brute-force handles `k ≤ BRUTE_LIMIT`, lattice handles the rest.
/// At 3, brute tops out at ~10⁷ shell points (~100 ms).
const BRUTE_LIMIT: u32 = 3;

/// Process-wide cache over [`phase1_brute`]: the shell enumeration is a
/// pure function of `k`, and optimal mode would otherwise re-run it 4×
/// per target. The cached unit-scale d = 0 float matrices
/// `(u1, −u2*, u2, u1*)/√2^k` let per-target scans use the cheap f64
/// prefilter [`brute_dist_est`] instead of MPFR distance on every shell
/// solution; accept/reject still goes through the exact MPFR path, so
/// decisions are bit-identical to the uncached scan.
struct BruteShell {
    sols: Vec<[i64; 16]>,
    mats: Vec<[Complex64; 4]>,
}

fn brute_shell_cached(k: u32) -> &'static BruteShell {
    use std::sync::OnceLock;
    const CELL: OnceLock<BruteShell> = OnceLock::new();
    static CACHE: [OnceLock<BruteShell>; (BRUTE_LIMIT + 1) as usize] =
        [CELL; (BRUTE_LIMIT + 1) as usize];
    debug_assert!(k <= BRUTE_LIMIT);
    CACHE[k as usize].get_or_init(|| {
        let sols = phase1_brute(k);
        let inv_scale = 1.0 / (2f64.powi(k as i32)).sqrt();
        // ζ₁₆^j basis at unit scale.
        let basis: [Complex64; 8] =
            std::array::from_fn(|j| Complex64::from_polar(1.0, j as f64 * PI / 8.0));
        let to_c = |s: &[i64]| -> Complex64 {
            (0..8).map(|j| basis[j] * s[j] as f64).sum::<Complex64>() * inv_scale
        };
        let mats = sols
            .iter()
            .map(|sol| {
                let u1 = to_c(&sol[0..8]);
                let u2 = to_c(&sol[8..16]);
                [u1, -u2.conj(), u2, u1.conj()]
            })
            .collect();
        BruteShell { sols, mats }
    })
}

/// f64 estimate of the diamond distance from the cached unit-scale
/// matrix and det-phase rotation `zd = ζ₁₆^d`. Conservative prefilter
/// only — callers skip the exact MPFR check when the estimate clears ε
/// by [`brute_prefilter_threshold`]'s margin, so no true ε-accept is
/// ever lost (estimator abs error ≲ 1e-14 on these O(1) entries).
#[inline]
fn brute_dist_est(m: &[Complex64; 4], zd: Complex64, target: &Mat2) -> f64 {
    let u = [m[0], zd * m[1], m[2], zd * m[3]];
    let t = [target[0][0], target[0][1], target[1][0], target[1][1]];
    let mut tr = Complex64::new(0.0, 0.0);
    let mut su = 0.0;
    let mut st = 0.0;
    for i in 0..4 {
        tr += u[i] * t[i].conj();
        su += u[i].norm_sqr();
        st += t[i].norm_sqr();
    }
    let fro = (su + st - 2.0 * tr.norm()).max(0.0);
    let d_sq = fro * (8.0 - fro) / 16.0;
    d_sq.max(0.0).sqrt()
}

/// The slack is ~3 orders of magnitude above the estimator's error
/// bound (and brute only runs at ε > 1e-8), so candidates with true
/// distance < ε always reach the exact check.
#[inline]
fn brute_prefilter_threshold(epsilon: f64) -> f64 {
    1.05 * epsilon + 1e-11
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
/// per-m phases (see `synthesize_optimal_certified`) — interleaved,
/// m=2's 6× prefix fan-out starves the deep m=1 units that hold the
/// decisive finds.
fn default_optimal_m_sweep(epsilon: f64) -> Vec<u32> {
    if epsilon >= 1e-6 {
        vec![1]
    } else {
        vec![1, 2]
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

/// MPFR-precision column-1 of `U_L† · target` as the alignment vector
/// `v_inner` — the deep-ε replacement for the f64
/// `u2q_dag_times_mat2` → `unitary_to_uv_zeta` chain. Why: the f64
/// product's ~1e-16 error matches the radial cap width ε² at ε = 1e-8
/// and displaces the constructed cap, and no enumeration bound recovers
/// a solution the cap no longer contains. `U_L` is exact ring data and
/// `target` exact f64 data, so the product carries full `prec` bits.
fn u2q_dag_v_inner_mpfr(u_l: &U2Q, target: &Mat2, prec: u32) -> [rug::Float; 4] {
    use rug::ops::Pow;
    use rug::Float as RF;
    // ζ^i = e^{iπ/8}: cos/sin tables at prec.
    let pi = RF::with_val(prec, rug::float::Constant::Pi);
    let cosv: [RF; 8] = std::array::from_fn(|i| {
        (RF::with_val(prec, &pi * (i as u32)) / 8u32).cos()
    });
    let sinv: [RF; 8] = std::array::from_fn(|i| {
        (RF::with_val(prec, &pi * (i as u32)) / 8u32).sin()
    });
    // (re, im) of a ZZeta numerator at prec. Prefix coefficients are
    // far inside i64 at any production lde; debug-guarded.
    let zz = |z: &crate::rings::ZZeta| -> (RF, RF) {
        let mut re = RF::with_val(prec, 0.0);
        let mut im = RF::with_val(prec, 0.0);
        for i in 0..8 {
            let c = crate::synthesis::lattice::lll::i256_to_f64(z.coeff(i));
            if c != 0.0 {
                re += RF::with_val(prec, &cosv[i] * c);
                im += RF::with_val(prec, &sinv[i] * c);
            }
        }
        (re, im)
    };
    // 1/√2^k at prec.
    let scale = RF::with_val(prec, RF::with_val(prec, 2.0).sqrt().pow(u_l.k)).recip();
    // U†'s row i is [conj(U[0][i]), conj(U[1][i])]; m_inner column 1:
    // mᵢ = Σⱼ conj(U[j][i])·t[j][0]. (a − bi)(c + di) = (ac+bd) + (ad−bc)i.
    let col = |z1: (RF, RF), z2: (RF, RF)| -> (RF, RF) {
        let (a1, b1) = z1;
        let (a2, b2) = z2;
        let (c1, d1) = (target[0][0].re, target[0][0].im);
        let (c2, d2) = (target[1][0].re, target[1][0].im);
        let re = RF::with_val(prec, &a1 * c1) + RF::with_val(prec, &b1 * d1)
            + RF::with_val(prec, &a2 * c2) + RF::with_val(prec, &b2 * d2);
        let im = RF::with_val(prec, &a1 * d1) - RF::with_val(prec, &b1 * c1)
            + RF::with_val(prec, &a2 * d2) - RF::with_val(prec, &b2 * c2);
        (re, im)
    };
    let (m00_re, m00_im) = col(zz(&u_l.u11), zz(&u_l.u21));
    let (m10_re, m10_im) = col(zz(&u_l.u12), zz(&u_l.u22));
    [
        m00_re * &scale,
        m00_im * &scale,
        m10_re * &scale,
        m10_im * &scale,
    ]
}

/// Rotate the complex pairs (v[0]+i·v[1], v[2]+i·v[3]) by e^{iπj/16}
/// in MPFR — the parity-branch rotation, applied AFTER exact v
/// derivation so the odd branch's cap is built from uncorrupted
/// geometry (the scalar rotation commutes with the prefix product).
fn rot32_mpfr(v: [rug::Float; 4], j: u32, prec: u32) -> [rug::Float; 4] {
    use rug::Float as RF;
    if j == 0 {
        return v;
    }
    let ang = RF::with_val(prec, rug::float::Constant::Pi) * j / 16u32;
    let c = ang.clone().cos();
    let s = ang.sin();
    let [a, b, x, y] = v;
    [
        RF::with_val(prec, &a * &c) - RF::with_val(prec, &b * &s),
        RF::with_val(prec, &a * &s) + RF::with_val(prec, &b * &c),
        RF::with_val(prec, &x * &c) - RF::with_val(prec, &y * &s),
        RF::with_val(prec, &x * &s) + RF::with_val(prec, &y * &c),
    ]
}

/// Deep-ε-aware phase1 router. At ε ≤ 2e-8 the radial cap width ε²/4
/// sits under the f64 ULP at unit scale, so an f64 y-chain corrupts Q,
/// the cap center, and the Cholesky factor — and an f64 prefix product
/// additionally displaces the cap itself ([`u2q_dag_v_inner_mpfr`]).
/// Those ε route through the MPFR entry with `v` derived from the most
/// exact source available; above 2e-8 the f64 path is safe and ~free.
#[allow(clippy::too_many_arguments)]
fn phase1_deep_aware<F>(
    scratch: &mut IntScratch16,
    v: [f64; 4],
    deep_v_src: Option<(&U2Q, &Mat2)>,
    rot_src: Option<&(Mat2, u32)>,
    k: u32,
    eps: f64,
    max_phase2_calls: u64,
    budget_hit: &std::sync::atomic::AtomicBool,
    should_stop: F,
    external_abort: Option<&std::sync::atomic::AtomicBool>,
    consumed: Option<&std::sync::atomic::AtomicU64>,
) -> Vec<[i64; 16]>
where
    F: Fn(&[i64; 16]) -> bool + Sync,
{
    if eps <= 2e-8 {
        let prec = scratch.prec_q;
        // Derive v from the most exact source available. With a
        // rot_src present, the caller's f64 `v` and `target` are the
        // ROTATED (f64-corrupted) forms — rebuild from the unrotated
        // original and rotate exactly in MPFR.
        let v_mpfr: [rug::Float; 4] = match (deep_v_src, rot_src) {
            (Some((u_l, _rotated)), Some((orig, j))) => {
                rot32_mpfr(u2q_dag_v_inner_mpfr(u_l, orig, prec), *j, prec)
            }
            (Some((u_l, target)), None) => u2q_dag_v_inner_mpfr(u_l, target, prec),
            (None, Some((orig, j))) => {
                let base: [rug::Float; 4] = [
                    rug::Float::with_val(prec, orig[0][0].re),
                    rug::Float::with_val(prec, orig[0][0].im),
                    rug::Float::with_val(prec, orig[1][0].re),
                    rug::Float::with_val(prec, orig[1][0].im),
                ];
                rot32_mpfr(base, *j, prec)
            }
            (None, None) => std::array::from_fn(|i| rug::Float::with_val(prec, v[i])),
        };
        let y_mpfr = uv_to_xy_zeta_mpfr(&v_mpfr, k, prec);
        phase1_with_stop_mpfr(
            scratch, &y_mpfr, &v_mpfr, k, eps, max_phase2_calls, budget_hit,
            should_stop, external_abort, consumed,
        )
    } else {
        let y = uv_to_xy_zeta(v, k);
        phase1_with_stop(
            scratch, &y, k, eps, max_phase2_calls, budget_hit, should_stop,
            external_abort, consumed,
        )
    }
}

/// Two-pass leaf-budget strategy: pass 1 bails fast on doomed lde levels;
/// budget-hit lde levels are queued for pass 2 with a much larger cap.
/// Preserves completeness — a budget-hit lde is never skipped.
const PASS1_CAP: u64 = 100_000_000;
const PASS2_CAP: u64 = 4_000_000_000;

/// CYCLOSYNTH_SCREEN_DIV: screen-lite budget divisor for the optimal
/// pipeline's stage-2 screen at deep ε (default 1 = full budgets).
/// A/B knob; becomes a constant once the sweep lands.
fn screen_div() -> u64 {
    use std::sync::OnceLock;
    static DIV: OnceLock<u64> = OnceLock::new();
    *DIV.get_or_init(|| {
        std::env::var("CYCLOSYNTH_SCREEN_DIV")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&d| d >= 1)
            .unwrap_or(1)
    })
}

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

/// Rayon `with_min_len` for `dc_search_q`'s **optimize-mode** prefix
/// par_iter. `0` = legacy `usable.len() / n_threads` chunking.
///
/// **A/B 2026-06-10 (1e-6 suite, 6 targets, seed 12648430):** `1`
/// (every prefix independently stealable) ABORTS — per-job `map_init`
/// scratch construction nests stolen `per_prefix` frames on rayon's
/// 2 MiB pool workers and overflows the stack (the coarse chunking only
/// survives because job count stays ≈ n_threads, bounding the nesting).
/// Keep 0; the cheap-prefix serialization issue is addressed by
/// [`OPTIMAL_PREFIX_INTERLEAVE`] instead, at unchanged job granularity.
const OPTIMAL_PAR_MIN_LEN: usize = 0;

/// Transpose-interleave the cost-sorted prefix list across rayon chunks
/// (chunk j gets cost ranks j, j+t, j+2t, …). Plain `len/n_threads`
/// chunking hands all the cheapest prefixes to one chunk, serializing
/// exactly the prefixes most likely to set the incumbent; interleaving
/// runs the t cheapest in parallel first so later prefixes see maximal
/// pruning. Stack-safe, unlike `with_min_len(1)` (above). 1.66× wall at
/// bit-identical costs.
const OPTIMAL_PREFIX_INTERLEAVE: bool = true;

/// Compute `U_L† · target` as a continuous Mat2.
/// `U_L` is exact (`U2Q`), `target` is float (`Mat2`). Mirrors the 8D
/// helper `clifford_t::u2t_dag_times_mat2` for use by the Z1 D&C path.
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
    /// Construct a synthesizer with ε-tuned defaults: Z1 D&C below 1e-6
    /// (single search becomes pathological at deeper ε) and BKZ-4 below
    /// 1e-7 (where the SE region is large enough to pay for the tighter
    /// Hermite factor).
    pub fn new(epsilon: f64) -> Self {
        // m=2 strict at deep ε (k_inner coverage); m=1 relaxed at 1e-6
        // (m=2 has structural gaps at low lde); single search above.
        let (dc_split, dc_dr_filter) = if epsilon <= 1e-7 {
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
            let cap = dc_pass1_cap_for(epsilon);
            let mult: u64 = if epsilon <= 1e-8 { 3 } else { 1 };
            (2, cap.saturating_mul(mult))
        } else {
            (1, 0)
        };

        Self {
            epsilon,
            min_lde,
            max_lde: max_lde_override,
            dc_split,
            dc_dr_filter,
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
            optimal_prefix_prune: true,
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

    /// Det-phase filter: only run the inner search for prefixes whose
    /// `d_R = (d_target − d_L) mod 16` is in the set (empty = no filter);
    /// the 16-valued analog of Clifford+T's `det_zeta_parity` check.
    /// Completeness caveat: a target's right factorization may not lie
    /// in any single d_R bucket — widening the set or iterating m covers
    /// more cases.
    pub fn with_dc_dr_filter(mut self, allowed_offsets: Vec<u32>) -> Self {
        self.dc_dr_filter = allowed_offsets;
        self
    }

    /// f64 GS state in LLL instead of MPFR. NS09's Theorem 2 doesn't
    /// cover d=16 in f64, but (per fplll's wrapper strategy) it
    /// converges and matches MPFR across our ε range, much faster.
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

    /// Toggle cost-optimal selection (vs first-hit). The enum-stage
    /// m-sweep stays owned by the constructor defaults; this only flips
    /// the flag.
    pub fn with_optimize_cost(mut self, on: bool) -> Self {
        self.optimize_cost = on;
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
    /// when `optimize_cost` is on. Default 2. Higher values reduce the
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

    /// Set (or clear) the anytime enum-stage deadline in milliseconds.
    /// See the `optimal_deadline_ms` field doc.
    pub fn with_optimal_deadline_ms(mut self, ms: Option<u64>) -> Self {
        self.optimal_deadline_ms = ms;
        self
    }

    /// Toggle certificate mode (see the `certify` field doc).
    pub fn with_certify(mut self, on: bool) -> Self {
        self.certify = on;
        self
    }

    /// Set the certify extension wall budget in milliseconds.
    pub fn with_certify_extra_ms(mut self, ms: u64) -> Self {
        self.certify_extra_ms = ms;
        self
    }

    /// Toggle the odd-Q-parity branch (see the field doc).
    pub fn with_odd_parity_branch(mut self, on: bool) -> Self {
        self.odd_parity_branch = on;
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
    /// **Backend**: hybrid — brute-force `phase1_brute` for `k ≤ BRUTE_LIMIT`
    /// (=3), then single-shot 16D L²-LLL + Schnorr-Euchner `phase1` (optionally
    /// BKZ-reduced) and an FGKM-prefix divide-and-conquer mode (`dc_search_q`)
    /// for larger / deep k.
    pub fn synthesize(&self, target: Mat2) -> Option<SynthResultQ> {
        self.synthesize_with_unclear(target, None)
    }

    /// First-hit node cap after the `budget_div` policy (min 1).
    fn cap_div(&self, base: u64) -> u64 {
        (base / self.budget_div.max(1)).max(1)
    }

    /// `max_lde` clamped by the live cross-parity incumbent when present
    /// (lde ≤ cost + 1 staircase bound); the incumbent tightens
    /// concurrently as the peer branch finds circuits.
    fn effective_max_lde(&self) -> u32 {
        let mut m = self.max_lde;
        if let Some(best) = &self.global_best_cost {
            let c = best.load(std::sync::atomic::Ordering::Relaxed);
            if c != usize::MAX {
                let c32 = c.min(u32::MAX as usize - 1) as u32;
                m = m.min(c32.saturating_add(1));
            }
        }
        m
    }

    /// [`Self::synthesize`] with an optional truncation out-param: a find
    /// at level `fl` short-circuits the pass-2 retry queue, so a
    /// truncated-and-never-cleared level below `fl` may still hold a
    /// solution. `unclear_out` receives exactly those levels so the
    /// cost-optimal enum stage can add them to its (lde, m) grid.
    fn synthesize_with_unclear(
        &self,
        target: Mat2,
        mut unclear_out: Option<&mut Vec<u32>>,
    ) -> Option<SynthResultQ> {
        use crate::synthesis::diag;
        use crate::synthesis::lattice_zeta::{set_verify_prune_mpfr, verify_prune_mpfr};
        crate::synthesis::ensure_rayon_stack();

        // Land the det exactly on a ζ₁₆ power first (lossless, see
        // `project_det_to_zeta_coset`) — generic U(2) inputs otherwise
        // carry a residual phase no completion can absorb.
        let target = project_det_to_zeta_coset(&target);

        if self.optimize_cost {
            return self.synthesize_optimal(target);
        }

        let trace = diag::trace_enabled();
        if trace {
            diag::reset_all();
        }

        // Auto-enable MPFR prune verification below 2e-8: near 1.5e-8
        // the f64 partial-Euclidean prune suffers catastrophic
        // cancellation and silently drops valid candidates. Restore the
        // prior flag on exit so other paths aren't affected.
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

        // `should_stop` short-circuits the walker on the first ε-close
        // leaf; optimize_cost returns false unconditionally so every
        // ε-close leaf is enumerated and check_sols picks the cheapest.
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
            let budget_hit = AtomicBool::new(false);
            let should_stop = |x: &[i64; 16]| -> bool {
                if optimize_cost { return false; }
                let cand = solution_to_u2q_d(x, k, d);
                diamond_distance_u2q_float(&cand, &target) < epsilon
            };
            let sols = phase1_deep_aware(
                s.as_mut(), v, None, self.deep_rot_src.as_ref(), k, epsilon, budget, &budget_hit, should_stop, None, None,
            );
            (sols, budget_hit.load(std::sync::atomic::Ordering::Relaxed))
        };

        let check_sols = |sols: &[[i64; 16]], k: u32| -> Option<SynthResultQ> {
            let cands = sols.iter().map(|sol| (solution_to_u2q_d(sol, k, d), k));
            self.best_costed(cands, &target, !optimize_cost).map(|(_, r)| r)
        };

        // Brute regime: iterate every k for exact small-T Clifford+√T finds.
        let zd = Complex64::from_polar(1.0, d as f64 * PI / 8.0);
        for k in self.min_lde..=BRUTE_LIMIT.min(self.max_lde) {
            let t_k = std::time::Instant::now();
            let shell = brute_shell_cached(k);
            let thr = brute_prefilter_threshold(self.epsilon);
            let close: Vec<[i64; 16]> = shell
                .sols
                .iter()
                .zip(&shell.mats)
                .filter(|(_, m)| brute_dist_est(m, zd, &target) < thr)
                .map(|(s, _)| *s)
                .collect();
            let r = check_sols(&close, k);
            if trace {
                eprintln!("[zeta] brute lde={k:>2}  sols={:>7} close={:>3}  {}  t={:.0}ms",
                    shell.sols.len(), close.len(),
                    if r.is_some() { "FOUND" } else { "none " },
                    t_k.elapsed().as_secs_f64() * 1000.0);
            }
            if let Some(r) = r {
                if trace {
                    diag::dump_zeta(&diag::snapshot(),
                        &format!("synthesize ε={:.0e} k={k}", self.epsilon));
                }
                return Some(r);
            }
        }

        // 2-pass dispatcher: pass 1 bails fast on doomed levels;
        // budget-hit levels are requeued at the pass-2 cap so a
        // budget-truncated lde is never silently skipped (min-lde
        // correctness) while easy targets stay cheap.
        if let Some(m_split) = self.dc_split {
            // Budget-hit levels the pass-2 queue will never retry (it
            // covers the main sweep, not this fallback; a find aborts
            // the queue) — reported through `unclear_out`.
            let mut unverified_small: Vec<u32> = Vec::new();
            // Sequential small-k pass: dc_search_q cannot help for k <= m_split
            // (k_inner ≤ 0). These are typically few levels near lattice_start.
            for k in lattice_start..=m_split.min(self.max_lde) {
                let t_k = std::time::Instant::now();
                let (sols, small_budget_hit) = try_lattice_k(k, self.cap_div(PASS1_CAP), &mut scratch);
                if let Some(r) = check_sols(&sols, k) {
                    if trace {
                        eprintln!("[zeta] dc lde={k:>2} (single fallback)  FOUND  dist={:.3e}  t={:.0}ms",
                            r.distance, t_k.elapsed().as_secs_f64() * 1000.0);
                    }
                    if let Some(out) = unclear_out.as_deref_mut() {
                        out.extend(unverified_small.iter().copied());
                    }
                    return Some(r);
                }
                if small_budget_hit {
                    unverified_small.push(k);
                }
                if trace {
                    eprintln!("[zeta] dc lde={k:>2} (single fallback)  none   t={:.0}ms",
                        t_k.elapsed().as_secs_f64() * 1000.0);
                }
            }

            use std::sync::Mutex;
            let pass2_collector: Mutex<Vec<u32>> = Mutex::new(Vec::new());

            // window == 1 takes a plain sequential loop with zero
            // speculation machinery: the shared consumed-counter alone
            // costs 30-50% wall on shallow-ε million-node walks, and
            // speculation only pays where no-solution levels burn
            // seconds (deep ε).
            if self.parallel_lde_window <= 1 {
                for k in (m_split + 1).max(lattice_start)..=self.max_lde {
                    if k > self.effective_max_lde() {
                        break;
                    }
                    let t_k = std::time::Instant::now();
                    let (result, budget_hit) = self.dc_search_q(
                        &target, k, m_split, None, self.cap_div(dc_pass1_cap_for(self.epsilon)),
                        None, None, None, None,
                    );
                    if let Some(r) = result {
                        if trace {
                            eprintln!("[zeta] dc lde={k:>2} m={m_split} pass1  FOUND  dist={:.3e}  t={:.0}ms",
                                r.distance, t_k.elapsed().as_secs_f64() * 1000.0);
                        }
                        // Find at k short-circuits the pass-2 retries:
                        // every queued (budget-hit) level < k stays
                        // unverified — report it for the enum grid.
                        if let Some(out) = unclear_out.as_deref_mut() {
                            out.extend(unverified_small.iter().copied());
                            out.extend(pass2_collector.lock().unwrap().iter().copied());
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
                // Levels retried at pass-2 cap that hit the budget AGAIN
                // without finding: still not exhaustively walked.
                let mut still_truncated: Vec<u32> = Vec::new();
                for k in pass2_queue {
                    if k > self.effective_max_lde() {
                        break;
                    }
                    let t_k = std::time::Instant::now();
                    if trace {
                        eprintln!("[zeta] dc lde={k:>2} m={m_split} pass2 dispatching ...");
                    }
                    let (result, budget_hit2) = self.dc_search_q(
                        &target, k, m_split, None, self.cap_div(dc_pass2_cap_for(self.epsilon)), None, None, None, None,
                    );
                    if let Some(r) = result {
                        if trace {
                            eprintln!("[zeta] dc lde={k:>2} m={m_split} pass2  FOUND  dist={:.3e}  t={:.0}ms",
                                r.distance, t_k.elapsed().as_secs_f64() * 1000.0);
                        }
                        if let Some(out) = unclear_out.as_deref_mut() {
                            out.extend(unverified_small.iter().copied());
                            out.extend(still_truncated.iter().copied());
                        }
                        return Some(r);
                    }
                    if budget_hit2 {
                        still_truncated.push(k);
                    }
                    if trace {
                        eprintln!("[zeta] dc lde={k:>2} m={m_split} pass2  none   t={:.0}ms",
                            t_k.elapsed().as_secs_f64() * 1000.0);
                    }
                }
                return None;
            }

            // window ≥ 2: speculate a window of lde levels concurrently;
            // the first find aborts in-flight peers. Hard-target wall
            // drops from "sum of no-sol burns + find" to "find alone",
            // paid for by thread dilution on easy targets — hence only
            // enabled where hard targets overshoot the predicted lde.
            let cross_lde_abort = AtomicBool::new(false);
            let lde_window_size: u32 = self.parallel_lde_window.max(1);
            let mut k_cursor = (m_split + 1).max(lattice_start);

            let parallel_result: Option<SynthResultQ> = 'outer: loop {
                if k_cursor > self.effective_max_lde() { break 'outer None; }
                if cross_lde_abort.load(Ordering::Relaxed) { break 'outer None; }

                let window_end = (k_cursor + lde_window_size - 1).min(self.max_lde);
                let lde_window: Vec<u32> = (k_cursor..=window_end).collect();
                if trace {
                    eprintln!("[zeta] dc m={m_split} pass1 parallel-lde window={:?} dispatching ...", lde_window);
                }
                let t_window = std::time::Instant::now();

                // Staggered speculation: task i waits for task i-1 to
                // burn `trigger_nodes` without finding (0 = launch
                // immediately).
                let trigger_nodes = self.parallel_lde_trigger_nodes;
                let consumed_counters: Vec<std::sync::Arc<std::sync::atomic::AtomicU64>> =
                    (0..lde_window.len())
                        .map(|_| std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)))
                        .collect();
                // LIVENESS: the gate below must also wake when its
                // predecessor FINISHES — a level can complete cleanly
                // below the trigger, and a successor polling a counter
                // that permanently stopped deadlocks the process.
                let finished_flags: Vec<std::sync::Arc<std::sync::atomic::AtomicBool>> =
                    (0..lde_window.len())
                        .map(|_| std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)))
                        .collect();
                let results: Mutex<Vec<(u32, Option<SynthResultQ>, bool)>> =
                    Mutex::new(Vec::new());
                std::thread::scope(|s| {
                    for (i, &k) in lde_window.iter().enumerate() {
                        let results_ref = &results;
                        let abort_ref = &cross_lde_abort;
                        let pass2_ref = &pass2_collector;
                        let my_consumed = consumed_counters[i].clone();
                        let my_finished = finished_flags[i].clone();
                        let predecessor_consumed: Option<std::sync::Arc<std::sync::atomic::AtomicU64>> =
                            if i > 0 { Some(consumed_counters[i - 1].clone()) } else { None };
                        let predecessor_finished: Option<std::sync::Arc<std::sync::atomic::AtomicBool>> =
                            if i > 0 { Some(finished_flags[i - 1].clone()) } else { None };
                        s.spawn(move || {
                            // RAII: mark this task finished on EVERY exit
                            // path (normal, abort early-return, panic) so
                            // a successor's gate can never be stranded.
                            struct FinishedGuard(std::sync::Arc<std::sync::atomic::AtomicBool>);
                            impl Drop for FinishedGuard {
                                fn drop(&mut self) {
                                    self.0.store(true, Ordering::Release);
                                }
                            }
                            let _finished_guard = FinishedGuard(my_finished);
                            // Wait for predecessor to consume `trigger_nodes`
                            // search-tree nodes, FINISH (a clean empty
                            // completion below the trigger means this level
                            // should start immediately), or cross-LDE abort.
                            if i > 0 && trigger_nodes > 0 {
                                let pred = predecessor_consumed.as_ref().unwrap();
                                let pred_done = predecessor_finished.as_ref().unwrap();
                                loop {
                                    if abort_ref.load(Ordering::Relaxed) { return; }
                                    if pred.load(Ordering::Relaxed) >= trigger_nodes { break; }
                                    if pred_done.load(Ordering::Acquire) { break; }
                                    std::thread::sleep(std::time::Duration::from_millis(50));
                                }
                                if abort_ref.load(Ordering::Relaxed) { return; }
                            }
                            let t_k = std::time::Instant::now();
                            // Pass shared signals only when they can
                            // fire: the walker pays a contended atomic
                            // per recurse-enter if either is Some.
                            let abort_opt = if lde_window_size > 1 { Some(abort_ref) } else { None };
                            let consumed_opt = if trigger_nodes > 0 {
                                Some(my_consumed.as_ref())
                            } else {
                                None
                            };
                            let (result, budget_hit) = self.dc_search_q(
                                &target, k, m_split, None, self.cap_div(dc_pass1_cap_for(self.epsilon)),
                                abort_opt,
                                consumed_opt,
                                None, None,
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
                    if let Some(out) = unclear_out.as_deref_mut() {
                        let found_k = r.lde;
                        out.extend(unverified_small.iter().copied());
                        // Budget-hit levels from this and earlier windows.
                        out.extend(
                            pass2_collector.lock().unwrap().iter().copied()
                                .filter(|&k| k < found_k),
                        );
                        // Window peers below the finder: a peer may have
                        // been cross-LDE-aborted mid-walk (or never
                        // launched behind the speculation trigger), which
                        // is indistinguishable here from a clean exhaust —
                        // conservatively report every non-finding peer
                        // level below found_k.
                        out.extend(lde_window.iter().copied().filter(|&k| k < found_k));
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
            let mut still_truncated: Vec<u32> = Vec::new();
            for k in pass2_queue {
                if k > self.effective_max_lde() {
                    break;
                }
                let t_k = std::time::Instant::now();
                if trace {
                    eprintln!("[zeta] dc lde={k:>2} m={m_split} pass2 dispatching ...");
                }
                let (result, budget_hit2) = self.dc_search_q(&target, k, m_split, None, self.cap_div(dc_pass2_cap_for(self.epsilon)), None, None, None, None);
                if let Some(r) = result {
                    if trace {
                        eprintln!("[zeta] dc lde={k:>2} m={m_split} pass2  FOUND  dist={:.3e}  t={:.0}ms",
                            r.distance, t_k.elapsed().as_secs_f64() * 1000.0);
                    }
                    if let Some(out) = unclear_out.as_deref_mut() {
                        out.extend(unverified_small.iter().copied());
                        out.extend(still_truncated.iter().copied());
                    }
                    return Some(r);
                }
                if budget_hit2 {
                    still_truncated.push(k);
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
            if k > self.effective_max_lde() {
                break;
            }
            let t_k = std::time::Instant::now();
            let (sols, budget_was_hit) = try_lattice_k(k, self.cap_div(PASS1_CAP), &mut scratch);
            if let Some(r) = check_sols(&sols, k) {
                if trace {
                    eprintln!("[zeta] pass1 lde={k:>2}  FOUND  dist={:.3e}  t={:.0}ms",
                        r.distance, t_k.elapsed().as_secs_f64() * 1000.0);
                    diag::dump_zeta(&diag::snapshot(),
                        &format!("synthesize ε={:.0e} k={k} (pass1)", self.epsilon));
                }
                // Queued (budget-hit) levels < k never get their pass-2
                // retry — same upward-bias class as the dc dispatcher.
                if let Some(out) = unclear_out.as_deref_mut() {
                    out.extend(pass2_queue.iter().copied());
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
        let mut still_truncated: Vec<u32> = Vec::new();
        for k in pass2_queue {
            if k > self.effective_max_lde() {
                break;
            }
            let t_k = std::time::Instant::now();
            let (sols, budget_hit2) = try_lattice_k(k, self.cap_div(PASS2_CAP), &mut scratch);
            if let Some(r) = check_sols(&sols, k) {
                if trace {
                    eprintln!("[zeta] pass2 lde={k:>2}  FOUND  dist={:.3e}  t={:.0}ms",
                        r.distance, t_k.elapsed().as_secs_f64() * 1000.0);
                    diag::dump_zeta(&diag::snapshot(),
                        &format!("synthesize ε={:.0e} k={k} (pass2)", self.epsilon));
                }
                if let Some(out) = unclear_out.as_deref_mut() {
                    out.extend(still_truncated.iter().copied());
                }
                return Some(r);
            }
            if budget_hit2 {
                still_truncated.push(k);
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

    /// Z[ζ_16] analog of Clifford+T's `dc_search`: for each prefix
    /// `U_L ∈ L_m^Q`, search the inner factor at `k_total − k_prefix` and
    /// compose; `d_R = (d_target − d_L) mod 16` parametrises the inner
    /// reconstruction so `U_L · U_R` matches the target's det phase.
    /// The returned bool reports any budget-hit prefix so the 2-pass
    /// dispatcher knows a deeper retry could still find something.
    /// Prefixes run under rayon with per-worker scratch; nested SE
    /// parallelism over-subscribes the pool, which work-stealing handles.
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
        shared_best_cost: Option<&std::sync::atomic::AtomicUsize>,
    ) -> (Option<SynthResultQ>, bool) {
        use rayon::prelude::*;
        use crate::synthesis::diag;

        let prefixes = build_l_q(m_split);
        let q_cost_x2 = self.q_cost_x2;
        let prefix_costs: Vec<usize> = build_l_q_tq(m_split)
            .iter()
            .map(|&(t, q)| 2 * t + q_cost_x2 * q)
            .collect();
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
        let mut cand_idx: Vec<(usize, usize)> = prefixes
            .iter()
            .enumerate()
            .filter(|(_, u_l)| u_l.k < k_total)
            .filter(|(_, u_l)| {
                if dc_dr_filter.is_empty() {
                    return true;
                }
                let d_l = det_phase_of(&u_l.to_float());
                let d_r = ((d_target as i32 - d_l as i32).rem_euclid(16)) as u32;
                dc_dr_filter.contains(&d_r)
            })
            .map(|(i, _)| (i, prefix_costs[i]))
            .collect();

        // Right-coset dedup: one min-cost rep per (orbit, k) class of
        // the usable set. The per-prefix cap scales by the dedup ratio
        // so the total leaf budget per orbit is invariant — without
        // that, the rep gets ONE cap-bounded draw where the orbit had
        // `ratio` independent ones, and the racy leaf-visit order can
        // flip a near-cap find to budget-hit. Exhausted walks (the
        // common no-solution case) are unaffected, preserving the wall
        // win.
        let mut per_prefix_cap = per_prefix_cap;
        if *ZETA_COSET_DEDUP && cand_idx.len() > 1 {
            let pre = cand_idx.len();
            let keys = build_l_q_coset_keys(m_split);
            let mask = coset_keep_mask(&cand_idx, &keys);
            let mut it = mask.iter();
            cand_idx.retain(|_| *it.next().unwrap());
            let post = cand_idx.len().max(1);
            let ratio = (pre.div_ceil(post)) as u64;
            per_prefix_cap = per_prefix_cap.saturating_mul(ratio.max(1));
        }

        let mut usable: Vec<(&U2Q, usize)> = cand_idx
            .into_iter()
            .map(|(i, c)| (&prefixes[i], c))
            .collect();

        if usable.is_empty() {
            return (None, false);
        }

        let optimize_cost = cost_min_override.unwrap_or(self.optimize_cost);
        let prefix_prune = self.optimal_prefix_prune;

        // Optimal mode sorts cheapest-first so the shared incumbent
        // drops quickly; first-hit keeps k_prefix-desc (small k_inner =
        // fast bail or hit).
        let n_threads = rayon::current_num_threads().max(1);
        if optimize_cost {
            usable.sort_by_key(|(_, c)| *c);
            // See `OPTIMAL_PREFIX_INTERLEAVE`: deal the cost-sorted list
            // round-robin across ~n_threads strides so each rayon chunk
            // covers the whole cost spectrum (cheapest first) instead of
            // one chunk hoarding every cheap prefix.
            let n = usable.len();
            if OPTIMAL_PREFIX_INTERLEAVE && n > n_threads {
                let mut interleaved: Vec<(&U2Q, usize)> = Vec::with_capacity(n);
                for j in 0..n_threads {
                    let mut idx = j;
                    while idx < n {
                        interleaved.push(usable[idx]);
                        idx += n_threads;
                    }
                }
                usable = interleaved;
            }
        } else {
            usable.sort_by(|(a, _), (b, _)| b.k.cmp(&a.k));
        }

        let chunk = (usable.len() / n_threads).max(1);
        let opt_chunk = if OPTIMAL_PAR_MIN_LEN == 0 { chunk } else { OPTIMAL_PAR_MIN_LEN };

        // Node-level incumbent abort: a watcher flags in-flight prefixes
        // whose static floor (cost(U_L) + class_cost_lb(d_R)) can no
        // longer beat the incumbent, killing hopeless walks mid-tree —
        // the leaf-level check alone never fires on walks that produce
        // no ε-close leaf. Sound: only cuts walks whose every candidate
        // costs ≥ the incumbent.
        struct PrefixWatch {
            abort: AtomicBool,
            active: AtomicBool,
            floor: usize,
        }
        let watches: Vec<PrefixWatch> = if optimize_cost {
            usable
                .iter()
                .map(|&(u_l, c)| {
                    let d_l = det_phase_of(&u_l.to_float());
                    let d_r = ((d_target as i32 - d_l as i32).rem_euclid(16)) as u32;
                    PrefixWatch {
                        abort: AtomicBool::new(false),
                        active: AtomicBool::new(false),
                        floor: c.saturating_add(
                            crate::synthesis::cost_bound::class_cost_lb_half_units(d_r),
                        ),
                    }
                })
                .collect()
        } else {
            Vec::new()
        };

        // Shared best-cost tracker; a caller-supplied atomic lets all
        // concurrent dc_search_q calls prune against one (pre-seeded)
        // global incumbent.
        let local_best_cost = std::sync::atomic::AtomicUsize::new(usize::MAX);
        let best_cost: &std::sync::atomic::AtomicUsize =
            shared_best_cost.unwrap_or(&local_best_cost);

        let per_prefix = |scratch: &mut IntScratch16,
                          idx: usize,
                          entry: &(&U2Q, usize)|
         -> Option<(usize, SynthResultQ)> {
            let (u_l, u_l_cost) = (entry.0, entry.1);
            let k_prefix = u_l.k;
            let k_inner = k_total - k_prefix;

            // d_L from prefix's float det.
            let d_l = det_phase_of(&u_l.to_float());
            let d_r = ((d_target as i32 - d_l as i32).rem_euclid(16)) as u32;

            if optimize_cost && prefix_prune {
                let cur_best = best_cost.load(std::sync::atomic::Ordering::Relaxed);
                // Sound because syllable costs are additive in normal
                // form: any U cheaper than `best` is reachable through
                // its canonical prefix with cost ≤ best − LB(suffix).
                // Only the det-phase Q-parity bound is a valid suffix
                // LB — L(k_inner) is NOT: the k_inner shell contains
                // √2-scaled images of every lower-lde suffix, which can
                // cost far less.
                let suffix_lb =
                    crate::synthesis::cost_bound::class_cost_lb_half_units(d_r);
                if u_l_cost.saturating_add(suffix_lb) > cur_best {
                    return None;
                }
            }

            // m_inner = U_L† · target as a continuous Mat2.
            let m_inner = u2q_dag_times_mat2(u_l, target);
            let v_inner = unitary_to_uv_zeta(&m_inner);

            let budget_hit = AtomicBool::new(false);
            let u_l_local = *u_l;
            let target_local = *target;
            let capture = diag::capture_enabled();
            let suffix_floor =
                crate::synthesis::cost_bound::class_cost_lb_half_units(d_r);
            let should_stop = |x: &[i64; 16]| -> bool {
                if optimize_cost {
                    // Stop the walk once the incumbent reaches this
                    // prefix's floor — only skips candidates costing ≥
                    // the incumbent (checked at leaf hits, free).
                    return best_cost.load(std::sync::atomic::Ordering::Relaxed)
                        <= u_l_cost.saturating_add(suffix_floor);
                }
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

            // Optimize mode routes the walker's abort signal through this
            // prefix's own flag (set by the incumbent watcher; it also
            // mirrors `external_abort` if the caller passed one).
            // First-hit mode passes the caller's signal straight through.
            let walk_abort: Option<&AtomicBool> = if optimize_cost {
                let w = &watches[idx];
                w.active.store(true, Ordering::Relaxed);
                Some(&w.abort)
            } else {
                external_abort
            };

            let sols = phase1_deep_aware(
                scratch, v_inner, Some((u_l, target)), self.deep_rot_src.as_ref(), k_inner, epsilon,
                per_prefix_cap, &budget_hit, should_stop,
                walk_abort, consumed,
            );
            if optimize_cost {
                watches[idx].active.store(false, Ordering::Relaxed);
            }

            if budget_hit.load(std::sync::atomic::Ordering::Relaxed) {
                any_budget_hit.store(true, std::sync::atomic::Ordering::Relaxed);
            }

            // First-hit returns the first ε-close sol; optimal keeps the
            // min-cost one and publishes it for the prefix prune.
            let mut best: Option<(usize, SynthResultQ)> = None;
            for sol in &sols {
                let u_r = solution_to_u2q_d(sol, k_inner, d_r);
                let u_full = u_l_local * u_r;
                let dist = diamond_distance_u2q_float(&u_full, target);
                if dist < epsilon {
                    let gates = BlochDecomposer.decompose(&u_full);
                    let cost = gates_cost(&gates, q_cost_x2);
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
                    // Relaxed is enough: the prune is a heuristic.
                    best_cost.fetch_min(*c, std::sync::atomic::Ordering::Relaxed);
                }
            }
            best
        };

        // Boxed so the per-worker scratch lives on the heap — rayon's
        // in-place execution can run these closures on the caller's
        // (possibly small) thread stack.
        let make_scratch = || {
            let mut s = Box::new(IntScratch16::new(epsilon));
            s.use_f64_gs = use_f64_gs;
            s.bkz_block_size = bkz_block_size;
            s
        };

        let result_pair: Option<(usize, SynthResultQ)> = if optimize_cost {
            // Min-cost reduce across prefixes; the scoped watcher kills
            // walks whose floor can no longer beat the incumbent.
            let walks_done = AtomicBool::new(false);
            // RAII: set `walks_done` even if the par_iter panics —
            // otherwise `thread::scope` would join a watcher that never
            // exits (deadlock on unwind).
            struct DoneGuard<'a>(&'a AtomicBool);
            impl Drop for DoneGuard<'_> {
                fn drop(&mut self) {
                    self.0.store(true, Ordering::Relaxed);
                }
            }
            std::thread::scope(|wscope| {
                let _done_guard = DoneGuard(&walks_done);
                let watches_ref = &watches;
                let walks_done_ref = &walks_done;
                wscope.spawn(move || {
                    while !walks_done_ref.load(Ordering::Relaxed) {
                        std::thread::sleep(std::time::Duration::from_millis(20));
                        let cur_best =
                            best_cost.load(std::sync::atomic::Ordering::Relaxed);
                        let ext = external_abort
                            .map(|a| a.load(Ordering::Relaxed))
                            .unwrap_or(false);
                        for w in watches_ref {
                            if w.active.load(Ordering::Relaxed)
                                && (ext || cur_best <= w.floor)
                            {
                                w.abort.store(true, Ordering::Relaxed);
                            }
                        }
                    }
                });
                let r = usable
                    .par_iter()
                    .enumerate()
                    .with_min_len(opt_chunk)
                    .map_init(make_scratch, |s, (i, e)| per_prefix(s, i, e))
                    .reduce(
                        || None,
                        |a, b| match (a, b) {
                            (None, x) | (x, None) => x,
                            (Some(a), Some(b)) => Some(if a.0 <= b.0 { a } else { b }),
                        },
                    );
                walks_done.store(true, Ordering::Relaxed);
                r
            })
        } else {
            // First-hit: abort other prefixes as soon as one finds.
            usable
                .par_iter()
                .enumerate()
                .with_min_len(chunk)
                .map_init(make_scratch, |s, (i, e)| per_prefix(s, i, e))
                .find_map_any(|x| x)
        };
        let result = result_pair.map(|(_, r)| r);

        let budget_hit = any_budget_hit.load(std::sync::atomic::Ordering::Relaxed);
        (result, budget_hit)
    }

    /// Dispatch the (k, m ≥ 1) arms to the merged frontier under the
    /// deadline — as sequential lowest-m-first phases below 1e-7
    /// (interleaved, m=2's ~6× fan-out starves the deep m=1 units that
    /// hold the decisive finds; the incumbent carries forward as each
    /// next phase's prune floor), interleaved above (measured better at
    /// 1e-6/1e-7). Phase shares roll forward by default: a phase whose
    /// frontier finishes early (incumbent-pruned levels) donates its
    /// surplus to later phases — beat or tied every hand-tuned split.
    /// Env: CYCLOSYNTH_SEQ_M (1/0 forces), CYCLOSYNTH_SEQ_M_SPLIT
    /// (explicit per-phase ms), CYCLOSYNTH_SEQ_ROLLFWD=0.
    fn run_frontier(
        &self,
        target: &Mat2,
        tasks: &[(u32, u32)],
        deadline_ms: u64,
        shared_best: &std::sync::atomic::AtomicUsize,
    ) -> (Option<(usize, SynthResultQ)>, Vec<bool>) {
        let seq_m = match std::env::var("CYCLOSYNTH_SEQ_M").as_deref() {
            Ok("1") => true,
            Ok("0") => false,
            _ => self.epsilon < 1e-7,
        };
        let mut m_groups: Vec<u32> = tasks.iter().map(|&(_, m)| m).collect();
        m_groups.sort_unstable();
        m_groups.dedup();
        if !seq_m || m_groups.len() <= 1 {
            return self.dc_frontier_q(
                target,
                tasks,
                std::time::Duration::from_millis(deadline_ms),
                shared_best,
            );
        }

        let split: Vec<u64> = std::env::var("CYCLOSYNTH_SEQ_M_SPLIT")
            .ok()
            .map(|s| s.split(',').filter_map(|p| p.trim().parse().ok()).collect())
            .unwrap_or_default();
        let equal_share = (deadline_ms / m_groups.len() as u64).max(1);
        let rollfwd = split.is_empty()
            && std::env::var("CYCLOSYNTH_SEQ_ROLLFWD").as_deref() != Ok("0");
        let t_phases = std::time::Instant::now();
        let mut best_fr: Option<(usize, SynthResultQ)> = None;
        let mut trunc_by_task: Vec<((u32, u32), bool)> = Vec::new();
        for (gi, &mg) in m_groups.iter().enumerate() {
            let share = if rollfwd {
                let left = deadline_ms
                    .saturating_sub(t_phases.elapsed().as_millis() as u64);
                (left / (m_groups.len() - gi) as u64).max(1)
            } else {
                split
                    .get(gi)
                    .or(split.last())
                    .copied()
                    .unwrap_or(equal_share)
                    .max(1)
            };
            let group: Vec<(u32, u32)> =
                tasks.iter().copied().filter(|&(_, m)| m == mg).collect();
            let (g_fr, g_tr) = self.dc_frontier_q(
                target,
                &group,
                std::time::Duration::from_millis(share),
                shared_best,
            );
            trunc_by_task.extend(group.iter().copied().zip(g_tr));
            if let Some((c, r)) = g_fr {
                if best_fr.as_ref().is_none_or(|(bc, _)| c < *bc) {
                    best_fr = Some((c, r));
                }
            }
        }
        let truncated = tasks
            .iter()
            .map(|t| {
                trunc_by_task
                    .iter()
                    .find(|(tt, _)| tt == t)
                    .map(|&(_, tr)| tr)
                    .unwrap_or(true)
            })
            .collect();
        (best_fr, truncated)
    }

    /// Anytime merged-frontier enum stage (fast path, certify off): the
    /// prefix work-units of every (k, m) arm, tagged with the sound
    /// floor `cost(U_L) + class_cost_lb(d_R)` (one currency across
    /// arms), sorted floor-ascending (k-ascending tie-break: smaller SE
    /// regions drop the incumbent faster), transpose-interleaved across
    /// chunks, and stopped by deadline or floor-exhaustion — both cut
    /// only candidates costing ≥ the incumbent. A large per-prefix node
    /// cap backstops pathological prefixes. `cost_lb(k_inner)` is NOT in
    /// the floor (unsound — see `dc_search_q`).
    ///
    /// Returns the min-cost find plus a per-level truncation flag
    /// (parallel to `levels`): a level is marked truncated when any of
    /// its units was deadline-skipped, deadline-aborted, or hit the
    /// backstop cap. Conservative over-marking (a walk that finished
    /// cleanly right at the deadline may be marked) keeps the ledger
    /// honest; sound floor-kills are NOT truncation, as today.
    fn dc_frontier_q(
        &self,
        target: &Mat2,
        levels: &[(u32, u32)],
        deadline: std::time::Duration,
        shared_best_cost: &std::sync::atomic::AtomicUsize,
    ) -> (Option<(usize, SynthResultQ)>, Vec<bool>) {
        use rayon::prelude::*;

        let q_cost_x2 = self.q_cost_x2;
        let d_target = det_phase_of(target);
        let epsilon = self.epsilon;
        let use_f64_gs = self.use_f64_gs;
        let bkz_block_size = self.bkz_block_size;
        let prefix_prune = self.optimal_prefix_prune;
        let best_cost = shared_best_cost;
        let start = std::time::Instant::now();

        // Backstop node cap per unit — generous (the deadline is the
        // primary stop), but bounded so one pathological prefix can't
        // monopolise the frontier.
        let per_prefix_cap = dc_pass2_cap_for(epsilon)
            .saturating_mul(self.optimal_budget_multiplier.max(1));

        // Keep the per-m prefix caches alive for the unit borrows below.
        let level_prefixes: Vec<Arc<Vec<U2Q>>> =
            levels.iter().map(|&(_, m)| build_l_q(m)).collect();
        let level_costs: Vec<Arc<Vec<(usize, usize)>>> =
            levels.iter().map(|&(_, m)| build_l_q_tq(m)).collect();

        #[derive(Clone, Copy)]
        struct Unit<'a> {
            u_l: &'a U2Q,
            k_total: u32,
            d_r: u32,
            /// `cost(U_L) + class_cost_lb_half_units(d_R)` — the sound
            /// per-prefix bound from `dc_search_q`, in the half-unit
            /// currency shared by every (k, m) arm.
            floor: usize,
            level_idx: usize,
        }

        let truncated: Vec<AtomicBool> =
            levels.iter().map(|_| AtomicBool::new(false)).collect();

        let mut units: Vec<Unit> = Vec::new();
        for (li, &(k_total, m)) in levels.iter().enumerate() {
            // Mirror `try_optimal_variant`: m ≥ k arms don't run (the
            // D&C split needs k_inner ≥ 1 for every prefix).
            if m == 0 || m >= k_total {
                continue;
            }
            // Same filter the task grid uses: open at ε ≤ 1e-5, else
            // the per-m first-hit defaults.
            let filter = if self.optimal_open_dr_filter {
                Vec::new()
            } else {
                default_dc_dr_filter(m)
            };
            // (prefix index, d_R, floor) candidates for this level.
            let mut cands: Vec<(usize, u32, usize)> = Vec::new();
            for (pi, (u_l, &(t, q))) in level_prefixes[li]
                .iter()
                .zip(level_costs[li].iter())
                .enumerate()
            {
                if u_l.k >= k_total {
                    continue;
                }
                let d_l = det_phase_of(&u_l.to_float());
                let d_r = ((d_target as i32 - d_l as i32).rem_euclid(16)) as u32;
                if !filter.is_empty() && !filter.contains(&d_r) {
                    continue;
                }
                let u_l_cost = 2 * t + q_cost_x2 * q;
                let floor = u_l_cost.saturating_add(
                    crate::synthesis::cost_bound::class_cost_lb_half_units(d_r),
                );
                cands.push((pi, d_r, floor));
            }
            // Right-coset dedup of this arm's post-filter set: one rep
            // per (orbit, k) class ∩ usable, the min-floor member (the
            // floor is the frontier's sort/prune currency).
            // CYCLOSYNTH_ZETA_COSET=0 disables. See `coset_keep_mask`.
            if *ZETA_COSET_DEDUP && cands.len() > 1 {
                let keys = build_l_q_coset_keys(m);
                let iw: Vec<(usize, usize)> =
                    cands.iter().map(|&(pi, _, f)| (pi, f)).collect();
                let mask = coset_keep_mask(&iw, &keys);
                let mut it = mask.iter();
                cands.retain(|_| *it.next().unwrap());
            }
            for (pi, d_r, floor) in cands {
                units.push(Unit {
                    u_l: &level_prefixes[li][pi],
                    k_total,
                    d_r,
                    floor,
                    level_idx: li,
                });
            }
        }

        if units.is_empty() {
            return (None, truncated.into_iter().map(|t| t.into_inner()).collect());
        }

        // Global ascending floor sort; tie-break k ascending (smaller SE
        // regions complete sooner → incumbent drops faster). Then the
        // cost-rank transpose-interleave across rayon's chunking.
        units.sort_by(|a, b| a.floor.cmp(&b.floor).then(a.k_total.cmp(&b.k_total)));
        let n_threads = rayon::current_num_threads().max(1);
        let n = units.len();
        if OPTIMAL_PREFIX_INTERLEAVE && n > n_threads {
            let mut interleaved: Vec<Unit> = Vec::with_capacity(n);
            for j in 0..n_threads {
                let mut idx = j;
                while idx < n {
                    interleaved.push(units[idx]);
                    idx += n_threads;
                }
            }
            units = interleaved;
        }
        let chunk = (units.len() / n_threads).max(1);
        let opt_chunk = if OPTIMAL_PAR_MIN_LEN == 0 { chunk } else { OPTIMAL_PAR_MIN_LEN };

        // Per-unit watch: the watcher enforces both the sound
        // incumbent-floor kill (as in `dc_search_q`) and the deadline
        // abort (which additionally marks the unit's level truncated —
        // the watcher is the only place that knows WHY it killed a walk).
        struct PrefixWatch {
            abort: AtomicBool,
            active: AtomicBool,
            floor: usize,
        }
        let watches: Vec<PrefixWatch> = units
            .iter()
            .map(|u| PrefixWatch {
                abort: AtomicBool::new(false),
                active: AtomicBool::new(false),
                floor: u.floor,
            })
            .collect();

        let per_unit = |scratch: &mut IntScratch16,
                        idx: usize,
                        u: &Unit|
         -> Option<(usize, SynthResultQ)> {
            // (a) deadline pre-dispatch: never-started units leave their
            // level truncated (work provably remained at the cutoff).
            if start.elapsed() >= deadline {
                truncated[u.level_idx].store(true, Ordering::Relaxed);
                return None;
            }
            // (b) floor-exhaustion: sound prune, NOT truncation.
            if prefix_prune
                && best_cost.load(std::sync::atomic::Ordering::Relaxed) <= u.floor
            {
                return None;
            }

            let k_inner = u.k_total - u.u_l.k;
            let m_inner = u2q_dag_times_mat2(u.u_l, target);
            let v_inner = unitary_to_uv_zeta(&m_inner);
            let budget_hit = AtomicBool::new(false);
            let u_l_local = *u.u_l;
            let floor = u.floor;
            let should_stop = |_x: &[i64; 16]| -> bool {
                // Incumbent-abort (sound) OR deadline (anytime cutoff).
                // Leaf hits only — a handful per walk, so the Instant
                // read is noise.
                best_cost.load(std::sync::atomic::Ordering::Relaxed) <= floor
                    || start.elapsed() >= deadline
            };
            let w = &watches[idx];
            w.active.store(true, Ordering::Relaxed);
            let sols = phase1_deep_aware(
                scratch, v_inner, Some((u.u_l, target)), self.deep_rot_src.as_ref(), k_inner, epsilon,
                per_prefix_cap, &budget_hit, should_stop,
                Some(&w.abort), None,
            );
            w.active.store(false, Ordering::Relaxed);

            // Backstop cap, or the walk ran into the deadline (whether
            // aborted mid-tree or merely unfinished business remains
            // indistinguishable here — mark conservatively).
            if budget_hit.load(std::sync::atomic::Ordering::Relaxed)
                || start.elapsed() >= deadline
            {
                truncated[u.level_idx].store(true, Ordering::Relaxed);
            }

            let mut best: Option<(usize, SynthResultQ)> = None;
            for sol in &sols {
                let u_r = solution_to_u2q_d(sol, k_inner, u.d_r);
                let u_full = u_l_local * u_r;
                let dist = diamond_distance_u2q_float(&u_full, target);
                if dist < epsilon {
                    let gates = BlochDecomposer.decompose(&u_full);
                    let cost = gates_cost(&gates, q_cost_x2);
                    match &best {
                        Some((bcost, _)) if *bcost <= cost => {}
                        _ => best = Some((cost, SynthResultQ {
                            gates: Some(gates),
                            lde: u.k_total,
                            distance: dist,
                        })),
                    }
                }
            }
            if let Some((c, _)) = &best {
                best_cost.fetch_min(*c, std::sync::atomic::Ordering::Relaxed);
            }
            best
        };

        let make_scratch = || {
            let mut s = Box::new(IntScratch16::new(epsilon));
            s.use_f64_gs = use_f64_gs;
            s.bkz_block_size = bkz_block_size;
            s
        };

        let walks_done = AtomicBool::new(false);
        struct DoneGuard<'a>(&'a AtomicBool);
        impl Drop for DoneGuard<'_> {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Relaxed);
            }
        }
        let result_pair: Option<(usize, SynthResultQ)> = std::thread::scope(|wscope| {
            let _done_guard = DoneGuard(&walks_done);
            let watches_ref = &watches;
            let units_ref = &units;
            let truncated_ref = &truncated;
            let walks_done_ref = &walks_done;
            wscope.spawn(move || {
                while !walks_done_ref.load(Ordering::Relaxed) {
                    std::thread::sleep(std::time::Duration::from_millis(20));
                    let cur_best =
                        best_cost.load(std::sync::atomic::Ordering::Relaxed);
                    let dl = start.elapsed() >= deadline;
                    for (i, w) in watches_ref.iter().enumerate() {
                        if !w.active.load(Ordering::Relaxed) {
                            continue;
                        }
                        if cur_best <= w.floor {
                            // Sound incumbent-floor kill — not truncation.
                            w.abort.store(true, Ordering::Relaxed);
                        } else if dl {
                            w.abort.store(true, Ordering::Relaxed);
                            truncated_ref[units_ref[i].level_idx]
                                .store(true, Ordering::Relaxed);
                        }
                    }
                }
            });
            let r = units
                .par_iter()
                .enumerate()
                .with_min_len(opt_chunk)
                .map_init(make_scratch, |s, (i, u)| per_unit(s, i, u))
                .reduce(
                    || None,
                    |a, b| match (a, b) {
                        (None, x) | (x, None) => x,
                        (Some(a), Some(b)) => Some(if a.0 <= b.0 { a } else { b }),
                    },
                );
            walks_done.store(true, Ordering::Relaxed);
            r
        });

        (
            result_pair,
            truncated.into_iter().map(|t| t.into_inner()).collect(),
        )
    }

    /// Single-search lattice probe at lde `k`, returning the best
    /// `(cost, SynthResultQ)` under the current `optimize_cost` mode.
    /// Mirrors the `try_lattice_k` + `check_sols` closures in
    /// [`Self::synthesize`] but as a method so it can be reused by the
    /// Tier-1 certified synthesis: exhaustively enumerate every
    /// Clifford+√T circuit with reduced lde ≤ `k_max` (single
    /// unbudgeted shell enumeration per parity branch — see
    /// [`CostCertificate`] for the covering argument), floor with the
    /// Clifford+T baseline, and report a proven optimality interval.
    ///
    /// Wall time grows exponentially with `k_max`; `certified_optimal`
    /// requires `upper ≤ cost_lb_half_units(k_max + 1)` ≈ k_max, so
    /// closing the certificate for a cost-C circuit needs k_max ≳ C
    /// half-units under the current (slope-1/2) staircase. Tightening
    /// the staircase (design doc P1') shrinks the required horizon
    /// proportionally without touching this code.
    pub fn synthesize_certified(
        &self,
        target: Mat2,
        k_max: u32,
    ) -> Option<(SynthResultQ, CostCertificate)> {
        let target = project_det_to_zeta_coset(&target);
        let g = Complex64::from_polar(1.0, PI / 16.0);
        let target_odd: Mat2 = [
            [target[0][0] * g, target[0][1] * g],
            [target[1][0] * g, target[1][1] * g],
        ];

        // T-baseline floor — only when the target's det class is even:
        // Clifford+T determinants are even ζ₁₆ powers, so an odd-class
        // target makes the baseline sweep its whole lde range rejecting
        // every prefix (trace-diagnosed: 100% mat_uv_rej, minutes of
        // futile work).
        let d_even = det_phase_of(&target) % 2 == 0;
        let baseline: Option<(usize, SynthResultQ)> = if !d_even { None } else {
            crate::synthesis::clifford_t::SynthesizerT::new(self.epsilon)
                .synthesize(target)
                .and_then(|r| {
                    if !(r.distance < self.epsilon) {
                        return None;
                    }
                    r.gates.map(|gs| {
                        let c = gates_cost(&gs, self.q_cost_x2);
                        (c, SynthResultQ { gates: Some(gs), lde: r.lde, distance: r.distance })
                    })
                })
        };

        // One full enumeration per parity branch at shell k_max. The
        // lattice pipeline (LLL + SE) is only well-behaved for
        // k > BRUTE_LIMIT — at tiny shells it degenerates (that's why
        // the production path brute-forces k ≤ 3) — so small horizons
        // route to the exact brute enumerator instead.
        let trace = crate::synthesis::diag::trace_enabled();
        let mut best: Option<(usize, SynthResultQ)> = baseline;
        for (label, t) in [("even", &target), ("odd", &target_odd)] {
            let t_branch = std::time::Instant::now();
            let d = det_phase_of(t);
            let found: Option<(usize, SynthResultQ)> = if k_max <= BRUTE_LIMIT {
                let mut branch_best: Option<(usize, SynthResultQ)> = None;
                let shell = brute_shell_cached(k_max);
                let zd = Complex64::from_polar(1.0, d as f64 * PI / 8.0);
                let thr = brute_prefilter_threshold(self.epsilon);
                for (sol, m) in shell.sols.iter().zip(&shell.mats) {
                    if brute_dist_est(m, zd, t) >= thr {
                        continue;
                    }
                    // Shells above the minimum contain √2-scaled images
                    // of lower-lde circuits (that's the covering
                    // mechanism); reduce before decomposing — the
                    // decomposer expects primitive denominators.
                    let cand: U2Q = solution_to_u2q_d(sol, k_max, d).reduced();
                    let dist = diamond_distance_u2q_float(&cand, t);
                    if dist < self.epsilon {
                        let gates = BlochDecomposer.decompose(&cand);
                        let c = gates_cost(&gates, self.q_cost_x2);
                        match &branch_best {
                            Some((bc, _)) if *bc <= c => {}
                            _ => branch_best = Some((c, SynthResultQ {
                                gates: Some(gates), lde: k_max, distance: dist,
                            })),
                        }
                    }
                }
                branch_best
            } else {
                let v = unitary_to_uv_zeta(t);
                let mut scratch: Option<Box<IntScratch16>> = None;
                self.run_single_optimal(
                    t, d, v, k_max, u64::MAX, &mut scratch, /*cost_min=*/true,
                )
                .0
            };
            if trace {
                eprintln!(
                    "[zeta] certified branch={label} k={k_max} d={d} {} t={:.0}ms",
                    found.as_ref().map(|(c, _)| format!("cost={c}"))
                        .unwrap_or_else(|| "none".into()),
                    t_branch.elapsed().as_secs_f64() * 1000.0,
                );
            }
            if let Some((c, r)) = found {
                match &best {
                    Some((bc, _)) if *bc <= c => {}
                    _ => best = Some((c, r)),
                }
            }
        }

        let (upper, result) = best?;
        let beyond = crate::synthesis::cost_bound::cost_lb_half_units(k_max + 1);
        let cert = CostCertificate {
            upper_half_units: upper,
            lower_half_units: upper.min(beyond),
            k_searched: k_max,
            certified_optimal: upper <= beyond,
        };
        Some((result, cert))
    }

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
    ) -> (Option<(usize, SynthResultQ)>, bool) {
        let epsilon = self.epsilon;
        let s = scratch.get_or_insert_with(|| {
            let mut sb = Box::new(IntScratch16::new(epsilon));
            sb.use_f64_gs = self.use_f64_gs;
            sb.bkz_block_size = self.bkz_block_size;
            sb
        });
        let budget_hit = AtomicBool::new(false);
        let should_stop = |x: &[i64; 16]| -> bool {
            if cost_min { return false; }
            let cand = solution_to_u2q_d(x, k, d);
            diamond_distance_u2q_float(&cand, target) < epsilon
        };
        let sols = phase1_deep_aware(
            s.as_mut(), v, None, self.deep_rot_src.as_ref(), k, epsilon, budget, &budget_hit, should_stop, None, None,
        );
        let hit = budget_hit.load(std::sync::atomic::Ordering::Relaxed);
        let cands = sols.iter().map(|sol| (solution_to_u2q_d(sol, k, d), k));
        (self.best_costed(cands, target, !cost_min), hit)
    }

    /// Cost-optimal synthesis. Three stages:
    ///
    /// 1. **Brute regime** (k ≤ BRUTE_LIMIT): `phase1_brute` enumerates
    ///    the full norm shell exactly, so the min-cost candidate at the
    ///    smallest feasible k is already optimal there.
    /// 2. **Screen**: run the *production first-hit path* (a clone with
    ///    `optimize_cost` off) to locate the smallest feasible lde.
    ///    This inherits everything the first-hit path has — deep-ε
    ///    parallel-LDE speculation, 2-pass budget completeness — and is
    ///    5-10× cheaper per no-solution lde than an enumerating sweep.
    /// 3. **Enum**: flatten `[find_lde .. find_lde+window] × m_sweep`
    ///    into independent parallel tasks, all pruning against one
    ///    shared best-cost tracker seeded with the screen candidate's
    ///    cost. The screen candidate is the floor for the final min, so
    ///    this stage can only improve it.
    fn synthesize_optimal(&self, target: Mat2) -> Option<SynthResultQ> {
        self.synthesize_optimal_certified(target).map(|(r, _)| r)
    }

    /// Production search + certificate: same hybrid search, with the
    /// truncation ledger folded into a [`CostCertificate`]. The lower
    /// bound comes from the coverage horizon: per parity branch, the
    /// largest level whose m = 0 task completed WITHOUT budget
    /// truncation (one full level covers all lower lde via √2-scaled
    /// points); anything above the smaller branch horizon costs at
    /// least `cost_lb_half_units(horizon + 1)`. With `certify` off no
    /// m = 0 tasks run and the certificate is vacuous (lower = 0).
    pub fn synthesize_with_certificate(
        &self,
        target: Mat2,
    ) -> Option<(SynthResultQ, CostCertificate)> {
        let mut certified = self.clone();
        certified.certify = true;
        certified.synthesize_optimal_certified(target)
    }

    fn synthesize_optimal_certified(
        &self,
        target: Mat2,
    ) -> Option<(SynthResultQ, CostCertificate)> {
        let branch_horizon = |ledger: &[(u32, u32, bool)]| -> u32 {
            ledger
                .iter()
                .filter(|(_, m, truncated)| *m == 0 && !truncated)
                .map(|(k, _, _)| *k)
                .max()
                .unwrap_or(0)
        };
        let finish = |r: SynthResultQ, horizon: u32, q_cost_x2: usize| {
            let upper = gates_cost(r.gates.as_deref().unwrap_or(""), q_cost_x2);
            let beyond = crate::synthesis::cost_bound::cost_lb_half_units(horizon + 1);
            let cert = CostCertificate {
                upper_half_units: upper,
                lower_half_units: upper.min(beyond),
                k_searched: horizon,
                certified_optimal: upper <= beyond,
            };
            (r, cert)
        };

        if !self.odd_parity_branch {
            let mut ledger = Vec::new();
            let r = self.synthesize_optimal_inner(target, /*with_baseline=*/true, &mut ledger)?;
            // Single-branch search covers only one parity class: the
            // other class is unsearched, so the horizon is vacuous.
            return Some(finish(r, 0, self.q_cost_x2));
        }
        // Parity branches: the pipeline pins det to ζ₁₆^{d(target)} and
        // Q-count ≡ d (mod 2), so one target reaches only half the pool.
        // Rotating by e^{iπ/16} shifts d by 1 and opens the odd-Q half;
        // diamond distance is phase-invariant, so odd finds are valid.
        // The Clifford+T baseline skips the odd branch (T-circuit dets
        // are even ζ₁₆ powers — it would burn max_lde finding nothing).
        let g = Complex64::from_polar(1.0, PI / 16.0);
        let target_odd: Mat2 = [
            [target[0][0] * g, target[0][1] * g],
            [target[1][0] * g, target[1][1] * g],
        ];
        // One shared incumbent serves both branches: costs are directly
        // comparable across parities, and the staircase bound
        // (cost < c̃ ⇒ lde ≤ c̃ + 1) lets each branch use it as a
        // dynamic lde clamp — which is what allows concurrency at all
        // (the old static odd.max_lde cap forced serial order). 16 MiB
        // stacks for the deep SE recursion.
        let global_best =
            std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(usize::MAX));
        let mut even_self = self.clone();
        even_self.global_best_cost = Some(global_best.clone());
        let mut odd_self = self.clone();
        odd_self.global_best_cost = Some(global_best.clone());
        odd_self.deep_rot_src = Some((target, 1));
        // Stage-2 handshake flags (see field docs): each branch's
        // frontier dispatch waits until the peer's screen is done.
        let even_screen_done = std::sync::Arc::new(AtomicBool::new(false));
        let odd_screen_done = std::sync::Arc::new(AtomicBool::new(false));
        even_self.my_screen_done = Some(even_screen_done.clone());
        even_self.peer_screen_done = Some(odd_screen_done.clone());
        odd_self.my_screen_done = Some(odd_screen_done.clone());
        odd_self.peer_screen_done = Some(even_screen_done.clone());
        let mut ledger_even = Vec::new();
        let mut ledger_odd = Vec::new();
        let trace = crate::synthesis::diag::trace_enabled();
        let t_branches = std::time::Instant::now();
        // Deep-ε parities run sequentially by measurement, not fear:
        // each branch already saturates the pool alone, so concurrency
        // dilutes both and costs ~2× wall (and trades ~+1pp cost at the
        // same wall). CYCLOSYNTH_SEQ_PARITY=0 re-enables concurrency —
        // the halved wall is a legitimate fast-mode trade. The shared
        // incumbent flows identically either way.
        let force_sequential = self.epsilon < 2.5e-8
            && std::env::var("CYCLOSYNTH_SEQ_PARITY").as_deref() != Ok("0");
        if force_sequential {
            // No peer exists in sequential mode — pre-set BOTH handshake
            // flags or the frontier dead-sleeps its full 4×deadline
            // bound waiting on a screen that never starts.
            even_screen_done.store(true, Ordering::Release);
            odd_screen_done.store(true, Ordering::Release);
            let r_e = even_self.synthesize_optimal_inner(
                target, /*with_baseline=*/ true, &mut ledger_even,
            );
            let r_o = odd_self.synthesize_optimal_inner(
                target_odd, /*with_baseline=*/ false, &mut ledger_odd,
            );
            let horizon =
                branch_horizon(&ledger_even).min(branch_horizon(&ledger_odd));
            return match (r_e, r_o) {
                (Some(a), Some(b)) => {
                    let ca =
                        gates_cost(a.gates.as_deref().unwrap_or(""), self.q_cost_x2);
                    let cb =
                        gates_cost(b.gates.as_deref().unwrap_or(""), self.q_cost_x2);
                    Some(finish(if cb < ca { b } else { a }, horizon, self.q_cost_x2))
                }
                (a, b) => a.or(b).map(|r| finish(r, horizon, self.q_cost_x2)),
            };
        }
        let (r_even, r_odd) = std::thread::scope(|s| {
            let even_ledger = &mut ledger_even;
            let odd_ledger = &mut ledger_odd;
            let even_ref = &even_self;
            let odd_ref = &odd_self;
            let even_done = &even_screen_done;
            let odd_done = &odd_screen_done;
            let h_even = std::thread::Builder::new()
                .stack_size(16 * 1024 * 1024)
                .spawn_scoped(s, move || {
                    let t0 = std::time::Instant::now();
                    let r = even_ref.synthesize_optimal_inner(
                        target, /*with_baseline=*/ true, even_ledger,
                    );
                    // Branch done ⇒ screen trivially "done" (covers
                    // returns before stage 2, e.g. stage-1 brute finds)
                    // so the peer's handshake wait can't outlive us.
                    even_done.store(true, Ordering::Release);
                    (r, t0.elapsed())
                })
                .expect("spawn even parity branch");
            let h_odd = std::thread::Builder::new()
                .stack_size(16 * 1024 * 1024)
                .spawn_scoped(s, move || {
                    let t0 = std::time::Instant::now();
                    let r = odd_ref.synthesize_optimal_inner(
                        target_odd, /*with_baseline=*/ false, odd_ledger,
                    );
                    odd_done.store(true, Ordering::Release);
                    (r, t0.elapsed())
                })
                .expect("spawn odd parity branch");
            let (r_even, dt_even) = h_even.join().unwrap();
            let (r_odd, dt_odd) = h_odd.join().unwrap();
            if trace {
                eprintln!(
                    "[zeta] optimal branches even={:.0}ms odd={:.0}ms scope={:.0}ms",
                    dt_even.as_secs_f64() * 1000.0,
                    dt_odd.as_secs_f64() * 1000.0,
                    t_branches.elapsed().as_secs_f64() * 1000.0,
                );
            }
            (r_even, r_odd)
        });
        // Coverage holds only up to the SMALLER branch horizon: a level
        // is closed only when both parity worlds enumerated it fully.
        let horizon = branch_horizon(&ledger_even).min(branch_horizon(&ledger_odd));
        match (r_even, r_odd) {
            (Some(a), Some(b)) => {
                let ca = gates_cost(a.gates.as_deref().unwrap_or(""), self.q_cost_x2);
                let cb = gates_cost(b.gates.as_deref().unwrap_or(""), self.q_cost_x2);
                Some(finish(if cb < ca { b } else { a }, horizon, self.q_cost_x2))
            }
            (a, b) => a.or(b).map(|r| finish(r, horizon, self.q_cost_x2)),
        }
    }

    /// Scan ε-close candidates, decompose each, and keep the min-cost
    /// one — or return the FIRST ε-close one when `first_hit` (the
    /// legacy non-optimal semantics, which must stay order-sensitive).
    fn best_costed<I>(
        &self,
        cands: I,
        target: &Mat2,
        first_hit: bool,
    ) -> Option<(usize, SynthResultQ)>
    where
        I: IntoIterator<Item = (U2Q, u32)>,
    {
        let mut best: Option<(usize, SynthResultQ)> = None;
        for (cand, lde) in cands {
            let dist = diamond_distance_u2q_float(&cand, target);
            if dist < self.epsilon {
                let gates = BlochDecomposer.decompose(&cand);
                let cost = gates_cost(&gates, self.q_cost_x2);
                let result = SynthResultQ { gates: Some(gates), lde, distance: dist };
                if first_hit {
                    return Some((cost, result));
                }
                match &best {
                    Some((bc, _)) if *bc <= cost => {}
                    _ => best = Some((cost, result)),
                }
            }
        }
        best
    }

    /// Stage 1 of the optimal pipeline: exact min-cost scan of the brute
    /// shells (k ≤ BRUTE_LIMIT). A find here is already optimal at the
    /// smallest feasible k.
    fn brute_min_cost(&self, target: &Mat2, d: u32) -> Option<(usize, SynthResultQ)> {
        let zd = Complex64::from_polar(1.0, d as f64 * PI / 8.0);
        let thr = brute_prefilter_threshold(self.epsilon);
        for k in self.min_lde..=BRUTE_LIMIT.min(self.max_lde) {
            let shell = brute_shell_cached(k);
            let close = shell
                .sols
                .iter()
                .zip(&shell.mats)
                .filter(|(_, m)| brute_dist_est(m, zd, target) < thr)
                .map(|(sol, _)| (solution_to_u2q_d(sol, k, d), k));
            let best = self.best_costed(close, target, false);
            if best.is_some() {
                return best;
            }
        }
        None
    }

    /// Stage 2 of the optimal pipeline: the first-hit screen and the
    /// Clifford+T baseline, in parallel. T-only solutions live at lde ≈
    /// T-count — far above the enum window — so covering them requires
    /// synthesizing them directly, which also makes the result
    /// never-worse-than-Clifford+T by construction and seeds the stage-3
    /// prune. Returns `(screen result, unclear levels, baseline as a
    /// √T-shaped (cost, result) candidate)`.
    fn screen_and_baseline(
        &self,
        target: Mat2,
        with_baseline: bool,
    ) -> (Option<SynthResultQ>, Vec<u32>, Option<(usize, SynthResultQ)>) {
        // Clifford+T dets are even ζ₁₆ powers — odd-class targets make
        // the baseline burn its whole lde sweep finding nothing.
        let with_baseline = with_baseline && det_phase_of(&target) % 2 == 0;
        let (first, unclear, t_baseline) = std::thread::scope(|s| {
            let baseline_handle = if with_baseline {
                Some(
                    std::thread::Builder::new()
                        // 16 MiB: deep SE recursion.
                        .stack_size(16 * 1024 * 1024)
                        .spawn_scoped(s, || {
                            crate::synthesis::clifford_t::SynthesizerT::new(self.epsilon)
                                .synthesize(target)
                        })
                        .expect("spawn clifford_t baseline thread"),
                )
            } else {
                None
            };
            let mut first_hit = self.clone();
            first_hit.optimize_cost = false;
            first_hit.odd_parity_branch = false;
            // Screen-lite (CYCLOSYNTH_SCREEN_DIV; see `budget_div` docs).
            if self.epsilon < 1e-7 {
                first_hit.budget_div = screen_div();
            }
            // Truncated-and-never-cleared levels below find-lde must
            // reach the enum grid or the window silently misses them.
            let mut unclear = Vec::new();
            let mut first = first_hit.synthesize_with_unclear(target, Some(&mut unclear));
            if first.is_none() && first_hit.budget_div > 1 {
                first_hit.budget_div = 1;
                unclear.clear();
                first = first_hit.synthesize_with_unclear(target, Some(&mut unclear));
            }
            (first, unclear, baseline_handle.and_then(|h| h.join().unwrap()))
        });
        // The baseline's gate string contains no Q, so its cost is
        // exactly 2·T_count half-units.
        let baseline: Option<(usize, SynthResultQ)> = t_baseline.and_then(|r| {
            let dist = r.distance;
            if !(dist < self.epsilon) {
                return None;
            }
            r.gates.map(|g| {
                let c = gates_cost(&g, self.q_cost_x2);
                (c, SynthResultQ { gates: Some(g), lde: r.lde, distance: dist })
            })
        });
        (first, unclear, baseline)
    }

    fn synthesize_optimal_inner(
        &self,
        target: Mat2,
        with_baseline: bool,
        ledger_out: &mut Vec<(u32, u32, bool)>,
    ) -> Option<SynthResultQ> {
        use crate::synthesis::diag;
        use crate::synthesis::lattice_zeta::{set_verify_prune_mpfr, verify_prune_mpfr};
        let trace = diag::trace_enabled();

        // Same verify-prune RAII dance as `synthesize` — the enum stage
        // runs SE walks of its own, so the guard must span both stages.
        let verify_was_on = verify_prune_mpfr();
        let need_verify = self.epsilon < 2e-8;
        if need_verify && !verify_was_on {
            set_verify_prune_mpfr(true);
        }
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

        // Stage 1: brute regime, exact min-cost at the smallest k.
        if let Some((c, r)) = self.brute_min_cost(&target, d) {
            // Publish the brute win before returning — otherwise gate-like
            // targets leave the peer branch's dynamic lde clamp unseeded
            // and its screen sweeps to max_lde for nothing.
            if let Some(g) = &self.global_best_cost {
                g.fetch_min(c, std::sync::atomic::Ordering::Relaxed);
            }
            return Some(r);
        }

        let t_s = std::time::Instant::now();
        let (first, mut screen_unclear, baseline) =
            self.screen_and_baseline(target, with_baseline);
        // Signal screen completion to the peer parity branch; the
        // matching wait sits just before the frontier dispatch below.
        if let Some(flag) = &self.my_screen_done {
            flag.store(true, Ordering::Release);
        }
        let baseline_cost = baseline.as_ref().map(|(c, _)| *c).unwrap_or(usize::MAX);

        // If the √T screen found nothing within the configured bounds
        // (max_lde, budgets), return None: the baseline is a cost floor
        // for comparison, not a fallback — returning it would silently
        // bypass the caller's search bounds.
        let first = first?;
        let fl = first.lde;
        let first_cost = first
            .gates
            .as_deref()
            .map(|g| gates_cost(g, self.q_cost_x2))
            .unwrap_or(usize::MAX);
        if trace {
            eprintln!(
                "[zeta] optimal screen lde={fl} cost={first_cost} baseline(T)={baseline_cost}  t={:.0}ms",
                t_s.elapsed().as_secs_f64() * 1000.0);
        }

        // Stage 3: enum over the (lde, m) grid against one pre-seeded
        // incumbent (fetch_min — a peer's earlier cheaper find must
        // survive). Certify adds m = 0 tasks per level: the only variant
        // whose untruncated completion proves a level exhausted, which
        // is what moves the certificate horizon.
        let local_best = std::sync::atomic::AtomicUsize::new(usize::MAX);
        let shared_best: &std::sync::atomic::AtomicUsize =
            self.global_best_cost.as_deref().unwrap_or(&local_best);
        shared_best.fetch_min(
            first_cost.min(baseline_cost),
            std::sync::atomic::Ordering::Relaxed,
        );
        let mut tasks: Vec<(u32, u32)> = (0..=self.optimal_lde_window)
            .map(|i| fl + i)
            .filter(|&k| k <= self.max_lde)
            .flat_map(|k| self.optimal_m_sweep.iter().map(move |&m| (k, m)))
            .collect();
        if self.certify {
            for i in 0..=self.optimal_lde_window {
                let k = fl + i;
                if k <= self.max_lde && !tasks.contains(&(k, 0)) {
                    tasks.push((k, 0));
                }
            }
        }
        // Unverified below-fl levels get the same arm set as window
        // levels — the find at fl short-circuited their pass-2 retry, so
        // they may still hold a cheaper candidate.
        screen_unclear.sort_unstable();
        screen_unclear.dedup();
        screen_unclear.retain(|&k| k < fl && k <= self.max_lde);
        if !screen_unclear.is_empty() {
            if trace {
                eprintln!("[zeta] optimal screen left levels {screen_unclear:?} unverified below fl={fl} — adding to enum grid");
            }
            for &k in &screen_unclear {
                for &m in &self.optimal_m_sweep {
                    if !tasks.contains(&(k, m)) {
                        tasks.push((k, m));
                    }
                }
                if self.certify && !tasks.contains(&(k, 0)) {
                    tasks.push((k, 0));
                }
            }
        }
        // ── Anytime merged frontier (fast path) ─────────────────────
        // With a deadline configured and certify off, all (k, m ≥ 1)
        // arms run as ONE floor-ordered prefix frontier under a wall
        // deadline instead of per-arm node budgets (see
        // `dc_frontier_q`). The legacy task grid below remains the
        // certify path (honest budget-truncation semantics) and the
        // deep-ε path (deadline default None), and still handles
        // m = 0 arms (single-shot probes are not prefix work-units).
        if !self.certify
            && !tasks.is_empty()
            && tasks.iter().all(|&(_, m)| m >= 1)
        {
            if let Some(deadline_ms) = self.optimal_deadline_ms {
                // Wait for the peer's screen before flooding the pool
                // (a frontier starves a running screen ~50×); bounded,
                // and the peer's branch-return store guarantees
                // progress on early exits.
                if let Some(peer) = &self.peer_screen_done {
                    let t_wait = std::time::Instant::now();
                    let cap = std::time::Duration::from_millis(4 * deadline_ms.max(100));
                    while !peer.load(Ordering::Acquire) && t_wait.elapsed() < cap {
                        std::thread::sleep(std::time::Duration::from_millis(1));
                    }
                    if trace {
                        eprintln!(
                            "[zeta] optimal frontier handshake wait={:.0}ms",
                            t_wait.elapsed().as_secs_f64() * 1000.0);
                    }
                }
                let t_w = std::time::Instant::now();
                let (fr, level_truncated) =
                    self.run_frontier(&target, &tasks, deadline_ms, shared_best);
                if trace {
                    eprintln!(
                        "[zeta] optimal frontier {:?} deadline={}ms t={:.0}ms truncated={:?}",
                        tasks, deadline_ms,
                        t_w.elapsed().as_secs_f64() * 1000.0,
                        tasks.iter().zip(level_truncated.iter())
                            .filter(|(_, &tr)| tr).map(|(t, _)| *t)
                            .collect::<Vec<_>>(),
                    );
                }
                let mut best: (usize, SynthResultQ) = (first_cost, first);
                if let Some((bc, br)) = baseline {
                    if bc < best.0 {
                        best = (bc, br);
                    }
                }
                if let Some((c, res)) = fr {
                    if trace {
                        eprintln!("[zeta]   frontier best lde={:>2} cost={c} dist={:.3e}",
                            res.lde, res.distance);
                    }
                    if c < best.0 {
                        best = (c, res);
                    }
                }
                *ledger_out = tasks
                    .iter()
                    .zip(level_truncated)
                    .map(|(&(k, m), tr)| (k, m, tr))
                    .collect();
                return Some(best.1);
            }
        }

        let t_w = std::time::Instant::now();
        let task_results: Vec<(u32, u32, bool, Option<(usize, SynthResultQ)>)> =
            std::thread::scope(|s| {
                let shared_best = shared_best;
                let handles: Vec<_> = tasks
                    .iter()
                    .map(|&(k, m)| {
                        // 16 MiB stack: these threads participate in
                        // rayon's in-place execution of dc_search_q,
                        // whose per-prefix scratch + SE recursion
                        // overflow the 2 MiB scoped-thread default
                        // (observed SIGABRT at lde_window = 2).
                        std::thread::Builder::new()
                            .stack_size(16 * 1024 * 1024)
                            .spawn_scoped(s, move || {
                                let (r, truncated) = self.try_optimal_variant(
                                    target, d, v, k, m, /*cost_min=*/true,
                                    Some(shared_best),
                                );
                                (k, m, truncated, r)
                            })
                            .expect("spawn lde-window thread")
                    })
                    .collect();
                handles.into_iter().map(|h| h.join().unwrap()).collect()
            });
        if trace {
            eprintln!("[zeta] optimal enum {:?} parallel t={:.0}ms",
                tasks, t_w.elapsed().as_secs_f64() * 1000.0);
        }
        let mut best: (usize, SynthResultQ) = (first_cost, first);
        if let Some((bc, br)) = baseline {
            if bc < best.0 {
                best = (bc, br);
            }
        }
        // Truncation ledger: (level, m, truncated) for every enum task.
        let mut ledger: Vec<(u32, u32, bool)> = Vec::new();
        for (k, m, truncated, r) in task_results {
            ledger.push((k, m, truncated));
            if let Some((c, res)) = r {
                if trace {
                    eprintln!("[zeta]   enum  lde={:>2}  cost={c} m={m} dist={:.3e}",
                        res.lde, res.distance);
                }
                if c < best.0 {
                    best = (c, res);
                }
            }
        }

        // Floor-driven extension (certify mode): keep running full m=0
        // levels above the window while the proven beyond-horizon floor
        // is still below the incumbent and the extension time budget
        // lasts. Every completed (untruncated) level raises the
        // certificate's lower bound by 4 half-units.
        if self.certify && self.certify_extra_ms > 0 {
            let t_ext = std::time::Instant::now();
            let mut k = fl + self.optimal_lde_window + 1;
            while k <= self.max_lde
                && crate::synthesis::cost_bound::cost_lb_half_units(k) < best.0
                && (t_ext.elapsed().as_millis() as u64) < self.certify_extra_ms
            {
                let (r, truncated) =
                    self.try_optimal_variant(target, d, v, k, 0, true, Some(shared_best));
                ledger.push((k, 0, truncated));
                if trace {
                    eprintln!("[zeta] certify-extend k={k} truncated={truncated} t={:.0}ms",
                        t_ext.elapsed().as_secs_f64() * 1000.0);
                }
                if let Some((c, res)) = r {
                    if c < best.0 {
                        best = (c, res);
                    }
                }
                if truncated {
                    break; // deeper levels will only be bigger
                }
                k += 1;
            }
        }

        *ledger_out = ledger;
        Some(best.1)
    }

    /// One (lde, m) variant of the optimal search: m=0 → single-shot
    /// lattice probe, m≥1 → FGKM-prefix D&C with the default d_R filter.
    /// Extracted from the m-sweep loop so the enum phase can run all
    /// (k, m) pairs as independent parallel tasks.
    fn try_optimal_variant(
        &self,
        target: Mat2,
        d: u32,
        v: [f64; 4],
        k: u32,
        m: u32,
        cost_min: bool,
        shared_best_cost: Option<&std::sync::atomic::AtomicUsize>,
    ) -> (Option<(usize, SynthResultQ)>, bool) {
        let budget_mult = self.optimal_budget_multiplier.max(1);
        if m == 0 {
            // In certify mode the m = 0 tasks are the coverage proof —
            // a truncated one contributes nothing to the horizon, so
            // give them room (32×) to actually finish the level.
            let cert_boost: u64 = if self.certify { 32 } else { 1 };
            let cap = PASS1_CAP
                .saturating_mul(budget_mult)
                .saturating_mul(cert_boost);
            let mut local_scratch: Option<Box<IntScratch16>> = None;
            let (r, hit) =
                self.run_single_optimal(&target, d, v, k, cap, &mut local_scratch, cost_min);
            if hit && crate::synthesis::diag::trace_enabled() {
                eprintln!("[zeta]   enum (k={k}, m=0) BUDGET-HIT — coverage lost");
            }
            (r, hit)
        } else if m < k {
            // The d_R filters were tuned for first-hit *speed*; in enum
            // mode they may exclude det-phase classes containing the
            // cost optimum. `optimal_open_dr_filter` lifts them.
            let filter = if self.optimal_open_dr_filter {
                Vec::new()
            } else {
                default_dc_dr_filter(m)
            };
            let cap = dc_pass1_cap_for(self.epsilon).saturating_mul(budget_mult);
            let (r, budget_hit) = self.dc_search_q(
                &target, k, m, Some(&filter), cap, None, None, Some(cost_min),
                shared_best_cost,
            );
            if budget_hit && crate::synthesis::diag::trace_enabled() {
                eprintln!("[zeta]   enum (k={k}, m={m}) BUDGET-HIT — level truncated");
            }
            (r.map(|res| {
                let c = gates_cost(res.gates.as_deref().unwrap_or(""), self.q_cost_x2);
                (c, res)
            }), budget_hit)
        } else {
            (None, false)
        }
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
    /// The optimize-cost hybrid runs a Clifford+T baseline and returns
    /// the min, so its weighted cost can never exceed the Clifford+T
    /// result on the same target. Guard that invariant.
    #[test]
    fn optimal_cost_never_exceeds_clifford_t() {
        fn rz(t: f64) -> Mat2 {
            [
                [Complex64::from_polar(1.0, -t/2.0), Complex64::new(0.0, 0.0)],
                [Complex64::new(0.0, 0.0), Complex64::from_polar(1.0, t/2.0)],
            ]
        }
        for &(theta, eps) in &[(0.3_f64, 1e-3_f64), (1.1, 1e-3), (2.37, 1e-4)] {
            let target = rz(theta);
            let rt = crate::synthesis::clifford_t::SynthesizerT::new(eps)
                .synthesize(target)
                .expect("clifford_t baseline should synthesize");
            let t_cost = gates_cost(rt.gates.as_deref().unwrap_or(""), 7);
            let rq = SynthesizerQ::new(eps)
                .with_optimize_cost(true)
                .with_optimal_lde_window(2)
                .synthesize(target)
                .expect("hybrid optimal should synthesize");
            assert!(rq.distance < eps);
            let q_cost = gates_cost(rq.gates.as_deref().unwrap_or(""), 7);
            assert!(
                q_cost <= t_cost,
                "hybrid cost {q_cost} > clifford_t cost {t_cost} at θ={theta}, ε={eps:e}"
            );
        }
    }

    /// Screen-truncation out-param plumbing: on an easy coarse-ε target
    /// no level hits a budget cap, so `synthesize_with_unclear` must
    /// report zero unclear levels and agree with the public entry point.
    /// (ε = 1e-2: near-z-axis diagonal targets at 1e-3 burn every
    /// level's budget for ~5 min — known sparse-region hardness, not a
    /// plumbing concern.)
    #[test]
    fn screen_unclear_empty_on_easy_target() {
        let target: Mat2 = [
            [Complex64::from_polar(1.0, -0.35), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::from_polar(1.0, 0.35)],
        ];
        let synth = SynthesizerQ::new(1e-2).with_optimize_cost(false);
        let mut unclear = Vec::new();
        let r1 = synth
            .synthesize_with_unclear(target, Some(&mut unclear))
            .expect("should synthesize");
        let r2 = synth.synthesize(target).expect("should synthesize");
        assert!(unclear.is_empty(), "unexpected unclear levels: {unclear:?}");
        assert_eq!(r1.lde, r2.lde);
        assert!(r1.distance < 1e-2);
    }

    /// Production-path certificate (items 1+2): the hybrid search with
    /// `certify` on must return a well-formed interval, and at coarse ε
    /// the floor-driven extension should CLOSE it on a generic target.
    #[test]
    fn production_certificate_well_formed_and_closes_at_coarse_eps() {
        fn rzm(t: f64) -> Mat2 {
            [
                [Complex64::from_polar(1.0, -t/2.0), Complex64::new(0.0, 0.0)],
                [Complex64::new(0.0, 0.0), Complex64::from_polar(1.0, t/2.0)],
            ]
        }
        let (r, cert) = SynthesizerQ::new(3e-2)
            .with_certify_extra_ms(20_000)
            .synthesize_with_certificate(rzm(0.7))
            .expect("should synthesize");
        assert!(r.distance < 3e-2);
        assert!(cert.lower_half_units <= cert.upper_half_units);
        assert_eq!(
            cert.upper_half_units,
            gates_cost(r.gates.as_deref().unwrap(), 7)
        );
        // At 3e-2 the optimum costs ~19 HU; the extension reaches the
        // closing horizon (k ≈ 6) within the budget.
        assert!(cert.certified_optimal,
            "expected closure at coarse ε: upper {} lower {} k {}",
            cert.upper_half_units, cert.lower_half_units, cert.k_searched);
    }

    /// Tier-1 closing certificate at the cheapest scale: a T gate costs
    /// 2 half-units and the beyond-horizon floor L(3) = 2 matches, so
    /// k_max = 2 must CLOSE the certificate. (Unbudgeted shell walks
    /// grow fast with k — a k=8 closure test ran minutes; keep tests at
    /// the smallest k that exercises the logic.)
    #[test]
    fn certificate_closes_on_t_target() {
        let t_f = U2Q::t().to_float();
        let g = Complex64::from_polar(1.0, -PI / 8.0); // det(T)=ζ₁₆² → g²=ζ₁₆⁻²
        let target: Mat2 = [
            [t_f[0][0] * g, t_f[0][1] * g],
            [t_f[1][0] * g, t_f[1][1] * g],
        ];
        let (r, cert) = SynthesizerQ::new(1e-3)
            .synthesize_certified(target, 2)
            .expect("certified synthesis should succeed");
        assert!(r.distance < 1e-3);
        assert_eq!(cert.upper_half_units, 2, "T circuit costs 2 HU");
        assert!(cert.certified_optimal,
            "upper {} vs floor {} at k=2",
            cert.upper_half_units, cert.lower_half_units);
        assert_eq!(cert.lower_half_units, cert.upper_half_units);
    }

    /// Tier-1 gap certificate on a generic target at a small horizon:
    /// interval well-formed, does not close.
    #[test]
    fn certificate_gap_on_generic_target() {
        fn rzm(t: f64) -> Mat2 {
            [
                [Complex64::from_polar(1.0, -t/2.0), Complex64::new(0.0, 0.0)],
                [Complex64::new(0.0, 0.0), Complex64::from_polar(1.0, t/2.0)],
            ]
        }
        let (r, cert) = SynthesizerQ::new(1e-2)
            .synthesize_certified(rzm(0.7), 4)
            .expect("certified synthesis should succeed");
        assert!(r.distance < 1e-2);
        assert!(cert.lower_half_units <= cert.upper_half_units);
        assert_eq!(cert.k_searched, 4);
        // A 1e-2 approximation of a generic angle costs well over
        // L(5) = 4 HU, so the interval stays open.
        assert!(!cert.certified_optimal,
            "unexpected closure: upper {} lower {}",
            cert.upper_half_units, cert.lower_half_units);
    }

    /// k = 8 closure on the single-Q target (cost 7 HU needs the L(9)=8
    /// floor). Minutes-scale unbudgeted walk — milestone runs only.
    #[test]
    #[ignore = "unbudgeted k=8 shell walk; run with --ignored"]
    fn certificate_closes_on_single_q_target_slow() {
        let g = Complex64::from_polar(1.0, -PI / 16.0);
        let hqh = (U2Q::h() * U2Q::q() * U2Q::h()).reduced().to_float();
        let target: Mat2 = [
            [hqh[0][0] * g, hqh[0][1] * g],
            [hqh[1][0] * g, hqh[1][1] * g],
        ];
        let (_, cert) = SynthesizerQ::new(1e-3)
            .synthesize_certified(target, 8)
            .expect("certified synthesis should succeed");
        assert_eq!(cert.upper_half_units, 7);
        assert!(cert.certified_optimal);
    }

    /// The odd-parity branch must reach circuits the single-target
    /// pipeline cannot: V = e^{-iπ/16}·(H·Q·H) has det 1 (even class),
    /// but its physical optimum is the single-Q circuit (odd class,
    /// cost 3.5). Without the branch the search can only offer even-Q
    /// approximations.
    #[test]
    fn odd_parity_branch_finds_single_q() {
        let g = Complex64::from_polar(1.0, -PI / 16.0);
        let hqh = {
            let u = (U2Q::h() * U2Q::q() * U2Q::h()).reduced();
            u.to_float()
        };
        let target: Mat2 = [
            [hqh[0][0] * g, hqh[0][1] * g],
            [hqh[1][0] * g, hqh[1][1] * g],
        ];
        let r = SynthesizerQ::new(1e-3)
            .synthesize(target)
            .expect("should synthesize");
        let gates = r.gates.expect("gates");
        let q = gates.chars().filter(|&c| c == 'Q').count();
        let t = gates.chars().filter(|&c| c == 'T').count();
        assert!(r.distance < 1e-3);
        assert_eq!((t, q), (0, 1),
            "odd branch should find the exact single-Q circuit, got {gates}");
    }

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
        let synth = SynthesizerQ::new(1e-8).with_optimize_cost(false).with_max_lde(2);
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
        // First-hit: this tests gate-string reconstruction, not the
        // cost-optimal pipeline.
        let synth = SynthesizerQ::new(1e-7).with_optimize_cost(false);
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
        let synth = SynthesizerQ::new(1e-3).with_optimize_cost(false).with_max_lde(15);
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

    /// Census: how much of `build_l_q(m)` is right-coset duplicate work
    /// under ⟨S,X⟩, and how much survives the d_R filters. Soundness
    /// premise: (U_L·C)·U_R = U_L·(C·U_R) on the same shell — the rep's
    /// search covers every mate's solutions with identical totals. The
    /// list member matched to u·C is ζ^p·(u·C), so mates' d_R differ by
    /// arbitrary EVEN offsets (the argument is d_R-agnostic).
    /// Run: `cargo test --release --lib zeta_coset_census -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn zeta_coset_census() {
        use std::collections::{HashMap, HashSet};

        // lde-0 Clifford subgroup as U2Q (rebuilt from table names, the
        // same route build_l_q_inner uses for its Clifford suffixes —
        // shared with the production orbit table).
        let lde0 = lde0_cliffords_q();
        for c in &lde0 {
            assert_eq!(c.k, 0, "lde-0 Clifford has k != 0 as U2Q");
        }

        for m in 1..=3u32 {
            let prefixes = build_l_q(m);
            let n = prefixes.len();
            let key_of: Vec<[i64; 8]> = prefixes.iter().map(canonical_key_q).collect();
            let idx_of: HashMap<[i64; 8], usize> =
                key_of.iter().enumerate().map(|(i, k)| (*k, i)).collect();

            // Right-coset orbits: orbit(u) = {u·c} is exactly the coset
            // u·⟨S,X⟩, so one multiplication sweep finds the whole orbit.
            // Orbit id = min member index. `missing` counts mates whose
            // canonical key is absent from L (float-key rounding or a
            // genuine coverage hole — must stay ~0 for the dedup claim).
            let mut orbit_id: Vec<usize> = (0..n).collect();
            let mut missing = 0usize;
            for i in 0..n {
                let mut mn = i;
                for c in &lde0 {
                    let key = canonical_key_q(&(prefixes[i] * *c));
                    match idx_of.get(&key) {
                        Some(&j) => mn = mn.min(j),
                        None => missing += 1,
                    }
                }
                orbit_id[i] = mn;
            }
            let orbits: HashSet<usize> = orbit_id.iter().copied().collect();
            eprintln!(
                "\nm={m}: |L|={n}  orbits={}  full-orbit ratio={:.2}x  (missing mate keys: {missing})",
                orbits.len(),
                n as f64 / orbits.len() as f64
            );

            // Self-consistency with the production dedup: the cached
            // orbit table the searches use must be IDENTICAL to the
            // census's locally computed linking (gate 5).
            assert_eq!(
                orbit_id,
                *build_l_q_orbits(m).as_ref(),
                "production build_l_q_orbits({m}) diverges from census linking"
            );

            // d_R-respecting census per filter. For each d_target the
            // usable set is {u : (d_target − d_L) mod 16 ∈ filter}; the
            // dedup that survives = |usable| / |orbits among usable|.
            // `classes` additionally splits orbits by the unreduced k —
            // the PRODUCTION dedup grouping (`build_l_q_coset_keys`;
            // cross-k orbit links are float-real but their coverage is
            // asymmetric, so the implementation keeps one rep per
            // (orbit, k) ∩ usable): the classes column is the actual
            // achieved reduction.
            let d_l: Vec<u32> = prefixes
                .iter()
                .map(|u| det_phase_of(&u.to_float()))
                .collect();
            let coset_keys = build_l_q_coset_keys(m);
            for (fname, filter) in [
                ("strict [0]   (m=2 1st-hit default)", vec![0u32]),
                ("relaxed [0,1,15] (m=1 default)", vec![0u32, 1, 15]),
                ("OPEN (optimal_open_dr_filter, prod at eps<=1e-5)", vec![]),
            ] {
                let mut tot_usable = 0usize;
                let mut tot_orbits = 0usize;
                let mut tot_classes = 0usize;
                let mut per_d: Vec<(u32, usize, usize)> = Vec::new();
                for d_target in 0..16u32 {
                    let usable: Vec<usize> = (0..n)
                        .filter(|&i| {
                            if filter.is_empty() {
                                return true;
                            }
                            let d_r = ((d_target as i32 - d_l[i] as i32)
                                .rem_euclid(16)) as u32;
                            filter.contains(&d_r)
                        })
                        .collect();
                    let uorb: HashSet<usize> =
                        usable.iter().map(|&i| orbit_id[i]).collect();
                    let uclass: HashSet<(usize, u32)> =
                        usable.iter().map(|&i| coset_keys[i]).collect();
                    tot_usable += usable.len();
                    tot_orbits += uorb.len();
                    tot_classes += uclass.len();
                    per_d.push((d_target, usable.len(), uorb.len()));
                }
                eprintln!(
                    "  filter {fname}: avg usable {:.1} -> orbits {:.1} (dedup {:.2}x) | (orbit,k) classes {:.1} (PROD dedup {:.2}x)",
                    tot_usable as f64 / 16.0,
                    tot_orbits as f64 / 16.0,
                    tot_usable as f64 / tot_orbits.max(1) as f64,
                    tot_classes as f64 / 16.0,
                    tot_usable as f64 / tot_classes.max(1) as f64
                );
                if m == 2 {
                    let row: Vec<String> = per_d
                        .iter()
                        .map(|(d, u, o)| format!("d{d}:{u}/{o}"))
                        .collect();
                    eprintln!("    per-d usable/orbits: {}", row.join(" "));
                }
            }
        }
    }

    /// Ring-exact soundness pin: every pair sharing a dedup class
    /// (orbit, k) satisfies `u_i = ζ^p · u_rep · C` for some lde-0 C —
    /// exactly the relation the dedup's coverage argument consumes.
    #[test]
    fn zeta_coset_orbits_sound() {
        let lde0 = lde0_cliffords_q();
        for c in &lde0 {
            assert_eq!(c.k, 0, "lde-0 Clifford has k != 0 as U2Q");
        }
        let scale = |u: &U2Q, z: ZZeta| -> U2Q {
            U2Q::new(z * u.u11, z * u.u12, z * u.u21, z * u.u22, u.k)
        };
        for m in 1..=2u32 {
            let prefixes = build_l_q(m);
            let keys = build_l_q_coset_keys(m);
            assert_eq!(prefixes.len(), keys.len());
            // First member per (orbit, k) class = the class rep ties
            // resolve to in production when costs tie.
            let mut rep_of: HashMap<(usize, u32), usize> = HashMap::new();
            let mut classes = 0usize;
            for (i, u) in prefixes.iter().enumerate() {
                assert!(keys[i].0 <= i, "orbit id must be a min index (m={m}, i={i})");
                assert_eq!(keys[i].1, u.k, "class k must be the prefix k");
                let rep = *rep_of.entry(keys[i]).or_insert_with(|| {
                    classes += 1;
                    i
                });
                if rep == i {
                    continue;
                }
                let r = &prefixes[rep];
                let mate = lde0.iter().any(|c| {
                    let rc = *r * *c;
                    (0..16u32).any(|p| scale(&rc, zeta_16_pow(p)) == *u)
                });
                assert!(
                    mate,
                    "class-mates not ring-exact coset mates (m={m}, i={i}, rep={rep})"
                );
            }
            assert!(
                classes < prefixes.len(),
                "coset dedup must merge something at m={m}"
            );
        }
    }

    /// Coset-regression probe (ignored): probe_t_vs_qt target 0
    /// (θ=2.37 φ=5.73 λ=3.33, seed 12648430) at ε=1e-6 optimal w2 —
    /// coset-off finds cost 52.5, coset-on falls to the T baseline 53.
    /// Runs ONE mode per process (env LazyLock): set the mode via the
    /// test name. Prints the enum trace for diffing.
    /// Run: cargo test --release --lib probe_zeta_coset_t0_off -- --ignored --nocapture
    #[test]
    #[ignore]
    fn probe_zeta_coset_t0_off() {
        probe_zeta_coset_target(0, 1e-6, "0");
    }
    #[test]
    #[ignore]
    fn probe_zeta_coset_t0_on() {
        probe_zeta_coset_target(0, 1e-6, "1");
    }
    /// 1e-8 flip probe: probe_t_vs_qt target 6 (θ=1.80 φ=0.59 λ=1.62)
    /// — coset-off screen finds lde=24 (cost 73.5), coset-on drifts to
    /// the lde-78 fallback (cost 78).
    #[test]
    #[ignore]
    fn probe_zeta_coset_t6_1e8_off() {
        probe_zeta_coset_target(6, 1e-8, "0");
    }
    #[test]
    #[ignore]
    fn probe_zeta_coset_t6_1e8_on() {
        probe_zeta_coset_target(6, 1e-8, "1");
    }
    fn probe_zeta_coset_target(index: usize, eps: f64, coset: &str) {
        unsafe {
            std::env::set_var("CYCLOSYNTH_ZETA_COSET", coset);
            std::env::set_var("CYCLOSYNTH_TRACE", "1");
        }
        // SplitMix64 target gen, first triple of seed 12648430
        // (probe_t_vs_qt's Xs) — replicated from tests/qt_guard_1e5.rs.
        struct Xs(u64);
        impl Xs {
            fn next(&mut self) -> u64 {
                self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
                let mut z = self.0;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
                z ^ (z >> 31)
            }
            fn unit(&mut self) -> f64 {
                (self.next() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
            }
            fn range(&mut self, lo: f64, hi: f64) -> f64 {
                lo + (hi - lo) * self.unit()
            }
        }
        let mut rng = Xs(12648430);
        let mut tpl = (0.0, 0.0, 0.0);
        for _ in 0..=index {
            tpl = (
                rng.range(0.2, PI - 0.2),
                rng.range(0.1, 2.0 * PI - 0.1),
                rng.range(0.1, 2.0 * PI - 0.1),
            );
        }
        let (th, ph, la) = tpl;
        eprintln!("[t{index}] θ={th:.3} φ={ph:.3} λ={la:.3} ε={eps:e} coset={coset}");
        let (c, s) = ((th / 2.0).cos(), (th / 2.0).sin());
        let eilam = Complex64::from_polar(1.0, la);
        let eiphi = Complex64::from_polar(1.0, ph);
        let g = Complex64::from_polar(1.0, -(ph + la) / 2.0);
        let target: Mat2 = [
            [Complex64::new(c, 0.0) * g, -eilam * s * g],
            [eiphi * s * g, eiphi * eilam * Complex64::new(c, 0.0) * g],
        ];
        let r = SynthesizerQ::new(eps)
            .with_optimize_cost(true)
            .with_optimal_lde_window(2)
            .synthesize(target);
        match r {
            Some(r) => {
                let g = r.gates.as_deref().unwrap_or("");
                let (t, q) = gates_tq(g);
                eprintln!(
                    "[t{index}] RESULT lde={} T={t} Q={q} cost={} dist={:.3e}",
                    r.lde,
                    t as f64 + 3.5 * q as f64,
                    r.distance
                );
            }
            None => eprintln!("[t{index}] RESULT NONE"),
        }
    }

    /// Is the strict-filter ([0]) deep-ε screen blind to non-class-0
    /// solutions? Re-running tie targets with the relaxed filter should
    /// collapse first-hit lde if so.
    /// Run: cargo test --release --lib h1_dr_filter_blindness -- --ignored --nocapture
    #[test]
    #[ignore]
    fn h1_dr_filter_blindness() {
        fn xorshift64(s: &mut u64) -> u64 { *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; *s }
        fn rand_angle(s: &mut u64) -> f64 {
            let b = xorshift64(s) >> 11;
            (b as f64) / ((1u64 << 53) as f64) * 2.0 * std::f64::consts::PI
        }
        let mut state: u64 = 12648430 | 1;
        // probe_t_vs_qt target gen: theta in (0.2, PI-0.2), phi/lambda in (0.1, 2PI-0.1)
        let mut angles = Vec::new();
        for _ in 0..2 {
            let t = 0.2 + rand_angle(&mut state) / (2.0 * std::f64::consts::PI)
                * (std::f64::consts::PI - 0.4);
            let p = 0.1 + rand_angle(&mut state) / (2.0 * std::f64::consts::PI)
                * (2.0 * std::f64::consts::PI - 0.2);
            let l = 0.1 + rand_angle(&mut state) / (2.0 * std::f64::consts::PI)
                * (2.0 * std::f64::consts::PI - 0.2);
            angles.push((t, p, l));
        }
        eprintln!("targets (must match probe rows 0,1: θ=2.37/1.17): {angles:?}");
        for (i, &(t, p, l)) in angles.iter().enumerate() {
            let ct = (t / 2.0).cos();
            let st = (t / 2.0).sin();
            let gp = Complex64::from_polar(1.0, -(p + l) / 2.0);
            let target: Mat2 = [
                [gp * Complex64::new(ct, 0.0), gp * (-Complex64::from_polar(st, l))],
                [gp * Complex64::from_polar(st, p), gp * Complex64::from_polar(ct, p + l)],
            ];
            for (label, filt) in [("strict[0]", vec![0u32]), ("relaxed[0,1,15]", vec![0u32, 1, 15])] {
                let synth = SynthesizerQ::new(1e-8).with_dc_dr_filter(filt);
                let t0 = std::time::Instant::now();
                let r = synth.synthesize(target);
                match r {
                    Some(r) => {
                        let g = r.gates.as_deref().unwrap_or("");
                        let (tc, qc) = gates_tq(g);
                        eprintln!(
                            "target {i} {label}: lde={} T={tc} Q={qc} cost={} dist={:.2e} t={:.1}s",
                            r.lde,
                            gates_cost(g, 7) as f64 / 2.0,
                            r.distance,
                            t0.elapsed().as_secs_f64()
                        );
                    }
                    None => eprintln!("target {i} {label}: NONE t={:.1}s", t0.elapsed().as_secs_f64()),
                }
            }
        }
    }
}
