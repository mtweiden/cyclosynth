//! FGKM prefix-set construction and right-coset dedup.

use super::*;

/// Per-m memoization cache (m = syllable count): an `Arc`-shared `Vec` so
/// cache hits are a refcount bump, not a deep clone.
type CacheByM<V> = LazyLock<Mutex<HashMap<u32, Arc<Vec<V>>>>>;

// ─── FGKM canonical-form prefix generation (syllable-count enumeration) ──
//
// Mirrors `clifford_t::build_ma_prefix_set`. Where Clifford+T enumerates
// Matsumoto–Amano words `T^{a₀} · ∏ (HS^bᵢ T) · C` of T-count t', this
// enumerates Forest–Gosset–Kliuchnikov–McKinnon words
// `∏ R_{pᵢ}(aᵢπ/8) · C` of syllable count m. A syllable is one
// `R_p(a·π/8)` with `p ∈ {x,y,z}, a ∈ {1,2,3}`; consecutive syllables
// have distinct axes (Lemma 3.1). m is the right enumeration coordinate
// because each syllable peels √2-exp by ≥1, matching the inner lde
// split; Q-count (Σaᵢ ∈ [m, 3m]) does not.

/// Global cache for `build_fgkm_prefix_set` results, keyed by syllable count `m`.
pub(crate) static FGKM_PREFIX_CACHE: CacheByM<U2Q> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Canonical float key for a `U2Q` matrix, invariant under global U(1)
/// phase. Mirrors `clifford_t::canonical_key`: rotates the flattened
/// matrix so the largest-magnitude entry is real-positive, then rounds to
/// 6 decimals. Used for O(n)-average dedup in `build_fgkm_prefix_set_inner`.
pub(crate) fn canonical_key_q(u: &U2Q) -> [i64; 8] {
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
pub fn build_fgkm_prefix_set(m: u32) -> Arc<Vec<U2Q>> {
    {
        let cache = FGKM_PREFIX_CACHE.lock().unwrap();
        if let Some(v) = cache.get(&m) {
            return Arc::clone(v);
        }
    }
    let result = Arc::new(build_fgkm_prefix_set_inner(m));
    FGKM_PREFIX_CACHE
        .lock()
        .unwrap()
        .insert(m, Arc::clone(&result));
    result
}

/// Cache for prefix `(T, Q)` gate counts (parallel to `FGKM_PREFIX_CACHE`).
pub(crate) static BUILD_L_Q_TQ_CACHE: CacheByM<(usize, usize)> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Pre-computed `(T_count, Q_count)` of the canonical [`BlochDecomposer`]
/// decomposition for each prefix in `build_fgkm_prefix_set(m)`, indexed parallel to
/// that Vec. Cached forever per `m`; the caller applies its own Q-cost
/// weight. NB: the weighted cost is **not a lower bound** on
/// `cost(U_L · U_R)` — U_R can cancel parts of U_L. It is used as a
/// heuristic ranking + prune, not a sound bound.
pub fn build_fgkm_prefix_gate_counts(m: u32) -> Arc<Vec<(usize, usize)>> {
    {
        let cache = BUILD_L_Q_TQ_CACHE.lock().unwrap();
        if let Some(v) = cache.get(&m) {
            return Arc::clone(v);
        }
    }
    let prefixes = build_fgkm_prefix_set(m);
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

/// Right-coset dedup gate for the ζ prefix lists.
/// `CYCLOSYNTH_ZETA_COSET=0` disables it; anything else (or unset)
/// enables. Read once per process.
pub(crate) static ZETA_COSET_DEDUP: LazyLock<bool> = LazyLock::new(|| {
    !matches!(std::env::var("CYCLOSYNTH_ZETA_COSET").as_deref(), Ok("0"))
});

/// The 8-element lde-0 Clifford subgroup ⟨S, X⟩ as U2Q, rebuilt from
/// [`crate::synthesis::cliffords::CLIFFORD_TABLE_T`] entry names via [`crate::synthesis::cliffords::CLIFFORD_LDE0_IDX`] — the same
/// name-folding route `build_fgkm_prefix_set_inner` uses for its Clifford suffixes
/// (NOT the det-1 U2T table matrices, which differ by ζ-power phases;
/// orbit keys must match the list's own construction including float
/// tie-breaking, see `build_fgkm_prefix_orbits`).
pub(crate) fn lde0_cliffords_q() -> [U2Q; 8] {
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
/// [`FGKM_PREFIX_CACHE`], keyed by syllable count `m`).
pub(crate) static FGKM_PREFIX_ORBIT_CACHE: CacheByM<usize> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Orbit id per prefix under RIGHT multiplication by the lde-0
/// Clifford subgroup ⟨S, X⟩, mod global phase (id = min list index
/// among key-matched mates). Mates whose float key is absent from the
/// list stay unlinked — conservative: less dedup, never less coverage.
/// The linking is by float value and `build_fgkm_prefix_set` stores the unreduced
/// peel-depth k, so an orbit can span several k; production dedup
/// groups by (orbit, k), within which mates are exact ring-unit coset
/// partners (pinned by `zeta_coset_orbits_sound`).
pub fn build_fgkm_prefix_orbits(m: u32) -> Arc<Vec<usize>> {
    {
        let cache = FGKM_PREFIX_ORBIT_CACHE.lock().unwrap();
        if let Some(v) = cache.get(&m) {
            return Arc::clone(v);
        }
    }
    let prefixes = build_fgkm_prefix_set(m);
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
    FGKM_PREFIX_ORBIT_CACHE
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
pub(crate) fn coset_keep_mask(cands: &[(usize, usize)], keys: &[(usize, u32)]) -> Vec<bool> {
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

/// Cached per-m `(orbit id, k)` dedup keys, parallel to `build_fgkm_prefix_set(m)`.
pub(crate) static BUILD_L_Q_COSET_KEY_CACHE: CacheByM<(usize, u32)> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// The `(orbit id, unreduced k)` dedup class per prefix of
/// `build_fgkm_prefix_set(m)` — the key [`coset_keep_mask`] groups by.
pub fn build_fgkm_prefix_coset_keys(m: u32) -> Arc<Vec<(usize, u32)>> {
    {
        let cache = BUILD_L_Q_COSET_KEY_CACHE.lock().unwrap();
        if let Some(v) = cache.get(&m) {
            return Arc::clone(v);
        }
    }
    let prefixes = build_fgkm_prefix_set(m);
    let orbit = build_fgkm_prefix_orbits(m);
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

pub(crate) fn build_fgkm_prefix_set_inner(m: u32) -> Vec<U2Q> {
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

    let mut bodies: Vec<U2Q> = Vec::new();
    enumerate_bodies(m, 3, U2Q::eye(), &syllables, &mut bodies);

    // Append every Clifford suffix to every body.
    //
    // The stored `k` is the UNREDUCED accumulation — a *peel-depth*
    // coordinate matching the inner-LLL+SE shell split (`lde_inner =
    // lde_total − u_l.k`), NOT the prefix's reduced matrix lde. Reducing it
    // makes z-axis and Clifford-heavy prefixes drop to k ≈ 0-1, so their
    // suffix searches run at nearly full depth (large wall regression); a
    // sound reduction needs a dual coordinate (reduced lde for cost, peel
    // depth for shell selection — see docs/design_certified_optimal_cost.md).
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
pub(crate) fn enumerate_bodies(
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
