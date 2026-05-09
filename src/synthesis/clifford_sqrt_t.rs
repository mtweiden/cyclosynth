//! Clifford+√T synthesis backend over Z[ζ_16].
//!
//! [`SynthesizerQ`] is one of two backends behind the unified user-facing
//! [`crate::synthesis::Synthesizer`]; the other is
//! [`crate::synthesis::clifford_t::SynthesizerT`] (Clifford+T, Z[ω]). Code
//! shouldn't construct `SynthesizerQ` directly — use `Synthesizer` with
//! `sqrt_t = true`. The struct stays public so the test suite can poke at
//! it (`pub` instead of `pub(crate)`).
//!
//! ## Backend (hybrid)
//!
//! For `k ≤ BRUTE_LIMIT` (=4): brute-force enumeration via
//! [`crate::synthesis::search_zeta::phase1_brute`] — cheap exact-find
//! for small Clifford+√T targets.
//!
//! For `k > BRUTE_LIMIT`: 16D L²-LLL + Schnorr-Euchner via
//! [`crate::synthesis::lenstra_zeta::phase1`] with adaptive leaf budget
//! scaling exponentially in `k`. Reaches ε ≲ 1e-5 at k ≈ 30.
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
use crate::synthesis::distance::{diamond_distance_float, diamond_distance_u2q_float, Mat2};
use crate::synthesis::lenstra_zeta::{phase1_with_stop, IntScratch16};
use crate::synthesis::search_zeta::{phase1_brute, uv_to_xy_zeta};
use num_complex::Complex64;
use std::collections::HashMap;
use std::f64::consts::PI;
use std::sync::{Arc, LazyLock, Mutex};
use std::sync::atomic::AtomicBool;

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
pub(crate) fn build_l_q(m: u32) -> Arc<Vec<U2Q>> {
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
}

/// k cutoff: brute-force handles `k ≤ BRUTE_LIMIT`, the 16D LLL+SE
/// backend handles larger k.
///
/// **Was 4** until profiling found that `phase1_brute(4)` (~5·10⁸ shell
/// points, ~10 s) was wasted on every approximation target at moderate-
/// or-deep ε, since the actual answer lives in the lattice regime at k≥5.
/// At BRUTE_LIMIT=3, brute tops out at ~10⁷ shell points (~100 ms) and
/// the lattice walker handles k=4 efficiently when needed.
const BRUTE_LIMIT: u32 = 3;

/// Estimate the smallest lde at which a generic SU(2) target is reachable
/// within ε. Empirical from the ε-1e-3 / ε-1e-4 / ε-1e-5 benches: lde lands
/// at roughly `⌈-log₂(ε)⌉ - 3`, with a per-target jitter of ±2. We start
/// the lattice search 2 below the estimate so that easy targets land
/// without an extra full-shell sweep, and harder ones advance into deeper k.
fn lattice_lde_estimate(epsilon: f64) -> u32 {
    if !(epsilon > 0.0 && epsilon < 1.0) {
        return 0;
    }
    let raw = (-epsilon.log2()).ceil() as i32 - 3;
    raw.max(0) as u32
}

/// Two-pass leaf-budget strategy (mirrors 8D `dc_search`):
///   - **Pass 1** is the aggressive cap: small enough that doomed k
///     (where no sol exists in this ε regime) bail quickly. At each k
///     in the lattice range we try Pass 1 first.
///   - If Pass 1 finds a sol, return.
///   - If Pass 1 exhausts the SE region (no budget hit, no sol), the
///     search was complete at this k — advance to k+1.
///   - If Pass 1 budget-hits without finding a sol, mark this k for
///     Pass 2 retry and continue to k+1.
///   - **Pass 2** is the unbounded cap: only run on k's that Pass 1
///     budget-hit, after the Pass-1 sweep finishes without finding a
///     sol elsewhere. Guarantees no completeness loss.
///
/// Empirically: at ε=1e-5 target_01 lands at lde=13 but k=12 has no
/// sol — single-pass with 4G budget burns ~30 s on k=12 before
/// advancing. Pass 1 at 100 M lets k=12 bail in ~7 s, k=13 finds
/// quickly.
const PASS1_CAP: u64 = 100_000_000;
const PASS2_CAP: u64 = 4_000_000_000;

/// Per-prefix budget for the Z1 D&C dispatcher's pass 1.
///
/// **Tried and abandoned**: a tiered budget by lde proximity to
/// `lattice_lde_estimate(eps)` — small cap at "below-expected" lde
/// levels (where the SE region is presumed empty), full cap at the
/// expected zone. Two values tested:
///
///   100K low_cap: target_02 ε=1e-7 regressed lde=19 → lde=20.
///                 SE budget is shared across 16 z[15]-subtree workers
///                 → 6K/worker, too few; some genuine answers missed.
///   500K low_cap: target_00 ε=1e-7 regressed lde=20 → lde=21.
///                 The pass-1 region for target_00's right prefix at
///                 lde=20 was empty within 5M budget, so the parallel
///                 race outcome shifted: at small budget the
///                 first-find-wins resolution returned a higher-lde
///                 answer.
///
/// The flat 5M cap preserves all baseline lde across our test set
/// while still being 2× faster than 10M on NO-levels.
const DC_PASS1_CAP: u64 = 5_000_000;

/// At deep ε the post-LLL SE region grows exponentially in k_inner. The
/// flat 5M leaf budget that works at ε≥1e-7 hits budget at every lde
/// 22..27 at ε=1e-8 without finding a single candidate (per probe in
/// `bin/probe_eps_1e8_v2.rs`). Scale pass1's per-prefix budget with ε so
/// the SE walk has room to reach the cap interior.
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

/// Pass 2: only runs on lde levels where pass 1's prefixes hit budget
/// without finding (= search may have missed a solution past pass-1
/// budget). 10M is generous enough to cover the hard-target cases where
/// the right prefix needs many millions of leaves to converge.
const DC_PASS2_CAP: u64 = 10_000_000;

/// MPFR-precision 2×2 complex matrix: `[[(re,im); 2]; 2]`.
///
/// Used by `synthesize_mpfr` and the MPFR D&C path to keep cap
/// localization at full precision down to ε ≤ 1e-8 (where f64 ULP
/// exceeds the cap-radial direction Δ_y/R = ε²/4).
pub type Mat2Mpfr = [[(rug::Float, rug::Float); 2]; 2];

/// Lift a f64 `Mat2` to `Mat2Mpfr` at the given precision. No precision
/// gain by itself; useful when callers have f64 targets but want to feed
/// the MPFR pipeline (e.g. for testing or for D&C structure access).
pub fn mat2_to_mat2_mpfr(target: &Mat2, prec: u32) -> Mat2Mpfr {
    use rug::Float as RFloat;
    [
        [
            (RFloat::with_val(prec, target[0][0].re), RFloat::with_val(prec, target[0][0].im)),
            (RFloat::with_val(prec, target[0][1].re), RFloat::with_val(prec, target[0][1].im)),
        ],
        [
            (RFloat::with_val(prec, target[1][0].re), RFloat::with_val(prec, target[1][0].im)),
            (RFloat::with_val(prec, target[1][1].re), RFloat::with_val(prec, target[1][1].im)),
        ],
    ]
}

/// Convert a `U2Q` to `Mat2Mpfr`. Reads ZZeta integer coefficients and
/// evaluates against `(cos(kπ/8), sin(kπ/8))` for k=0..7 in MPFR, then
/// divides by `√2^k` in MPFR (binary shift + optional ×1/√2).
pub fn u2q_to_mat2_mpfr(u: &U2Q, prec: u32) -> Mat2Mpfr {
    use rug::Float as RFloat;
    use std::f64::consts::PI;
    use crate::rings::types::int_to_f64;

    let two = RFloat::with_val(prec, 2.0);
    let inv_sqrt2 = RFloat::with_val(prec, 1.0) / two.clone().sqrt();
    let half_k = u.k / 2;
    let mut inv_scale = RFloat::with_val(prec, 1.0);
    inv_scale >>= half_k;
    if u.k % 2 == 1 {
        inv_scale *= &inv_sqrt2;
    }

    let basis: [(RFloat, RFloat); 8] = std::array::from_fn(|k| {
        let theta = (k as f64) * PI / 8.0;
        (RFloat::with_val(prec, theta.cos()), RFloat::with_val(prec, theta.sin()))
    });

    let zzeta_to_re_im = |z: &crate::rings::ZZeta| -> (RFloat, RFloat) {
        let coeffs = [
            int_to_f64(z.a), int_to_f64(z.b), int_to_f64(z.c), int_to_f64(z.d),
            int_to_f64(z.e), int_to_f64(z.f), int_to_f64(z.g), int_to_f64(z.h),
        ];
        let mut re = RFloat::with_val(prec, 0.0);
        let mut im = RFloat::with_val(prec, 0.0);
        for k in 0..8 {
            let c = RFloat::with_val(prec, coeffs[k]);
            re += RFloat::with_val(prec, &c * &basis[k].0);
            im += RFloat::with_val(prec, &c * &basis[k].1);
        }
        (re * &inv_scale, im * &inv_scale)
    };

    [
        [zzeta_to_re_im(&u.u11), zzeta_to_re_im(&u.u12)],
        [zzeta_to_re_im(&u.u21), zzeta_to_re_im(&u.u22)],
    ]
}

/// MPFR analog of `u2q_dag_times_mat2`. Computes `U_L† · target` at MPFR
/// precision, lifting `U_L`'s exact ZZeta entries to MPFR via
/// `u2q_to_mat2_mpfr`. The `target` precision sets the result precision.
pub fn u2q_dag_times_mat2_mpfr(u_l: &U2Q, target: &Mat2Mpfr, prec: u32) -> Mat2Mpfr {
    use rug::Float as RFloat;
    let u = u2q_to_mat2_mpfr(u_l, prec);
    // (U†)[i][j] = conj(U[j][i])
    let ud00 = (u[0][0].0.clone(), RFloat::with_val(prec, -&u[0][0].1));
    let ud01 = (u[1][0].0.clone(), RFloat::with_val(prec, -&u[1][0].1));
    let ud10 = (u[0][1].0.clone(), RFloat::with_val(prec, -&u[0][1].1));
    let ud11 = (u[1][1].0.clone(), RFloat::with_val(prec, -&u[1][1].1));
    let mul = |a: &(RFloat, RFloat), b: &(RFloat, RFloat)| -> (RFloat, RFloat) {
        let re = RFloat::with_val(prec, &a.0 * &b.0)
            - RFloat::with_val(prec, &a.1 * &b.1);
        let im = RFloat::with_val(prec, &a.0 * &b.1)
            + RFloat::with_val(prec, &a.1 * &b.0);
        (RFloat::with_val(prec, re), RFloat::with_val(prec, im))
    };
    let add = |a: (RFloat, RFloat), b: (RFloat, RFloat)| -> (RFloat, RFloat) {
        (RFloat::with_val(prec, &a.0 + &b.0), RFloat::with_val(prec, &a.1 + &b.1))
    };
    [
        [
            add(mul(&ud00, &target[0][0]), mul(&ud01, &target[1][0])),
            add(mul(&ud00, &target[0][1]), mul(&ud01, &target[1][1])),
        ],
        [
            add(mul(&ud10, &target[0][0]), mul(&ud11, &target[1][0])),
            add(mul(&ud10, &target[0][1]), mul(&ud11, &target[1][1])),
        ],
    ]
}

/// Column-1 of an MPFR target as `(Re V₀₀, Im V₀₀, Re V₁₀, Im V₁₀)`.
/// MPFR analog of `unitary_to_uv_zeta`.
pub fn unitary_to_uv_zeta_mpfr(target: &Mat2Mpfr) -> [rug::Float; 4] {
    [
        target[0][0].0.clone(),
        target[0][0].1.clone(),
        target[1][0].0.clone(),
        target[1][0].1.clone(),
    ]
}

/// MPFR analog of `det_phase_of`. Computes `det = a·d − b·c` (complex)
/// in MPFR, then `arg(det) / (π/8) → Z/16`. Returns the phase index in
/// 0..16. f64 is sufficient for the angle quantization step (`atan2`
/// resolves 1/16 well enough), but the determinant's complex value is
/// computed in MPFR for stability when the target is near a det-phase
/// boundary.
pub fn det_phase_of_mat2_mpfr(target: &Mat2Mpfr) -> u32 {
    use rug::Float as RFloat;
    use std::f64::consts::PI;
    let prec = target[0][0].0.prec();
    let a = &target[0][0];
    let b = &target[0][1];
    let c = &target[1][0];
    let d = &target[1][1];
    // a·d
    let ad_re = RFloat::with_val(prec, &a.0 * &d.0)
        - RFloat::with_val(prec, &a.1 * &d.1);
    let ad_im = RFloat::with_val(prec, &a.0 * &d.1)
        + RFloat::with_val(prec, &a.1 * &d.0);
    // b·c
    let bc_re = RFloat::with_val(prec, &b.0 * &c.0)
        - RFloat::with_val(prec, &b.1 * &c.1);
    let bc_im = RFloat::with_val(prec, &b.0 * &c.1)
        + RFloat::with_val(prec, &b.1 * &c.0);
    let det_re = RFloat::with_val(prec, &ad_re - &bc_re);
    let det_im = RFloat::with_val(prec, &ad_im - &bc_im);
    let arg = det_im.to_f64().atan2(det_re.to_f64());
    let normalized = (arg / (PI / 8.0)).round() as i32;
    normalized.rem_euclid(16) as u32
}

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
    /// `max_lde = 30`: high enough to reach ε ≲ 1e-5 via the LLL backend.
    /// Override via [`with_max_lde`] for a tighter (faster) ceiling, e.g.
    /// `with_max_lde(4)` to stay in the brute regime.
    /// Construct a synthesizer with sensible defaults. **Z1 D&C is
    /// auto-enabled at ε ≤ 1e-6**: empirically, single search becomes
    /// pathological at this depth (per-target time can run into minutes,
    /// 1/3 fail at max_lde=30), while filtered D&C stays in the seconds.
    /// At ε > 1e-6 we keep single search since it's always faster on
    /// the lattice levels it needs to reach.
    ///
    /// Override either via [`Self::with_dc_split`] / [`Self::with_dc_dr_filter`].
    pub fn new(epsilon: f64) -> Self {
        // At deep ε, auto-enable D&C with the empirically-best filter
        // (m=1, |d_R|≤1). See `project_zeta_z1_empirics.md` for the data
        // — wins 22-1534× over single at ε=1e-7. Threshold of 1e-6 is
        // a touch aggressive (single still has variance there), but the
        // upside on slow targets (hours → seconds) dominates the
        // downside on easy targets.
        // Auto-D&C config:
        //   ε > 1e-6:   single search (D&C overhead not worth it)
        //   ε ∈ [1e-7, 1e-6]: m=1, |d_R|≤1 (relaxed). 36 prefixes; the
        //                    relaxed filter avoids structural gaps at
        //                    low lde where m=2 strict misses.
        //   ε ≤ 1e-7:   m=2, d_R=0 (strict). 144 prefixes but each at
        //               higher k_prefix → smaller k_inner → faster SE
        //               per prefix. Empirically wins lde quality
        //               (consistent minimum-lde finds) AND speed at
        //               this depth (27% faster vs m=1 on 8-target
        //               bench at ε=1e-7). Lde quality wins because the
        //               m=2 prefix set has more k_inner coverage at
        //               deep lde. Avoid at moderate ε (1e-6) because
        //               the strict filter creates structural gaps in
        //               the prefix set there (lde regression seen).
        // Auto-D&C config:
        //   ε > 1e-6:   single search (D&C overhead not worth it)
        //   ε ∈ (1e-7, 1e-6]: m=1, |d_R|≤1 (relaxed). 36 prefixes; the
        //                    relaxed filter avoids structural gaps at
        //                    low lde where m=2 strict misses.
        //   ε ≤ 1e-7:   m=2, d_R=0 (strict). 144 prefixes but each at
        //               higher k_prefix → smaller k_inner → faster SE
        //               per prefix. Empirically (8-target bench at
        //               ε=1e-7, time_zeta_synthesis seed):
        //                 m=1: 2856 ms/target avg, worst lde=24
        //                 m=2: 2036 ms/target avg, worst lde=20
        //               m=2 wins 40% on speed AND has consistent
        //               lde=19-20 finds (m=1 has occasional lde=24
        //               regressions on hard targets). Avoid at moderate
        //               ε (1e-6) because the strict filter creates
        //               structural gaps at low lde (lde regressions
        //               seen 15 → 17).
        let (dc_split, dc_dr_filter) = if epsilon <= 1e-7 {
            (Some(2u32), vec![0u32])
        } else if epsilon <= 1e-6 {
            (Some(1u32), vec![0u32, 1, 15])
        } else {
            (None, Vec::new())
        };
        // Default max_lde scales with ε. At ε=1e-7 we observed
        // single-search lde=19; D&C may need lde=20-21 with the
        // |d_R|≤1 filter (each prefix's k_inner = k_total − k_prefix).
        // 35 is safe for ε down to ~1e-9.
        let max_lde = if epsilon <= 1e-7 { 35 } else { 30 };
        // **Adaptive precision default**: f64 GS works through ε=1e-7
        // (52-bit mantissa vs ~46-bit requirement). At ε ≤ 1e-8 the
        // requirement crosses ~50 bits, leaving f64 with a 2-bit margin —
        // empirically the LLL spends much of its time in escalation,
        // doubling LLL cost vs going to MPFR-80 directly. Skip the f64
        // attempt entirely there.
        //
        // Inside `phase1_with_stop` the precision ladder still escalates
        // f64 → MPFR-80 if any individual LLL call fails the unimodularity
        // check, so users who manually call `with_f64_gs(true)` at deep ε
        // get correctness via the ladder (just slower than the
        // MPFR-only path at ε ≤ 1e-8).
        let use_f64_gs = epsilon > 1e-8;

        // **Min/max lde scaling at deep ε**: Z[ζ_16] needs ~0.30× the lde
        // of Z[ω] to reach the same ε (per `project_baseline_2026_05_07.md`
        // empirics). Without scaling, `min_lde=0` wastes time scanning
        // low-lde levels guaranteed to fail, and `max_lde=35` is at the
        // boundary for ε=1e-8 where the predicted lde is ~24-30.
        // Empirics: ε=1e-7 lands at lde=19-20. At ε=1e-8 (10× tighter,
        // ~+2 ldes per density argument), expect lde≈22-24 typical, with
        // hard targets needing 28-32. So at ε=1e-8 use min_lde=18, max_lde=45.
        let log2_recip = if epsilon > 0.0 && epsilon < 1.0 {
            (1.0 / epsilon).log2()
        } else { 0.0 };
        let min_lde = if epsilon <= 1e-8 {
            // ~0.7×log2(1/ε) — leaves a small buffer below the typical landing
            (0.7 * log2_recip).floor() as u32
        } else {
            0
        };
        let max_lde_override = if epsilon <= 1e-8 {
            // 1.7×log2(1/ε) covers ~+15 ldes above the typical landing.
            (1.7 * log2_recip).ceil() as u32
        } else {
            max_lde
        };

        // **Auto-BKZ default**: enable BKZ-4 only at ε ≤ 1e-7. Empirically
        // (8-target Rz·Ry·Rz bench, seed 0xC0FFEEBAADD0E):
        //   ε=1e-5: 0.37× (BKZ overhead crushes already-cheap SE walks)
        //   ε=1e-6: 1.04× (break-even, high variance)
        //   ε=1e-7: 1.44× (consistent win; one lde improvement)
        // At ε≤1e-7 the post-LLL SE region is large enough that BKZ-4's
        // tighter Hermite factor pays for the per-LLL-call tour cost.
        // Above that threshold, SE is already cheap and BKZ adds pure
        // overhead. Override via `with_bkz(β)`.
        let bkz_block_size = if epsilon <= 1e-7 { 4 } else { 0 };
        Self {
            epsilon,
            min_lde,
            max_lde: max_lde_override,
            dc_split,
            dc_dr_filter,
            use_f64_gs,
            bkz_block_size,
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

    /// MPFR-precision entry point for synthesis: caller provides
    /// `v_mpfr = (Re V₁₁, Im V₁₁, Re V₂₁, Im V₂₁)` at MPFR precision.
    ///
    /// **Why**: at ε ≤ 1e-8 the cap-radial direction `Δ_y/R = ε²/4` falls
    /// below f64 ULP at unit scale (~2.2e-16). The default
    /// [`synthesize`] receives a f64 `Mat2` target and so the lattice
    /// cap is localized with > 1 cap-width of slop in the radial axis;
    /// the SE walk explores the wrong region and the synthesizer never
    /// finds. This entry point lifts `v` to MPFR before computing the
    /// cap, bypassing the f64 floor.
    ///
    /// `target` (f64) is still used for the `diamond_distance_u2q_float`
    /// final check — that fn evaluates the candidate's ZZeta entries
    /// directly in MPFR, so f64-target precision is sufficient there
    /// (see `feedback_diamond_distance_frobenius.md`).
    ///
    /// Single-search path only (no D&C). For ε < 1e-7 callers.
    pub fn synthesize_v_mpfr(&self, v_mpfr: &[rug::Float; 4], target: Mat2) -> Option<SynthResultQ> {
        use crate::synthesis::diag;
        use crate::synthesis::lenstra_zeta::phase1_with_stop_mpfr;
        use crate::synthesis::search_zeta::uv_to_xy_zeta_mpfr;
        use crate::synthesis::lenstra_zeta::scratch::compute_prec_q;
        let trace = diag::trace_enabled();
        if trace {
            diag::reset_all();
        }

        let epsilon = self.epsilon;
        let d = det_phase_of(&target);
        let lattice_start = lattice_lde_estimate(epsilon)
            .saturating_sub(2)
            .max(BRUTE_LIMIT + 1)
            .max(self.min_lde);

        let prec = compute_prec_q(epsilon);
        let mut scratch = IntScratch16::new(epsilon);
        scratch.use_f64_gs = self.use_f64_gs;
        scratch.bkz_block_size = self.bkz_block_size;

        for k in lattice_start..=self.max_lde {
            let t_k = std::time::Instant::now();
            let y_mpfr = uv_to_xy_zeta_mpfr(v_mpfr, k, prec);
            let budget_hit = AtomicBool::new(false);
            let target_local = target;
            let should_stop = |x: &[i64; 16]| -> bool {
                let cand = solution_to_u2q_d(x, k, d);
                diamond_distance_u2q_float(&cand, &target_local) < epsilon
            };
            let sols = phase1_with_stop_mpfr(
                &mut scratch, &y_mpfr, v_mpfr, k, epsilon,
                PASS1_CAP, &budget_hit, should_stop,
            );
            for sol in &sols {
                let cand: U2Q = solution_to_u2q_d(sol, k, d);
                let dist = diamond_distance_u2q_float(&cand, &target);
                if dist < epsilon {
                    if trace {
                        eprintln!("[zeta-mpfr] lde={k:>2}  FOUND  dist={:.3e}  t={:.0}ms",
                            dist, t_k.elapsed().as_secs_f64() * 1000.0);
                    }
                    let gates = BlochDecomposer.decompose(&cand);
                    return Some(SynthResultQ {
                        gates: Some(gates),
                        lde: k,
                        distance: dist,
                    });
                }
            }
            if trace {
                eprintln!("[zeta-mpfr] lde={k:>2}  none{}  t={:.0}ms",
                    if budget_hit.load(std::sync::atomic::Ordering::Relaxed) { " (budget hit)" } else { "" },
                    t_k.elapsed().as_secs_f64() * 1000.0);
            }
        }
        None
    }

    /// Full MPFR-precision synthesis. Mirrors `synthesize` but with
    /// `Mat2Mpfr` target, `Mat2Mpfr` per-prefix `m_inner`, MPFR `v` and
    /// MPFR `y` throughout. Required at ε ≤ 1e-8 where f64 ULP exceeds
    /// the cap-radial direction Δ_y/R.
    ///
    /// Brute regime + single-search lattice + 2-pass D&C dispatcher.
    pub fn synthesize_mpfr(&self, target_mpfr: &Mat2Mpfr) -> Option<SynthResultQ> {
        use crate::synthesis::diag;
        use crate::synthesis::lenstra_zeta::phase1_with_stop_mpfr;
        use crate::synthesis::search_zeta::uv_to_xy_zeta_mpfr;
        use crate::synthesis::distance::diamond_distance_u2q_mpfr_target;

        let trace = diag::trace_enabled();
        if trace { diag::reset_all(); }

        let prec = target_mpfr[0][0].0.prec();
        let epsilon = self.epsilon;
        let d_target = det_phase_of_mat2_mpfr(target_mpfr);
        let v_outer = unitary_to_uv_zeta_mpfr(target_mpfr);

        let lattice_start = lattice_lde_estimate(epsilon)
            .saturating_sub(2)
            .max(BRUTE_LIMIT + 1)
            .max(self.min_lde);

        // Brute regime
        for k in self.min_lde..=BRUTE_LIMIT.min(self.max_lde) {
            let sols = phase1_brute(k);
            for sol in &sols {
                let cand: U2Q = solution_to_u2q_d(sol, k, d_target);
                let dist = diamond_distance_u2q_mpfr_target(&cand, target_mpfr);
                if dist < epsilon {
                    let gates = BlochDecomposer.decompose(&cand);
                    return Some(SynthResultQ { gates: Some(gates), lde: k, distance: dist });
                }
            }
        }

        // D&C path
        if let Some(m_split) = self.dc_split {
            let mut pass2_queue: Vec<u32> = Vec::new();
            for k in lattice_start..=self.max_lde {
                let t_k = std::time::Instant::now();
                if k <= m_split { continue; }
                if trace {
                    eprintln!("[zeta-mpfr] dc lde={k:>2} m={m_split} pass1 dispatching ...");
                }
                let (result, budget_hit) = self.dc_search_q_mpfr(
                    target_mpfr, k, m_split, dc_pass1_cap_for(epsilon),
                );
                if let Some(r) = result {
                    if trace {
                        eprintln!("[zeta-mpfr] dc lde={k:>2} pass1 FOUND dist={:.3e} t={:.0}ms",
                            r.distance, t_k.elapsed().as_secs_f64() * 1000.0);
                    }
                    return Some(r);
                }
                if trace {
                    eprintln!("[zeta-mpfr] dc lde={k:>2} pass1 none{} t={:.0}ms",
                        if budget_hit { " (budget hit)" } else { "" },
                        t_k.elapsed().as_secs_f64() * 1000.0);
                    let s = diag::snapshot();
                    eprintln!("  [diag k={k}] se_leaves={} norm_rej={} bilin_rej={} align_rej={} sols={}",
                        s.se_callbacks, s.norm_rejected, s.bilinear_rejected,
                        s.align_rejected, s.sols_returned);
                    diag::reset_all();
                }
                if budget_hit { pass2_queue.push(k); }
            }
            for k in pass2_queue {
                let t_k = std::time::Instant::now();
                if trace {
                    eprintln!("[zeta-mpfr] dc lde={k:>2} m={m_split} pass2 dispatching ...");
                }
                let (result, _) = self.dc_search_q_mpfr(
                    target_mpfr, k, m_split, dc_pass2_cap_for(epsilon),
                );
                if let Some(r) = result {
                    if trace {
                        eprintln!("[zeta-mpfr] dc lde={k:>2} pass2 FOUND dist={:.3e} t={:.0}ms",
                            r.distance, t_k.elapsed().as_secs_f64() * 1000.0);
                    }
                    return Some(r);
                }
                if trace {
                    eprintln!("[zeta-mpfr] dc lde={k:>2} pass2 none t={:.0}ms",
                        t_k.elapsed().as_secs_f64() * 1000.0);
                }
            }
            return None;
        }

        // Single-search path
        let mut scratch = IntScratch16::new(epsilon);
        scratch.use_f64_gs = self.use_f64_gs;
        scratch.bkz_block_size = self.bkz_block_size;
        for k in lattice_start..=self.max_lde {
            let t_k = std::time::Instant::now();
            let y_mpfr = uv_to_xy_zeta_mpfr(&v_outer, k, prec);
            let budget_hit = AtomicBool::new(false);
            let target_local = target_mpfr.clone();
            let should_stop = |x: &[i64; 16]| -> bool {
                let cand = solution_to_u2q_d(x, k, d_target);
                diamond_distance_u2q_mpfr_target(&cand, &target_local) < epsilon
            };
            let sols = phase1_with_stop_mpfr(
                &mut scratch, &y_mpfr, &v_outer, k, epsilon,
                PASS1_CAP, &budget_hit, should_stop,
            );
            for sol in &sols {
                let cand: U2Q = solution_to_u2q_d(sol, k, d_target);
                let dist = diamond_distance_u2q_mpfr_target(&cand, target_mpfr);
                if dist < epsilon {
                    if trace {
                        eprintln!("[zeta-mpfr] single lde={k:>2} FOUND dist={:.3e} t={:.0}ms",
                            dist, t_k.elapsed().as_secs_f64() * 1000.0);
                    }
                    let gates = BlochDecomposer.decompose(&cand);
                    return Some(SynthResultQ { gates: Some(gates), lde: k, distance: dist });
                }
            }
            if trace {
                eprintln!("[zeta-mpfr] single lde={k:>2} none{} t={:.0}ms",
                    if budget_hit.load(std::sync::atomic::Ordering::Relaxed) { " (budget hit)" } else { "" },
                    t_k.elapsed().as_secs_f64() * 1000.0);
            }
        }
        None
    }

    /// MPFR-precision Z1 D&C dispatcher. Mirror of `dc_search_q` but
    /// computes `m_inner = U_L† · target` in MPFR, derives MPFR
    /// `v_inner` and `y_inner`, and calls `phase1_with_stop_mpfr`.
    fn dc_search_q_mpfr(
        &self,
        target: &Mat2Mpfr,
        k_total: u32,
        m_split: u32,
        per_prefix_cap: u64,
    ) -> (Option<SynthResultQ>, bool) {
        use rayon::prelude::*;
        use crate::synthesis::lenstra_zeta::phase1_with_stop_mpfr;
        use crate::synthesis::search_zeta::uv_to_xy_zeta_mpfr;
        use crate::synthesis::distance::diamond_distance_u2q_mpfr_target;

        let prefixes = build_l_q(m_split);
        let d_target = det_phase_of_mat2_mpfr(target);
        let epsilon = self.epsilon;
        let use_f64_gs = self.use_f64_gs;
        let bkz_block_size = self.bkz_block_size;
        let prec = target[0][0].0.prec();

        let any_budget_hit = Arc::new(AtomicBool::new(false));

        let dc_dr_filter = &self.dc_dr_filter;
        let mut usable: Vec<&U2Q> = prefixes
            .iter()
            .filter(|u_l| u_l.k < k_total)
            .filter(|u_l| {
                if dc_dr_filter.is_empty() { return true; }
                let d_l = det_phase_of(&u_l.to_float());
                let d_r = ((d_target as i32 - d_l as i32).rem_euclid(16)) as u32;
                dc_dr_filter.contains(&d_r)
            })
            .collect();

        if usable.is_empty() { return (None, false); }
        usable.sort_by(|a, b| b.k.cmp(&a.k));

        let n_threads = rayon::current_num_threads().max(1);
        let chunk = (usable.len() / n_threads).max(1);

        let result = usable
            .par_iter()
            .with_min_len(chunk)
            .map_init(
                || {
                    let mut s = IntScratch16::new(epsilon);
                    s.use_f64_gs = use_f64_gs;
                    s.bkz_block_size = bkz_block_size;
                    s
                },
                |scratch, u_l| -> Option<SynthResultQ> {
                    let k_prefix = u_l.k;
                    let k_inner = k_total - k_prefix;
                    let m_inner = u2q_dag_times_mat2_mpfr(u_l, target, prec);
                    let v_inner = unitary_to_uv_zeta_mpfr(&m_inner);
                    let d_l = det_phase_of(&u_l.to_float());
                    let d_r = ((d_target as i32 - d_l as i32).rem_euclid(16)) as u32;

                    let y_inner = uv_to_xy_zeta_mpfr(&v_inner, k_inner, prec);
                    let budget_hit = AtomicBool::new(false);
                    let u_l_local = **u_l;
                    let target_local = target.clone();
                    let should_stop = |x: &[i64; 16]| -> bool {
                        let u_r = solution_to_u2q_d(x, k_inner, d_r);
                        let u_full = u_l_local * u_r;
                        diamond_distance_u2q_mpfr_target(&u_full, &target_local) < epsilon
                    };

                    let sols = phase1_with_stop_mpfr(
                        scratch, &y_inner, &v_inner, k_inner, epsilon,
                        per_prefix_cap, &budget_hit, should_stop,
                    );

                    if budget_hit.load(std::sync::atomic::Ordering::Relaxed) {
                        any_budget_hit.store(true, std::sync::atomic::Ordering::Relaxed);
                    }

                    for sol in &sols {
                        let u_r = solution_to_u2q_d(sol, k_inner, d_r);
                        let u_full = u_l_local * u_r;
                        let dist = diamond_distance_u2q_mpfr_target(&u_full, target);
                        if dist < epsilon {
                            let gates = BlochDecomposer.decompose(&u_full);
                            return Some(SynthResultQ {
                                gates: Some(gates),
                                lde: k_total,
                                distance: dist,
                            });
                        }
                    }
                    None
                },
            )
            .find_map_any(|x| x);

        let budget_hit = any_budget_hit.load(std::sync::atomic::Ordering::Relaxed);
        (result, budget_hit)
    }

    /// Find a minimum-lde Clifford+√T circuit approximating `target`.
    ///
    /// Returns `None` if no circuit within `max_lde` achieves diamond
    /// distance < `epsilon`. Returns the FIRST candidate found at the
    /// smallest k that works (not necessarily √T-count optimal).
    ///
    /// **Backend**: hybrid — brute-force `phase1_brute` for `k ≤ BRUTE_LIMIT`,
    /// 16D L²-LLL + Schnorr-Euchner `phase1` for larger k.
    pub fn synthesize(&self, target: Mat2) -> Option<SynthResultQ> {
        use crate::synthesis::diag;
        let trace = diag::trace_enabled();
        if trace {
            diag::reset_all();
        }

        let d = det_phase_of(&target);
        let v = unitary_to_uv_zeta(&target);

        // Lattice scratch is allocated lazily on first lattice call.
        let mut scratch: Option<Box<IntScratch16>> = None;

        // k schedule:
        //   - Brute regime [min_lde .. BRUTE_LIMIT]: cheap exact-find for
        //     small Clifford+√T targets (Q, T, H, …).
        //   - Lattice regime: skip k that the empirical ε→lde fit
        //     `lde ≈ ⌈-log₂(ε)⌉ - 3` says are too small. Start 2 below the
        //     estimate to absorb per-target jitter, then advance to
        //     `max_lde` with two-pass budgeting.
        let lattice_start = lattice_lde_estimate(self.epsilon)
            .saturating_sub(2)
            .max(BRUTE_LIMIT + 1)
            .max(self.min_lde);

        // Lattice search with early-exit. The `should_stop` predicate
        // runs only on leaves that pass the integer-exact filter (norm
        // shell, bilinear, alignment) — typically a handful per call —
        // and short-circuits the walker once we find a candidate whose
        // diamond distance to `target` is already below ε. At deep ε this
        // can cut the walk by orders of magnitude.
        let epsilon = self.epsilon;
        let use_f64_gs = self.use_f64_gs;
        let bkz_block_size = self.bkz_block_size;
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
                let cand = solution_to_u2q_d(x, k, d);
                diamond_distance_u2q_float(&cand, &target) < epsilon
            };
            let sols = phase1_with_stop(
                s.as_mut(), &y, k, epsilon, budget, &budget_hit, should_stop,
            );
            (sols, budget_hit.load(std::sync::atomic::Ordering::Relaxed))
        };

        let check_sols = |sols: &[[i64; 16]], k: u32| -> Option<SynthResultQ> {
            for sol in sols {
                let cand: U2Q = solution_to_u2q_d(sol, k, d);
                let dist = diamond_distance_u2q_float(&cand, &target);
                if dist < self.epsilon {
                    let gates = BlochDecomposer.decompose(&cand);
                    return Some(SynthResultQ {
                        gates: Some(gates),
                        lde: k,
                        distance: dist,
                    });
                }
            }
            None
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
            let mut pass2_queue: Vec<u32> = Vec::new();
            for k in lattice_start..=self.max_lde {
                let t_k = std::time::Instant::now();
                if k <= m_split {
                    // k_inner would be ≤ 0 for the smallest-k_prefix prefixes,
                    // so D&C can't help here — still need to handle this k via
                    // the single-search path to stay complete. Fall through to
                    // the existing pass1 logic for these k values.
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
                    continue;
                }
                if trace {
                    eprintln!("[zeta] dc lde={k:>2} m={m_split} pass1 dispatching ...");
                }
                let (result, budget_hit) = self.dc_search_q(&target, k, m_split, dc_pass1_cap_for(self.epsilon));
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
                if budget_hit {
                    pass2_queue.push(k);
                }
            }

            // Pass 2 retries: only the lde levels where pass 1's prefixes
            // hit budget without finding. Other lde levels were
            // exhausted at pass 1 (no solution exists at that lde).
            for k in pass2_queue {
                let t_k = std::time::Instant::now();
                if trace {
                    eprintln!("[zeta] dc lde={k:>2} m={m_split} pass2 dispatching ...");
                }
                let (result, _) = self.dc_search_q(&target, k, m_split, dc_pass2_cap_for(self.epsilon));
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

    /// Synthesize with a Clifford+T fallback if the Q-only path returns
    /// `None`. ZOmega ⊂ ZZeta via ω = ζ², so any U2T is a valid U2Q with
    /// the same matrix value (just non-√T-optimal). Used at deep ε
    /// where the Q-lattice search may exhaust `max_lde` without finding.
    ///
    /// Returns the Q-optimal result if Q finds; otherwise returns the
    /// embedded CT result (which has a higher lde but a valid circuit).
    /// Returns `None` only if both backends fail.
    pub fn synthesize_with_ct_fallback(&self, target: Mat2) -> Option<SynthResultQ> {
        // Skip the Q-only path at ε ≤ 1e-8 — empirically the lde-threshold
        // for typical SU(2) targets is past max_lde (per the probes in
        // bin/probe_eps_1e8_dc_mpfr.rs) and Q wastes ~hours scanning.
        // CT works at this depth (~7 s/target for lde=80).
        if self.epsilon > 1e-8 {
            if let Some(r) = self.synthesize(target) {
                return Some(r);
            }
        }
        // Fall back to CT.
        use crate::synthesis::clifford_t::SynthesizerT;
        let synth_t = SynthesizerT::new(self.epsilon);
        let r_t = synth_t.synthesize(target)?;
        // Embed U2T → U2Q. The gate decomposition string from CT uses
        // T/H/S/X/Y/Z which all have valid U2Q forms (T = Q² in Z[ζ_16]).
        // Easiest path: rebuild the gate string into U2Q.
        let gates = r_t.gates?;
        let mut u2q = U2Q::eye();
        for ch in gates.chars() {
            u2q = match ch {
                'T' => u2q * U2Q::t(),
                'H' => u2q * U2Q::h(),
                'S' => u2q * U2Q::s(),
                'X' => u2q * U2Q::x(),
                'Y' => u2q * U2Q::y(),
                'Z' => u2q * U2Q::z(),
                'Q' => u2q * U2Q::q(),
                _ => return None,
            };
        }
        let dist = diamond_distance_u2q_float(&u2q, &target);
        if dist >= self.epsilon {
            return None;
        }
        let q_gates = BlochDecomposer.decompose(&u2q);
        Some(SynthResultQ {
            gates: Some(q_gates),
            lde: u2q.k,
            distance: dist,
        })
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
    fn dc_search_q(
        &self,
        target: &Mat2,
        k_total: u32,
        m_split: u32,
        per_prefix_cap: u64,
    ) -> (Option<SynthResultQ>, bool) {
        use rayon::prelude::*;

        let prefixes = build_l_q(m_split);
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
        // required d_R isn't in the allowed-offsets set. We collect into
        // a Vec of references so the per-thread closure does cheap reads
        // only.
        let dc_dr_filter = &self.dc_dr_filter;
        let mut usable: Vec<&U2Q> = prefixes
            .iter()
            .filter(|u_l| u_l.k < k_total)
            .filter(|u_l| {
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

        // **Prefix prioritisation by k_prefix (descending)**: prefixes
        // with high k_prefix have small `k_inner = k_total − k_prefix` →
        // tiny SE region per prefix → fast bail or fast hit. Sorting
        // before dispatch matters when |usable| > num_cores (the m=2
        // case has ~108 usable, more than 8 cores), so the work-stealing
        // queue picks the cheap high-k ones first. At m=1 with 36
        // usable on 8 cores there's also some benefit since rayon's
        // chunking respects the iter order.
        usable.sort_by(|a, b| b.k.cmp(&a.k));

        // Distribute prefixes across rayon workers. `with_min_len(chunk)`
        // ensures each worker gets at least `chunk` prefixes so the
        // per-prefix scratch allocation amortises over the chunk.
        let n_threads = rayon::current_num_threads().max(1);
        let chunk = (usable.len() / n_threads).max(1);

        let result = usable
            .par_iter()
            .with_min_len(chunk)
            .map_init(
                || {
                    let mut s = IntScratch16::new(epsilon);
                    s.use_f64_gs = use_f64_gs;
                    s.bkz_block_size = bkz_block_size;
                    s
                },
                |scratch, u_l| -> Option<SynthResultQ> {
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
                    let u_l_local = **u_l;
                    let target_local = *target;
                    let should_stop = |x: &[i64; 16]| -> bool {
                        let u_r = solution_to_u2q_d(x, k_inner, d_r);
                        let u_full = u_l_local * u_r;
                        diamond_distance_u2q_float(&u_full, &target_local) < epsilon
                    };

                    let sols = phase1_with_stop(
                        scratch, &y, k_inner, epsilon,
                        per_prefix_cap, &budget_hit, should_stop,
                    );

                    // Propagate this prefix's budget-hit signal to the
                    // dispatcher level. We OR in our local flag; the
                    // dispatcher reads `any_budget_hit` after the
                    // par_iter completes (or aborts via find_map_any).
                    if budget_hit.load(std::sync::atomic::Ordering::Relaxed) {
                        any_budget_hit.store(true, std::sync::atomic::Ordering::Relaxed);
                    }

                    // Score each returned solution and return the first
                    // that satisfies the ε bound.
                    for sol in &sols {
                        let u_r = solution_to_u2q_d(sol, k_inner, d_r);
                        let u_full = u_l_local * u_r;
                        let dist = diamond_distance_u2q_float(&u_full, target);
                        if dist < epsilon {
                            let gates = BlochDecomposer.decompose(&u_full);
                            return Some(SynthResultQ {
                                gates: Some(gates),
                                lde: k_total,
                                distance: dist,
                            });
                        }
                    }
                    None
                },
            )
            .find_map_any(|x| x);

        let budget_hit = any_budget_hit.load(std::sync::atomic::Ordering::Relaxed);
        (result, budget_hit)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
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

    /// Z1 filter — does the float-domain FGKM faithfully predict the
    /// integer-domain canonical-form syllables? Build many exact
    /// Clifford+√T targets, compare predictions step-by-step.
    #[test]
    fn z1_float_fgkm_agreement() {
        use crate::synthesis::decomposer::{canonical_form_axes_q, canonical_form_axes_q_float};
        use crate::matrix::so3::SO3;
        use crate::rings::ZZeta;
        use rand::{Rng, SeedableRng};
        use rand::rngs::StdRng;

        let bases: [U2Q; 3] = [
            U2Q::h() * U2Q::q() * U2Q::h(),
            U2Q::s() * U2Q::h() * U2Q::q() * U2Q::h() * U2Q::s().dagger(),
            U2Q::q(),
        ];
        let mut syllables: [[U2Q; 3]; 3] = [[U2Q::eye(); 3]; 3];
        for (axis, base) in bases.iter().enumerate() {
            let mut acc = U2Q::eye();
            for a in 0..3 {
                acc = acc * *base;
                syllables[axis][a] = acc;
            }
        }
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

        let mut rng = StdRng::seed_from_u64(0xC1FF1);
        let n_trials = 2000;

        for &target_len in &[3usize, 5, 8, 12] {
            let mut step_match = vec![0u64; target_len];
            let mut step_total = vec![0u64; target_len];
            let mut full_match: u64 = 0;
            let mut full_first_match: u64 = 0;

            for _ in 0..n_trials {
                let mut prev_axis: usize = 3;
                let mut body = U2Q::eye();
                for _ in 0..target_len {
                    let mut axis = rng.random_range(0..3);
                    if axis == prev_axis { axis = (axis + 1) % 3; }
                    if axis == prev_axis { axis = (axis + 1) % 3; }
                    let a = rng.random_range(0..3);
                    body = body * syllables[axis][a];
                    prev_axis = axis;
                }
                let c_idx = rng.random_range(0..cliffords_q.len());
                let target = body * cliffords_q[c_idx];

                let canon_int = canonical_form_axes_q(&target);
                // Build float SO3 from the integer SO3 of target.
                let so3_int = SO3::<crate::matrix::so3::R4>::from_u2(&target);
                let mut so3_f = so3_int.to_float();
                let canon_float = canonical_form_axes_q_float(&mut so3_f, target_len + 4);

                let n = canon_int.len().min(canon_float.len()).min(target_len);
                let mut all_match = canon_int.len() == canon_float.len();
                for i in 0..n {
                    step_total[i] += 1;
                    if canon_int[i] == canon_float[i] {
                        step_match[i] += 1;
                    } else {
                        all_match = false;
                    }
                }
                if !canon_int.is_empty() && !canon_float.is_empty()
                   && canon_int[0] == canon_float[0] {
                    full_first_match += 1;
                }
                if all_match { full_match += 1; }
            }

            eprintln!(
                "\n=== target_len={target_len}, n_trials={n_trials} ==="
            );
            eprintln!(
                "  full sequence agreement     : {}/{} = {:.1}%",
                full_match, n_trials,
                100.0 * (full_match as f64) / (n_trials as f64)
            );
            eprintln!(
                "  first-syllable agreement     : {}/{} = {:.1}%",
                full_first_match, n_trials,
                100.0 * (full_first_match as f64) / (n_trials as f64)
            );
            eprintln!("  per-step agreement:");
            for i in 0..target_len {
                if step_total[i] == 0 { continue; }
                eprintln!(
                    "    step {i:>2}: {:>5}/{:>5} = {:.1}%",
                    step_match[i], step_total[i],
                    100.0 * (step_match[i] as f64) / (step_total[i] as f64)
                );
            }
        }
    }

    /// Z1 filter investigation: build random Clifford+√T targets by composing
    /// known FGKM bodies, then look at where the canonical-form *first*
    /// syllable lands. If the distribution is uniform across all 9 (axis, a)
    /// cells, no filter via this angle. If biased, follow up with cheap-
    /// predictor experiments.
    #[test]
    fn z1_first_syllable_distribution() {
        use crate::synthesis::decomposer::canonical_form_axes_q;
        use rand::{Rng, SeedableRng};
        use rand::rngs::StdRng;

        // Build the 9 base syllables (same as build_l_q_inner).
        let bases: [U2Q; 3] = [
            U2Q::h() * U2Q::q() * U2Q::h(),
            U2Q::s() * U2Q::h() * U2Q::q() * U2Q::h() * U2Q::s().dagger(),
            U2Q::q(),
        ];
        let mut syllables: [[U2Q; 3]; 3] = [[U2Q::eye(); 3]; 3];
        for (axis, base) in bases.iter().enumerate() {
            let mut acc = U2Q::eye();
            for a in 0..3 {
                acc = acc * *base;
                syllables[axis][a] = acc;
            }
        }
        // Cliffords as U2Q.
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

        let mut rng = StdRng::seed_from_u64(0xC1FF0);
        let n_trials = 5000;

        for &target_len in &[3usize, 5, 8, 12] {
            // Bin: hist[axis][a-1] for first syllable.
            let mut first_hist = [[0u64; 3]; 3];
            // Compare to bin of *body construction's* first syllable.
            let mut construct_hist = [[0u64; 3]; 3];
            // Track length-mismatches (canonical-form length vs constructed length).
            let mut len_short = 0u64;
            let mut len_match = 0u64;
            let mut len_long = 0u64;
            // Sample variability in returned length.
            let mut sum_len = 0u64;

            for _ in 0..n_trials {
                // Build a random length-target_len FGKM body with adjacency.
                let mut prev_axis: usize = 3;
                let mut body = U2Q::eye();
                let mut constructed: Vec<(u8, u8)> = Vec::new();
                for _ in 0..target_len {
                    // pick axis ≠ prev_axis
                    let mut axis = rng.random_range(0..3);
                    if axis == prev_axis {
                        axis = (axis + 1) % 3;
                    }
                    if axis == prev_axis {
                        axis = (axis + 1) % 3;
                    }
                    let a = rng.random_range(0..3); // 0..=2 → a∈{1,2,3}
                    body = body * syllables[axis][a];
                    constructed.push((axis as u8, (a + 1) as u8));
                    prev_axis = axis;
                }
                let c_idx = rng.random_range(0..cliffords_q.len());
                let target = body * cliffords_q[c_idx];

                // Canonical-form decomposition.
                let canon = canonical_form_axes_q(&target);
                if !canon.is_empty() {
                    let (p, a) = canon[0];
                    first_hist[p as usize][(a - 1) as usize] += 1;
                }
                let (cp, ca) = constructed[0];
                construct_hist[cp as usize][(ca - 1) as usize] += 1;

                sum_len += canon.len() as u64;
                if canon.len() < target_len { len_short += 1; }
                else if canon.len() == target_len { len_match += 1; }
                else { len_long += 1; }
            }

            eprintln!(
                "\n=== target_len={target_len}, n_trials={n_trials} ==="
            );
            eprintln!(
                "Canonical-form length:  exact={len_match}  shorter={len_short}  longer={len_long}  avg={:.2}",
                (sum_len as f64) / (n_trials as f64)
            );
            eprintln!("Constructed-first vs canonical-form-first  (counts, % of trials)");
            eprintln!("        constructed                canonical");
            eprintln!("        a=1   a=2   a=3            a=1   a=2   a=3");
            let labels = ["x", "y", "z"];
            for axis in 0..3 {
                eprint!("  {}    ", labels[axis]);
                for a in 0..3 {
                    eprint!("{:>5} ", construct_hist[axis][a]);
                }
                eprint!("           ");
                for a in 0..3 {
                    eprint!("{:>5} ", first_hist[axis][a]);
                }
                eprintln!();
            }
            // Quick numeric summary of canonical-form-first balance.
            let total: u64 = first_hist.iter().flatten().sum();
            let z_share: u64 = first_hist[2].iter().sum();
            let nonz_share = total - z_share;
            eprintln!(
                "  first-syllable z-axis share: {} / {} = {:.1}%   (uniform = 33.3%)",
                z_share, total, 100.0 * (z_share as f64) / (total as f64)
            );
            eprintln!(
                "  first-syllable non-z share : {} / {} = {:.1}%   (uniform = 66.7%)",
                nonz_share, total, 100.0 * (nonz_share as f64) / (total as f64)
            );
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

    /// Z1 viability at ε=1e-7: single search becomes unreliable here (per
    /// perf-state memo, 1/3 fail at max_lde=30). Test whether filtered D&C
    /// reliably finds solutions and how long it takes.
    #[test]
    #[ignore]  // long: per-target time can run into minutes
    fn z1_dc_eps_1e_7() {
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

        let mut rng = StdRng::seed_from_u64(0x1E7);
        let n = 3;
        let eps = 1e-7_f64;

        eprintln!("\n=== ε={eps:.0e}, {n} targets, max_lde=35 ===");
        for i in 0..n {
            let alpha = 2.0 * std::f64::consts::PI * rng.random::<f64>();
            let beta = 2.0 * std::f64::consts::PI * rng.random::<f64>();
            let gamma = 2.0 * std::f64::consts::PI * rng.random::<f64>();
            let target = matmul(matmul(rz(alpha), ry(beta)), rz(gamma));

            // Single search.
            let synth_s = SynthesizerQ::new(eps).with_max_lde(35);
            let t0 = std::time::Instant::now();
            let r_s = synth_s.synthesize(target);
            let ts = t0.elapsed().as_secs_f64();
            let s_label = match &r_s {
                Some(r) => format!("lde={} dist={:.2e}", r.lde, r.distance),
                None => "FAILED".to_string(),
            };
            eprintln!("  trial {i}  single   {s_label:<32} t={ts:>7.1}s");

            // m=1 relaxed.
            let synth_m1 = SynthesizerQ::new(eps).with_max_lde(35)
                .with_dc_split(1).with_dc_dr_filter(vec![0, 1, 15]);
            let t0 = std::time::Instant::now();
            let r_m1 = synth_m1.synthesize(target);
            let tm1 = t0.elapsed().as_secs_f64();
            let m1_label = match &r_m1 {
                Some(r) => format!("lde={} dist={:.2e}", r.lde, r.distance),
                None => "FAILED".to_string(),
            };
            eprintln!("  trial {i}  m1_relax {m1_label:<32} t={tm1:>7.1}s  ({:.2}× vs single)",
                if tm1 > 0.0 { ts/tm1 } else { 0.0 });

            // m=2 strict.
            let synth_m2 = SynthesizerQ::new(eps).with_max_lde(35)
                .with_dc_split(2).with_dc_dr_filter(vec![0]);
            let t0 = std::time::Instant::now();
            let r_m2 = synth_m2.synthesize(target);
            let tm2 = t0.elapsed().as_secs_f64();
            let m2_label = match &r_m2 {
                Some(r) => format!("lde={} dist={:.2e}", r.lde, r.distance),
                None => "FAILED".to_string(),
            };
            eprintln!("  trial {i}  m2_strict{m2_label:<32} t={tm2:>7.1}s  ({:.2}× vs single)",
                if tm2 > 0.0 { ts/tm2 } else { 0.0 });
            eprintln!();
        }
    }

    /// Compare D&C split-parameter configurations at deep ε.
    /// m=1 |d_R|≤1 (default, 36 prefixes) vs m=2 d_R=0 (144 prefixes
    /// but each at higher k_prefix → smaller k_inner → faster SE).
    ///
    /// Decides whether to switch the auto-D&C default at very deep ε.
    #[test]
    #[ignore]
    fn z1_m1_vs_m2_deep_eps() {
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

        let mut rng = StdRng::seed_from_u64(0xC0FFEE);
        let n: usize = std::env::var("M_N")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(3);
        let eps: f64 = std::env::var("M_EPS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1e-7);
        eprintln!("\n=== ε={eps:.0e}, {n} random U3 targets ===");

        for i in 0..n {
            let alpha = 2.0 * std::f64::consts::PI * rng.random::<f64>();
            let beta = 2.0 * std::f64::consts::PI * rng.random::<f64>();
            let gamma = 2.0 * std::f64::consts::PI * rng.random::<f64>();
            let target = matmul(matmul(rz(alpha), ry(beta)), rz(gamma));

            // m=1 |d_R|≤1 (current default).
            let synth_m1 = SynthesizerQ::new(eps).with_max_lde(30);
            let t0 = std::time::Instant::now();
            let r_m1 = synth_m1.synthesize(target);
            let t_m1 = t0.elapsed();

            // m=2 strict d_R=0.
            let synth_m2 = SynthesizerQ::new(eps).with_max_lde(30)
                .with_dc_split(2).with_dc_dr_filter(vec![0]);
            let t0 = std::time::Instant::now();
            let r_m2 = synth_m2.synthesize(target);
            let t_m2 = t0.elapsed();

            let ms = |t: std::time::Duration| t.as_secs_f64() * 1000.0;
            eprintln!(
                "  trial {i}  m=1: lde={:?} t={:.0}ms  | m=2: lde={:?} t={:.0}ms  ratio={:.2}×",
                r_m1.as_ref().map(|r| r.lde),
                ms(t_m1),
                r_m2.as_ref().map(|r| r.lde),
                ms(t_m2),
                ms(t_m1) / ms(t_m2),
            );
        }
    }

    /// Quick diagnostic at ε=1e-8: f64 vs MPFR, single target, short
    /// timeout. Reports first lde where solution is found (if any), and
    /// whether the path completes.
    #[test]
    #[ignore]
    fn z1_eps_1e_8_diag() {
        let theta = 0.3_f64;
        let target: Mat2 = [
            [Complex64::from_polar(1.0, -theta / 2.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::from_polar(1.0, theta / 2.0)],
        ];
        let eps = 1e-8_f64;
        // MPFR first (the path that should actually work at this depth),
        // then f64 with the ladder (slower / may not finish).
        for (label, use_f64) in &[("MPFR", false), ("f64+ladder", true)] {
            // Cap max_lde at 35 to bound runtime.
            let synth = SynthesizerQ::new(eps).with_max_lde(35).with_f64_gs(*use_f64);
            let t0 = std::time::Instant::now();
            let r = synth.synthesize(target);
            let dt = t0.elapsed();
            eprintln!(
                "  {label:<10}: lde={:?}  dist={:?}  t={:.0}ms",
                r.as_ref().map(|r| r.lde),
                r.as_ref().map(|r| r.distance),
                dt.as_secs_f64() * 1000.0
            );
        }
    }

    /// f64 GS at deep ε via the SynthesizerQ builder. Apples-to-apples
    /// comparison: same code path, only `use_f64_gs` differs.
    #[test]
    #[ignore]
    fn z1_f64_gs_deep_eps() {
        let theta = 0.3_f64;
        let target: Mat2 = [
            [Complex64::from_polar(1.0, -theta / 2.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::from_polar(1.0, theta / 2.0)],
        ];

        for &eps in &[1e-5, 1e-6, 1e-7_f64] {
            eprintln!("\n=== ε={eps:.0e} ===");

            let synth_mpfr = SynthesizerQ::new(eps);
            let t0 = std::time::Instant::now();
            let r_mpfr = synth_mpfr.synthesize(target);
            let t_mpfr = t0.elapsed();
            eprintln!("  MPFR: lde={:?} dist={:?} t={:.0}ms",
                r_mpfr.as_ref().map(|r| r.lde),
                r_mpfr.as_ref().map(|r| r.distance),
                t_mpfr.as_secs_f64() * 1000.0);

            let synth_f64 = SynthesizerQ::new(eps).with_f64_gs(true);
            let t0 = std::time::Instant::now();
            let r_f64 = synth_f64.synthesize(target);
            let t_f64 = t0.elapsed();
            let speedup = t_mpfr.as_secs_f64() / t_f64.as_secs_f64();
            eprintln!("  f64 : lde={:?} dist={:?} t={:.0}ms  ({speedup:.2}× vs MPFR)",
                r_f64.as_ref().map(|r| r.lde),
                r_f64.as_ref().map(|r| r.distance),
                t_f64.as_secs_f64() * 1000.0);
        }
    }

    /// f64 GS state experiment: try the f64 LLL path on synth at various ε
    /// and compare to the MPFR path. Reports correctness (does it find the
    /// answer? at the same lde?) and speed.
    #[test]
    fn z1_f64_gs_experiment() {
        use crate::synthesis::diag;
        use crate::synthesis::lenstra_zeta::lll_f64::{run_lll_16_f64};
        use crate::synthesis::lenstra_zeta::lll::run_lll_16;

        let theta = 0.3_f64;
        let target: Mat2 = [
            [Complex64::from_polar(1.0, -theta / 2.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::from_polar(1.0, theta / 2.0)],
        ];
        let v = unitary_to_uv_zeta(&target);

        for &eps in &[1e-3, 1e-4, 1e-5_f64] {
            eprintln!("\n=== ε={eps:.0e} ===");
            let max_lde = if eps >= 1e-4 { 14u32 } else { 16 };

            // Phase A: pure LLL timing — set up scratch, run LLL only.
            // Same Q, side-by-side timing of MPFR vs f64.
            let mut s_mpfr = IntScratch16::new(eps);
            let mut s_f64 = IntScratch16::new(eps);

            // Pick a representative k for timing.
            let k = max_lde - 2;
            let y = uv_to_xy_zeta(v, k);

            // Build Q + Gram for both scratches identically (no LLL yet).
            // Since `phase1_with_stop` is the easiest entry, use it twice:
            // once with use_f64_gs=false, once true. Time only.
            let runs = 5;
            let mut t_mpfr_total_us = 0u128;
            let mut t_f64_total_us = 0u128;
            let mut sols_mpfr_count = 0;
            let mut sols_f64_count = 0;
            for _ in 0..runs {
                diag::reset_all();
                let budget_hit = AtomicBool::new(false);
                s_mpfr.use_f64_gs = false;
                let t = std::time::Instant::now();
                let sols = phase1_with_stop(
                    &mut s_mpfr, &y, k, eps, 100_000_000, &budget_hit, |_| false
                );
                t_mpfr_total_us += t.elapsed().as_nanos() as u128 / 1000;
                sols_mpfr_count += sols.len();

                diag::reset_all();
                let budget_hit = AtomicBool::new(false);
                s_f64.use_f64_gs = true;
                let t = std::time::Instant::now();
                let sols = phase1_with_stop(
                    &mut s_f64, &y, k, eps, 100_000_000, &budget_hit, |_| false
                );
                t_f64_total_us += t.elapsed().as_nanos() as u128 / 1000;
                sols_f64_count += sols.len();
            }
            eprintln!("  phase1 single call (k={k}, {runs} runs):");
            eprintln!("    MPFR: avg {:.0} μs, {} total sols",
                (t_mpfr_total_us as f64) / runs as f64, sols_mpfr_count);
            eprintln!("    f64 : avg {:.0} μs, {} total sols",
                (t_f64_total_us as f64) / runs as f64, sols_f64_count);
            let speedup = (t_mpfr_total_us as f64) / (t_f64_total_us as f64);
            eprintln!("    speedup: {speedup:.2}×");

            // Phase B: end-to-end synth, MPFR vs f64.
            let synth_mpfr = SynthesizerQ::new(eps).with_max_lde(max_lde);
            let t0 = std::time::Instant::now();
            let r_mpfr = synth_mpfr.synthesize(target);
            let t_mpfr = t0.elapsed();

            // We hijack a synth via direct phase1 calls — synth doesn't yet
            // expose use_f64_gs as a builder. Build a minimal harness:
            let mut s = IntScratch16::new(eps);
            s.use_f64_gs = true;
            let mut found = None;
            let t0 = std::time::Instant::now();
            for k_try in 4..=max_lde {
                let y = uv_to_xy_zeta(v, k_try);
                let budget_hit = AtomicBool::new(false);
                let target_local = target;
                let d = det_phase_of(&target);
                let should_stop = |x: &[i64; 16]| -> bool {
                    let cand = solution_to_u2q_d(x, k_try, d);
                    diamond_distance_float(&cand.to_float(), &target_local) < eps
                };
                let sols = phase1_with_stop(&mut s, &y, k_try, eps, 100_000_000, &budget_hit, should_stop);
                for sol in &sols {
                    let cand = solution_to_u2q_d(sol, k_try, d);
                    let dist = diamond_distance_float(&cand.to_float(), &target_local);
                    if dist < eps {
                        found = Some((k_try, dist));
                        break;
                    }
                }
                if found.is_some() { break; }
            }
            let t_f64 = t0.elapsed();

            eprintln!("  end-to-end synth:");
            eprintln!("    MPFR: lde={:?} dist={:?} t={:.1}ms",
                r_mpfr.as_ref().map(|r| r.lde),
                r_mpfr.as_ref().map(|r| r.distance),
                t_mpfr.as_secs_f64() * 1000.0);
            eprintln!("    f64 : found={:?}  t={:.1}ms",
                found, t_f64.as_secs_f64() * 1000.0);
        }
    }

    /// LLL precision sweep: time single search at different GS_PREC values
    /// to see if lowering precision speeds up LLL without losing
    /// correctness.
    #[test]
    fn z1_gs_prec_sweep() {
        // Need to import IntScratch16 builder and run a single phase1 a few
        // times manually, since SynthesizerQ doesn't expose gs_prec.
        let theta = 0.3_f64;
        let target: Mat2 = [
            [Complex64::from_polar(1.0, -theta / 2.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::from_polar(1.0, theta / 2.0)],
        ];
        let eps = 1e-4_f64;
        let v = unitary_to_uv_zeta(&target);

        for &gs_prec in &[64u32, 80, 96, 128] {
            let mut scratch = IntScratch16::with_gs_prec(eps, gs_prec);
            // Run phase1 at lde=10, 11, 12 (where ε=1e-4 typically lands).
            let mut total_ns = 0u128;
            let mut found = false;
            let mut found_dist = f64::INFINITY;
            for k in 9..=14u32 {
                let y = uv_to_xy_zeta(v, k);
                let budget_hit = AtomicBool::new(false);
                let target_local = target;
                let d = det_phase_of(&target);
                let should_stop = |x: &[i64; 16]| -> bool {
                    let cand = solution_to_u2q_d(x, k, d);
                    diamond_distance_float(&cand.to_float(), &target_local) < eps
                };
                let t0 = std::time::Instant::now();
                let sols = phase1_with_stop(
                    &mut scratch, &y, k, eps, 100_000_000, &budget_hit, should_stop
                );
                total_ns += t0.elapsed().as_nanos();
                for sol in &sols {
                    let cand = solution_to_u2q_d(sol, k, d);
                    let dist = diamond_distance_float(&cand.to_float(), &target_local);
                    if dist < eps && !found {
                        found = true;
                        found_dist = dist;
                        eprintln!(
                            "  gs_prec={gs_prec:>3}  found at lde={k}  dist={dist:.3e}  k={k} cum_t={:.0}ms",
                            (total_ns as f64) / 1e6
                        );
                        break;
                    }
                }
                if found { break; }
            }
            if !found {
                eprintln!("  gs_prec={gs_prec:>3}  NOT FOUND  cum_t={:.0}ms", (total_ns as f64) / 1e6);
            } else {
                let _ = found_dist;
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

    /// Sweep filter performance across ε to find the crossover where
    /// D&C with strict d_R filter beats single search.
    #[test]
    fn z1_dc_dr_filter_eps_sweep() {
        let theta = 0.3_f64;
        let target: Mat2 = [
            [Complex64::from_polar(1.0, -theta / 2.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::from_polar(1.0, theta / 2.0)],
        ];
        for &eps in &[1e-3, 1e-4, 1e-5_f64] {
            eprintln!("\n=== ε={eps:.0e} ===");
            let max_lde = ((-eps.log2() * 2.0) as u32).max(15);

            let synth_single = SynthesizerQ::new(eps).with_max_lde(max_lde);
            let t0 = std::time::Instant::now();
            let r_single = synth_single.synthesize(target);
            let t_single = t0.elapsed();
            let single_ms = t_single.as_secs_f64() * 1000.0;
            eprintln!("  single                  lde={:?}  t={single_ms:.0}ms",
                r_single.as_ref().map(|r| r.lde));

            let configs: &[(u32, &[u32], &str)] = &[
                (1, &[0, 1, 15], "m=1 |d_R|≤1"),
                (2, &[0], "m=2 strict"),
                (2, &[0, 1, 15], "m=2 |d_R|≤1"),
            ];
            for (m, filter, label) in configs {
                let synth = SynthesizerQ::new(eps)
                    .with_max_lde(max_lde)
                    .with_dc_split(*m)
                    .with_dc_dr_filter(filter.to_vec());
                let t0 = std::time::Instant::now();
                let r = synth.synthesize(target);
                let dt = t0.elapsed();
                let ms = dt.as_secs_f64() * 1000.0;
                let speedup = single_ms / ms;
                eprintln!("  {label:<22}  lde={:?}  t={ms:>7.0}ms  ({speedup:>5.2}× vs single)",
                    r.as_ref().map(|r| r.lde));
            }
        }
    }

    /// Z1 det-phase filter at deep ε. Single search is slow there so D&C
    /// has more room to win.
    #[test]
    #[ignore]
    fn z1_dc_dr_filter_deep_eps() {
        let theta = 0.3_f64;
        let target: Mat2 = [
            [Complex64::from_polar(1.0, -theta / 2.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::from_polar(1.0, theta / 2.0)],
        ];
        let eps = 1e-6_f64;

        let synth_single = SynthesizerQ::new(eps).with_max_lde(35);
        let t0 = std::time::Instant::now();
        let r_single = synth_single.synthesize(target);
        let t_single = t0.elapsed();
        eprintln!(
            "single  lde={:?}  t={:.0}ms",
            r_single.as_ref().map(|r| r.lde),
            t_single.as_secs_f64() * 1000.0
        );

        let configs: &[(u32, &[u32], &str)] = &[
            (1, &[0, 1, 15], "m=1 |d_R|≤1"),
            (1, &[0, 1, 2, 14, 15], "m=1 |d_R|≤2"),
            (2, &[0], "m=2 strict d_R=0"),
            (2, &[0, 1, 15], "m=2 |d_R|≤1"),
            (3, &[0], "m=3 strict d_R=0"),
        ];
        for (m, filter, label) in configs {
            let synth = SynthesizerQ::new(eps)
                .with_max_lde(35)
                .with_dc_split(*m)
                .with_dc_dr_filter(filter.to_vec());
            let t0 = std::time::Instant::now();
            let r = synth.synthesize(target);
            let dt = t0.elapsed();
            eprintln!(
                "  {label:<22} lde={:?}  t={:.0}ms",
                r.as_ref().map(|r| r.lde),
                dt.as_secs_f64() * 1000.0
            );
        }
    }

    /// Z1 det-phase filter test: with various allowed-d_R sets, see how
    /// many prefixes pass the filter and how the dispatcher does.
    #[test]
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

    /// Z1 pre-filter probe: for each prefix, compute the rounded integer
    /// encoding of v_inner and evaluate the bilinear forms (the
    /// totally-real-subring decomposition of unitarity). Classify prefixes
    /// by max|B_i| and look at how the distribution separates.
    ///
    /// If "the right" prefix at this k_total has small |B_i| while most
    /// wrong prefixes have large |B_i|, we have a cheap algebraic filter.
    #[test]
    fn z1_prefilter_bilinear_distribution() {
        use crate::synthesis::lenstra_zeta::se::bilinear_forms;
        let theta = 0.3_f64;
        let target: Mat2 = [
            [Complex64::from_polar(1.0, -theta / 2.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::from_polar(1.0, theta / 2.0)],
        ];
        let m = 3u32;
        let k_total = 10u32;

        let prefixes = build_l_q(m);

        // Also need to know the "right" prefix's bilinear so we can see
        // separation. Use a known lattice solution from a single-search
        // run for comparison — but for this test we just look at the
        // distribution and the minimum.

        // Bin prefixes by max|B_i| of rounded scaled y.
        // Buckets: |B|=0..=3, 4..=15, 16..=63, 64..=255, ..., else.
        let bin_of = |max_b: i128| -> usize {
            if max_b == 0 { 0 }
            else if max_b <= 3 { 1 }
            else if max_b <= 15 { 2 }
            else if max_b <= 63 { 3 }
            else if max_b <= 255 { 4 }
            else if max_b <= 1023 { 5 }
            else if max_b <= 4095 { 6 }
            else { 7 }
        };
        let bin_label = ["=0", "1-3", "4-15", "16-63", "64-255",
                         "256-1023", "1024-4095", ">4095"];
        let mut bins = vec![0u64; bin_label.len()];
        let mut min_b: i128 = i128::MAX;
        let mut min_b_kprefix: u32 = 0;

        for u_l in prefixes.iter() {
            if u_l.k >= k_total { continue; }
            let k_inner = k_total - u_l.k;
            let m_inner = u2q_dag_times_mat2(u_l, &target);
            let v_inner = unitary_to_uv_zeta(&m_inner);
            let y = uv_to_xy_zeta(v_inner, k_inner);
            let y_int: [i64; 16] = std::array::from_fn(|i| y[i].round() as i64);
            let (b1, b2, b3) = bilinear_forms(&y_int);
            let max_b = b1.abs().max(b2.abs()).max(b3.abs());
            bins[bin_of(max_b)] += 1;
            if max_b < min_b {
                min_b = max_b;
                min_b_kprefix = u_l.k;
            }
        }

        eprintln!("\n=== bilinear_forms(round(y)) over L_{m}^Q  k_total={k_total} ===");
        for (i, &c) in bins.iter().enumerate() {
            eprintln!("  max|B_i| {:>10}: {:>6}  ({:>5.1}%)",
                bin_label[i], c, 100.0 * (c as f64) / (prefixes.len() as f64));
        }
        eprintln!("  min max|B_i| = {min_b}  (at a prefix with k_prefix={min_b_kprefix})");
    }

    /// Z1 phase-by-phase breakdown using the diag counters. Run with
    /// `CYCLOSYNTH_TRACE=1` to populate T_BUILD_NS, T_LLL_NS, etc.
    /// across all prefixes at a fixed (m, k_total).
    #[test]
    fn z1_dc_phase_breakdown() {
        use crate::synthesis::diag;
        let theta = 0.3_f64;
        let target: Mat2 = [
            [Complex64::from_polar(1.0, -theta / 2.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::from_polar(1.0, theta / 2.0)],
        ];
        let eps = 1e-3_f64;
        let m = 3u32;
        let k_total = 10u32;

        if !diag::trace_enabled() {
            eprintln!("(set CYCLOSYNTH_TRACE=1 to see phase breakdown)");
        }
        diag::reset_all();

        let prefixes = build_l_q(m);
        let d_target = det_phase_of(&target);
        let mut scratch = IntScratch16::new(eps);
        scratch.use_f64_gs = true;
        let first_call = true;
        let _ = first_call;

        let t_total = std::time::Instant::now();
        let mut n_processed = 0u64;
        for u_l in prefixes.iter() {
            if u_l.k >= k_total { continue; }
            let k_inner = k_total - u_l.k;
            let m_inner = u2q_dag_times_mat2(u_l, &target);
            let v_inner = unitary_to_uv_zeta(&m_inner);
            let d_l = det_phase_of(&u_l.to_float());
            let _d_r = ((d_target as i32 - d_l as i32).rem_euclid(16)) as u32;
            let y = uv_to_xy_zeta(v_inner, k_inner);
            let budget_hit = AtomicBool::new(false);
            let _ = first_call;
            let _sols = phase1_with_stop(
                &mut scratch, &y, k_inner, eps,
                10_000, &budget_hit, |_| false,
            );
            n_processed += 1;
        }
        let total_ms = t_total.elapsed().as_secs_f64() * 1000.0;

        let s = diag::snapshot();
        eprintln!("\n=== dc m={m}  k_total={k_total}  prefixes processed {n_processed}  wall {total_ms:.0}ms ===");
        eprintln!("  build  {:>8.1} ms", s.t_build_ms);
        eprintln!("  lll    {:>8.1} ms  (iters total {}, avg {:.1}, max {})",
            s.t_lll_ms, s.lll_iters_total,
            (s.lll_iters_total as f64) / (n_processed.max(1) as f64),
            s.lll_iters_max);
        eprintln!("  chol   {:>8.1} ms", s.t_cholesky_ms);
        eprintln!("  lu     {:>8.1} ms", s.t_lu_ms);
        eprintln!("  se     {:>8.1} ms", s.t_se_ms);
        let phase_sum = s.t_build_ms + s.t_lll_ms + s.t_cholesky_ms + s.t_lu_ms + s.t_se_ms;
        eprintln!("  sum    {:>8.1} ms  ({:>5.1}% of wall)", phase_sum, 100.0 * phase_sum / total_ms);
        if n_processed > 0 {
            eprintln!("  per-prefix breakdown (μs each):");
            eprintln!("    build  {:>8.1}", s.t_build_ms * 1000.0 / n_processed as f64);
            eprintln!("    lll    {:>8.1}", s.t_lll_ms * 1000.0 / n_processed as f64);
            eprintln!("    chol   {:>8.1}", s.t_cholesky_ms * 1000.0 / n_processed as f64);
            eprintln!("    lu     {:>8.1}", s.t_lu_ms * 1000.0 / n_processed as f64);
            eprintln!("    se     {:>8.1}", s.t_se_ms * 1000.0 / n_processed as f64);
        }
    }

    /// Z1 instrumentation: time each prefix separately and bin by k_inner
    /// to see *where* the per-prefix cost is going. Repeats the inner search
    /// inline rather than calling dc_search_q so we can attach per-prefix
    /// timing without reaching across modules.
    #[test]
    fn z1_dc_per_prefix_breakdown() {
        let theta = 0.3_f64;
        let target: Mat2 = [
            [Complex64::from_polar(1.0, -theta / 2.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::from_polar(1.0, theta / 2.0)],
        ];
        let eps = 1e-3_f64;
        let m = 3u32;
        let k_total = 10u32;

        let prefixes = build_l_q(m);
        let d_target = det_phase_of(&target);
        let mut scratch = IntScratch16::new(eps);

        // Bin per-prefix time by k_inner.
        let mut bin_count: Vec<u64> = vec![0; (k_total + 1) as usize];
        let mut bin_time_ns: Vec<u64> = vec![0; (k_total + 1) as usize];
        let mut bin_sols: Vec<u64> = vec![0; (k_total + 1) as usize];
        let mut total_skipped = 0u64;

        let t_total = std::time::Instant::now();
        for u_l in prefixes.iter() {
            let k_prefix = u_l.k;
            if k_prefix >= k_total {
                total_skipped += 1;
                continue;
            }
            let k_inner = k_total - k_prefix;
            let m_inner = u2q_dag_times_mat2(u_l, &target);
            let v_inner = unitary_to_uv_zeta(&m_inner);
            let d_l = det_phase_of(&u_l.to_float());
            let d_r = ((d_target as i32 - d_l as i32).rem_euclid(16)) as u32;

            let y = uv_to_xy_zeta(v_inner, k_inner);
            let budget_hit = AtomicBool::new(false);

            let t_prefix = std::time::Instant::now();
            let _sols = phase1_with_stop(
                &mut scratch, &y, k_inner, eps,
                10_000, &budget_hit, |_| false,
            );
            let dt = t_prefix.elapsed().as_nanos() as u64;
            // Just counting non-trivial returns; not validating distance here.
            bin_count[k_inner as usize] += 1;
            bin_time_ns[k_inner as usize] += dt;
            bin_sols[k_inner as usize] += _sols.len() as u64;
            let _ = (m_inner, d_r);
        }
        let total_time = t_total.elapsed();

        eprintln!("\n=== dc m={m}  k_total={k_total}  total {:.0} ms  skipped {total_skipped}/{} ===",
            total_time.as_secs_f64() * 1000.0,
            prefixes.len()
        );
        eprintln!("  k_inner   count    total_ms    per_prefix_us    sols");
        for k in 0..=k_total {
            let n = bin_count[k as usize];
            if n == 0 { continue; }
            let total_ms = (bin_time_ns[k as usize] as f64) / 1e6;
            let per_us = (bin_time_ns[k as usize] as f64) / (n as f64) / 1e3;
            eprintln!("    {k:>3}  {:>7}  {total_ms:>10.1}  {per_us:>14.0}  {:>6}",
                n, bin_sols[k as usize]);
        }
    }

    /// Z1 prototype smoke test: synthesize Rz(0.3) at ε=1e-3 with D&C
    /// at m=3, verify it lands at a similar lde to single-search and
    /// time both for an A/B comparison.
    #[test]
    #[ignore]  // long: ~1 minute with 10M leaf budget
    fn z1_dc_smoke_rz_eps_1e_5() {
        let theta = 0.3_f64;
        let target: Mat2 = [
            [Complex64::from_polar(1.0, -theta / 2.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::from_polar(1.0, theta / 2.0)],
        ];
        let eps = 1e-5_f64;

        let synth_single = SynthesizerQ::new(eps).with_max_lde(28);
        let t0 = std::time::Instant::now();
        let r_single = synth_single.synthesize(target);
        let t_single = t0.elapsed();
        eprintln!(
            "single: lde={:?} dist={:?} t={:.0}ms",
            r_single.as_ref().map(|r| r.lde),
            r_single.as_ref().map(|r| r.distance),
            t_single.as_secs_f64() * 1000.0
        );

        for m in [1u32, 2, 3] {
            let synth_dc = SynthesizerQ::new(eps).with_max_lde(28).with_dc_split(m);
            let t1 = std::time::Instant::now();
            let r_dc = synth_dc.synthesize(target);
            let t_dc = t1.elapsed();
            let l_size = build_l_q(m).len();
            eprintln!(
                "  d&c m={m}: |L|={l_size}  lde={:?}  dist={:?}  t={:.0}ms",
                r_dc.as_ref().map(|r| r.lde),
                r_dc.as_ref().map(|r| r.distance),
                t_dc.as_secs_f64() * 1000.0
            );
        }
    }

    #[test]
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
        for &eps in &[1e-8, 1e-9_f64] {
            assert!(!SynthesizerQ::new(eps).use_f64_gs, "f64 default should be OFF at ε={eps:.0e}");
        }

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
        for &eps in &[1e-7, 1e-8, 1e-9_f64] {
            assert_eq!(SynthesizerQ::new(eps).bkz_block_size, 4,
                "BKZ default should be 4 at ε={eps:.0e}");
        }
    }

    #[test]
    fn synthesize_identity_at_k_0() {
        let one = Complex64::new(1.0, 0.0);
        let zero = Complex64::new(0.0, 0.0);
        let target = complex_target([[one, zero], [zero, one]]);
        let synth = SynthesizerQ::new(1e-9);
        let result = synth.synthesize(target).expect("identity should synthesize");
        assert_eq!(result.lde, 0, "identity should be at k=0");
        assert!(result.distance < 1e-9);
    }

    #[test]
    fn synthesize_q_gate() {
        let q = U2Q::q();
        let target = q.to_float();
        let synth = SynthesizerQ::new(1e-9);
        let result = synth.synthesize(target).expect("Q should synthesize");
        assert_eq!(result.lde, 0, "Q should be found at k=0");
        assert!(result.distance < 1e-9);
        // The synthesized gate string, when applied, should give back Q.
        assert!(result.gates.is_some());
    }

    #[test]
    fn synthesize_t_gate() {
        let t = U2Q::t();
        let target = t.to_float();
        let synth = SynthesizerQ::new(1e-9);
        let result = synth.synthesize(target).expect("T should synthesize");
        assert_eq!(result.lde, 0, "T should be found at k=0");
        assert!(result.distance < 1e-9);
    }

    #[test]
    fn synthesize_hqh() {
        let hqh: U2Q = U2Q::h() * U2Q::q() * U2Q::h();
        let target = hqh.to_float();
        let synth = SynthesizerQ::new(1e-9);
        let result = synth.synthesize(target).expect("HQH should synthesize");
        // HQH has k=2 (1 from each H).
        assert_eq!(result.lde, 2);
        assert!(result.distance < 1e-9);
    }

    #[test]
    fn synthesize_qhq() {
        let qhq: U2Q = U2Q::q() * U2Q::h() * U2Q::q();
        let target = qhq.to_float();
        let synth = SynthesizerQ::new(1e-9);
        let result = synth.synthesize(target).expect("QHQ should synthesize");
        assert_eq!(result.lde, 1);
        assert!(result.distance < 1e-9);
    }

    #[test]
    fn synthesize_h_gate() {
        // H has k=1 (one H gate). Should be found at k=1.
        let h = U2Q::h();
        let target = h.to_float();
        let synth = SynthesizerQ::new(1e-9);
        let result = synth.synthesize(target).expect("H should synthesize");
        assert_eq!(result.lde, 1);
        assert!(result.distance < 1e-9);
    }

    #[test]
    fn synthesize_returns_none_when_unreachable() {
        // Target Rx(π/16) — angle isn't a multiple of π/8, so the closest
        // Clifford+√T circuit at any small k is bounded away from it. With
        // ε=1e-9 (tight) and max_lde=2 (so the test stays under a second),
        // should return None.
        let theta = PI / 16.0;
        let c = (theta / 2.0).cos();
        let s = (theta / 2.0).sin();
        let i = Complex64::new(0.0, 1.0);
        let target: Mat2 = [
            [Complex64::new(c, 0.0), -i * s],
            [-i * s, Complex64::new(c, 0.0)],
        ];
        let synth = SynthesizerQ::new(1e-9).with_max_lde(2);
        let result = synth.synthesize(target);
        assert!(result.is_none(),
            "Rx(π/16) should not be reachable in Clifford+√T at k≤2 with ε=1e-9");
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
        let synth = SynthesizerQ::new(1e-9);
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
