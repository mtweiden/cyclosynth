//! Exact Clifford+T synthesis (Algorithm 3.14 of arXiv:2510.05816).
//!
//! Given a target unitary `V` and tolerance `ε`, finds a minimum-T-count
//! Clifford+T circuit `U` with `d_diamond(U, V) < ε`.
//!
//! # Search modes
//!
//! The [`SynthesizerT::synthesize`] entry point drives a search over T-count
//! `t = 0, 1, 2, …`, trying two backends depending on `t`:
//!
//! - [`direct_search`] (`t ≤ direct_limit`, default 6): brute-force
//!   enumeration over the norm shell `‖x‖² = 2^t` via
//!   [`crate::synthesis::search::brute_aligned_search`]. Tries even, T, and T†
//!   right-side branches, each combined with all 24 Clifford left
//!   prefixes. Fast for small `t`; exponential beyond that.
//!
//! - [`dc_search`] (`t > direct_limit`, Algorithm 3.11): divide-and-
//!   conquer using Matsumoto–Amano left prefixes `L_{t'}`. Splits at
//!   `t' = max(t − direct_limit, ⌈t − 5/2·log₂(1/ε)⌉)`. For each prefix
//!   `U_L ∈ L_{t'}`, searches for the right factor via
//!   [`lll_aligned_search`] at inner lde `k_inner` (see below).
//!   Tries even (U_L·U_R) and odd (U_L·U_R·T) inner branches.
//!
//! # Inner-lde convention
//!
//! [`lll_aligned_search`] uses `k_inner = T_inner/2 + 1` (norm shell
//! `2^k_inner`), not the T-count itself:
//!
//!   k_inner = t_inner / 2 + 1            (even t_inner)
//!   k_inner = (t_inner - 1) / 2 + 1      (odd t_inner)

use num_complex::Complex;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex};

use crate::matrix::U2T;
use crate::rings::types::{Float, Int};
use crate::rings::ZOmega;
use crate::synthesis::cliffords::{CLIFFORD_LDE0_IDX, CLIFFORD_TABLE_T};
use crate::synthesis::decomposer::BlochDecomposer;
use crate::synthesis::distance::{diamond_distance_u2t_float, Mat2};
use crate::synthesis::search::{
    brute_aligned_search, apply_t_dag_to_uv, apply_t_to_uv, apply_u2t_dag_to_uv, compute_align_vec,
    normalize4,
};

/// Global cache for `build_l_reference` results, keyed by `(t_prime, coset_dedup)`.
/// Values are wrapped in `Arc` so cache hits return an `O(1)` refcount bump
/// rather than cloning the full prefix list (at t'=14 that vector holds
/// ~329 k U2T values, ~32 MB).
static BUILD_L_CACHE: LazyLock<Mutex<HashMap<(u32, bool), Arc<Vec<U2T>>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Extract uv = [Re(u1), Im(u1), Re(u2), Im(u2)] from a 2×2 unitary matrix.
///
/// Normalizes to SU(2) first by dividing by √det, so that targets like
/// diag(1, i) (which has det=i) map to the same search direction as their
/// SU(2) representative diag(e^{−iπ/4}, e^{iπ/4}).
///
/// Convention: V ≈ e^{iφ} · [[u1, −ū2],[u2, ū1]].
fn unitary_to_uv(v: &Mat2) -> [Float; 4] {
    let det = v[0][0] * v[1][1] - v[0][1] * v[1][0];
    let phase = det.sqrt(); // principal square root of det
    if phase.norm() > 1e-12 {
        let u1 = v[0][0] / phase;
        let u2 = v[1][0] / phase;
        [u1.re, u1.im, u2.re, u2.im]
    } else {
        [v[0][0].re, v[0][0].im, v[1][0].re, v[1][0].im]
    }
}

/// Convert a 2×2 unitary to uv by trying all 8 global phases e^{ikπ/4} to find
/// the SU(2) form [[u1, −ū2], [u2, ū1]]. Returns None if no phase works.
///
/// Matches Python's mat_to_uv in bandb6.py.  The 8 phases correspond to the
/// possible determinants of Clifford+T products (det ∈ {e^{ikπ/4}}).
fn mat_to_uv(u: &Mat2) -> Option<[Float; 4]> {
    use std::f64::consts::FRAC_PI_4;
    for k in 0..8 {
        let ph = Complex::from_polar(1.0, k as Float * FRAC_PI_4);
        let m00 = ph * u[0][0];
        let m01 = ph * u[0][1];
        let m10 = ph * u[1][0];
        let m11 = ph * u[1][1];
        // Check [[u1, -ū2], [u2, ū1]]: u1 = m00, u2 = m10.
        // Need: m11 == conj(m00) and m01 == -conj(m10).
        let d11 = m11 - Complex::new(m00.re, -m00.im);
        let d01 = m01 - Complex::new(-m10.re, m10.im);
        if d11.norm() < 1e-9 && d01.norm() < 1e-9 {
            let u1 = m00;
            let u2 = m10;
            let v = [u1.re, u1.im, u2.re, u2.im];
            let n: Float = v.iter().map(|x| x * x).sum::<Float>().sqrt();
            if n > 1e-12 {
                return Some(v.map(|x| x / n));
            }
        }
    }
    None
}

/// Return 0 if det(m) is approximately ±1 or ±i (even ζ-power, ζ = e^{iπ/4}),
/// 1 if det(m) is at the half-integer positions ζ^{odd}, or None if det is
/// not on the 8th-root-of-unity circle.
///
/// Used as an upstream algebraic filter for `dc_search`: the `mat_to_uv`
/// rejection condition is exactly `det(U_L† · target) ∉ {ζ^{even}}`, which
/// reduces to `parity(det(U_L)) ≠ parity(det(target))`. Skipping prefixes
/// whose parity mismatches the target is provably equivalent to skipping
/// prefixes that mat_to_uv would have rejected — no heuristic, no
/// completeness loss.
fn det_zeta_parity(m: &Mat2) -> Option<u8> {
    let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    let mag_sq = det.norm_sqr();
    if (mag_sq - 1.0).abs() > 1e-3 {
        return None;
    }
    let max_axis = det.re.abs().max(det.im.abs());
    // Even ζ-powers ({1, i, -1, -i}): max(|re|, |im|) = 1.
    // Odd  ζ-powers ({ζ, ζ³, ζ⁵, ζ⁷}): max(|re|, |im|) = √2/2 ≈ 0.707.
    if max_axis > 0.9 {
        Some(0)
    } else if max_axis > 0.6 && max_axis < 0.85 {
        Some(1)
    } else {
        None
    }
}

/// Compute U_L† · target as a float matrix.
/// U_L is stored as U2T (exact), target as Mat2 (float).
fn u2t_dag_times_mat2(u_l: &U2T, target: &Mat2) -> Mat2 {
    let u_f = u_l.to_float();
    // (U_L†)[i][j] = conj(U_L[j][i])
    let ud00 = Complex::new(u_f[0][0].re, -u_f[0][0].im);
    let ud01 = Complex::new(u_f[1][0].re, -u_f[1][0].im);
    let ud10 = Complex::new(u_f[0][1].re, -u_f[0][1].im);
    let ud11 = Complex::new(u_f[1][1].re, -u_f[1][1].im);
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

// ─── MA prefix generation (Lemma 3.10) ───────────────────────────────────────

/// Canonical float key for a U2T matrix, invariant under global U(1) phase.
///
/// Rotates the flattened matrix so the largest-magnitude element becomes
/// real-positive, then rounds to 6 decimal places.  Used for O(n)-average
/// deduplication in build_L, matching Python's `_canonical_key`.
fn canonical_key(u: &U2T) -> [i64; 8] {
    let m = u.to_float(); // [[Complex; 2]; 2]
    let flat = [m[0][0], m[0][1], m[1][0], m[1][1]];

    // Find element with largest magnitude
    let (idx, _) = flat.iter().enumerate()
        .max_by(|(_, a), (_, b)| a.norm_sqr().partial_cmp(&b.norm_sqr()).unwrap())
        .unwrap();
    let piv = flat[idx];

    // Rotate so pivot is real-positive
    let rot: Vec<_> = if piv.norm() < 1e-12 {
        flat.iter().flat_map(|c| [c.re, c.im]).collect()
    } else {
        let phase = piv / piv.norm();
        flat.iter().flat_map(|c| {
            let r = c / phase;
            [r.re, r.im]
        }).collect()
    };

    // Round to 6 decimal places and encode as i64 (multiply by 1e6)
    rot.iter().map(|x| (x * 1_000_000.0).round() as i64).collect::<Vec<_>>()
        .try_into().unwrap()
}

/// Right-coset dedup gate for `build_l_reference` (stage 1 of
/// docs/plan_8d_prefix_rework.md, lever B1). Tri-state:
/// `CYCLOSYNTH_L_COSET=0` forces plain phase-dedup, `=1` forces coset
/// dedup at every ε, unset defers to the [`COSET_EPS_FLOOR`] rule.
/// Read once per process (LazyLock).
static L_COSET_DEDUP: LazyLock<Option<bool>> = LazyLock::new(|| {
    match std::env::var("CYCLOSYNTH_L_COSET").as_deref() {
        Ok("0") => Some(false),
        Ok("1") => Some(true),
        _ => None,
    }
});

/// ε floor for default-on right-coset dedup. Below it the radial cap
/// half-width ≈ ε²/4 falls under the f64 alignment chain's noise: cap
/// centers are misplaced by several half-widths and only the 8× coset-
/// mate redundancy lets SOME frame land inside the walk bound — dedup
/// there flips FOUND→none. Stays off until the 8D y chain is rebuilt
/// above f64 (the sibling of the 16D MPFR fix).
const COSET_EPS_FLOOR: Float = 1e-7;

/// Resolve the dedup mode for a given ε (env override first).
fn coset_mode_for(eps: Float) -> bool {
    (*L_COSET_DEDUP).unwrap_or(eps >= COSET_EPS_FLOOR)
}

/// Build L_{t'}: the Matsumoto–Amano prefix set with Clifford postmultiplication.
///
/// Matches Python's `build_L`:
///   L_0 = {I}
///   L_n (n≥1):
///     even branch: (HS^{b_n}T)·…·(HS^{b_1}T) · C  for b_i ∈ {0,1}, C ∈ C_1
///     odd  branch: T · (HS^{b_{n-1}}T)·…·(HS^{b_1}T) · C
///   deduplicated up to global phase, then (at ε ≥ COSET_EPS_FLOOR) up
///   to RIGHT cosets of the lde-0 Clifford subgroup ⟨S,X⟩: `(U_L·C)·U_R
///   = U_L·(C·U_R)` on the same shell, so one rep per coset preserves
///   completeness exactly — PROVIDED the per-frame walk is complete,
///   which holds above the floor only.
#[cfg(test)]
pub(crate) fn build_l_reference(t_prime: u32) -> Arc<Vec<U2T>> {
    // Legacy entry point (probes/tests): plain phase-dedup unless the env
    // forces coset mode. Production (`dc_search` + the prewarm) goes
    // through `build_l` with `coset_mode_for(eps)`.
    build_l(t_prime, (*L_COSET_DEDUP).unwrap_or(false))
}

/// `build_l_reference` with an explicit dedup mode; results cached per
/// `(t_prime, coset_dedup)`.
pub fn build_l(t_prime: u32, coset_dedup: bool) -> Arc<Vec<U2T>> {
    let key = (t_prime, coset_dedup);
    // Check cache first; clone of `Arc` is just a refcount bump.
    {
        let cache = BUILD_L_CACHE.lock().unwrap();
        if let Some(v) = cache.get(&key) {
            return Arc::clone(v);
        }
    }

    let result = Arc::new(build_l_inner_with(t_prime, coset_dedup));

    // Race-tolerant insert: another thread may have populated this entry while
    // we were computing; either copy is identical so an overwrite is harmless.
    BUILD_L_CACHE
        .lock()
        .unwrap()
        .insert(key, Arc::clone(&result));
    result
}

/// `build_l_inner` with an explicit dedup mode (the M1 census probe
/// compares both modes in one process, bypassing the env gate + cache).
fn build_l_inner_with(t_prime: u32, coset_dedup: bool) -> Vec<U2T> {
    if t_prime == 0 {
        return vec![U2T::eye()];
    }

    let h = U2T::h();
    let s = U2T::s();
    let t = U2T::t();
    let hs0t = h * t;        // H·T
    let hs1t = h * s * t;   // H·S·T

    let mut candidates: Vec<U2T> = Vec::new();

    // Even branch: length-t' product of (HS^b T) blocks, then · C
    let n = 1u32 << t_prime;
    for bits in 0..n {
        let mut u = U2T::eye();
        for i in 0..t_prime {
            let gate = if (bits >> i) & 1 == 1 { hs1t } else { hs0t };
            u = u * gate;
        }
        for (_, c_u2t) in CLIFFORD_TABLE_T {
            candidates.push(u * *c_u2t);
        }
    }

    // Odd branch: T · length-(t'-1) product · C
    let n2 = 1u32 << (t_prime - 1);
    for bits in 0..n2 {
        let mut u = t;
        for i in 0..(t_prime - 1) {
            let gate = if (bits >> i) & 1 == 1 { hs1t } else { hs0t };
            u = u * gate;
        }
        for (_, c_u2t) in CLIFFORD_TABLE_T {
            candidates.push(u * *c_u2t);
        }
    }

    // Deduplicate up to global phase (and, in coset mode, up to the right
    // coset u·⟨S,X⟩). Coset mode inserts the canonical key of every orbit
    // member u·c when a representative u is KEPT, so later coset-mates hit
    // `contains` with a single key computation each — ~2.3n keys total vs
    // 8n for a min-over-orbit key.
    let mut seen: std::collections::HashSet<[i64; 8]> = std::collections::HashSet::new();
    let mut unique: Vec<U2T> = Vec::new();
    for u in candidates {
        let key = canonical_key(&u);
        if seen.contains(&key) {
            continue;
        }
        unique.push(u);
        if coset_dedup {
            // c = I is CLIFFORD_LDE0_IDX[0], which re-inserts `key` itself.
            for &ci in CLIFFORD_LDE0_IDX.iter() {
                seen.insert(canonical_key(&(u * CLIFFORD_TABLE_T[ci].1)));
            }
        } else {
            seen.insert(key);
        }
    }
    unique
}

// ─── Solution conversion ──────────────────────────────────────────────────────

/// Build U2T from an integer lattice solution and denominator exponent.
///
/// sol = [a,b,c,d, e,f,g,h] encodes u1=(a,b,c,d), u2=(e,f,g,h) in ZOmega,
/// with U = [[u1, -ū2], [u2, ū1]] / √2^k (SU(2) convention).
fn solution_to_u2t(sol: &[i64; 8], k: u32) -> U2T {
    let u1 = ZOmega::new(
        Int::from_i64(sol[0]), Int::from_i64(sol[1]),
        Int::from_i64(sol[2]), Int::from_i64(sol[3]),
    );
    let u2 = ZOmega::new(
        Int::from_i64(sol[4]), Int::from_i64(sol[5]),
        Int::from_i64(sol[6]), Int::from_i64(sol[7]),
    );
    U2T::new(u1, -u2.conj(), u2, u1.conj(), k)
}

/// Decompose a lattice solution into a Clifford+T gate string.
fn solution_to_gates(sol: &[i64; 8], k: u32) -> String {
    BlochDecomposer.decompose(&solution_to_u2t(sol, k))
}

// ─── Trace output helper ─────────────────────────────────────────────────────

/// Emit one pass of the per-lde diagnostic block on stderr. Called at the
/// end of each `dc_search` invocation when `CYCLOSYNTH_TRACE=1` is set.
fn trace_dump_pass(
    t: u32,
    t_prime: u32,
    pass: u8,
    s: &crate::synthesis::diag::Snapshot,
    budget_hit: bool,
    pass_ms: f64,
    found: bool,
) {
    eprintln!(
        "[trace] lde={:>2} pass{} t'={:>2} prefixes={:>6} mat_uv_rej={:>6} \
         se_cb={:>9} se_nodes={:>11} (max/walk {:>9}) dist_rej={} budget={} {:>9.1}ms result={}",
        t, pass, t_prime, s.prefixes, s.mat_to_uv_rejected, s.se_callbacks,
        s.se_nodes, s.se_nodes_max, s.dist_rejected, budget_hit as u8, pass_ms,
        if found { "FOUND" } else { "none" }
    );
    let phase_total = s.t_build_ms + s.t_lll_ms + s.t_cholesky_ms + s.t_lu_ms + s.t_se_ms;
    if phase_total > 0.0 {
        eprintln!(
            "[trace]            phase_ms (cpu-summed) build={:>7.1} lll={:>7.1} \
             chol={:>7.1} lu={:>7.1} se={:>7.1} sum={:>7.1}",
            s.t_build_ms, s.t_lll_ms, s.t_cholesky_ms, s.t_lu_ms, s.t_se_ms, phase_total
        );
        let n_lll_calls = s.prefixes.saturating_sub(s.mat_to_uv_rejected);
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

// ─── LLL-based aligned search (used by dc_search inner step) ─────────────────

/// Scale a 4-element alignment vector `v` to the 8-element y vector used by
/// the lattice pipeline. `y = compute_align_vec(v) · sqrt(2^k) / 2`,
/// satisfying `‖y‖² = 2^(k-1)`. Used `powf` (not bit-shift) so `k ≥ 64`
/// stays well-defined.
fn uv_to_xy(v: [Float; 4], k: u32) -> [Float; 8] {
    let scale = 2.0_f64.powf(k as f64 / 2.0 - 1.0);
    compute_align_vec(v).map(|x| x * scale)
}

/// Per-prefix budget cap on Schnorr-Euchner leaf-callback invocations,
/// shared across the parallel inner-loop for a single MA prefix. When hit,
/// the search bails out of that prefix and signals the dispatcher via
/// `budget_hit`. The dispatcher uses a two-pass strategy at each lde:
///   - Pass 1 with `PASS1_CAP` (aggressive — bails unproductive prefixes
///     quickly).
///   - If no solution and at least one prefix tripped the cap, Pass 2 with
///     `PASS2_CAP` (effectively unbounded) before advancing to lde+1.
const PASS1_CAP: u64 = 2_000_000;
const PASS2_CAP: u64 = u64::MAX;

/// NODE budgets per `phase1` call: the leaf caps never bind on a
/// no-solution level (almost nothing reaches a leaf), so an empty
/// level used to walk unbudgeted to exhaustion. Pass 1 sits ≥ 700×
/// above every observed completing walk (empty levels are expensive
/// through prefix COUNT, not any single walk); a level that cannot
/// finish a walk in the pass-2 cap is pathological exhaustion, skipped
/// under the accepted speed-over-completeness rule.
const PASS1_NODE_CAP: u64 = 2_000_000;
const PASS2_NODE_CAP: u64 = 50_000_000;

/// Candidates collected per DC walk before the ε-distance check. The
/// coset dedup collapsed up to 8 independent first-hit draws into one
/// frame, so that frame must yield up to 8 candidates to keep the same
/// robustness against a borderline first candidate (passes the MPFR
/// alignment cap, fails f64 distance by ~1 ulp). Identical to
/// max_solutions=1 whenever the first candidate is good.
const DC_WALK_MAX_SOLUTIONS: usize = 8;

/// LLL-based aligned search for a right factor `U_R` of given lde `k`
/// matching the alignment vector `v`. Finds integer 8-vectors satisfying
/// the norm-shell, bilinear-form, and alignment constraints.
///
/// `max_solutions` caps how many candidates are returned (1 = historical
/// first-hit walk). `max_phase2_calls` caps the per-prefix SE leaf budget
/// and `max_nodes` the per-prefix SE NODE budget; if either is reached,
/// `budget_hit` is set so the caller can retry with a larger budget.
/// `external_abort` is the cross-branch winner signal (checked at every SE
/// recurse-entry; does not set `budget_hit`).
#[allow(clippy::too_many_arguments)]
fn lll_aligned_search(
    scratch: &mut crate::synthesis::lattice::scratch::IntScratch,
    v: [Float; 4],
    k: u32,
    eps: Float,
    max_solutions: usize,
    max_phase2_calls: u64,
    max_nodes: u64,
    budget_hit: &std::sync::atomic::AtomicBool,
    external_abort: Option<&std::sync::atomic::AtomicBool>,
) -> Vec<[i64; 8]> {
    // Old guard was `k > 62` because `target_norm = 1i64 << k` overflowed at
    // k ≥ 63. Now that target_norm is i128 and uv_to_xy uses powf, the safe
    // ceiling is much higher. Cap at 110 to stay comfortably below i128 range
    // (target_norm = 2^k must fit, and Σ-products in bilinear_b can reach
    // ~k+log₂(8) bits — 2^110 + log₂(8) ≈ 2^113, well within i128 = 2^127).
    if max_solutions == 0 || k > 110 {
        return Vec::new();
    }
    let y = uv_to_xy(v, k);
    // Lenstra-style 8D enumeration (Algorithm 3.6 of arXiv:2510.05816), with
    // MPFR (rug) at adaptive precision in the LLL+Cholesky setup phase. The
    // SE step downcasts to f64. Scratch is reused across all prefixes within
    // one rayon worker via map_init in dc_search.
    crate::synthesis::lattice::phase1(
        scratch, &y, k, eps, max_solutions, max_phase2_calls, max_nodes,
        budget_hit, external_abort,
    )
}

// ─── Optimal D&C split (Proposition 3.13) ─────────────────────────────────────

/// Compute the optimal t' for the divide-and-conquer split (Proposition 3.13).
///
/// t' = max(0, ⌈t − 5/2 · log₂(1/ε)⌉)
/// t_inner = t − t' is the residual lde passed to direct_search.
///
/// DC beats direct when t' > 0, i.e. t > 5/2·log₂(1/ε):
///   ε = 0.1  → threshold ≈  8.3,  DC kicks in at t ≥  9
///   ε = 0.01 → threshold ≈ 16.6,  DC kicks in at t ≥ 17
///   ε = 0.001→ threshold ≈ 24.9,  DC kicks in at t ≥ 25
///
/// When ε ≥ 1 the threshold is 0 and DC never helps, so t' = 0.
fn optimal_t_prime(t: u32, eps: Float) -> u32 {
    if eps >= 1.0 {
        return 0;
    }
    let threshold = (5.0 / 2.0) * (1.0 / eps).log2();
    if t as Float <= threshold {
        0
    } else {
        // ceil(t - threshold)
        let raw = t as Float - threshold;
        raw.ceil() as u32
    }
}

// ─── Result type ──────────────────────────────────────────────────────────────

/// Result of a successful synthesis.
pub struct SynthResultT {
    /// Clifford+T gate string (leftmost = first gate applied).
    /// `None` if the gate string could not be extracted.
    pub gates: Option<String>,
    /// Denominator exponent of the synthesized unitary.
    pub lde: u32,
    /// Diamond distance to the target.
    pub distance: Float,
}

// ─── Direct search branch tags ────────────────────────────────────────────────

enum DirectBranch {
    Plain,
    T,
    Tdg,
    ClifEven(usize),
    ClifT(usize),
    ClifTdg(usize),
}

// ─── Synthesizer ──────────────────────────────────────────────────────────────

/// Clifford+T synthesis backend, implementing Algorithm 3.14 of
/// arXiv:2510.05816. One of two backends behind the unified user-facing
/// [`crate::synthesis::Synthesizer`] (the other is
/// [`crate::synthesis::clifford_sqrt_t::SynthesizerQ`] for Clifford+√T).
/// Code shouldn't construct `SynthesizerT` directly — use `Synthesizer`
/// without `sqrt_t = true`. Public for direct access from tests.
pub struct SynthesizerT {
    /// Approximation precision in diamond distance.
    pub epsilon: Float,
    /// Maximum lde to search before giving up.
    pub max_lde: u32,
    /// Minimum lde to start searching from.
    /// Defaults to floor(3/2 · log₂(1/ε)), the information-theoretic lower bound
    /// on the minimum T-count for a generic SU(2) rotation.  Set to 0 to find
    /// exact low-T-count solutions for Cliffords and other special gates.
    pub min_lde: u32,
    /// Maximum lde for direct_search (brute-force brute_aligned_search).
    /// For t > direct_limit, skip direct_search and go straight to dc_search
    /// regardless of the optimal t' split. This prevents brute_aligned_search from
    /// hanging at large lde where it becomes O(2^(4t)) intractable.
    /// Default: 6 (brute_aligned_search is fast up to norm shell 2^6=64; beyond that
    /// DC with forced t_prime = t - direct_limit is used).
    pub direct_limit: u32,
}

impl SynthesizerT {
    /// ε-tuned defaults. min_lde's coefficient ramps 1.5 → 2.8 in
    /// log10(1/ε): shallow ε must give small-T/identity-like targets a
    /// chance below the generic floor, deep ε can skip known-empty
    /// levels. max_lde scales at 3.1× with headroom for worst-case
    /// angles. direct_limit is large only at moderate ε, where it
    /// covers the gap below the t' > 0 threshold; direct search is
    /// exponential in t, so deep ε keeps it small.
    pub fn new(epsilon: Float) -> Self {
        let (min_lde, max_lde) = if epsilon > 0.0 && epsilon < 1.0 {
            let log2_recip  = (1.0 / epsilon).log2();
            let log10_recip = (1.0 / epsilon).log10();
            let coef = if log10_recip <= 4.0 {
                1.5
            } else if log10_recip >= 6.0 {
                2.8
            } else {
                // Linear in log10(1/ε): 1.5 at decade 4, 2.8 at decade 6.
                1.5 + 0.65 * (log10_recip - 4.0)
            };
            let min_lde = (coef * log2_recip).floor() as u32;
            let max_lde = ((3.1 * log2_recip).ceil() as u32 + 2).max(50);
            (min_lde, max_lde)
        } else {
            (0, 50)
        };
        let direct_limit = if epsilon >= 1e-4 { 8 } else { 6 };
        Self { epsilon, max_lde, min_lde, direct_limit }
    }

    pub fn with_max_lde(mut self, max_lde: u32) -> Self {
        self.max_lde = max_lde;
        self
    }

    pub fn with_min_lde(mut self, min_lde: u32) -> Self {
        self.min_lde = min_lde;
        self
    }

    /// Find a minimum-lde Clifford+T circuit approximating `target`.
    ///
    /// Returns `None` if no circuit within `max_lde` achieves distance < `epsilon`.
    ///
    /// # Performance
    ///
    /// All ε regimes go through the unified Lenstra 8D pipeline (L²-LLL over
    /// an exact i256 Gram + f64 Gram-Schmidt + MPFR-128 Schnorr-Euchner +
    /// MPFR-scaled-precision LU for the cap-center solve). MPFR precision
    /// scales with ε via `compute_prec_q` (build_q at `8·log₂(1/ε)` bits)
    /// and `compute_lu_prec` (LU at `6·log₂(1/ε)`).
    ///
    /// Typical synth times:
    /// - ε ≥ 1e-3: 1–15 ms
    /// - ε = 1e-5: 70–400 ms
    /// - ε = 1e-7: 0.15–2 s
    /// - ε = 1e-8: ~20 s
    pub fn synthesize(&self, target: Mat2) -> Option<SynthResultT> {
        // First-touch rayon init with 16 MiB worker stacks — the 8D path
        // races the ζ₁₆ entries for global-pool initialisation, and a
        // 2 MiB-stack pool installed here overflows later deep walks
        // (see ensure_rayon_stack).
        crate::synthesis::ensure_rayon_stack();
        let raw_uv = unitary_to_uv(&target);
        let v = normalize4(raw_uv).unwrap_or([1.0, 0.0, 0.0, 0.0]);

        // Phase 1: direct_search for small t.  Starts at min_lde (not 0) because
        // no generic rotation can be approximated to within ε with fewer T-gates.
        for t in self.min_lde..=self.direct_limit {
            let result = self.try_at_lde(&target, v, t);
            if result.is_some() {
                return result;
            }
        }

        // Phase 2: DC regime — skip the gap where dc_search exists but prefix lists
        // are tiny and lll_aligned_search is cheap anyway (t=direct_limit+1 .. t_dc_start-1)
        let t_dc_start = if self.epsilon < 1.0 {
            let raw = (5.0 / 2.0) * (1.0 / self.epsilon).log2();
            (raw.ceil() as u32).max(self.direct_limit + 1)
        } else {
            self.direct_limit + 1
        };
        let t_dc_start = t_dc_start.max(self.min_lde);

        // Pre-warm the L cache in parallel for the t_prime values expected in the first
        // few steps of the t-loop.  build_l_reference is expensive (O(2^t_prime)) and lazily
        // populated; doing it here fills all cores before the search loop starts.
        // Cap at a 5-step horizon: solutions are almost always found within the first
        // few t values above t_dc_start, so building larger L sets is wasteful.
        if t_dc_start <= self.max_lde {
            let horizon = (t_dc_start + 5).min(self.max_lde);
            let needed: Vec<u32> = {
                let mut seen = std::collections::HashSet::new();
                (t_dc_start..=horizon)
                    .filter_map(|t| {
                        let tp = optimal_t_prime(t, self.epsilon);
                        if tp > 0 && seen.insert(tp) { Some(tp) } else { None }
                    })
                    .collect()
            };
            let coset = coset_mode_for(self.epsilon);
            needed.into_par_iter().for_each(|tp| { build_l(tp, coset); });
        }

        for t in t_dc_start..=self.max_lde {
            let result = self.try_at_lde(&target, v, t);
            if result.is_some() {
                return result;
            }
        }
        None
    }

    /// Try to find a solution at denominator exponent `t`.
    ///
    /// Dispatches to direct_search or dc_search:
    ///   - If t <= direct_limit AND optimal_t_prime == 0: direct_search (fast brute-force).
    ///   - Otherwise: dc_search with adaptive cap retry.
    ///
    /// Adaptive cap: first try dc_search with PASS1_CAP (aggressive — bails unproductive
    /// prefixes quickly). If no solution found AND budget was actually exhausted, retry
    /// with PASS2_CAP (full budget). If pass 1 found no solution and budget was *not*
    /// exhausted, the search was already exhaustive at this lde — skip pass 2 and let the
    /// caller advance to lde+1.
    fn try_at_lde(&self, target: &Mat2, v: [Float; 4], t: u32) -> Option<SynthResultT> {
        let trace = crate::synthesis::diag::trace_enabled();
        if t <= self.direct_limit {
            let t_start = std::time::Instant::now();
            let result = self.direct_search(target, v, t);
            if trace {
                eprintln!(
                    "[trace] lde={:>2} direct_search    {:>9.1}ms  result={}",
                    t,
                    t_start.elapsed().as_secs_f64() * 1000.0,
                    if result.is_some() { "FOUND" } else { "none" }
                );
            }
            result
        } else {
            if trace {
                crate::synthesis::diag::reset_all();
            }
            let t_start = std::time::Instant::now();
            let (result, budget_hit) =
                self.dc_search(target, v, t, PASS1_CAP, PASS1_NODE_CAP);
            let pass1_ms = t_start.elapsed().as_secs_f64() * 1000.0;
            if trace {
                let s = crate::synthesis::diag::snapshot();
                trace_dump_pass(t, optimal_t_prime(t, self.epsilon), 1, &s, budget_hit, pass1_ms, result.is_some());
            }
            if result.is_some() {
                return result;
            }
            if !budget_hit {
                // Search was exhaustive at PASS1_CAP — no solution exists at this lde.
                return None;
            }
            // Some prefix's budget was exhausted; the solution might be deeper.
            if trace {
                crate::synthesis::diag::reset_all();
            }
            let t_start2 = std::time::Instant::now();
            let (result2, budget_hit2) =
                self.dc_search(target, v, t, PASS2_CAP, PASS2_NODE_CAP);
            if trace {
                let s = crate::synthesis::diag::snapshot();
                trace_dump_pass(
                    t, optimal_t_prime(t, self.epsilon), 2, &s, budget_hit2,
                    t_start2.elapsed().as_secs_f64() * 1000.0,
                    result2.is_some(),
                );
            }
            result2
        }
    }

    /// Algorithm 3.6: direct search at lde `t`.
    ///
    /// Uses `search::brute_aligned_search` (fast brute-force with Cauchy-Schwarz pruning)
    /// for the inner lattice search.  `lll_aligned_search` (LLL+CVP) is reserved
    /// for the DC path where the inner lde is large and the CVP target is tight.
    ///
    /// Tries three top-level branches:
    ///   Even:  U ≈ target           → search at uv(target)
    ///   T:     U·T ≈ target         → search at uv(target·T†)
    ///   T†:    U·T† ≈ target        → search at uv(target·T)
    ///
    /// Then for each of the 24 Cliffords C:
    ///   Even:  C·U ≈ target         → search at uv(C†·target)
    ///   T:     C·U·T ≈ target       → search at uv(C†·target·T†)
    ///   T†:    C·U·T† ≈ target      → search at uv(C†·target·T)
    fn direct_search(&self, target: &Mat2, v: [Float; 4], t: u32) -> Option<SynthResultT> {
        let eps = self.epsilon;

        // Pre-compute search directions for all 24 Clifford left-prefixes.
        let clif_vs: Vec<[Float; 4]> = CLIFFORD_TABLE_T.iter()
            .map(|(_, c_u2t)| apply_u2t_dag_to_uv(c_u2t, v))
            .collect();

        // Build all 75 (v_search, tag) branches: 3 top-level + 23 Cliffords × 3.
        // Index 0 of CLIFFORD_TABLE_T is "I", covered by Plain/T/Tdg already.
        let mut branches: Vec<([Float; 4], DirectBranch)> = Vec::with_capacity(75);
        branches.push((v, DirectBranch::Plain));
        branches.push((apply_t_dag_to_uv(v), DirectBranch::T));
        branches.push((apply_t_to_uv(v), DirectBranch::Tdg));
        for i in 1..CLIFFORD_TABLE_T.len() {
            let vi = clif_vs[i];
            branches.push((vi, DirectBranch::ClifEven(i)));
            branches.push((apply_t_dag_to_uv(vi), DirectBranch::ClifT(i)));
            branches.push((apply_t_to_uv(vi), DirectBranch::ClifTdg(i)));
        }

        branches.par_iter().find_map_any(|(v_s, tag)| {
            for sol in brute_aligned_search(*v_s, t, eps, 1) {
                let (u2t, gates) = match tag {
                    DirectBranch::Plain => (
                        solution_to_u2t(&sol, t),
                        solution_to_gates(&sol, t),
                    ),
                    DirectBranch::T => (
                        solution_to_u2t(&sol, t) * U2T::t(),
                        solution_to_gates(&sol, t) + "T",
                    ),
                    DirectBranch::Tdg => (
                        solution_to_u2t(&sol, t) * U2T::t().dagger(),
                        solution_to_gates(&sol, t) + "SSST",
                    ),
                    DirectBranch::ClifEven(i) => {
                        let (c_str, c_u2t) = &CLIFFORD_TABLE_T[*i];
                        (*c_u2t * solution_to_u2t(&sol, t), solution_to_gates(&sol, t) + c_str)
                    },
                    DirectBranch::ClifT(i) => {
                        let (c_str, c_u2t) = &CLIFFORD_TABLE_T[*i];
                        (
                            *c_u2t * solution_to_u2t(&sol, t) * U2T::t(),
                            solution_to_gates(&sol, t) + "T" + c_str,
                        )
                    },
                    DirectBranch::ClifTdg(i) => {
                        let (c_str, c_u2t) = &CLIFFORD_TABLE_T[*i];
                        (
                            *c_u2t * solution_to_u2t(&sol, t) * U2T::t().dagger(),
                            solution_to_gates(&sol, t) + "SSST" + c_str,
                        )
                    },
                };
                let dist = diamond_distance_u2t_float(&u2t, target);
                if dist < eps {
                    return Some(SynthResultT { gates: Some(gates), lde: t, distance: dist });
                }
            }
            None
        })
    }

    /// Algorithm 3.11: divide-and-conquer with MA left prefixes.
    ///
    /// Optimal split t' = max(0, ceil(t - 5/2*log2(1/eps))) from Prop 3.13.
    /// Inner step uses lll_aligned_search (CVP-based), which is O(1) near a
    /// solution — fast exactly when DC is needed (large t, small eps).
    /// Even and odd inner branches are both tried per prefix.
    /// `max_phase2_calls` (SE leaf budget) and `max_nodes` (SE node budget) are
    /// forwarded to lll_aligned_search → lattice::phase1, per prefix × branch.
    /// Returns `(solution, budget_was_hit)` where `budget_was_hit=true` means at least
    /// one phase1 invocation exhausted an SE budget — the caller may want to
    /// retry at the same lde with a larger budget. If `false` and `solution` is `None`,
    /// the search was exhaustive at this lde and the caller should advance to lde+1.
    ///
    /// Cross-branch abort: the first prefix to find an ε-close solution sets
    /// `found_abort`; every other in-flight SE walk sees it at its next
    /// recurse-entry and unwinds (the 16D `external_abort` pattern). Without
    /// it, `find_any` only stops SCHEDULING new prefixes — the winning branch
    /// still waited for the slowest already-running loser. First-hit mode
    /// returns any valid find (speed > completeness), so abort-racing is
    /// acceptable; the per-lde loop structure (hence the reported lde) is
    /// unchanged.
    fn dc_search(
        &self,
        target: &Mat2,
        v: [Float; 4],
        t: u32,
        max_phase2_calls: u64,
        max_nodes: u64,
    ) -> (Option<SynthResultT>, bool) {
        let eps = self.epsilon;

        // Compute t_prime: use the optimal split from Prop 3.13, but if that gives
        // t_prime == 0 (meaning the formula says direct search is fine), force
        // t_prime = t - direct_limit so the inner LLL/CVP search stays within the
        // brute-force-tractable regime (inner T-count = direct_limit).
        // Guard: if opt==0 the formula says t is below the DC threshold, meaning
        // no split is theoretically needed and forcing t_prime = t-direct_limit
        // would produce an exponentially large prefix set.  Return None so the
        // outer loop advances to the next (higher) t where opt > 0.
        let t_prime = {
            let opt = optimal_t_prime(t, eps);
            if opt == 0 && t > self.direct_limit {
                return (None, false);
            }
            opt
        };

        if t_prime == 0 || t_prime > t {
            return (self.direct_search(target, v, t), false);
        }
        let t_inner = t - t_prime;

        // Convert t_inner (T-count) to the k convention used by lll_aligned_search.
        //   even T-count: k = t_inner/2 + 1
        //   odd  T-count: k = (t_inner-1)/2 + 1
        let odd_inner = t_inner % 2 == 1;
        let k_inner: u32 = if odd_inner {
            (t_inner - 1) / 2 + 1
        } else {
            t_inner / 2 + 1
        };

        let prefixes = build_l(t_prime, coset_mode_for(eps));
        if crate::synthesis::diag::trace_enabled() {
            crate::synthesis::diag::N_PREFIXES
                .fetch_add(prefixes.len() as u64, std::sync::atomic::Ordering::Relaxed);
        }

        // Parallel search over all left prefixes.
        // find_any stops scheduling new prefixes as soon as any one returns
        // Some(...); with_min_len ensures rayon distributes work evenly
        // rather than keeping everything on one thread when items complete
        // quickly.
        let n_threads = rayon::current_num_threads();
        let n = prefixes.len();
        let chunk = (n / n_threads).max(1);
        let budget_hit = std::sync::atomic::AtomicBool::new(false);
        // Cross-branch winner signal: set by the first prefix that finds an
        // ε-close solution; checked at every SE recurse-entry of every
        // other in-flight walk.
        let found_abort = std::sync::atomic::AtomicBool::new(false);

        // build_l_reference order correlates position with structure, so
        // contiguous chunks concentrate similar prefixes on one worker;
        // dealing lowers time-to-first-hit under find_any.
        let indices: Vec<u32> = (0..n as u32).collect();
        let order = crate::synthesis::stride_interleave(&indices, n_threads);

        // Algebraic parity pre-filter: `mat_to_uv(U_L† · target)` succeeds
        // iff `parity(det(U_L)) == parity(det(target))`. Skipping prefixes
        // with mismatched parity short-circuits before `u2t_dag_times_mat2`
        // and saves the per-prefix float matmul + 8-phase trial. Provably
        // equivalent to mat_to_uv's rejection condition; no completeness loss.
        // `None` = target det not on the 8th-root-of-unity circle (e.g. an
        // arbitrary unitary), in which case we fall through to the original
        // mat_to_uv check.
        let target_parity = det_zeta_parity(target);

        // Inner branches (`odd` flags) run per prefix: even (U_L·U_R) and,
        // when t_inner > 0, odd (U_L·U_R·T). A branch-ordered "two-sweep"
        // variant was tried and killed — M2 measured branch wins at ~50/50
        // with no t_inner-parity rule, so no sweep order dominates
        // (docs/w_8d_rework_notes.md; removed 2026-06-12).
        let plans: Vec<Vec<bool>> = if t_inner == 0 {
            vec![vec![false]]
        } else {
            vec![vec![false, true]]
        };

        // Per-worker scratch: rayon's `map_init` allocates one
        // `IntScratch` (pre-allocated MPFR/i256 buffers at the right
        // precision for `eps`) per worker thread and reuses it across every
        // prefix that worker handles, avoiding per-op allocation in the
        // hot path.
        //
        // `budget_hit` and `found_abort` are shared across sweeps:
        // budget_hit ORs (a sweep-1 budget trip must surface even when
        // sweep 2 completes exhaustively — the 2-pass requeue depends on
        // it), and found_abort can only be set by a winner, which ends the
        // sweep loop anyway.
        let mut result: Option<SynthResultT> = None;
        for plan in &plans {
            result = order
                .par_iter()
                .enumerate()
                .with_min_len(chunk)
                .map_init(
                    || crate::synthesis::lattice::scratch::IntScratch::new(eps),
                    |scratch, (pos, &pi)| -> Option<SynthResultT> {
                        let u_l = &prefixes[pi as usize];
                        if let Some(tp) = target_parity {
                            if det_zeta_parity(&u_l.to_float()) != Some(tp) {
                                crate::synthesis::diag::N_MAT_TO_UV_REJECTED
                                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                return None;
                            }
                        }
                        let m_inner = u2t_dag_times_mat2(u_l, target);
                        let v_inner = match mat_to_uv(&m_inner) {
                            Some(v) => v,
                            None => {
                                crate::synthesis::diag::N_MAT_TO_UV_REJECTED
                                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                return None;
                            }
                        };

                        for &odd in plan {
                            // Even inner branch: U_L · U_R ≈ target
                            // Odd  inner branch: U_L · U_R · T ≈ target
                            let v_branch = if odd {
                                apply_t_dag_to_uv(v_inner)
                            } else {
                                v_inner
                            };
                            for sol in lll_aligned_search(
                                scratch, v_branch, k_inner, eps,
                                DC_WALK_MAX_SOLUTIONS, max_phase2_calls,
                                max_nodes, &budget_hit, Some(&found_abort),
                            ) {
                                let u2t = if odd {
                                    *u_l * solution_to_u2t(&sol, k_inner) * U2T::t()
                                } else {
                                    *u_l * solution_to_u2t(&sol, k_inner)
                                };
                                let dist = diamond_distance_u2t_float(&u2t, target);
                                if dist < eps {
                                    found_abort
                                        .store(true, std::sync::atomic::Ordering::Relaxed);
                                    // M2: branch-win + sweep-position telemetry.
                                    crate::synthesis::diag::record_branch_win(
                                        odd, pos, n, t,
                                    );
                                    return Some(SynthResultT {
                                        gates: Some(BlochDecomposer.decompose(&u2t)),
                                        lde: t,
                                        distance: dist,
                                    });
                                }
                                crate::synthesis::diag::N_DIST_REJECTED
                                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            }
                        }

                        None
                    },
                )
                .find_any(|r| r.is_some())
                .flatten();
            if result.is_some() {
                break;
            }
        }

        (result, budget_hit.load(std::sync::atomic::Ordering::Relaxed))
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthesis::distance::diamond_distance_float;
    use std::{f64::consts::{FRAC_1_SQRT_2, PI}};

    fn rz(theta: Float) -> Mat2 {
        [
            [Complex::from_polar(1., -theta / 2.), Complex::new(0., 0.)],
            [Complex::new(0., 0.), Complex::from_polar(1., theta / 2.)],
        ]
    }

    fn ry(theta: Float) -> Mat2 {
        let c = (theta / 2.).cos();
        let s = (theta / 2.).sin();
        [
            [Complex::new(c, 0.), Complex::new(-s, 0.)],
            [Complex::new(s, 0.), Complex::new(c, 0.)],
        ]
    }

    fn mat_mul(a: Mat2, b: Mat2) -> Mat2 {
        [
            [a[0][0]*b[0][0] + a[0][1]*b[1][0], a[0][0]*b[0][1] + a[0][1]*b[1][1]],
            [a[1][0]*b[0][0] + a[1][1]*b[1][0], a[1][0]*b[0][1] + a[1][1]*b[1][1]],
        ]
    }

    /// Same convention as bin/time_synthesis: U3(a,b,c) = Rz(a)·Ry(b)·Rz(c).
    fn u3(a: Float, b: Float, c: Float) -> Mat2 {
        mat_mul(mat_mul(rz(a), ry(b)), rz(c))
    }

    fn check_result(result: &SynthResultT, _target: &Mat2, eps: Float) {
        assert!(
            result.distance < eps,
            "distance={:.6e} ≥ epsilon={:.6e}",
            result.distance, eps
        );
    }

    /// Re-build a U2T from the synthesized gate string by parsing left-to-right.
    fn gates_to_u2t_verify(gate_str: &str) -> crate::matrix::U2T {
        use crate::matrix::U2T;
        let mut u = U2T::eye();
        for ch in gate_str.chars() {
            let g = match ch {
                'H' => U2T::h(),
                'S' => U2T::s(),
                'T' => U2T::t(),
                'Z' => U2T::z(),
                'X' => U2T::x(),
                'Y' => U2T::y(),
                'I' => U2T::eye(),
                _ => panic!("unexpected gate char: {ch}"),
            };
            u = u * g;
        }
        u
    }

    /// End-to-end correctness verification: synthesize, then independently
    /// re-evaluate the gate string and confirm the result still satisfies the
    /// approximation bound. Validates that:
    ///   1. result.distance < eps (reported distance is below threshold)
    ///   2. The gate string parses to a U2T whose lde matches result.lde
    ///   3. Re-evaluated diamond distance to target matches result.distance
    ///   4. T-count of the gate string is consistent with the lde
    fn verify_synthesis_round_trip(target: &Mat2, eps: Float, label: &str) {
        // max_lde generously oversized so very tight ε (1e-5+) has room.
        let synth = SynthesizerT::new(eps).with_max_lde(80);
        let result = synth
            .synthesize(*target)
            .unwrap_or_else(|| panic!("{label}: synthesis returned None"));

        // Check 1: reported distance under threshold
        assert!(
            result.distance < eps,
            "{label}: result.distance={:.6e} ≥ eps={:.6e}",
            result.distance,
            eps
        );

        // Check 2: gate string round-trips. Re-build the U2T from the gate
        // string and verify the diamond distance is the same as reported.
        let gates = result
            .gates
            .as_ref()
            .unwrap_or_else(|| panic!("{label}: result.gates is None"));
        let rebuilt = gates_to_u2t_verify(gates);
        let rebuilt_float = rebuilt.to_float();
        let recomputed_dist = diamond_distance_float(&rebuilt_float, target);
        assert!(
            recomputed_dist < eps,
            "{label}: re-evaluated distance={:.6e} ≥ eps={:.6e} (gate string does not approximate target)",
            recomputed_dist,
            eps
        );
        // Reported and rebuilt distances should agree to FP precision (the
        // synth doesn't round-trip through the gate string internally, so
        // small rounding from to_float()/diamond_distance_float() is expected,
        // but they should agree to ~1e-12).
        let dist_consistency = (recomputed_dist - result.distance).abs();
        // Tolerance: diamond distance involves catastrophic cancellation in
        // `1 − |tr(U·V†)|²/4` when U is close to V. Plus the rebuilt path
        // accumulates f64 error through ~n_gates U2T products. Empirically
        // ~n_gates · 1e-12 covers the round-trip noise even for 200+ gate
        // sequences at ε=1e-7. Floor at 1e-10 for short sequences; the
        // tolerance must remain << ε so the "within ε" guarantee isn't
        // compromised.
        let n_gates = result.gates.as_ref().map(|s| s.len()).unwrap_or(0) as f64;
        // Per-gate bound + floor. The `dist < ε` check above is the real
        // correctness gate; this consistency check is a self-sanity ratchet
        // against silent algorithmic divergence between synth.synthesize's
        // reported distance and the gate-replay distance.
        let tol = (n_gates * 5e-11).max(1e-9);
        assert!(
            dist_consistency < tol,
            "{label}: rebuilt distance ({:.6e}) differs from reported ({:.6e}) by {:e} (tol={:e}, gates_len={})",
            recomputed_dist,
            result.distance,
            dist_consistency,
            tol,
            n_gates as usize
        );

        // Check 3: T-count of the gate string. result.lde holds the
        // synthesizer's t-loop value (the *target* T-count for the search).
        // The actual gate string can have at most that many T gates.
        let t_count = gates.chars().filter(|&c| c == 'T').count() as u32;
        // We accept up to lde + a few (the BlochDecomposer can introduce
        // small constant overhead from final Clifford fixup).
        assert!(
            t_count <= result.lde + 8,
            "{label}: T-count={} far exceeds reported lde={}",
            t_count,
            result.lde
        );

        eprintln!(
            "[verify] {label}: lde={} dist={:.4e} (rebuilt: {:.4e}) T-count={} gates_len={} U2T_k={}",
            result.lde,
            result.distance,
            recomputed_dist,
            t_count,
            gates.len(),
            rebuilt.k
        );
    }

    #[test]
    fn verify_correctness_at_1e_3_rz_03() {
        verify_synthesis_round_trip(&rz(0.30), 1e-3, "Rz(0.30) @ 1e-3");
    }

    #[test]
    fn verify_correctness_at_1e_4_rz_03() {
        verify_synthesis_round_trip(&rz(0.30), 1e-4, "Rz(0.30) @ 1e-4");
    }

    #[test]
    fn verify_correctness_at_1e_4_rz_pi7() {
        verify_synthesis_round_trip(&rz(PI / 7.0), 1e-4, "Rz(π/7) @ 1e-4");
    }

    #[test]
    fn verify_correctness_at_1e_4_u3() {
        verify_synthesis_round_trip(&u3(0.3, 0.7, 1.2), 1e-4, "U3(0.3,0.7,1.2) @ 1e-4");
    }

    #[test]
    fn verify_correctness_at_1e_5_rz_03() {
        verify_synthesis_round_trip(&rz(0.30), 1e-5, "Rz(0.30) @ 1e-5");
    }

    /// Round-trip at ε=1e-7. Validates the L²-LLL backend at deeper ε.
    /// Fast (~40 ms) on `Rz(0.30)` after the post-Frobenius perf fixes.
    #[test]
    fn verify_correctness_at_1e_7_rz_03() {
        verify_synthesis_round_trip(&rz(0.30), 1e-7, "Rz(0.30) @ 1e-7");
    }

    /// Round-trip at ε=1e-7 on `Rz(π/7)` — the worst-case 1e-7 target in
    /// the bench (lde=70 vs typical 66). Slowest test in the suite (~2 s),
    /// kept in the default run because it's the only direct guard for the
    /// "outlier-target at deep ε" failure mode that motivated the
    /// MPFR-alignment / Frobenius-distance fixes.
    #[test]
    fn verify_correctness_at_1e_7_rz_pi7() {
        verify_synthesis_round_trip(&rz(PI / 7.0), 1e-7, "Rz(π/7) @ 1e-7");
    }

    #[test]
    fn test_synthesize_identity() {
        let id: Mat2 = [[Complex::new(1., 0.), Complex::new(0., 0.)], [Complex::new(0., 0.), Complex::new(1., 0.)]];
        // with_min_lde(0): identity is a Clifford with exact solution at lde=0.
        let synth = SynthesizerT::new(0.01).with_min_lde(0);
        let result = synth.synthesize(id).expect("Should synthesize identity");
        check_result(&result, &id, 0.01);
        assert_eq!(result.lde, 0, "Identity should have lde=0");
    }

    #[test]
    fn test_synthesize_s_gate() {
        let s: Mat2 = [
            [Complex::new(1., 0.), Complex::new(0., 0.)],
            [Complex::new(0., 0.), Complex::new(0., 1.)],
        ];
        // with_min_lde(0): S is a Clifford with exact solution at lde=0.
        let synth = SynthesizerT::new(0.01).with_min_lde(0);
        let result = synth.synthesize(s).expect("Should synthesize S");
        println!("{:?}", result.gates);
        check_result(&result, &s, 0.01);
        assert_eq!(result.lde, 0, "S is a Clifford, should need lde=0");
    }

    #[test]
    fn test_synthesize_h_gate() {
        let r = FRAC_1_SQRT_2 as Float;
        let h: Mat2 = [
            [Complex::new(r, 0.), Complex::new(r, 0.)],
            [Complex::new(r, 0.), Complex::new(-r, 0.)],
        ];
        let synth = SynthesizerT::new(0.01);
        let result = synth.synthesize(h).expect("Should synthesize H");
        check_result(&result, &h, 0.01);
    }

    #[test]
    fn test_synthesize_rz_small() {
        // Rz(π/4) = T gate, should need lde=1.
        let target = rz(PI as Float / 4.);
        let synth = SynthesizerT::new(0.01);
        let result = synth.synthesize(target).expect("Should synthesize Rz(π/4)");
        println!("{:?}", result.gates);

        check_result(&result, &target, 0.01);
    }

    #[test]
    fn test_synthesize_rz_moderate_1() {
        let target = rz(0.3);
        let synth = SynthesizerT::new(0.01);
        let result = synth.synthesize(target).expect("Should synthesize Rz(0.3)");
        println!("{:?}", result.gates);
        check_result(&result, &target, 0.01);
    }

    #[test]
    fn test_synthesize_rz_moderate_2() {
        let target = rz(1.34);
        let synth = SynthesizerT::new(0.01);
        let result = synth.synthesize(target).expect("Should synthesize Rz(1.34)");
        println!("{:?}", result.gates);
        check_result(&result, &target, 0.01);
    }

    #[test]
    fn test_synthesize_rz_hard_1() {
        // eps=0.01: needs t~26, DC kicks in at t>=17 (t'=t-17, t_inner=17).
        // Much faster than eps=0.001 which needs t~40.
        let target = rz(0.3);
        let synth = SynthesizerT::new(0.01);
        let result = synth.synthesize(target).expect("Should synthesize Rz(0.3) at eps=0.01");
        println!("{:?}", result.gates);
        check_result(&result, &target, 0.01);
    }

    #[test]
    fn test_synthesize_rz_hard_2() {
        let target = rz(1.34);
        let synth = SynthesizerT::new(0.001);
        let result = synth.synthesize(target).expect("Should synthesize Rz(1.34) at eps=0.01");
        println!("{:?}", result.gates);

        check_result(&result, &target, 0.01);
    }

    /// Empirical sister of `clifford_sqrt_t::tests::build_l_q_dc_cost_ratio`.
    /// Computes the same `S(t', α)` cost-ratio (Σ count(t', k)/α^k) for
    /// Clifford+T's `build_l_reference` so we can directly compare what the naive
    /// cost model predicts for D&C in each ring.
    #[test]
    fn build_l_size_and_cost_ratio() {
        eprintln!("\n|L_{{t'}}| sizes:");
        for t_prime in 0..=10 {
            let l = build_l_reference(t_prime);
            eprintln!("  t'={t_prime:>2}  |L_{{t'}}|={:>8}", l.len());
        }

        eprintln!("\nk_prefix histogram (Clifford+T, build_l_reference):");
        for t_prime in 1..=8 {
            let l = build_l_reference(t_prime);
            let mut counts: Vec<u64> = vec![0; 64];
            for u in l.iter() {
                let k = u.k as usize;
                if k < counts.len() { counts[k] += 1; }
            }
            let mut k_min = u32::MAX; let mut k_max = 0;
            for u in l.iter() { k_min = k_min.min(u.k); k_max = k_max.max(u.k); }
            eprintln!(
                "  t'={t_prime:>2}  total={:>8}  k range [{k_min}, {k_max}]",
                l.len()
            );
        }

        eprintln!("\nS(t', α) = Σ_k count(t', k) / α^k  (D&C cost ratio):");
        eprintln!("  t'  total      α=2.0    α=2.5    α=3.0    α=3.5    α=4.0");
        for t_prime in 1..=10 {
            let l = build_l_reference(t_prime);
            let mut counts: Vec<u64> = vec![0; 64];
            for u in l.iter() {
                let k = u.k as usize;
                if k < counts.len() { counts[k] += 1; }
            }
            eprint!("  {t_prime:>2}  {:>8}", l.len());
            for &alpha in &[2.0_f64, 2.5, 3.0, 3.5, 4.0] {
                let s: f64 = counts
                    .iter()
                    .enumerate()
                    .map(|(k, &c)| (c as f64) / alpha.powi(k as i32))
                    .sum();
                eprint!("   {s:>8.2}");
            }
            eprintln!();
        }
    }

    /// Stage-2 contract: `dc_search`'s `budget_hit` is shared across the
    /// branch sweeps (OR semantics) and surfaces to the caller — the 2-pass
    /// requeue depends on it. With a 1-node SE budget every walk in BOTH
    /// sweeps trips immediately on an empty level → `(None, true)`; the
    /// same level at the production pass-1 caps completes exhaustively →
    /// `(None, false)`. Together these pin the budget-driven requeue
    /// signal through the two-sweep restructure.
    #[test]
    fn budget_hit_ors_across_sweeps() {
        // Rz(π/7) @ 1e-5 first hits around lde 51 — the DC band (t'=1 at
        // t=42) below it is a wide stretch of cheap empty levels.
        let target = rz(PI / 7.0);
        let eps = 1e-5_f64;
        let synth = SynthesizerT::new(eps);
        let raw_uv = unitary_to_uv(&target);
        let v = normalize4(raw_uv).unwrap();
        // Scan UP from the DC threshold for an EMPTY level (production
        // caps → (None, false), i.e. exhaustive) with surviving prefixes,
        // then verify that a 1-node SE budget on that level — which trips
        // every walk in BOTH sweeps at its first recurse-entry — surfaces
        // through the shared budget_hit as (None, true). The scan STOPS at
        // the first FOUND level: empty levels live below first-hit, and
        // climbing past it would build exponentially larger L_{t'} sets
        // for nothing.
        let mut verified = false;
        for t in 42..=52u32 {
            if optimal_t_prime(t, eps) == 0 {
                continue;
            }
            let (res, hit) = synth.dc_search(&target, v, t, PASS1_CAP, PASS1_NODE_CAP);
            if res.is_some() {
                break; // first-hit reached; no empty levels above
            }
            assert!(!hit, "production caps should be exhaustive at lde={t}");
            let (res1, hit1) = synth.dc_search(&target, v, t, u64::MAX, 1);
            assert!(res1.is_none(), "no solution reachable on a 1-node budget (lde={t})");
            if hit1 {
                verified = true;
                break;
            }
            // else: level had no surviving prefixes (odd-t' parity
            // wipeout) — no walk ran, keep scanning.
        }
        assert!(
            verified,
            "no empty dc level with surviving prefixes found below first-hit — \
             budget_hit OR-across-sweeps could not be exercised"
        );
    }

    /// Structural soundness of the right-coset dedup (stage 1, lever B1):
    /// every plain-dedup prefix must be reachable as `rep · c` for some
    /// kept representative `rep` and lde-0 Clifford `c` — i.e. the coset
    /// orbits of the kept reps COVER the full prefix set, so no subproblem
    /// is lost. Checked exactly (canonical-key equality, the same
    /// equivalence the production dedup uses) for t' = 1..6.
    #[test]
    fn coset_dedup_covers_all_prefixes() {
        for tp in 1..=6 {
            let plain = build_l_inner_with(tp, false);
            let coset = build_l_inner_with(tp, true);
            assert!(coset.len() < plain.len(), "t'={tp}: coset dedup removed nothing");
            let mut covered: std::collections::HashSet<[i64; 8]> =
                std::collections::HashSet::new();
            for u in &coset {
                for &ci in CLIFFORD_LDE0_IDX.iter() {
                    covered.insert(canonical_key(&(*u * CLIFFORD_TABLE_T[ci].1)));
                }
            }
            for (i, u) in plain.iter().enumerate() {
                assert!(
                    covered.contains(&canonical_key(u)),
                    "t'={tp}: plain prefix {i} not covered by any kept coset orbit"
                );
            }
        }
    }

    /// Diagnostic probe (ignored): the t_identity target-2 @1e-5 FOUND→none
    /// flip under coset dedup, reproduced at level t=47 (t'=6). Finds every
    /// PLAIN prefix that yields an ε-valid solution, maps each winner to its
    /// kept coset representative, reruns the rep's two branches, and checks
    /// whether the image solution c·U_R appears — pinpointing where the
    /// Q-isometric-bijection argument breaks in practice.
    /// Run: `cargo test --release --lib probe_coset_flip_t47 -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn probe_coset_flip_t47() {
        // SplitMix64(0xC0FFEE) — t_identity_1e5's generator; target idx 2.
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
        // Widen the SE walk bound for the WHOLE probe (LazyLock-once; must
        // precede the first phase1 call). If the rep's missing image shows
        // up at bound 4.0, its Q-norm in that frame is in (1.51, 4.0] and
        // the Q-band model is frame-fragile; if it stays missing, the f64
        // partial-eucl norm prune (the known 1.5e-8-cliff mechanism) is
        // killing the branch.
        std::env::set_var("CYCLOSYNTH_SE_BOUND_8D", "4.0");
        let mut rng = Xs(0xC0FFEE);
        let mut tri = (0.0, 0.0, 0.0);
        for _ in 0..3 {
            tri = (
                rng.range(0.2, PI - 0.2),
                rng.range(0.1, 2.0 * PI - 0.1),
                rng.range(0.1, 2.0 * PI - 0.1),
            );
        }
        let (th, ph, la) = tri;
        // u3 with the t_identity convention (global-phase normalized).
        let (c, s) = ((th / 2.0).cos(), (th / 2.0).sin());
        let eilam = Complex::from_polar(1.0, la);
        let eiphi = Complex::from_polar(1.0, ph);
        let g = Complex::from_polar(1.0, -(ph + la) / 2.0);
        let target: Mat2 = [
            [Complex::new(c, 0.0) * g, -eilam * s * g],
            [eiphi * s * g, eiphi * eilam * Complex::new(c, 0.0) * g],
        ];

        coset_flip_probe(target, 1e-5, 47);
    }

    /// Same forensic probe at the bench-suite 1e-8 flip: time_synthesis
    /// target_00 (xorshift64, seed 0xC0FFEEBAADD0E|1), lde 78 (t'=12),
    /// which still drifts to 80 under coset dedup after the
    /// euclidean_cholesky trust guards.
    /// Run: `cargo test --release --lib probe_coset_flip_t78 -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn probe_coset_flip_t78() {
        std::env::set_var("CYCLOSYNTH_SE_BOUND_8D", "4.0");
        fn xorshift64(s: &mut u64) -> u64 {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
            *s
        }
        fn rand_angle(s: &mut u64) -> f64 {
            let b = xorshift64(s) >> 11;
            (b as f64) / ((1u64 << 53) as f64) * 2.0 * PI
        }
        let mut state: u64 = 0xC0FFEE_BAADD0E_u64 | 1;
        let a = rand_angle(&mut state);
        let b = rand_angle(&mut state);
        let c = rand_angle(&mut state);
        let target = u3(a, b, c);
        coset_flip_probe(target, 1e-8, 78);
    }

    #[allow(clippy::needless_range_loop)]
    fn coset_flip_probe(target: Mat2, eps: Float, t: u32) {
        use std::sync::atomic::AtomicBool;
        let t_prime = optimal_t_prime(t, eps);
        let t_inner = t - t_prime;
        let k_inner: u32 = if t_inner % 2 == 1 { (t_inner - 1) / 2 + 1 } else { t_inner / 2 + 1 };
        eprintln!("t={t} t'={t_prime} t_inner={t_inner} k_inner={k_inner}");

        let plain = build_l_inner_with(t_prime, false);
        let coset = build_l_inner_with(t_prime, true);
        eprintln!("|plain|={} |coset|={}", plain.len(), coset.len());

        let run_prefix = |u_l: &U2T, max_sols: usize| -> Vec<(bool, [i64; 8], f64)> {
            let mut out = Vec::new();
            let m_inner = u2t_dag_times_mat2(u_l, &target);
            let Some(v_inner) = mat_to_uv(&m_inner) else { return out };
            let mut scratch = crate::synthesis::lattice::scratch::IntScratch::new(eps);
            for odd in [false, true] {
                let v_b = if odd { apply_t_dag_to_uv(v_inner) } else { v_inner };
                let hit = AtomicBool::new(false);
                for sol in lll_aligned_search(
                    &mut scratch, v_b, k_inner, eps, max_sols, u64::MAX,
                    50_000_000, &hit, None,
                ) {
                    let u2t = if odd {
                        *u_l * solution_to_u2t(&sol, k_inner) * U2T::t()
                    } else {
                        *u_l * solution_to_u2t(&sol, k_inner)
                    };
                    let dist = diamond_distance_u2t_float(&u2t, &target);
                    out.push((odd, sol, dist));
                }
            }
            out
        };

        // 1) all plain winners.
        let winners: Vec<(usize, bool, [i64; 8], f64)> = plain
            .par_iter()
            .enumerate()
            .flat_map_iter(|(i, u_l)| {
                run_prefix(u_l, 16)
                    .into_iter()
                    .filter(|&(_, _, d)| d < eps)
                    .map(move |(odd, sol, d)| (i, odd, sol, d))
                    .collect::<Vec<_>>()
            })
            .collect();
        eprintln!("plain winners: {}", winners.len());
        for &(i, odd, sol, d) in winners.iter().take(8) {
            eprintln!("  plain[{i}] odd={odd} sol={sol:?} dist={d:.3e}");
        }

        // 2) orbit-key → rep map for the coset set.
        let mut rep_of: HashMap<[i64; 8], usize> = HashMap::new();
        for (ri, r) in coset.iter().enumerate() {
            for &ci in CLIFFORD_LDE0_IDX.iter() {
                rep_of.entry(canonical_key(&(*r * CLIFFORD_TABLE_T[ci].1))).or_insert(ri);
            }
        }

        for &(i, _odd, sol, d) in winners.iter().take(4) {
            let w = &plain[i];
            let Some(&ri) = rep_of.get(&canonical_key(w)) else {
                eprintln!("plain[{i}]: NO REP FOUND (coverage hole!)");
                continue;
            };
            let r = &coset[ri];
            // which c maps rep -> winner? r·c ≡ w (up to phase).
            let c_idx = CLIFFORD_LDE0_IDX.iter().copied().find(|&ci| {
                canonical_key(&(*r * CLIFFORD_TABLE_T[ci].1)) == canonical_key(w)
            });
            eprintln!(
                "plain[{i}] (dist {d:.3e}) -> rep coset[{ri}] via c={:?} (rep==winner: {})",
                c_idx.map(|ci| CLIFFORD_TABLE_T[ci].0),
                ri_eq(r, w),
            );
            // 3) rerun the rep with a deep candidate budget.
            let rsols = run_prefix(r, 4096);
            let n_close = rsols.iter().filter(|&&(_, _, d)| d < eps).count();
            eprintln!(
                "  rep sols={} eps-close={} dists(first 6)={:?}",
                rsols.len(),
                n_close,
                rsols.iter().take(6).map(|&(o, _, d)| (o, d)).collect::<Vec<_>>()
            );
            // image solution: x_img = c · x_w (matrix-vector in the ring).
            if let Some(ci) = c_idx {
                let c_mat = &CLIFFORD_TABLE_T[ci].1;
                // w ≈ r·c  ⇒  w·U(sol) = r·(c·U(sol)); image x = first col
                // of c·U(sol). Winner was ODD branch: total = r·img·T.
                let img_u2t = *c_mat * solution_to_u2t(&sol, k_inner);
                eprintln!(
                    "  image k={} (k_inner={k_inner}); in rep sols: {}",
                    img_u2t.k,
                    rsols.iter().any(|(_, s, _)| solution_to_u2t(s, k_inner).diamond_distance(&img_u2t) < 1e-9),
                );
                let img_total = *r * img_u2t * U2T::t();
                eprintln!(
                    "  dist(r·img·T, target) = {:.3e}",
                    diamond_distance_u2t_float(&img_total, &target)
                );
                // Geometry of x_img in the rep's ODD frame.
                let m_inner_r = u2t_dag_times_mat2(r, &target);
                let v_inner_r = mat_to_uv(&m_inner_r).expect("rep mat_to_uv");
                let v_odd_r = apply_t_dag_to_uv(v_inner_r);
                let y = uv_to_xy(v_odd_r, k_inner);
                // x_img integer coords: (u1, u2) coefficients of img_u2t.
                let gi = |z: &crate::rings::ZOmega| -> [f64; 4] {
                    use crate::rings::types::int_to_f64;
                    [
                        int_to_f64(z.a),
                        int_to_f64(z.b),
                        int_to_f64(z.c),
                        int_to_f64(z.d),
                    ]
                };
                let (i1, i2) = (gi(&img_u2t.u11), gi(&img_u2t.u21));
                let x_img: [f64; 8] =
                    [i1[0], i1[1], i1[2], i1[3], i2[0], i2[1], i2[2], i2[3]];
                let dot: f64 = (0..8).map(|j| y[j] * x_img[j]).sum();
                let norm_sq: f64 = x_img.iter().map(|v| v * v).sum();
                let thresh = (1.0 - eps * eps) * 2f64.powi(2 * k_inner as i32) / 4.0;
                eprintln!(
                    "  x_img: |x|^2/2^k = {:.6}  dot^2/thresh - 1 = {:+.6e}",
                    norm_sq / 2f64.powi(k_inner as i32),
                    dot * dot / thresh - 1.0,
                );
                // Q-norm of x_img from the rep's odd frame, evaluated in
                // MPFR at the scratch precision (an f64 eval of this form
                // is garbage: Q eigenvalues reach 1/Δ_y² ~ 1e14 at 1e-5 and
                // the form only stays O(1) through cancellation).
                use crate::synthesis::lattice::{q_metric::build_q_mpfr, scratch::IntScratch};
                use rug::Float as RFloat;
                let mut qs = IntScratch::new(eps);
                build_q_mpfr(&mut qs, &y, k_inner, eps);
                let prec = qs.q_mpfr[0][0].prec();
                let mut qn = RFloat::with_val(prec, 0.0);
                for a in 0..8 {
                    for b in 0..8 {
                        let da = RFloat::with_val(prec, x_img[a]) - &qs.c[a];
                        let db = RFloat::with_val(prec, x_img[b]) - &qs.c[b];
                        qn += da * db * &qs.q_mpfr[a][b];
                    }
                }
                eprintln!(
                    "  x_img Q-norm in rep odd frame = {:.6} (walk bound 1.51; probe bound 4.0)",
                    qn.to_f64()
                );
                // Call integer::phase1 directly to expose should_escalate
                // (mod.rs's wrapper silently drops it).
                {
                    use std::sync::atomic::AtomicBool;
                    let mut s2 = IntScratch::new(eps);
                    s2.reset_basis();
                    let hit = AtomicBool::new(false);
                    let out = crate::synthesis::lattice::integer::phase1(
                        &mut s2, &y, k_inner, eps, usize::MAX, u64::MAX,
                        50_000_000, &hit, None,
                    );
                    eprintln!(
                        "  rep odd frame direct phase1: sols={} should_escalate={} budget_hit={}",
                        out.solutions.len(),
                        out.should_escalate,
                        hit.load(std::sync::atomic::Ordering::Relaxed),
                    );
                }
                // SE-walk replay: reproduce phase1's setup, locate x_img's
                // z-path, and print the walker's own per-depth partials to
                // find which level excludes it.
                {
                    use crate::synthesis::lattice::{
                        cholesky_lu::{cholesky_f64_8, lu_solve_int_inplace},
                        lll::lll_l2_8,
                        q_metric::build_q_int,
                        se::{bilinear_b, euclidean_cholesky, reconstruct_x},
                    };
                    use rug::Assign;
                    let mut s3 = IntScratch::new(eps);
                    s3.reset_basis();
                    build_q_mpfr(&mut s3, &y, k_inner, eps);
                    build_q_int(&mut s3);
                    let lll_res = lll_l2_8(&mut s3);
                    eprintln!("  replay: lll={lll_res:?} scale_bits={}", s3.scale_bits);
                    let basis = s3.basis;
                    let chol_ok = cholesky_f64_8(&mut s3);
                    for i in 0..8 {
                        for j in 0..8 {
                            let v = basis[j][i] as f64;
                            s3.lu_a[i][j].assign(v);
                        }
                        let ci = s3.c[i].clone();
                        s3.lu_rhs[i].assign(&ci);
                    }
                    let lu_ok = lu_solve_int_inplace(&mut s3);
                    eprintln!("  replay: chol_ok={chol_ok} lu_ok={lu_ok}");
                    let z_c: [f64; 8] = std::array::from_fn(|i| s3.lu_x[i].to_f64());
                    // Solve B^T z = x_img EXACTLY (det ±1) with rug::Integer
                    // adjugate (an f64 solve fails here — basis dynamic range
                    // is huge; scale_bits=132). z = adj(A)·x / det(A).
                    use rug::Integer as RInt;
                    let aij = |i: usize, j: usize| RInt::from(basis[j][i]);
                    // det via cofactor expansion is fine at 8x8 with exact
                    // ints? Too slow (8!). Use fraction-free Bareiss.
                    let mut m: Vec<Vec<RInt>> = (0..8)
                        .map(|i| {
                            let mut row: Vec<RInt> =
                                (0..8).map(|j| aij(i, j)).collect();
                            row.push(RInt::from(x_img[i] as i64));
                            row
                        })
                        .collect();
                    let mut sign = 1i32;
                    let mut prev = RInt::from(1);
                    for col in 0..8 {
                        if m[col][col] == 0 {
                            let p = (col + 1..8).find(|&r1| m[r1][col] != 0).unwrap();
                            m.swap(col, p);
                            sign = -sign;
                        }
                        for r2 in (col + 1)..8 {
                            for cc in (col + 1)..9 {
                                let t1 = RInt::from(&m[col][col] * &m[r2][cc]);
                                let t2 = RInt::from(&m[r2][col] * &m[col][cc]);
                                let num = t1 - t2;
                                let (q, rem) = num.div_rem(prev.clone());
                                assert!(rem == 0, "Bareiss exact division failed");
                                m[r2][cc] = q;
                            }
                            m[r2][col] = RInt::from(0);
                        }
                        prev = m[col][col].clone();
                    }
                    // After Bareiss, m[7][7] = det·sign' and back-substitution
                    // on the triangular system is exact.
                    let det = RInt::from(&m[7][7] * sign);
                    eprintln!("  replay: det(B^T) = {det}");
                    let mut z_big: Vec<RInt> = vec![RInt::from(0); 8];
                    for r2 in (0..8).rev() {
                        let mut v = m[r2][8].clone();
                        for cc in (r2 + 1)..8 {
                            v -= RInt::from(&m[r2][cc] * &z_big[cc]);
                        }
                        let (q, rem) = v.div_rem(m[r2][r2].clone());
                        assert!(rem == 0, "back-substitution not integral at {r2}");
                        z_big[r2] = q;
                    }
                    let mut z_img = [0i64; 8];
                    for (zi, zb) in z_img.iter_mut().zip(z_big.iter()) {
                        *zi = zb.to_i64().expect("z fits i64");
                    }
                    let x_chk = reconstruct_x(&basis, &z_img);
                    let x_int: [i64; 8] = std::array::from_fn(|i| x_img[i] as i64);
                    eprintln!(
                        "  replay: z_img={z_img:?} reconstruct==x_img: {}  bilinear_b={}",
                        x_chk == x_int,
                        bilinear_b(&x_int)
                    );
                    // Walker partials: R = l_f64^T (Q-metric), per depth d:
                    // partial_d = sum_{i>=d} (sum_{j>=i} R[i][j] (z[j]-z_c[j]))^2.
                    let mut rq = [[0.0f64; 8]; 8];
                    for i in 0..8 {
                        for j in 0..8 {
                            rq[i][j] = s3.l_f64[j][i];
                        }
                    }
                    let mut pq = [0.0f64; 9]; // pq[d] = partial entering depth d-1
                    for d in (0..8).rev() {
                        let mut lvl = 0.0;
                        for j in d..8 {
                            lvl += rq[d][j] * (z_img[j] as f64 - z_c[j]);
                        }
                        pq[d] = pq[d + 1] + lvl * lvl;
                    }
                    eprintln!("  replay: Q-partials by depth (7..0): {:?}",
                        (0..8).rev().map(|d| (d, pq[d])).collect::<Vec<_>>());
                    if let Some(re) = euclidean_cholesky(&basis) {
                        let mut pe = [0.0f64; 9];
                        for d in (0..8).rev() {
                            let mut lvl = 0.0;
                            for j in d..8 {
                                lvl += re[d][j] * z_img[j] as f64;
                            }
                            pe[d] = pe[d + 1] + lvl * lvl;
                        }
                        let tgt = 2f64.powi(k_inner as i32);
                        eprintln!(
                            "  replay: eucl partials/2^k by depth (7..0): {:?} (cut if > 1 + 2^-k)",
                            (0..8).rev().map(|d| (d, pe[d] / tgt)).collect::<Vec<_>>()
                        );
                    } else {
                        eprintln!("  replay: euclidean_cholesky FAILED (prune disabled)");
                    }
                }
            }
        }

        fn ri_eq(a: &U2T, b: &U2T) -> bool {
            canonical_key(a) == canonical_key(b)
        }
    }

    /// M1 census probe (stage 0 of docs/plan_8d_prefix_rework.md):
    /// |L_{t'}| with vs without right-coset dedup, t' = 1..13. Lever B1
    /// predicts 4.5-8×; kill if < 2×.
    /// Run: `cargo test --release --lib l_coset_census -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn l_coset_census() {
        eprintln!("\nM1 census: build_l_reference plain phase-dedup vs right-coset dedup");
        eprintln!("  t'   |L| plain   |L| coset   ratio");
        for tp in 1..=13u32 {
            let t0 = std::time::Instant::now();
            let plain = build_l_inner_with(tp, false).len();
            let t_plain = t0.elapsed().as_secs_f64() * 1000.0;
            let t0 = std::time::Instant::now();
            let coset = build_l_inner_with(tp, true).len();
            let t_coset = t0.elapsed().as_secs_f64() * 1000.0;
            eprintln!(
                "  {tp:>2}   {plain:>9}   {coset:>9}   {:>5.2}x   (build {t_plain:.0} / {t_coset:.0} ms)",
                plain as f64 / coset as f64
            );
        }
    }

    /// Test that DC (Algorithm 3.11) fires and finds a solution.
    /// Uses a tight eps where direct_search would hang but DC with MA prefixes
    /// and LLL/CVP inner search should terminate quickly.
    #[test]
    fn test_dc_fires_and_finds_solution() {
        // eps=0.01, Rz(0.3): DC fires at t>=17.  We go straight to t=20 to
        // ensure dc_search is exercised (t'=4, t_inner=16, |L|~16).
        let target = rz(0.3);
        let eps = 0.01_f64;
        let synth = SynthesizerT::new(eps).with_max_lde(35);
        assert!(optimal_t_prime(20, eps) > 0, "DC should fire at t=20 for eps=0.01");
        let result = synth.synthesize(target).expect("Should find a solution");
        check_result(&result, &target, eps);
        // Verify that a solution was found via DC (lde > direct_limit)
        //println!("lde={}, dist={:.4e}", result.lde, result.distance);
    }

    /// Test that optimal_t_prime gives correct thresholds (Proposition 3.13).
    #[test]
    fn test_optimal_t_prime_thresholds() {
        // ε=0.1: threshold ≈ 8.3, so t'=0 for t<=8, t'>=1 for t>=9.
        assert_eq!(optimal_t_prime(8, 0.1), 0);
        assert!(optimal_t_prime(9, 0.1) >= 1);
        // ε=0.01: threshold ≈ 16.6, so t'=0 for t<=16, t'>=1 for t>=17.
        assert_eq!(optimal_t_prime(16, 0.01), 0);
        assert!(optimal_t_prime(17, 0.01) >= 1);
        // t_inner = t - t' should satisfy: t_inner <= threshold (i.e. t' >= t - threshold).
        for &eps in &[0.1_f64, 0.01, 0.001] {
            for t in 0u32..30 {
                let tp = optimal_t_prime(t, eps);
                let t_inner = t - tp;
                let threshold = (5.0 / 2.0) * (1.0 / eps).log2();
                // t_inner should be <= threshold (direct_search is cheap enough).
                assert!(
                    t_inner as Float <= threshold + 1.0,
                    "t={t}, eps={eps}: t_inner={t_inner} > threshold={threshold:.1}"
                );
            }
        }
    }

    /// Test that DC is never triggered at t=0 (no prefix possible).
    #[test]
    fn test_dc_not_triggered_at_t0() {
        for &eps in &[0.1_f64, 0.01, 0.001] {
            assert_eq!(optimal_t_prime(0, eps), 0, "t'=0 always for t=0");
        }
    }

    /// Synthesize a Haar-random SU(2) unitary at ε=1e-3. Exercises the
    /// dc_search path on a non-trivial target (not just Rz/Ry); the named
    /// tests above mostly cover axis-aligned rotations.
    #[test]
    fn test_synthesize_random_unitary() {
        use rand::{SeedableRng, rngs::StdRng, Rng};

        let mut rng = StdRng::seed_from_u64(42);
        let eps = 0.001_f64;

        let theta: Float = rng.random::<Float>() * (2.0 * std::f64::consts::PI);
        let phi: Float = rng.random::<Float>() * (2.0 * std::f64::consts::PI);
        let lambda: Float = rng.random::<Float>() * (2.0 * std::f64::consts::PI);
        
        let ct = (theta / 2.0).cos();
        let st = (theta / 2.0).sin();

        // U3(θ,φ,λ) has det = e^{i(φ+λ)}, which is SU(2) only if φ+λ=0.
        // Normalize to SU(2) by multiplying by e^{-i(φ+λ)/2}.
        let global_phase = Complex::from_polar(1.0, -(phi + lambda) / 2.0);
        let target: Mat2 = [
            [global_phase * Complex::new(ct, 0.0), global_phase * (-Complex::from_polar(st, lambda))],
            [global_phase * Complex::from_polar(st, phi), global_phase * Complex::from_polar(ct, phi + lambda)],
        ];
        println!("Target unitary:\n{:?}", target);

        let synth = SynthesizerT::new(eps);
        let result = synth.synthesize(target).expect("Should synthesize random unitary");
        println!("Random unitary synthesis result: gates={:?}, lde={}, distance={:.6e}",
            result.gates, result.lde, result.distance);
        assert!(result.distance < eps,
            "distance={:.6e} >= epsilon={:.6e}", result.distance, eps);
    }

    /// Telemetry (ignored): geometric Q-norm² distribution of ε-close 8D
    /// solutions, the Z[ω] mirror of the 16D `q_telemetry_sweep` that
    /// found the ζ₁₆ band [0.875, 1.25] and dropped that bound 8 → 1.5.
    /// The 8D SE bound is the empirical 1.51 (lattice/integer.rs); this
    /// measures where ε-close solutions actually sit, from the TRUE cap
    /// center (the 8D walk already uses a fractional center, so measured
    /// ≈ geometric — no rounding-inflation step needed). If the max pins
    /// well below 1.51, a tightened bound buys (1.51/max)⁴ fewer nodes.
    ///
    /// 2026-06-11: collects ALL in-region solutions per level
    /// (`max_solutions = usize::MAX`) — the earlier first-hit numbers
    /// ([0.75, 0.94]) were maximally center-biased because phase1 stopped
    /// at the first hit of a distance-ordered walk. Walks are bounded by
    /// the new node budget (T8_NODES, default 50M per branch walk), which
    /// is what makes ε=1e-3 runnable at all (empty/slow branches used to
    /// walk unbudgeted for tens of minutes). Optionally widen the walk
    /// region via CYCLOSYNTH_SE_BOUND_8D (e.g. 2.5) to check for
    /// solutions ABOVE the production bound.
    /// Run: `cargo test --release --lib q_telemetry_sweep_8d -- --ignored --nocapture`
    /// Env: T8_EPS (default sweeps 3e-2 and 1e-3), T8_BUDGET (default 20M),
    /// T8_NODES (default 50M).
    #[test]
    #[ignore]
    fn q_telemetry_sweep_8d() {
        use crate::synthesis::lattice::{integer::phase1, q_metric::build_q_mpfr};
        use crate::synthesis::lattice::scratch::IntScratch;
        use std::sync::atomic::AtomicBool;

        let budget: u64 = std::env::var("T8_BUDGET").ok()
            .and_then(|s| s.parse().ok()).unwrap_or(20_000_000);
        let nodes: u64 = std::env::var("T8_NODES").ok()
            .and_then(|s| s.parse().ok()).unwrap_or(50_000_000);
        let mut global_max_close = 0.0f64;
        let mut global_min_close = f64::INFINITY;
        let mut total_close = 0usize;

        // t (lde) scan ranges per ε. CAUTION (learned the 46-minute way,
        // twice): `max_phase2_calls` caps CANDIDATE COMPLETIONS, not raw
        // nodes — on a no-solution level almost nothing reaches
        // candidacy, so the walk runs effectively unbudgeted and a
        // single below-first-hit level burns tens of minutes on one
        // core. Per-θ first-hit levels can't be reliably guessed, so
        // scan DOWNWARD from t_hi: every level at-or-above first-hit is
        // solution-dense and returns fast, and the two-level early-stop
        // fires before the scan can descend into empty territory.
        // Optional deep entry (T8_DEEP=1): ε=1e-5, scanned down from
        // t_hi=46 (typical first-hit lde ≈ 40-44 across these θ).
        let deep = std::env::var("T8_DEEP").as_deref() == Ok("1");
        let mut grid: Vec<(f64, u32, u32)> =
            vec![(3e-2f64, 8u32, 14u32), (1e-3, 27, 34)];
        if deep {
            grid.push((1e-5, 38, 46));
        }
        for &theta in &[0.3f64, 0.55, 0.8, 1.05, 1.3] {
            let target = rz(theta);
            let raw_uv = unitary_to_uv(&target);
            let v = normalize4(raw_uv).unwrap_or([1.0, 0.0, 0.0, 0.0]);
            for &(eps, t_lo, t_hi) in &grid {
                let mut levels_with_sols = 0;
                'levels: for t in (t_lo..=t_hi).rev() {
                    if levels_with_sols >= 2 {
                        break;
                    }
                    // Probe the PRODUCTION geometry for this level. With
                    // t' = optimal_t_prime == 0 that's the three direct
                    // branches at k = t; with t' > 0 it's the MA-prefix
                    // inner frames at k_inner — the frames whose walks the
                    // 1.51 bound actually governs. (Direct full-lde
                    // probing at ε ≤ 1e-3 is hopeless: the t=27..34 region
                    // is so large that a 50M-node budgeted walk finds
                    // nothing — the pre-fix 46-minute deadlock geometry.)
                    // Q/c are built in each frame's own (y, k); phase1
                    // sols have already passed the alignment-cap leaf
                    // check, which is exactly the in-cap criterion the
                    // bound governs.
                    let t_level = std::time::Instant::now();
                    let t_prime = optimal_t_prime(t, eps);
                    let mut frames: Vec<([Float; 4], u32)> = Vec::new();
                    if t_prime == 0 || t_prime > t {
                        for v_s in [v, apply_t_dag_to_uv(v), apply_t_to_uv(v)] {
                            frames.push((v_s, t));
                        }
                    } else {
                        let t_inner = t - t_prime;
                        let k_inner: u32 = if t_inner % 2 == 1 {
                            (t_inner - 1) / 2 + 1
                        } else {
                            t_inner / 2 + 1
                        };
                        let target_parity = det_zeta_parity(&target);
                        for u_l in build_l_reference(t_prime).iter() {
                            if frames.len() >= 64 {
                                break;
                            }
                            if let Some(tp) = target_parity {
                                if det_zeta_parity(&u_l.to_float()) != Some(tp) {
                                    continue;
                                }
                            }
                            let m_inner = u2t_dag_times_mat2(u_l, &target);
                            let Some(v_inner) = mat_to_uv(&m_inner) else { continue };
                            frames.push((v_inner, k_inner));
                            if t_inner > 0 {
                                frames.push((apply_t_dag_to_uv(v_inner), k_inner));
                            }
                        }
                    }

                    let mut min_close = f64::INFINITY;
                    let mut max_close = 0.0f64;
                    let mut n_close = 0usize;
                    let mut sol_frames = 0usize;
                    let mut any_trunc = false;
                    let mut k_probed = t;
                    let mut breaker = false;
                    for &(v_s, k_f) in &frames {
                        // Sample at most 6 solution-bearing frames per level.
                        if sol_frames >= 6 {
                            break;
                        }
                        k_probed = k_f;
                        let y = uv_to_xy(v_s, k_f);
                        let mut s = IntScratch::new(eps);
                        let hit = AtomicBool::new(false);
                        let out = phase1(
                            &mut s, &y, k_f, eps, usize::MAX, budget, nodes, &hit, None,
                        );
                        // Circuit breaker: this level is expensive territory.
                        if t_level.elapsed().as_secs() > 60 {
                            breaker = true;
                            break;
                        }
                        if out.solutions.is_empty() {
                            continue;
                        }
                        sol_frames += 1;
                        any_trunc |= hit.load(std::sync::atomic::Ordering::Relaxed);
                        // Fresh scratch for Q + cap center in THIS frame:
                        // phase1's LLL may have mutated downstream state;
                        // build_q alone is cheap and sets q_mpfr and c.
                        let mut qs = IntScratch::new(eps);
                        build_q_mpfr(&mut qs, &y, k_f, eps);
                        let q: [[f64; 8]; 8] = std::array::from_fn(|i| {
                            std::array::from_fn(|j| qs.q_mpfr[i][j].to_f64())
                        });
                        let c: [f64; 8] = std::array::from_fn(|i| qs.c[i].to_f64());
                        for sol in &out.solutions {
                            let dvec: [f64; 8] =
                                std::array::from_fn(|i| sol[i] as f64 - c[i]);
                            let mut qn = 0.0;
                            for i in 0..8 {
                                for j in 0..8 {
                                    qn += dvec[i] * q[i][j] * dvec[j];
                                }
                            }
                            max_close = max_close.max(qn);
                            min_close = min_close.min(qn);
                            n_close += 1;
                        }
                    }
                    if n_close > 0 {
                        levels_with_sols += 1;
                        eprintln!(
                            "θ={theta:<4} ε={eps:.0e} t={t:<2} k={k_probed:<2} frames={sol_frames} close={n_close:<5} Q∈[{min_close:.4}, {max_close:.4}]{}",
                            if any_trunc { "  (TRUNCATED walk)" } else { "" }
                        );
                        global_max_close = global_max_close.max(max_close);
                        global_min_close = global_min_close.min(min_close);
                        total_close += n_close;
                    } else if breaker {
                        break 'levels;
                    }
                }
            }
        }
        eprintln!(
            "GLOBAL 8D: eps-close sols={total_close}  Q∈[{global_min_close:.4}, {global_max_close:.4}]  (walk bound: 1.51)"
        );
        assert!(total_close > 0, "telemetry collected no eps-close solutions");
    }

    /// Telemetry (ignored): W0-style yardstick for ONE 8D level walk —
    /// wall, CPU utilization (process cpu-time / wall), solutions. The
    /// 16D version of this measurement (util 1.08× on 14 threads)
    /// motivated the W1 flat-frontier parallelization (~10×). Whether
    /// the port pays here depends on the T-baseline's wall at deep ε.
    /// Run: `cargo test --release --lib w1_telemetry_8d -- --ignored --nocapture`
    /// Env: T8_THETA (0.7), T8_EPS (1e-3), T8_LDE (30), T8_BUDGET (500M),
    /// T8_NODES (node budget, default 200M ≈ a few minutes single-core —
    /// a full-lde frame at ε=1e-3 is the 46-minute-runaway geometry, so
    /// an unbounded default is a footgun; raise it explicitly for a pure
    /// yardstick). 2026-06-11: with the node budget landed, an EMPTY
    /// level is also safe to measure — that's the budgeted-empty
    /// yardstick configuration.
    #[test]
    #[ignore]
    fn w1_telemetry_8d() {
        use std::sync::atomic::AtomicBool;

        fn envf(name: &str, default: f64) -> f64 {
            std::env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
        }
        let theta = envf("T8_THETA", 0.7);
        let eps = envf("T8_EPS", 1e-3);
        let t = envf("T8_LDE", 30.0) as u32;
        let budget = envf("T8_BUDGET", 500_000_000.0) as u64;
        let nodes: u64 = std::env::var("T8_NODES").ok()
            .and_then(|s| s.parse().ok()).unwrap_or(200_000_000);

        // CLOCK_PROCESS_CPUTIME_ID = 12 on macOS (same constant the 16D
        // w1_walk_bench uses).
        #[repr(C)]
        struct Timespec { tv_sec: i64, tv_nsec: i64 }
        extern "C" {
            fn clock_gettime(clk_id: i32, tp: *mut Timespec) -> i32;
        }
        fn cpu_time_s() -> f64 {
            let mut ts = Timespec { tv_sec: 0, tv_nsec: 0 };
            let rc = unsafe { clock_gettime(12, &mut ts) };
            if rc != 0 { return f64::NAN; }
            ts.tv_sec as f64 + ts.tv_nsec as f64 * 1e-9
        }

        let target = rz(theta);
        let raw_uv = unitary_to_uv(&target);
        let v = normalize4(raw_uv).unwrap_or([1.0, 0.0, 0.0, 0.0]);
        let mut scratch = crate::synthesis::lattice::scratch::IntScratch::new(eps);
        let hit = AtomicBool::new(false);

        crate::synthesis::diag::N_SE_NODES
            .store(0, std::sync::atomic::Ordering::Relaxed);
        let cpu0 = cpu_time_s();
        let t0 = std::time::Instant::now();
        let sols = lll_aligned_search(
            &mut scratch, v, t, eps, usize::MAX, budget, nodes, &hit, None,
        );
        let wall = t0.elapsed().as_secs_f64();
        let cpu = cpu_time_s() - cpu0;
        let n_nodes = crate::synthesis::diag::N_SE_NODES
            .load(std::sync::atomic::Ordering::Relaxed);

        let n_close = sols.iter().filter(|sol| {
            diamond_distance_float(&solution_to_u2t(sol, t).to_float(), &target) <= eps
        }).count();
        eprintln!(
            "8D walk: rz({theta}) ε={eps:e} t={t} | wall {wall:.3} s | cpu-util {:.2}x | nodes {n_nodes} ({:.2} Mnode/s) | sols {} (eps-close {n_close}) | budget_hit={}",
            cpu / wall.max(1e-9),
            n_nodes as f64 / wall.max(1e-9) / 1e6,
            sols.len(),
            hit.load(std::sync::atomic::Ordering::Relaxed),
        );
    }

    /// Stage-4 warm-LLL gate experiment (docs/plan_8d_prefix_rework.md
    /// lever C): on a captured set of production prefixes (bench
    /// target_00, found/empty levels at 1e-7 and 1e-8), compare
    /// `lll_l2_8` iteration counts between the identity start and a seed
    /// = the LLL-reduced basis of the prefix-independent Q_base(k, ε).
    /// Adoption gate: ≥25% total iteration reduction, else kill (16D
    /// precedent).
    /// Env: WARM_N (prefix cap per level, default 400).
    /// Run: `cargo test --release --lib warm_lll_gate -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn warm_lll_gate() {
        use crate::synthesis::lattice::lll::{lll_l2_8_seeded, LllResult};
        use crate::synthesis::lattice::q_metric::{build_q_int, build_q_mpfr};
        use crate::synthesis::lattice::scratch::IntScratch;
        use rug::Assign;

        fn xorshift64(s: &mut u64) -> u64 {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
            *s
        }
        fn rand_angle(s: &mut u64) -> f64 {
            let b = xorshift64(s) >> 11;
            (b as f64) / ((1u64 << 53) as f64) * 2.0 * PI
        }
        let mut state: u64 = 0xC0FFEE_BAADD0E_u64 | 1;
        let a = rand_angle(&mut state);
        let b = rand_angle(&mut state);
        let c = rand_angle(&mut state);
        let target = u3(a, b, c);
        let n_cap: usize = std::env::var("WARM_N")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(400);

        // (eps, t): found levels 66@1e-7 / 76@1e-8 plus the expensive
        // empty level 74@1e-8 (M0-refresh structure, fixed pipeline).
        for &(eps, t) in &[(1e-7f64, 66u32), (1e-8, 74), (1e-8, 76)] {
            let t_prime = optimal_t_prime(t, eps);
            let t_inner = t - t_prime;
            let k_inner: u32 = if t_inner % 2 == 1 {
                (t_inner - 1) / 2 + 1
            } else {
                t_inner / 2 + 1
            };
            let prefixes = build_l(t_prime, coset_mode_for(eps));
            let target_parity = det_zeta_parity(&target);

            // Capture surviving prefixes' y vectors (both inner branches,
            // like production).
            let mut ys: Vec<[Float; 8]> = Vec::new();
            for u_l in prefixes.iter() {
                if ys.len() >= n_cap {
                    break;
                }
                if let Some(tp) = target_parity {
                    if det_zeta_parity(&u_l.to_float()) != Some(tp) {
                        continue;
                    }
                }
                let m_inner = u2t_dag_times_mat2(u_l, &target);
                let Some(v_inner) = mat_to_uv(&m_inner) else { continue };
                ys.push(uv_to_xy(v_inner, k_inner));
                if t_inner > 0 && ys.len() < n_cap {
                    ys.push(uv_to_xy(apply_t_dag_to_uv(v_inner), k_inner));
                }
            }
            if ys.is_empty() {
                eprintln!("eps={eps:e} t={t}: no surviving prefixes (parity-dead level), skipping");
                continue;
            }

            let mut s = IntScratch::new(eps);
            // Warm seed: LLL-reduce Q_base itself. Populate q_base via one
            // build_q_mpfr call, copy it into q_mpfr, snapshot, reduce.
            build_q_mpfr(&mut s, &ys[0], k_inner, eps);
            for i in 0..8 {
                for j in 0..8 {
                    s.q_mpfr[i][j].assign(&s.q_base[i][j]);
                }
            }
            build_q_int(&mut s);
            let (res_base, it_base) = lll_l2_8_seeded(&mut s, None);
            let warm = s.basis;
            eprintln!(
                "eps={eps:e} t={t} t'={t_prime} k_inner={k_inner} captured={} \
                 | q_base LLL: {res_base:?} iters={it_base}",
                ys.len()
            );

            let (mut tot_cold, mut tot_warm) = (0u64, 0u64);
            let mut nonconv = 0usize;
            for y in &ys {
                build_q_mpfr(&mut s, y, k_inner, eps);
                build_q_int(&mut s);
                let (rc, ic) = lll_l2_8_seeded(&mut s, None);
                let (rw, iw) = lll_l2_8_seeded(&mut s, Some(&warm));
                if !matches!(rc, LllResult::Converged)
                    || !matches!(rw, LllResult::Converged)
                {
                    nonconv += 1;
                }
                tot_cold += ic as u64;
                tot_warm += iw as u64;
            }
            eprintln!(
                "  cold_iters={tot_cold} warm_iters={tot_warm} \
                 warm/cold={:.3} (gate: <=0.75) nonconverged={nonconv}",
                tot_warm as f64 / tot_cold.max(1) as f64
            );
        }
    }

}
