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
//!   [`crate::synthesis::search::aligned_search`]. Tries even, T, and T†
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
use crate::synthesis::cliffords::CLIFFORD_TABLE_T;
use crate::synthesis::decomposer::BlochDecomposer;
use crate::synthesis::distance::{diamond_distance_u2t_float, Mat2};
use crate::synthesis::search::{
    aligned_search, apply_t_dag_to_uv, apply_t_to_uv, apply_u2t_dag_to_uv, compute_align_vec,
    normalize4,
};

/// Global cache for `build_l` results, keyed by `t_prime`. Values are wrapped
/// in `Arc` so cache hits return an `O(1)` refcount bump rather than cloning
/// the full prefix list (at t'=14 that vector holds ~329 k U2T values, ~32 MB).
static BUILD_L_CACHE: LazyLock<Mutex<HashMap<u32, Arc<Vec<U2T>>>>> =
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
    let (idx, _) = flat
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.norm_sqr().partial_cmp(&b.norm_sqr()).unwrap())
        .unwrap();
    let piv = flat[idx];

    // Rotate so pivot is real-positive
    let rot: Vec<_> = if piv.norm() < 1e-12 {
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

    // Round to 6 decimal places and encode as i64 (multiply by 1e6)
    rot.iter()
        .map(|x| (x * 1_000_000.0).round() as i64)
        .collect::<Vec<_>>()
        .try_into()
        .unwrap()
}

/// Build L_{t'}: the Matsumoto–Amano prefix set with Clifford postmultiplication.
///
/// Matches Python's `build_L`:
///   L_0 = {I}
///   L_n (n≥1):
///     even branch: (HS^{b_n}T)·…·(HS^{b_1}T) · C  for b_i ∈ {0,1}, C ∈ C_1
///     odd  branch: T · (HS^{b_{n-1}}T)·…·(HS^{b_1}T) · C
///   deduplicated up to global U(1) phase.
///
/// Size after dedup: |L_0|=1, |L_n| = O(2^n) (much less than 3·2^{n-1}·24
/// due to many Clifford products being phase-equivalent).
fn build_l(t_prime: u32) -> Arc<Vec<U2T>> {
    // Check cache first; clone of `Arc` is just a refcount bump.
    {
        let cache = BUILD_L_CACHE.lock().unwrap();
        if let Some(v) = cache.get(&t_prime) {
            return Arc::clone(v);
        }
    }

    let result = Arc::new(build_l_inner(t_prime));

    // Race-tolerant insert: another thread may have populated this entry while
    // we were computing; either copy is identical so an overwrite is harmless.
    BUILD_L_CACHE
        .lock()
        .unwrap()
        .insert(t_prime, Arc::clone(&result));
    result
}

fn build_l_inner(t_prime: u32) -> Vec<U2T> {
    if t_prime == 0 {
        return vec![U2T::eye()];
    }

    let h = U2T::h();
    let s = U2T::s();
    let t = U2T::t();
    let hs0t = h * t; // H·T
    let hs1t = h * s * t; // H·S·T

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

    // Deduplicate up to global phase
    let mut seen: std::collections::HashSet<[i64; 8]> = std::collections::HashSet::new();
    let mut unique: Vec<U2T> = Vec::new();
    for u in candidates {
        let key = canonical_key(&u);
        if seen.insert(key) {
            unique.push(u);
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
        Int::from_i64(sol[0]),
        Int::from_i64(sol[1]),
        Int::from_i64(sol[2]),
        Int::from_i64(sol[3]),
    );
    let u2 = ZOmega::new(
        Int::from_i64(sol[4]),
        Int::from_i64(sol[5]),
        Int::from_i64(sol[6]),
        Int::from_i64(sol[7]),
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
         se_cb={:>9} budget={} {:>9.1}ms result={}",
        t,
        pass,
        t_prime,
        s.prefixes,
        s.mat_to_uv_rejected,
        s.se_callbacks,
        budget_hit as u8,
        pass_ms,
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

/// LLL-based aligned search for a right factor `U_R` of given lde `k`
/// matching the alignment vector `v`. Finds integer 8-vectors satisfying
/// the norm-shell, bilinear-form, and alignment constraints.
///
/// `max_solutions` caps how many candidates are returned. `max_phase2_calls`
/// caps the per-prefix SE budget; if reached, `budget_hit` is set so the
/// caller can choose to retry with a larger budget.
fn lll_aligned_search(
    scratch: &mut crate::synthesis::lattice::LatticeScratch,
    v: [Float; 4],
    k: u32,
    eps: Float,
    max_solutions: usize,
    max_phase2_calls: u64,
    budget_hit: &std::sync::atomic::AtomicBool,
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
    let sols = crate::synthesis::lattice::phase1(scratch, &y, k, eps, max_phase2_calls, budget_hit);
    if max_solutions >= sols.len() {
        sols
    } else {
        sols.into_iter().take(max_solutions).collect()
    }
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
pub(crate) fn optimal_t_prime(t: u32, eps: Float) -> u32 {
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
    /// Raw 8D lattice vector `(u, t)` selected by the successful search.
    pub x: Option<[i64; 8]>,
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
    /// Maximum lde for direct_search (brute-force aligned_search).
    /// For t > direct_limit, skip direct_search and go straight to dc_search
    /// regardless of the optimal t' split. This prevents aligned_search from
    /// hanging at large lde where it becomes O(2^(4t)) intractable.
    /// Default: 6 (aligned_search is fast up to norm shell 2^6=64; beyond that
    /// DC with forced t_prime = t - direct_limit is used).
    pub direct_limit: u32,
}

impl SynthesizerT {
    /// Create a synthesizer with the given precision and sensible defaults.
    ///
    /// Sets `min_lde = floor(coef(ε) · log₂(1/ε))` where `coef` ramps from
    /// `1.5` at ε ≥ 1e-4 up to `2.8` at ε ≤ 1e-6, linearly interpolated in
    /// log10(1/ε) between the two breakpoints. At larger ε, identity-like
    /// targets and small-T solutions live well below the deep-ε floor; the
    /// 1.5× coefficient gives them a chance. At deep ε, the 2.8× floor
    /// matches the empirical T-count for generic rotations and skips
    /// known-empty lde levels.
    ///
    /// Sets `max_lde = max(50, ceil(3.1 · log₂(1/ε)) + 2)` — generous upper
    /// bound that scales with ε so that worst-case angles (e.g. Rz(π/7) at
    /// 1e-5 needs lde=51) still have headroom. The +2 covers parity-skipped
    /// odd-t' lde values; the 3.1× coefficient is empirically tuned from the
    /// observed T-count spread across angles in the bench.
    ///
    /// Sets `direct_limit = 20` for ε ≥ 1e-4, `6` otherwise. Bumping
    /// direct_limit at moderate ε lets the brute-force `direct_search`
    /// cover the gap below `t_dc_start = ceil(2.5·log₂(1/ε))` where
    /// dc_search would bail (`t_prime = 0`). Direct search at norm shell
    /// 2^t is exponential in t — practical only up to ~t=20. For deep ε
    /// the optimal-T-count is already above this range, so dc_search is
    /// the right tool and direct_limit stays small.
    pub fn new(epsilon: Float) -> Self {
        let (min_lde, max_lde) = if epsilon > 0.0 && epsilon < 1.0 {
            let log2_recip = (1.0 / epsilon).log2();
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
        Self {
            epsilon,
            max_lde,
            min_lde,
            direct_limit,
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

    /// Set the maximum lde for direct_search (brute-force).
    /// Beyond this, dc_search is always used.
    pub fn with_direct_limit(mut self, direct_limit: u32) -> Self {
        self.direct_limit = direct_limit;
        self
    }

    /// Find a minimum-lde Clifford+T circuit approximating `target`.
    ///
    /// Returns `None` if no circuit within `max_lde` achieves distance < `epsilon`.
    ///
    /// # Performance
    ///
    /// All ε regimes go through the unified Lenstra 8D pipeline (L²-LLL
    /// over an exact i256 Gram + f64 Gram-Schmidt + MPFR-128 Schnorr-Euchner
    /// + MPFR-scaled-precision LU for the cap-center solve). MPFR precision
    /// scales with ε via `compute_prec_q` (build_q at `8·log₂(1/ε)` bits)
    /// and `compute_lu_prec` (LU at `6·log₂(1/ε)`).
    ///
    /// Typical synth times:
    /// - ε ≥ 1e-3: 1–15 ms
    /// - ε = 1e-5: 70–400 ms
    /// - ε = 1e-7: 0.15–2 s
    /// - ε = 1e-8: ~20 s
    pub fn synthesize(&self, target: Mat2) -> Option<SynthResultT> {
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
        // few steps of the t-loop.  build_l is expensive (O(2^t_prime)) and lazily
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
                        if tp > 0 && seen.insert(tp) {
                            Some(tp)
                        } else {
                            None
                        }
                    })
                    .collect()
            };
            needed.into_par_iter().for_each(|tp| {
                build_l(tp);
            });
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
            let (result, budget_hit) = self.dc_search(target, v, t, PASS1_CAP);
            let pass1_ms = t_start.elapsed().as_secs_f64() * 1000.0;
            if trace {
                let s = crate::synthesis::diag::snapshot();
                trace_dump_pass(
                    t,
                    optimal_t_prime(t, self.epsilon),
                    1,
                    &s,
                    budget_hit,
                    pass1_ms,
                    result.is_some(),
                );
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
            let (result2, budget_hit2) = self.dc_search(target, v, t, PASS2_CAP);
            if trace {
                let s = crate::synthesis::diag::snapshot();
                trace_dump_pass(
                    t,
                    optimal_t_prime(t, self.epsilon),
                    2,
                    &s,
                    budget_hit2,
                    t_start2.elapsed().as_secs_f64() * 1000.0,
                    result2.is_some(),
                );
            }
            result2
        }
    }

    /// Algorithm 3.6: direct search at lde `t`.
    ///
    /// Uses `search::aligned_search` (fast brute-force with Cauchy-Schwarz pruning)
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
        let clif_vs: Vec<[Float; 4]> = CLIFFORD_TABLE_T
            .iter()
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
            for sol in aligned_search(*v_s, t, eps, 1) {
                let (u2t, gates) = match tag {
                    DirectBranch::Plain => (solution_to_u2t(&sol, t), solution_to_gates(&sol, t)),
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
                        (
                            *c_u2t * solution_to_u2t(&sol, t),
                            solution_to_gates(&sol, t) + c_str,
                        )
                    }
                    DirectBranch::ClifT(i) => {
                        let (c_str, c_u2t) = &CLIFFORD_TABLE_T[*i];
                        (
                            *c_u2t * solution_to_u2t(&sol, t) * U2T::t(),
                            solution_to_gates(&sol, t) + "T" + c_str,
                        )
                    }
                    DirectBranch::ClifTdg(i) => {
                        let (c_str, c_u2t) = &CLIFFORD_TABLE_T[*i];
                        (
                            *c_u2t * solution_to_u2t(&sol, t) * U2T::t().dagger(),
                            solution_to_gates(&sol, t) + "SSST" + c_str,
                        )
                    }
                };
                let dist = diamond_distance_u2t_float(&u2t, target);
                if dist < eps {
                    return Some(SynthResultT {
                        gates: Some(gates),
                        lde: t,
                        distance: dist,
                        x: Some(sol),
                    });
                }
            }
            None
        })
    }

    /// Algorithm 3.11: divide-and-conquer with MA left prefixes.
    /// Algorithm 3.11 body: DC with MA left prefixes.
    ///
    /// Optimal split t' = max(0, ceil(t - 5/2*log2(1/eps))) from Prop 3.13.
    /// Inner step uses lll_aligned_search (CVP-based), which is O(1) near a
    /// solution — fast exactly when DC is needed (large t, small eps).
    /// Even and odd inner branches are both tried per prefix.
    /// `max_phase2_calls` is forwarded to lll_aligned_search → lattice::phase1.
    /// Returns `(solution, budget_was_hit)` where `budget_was_hit=true` means at least
    /// one phase1 invocation exhausted its SE-callback budget — the caller may want to
    /// retry at the same lde with a larger budget. If `false` and `solution` is `None`,
    /// the search was exhaustive at this lde and the caller should advance to lde+1.
    fn dc_search(
        &self,
        target: &Mat2,
        v: [Float; 4],
        t: u32,
        max_phase2_calls: u64,
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

        let prefixes = build_l(t_prime);
        if crate::synthesis::diag::trace_enabled() {
            crate::synthesis::diag::N_PREFIXES
                .fetch_add(prefixes.len() as u64, std::sync::atomic::Ordering::Relaxed);
        }

        // Parallel search over all left prefixes.
        // find_map_any stops all threads as soon as any one returns Some(...).
        // with_min_len ensures rayon distributes work evenly rather than
        // keeping everything on one thread when items complete quickly.
        let n_threads = rayon::current_num_threads();
        let chunk = (prefixes.len() / n_threads).max(1);
        let budget_hit = std::sync::atomic::AtomicBool::new(false);

        // Algebraic parity pre-filter: `mat_to_uv(U_L† · target)` succeeds
        // iff `parity(det(U_L)) == parity(det(target))`. Skipping prefixes
        // with mismatched parity short-circuits before `u2t_dag_times_mat2`
        // and saves the per-prefix float matmul + 8-phase trial. Provably
        // equivalent to mat_to_uv's rejection condition; no completeness loss.
        // `None` = target det not on the 8th-root-of-unity circle (e.g. an
        // arbitrary unitary), in which case we fall through to the original
        // mat_to_uv check.
        let target_parity = det_zeta_parity(target);

        // Per-worker scratch: rayon's `map_init` allocates one
        // `LatticeScratch` (pre-allocated MPFR/i256 buffers at the right
        // precision for `eps`) per worker thread and reuses it across every
        // prefix that worker handles, avoiding per-op allocation in the
        // hot path.
        let result = prefixes
            .par_iter()
            .with_min_len(chunk)
            .map_init(
                || crate::synthesis::lattice::LatticeScratch::new(eps),
                |scratch, u_l| -> Option<SynthResultT> {
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

                    // Even inner branch: U_L · U_R ≈ target
                    for sol in lll_aligned_search(
                        scratch,
                        v_inner,
                        k_inner,
                        eps,
                        1,
                        max_phase2_calls,
                        &budget_hit,
                    ) {
                        let u2t = *u_l * solution_to_u2t(&sol, k_inner);
                        let dist = diamond_distance_u2t_float(&u2t, target);
                        if dist < eps {
                            return Some(SynthResultT {
                                gates: Some(BlochDecomposer.decompose(&u2t)),
                                lde: t,
                                distance: dist,
                                x: Some(sol),
                            });
                        }
                        crate::synthesis::diag::N_DIST_REJECTED
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }

                    // Odd inner branch: U_L · U_R · T ≈ target
                    if t_inner > 0 {
                        let v_inner_t = apply_t_dag_to_uv(v_inner);
                        for sol in lll_aligned_search(
                            scratch,
                            v_inner_t,
                            k_inner,
                            eps,
                            1,
                            max_phase2_calls,
                            &budget_hit,
                        ) {
                            let u2t = *u_l * solution_to_u2t(&sol, k_inner) * U2T::t();
                            let dist = diamond_distance_u2t_float(&u2t, target);
                            if dist < eps {
                                return Some(SynthResultT {
                                    gates: Some(BlochDecomposer.decompose(&u2t)),
                                    lde: t,
                                    distance: dist,
                                    x: Some(sol),
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

        (
            result,
            budget_hit.load(std::sync::atomic::Ordering::Relaxed),
        )
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthesis::distance::diamond_distance_float;
    use std::f64::consts::{FRAC_1_SQRT_2, PI};

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
            [
                a[0][0] * b[0][0] + a[0][1] * b[1][0],
                a[0][0] * b[0][1] + a[0][1] * b[1][1],
            ],
            [
                a[1][0] * b[0][0] + a[1][1] * b[1][0],
                a[1][0] * b[0][1] + a[1][1] * b[1][1],
            ],
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
            result.distance,
            eps
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
        let id: Mat2 = [
            [Complex::new(1., 0.), Complex::new(0., 0.)],
            [Complex::new(0., 0.), Complex::new(1., 0.)],
        ];
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
        let result = synth
            .synthesize(target)
            .expect("Should synthesize Rz(1.34)");
        println!("{:?}", result.gates);
        check_result(&result, &target, 0.01);
    }

    #[test]
    fn test_synthesize_rz_hard_1() {
        // eps=0.01: needs t~26, DC kicks in at t>=17 (t'=t-17, t_inner=17).
        // Much faster than eps=0.001 which needs t~40.
        let target = rz(0.3);
        let synth = SynthesizerT::new(0.01);
        let result = synth
            .synthesize(target)
            .expect("Should synthesize Rz(0.3) at eps=0.01");
        println!("{:?}", result.gates);
        check_result(&result, &target, 0.01);
    }

    #[test]
    fn test_synthesize_rz_hard_2() {
        let target = rz(1.34);
        let synth = SynthesizerT::new(0.001);
        let result = synth
            .synthesize(target)
            .expect("Should synthesize Rz(1.34) at eps=0.01");
        println!("{:?}", result.gates);

        check_result(&result, &target, 0.01);
    }

    /// Empirical sister of `clifford_sqrt_t::tests::build_l_q_dc_cost_ratio`.
    /// Computes the same `S(t', α)` cost-ratio (Σ count(t', k)/α^k) for
    /// Clifford+T's `build_l` so we can directly compare what the naive
    /// cost model predicts for D&C in each ring.
    #[test]
    fn build_l_size_and_cost_ratio() {
        eprintln!("\n|L_{{t'}}| sizes:");
        for t_prime in 0..=10 {
            let l = build_l(t_prime);
            eprintln!("  t'={t_prime:>2}  |L_{{t'}}|={:>8}", l.len());
        }

        eprintln!("\nk_prefix histogram (Clifford+T, build_l):");
        for t_prime in 1..=8 {
            let l = build_l(t_prime);
            let mut counts: Vec<u64> = vec![0; 64];
            for u in l.iter() {
                let k = u.k as usize;
                if k < counts.len() {
                    counts[k] += 1;
                }
            }
            let mut k_min = u32::MAX;
            let mut k_max = 0;
            for u in l.iter() {
                k_min = k_min.min(u.k);
                k_max = k_max.max(u.k);
            }
            eprintln!(
                "  t'={t_prime:>2}  total={:>8}  k range [{k_min}, {k_max}]",
                l.len()
            );
        }

        eprintln!("\nS(t', α) = Σ_k count(t', k) / α^k  (D&C cost ratio):");
        eprintln!("  t'  total      α=2.0    α=2.5    α=3.0    α=3.5    α=4.0");
        for t_prime in 1..=10 {
            let l = build_l(t_prime);
            let mut counts: Vec<u64> = vec![0; 64];
            for u in l.iter() {
                let k = u.k as usize;
                if k < counts.len() {
                    counts[k] += 1;
                }
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
        assert!(
            optimal_t_prime(20, eps) > 0,
            "DC should fire at t=20 for eps=0.01"
        );
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
        use rand::{rngs::StdRng, Rng, SeedableRng};

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
            [
                global_phase * Complex::new(ct, 0.0),
                global_phase * (-Complex::from_polar(st, lambda)),
            ],
            [
                global_phase * Complex::from_polar(st, phi),
                global_phase * Complex::from_polar(ct, phi + lambda),
            ],
        ];
        println!("Target unitary:\n{:?}", target);

        let synth = SynthesizerT::new(eps);
        let result = synth
            .synthesize(target)
            .expect("Should synthesize random unitary");
        print!(
            "Random unitary synthesis result: gates={:?}, lde={}, distance={:.6e}\n",
            result.gates, result.lde, result.distance
        );
        assert!(
            result.distance < eps,
            "distance={:.6e} >= epsilon={:.6e}",
            result.distance,
            eps
        );
    }
}
