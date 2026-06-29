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
//! - `direct_search` (`t ≤ direct_limit`; 8 at ε ≥ 1e-4, else 6):
//!   brute-force enumeration over the norm shell `‖x‖² = 2^t` via
//!   [`crate::synthesis::lattice::omega::brute::brute_aligned_search`]. Tries even, T, and T†
//!   right-side branches, each combined with all 24 Clifford left
//!   prefixes. Fast for small `t`; exponential beyond that.
//!
//! - `prefix_split_search` (`t > direct_limit`, Algorithm 3.11): divide-and-
//!   conquer using Matsumoto–Amano left prefixes `L_{t'}`. Splits at
//!   `t' = max(0, ⌈t − 5/2·log₂(1/ε)⌉)` (returning `None` when that is 0).
//!   For each prefix
//!   `U_L ∈ L_{t'}`, searches for the right factor via
//!   `lll_aligned_search` at inner lde `lde_inner` (see below).
//!   Tries even (U_L·U_R) and odd (U_L·U_R·T) inner branches.
//!
//! # Inner-lde convention
//!
//! `lll_aligned_search` uses `lde_inner = T_inner/2 + 1` (norm shell
//! `2^lde_inner`), not the T-count itself:
//!
//!   lde_inner = t_inner / 2 + 1            (even t_inner)
//!   lde_inner = (t_inner - 1) / 2 + 1      (odd t_inner)

use num_complex::Complex;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex};

use crate::matrix::U2T;
use crate::rings::types::{Float, Int};
use crate::rings::ZOmega;
use crate::synthesis::cliffords::{CLIFFORD_LDE0_IDX, CLIFFORD_TABLE_T};
use crate::synthesis::decomposer::BlochDecomposer;
use crate::synthesis::distance::{diamond_distance_u2t_float, to_su2, Mat2};
use crate::rings::MpFloat;
use crate::synthesis::lattice::omega::brute::{
    brute_aligned_search, apply_t_dag_to_uv, apply_t_dag_to_uv_mpfr, apply_t_to_uv,
    apply_u2t_dag_to_uv, apply_u2t_dag_to_uv_mpfr, compute_align_vec, normalize4,
};
use crate::synthesis::lattice::omega::find_aligned_lattice_points_exact;
use crate::synthesis::lattice::omega::q_metric::uv_to_lattice_y_mpfr;

/// At ε ≤ this, the deep-ε MPFR alignment path replaces the f64 chain (the
/// f64 prefix residual and lattice y lose precision once the cap half-width
/// ε²/4 nears the f64 ULP). `CYCLOSYNTH_OMEGA_FORCE_EXACT=1` forces it on at
/// any ε (used to check exact-vs-f64 equivalence without deep-ε runs).
const OMEGA_EXACT_EPS: f64 = 2e-8;

fn omega_force_exact() -> bool {
    static F: LazyLock<bool> =
        LazyLock::new(|| std::env::var("CYCLOSYNTH_OMEGA_FORCE_EXACT").as_deref() == Ok("1"));
    *F
}

fn omega_use_exact(eps: f64) -> bool {
    eps <= OMEGA_EXACT_EPS || omega_force_exact()
}

/// `(t', coset_dedup)` → MA prefix list. `Arc`-wrapped so cache hits are a
/// refcount bump, not a clone of the full prefix list (~329 k U2T at t'=14).
type MaPrefixCache = LazyLock<Mutex<HashMap<(u32, bool), Arc<Vec<U2T>>>>>;
static MA_PREFIX_CACHE: MaPrefixCache = LazyLock::new(|| Mutex::new(HashMap::new()));

/// `uv = [Re(u1), Im(u1), Re(u2), Im(u2)]` from a 2×2 unitary, normalized
/// to SU(2) by dividing by √det so that e.g. diag(1, i) (det = i) maps to
/// the same search direction as its SU(2) representative.
/// Convention: `V ≈ e^{iφ} · [[u1, −ū2],[u2, ū1]]`.
pub fn unitary_to_uv(v: &Mat2) -> [Float; 4] {
    let det = v[0][0] * v[1][1] - v[0][1] * v[1][0];
    let phase = det.sqrt();
    if phase.norm() > 1e-12 {
        let u1 = v[0][0] / phase;
        let u2 = v[1][0] / phase;
        [u1.re, u1.im, u2.re, u2.im]
    } else {
        [v[0][0].re, v[0][0].im, v[1][0].re, v[1][0].im]
    }
}

/// `uv` of the SU(2) form `[[u1, −ū2], [u2, ū1]]`, found by trying the 8
/// global phases e^{ikπ/4} (the possible Clifford+T determinants). `None`
/// if no phase yields that form.
fn try_unitary_to_uv(u: &Mat2) -> Option<[Float; 4]> {
    use std::f64::consts::FRAC_PI_4;
    for k in 0..8 {
        let ph = Complex::from_polar(1.0, k as Float * FRAC_PI_4);
        let m00 = ph * u[0][0];
        let m01 = ph * u[0][1];
        let m10 = ph * u[1][0];
        let m11 = ph * u[1][1];
        // Need m11 == conj(m00) and m01 == -conj(m10).
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

/// Parity of det(m) as a ζ-power (ζ = e^{iπ/4}): `0` for even powers
/// ({±1, ±i}), `1` for odd, `None` if det isn't on the 8th-root circle.
/// Upstream filter for `prefix_split_search`: `try_unitary_to_uv` rejects
/// exactly when `parity(det(U_L)) ≠ parity(det(target))`, so this prunes
/// the same prefixes one float matmul earlier — no completeness loss.
fn det_zeta_parity(m: &Mat2) -> Option<u8> {
    let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    let mag_sq = det.norm_sqr();
    if (mag_sq - 1.0).abs() > 1e-3 {
        return None;
    }
    // Even ζ-powers have max(|re|, |im|) = 1; odd have √2/2 ≈ 0.707.
    let max_axis = det.re.abs().max(det.im.abs());
    if max_axis > 0.9 {
        Some(0)
    } else if max_axis > 0.6 && max_axis < 0.85 {
        Some(1)
    } else {
        None
    }
}

/// `U_L† · target` as a float matrix (`U_L` exact `U2T`, `target` float).
fn prefix_dag_times_target(u_l: &U2T, target: &Mat2) -> Mat2 {
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

/// Phase-invariant canonical key for dedup: rotate so the
/// largest-magnitude entry is real-positive, then round to 6 decimals.
fn canonical_key(u: &U2T) -> [i64; 8] {
    let m = u.to_float();
    let flat = [m[0][0], m[0][1], m[1][0], m[1][1]];

    let (idx, _) = flat.iter().enumerate()
        .max_by(|(_, a), (_, b)| a.norm_sqr().partial_cmp(&b.norm_sqr()).unwrap())
        .unwrap();
    let piv = flat[idx];

    let rot: Vec<_> = if piv.norm() < 1e-12 {
        flat.iter().flat_map(|c| [c.re, c.im]).collect()
    } else {
        let phase = piv / piv.norm();
        flat.iter().flat_map(|c| {
            let r = c / phase;
            [r.re, r.im]
        }).collect()
    };

    rot.iter().map(|x| (x * 1_000_000.0).round() as i64).collect::<Vec<_>>()
        .try_into().unwrap()
}

/// Right-coset dedup gate for `build_ma_prefix_set`. Tri-state:
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

/// Probe/test entry: plain phase-dedup unless the env forces coset mode.
/// Production goes through `build_ma_prefix_set` with `coset_mode_for(eps)`.
#[cfg(test)]
pub(crate) fn build_ma_prefix_set_reference(t_prime: u32) -> Arc<Vec<U2T>> {
    build_ma_prefix_set(t_prime, (*L_COSET_DEDUP).unwrap_or(false))
}

/// The Matsumoto–Amano prefix set L_{t'} with Clifford postmultiplication,
/// cached per `(t_prime, coset_dedup)`.
pub fn build_ma_prefix_set(t_prime: u32, coset_dedup: bool) -> Arc<Vec<U2T>> {
    let key = (t_prime, coset_dedup);
    {
        let cache = MA_PREFIX_CACHE.lock().unwrap();
        if let Some(v) = cache.get(&key) {
            return Arc::clone(v);
        }
    }
    let result = Arc::new(build_l_inner_with(t_prime, coset_dedup));
    // A racing thread may have inserted an identical copy; overwrite is harmless.
    MA_PREFIX_CACHE
        .lock()
        .unwrap()
        .insert(key, Arc::clone(&result));
    result
}

/// L_{t'} construction (cache + env bypassed so a probe can compare dedup
/// modes in one process). L_0 = {I}; otherwise an even branch of
/// (HS^b·T) products and an odd branch prefixed with T, each times every
/// Clifford, deduplicated up to global phase (and, in coset mode, up to
/// the right coset u·⟨S,X⟩).
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

    // Even branch: length-t' product of (HS^b·T) blocks, then · C.
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

    // Odd branch: T · length-(t'-1) product · C.
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

    // Coset mode inserts every orbit member u·c's key when a rep u is kept,
    // so later mates dedup with one key computation each (~2.3n keys vs 8n
    // for a min-over-orbit key).
    let mut seen: std::collections::HashSet<[i64; 8]> = std::collections::HashSet::new();
    let mut unique: Vec<U2T> = Vec::new();
    for u in candidates {
        let key = canonical_key(&u);
        if seen.contains(&key) {
            continue;
        }
        unique.push(u);
        if coset_dedup {
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

/// `U2T` from a lattice solution: `sol = [u1(0..4), u2(4..8)]` in ZOmega,
/// `U = [[u1, -ū2], [u2, ū1]] / √2^k`.
pub fn solution_to_u2t(sol: &[i64; 8], k: u32) -> U2T {
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
/// end of each `prefix_split_search` invocation when `CYCLOSYNTH_TRACE=1` is set.
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
        t, pass, t_prime, s.prefixes, s.uv_extract_rejected, s.se_callbacks,
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

// ─── LLL-based aligned search (used by prefix_split_search inner step) ─────────────────

/// Scale a 4-element alignment vector `v` to the 8-element y vector used by
/// the lattice pipeline. `y = compute_align_vec(v) · sqrt(2^k) / 2`,
/// satisfying `‖y‖² = 2^(k-1)`. Used `powf` (not bit-shift) so `k ≥ 64`
/// stays well-defined.
pub fn uv_to_lattice_y(v: [Float; 4], k: u32) -> [Float; 8] {
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

/// NODE budgets per `find_aligned_lattice_points` call: the leaf caps
/// never bind on a no-solution level (almost nothing reaches a leaf), so
/// without a node cap an empty level walks to exhaustion. Pass 1 sits far
/// above every completing walk (empty levels are expensive through prefix
/// COUNT, not any single walk); a walk that can't finish under the pass-2
/// cap is pathological exhaustion, skipped per the speed-over-completeness
/// rule.
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
/// first-hit walk). `max_leaf_checks` caps the per-prefix SE leaf budget
/// and `max_nodes` the per-prefix SE NODE budget; if either is reached,
/// `budget_hit` is set so the caller can retry with a larger budget.
/// `external_abort` is the cross-branch winner signal (checked at every SE
/// recurse-entry; does not set `budget_hit`).
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
fn lll_aligned_search(
    scratch: &mut crate::synthesis::lattice::omega::scratch::IntScratch,
    v: [Float; 4],
    v_mpfr: Option<&[MpFloat; 4]>,
    k: u32,
    eps: Float,
    max_solutions: usize,
    max_leaf_checks: u64,
    max_nodes: u64,
    budget_hit: &std::sync::atomic::AtomicBool,
    external_abort: Option<&std::sync::atomic::AtomicBool>,
) -> Vec<[i64; 8]> {
    // k ≤ 110 keeps everything in i128: target_norm = 2^k and the
    // Σ-products in bilinear_b reach ~k+3 bits ≈ 2^113 < 2^127.
    if max_solutions == 0 || k > 110 {
        return Vec::new();
    }
    // Lenstra-style 8D enumeration (Algorithm 3.6 of arXiv:2510.05816):
    // MPFR at adaptive precision for LLL+Cholesky, f64 for the SE step.
    // At deep ε an exact MPFR alignment vector keeps the cap center and SE
    // dot exact below the f64 ULP.
    if let Some(vm) = v_mpfr.filter(|_| omega_use_exact(eps)) {
        let y_q = uv_to_lattice_y_mpfr(vm, k, scratch.prec_q);
        return find_aligned_lattice_points_exact(
            scratch, &y_q, k, eps, max_solutions, max_leaf_checks, max_nodes,
            budget_hit, external_abort,
        );
    }
    let y = uv_to_lattice_y(v, k);
    crate::synthesis::lattice::omega::find_aligned_lattice_points(
        scratch, &y, k, eps, max_solutions, max_leaf_checks, max_nodes,
        budget_hit, external_abort,
    )
}

// ─── Optimal prefix-split point (Proposition 3.13) ──────────────────────────────────────────────────────────────

/// Optimal D&C split exponent (Proposition 3.13):
/// `t' = max(0, ⌈t − 5/2·log₂(1/ε)⌉)`. `t' > 0` (D&C beats direct) once
/// `t > 5/2·log₂(1/ε)`; `t' = 0` for ε ≥ 1.
fn optimal_t_prime(t: u32, eps: Float) -> u32 {
    if eps >= 1.0 {
        return 0;
    }
    let threshold = (5.0 / 2.0) * (1.0 / eps).log2();
    if t as Float <= threshold {
        0
    } else {
        (t as Float - threshold).ceil() as u32
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

/// Clifford+T synthesis backend (Algorithm 3.14 of arXiv:2510.05816).
/// Prefer the unified [`crate::synthesis::Synthesizer`]; public for tests.
pub struct SynthesizerT {
    /// Approximation precision in diamond distance.
    pub epsilon: Float,
    pub max_lde: u32,
    /// Defaults to floor(coef·log₂(1/ε)) with coef ramping 1.5 → 2.8 over
    /// ε (see [`Self::new`]); ~the information-theoretic T-count lower bound
    /// for a generic SU(2) rotation. Set 0 for exact low-T-count solutions
    /// of Cliffords and other special gates.
    pub min_lde: u32,
    /// Max lde for direct_search; above it, skip straight to
    /// prefix_split_search since brute_aligned_search becomes O(2^4ᵗ).
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

    /// Find a minimum-lde Clifford+T circuit approximating `target`, or
    /// `None` if none within `max_lde` reaches distance < `epsilon`. All ε
    /// route through the Lenstra 8D pipeline (L²-LLL over an exact i256
    /// Gram + f64 GS + MPFR-128 Schnorr-Euchner + MPFR LU for the
    /// cap-center), with MPFR precision scaled to ε.
    pub fn synthesize(&self, target: Mat2) -> Option<SynthResultT> {
        self.run(target, None)
    }

    /// Synthesize with a higher-precision target column `exact_col` (the
    /// √det-normalized first column of the SU(2) target, e.g. from exact
    /// rational-π angles). At deep ε the search aligns to this MPFR column
    /// instead of the f64 chain, reaching ε below the f64 ULP wall. `target`
    /// (f64) is still used for the diamond-distance acceptance check.
    pub fn synthesize_with_exact_col(
        &self,
        target: Mat2,
        exact_col: &[MpFloat; 4],
    ) -> Option<SynthResultT> {
        self.run(target, Some(exact_col))
    }

    fn run(&self, target: Mat2, exact_col: Option<&[MpFloat; 4]>) -> Option<SynthResultT> {
        // Project to SU(2): the search assumes det = 1 (see `to_su2`).
        let target = to_su2(&target);
        // 16 MiB worker stacks: the 8D path races the ζ₁₆ entries for
        // global-pool init, and a 2 MiB pool overflows later deep walks.
        crate::synthesis::ensure_rayon_stack();
        let raw_uv = unitary_to_uv(&target);
        let v = normalize4(raw_uv).unwrap_or([1.0, 0.0, 0.0, 0.0]);

        // Direct search starts at min_lde, not 0: no generic rotation
        // reaches ε with fewer T-gates.
        for t in self.min_lde..=self.direct_limit {
            let result = self.try_at_lde(&target, v, exact_col, t);
            if result.is_some() {
                return result;
            }
        }

        // Skip the gap where prefix_split_search exists but prefix lists are
        // tiny and the inner search is cheap anyway.
        let t_dc_start = if self.epsilon < 1.0 {
            let raw = (5.0 / 2.0) * (1.0 / self.epsilon).log2();
            (raw.ceil() as u32).max(self.direct_limit + 1)
        } else {
            self.direct_limit + 1
        };
        let t_dc_start = t_dc_start.max(self.min_lde);

        // Pre-warm the prefix-set cache in parallel for the t_prime values
        // the first few t-loop steps will need. `build_ma_prefix_set` is
        // O(2^t_prime) and lazily populated, so building here fills all
        // cores before the search loop. Cap at a 5-step horizon: solutions
        // almost always land within a few t of t_dc_start.
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
            needed.into_par_iter().for_each(|tp| { build_ma_prefix_set(tp, coset); });
        }

        for t in t_dc_start..=self.max_lde {
            let result = self.try_at_lde(&target, v, exact_col, t);
            if result.is_some() {
                return result;
            }
        }
        None
    }

    /// Find a solution at lde `t`: direct_search for `t ≤ direct_limit`,
    /// else prefix_split_search with an adaptive 2-pass cap — PASS1 bails
    /// unproductive prefixes fast, and PASS2's full budget runs only if
    /// pass 1 actually exhausted its budget (otherwise pass 1 was already
    /// exhaustive and no solution exists at this lde).
    fn try_at_lde(
        &self,
        target: &Mat2,
        v: [Float; 4],
        exact_col: Option<&[MpFloat; 4]>,
        t: u32,
    ) -> Option<SynthResultT> {
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
                self.prefix_split_search(target, v, exact_col, t, PASS1_CAP, PASS1_NODE_CAP);
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
                self.prefix_split_search(target, v, exact_col, t, PASS2_CAP, PASS2_NODE_CAP);
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

    /// Direct search at lde `t` (Algorithm 3.6): `brute_aligned_search`
    /// over 72 directions in parallel — the 3 top-level branches
    /// (U, U·T, U·T† ≈ target) plus 3 for each of the 23 non-identity
    /// left-Cliffords C. LLL+CVP (`lll_aligned_search`) is reserved for
    /// the D&C path where the inner lde is large.
    fn direct_search(&self, target: &Mat2, v: [Float; 4], t: u32) -> Option<SynthResultT> {
        let eps = self.epsilon;

        let clif_vs: Vec<[Float; 4]> = CLIFFORD_TABLE_T.iter()
            .map(|(_, c_u2t)| apply_u2t_dag_to_uv(c_u2t, v))
            .collect();

        // 3 top-level + 23 Cliffords × 3 (index 0 = "I", already covered).
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
    /// `max_leaf_checks` (SE leaf budget) and `max_nodes` (SE node budget) are
    /// forwarded to lll_aligned_search → lattice::find_aligned_lattice_points, per prefix × branch.
    /// Returns `(solution, budget_was_hit)` where `budget_was_hit=true` means at least
    /// one find_aligned_lattice_points invocation exhausted an SE budget — the caller may want to
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
    #[allow(clippy::too_many_arguments)]
    fn prefix_split_search(
        &self,
        target: &Mat2,
        v: [Float; 4],
        exact_col: Option<&[MpFloat; 4]>,
        t: u32,
        max_leaf_checks: u64,
        max_nodes: u64,
    ) -> (Option<SynthResultT>, bool) {
        let eps = self.epsilon;

        // t_prime is the optimal split from Prop 3.13. When it is 0 the
        // formula says no split is needed; if t is nonetheless past the
        // direct-search limit, return None so the outer loop advances to the
        // next (higher) t where the split is non-trivial. (Forcing a split
        // here would produce an exponentially large prefix set.)
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
        let lde_inner: u32 = if odd_inner {
            (t_inner - 1) / 2 + 1
        } else {
            t_inner / 2 + 1
        };

        let prefixes = build_ma_prefix_set(t_prime, coset_mode_for(eps));
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

        // build_ma_prefix_set order correlates position with structure, so
        // contiguous chunks concentrate similar prefixes on one worker;
        // dealing lowers time-to-first-hit under find_any.
        let indices: Vec<u32> = (0..n as u32).collect();
        let order = crate::synthesis::stride_interleave(&indices, n_threads);

        // Algebraic parity pre-filter: `try_unitary_to_uv(U_L† · target)` succeeds
        // iff `parity(det(U_L)) == parity(det(target))`. Skipping prefixes
        // with mismatched parity short-circuits before `prefix_dag_times_target`
        // and saves the per-prefix float matmul + 8-phase trial. Provably
        // equivalent to try_unitary_to_uv's rejection condition; no completeness loss.
        // `None` = target det not on the 8th-root-of-unity circle (e.g. an
        // arbitrary unitary), in which case we fall through to the original
        // try_unitary_to_uv check.
        let target_parity = det_zeta_parity(target);

        // Inner branches (`odd` flags) run per prefix: even (U_L·U_R) and,
        // when t_inner > 0, odd (U_L·U_R·T). Branch wins split ~50/50 with
        // no t_inner-parity rule, so neither branch order dominates and the
        // two are swept together.
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
                    || crate::synthesis::lattice::omega::scratch::IntScratch::new(eps),
                    |scratch, (pos, &pi)| -> Option<SynthResultT> {
                        let u_l = &prefixes[pi as usize];
                        if let Some(tp) = target_parity {
                            if det_zeta_parity(&u_l.to_float()) != Some(tp) {
                                if crate::synthesis::diag::trace_enabled() {
                                    crate::synthesis::diag::N_UV_EXTRACT_REJECTED
                                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                }
                                return None;
                            }
                        }
                        let m_inner = prefix_dag_times_target(u_l, target);
                        let v_inner = match try_unitary_to_uv(&m_inner) {
                            Some(v) => v,
                            None => {
                                if crate::synthesis::diag::trace_enabled() {
                                    crate::synthesis::diag::N_UV_EXTRACT_REJECTED
                                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                }
                                return None;
                            }
                        };
                        // Exact MPFR residual for the deep-ε alignment vector
                        // (same √det-normalized column as v_inner, kept exact).
                        let v_inner_mpfr: Option<[MpFloat; 4]> =
                            exact_col.map(|col| apply_u2t_dag_to_uv_mpfr(u_l, col, scratch.prec_q));

                        for &odd in plan {
                            // Even inner branch: U_L · U_R ≈ target
                            // Odd  inner branch: U_L · U_R · T ≈ target
                            let v_branch = if odd {
                                apply_t_dag_to_uv(v_inner)
                            } else {
                                v_inner
                            };
                            let v_branch_mpfr: Option<[MpFloat; 4]> =
                                v_inner_mpfr.as_ref().map(|vm| {
                                    if odd {
                                        apply_t_dag_to_uv_mpfr(vm, scratch.prec_q)
                                    } else {
                                        vm.clone()
                                    }
                                });
                            for sol in lll_aligned_search(
                                scratch, v_branch, v_branch_mpfr.as_ref(), lde_inner, eps,
                                DC_WALK_MAX_SOLUTIONS, max_leaf_checks,
                                max_nodes, &budget_hit, Some(&found_abort),
                            ) {
                                let u2t = if odd {
                                    *u_l * solution_to_u2t(&sol, lde_inner) * U2T::t()
                                } else {
                                    *u_l * solution_to_u2t(&sol, lde_inner)
                                };
                                let dist = diamond_distance_u2t_float(&u2t, target);
                                if dist < eps {
                                    found_abort
                                        .store(true, std::sync::atomic::Ordering::Relaxed);
                                    crate::synthesis::diag::record_branch_win(
                                        odd, pos, n, t,
                                    );
                                    return Some(SynthResultT {
                                        gates: Some(BlochDecomposer.decompose(&u2t)),
                                        lde: t,
                                        distance: dist,
                                    });
                                }
                                if crate::synthesis::diag::trace_enabled() {
                                    crate::synthesis::diag::N_DIST_REJECTED
                                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                }
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
mod tests;
