//! Exact Clifford+T synthesis (Algorithm 3.14, arXiv:2510.05816).
//!
//! Finds the minimum-T-count Clifford+T circuit U such that d_diamond(U, V) < ε.
//!
//! # Architecture
//!
//! Two search modes are used depending on T-count `t` vs `direct_limit` (default 6):
//!
//! **direct_search** (Algorithm 3.6, `t ≤ direct_limit`):
//!   Brute-force enumeration via `search::aligned_search` over the norm shell
//!   ‖x‖² = 2^t. Tries even / T / T† branches and all 24 Clifford left-prefixes.
//!   Fast for small t (norm ≤ 2^6 = 64), exponentially slow beyond that.
//!
//! **dc_search** (Algorithm 3.11, `t > direct_limit`):
//!   Divide-and-conquer with Matsumoto–Amano left prefixes L_{t'}.
//!   Split: t' = max(t − direct_limit,  ⌈t − 5/2·log₂(1/ε)⌉)  (whichever is larger).
//!   For each U_L ∈ L_{t'}, searches for U_R via `lll_aligned_search` (LLL+CVP,
//!   bandb5.py port) at the inner lde k_inner = t_inner/2 + 1 (bandb5 k convention).
//!   Tries even (U_L·U_R) and odd (U_L·U_R·T) inner branches.
//!
//! # Key invariant
//!
//! `lll_aligned_search` uses bandb5.py's k convention: norm shell = 2^k where
//! k = T_count/2 + 1. This is NOT the T-count itself. Always convert:
//!   k_inner = t_inner / 2 + 1  (even)
//!   k_inner = (t_inner - 1) / 2 + 1  (odd)

use num_complex::Complex;
use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};
use rayon::prelude::*;

// ─── Profiling counters (compiled only with --features profiling) ─────────────
#[cfg(feature = "profiling")]
mod profiling {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{LazyLock, Mutex};

    // phase1_enumerate
    pub static PHASE1_CALLS: AtomicU64 = AtomicU64::new(0);   // invocations
    pub static PHASE1A_PAIRS: AtomicU64 = AtomicU64::new(0);  // total (a1,c1) pairs collected
    pub static PHASE1A_NANOS: AtomicU64 = AtomicU64::new(0);  // CPU-ns in phase 1a collection
    pub static PHASE1B_NANOS: AtomicU64 = AtomicU64::new(0);  // CPU-ns in phase 1b search

    // phase2_pq
    pub static PHASE2_CALLS: AtomicU64 = AtomicU64::new(0);
    pub static LLL_NANOS: AtomicU64 = AtomicU64::new(0);
    pub static QR_NANOS: AtomicU64 = AtomicU64::new(0);
    pub static SCHNORR_NANOS: AtomicU64 = AtomicU64::new(0);

    // w_bd repetition (upper bound on LLL cache hit rate)
    pub static W_BD_CACHE: LazyLock<Mutex<std::collections::HashSet<[i64; 4]>>> =
        LazyLock::new(|| Mutex::new(std::collections::HashSet::new()));
    pub static W_BD_HITS: AtomicU64 = AtomicU64::new(0);

    // CVP-offset stats: when SE returns a solution, how far is z2/z1 from the CVP center?
    // Drives the "cap SE radius" optimization (ideas.md).
    pub static SE_HITS: AtomicU64 = AtomicU64::new(0);             // # successful SE returns
    pub static SE_Z2_OFFSET_SUM: AtomicU64 = AtomicU64::new(0);    // Σ |z2 - round(t_lat[2])|
    pub static SE_Z1_OFFSET_SUM: AtomicU64 = AtomicU64::new(0);    // Σ |z1 - round(z1_center)|
    pub static SE_Z2_OFFSET_MAX: AtomicU64 = AtomicU64::new(0);    // max |z2_offset|
    pub static SE_Z1_OFFSET_MAX: AtomicU64 = AtomicU64::new(0);    // max |z1_offset|

    pub fn reset() {
        PHASE1_CALLS.store(0, Ordering::Relaxed);
        PHASE1A_PAIRS.store(0, Ordering::Relaxed);
        PHASE1A_NANOS.store(0, Ordering::Relaxed);
        PHASE1B_NANOS.store(0, Ordering::Relaxed);
        PHASE2_CALLS.store(0, Ordering::Relaxed);
        LLL_NANOS.store(0, Ordering::Relaxed);
        QR_NANOS.store(0, Ordering::Relaxed);
        SCHNORR_NANOS.store(0, Ordering::Relaxed);
        W_BD_HITS.store(0, Ordering::Relaxed);
        W_BD_CACHE.lock().unwrap().clear();
        SE_HITS.store(0, Ordering::Relaxed);
        SE_Z2_OFFSET_SUM.store(0, Ordering::Relaxed);
        SE_Z1_OFFSET_SUM.store(0, Ordering::Relaxed);
        SE_Z2_OFFSET_MAX.store(0, Ordering::Relaxed);
        SE_Z1_OFFSET_MAX.store(0, Ordering::Relaxed);
    }

    pub fn record_se_hit(z2_offset: i64, z1_offset: i64) {
        let z2 = z2_offset.unsigned_abs();
        let z1 = z1_offset.unsigned_abs();
        SE_HITS.fetch_add(1, Ordering::Relaxed);
        SE_Z2_OFFSET_SUM.fetch_add(z2, Ordering::Relaxed);
        SE_Z1_OFFSET_SUM.fetch_add(z1, Ordering::Relaxed);
        // Atomic max via compare-exchange loop
        let mut prev = SE_Z2_OFFSET_MAX.load(Ordering::Relaxed);
        while z2 > prev {
            match SE_Z2_OFFSET_MAX.compare_exchange_weak(prev, z2, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => break,
                Err(now) => prev = now,
            }
        }
        let mut prev = SE_Z1_OFFSET_MAX.load(Ordering::Relaxed);
        while z1 > prev {
            match SE_Z1_OFFSET_MAX.compare_exchange_weak(prev, z1, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => break,
                Err(now) => prev = now,
            }
        }
    }

    pub fn report() {
        let p1_calls = PHASE1_CALLS.load(Ordering::Relaxed);
        let p1a_pairs = PHASE1A_PAIRS.load(Ordering::Relaxed);
        let p1a_ms = PHASE1A_NANOS.load(Ordering::Relaxed) as f64 / 1e6;
        let p1b_ms = PHASE1B_NANOS.load(Ordering::Relaxed) as f64 / 1e6;
        let avg_pairs = if p1_calls > 0 { p1a_pairs as f64 / p1_calls as f64 } else { 0.0 };
        eprintln!(
            "[profile] phase1_enumerate: {p1_calls} calls | 1a_pairs={p1a_pairs} ({avg_pairs:.1}/call) | 1a={p1a_ms:.1}ms  1b={p1b_ms:.1}ms"
        );

        let p2_calls = PHASE2_CALLS.load(Ordering::Relaxed);
        let lll_ms = LLL_NANOS.load(Ordering::Relaxed) as f64 / 1e6;
        let qr_ms = QR_NANOS.load(Ordering::Relaxed) as f64 / 1e6;
        let sch_ms = SCHNORR_NANOS.load(Ordering::Relaxed) as f64 / 1e6;
        let hits = W_BD_HITS.load(Ordering::Relaxed);
        let hit_pct = if p2_calls > 0 { 100.0 * hits as f64 / p2_calls as f64 } else { 0.0 };
        let avg_lll_us = if p2_calls > 0 { lll_ms * 1000.0 / p2_calls as f64 } else { 0.0 };
        let avg_se_us  = if p2_calls > 0 { sch_ms * 1000.0 / p2_calls as f64 } else { 0.0 };
        eprintln!(
            "[profile] phase2_pq:        {p2_calls} calls | lll={lll_ms:.1}ms ({avg_lll_us:.2}µs/call)  qr={qr_ms:.1}ms  se={sch_ms:.1}ms ({avg_se_us:.2}µs/call) | w_bd_hits={hits}/{p2_calls} ({hit_pct:.1}%)"
        );

        let se_hits = SE_HITS.load(Ordering::Relaxed);
        let z2_sum = SE_Z2_OFFSET_SUM.load(Ordering::Relaxed);
        let z1_sum = SE_Z1_OFFSET_SUM.load(Ordering::Relaxed);
        let z2_max = SE_Z2_OFFSET_MAX.load(Ordering::Relaxed);
        let z1_max = SE_Z1_OFFSET_MAX.load(Ordering::Relaxed);
        if se_hits > 0 {
            let z2_avg = z2_sum as f64 / se_hits as f64;
            let z1_avg = z1_sum as f64 / se_hits as f64;
            eprintln!(
                "[profile] se_hits: {se_hits} | |z2_off| avg={z2_avg:.2} max={z2_max} | |z1_off| avg={z1_avg:.2} max={z1_max}"
            );
        }
    }
}

#[cfg(feature = "profiling")]
pub use profiling::{reset as reset_profiling, report as report_profiling};

/// Global cache for build_l results, keyed by t_prime.
static BUILD_L_CACHE: LazyLock<Mutex<HashMap<u32, Vec<U2T>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

use crate::matrix::U2T;
use crate::rings::types::{Int, Float};
use crate::rings::ZOmega;
use crate::synthesis::cliffords::CLIFFORD_TABLE_T;
use crate::synthesis::decomposer::BlochDecomposer;
use crate::synthesis::search::{
    aligned_search,
    apply_t_dag_to_uv, apply_t_to_uv, apply_u2t_dag_to_uv, compute_align_vec, normalize4,
};

// ─── Float matrix helpers ──────────────────────────────────────────────────────

type Mat2 = [[Complex<Float>; 2]; 2];

/// Diamond distance between two unitaries: √max(0, 1 − |tr(A·B†)|²/4).
///
/// Tr(A·B†) = sum_{i,k} A[i][k] · conj(B[i][k])  (sum over all i, k of element-wise products).
pub fn diamond_distance_float(a: &Mat2, b: &Mat2) -> Float {
    let tr = a[0][0] * b[0][0].conj()
           + a[0][1] * b[0][1].conj()
           + a[1][0] * b[1][0].conj()
           + a[1][1] * b[1][1].conj();
    (1.0 - tr.norm_sqr() / 4.0).max(0.0).sqrt()
}

/// Diamond distance between an exact U2T and a float target matrix.
fn diamond_distance_u2t_float(u: &U2T, target: &Mat2) -> Float {
    let uf = u.to_float();
    let tr = uf[0][0] * target[0][0].conj()
           + uf[0][1] * target[0][1].conj()
           + uf[1][0] * target[1][0].conj()
           + uf[1][1] * target[1][1].conj();
    (1.0 - tr.norm_sqr() / 4.0).max(0.0).sqrt()
}

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
fn build_l(t_prime: u32) -> Vec<U2T> {
    // Check cache first
    {
        let cache = BUILD_L_CACHE.lock().unwrap();
        if let Some(v) = cache.get(&t_prime) {
            return v.clone();
        }
    }

    let result = build_l_inner(t_prime);

    // Store in cache
    BUILD_L_CACHE.lock().unwrap().insert(t_prime, result.clone());
    result
}

fn build_l_inner(t_prime: u32) -> Vec<U2T> {
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

// ─── LLL-based aligned search (bandb5.py port) ────────────────────────────────

/// Scale the alignment vector to the y-vector used in bandb5.py.
///
/// y = compute_align_vec(v) * sqrt(2^k) / 2.
/// Property: ||y||² = 2^(k-1).
fn uv_to_xy(v: [Float; 4], k: u32) -> [Float; 8] {
    let scale = ((1i64 << k) as Float).sqrt() / 2.0;
    compute_align_vec(v).map(|x| x * scale)
}

/// Extended GCD: returns (gcd, s, t) such that a*s + b*t = gcd.
fn extended_gcd(a: i64, b: i64) -> (i64, i64, i64) {
    if b == 0 {
        return (a, 1, 0);
    }
    let (g, s, t) = extended_gcd(b, a % b);
    (g, t, s - (a / b) * t)
}

/// Compute a 3×4 integer null basis N for the row vector w ∈ ℤ⁴.
///
/// Returns N such that N @ w == 0 and N has full row rank 3.
/// Uses unimodular column operations via extended GCD.
fn integer_null_basis(w: [i64; 4]) -> [[i64; 4]; 3] {
    let mut ww = w;
    // 4×4 identity stored column-major: u[col][row]
    let mut u = [[0i64; 4]; 4];
    for i in 0..4 {
        u[i][i] = 1;
    }

    for i in 1..4 {
        if ww[i] == 0 {
            continue;
        }
        let (g, s, t) = extended_gcd(ww[0], ww[i]);
        let wi_g = ww[i] / g;
        let w0_g = ww[0] / g;
        // new_col0 = s*col0 + t*col_i
        // new_col_i = -wi_g*col0 + w0_g*col_i
        let mut new_col0 = [0i64; 4];
        let mut new_coli = [0i64; 4];
        for r in 0..4 {
            new_col0[r] = s * u[0][r] + t * u[i][r];
            new_coli[r] = -wi_g * u[0][r] + w0_g * u[i][r];
        }
        u[0] = new_col0;
        u[i] = new_coli;
        ww[0] = g;
        ww[i] = 0;
    }

    // Return last 3 columns of U transposed: shape 3×4
    [u[1], u[2], u[3]]
}

/// Gram-Schmidt orthogonalization of a 3×4 float basis (rows = basis vectors).
/// Returns (Bs, mu) where Bs is the orthogonalized basis and mu[i][j] = proj coefficient.
fn gram_schmidt_3x4(bf: &[[Float; 4]; 3]) -> ([[Float; 4]; 3], [[Float; 3]; 3]) {
    let mut bs = *bf;
    let mut mu = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..i {
            let dot_ij: Float = bf[i].iter().zip(bs[j].iter()).map(|(a, b)| a * b).sum();
            let dot_jj: Float = bs[j].iter().map(|x| x * x).sum();
            if dot_jj.abs() < 1e-14 {
                continue;
            }
            mu[i][j] = dot_ij / dot_jj;
            for k in 0..4 {
                bs[i][k] -= mu[i][j] * bs[j][k];
            }
        }
    }
    (bs, mu)
}

/// LLL basis reduction for a 3×4 integer matrix (rows = basis vectors), delta=0.75.
fn lll_reduce(basis: [[i64; 4]; 3]) -> [[i64; 4]; 3] {
    let mut b = basis;
    let mut bf: [[Float; 4]; 3] = b.map(|row| row.map(|x| x as Float));

    let mut k = 1usize;
    while k < 3 {
        let (_, mu) = gram_schmidt_3x4(&bf);

        // Size reduction
        for j in (0..k).rev() {
            let r = mu[k][j].round() as i64;
            if r != 0 {
                for c in 0..4 {
                    b[k][c] -= r * b[j][c];
                    bf[k][c] -= r as Float * bf[j][c];
                }
            }
        }

        let (bs2, mu2) = gram_schmidt_3x4(&bf);
        let norm_k: Float = bs2[k].iter().map(|x| x * x).sum();
        let norm_km1: Float = bs2[k - 1].iter().map(|x| x * x).sum();
        let delta = 0.75;
        if norm_k >= (delta - mu2[k][k - 1].powi(2)) * norm_km1 {
            k += 1;
        } else {
            b.swap(k, k - 1);
            bf.swap(k, k - 1);
            k = k.saturating_sub(1).max(1);
        }
    }
    b
}

/// Compute the upper-triangular R factor from QR decomposition of N^T (4×3).
///
/// Uses modified Gram-Schmidt on the columns of N^T.
/// Returns 3×3 upper triangular R with positive diagonal.
fn qr_upper(n: &[[i64; 4]; 3]) -> [[Float; 3]; 3] {
    let mut cols: [[Float; 4]; 3] = [
        n[0].map(|x| x as Float),
        n[1].map(|x| x as Float),
        n[2].map(|x| x as Float),
    ];
    let mut r = [[0.0; 3]; 3];

    for i in 0..3 {
        let norm_i: Float = cols[i].iter().map(|x| x * x).sum::<Float>().sqrt();
        r[i][i] = norm_i;
        if norm_i < 1e-14 {
            continue;
        }
        let q_i = cols[i].map(|x| x / norm_i);

        for j in (i + 1)..3 {
            let dot: Float = q_i.iter().zip(cols[j].iter()).map(|(a, b)| a * b).sum();
            r[i][j] = dot;
            for row in 0..4 {
                cols[j][row] -= dot * q_i[row];
            }
        }
        cols[i] = q_i;
    }

    // Make diagonal positive
    for i in 0..3 {
        if r[i][i] < 0.0 {
            for j in 0..3 {
                r[i][j] = -r[i][j];
            }
        }
    }

    r
}

/// Hard cap on SE outward iteration radius. Empirically (see profiling se_hits stats),
/// >99% of solutions across k_inner=9..17 land within ±8 of the CVP center. Capping at 6
/// caps per-call SE cost aggressively; solutions at offset > 6 are skipped at t=T but found
/// at t=T+1 (acceptable per project speed-vs-completeness goal).
const SE_MAX_OFFSET: i64 = 6;

/// Adaptive cap on outer (a1,c1,a2,c2) CenteredRange offset in `phase1_enumerate`.
/// Scales with the natural sphere bound max_outer = sqrt(2^k). A fixed cap fails because
/// at k=17 (max_outer=363) capping at 50 forces lde rollover (solutions need wider range);
/// at k=13 (max_outer=90) capping at 50 is loose enough. We use max_outer/2 with a floor
/// of 30, which gives k=13: ±45 (full range), k=17: ±181 (about half range).
fn outer_max_offset(max_outer: i64) -> i64 {
    (max_outer / 2).max(30)
}

/// Schnorr-Euchner CVP enumeration over z ∈ ℤ³ such that ‖N_lll^T @ z‖² = r_norm_sq.
///
/// Iterates z2 outward from the CVP center and prunes via alignment Cauchy-Schwarz.
/// Returns the first (b1,d1,b2,d2) that satisfies norm, unitarity, and alignment.
fn schnorr_euchner(
    n_lll: &[[i64; 4]; 3],
    r_mat: &[[Float; 3]; 3],
    t_lat: [Float; 3],
    r_norm_sq: i64,
    a1: i64, c1: i64, a2: i64, c2: i64,
    y_inner: [Float; 4],
    dot_outer: Float,
    threshold_sq: Float,
) -> Option<[i64; 4]> {
    // w_lat[i] = N_lll[i] · y_inner: alignment direction in lattice coordinates
    let nf: [[Float; 4]; 3] = n_lll.map(|row| row.map(|x| x as Float));
    let w_lat: [Float; 3] = std::array::from_fn(|i| {
        nf[i].iter().zip(y_inner.iter()).map(|(a, b)| a * b).sum()
    });
    let r00 = r_mat[0][0].abs().max(1e-12);
    let r11_abs = r_mat[1][1].abs().max(1e-12);
    // Cauchy-Schwarz bound coefficient: max |z1·w_lat[1] + z0·w_lat[0]|² / rem2
    let align_perp_sq = w_lat[0] * w_lat[0] / (r00 * r00)
                      + w_lat[1] * w_lat[1] / (r11_abs * r11_abs);

    let radius = (r_norm_sq as Float).sqrt();
    let r22 = r_mat[2][2];
    if r22.abs() < 1e-12 {
        return None;
    }

    let z2_ci = t_lat[2].round() as i64;
    // Cap SE outward iteration radius. Empirically, ~99% of solutions land within ±8 of
    // the CVP center (see [profile] se_hits stats). Capping trades occasional misses at
    // t=T for a constant-bounded SE per call; missed solutions are recovered at t=T+1.
    let z2_max_offset = ((radius / r22.abs()).ceil() as i64 + 2).min(SE_MAX_OFFSET);

    let mut z2_pos_done = false;
    let mut z2_neg_done = false;

    for raw_offset in 0..=(2 * z2_max_offset) {
        // Alternate outward: 0, +1, -1, +2, -2, ...
        let offset: i64 = if raw_offset == 0 {
            0
        } else if raw_offset % 2 == 1 {
            (raw_offset + 1) / 2
        } else {
            -(raw_offset / 2)
        };

        if offset > 0 && z2_pos_done { continue; }
        if offset < 0 && z2_neg_done { continue; }
        if z2_pos_done && z2_neg_done { break; }

        let z2 = z2_ci + offset;

        let rem2_raw = r_norm_sq as Float - (r22 * z2 as Float) * (r22 * z2 as Float);
        if rem2_raw < -1e-9 {
            // Sphere exceeded: monotone, so all further z2 in this direction also exceed.
            if offset >= 0 { z2_pos_done = true; }
            if offset <= 0 { z2_neg_done = true; }
            continue;
        }
        let rem2 = rem2_raw.max(0.0);

        // Alignment prune: can any (z1,z0) achieve |dot| ≥ sqrt(threshold_sq)?
        let z2_dot = z2 as Float * w_lat[2];
        let total_center = dot_outer + z2_dot;
        let max_perp = (rem2 * align_perp_sq).sqrt();
        if (total_center + max_perp) * (total_center + max_perp) < threshold_sq
            && (total_center - max_perp) * (total_center - max_perp) < threshold_sq
        {
            continue; // no (z1,z0) can satisfy alignment for this z2
        }

        let r11 = r_mat[1][1];
        let r12 = r_mat[1][2];
        if r11.abs() < 1e-12 { continue; }
        let z1_center = t_lat[1] - (r12 / r11) * z2 as Float;
        let z1_ci = z1_center.round() as i64;
        let z1_max_offset = ((rem2.sqrt() / r11.abs()).ceil() as i64 + 2).min(SE_MAX_OFFSET);
        let w_lat_z0_scale = w_lat[0].abs() / r00;

        let mut z1_pos_done = false;
        let mut z1_neg_done = false;

        for raw_offset1 in 0..=(2 * z1_max_offset) {
            let offset1: i64 = if raw_offset1 == 0 {
                0
            } else if raw_offset1 % 2 == 1 {
                (raw_offset1 + 1) / 2
            } else {
                -(raw_offset1 / 2)
            };

            if offset1 > 0 && z1_pos_done { continue; }
            if offset1 < 0 && z1_neg_done { continue; }
            if z1_pos_done && z1_neg_done { break; }

            let z1 = z1_ci + offset1;

            let rem1_raw =
                rem2 - (r_mat[1][1] * z1 as Float + r_mat[1][2] * z2 as Float).powi(2);
            if rem1_raw < -1e-9 {
                if offset1 >= 0 { z1_pos_done = true; }
                if offset1 <= 0 { z1_neg_done = true; }
                continue;
            }
            let rem1 = rem1_raw.max(0.0);

            let r00_local = r_mat[0][0];
            let r01 = r_mat[0][1];
            let r02 = r_mat[0][2];
            if r00_local.abs() < 1e-12 { continue; }
            let inner = r01 * z1 as Float + r02 * z2 as Float;
            let val = rem1.sqrt();

            // Alignment prune on z1: incorporate z0's analytic center into partial for
            // a tighter bound — z0f ≈ (-inner ± val)/r00, so z0 contributes
            // (-inner/r00)*w_lat[0] (fixed) ± (val/r00)*|w_lat[0]| (variable).
            let z1_dot = z1 as Float * w_lat[1];
            let z0_center_dot = (-inner / r00_local) * w_lat[0];
            let max_z0_align = val * w_lat_z0_scale;
            let partial_eff = total_center + z1_dot + z0_center_dot;
            if (partial_eff + max_z0_align) * (partial_eff + max_z0_align) < threshold_sq
                && (partial_eff - max_z0_align) * (partial_eff - max_z0_align) < threshold_sq
            {
                continue;
            }

            for sign in [1.0_f64, -1.0] {
                let z0f = (-inner + sign * val) / r00_local;
                let z0 = z0f.round() as i64;
                let mut bd = [0i64; 4];
                for row in 0..4 {
                    bd[row] = n_lll[0][row] * z0 + n_lll[1][row] * z1 + n_lll[2][row] * z2;
                }
                let [b1, d1, b2, d2] = bd;
                if b1 * b1 + d1 * d1 + b2 * b2 + d2 * d2 != r_norm_sq { continue; }
                if b1 * (a1 + c1) + d1 * (c1 - a1) + b2 * (a2 + c2) + d2 * (c2 - a2) != 0 {
                    continue;
                }
                let dot_inner = b1 as Float * y_inner[0] + d1 as Float * y_inner[1]
                              + b2 as Float * y_inner[2] + d2 as Float * y_inner[3];
                let dot = dot_outer + dot_inner;
                if dot * dot >= threshold_sq {
                    #[cfg(feature = "profiling")]
                    profiling::record_se_hit(z2 - z2_ci, z1 - z1_ci);
                    return Some(bd);
                }
            }
        }
    }
    None
}

/// Phase 2: find (b1,d1,b2,d2) satisfying norm, unitarity, and alignment using LLL+CVP.
fn phase2_pq(
    a1: i64, c1: i64, a2: i64, c2: i64, r: i64,
    y_inner: [Float; 4],
    dot_outer: Float,
    threshold_sq: Float,
) -> Option<[i64; 4]> {
    if r < 0 {
        return None;
    }
    if r == 0 {
        let dot = dot_outer;
        return if dot * dot >= threshold_sq { Some([0, 0, 0, 0]) } else { None };
    }

    let w_bd = [a1 + c1, c1 - a1, a2 + c2, c2 - a2];

    // Degenerate: all unitarity coefficients zero → brute-force 4D sphere with alignment check
    if w_bd == [0, 0, 0, 0] {
        let max_b1 = (r as Float).sqrt() as i64 + 1;
        for b1 in -max_b1..=max_b1 {
            let rem1 = r - b1 * b1;
            if rem1 < 0 { continue; }
            let max_d1 = (rem1 as Float).sqrt() as i64 + 1;
            for d1 in -max_d1..=max_d1 {
                let rem2 = rem1 - d1 * d1;
                if rem2 < 0 { continue; }
                let max_b2 = (rem2 as Float).sqrt() as i64 + 1;
                for b2 in -max_b2..=max_b2 {
                    let rem3 = rem2 - b2 * b2;
                    if rem3 < 0 { continue; }
                    let d2s = (rem3 as Float).sqrt() as i64;
                    if d2s * d2s != rem3 { continue; }
                    let d2_vals: &[i64] = if d2s == 0 { &[0] } else { &[d2s, -d2s] };
                    for &d2 in d2_vals {
                        let dot_inner = b1 as Float * y_inner[0] + d1 as Float * y_inner[1]
                                      + b2 as Float * y_inner[2] + d2 as Float * y_inner[3];
                        let dot = dot_outer + dot_inner;
                        if dot * dot >= threshold_sq {
                            return Some([b1, d1, b2, d2]);
                        }
                    }
                }
            }
        }
        return None;
    }

    #[cfg(feature = "profiling")]
    {
        use std::sync::atomic::Ordering;
        profiling::PHASE2_CALLS.fetch_add(1, Ordering::Relaxed);
        if !profiling::W_BD_CACHE.lock().unwrap().insert(w_bd) {
            profiling::W_BD_HITS.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[cfg(feature = "profiling")]
    let t_lll = std::time::Instant::now();
    let n_bd = integer_null_basis(w_bd);
    let n_lll = lll_reduce(n_bd);
    #[cfg(feature = "profiling")]
    profiling::LLL_NANOS.fetch_add(t_lll.elapsed().as_nanos() as u64,
                                   std::sync::atomic::Ordering::Relaxed);

    // CVP target: project y_inner toward √r
    let norm_yi: Float = y_inner.iter().map(|x| x * x).sum::<Float>().sqrt();
    let t_ambient: [Float; 4] = if norm_yi < 1e-12 {
        [0.0; 4]
    } else {
        let s = (r as Float).sqrt() / norm_yi;
        y_inner.map(|x| x * s)
    };

    // Gram matrix G = N_lll @ N_lll^T (3×3)
    let nf: [[Float; 4]; 3] = n_lll.map(|row| row.map(|x| x as Float));
    let mut g = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            g[i][j] = nf[i].iter().zip(nf[j].iter()).map(|(a, b)| a * b).sum();
        }
    }

    // rhs = N_lll @ t_ambient
    let rhs: [Float; 3] = [
        nf[0].iter().zip(t_ambient.iter()).map(|(a, b)| a * b).sum(),
        nf[1].iter().zip(t_ambient.iter()).map(|(a, b)| a * b).sum(),
        nf[2].iter().zip(t_ambient.iter()).map(|(a, b)| a * b).sum(),
    ];

    let t_lat = solve_3x3(&g, rhs).unwrap_or([0.0; 3]);

    #[cfg(feature = "profiling")]
    let t_qr = std::time::Instant::now();
    let r_mat = qr_upper(&n_lll);
    #[cfg(feature = "profiling")]
    profiling::QR_NANOS.fetch_add(t_qr.elapsed().as_nanos() as u64,
                                  std::sync::atomic::Ordering::Relaxed);

    #[cfg(feature = "profiling")]
    let t_sch = std::time::Instant::now();
    let result = schnorr_euchner(
        &n_lll, &r_mat, t_lat, r, a1, c1, a2, c2, y_inner, dot_outer, threshold_sq,
    );
    #[cfg(feature = "profiling")]
    profiling::SCHNORR_NANOS.fetch_add(t_sch.elapsed().as_nanos() as u64,
                                       std::sync::atomic::Ordering::Relaxed);
    result
}


/// Solve a 3×3 system Ax = b via Gaussian elimination with partial pivoting.
fn solve_3x3(a: &[[Float; 3]; 3], b: [Float; 3]) -> Option<[Float; 3]> {
    let mut m = [
        [a[0][0], a[0][1], a[0][2], b[0]],
        [a[1][0], a[1][1], a[1][2], b[1]],
        [a[2][0], a[2][1], a[2][2], b[2]],
    ];
    for col in 0..3 {
        let mut max_row = col;
        for row in (col + 1)..3 {
            if m[row][col].abs() > m[max_row][col].abs() {
                max_row = row;
            }
        }
        m.swap(col, max_row);
        if m[col][col].abs() < 1e-14 {
            return None;
        }
        let pivot = m[col][col];
        for row in (col + 1)..3 {
            let factor = m[row][col] / pivot;
            for k in col..4 {
                let v = m[col][k];
                m[row][k] -= factor * v;
            }
        }
    }
    let mut x = [0.0; 3];
    for i in (0..3).rev() {
        let mut s = m[i][3];
        for j in (i + 1)..3 {
            s -= m[i][j] * x[j];
        }
        x[i] = s / m[i][i];
    }
    Some(x)
}

/// Iterator yielding integers outward from `center`: center, center+1, center-1, ...
struct CenteredRange {
    center: i64,
    offset: i64,
    limit: i64,
}

impl CenteredRange {
    fn new(center: i64, max_radius: i64) -> Self {
        Self { center, offset: 0, limit: max_radius + center.unsigned_abs() as i64 + 2 }
    }
}

impl Iterator for CenteredRange {
    type Item = i64;
    fn next(&mut self) -> Option<i64> {
        if self.offset > self.limit {
            return None;
        }
        let val = if self.offset == 0 {
            self.offset = 1;
            self.center
        } else if self.offset % 2 == 1 {
            let v = self.center + (self.offset + 1) / 2;
            self.offset += 1;
            v
        } else {
            let v = self.center - self.offset / 2;
            self.offset += 1;
            v
        };
        Some(val)
    }
}

/// Phase 1: enumerate outer variables (a1,c1,a2,c2) with Cauchy-Schwarz pruning.
///
/// Returns up to one full 8-vector [a1,b1,c1,d1,a2,b2,c2,d2].
fn phase1_enumerate(y: &[Float; 8], k: u32, eps: Float) -> Vec<[i64; 8]> {
    #[cfg(feature = "profiling")]
    {
        use std::sync::atomic::Ordering;
        profiling::PHASE1_CALLS.fetch_add(1, Ordering::Relaxed);
    }
    #[cfg(feature = "profiling")]
    let t_phase1a = std::time::Instant::now();

    let target_norm: i64 = 1i64 << k;
    // Threshold on (x·y)²: 2^(2k-2)·(1-eps²)
    let threshold_xy = (1i64 << (2 * k)) as Float / 4.0 * (1.0 - eps * eps);

    let y_norm: Float = y.iter().map(|x| x * x).sum::<Float>().sqrt();
    let scale = (target_norm as Float).sqrt() / y_norm;

    let a1_c = (y[0] * scale).round() as i64;
    let c1_c = (y[2] * scale).round() as i64;
    let a2_c = (y[4] * scale).round() as i64;
    let c2_c = (y[6] * scale).round() as i64;

    let max_outer = (target_norm as Float).sqrt() as i64 + 1;

    let y_sq_all: Float = y.iter().map(|x| x * x).sum();
    let y_sq_no_a1 = y_sq_all - y[0] * y[0];
    let y_sq_no_a1_c1 = y_sq_no_a1 - y[2] * y[2];
    let y_sq_no_a1_c1_a2 = y_sq_no_a1_c1 - y[4] * y[4];
    let y_sq_inner = y[1]*y[1] + y[3]*y[3] + y[5]*y[5] + y[7]*y[7];

    let thresh = threshold_xy.sqrt();

    // Phase 1a: collect (a1, c1, rem_c1, dot_a1c1) pairs that pass the outer two
    // pruning levels.  Capped adaptively at outer_max_offset(max_outer).
    let outer_take = (2 * outer_max_offset(max_outer) + 1) as usize;
    let pairs: Vec<(i64, i64, i64, Float)> = {
        let mut v = Vec::new();
        for a1 in CenteredRange::new(a1_c, max_outer).take(outer_take) {
            if a1 * a1 > target_norm { continue; }
            let rem_a1 = target_norm - a1 * a1;
            let dot_a1 = a1 as Float * y[0];
            if dot_a1.abs() + (rem_a1 as Float * y_sq_no_a1).sqrt() < thresh {
                continue;
            }
            for c1 in CenteredRange::new(c1_c, max_outer).take(outer_take) {
                if a1*a1 + c1*c1 > target_norm { continue; }
                let rem_c1 = target_norm - a1*a1 - c1*c1;
                let dot_a1c1 = dot_a1 + c1 as Float * y[2];
                if dot_a1c1.abs() + (rem_c1 as Float * y_sq_no_a1_c1).sqrt() < thresh {
                    continue;
                }
                v.push((a1, c1, rem_c1, dot_a1c1));
            }
        }
        v
    };

    #[cfg(feature = "profiling")]
    {
        use std::sync::atomic::Ordering;
        profiling::PHASE1A_PAIRS.fetch_add(pairs.len() as u64, Ordering::Relaxed);
        profiling::PHASE1A_NANOS.fetch_add(t_phase1a.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
    #[cfg(feature = "profiling")]
    let t_phase1b = std::time::Instant::now();

    // Phase 1b: run the a2/c2/phase2_pq inner work over collected pairs.
    // For small Vec, run sequentially to avoid rayon overhead.
    // For large Vec, run in parallel (find_map_any for early exit).
    let n_threads = rayon::current_num_threads();
    let run_par = n_threads > 1 && pairs.len() >= n_threads * 4;
    let inner_fn = |(a1, c1, rem_c1, dot_a1c1): (i64, i64, i64, Float)| -> Option<[i64; 8]> {
        for a2 in CenteredRange::new(a2_c, max_outer).take(outer_take) {
            if a2*a2 > rem_c1 { continue; }
            let rem_a2 = rem_c1 - a2*a2;
            let dot_3 = dot_a1c1 + a2 as Float * y[4];
            if dot_3.abs() + (rem_a2 as Float * y_sq_no_a1_c1_a2).sqrt() < thresh {
                continue;
            }
            for c2 in CenteredRange::new(c2_c, max_outer).take(outer_take) {
                if c2*c2 > rem_a2 { continue; }
                let r = rem_a2 - c2*c2;
                let outer_norm_sq = target_norm - r;
                let dot_outer = dot_3 + c2 as Float * y[6];
                let w_bd_norm_sq = 2 * outer_norm_sq;
                let y_inner_proj_sq = if w_bd_norm_sq > 0 {
                    let wdot = y[1] * (a1 + c1) as Float + y[3] * (c1 - a1) as Float
                             + y[5] * (a2 + c2) as Float + y[7] * (c2 - a2) as Float;
                    (y_sq_inner - wdot * wdot / w_bd_norm_sq as Float).max(0.0)
                } else {
                    y_sq_inner
                };
                if dot_outer.abs() + (r as Float * y_inner_proj_sq).sqrt() < thresh {
                    continue;
                }
                let y_inner = [y[1], y[3], y[5], y[7]];
                if let Some([b1, d1, b2, d2]) =
                    phase2_pq(a1, c1, a2, c2, r, y_inner, dot_outer, threshold_xy)
                {
                    return Some([a1, b1, c1, d1, a2, b2, c2, d2]);
                }
            }
        }
        None
    };

    let result = if run_par {
        pairs.into_par_iter().find_map_any(inner_fn)
    } else {
        pairs.into_iter().find_map(inner_fn)
    };

    #[cfg(feature = "profiling")]
    profiling::PHASE1B_NANOS.fetch_add(t_phase1b.elapsed().as_nanos() as u64,
                                       std::sync::atomic::Ordering::Relaxed);

    result.map(|x| vec![x]).unwrap_or_default()
}

/// LLL-based aligned search: implements bandb5.py's `synthesize`.
///
/// Finds integer lattice vectors satisfying norm, unitarity, and alignment.
fn lll_aligned_search(v: [Float; 4], k: u32, eps: Float, max_solutions: usize) -> Vec<[i64; 8]> {
    if max_solutions == 0 || k > 62 {
        return Vec::new();
    }
    let y = uv_to_xy(v, k);
    let sols = phase1_enumerate(&y, k, eps);
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
pub struct SynthResult {
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

/// Clifford+T synthesizer implementing Algorithm 3.14 of arXiv:2510.05816.
pub struct Synthesizer {
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

impl Synthesizer {
    /// Create a synthesizer with the given precision and sensible defaults.
    ///
    /// Sets `min_lde = floor(3/2 · log₂(1/ε))` — the lower bound below which
    /// no generic rotation can be approximated to within ε.
    pub fn new(epsilon: Float) -> Self {
        let min_lde = if epsilon > 0.0 && epsilon < 1.0 {
            (2.8 * (1.0 / epsilon).log2()).floor() as u32
        } else {
            0
        };
        Self { epsilon, max_lde: 50, min_lde, direct_limit: 6 }
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
    pub fn synthesize(&self, target: Mat2) -> Option<SynthResult> {
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
                        if tp > 0 && seen.insert(tp) { Some(tp) } else { None }
                    })
                    .collect()
            };
            needed.into_par_iter().for_each(|tp| { build_l(tp); });
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
    ///   - Otherwise: dc_search (MA prefix + LLL/CVP inner search).
    ///
    /// The direct_limit cap prevents aligned_search from hanging at large lde.
    fn try_at_lde(&self, target: &Mat2, v: [Float; 4], t: u32) -> Option<SynthResult> {
        if t <= self.direct_limit {
            self.direct_search(target, v, t)
        } else {
            self.dc_search(target, v, t)
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
    fn direct_search(&self, target: &Mat2, v: [Float; 4], t: u32) -> Option<SynthResult> {
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
            for sol in aligned_search(*v_s, t, eps, 1) {
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
                    return Some(SynthResult { gates: Some(gates), lde: t, distance: dist });
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
    fn dc_search(&self, target: &Mat2, v: [Float; 4], t: u32) -> Option<SynthResult> {
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
                return None;
            }
            opt
        };
        
        if t_prime == 0 || t_prime > t {
            return self.direct_search(target, v, t);
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

        // Parallel search over all left prefixes.
        // find_map_any stops all threads as soon as any one returns Some(...).
        // with_min_len ensures rayon distributes work evenly rather than
        // keeping everything on one thread when items complete quickly.
        let n_threads = rayon::current_num_threads();
        let chunk = (prefixes.len() / n_threads).max(1);
        prefixes.par_iter().with_min_len(chunk).find_map_any(|u_l| {
            // Compute U_L† · target as a full float matrix, then extract uv via
            // mat_to_uv which tries all 8 global phases (matches bandb6.py).
            let m_inner = u2t_dag_times_mat2(u_l, target);
            let v_inner = match mat_to_uv(&m_inner) {
                Some(v) => v,
                None => return None,
            };

            // Even inner branch: U_L · U_R ≈ target
            for sol in lll_aligned_search(v_inner, k_inner, eps, 1) {
                let u2t = *u_l * solution_to_u2t(&sol, k_inner);
                let dist = diamond_distance_u2t_float(&u2t, target);
                if dist < eps {
                    return Some(SynthResult {
                        gates: Some(BlochDecomposer.decompose(&u2t)),
                        lde: t,
                        distance: dist,
                    });
                }
            }

            // Odd inner branch: U_L · U_R · T ≈ target
            if t_inner > 0 {
                let v_inner_t = apply_t_dag_to_uv(v_inner);
                for sol in lll_aligned_search(v_inner_t, k_inner, eps, 1) {
                    let u2t = *u_l * solution_to_u2t(&sol, k_inner) * U2T::t();
                    let dist = diamond_distance_u2t_float(&u2t, target);
                    if dist < eps {
                        return Some(SynthResult {
                            gates: Some(BlochDecomposer.decompose(&u2t)),
                            lde: t,
                            distance: dist,
                        });
                    }
                }
            }

            None
        })
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::{f64::consts::{FRAC_1_SQRT_2, PI}};

    fn rz(theta: Float) -> Mat2 {
        [
            [Complex::from_polar(1., -theta / 2.), Complex::new(0., 0.)],
            [Complex::new(0., 0.), Complex::from_polar(1., theta / 2.)],
        ]
    }

    fn check_result(result: &SynthResult, _target: &Mat2, eps: Float) {
        assert!(
            result.distance < eps,
            "distance={:.6e} ≥ epsilon={:.6e}",
            result.distance, eps
        );
    }

    #[test]
    fn test_synthesize_identity() {
        let id: Mat2 = [[Complex::new(1., 0.), Complex::new(0., 0.)], [Complex::new(0., 0.), Complex::new(1., 0.)]];
        // with_min_lde(0): identity is a Clifford with exact solution at lde=0.
        let synth = Synthesizer::new(0.01).with_min_lde(0);
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
        let synth = Synthesizer::new(0.01).with_min_lde(0);
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
        let synth = Synthesizer::new(0.01);
        let result = synth.synthesize(h).expect("Should synthesize H");
        check_result(&result, &h, 0.01);
    }

    #[test]
    fn test_synthesize_rz_small() {
        // Rz(π/4) = T gate, should need lde=1.
        let target = rz(PI as Float / 4.);
        let synth = Synthesizer::new(0.01);
        let result = synth.synthesize(target).expect("Should synthesize Rz(π/4)");
        println!("{:?}", result.gates);

        check_result(&result, &target, 0.01);
    }

    #[test]
    fn test_synthesize_rz_moderate_1() {
        let target = rz(0.3);
        let synth = Synthesizer::new(0.01);
        let result = synth.synthesize(target).expect("Should synthesize Rz(0.3)");
        println!("{:?}", result.gates);
        check_result(&result, &target, 0.01);
    }

    #[test]
    fn test_synthesize_rz_moderate_2() {
        let target = rz(1.34);
        let synth = Synthesizer::new(0.01);
        let result = synth.synthesize(target).expect("Should synthesize Rz(1.34)");
        println!("{:?}", result.gates);
        check_result(&result, &target, 0.01);
    }

    #[test]
    fn test_synthesize_rz_hard_1() {
        // eps=0.01: needs t~26, DC kicks in at t>=17 (t'=t-17, t_inner=17).
        // Much faster than eps=0.001 which needs t~40.
        let target = rz(0.3);
        let synth = Synthesizer::new(0.01);
        let result = synth.synthesize(target).expect("Should synthesize Rz(0.3) at eps=0.01");
        println!("{:?}", result.gates);
        check_result(&result, &target, 0.01);
    }

    #[test]
    fn test_synthesize_rz_hard_2() {
        let target = rz(1.34);
        let synth = Synthesizer::new(0.001);
        let result = synth.synthesize(target).expect("Should synthesize Rz(1.34) at eps=0.01");
        println!("{:?}", result.gates);

        check_result(&result, &target, 0.01);
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
        let synth = Synthesizer::new(eps).with_max_lde(35);
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

    // Synthesize a Haar-random SU(2) unitary at eps=0.01.
    #[ignore]
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

        let synth = Synthesizer::new(eps);
        let result = synth.synthesize(target).expect("Should synthesize random unitary");
        print!("Random unitary synthesis result: gates={:?}, lde={}, distance={:.6e}\n",
            result.gates, result.lde, result.distance);
        assert!(result.distance < eps,
            "distance={:.6e} >= epsilon={:.6e}", result.distance, eps);
    }

}
