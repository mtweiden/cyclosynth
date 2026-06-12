//! Brute shell cache + f64 distance prefilter for k ≤ BRUTE_LIMIT.

use super::*;

/// k cutoff: brute-force handles `k ≤ BRUTE_LIMIT`, lattice handles the rest.
/// At 3, brute tops out at ~10⁷ shell points (~100 ms).
pub(crate) const BRUTE_LIMIT: u32 = 3;

/// Process-wide cache over [`enumerate_unitary_norm_shell`]: the shell enumeration is a
/// pure function of `k`, and optimal mode would otherwise re-run it 4×
/// per target. The cached unit-scale d = 0 float matrices
/// `(u1, −u2*, u2, u1*)/√2^k` let per-target scans use the cheap f64
/// prefilter [`brute_dist_est`] instead of MPFR distance on every shell
/// solution; accept/reject still goes through the exact MPFR path, so
/// decisions are bit-identical to the uncached scan.
pub(crate) struct BruteShell {
    pub(crate) sols: Vec<[i64; 16]>,
    pub(crate) mats: Vec<[Complex64; 4]>,
}

pub(crate) fn brute_shell_cached(k: u32) -> &'static BruteShell {
    use std::sync::OnceLock;
    const CELL: OnceLock<BruteShell> = OnceLock::new();
    static CACHE: [OnceLock<BruteShell>; (BRUTE_LIMIT + 1) as usize] =
        [CELL; (BRUTE_LIMIT + 1) as usize];
    debug_assert!(k <= BRUTE_LIMIT);
    CACHE[k as usize].get_or_init(|| {
        let sols = enumerate_unitary_norm_shell(k);
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
pub(crate) fn brute_dist_est(m: &[Complex64; 4], zd: Complex64, target: &Mat2) -> f64 {
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
pub(crate) fn brute_prefilter_threshold(epsilon: f64) -> f64 {
    1.05 * epsilon + 1e-11
}

