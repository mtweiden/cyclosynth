//! L²-LLL (Nguyen-Stehlé 2009) on the 16D Z[ζ_16] integer lattice.
//!
//! ## What's the same as the 8D version?
//!
//! The algorithmic skeleton: Figure 6 main loop (lazy size-reduce + Lovász
//! cascade with INSERT semantics), Figure 5 lazy size-reduction, Figure 4
//! Cholesky factorisation. Exact i256 Gram on the side; basis kept as i64.
//!
//! ## What's different at d=16?
//!
//! **MPFR Gram-Schmidt is mandatory.** Theorem 2 of Nguyen-Stehlé 2009
//! proves that f64 (ℓ=52 mantissa bits) is sufficient for L²-LLL at
//! `(δ=0.75, η=0.55)` only when `d ≤ 11`. At d=16 the proof's headroom
//! disappears (the precision requirement
//! `ℓ ≥ 5 + 2·log d − log ε + d·log ρ` grows roughly as `d·log ρ`,
//! hitting ~50 bits at d=16 ε=1e-7, leaving f64 with no margin). We use
//! MPFR `RFloat` at [`super::scratch::GS_PREC`] = 128 bits everywhere f64
//! was used in the 8D path, giving ~78 bits of margin.
//!
//! The exact integer Gram (i256) is unchanged structurally; only the
//! analysis of dimension-16 growth differs (see `super::scratch` overflow
//! analysis).
//!
//! All public LLL entry-points return [`LllResult`] indicating whether the
//! basis converged, hit Gram-overflow, or fell into the iteration cap.

#![allow(clippy::needless_range_loop)]

use i256::i256;
use rug::{Assign, Float as RFloat};

use super::scratch::{IntScratch16, GRAM_OVERFLOW_THRESHOLD_BITS};

// ─── L²-LLL parameters ───────────────────────────────────────────────────────

// ─── L²-LLL parameters & result type — moved to lattice_common ──────────────

pub use crate::synthesis::lattice_common::{
    LllResult, L2_DELTA, L2_DELTA_BAR, L2_ETA, L2_ETA_BAR, MAX_LAZY_PASSES,
};

/// Hard cap on outer L²-LLL iterations. **lattice_zeta-specific**: the 8D
/// path doesn't have an iter cap because empirically the 8D loop always
/// converges fast; the 16D loop in our regime averages ~230 iterations
/// and the cap is a safety net.
pub const MAX_LLL_ITERS: usize = 50_000;

// ─── i256 → MPFR conversion ──────────────────────────────────────────────────

/// Set `dst` to the value of i256 `v` (lossless). i256 is at most 256 bits,
/// so any GS_PREC ≥ 256 is exact; at GS_PREC=128 we get the leading 128
/// bits, which is the same precision as f64 + 75 bits — still well above
/// the L²-LLL precision requirement at d=16.
#[inline]
pub fn i256_to_rfloat_inplace(v: i256, dst: &mut RFloat) {
    super::q_metric::i256_to_rfloat(v, dst);
}

/// Bit count of |v| (≈ ⌈log₂(|v|)⌉, returns -1 for v=0).
pub(super) fn i256_log2_ceil(v: &i256) -> i32 {
    let zero = i256::from_i64(0);
    if *v == zero {
        return -1;
    }
    let abs = if *v < zero { -*v } else { *v };
    let bytes = abs.to_le_bytes();
    let mut leading_zeros: u32 = 0;
    for byte in bytes.iter().rev() {
        if *byte == 0 {
            leading_zeros += 8;
        } else {
            leading_zeros += byte.leading_zeros();
            break;
        }
    }
    (256 - leading_zeros as i32) - 1
}

/// Check whether any Gram entry exceeds the overflow threshold.
pub(super) fn gram_overflow_check(scratch: &IntScratch16) -> bool {
    let thresh = GRAM_OVERFLOW_THRESHOLD_BITS as i32;
    for i in 0..16 {
        for j in 0..16 {
            if i256_log2_ceil(&scratch.gram[i][j]) > thresh {
                return true;
            }
        }
    }
    false
}

// ─── Cholesky Factorization Algorithm (Figure 4) — MPFR ─────────────────────

/// Row-at-a-time CFA (Figure 4 of Nguyen-Stehlé 2009) at MPFR precision.
/// Computes `r_bar[i][*]`, `mu_bar[i][*]`, `s_bar[i][*]` given rows 0..i are
/// already populated. Reads gram entries via `i256_to_rfloat`.
pub fn cfa_row(scratch: &mut IntScratch16, i: usize) {
    let prec = scratch.gs_prec;

    // Off-diagonal entries: j = 0..i-1
    for j in 0..i {
        // r̄_{i,j} = <b_i, b_j> from i256 Gram.
        i256_to_rfloat_inplace(scratch.gram[i][j], &mut scratch.r_bar[i][j]);
        // r̄_{i,j} -= Σ_{k<j} μ̄_{j,k} · r̄_{i,k}.
        for k in 0..j {
            // tmp = μ̄_{j,k} · r̄_{i,k}
            scratch
                .tmp_a
                .assign(&scratch.mu_bar[j][k] * &scratch.r_bar[i][k]);
            scratch.tmp_b.assign(&scratch.r_bar[i][j] - &scratch.tmp_a);
            scratch.r_bar[i][j].assign(&scratch.tmp_b);
        }
        // μ̄_{i,j} = r̄_{i,j} / r̄_{j,j}, with degenerate-case guard.
        let r_jj_f = scratch.r_bar[j][j].to_f64();
        if r_jj_f.abs() < 1e-300 {
            scratch.mu_bar[i][j].assign(0.0_f64);
        } else {
            scratch.mu_bar[i][j].assign(&scratch.r_bar[i][j] / &scratch.r_bar[j][j]);
        }
    }

    // Diagonal: s̄_{i,*} sequence, r̄_{i,i} = s̄_{i,i}.
    i256_to_rfloat_inplace(scratch.gram[i][i], &mut scratch.s_bar[i][0]);
    for j in 1..=i {
        // s̄_{i,j} = s̄_{i,j-1} - μ̄_{i,j-1} · r̄_{i,j-1}.
        scratch
            .tmp_a
            .assign(&scratch.mu_bar[i][j - 1] * &scratch.r_bar[i][j - 1]);
        scratch
            .tmp_b
            .assign(&scratch.s_bar[i][j - 1] - &scratch.tmp_a);
        scratch.s_bar[i][j].assign(&scratch.tmp_b);
    }
    let _ = prec;
    scratch.r_bar[i][i].assign(&scratch.s_bar[i][i]);
}

/// Run CFA for ALL rows 0..16.
pub fn cfa_full(scratch: &mut IntScratch16) {
    for i in 0..16 {
        cfa_row(scratch, i);
    }
}

// ─── Incremental Gram update for size-reduce + swap ──────────────────────────

/// Apply `b_k -= r·b_j` to the i256 Gram in O(d) ops.
/// Math: `B_new = M·B` where `M = I − r·E_kj`, hence `G_new = M·G·Mᵀ`.
pub(super) fn gram_update_size_reduce(scratch: &mut IntScratch16, k: usize, j: usize, r: i64) {
    if r == 0 {
        return;
    }
    let r256 = i256::from_i64(r);
    // Step 1: row k.
    let row_j_snapshot: [i256; 16] = scratch.gram[j];
    for m in 0..16 {
        scratch.gram[k][m] -= r256 * row_j_snapshot[m];
    }
    // Step 2: column k.
    let mut col_j_snapshot = [i256::from_i64(0); 16];
    for i in 0..16 {
        col_j_snapshot[i] = scratch.gram[i][j];
    }
    for i in 0..16 {
        scratch.gram[i][k] -= r256 * col_j_snapshot[i];
    }
}

/// Swap rows a and b in the (symmetric) Gram: swap rows AND columns.
pub(super) fn gram_update_swap(scratch: &mut IntScratch16, a: usize, b: usize) {
    if a == b {
        return;
    }
    scratch.gram.swap(a, b);
    for i in 0..16 {
        scratch.gram[i].swap(a, b);
    }
}

/// L² INSERT operation: move basis row `kappa_orig` to `kappa_insert`.
pub(super) fn basis_insert(scratch: &mut IntScratch16, kappa_orig: usize, kappa_insert: usize) {
    debug_assert!(kappa_insert <= kappa_orig);
    let mut current = kappa_orig;
    while current > kappa_insert {
        scratch.basis.swap(current, current - 1);
        gram_update_swap(scratch, current, current - 1);
        current -= 1;
    }
}

// ─── Full Gram computation: G = B · Q_int · Bᵀ ───────────────────────────────

/// Compute G = B · Q_int · Bᵀ entirely in i256, into `scratch.gram`. Uses
/// `scratch.temp_bq` as intermediate (= B · Q_int). Returns `false` if any
/// Gram entry exceeds `2^GRAM_OVERFLOW_THRESHOLD_BITS`.
pub fn compute_gram_full(scratch: &mut IntScratch16) -> bool {
    let zero = i256::from_i64(0);

    // temp_bq[i][b] = sum_a B[i][a] · Q_int[a][b]
    for i in 0..16 {
        for b in 0..16 {
            let mut acc = zero;
            for a in 0..16 {
                let bi_a = i256::from_i64(scratch.basis[i][a]);
                acc += bi_a * scratch.q_int[a][b];
            }
            scratch.temp_bq[i][b] = acc;
        }
    }

    // gram[i][j] = sum_b temp_bq[i][b] · B[j][b]
    let mut max_abs_log2: i32 = -1;
    for i in 0..16 {
        for j in 0..16 {
            let mut acc = zero;
            for b in 0..16 {
                let bj_b = i256::from_i64(scratch.basis[j][b]);
                acc += scratch.temp_bq[i][b] * bj_b;
            }
            scratch.gram[i][j] = acc;
            let bits = i256_log2_ceil(&acc);
            if bits > max_abs_log2 {
                max_abs_log2 = bits;
            }
        }
    }
    max_abs_log2 <= GRAM_OVERFLOW_THRESHOLD_BITS as i32
}

// ─── Lazy size-reduce (Figure 5) — MPFR ──────────────────────────────────────

/// Lazy floating-point size-reduction at MPFR precision.
///
/// Reduces row κ against rows 0..κ-1 such that `|μ̄_{κ,j}| ≤ η̄` for all
/// `j < κ`, where η̄ = (η + 1/2) / 2. Operates iteratively: each pass
/// computes CFA for row κ, predicts X_i = round(μ̄_{κ,i}), updates μ̄_{κ,j}
/// predictively, then applies the basis transform `b_κ -= Σ X_i b_i` and
/// updates the i256 Gram. Repeats until convergence.
pub fn lazy_size_reduce(scratch: &mut IntScratch16, kappa: usize) -> usize {
    let mut x = [0i64; 16];

    for pass in 0..MAX_LAZY_PASSES {
        // Step 2: compute CFA for row κ.
        cfa_row(scratch, kappa);

        // Step 3: convergence check on max(|μ̄_{κ,j}|).
        let mut max_mu: f64 = 0.0;
        for j in 0..kappa {
            let m = scratch.mu_bar[kappa][j].to_f64().abs();
            if m > max_mu {
                max_mu = m;
            }
        }
        if max_mu <= L2_ETA_BAR {
            if crate::synthesis::diag::trace_enabled() {
                crate::synthesis::diag::record_lazy_passes((pass + 1) as u64);
            }
            return pass;
        }

        // Steps 4-5: compute X_i descending from κ-1 to 0, predictively
        // shrinking μ̄_{κ,j} as we go.
        for i in (0..kappa).rev() {
            let xi = scratch.mu_bar[kappa][i].to_f64().round() as i64;
            x[i] = xi;
            if xi != 0 {
                let xi_f = rug::Float::with_val(scratch.gs_prec, xi as f64);
                for j in 0..i {
                    // μ̄_{κ,j} -= xi · μ̄_{i,j}
                    scratch.tmp_a.assign(&xi_f * &scratch.mu_bar[i][j]);
                    scratch
                        .tmp_b
                        .assign(&scratch.mu_bar[kappa][j] - &scratch.tmp_a);
                    scratch.mu_bar[kappa][j].assign(&scratch.tmp_b);
                }
            }
        }

        // Step 6: apply basis update + Gram update for each non-zero X_i.
        for i in 0..kappa {
            if x[i] != 0 {
                for c in 0..16 {
                    scratch.basis[kappa][c] -= x[i] * scratch.basis[i][c];
                }
                gram_update_size_reduce(scratch, kappa, i, x[i]);
                x[i] = 0;
            }
        }
    }
    if crate::synthesis::diag::trace_enabled() {
        crate::synthesis::diag::record_lazy_passes(MAX_LAZY_PASSES as u64);
    }
    MAX_LAZY_PASSES
}

// ─── L²-LLL main loop (Figure 6) ─────────────────────────────────────────────

/// Run L²-LLL on the 16×16 Q-metric Gram already snapshotted into
/// `scratch.gram`. Builds an LLL-reduced basis recorded in `scratch.basis`;
/// MPFR Gram-Schmidt state lives in `scratch.r_bar`/`mu_bar`/`s_bar` and
/// the i256 `gram`.
///
/// **Precondition:** `scratch.basis` is the identity, and `scratch.q_int`
/// holds the integer-scaled Q. The caller (or [`run_lll_16`]) is responsible
/// for invoking [`compute_gram_full`] before this function.
pub fn lll_l2_16(scratch: &mut IntScratch16) -> LllResult {
    let max_iter = MAX_LLL_ITERS;
    let mut iters: usize = 0;

    // Step 2: initialize r̄_{0,0} = ‖b_0‖² (CFA on row 0).
    cfa_row(scratch, 0);
    let mut kappa = 1usize;

    // Pre-allocate scratch for the Lovász test.
    let delta_bar = RFloat::with_val(scratch.gs_prec, L2_DELTA_BAR);
    let mut lhs = RFloat::with_val(scratch.gs_prec, 0.0);

    while kappa < 16 && iters < max_iter {
        iters += 1;

        // Step 3: lazy size-reduce row κ.
        let _passes = lazy_size_reduce(scratch, kappa);

        if gram_overflow_check(scratch) {
            if crate::synthesis::diag::trace_enabled() {
                crate::synthesis::diag::record_lll_iters(iters as u64, max_iter as u64);
            }
            return LllResult::GramOverflow;
        }

        // Step 4: Lovász cascade. δ̄ · r̄_{κ-1,κ-1} > s̄_{κ_orig}^{(κ-1)} ?
        let kappa_orig = kappa;
        loop {
            if kappa == 0 {
                break;
            }
            // lhs = δ̄ · r̄_{κ-1,κ-1}
            lhs.assign(&delta_bar * &scratch.r_bar[kappa - 1][kappa - 1]);
            // rhs = s̄_{κ_orig,κ-1}
            // Compare via subtraction sign in MPFR (lossless under GS_PREC).
            let cmp = lhs.partial_cmp(&scratch.s_bar[kappa_orig][kappa - 1]);
            let lhs_greater = matches!(cmp, Some(std::cmp::Ordering::Greater));
            if !lhs_greater {
                break;
            }
            if kappa <= 1 {
                kappa = 0;
                break;
            }
            kappa -= 1;
        }

        // If insertion position is shallower than the original frontier,
        // rotate the basis (and Gram) so the reduced row lands at kappa.
        if kappa < kappa_orig {
            basis_insert(scratch, kappa_orig, kappa);
            cfa_row(scratch, kappa);
        }
        kappa += 1;
    }

    if crate::synthesis::diag::trace_enabled() {
        crate::synthesis::diag::record_lll_iters(iters as u64, max_iter as u64);
    }
    if iters >= max_iter {
        LllResult::IterCap
    } else {
        LllResult::Converged
    }
}

/// Convenience: prepare Gram from current basis and Q_int, then run LLL.
/// Honours `scratch.warm_lll` — when true, the caller-supplied basis is
/// reused as the LLL starting point (Z1 D&C amortisation); otherwise we
/// reset to identity (default single-search behaviour).
pub fn run_lll_16(scratch: &mut IntScratch16) -> LllResult {
    if !scratch.warm_lll {
        scratch.reset_basis();
    }
    if !compute_gram_full(scratch) {
        return LllResult::GramOverflow;
    }
    lll_l2_16(scratch)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::q_metric::{build_q_int_zeta, build_q_mpfr_zeta};
    use super::super::scratch::IntScratch16;
    use super::*;

    /// Compute the determinant of a 16×16 i64 matrix exactly via Bareiss
    /// in i256. Returns `None` on i64 overflow at the end (shouldn't happen
    /// for a unimodular post-LLL basis).
    fn det16_exact(m: &[[i64; 16]; 16]) -> Option<i64> {
        let mut a: [[i256; 16]; 16] =
            std::array::from_fn(|i| std::array::from_fn(|j| i256::from_i64(m[i][j])));
        let mut sign: i64 = 1;
        let mut prev = i256::from_i64(1);
        let zero = i256::from_i64(0);

        for k in 0..16 {
            if a[k][k] == zero {
                let mut found = false;
                for i in (k + 1)..16 {
                    if a[i][k] != zero {
                        a.swap(k, i);
                        sign = -sign;
                        found = true;
                        break;
                    }
                }
                if !found {
                    return Some(0);
                }
            }
            let pivot = a[k][k];
            for i in (k + 1)..16 {
                for j in (k + 1)..16 {
                    let lhs = a[i][j] * pivot;
                    let rhs = a[i][k] * a[k][j];
                    a[i][j] = (lhs - rhs) / prev;
                }
                a[i][k] = zero;
            }
            prev = pivot;
        }
        let det = a[15][15];
        let det_signed = if sign < 0 { -det } else { det };
        let lo = det_signed.as_i128();
        if lo >= i64::MIN as i128 && lo <= i64::MAX as i128 {
            Some(lo as i64)
        } else {
            None
        }
    }

    fn realistic_v() -> [f64; 4] {
        // Generic (non-canonical-axis) 4-direction normalized.
        let v = [0.5, 0.3, 0.7, -0.4];
        let n: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        std::array::from_fn(|i| v[i] / n)
    }

    fn run_for(eps: f64, k: u32) -> (LllResult, IntScratch16) {
        let v = realistic_v();
        let mut s = IntScratch16::new(eps);
        build_q_mpfr_zeta(&mut s, v, k, eps);
        build_q_int_zeta(&mut s);
        let r = run_lll_16(&mut s);
        (r, s)
    }

    #[test]
    fn lll16_unimodular_eps_1e_3() {
        let (r, s) = run_for(1e-3, 6);
        assert_eq!(r, LllResult::Converged, "LLL did not converge at ε=1e-3");
        let det = det16_exact(&s.basis).expect("det16 overflow");
        assert!(det == 1 || det == -1, "non-unimodular: det={}", det);
    }

    #[test]
    fn lll16_unimodular_eps_1e_5() {
        let (r, s) = run_for(1e-5, 14);
        assert!(
            matches!(r, LllResult::Converged | LllResult::IterCap),
            "unexpected result: {:?}",
            r
        );
        if !matches!(r, LllResult::GramOverflow) {
            let det = det16_exact(&s.basis).expect("det16 overflow");
            assert!(det == 1 || det == -1, "non-unimodular: det={}", det);
        }
    }

    #[test]
    fn lll16_size_reduction_invariant() {
        // After LLL, |μ̄_{i,j}| ≤ η for all i > j.
        let (_, mut s) = run_for(1e-4, 10);
        cfa_full(&mut s);
        for i in 1..16 {
            for j in 0..i {
                let m = s.mu_bar[i][j].to_f64().abs();
                assert!(
                    m <= L2_ETA + 1e-9,
                    "size-reduction violated: |μ̄[{}][{}]| = {} > η = {}",
                    i,
                    j,
                    m,
                    L2_ETA
                );
            }
        }
    }

    #[test]
    fn lll16_lovasz_invariant() {
        // δ · r̄_{κ-1,κ-1} ≤ s̄_{κ-1,κ-1}^{(κ)} = s̄[κ][κ-1] for κ = 1..16.
        let (_, mut s) = run_for(1e-4, 10);
        cfa_full(&mut s);
        for kappa in 1..16 {
            let lhs = L2_DELTA * s.r_bar[kappa - 1][kappa - 1].to_f64();
            let rhs = s.s_bar[kappa][kappa - 1].to_f64();
            assert!(
                lhs <= rhs + 1e-9 * rhs.abs().max(1.0),
                "Lovász violated at κ={}: δ·r̄_{}={} > s̄_{}^{}_={}",
                kappa,
                kappa - 1,
                lhs,
                kappa - 1,
                kappa,
                rhs
            );
        }
    }

    #[test]
    fn lll16_first_basis_vector_q_norm_small() {
        // Theorem 2 of Lenstra-Lenstra-Lovász: ‖b_1‖² ≤ (4/3)^(d-1) · λ_1²
        // where λ_1 is the shortest Q-norm in the lattice. We don't have λ_1
        // a priori, but we can check that b_1 is *much* smaller than a
        // typical row of the original basis (which is just an identity row).
        // The original Q[i][i] for i=0..15 is the diagonal; a typical entry
        // is dominated by 1/Δ_y² (the cap-radial term, ~10^12 at ε=1e-3).
        // After LLL the first basis vector should have Q-norm at the lattice
        // shortest-vector scale, which is many orders of magnitude smaller.
        let (r, s) = run_for(1e-3, 6);
        assert_eq!(r, LllResult::Converged);
        // Q-norm² of b_0 = (b_0)ᵀ · Q · b_0 = G[0][0] / 2^scale_bits.
        // We compare to the smallest diagonal of the original Q (which would
        // be the first basis vector if no reduction had happened).
        let mut min_orig_diag = f64::INFINITY;
        for i in 0..16 {
            let d = s.q_mpfr[i][i].to_f64();
            if d < min_orig_diag {
                min_orig_diag = d;
            }
        }
        // Convert G[0][0] back to natural Q-scale.
        let scale = 2.0f64.powi(-s.scale_bits);
        let mut g00_rf = rug::Float::with_val(s.gs_prec, 0.0);
        super::super::q_metric::i256_to_rfloat(s.gram[0][0], &mut g00_rf);
        let g00 = g00_rf.to_f64() * scale;
        // The reduced first vector should have Q-norm ≤ min original diag
        // (if it's larger, the LLL hasn't reduced anything — bug). At d=16
        // the actual ratio is much better; we keep this conservative.
        assert!(
            g00 <= min_orig_diag * 1.05,
            "b_0 Q-norm {} > smallest original diag {} (× 1.05): no reduction!",
            g00,
            min_orig_diag
        );
    }

    #[test]
    fn lll16_basis_recovers_original_lattice() {
        // The basis B is unimodular, so it spans the same lattice as the
        // identity. The fact that det = ±1 already proves this — we test
        // that explicitly here as a sanity check that the basis hasn't
        // accidentally been zeroed or duplicated.
        let (r, s) = run_for(1e-3, 6);
        assert_eq!(r, LllResult::Converged);
        // No row should be identically zero.
        for i in 0..16 {
            let nz = s.basis[i].iter().any(|&v| v != 0);
            assert!(nz, "row {} of LLL basis is all zero", i);
        }
        // No two rows should be identical.
        for i in 0..16 {
            for j in (i + 1)..16 {
                assert!(
                    s.basis[i] != s.basis[j],
                    "rows {} and {} of LLL basis are identical",
                    i,
                    j
                );
            }
        }
    }

    /// Performance smoke: at k=8 (where brute-force k=8 is intractable —
    /// the norm-shell `r_16(2^8) ~ 10¹³` points), the LLL+CFA pass should
    /// finish in well under a second. Together with the upcoming M4 SE
    /// (Q-bound pruned walk) the full pipeline should complete in low
    /// seconds at moderate ε.
    #[test]
    fn lll16_perf_at_k_8_completes() {
        let start = std::time::Instant::now();
        let v = realistic_v();
        let mut s = IntScratch16::new(1e-3);
        build_q_mpfr_zeta(&mut s, v, 8, 1e-3);
        build_q_int_zeta(&mut s);
        let r = run_lll_16(&mut s);
        let elapsed = start.elapsed();
        eprintln!("lll16_perf_at_k_8: {:?}, elapsed={:?}", r, elapsed);
        assert!(
            matches!(r, LllResult::Converged | LllResult::IterCap),
            "unexpected at k=8: {:?}",
            r
        );
        // 5 seconds is a generous budget; the actual 16D LLL with MPFR
        // GS should finish in ~100-500ms.
        assert!(
            elapsed.as_secs() < 5,
            "LLL at k=8 took {:?}; budget was 5s",
            elapsed
        );
    }

    /// Deeper-ε smoke test: at ε=1e-5, k=14, LLL should still converge in
    /// reasonable time. `prec_q` scales linearly with `log(1/ε)` so MPFR
    /// ops slow ~3× from ε=1e-3 to ε=1e-5.
    #[test]
    fn lll16_perf_at_eps_1e_5_completes() {
        let start = std::time::Instant::now();
        let v = realistic_v();
        let mut s = IntScratch16::new(1e-5);
        build_q_mpfr_zeta(&mut s, v, 14, 1e-5);
        build_q_int_zeta(&mut s);
        let r = run_lll_16(&mut s);
        let elapsed = start.elapsed();
        eprintln!("lll16_perf_at_eps_1e_5: {:?}, elapsed={:?}", r, elapsed);
        assert!(
            matches!(r, LllResult::Converged | LllResult::IterCap),
            "unexpected at ε=1e-5: {:?}",
            r
        );
        // 30 seconds is a generous budget for deeper ε.
        assert!(
            elapsed.as_secs() < 30,
            "LLL at ε=1e-5 took {:?}; budget was 30s",
            elapsed
        );
    }

    /// Build Q at a random direction and confirm LLL converges with
    /// unimodular output. Stress-test across multiple random targets.
    #[test]
    fn lll16_random_directions_unimodular() {
        use rand::Rng;
        let mut rng = rand::rng();
        for _ in 0..5 {
            let mut v = [0.0f64; 4];
            for x in v.iter_mut() {
                *x = rng.random::<f64>() * 2.0 - 1.0;
            }
            let n: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            for x in v.iter_mut() {
                *x /= n;
            }
            let mut s = IntScratch16::new(1e-3);
            build_q_mpfr_zeta(&mut s, v, 6, 1e-3);
            build_q_int_zeta(&mut s);
            let r = run_lll_16(&mut s);
            assert!(
                matches!(r, LllResult::Converged | LllResult::IterCap),
                "unexpected at v={:?}: {:?}",
                v,
                r
            );
            if !matches!(r, LllResult::GramOverflow) {
                let det = det16_exact(&s.basis).expect("det16 overflow");
                assert!(
                    det == 1 || det == -1,
                    "non-unimodular at v={:?}: det={}",
                    v,
                    det
                );
            }
        }
    }
}
