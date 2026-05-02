//! Audit of `hyperprec::f256` (4-limb quad-double, ~212-bit) for the LLL +
//! Cholesky pipeline that dominates the per-prefix cost at deep ε. Goal: prove
//! that this stack-allocated Copy type can replace the heap-backed MPFR rug
//! buffers without losing the unimodularity / Cholesky-PD properties.
//!
//! Audit gate (per buddy's recommendation):
//!   1. Build Q at f256 precision, using the same anisotropic ellipsoid metric
//!      as `lenstra_heavy::build_q`.
//!   2. Run LLL with δ=0.75 on the Q-Gram metric.
//!   3. Verify det = ±1 on the resulting integer basis.
//!   4. Run Cholesky on the post-LLL Gram and verify all diagonals positive.
//!
//! Run with:
//!   cargo test --release --lib qd_audit -- --ignored --nocapture
//!
//! All arithmetic is on `MultiFloat<4>` (= f256). Operators are auto-`Copy`,
//! no scratch struct, no per-op allocation.

#![cfg(test)]

use crate::rings::Float;
use hyperprec::f256;

// ─── Type aliases for clarity ────────────────────────────────────────────────

type Mat8 = [[f256; 8]; 8];
type Vec8 = [f256; 8];
type IMat8 = [[i64; 8]; 8];

#[inline]
fn mat_zero() -> Mat8 {
    [[f256::ZERO; 8]; 8]
}

#[inline]
fn vec_zero() -> Vec8 {
    [f256::ZERO; 8]
}

#[inline]
fn identity_basis() -> IMat8 {
    std::array::from_fn(|i| {
        let mut row = [0i64; 8];
        row[i] = 1;
        row
    })
}

// ─── Σ matrix from arXiv:2510.05816 eq (3.15) ───────────────────────────────

fn sigma_matrix() -> Mat8 {
    // Pattern: 1 = +1, -1 = -1, 2 = +1/√2, -2 = -1/√2, 0 = 0
    let pattern: [[i32; 8]; 8] = [
        [1, 2, 0, -2, 0, 0, 0, 0],
        [0, 2, 1, 2, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 2, 0, -2],
        [0, 0, 0, 0, 0, 2, 1, 2],
        [1, -2, 0, 2, 0, 0, 0, 0],
        [0, -2, 1, -2, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, -2, 0, 2],
        [0, 0, 0, 0, 0, -2, 1, -2],
    ];
    let two = f256::from_f64(2.0);
    let r2 = two.sqrt().recip(); // 1/√2 in f256
    let nr2 = -r2;
    let one = f256::from_f64(1.0);
    let none = -one;
    let zero = f256::ZERO;

    let mut sigma = mat_zero();
    for i in 0..8 {
        for j in 0..8 {
            sigma[i][j] = match pattern[i][j] {
                1 => one,
                -1 => none,
                2 => r2,
                -2 => nr2,
                _ => zero,
            };
        }
    }
    sigma
}

// ─── build_q: anisotropic ellipsoid metric Q (same logic as rug build_q) ────

fn build_q(y: &[Float; 8], k: u32, eps: Float) -> Mat8 {
    let sigma = sigma_matrix();
    let two = f256::from_f64(2.0);
    let half = f256::from_f64(0.5);
    let one = f256::from_f64(1.0);

    // R² = 2^k, R = √(2^k)
    let r_sq = f256::from_f64((1u64 << k) as f64);
    let r = r_sq.sqrt();
    let eps_f = f256::from_f64(eps);

    // Δ_y = R · ε² / (2·(1 + √(1−ε²)))
    let eps_sq = eps_f * eps_f;
    let one_minus_eps_sq = one - eps_sq;
    let sqrt_1m = one_minus_eps_sq.sqrt();
    let denom = two * (one + sqrt_1m);
    let delta_y = (r * eps_sq) / denom;
    let delta_perp = r * eps_f;

    let inv_dy_sq = one / (delta_y * delta_y);
    let inv_dp_sq = one / (delta_perp * delta_perp);
    let inv_r_sq = one / r_sq;

    // y in f256, ‖y‖²
    let mut y_f = vec_zero();
    for i in 0..8 {
        y_f[i] = f256::from_f64(y[i]);
    }
    let mut y_norm_sq = f256::ZERO;
    for i in 0..8 {
        y_norm_sq = y_norm_sq + y_f[i] * y_f[i];
    }
    let inv_y_norm_sq = one / y_norm_sq;

    // ŷŷᵀ
    let mut yhat_yhat_t = mat_zero();
    for i in 0..8 {
        for j in 0..8 {
            yhat_yhat_t[i][j] = y_f[i] * y_f[j] * inv_y_norm_sq;
        }
    }

    // P_u = ½·Σ_topᵀ·Σ_top, P_{u•} = ½·Σ_botᵀ·Σ_bot
    let mut p_u = mat_zero();
    let mut p_ub = mat_zero();
    for i in 0..8 {
        for j in 0..8 {
            let mut acc_u = f256::ZERO;
            let mut acc_ub = f256::ZERO;
            for r_idx in 0..4 {
                acc_u = acc_u + sigma[r_idx][i] * sigma[r_idx][j];
                acc_ub = acc_ub + sigma[r_idx + 4][i] * sigma[r_idx + 4][j];
            }
            p_u[i][j] = acc_u * half;
            p_ub[i][j] = acc_ub * half;
        }
    }

    // Q = inv_dy_sq · ŷŷᵀ + inv_dp_sq · (P_u − ŷŷᵀ) + inv_r_sq · P_{u•}
    let mut q = mat_zero();
    for i in 0..8 {
        for j in 0..8 {
            let term1 = inv_dy_sq * yhat_yhat_t[i][j];
            let term2 = inv_dp_sq * (p_u[i][j] - yhat_yhat_t[i][j]);
            let term3 = inv_r_sq * p_ub[i][j];
            q[i][j] = term1 + term2 + term3;
        }
    }
    q
}

// ─── compute_qgram: G = B · Q · Bᵀ in f256 ──────────────────────────────────

fn compute_qgram(basis: &IMat8, q: &Mat8) -> Mat8 {
    // temp_g = Q · Bᵀ; g = B · temp_g
    let mut temp_g = mat_zero();
    for i in 0..8 {
        for j in 0..8 {
            let mut acc = f256::ZERO;
            for c in 0..8 {
                let bc = f256::from_f64(basis[j][c] as f64);
                acc = acc + q[i][c] * bc;
            }
            temp_g[i][j] = acc;
        }
    }
    let mut g = mat_zero();
    for i in 0..8 {
        for j in 0..8 {
            let mut acc = f256::ZERO;
            for c in 0..8 {
                let bc = f256::from_f64(basis[i][c] as f64);
                acc = acc + bc * temp_g[c][j];
            }
            g[i][j] = acc;
        }
    }
    g
}

// ─── GS in the Q-Gram metric ────────────────────────────────────────────────

#[derive(Default)]
struct Gs {
    mu: Mat8,
    g_star: Mat8,
    gnorm_sq: Vec8,
}

fn gs_qgram(g_lll: &Mat8) -> Gs {
    let mut mu = mat_zero();
    let mut g_star = mat_zero();
    let mut gnorm_sq = vec_zero();
    let tiny = f256::from_f64(1e-300_f64);

    for j in 0..8 {
        for i in j..8 {
            let mut acc = g_lll[i][j];
            for k in 0..j {
                acc = acc - mu[j][k] * g_star[i][k];
            }
            g_star[i][j] = acc;
        }
        gnorm_sq[j] = g_star[j][j];
        if gnorm_sq[j].abs() < tiny {
            for i in (j + 1)..8 {
                mu[i][j] = f256::ZERO;
            }
            continue;
        }
        for i in (j + 1)..8 {
            mu[i][j] = g_star[i][j] / gnorm_sq[j];
        }
    }
    Gs { mu, g_star, gnorm_sq }
}

// ─── LLL with δ=0.75 in the Q-Gram metric ───────────────────────────────────

fn lll_qgram_8(q: &Mat8) -> IMat8 {
    let mut basis = identity_basis();
    let delta_lll = f256::from_f64(0.75);
    let max_iter = 10_000usize;
    let mut iters = 0usize;
    let mut k = 1usize;

    while k < 8 && iters < max_iter {
        iters += 1;
        let g = compute_qgram(&basis, q);
        let gs = gs_qgram(&g);

        // Size reduction
        for j in (0..k).rev() {
            let r_round = gs.mu[k][j].to_f64().round() as i64;
            if r_round != 0 {
                for c in 0..8 {
                    basis[k][c] -= r_round * basis[j][c];
                }
            }
        }

        // Recompute Gram + GS
        let g = compute_qgram(&basis, q);
        let gs = gs_qgram(&g);

        // Lovász: gnorm[k] ≥ (δ − μ[k][k-1]²) · gnorm[k-1]
        let mu_sq = gs.mu[k][k - 1] * gs.mu[k][k - 1];
        let bound = (delta_lll - mu_sq) * gs.gnorm_sq[k - 1];
        if gs.gnorm_sq[k] >= bound {
            k += 1;
        } else {
            basis.swap(k, k - 1);
            k = k.saturating_sub(1).max(1);
        }
    }
    basis
}

// ─── Cholesky on the post-LLL Gram (returns L lower-triangular) ─────────────

fn cholesky_8(g: &Mat8) -> Option<Mat8> {
    let mut l = mat_zero();
    let zero = f256::ZERO;
    for i in 0..8 {
        for j in 0..=i {
            let mut acc = g[i][j];
            for k in 0..j {
                acc = acc - l[i][k] * l[j][k];
            }
            if i == j {
                if acc <= zero {
                    return None;
                }
                l[i][j] = acc.sqrt();
            } else {
                l[i][j] = acc / l[j][j];
            }
        }
    }
    Some(l)
}

// ─── det8_exact (i64 cofactor expansion, copied logic from rug version) ─────

fn det8_exact(m: &IMat8) -> Option<i64> {
    fn det_inner(matrix: &[Vec<i64>]) -> Option<i64> {
        let n = matrix.len();
        if n == 1 {
            return Some(matrix[0][0]);
        }
        if n == 2 {
            return matrix[0][0]
                .checked_mul(matrix[1][1])?
                .checked_sub(matrix[0][1].checked_mul(matrix[1][0])?);
        }
        let mut total: i64 = 0;
        for c in 0..n {
            let v = matrix[0][c];
            if v == 0 {
                continue;
            }
            let minor: Vec<Vec<i64>> = matrix[1..]
                .iter()
                .map(|row| {
                    row.iter()
                        .enumerate()
                        .filter(|(j, _)| *j != c)
                        .map(|(_, x)| *x)
                        .collect()
                })
                .collect();
            let cof = det_inner(&minor)?;
            let term = v.checked_mul(cof)?;
            if c % 2 == 0 {
                total = total.checked_add(term)?;
            } else {
                total = total.checked_sub(term)?;
            }
        }
        Some(total)
    }
    let v: Vec<Vec<i64>> = m.iter().map(|row| row.to_vec()).collect();
    det_inner(&v)
}

// ─── Tests ──────────────────────────────────────────────────────────────────

fn realistic_y(k: u32) -> [Float; 8] {
    let r2 = 1.0 / 2.0_f64.sqrt();
    let s = ((1u64 << k) as f64).sqrt() / 2.0;
    let c = 0.15_f64.cos();
    let ns = -0.15_f64.sin();
    [
        s * c,
        s * (c + ns) * r2,
        s * ns,
        s * (-c + ns) * r2,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
}

fn run_audit_at(k: u32, eps: Float) -> Result<(f64, f64, usize), String> {
    let y = realistic_y(k);
    let q = build_q(&y, k, eps);

    let basis = lll_qgram_8(&q);
    let det = det8_exact(&basis).ok_or("det8 overflow")?;
    if det != 1 && det != -1 {
        return Err(format!("non-unimodular: det={det}"));
    }

    let g = compute_qgram(&basis, &q);
    let l = cholesky_8(&g).ok_or("Cholesky failed")?;

    // L diagonals: extract f64 magnitudes for ratio comparison
    let mut min_d = f64::INFINITY;
    let mut max_d = 0.0_f64;
    for i in 0..8 {
        let d = l[i][i].to_f64();
        if !d.is_finite() || d <= 0.0 {
            return Err(format!("L[{i}][{i}]={d}"));
        }
        if d < min_d {
            min_d = d;
        }
        if d > max_d {
            max_d = d;
        }
    }
    let max_basis_entry = basis.iter().flatten().map(|&v| v.abs()).max().unwrap_or(0);
    Ok((min_d, max_d, max_basis_entry as usize))
}

#[test]
#[ignore = "qd audit; run with --ignored --nocapture"]
fn qd_audit_eps_sweep() {
    let cases: &[(Float, u32)] = &[
        (1e-3, 14),
        (1e-4, 17),
        (1e-5, 21),
        (1e-6, 25),
        (1e-7, 29),
    ];
    let mut all_pass = true;
    for &(eps, k) in cases {
        let t0 = std::time::Instant::now();
        match run_audit_at(k, eps) {
            Ok((min_d, max_d, max_b)) => {
                let elapsed_us = t0.elapsed().as_secs_f64() * 1e6;
                eprintln!(
                    "[qd audit] ε={eps:e} k={k}: OK  L_diag ∈ [{min_d:.3e}, {max_d:.3e}], \
                     ratio={:.2e}  max_basis={max_b}  {:.1}μs",
                    max_d / min_d,
                    elapsed_us
                );
            }
            Err(e) => {
                eprintln!("[qd audit] ε={eps:e} k={k}: FAIL ({e})");
                all_pass = false;
            }
        }
    }
    assert!(all_pass, "qd audit had failures — see stderr above");
}

#[test]
#[ignore = "qd audit timing; run with --ignored --nocapture"]
fn qd_audit_timing_vs_rug() {
    use crate::synthesis::lenstra_heavy::{HeavyScratch, build_q as rug_build_q, lll_qgram_8 as rug_lll, compute_qgram_inplace, cholesky_8 as rug_chol, det8_exact as rug_det8, compute_prec};

    let cases: &[(Float, u32)] = &[(1e-4, 17), (1e-5, 21), (1e-6, 25), (1e-7, 29)];
    let n_runs = 30;

    for &(eps, k) in cases {
        let y = realistic_y(k);

        let t0 = std::time::Instant::now();
        for _ in 0..n_runs {
            let q = build_q(&y, k, eps);
            let basis = lll_qgram_8(&q);
            let _ = det8_exact(&basis);
            let g = compute_qgram(&basis, &q);
            let _ = cholesky_8(&g);
        }
        let qd_us = t0.elapsed().as_secs_f64() * 1e6 / n_runs as f64;

        let prec = compute_prec(eps);
        let mut scratch = HeavyScratch::new(prec);
        let t0 = std::time::Instant::now();
        for _ in 0..n_runs {
            rug_build_q(&mut scratch, &y, k, eps);
            rug_lll(&mut scratch);
            let _ = rug_det8(&scratch.basis);
            compute_qgram_inplace(&mut scratch);
            let _ = rug_chol(&mut scratch);
        }
        let rug_us = t0.elapsed().as_secs_f64() * 1e6 / n_runs as f64;

        eprintln!(
            "[qd timing] ε={eps:e} k={k:>2} prec={prec:>3}b  f256={qd_us:>8.1}μs  rug={rug_us:>8.1}μs  speedup={:.2}×",
            rug_us / qd_us
        );
    }
}
