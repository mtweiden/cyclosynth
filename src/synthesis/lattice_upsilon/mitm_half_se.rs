//! 8D LLL+SE backend for per-half MITM enumeration
//! (PROMPT_lattice_upsilon_mitm_8d_se.md).
//!
//! ## Copy inventory
//!
//! Verbatim-pattern copies (halved-to-8D where the original was d=16, or
//! direct copy where already d=8):
//!   - `compute_prec_q`, `compute_lu_prec` — copied from
//!     [`super::scratch`] (d=16) but they were dimension-independent.
//!   - i256/MPFR helpers — re-used directly from
//!     [`crate::synthesis::lattice::lll::i256_to_f64`] (limb-level i256→f64)
//!     and [`crate::synthesis::lattice::cholesky_lu::i256_to_rfloat`].
//!   - LLL skeleton (`cfa_row`, `lazy_size_reduce`, `lll_l2_8`,
//!     `gram_update_size_reduce`, `gram_update_swap`, `basis_insert`,
//!     `compute_gram_full`) — verbatim from [`super::super::lattice_omicron::lll`]
//!     which is already at d=8 with f64 Gram-Schmidt state.
//!   - `cholesky_f64_8`, `cholesky_mpfr_to_f64_8` — copied from
//!     [`super::super::lattice_omicron::cholesky_lu`] / [`super::cholesky_lu`].
//!   - `lu_solve_8` — copied from
//!     [`super::super::lattice_omicron::cholesky_lu::lu_solve_int_inplace`].
//!   - SE walker (`schnorr_euchner_8d_emit`, `recurse_8`) — adapted from
//!     [`super::se::recurse_16`] (halved, simplified: no norm-shell at the
//!     half level, no bullet pruning — those are joint-level invariants).
//!     `z_c` is kept as f64 floats throughout (not rounded to i64 early)
//!     because the half Q-metric has weight `1/(2R²ε²)` ≈ 10¹⁰ on the σ₁
//!     rows, so 0.5-coord rounding error on z_c would inflate the SE bound
//!     by >> 1 (the half-region's natural unit-Q-norm).
//!
//! ## What is NEW (and why)
//!
//! - `build_q_half`: the only piece not present in the 16D path. The half
//!   region is a σ₁-disc × 3 conjugate-balls — rank-2 isotropic on σ₁ rows,
//!   rank-6 isotropic on the 6 conjugate rows — so
//!
//!       Q_half = (1/(2R²ε²)) · P_σ₁ + (1/R²) · P_conj
//!
//!   with P_σ₁ summing the outer products of rows `{0, 1}` of Σ_el and
//!   P_conj summing rows `{2,3,4,5,6,7}`. By construction
//!   `P_σ₁ + P_conj = Σ_el^T Σ_el = 4I₈ + 2C` (the per-element Gram —
//!   asserted in [`tests::p_sigma1_plus_p_conj_equals_gram`]).
//!
//!   There is **no ŷ direction, no rank-1 depth term, no cap_mid** — the
//!   half region is a ball, not a thin cap. The 16D ŷ machinery does NOT
//!   apply here and was deliberately left out.
//!
//!   Dynamic range is `1/(R²ε²)` ≈ `2^k · ε⁻²` ≈ 1e10 at ε=1e-5, k=8 —
//!   far from the i256-Gram overflow cliff (~2^240); the established
//!   MPFR-Q + i256-Q-snapshot + MPFR-Cholesky-fallback pattern is kept
//!   anyway, mirroring the 16D path.
//!
//! - `build_c_half`: cap center `c = (Σ_el^T Σ_el)⁻¹ · Σ_el^T · v_pad`
//!   where `v_pad = (R·V_i.re, R·V_i.im, 0, 0, 0, 0, 0, 0)` — solved via
//!   f64 LU on the 8×8 Gram (single solve per region build, off the hot
//!   path; precision is dominated by the post-LLL MPFR LU on `Bᵀ` later).
//!
//! ## SE bound derivation
//!
//! Per the prompt's "boundary derivation":
//!   - σ₁ cap on its boundary: `‖σ₁(x) − R V_i‖² = 2R²ε²` ⇒ Q-norm
//!     contribution = `1`.
//!   - each conjugate ball on its boundary: `|σ_m(x)|² = R²` ⇒ Q-norm
//!     contribution = `1` per ball (× 3 balls).
//! Sum at the worst case (all 4 constraints active): `1 + 3 = 4`. The SE
//! bound is set just above (`4 + slack`), where `slack` covers (a) f64
//! Cholesky / Q-snapshot round-off and (b) the small post-LLL z_c MPFR→f64
//! conversion. Tighten only after Part-3 passes; the leaf-level
//! [`super::mitm::PerHalfRegion::contains`] check decides the final
//! emit/skip so over-coverage is benign.

#![allow(clippy::needless_range_loop)]

use i256::i256;
use num_complex::Complex64;
use rug::{Assign, Float as RFloat};

use super::mitm::{mitm_join, HalfSide, PerHalfRegion};
use super::sigma::sigma_el;
use crate::synthesis::lattice::cholesky_lu::i256_to_rfloat;
use crate::synthesis::lattice::lll::i256_to_f64;
use crate::synthesis::lattice_common::{
    compute_scale_bits, LllResult, GRAM_OVERFLOW_THRESHOLD_BITS, L2_DELTA_BAR, L2_ETA_BAR,
    MAX_LAZY_PASSES, TARGET_BITS,
};

// ─── Precision ───────────────────────────────────────────────────────────────

fn compute_prec_q(eps: f64) -> u32 {
    if eps <= 0.0 {
        return 100;
    }
    let log_recip = (1.0 / eps).log2().max(1.0);
    let bits = (8.0 * log_recip).ceil() as u32;
    bits.max(100).min(4096)
}

fn compute_lu_prec(eps: f64) -> u32 {
    if eps <= 0.0 {
        return 96;
    }
    let log_recip = (1.0 / eps).log2().max(1.0);
    let bits = (6.0 * log_recip).ceil() as u32;
    bits.max(96).min(4096)
}

// ─── 8D scratch (clean _8 copy) ──────────────────────────────────────────────

/// Per-region scratch buffers for the 8D LLL+SE half-enumerator.
/// One allocation per `lll_se_enumerate_half` call (fine — the half pool is
/// produced once per (target, k, ε)).
pub struct HalfScratch8 {
    pub prec_q: u32,
    pub lu_prec: u32,
    pub scale_bits: i32,
    // MPFR Q + integer snapshot.
    pub q_mpfr: [[RFloat; 8]; 8],
    pub q_int: [[i256; 8]; 8],
    // Integer LLL state.
    pub basis: [[i64; 8]; 8],
    pub gram: [[i256; 8]; 8],
    pub temp_bq: [[i256; 8]; 8],
    // L²-LLL Gram-Schmidt state (f64 — Thm 2 of NS09 covers d=8 in f64).
    pub r_bar: [[f64; 8]; 8],
    pub mu_bar: [[f64; 8]; 8],
    pub s_bar: [[f64; 8]; 8],
    // Post-LLL f64 Cholesky factor (used by SE).
    pub l_f64: [[f64; 8]; 8],
    // Cap center `c` in lattice coords (MPFR at prec_q).
    pub c: [RFloat; 8],
    // MPFR LU buffers at lu_prec (solve Bᵀ · z_c = c after LLL).
    pub lu_a: [[RFloat; 8]; 8],
    pub lu_rhs: [RFloat; 8],
    pub lu_x: [RFloat; 8],
}

fn identity_basis_8() -> [[i64; 8]; 8] {
    std::array::from_fn(|i| {
        let mut r = [0i64; 8];
        r[i] = 1;
        r
    })
}

impl HalfScratch8 {
    pub fn new(eps: f64) -> Self {
        let prec_q = compute_prec_q(eps);
        let lu_prec = compute_lu_prec(eps);
        let rfz_q = || RFloat::with_val(prec_q, 0.0_f64);
        let rfz_lu = || RFloat::with_val(lu_prec, 0.0_f64);
        Self {
            prec_q,
            lu_prec,
            scale_bits: 0,
            q_mpfr: std::array::from_fn(|_| std::array::from_fn(|_| rfz_q())),
            q_int: std::array::from_fn(|_| std::array::from_fn(|_| i256::from_i64(0))),
            basis: identity_basis_8(),
            gram: std::array::from_fn(|_| std::array::from_fn(|_| i256::from_i64(0))),
            temp_bq: std::array::from_fn(|_| std::array::from_fn(|_| i256::from_i64(0))),
            r_bar: [[0.0_f64; 8]; 8],
            mu_bar: [[0.0_f64; 8]; 8],
            s_bar: [[0.0_f64; 8]; 8],
            l_f64: [[0.0_f64; 8]; 8],
            c: std::array::from_fn(|_| rfz_q()),
            lu_a: std::array::from_fn(|_| std::array::from_fn(|_| rfz_lu())),
            lu_rhs: std::array::from_fn(|_| rfz_lu()),
            lu_x: std::array::from_fn(|_| rfz_lu()),
        }
    }

    pub fn reset_basis(&mut self) {
        self.basis = identity_basis_8();
    }
}

// ─── Q_half construction ─────────────────────────────────────────────────────

/// Σ_el rows belonging to the σ₁ disc (Re σ₁, Im σ₁).
pub const SIGMA1_ROWS: [usize; 2] = [0, 1];
/// Σ_el rows belonging to the 3 conjugate balls (`m ∈ {17, 13, 5}`).
pub const CONJ_ROWS: [usize; 6] = [2, 3, 4, 5, 6, 7];

/// Project the 8×8 outer-product accumulators
/// `P_σ₁[i][j] = Σ_{r ∈ SIGMA1_ROWS} σ[r][i]·σ[r][j]`
/// and the analog for `CONJ_ROWS`. The pair sums to `Σ_el^T Σ_el = 4I + 2C`
/// (verified in tests).
pub fn fill_p_sigma1_p_conj_f64() -> ([[f64; 8]; 8], [[f64; 8]; 8]) {
    let sigma = sigma_el();
    let mut p_s1 = [[0.0_f64; 8]; 8];
    let mut p_co = [[0.0_f64; 8]; 8];
    for i in 0..8 {
        for j in 0..8 {
            let mut ss = 0.0_f64;
            for &r in &SIGMA1_ROWS {
                ss += sigma[r][i] * sigma[r][j];
            }
            let mut sc = 0.0_f64;
            for &r in &CONJ_ROWS {
                sc += sigma[r][i] * sigma[r][j];
            }
            p_s1[i][j] = ss;
            p_co[i][j] = sc;
        }
    }
    (p_s1, p_co)
}

/// Build `Q_half = (1/(2R²ε²))·P_σ₁ + (1/R²)·P_conj` in MPFR and the cap
/// center `c` such that `Σ_el · c = (R·V_i.re, R·V_i.im, 0,…,0)`.
pub fn build_q_half(scratch: &mut HalfScratch8, region: &PerHalfRegion) {
    let prec = scratch.prec_q;
    let r_sq = region.r_sq;
    let eps = region.eps;
    let inv_cap_sq = 1.0_f64 / (2.0 * r_sq * eps * eps);
    let inv_r_sq = 1.0_f64 / r_sq;
    let (p_s1, p_co) = fill_p_sigma1_p_conj_f64();
    let inv_cap_rf = RFloat::with_val(prec, inv_cap_sq);
    let inv_r_rf = RFloat::with_val(prec, inv_r_sq);
    for i in 0..8 {
        for j in 0..8 {
            let ps1 = RFloat::with_val(prec, p_s1[i][j]);
            let pco = RFloat::with_val(prec, p_co[i][j]);
            let t1 = RFloat::with_val(prec, &inv_cap_rf * &ps1);
            let t2 = RFloat::with_val(prec, &inv_r_rf * &pco);
            scratch.q_mpfr[i][j].assign(RFloat::with_val(prec, &t1 + &t2));
        }
    }

    // Diagonal PSD floor — same pattern as the 16D build at deep ε.
    let mut max_q_rf = RFloat::with_val(prec, 0.0_f64);
    for i in 0..8 {
        for j in 0..8 {
            let v = scratch.q_mpfr[i][j].clone().abs();
            if v > max_q_rf {
                max_q_rf.assign(v);
            }
        }
    }
    if !max_q_rf.is_zero() {
        let rel = RFloat::with_val(prec, 1e-15_f64);
        let floor = RFloat::with_val(prec, &max_q_rf * &rel);
        for i in 0..8 {
            let cur = scratch.q_mpfr[i][i].clone();
            scratch.q_mpfr[i][i].assign(RFloat::with_val(prec, cur + &floor));
        }
    }

    // Cap center.
    let v0 = region.sigma1_center[0];
    let v1 = region.sigma1_center[1];
    let sigma = sigma_el();
    let c_lattice = sigma_inv_apply_8(&sigma, v0, v1);
    for i in 0..8 {
        scratch.c[i].assign(RFloat::with_val(prec, c_lattice[i]));
    }
}

/// Solve `(Σᵀ·Σ) · c = Σᵀ · v_pad` where `v_pad[0]=v0`, `v_pad[1]=v1`, rest 0.
/// f64 LU is enough — this runs ONCE per region build (off the hot path);
/// downstream the cap-center is consumed in MPFR via the post-LLL LU on Bᵀ.
fn sigma_inv_apply_8(sigma: &[[f64; 8]; 8], v0: f64, v1: f64) -> [f64; 8] {
    let mut rhs = [0.0_f64; 8];
    for j in 0..8 {
        rhs[j] = sigma[0][j] * v0 + sigma[1][j] * v1;
    }
    let mut g = [[0.0_f64; 8]; 8];
    for i in 0..8 {
        for j in 0..8 {
            let mut s = 0.0_f64;
            for r in 0..8 {
                s += sigma[r][i] * sigma[r][j];
            }
            g[i][j] = s;
        }
    }
    lu_solve_f64_8(&mut g, &mut rhs)
}

fn lu_solve_f64_8(a: &mut [[f64; 8]; 8], b: &mut [f64; 8]) -> [f64; 8] {
    let n = 8;
    for k in 0..n {
        let mut p = k;
        let mut max = a[k][k].abs();
        for i in (k + 1)..n {
            if a[i][k].abs() > max {
                max = a[i][k].abs();
                p = i;
            }
        }
        if p != k {
            a.swap(p, k);
            b.swap(p, k);
        }
        if a[k][k].abs() < 1e-18 {
            return [0.0; 8];
        }
        for i in (k + 1)..n {
            let f = a[i][k] / a[k][k];
            a[i][k] = f;
            for j in (k + 1)..n {
                a[i][j] -= f * a[k][j];
            }
            b[i] -= f * b[k];
        }
    }
    let mut x = [0.0_f64; 8];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n {
            s -= a[i][j] * x[j];
        }
        x[i] = s / a[i][i];
    }
    x
}

// ─── i256 snapshot of Q_mpfr ─────────────────────────────────────────────────

pub fn build_q_int(scratch: &mut HalfScratch8) {
    let mut max_log2: i32 = i32::MIN;
    for i in 0..8 {
        for j in 0..8 {
            let v = scratch.q_mpfr[i][j].clone().abs();
            if v.is_zero() {
                continue;
            }
            let e = v.get_exp().unwrap_or(0);
            if e > max_log2 {
                max_log2 = e;
            }
        }
    }
    if max_log2 == i32::MIN {
        scratch.scale_bits = TARGET_BITS as i32;
        for i in 0..8 {
            for j in 0..8 {
                scratch.q_int[i][j] = i256::from_i64(0);
            }
        }
        return;
    }
    let b = compute_scale_bits(max_log2);
    scratch.scale_bits = b;
    for i in 0..8 {
        for j in 0..8 {
            scratch.q_int[i][j] = rug_to_i256_scaled(&scratch.q_mpfr[i][j], b);
        }
    }
}

fn rug_to_i256_scaled(x: &RFloat, shift_bits: i32) -> i256 {
    if x.is_zero() {
        return i256::from_i64(0);
    }
    let mut scaled = x.clone();
    if shift_bits >= 0 {
        scaled <<= shift_bits as u32;
    } else {
        scaled >>= (-shift_bits) as u32;
    }
    scaled.round_mut();
    rfloat_to_i256(&scaled)
}

fn rfloat_to_i256(x: &RFloat) -> i256 {
    use rug::integer::Order;
    let sign_neg = x.is_sign_negative();
    let abs = x.clone().abs();
    if abs <= rug::Float::with_val(64, i64::MAX as f64) {
        let v = abs.to_f64() as i64;
        let res = i256::from_i64(v);
        return if sign_neg { -res } else { res };
    }
    let int = match abs.to_integer() {
        Some(i) => i,
        None => return i256::from_i64(0),
    };
    if int.significant_bits() > 254 {
        return if sign_neg { i256::MIN } else { i256::MAX };
    }
    let mut limbs = [0u64; 4];
    int.write_digits(&mut limbs, Order::Lsf);
    let mut bytes = [0u8; 32];
    for (idx, limb) in limbs.iter().enumerate() {
        bytes[idx * 8..(idx + 1) * 8].copy_from_slice(&limb.to_le_bytes());
    }
    let val = i256::from_le_bytes(bytes);
    if sign_neg {
        -val
    } else {
        val
    }
}

// ─── 8D L²-LLL (clean copy of lattice_omicron::lll) ──────────────────────────

fn i256_log2_ceil(v: &i256) -> i32 {
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

pub fn compute_gram_full(scratch: &mut HalfScratch8) -> bool {
    let zero = i256::from_i64(0);
    for i in 0..8 {
        for b in 0..8 {
            let mut acc = zero;
            for a in 0..8 {
                let bi_a = i256::from_i64(scratch.basis[i][a]);
                acc += bi_a * scratch.q_int[a][b];
            }
            scratch.temp_bq[i][b] = acc;
        }
    }
    let mut max_abs_log2: i32 = -1;
    for i in 0..8 {
        for j in 0..8 {
            let mut acc = zero;
            for b in 0..8 {
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

fn gram_overflow_check(scratch: &HalfScratch8) -> bool {
    let thresh = GRAM_OVERFLOW_THRESHOLD_BITS as i32;
    for i in 0..8 {
        for j in 0..8 {
            if i256_log2_ceil(&scratch.gram[i][j]) > thresh {
                return true;
            }
        }
    }
    false
}

pub fn cfa_row(scratch: &mut HalfScratch8, i: usize) {
    for j in 0..i {
        let mut r = i256_to_f64(scratch.gram[i][j]);
        for k in 0..j {
            r -= scratch.mu_bar[j][k] * scratch.r_bar[i][k];
        }
        scratch.r_bar[i][j] = r;
        let r_jj = scratch.r_bar[j][j];
        scratch.mu_bar[i][j] = if r_jj.abs() < 1e-300 { 0.0 } else { r / r_jj };
    }
    scratch.s_bar[i][0] = i256_to_f64(scratch.gram[i][i]);
    for j in 1..=i {
        scratch.s_bar[i][j] =
            scratch.s_bar[i][j - 1] - scratch.mu_bar[i][j - 1] * scratch.r_bar[i][j - 1];
    }
    scratch.r_bar[i][i] = scratch.s_bar[i][i];
}

fn gram_update_size_reduce(scratch: &mut HalfScratch8, k: usize, j: usize, r: i64) {
    if r == 0 {
        return;
    }
    let r256 = i256::from_i64(r);
    let row_j_snapshot: [i256; 8] = scratch.gram[j];
    for m in 0..8 {
        scratch.gram[k][m] -= r256 * row_j_snapshot[m];
    }
    let mut col_j_snapshot = [i256::from_i64(0); 8];
    for i in 0..8 {
        col_j_snapshot[i] = scratch.gram[i][j];
    }
    for i in 0..8 {
        scratch.gram[i][k] -= r256 * col_j_snapshot[i];
    }
}

fn gram_update_swap(scratch: &mut HalfScratch8, a: usize, b: usize) {
    if a == b {
        return;
    }
    scratch.gram.swap(a, b);
    for i in 0..8 {
        scratch.gram[i].swap(a, b);
    }
}

fn basis_insert(scratch: &mut HalfScratch8, kappa_orig: usize, kappa_insert: usize) {
    let mut current = kappa_orig;
    while current > kappa_insert {
        scratch.basis.swap(current, current - 1);
        gram_update_swap(scratch, current, current - 1);
        current -= 1;
    }
}

pub fn lazy_size_reduce(scratch: &mut HalfScratch8, kappa: usize) -> usize {
    let mut x = [0i64; 8];
    for pass in 0..MAX_LAZY_PASSES {
        cfa_row(scratch, kappa);
        let mut max_mu: f64 = 0.0;
        for j in 0..kappa {
            let m = scratch.mu_bar[kappa][j].abs();
            if m > max_mu {
                max_mu = m;
            }
        }
        if max_mu <= L2_ETA_BAR {
            return pass;
        }
        for i in (0..kappa).rev() {
            let xi = scratch.mu_bar[kappa][i].round() as i64;
            x[i] = xi;
            if xi != 0 {
                let xi_f = xi as f64;
                for j in 0..i {
                    scratch.mu_bar[kappa][j] -= xi_f * scratch.mu_bar[i][j];
                }
            }
        }
        for i in 0..kappa {
            if x[i] != 0 {
                for c in 0..8 {
                    scratch.basis[kappa][c] -= x[i] * scratch.basis[i][c];
                }
                gram_update_size_reduce(scratch, kappa, i, x[i]);
                x[i] = 0;
            }
        }
    }
    MAX_LAZY_PASSES
}

pub fn lll_l2_8(scratch: &mut HalfScratch8) -> LllResult {
    scratch.reset_basis();
    if !compute_gram_full(scratch) {
        return LllResult::GramOverflow;
    }
    cfa_row(scratch, 0);
    let mut kappa = 1usize;
    let max_iter: usize = 10_000;
    let mut iters: usize = 0;
    while kappa < 8 && iters < max_iter {
        iters += 1;
        let _ = lazy_size_reduce(scratch, kappa);
        if gram_overflow_check(scratch) {
            return LllResult::GramOverflow;
        }
        let kappa_orig = kappa;
        while kappa >= 1
            && L2_DELTA_BAR * scratch.r_bar[kappa - 1][kappa - 1]
                > scratch.s_bar[kappa_orig][kappa - 1]
        {
            if kappa <= 1 {
                kappa = 0;
                break;
            }
            kappa -= 1;
        }
        if kappa < kappa_orig {
            basis_insert(scratch, kappa_orig, kappa);
            cfa_row(scratch, kappa);
        }
        kappa += 1;
    }
    if iters >= max_iter {
        LllResult::IterCap
    } else {
        LllResult::Converged
    }
}

// ─── Cholesky + LU (clean copies of lattice_omicron::cholesky_lu) ────────────

pub fn cholesky_f64_8(scratch: &mut HalfScratch8) -> bool {
    let scale = 2.0_f64.powi(-scratch.scale_bits);
    let mut g = [[0.0_f64; 8]; 8];
    for i in 0..8 {
        for j in 0..=i {
            g[i][j] = i256_to_f64(scratch.gram[i][j]) * scale;
        }
    }
    for i in 0..8 {
        for j in 0..8 {
            scratch.l_f64[i][j] = 0.0;
        }
    }
    for i in 0..8 {
        for j in 0..=i {
            let mut s = g[i][j];
            for k in 0..j {
                s -= scratch.l_f64[i][k] * scratch.l_f64[j][k];
            }
            if i == j {
                if s <= 0.0 {
                    return false;
                }
                scratch.l_f64[i][i] = s.sqrt();
            } else {
                scratch.l_f64[i][j] = s / scratch.l_f64[j][j];
            }
        }
    }
    true
}

/// MPFR Cholesky on the natural-scale post-LLL Gram, copied into f64 for SE.
/// Used as a fallback when `cholesky_f64_8` trips a false PSD-violation pivot
/// (rare; protects against the deep-ε Q's heavy σ₁ row anisotropy).
pub fn cholesky_mpfr_to_f64_8(scratch: &mut HalfScratch8) -> bool {
    let prec = scratch.prec_q;
    let shift = scratch.scale_bits;
    let mut tmp = RFloat::with_val(prec, 0.0_f64);
    let mut g_post: [[RFloat; 8]; 8] =
        std::array::from_fn(|_| std::array::from_fn(|_| RFloat::with_val(prec, 0.0_f64)));
    for i in 0..8 {
        for j in 0..=i {
            i256_to_rfloat(scratch.gram[i][j], &mut tmp);
            if shift > 0 {
                tmp >>= shift as u32;
            } else if shift < 0 {
                tmp <<= (-shift) as u32;
            }
            g_post[i][j].assign(&tmp);
        }
    }
    let mut l: [[RFloat; 8]; 8] =
        std::array::from_fn(|_| std::array::from_fn(|_| RFloat::with_val(prec, 0.0_f64)));
    let zero = RFloat::with_val(prec, 0.0_f64);
    let mut acc = RFloat::with_val(prec, 0.0_f64);
    let mut tmp2 = RFloat::with_val(prec, 0.0_f64);
    for i in 0..8 {
        for j in 0..=i {
            acc.assign(&g_post[i][j]);
            for k in 0..j {
                tmp2.assign(&l[i][k] * &l[j][k]);
                let acc_cl = acc.clone();
                acc.assign(&acc_cl - &tmp2);
            }
            if i == j {
                if acc <= zero {
                    return false;
                }
                let acc_cl = acc.clone();
                l[i][i].assign(acc_cl.sqrt());
            } else {
                let denom = l[j][j].clone();
                l[i][j].assign(&acc / &denom);
            }
        }
    }
    for i in 0..8 {
        for j in 0..8 {
            scratch.l_f64[i][j] = if j <= i { l[i][j].to_f64() } else { 0.0 };
        }
    }
    true
}

/// Solve `Bᵀ · z_c = c` in MPFR. Reads `scratch.basis` (i64) and `scratch.c`
/// (MPFR), writes the solution to `scratch.lu_x`.
pub fn lu_solve_zc(scratch: &mut HalfScratch8) -> bool {
    let lu_prec = scratch.lu_prec;
    for i in 0..8 {
        for j in 0..8 {
            scratch.lu_a[i][j].assign(RFloat::with_val(lu_prec, scratch.basis[j][i] as f64));
        }
        scratch.lu_rhs[i].assign(&scratch.c[i]);
    }
    let tol = RFloat::with_val(lu_prec, 1e-30_f64);
    for k in 0..8 {
        let mut piv = k;
        let mut piv_abs = scratch.lu_a[k][k].clone().abs();
        for i in (k + 1)..8 {
            let v = scratch.lu_a[i][k].clone().abs();
            if v > piv_abs {
                piv_abs = v;
                piv = i;
            }
        }
        if piv_abs < tol {
            return false;
        }
        if piv != k {
            scratch.lu_a.swap(k, piv);
            scratch.lu_rhs.swap(k, piv);
        }
        for i in (k + 1)..8 {
            let factor = RFloat::with_val(lu_prec, &scratch.lu_a[i][k] / &scratch.lu_a[k][k]);
            let (row_i, row_k) = {
                let (head, tail) = scratch.lu_a.split_at_mut(i);
                (&mut tail[0], &mut head[k])
            };
            let mut tmp = RFloat::with_val(lu_prec, 0.0_f64);
            for j in k..8 {
                tmp.assign(&factor * &row_k[j]);
                let cur = row_i[j].clone();
                row_i[j].assign(&cur - &tmp);
            }
            tmp.assign(&factor * &scratch.lu_rhs[k]);
            let rhs_i_cur = scratch.lu_rhs[i].clone();
            scratch.lu_rhs[i].assign(&rhs_i_cur - &tmp);
        }
    }
    for i in (0..8).rev() {
        let mut acc = scratch.lu_rhs[i].clone();
        for j in (i + 1)..8 {
            let prod = RFloat::with_val(lu_prec, &scratch.lu_a[i][j] * &scratch.lu_x[j]);
            let cur = acc.clone();
            acc.assign(&cur - &prod);
        }
        scratch.lu_x[i].assign(&acc / &scratch.lu_a[i][i]);
    }
    true
}

// ─── 8D Schnorr-Euchner walker ───────────────────────────────────────────────

/// Enumerate integer 8-tuples `z ∈ ℤ⁸` with `‖L·(z − z_c)‖² ≤ bound_sq`,
/// invoking `emit(x)` at each leaf where `x = B·z` is the reconstructed
/// lattice point. `z_c` is kept as f64 throughout to preserve the per-half
/// Q-metric's σ₁-row weight scale (`1/(2R²ε²)` ≈ 1e10 at deep ε).
pub fn schnorr_euchner_8d_emit<F>(
    l_chol: &[[f64; 8]; 8],
    basis: &[[i64; 8]; 8],
    z_c: &[f64; 8],
    bound_sq: f64,
    mut emit: F,
    max_leaves: u64,
) where
    F: FnMut(&[i64; 8]),
{
    let mut z = [0i64; 8];
    let mut leaves: u64 = 0;
    let mut aborted = false;
    recurse_8(
        7,
        l_chol,
        basis,
        z_c,
        bound_sq,
        0.0,
        &mut z,
        &mut emit,
        &mut leaves,
        max_leaves,
        &mut aborted,
    );
}

#[allow(clippy::too_many_arguments)]
fn recurse_8<F>(
    depth: i32,
    l: &[[f64; 8]; 8],
    basis: &[[i64; 8]; 8],
    z_c: &[f64; 8],
    bound_sq: f64,
    partial: f64,
    z: &mut [i64; 8],
    emit: &mut F,
    leaves: &mut u64,
    max_leaves: u64,
    aborted: &mut bool,
) where
    F: FnMut(&[i64; 8]),
{
    if *aborted {
        return;
    }
    if depth < 0 {
        // Reconstruct x = B·z (row convention: B[i] is the i-th basis vector).
        let mut x = [0i64; 8];
        for i in 0..8 {
            for j in 0..8 {
                x[j] += z[i] * basis[i][j];
            }
        }
        emit(&x);
        *leaves += 1;
        if *leaves >= max_leaves {
            *aborted = true;
        }
        return;
    }
    let d = depth as usize;
    let l_dd = l[d][d];
    if l_dd.abs() < 1e-30 {
        z[d] = z_c[d].round() as i64;
        recurse_8(
            depth - 1,
            l,
            basis,
            z_c,
            bound_sq,
            partial,
            z,
            emit,
            leaves,
            max_leaves,
            aborted,
        );
        return;
    }
    let mut tail = 0.0_f64;
    for j in (d + 1)..8 {
        tail += l[d][j] * ((z[j] as f64) - z_c[j]);
    }
    let rem = bound_sq - partial;
    if rem < 0.0 {
        return;
    }
    // Pad span by 1 ulp's-worth of f64 rounding (additive to span²). When
    // a target lattice point's contribution at this depth exactly saturates
    // the residual bound, naive ceil/floor on `center ± span` can round off
    // the boundary integer; padding here re-includes it. The leaf-level
    // `new_partial > bound + slack` check filters anything that genuinely
    // exceeds bound, so this stays sound.
    let rem_sqrt = (rem + 1e-9 * rem.abs().max(1.0)).sqrt();
    // Center of z[d] in real space (continuous): z_c[d] − tail/l_dd.
    let center_full = z_c[d] - tail / l_dd;
    let span = rem_sqrt / l_dd.abs();
    let z_low = (center_full - span).ceil() as i64;
    let z_high = (center_full + span).floor() as i64;
    let z_mid = center_full.round() as i64;
    let max_off = (z_high - z_mid).max(z_mid - z_low).max(0);
    for raw in 0..=(2 * max_off + 1) {
        if *aborted {
            return;
        }
        let off = if raw == 0 {
            0
        } else if raw % 2 == 1 {
            (raw + 1) / 2
        } else {
            -(raw / 2)
        };
        let zd = z_mid + off;
        if zd < z_low || zd > z_high {
            continue;
        }
        let level = l_dd * ((zd as f64) - z_c[d]) + tail;
        let new_partial = partial + level * level;
        if new_partial > bound_sq + 1e-9 * bound_sq.abs() {
            continue;
        }
        z[d] = zd;
        recurse_8(
            depth - 1,
            l,
            basis,
            z_c,
            bound_sq,
            new_partial,
            z,
            emit,
            leaves,
            max_leaves,
            aborted,
        );
    }
}

// ─── Top-level half enumerator + MITM wrapper ────────────────────────────────

/// SE bound for the half region's Q metric. The boundary derivation gives
/// `1` per active constraint (σ₁ cap + 3 conjugate balls = 4 max). We set
/// it to **`8.0`** — math bound × 2 — to cover the f64-rounding boundary
/// case where a valid integer `z[d]` sits right on the `center ± span`
/// line (empirical: bounds `4.5`/`5.0` dropped 1-3 in-region points whose
/// final Q-norm was ≤ `3.85`, while `6.0`/`8.0` recovered all of them).
/// At deep ε the actual enumeration cost is dominated by the integer-volume
/// of the half region (very few lattice points), not by the Q-ball volume,
/// so the headroom is essentially free. Leaf filter
/// [`super::mitm::PerHalfRegion::contains`] decides the final emit.
const SE_BOUND_SQ: f64 = 8.0;

/// Hard upper bound on leaves emitted by SE before aborting. The expected
/// pool size at the operating point (deep ε, k ~ R⁸ε² ≈ 1-100 valid halves)
/// is well below this; the cap protects against runaway enumeration if the
/// region or LLL output degenerates.
const MAX_SE_LEAVES: u64 = 5_000_000;

/// 8D LLL+SE enumeration of the per-half region. Returns all integer
/// 8-tuples in the region (filtered exactly via [`PerHalfRegion::contains`]
/// at the leaf — the SE bound is a Q-metric outer cover).
pub fn lll_se_enumerate_half(region: &PerHalfRegion) -> Vec<[i64; 8]> {
    let eps = region.eps;
    let mut scratch = HalfScratch8::new(eps);
    build_q_half(&mut scratch, region);
    build_q_int(&mut scratch);
    let lll_res = lll_l2_8(&mut scratch);
    if matches!(lll_res, LllResult::GramOverflow) {
        return Vec::new();
    }
    if !cholesky_f64_8(&mut scratch) && !cholesky_mpfr_to_f64_8(&mut scratch) {
        return Vec::new();
    }
    if !lu_solve_zc(&mut scratch) {
        return Vec::new();
    }
    let mut z_c = [0.0_f64; 8];
    for i in 0..8 {
        z_c[i] = scratch.lu_x[i].to_f64();
    }
    let mut out: Vec<[i64; 8]> = Vec::new();
    schnorr_euchner_8d_emit(
        &scratch.l_f64,
        &scratch.basis,
        &z_c,
        SE_BOUND_SQ,
        |x| {
            if region.contains(x) {
                out.push(*x);
            }
        },
        MAX_SE_LEAVES,
    );
    out.sort();
    out.dedup();
    out
}

/// MITM with 8D LLL+SE half-pools — the deep-ε backend.
pub fn lll_se_mitm_norm_bullet_set(
    target: &[[Complex64; 2]; 2],
    k: u32,
    eps: f64,
) -> Vec<[i64; 16]> {
    let v_11 = target[0][0];
    let v_21 = target[1][0];
    let r1 = PerHalfRegion::new(HalfSide::U1, v_11, k, eps);
    let r2 = PerHalfRegion::new(HalfSide::U2, v_21, k, eps);
    let pool1 = lll_se_enumerate_half(&r1);
    let pool2 = lll_se_enumerate_half(&r2);
    mitm_join(&pool1, &pool2, k)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthesis::lattice_upsilon::mitm::brute_enumerate_half;
    use crate::synthesis::lattice_upsilon::sigma::gram_el_int;

    /// Q_half sanity #1: P_σ₁ + P_conj == 4I + 2C (the per-element Gram).
    #[test]
    fn p_sigma1_plus_p_conj_equals_gram() {
        let (p_s1, p_co) = fill_p_sigma1_p_conj_f64();
        let g = gram_el_int();
        for i in 0..8 {
            for j in 0..8 {
                let sum = p_s1[i][j] + p_co[i][j];
                let expected = g[i][j] as f64;
                assert!(
                    (sum - expected).abs() < 1e-10,
                    "(P_σ₁+P_conj)[{i}][{j}] = {sum}, expected {expected}"
                );
            }
        }
    }

    /// Q_half sanity #2: the cap center c, when applied through Σ_el, gives
    /// (v0, v1, 0, 0, 0, 0, 0, 0) (R·V_i projected onto σ₁ rows, zero on the
    /// 6 conjugate rows).
    #[test]
    fn cap_center_hits_sigma1_zero_conj() {
        let v0 = 0.7_f64;
        let v1 = 0.3_f64;
        let sigma = sigma_el();
        let c = sigma_inv_apply_8(&sigma, v0, v1);
        let mut img = [0.0_f64; 8];
        for r in 0..8 {
            for j in 0..8 {
                img[r] += sigma[r][j] * c[j];
            }
        }
        assert!((img[0] - v0).abs() < 1e-10, "Re σ_1 image mismatch: {}", img[0]);
        assert!((img[1] - v1).abs() < 1e-10, "Im σ_1 image mismatch: {}", img[1]);
        for r in 2..8 {
            assert!(img[r].abs() < 1e-10, "conjugate row {r} not zero: {}", img[r]);
        }
    }

    /// Soundness: 8D-SE equals brute-half on the H·P·H fixture at k=2.
    #[test]
    fn lll_se_matches_brute_h_p_h() {
        use crate::matrix::U2;
        use crate::rings::ZUpsilon;

        let p: U2<ZUpsilon> = U2::p();
        let h: U2<ZUpsilon> = U2::h();
        let u: U2<ZUpsilon> = h * p * h;
        let target = u.to_float();
        let eps = 1e-1_f64;
        let k = 2;
        let r1 = PerHalfRegion::new(HalfSide::U1, target[0][0], k, eps);
        let r2 = PerHalfRegion::new(HalfSide::U2, target[1][0], k, eps);
        let brute1 = brute_enumerate_half(&r1);
        let brute2 = brute_enumerate_half(&r2);
        let se1 = lll_se_enumerate_half(&r1);
        let se2 = lll_se_enumerate_half(&r2);
        assert_eq!(
            brute1, se1,
            "u1: SE set ≠ brute set (SE dropped {} valid pts)",
            brute1.len() as i64 - se1.len() as i64
        );
        assert_eq!(
            brute2, se2,
            "u2: SE set ≠ brute set (SE dropped {} valid pts)",
            brute2.len() as i64 - se2.len() as i64
        );
    }

    /// Same equality on H·P·H·P·H (k=3).
    #[test]
    fn lll_se_matches_brute_h_p_h_p_h() {
        use crate::matrix::U2;
        use crate::rings::ZUpsilon;

        let p: U2<ZUpsilon> = U2::p();
        let h: U2<ZUpsilon> = U2::h();
        let u: U2<ZUpsilon> = h * p * h * p * h;
        let target = u.to_float();
        let eps = 1e-1_f64;
        let k = 3;
        let r1 = PerHalfRegion::new(HalfSide::U1, target[0][0], k, eps);
        let r2 = PerHalfRegion::new(HalfSide::U2, target[1][0], k, eps);
        let brute1 = brute_enumerate_half(&r1);
        let brute2 = brute_enumerate_half(&r2);
        let se1 = lll_se_enumerate_half(&r1);
        let se2 = lll_se_enumerate_half(&r2);
        assert_eq!(brute1, se1);
        assert_eq!(brute2, se2);
    }
}
