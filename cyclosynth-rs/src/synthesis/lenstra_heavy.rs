//! 8D output-sensitive integer enumeration for Clifford+T synthesis (Algorithm 3.6
//! from arXiv:2510.05816), with `rug` (MPFR) at adaptive precision for the LLL +
//! Cholesky setup phase.
//!
//! Pipeline per phase1 call:
//!  1. Build anisotropic ellipsoid metric Q (8×8 SPD) bounding the cap × ball body,
//!     in MPFR at `prec ≈ 8·log₂(1/ε)` bits (with floor 100, generous safety
//!     margin over the empirical `4·log₂(1/ε)+30` minimum).
//!  2. LLL-reduce ℤ⁸ identity basis using Q as the inner product (MPFR).
//!  3. Cholesky factor G_LLL = B_LLL · Q · B_LLLᵀ = L Lᵀ (MPFR).
//!  4. Solve B_LLLᵀ · z_c = c for the cap-center in lattice coordinates (MPFR LU
//!     with partial pivoting).
//!  5. Schnorr-Euchner enumerate z ∈ ℤ⁸ with ‖Lᵀ·(z − z_c)‖² ≤ 1.51 (f64).
//!  6. For each candidate, reconstruct x = B_LLL · z (i64 exact), check
//!     ‖x‖² == 2^k AND B(x) == 0 AND |y·x|² ≥ thresh_xy.
//!
//! All rug operations use the in-place `Float::assign(&_ + &_)` (Incomplete +
//! Assign) pattern via the `r_*!` macros below. Combined with a per-thread
//! pre-allocated `HeavyScratch` (allocated once via rayon's `map_init`), this
//! avoids per-op heap allocations that would otherwise cause severe global-
//! allocator contention on the 8-thread parallel iterator over MA prefixes.

#![allow(dead_code)]

use crate::rings::Float;
use rug::{Assign, Float as RFloat};
use std::sync::atomic::AtomicBool;
use twofloat::TwoFloat as Tf;

// ─── Types ────────────────────────────────────────────────────────────────────

type IMat8 = [[i64; 8]; 8];
type RMat8 = [[RFloat; 8]; 8];
type RVec8 = [RFloat; 8];

#[inline]
fn rfz(prec: u32) -> RFloat {
    RFloat::with_val(prec, 0.0_f64)
}

#[inline]
fn rfv(prec: u32, x: f64) -> RFloat {
    RFloat::with_val(prec, x)
}

fn rmat_zero(prec: u32) -> RMat8 {
    std::array::from_fn(|_| std::array::from_fn(|_| rfz(prec)))
}

fn rvec_zero(prec: u32) -> RVec8 {
    std::array::from_fn(|_| rfz(prec))
}

// ─── Adaptive precision ───────────────────────────────────────────────────────

/// Precision in bits for `rug::Float` ops at this ε. Empirical minimum found
/// in the precision audit is ≈ 4·log₂(1/ε)+30; we use 8·log₂(1/ε) with floor 100
/// for a 2× safety margin (still ~0.5–1 ms per LLL setup at ε=1e-7 vs. the
/// hot path's ~5 µs per prefix at ε=1e-4 — a tiny absolute slowdown).
pub fn compute_prec(eps: Float) -> u32 {
    let log_recip = (1.0 / eps).log2().max(1.0);
    let bits = (8.0 * log_recip).ceil() as u32;
    bits.max(100)
}

/// Reduced precision for the first attempt of the adaptive Heavy pipeline.
/// 100-bit floor keeps the Stehlé-Pujol bound for ε ≥ 1e-6 while staying
/// within the 2-limb (128-bit storage) MPFR tier — same per-op cost band as
/// the floor, ~2× cheaper than the 3-limb tier that `compute_prec` reaches at
/// ε=1e-5. Failures (det/Cholesky/LU or SE blowup) escalate to `compute_prec`.
///
/// Bit-counts (for reference):
///   ε=1e-4 → 100  (full: 107)
///   ε=1e-5 → 100  (full: 134)   ← key savings here (2 limbs vs 3)
///   ε=1e-6 → 100  (full: 160)
///   ε=1e-7 → 117  (full: 187)
pub fn compute_prec_low(eps: Float) -> u32 {
    let log_recip = (1.0 / eps).log2().max(1.0);
    let bits = (5.0 * log_recip).ceil() as u32;
    bits.max(100)
}

/// SE callback-count threshold. If the inner loop evaluates more leaves than
/// this without finding a solution, treat the LLL/Cholesky setup as having
/// silently lost orthogonalization precision (fat-ellipsoid signature) and
/// signal escalation to full precision. Tuned empirically: healthy SE walks
/// at ε=1e-5 visit ~50–500 leaves.
pub const SE_ESCALATE_THRESHOLD: u64 = 5_000;

// ─── In-place rug op macros ───────────────────────────────────────────────────
//
// `&a OP &b` returns a `_Incomplete<'_>`; assigning that to a `&mut Float`
// resolves it directly into the destination, allocating zero new MPFR objects.
// These macros encode that pattern uniformly so the call sites stay readable.

macro_rules! r_mul {
    ($dst:expr, $a:expr, $b:expr) => {
        $dst.assign(&$a * &$b)
    };
}
macro_rules! r_add {
    ($dst:expr, $a:expr, $b:expr) => {
        $dst.assign(&$a + &$b)
    };
}
macro_rules! r_sub {
    ($dst:expr, $a:expr, $b:expr) => {
        $dst.assign(&$a - &$b)
    };
}
macro_rules! r_div {
    ($dst:expr, $a:expr, $b:expr) => {
        $dst.assign(&$a / &$b)
    };
}

/// Helper to satisfy the borrow checker when an inner-loop op needs distinct
/// rows of the same matrix as immutable + mutable references simultaneously.
fn two_rows_mut(matrix: &mut RMat8, i: usize, j: usize) -> (&mut [RFloat; 8], &mut [RFloat; 8]) {
    debug_assert!(i != j);
    if i < j {
        let (head, tail) = matrix.split_at_mut(j);
        (&mut head[i], &mut tail[0])
    } else {
        let (head, tail) = matrix.split_at_mut(i);
        (&mut tail[0], &mut head[j])
    }
}

// ─── HeavyScratch: per-thread pre-allocated working buffers ────────────────

/// All MPFR working storage needed for one `phase1_lenstra` call. Allocated
/// once per rayon worker via `map_init`, then reused across all prefixes that
/// worker handles. Every `RFloat` is created with `prec` bits up front; no
/// allocation happens inside the LLL inner loop.
pub struct HeavyScratch {
    pub prec: u32,

    // Inputs (filled in by build_q / build_center)
    pub q: RMat8,
    pub c: RVec8,

    // Constants computed once (depend only on prec)
    pub sigma: RMat8,
    pub one: RFloat,
    pub two: RFloat,
    pub half: RFloat,

    // Reusable temporaries
    pub tmp: RFloat,
    pub tmp2: RFloat,
    pub tmp3: RFloat,
    pub acc: RFloat,

    // LLL working buffers
    pub basis: IMat8,
    pub g_lll: RMat8,
    pub temp_g: RMat8, // used for compute_qgram intermediate
    pub mu: RMat8,
    pub g_star: RMat8,
    pub gnorm_sq: RVec8,
    pub delta_lll: RFloat,

    // Cholesky factor L (lower-triangular)
    pub l: RMat8,

    // Q construction temporaries
    pub p_u: RMat8,
    pub p_ub: RMat8,
    pub yhat_yhat_t: RMat8,
    pub y_rf: RVec8,
    pub eps_rf: RFloat,
    pub r: RFloat,
    pub r_sq: RFloat,
    pub delta_y: RFloat,
    pub delta_perp: RFloat,
    pub inv_dy_sq: RFloat,
    pub inv_dp_sq: RFloat,
    pub inv_r_sq: RFloat,
    pub y_norm_sq: RFloat,
    pub inv_y_norm_sq: RFloat,
    pub cap_mid: RFloat,

    // LU working buffers
    pub lu_a: RMat8,
    pub lu_rhs: RVec8,
    pub lu_x: RVec8,
}

impl HeavyScratch {
    pub fn new(prec: u32) -> Self {
        let mut s = Self {
            prec,
            q: rmat_zero(prec),
            c: rvec_zero(prec),
            sigma: rmat_zero(prec),
            one: rfv(prec, 1.0),
            two: rfv(prec, 2.0),
            half: rfv(prec, 0.5),
            tmp: rfz(prec),
            tmp2: rfz(prec),
            tmp3: rfz(prec),
            acc: rfz(prec),
            basis: identity_basis(),
            g_lll: rmat_zero(prec),
            temp_g: rmat_zero(prec),
            mu: rmat_zero(prec),
            g_star: rmat_zero(prec),
            gnorm_sq: rvec_zero(prec),
            delta_lll: rfv(prec, 0.75),
            l: rmat_zero(prec),
            p_u: rmat_zero(prec),
            p_ub: rmat_zero(prec),
            yhat_yhat_t: rmat_zero(prec),
            y_rf: rvec_zero(prec),
            eps_rf: rfz(prec),
            r: rfz(prec),
            r_sq: rfz(prec),
            delta_y: rfz(prec),
            delta_perp: rfz(prec),
            inv_dy_sq: rfz(prec),
            inv_dp_sq: rfz(prec),
            inv_r_sq: rfz(prec),
            y_norm_sq: rfz(prec),
            inv_y_norm_sq: rfz(prec),
            cap_mid: rfz(prec),
            lu_a: rmat_zero(prec),
            lu_rhs: rvec_zero(prec),
            lu_x: rvec_zero(prec),
        };
        // Σ matrix is constant — fill once.
        fill_sigma(&mut s.sigma, prec);
        s
    }

    /// Reset for a fresh call (reset basis to identity; other buffers will be
    /// overwritten as needed during build_q / LLL / etc).
    pub fn reset_basis(&mut self) {
        self.basis = identity_basis();
    }
}

fn identity_basis() -> IMat8 {
    std::array::from_fn(|i| {
        let mut row = [0i64; 8];
        row[i] = 1;
        row
    })
}

/// Fill a pre-allocated 8×8 buffer with the Σ matrix from arXiv:2510.05816 eq (3.15).
fn fill_sigma(sigma: &mut RMat8, prec: u32) {
    // Pattern: 1 = +1, -1 = -1, 2 = +1/√2, -2 = -1/√2, 0 = 0
    let pattern: [[i32; 8]; 8] = [
        [1,  2, 0, -2, 0,  0, 0,  0],
        [0,  2, 1,  2, 0,  0, 0,  0],
        [0,  0, 0,  0, 1,  2, 0, -2],
        [0,  0, 0,  0, 0,  2, 1,  2],
        [1, -2, 0,  2, 0,  0, 0,  0],
        [0, -2, 1, -2, 0,  0, 0,  0],
        [0,  0, 0,  0, 1, -2, 0,  2],
        [0,  0, 0,  0, 0, -2, 1, -2],
    ];
    let two = rfv(prec, 2.0);
    let r2 = two.sqrt().recip(); // 1/√2 at full prec
    let mut nr2 = rfz(prec);
    nr2.assign(-&r2);
    let one = rfv(prec, 1.0);
    let mut none = rfz(prec);
    none.assign(-&one);
    let zero = rfz(prec);
    for i in 0..8 {
        for j in 0..8 {
            sigma[i][j].assign(match pattern[i][j] {
                1 => &one,
                -1 => &none,
                2 => &r2,
                -2 => &nr2,
                _ => &zero,
            });
        }
    }
}

// ─── build_q + build_center (in-place, scratch-driven) ────────────────────────

/// Compute the anisotropic ellipsoid metric Q in-place into `scratch.q`.
/// Q = (1/Δ_y²)·ŷŷᵀ + (1/Δ_⊥²)·(P_u − ŷŷᵀ) + (1/R²)·P_{u•}.
pub fn build_q(scratch: &mut HeavyScratch, y: &[Float; 8], k: u32, eps: Float) {
    let prec = scratch.prec;

    // r_sq = 2^k; r = √(2^k); eps_rf
    scratch.r_sq.assign(rfv(prec, (1u64 << k) as f64));
    scratch.r.assign(scratch.r_sq.clone().sqrt());
    scratch.eps_rf.assign(rfv(prec, eps));

    // Δ_y = R · ε² / (2·(1 + √(1−ε²))) — safe form
    r_mul!(scratch.tmp, scratch.eps_rf, scratch.eps_rf); // tmp = ε²
    r_sub!(scratch.tmp2, scratch.one, scratch.tmp); // tmp2 = 1 - ε²
    let sqrt_1m = scratch.tmp2.clone().sqrt();
    r_add!(scratch.tmp2, scratch.one, sqrt_1m); // tmp2 = 1 + √(1−ε²)
    r_mul!(scratch.tmp3, scratch.tmp2, scratch.two); // tmp3 = 2(1+√)
    r_mul!(scratch.acc, scratch.r, scratch.tmp); // acc = R·ε²
    r_div!(scratch.delta_y, scratch.acc, scratch.tmp3); // Δ_y

    // Δ_⊥ = R · ε
    r_mul!(scratch.delta_perp, scratch.r, scratch.eps_rf);

    // 1/Δ_y², 1/Δ_⊥², 1/R²
    r_mul!(scratch.tmp, scratch.delta_y, scratch.delta_y);
    r_div!(scratch.inv_dy_sq, scratch.one, scratch.tmp);
    r_mul!(scratch.tmp, scratch.delta_perp, scratch.delta_perp);
    r_div!(scratch.inv_dp_sq, scratch.one, scratch.tmp);
    r_div!(scratch.inv_r_sq, scratch.one, scratch.r_sq);

    // y_rf, ‖y‖², 1/‖y‖²
    for i in 0..8 {
        scratch.y_rf[i].assign(rfv(prec, y[i]));
    }
    scratch.y_norm_sq.assign(0.0_f64);
    for i in 0..8 {
        r_mul!(scratch.tmp, scratch.y_rf[i], scratch.y_rf[i]);
        let acc_clone = scratch.y_norm_sq.clone();
        r_add!(scratch.y_norm_sq, acc_clone, scratch.tmp);
    }
    r_div!(scratch.inv_y_norm_sq, scratch.one, scratch.y_norm_sq);

    // ŷŷᵀ = (y_i · y_j) / ‖y‖²
    for i in 0..8 {
        for j in 0..8 {
            r_mul!(scratch.tmp, scratch.y_rf[i], scratch.y_rf[j]);
            r_mul!(scratch.yhat_yhat_t[i][j], scratch.tmp, scratch.inv_y_norm_sq);
        }
    }

    // P_u = ½·Σ_topᵀ·Σ_top, P_{u•} = ½·Σ_botᵀ·Σ_bot
    for i in 0..8 {
        for j in 0..8 {
            scratch.acc.assign(0.0_f64);
            scratch.tmp2.assign(0.0_f64);
            for r_idx in 0..4 {
                r_mul!(scratch.tmp, scratch.sigma[r_idx][i], scratch.sigma[r_idx][j]);
                let acc_clone = scratch.acc.clone();
                r_add!(scratch.acc, acc_clone, scratch.tmp);
                r_mul!(
                    scratch.tmp,
                    scratch.sigma[r_idx + 4][i],
                    scratch.sigma[r_idx + 4][j]
                );
                let tmp2_clone = scratch.tmp2.clone();
                r_add!(scratch.tmp2, tmp2_clone, scratch.tmp);
            }
            r_mul!(scratch.p_u[i][j], scratch.acc, scratch.half);
            r_mul!(scratch.p_ub[i][j], scratch.tmp2, scratch.half);
        }
    }

    // Q = inv_dy_sq · ŷŷᵀ + inv_dp_sq · (P_u − ŷŷᵀ) + inv_r_sq · P_{u•}
    for i in 0..8 {
        for j in 0..8 {
            // term1 = inv_dy_sq · yhat_yhat_t[i][j]
            r_mul!(scratch.tmp, scratch.inv_dy_sq, scratch.yhat_yhat_t[i][j]);
            // term2 = inv_dp_sq · (p_u[i][j] - yhat_yhat_t[i][j])
            r_sub!(scratch.tmp2, scratch.p_u[i][j], scratch.yhat_yhat_t[i][j]);
            r_mul!(scratch.tmp3, scratch.inv_dp_sq, scratch.tmp2);
            // term3 = inv_r_sq · p_ub[i][j]
            r_mul!(scratch.acc, scratch.inv_r_sq, scratch.p_ub[i][j]);
            // q[i][j] = term1 + term2 + term3
            let tmp_clone = scratch.tmp.clone();
            r_add!(scratch.tmp, tmp_clone, scratch.tmp3);
            r_add!(scratch.q[i][j], scratch.tmp, scratch.acc);
        }
    }
}

/// Compute the cap center c = y · cap_mid in-place into `scratch.c`.
pub fn build_center(scratch: &mut HeavyScratch, y: &[Float; 8], _k: u32, eps: Float) {
    let prec = scratch.prec;
    scratch.eps_rf.assign(rfv(prec, eps));
    r_mul!(scratch.tmp, scratch.eps_rf, scratch.eps_rf);
    r_sub!(scratch.tmp2, scratch.one, scratch.tmp); // 1 − ε²
    let sqrt_1m = scratch.tmp2.clone().sqrt();
    r_add!(scratch.tmp, scratch.one, sqrt_1m);
    r_div!(scratch.cap_mid, scratch.tmp, scratch.two); // (1 + √(1−ε²))/2

    for i in 0..8 {
        scratch.tmp.assign(rfv(prec, y[i]));
        r_mul!(scratch.c[i], scratch.tmp, scratch.cap_mid);
    }
}

// ─── Q-Gram + Gram-Schmidt + LLL (in-place rug) ──────────────────────────────

/// Compute G = B Q Bᵀ in `scratch.g_lll` from `scratch.basis` and `scratch.q`.
/// Uses `scratch.temp_g` as intermediate (= B · Q).
fn compute_qgram_inplace(scratch: &mut HeavyScratch) {
    let prec = scratch.prec;
    // temp[i][b] = sum_a basis[i][a] · Q[a][b]
    for i in 0..8 {
        for b in 0..8 {
            scratch.acc.assign(0.0_f64);
            for a in 0..8 {
                scratch.tmp.assign(rfv(prec, scratch.basis[i][a] as f64));
                r_mul!(scratch.tmp2, scratch.tmp, scratch.q[a][b]);
                let acc_clone = scratch.acc.clone();
                r_add!(scratch.acc, acc_clone, scratch.tmp2);
            }
            scratch.temp_g[i][b].assign(&scratch.acc);
        }
    }
    // g[i][j] = sum_b temp[i][b] · basis[j][b]
    for i in 0..8 {
        for j in 0..8 {
            scratch.acc.assign(0.0_f64);
            for b in 0..8 {
                scratch.tmp.assign(rfv(prec, scratch.basis[j][b] as f64));
                r_mul!(scratch.tmp2, scratch.temp_g[i][b], scratch.tmp);
                let acc_clone = scratch.acc.clone();
                r_add!(scratch.acc, acc_clone, scratch.tmp2);
            }
            scratch.g_lll[i][j].assign(&scratch.acc);
        }
    }
}

/// Gram-Schmidt orthogonalization in Q metric, working from `scratch.g_lll`.
/// Output: `scratch.mu` and `scratch.gnorm_sq`. `scratch.g_star` is intermediate.
fn gs_qgram_inplace(scratch: &mut HeavyScratch) {
    let prec = scratch.prec;
    let tiny = rfv(prec, 1e-300);
    for j in 0..8 {
        for i in j..8 {
            scratch.acc.assign(&scratch.g_lll[i][j]);
            for k in 0..j {
                r_mul!(scratch.tmp, scratch.mu[j][k], scratch.g_star[i][k]);
                let acc_clone = scratch.acc.clone();
                r_sub!(scratch.acc, acc_clone, scratch.tmp);
            }
            scratch.g_star[i][j].assign(&scratch.acc);
        }
        scratch.gnorm_sq[j].assign(&scratch.g_star[j][j]);
        if scratch.gnorm_sq[j].clone().abs() < tiny {
            for i in (j + 1)..8 {
                scratch.mu[i][j].assign(0.0_f64);
            }
            continue;
        }
        for i in (j + 1)..8 {
            r_div!(scratch.mu[i][j], scratch.g_star[i][j], scratch.gnorm_sq[j]);
        }
    }
}

/// Run LLL in-place: `scratch.basis` is reset to identity and reduced using
/// `scratch.q` as the inner product metric, δ = 0.75.
pub fn lll_qgram_8(scratch: &mut HeavyScratch) {
    scratch.reset_basis();
    let max_iter = 10_000usize;
    let mut iters = 0usize;
    let mut k = 1usize;
    while k < 8 && iters < max_iter {
        iters += 1;
        // Compute Gram + GS for size reduction
        compute_qgram_inplace(scratch);
        gs_qgram_inplace(scratch);

        // Size reduction: for j from k-1 down to 0,
        //   r = round(mu[k][j]); if r != 0: b[k] -= r·b[j]
        for j in (0..k).rev() {
            let r_round = scratch.mu[k][j].to_f64().round() as i64;
            if r_round != 0 {
                for c in 0..8 {
                    scratch.basis[k][c] -= r_round * scratch.basis[j][c];
                }
            }
        }

        // Recompute Gram+GS to check Lovász
        compute_qgram_inplace(scratch);
        gs_qgram_inplace(scratch);

        // Lovász: gnorm[k] ≥ (δ − μ[k][k-1]²) · gnorm[k-1]
        r_mul!(scratch.tmp, scratch.mu[k][k - 1], scratch.mu[k][k - 1]);
        r_sub!(scratch.tmp2, scratch.delta_lll, scratch.tmp);
        r_mul!(scratch.tmp, scratch.tmp2, scratch.gnorm_sq[k - 1]);
        if scratch.gnorm_sq[k] >= scratch.tmp {
            k += 1;
        } else {
            scratch.basis.swap(k, k - 1);
            k = k.saturating_sub(1).max(1);
        }
    }
}

// ─── Cholesky (in-place rug) ─────────────────────────────────────────────────

/// Cholesky of `scratch.g_lll` → `scratch.l` (lower-triangular). Returns
/// `false` and leaves `scratch.l` partially-filled if a diagonal computes as
/// non-positive. With high-precision MPFR, this should only happen if Q is
/// genuinely not PD (a bug, not a precision issue).
pub fn cholesky_8(scratch: &mut HeavyScratch) -> bool {
    let prec = scratch.prec;
    // Zero L
    for i in 0..8 {
        for j in 0..8 {
            scratch.l[i][j].assign(0.0_f64);
        }
    }
    let zero = rfz(prec);
    for i in 0..8 {
        for j in 0..=i {
            scratch.acc.assign(&scratch.g_lll[i][j]);
            for k in 0..j {
                r_mul!(scratch.tmp, scratch.l[i][k], scratch.l[j][k]);
                let acc_clone = scratch.acc.clone();
                r_sub!(scratch.acc, acc_clone, scratch.tmp);
            }
            if i == j {
                if scratch.acc <= zero {
                    return false;
                }
                let acc_clone = scratch.acc.clone();
                scratch.l[i][i].assign(acc_clone.sqrt());
            } else {
                // Stage l[j][j] into a separate scratch field to break the
                // simultaneous mutable+immutable borrow on scratch.l.
                scratch.tmp2.assign(&scratch.l[j][j]);
                r_div!(scratch.l[i][j], scratch.acc, scratch.tmp2);
            }
        }
    }
    true
}

// ─── LU solve with partial pivoting (in-place rug) ───────────────────────────

/// Solve A · x = b for x where A is loaded into `scratch.lu_a` and b into
/// `scratch.lu_rhs`; result lands in `scratch.lu_x`. Both `lu_a` and `lu_rhs`
/// are mutated by the elimination. Returns `false` on numerical singularity.
fn lu_solve_inplace(scratch: &mut HeavyScratch) -> bool {
    let prec = scratch.prec;
    let tol = rfv(prec, 1e-30);

    for k in 0..8 {
        // Find pivot row in column k (rows k..8)
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
        // Eliminate below the pivot
        for i in (k + 1)..8 {
            // factor = a[i][k] / a[k][k]
            r_div!(scratch.tmp, scratch.lu_a[i][k], scratch.lu_a[k][k]);
            // Save factor (we'll need it after the borrow below ends)
            let factor = scratch.tmp.clone();
            // a[i][j] -= factor · a[k][j]  for j in k..8
            // Avoid simultaneous borrow of rows i and k.
            let (row_i, row_k) = two_rows_mut(&mut scratch.lu_a, i, k);
            for j in k..8 {
                scratch.tmp.assign(&factor * &row_k[j]);
                let cur = row_i[j].clone();
                r_sub!(row_i[j], cur, scratch.tmp);
            }
            // rhs[i] -= factor · rhs[k]
            scratch.tmp.assign(&factor * &scratch.lu_rhs[k]);
            let rhs_i_cur = scratch.lu_rhs[i].clone();
            r_sub!(scratch.lu_rhs[i], rhs_i_cur, scratch.tmp);
        }
    }
    // Back substitution
    for i in (0..8).rev() {
        scratch.acc.assign(&scratch.lu_rhs[i]);
        for j in (i + 1)..8 {
            r_mul!(scratch.tmp, scratch.lu_a[i][j], scratch.lu_x[j]);
            let cur = scratch.acc.clone();
            r_sub!(scratch.acc, cur, scratch.tmp);
        }
        let acc_clone = scratch.acc.clone();
        r_div!(scratch.lu_x[i], acc_clone, scratch.lu_a[i][i]);
    }
    true
}

// ─── det8_exact: i256-based unimodularity check ──────────────────────────────

/// Compute the determinant of an 8×8 i64 matrix via Bareiss algorithm in i256
/// (so corrupt LLL output doesn't itself overflow). Returns det as i64 if it
/// fits, else None.
pub fn det8_exact(m: &IMat8) -> Option<i64> {
    use i256::i256;
    let mut a: [[i256; 8]; 8] =
        std::array::from_fn(|i| std::array::from_fn(|j| i256::from_i64(m[i][j])));
    let mut sign: i64 = 1;
    let mut prev = i256::from_i64(1);
    let zero = i256::from_i64(0);

    for k in 0..8 {
        if a[k][k] == zero {
            let mut found = false;
            for i in (k + 1)..8 {
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
        for i in (k + 1)..8 {
            for j in (k + 1)..8 {
                let lhs = a[i][j] * pivot;
                let rhs = a[i][k] * a[k][j];
                a[i][j] = (lhs - rhs) / prev;
            }
            a[i][k] = zero;
        }
        prev = pivot;
    }
    let det = a[7][7];
    let det_signed = if sign < 0 { -det } else { det };
    let lo = det_signed.as_i128();
    if lo >= i64::MIN as i128 && lo <= i64::MAX as i128 {
        Some(lo as i64)
    } else {
        None
    }
}

// ─── 8D Schnorr-Euchner (f64) ─────────────────────────────────────────────────

/// Convert a `rug::Float` to a `TwoFloat`, preserving precision up to the
/// ~104-bit mantissa of double-double. Used to produce a higher-precision
/// `R_chol` and `z_c` for the SE step at extreme ε where f64's 53-bit mantissa
/// would lose enough precision in the squared-norm sum to mis-bound the SE
/// (ghost-node blowup at L_diag ratio ≳ 10¹⁰).
fn rug_to_tf(r: &RFloat) -> Tf {
    // Shannon decomposition: hi = nearest f64 to r; lo = nearest f64 to (r − hi).
    let hi = r.to_f64();
    let mut resid = r.clone();
    resid -= hi;
    let lo = resid.to_f64();
    Tf::new_add(hi, lo)
}

/// Compute the upper-triangular Cholesky factor R of B·Bᵀ for an integer LLL
/// basis B. Used for the Euclidean-norm partial-prune below: at SE depth d,
/// ∑_{i≥d} (Rz)_i² is a strict lower bound on ‖x‖² regardless of how the
/// remaining z[<d] are chosen (each level contributes a non-negative
/// squared term in the GS decomposition).
fn compute_r_eucl(basis: &IMat8) -> Option<[[f64; 8]; 8]> {
    // Gram = B · Bᵀ, exact i64
    let mut gram = [[0_i64; 8]; 8];
    for i in 0..8 {
        for j in 0..8 {
            let mut s = 0_i64;
            for k in 0..8 {
                s += basis[i][k] * basis[j][k];
            }
            gram[i][j] = s;
        }
    }
    // Cholesky in f64 (gram entries up to ~10⁹ for typical LLL output, well
    // within f64's 15-digit margin)
    let mut l = [[0.0_f64; 8]; 8];
    for i in 0..8 {
        for j in 0..=i {
            let mut s = gram[i][j] as f64;
            for k in 0..j {
                s -= l[i][k] * l[j][k];
            }
            if i == j {
                if s <= 0.0 {
                    return None;
                }
                l[i][i] = s.sqrt();
            } else {
                l[i][j] = s / l[j][j];
            }
        }
    }
    // Transpose to upper-triangular R = Lᵀ
    let mut r = [[0.0_f64; 8]; 8];
    for i in 0..8 {
        for j in 0..8 {
            r[i][j] = l[j][i];
        }
    }
    Some(r)
}

/// 8D Schnorr-Euchner enumeration in `TwoFloat` precision (~32 decimal digits)
/// for the Q-metric ellipsoid bound, with an additional f64 Euclidean-norm
/// prune (`r_eucl_opt`) to short-circuit branches whose partial GS norm
/// already exceeds 2^k. Pass `None` for `r_eucl_opt` to disable the prune.
///
/// `abort` is checked at every recursion entry — when set, the enumeration
/// returns immediately without setting `result`. The caller (in `phase1_lenstra`)
/// uses this for the SE-node-count circuit breaker that triggers precision
/// escalation in the adaptive Heavy pipeline.
fn se_8d_tf<F>(
    r_chol: &[[Tf; 8]; 8],
    z_c: &[Tf; 8],
    bound: Tf,
    r_eucl_opt: Option<&[[f64; 8]; 8]>,
    target_norm_f: f64,
    abort: &std::sync::atomic::AtomicBool,
    mut callback: F,
) -> Option<[i64; 8]>
where
    F: FnMut(&[i64; 8]) -> Option<[i64; 8]>,
{
    use std::sync::atomic::Ordering;
    let mut z = [0i64; 8];
    let result = std::cell::RefCell::new(None);
    let zero = Tf::from(0.0_f64);

    fn recurse<F>(
        depth: i32,
        r_chol: &[[Tf; 8]; 8],
        z_c: &[Tf; 8],
        bound: Tf,
        r_eucl_opt: Option<&[[f64; 8]; 8]>,
        target_norm_f: f64,
        partial_eucl: f64,
        z: &mut [i64; 8],
        partial: Tf,
        abort: &std::sync::atomic::AtomicBool,
        callback: &mut F,
        result: &std::cell::RefCell<Option<[i64; 8]>>,
    ) where
        F: FnMut(&[i64; 8]) -> Option<[i64; 8]>,
    {
        if result.borrow().is_some() || abort.load(Ordering::Relaxed) {
            return;
        }
        if depth < 0 {
            if let Some(r) = callback(z) {
                *result.borrow_mut() = Some(r);
            }
            return;
        }
        let d = depth as usize;
        let r_dd = r_chol[d][d];
        // Use f64 magnitude for the threshold check (cheap, only matters as a
        // structural guard against zero diagonals).
        if f64::from(r_dd.abs()) < 1e-30 {
            z[d] = f64::from(z_c[d]).round() as i64;
            recurse(
                depth - 1, r_chol, z_c, bound, r_eucl_opt, target_norm_f,
                partial_eucl, z, partial, abort, callback, result,
            );
            return;
        }
        // tail = ∑_{j>d} R[d][j] · (z[j] − z_c[j])
        let mut tail = Tf::from(0.0_f64);
        for j in (d + 1)..8 {
            let zj = Tf::from(z[j] as f64);
            let diff = zj - z_c[j];
            tail = tail + r_chol[d][j] * diff;
        }
        let rem = bound - partial;
        if f64::from(rem) < 0.0 {
            return;
        }
        let rem_sqrt = rem.sqrt();
        // For deciding the integer iteration range we drop to f64 — the range
        // is computed via center ± span, and we want integer bounds. Twofloat
        // precision in the bounds doesn't change which integers fall inside
        // them (only matters at the unlikely 1-ULP edge).
        let r_dd_f = f64::from(r_dd);
        let z_c_d_f = f64::from(z_c[d]);
        let center_off = -f64::from(tail) / r_dd_f;
        let span = f64::from(rem_sqrt) / r_dd_f.abs();
        let z_low = (z_c_d_f + center_off - span).ceil() as i64;
        let z_high = (z_c_d_f + center_off + span).floor() as i64;
        let z_mid = (z_c_d_f + center_off).round() as i64;
        let max_off = (z_high - z_mid).max(z_mid - z_low).max(0);

        // Pre-compute the Euclidean tail for the GS-decomposition contribution
        // at level d (uses fixed levels j > d).
        let tail_eucl = if let Some(r_eucl) = r_eucl_opt {
            let mut t = 0.0_f64;
            for j in (d + 1)..8 {
                t += r_eucl[d][j] * (z[j] as f64);
            }
            t
        } else {
            0.0
        };

        for raw in 0..=(2 * max_off + 1) {
            if result.borrow().is_some() || abort.load(Ordering::Relaxed) {
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
            // The squared distance accumulation IS in twofloat — this is the
            // bit that f64 was getting wrong.
            let zd_tf = Tf::from(zd as f64);
            let level = r_dd * (zd_tf - z_c[d]) + tail;
            let level_sq = level * level;
            let new_partial = partial + level_sq;
            // 1e-9 noise margin in the bound check — twofloat precision is
            // ample to make this exact, but keep the slack for safety.
            if f64::from(new_partial - bound) > 1e-9 {
                continue;
            }
            // Euclidean norm prune (when r_eucl provided): partial GS sum
            // ∑_{i≥d} (Rz)_i² is a strict lower bound on ‖x‖². If this
            // already exceeds 2^k by more than f64-noise margin, the
            // unitarity-shell constraint can't be met. Margin generous (1.0)
            // to absorb f64 accumulation noise.
            let new_partial_eucl = if let Some(r_eucl) = r_eucl_opt {
                let level_eucl = r_eucl[d][d] * (zd as f64) + tail_eucl;
                let p = partial_eucl + level_eucl * level_eucl;
                if p > target_norm_f + 1.0 {
                    continue;
                }
                p
            } else {
                partial_eucl
            };
            z[d] = zd;
            recurse(
                depth - 1, r_chol, z_c, bound, r_eucl_opt, target_norm_f,
                new_partial_eucl, z, new_partial, abort, callback, result,
            );
        }
    }

    recurse(
        7, r_chol, z_c, bound, r_eucl_opt, target_norm_f, 0.0,
        &mut z, zero, abort, &mut callback, &result,
    );
    result.into_inner()
}

#[allow(dead_code)] // kept for diagnostic / fallback comparison; Heavy path uses se_8d_tf
fn se_8d_f64<F>(
    r_chol: &[[f64; 8]; 8],
    z_c: &[f64; 8],
    bound: f64,
    mut callback: F,
) -> Option<[i64; 8]>
where
    F: FnMut(&[i64; 8]) -> Option<[i64; 8]>,
{
    let mut z = [0i64; 8];
    let result = std::cell::RefCell::new(None);

    fn recurse<F>(
        depth: i32,
        r_chol: &[[f64; 8]; 8],
        z_c: &[f64; 8],
        bound: f64,
        z: &mut [i64; 8],
        partial: f64,
        callback: &mut F,
        result: &std::cell::RefCell<Option<[i64; 8]>>,
    ) where
        F: FnMut(&[i64; 8]) -> Option<[i64; 8]>,
    {
        if result.borrow().is_some() {
            return;
        }
        if depth < 0 {
            if let Some(r) = callback(z) {
                *result.borrow_mut() = Some(r);
            }
            return;
        }
        let d = depth as usize;
        let r_dd = r_chol[d][d];
        if r_dd.abs() < 1e-30 {
            z[d] = z_c[d].round() as i64;
            recurse(depth - 1, r_chol, z_c, bound, z, partial, callback, result);
            return;
        }
        let mut tail = 0.0;
        for j in (d + 1)..8 {
            tail += r_chol[d][j] * (z[j] as f64 - z_c[j]);
        }
        let rem = bound - partial;
        if rem < 0.0 {
            return;
        }
        let rem_sqrt = rem.sqrt();
        let center_off = -tail / r_dd;
        let span = rem_sqrt / r_dd.abs();
        let z_low = (z_c[d] + center_off - span).ceil() as i64;
        let z_high = (z_c[d] + center_off + span).floor() as i64;
        let z_mid = (z_c[d] + center_off).round() as i64;
        let max_off = (z_high - z_mid).max(z_mid - z_low).max(0);

        for raw in 0..=(2 * max_off + 1) {
            if result.borrow().is_some() {
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
            let level = r_dd * (zd as f64 - z_c[d]) + tail;
            let new_partial = partial + level * level;
            if new_partial > bound + 1e-9 {
                continue;
            }
            z[d] = zd;
            recurse(depth - 1, r_chol, z_c, bound, z, new_partial, callback, result);
        }
    }

    recurse(7, r_chol, z_c, bound, &mut z, 0.0, &mut callback, &result);
    result.into_inner()
}

// ─── Top-level phase1_lenstra ─────────────────────────────────────────────────

#[inline]
fn bilinear_b(x: &[i64; 8]) -> i64 {
    let (a1, b1, c1, d1) = (x[0], x[1], x[2], x[3]);
    let (a2, b2, c2, d2) = (x[4], x[5], x[6], x[7]);
    a1 * b1 - a1 * d1 + b1 * c1 + c1 * d1 + a2 * b2 - a2 * d2 + b2 * c2 + c2 * d2
}

#[inline]
fn reconstruct_x(b_lll: &IMat8, z: &[i64; 8]) -> [i64; 8] {
    let mut x = [0i64; 8];
    for i in 0..8 {
        for j in 0..8 {
            x[j] += z[i] * b_lll[i][j];
        }
    }
    x
}

/// Outcome of one `phase1_lenstra_attempt` call. The `should_escalate` flag
/// signals to the dispatch layer that this attempt's precision was insufficient
/// — either a hard fail (det/Cholesky/LU) or a soft fail (SE-node circuit
/// breaker tripped without a solution). On `true`, the dispatch should retry
/// at higher precision before reporting "no solution at this lde".
pub struct AttemptOutcome {
    pub solutions: Vec<[i64; 8]>,
    pub should_escalate: bool,
}

/// Wrapper preserving the original signature for any external callers / tests.
/// Runs as a single non-escalating attempt (no SE-node circuit breaker).
pub fn phase1_lenstra(
    scratch: &mut HeavyScratch,
    y: &[Float; 8],
    k: u32,
    eps: Float,
    max_phase2_calls: u64,
    budget_hit: &AtomicBool,
) -> Vec<[i64; 8]> {
    phase1_lenstra_attempt(scratch, y, k, eps, max_phase2_calls, budget_hit, false).solutions
}

/// Run the full 8D Lenstra pipeline for one MA-prefix's `(y, k, eps)` setup.
/// Pre-allocated `scratch` is reused (typically allocated per-rayon-worker via
/// `map_init`). Returns an `AttemptOutcome` with up to one valid 8-vector
/// solution and a flag indicating whether the dispatch should retry at higher
/// precision.
///
/// `enable_escalation` controls whether the SE-node circuit breaker is active.
/// Pass `true` for the low-precision attempt (will signal escalation on
/// excessive node count) and `false` for the high-precision attempt (the SE
/// runs to completion, bounded only by `max_phase2_calls`).
pub fn phase1_lenstra_attempt(
    scratch: &mut HeavyScratch,
    y: &[Float; 8],
    k: u32,
    eps: Float,
    max_phase2_calls: u64,
    budget_hit: &AtomicBool,
    enable_escalation: bool,
) -> AttemptOutcome {
    use std::sync::atomic::{AtomicU64, Ordering};

    if enable_escalation {
        crate::synthesis::diag::N_LOW_ATTEMPT.fetch_add(1, Ordering::Relaxed);
    } else {
        crate::synthesis::diag::N_HIGH_ATTEMPT.fetch_add(1, Ordering::Relaxed);
    }

    let target_norm: i64 = 1i64 << k;
    let threshold_xy = (1i64 << (2 * k)) as Float / 4.0 * (1.0 - eps * eps);

    let trace = crate::synthesis::diag::trace_enabled();

    // Step 1: build Q and center c
    let t_phase = if trace { Some(std::time::Instant::now()) } else { None };
    build_q(scratch, y, k, eps);
    build_center(scratch, y, k, eps);
    if let Some(t0) = t_phase {
        crate::synthesis::diag::T_BUILD_NS
            .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

    // Step 2: LLL-reduce ℤ⁸ identity using Q metric
    let t_phase = if trace { Some(std::time::Instant::now()) } else { None };
    lll_qgram_8(scratch);
    if let Some(t0) = t_phase {
        crate::synthesis::diag::T_LLL_NS
            .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

    // Step 3: assert det = ±1 (catches genuine pathology — escalate on failure
    // since loss of unimodularity at this precision means the LLL's GS
    // orthogonalization broke down silently)
    let basis = scratch.basis;
    match det8_exact(&basis) {
        Some(1) | Some(-1) => {}
        Some(d) => {
            if enable_escalation {
                crate::synthesis::diag::N_LOW_ESCALATE.fetch_add(1, Ordering::Relaxed);
            } else {
                eprintln!(
                    "[lenstra] LLL non-unimodular (det={}) at full prec; k={}, ε={:e}; bailing.",
                    d, k, eps
                );
            }
            return AttemptOutcome { solutions: Vec::new(), should_escalate: enable_escalation };
        }
        None => {
            if enable_escalation {
                crate::synthesis::diag::N_LOW_ESCALATE.fetch_add(1, Ordering::Relaxed);
            } else {
                eprintln!(
                    "[lenstra] det8_exact overflow at full prec; k={}, ε={:e}; bailing.",
                    k, eps
                );
            }
            return AttemptOutcome { solutions: Vec::new(), should_escalate: enable_escalation };
        }
    }

    // Step 4: Cholesky of G_LLL = B Q Bᵀ (computed inline by the LLL's last
    // gram pass — but we need to recompute since LLL's last GS may have been
    // for the post-swap basis).
    let t_phase = if trace { Some(std::time::Instant::now()) } else { None };
    compute_qgram_inplace(scratch);
    let chol_ok = cholesky_8(scratch);
    if let Some(t0) = t_phase {
        crate::synthesis::diag::T_CHOLESKY_NS
            .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
    if !chol_ok {
        if enable_escalation {
            crate::synthesis::diag::N_LOW_ESCALATE.fetch_add(1, Ordering::Relaxed);
        } else {
            eprintln!(
                "[lenstra] Cholesky failed at full prec; k={}, ε={:e}; bailing.",
                k, eps
            );
        }
        return AttemptOutcome { solutions: Vec::new(), should_escalate: enable_escalation };
    }

    // Convert R_chol = Lᵀ to TwoFloat (~104 bits) for the SE step. f64
    // (53 bits) was insufficient at ε ≤ 1e-5 — the L_diag entry ratios reach
    // ~10¹⁰, leaving only 5 digits of working margin in the squared-norm sum
    // and causing "ghost node" SE blowups (millions of false candidates).
    let r_chol_tf: [[Tf; 8]; 8] = std::array::from_fn(|i| {
        std::array::from_fn(|j| rug_to_tf(&scratch.l[j][i]))
    });

    // Step 5: solve B_LLLᵀ · z_c = c via LU with partial pivoting.
    // (B_LLL has rows = basis vectors, so x = B_LLLᵀ · z. To find z given x = c,
    // solve B_LLLᵀ · z = c.)
    let t_phase = if trace { Some(std::time::Instant::now()) } else { None };
    for i in 0..8 {
        for j in 0..8 {
            scratch.lu_a[i][j].assign(rfv(scratch.prec, basis[j][i] as f64));
        }
        scratch.lu_rhs[i].assign(&scratch.c[i]);
    }
    let lu_ok = lu_solve_inplace(scratch);
    if let Some(t0) = t_phase {
        crate::synthesis::diag::T_LU_NS
            .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
    if !lu_ok {
        if enable_escalation {
            crate::synthesis::diag::N_LOW_ESCALATE.fetch_add(1, Ordering::Relaxed);
        } else {
            eprintln!("[lenstra] LU solve failed at full prec; bailing.");
        }
        return AttemptOutcome { solutions: Vec::new(), should_escalate: enable_escalation };
    }
    let z_c_tf: [Tf; 8] = std::array::from_fn(|i| rug_to_tf(&scratch.lu_x[i]));

    // Compute R_eucl (Euclidean Cholesky factor of B B^T) for the safe
    // Euclidean-norm pruner inside SE. Skipped silently if not PD (rare).
    let r_eucl = compute_r_eucl(&basis);
    let target_norm_f = target_norm as f64;

    // Step 6: 8D SE in twofloat with bound 1.51 (cap volume max + 0.01 noise).
    // The `abort` flag is set by the callback when the SE-node circuit breaker
    // trips (count > SE_ESCALATE_THRESHOLD), signalling fat-ellipsoid loss of
    // GS bounds at the current precision. The dispatch retries at higher
    // precision when this fires without finding a solution.
    let count = AtomicU64::new(0);
    let abort = AtomicBool::new(false);
    let bound_tf = Tf::from(1.51_f64);
    let escalate_at = if enable_escalation { SE_ESCALATE_THRESHOLD } else { u64::MAX };
    let t_phase = if trace { Some(std::time::Instant::now()) } else { None };
    let result = se_8d_tf(&r_chol_tf, &z_c_tf, bound_tf, r_eucl.as_ref(), target_norm_f, &abort, |z: &[i64; 8]| {
        let n_so_far = count.load(Ordering::Relaxed);
        if n_so_far >= max_phase2_calls {
            budget_hit.store(true, Ordering::Relaxed);
            return None;
        }
        if n_so_far >= escalate_at {
            abort.store(true, Ordering::Relaxed);
            return None;
        }
        count.fetch_add(1, Ordering::Relaxed);
        let x = reconstruct_x(&basis, z);
        let n: i64 = x.iter().map(|&v| v * v).sum();
        if n != target_norm {
            return None;
        }
        if bilinear_b(&x) != 0 {
            return None;
        }
        let dot: Float = x
            .iter()
            .zip(y.iter())
            .map(|(a, b)| *a as Float * b)
            .sum();
        if dot * dot < threshold_xy {
            return None;
        }
        Some(x)
    });

    if let Some(t0) = t_phase {
        crate::synthesis::diag::T_SE_NS
            .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
    crate::synthesis::diag::N_SE_CALLBACKS
        .fetch_add(count.load(Ordering::Relaxed), Ordering::Relaxed);
    let aborted = abort.load(Ordering::Relaxed);
    match result {
        Some(x) => {
            if enable_escalation {
                crate::synthesis::diag::N_LOW_FOUND.fetch_add(1, Ordering::Relaxed);
            } else {
                crate::synthesis::diag::N_HIGH_FOUND.fetch_add(1, Ordering::Relaxed);
            }
            AttemptOutcome { solutions: vec![x], should_escalate: false }
        }
        None => {
            if aborted {
                crate::synthesis::diag::N_LOW_ESCALATE.fetch_add(1, Ordering::Relaxed);
            }
            AttemptOutcome { solutions: Vec::new(), should_escalate: aborted }
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn det8_known_unimodular() {
        let id: IMat8 = identity_basis();
        assert_eq!(det8_exact(&id), Some(1));
        let mut swapped = id;
        swapped.swap(2, 5);
        assert_eq!(det8_exact(&swapped), Some(-1));
        let mut shifted = id;
        for c in 0..8 {
            shifted[1][c] += shifted[0][c];
        }
        assert_eq!(det8_exact(&shifted), Some(1));
    }

    #[test]
    fn lll_returns_unimodular_at_eps_1e_4() {
        let prec = compute_prec(1e-4);
        let mut scratch = HeavyScratch::new(prec);
        let r2 = 1.0 / 2.0_f64.sqrt();
        let s = ((1u64 << 17) as f64).sqrt() / 2.0;
        let y = [s, s * r2, 0.0, -s * r2, 0.0, 0.0, 0.0, 0.0];
        build_q(&mut scratch, &y, 17, 1e-4);
        lll_qgram_8(&mut scratch);
        let det = det8_exact(&scratch.basis).expect("det fits");
        assert!(det == 1 || det == -1, "det = {}", det);
    }

    #[test]
    fn cholesky_succeeds_at_eps_1e_5() {
        let prec = compute_prec(1e-5);
        let mut scratch = HeavyScratch::new(prec);
        let r2 = 1.0 / 2.0_f64.sqrt();
        let s = ((1u64 << 21) as f64).sqrt() / 2.0;
        let y = [s, s * r2, 0.0, -s * r2, 0.0, 0.0, 0.0, 0.0];
        build_q(&mut scratch, &y, 21, 1e-5);
        lll_qgram_8(&mut scratch);
        compute_qgram_inplace(&mut scratch);
        assert!(cholesky_8(&mut scratch));
    }

    #[test]
    fn compute_prec_examples() {
        assert_eq!(compute_prec(1e-2), 100); // floor
        assert_eq!(compute_prec(1e-3), 100); // floor
        // ε=1e-4: 8·log₂(1e4) = 8·13.29 = 106 → 107
        assert!(compute_prec(1e-4) >= 100 && compute_prec(1e-4) <= 110);
        // ε=1e-5: ~133
        assert!(compute_prec(1e-5) >= 130 && compute_prec(1e-5) <= 140);
    }
}

// ─── Precision audit (slow; run via --ignored) ───────────────────────────────
//
// Diagnostic tool for measuring the precision required at very tight ε. Not run
// by default since it uses 200+ bit MPFR math. Re-run via:
//   cargo test --release --lib precision_audit -- --ignored --nocapture

#[cfg(test)]
mod precision_audit {
    use super::*;

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

    fn run_at(prec: u32, k: u32, eps: Float) -> Result<(f64, f64), String> {
        let mut scratch = HeavyScratch::new(prec);
        let y = realistic_y(k);
        build_q(&mut scratch, &y, k, eps);
        lll_qgram_8(&mut scratch);
        let det = det8_exact(&scratch.basis).ok_or("det overflow")?;
        if det != 1 && det != -1 {
            return Err(format!("non-unimodular: det={det}"));
        }
        compute_qgram_inplace(&mut scratch);
        if !cholesky_8(&mut scratch) {
            return Err("Cholesky failed".to_string());
        }
        let mut min_d = f64::INFINITY;
        let mut max_d = 0.0_f64;
        for i in 0..8 {
            let d = scratch.l[i][i].to_f64();
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
        Ok((min_d, max_d))
    }

    #[test]
    #[ignore = "slow; run with --ignored to characterize precision ceiling"]
    fn audit_precision_ceiling_sweep() {
        let cases: &[(Float, u32, &[u32])] = &[
            (1e-3_f64, 14, &[60, 80, 100, 150]),
            (1e-4_f64, 17, &[80, 100, 150]),
            (1e-5_f64, 21, &[100, 150, 200]),
            (1e-6_f64, 25, &[150, 200, 250]),
            (1e-7_f64, 29, &[200, 250, 300]),
            (1e-9_f64, 36, &[300, 400, 500]),
        ];
        for &(eps, k, precs) in cases {
            let mut succeeded_at: Option<u32> = None;
            for &p in precs {
                match run_at(p, k, eps) {
                    Ok((min_d, max_d)) => {
                        eprintln!(
                            "[audit] ε={eps:e} k={k} prec={p}: OK L_diag ∈ [{min_d:.3e}, {max_d:.3e}], ratio={:.2e}",
                            max_d / min_d
                        );
                        succeeded_at = Some(p);
                        break;
                    }
                    Err(e) => {
                        eprintln!("[audit] ε={eps:e} k={k} prec={p}: FAIL ({e})");
                    }
                }
            }
            assert!(
                succeeded_at.is_some(),
                "no precision in {precs:?} succeeded at ε={eps:e}"
            );
        }
    }
}
