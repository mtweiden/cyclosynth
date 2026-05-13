//! Per-thread scratch buffers for the 16D Z[ζ_16] L²-LLL pipeline.
//!
//! Mirrors `super::super::lenstra::scratch` but extended to dimension 16:
//!
//!   - `q_int`: i256 16x16 scaled Q-metric snapshot.
//!   - `basis`: i64 16x16 LLL basis (rows = basis vectors).
//!   - `gram` / `temp_bq`: i256 16x16 working buffers for `G = B·Q·Bᵀ`.
//!   - `r_bar`, `mu_bar`, `s_bar`: MPFR Gram-Schmidt state at
//!     [`GS_PREC`] bits. **MPFR is mandatory at d=16** (Theorem 2 of
//!     Nguyen-Stehlé 2009 covers only d ≤ 11 in f64).
//!
//! ## Overflow analysis at d=16
//!
//! With `TARGET_BITS = 180` and a basis post-LLL with `max(|B|) ≤ 2^15`:
//!
//!   `|G[i][j]| ≤ 16 · max(|B|)² · max(|Q_int|)` (sum of 16 products)
//!                `≤ 2^4 · 2^30 · 2^180 = 2^214`
//!
//! With overflow threshold at `2^240`, this leaves 26 bits of headroom for
//! transient B-growth during LLL swaps. At deep ε with B inflated to ~2^25
//! the transient Gram entries can hit ~2^234, still under threshold; beyond
//! that the i256 path will trip `GramOverflow` and the caller must escalate
//! to a larger integer type (e.g. `rug::Integer`). For the M3 deliverable
//! targeting moderate ε (≥ 10⁻⁵, k ≤ 30) this is comfortable.

#![allow(dead_code)]
#![allow(clippy::needless_range_loop)]

use crate::rings::Float;
use i256::i256;
use rug::Float as RFloat;

// ─── Adaptive precision constants ────────────────────────────────────────────

/// Target effective precision for `Q_int` entries: max(|Q_int|) ≈ 2^TARGET_BITS.
/// Same value as the 8D pipeline; the 16D dimensional growth in the Gram
/// budget is absorbed via `GRAM_OVERFLOW_THRESHOLD_BITS` headroom rather
/// than a smaller TARGET_BITS, which would hurt LLL precision near deep ε.
pub const TARGET_BITS: u32 = 180;

/// Magnitude threshold for Gram-entry overflow detection: 2^240, leaving
/// 16-bit margin to i256::MAX. At d=16 the safe operating range is
/// `max(|B|)² · max(|Q_int|) · 16 ≤ 2^240`, which holds for typical
/// post-LLL bases at ε ≥ 1e-5.
pub const GRAM_OVERFLOW_THRESHOLD_BITS: u32 = 240;

/// MPFR Gram-Schmidt precision. Theorem 2 of Nguyen-Stehlé 2009 covers d ≤ 11
/// at L²-LLL parameters (δ=0.75, η=0.55) in f64; for d=16 the proof doesn't
/// apply, so we use MPFR. 128 bits gives ~75 mantissa bits of headroom over
/// the f64 ℓ=52 used at d=8, well above the algorithmic requirement
/// `ℓ ≥ 5 + 2·log d − log ε + d·log ρ` (which extrapolates to ~50 bits at
/// d=16, ε=1e-7, ρ ≈ 1.05). Configurable per-construction in case a deeper
/// ε regime needs more.
pub const GS_PREC: u32 = 128;

/// Compute the bit-shift `B` such that `round(2^B · Q[i][j])` lands in i256
/// with max entry ≈ 2^TARGET_BITS.
pub fn compute_scale_bits(max_q_log2: i32) -> i32 {
    TARGET_BITS as i32 - max_q_log2
}

/// MPFR precision in bits used to construct the anisotropic Q metric.
pub fn compute_prec_q(eps: Float) -> u32 {
    let log_recip = (1.0 / eps).log2().max(1.0);
    let bits = (8.0 * log_recip).ceil() as u32;
    bits.max(100)
}

/// MPFR precision used by the cap-center LU solve. The 16x16 partial-pivoting
/// LU on the post-LLL basis B (det = ±1, entries up to ~2^15-2^41) develops
/// pivot ratios that grow with ε; 6·log₂(1/ε) bits leaves headroom past SE's
/// 10⁻⁹ tolerance.
pub fn compute_lu_prec(eps: Float) -> u32 {
    let log_recip = (1.0 / eps).log2().max(1.0);
    let bits = (6.0 * log_recip).ceil() as u32;
    bits.max(96)
}

// ─── Type aliases ────────────────────────────────────────────────────────────

pub type IMat16 = [[i64; 16]; 16];
pub type Mat256_16 = [[i256; 16]; 16];

// ─── Helpers ─────────────────────────────────────────────────────────────────

pub fn rfz(prec: u32) -> RFloat {
    RFloat::with_val(prec, 0.0_f64)
}

pub fn rfv(prec: u32, x: f64) -> RFloat {
    RFloat::with_val(prec, x)
}

pub fn rmat_zero_16(prec: u32) -> [[RFloat; 16]; 16] {
    std::array::from_fn(|_| std::array::from_fn(|_| rfz(prec)))
}

pub fn rvec_zero_16(prec: u32) -> [RFloat; 16] {
    std::array::from_fn(|_| rfz(prec))
}

pub fn imat_zero_16() -> Mat256_16 {
    let z = i256::from_i64(0);
    std::array::from_fn(|_| std::array::from_fn(|_| z))
}

pub fn identity_basis_16() -> IMat16 {
    std::array::from_fn(|i| {
        let mut row = [0i64; 16];
        row[i] = 1;
        row
    })
}

// ─── IntScratch16 ────────────────────────────────────────────────────────────

/// Per-thread scratch buffers for the 16D pipeline. All MPFR / i256 storage
/// is allocated up front; the inner LLL loop performs zero allocation.
pub struct IntScratch16 {
    /// MPFR precision used for build_q.
    pub prec_q: u32,
    /// MPFR precision used for the LLL Gram-Schmidt state.
    pub gs_prec: u32,
    /// Adaptive scale `B` such that `Q_int[i][j] ≈ 2^B · Q[i][j]`.
    pub scale_bits: i32,

    // ── MPFR buffers for build_q ──
    pub q_mpfr: [[RFloat; 16]; 16],

    // ── Integer LLL buffers ──
    pub q_int: Mat256_16,
    pub basis: IMat16,
    pub gram: Mat256_16,
    pub temp_bq: Mat256_16,

    // ── L²-LLL Gram-Schmidt state, MPFR at GS_PREC bits ──
    //
    // r_bar[i][j] = <b_i*, b_j*>           (Gram-Schmidt inner products)
    // mu_bar[i][j] = r_bar[i][j] / r_bar[j][j]
    // s_bar[i][j] = r_bar[i][i] - Σ_{k<j} mu_bar[i][k] · r_bar[i][k]
    //                (Lovász partial sums)
    pub r_bar: [[RFloat; 16]; 16],
    pub mu_bar: [[RFloat; 16]; 16],
    pub s_bar: [[RFloat; 16]; 16],

    // ── Scratch RFloats reused inside LLL ──
    pub tmp_a: RFloat,
    pub tmp_b: RFloat,
    pub tmp_c: RFloat,

    // ── post-LLL Cholesky output (f64) ──
    pub l_f64: [[f64; 16]; 16],

    // ── Cap center in MPFR (for the post-LLL LU solve) ──
    /// `c[i] = y[i] · (1 + √(1−ε²))/2` — the cap-center in lattice coords,
    /// computed by `build_q_mpfr_zeta`. After LLL this is solved against
    /// `Bᵀ` to recover the cap-center in basis coords (`z_c`), which is the
    /// SE walk's recursion center.
    pub c: [RFloat; 16],

    // ── MPFR LU buffers at lu_prec (scales with ε) ──
    pub lu_prec: u32,
    pub lu_a: [[RFloat; 16]; 16],
    pub lu_rhs: [RFloat; 16],
    pub lu_x: [RFloat; 16],
    pub lu_tmp: RFloat,
    pub lu_acc: RFloat,
}

impl IntScratch16 {
    pub fn new(eps: Float) -> Self {
        let prec_q = compute_prec_q(eps);
        let gs_prec = GS_PREC;
        let lu_prec = compute_lu_prec(eps);
        Self {
            prec_q,
            gs_prec,
            scale_bits: 0,
            q_mpfr: rmat_zero_16(prec_q),
            q_int: imat_zero_16(),
            basis: identity_basis_16(),
            gram: imat_zero_16(),
            temp_bq: imat_zero_16(),
            r_bar: rmat_zero_16(gs_prec),
            mu_bar: rmat_zero_16(gs_prec),
            s_bar: rmat_zero_16(gs_prec),
            tmp_a: rfz(gs_prec),
            tmp_b: rfz(gs_prec),
            tmp_c: rfz(gs_prec),
            l_f64: [[0.0_f64; 16]; 16],
            c: rvec_zero_16(prec_q),
            lu_prec,
            lu_a: std::array::from_fn(|_| std::array::from_fn(|_| rfz(lu_prec))),
            lu_rhs: std::array::from_fn(|_| rfz(lu_prec)),
            lu_x: std::array::from_fn(|_| rfz(lu_prec)),
            lu_tmp: rfz(lu_prec),
            lu_acc: rfz(lu_prec),
        }
    }

    pub fn reset_basis(&mut self) {
        self.basis = identity_basis_16();
    }
}
