//! Per-thread scratch buffers for the 16D Z[ζ_16] L²-LLL pipeline, the
//! dimension-16 analog of `super::super::lattice::scratch`. All MPFR/i256
//! storage is allocated up front so the inner LLL loop never allocates.
//! MPFR Gram-Schmidt is mandatory at d=16 (Theorem 2 of Nguyen-Stehlé 2009
//! covers only d ≤ 11 in f64). Gram entries reach ~2^214 (= 16·2^30·2^180
//! from B² · Q_int), under the 2^240 overflow threshold; only extreme deep
//! ε trips `GramOverflow`, where the caller must escalate the integer type.

#![allow(clippy::needless_range_loop)]

use crate::rings::Float;
use i256::i256;
use rug::Float as RFloat;

// ─── Adaptive precision constants ────────────────────────────────────────────

pub use crate::synthesis::lattice::common::{
    compute_lu_prec, compute_prec_q, compute_scale_bits, rfv, rfz,
    GRAM_OVERFLOW_THRESHOLD_BITS, TARGET_BITS,
};

/// MPFR Gram-Schmidt precision. d=16 is outside the NS09 f64 proof (d ≤ 11),
/// and fplll's `l2_min_prec` needs ~42 bits at ε=1e-7; 80 leaves ~40-bit
/// headroom and was the fastest in a sweep. Per-construction override:
/// [`IntScratch16::with_gs_prec`].
pub const GS_PREC: u32 = 80;

// ─── Type aliases ────────────────────────────────────────────────────────────

pub type IMat16 = [[i64; 16]; 16];
pub type Mat256_16 = [[i256; 16]; 16];

// ─── Helpers ─────────────────────────────────────────────────────────────────

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
    // s_bar[i][j] = ‖b_i‖² - Σ_{k<j} mu_bar[i][k] · r_bar[i][k]
    //                (Lovász partial sums; s_bar[i][0] = ‖b_i‖² = gram[i][i],
    //                 and r_bar[i][i] is defined as the final s_bar[i][i])
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

    /// When true, the LLL warm-starts from the previous call's reduced
    /// basis instead of resetting to identity. The caller sets it after
    /// the first call (which needs a clean start). Default false.
    pub warm_lll: bool,

    /// Per-(k, ε) warm-LLL seed: the LLL-reduced basis of the
    /// prefix-independent part of the metric (everything but the rank-1
    /// per-prefix term). Computed lazily once per (k, ε) and reused via
    /// `warm_lll`; `None` after a non-converged seed reduction.
    pub q_base_seed: Option<IMat16>,
    pub q_base_seed_key: Option<(u32, u64)>,

    /// Use the experimental f64 Gram-Schmidt LLL (`lll_f64::run_lll_16_f64`)
    /// instead of MPFR — outside the NS09 d ≤ 11 proof, but fplll tries f64
    /// first at every dim. Default false.
    pub use_f64_gs: bool,

    /// BKZ block size (0 = off, 3..=8 = β-block SVP tours after LLL,
    /// strengthening the basis most at deep ε). β=2 is LLL-equivalent.
    pub bkz_block_size: u32,

    // f64 mirrors of r_bar/mu_bar/s_bar for the lll_f64 path (~5× per-iter
    // faster when it converges).
    pub r_bar_f64: [[f64; 16]; 16],
    pub mu_bar_f64: [[f64; 16]; 16],
    pub s_bar_f64: [[f64; 16]; 16],
}

impl IntScratch16 {
    pub fn new(eps: Float) -> Self {
        Self::with_gs_prec(eps, GS_PREC)
    }

    /// Construct with an overridden Gram-Schmidt precision (lower = faster
    /// MPFR ops, less correctness margin). The default is [`GS_PREC`].
    pub fn with_gs_prec(eps: Float, gs_prec: u32) -> Self {
        let prec_q = compute_prec_q(eps);
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
            warm_lll: false,
            q_base_seed: None,
            q_base_seed_key: None,
            r_bar_f64: [[0.0; 16]; 16],
            mu_bar_f64: [[0.0; 16]; 16],
            s_bar_f64: [[0.0; 16]; 16],
            use_f64_gs: false,
            bkz_block_size: 0,
        }
    }

    pub fn reset_basis(&mut self) {
        self.basis = identity_basis_16();
    }
}
