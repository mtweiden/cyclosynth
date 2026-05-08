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

// TARGET_BITS, GRAM_OVERFLOW_THRESHOLD_BITS, and compute_scale_bits live
// in lenstra_common — same values for both backends. (At d=16 the
// dimensional growth in the Gram budget is absorbed via the threshold
// headroom rather than a smaller TARGET_BITS.)
pub use crate::synthesis::lenstra_common::{
    compute_scale_bits, GRAM_OVERFLOW_THRESHOLD_BITS, TARGET_BITS,
};

/// MPFR Gram-Schmidt precision. Theorem 2 of Nguyen-Stehlé 2009 covers d ≤ 11
/// at L²-LLL parameters (δ=0.75, η=0.55) in f64; for d=16 the proof doesn't
/// apply, so we use MPFR. The algorithmic requirement is
/// `ℓ ≥ 10 + 2·log d − log ε + d·log ρ` (fplll's `l2_min_prec`), which is
/// ~32 bits at d=16, ε=1e-4 and ~42 bits at ε=1e-7. **80 bits** leaves
/// ~40-bit headroom over the deep-ε requirement and was empirically the
/// sweet spot in a sweep at ε=1e-4 (8% faster than 128, with no
/// correctness regressions and *better* lde landing). Configurable
/// per-construction via [`IntScratch16::with_gs_prec`].
pub const GS_PREC: u32 = 80;

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

    /// Z1 D&C optimisation: when true, `phase1_with_stop` skips the
    /// `reset_basis()` call and lets LLL warm-start from the previous
    /// call's reduced basis. Caller is responsible for setting this
    /// after the first call (so LLL gets a clean identity start once).
    /// Default: false (cold start, single-search behaviour).
    pub warm_lll: bool,

    /// Experimental f64 GS state: when true, `phase1_with_stop` calls
    /// `lll_f64::run_lll_16_f64` instead of the MPFR-based `run_lll_16`.
    /// Theorem 2 of Nguyen-Stehlé 2009 doesn't cover d=16 in f64, but
    /// fplll's `wrapper.cpp` tries `double` first at every dim — we test
    /// whether it converges in our regime. Default: false (MPFR path).
    pub use_f64_gs: bool,

    // ── f64 GS state (experimental, fplll-style) ──
    //
    // Parallel buffers to `r_bar`/`mu_bar`/`s_bar` but in plain f64.
    // Used by the experimental [`super::lll_f64`] path: bypasses MPFR for
    // the GS state during LLL. Theorem 2 of Nguyen-Stehlé 2009 doesn't
    // cover d=16 in f64, but fplll's wrapper tries `double` first at every
    // dim. If LLL converges and produces a valid unimodular basis, this
    // gives a ~5× per-LLL-iter speedup vs the MPFR path.
    pub r_bar_f64: [[f64; 16]; 16],
    pub mu_bar_f64: [[f64; 16]; 16],
    pub s_bar_f64: [[f64; 16]; 16],
}

impl IntScratch16 {
    pub fn new(eps: Float) -> Self {
        Self::with_gs_prec(eps, GS_PREC)
    }

    /// Construct a scratch with an overridden Gram-Schmidt precision.
    /// The default `GS_PREC=128` has ~78 bits of margin over the
    /// Nguyen-Stehlé requirement at ε=1e-7. Lower values trade
    /// correctness margin for faster MPFR ops in the LLL hot path.
    /// Used by Z1 D&C experiments and benchmarks.
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
            r_bar_f64: [[0.0; 16]; 16],
            mu_bar_f64: [[0.0; 16]; 16],
            s_bar_f64: [[0.0; 16]; 16],
            use_f64_gs: false,
        }
    }

    pub fn reset_basis(&mut self) {
        self.basis = identity_basis_16();
    }
}
