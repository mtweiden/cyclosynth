//! Per-thread scratch buffers for the 16D Z[ζ_16] L²-LLL pipeline, the
//! dimension-16 analog of `super::super::omega::scratch`. All MPFR/i256
//! storage is allocated up front so the inner LLL loop never allocates.
//! MPFR Gram-Schmidt is mandatory at d=16 (Theorem 2 of Nguyen-Stehlé 2009
//! covers only d ≤ 11 in f64). Gram entries reach ~2^214 (= 16·2^30·2^180
//! from B² · Q_int), under the 2^240 overflow threshold; only extreme deep
//! ε trips `GramOverflow`, where the caller must escalate the integer type.

#![allow(clippy::needless_range_loop)]

use i256::i256;
use crate::rings::MpFloat;

// ─── Adaptive precision constants ────────────────────────────────────────────

pub use crate::synthesis::lattice::common::{
    compute_lu_prec, compute_prec_q, compute_scale_bits, identity_basis, imat_zero,
    rfv, rfz, rmat_zero, rvec_zero, GRAM_OVERFLOW_THRESHOLD_BITS, TARGET_BITS,
};

/// MPFR Gram-Schmidt precision. d=16 is outside the NS09 f64 proof (d ≤ 11),
/// and fplll's `l2_min_prec` needs ~42 bits at ε=1e-7; 80 leaves ~40-bit
/// headroom while staying the fastest precision in practice. Per-construction
/// override: [`IntScratch16::with_gs_prec`].
pub const GS_PREC: u32 = 80;

// ─── Type aliases ────────────────────────────────────────────────────────────

pub type IMat16 = [[i64; 16]; 16];
pub type Mat256_16 = [[i256; 16]; 16];


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
    pub q_mpfr: [[MpFloat; 16]; 16],

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
    pub r_bar: [[MpFloat; 16]; 16],
    pub mu_bar: [[MpFloat; 16]; 16],
    pub s_bar: [[MpFloat; 16]; 16],

    // ── Scratch MpFloats reused inside LLL ──
    pub tmp_a: MpFloat,
    pub tmp_b: MpFloat,

    // ── post-LLL Cholesky output (f64) ──
    pub l_f64: [[f64; 16]; 16],

    // ── Cap center in MPFR (for the post-LLL LU solve) ──
    /// `c[i] = y[i] · (1 + √(1−ε²))/2` — the cap-center in lattice coords,
    /// computed by `build_q_mpfr_zeta`. After LLL this is solved against
    /// `Bᵀ` to recover the cap-center in basis coords (`z_c`), which is the
    /// SE walk's recursion center.
    pub c: [MpFloat; 16],

    // ── MPFR LU buffers at lu_prec (scales with ε) ──
    pub lu_prec: u32,
    pub lu_a: [[MpFloat; 16]; 16],
    pub lu_rhs: [MpFloat; 16],
    pub lu_x: [MpFloat; 16],
    pub lu_tmp: MpFloat,
    pub lu_acc: MpFloat,

    /// BKZ block size (0 = off, 3..=8 = β-block SVP tours after LLL,
    /// strengthening the basis most at deep ε). β=2 is LLL-equivalent.
    pub bkz_block_size: u32,

    /// Re-check every f64 prune-fire of the SE walk's Euclidean partial-norm
    /// prune at dd precision (the dd verdict wins). Needed at ε < 2e-8 where
    /// the f64 dot product suffers catastrophic cancellation and silently
    /// drops valid candidates; pure overhead at shallower ε. Defaults to
    /// `false`; the `clifford_sqrt_t` synthesize drivers set it per scratch
    /// (see `first_hit::verify_prune_mpfr_for`), and
    /// `find_aligned_lattice_points_mpfr` copies it onto the walk's
    /// `SeCenter16`.
    pub(crate) verify_prune_mpfr: bool,
}

impl IntScratch16 {
    pub fn new(eps: f64) -> Self {
        Self::with_gs_prec(eps, GS_PREC)
    }

    /// Construct with an overridden Gram-Schmidt precision (lower = faster
    /// MPFR ops, less correctness margin). The default is [`GS_PREC`].
    pub fn with_gs_prec(eps: f64, gs_prec: u32) -> Self {
        let prec_q = compute_prec_q(eps);
        let lu_prec = compute_lu_prec(eps);
        Self {
            prec_q,
            gs_prec,
            scale_bits: 0,
            q_mpfr: rmat_zero(prec_q),
            q_int: imat_zero(),
            basis: identity_basis(),
            gram: imat_zero(),
            temp_bq: imat_zero(),
            r_bar: rmat_zero(gs_prec),
            mu_bar: rmat_zero(gs_prec),
            s_bar: rmat_zero(gs_prec),
            tmp_a: rfz(gs_prec),
            tmp_b: rfz(gs_prec),
            l_f64: [[0.0_f64; 16]; 16],
            c: rvec_zero(prec_q),
            lu_prec,
            lu_a: std::array::from_fn(|_| std::array::from_fn(|_| rfz(lu_prec))),
            lu_rhs: std::array::from_fn(|_| rfz(lu_prec)),
            lu_x: std::array::from_fn(|_| rfz(lu_prec)),
            lu_tmp: rfz(lu_prec),
            lu_acc: rfz(lu_prec),
            bkz_block_size: 0,
            verify_prune_mpfr: false,
        }
    }

    pub fn reset_basis(&mut self) {
        self.basis = identity_basis();
    }
}
