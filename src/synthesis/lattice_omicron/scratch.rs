//! Per-thread scratch buffers and shared infrastructure for the L²-LLL
//! pipeline: precision constants, type aliases, MPFR macros, and the
//! `IntScratch` struct that pre-allocates every buffer up front so the
//! inner LLL loop has zero allocation.
//!
//! This is the n=6 (Z[ξ], ξ=e^{iπ/6}) variant. The only difference from
//! the n=4 version (`lattice::scratch`) is `fill_sigma`, which uses the
//! n=6 embedding Σ (entries √3/2, 1/2, 1 instead of √2/2, 1).

#![allow(dead_code)]
// 8×8 matrix code reads more clearly with explicit (i, j) indexing.
#![allow(clippy::needless_range_loop)]

use crate::rings::Float;
use i256::i256;
use rug::{Assign, Float as RFloat};

// ─── Adaptive precision constants — re-exported from lattice_common ─────────

pub use crate::synthesis::lattice_common::{
    compute_scale_bits, GRAM_OVERFLOW_THRESHOLD_BITS, TARGET_BITS,
};

/// MPFR precision in bits used to construct the anisotropic Q metric.
/// `8·log₂(1/ε)` covers κ(Q) ≈ 16/ε⁴ with safety margin; floor at 100 bits
/// for moderate ε where the formula otherwise underflows.
pub fn compute_prec_q(eps: Float) -> u32 {
    if eps <= 0.0 { return 100; }
    let log_recip = (1.0 / eps).log2().max(1.0);
    let bits = (8.0 * log_recip).ceil() as u32;
    bits.max(100).min(4096)
}

/// MPFR precision used by the cap-center LU solve, scaled with ε.
///
/// The basis `B` has det=±1 but its entries grow with ε (up to ~2¹⁵ at
/// ε=1e-5, ~2⁴¹ at ε=1e-8). Partial-pivoting LU on this basis can develop
/// pivot ratios up to ~max(|B|)^(d-1) in pathological cases — usually
/// much tighter, but enough to consume meaningful precision at deep ε.
/// Empirically at ε=1e-8 a 96-bit LU loses enough precision in z_c that SE
/// misses the canonical-lde solution; 6·log₂(1/ε) bits leaves margin.
///
/// Versus `prec_q = 8·log₂(1/ε)` this is 75% of the precision, so each MPFR
/// op is ~1.3× cheaper.
pub fn compute_lu_prec(eps: Float) -> u32 {
    if eps <= 0.0 { return 96; }
    let log_recip = (1.0 / eps).log2().max(1.0);
    let bits = (6.0 * log_recip).ceil() as u32;
    bits.max(96).min(4096)
}

// ─── Type aliases ────────────────────────────────────────────────────────────

pub type IMat8 = [[i64; 8]; 8];
pub type Mat256 = [[i256; 8]; 8];

// ─── In-place MPFR op macros via gmp-mpfr-sys ────────────────────────────────

macro_rules! r_mul {
    ($dst:expr, $a:expr, $b:expr) => {
        unsafe {
            ::gmp_mpfr_sys::mpfr::mul(
                $dst.as_raw_mut(),
                $a.as_raw(),
                $b.as_raw(),
                ::gmp_mpfr_sys::mpfr::rnd_t::RNDN,
            );
        }
    };
}
macro_rules! r_add {
    ($dst:expr, $a:expr, $b:expr) => {
        unsafe {
            ::gmp_mpfr_sys::mpfr::add(
                $dst.as_raw_mut(),
                $a.as_raw(),
                $b.as_raw(),
                ::gmp_mpfr_sys::mpfr::rnd_t::RNDN,
            );
        }
    };
}
macro_rules! r_sub {
    ($dst:expr, $a:expr, $b:expr) => {
        unsafe {
            ::gmp_mpfr_sys::mpfr::sub(
                $dst.as_raw_mut(),
                $a.as_raw(),
                $b.as_raw(),
                ::gmp_mpfr_sys::mpfr::rnd_t::RNDN,
            );
        }
    };
}
macro_rules! r_div {
    ($dst:expr, $a:expr, $b:expr) => {
        unsafe {
            ::gmp_mpfr_sys::mpfr::div(
                $dst.as_raw_mut(),
                $a.as_raw(),
                $b.as_raw(),
                ::gmp_mpfr_sys::mpfr::rnd_t::RNDN,
            );
        }
    };
}

pub(crate) use r_add;
pub(crate) use r_div;
pub(crate) use r_mul;
pub(crate) use r_sub;

// ─── IntScratch struct ───────────────────────────────────────────────────────

/// Per-thread scratch for the L²-LLL pipeline. Allocated once per rayon
/// worker via `map_init`, reused across all MA prefixes that worker handles.
/// Every MPFR buffer is pre-allocated at `prec_q` bits up front; no
/// allocation happens inside the LLL inner loop.
pub struct IntScratch {
    /// MPFR precision used for build_q + Cholesky + LU (post-LLL phases).
    pub prec_q: u32,
    /// Adaptive scale `B` such that `Q_int[i][j] ≈ 2^B · Q[i][j]`. Picked
    /// per phase1 call so `max(|Q_int|) ≈ 2^TARGET_BITS`.
    pub scale_bits: i32,

    // ── MPFR buffers for build_q (constants + per-call working values) ──
    pub q_mpfr: [[RFloat; 8]; 8],
    pub c: [RFloat; 8],
    pub sigma: [[RFloat; 8]; 8],
    pub one: RFloat,
    pub two: RFloat,
    pub half: RFloat,
    pub tmp: RFloat,
    pub tmp2: RFloat,
    pub tmp3: RFloat,
    pub acc: RFloat,
    pub p_u: [[RFloat; 8]; 8],
    pub p_ub: [[RFloat; 8]; 8],
    pub yhat_yhat_t: [[RFloat; 8]; 8],
    pub y_rf: [RFloat; 8],
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

    // ── Integer LLL buffers ──
    pub q_int: Mat256,
    pub basis: IMat8,
    pub gram: Mat256,        // G = B · Q_int · Bᵀ
    pub temp_bq: Mat256,     // intermediate = B · Q_int

    // ── L²-LLL state (Nguyen-Stehlé 2009): pure f64. ──
    pub r_bar: [[f64; 8]; 8],
    pub mu_bar: [[f64; 8]; 8],
    pub s_bar: [[f64; 8]; 8],

    // ── post-LLL Cholesky output (f64 production path) ──
    pub l_f64: [[f64; 8]; 8],

    // ── MPFR Cholesky buffers (test-suite oracle only) ──
    pub g_post_lll: [[RFloat; 8]; 8],
    pub l: [[RFloat; 8]; 8],

    // ── MPFR LU buffers at lu_prec ──
    pub lu_prec: u32,
    pub lu_a: [[RFloat; 8]; 8],
    pub lu_rhs: [RFloat; 8],
    pub lu_x: [RFloat; 8],
    pub lu_tmp: RFloat,
    pub lu_acc: RFloat,
}

// ─── MPFR/i256 zero-fill helpers ─────────────────────────────────────────────

pub fn rfz(prec: u32) -> RFloat {
    RFloat::with_val(prec, 0.0_f64)
}

pub fn rfv(prec: u32, x: f64) -> RFloat {
    RFloat::with_val(prec, x)
}

pub fn rmat_zero(prec: u32) -> [[RFloat; 8]; 8] {
    std::array::from_fn(|_| std::array::from_fn(|_| rfz(prec)))
}

pub fn rvec_zero(prec: u32) -> [RFloat; 8] {
    std::array::from_fn(|_| rfz(prec))
}

pub fn imat_zero() -> Mat256 {
    let z = i256::from_i64(0);
    std::array::from_fn(|_| std::array::from_fn(|_| z))
}

pub fn identity_basis() -> IMat8 {
    std::array::from_fn(|i| {
        let mut row = [0i64; 8];
        row[i] = 1;
        row
    })
}

// ─── Σ-derived constants (filled once per IntScratch::new) ───────────────────

/// Populate Σ for n=6 (Z[ξ], ξ=e^{iπ/6}).
///
/// The 8×8 real embedding has row ordering (Re u, Im u, Re u•, Im u•,
/// Re t, Im t, Re t•, Im t•) and column ordering (a₀,a₁,a₂,a₃, b₀,b₁,b₂,b₃).
///
/// Concretely (s = √3/2):
///   Row 0 (Re u):   [ 1,  s,  ½,  0 | 0, 0, 0, 0 ]
///   Row 1 (Im u):   [ 0,  ½,  s,  1 | 0, 0, 0, 0 ]
///   Row 2 (Re u•):  [ 1, -s,  ½,  0 | 0, 0, 0, 0 ]  (bullet: √3→−√3)
///   Row 3 (Im u•):  [ 0,  ½, -s,  1 | 0, 0, 0, 0 ]
///   Row 4 (Re t):   [ 0, 0, 0, 0 | 1,  s,  ½,  0 ]
///   Row 5 (Im t):   [ 0, 0, 0, 0 | 0,  ½,  s,  1 ]
///   Row 6 (Re t•):  [ 0, 0, 0, 0 | 1, -s,  ½,  0 ]
///   Row 7 (Im t•):  [ 0, 0, 0, 0 | 0,  ½, -s,  1 ]
///
/// Entry encoding used in the pattern array:
///   0 → 0, 1 → 1, -1 → -1, 2 → √3/2, -2 → -√3/2, 3 → ½
fn fill_sigma(sigma: &mut [[RFloat; 8]; 8], prec: u32) {
    // Row × column pattern (see doc above).
    // Values: 0=0, 1=+1, -1=-1, 2=+√3/2, -2=-√3/2, 3=+½
    let pattern: [[i32; 8]; 8] = [
        [ 1,  2,  3,  0,  0,  0,  0,  0],  // Re u
        [ 0,  3,  2,  1,  0,  0,  0,  0],  // Im u
        [ 1, -2,  3,  0,  0,  0,  0,  0],  // Re u• (bullet: √3 → -√3)
        [ 0,  3, -2,  1,  0,  0,  0,  0],  // Im u•
        [ 0,  0,  0,  0,  1,  2,  3,  0],  // Re t
        [ 0,  0,  0,  0,  0,  3,  2,  1],  // Im t
        [ 0,  0,  0,  0,  1, -2,  3,  0],  // Re t•
        [ 0,  0,  0,  0,  0,  3, -2,  1],  // Im t•
    ];

    // Precompute the distinct values at full MPFR precision.
    let zero  = rfv(prec, 0.0);
    let one   = rfv(prec, 1.0);
    let mut none = rfz(prec); none.assign(-&one);
    // √3/2
    let mut s32 = rfz(prec);
    s32.assign(rfv(prec, 3.0_f64).sqrt());
    s32 /= 2u32;
    let mut ns32 = rfz(prec); ns32.assign(-&s32);
    // ½
    let half = rfv(prec, 0.5);

    for i in 0..8 {
        for j in 0..8 {
            sigma[i][j].assign(match pattern[i][j] {
                 1 => &one,
                -1 => &none,
                 2 => &s32,
                -2 => &ns32,
                 3 => &half,
                 _ => &zero,
            });
        }
    }
}

/// Compute `p_u[i][j]` and `p_ub[i][j]` using the n=6 row ordering
/// (Re u, Im u, Re u•, Im u•, Re t, Im t, Re t•, Im t•).
///
/// Standard (non-bullet) rows: {0, 1, 4, 5}
/// Bullet rows:                {2, 3, 6, 7}
///
/// p_u[i][j]  = ½·Σ_{r ∈ standard rows} σ[r][i]·σ[r][j]
/// p_ub[i][j] = ½·Σ_{r ∈ bullet rows}   σ[r][i]·σ[r][j]
fn fill_p_u_p_ub(scratch: &mut IntScratch) {
    let std_rows: [usize; 4] = [0, 1, 4, 5];
    let bullet_rows: [usize; 4] = [2, 3, 6, 7];

    for i in 0..8 {
        for j in 0..8 {
            scratch.acc.assign(0.0_f64);
            scratch.tmp2.assign(0.0_f64);
            for &r in &std_rows {
                r_mul!(scratch.tmp, scratch.sigma[r][i], scratch.sigma[r][j]);
                let acc_clone = scratch.acc.clone();
                r_add!(scratch.acc, acc_clone, scratch.tmp);
            }
            for &r in &bullet_rows {
                r_mul!(scratch.tmp, scratch.sigma[r][i], scratch.sigma[r][j]);
                let tmp2_clone = scratch.tmp2.clone();
                r_add!(scratch.tmp2, tmp2_clone, scratch.tmp);
            }
            r_mul!(scratch.p_u[i][j], scratch.acc, scratch.half);
            r_mul!(scratch.p_ub[i][j], scratch.tmp2, scratch.half);
        }
    }
}

impl IntScratch {
    pub fn new(eps: Float) -> Self {
        let prec_q = compute_prec_q(eps);
        let lu_prec = compute_lu_prec(eps);
        let mut s = Self {
            prec_q,
            scale_bits: 0,
            q_mpfr: rmat_zero(prec_q),
            c: rvec_zero(prec_q),
            sigma: rmat_zero(prec_q),
            one: rfv(prec_q, 1.0),
            two: rfv(prec_q, 2.0),
            half: rfv(prec_q, 0.5),
            tmp: rfz(prec_q),
            tmp2: rfz(prec_q),
            tmp3: rfz(prec_q),
            acc: rfz(prec_q),
            p_u: rmat_zero(prec_q),
            p_ub: rmat_zero(prec_q),
            yhat_yhat_t: rmat_zero(prec_q),
            y_rf: rvec_zero(prec_q),
            eps_rf: rfz(prec_q),
            r: rfz(prec_q),
            r_sq: rfz(prec_q),
            delta_y: rfz(prec_q),
            delta_perp: rfz(prec_q),
            inv_dy_sq: rfz(prec_q),
            inv_dp_sq: rfz(prec_q),
            inv_r_sq: rfz(prec_q),
            y_norm_sq: rfz(prec_q),
            inv_y_norm_sq: rfz(prec_q),
            cap_mid: rfz(prec_q),
            q_int: imat_zero(),
            basis: identity_basis(),
            gram: imat_zero(),
            temp_bq: imat_zero(),
            r_bar: [[0.0_f64; 8]; 8],
            mu_bar: [[0.0_f64; 8]; 8],
            s_bar: [[0.0_f64; 8]; 8],
            l_f64: [[0.0_f64; 8]; 8],
            g_post_lll: rmat_zero(prec_q),
            l: rmat_zero(prec_q),
            lu_prec,
            lu_a: rmat_zero(lu_prec),
            lu_rhs: rvec_zero(lu_prec),
            lu_x: rvec_zero(lu_prec),
            lu_tmp: rfz(lu_prec),
            lu_acc: rfz(lu_prec),
        };
        fill_sigma(&mut s.sigma, prec_q);
        fill_p_u_p_ub(&mut s);
        s
    }

    pub fn reset_basis(&mut self) {
        self.basis = identity_basis();
    }
}
