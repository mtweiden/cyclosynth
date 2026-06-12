//! Per-thread scratch buffers and shared infrastructure for the L²-LLL
//! pipeline: precision constants, type aliases, MPFR macros, and the
//! `IntScratch` struct that pre-allocates every buffer up front so the
//! inner LLL loop has zero allocation.

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
    let log_recip = (1.0 / eps).log2().max(1.0);
    let bits = (8.0 * log_recip).ceil() as u32;
    bits.max(100)
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
/// op is ~1.3× cheaper. Applied to ~13 s of LU CPU at lde=80 this saves
/// ~3-4 s CPU, ~0.4 s wall.
pub fn compute_lu_prec(eps: Float) -> u32 {
    let log_recip = (1.0 / eps).log2().max(1.0);
    let bits = (6.0 * log_recip).ceil() as u32;
    bits.max(96)
}

// ─── Type aliases ────────────────────────────────────────────────────────────

pub type IMat8 = [[i64; 8]; 8];
pub type Mat256 = [[i256; 8]; 8];

// ─── In-place MPFR op macros via gmp-mpfr-sys ────────────────────────────────
//
// Each macro calls the corresponding `mpfr::{add,sub,mul,div}` directly on
// the underlying `mpfr_t` (via `as_raw_mut` / `as_raw`). The naive
// `$dst.assign(&$a OP &$b)` pattern allocates a `rug::Incomplete` struct per
// op; this version is zero-allocation. mpfr's API explicitly permits aliasing
// rop with op1/op2, so dst == a or dst == b is safe. Macros use absolute
// paths so importers don't need a matching `use`.

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
    /// per find_aligned_lattice_points call so `max(|Q_int|) ≈ 2^TARGET_BITS`.
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

    // ── Q_base hoist (stage 4, docs/plan_8d_prefix_rework.md lever C) ──
    /// Prefix-independent part of the Q metric:
    /// `q_base[i][j] = inv_dp_sq·p_u[i][j] + inv_r_sq·p_ub[i][j]`.
    /// Valid for the `(k, eps)` recorded in `q_base_key`; rebuilt by
    /// `build_q_mpfr` only when the key changes (within one `prefix_split_search`
    /// level k is fixed, so this runs once per worker per level).
    pub q_base: [[RFloat; 8]; 8],
    /// Scalar weight of the prefix-dependent rank-1 term:
    /// `coef_y = inv_dy_sq − inv_dp_sq` (no cancellation — the terms
    /// differ by a factor ≈ (4/ε)²), so
    /// `Q = q_base + (coef_y/‖y‖²)·y·yᵀ`.
    pub coef_y: RFloat,
    /// `(k, eps.to_bits())` the cached q_base/coef_y/cap_mid were built
    /// for; `None` until the first build_q_mpfr call.
    pub q_base_key: Option<(u32, u64)>,
    /// LLL-reduced unimodular basis of `q_base` alone, used to warm-seed
    /// every per-prefix LLL at the same `(k, ε)`: the prefix-dependent
    /// rank-1 term carries ~half the anisotropy bits, so the Q_base
    /// reduction is most of the shared work (measured warm/cold iters
    /// ≈ 0.60 on 400-prefix captures, `warm_lll_gate` test). Keyed
    /// separately from `q_base_key`: computed lazily by `find_aligned_lattice_points` (it
    /// needs an LLL run, which `build_q_mpfr` must not recurse into).
    pub q_base_seed: Option<IMat8>,
    pub q_base_seed_key: Option<(u32, u64)>,

    // ── Integer LLL buffers ──
    pub q_int: Mat256,
    pub basis: IMat8,
    pub gram: Mat256,        // G = B · Q_int · Bᵀ
    pub temp_bq: Mat256,     // intermediate = B · Q_int

    // ── L²-LLL state (Nguyen-Stehlé 2009): pure f64. Theorem 2 + Figure 7
    // prove this precision is sufficient for d ≤ 11; we operate at d=8.
    //
    // r̄_{i,j} = <b_i*, b_j*> for i ≥ j  (FP-approx of GSO inner products)
    // μ̄_{i,j} = r̄_{i,j} / r̄_{j,j}     (FP-approx GSO coefficients)
    // s̄_j^{(i)} = r̄_{i,i} - Σ_{k<j} μ̄_{i,k}·r̄_{i,k}  (Lovász partial sums)
    //
    // The exact integer Gram lives in `gram` (i256). f64 entries are derived
    // on demand via i256→f64 (mantissa truncation; exponent has 1024-bit
    // range so our 2^240 max gram entries fit with no overflow).
    pub r_bar: [[f64; 8]; 8],
    pub mu_bar: [[f64; 8]; 8],
    pub s_bar: [[f64; 8]; 8],

    // ── post-LLL Cholesky output (f64 production path) ──
    /// Lower-triangular Cholesky factor of the natural-scale post-LLL Gram,
    /// computed in f64 directly from the i256 Gram. Justified by the LLL
    /// invariant κ(G) ≤ (4/3)^(d-1) ≤ 16 at d=8: f64's 53-bit mantissa
    /// yields ~10⁻¹⁵ absolute error at the SE unit-scale bound check, six
    /// orders of magnitude below SE's 10⁻⁹ tolerance.
    pub l_f64: [[f64; 8]; 8],

    // ── MPFR Cholesky buffers (test-suite oracle only) ──
    /// Kept so the test suite can run `cholesky_int_8` as a reference oracle
    /// against `cholesky_f64_8` across ε regimes. Not used in production.
    pub g_post_lll: [[RFloat; 8]; 8],
    pub l: [[RFloat; 8]; 8],

    // ── MPFR LU buffers at lu_prec (scales with ε, ~75% of prec_q) ──
    /// Decoupled from `prec_q` so each MPFR op in the LU runs at lower
    /// precision (~1.3× cheaper) without affecting build_q's higher-precision
    /// requirement.
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

/// Populate Σ — the 8×8 real embedding of Z[ω,√2]² into ℝ⁸ via two Galois
/// embeddings (rows 0–3: √2 → +√2; rows 4–7: √2 → −√2). Pattern entries
/// {0, ±1, ±2} map to {0, ±1, ±1/√2}.
fn fill_sigma(sigma: &mut [[RFloat; 8]; 8], prec: u32) {
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
    let r2 = two.sqrt().recip();
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

/// Compute `p_u[i][j] = ½·Σ_{r=0..3} σ[r][i]·σ[r][j]` and
/// `p_ub[i][j] = ½·Σ_{r=4..7} σ[r][i]·σ[r][j]`. Depends only on the
/// constant Σ matrix, so it runs once at scratch construction; the values
/// persist across every build_q_mpfr call. Eliminates ~512 MPFR mul + 512
/// MPFR add ops per find_aligned_lattice_points invocation.
fn fill_p_u_p_ub(scratch: &mut IntScratch) {
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
            q_base: rmat_zero(prec_q),
            coef_y: rfz(prec_q),
            q_base_key: None,
            q_base_seed: None,
            q_base_seed_key: None,
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
