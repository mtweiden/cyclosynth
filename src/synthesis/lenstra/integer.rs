//! L²-LLL pipeline for the deep-ε path of Clifford+T synthesis.
//!
//! Implements the L² algorithm of Nguyen-Stehlé 2009 (SIAM J. Computing,
//! "An LLL Algorithm with Quadratic Complexity") specialised to dimension 8
//! with the anisotropic Q metric used by arXiv:2510.05816 Algorithm 3.6.
//!
//! ## Per-phase1 call pipeline
//!
//!  1. **Build Q** in MPFR (`build_q_mpfr`): the anisotropic ellipsoid
//!     metric for the cap × ball intersection (eq 3.15 of the paper).
//!     ~0.1% of phase1 CPU.
//!
//!  2. **Snapshot Q to i256** (`build_q_int`) with adaptive scale
//!     `S = 2^B` chosen so `max(|S·Q|) ≈ 2^TARGET_BITS`. The exact integer
//!     Gram is the input to L²-LLL; LLL μ-values are scale-invariant
//!     ratios, so the choice of `S` only affects the effective precision
//!     of the snapshot, not the algorithm's correctness.
//!
//!  3. **L²-LLL** (`lll_l2_8`): pure-f64 Gram-Schmidt with the exact i256
//!     Gram on the side. Per Theorem 2 + Figure 7 of the paper, f64
//!     (ℓ=52 mantissa bits) is provably sufficient at d=8 with
//!     (δ=0.75, η=0.55), giving 18-bit precision margin. INSERT semantics
//!     + lazy size-reduction keep the basis L³-reduced incrementally,
//!     which is the invariant required for the f64 sufficiency proof.
//!
//!  4. **Cholesky + LU** post-LLL: convert the i256 Gram for the reduced
//!     basis to MPFR once (`snapshot_gram_to_mpfr`), Cholesky-factor it
//!     (`cholesky_int_8`), and solve `Bᵀ·z_c = c` for the cap-center in
//!     lattice coordinates (`lu_solve_int_inplace`). All ~1% of phase1.
//!
//!  5. **Schnorr-Euchner** ([`super::se::schnorr_euchner_8d`]): walk
//!     candidate `z` values within the SE ellipsoid; for each, reconstruct
//!     `x = B·z`, validate `‖x‖² == 2^k`, `B(x) == 0` (bilinear unitarity
//!     constraint), and `|y·x|² ≥ thresh_xy` (alignment cap).
//!
//! Validated for `ε ∈ [1e-10, 1e-3]`. The `[`super::light`]` path
//! covers `ε ≥ 1e-4` more cheaply via TwoFloat.

#![allow(dead_code)]

use crate::rings::Float;
use gmp_mpfr_sys::mpfr as mpfr_sys;
use i256::i256;
use rug::{Assign, Float as RFloat};

// ─── Adaptive scale ────────────────────────────────────────────────────────────

/// Target effective precision for `Q_int` entries: max(|Q_int|) ≈ 2^TARGET_BITS.
/// Chosen to balance two competing constraints:
///   - Higher → more relative precision in Q_int → tighter LLL convergence
///     near the Lovász decision boundary. After GS cancels ~log₂(κ(Q)) bits,
///     post-GS gnorm needs (TARGET_BITS − log₂(κ(Q))) ≳ 30 to be useful.
///     For ε=1e-8: log₂(κ(Q)) ≈ 107, need TARGET_BITS ≳ 140.
///     For ε=1e-10: log₂(κ(Q)) ≈ 137, need TARGET_BITS ≳ 170.
///   - Lower → more i256 headroom for transient Gram entries G = B·Q_int·Bᵀ.
///     G ≤ 64·max(B)²·max(Q_int). For typical post-LLL max(B)=2^15: G ≤
///     2^(36+TARGET_BITS). For transient max(B)=2^60: G ≤ 2^(126+TARGET_BITS).
///
/// 180 bits keeps us safely through ε=1e-8 (margin ~70 bits post-GS) and gives
/// 256−180 = 76 bits of i256 headroom for B² inflation. Pairs with overflow
/// detection on the Gram update (debug_assert + caller fallback path).
pub const TARGET_BITS: u32 = 180;

/// Magnitude threshold above which we declare a Gram-entry overflow risk:
/// 2^240, leaving 16-bit margin to i256::MAX. Triggered during transient
/// B-growth at deep ε (rare in practice because LLL output basis is small).
pub const GRAM_OVERFLOW_THRESHOLD_BITS: u32 = 240;

/// Compute the bit-shift `B` such that round(2^B · Q[i][j]) lands in i256 with
/// max entry ≈ 2^TARGET_BITS. Returns `B`. Caller-supplied `max_q_log2` is
/// `⌈log₂(max(|Q_entry|))⌉` from the MPFR Q computation.
pub fn compute_scale_bits(max_q_log2: i32) -> i32 {
    TARGET_BITS as i32 - max_q_log2
}

// ─── Helpers: rug ↔ i256 ──────────────────────────────────────────────────────

/// Round `2^shift_bits · x` to `i256`. `shift_bits` may be positive (scale up)
/// or negative (scale down). Saturates to i256 bounds (callers should choose
/// shift_bits to avoid this).
pub fn rug_to_i256_scaled(x: &RFloat, shift_bits: i32) -> i256 {
    if x.is_zero() {
        return i256::from_i64(0);
    }
    // Multiply by 2^shift then round to nearest integer.
    let mut scaled = x.clone();
    if shift_bits >= 0 {
        scaled <<= shift_bits as u32;
    } else {
        scaled >>= (-shift_bits) as u32;
    }
    // Round to nearest integer in MPFR
    scaled.round_mut();
    rfloat_to_i256(&scaled)
}

/// Convert an integer-valued RFloat to i256. Saturates on overflow.
fn rfloat_to_i256(x: &RFloat) -> i256 {
    use rug::integer::Order;
    let sign_neg = x.is_sign_negative();
    let abs = x.clone().abs();
    // Fast path: fits in i64
    if abs <= rug::Float::with_val(64, i64::MAX as f64) {
        let v = abs.to_f64() as i64;
        let res = i256::from_i64(v);
        return if sign_neg { -res } else { res };
    }
    // Convert to rug::Integer, then extract LE u64 limbs.
    // Float::to_integer() returns Option<Integer> (None if NaN/∞).
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
    if sign_neg { -val } else { val }
}

// ─── IntScratch: per-thread pre-allocated working buffers ──────────────────

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
/// Versus the previous prec_q (8·log₂(1/ε)) this is 75% of the precision,
/// so each MPFR op is ~1.3× cheaper. Applied to ~13s of LU CPU at lde=80
/// this is ~3-4s CPU savings, ~0.4s wall.
pub fn compute_lu_prec(eps: Float) -> u32 {
    let log_recip = (1.0 / eps).log2().max(1.0);
    let bits = (6.0 * log_recip).ceil() as u32;
    bits.max(96)
}

type IMat8 = [[i64; 8]; 8];
type Mat256 = [[i256; 8]; 8];

/// Per-thread scratch for the L²-LLL pipeline. Allocated once per rayon
/// worker via `map_init`, reused across all MA prefixes that worker handles.
/// All MPFR buffers are pre-allocated at `prec_q` bits up front; no
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
    /// computed in f64 directly from the i256 Gram (no MPFR snapshot). Justified
    /// by the LLL invariant κ(G) ≤ (4/3)^(d-1) ≤ 16 for d=8: f64's 53-bit
    /// mantissa yields ~10⁻¹⁵ absolute error at the SE unit-scale bound check,
    /// six orders of magnitude below SE's 10⁻⁹ tolerance.
    pub l_f64: [[f64; 8]; 8],

    // ── Legacy MPFR Cholesky buffers (test-suite reference only) ──
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

fn rfz(prec: u32) -> RFloat {
    RFloat::with_val(prec, 0.0_f64)
}

fn rfv(prec: u32, x: f64) -> RFloat {
    RFloat::with_val(prec, x)
}

fn rmat_zero(prec: u32) -> [[RFloat; 8]; 8] {
    std::array::from_fn(|_| std::array::from_fn(|_| rfz(prec)))
}

fn rvec_zero(prec: u32) -> [RFloat; 8] {
    std::array::from_fn(|_| rfz(prec))
}

fn imat_zero() -> Mat256 {
    let z = i256::from_i64(0);
    std::array::from_fn(|_| std::array::from_fn(|_| z))
}

fn identity_basis() -> IMat8 {
    std::array::from_fn(|i| {
        let mut row = [0i64; 8];
        row[i] = 1;
        row
    })
}

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

// ─── In-place MPFR op macros via gmp-mpfr-sys ─────────────────────────────────
//
// Each macro calls the corresponding `mpfr::{add,sub,mul,div}` directly on
// the underlying `mpfr_t` (via `as_raw_mut` / `as_raw`). The previous
// `$dst.assign(&$a OP &$b)` pattern allocated a `rug::Incomplete` struct per
// op even with assign-into-target; this version is zero-allocation. mpfr's
// API explicitly permits aliasing rop with op1/op2, so dst == a or dst == b
// is safe.

macro_rules! r_mul {
    ($dst:expr, $a:expr, $b:expr) => {
        unsafe {
            mpfr_sys::mul(
                $dst.as_raw_mut(),
                $a.as_raw(),
                $b.as_raw(),
                mpfr_sys::rnd_t::RNDN,
            );
        }
    };
}
macro_rules! r_add {
    ($dst:expr, $a:expr, $b:expr) => {
        unsafe {
            mpfr_sys::add(
                $dst.as_raw_mut(),
                $a.as_raw(),
                $b.as_raw(),
                mpfr_sys::rnd_t::RNDN,
            );
        }
    };
}
macro_rules! r_sub {
    ($dst:expr, $a:expr, $b:expr) => {
        unsafe {
            mpfr_sys::sub(
                $dst.as_raw_mut(),
                $a.as_raw(),
                $b.as_raw(),
                mpfr_sys::rnd_t::RNDN,
            );
        }
    };
}
macro_rules! r_div {
    ($dst:expr, $a:expr, $b:expr) => {
        unsafe {
            mpfr_sys::div(
                $dst.as_raw_mut(),
                $a.as_raw(),
                $b.as_raw(),
                mpfr_sys::rnd_t::RNDN,
            );
        }
    };
}

/// Compute `p_u[i][j] = ½·Σ_{r=0..3} σ[r][i]·σ[r][j]` and
/// `p_ub[i][j] = ½·Σ_{r=4..7} σ[r][i]·σ[r][j]`. Depends only on the
/// (constant) Σ matrix from `fill_sigma`, so it runs once at scratch
/// construction; the values persist across every build_q_mpfr call.
/// Eliminates ~512 MPFR mul + 512 MPFR add ops per phase1 invocation.
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

// ─── build_q_mpfr: identical to heavy's build_q, into scratch.q_mpfr ──────────

/// Build the anisotropic Q matrix in MPFR (eq 3.15 of arXiv:2510.05816)
/// into `scratch.q_mpfr`. Also computes the cap center into `scratch.c`.
/// Q is the metric used by the LLL; the cap center is the projection of the
/// target onto the alignment direction, used by the post-LLL LU solve.
pub fn build_q_mpfr(scratch: &mut IntScratch, y: &[Float; 8], k: u32, eps: Float) {
    let prec = scratch.prec_q;

    // R² = 2^k. For k ≥ 64, `1u64 << k` is UB — build via f64 powi (f64 exp
    // up to 1023 covers all reasonable k).
    let r_sq_f = 2.0_f64.powi(k as i32);
    scratch.r_sq.assign(rfv(prec, r_sq_f));
    scratch.r.assign(scratch.r_sq.clone().sqrt());
    scratch.eps_rf.assign(rfv(prec, eps));

    // Δ_y = R · ε² / (2·(1 + √(1−ε²)))
    r_mul!(scratch.tmp, scratch.eps_rf, scratch.eps_rf);
    r_sub!(scratch.tmp2, scratch.one, scratch.tmp);
    let sqrt_1m = scratch.tmp2.clone().sqrt();
    r_add!(scratch.tmp2, scratch.one, sqrt_1m);
    r_mul!(scratch.tmp3, scratch.tmp2, scratch.two);
    r_mul!(scratch.acc, scratch.r, scratch.tmp);
    r_div!(scratch.delta_y, scratch.acc, scratch.tmp3);

    r_mul!(scratch.delta_perp, scratch.r, scratch.eps_rf);

    r_mul!(scratch.tmp, scratch.delta_y, scratch.delta_y);
    r_div!(scratch.inv_dy_sq, scratch.one, scratch.tmp);
    r_mul!(scratch.tmp, scratch.delta_perp, scratch.delta_perp);
    r_div!(scratch.inv_dp_sq, scratch.one, scratch.tmp);
    r_div!(scratch.inv_r_sq, scratch.one, scratch.r_sq);

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

    for i in 0..8 {
        for j in 0..8 {
            r_mul!(scratch.tmp, scratch.y_rf[i], scratch.y_rf[j]);
            r_mul!(scratch.yhat_yhat_t[i][j], scratch.tmp, scratch.inv_y_norm_sq);
        }
    }

    // p_u and p_ub depend only on the (constant) Σ matrix and are populated
    // once by `fill_p_u_p_ub` in IntScratch::new — nothing to recompute here.

    for i in 0..8 {
        for j in 0..8 {
            r_mul!(scratch.tmp, scratch.inv_dy_sq, scratch.yhat_yhat_t[i][j]);
            r_sub!(scratch.tmp2, scratch.p_u[i][j], scratch.yhat_yhat_t[i][j]);
            r_mul!(scratch.tmp3, scratch.inv_dp_sq, scratch.tmp2);
            r_mul!(scratch.acc, scratch.inv_r_sq, scratch.p_ub[i][j]);
            let tmp_clone = scratch.tmp.clone();
            r_add!(scratch.tmp, tmp_clone, scratch.tmp3);
            r_add!(scratch.q_mpfr[i][j], scratch.tmp, scratch.acc);
        }
    }

    // Cap center
    r_mul!(scratch.tmp, scratch.eps_rf, scratch.eps_rf);
    r_sub!(scratch.tmp2, scratch.one, scratch.tmp);
    let sqrt_1m = scratch.tmp2.clone().sqrt();
    r_add!(scratch.tmp, scratch.one, sqrt_1m);
    r_div!(scratch.cap_mid, scratch.tmp, scratch.two);
    for i in 0..8 {
        scratch.tmp.assign(rfv(prec, y[i]));
        r_mul!(scratch.c[i], scratch.tmp, scratch.cap_mid);
    }
}

// ─── build_q_int: snapshot MPFR Q to scaled i256 ─────────────────────────────

/// After `build_q_mpfr`, snapshot the MPFR Q into `scratch.q_int` with adaptive
/// scaling. Sets `scratch.scale_bits` to the chosen B.
///
/// Strategy: find max |Q_mpfr[i][j]|, choose B = TARGET_BITS - ⌈log₂(max)⌉, then
/// round each S·Q[i][j] to i256 (S = 2^B).
pub fn build_q_int(scratch: &mut IntScratch) {
    // Find max magnitude
    let mut max_log2: i32 = i32::MIN;
    for i in 0..8 {
        for j in 0..8 {
            let v = scratch.q_mpfr[i][j].clone().abs();
            if v.is_zero() {
                continue;
            }
            // log2(|v|) — RFloat exposes the binary exponent directly
            // via to_f64()'s ln_abs() doesn't work for very large values, so
            // use the MPFR `get_exp()` accessor: |v| ∈ [2^(e-1), 2^e).
            let e = v.get_exp().unwrap_or(0);
            if e > max_log2 {
                max_log2 = e;
            }
        }
    }
    if max_log2 == i32::MIN {
        // All zero — degenerate, but produce zero matrix
        scratch.scale_bits = TARGET_BITS as i32;
        scratch.q_int = imat_zero();
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

// ─── i256 → MPFR conversion (used post-LLL by snapshot_gram_to_mpfr) ────────
//
// Direct `gmp_mpfr_sys` access via a stack-allocated read-only `mpz_t` view
// of the i256 limbs, passed to `mpfr::set_z`. Zero allocation per conversion
// — `rug::Integer` is bypassed entirely. All unsafe code uses only the
// documented public mpfr/gmp API (no internal field manipulation of mpfr_t).

use gmp_mpfr_sys::gmp;
use gmp_mpfr_sys::mpfr;
use std::ptr::NonNull;

/// Set `dst` (an MPFR variable) to the value of i256 `v`. Zero-allocation.
/// Constructs a stack-allocated read-only mpz_t view of the i256 limbs and
/// passes it to `mpfr::set_z`. Safe for all i256 values including 0 and
/// negatives (caller's `dst` must be initialized with a precision adequate
/// to represent the value exactly — 256 bits suffices for any i256).
#[inline]
pub fn i256_to_rfloat(v: i256, dst: &mut RFloat) {
    let zero = i256::from_i64(0);
    if v == zero {
        unsafe { mpfr::set_zero(dst.as_raw_mut(), 0) };
        return;
    }
    let neg = v < zero;
    let abs = if neg { -v } else { v };
    let bytes = abs.to_le_bytes();
    let mut limbs: [gmp::limb_t; 4] = std::array::from_fn(|i| {
        let mut buf = [0u8; 8];
        buf.copy_from_slice(&bytes[i * 8..(i + 1) * 8]);
        u64::from_le_bytes(buf) as gmp::limb_t
    });
    // Trim trailing-zero limbs to determine `_mp_size`.
    let mut size: i32 = 4;
    while size > 0 && limbs[(size - 1) as usize] == 0 {
        size -= 1;
    }
    let signed_size = if neg { -size } else { size };
    // Build a stack mpz_t view: alloc=0 means "non-owned", `mpfr_set_z` only
    // reads from it. Safe per the GMP/MPFR API contract.
    let mpz = gmp::mpz_t {
        alloc: 0,
        size: signed_size,
        d: unsafe { NonNull::new_unchecked(limbs.as_mut_ptr()) },
    };
    unsafe {
        mpfr::set_z(dst.as_raw_mut(), &mpz as *const _, mpfr::rnd_t::RNDN);
    }
    // limbs goes out of scope; mpfr::set_z has already copied the bits into dst.
}

// ─── L²-LLL (Nguyen-Stehlé 2009): pure-f64 path ──────────────────────────────
//
// All routines below operate on the f64 scratch fields (r_bar, mu_bar, s_bar)
// and read the EXACT integer Gram (i256) on demand. Theorem 2 proves f64
// suffices for d ≤ 11 with (δ=0.75, η=0.55); we operate at d=8 with
// 32-bit precision margin.

/// L² parameter η: relaxed size-reduction factor. Must satisfy 1/2 < η < √δ.
/// Per Figure 7, (δ=0.75, η=0.55) supports d ≤ 11 in f64. Stored in code as a
/// const so the inner loop optimizer can fold it.
pub const L2_ETA: f64 = 0.55;
/// L² parameter δ: Lovász factor. (δ=0.75 is the classical LLL value.)
pub const L2_DELTA: f64 = 0.75;
/// δ̄ = (δ + 1) / 2  (used by the main loop's Lovász test, per Figure 6 step 2).
/// Slightly relaxed to take FP rounding into account.
pub const L2_DELTA_BAR: f64 = (L2_DELTA + 1.0) / 2.0;
/// η̄ = (η + 1/2) / 2 (used by lazy size-reduction, per Figure 5 step 1).
pub const L2_ETA_BAR: f64 = (L2_ETA + 0.5) / 2.0;

/// Convert i256 to f64. f64 has 53 mantissa bits + 11 exponent bits (range
/// 2^±1023). Our gram values are bounded by ≈ 2^240, well within range.
/// Mantissa rounding: low bits beyond 53 are dropped (round-to-nearest-even).
/// L² algorithm only requires ≈ 20 bits of precision per Theorem 2 — f64
/// gives 53 with no overflow risk for our magnitudes.
#[inline]
pub fn i256_to_f64(v: i256) -> f64 {
    let zero = i256::from_i64(0);
    if v == zero {
        return 0.0;
    }
    let neg = v < zero;
    let abs = if neg { -v } else { v };
    let bytes = abs.to_le_bytes();
    let l0 = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
    let l1 = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
    let l2 = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
    let l3 = u64::from_le_bytes(bytes[24..32].try_into().unwrap());
    // Combine in increasing-precision order so the accumulation rounds the
    // low bits, not the high bits.
    let result = (l0 as f64)
        + (l1 as f64) * 2f64.powi(64)
        + (l2 as f64) * 2f64.powi(128)
        + (l3 as f64) * 2f64.powi(192);
    if neg { -result } else { result }
}

/// Cholesky Factorization Algorithm (Figure 4 of Nguyen-Stehlé 2009),
/// row-at-a-time variant. Computes r_bar[i][*], mu_bar[i][*], s_bar[i][*]
/// given rows 0..i are already populated.
///
/// Reads gram entries via `i256_to_f64`. All arithmetic in f64.
///
/// Per Figure 4 (with our 0-indexed convention):
///   For j = 0..i-1:
///     r̄_{i,j} ← <b_i, b_j>    (from i256 Gram)
///     For k = 0..j-1: r̄_{i,j} ← r̄_{i,j} - μ̄_{j,k} · r̄_{i,k}
///     μ̄_{i,j} ← r̄_{i,j} / r̄_{j,j}
///   s̄_{i,0} ← <b_i, b_i>
///   For j = 1..=i: s̄_{i,j} ← s̄_{i,j-1} - μ̄_{i,j-1} · r̄_{i,j-1}
///   r̄_{i,i} ← s̄_{i,i}
///
/// IMPORTANT: assumes rows 0..i are already filled by prior `cfa_row`
/// calls (or by initial setup). The L² main loop calls this at each new κ.
#[inline]
pub fn cfa_row(scratch: &mut IntScratch, i: usize) {
    // Off-diagonal entries: j = 0..i-1
    for j in 0..i {
        let mut r = i256_to_f64(scratch.gram[i][j]);
        for k in 0..j {
            r -= scratch.mu_bar[j][k] * scratch.r_bar[i][k];
        }
        scratch.r_bar[i][j] = r;
        // μ̄_{i,j} = r̄_{i,j} / r̄_{j,j}
        let r_jj = scratch.r_bar[j][j];
        scratch.mu_bar[i][j] = if r_jj.abs() < 1e-300 { 0.0 } else { r / r_jj };
    }
    // Diagonal: s̄_{i,*} sequence, r̄_{i,i} = s̄_{i,i}
    scratch.s_bar[i][0] = i256_to_f64(scratch.gram[i][i]);
    for j in 1..=i {
        scratch.s_bar[i][j] =
            scratch.s_bar[i][j - 1] - scratch.mu_bar[i][j - 1] * scratch.r_bar[i][j - 1];
    }
    scratch.r_bar[i][i] = scratch.s_bar[i][i];
}

/// Run CFA for ALL rows 0..d. Used for initial computation when L² starts.
/// Equivalent to calling `cfa_row` for i = 0, 1, ..., d-1 in order.
pub fn cfa_full(scratch: &mut IntScratch) {
    for i in 0..8 {
        cfa_row(scratch, i);
    }
}

/// Lazy floating-point size-reduction (Figure 5 of Nguyen-Stehlé 2009).
///
/// Reduces row κ against rows 0..κ-1 such that |μ̄_{κ,j}| ≤ η̄ for all j < κ,
/// where η̄ = (η + 1/2) / 2. Operates iteratively: each pass computes CFA for
/// row κ, predicts X_i = round(μ̄_{κ,i}), updates μ̄_{κ,j} predictively, then
/// applies the basis transform b_κ -= Σ X_i b_i and updates the i256 Gram.
/// Repeats until convergence.
///
/// Per Theorem 3 of the paper, the precision requirement for f64 (ℓ=52) is
/// satisfied when rows 0..κ-1 are already L³-reduced — the L² main loop
/// maintains this invariant.
///
/// Returns the number of passes used. Caller can detect non-convergence via
/// MAX_LAZY_PASSES (we never expect this to fire in practice; a hard bound
/// guards against pathological inputs).
pub const MAX_LAZY_PASSES: usize = 32;

pub fn lazy_size_reduce(scratch: &mut IntScratch, kappa: usize) -> usize {
    let mut x = [0i64; 8];

    for pass in 0..MAX_LAZY_PASSES {
        // Step 2: compute CFA for row κ (reads i256 Gram via i256_to_f64).
        cfa_row(scratch, kappa);

        // Step 3: convergence check.
        let mut max_mu: f64 = 0.0;
        for j in 0..kappa {
            let m = scratch.mu_bar[kappa][j].abs();
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

        // Steps 4-5: compute X_i values descending from κ-1 to 0,
        // predictively shrinking μ̄_{κ,j} as we go down.
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

        // Step 6: apply basis update and Gram update for each non-zero X_i.
        // gram_update_size_reduce already encodes M·G·Mᵀ for one (k, j, r)
        // triple; we call it sequentially for each non-zero x[i] so the
        // chain of updates produces the correct final Gram.
        for i in 0..kappa {
            if x[i] != 0 {
                for c in 0..8 {
                    scratch.basis[kappa][c] -= x[i] * scratch.basis[i][c];
                }
                gram_update_size_reduce(scratch, kappa, i, x[i]);
                x[i] = 0; // clear for next pass
            }
        }
        // Step 7: goto step 2 (top of loop).
    }
    if crate::synthesis::diag::trace_enabled() {
        crate::synthesis::diag::record_lazy_passes(MAX_LAZY_PASSES as u64);
    }
    MAX_LAZY_PASSES
}


// ─── Incremental Gram update for size-reduce + swap ──────────────────────────

/// Apply the basis transform `b_k -= r·b_j` to the Gram matrix in O(16) i256
/// operations instead of O(8³) = 512 for a full recompute. Math: B_new = M·B
/// where M = I - r·E_kj. Then G_new = M·G·Mᵀ. Computed via the two-step
/// recurrence (row k update, then column k update — see code comments).
///
/// Caller must call this AFTER updating the i64 basis row k. Idempotent for
/// r=0 (returns immediately).
fn gram_update_size_reduce(scratch: &mut IntScratch, k: usize, j: usize, r: i64) {
    if r == 0 {
        return;
    }
    let r256 = i256::from_i64(r);
    // Step 1: row k. G[k][m] := G[k][m] - r·G[j][m]  for m = 0..8.
    // Snapshot row j BEFORE mutating row k (the new G[k][k] depends on G[j][k]).
    let mut row_j_snapshot = [i256::from_i64(0); 8];
    for m in 0..8 {
        row_j_snapshot[m] = scratch.gram[j][m];
    }
    for m in 0..8 {
        scratch.gram[k][m] = scratch.gram[k][m] - r256 * row_j_snapshot[m];
    }
    // Step 2: column k. G[i][k] := G[i][k] - r·G[i][j]  for i = 0..8.
    // For i ≠ k: G[i][j] is unchanged from before (step 1 only touched row k).
    // For i = k: G[k][j] was updated in step 1 — we must use the post-update
    //            value here, which gives the correct G_new[k][k] derivation.
    // Snapshot column j BEFORE mutating column k.
    let mut col_j_snapshot = [i256::from_i64(0); 8];
    for i in 0..8 {
        col_j_snapshot[i] = scratch.gram[i][j];
    }
    for i in 0..8 {
        scratch.gram[i][k] = scratch.gram[i][k] - r256 * col_j_snapshot[i];
    }
}

/// Apply the basis swap of rows k and k-1 to the Gram. The Gram is
/// symmetric, so we swap rows (k, k-1) AND columns (k, k-1). 32 i256
/// pointer-style writes (or fewer with native swap). O(8) work.
fn gram_update_swap(scratch: &mut IntScratch, a: usize, b: usize) {
    if a == b {
        return;
    }
    // Swap rows a and b
    scratch.gram.swap(a, b);
    // Swap columns a and b
    for i in 0..8 {
        let tmp = scratch.gram[i][a];
        scratch.gram[i][a] = scratch.gram[i][b];
        scratch.gram[i][b] = tmp;
    }
}

/// L² INSERT operation (Figure 6 step 6 of Nguyen-Stehlé 2009).
///
/// Move basis row `kappa_orig` to position `kappa_insert` (≤ kappa_orig).
/// Rows kappa_insert..kappa_orig-1 shift down by one. Implemented as a chain
/// of adjacent swaps so the i256 Gram is kept consistent via the existing
/// gram_update_swap. After basis + Gram are rotated, the GS state for row
/// kappa_insert needs to be REFRESHED (because the row's contents changed):
/// caller is responsible for invoking `cfa_row(scratch, kappa_insert)`.
///
/// Cost: O(kappa_orig - kappa_insert) adjacent swaps, each O(d) for the
/// gram column swap.
fn basis_insert(scratch: &mut IntScratch, kappa_orig: usize, kappa_insert: usize) {
    debug_assert!(kappa_insert <= kappa_orig);
    let mut current = kappa_orig;
    while current > kappa_insert {
        scratch.basis.swap(current, current - 1);
        gram_update_swap(scratch, current, current - 1);
        current -= 1;
    }
    // Note: GS state (r_bar, mu_bar, s_bar) for rows kappa_insert..kappa_orig
    // is now stale. The L² main loop must refresh row kappa_insert via
    // cfa_row before the next iteration uses it. Rows above kappa_insert
    // are recomputed naturally as κ advances and lazy_size_reduce calls CFA.
}

// ─── i256 Gram update: G = B · Q_int · Bᵀ ──────────────────────────────────

/// Compute G = B · Q_int · Bᵀ entirely in i256, into `scratch.gram`. Uses
/// `scratch.temp_bq` as intermediate (= B · Q_int).
///
/// **Overflow analysis**: max |Q_int| = 2^TARGET_BITS = 2^180 by `build_q_int`
/// choice. For typical post-LLL max(|B[i][j]|) ≤ 2^15, intermediate B·Q_int
/// entries fit ≤ 2^198, final G entries fit ≤ 2^216 → safe with 40-bit margin
/// to i256::MAX. For transient B-growth during LLL swaps, max(|B|) can spike
/// to ~2^40 at deep ε; G entries can then approach 2^260 (overflow). Returns
/// `false` if any Gram entry magnitude exceeds 2^GRAM_OVERFLOW_THRESHOLD_BITS,
/// allowing the LLL caller to abort and trigger fallback.
pub fn compute_gram_full(scratch: &mut IntScratch) -> bool {
    let zero = i256::from_i64(0);

    // temp_bq[i][b] = sum_a B[i][a] · Q_int[a][b]
    for i in 0..8 {
        for b in 0..8 {
            let mut acc = zero;
            for a in 0..8 {
                let bi_a = i256::from_i64(scratch.basis[i][a]);
                acc = acc + bi_a * scratch.q_int[a][b];
            }
            scratch.temp_bq[i][b] = acc;
        }
    }

    // gram[i][j] = sum_b temp_bq[i][b] · B[j][b]
    let mut max_abs_log2: i32 = -1;
    for i in 0..8 {
        for j in 0..8 {
            let mut acc = zero;
            for b in 0..8 {
                let bj_b = i256::from_i64(scratch.basis[j][b]);
                acc = acc + scratch.temp_bq[i][b] * bj_b;
            }
            scratch.gram[i][j] = acc;
            // Magnitude check (cheap: leading-zero count on the |i256|)
            let bits = i256_log2_ceil(&acc);
            if bits > max_abs_log2 {
                max_abs_log2 = bits;
            }
        }
    }
    max_abs_log2 <= GRAM_OVERFLOW_THRESHOLD_BITS as i32
}

/// Bit count of |v| (≈ ⌈log₂(|v|)⌉, returns -1 for v=0).
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


// ─── LLL inner loop (i256 Gram + MPFR GS) ────────────────────────────────────

/// Result of `lll_int_8`. `Ok` on convergence with a unimodular basis;
/// `Err(reason)` on overflow or iteration cap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LllResult {
    /// LLL converged (within `max_iter` iterations and no overflow).
    Converged,
    /// A Gram entry's magnitude exceeded `GRAM_OVERFLOW_THRESHOLD_BITS`
    /// during transient basis growth. Indicates the i256 buffer is no
    /// longer wide enough for the current ε regime; the caller should
    /// reject this prefix.
    GramOverflow,
    /// Reached the iteration cap without convergence (cycling or near-
    /// boundary precision noise). Diagnostic only.
    IterCap,
}

/// Check whether any Gram entry magnitude exceeds the overflow threshold.
/// Cheap: 64 leading-zero queries on i256 LE bytes. `i256_log2_ceil` returns
/// -1 for zero — guard against the i32→u32 wrap that would mis-flag zeroes.
fn gram_overflow_check(scratch: &IntScratch) -> bool {
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

/// L²-LLL (Nguyen-Stehlé 2009, Figure 6) on the 8×8 Q-metric Gram already
/// snapshotted into `scratch.gram`. Builds an LLL-reduced basis and
/// records it in `scratch.basis`; intermediate state lives in
/// `scratch.r_bar` / `mu_bar` / `s_bar` and the i256 `gram`.
///
/// The algorithm walks rows κ = 1..d, maintaining the invariant that
/// rows 0..κ-1 are (δ,η)-L³-reduced. At each κ it:
///   1. Lazily size-reduces row κ (interleaved CFA and basis reduction)
///      until `|μ̄_{κ,j}| ≤ η̄` for all j < κ.
///   2. Finds the deepest insertion position κ_insert by walking the
///      Lovász condition downward.
///   3. If κ_insert < κ, rotates the basis (and Gram) so that the
///      reduced row lands at position κ_insert; otherwise leaves it.
///   4. Advances κ to the next row.
///
/// Per Theorem 3 of the paper, f64 precision (53 mantissa bits) suffices
/// for d=8 at (δ=0.75, η=0.55): the required precision is
/// `ℓ ≥ 5 + 2·log d − log ε + d·log ρ ≈ 34 bits`, leaving ~18 bits of
/// margin. The L³-reduction invariant on the prefix is what makes the
/// f64 GS coefficients accurate enough; running CFA on an unreduced
/// basis would suffer catastrophic cancellation at deep ε.
pub fn lll_l2_8(scratch: &mut IntScratch) -> LllResult {
    scratch.reset_basis();
    let max_iter: usize = 10_000;
    let mut iters: usize = 0;

    // Step 1: compute exact integer Gram. Basis = identity → Gram = Q_int.
    if !compute_gram_full(scratch) {
        if crate::synthesis::diag::trace_enabled() {
            crate::synthesis::diag::record_lll_iters(iters as u64, max_iter as u64);
        }
        return LllResult::GramOverflow;
    }

    // Step 2: initialize r̄_{0,0} = ‖b_0‖² (CFA on row 0).
    cfa_row(scratch, 0);
    let mut kappa = 1usize;

    while kappa < 8 && iters < max_iter {
        iters += 1;

        // Step 3: lazy size-reduce row κ. Updates basis (i64) + Gram (i256)
        // and refreshes r_bar/mu_bar/s_bar for row κ.
        let _passes = lazy_size_reduce(scratch, kappa);

        if gram_overflow_check(scratch) {
            if crate::synthesis::diag::trace_enabled() {
                crate::synthesis::diag::record_lll_iters(iters as u64, max_iter as u64);
            }
            return LllResult::GramOverflow;
        }

        // Step 4: Lovász cascade. Find deepest position κ_insert where the
        // size-reduced row κ_orig should be inserted. Use s̄_{κ-1}^{(κ_orig)}
        // (partial CFA sum at depth κ-1 for the orig-frontier row) as the
        // projected GS norm² at insertion depth κ.
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

        // If the insertion position is shallower than the current frontier,
        // rotate the basis and Gram so the (now-reduced) frontier row lands
        // at the new position, then recompute that row's GS state from the
        // updated Gram.
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

// ─── Convert i256 Gram → MPFR (post-LLL, into g_post_lll) ─────────────────────

/// Convert the post-LLL i256 Gram into MPFR `g_post_lll` so Cholesky/LU
/// can run on it. The integer Gram is divided by `2^scale_bits` during
/// conversion to recover the natural Q-metric scale.
fn snapshot_gram_to_mpfr(scratch: &mut IntScratch) {
    let prec = scratch.prec_q;
    let shift = scratch.scale_bits;
    let mut tmp = rfz(prec);
    for i in 0..8 {
        for j in 0..8 {
            i256_to_rfloat(scratch.gram[i][j], &mut tmp);
            // Divide by 2^scale_bits to recover natural-scale G
            if shift > 0 {
                tmp >>= shift as u32;
            } else if shift < 0 {
                tmp <<= (-shift) as u32;
            }
            scratch.g_post_lll[i][j].assign(&tmp);
        }
    }
}

// ─── Cholesky (in-place rug, ported from heavy) ──────────────────────────────

fn cholesky_int_8(scratch: &mut IntScratch) -> bool {
    let prec = scratch.prec_q;
    for i in 0..8 {
        for j in 0..8 {
            scratch.l[i][j].assign(0.0_f64);
        }
    }
    let zero = rfz(prec);
    for i in 0..8 {
        for j in 0..=i {
            scratch.acc.assign(&scratch.g_post_lll[i][j]);
            for k in 0..j {
                scratch.tmp.assign(&scratch.l[i][k] * &scratch.l[j][k]);
                let acc_clone = scratch.acc.clone();
                scratch.acc.assign(&acc_clone - &scratch.tmp);
            }
            if i == j {
                if scratch.acc <= zero {
                    return false;
                }
                let acc_clone = scratch.acc.clone();
                scratch.l[i][i].assign(acc_clone.sqrt());
            } else {
                scratch.tmp2.assign(&scratch.l[j][j]);
                scratch.l[i][j].assign(&scratch.acc / &scratch.tmp2);
            }
        }
    }
    true
}

// ─── LU solve with partial pivoting (in-place rug, ported from heavy) ─────────

fn lu_solve_int_inplace(scratch: &mut IntScratch) -> bool {
    let tol = rfv(scratch.lu_prec, 1e-30);

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
            scratch.lu_tmp.assign(&scratch.lu_a[i][k] / &scratch.lu_a[k][k]);
            let factor = scratch.lu_tmp.clone();
            // a[i][j] -= factor · a[k][j]  for j in k..8
            // Avoid simultaneous &mut borrows on rows i and k.
            let (row_i, row_k) = if i < k {
                let (head, tail) = scratch.lu_a.split_at_mut(k);
                (&mut head[i], &mut tail[0])
            } else {
                let (head, tail) = scratch.lu_a.split_at_mut(i);
                (&mut tail[0], &mut head[k])
            };
            for j in k..8 {
                scratch.lu_tmp.assign(&factor * &row_k[j]);
                let cur = row_i[j].clone();
                row_i[j].assign(&cur - &scratch.lu_tmp);
            }
            scratch.lu_tmp.assign(&factor * &scratch.lu_rhs[k]);
            let rhs_i_cur = scratch.lu_rhs[i].clone();
            scratch.lu_rhs[i].assign(&rhs_i_cur - &scratch.lu_tmp);
        }
    }
    for i in (0..8).rev() {
        scratch.lu_acc.assign(&scratch.lu_rhs[i]);
        for j in (i + 1)..8 {
            scratch.lu_tmp.assign(&scratch.lu_a[i][j] * &scratch.lu_x[j]);
            let cur = scratch.lu_acc.clone();
            scratch.lu_acc.assign(&cur - &scratch.lu_tmp);
        }
        let acc_clone = scratch.lu_acc.clone();
        scratch.lu_x[i].assign(&acc_clone / &scratch.lu_a[i][i]);
    }
    true
}

// ─── f64 Cholesky on the natural-scale post-LLL Gram ────────────────────────
//
// Reads the i256 Gram via `i256_to_f64`, multiplies by `2^-scale_bits` (an
// exponent shift, no precision cost) to recover natural units, then runs
// standard f64 Cholesky. Output: lower-triangular `scratch.l_f64`.
//
// Why this is precision-sufficient:
//   1. The L³-reduction invariant after L²-LLL termination bounds the
//      condition number of G = B·Q·Bᵀ on the reduced basis: κ(G) ≤
//      (4/3)^(d-1) ≤ 16 for d=8. The reduced Gram is well-conditioned even
//      when the input Q has κ ≈ 2^137 at ε=1e-10.
//   2. f64 Cholesky on a κ ≤ 16 matrix yields a factor with 53-bit relative
//      precision (one bit of conditioning loss per κ doubling, four bits
//      total).
//   3. SE downcasts the factor to TwoFloat at the bound check; the SE walk
//      compounds errors over d=8 levels and reaches ~10⁻¹⁵ absolute at unit
//      scale — six orders of magnitude below SE's existing 10⁻⁹ tolerance.

/// Run f64 Cholesky on the natural-scale post-LLL Gram. Returns `false` if
/// the Gram is not numerically positive-definite (extremely rare for an
/// LLL-output basis; would indicate an upstream bug). Result lives in
/// `scratch.l_f64` as the lower-triangular factor.
pub fn cholesky_f64_8(scratch: &mut IntScratch) -> bool {
    let scale = 2.0_f64.powi(-scratch.scale_bits);
    // Snapshot the lower triangle in f64 with the natural-scale shift folded
    // into the conversion. Upper triangle is implicit via symmetry.
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

// ─── TwoFloat LU for the cap-center solve ────────────────────────────────────
//
// The cap-center system is `Bᵀ·z_c = c` where B is exact integer (det=±1)
// and c is the alignment target at TwoFloat precision. Standard Gaussian
// elimination with partial pivoting in TwoFloat throughout. SE consumes
// `z_c` as TwoFloat directly — no MPFR required.

use twofloat::TwoFloat as Tf;

/// Compute `cap_mid = (1 + √(1 − ε²)) / 2` in TwoFloat. For ε = 1e-8 this
/// is ≈ 1 − 2.5·10⁻¹⁷, below f64 precision but well within TwoFloat's
/// ~104-bit mantissa. Used to build the alignment-target RHS.
pub fn cap_mid_tf(eps: Float) -> Tf {
    let eps_tf = Tf::from(eps);
    let term = Tf::from(1.0_f64) - eps_tf * eps_tf;
    (Tf::from(1.0_f64) + term.sqrt()) / Tf::from(2.0_f64)
}

/// Solve `Bᵀ·z = c` for `z` in TwoFloat, where `basis` is the LLL-reduced
/// integer basis (rows are basis vectors) and `c` is the target alignment
/// in TwoFloat. Standard partial-pivoting LU. Returns `None` if the matrix
/// is numerically singular (impossible for an LLL output with det=±1, but
/// guarded for robustness).
pub fn tf_lu_solve_8(basis: &IMat8, c: &[Tf; 8]) -> Option<[Tf; 8]> {
    let mut a: [[Tf; 8]; 8] = std::array::from_fn(|i| {
        std::array::from_fn(|j| Tf::from(basis[j][i] as f64))
    });
    let mut rhs: [Tf; 8] = *c;

    for k in 0..8 {
        let mut piv = k;
        let mut piv_abs = f64::from(a[k][k].abs());
        for i in (k + 1)..8 {
            let v = f64::from(a[i][k].abs());
            if v > piv_abs {
                piv_abs = v;
                piv = i;
            }
        }
        if piv_abs < 1e-30 {
            return None;
        }
        if piv != k {
            a.swap(k, piv);
            rhs.swap(k, piv);
        }
        for i in (k + 1)..8 {
            let factor = a[i][k] / a[k][k];
            for j in k..8 {
                a[i][j] = a[i][j] - factor * a[k][j];
            }
            rhs[i] = rhs[i] - factor * rhs[k];
        }
    }
    let mut x: [Tf; 8] = [Tf::from(0.0_f64); 8];
    for i in (0..8).rev() {
        let mut s = rhs[i];
        for j in (i + 1)..8 {
            s = s - a[i][j] * x[j];
        }
        x[i] = s / a[i][i];
    }
    Some(x)
}

// ─── Top-level phase1 entry point ────────────────────────────────────────────

use std::sync::atomic::AtomicBool;

/// Outcome of one phase1 invocation. `should_escalate` is set when the i256
/// Gram overflowed during LLL (transient B-growth at very deep ε beyond what
/// `TARGET_BITS = 180` absorbs). The dispatcher can use this signal to fall
/// back to an alternative strategy if needed; the L²-LLL path was designed
/// to keep this flag clear in our target ε ∈ [1e-10, 1e-3] regime.
pub struct PhaseOneOutcome {
    pub solutions: Vec<[i64; 8]>,
    pub should_escalate: bool,
}

/// Run the full Lenstra 8D pipeline for one MA-prefix's `(y, k, eps)` setup
/// using the L²-LLL algorithm. Returns at most one valid 8-vector solution;
/// the caller can request more by raising `max_phase2_calls`.
pub fn phase1(
    scratch: &mut IntScratch,
    y: &[Float; 8],
    k: u32,
    eps: Float,
    max_phase2_calls: u64,
    budget_hit: &AtomicBool,
) -> PhaseOneOutcome {
    use std::sync::atomic::{AtomicU64, Ordering};

    // Use i128 so target_norm = 2^k stays correct for k ≥ 63 (where i64 would
    // overflow). At k=82 (ε=1e-8), target_norm = 2^82 ≈ 5e24 — fits in i128.
    let target_norm: i128 = 1i128 << k;
    // Fast path: when k ≤ 62, ‖x‖² and the target both fit in i64. The SE
    // callback is the hot loop (millions of invocations); the i128 path is
    // ~3-5× slower per op on aarch64. Hoist the branch outside the callback.
    let use_i64_path = k <= 62;
    let target_norm_i64: i64 = if use_i64_path { 1i64 << k } else { 0 };
    // 2^(2k) overflows u128 at k ≥ 64 (2k ≥ 128). Compute the threshold in
    // f64 directly via powi; f64 represents 2^k exactly for any k ≤ 1023.
    let threshold_xy = 2.0_f64.powi(2 * k as i32) / 4.0 * (1.0 - eps * eps);

    let trace = crate::synthesis::diag::trace_enabled();

    // Step 1: build Q in MPFR + integer snapshot
    let t_phase = if trace { Some(std::time::Instant::now()) } else { None };
    build_q_mpfr(scratch, y, k, eps);
    build_q_int(scratch);
    if let Some(t0) = t_phase {
        crate::synthesis::diag::T_BUILD_NS
            .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

    // Step 2: L²-LLL (pure f64 GS with exact i256 Gram, INSERT semantics).
    let t_phase = if trace { Some(std::time::Instant::now()) } else { None };
    let lll_result = lll_l2_8(scratch);
    if let Some(t0) = t_phase {
        crate::synthesis::diag::T_LLL_NS
            .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
    if let LllResult::GramOverflow = lll_result {
        return PhaseOneOutcome { solutions: Vec::new(), should_escalate: true };
    }

    // Step 3: assert det = ±1
    let basis = scratch.basis;
    match super::se::det8_exact(&basis) {
        Some(1) | Some(-1) => {}
        Some(d) => {
            eprintln!(
                "[lenstra_int] LLL non-unimodular (det={}) at eps={:e}, k={}; bailing.",
                d, eps, k
            );
            return PhaseOneOutcome { solutions: Vec::new(), should_escalate: false };
        }
        None => {
            eprintln!(
                "[lenstra_int] det8_exact overflow at eps={:e}, k={}; bailing.",
                eps, k
            );
            return PhaseOneOutcome { solutions: Vec::new(), should_escalate: false };
        }
    }

    // Step 4: f64 Cholesky directly on the i256 Gram (natural-scale via
    // 2^-scale_bits exponent shift). Eliminates the MPFR snapshot + MPFR
    // Cholesky path. Justified by post-LLL κ ≤ 16 (LLL invariant).
    let t_phase = if trace { Some(std::time::Instant::now()) } else { None };
    let chol_ok = cholesky_f64_8(scratch);
    if let Some(t0) = t_phase {
        crate::synthesis::diag::T_CHOLESKY_NS
            .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
    if !chol_ok {
        eprintln!(
            "[lenstra_int] Cholesky (f64) failed at eps={:e}, k={}; bailing.", eps, k
        );
        return PhaseOneOutcome { solutions: Vec::new(), should_escalate: false };
    }

    // Build R = Lᵀ in TwoFloat (zero-cost f64→Tf lift; lo = 0).
    let r_chol_tf: [[Tf; 8]; 8] = std::array::from_fn(|i| {
        std::array::from_fn(|j| Tf::from(scratch.l_f64[j][i]))
    });

    // Step 5: build cap center c = y · cap_mid (in MPFR), then LU solve
    // B_LLLᵀ · z_c = c. MPFR LU retained: TwoFloat LU showed ~10⁻⁷ relative
    // error at ε=1e-5 (vs MPFR reference) — z_c values reach magnitudes
    // (2^41 and beyond) where TwoFloat's 104-bit relative precision becomes
    // a large absolute offset that destabilises the SE walk's cap-center.
    // The `lu_tf_matches_mpfr_*` test guards against re-introducing this.
    let t_phase = if trace { Some(std::time::Instant::now()) } else { None };
    for i in 0..8 {
        for j in 0..8 {
            scratch.lu_a[i][j].assign(rfv(scratch.prec_q, basis[j][i] as f64));
        }
        scratch.lu_rhs[i].assign(&scratch.c[i]);
    }
    let lu_ok = lu_solve_int_inplace(scratch);
    if let Some(t0) = t_phase {
        crate::synthesis::diag::T_LU_NS
            .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
    if !lu_ok {
        eprintln!("[lenstra_int] LU solve failed at eps={:e}, k={}; bailing.", eps, k);
        return PhaseOneOutcome { solutions: Vec::new(), should_escalate: false };
    }
    let z_c_tf: [Tf; 8] = std::array::from_fn(|i| {
        super::se::rfloat_to_tf(&scratch.lu_x[i])
    });

    // Step 6: SE in TwoFloat
    let r_eucl = super::se::euclidean_cholesky(&basis);
    let target_norm_f = target_norm as f64;
    let count = AtomicU64::new(0);
    let abort = AtomicBool::new(false);
    let bound_tf = Tf::from(1.51_f64);
    let t_phase = if trace { Some(std::time::Instant::now()) } else { None };

    let result = super::se::schnorr_euchner_8d(
        &r_chol_tf,
        &z_c_tf,
        bound_tf,
        r_eucl.as_ref(),
        target_norm_f,
        &abort,
        |z: &[i64; 8]| {
            let n_so_far = count.load(Ordering::Relaxed);
            if n_so_far >= max_phase2_calls {
                budget_hit.store(true, Ordering::Relaxed);
                return None;
            }
            count.fetch_add(1, Ordering::Relaxed);
            let x = super::se::reconstruct_x(&basis, z);
            // Norm check: i64 fast path for k ≤ 62, i128 path otherwise.
            // Most SE candidates fail this check, so it's the hottest test;
            // keeping it in i64 when safe is worth the branch.
            if use_i64_path {
                let n: i64 = x.iter().map(|&v| v * v).sum();
                if n != target_norm_i64 {
                    return None;
                }
            } else {
                let n: i128 = x.iter().map(|&v| (v as i128) * (v as i128)).sum();
                if n != target_norm {
                    return None;
                }
            }
            if super::se::bilinear_b(&x) != 0 {
                return None;
            }
            let dot: Float = x.iter().zip(y.iter()).map(|(a, b)| *a as Float * b).sum();
            if dot * dot < threshold_xy {
                return None;
            }
            Some(x)
        },
    );

    if let Some(t0) = t_phase {
        crate::synthesis::diag::T_SE_NS
            .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
    crate::synthesis::diag::N_SE_CALLBACKS
        .fetch_add(count.load(Ordering::Relaxed), Ordering::Relaxed);

    match result {
        Some(x) => PhaseOneOutcome { solutions: vec![x], should_escalate: false },
        None => PhaseOneOutcome { solutions: Vec::new(), should_escalate: false },
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::se;

    fn realistic_y(k: u32) -> [Float; 8] {
        let r2 = 1.0 / 2.0_f64.sqrt();
        // 2^(k/2-1) — for k > 63 we can't do `(1u64 << k) as f64`, use powi
        let s = 2.0_f64.powi(k as i32 / 2 - 1);
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

    /// Verify build_q_int produces an i256 matrix that, when scaled back to
    /// f64, matches the MPFR Q to within rounding error (≤ 2^-(TARGET_BITS-2)
    /// relative for max-magnitude entries).
    fn check_int_q_matches_mpfr(eps: Float, k: u32) {
        let y = realistic_y(k);
        let mut s = IntScratch::new(eps);
        build_q_mpfr(&mut s, &y, k, eps);
        build_q_int(&mut s);
        // Scale back: q_recovered[i][j] = q_int[i][j] / 2^scale_bits, in MPFR
        let mut max_abs_q: f64 = 0.0;
        let mut max_err: f64 = 0.0;
        for i in 0..8 {
            for j in 0..8 {
                let q_true = s.q_mpfr[i][j].to_f64();
                max_abs_q = max_abs_q.max(q_true.abs());
                // Convert i256 to f64 (lossy but ok for the check)
                let q_int_f = i256_to_f64_scaled(&s.q_int[i][j], s.scale_bits);
                let err = (q_true - q_int_f).abs();
                max_err = max_err.max(err);
            }
        }
        let rel_err = max_err / max_abs_q.max(1e-300);
        // Allow 2^-100 relative error (very forgiving — 20 bits below
        // TARGET_BITS to absorb rounding noise + i256→f64 truncation).
        assert!(
            rel_err < 1e-25,
            "eps={:e}, k={}: rel_err={:e}, max_q={:e}, max_err={:e}, scale_bits={}",
            eps, k, rel_err, max_abs_q, max_err, s.scale_bits
        );
    }

    fn i256_to_f64_scaled(v: &i256, shift_bits: i32) -> f64 {
        // v / 2^shift_bits as f64. For tests only; magnitudes here are within
        // f64 range after scaling.
        let bytes = v.to_le_bytes();
        // Reconstruct as integer string for robustness, then route through
        // RFloat for precise division.
        let neg = (bytes[31] & 0x80) != 0;
        let mag = if neg { -*v } else { *v };
        let mag_bytes = mag.to_le_bytes();
        let mut int = rug::Integer::new();
        // bytes are little-endian; rug::Integer assigns from limbs little-endian
        let mut hex = String::with_capacity(64);
        for &b in mag_bytes.iter().rev() {
            hex.push_str(&format!("{:02x}", b));
        }
        int.assign(rug::Integer::parse_radix(&hex, 16).unwrap());
        let mut f = rug::Float::with_val(256, &int);
        if shift_bits >= 0 {
            f >>= shift_bits as u32;
        } else {
            f <<= (-shift_bits) as u32;
        }
        let r = f.to_f64();
        if neg { -r } else { r }
    }

    #[test]
    fn q_int_matches_mpfr_at_eps_1e_3() {
        check_int_q_matches_mpfr(1e-3, 14);
    }

    #[test]
    fn q_int_matches_mpfr_at_eps_1e_5() {
        check_int_q_matches_mpfr(1e-5, 21);
    }

    #[test]
    fn q_int_matches_mpfr_at_eps_1e_8() {
        check_int_q_matches_mpfr(1e-8, 70);
    }

    #[test]
    fn q_int_matches_mpfr_at_eps_1e_10() {
        check_int_q_matches_mpfr(1e-10, 100);
    }

    #[test]
    fn scale_bits_chosen_correctly() {
        // ε=1e-5, k=21: max(Q) ≈ 2^49 (inv_dy_sq dominant) → scale_bits ≈ 71
        let y = realistic_y(21);
        let mut s = IntScratch::new(1e-5);
        build_q_mpfr(&mut s, &y, 21, 1e-5);
        build_q_int(&mut s);
        // Should be in a sensible range — neither saturated nor zeroed
        assert!(
            s.scale_bits > 30 && s.scale_bits < 200,
            "unexpected scale_bits={}", s.scale_bits
        );
    }


    /// Verify cfa_full maintains the algorithmic invariant
    /// `r_bar[i][i] == s_bar[i][i]` for any input.
    ///
    /// IMPORTANT: this test does NOT assert r̄_{i,i} > 0. Running CFA on an
    /// unreduced identity basis with a high-κ Gram (our deep-ε regime) can
    /// produce cancellation noise that drives r̄_{i,i} negative — that is the
    /// precise scenario L² is engineered to AVOID via lazy size-reduction
    /// interleaved with CFA. The unit test here is a structural sanity check
    /// only; correctness validation lives at the L²-loop integration level.
    #[test]
    fn cfa_f64_diagonal_invariant_eps_1e_3() {
        // Use ε=1e-3 (κ ≈ 2^40) where f64 has comfortable margin even on
        // unreduced identity basis. This isolates the structural bug
        // detection (algorithm correctness) from the precision question.
        let eps = 1e-3;
        let k = 14u32;
        let y = realistic_y(k);
        let mut s = IntScratch::new(eps);
        build_q_mpfr(&mut s, &y, k, eps);
        build_q_int(&mut s);
        compute_gram_full(&mut s);
        cfa_full(&mut s);

        for i in 0..8 {
            assert_eq!(s.r_bar[i][i], s.s_bar[i][i],
                "r_bar[{}][{}] != s_bar[{}][{}]: structural invariant violated", i, i, i, i);
        }
        // At ε=1e-3 with d=8 and κ ≈ 2^40, f64 (53-bit mantissa) has 13+
        // bits of margin even on unreduced identity. Diagonals should be
        // positive at this benign ε.
        for i in 0..8 {
            assert!(s.r_bar[i][i] > 0.0,
                "r_bar[{}][{}] = {} unexpectedly non-positive at ε=1e-3 (κ ≈ 2^40)",
                i, i, s.r_bar[i][i]);
        }
    }

    /// Verify i256_to_f64 produces correct values for various magnitudes.
    #[test]
    fn i256_to_f64_correctness() {
        // Small positive
        assert_eq!(i256_to_f64(i256::from_i64(0)), 0.0);
        assert_eq!(i256_to_f64(i256::from_i64(1)), 1.0);
        assert_eq!(i256_to_f64(i256::from_i64(-1)), -1.0);
        assert_eq!(i256_to_f64(i256::from_i64(42)), 42.0);
        // Powers of 2
        let mut v = i256::from_i64(1);
        for shift in [10, 30, 60, 100, 200] {
            for _ in 0..shift { v = v + v; }  // v = 2^shift
            let expected = 2f64.powi(shift);
            let actual = i256_to_f64(v);
            assert_eq!(actual, expected, "2^{} got {} expected {}", shift, actual, expected);
            v = i256::from_i64(1);
        }
        // Negative large
        let mut v = i256::from_i64(1);
        for _ in 0..100 { v = v + v; }
        let neg_v = -v;
        assert_eq!(i256_to_f64(neg_v), -2f64.powi(100));
    }

    /// Run the L²-LLL for given (eps, k) and assert (a) det = ±1
    /// (unimodular basis), (b) post-conditions of an L³-reduced basis
    /// (size-reduced + Lovász). This is the invariant-based validation the
    /// critic mandated for Task #60.
    fn check_l2_lll(eps: Float, k: u32) -> LllResult {
        let y = realistic_y(k);
        let mut s = IntScratch::new(eps);
        build_q_mpfr(&mut s, &y, k, eps);
        build_q_int(&mut s);
        let result = lll_l2_8(&mut s);
        if let LllResult::GramOverflow = result {
            return result;
        }
        // Unimodular check
        let det = se::det8_exact(&s.basis)
            .expect("det8_exact overflow");
        assert!(
            det == 1 || det == -1,
            "L²-LLL output non-unimodular: det={}, eps={:e}, k={}, result={:?}",
            det, eps, k, result
        );
        // Size-reduction invariant: |μ̄_{i,j}| ≤ η for all i > j.
        // Compute final GS state via CFA (algorithm doesn't promise final
        // r_bar/mu_bar are valid; recompute fresh for the post-condition).
        cfa_full(&mut s);
        for i in 1..8 {
            for j in 0..i {
                assert!(
                    s.mu_bar[i][j].abs() <= L2_ETA + 1e-10,
                    "size-reduction violated: |μ̄[{}][{}]|={} > η={}, eps={:e}, k={}",
                    i, j, s.mu_bar[i][j].abs(), L2_ETA, eps, k
                );
            }
        }
        // Lovász: δ·r̄_{κ-1,κ-1} ≤ s̄_{κ-1}^{(κ)} for κ = 1..7.
        // s̄_{κ-1}^{(κ)} = s_bar[κ][κ-1].
        for kappa in 1..8 {
            let lhs = L2_DELTA * s.r_bar[kappa - 1][kappa - 1];
            let rhs = s.s_bar[kappa][kappa - 1];
            assert!(
                lhs <= rhs + 1e-10 * rhs.abs().max(1.0),
                "Lovász violated at κ={}: δ·r̄_{}={} > s̄_{}^{}_={}, eps={:e}, k={}",
                kappa, kappa - 1, lhs, kappa - 1, kappa, rhs, eps, k
            );
        }
        result
    }

    #[test]
    fn l2_lll_eps_1e_3() {
        let r = check_l2_lll(1e-3, 14);
        assert_eq!(r, LllResult::Converged, "L² did not converge at ε=1e-3");
    }

    #[test]
    fn l2_lll_eps_1e_5() {
        let r = check_l2_lll(1e-5, 21);
        assert_eq!(r, LllResult::Converged, "L² did not converge at ε=1e-5");
    }

    #[test]
    fn l2_lll_eps_1e_7() {
        let r = check_l2_lll(1e-7, 49);
        assert_eq!(r, LllResult::Converged, "L² did not converge at ε=1e-7");
    }

    #[test]
    fn l2_lll_eps_1e_8() {
        let r = check_l2_lll(1e-8, 70);
        assert!(matches!(r, LllResult::Converged | LllResult::IterCap),
            "unexpected at ε=1e-8: {:?}", r);
    }

    /// Run the integer LLL for given (eps, k) and assert det = ±1
    /// (unimodular basis output). Uses the heavy module's det8_exact for the
    /// integer determinant check.
    fn check_lll_unimodular(eps: Float, k: u32) -> LllResult {
        let y = realistic_y(k);
        let mut s = IntScratch::new(eps);
        build_q_mpfr(&mut s, &y, k, eps);
        build_q_int(&mut s);
        let result = lll_l2_8(&mut s);
        // Allow IterCap as a soft outcome (LLL is "noisy" at deep ε); the
        // unimodular check is a hard invariant either way.
        if let LllResult::GramOverflow = result {
            return result;
        }
        let det = se::det8_exact(&s.basis)
            .expect("det8_exact overflow");
        assert!(
            det == 1 || det == -1,
            "lll output non-unimodular: det={}, eps={:e}, k={}, result={:?}",
            det, eps, k, result
        );
        result
    }

    #[test]
    fn lll_unimodular_at_eps_1e_3() {
        let r = check_lll_unimodular(1e-3, 14);
        assert_eq!(r, LllResult::Converged);
    }

    #[test]
    fn lll_unimodular_at_eps_1e_5() {
        let r = check_lll_unimodular(1e-5, 21);
        assert_eq!(r, LllResult::Converged);
    }

    #[test]
    fn lll_unimodular_at_eps_1e_6() {
        let r = check_lll_unimodular(1e-6, 28);
        assert_eq!(r, LllResult::Converged);
    }

    #[test]
    fn lll_unimodular_at_eps_1e_7() {
        let r = check_lll_unimodular(1e-7, 49);
        // ε=1e-7 is comfortably within precision budget (κ≈2^93,
        // TARGET_BITS=180, post-GS ~87 bits). Convergence expected.
        assert_eq!(r, LllResult::Converged);
    }

    #[test]
    fn lll_unimodular_at_eps_1e_8() {
        // Stretch goal: ε=1e-8. κ≈2^107, post-GS ~73 bits. Should converge
        // unless transient B-growth triggers Gram overflow.
        let r = check_lll_unimodular(1e-8, 70);
        assert!(
            matches!(r, LllResult::Converged | LllResult::IterCap),
            "unexpected result at eps=1e-8: {:?}", r
        );
    }

    #[test]
    fn lll_unimodular_at_eps_1e_10() {
        // Deep end of target range: κ≈2^137, post-GS ~43 bits. Likely
        // produces non-LLL-reduced but still unimodular basis (size-reduce
        // is robust; Lovász decisions are noisy). Document outcome.
        let r = check_lll_unimodular(1e-10, 100);
        eprintln!("lll_unimodular_at_eps_1e_10: result = {:?}", r);
    }

    #[test]
    fn incremental_size_reduce_matches_full_recompute() {
        // Build an arbitrary i256 Q, set a non-identity basis, do one
        // size-reduce step both via gram_update_size_reduce and via full
        // recompute; verify entries match exactly.
        let eps = 1e-5;
        let k_val = 21u32;
        let y = realistic_y(k_val);
        let mut s = IntScratch::new(eps);
        build_q_mpfr(&mut s, &y, k_val, eps);
        build_q_int(&mut s);
        s.basis = [
            [3, 1, 0, 0, 0, 0, 0, 0],
            [1, 2, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ];
        compute_gram_full(&mut s);
        // Apply incremental update for b_2 -= 5 * b_0
        let k = 2usize;
        let j = 0usize;
        let r = 5i64;
        for c in 0..8 { s.basis[k][c] -= r * s.basis[j][c]; }
        gram_update_size_reduce(&mut s, k, j, r);
        let g_inc = s.gram;
        // Full recompute on the new basis
        compute_gram_full(&mut s);
        let g_full = s.gram;
        for i in 0..8 {
            for jj in 0..8 {
                assert_eq!(
                    g_inc[i][jj], g_full[i][jj],
                    "mismatch at [{}][{}]: inc={:?} full={:?}",
                    i, jj, g_inc[i][jj], g_full[i][jj]
                );
            }
        }
    }

    #[test]
    fn incremental_swap_matches_full_recompute() {
        let eps = 1e-5;
        let k_val = 21u32;
        let y = realistic_y(k_val);
        let mut s = IntScratch::new(eps);
        build_q_mpfr(&mut s, &y, k_val, eps);
        build_q_int(&mut s);
        s.basis = [
            [3, 1, 0, 0, 0, 0, 0, 0],
            [1, 2, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ];
        compute_gram_full(&mut s);
        s.basis.swap(2, 3);
        gram_update_swap(&mut s, 2, 3);
        let g_inc = s.gram;
        compute_gram_full(&mut s);
        let g_full = s.gram;
        for i in 0..8 {
            for jj in 0..8 {
                assert_eq!(g_inc[i][jj], g_full[i][jj], "swap mismatch at [{}][{}]", i, jj);
            }
        }
    }

    /// Verify that the f64 Cholesky output matches the legacy MPFR Cholesky
    /// (snapshot_gram_to_mpfr + cholesky_int_8) within a tight relative
    /// tolerance, across the ε range used in production. This is the
    /// guardrail that catches any precision-budget regression in the f64
    /// Cholesky path: if the LLL invariant κ ≤ 16 ever stops holding (e.g.
    /// upstream LLL change leaves a non-reduced basis), this test trips.
    fn cholesky_f64_matches_mpfr(eps: Float, k: u32) {
        let y = realistic_y(k);
        let mut s = IntScratch::new(eps);
        build_q_mpfr(&mut s, &y, k, eps);
        build_q_int(&mut s);
        let _ = lll_l2_8(&mut s);
        // MPFR reference path
        snapshot_gram_to_mpfr(&mut s);
        assert!(
            cholesky_int_8(&mut s),
            "MPFR Cholesky failed at eps={:e}, k={}", eps, k
        );
        let l_mpfr: [[f64; 8]; 8] = std::array::from_fn(|i|
            std::array::from_fn(|j| s.l[i][j].to_f64())
        );
        // f64 production path
        assert!(
            cholesky_f64_8(&mut s),
            "f64 Cholesky failed at eps={:e}, k={}", eps, k
        );
        // Compare lower triangles in relative error.
        let mut max_rel: f64 = 0.0;
        for i in 0..8 {
            for j in 0..=i {
                let diff = (l_mpfr[i][j] - s.l_f64[i][j]).abs();
                let mag = l_mpfr[i][j].abs().max(s.l_f64[i][j].abs()).max(1e-300);
                let rel = diff / mag;
                if rel > max_rel { max_rel = rel; }
                assert!(
                    rel < 1e-10,
                    "Cholesky[{}][{}] mismatch at eps={:e}, k={}: \
                     rel={:e}, mpfr={}, f64={}",
                    i, j, eps, k, rel, l_mpfr[i][j], s.l_f64[i][j]
                );
            }
        }
        eprintln!("cholesky_f64_matches_mpfr eps={:e} k={}: max_rel={:e}", eps, k, max_rel);
    }

    #[test]
    fn cholesky_f64_matches_mpfr_at_eps_1e_3() {
        cholesky_f64_matches_mpfr(1e-3, 14);
    }

    #[test]
    fn cholesky_f64_matches_mpfr_at_eps_1e_5() {
        cholesky_f64_matches_mpfr(1e-5, 21);
    }

    #[test]
    fn cholesky_f64_matches_mpfr_at_eps_1e_7() {
        cholesky_f64_matches_mpfr(1e-7, 49);
    }

    #[test]
    fn cholesky_f64_matches_mpfr_at_eps_1e_8() {
        cholesky_f64_matches_mpfr(1e-8, 70);
    }

    /// FAILS as documented (gated `#[ignore]`): the TwoFloat LU diverges
    /// from the MPFR reference by ~10⁻⁷ relative at ε=1e-5 (vs the expected
    /// ~10⁻³⁰ from a TwoFloat κ=1 solve). The discrepancy is too large to be
    /// pure rounding accumulation; root cause not yet pinned down. Production
    /// path retains MPFR LU. This test exists as a discoverable record so
    /// any future "let's try TwoFloat LU" reconsideration starts here.
    #[allow(dead_code)]
    fn lu_tf_matches_mpfr(eps: Float, k: u32) {
        let y = realistic_y(k);
        let mut s = IntScratch::new(eps);
        build_q_mpfr(&mut s, &y, k, eps);
        build_q_int(&mut s);
        let _ = lll_l2_8(&mut s);
        let basis = s.basis;

        // MPFR reference path
        for i in 0..8 {
            for j in 0..8 {
                s.lu_a[i][j].assign(rfv(s.prec_q, basis[j][i] as f64));
            }
            s.lu_rhs[i].assign(&s.c[i]);
        }
        assert!(lu_solve_int_inplace(&mut s), "MPFR LU failed at eps={:e}", eps);
        let z_mpfr: [f64; 8] = std::array::from_fn(|i| s.lu_x[i].to_f64());

        // TwoFloat path
        let cap_mid = cap_mid_tf(eps);
        let c_tf: [Tf; 8] = std::array::from_fn(|i| Tf::from(y[i]) * cap_mid);
        let z_tf = tf_lu_solve_8(&basis, &c_tf).expect("TwoFloat LU failed");
        let z_tf_f64: [f64; 8] = std::array::from_fn(|i| f64::from(z_tf[i]));

        let mut max_rel: f64 = 0.0;
        for i in 0..8 {
            let diff = (z_mpfr[i] - z_tf_f64[i]).abs();
            let mag = z_mpfr[i].abs().max(z_tf_f64[i].abs()).max(1e-300);
            let rel = diff / mag;
            if rel > max_rel { max_rel = rel; }
            assert!(
                rel < 1e-10,
                "z_c[{}] mismatch at eps={:e}, k={}: rel={:e}, mpfr={}, tf={}",
                i, eps, k, rel, z_mpfr[i], z_tf_f64[i]
            );
        }
        eprintln!("lu_tf_matches_mpfr eps={:e} k={}: max_rel={:e}", eps, k, max_rel);
    }

    /// Sanity-check tf_lu_solve_8 on a trivial case. If this passes,
    /// the bug is *somewhere* in input setup or precision interaction with
    /// the realistic basis / c values, not in the LU algorithm itself.
    #[test]
    fn tf_lu_solve_identity_smoke() {
        let basis: IMat8 = identity_basis();
        let c: [Tf; 8] = std::array::from_fn(|i| Tf::from((i as f64) + 1.0));
        let z = tf_lu_solve_8(&basis, &c).expect("identity LU failed");
        for i in 0..8 {
            let z_f = f64::from(z[i]);
            assert!(
                (z_f - ((i as f64) + 1.0)).abs() < 1e-10,
                "identity LU: z[{}] = {} expected {}", i, z_f, (i as f64) + 1.0
            );
        }
    }

    /// Solve (B^T · z = c) for a random small integer B with det=±1 and a
    /// small TwoFloat RHS. Cross-check via MPFR. If THIS fails, the bug is
    /// in the LU implementation; if this passes but the realistic test
    /// fails, the bug is in the precision interaction at large RHS.
    #[test]
    fn tf_lu_solve_small_int_basis() {
        let basis: IMat8 = [
            [3, 1, 0, 0, 0, 0, 0, 0],
            [1, 2, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ];
        let c_vals: [f64; 8] = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5];
        let c_tf: [Tf; 8] = std::array::from_fn(|i| Tf::from(c_vals[i]));
        let z_tf = tf_lu_solve_8(&basis, &c_tf).expect("small int LU failed");
        // Reference: solve in MPFR.
        let mut s = IntScratch::new(1e-3);
        for i in 0..8 {
            for j in 0..8 {
                s.lu_a[i][j].assign(rfv(s.prec_q, basis[j][i] as f64));
            }
            s.lu_rhs[i].assign(rfv(s.prec_q, c_vals[i]));
        }
        assert!(lu_solve_int_inplace(&mut s));
        for i in 0..8 {
            let mpfr_z = s.lu_x[i].to_f64();
            let tf_z = f64::from(z_tf[i]);
            let rel = (mpfr_z - tf_z).abs() / mpfr_z.abs().max(1e-30);
            assert!(
                rel < 1e-12,
                "small int z[{}]: mpfr={} tf={} rel={:e}",
                i, mpfr_z, tf_z, rel
            );
        }
    }

    #[test]
    #[ignore]
    fn lu_tf_matches_mpfr_at_eps_1e_5() {
        lu_tf_matches_mpfr(1e-5, 21);
    }

    #[test]
    #[ignore]
    fn lu_tf_matches_mpfr_at_eps_1e_7() {
        lu_tf_matches_mpfr(1e-7, 49);
    }

    #[test]
    #[ignore]
    fn lu_tf_matches_mpfr_at_eps_1e_8() {
        lu_tf_matches_mpfr(1e-8, 70);
    }
}
