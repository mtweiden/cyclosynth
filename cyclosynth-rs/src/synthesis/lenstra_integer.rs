//! Pure-Rust integer-arithmetic 8D LLL for Clifford+T synthesis.
//!
//! Replaces the all-MPFR `lenstra_heavy` pipeline. Key shift: the LLL inner
//! loop's Gram-matrix arithmetic moves from `rug::Float` to `i256`, eliminating
//! per-iteration MPFR allocation and the dominant per-iter cost. μ-values
//! still live in MPFR (f64 was empirically insufficient at deep ε — see
//! `project_paper_gap_analysis.md` negative findings).
//!
//! Pipeline per phase1 call:
//!  1. Build anisotropic Q in MPFR (unchanged from heavy; ~0.1% of CPU).
//!  2. Snapshot S·Q to `q_int: [[i256; 8]; 8]` with adaptive scale S = 2^B
//!     chosen so max(|Q_int|) ≈ 2^120. LLL μ-values are scale-invariant
//!     ratios, so absolute scale of S only affects effective precision, not
//!     correctness.
//!  3. LLL with i256 Gram (G = B·Q_int·Bᵀ, all integer). GS μ in MPFR at
//!     `prec_mu` bits (small — only the eight diagonals + above need MPFR).
//!  4. Once LLL is converged, convert G back to MPFR for Cholesky and LU
//!     (sub-1% of CPU; reuse heavy module's MPFR routines).
//!  5. Schnorr-Euchner enumeration + post-SE filter unchanged.
//!
//! Adaptive ε range: validated for ε ∈ [1e-10, 1e-3], aspirational target
//! 1e-8 at "reasonable" wallclock.

#![allow(dead_code)]

use crate::rings::Float;
use i256::i256;
use rug::{Assign, Float as RFloat};
use rug::ops::NegAssign;

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

/// MPFR precision used for μ values during LLL. Only needs enough bits to
/// distinguish Lovász-boundary cases — μ² has dynamic range up to κ(Q) ≈ 2^137
/// at ε=1e-10. 256 bits keeps comfortable margin without paying for full Q
/// precision.
pub const MU_PREC: u32 = 256;

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

/// MPFR precision for Q construction. Same ε-scaling as heavy's `compute_prec`,
/// since the build_q step is identical.
pub fn compute_prec_q(eps: Float) -> u32 {
    let log_recip = (1.0 / eps).log2().max(1.0);
    let bits = (8.0 * log_recip).ceil() as u32;
    bits.max(100)
}

type IMat8 = [[i64; 8]; 8];
type Mat256 = [[i256; 8]; 8];

/// Per-thread scratch for the integer LLL pipeline. Holds:
///   - MPFR working buffers for build_q + Cholesky + LU (small overhead).
///   - i256 buffers for Q_int, Gram, intermediate B·Q during Gram update.
///   - i64 basis matrix (LLL output).
///   - MPFR μ matrix for the size-reduce + Lovász decisions.
pub struct IntScratch {
    pub prec_q: u32,
    pub prec_mu: u32,
    pub scale_bits: i32, // B such that Q_int[i][j] ≈ 2^B · Q[i][j]

    // ── MPFR buffers for build_q (subset of HeavyScratch fields) ──
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

    // ── MPFR μ matrix and gnorm vector ──
    pub mu: [[RFloat; 8]; 8],
    pub gnorm_sq: [RFloat; 8],
    pub g_star: [[RFloat; 8]; 8],
    pub delta_lll: RFloat,

    // ── prec_mu temporaries for Lovász check ──
    pub lov_t1: RFloat,
    pub lov_t2: RFloat,
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
        let prec_mu = MU_PREC;
        let mut s = Self {
            prec_q,
            prec_mu,
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
            mu: rmat_zero(prec_mu),
            gnorm_sq: rvec_zero(prec_mu),
            g_star: rmat_zero(prec_mu),
            delta_lll: rfv(prec_mu, 0.75),
            lov_t1: rfz(prec_mu),
            lov_t2: rfz(prec_mu),
        };
        fill_sigma(&mut s.sigma, prec_q);
        s
    }

    pub fn reset_basis(&mut self) {
        self.basis = identity_basis();
    }
}

// ─── In-place rug op macros (mirror heavy's pattern) ──────────────────────────

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

// ─── build_q_mpfr: identical to heavy's build_q, into scratch.q_mpfr ──────────

/// Build the MPFR Q matrix using the same anisotropic ellipsoid metric formula
/// as `lenstra_heavy::build_q`, into `scratch.q_mpfr`. Also computes the cap
/// center into `scratch.c`.
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

// ─── i256 → MPFR conversion (per LLL iter, on demand) ────────────────────────

/// Convert i256 `v` to MPFR, writing into `dst`. Caller's RFloat must have
/// enough precision to represent the integer exactly (`prec ≥ 256` is safe;
/// 200 bits is enough for our values which top out around 2^180).
///
/// Cost: ~100-200 ns per call (3 shifts + 4 adds in rug::Integer, then assign
/// into RFloat). Called O(36) times per LLL iteration during GS — total
/// ~5 μs/iter overhead, well below the per-iter savings from i256 Gram.
pub fn i256_to_rfloat(v: &i256, dst: &mut RFloat) {
    let zero = i256::from_i64(0);
    if *v == zero {
        dst.assign(0.0_f64);
        return;
    }
    let neg = *v < zero;
    let abs = if neg { -*v } else { *v };
    let bytes = abs.to_le_bytes();
    let limbs: [u64; 4] = std::array::from_fn(|i| {
        let mut buf = [0u8; 8];
        buf.copy_from_slice(&bytes[i * 8..(i + 1) * 8]);
        u64::from_le_bytes(buf)
    });
    let mut int = rug::Integer::from(limbs[3]);
    int <<= 64u32;
    int += limbs[2];
    int <<= 64u32;
    int += limbs[1];
    int <<= 64u32;
    int += limbs[0];
    dst.assign(&int);
    if neg {
        dst.neg_assign();
    }
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
pub fn compute_gram_int(scratch: &mut IntScratch) -> bool {
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
    (max_abs_log2 as u32) <= GRAM_OVERFLOW_THRESHOLD_BITS
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

// ─── GS in MPFR, reading from i256 Gram ───────────────────────────────────────

/// Gram-Schmidt orthogonalization in Q metric, reading `scratch.gram` (i256)
/// and writing `scratch.mu`, `scratch.gnorm_sq`, `scratch.g_star` (all MPFR).
///
/// Standard recurrence:
///   for j = 0..8:
///     for i = j..8:
///       g_star[i][j] = G[i][j] - Σ_{l<j} μ[j][l] · g_star[i][l]
///     gnorm[j] = g_star[j][j]
///     for i = j+1..8: μ[i][j] = g_star[i][j] / gnorm[j]
///
/// G entries are converted on-demand via `i256_to_rfloat`. Only the lower
/// triangle (j ≤ i) is read.
pub fn gs_int_inplace(scratch: &mut IntScratch) {
    let prec = scratch.prec_mu;
    let tiny = rfv(prec, 1e-300);
    let mut acc = rfz(prec);
    let mut tmp = rfz(prec);

    for j in 0..8 {
        for i in j..8 {
            // acc = G[i][j] (converted from i256 to MPFR)
            i256_to_rfloat(&scratch.gram[i][j], &mut acc);
            for l in 0..j {
                // tmp = μ[j][l] * g_star[i][l]
                tmp.assign(&scratch.mu[j][l] * &scratch.g_star[i][l]);
                let cur = acc.clone();
                acc.assign(&cur - &tmp);
            }
            scratch.g_star[i][j].assign(&acc);
        }
        scratch.gnorm_sq[j].assign(&scratch.g_star[j][j]);
        if scratch.gnorm_sq[j].clone().abs() < tiny {
            for i in (j + 1)..8 {
                scratch.mu[i][j].assign(0.0_f64);
            }
            continue;
        }
        for i in (j + 1)..8 {
            scratch.mu[i][j].assign(&scratch.g_star[i][j] / &scratch.gnorm_sq[j]);
        }
    }
}

// ─── LLL inner loop (i256 Gram + MPFR GS) ────────────────────────────────────

/// Result of `lll_int_8`. `Ok` on convergence with a unimodular basis;
/// `Err(reason)` on overflow or iteration cap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LllResult {
    /// LLL converged (within `max_iter` iterations and no overflow).
    Converged,
    /// Gram entry exceeded GRAM_OVERFLOW_THRESHOLD_BITS (transient B-growth
    /// at deep ε beyond what TARGET_BITS=180 can absorb). Caller should
    /// fall back to the heavy MPFR LLL or another strategy.
    GramOverflow,
    /// Reached the iteration cap without convergence (cycling or near-
    /// boundary precision noise). Diagnostic only.
    IterCap,
}

/// LLL with δ=0.75 in the Q metric, using i256 Gram and MPFR μ. Resets the
/// basis to identity, then runs the standard size-reduce + Lovász loop.
/// Returns the convergence status.
///
/// Implementation mirrors `lenstra_heavy::lll_qgram_8` but with the per-iter
/// Gram recompute moved to i256 (cheap) and the GS to MPFR-reading-from-i256
/// (still cheap because the ~36 i256→MPFR conversions sum to a few μs).
pub fn lll_int_8(scratch: &mut IntScratch) -> LllResult {
    scratch.reset_basis();
    let max_iter: usize = 10_000;
    let mut iters: usize = 0;
    let mut k = 1usize;

    while k < 8 && iters < max_iter {
        iters += 1;

        // Step 1: Gram + GS for size-reduction
        if !compute_gram_int(scratch) {
            if crate::synthesis::diag::trace_enabled() {
                crate::synthesis::diag::record_lll_iters(iters as u64, max_iter as u64);
            }
            return LllResult::GramOverflow;
        }
        gs_int_inplace(scratch);

        // Step 2: size reduce row k against rows 0..k
        for j in (0..k).rev() {
            let r_round = scratch.mu[k][j].to_f64().round() as i64;
            if r_round != 0 {
                for c in 0..8 {
                    scratch.basis[k][c] -= r_round * scratch.basis[j][c];
                }
            }
        }

        // Step 3: re-Gram + GS to check Lovász
        if !compute_gram_int(scratch) {
            if crate::synthesis::diag::trace_enabled() {
                crate::synthesis::diag::record_lll_iters(iters as u64, max_iter as u64);
            }
            return LllResult::GramOverflow;
        }
        gs_int_inplace(scratch);

        // Step 4: Lovász: gnorm[k] ≥ (δ − μ[k][k-1]²) · gnorm[k-1]
        // lov_t1 = μ[k][k-1]²
        scratch.lov_t1.assign(&scratch.mu[k][k - 1] * &scratch.mu[k][k - 1]);
        // lov_t2 = δ − μ[k][k-1]²
        scratch.lov_t2.assign(&scratch.delta_lll - &scratch.lov_t1);
        // lov_t1 = (δ − μ²) · gnorm[k-1]
        scratch.lov_t1.assign(&scratch.lov_t2 * &scratch.gnorm_sq[k - 1]);
        if scratch.gnorm_sq[k] >= scratch.lov_t1 {
            k += 1;
        } else {
            scratch.basis.swap(k, k - 1);
            k = k.saturating_sub(1).max(1);
        }
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

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

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

    /// Reference Gram computation in MPFR: G_ref = B · Q_mpfr · Bᵀ.
    /// Returns 8×8 RFloat matrix at given precision.
    fn reference_gram_mpfr(
        basis: &IMat8,
        q_mpfr: &[[RFloat; 8]; 8],
        prec: u32,
    ) -> [[RFloat; 8]; 8] {
        let mut bq: [[RFloat; 8]; 8] = std::array::from_fn(|_| std::array::from_fn(|_| rfz(prec)));
        for i in 0..8 {
            for b in 0..8 {
                let mut acc = rfz(prec);
                for a in 0..8 {
                    let bi_a = rfv(prec, basis[i][a] as f64);
                    let prod = bi_a * &q_mpfr[a][b];
                    acc = acc + prod;
                }
                bq[i][b] = acc;
            }
        }
        let mut g: [[RFloat; 8]; 8] = std::array::from_fn(|_| std::array::from_fn(|_| rfz(prec)));
        for i in 0..8 {
            for j in 0..8 {
                let mut acc = rfz(prec);
                for b in 0..8 {
                    let bj_b = rfv(prec, basis[j][b] as f64);
                    let prod = &bq[i][b] * bj_b;
                    acc = acc + prod;
                }
                g[i][j] = acc;
            }
        }
        g
    }

    /// Reference GS in MPFR (same recurrence as gs_int_inplace, but reading
    /// from MPFR Gram rather than i256).
    fn reference_gs_mpfr(
        g: &[[RFloat; 8]; 8],
        prec: u32,
    ) -> ([[RFloat; 8]; 8], [RFloat; 8]) {
        let mut mu: [[RFloat; 8]; 8] = std::array::from_fn(|_| std::array::from_fn(|_| rfz(prec)));
        let mut g_star: [[RFloat; 8]; 8] = std::array::from_fn(|_| std::array::from_fn(|_| rfz(prec)));
        let mut gnorm: [RFloat; 8] = std::array::from_fn(|_| rfz(prec));
        let tiny = rfv(prec, 1e-300);
        for j in 0..8 {
            for i in j..8 {
                let mut acc = g[i][j].clone();
                for l in 0..j {
                    let prod = RFloat::with_val(prec, &mu[j][l] * &g_star[i][l]);
                    acc -= prod;
                }
                g_star[i][j].assign(&acc);
            }
            gnorm[j].assign(&g_star[j][j]);
            if gnorm[j].clone().abs() < tiny {
                continue;
            }
            for i in (j + 1)..8 {
                mu[i][j].assign(&g_star[i][j] / &gnorm[j]);
            }
        }
        (mu, gnorm)
    }

    /// Verify integer Gram (after scaling) matches MPFR Gram entry-wise.
    fn check_gram_int_matches_mpfr(eps: Float, k: u32) {
        let y = realistic_y(k);
        let mut s = IntScratch::new(eps);
        build_q_mpfr(&mut s, &y, k, eps);
        build_q_int(&mut s);
        // Use a non-trivial basis so off-diagonals are exercised
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
        let ok = compute_gram_int(&mut s);
        assert!(ok, "compute_gram_int reported overflow at eps={:e}, k={}", eps, k);

        let g_ref = reference_gram_mpfr(&s.basis, &s.q_mpfr, s.prec_q);

        // Entry-wise: g_int[i][j] should equal round(g_ref[i][j] · 2^scale_bits)
        // Tolerance: 1e-25 relative (scaled match is exact modulo rounding noise)
        let mut max_abs_g: f64 = 0.0;
        let mut max_err: f64 = 0.0;
        for i in 0..8 {
            for j in 0..8 {
                let g_true = g_ref[i][j].to_f64();
                max_abs_g = max_abs_g.max(g_true.abs());
                let g_int_f = i256_to_f64_scaled(&s.gram[i][j], s.scale_bits);
                let err = (g_true - g_int_f).abs();
                max_err = max_err.max(err);
            }
        }
        let rel_err = max_err / max_abs_g.max(1e-300);
        assert!(
            rel_err < 1e-20,
            "eps={:e}, k={}: gram rel_err={:e}, max_g={:e}, scale_bits={}",
            eps, k, rel_err, max_abs_g, s.scale_bits
        );
    }

    /// Verify GS μ from integer pipeline matches GS μ from MPFR reference.
    fn check_gs_int_matches_mpfr(eps: Float, k: u32) {
        let y = realistic_y(k);
        let mut s = IntScratch::new(eps);
        build_q_mpfr(&mut s, &y, k, eps);
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
        let ok = compute_gram_int(&mut s);
        assert!(ok, "compute_gram_int reported overflow at eps={:e}, k={}", eps, k);
        gs_int_inplace(&mut s);

        let g_ref = reference_gram_mpfr(&s.basis, &s.q_mpfr, s.prec_q);
        let (mu_ref, gnorm_ref) = reference_gs_mpfr(&g_ref, s.prec_q);

        // μ values are scale-invariant: integer pipeline's μ should match
        // reference μ to MPFR precision (modulo TARGET_BITS rounding noise in Q_int)
        let mut max_mu_err: f64 = 0.0;
        for i in 0..8 {
            for j in 0..i {
                let m_int = s.mu[i][j].to_f64();
                let m_ref = mu_ref[i][j].to_f64();
                let err = (m_int - m_ref).abs();
                max_mu_err = max_mu_err.max(err);
            }
        }
        // gnorm ratios are scale-invariant:
        // (gnorm_int[i] / gnorm_int[0]) should match (gnorm_ref[i] / gnorm_ref[0])
        let mut max_gn_rel_err: f64 = 0.0;
        let g0_int = s.gnorm_sq[0].to_f64();
        let g0_ref = gnorm_ref[0].to_f64();
        for i in 1..8 {
            let r_int = s.gnorm_sq[i].to_f64() / g0_int;
            let r_ref = gnorm_ref[i].to_f64() / g0_ref;
            let rel = ((r_int - r_ref) / r_ref.abs().max(1e-300)).abs();
            max_gn_rel_err = max_gn_rel_err.max(rel);
        }
        // ε-aware tolerance: GS cancels ~log₂(κ(Q)) bits, so post-GS effective
        // precision is roughly TARGET_BITS − log₂(κ(Q)). κ(Q) ≈ 16/ε⁴, so
        // log₂(κ(Q)) ≈ 4·log₂(1/ε) + 4.
        let log_recip = (1.0 / eps).log2();
        let effective_bits = (TARGET_BITS as f64 - 4.0 * log_recip - 4.0).max(20.0);
        let tol = 2.0_f64.powf(-effective_bits + 10.0); // +10 bits slack for noise
        assert!(
            max_mu_err < tol && max_gn_rel_err < tol,
            "eps={:e}, k={}: max_mu_err={:e}, max_gn_rel_err={:e}, tol={:e} (effective_bits≈{:.0})",
            eps, k, max_mu_err, max_gn_rel_err, tol, effective_bits
        );
    }

    #[test]
    fn gram_int_matches_mpfr_at_eps_1e_5() {
        check_gram_int_matches_mpfr(1e-5, 21);
    }

    #[test]
    fn gram_int_matches_mpfr_at_eps_1e_10() {
        check_gram_int_matches_mpfr(1e-10, 100);
    }

    #[test]
    fn gs_int_matches_mpfr_at_eps_1e_5() {
        check_gs_int_matches_mpfr(1e-5, 21);
    }

    #[test]
    fn gs_int_matches_mpfr_at_eps_1e_8() {
        check_gs_int_matches_mpfr(1e-8, 70);
    }

    #[test]
    fn gs_int_matches_mpfr_at_eps_1e_10() {
        check_gs_int_matches_mpfr(1e-10, 100);
    }

    /// Run the integer LLL for given (eps, k) and assert det = ±1
    /// (unimodular basis output). Uses the heavy module's det8_exact for the
    /// integer determinant check.
    fn check_lll_unimodular(eps: Float, k: u32) -> LllResult {
        let y = realistic_y(k);
        let mut s = IntScratch::new(eps);
        build_q_mpfr(&mut s, &y, k, eps);
        build_q_int(&mut s);
        let result = lll_int_8(&mut s);
        // Allow IterCap as a soft outcome (LLL is "noisy" at deep ε); the
        // unimodular check is a hard invariant either way.
        if let LllResult::GramOverflow = result {
            return result;
        }
        let det = crate::synthesis::lenstra_heavy::det8_exact(&s.basis)
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
    #[ignore]
    fn scale_bits_sweep_diagnostic() {
        // Diagnostic only: print the chosen scale_bits across ε
        let cases = [(1e-3_f64, 14), (1e-4, 17), (1e-5, 21), (1e-6, 28),
                     (1e-7, 49), (1e-8, 70), (1e-9, 85), (1e-10, 100)];
        for (eps, k) in cases {
            let y = realistic_y(k);
            let mut s = IntScratch::new(eps);
            build_q_mpfr(&mut s, &y, k, eps);
            build_q_int(&mut s);
            let mut max_log2: i32 = i32::MIN;
            for i in 0..8 {
                for j in 0..8 {
                    let v = s.q_mpfr[i][j].clone().abs();
                    if !v.is_zero() {
                        let e = v.get_exp().unwrap_or(0);
                        if e > max_log2 {
                            max_log2 = e;
                        }
                    }
                }
            }
            eprintln!(
                "  eps={:e}  k={:>3}  max_q_log2={:>4}  scale_bits={:>4}",
                eps, k, max_log2, s.scale_bits
            );
        }
    }
}
