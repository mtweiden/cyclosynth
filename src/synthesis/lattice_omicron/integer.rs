//! L²-LLL pipeline for the Clifford+R (R=Rz(π/6)) n=6 synthesis Lenstra path.
//!
//! Mirrors `lattice::integer` but adapted for the Z[ξ] ring (ξ=e^{iπ/6}):
//!
//! ## Ring-specific differences from the n=4 (`lattice::integer`) version
//!
//! 1. **Norm check**: for n=6 the SU(2) unitarity norm equation is
//!    `‖x‖² + (a₀a₂ + a₁a₃ + b₀b₂ + b₁b₃) = 2^k` (the Gram is non-scalar).
//!    n=4 used the simpler `‖x‖² = 2^k` (Gram = 2I, scalar).
//!
//! 2. **Alignment threshold**: for n=6 the threshold on |x·y|² is `2^k·(1−ε²)`
//!    (matching `clifford_pi6::check_alignment`). n=4 used `2^(2k)·(1−ε²)/4`
//!    which reflects n=4's different y-normalization.
//!
//! 3. **Euclidean-prune bound**: for n=6 valid x has Euclidean norm ≤ 2·2^k
//!    (since the cross term can be negative, shifting ‖x‖² up from 2^k).
//!    n=4 used exactly 2^k.
//!
//! 4. **Bilinear form**: calls `super::se::bilinear_b` which for n=6 is
//!    `a₀a₁+a₁a₂+a₂a₃+b₀b₁+b₁b₂+b₂b₃` (consecutive-coord, all +).

#![allow(dead_code)]
#![allow(clippy::needless_range_loop)]

use rug::{Assign, Float as RFloat};
use std::sync::atomic::AtomicBool;

use super::cholesky_lu::{cholesky_f64_8, lu_solve_int_inplace};
use super::lll::{lll_l2_8, LllResult};
use super::q_metric::{build_q_int, build_q_mpfr};
use super::scratch::{rfv, IntScratch};
use crate::rings::Float;

/// Outcome of one `phase1` invocation.
pub struct PhaseOneOutcome {
    pub solutions: Vec<[i64; 8]>,
    pub should_escalate: bool,
}

/// Run the full Lenstra 8D pipeline for one (y, k, eps) target.
/// Returns at most one valid 8-vector solution; raise `max_phase2_calls`
/// to get more.
pub fn phase1(
    scratch: &mut IntScratch,
    y: &[Float; 8],
    k: u32,
    eps: Float,
    max_phase2_calls: u64,
    budget_hit: &AtomicBool,
) -> PhaseOneOutcome {
    use std::sync::atomic::{AtomicU64, Ordering};

    // target_norm = 2^k.  For n=6 the norm equation is
    //   ‖x‖² + cross = 2^k  (cross = a₀a₂+a₁a₃+b₀b₂+b₁b₃)
    // so target_norm is the RHS of that equation.
    let target_norm: i128 = 1i128 << k;
    let use_i64_path = k <= 62;
    let target_norm_i64: i64 = if use_i64_path { 1i64 << k } else { 0 };

    // ── Alignment threshold (n=6) ──────────────────────────────────────────
    // |x · y|² ≥ 2^k · (1 − ε²)  (STEP 2 alignment bound, base 2).
    // This matches clifford_pi6::check_alignment exactly.
    let prec = super::se::SE_PREC;
    let two_to_k = RFloat::with_val(prec, 1.0) << k;
    let eps_rf = RFloat::with_val(prec, eps);
    let one_minus_eps_sq =
        RFloat::with_val(prec, 1.0) - eps_rf.clone() * &eps_rf;
    let threshold_xy_mpfr =
        RFloat::with_val(prec, &two_to_k * &one_minus_eps_sq);
    let y_mpfr: [RFloat; 8] = std::array::from_fn(|i| RFloat::with_val(prec, y[i]));

    let trace = crate::synthesis::diag::trace_enabled();

    // Step 1: build Q in MPFR + integer snapshot.
    let t_phase = if trace { Some(std::time::Instant::now()) } else { None };
    build_q_mpfr(scratch, y, k, eps);
    build_q_int(scratch);
    if let Some(t0) = t_phase {
        crate::synthesis::diag::T_BUILD_NS
            .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

    // Step 2: L²-LLL (f64 GS over exact i256 Gram + INSERT semantics).
    let t_phase = if trace { Some(std::time::Instant::now()) } else { None };
    let lll_result = lll_l2_8(scratch);
    if let Some(t0) = t_phase {
        crate::synthesis::diag::T_LLL_NS
            .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
    if let LllResult::GramOverflow = lll_result {
        return PhaseOneOutcome { solutions: Vec::new(), should_escalate: true };
    }

    // Step 3: assert det(B) = ±1 (unimodular basis output).
    let basis = scratch.basis;
    match super::se::det8_exact(&basis) {
        Some(1) | Some(-1) => {}
        Some(d) => {
            eprintln!(
                "[lattice_omicron] LLL non-unimodular (det={}) at eps={:e}, k={}; bailing.",
                d, eps, k
            );
            return PhaseOneOutcome { solutions: Vec::new(), should_escalate: false };
        }
        None => {
            eprintln!(
                "[lattice_omicron] det8_exact overflow at eps={:e}, k={}; bailing.",
                eps, k
            );
            return PhaseOneOutcome { solutions: Vec::new(), should_escalate: false };
        }
    }

    // Step 4: f64 Cholesky on the i256 Gram.
    let t_phase = if trace { Some(std::time::Instant::now()) } else { None };
    let chol_ok = cholesky_f64_8(scratch);
    if let Some(t0) = t_phase {
        crate::synthesis::diag::T_CHOLESKY_NS
            .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
    if !chol_ok {
        eprintln!(
            "[lattice_omicron] Cholesky (f64) failed at eps={:e}, k={}; bailing.",
            eps, k
        );
        return PhaseOneOutcome { solutions: Vec::new(), should_escalate: false };
    }

    // Build R = Lᵀ at SE working precision (128-bit MPFR).
    let r_chol_se: [[RFloat; 8]; 8] = std::array::from_fn(|i| {
        std::array::from_fn(|j| RFloat::with_val(super::se::SE_PREC, scratch.l_f64[j][i]))
    });

    // Step 5: solve B_LLLᵀ · z_c = c for the cap-center in lattice coords.
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
        eprintln!("[lattice_omicron] LU solve failed at eps={:e}, k={}; bailing.", eps, k);
        return PhaseOneOutcome { solutions: Vec::new(), should_escalate: false };
    }
    let z_c_se: [RFloat; 8] = std::array::from_fn(|i| {
        super::se::rfloat_to_se(&scratch.lu_x[i])
    });

    // Step 6: Schnorr-Euchner walk at MPFR-128.
    //
    // Euclidean prune: for n=6 valid x satisfies ‖x‖²_euclid ≤ 2·2^k
    // (the cross term a₀a₂+… can be negative, so euclid = 2^k − cross ≤ 2·2^k).
    // Pass 2·target_norm as the upper bound so no valid candidate is pruned.
    let r_eucl = super::se::euclidean_cholesky(&basis);
    let target_norm_eucl_f = 2.0 * target_norm as f64;
    let count = AtomicU64::new(0);
    let abort = AtomicBool::new(false);
    let bound_se = RFloat::with_val(super::se::SE_PREC, 1.51_f64);
    let t_phase = if trace { Some(std::time::Instant::now()) } else { None };

    let result = super::se::schnorr_euchner_8d(
        &r_chol_se,
        &z_c_se,
        &bound_se,
        r_eucl.as_ref(),
        target_norm_eucl_f,
        &abort,
        |z: &[i64; 8]| {
            let n_so_far = count.load(Ordering::Relaxed);
            if n_so_far >= max_phase2_calls {
                budget_hit.store(true, Ordering::Relaxed);
                return None;
            }
            count.fetch_add(1, Ordering::Relaxed);
            let x = super::se::reconstruct_x(&basis, z);

            // ── Norm check (n=6) ─────────────────────────────────────────
            // Equation: ‖x‖² + (a₀a₂ + a₁a₃ + b₀b₂ + b₁b₃) = 2^k
            if use_i64_path {
                let euclid: i64 = x.iter().map(|&v| v * v).sum();
                let cross: i64 = x[0]*x[2] + x[1]*x[3] + x[4]*x[6] + x[5]*x[7];
                if euclid + cross != target_norm_i64 {
                    return None;
                }
            } else {
                let euclid: i128 = x.iter().map(|&v| (v as i128) * (v as i128)).sum();
                let cross: i128 = (x[0] as i128)*(x[2] as i128)
                    + (x[1] as i128)*(x[3] as i128)
                    + (x[4] as i128)*(x[6] as i128)
                    + (x[5] as i128)*(x[7] as i128);
                if euclid + cross != target_norm {
                    return None;
                }
            }

            // ── Bilinear check (n=6) ─────────────────────────────────────
            // a₀a₁ + a₁a₂ + a₂a₃ + b₀b₁ + b₁b₂ + b₂b₃ = 0
            if super::se::bilinear_b(&x) != 0 {
                return None;
            }

            // ── Alignment check at MPFR-128 ──────────────────────────────
            // |x · y|² ≥ 2^k · (1 − ε²)
            let mut tmp = RFloat::with_val(prec, 0.0);
            let mut dot_acc = RFloat::with_val(prec, 0.0);
            for (xv, yv) in x.iter().zip(y_mpfr.iter()) {
                tmp.assign(*xv);
                tmp *= yv;
                dot_acc += &tmp;
            }
            tmp.assign(&dot_acc * &dot_acc);
            if tmp < threshold_xy_mpfr {
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
