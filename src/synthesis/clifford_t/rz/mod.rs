//! Native Ross–Selinger z-rotation synthesis (no external crates).
//!
//! A from-scratch Rust implementation of the Ross–Selinger z-rotation
//! algorithm (arXiv:1403.2975) on cyclosynth-native machinery:
//! `rug::Integer` ring coefficients, `MpFloat` reals with explicit
//! precision (no process-global state), a deterministic SplitMix64 for
//! the Las Vegas factoring, and the crate's own `BlochDecomposer` for
//! the final exact unitary → gate-string step. Portions derived from
//! pygridsynth (MIT — see THIRD_PARTY.md); outputs are validated against
//! the reference implementations in the probe battery. The module layout
//! matches the Clifford+√T route (`clifford_sqrt_t::rz`).
//!
//! Layout mirrors the algorithm: the rings live in `crate::rings`
//! (`real`/`dyadic`/big cyclotomics) → `geometry` (ε-cap + disks, grid
//! operators, upright reduction) → `grid_solver` (1D/2D grid enumeration
//! by increasing k) → `norm_eq` (the Diophantine step t†t = ξ, on the
//! shared `synthesis::factor` number theory) → this driver (k/phase
//! schedule + assembly).

// Ported numeric code: k/denominator-exponent conversions are range-safe
// by construction (k ≲ 4·log₂(1/ε) ≪ 2³²) and the factoring-budget floats
// are heuristics mirroring the reference implementation.
#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss, clippy::cast_sign_loss)]

pub(crate) mod ladder;
pub(crate) mod geometry;
pub(crate) mod grid_solver;
pub(crate) mod norm_eq;

use rug::Integer;

use crate::matrix::U2T;
use crate::rings::types::MpFloat;
use crate::rings::ZOmega;
use crate::synthesis::decomposer::BlochDecomposer;

use crate::synthesis::factor::{Budget, Rng};
use norm_eq::diophantine;
use crate::rings::dyadic::{DOmega, DRootTwo};
use crate::rings::real::ZRootTwo;
use crate::rings::zomega::ZOmegaBig;
use grid_solver::{ConvexSet, EpsilonRegion, UnitDisk};
use grid_solver::solve_tdgp_visit;
use geometry::{to_upright_ellipse_pair, to_upright_set_pair};

/// Deterministic seed for the Diophantine factoring randomness.
const GS_SEED: u64 = 0xC0FFEE;

/// xi = ξ/√2^k → w with w†w = xi.
fn diophantine_dyadic(xi: &DRootTwo, rng: &mut Rng, budget: &mut Budget) -> Option<DOmega> {
    let k_div_2 = xi.k >> 1;
    let k_mod_2 = xi.k & 1;
    let arg = if k_mod_2 == 1 {
        xi.alpha.mul(&ZRootTwo::lambda())
    } else {
        xi.alpha.clone()
    };
    let t = diophantine(&arg, rng, budget)?;
    let t = if k_mod_2 == 1 {
        t.mul(&ZOmegaBig::from_i64(0, 1, -1, 0))
    } else {
        t
    };
    Some(DOmega::new(t, k_div_2 + k_mod_2))
}

/// rug Integer → I256 coefficient (None above 254 bits — the final
/// unitary's coefficients are ~2^(k/2), so this bounds ε at ~1e-100).
pub(crate) fn integer_to_i256(x: &Integer) -> Option<crate::rings::types::Int> {
    use crate::rings::types::Int;
    if x.significant_bits() > 254 {
        return None;
    }
    let mut mag = x.clone().abs();
    let mut acc = Int::from_i8(0);
    let mut shift = 0u32;
    while mag != 0 {
        let limb = mag.to_u64_wrapping();
        acc += Int::from_i128(i128::from(limb)) << shift;
        mag >>= 64u32;
        shift += 64;
    }
    Some(if *x < 0 { -acc } else { acc })
}

/// Width conversion only — both types share the (1, ω, ω², ω³) basis.
fn to_crate_zomega(z: &ZOmegaBig) -> Option<ZOmega> {
    Some(ZOmega::new(
        integer_to_i256(&z.a)?,
        integer_to_i256(&z.b)?,
        integer_to_i256(&z.c)?,
        integer_to_i256(&z.d)?,
    ))
}

/// Assemble U = [[z, −w†ωⁿ], [w, z†ωⁿ]] / √2^k and decompose to gates.
fn assemble_gates(z: &DOmega, w: &DOmega, n: i32) -> Option<String> {
    let k = z.k.max(w.k);
    // The decomposer's exact SO(3) representation carries numerators ~2^k
    // in I256: beyond k = 250 it would overflow silently (T ≈ 500,
    // ε ≈ 1e-50). Refuse cleanly — the driver reports Fatal → None.
    if k > 250 {
        return None;
    }
    let z = z.renew_denomexp(k);
    let w = w.renew_denomexp(k);
    let u11 = to_crate_zomega(&z.u)?;
    let u12 = to_crate_zomega(&w.u.conj().mul_by_omega_power(n).neg())?;
    let u21 = to_crate_zomega(&w.u)?;
    let u22 = to_crate_zomega(&z.u.conj().mul_by_omega_power(n))?;
    let u2t = U2T::new(u11, u12, u21, u22, k).reduced();
    Some(BlochDecomposer.decompose(&u2t))
}

/// Try TDGP candidates at level k in enumeration order until one admits a
/// Diophantine companion; stops the enumeration at the first success
///.
#[allow(clippy::too_many_arguments)]
fn try_fixed_k(
    set_a: &dyn ConvexSet,
    set_b: &dyn ConvexSet,
    op_g: &geometry::GridOp,
    bbox_a: &geometry::Rectangle,
    bbox_b: &geometry::Rectangle,
    k: u32,
    has_phase: bool,
    prec: u32,
    rng: &mut Rng,
) -> LevelOutcome {
    let mut found: Option<LevelOutcome> = None;
    let mut visit = |mut z: DOmega| -> bool {
        if z.mul(&z.conj()).residue() == 0 {
            return false;
        }
        if has_phase {
            z = z.mul(&DOmega::new(ZOmegaBig::from_i64(0, 1, -1, 0), 1));
        }
        let xi = DRootTwo::from_int_i64(1).sub(&DRootTwo::from_domega(&z.conj().mul(&z)));
        let mut budget = Budget::default();
        let Some(w) = diophantine_dyadic(&xi, rng, &mut budget) else {
            return false;
        };
        let mut z = z.reduce_denomexp();
        let mut w = w.reduce_denomexp();
        let k_common = z.k.max(w.k);
        z = z.renew_denomexp(k_common);
        w = w.renew_denomexp(k_common);
        let k1 = z.add(&w).reduce_denomexp().k;
        let k2 = z.add(&w.mul_by_omega()).reduce_denomexp().k;
        let k3 = z.add(&w.mul_by_omega_inv()).reduce_denomexp().k;
        let (w_sel, n) = if has_phase {
            if k1 <= k2 && k1 <= k3 {
                (w, -1)
            } else {
                (w.mul_by_omega_inv(), -1)
            }
        } else if k1 <= k2 {
            (w, 0)
        } else {
            (w.mul_by_omega(), 0)
        };
        // Assembly fails only on coefficient overflow, which is fatal for
        // the whole call: coefficients only grow with k. Abort fast rather
        // than scanning every deeper level with the same outcome.
        found = Some(match assemble_gates(&z, &w_sel, n) {
            Some(gates) => LevelOutcome::Found(gates),
            None => LevelOutcome::Fatal,
        });
        true
    };
    solve_tdgp_visit(set_a, set_b, op_g, bbox_a, bbox_b, k, prec, &mut visit);
    found.unwrap_or(LevelOutcome::Empty)
}

/// Outcome of scanning one (k, phase) level.
enum LevelOutcome {
    /// No candidate admitted a companion — try the next level.
    Empty,
    Found(String),
    /// Result unrepresentable (coefficients beyond I256) — give up.
    Fatal,
}

/// Synthesize Rz(θ) over Clifford+T with the up-to-phase criterion.
/// Returns the gate string (matrix order, {H,S,T,X,Y,Z}); the caller
/// verifies the diamond distance with the crate's own acceptance check.
pub(crate) fn gridsynth_gates_native(theta: &MpFloat, epsilon: f64, prec: u32) -> Option<String> {
    let theta = MpFloat::with_val(prec, theta);
    let eps = MpFloat::with_val(prec, epsilon);
    let mut rng = Rng::new(GS_SEED);

    let region0 = EpsilonRegion::new(&theta, &eps, ZRootTwo::one(), prec);
    let disk0 = UnitDisk::new(ZRootTwo::one(), prec);
    let region1 = EpsilonRegion::new(&theta, &eps, ZRootTwo::from_i64(2, 1), prec);
    let disk1 = UnitDisk::new(ZRootTwo::from_i64(2, -1), prec);

    let op_g = to_upright_ellipse_pair(region0.ellipse(), disk0.ellipse(), prec);
    let (g0, _ea0, _eb0, bbox_a0, bbox_b0) =
        to_upright_set_pair(region0.ellipse(), disk0.ellipse(), Some(op_g.clone()), prec);
    let (g1, _ea1, _eb1, bbox_a1, bbox_b1) =
        to_upright_set_pair(region1.ellipse(), disk1.ellipse(), Some(op_g), prec);

    // Interleave the plain and ω-phase-shifted branches per k so the
    // first (lowest-k) hit across both wins.
    let mut k: u32 = 0;
    let mut has_phase = false;
    // Generous stop: ~4·log2(1/ε) + margin levels, the worst-case bound.
    let k_cap = (4.0 * (1.0 / epsilon).log2()).ceil() as u32 + 40;
    loop {
        let found = if has_phase {
            try_fixed_k(&region1, &disk1, &g1, &bbox_a1, &bbox_b1, k, true, prec, &mut rng)
        } else {
            try_fixed_k(&region0, &disk0, &g0, &bbox_a0, &bbox_b0, k, false, prec, &mut rng)
        };
        match found {
            LevelOutcome::Found(gates) => return Some(gates),
            LevelOutcome::Fatal => return None,
            LevelOutcome::Empty => {}
        }
        if k >= 2 {
            has_phase = !has_phase;
            if has_phase {
                k += 1;
            }
        } else if k == 0 && !has_phase {
            k = 1;
        } else if k == 1 && !has_phase {
            k = 0;
            has_phase = true;
        } else if k == 0 && has_phase {
            k = 1;
        } else {
            // k == 1 && has_phase
            k = 2;
            has_phase = false;
        }
        if k > k_cap {
            return None;
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Which (k, phase) level produced the ε=1e-28 generic result, and is
    /// the assembled unitary S-shifted? (bug bisect probe)
    /// Run: `cargo test --release --lib gridsynth_phase_branch_bisect -- --ignored --nocapture`
    #[test]
    #[ignore = "bisect probe, print-only"]
    fn gridsynth_phase_branch_bisect() {
        let eps = 1e-28_f64;
        let prec = crate::synthesis::clifford_t::rz::ladder::prec_for_epsilon(eps);
        let th = 1.0472_f64;
        let theta = MpFloat::with_val(prec, th);
        let eps_f = MpFloat::with_val(prec, eps);
        let mut rng = Rng::new(GS_SEED);
        let region0 = EpsilonRegion::new(&theta, &eps_f, ZRootTwo::one(), prec);
        let disk0 = UnitDisk::new(ZRootTwo::one(), prec);
        let region1 = EpsilonRegion::new(&theta, &eps_f, ZRootTwo::from_i64(2, 1), prec);
        let disk1 = UnitDisk::new(ZRootTwo::from_i64(2, -1), prec);
        let op_g = to_upright_ellipse_pair(region0.ellipse(), disk0.ellipse(), prec);
        let (g0, _, _, ba0, bb0) = to_upright_set_pair(
            region0.ellipse(), disk0.ellipse(), Some(op_g.clone()), prec);
        let (g1, _, _, ba1, bb1) = to_upright_set_pair(
            region1.ellipse(), disk1.ellipse(), Some(op_g), prec);
        let mut k: u32 = 0;
        let mut has_phase = false;
        for _ in 0..300 {
            let out = if has_phase {
                try_fixed_k(&region1, &disk1, &g1, &ba1, &bb1, k, true, prec, &mut rng)
            } else {
                try_fixed_k(&region0, &disk0, &g0, &ba0, &bb0, k, false, prec, &mut rng)
            };
            match out {
                LevelOutcome::Found(g) => {
                    eprintln!(
                        "FOUND at k={k} has_phase={has_phase} T={}",
                        g.chars().filter(|&c| c == 'T' || c == 't').count()
                    );
                    // Split assembly vs decomposer: re-derive the found
                    // candidate deterministically (same rng state replay is
                    // impractical; instead re-scan this level capturing z/w).
                    let mut rng2 = Rng::new(GS_SEED);
                    let mut captured: Option<(DOmega, DOmega, i32)> = None;
                    {
                        let cap = &mut captured;
                        let rng2 = &mut rng2;
                        let mut visit = |z: DOmega| -> bool {
                            if z.mul(&z.conj()).residue() == 0 {
                                return false;
                            }
                            let xi = DRootTwo::from_int_i64(1)
                                .sub(&DRootTwo::from_domega(&z.conj().mul(&z)));
                            let mut budget = Budget::default();
                            let Some(w) = diophantine_dyadic(&xi, rng2, &mut budget) else {
                                return false;
                            };
                            let mut z = z.reduce_denomexp();
                            let mut w = w.reduce_denomexp();
                            let kc = z.k.max(w.k);
                            z = z.renew_denomexp(kc);
                            w = w.renew_denomexp(kc);
                            let k1 = z.add(&w).reduce_denomexp().k;
                            let k2 = z.add(&w.mul_by_omega()).reduce_denomexp().k;
                            let w_sel = if k1 <= k2 { w } else { w.mul_by_omega() };
                            *cap = Some((z, w_sel, 0));
                            true
                        };
                        solve_tdgp_visit(&region0, &disk0, &g0, &ba0, &bb0, k, prec, &mut visit);
                    }
                    let (z, w_sel, n) = captured.expect("re-capture");
                    // Verify the assembled U2T directly (pre-decompose).
                    let kk = z.k.max(w_sel.k);
                    let z2 = z.renew_denomexp(kk);
                    let w2 = w_sel.renew_denomexp(kk);
                    let u11 = to_crate_zomega(&z2.u).expect("fits");
                    let u12 = to_crate_zomega(&w2.u.conj().mul_by_omega_power(n).neg()).expect("fits");
                    let u21 = to_crate_zomega(&w2.u).expect("fits");
                    let u22 = to_crate_zomega(&z2.u.conj().mul_by_omega_power(n)).expect("fits");
                    let u2t = U2T::new(u11, u12, u21, u22, kk).reduced();
                    let vp = prec + 128;
                    let dd_of = |u: &U2T| -> f64 {
                        let theta_v = MpFloat::with_val(vp, th);
                        let half = MpFloat::with_val(vp, &theta_v / 2u32);
                        let (t_re, t_im) = (half.clone().cos(), half.sin());
                        let sc = crate::rings::dyadic::sqrt2_pow_f(u.k, vp);
                        let inv_sqrt2 = MpFloat::with_val(vp, 0.5).sqrt();
                        let exact = |x| {
                            let i = Integer::from_str_radix(&format!("{x}"), 10).expect("dec");
                            MpFloat::with_val(vp, &i)
                        };
                        let entry = |zz: &ZOmega| {
                            let (a, b, c, d) = (exact(zz.a), exact(zz.b), exact(zz.c), exact(zz.d));
                            let re = (a + MpFloat::with_val(vp, &b - &d) * &inv_sqrt2) / &sc;
                            let im = (c + MpFloat::with_val(vp, &b + &d) * &inv_sqrt2) / &sc;
                            (re, im)
                        };
                        let (u00r, u00i) = entry(&u.u11);
                        let (u11r, u11i) = entry(&u.u22);
                        let tr_re = MpFloat::with_val(vp, &u00r * &t_re)
                            - MpFloat::with_val(vp, &u00i * &t_im)
                            + MpFloat::with_val(vp, &u11r * &t_re)
                            + MpFloat::with_val(vp, &u11i * &t_im);
                        let tr_im = MpFloat::with_val(vp, &u00r * &t_im)
                            + MpFloat::with_val(vp, &u00i * &t_re)
                            - MpFloat::with_val(vp, &u11r * &t_im)
                            + MpFloat::with_val(vp, &u11i * &t_re);
                        let tr_abs = (MpFloat::with_val(vp, &tr_re * &tr_re)
                            + MpFloat::with_val(vp, &tr_im * &tr_im)).sqrt();
                        let q = MpFloat::with_val(vp, 4.0) - MpFloat::with_val(vp, &tr_abs * 2u32);
                        let q8 = MpFloat::with_val(vp, 8.0) - &q;
                        (MpFloat::with_val(vp, &q * &q8) / 16u32)
                            .max(&MpFloat::with_val(vp, 0.0))
                            .sqrt()
                            .to_f64()
                    };
                    eprintln!("assembled U2T dd = {:.3e} (k={})", dd_of(&u2t), u2t.k);
                    let re_evaled = crate::synthesis::near_clifford::eval_gates_t(&g).expect("eval");
                    eprintln!("gates-eval  dd = {:.3e} (k={})", dd_of(&re_evaled), re_evaled.k);
                    // Mismatch M = gates† · u2t: if the decomposer dropped its
                    // Clifford suffix, M is that Clifford exactly.
                    let m = (re_evaled.dagger() * u2t).reduced();
                    eprintln!("mismatch k={} ", m.k);
                    for (nm, c) in crate::synthesis::cliffords::CLIFFORD_TABLE_T {
                        let d = c.diamond_distance(&m);
                        if d < 1e-6 {
                            eprintln!("mismatch is Clifford {nm} (d={d:.2e})");
                        }
                    }
                    return;
                }
                LevelOutcome::Fatal => {
                    eprintln!("FATAL at k={k} has_phase={has_phase}");
                    return;
                }
                LevelOutcome::Empty => {}
            }
            if k >= 2 {
                has_phase = !has_phase;
                if has_phase { k += 1; }
            } else if k == 0 && !has_phase { k = 1; }
            else if k == 1 && !has_phase { k = 0; has_phase = true; }
            else if k == 0 && has_phase { k = 1; }
            else { k = 2; has_phase = false; }
        }
    }

    /// Benchmark battery: single native calls (no ladder) across a grid of
    /// angles × ε. Prints per-case time, a total, and an order-sensitive
    /// hash of all gate strings — the regression guard for optimization
    /// passes (any behavior change shows up as a hash change).
    /// Run: `cargo test --release --lib bench_gridsynth_battery -- --ignored --nocapture`
    #[test]
    #[ignore = "bench probe, print-only"]
    fn bench_gridsynth_battery() {
        let angles: Vec<(String, f64)> = {
            let mut v: Vec<(String, f64)> = Vec::new();
            // Generic angles (fixed pseudo-random).
            let mut s = 0x1234_5678_u64;
            for i in 0..6 {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                #[allow(clippy::cast_precision_loss)]
                let x = (s >> 11) as f64 / (1u64 << 53) as f64 * std::f64::consts::TAU;
                v.push((format!("rand{i}"), x));
            }
            // QFT ladder.
            for k in [4i32, 10, 16, 23, 30] {
                v.push((format!("pi/2^{k}"), std::f64::consts::PI * 2f64.powi(-k)));
            }
            // Near-Clifford ring.
            for (name, d) in [("d=2e-8", 2e-8), ("d=3.7e-7", 3.74507e-7), ("d=1e-5", 1e-5)] {
                v.push((name.into(), d));
                v.push((format!("S+{name}"), std::f64::consts::FRAC_PI_2 + d));
            }
            v
        };
        let mut hash: u64 = 0xcbf29ce484222325;
        let mut total = std::time::Duration::ZERO;
        let mut rows: Vec<(String, f64, f64)> = Vec::new();
        for eps in [1e-4_f64, 1e-6, 1e-8, 1e-10] {
            let prec = crate::synthesis::clifford_t::rz::ladder::prec_for_epsilon(eps);
            for (name, th) in &angles {
                if *th < eps {
                    continue;
                }
                let theta = MpFloat::with_val(prec, *th);
                let t0 = std::time::Instant::now();
                let g = gridsynth_gates_native(&theta, eps, prec).expect("must solve");
                let dt = t0.elapsed();
                total += dt;
                for b in g.bytes() {
                    hash = (hash ^ u64::from(b)).wrapping_mul(0x100000001b3);
                }
                rows.push((format!("{name}@{eps:.0e}"), dt.as_secs_f64() * 1e3, {
                    #[allow(clippy::cast_precision_loss)]
                    let t = g.matches('T').count() as f64;
                    t
                }));
            }
        }
        rows.sort_by(|a, b| b.1.total_cmp(&a.1));
        eprintln!("slowest 12:");
        for (name, ms, t) in rows.iter().take(12) {
            eprintln!("  {name:>18}  {ms:8.2}ms  T={t}");
        }
        eprintln!("cases={} total={:.1}ms hash={hash:016x}", rows.len(), total.as_secs_f64() * 1e3);
    }

    /// Depth limit: time single native calls at ever-deeper ε until a
    /// call exceeds 1 s or the assembly's i128 import overflows.
    /// Run: `cargo test --release --lib bench_gridsynth_depth_limit -- --ignored --nocapture`
    #[test]
    #[ignore = "bench probe, print-only"]
    fn bench_gridsynth_depth_limit() {
        let mut bail = false;
        for decades in [14u32, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52] {
            if bail {
                break;
            }
            let eps = 10f64.powi(-i32::try_from(decades).expect("small"));
            let prec = crate::synthesis::clifford_t::rz::ladder::prec_for_epsilon(eps);
            for (name, th) in [("generic", 1.0472_f64), ("near-id", 3.74507e-7)] {
                let theta = MpFloat::with_val(prec, th);
                let t0 = std::time::Instant::now();
                let g = gridsynth_gates_native(&theta, eps, prec);
                let dt = t0.elapsed();
                match g {
                    Some(g) => {
                        let t_count = g.chars().filter(|&c| c == 'T' || c == 't').count();
                        // High-precision phase-invariant check at prec+128:
                        // D² = q(8−q)/16 with q = 4 − 2|tr(U·Rz(θ)†)|, the
                        // library's diamond formula on exact coefficients.
                        let vp = prec + 128;
                        let u = crate::synthesis::near_clifford::eval_gates_t(&g)
                            .expect("evaluable");
                        let theta_v = MpFloat::with_val(vp, th);
                        let half = MpFloat::with_val(vp, &theta_v / 2u32);
                        let (t_re, t_im) = (half.clone().cos(), half.sin());
                        let sc = crate::rings::dyadic::sqrt2_pow_f(u.k, vp);
                        let inv_sqrt2 = MpFloat::with_val(vp, 0.5).sqrt();
                        // Exact I256 → MPFR via decimal strings (f64 would
                        // truncate the 200+-bit coefficients at deep ε).
                        let exact = |x| {
                            let i = Integer::from_str_radix(&format!("{x}"), 10)
                                .expect("decimal I256");
                            MpFloat::with_val(vp, &i)
                        };
                        let entry = |z: &ZOmega| {
                            let (a, b, c, d) = (exact(z.a), exact(z.b), exact(z.c), exact(z.d));
                            let re = (a + MpFloat::with_val(vp, &b - &d) * &inv_sqrt2) / &sc;
                            let im = (c + MpFloat::with_val(vp, &b + &d) * &inv_sqrt2) / &sc;
                            (re, im)
                        };
                        let (u00r, u00i) = entry(&u.u11);
                        let (u11r, u11i) = entry(&u.u22);
                        // tr(U·T†) = u00·e^{−iθ/2}∗ + u11·e^{iθ/2}∗
                        //          = u00·(cos+i·sin) + u11·(cos−i·sin).
                        let tr_re = MpFloat::with_val(vp, &u00r * &t_re)
                            - MpFloat::with_val(vp, &u00i * &t_im)
                            + MpFloat::with_val(vp, &u11r * &t_re)
                            + MpFloat::with_val(vp, &u11i * &t_im);
                        let tr_im = MpFloat::with_val(vp, &u00r * &t_im)
                            + MpFloat::with_val(vp, &u00i * &t_re)
                            - MpFloat::with_val(vp, &u11r * &t_im)
                            + MpFloat::with_val(vp, &u11i * &t_re);
                        let tr_abs = (MpFloat::with_val(vp, &tr_re * &tr_re)
                            + MpFloat::with_val(vp, &tr_im * &tr_im))
                            .sqrt();
                        let q = MpFloat::with_val(vp, 4.0) - MpFloat::with_val(vp, &tr_abs * 2u32);
                        let q8 = MpFloat::with_val(vp, 8.0) - &q;
                        let dist = (MpFloat::with_val(vp, &q * &q8) / 16u32)
                            .max(&MpFloat::with_val(vp, 0.0))
                            .sqrt()
                            .to_f64();
                        let ok = if dist < eps { "ok " } else { "BAD" };
                        eprintln!(
                            "eps=1e-{decades:<3} prec={prec:<5} {name:8} T={t_count:<5} dd={dist:9.2e} {ok} {dt:>9.2?}"
                        );
                        if dist >= eps {
                            bail = true;
                        }
                    }
                    None => {
                        eprintln!("eps=1e-{decades:<3} prec={prec:<5} {name:8} FAILED {dt:>9.2?}");
                        bail = true;
                    }
                }
                if dt > std::time::Duration::from_millis(1500) {
                    bail = true;
                }
            }
        }
    }

    /// Deep-ε scaling + parallel-throughput checks (print-only).
    /// Run: `cargo test --release --lib bench_gridsynth_scaling -- --ignored --nocapture`
    #[test]
    #[ignore = "bench probe, print-only"]
    fn bench_gridsynth_scaling() {
        // Deep-ε scaling: single calls, one generic + one near-Clifford angle.
        for eps in [1e-6_f64, 1e-8, 1e-10, 1e-12, 1e-14] {
            let prec = crate::synthesis::clifford_t::rz::ladder::prec_for_epsilon(eps);
            for (name, th) in [("generic", 1.0472_f64), ("near-id", 3.74507e-7)] {
                let theta = MpFloat::with_val(prec, th);
                let t0 = std::time::Instant::now();
                let g = gridsynth_gates_native(&theta, eps, prec).expect("solve");
                eprintln!(
                    "eps={eps:<8.0e} {name:8} T={:<4} {:>8.2?}",
                    g.matches('T').count(),
                    t0.elapsed()
                );
            }
        }
        // Parallel throughput: 8 threads × 40 calls each, no shared state.
        let t0 = std::time::Instant::now();
        std::thread::scope(|scope| {
            for tid in 0..8u64 {
                scope.spawn(move || {
                    let eps = 1e-8_f64;
                    let prec = crate::synthesis::clifford_t::rz::ladder::prec_for_epsilon(eps);
                    for i in 0..40u64 {
                        #[allow(clippy::cast_precision_loss)]
                        let th = 0.1 + (tid * 40 + i) as f64 * 0.007;
                        let theta = MpFloat::with_val(prec, th);
                        let _ = gridsynth_gates_native(&theta, eps, prec).expect("solve");
                    }
                });
            }
        });
        let dt = t0.elapsed();
        eprintln!(
            "parallel: 320 calls on 8 threads in {dt:.2?} ({:.2}ms/call wall)",
            dt.as_secs_f64() * 1e3 / 320.0
        );
    }

    /// Phase attribution over the bench battery: where does the time go
    /// (to_upright | TDGP enumeration | Diophantine | assembly/decompose)?
    /// Run: `cargo test --release --lib bench_gridsynth_phases -- --ignored --nocapture`
    #[test]
    #[ignore = "bench probe, print-only"]
    #[allow(clippy::too_many_lines)]
    fn bench_gridsynth_phases() {
        use std::time::{Duration, Instant};
        let mut t_upright = Duration::ZERO;
        let mut t_enum = Duration::ZERO;
        let mut t_dio = Duration::ZERO;
        let mut t_asm = Duration::ZERO;
        let mut n_dio = 0u64;
        let mut n_cand = 0u64;
        let mut total = Duration::ZERO;

        let angles: Vec<f64> = {
            let mut v = vec![1.0472, 0.1234, 2.9876];
            for k in [4i32, 10, 23, 30] {
                v.push(std::f64::consts::PI * 2f64.powi(-k));
            }
            v.push(2e-8);
            v.push(3.74507e-7);
            v.push(std::f64::consts::FRAC_PI_2 + 3.74507e-7);
            v
        };
        for eps in [1e-4_f64, 1e-6, 1e-8, 1e-10] {
            let prec = crate::synthesis::clifford_t::rz::ladder::prec_for_epsilon(eps);
            for &th in &angles {
                if th < eps {
                    continue;
                }
                let t_call = Instant::now();
                let theta = MpFloat::with_val(prec, th);
                let eps_f = MpFloat::with_val(prec, eps);
                let mut rng = Rng::new(GS_SEED);
                let region0 = EpsilonRegion::new(&theta, &eps_f, ZRootTwo::one(), prec);
                let disk0 = UnitDisk::new(ZRootTwo::one(), prec);
                let region1 = EpsilonRegion::new(&theta, &eps_f, ZRootTwo::from_i64(2, 1), prec);
                let disk1 = UnitDisk::new(ZRootTwo::from_i64(2, -1), prec);
                let t0 = Instant::now();
                let op_g = to_upright_ellipse_pair(region0.ellipse(), disk0.ellipse(), prec);
                let (g0, _, _, ba0, bb0) = to_upright_set_pair(
                    region0.ellipse(), disk0.ellipse(), Some(op_g.clone()), prec);
                let (g1, _, _, ba1, bb1) = to_upright_set_pair(
                    region1.ellipse(), disk1.ellipse(), Some(op_g), prec);
                t_upright += t0.elapsed();

                let mut k: u32 = 0;
                let mut has_phase = false;
                let mut done = false;
                while !done {
                    let (sa, sb, g, ba, bb): (&dyn ConvexSet, &dyn ConvexSet, _, _, _) =
                        if has_phase {
                            (&region1, &disk1, &g1, &ba1, &bb1)
                        } else {
                            (&region0, &disk0, &g0, &ba0, &bb0)
                        };
                    let t1 = Instant::now();
                    let mut inner_dio = Duration::ZERO;
                    let mut inner_asm = Duration::ZERO;
                    {
                        let rng = &mut rng;
                        let mut visit = |mut z: DOmega| -> bool {
                            n_cand += 1;
                            if z.mul(&z.conj()).residue() == 0 {
                                return false;
                            }
                            if has_phase {
                                z = z.mul(&DOmega::new(ZOmegaBig::from_i64(0, 1, -1, 0), 1));
                            }
                            let xi = DRootTwo::from_int_i64(1)
                                .sub(&DRootTwo::from_domega(&z.conj().mul(&z)));
                            let mut budget = Budget::default();
                            n_dio += 1;
                            let td = Instant::now();
                            let w = diophantine_dyadic(&xi, rng, &mut budget);
                            inner_dio += td.elapsed();
                            let Some(w) = w else { return false };
                            let ta = Instant::now();
                            let mut z = z.reduce_denomexp();
                            let mut w = w.reduce_denomexp();
                            let kc = z.k.max(w.k);
                            z = z.renew_denomexp(kc);
                            w = w.renew_denomexp(kc);
                            let k1 = z.add(&w).reduce_denomexp().k;
                            let k2 = z.add(&w.mul_by_omega()).reduce_denomexp().k;
                            let k3 = z.add(&w.mul_by_omega_inv()).reduce_denomexp().k;
                            let (w_sel, n) = if has_phase {
                                if k1 <= k2 && k1 <= k3 { (w, -1) } else { (w.mul_by_omega_inv(), -1) }
                            } else if k1 <= k2 {
                                (w, 0)
                            } else {
                                (w.mul_by_omega(), 0)
                            };
                            let ok = assemble_gates(&z, &w_sel, n).is_some();
                            inner_asm += ta.elapsed();
                            ok
                        };
                        solve_tdgp_visit(sa, sb, g, ba, bb, k, prec, &mut visit);
                    }
                    let level = t1.elapsed();
                    t_enum += level.saturating_sub(inner_dio + inner_asm);
                    t_dio += inner_dio;
                    t_asm += inner_asm;
                    done = inner_asm > Duration::ZERO && {
                        // found ⟺ assemble ran and returned Some (visit stops)
                        true
                    };
                    if !done {
                        if k >= 2 {
                            has_phase = !has_phase;
                            if has_phase { k += 1; }
                        } else if k == 0 && !has_phase { k = 1; }
                        else if k == 1 && !has_phase { k = 0; has_phase = true; }
                        else if k == 0 && has_phase { k = 1; }
                        else { k = 2; has_phase = false; }
                    }
                }
                total += t_call.elapsed();
            }
        }
        let pct = |d: Duration| 100.0 * d.as_secs_f64() / total.as_secs_f64();
        eprintln!("total          {:>9.1?}", total);
        eprintln!("  to_upright   {:>9.1?}  {:>5.1}%", t_upright, pct(t_upright));
        eprintln!("  tdgp enum    {:>9.1?}  {:>5.1}%", t_enum, pct(t_enum));
        eprintln!("  diophantine  {:>9.1?}  {:>5.1}%  ({n_dio} calls)", t_dio, pct(t_dio));
        eprintln!("  assemble     {:>9.1?}  {:>5.1}%", t_asm, pct(t_asm));
        eprintln!("  candidates visited: {n_cand}");
    }

    /// Driver-shaped profile: per (k, phase) candidate/diophantine counts
    /// and timing for the adversarial θ=2e-8 @ ε=1e-8 case (print-only).
    /// Run: `cargo test --release --lib gridsynth_driver_profile -- --ignored --nocapture`
    #[test]
    #[ignore = "profiling probe, print-only"]
    fn gridsynth_driver_profile() {
        let epsilon = 1e-8_f64;
        let prec = 160;
        let theta = MpFloat::with_val(prec, 2e-8);
        let eps = MpFloat::with_val(prec, epsilon);
        let mut rng = Rng::new(GS_SEED);
        let region0 = EpsilonRegion::new(&theta, &eps, ZRootTwo::one(), prec);
        let disk0 = UnitDisk::new(ZRootTwo::one(), prec);
        let region1 = EpsilonRegion::new(&theta, &eps, ZRootTwo::from_i64(2, 1), prec);
        let disk1 = UnitDisk::new(ZRootTwo::from_i64(2, -1), prec);
        let op_g = to_upright_ellipse_pair(region0.ellipse(), disk0.ellipse(), prec);
        let (g0, _, _, ba0, bb0) =
            to_upright_set_pair(region0.ellipse(), disk0.ellipse(), Some(op_g.clone()), prec);
        let (g1, _, _, ba1, bb1) =
            to_upright_set_pair(region1.ellipse(), disk1.ellipse(), Some(op_g), prec);
        let mut k: u32 = 0;
        let mut has_phase = false;
        for _step in 0..120 {
            let t0 = std::time::Instant::now();
            let mut cands = 0u64;
            let mut dio_calls = 0u64;
            let mut dio_time = std::time::Duration::ZERO;
            let mut found = false;
            {
                let rng = &mut rng;
                let mut visit = |mut z: DOmega| -> bool {
                    cands += 1;
                    if z.mul(&z.conj()).residue() == 0 {
                        return false;
                    }
                    if has_phase {
                        z = z.mul(&DOmega::new(ZOmegaBig::from_i64(0, 1, -1, 0), 1));
                    }
                    let xi = DRootTwo::from_int_i64(1)
                        .sub(&DRootTwo::from_domega(&z.conj().mul(&z)));
                    let mut budget = Budget::default();
                    dio_calls += 1;
                    let td = std::time::Instant::now();
                    let w = diophantine_dyadic(&xi, rng, &mut budget);
                    dio_time += td.elapsed();
                    if w.is_some() {
                        found = true;
                        return true;
                    }
                    false
                };
                if has_phase {
                    solve_tdgp_visit(&region1, &disk1, &g1, &ba1, &bb1, k, prec, &mut visit);
                } else {
                    solve_tdgp_visit(&region0, &disk0, &g0, &ba0, &bb0, k, prec, &mut visit);
                }
            }
            eprintln!(
                "k={k:>2} phase={} cands={cands:>6} dio={dio_calls:>6} dio_t={dio_time:>9.2?} total={:>9.2?} found={found}",
                u8::from(has_phase),
                t0.elapsed()
            );
            if found {
                break;
            }
            if k >= 2 {
                has_phase = !has_phase;
                if has_phase {
                    k += 1;
                }
            } else if k == 0 && !has_phase {
                k = 1;
            } else if k == 1 && !has_phase {
                k = 0;
                has_phase = true;
            } else if k == 0 && has_phase {
                k = 1;
            } else {
                k = 2;
                has_phase = false;
            }
        }
    }


    /// Print our upright bbox widths for direct comparison with pygridsynth.
    /// Run: `cargo test --release --lib gridsynth_bbox_probe -- --ignored --nocapture`
    #[test]
    #[ignore = "profiling probe, print-only"]
    fn gridsynth_bbox_probe() {
        let prec = 160;
        let theta = MpFloat::with_val(prec, 2e-8);
        let eps = MpFloat::with_val(prec, 1e-8_f64);
        let region0 = EpsilonRegion::new(&theta, &eps, ZRootTwo::one(), prec);
        let disk0 = UnitDisk::new(ZRootTwo::one(), prec);
        let op_g = to_upright_ellipse_pair(region0.ellipse(), disk0.ellipse(), prec);
        let (_g0, ea, eb, ba, bb) =
            to_upright_set_pair(region0.ellipse(), disk0.ellipse(), Some(op_g), prec);
        eprintln!(
            "bboxA widths: x={:.6e} y={:.6e}",
            ba.ix.width().to_f64(),
            ba.iy.width().to_f64()
        );
        eprintln!(
            "bboxB widths: x={:.6e} y={:.6e}",
            bb.ix.width().to_f64(),
            bb.iy.width().to_f64()
        );
        eprintln!(
            "elA: a={:.3e} b={:.3e} d={:.3e}  elB: a={:.3e} b={:.3e} d={:.3e}",
            ea.a.to_f64(), ea.b.to_f64(), ea.d.to_f64(),
            eb.a.to_f64(), eb.b.to_f64(), eb.d.to_f64()
        );
        for k in [48u32, 50] {
            let fa = MpFloat::with_val(prec, &ba.iy.width() * &MpFloat::with_val(prec, 1e-4));
            let fb = MpFloat::with_val(prec, &bb.iy.width() * &MpFloat::with_val(prec, 1e-4));
            let t0 = std::time::Instant::now();
            let ys = grid_solver::solve_scaled_odgp(&ba.iy.fatten(&fa), &bb.iy.fatten(&fb), k + 1, prec);
            eprintln!("k={k}: our n_beta = {} in {:.2?}", ys.len(), t0.elapsed());
        }
    }

    /// Per-level cost profile for a near-Clifford angle (print-only).
    /// Run: `cargo test --release --lib gridsynth_klevel_profile -- --ignored --nocapture`
    #[test]
    #[ignore = "profiling probe, print-only"]
    fn gridsynth_klevel_profile() {
        let epsilon = 1e-8;
        let prec = 160;
        let theta = MpFloat::with_val(prec, 3.74507e-7);
        let eps = MpFloat::with_val(prec, epsilon);
        let region0 = EpsilonRegion::new(&theta, &eps, ZRootTwo::one(), prec);
        let disk0 = UnitDisk::new(ZRootTwo::one(), prec);
        let op_g = to_upright_ellipse_pair(region0.ellipse(), disk0.ellipse(), prec);
        let (g0, _a, _b, bbox_a, bbox_b) =
            to_upright_set_pair(region0.ellipse(), disk0.ellipse(), Some(op_g), prec);
        for k in 0..50u32 {
            let t0 = std::time::Instant::now();
            let mut count = 0u64;
            let mut visit = |_z: DOmega| -> bool {
                count += 1;
                false
            };
            solve_tdgp_visit(&region0, &disk0, &g0, &bbox_a, &bbox_b, k, prec, &mut visit);
            eprintln!("k={k:>2}  cands={count:>4}  {:>9.2?}", t0.elapsed());
        }
    }
}
