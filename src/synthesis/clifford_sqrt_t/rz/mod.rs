//! Native gridsynth for Clifford+√T z-rotations.
//!
//! The Ross–Selinger structure generalized to Z[ζ₁₆]: the 1D grid
//! problems over Z[√2] become an 8D lattice problem over Z[g]²
//! (g = 2cos π/8), and the Diophantine step becomes a relative norm
//! equation for the CM extension Z[ζ₁₆]/Z[g]. `clifford_t::rz` is the
//! Clifford+T counterpart this mirrors.

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss, clippy::cast_sign_loss)]

pub(crate) mod ladder;
pub(crate) mod grid;
pub(crate) mod norm_eq;

use crate::matrix::U2Q;
use crate::rings::types::MpFloat;
use crate::rings::ZZeta;
use crate::synthesis::decomposer::BlochDecomposer;
use crate::synthesis::factor::{Budget, Rng};
use crate::synthesis::clifford_t::rz::integer_to_i256;

use grid::GridCtx;
use norm_eq::solve_rel_norm;
use crate::rings::zzeta::ZZetaBig;

/// Deterministic seed (mirrors the Clifford+T driver).
const GSQ_SEED: u64 = 0x51C0FFEE;

/// ZZetaBig (rug) → crate ZZeta (I256); None on coefficient overflow.
fn to_crate_zzeta(z: &ZZetaBig) -> Option<ZZeta> {
    let cs = z.coeffs();
    let f = |i: usize| integer_to_i256(cs[i]);
    Some(ZZeta::new(
        f(0)?, f(1)?, f(2)?, f(3)?, f(4)?, f(5)?, f(6)?, f(7)?,
    ))
}

/// Assemble U = [[u, −(wζᵐ)‾], [wζᵐ, ū]]/√2^k and decompose to gates.
/// With `phase_sweep` all 8 legal off-diagonal phases m are decomposed
/// and the cheapest circuit (T + q_cost_x2/2·√T) wins; otherwise m=0.
fn assemble_gates_q(
    u: &ZZetaBig,
    w: &ZZetaBig,
    k: u32,
    phase_sweep: bool,
    q_cost_x2: usize,
) -> Option<String> {
    use crate::synthesis::clifford_sqrt_t::gates_cost;
    // Decomposer SO(3) guard (numerators ~2^k in I256), mirroring the
    // Clifford+T driver's k-wall.
    if k > 250 {
        return None;
    }
    // Unitary completions [[u, −(wζᵐ)‾], [wζᵐ, ū]]/√2^k, det 1: any m
    // approximates the same diagonal target (ζᵐ only rotates the ε-small
    // off-diagonal), so all 8 are legal circuits whose decompositions
    // differ in syllable structure and hence cost. A ζⁿ dressing of the
    // SECOND column is NOT legal — it multiplies det by ζⁿ and shifts
    // the implemented rotation by nπ/8 (the depth probe catches this as
    // dist = sin(nπ/16)). No variant changes the reduced k (units never
    // change √2-divisibility).
    // The −w variants (m = 8..16) yield no additional distinct costs.
    let sweep = if phase_sweep { 8 } else { 1 };
    (0..sweep)
        .filter_map(|m| {
            let w_sel = w.mul_by_zeta_power(m);
            let u12 = w_sel.conj().neg();
            let u22 = u.conj();
            let (a11, a12, a21, a22) = (
                to_crate_zzeta(u)?,
                to_crate_zzeta(&u12)?,
                to_crate_zzeta(&w_sel)?,
                to_crate_zzeta(&u22)?,
            );
            let mat = U2Q::new(a11, a12, a21, a22, k).reduced();
            Some(BlochDecomposer.decompose(&mat))
        })
        .min_by_key(|g| gates_cost(g, q_cost_x2))
}

/// Cost-optimization knobs for the native √T driver. Defaults mirror
/// the 16D pipeline's optimize_cost semantics; probes sweep them.
#[derive(Clone, Copy)]
pub(crate) struct QCostCfg {
    /// Levels searched past the first solvable one.
    pub window: u32,
    /// Candidates attempted per level.
    pub per_level_cap: u32,
    /// Decompose all 8 legal off-diagonal phases ζᵐ, not just m=0.
    pub phase_sweep: bool,
}

impl Default for QCostCfg {
    fn default() -> Self {
        // Cost-first policy (sweep-verified, gridsynth_q_cost_sweep):
        // the m-sweep buys ~5% cost at 1e-5 and per_level_cap 256 ~1.5%
        // more; window 4 / cap 1024 measured no further gain. Runtime is
        // secondary to circuit cost for this route.
        Self { window: 2, per_level_cap: 256, phase_sweep: true }
    }
}

/// Synthesize Rz(θ) over Clifford+√T (up to global phase).
///
/// Finds the first level k with a norm-equation-solvable candidate, then
/// optimizes gate cost (T + q_cost_x2/2·√T) over up to `cfg.per_level_cap`
/// candidates at levels k..k+`cfg.window` — mirroring the 16D pipeline's
/// optimize_cost + lde_window semantics on the native path.
pub(crate) fn gridsynth_q_gates_native(
    theta: &MpFloat,
    epsilon: f64,
    prec: u32,
    q_cost_x2: usize,
) -> Option<String> {
    gridsynth_q_gates_cfg(theta, epsilon, prec, q_cost_x2, QCostCfg::default())
}

pub(crate) fn gridsynth_q_gates_cfg(
    theta: &MpFloat,
    epsilon: f64,
    prec: u32,
    q_cost_x2: usize,
    cfg: QCostCfg,
) -> Option<String> {
    use crate::synthesis::clifford_sqrt_t::gates_cost;
    // Empirically the m-sweep moves a candidate's cost by a few gates;
    // only candidates within this margin of the incumbent are re-swept.
    const SWEEP_MARGIN: usize = 8;
    use rayon::prelude::*;
    let ctx = GridCtx::new(theta, epsilon, prec);
    let k_cap = (2.5 * (1.0 / epsilon).log2()).ceil() as u32 + 24;
    let mut best: Option<(usize, String)> = None;
    let mut k_star: Option<u32> = None;
    // (first-tie cost, u, w, k) contenders for the phase-sweep stage.
    let mut contenders: Vec<(usize, ZZetaBig, ZZetaBig, u32)> = Vec::new();
    for k in 0..=k_cap {
        if let Some(ks) = k_star {
            if k > ks + cfg.window {
                break;
            }
        }
        let mut level_cands: Vec<grid::Candidate> = Vec::new();
        ctx.visit_level(k, &mut |cand| {
            level_cands.push(cand);
            level_cands.len() >= cfg.per_level_cap as usize
        });
        // Norm equation + assembly per candidate, in parallel with a
        // deterministic per-candidate rng stream (index-order collect
        // keeps the min selection reproducible).
        let solved: Vec<(usize, String, ZZetaBig, ZZetaBig)> = level_cands
            .into_par_iter()
            .enumerate()
            .filter_map(|(idx, cand)| {
                let mut rng =
                    Rng::new(GSQ_SEED ^ (u64::from(k) << 32) ^ (idx as u64));
                let mut budget = Budget::default();
                let w = solve_rel_norm(&cand.xi, &mut rng, &mut budget)?;
                let u = ZZetaBig::from_module(&cand.a, &cand.b);
                let g = assemble_gates_q(&u, &w, k, false, q_cost_x2)?;
                Some((gates_cost(&g, q_cost_x2), g, u, w))
            })
            .collect();
        for (cost, g, u, w) in solved {
            if k_star.is_none() {
                k_star = Some(k);
            }
            if best.as_ref().is_none_or(|(bc, _)| cost < *bc) {
                best = Some((cost, g));
            }
            if cfg.phase_sweep {
                contenders.push((cost, u, w, k));
            }
        }
    }
    if cfg.phase_sweep {
        if let Some((best_cost, _)) = &best {
            let cut = best_cost + SWEEP_MARGIN;
            let swept: Vec<(usize, String)> = contenders
                .into_par_iter()
                .filter_map(|(c, u, w, k)| {
                    if c > cut {
                        return None;
                    }
                    let g = assemble_gates_q(&u, &w, k, true, q_cost_x2)?;
                    Some((gates_cost(&g, q_cost_x2), g))
                })
                .collect();
            for (cost, g) in swept {
                if best.as_ref().is_none_or(|(bc, _)| cost < *bc) {
                    best = Some((cost, g));
                }
            }
        }
    }
    best.map(|(_, g)| g)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthesis::distance::Mat2;

    fn rz_target(theta: f64) -> Mat2 {
        [
            [
                num_complex::Complex::from_polar(1.0, -theta / 2.0),
                num_complex::Complex::new(0.0, 0.0),
            ],
            [
                num_complex::Complex::new(0.0, 0.0),
                num_complex::Complex::from_polar(1.0, theta / 2.0),
            ],
        ]
    }

    /// Rejection anatomy at 1e-16: for raw SE hits, how far outside the
    /// cap / disks does each land?
    /// Run: `cargo test --release --lib gridsynth_q_reject_diag -- --ignored --nocapture`
    #[test]
    #[ignore = "diagnostic probe, print-only"]
    fn gridsynth_q_reject_diag() {
        for eps in [1e-14_f64, 1e-16] {
            let prec = crate::synthesis::clifford_t::rz::ladder::prec_for_epsilon(eps);
            let theta = MpFloat::with_val(prec, 0.7_f64);
            let ctx = GridCtx::new(&theta, eps, prec);
            let decades = -eps.log10();
            let k = (0.7 * decades * 3.32) as u32 + 7;
            eprintln!("eps={eps:.0e} k={k}:");
            let stats = ctx.reject_anatomy(k, 6);
            for (i, (cos_margin, xi_ok)) in stats.iter().enumerate() {
                eprintln!(
                    "  hit{i}: cap cos-margin/eps^2R = {cos_margin:+.3e}  xi_tot_nonneg={xi_ok}"
                );
            }
        }
    }

    /// Paired Clifford+T vs Clifford+√T cost comparison: same angles,
    /// same ε, both native Rz routes, both scored as 2·(T + 3·Q).
    /// Run: `cargo test --release --lib gridsynth_t_vs_q_paired -- --ignored --nocapture`
    #[test]
    #[ignore = "bench probe, print-only"]
    fn gridsynth_t_vs_q_paired() {
        use crate::synthesis::clifford_sqrt_t::gates_cost;
        use crate::synthesis::clifford_t::rz::gridsynth_gates_native;
        let angles = [0.7_f64, 1.9, 0.3, 0.196_349_540_849_362_07, 2.6];
        for eps in [
            1e-2_f64, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14, 1e-16, 1e-18,
            1e-24, 1e-32, 1e-40, 1e-48,
        ] {
            let prec = crate::synthesis::clifford_t::rz::ladder::prec_for_epsilon(eps);
            let (mut sum_t, mut sum_q) = (0usize, 0usize);
            let mut line = String::new();
            let t0 = std::time::Instant::now();
            for th in angles {
                let theta = MpFloat::with_val(prec, th);
                let gt = gridsynth_gates_native(&theta, eps, prec).expect("T solves");
                let gq = gridsynth_q_gates_native(&theta, eps, prec, 6).expect("Q solves");
                let (ct, cq) = (gates_cost(&gt, 6), gates_cost(&gq, 6));
                sum_t += ct;
                sum_q += cq;
                line.push_str(&format!("  {ct}/{cq}"));
                let _ = th;
            }
            let dt = t0.elapsed();
            eprintln!(
                "eps=1e-{:<3} T/Q pairs:{line}   sum {sum_t}/{sum_q}  ratio={:.3}  {dt:.2?}",
                -eps.log10() as i32,
                sum_q as f64 / sum_t as f64
            );
        }
    }

    /// Cost-vs-config sweep: what do bigger windows / caps / the (m, n)
    /// phase sweep buy over the default optimizer?
    /// Run: `cargo test --release --lib gridsynth_q_cost_sweep -- --ignored --nocapture`
    #[test]
    #[ignore = "bench probe, print-only"]
    fn gridsynth_q_cost_sweep() {
        use rayon::prelude::*;
        let cfgs: [(&str, QCostCfg); 9] = [
            ("first-hit", QCostCfg { window: 0, per_level_cap: 1, phase_sweep: false }),
            ("default", QCostCfg::default()),
            ("no-phase", QCostCfg { phase_sweep: false, ..QCostCfg::default() }),
            ("w4", QCostCfg { window: 4, ..QCostCfg::default() }),
            ("w8", QCostCfg { window: 8, ..QCostCfg::default() }),
            ("w12", QCostCfg { window: 12, ..QCostCfg::default() }),
            ("cap256", QCostCfg { per_level_cap: 256, ..QCostCfg::default() }),
            ("c256+ph", QCostCfg { per_level_cap: 256, phase_sweep: true, ..QCostCfg::default() }),
            ("max", QCostCfg { window: 4, per_level_cap: 1024, phase_sweep: true }),
        ];
        let angles = [0.7_f64, 1.9, 0.3, 2.51, 1.1];
        for eps in [1e-3_f64, 1e-5] {
            let prec = crate::synthesis::clifford_t::rz::ladder::prec_for_epsilon(eps);
            for (name, cfg) in &cfgs {
                let t0 = std::time::Instant::now();
                let costs: Vec<usize> = angles
                    .par_iter()
                    .map(|&th| {
                        let theta = MpFloat::with_val(prec, th);
                        let g = gridsynth_q_gates_cfg(&theta, eps, prec, 6, *cfg)
                            .expect("solves");
                        crate::synthesis::clifford_sqrt_t::gates_cost(&g, 6)
                    })
                    .collect();
                let dt = t0.elapsed();
                let sum: usize = costs.iter().sum();
                eprintln!("eps={eps:.0e} {name:9} sum={sum:<4} costs={costs:?} {dt:>8.2?}");
            }
        }
    }

    /// Geometry comparison 1e-14 (works) vs 1e-16 (empty).
    /// Run: `cargo test --release --lib gridsynth_q_geo_compare -- --ignored --nocapture`
    #[test]
    #[ignore = "diagnostic probe, print-only"]
    fn gridsynth_q_geo_compare() {
        for eps in [1e-14_f64, 1e-16] {
            let prec = crate::synthesis::clifford_t::rz::ladder::prec_for_epsilon(eps);
            let theta = MpFloat::with_val(prec, 0.7_f64);
            let ctx = GridCtx::new(&theta, eps, prec);
            let decades = -eps.log10();
            let k_lo = (0.7 * decades * 3.32) as u32;
            eprintln!("eps={eps:.0e}:");
            for k in k_lo..(k_lo + 10) {
                let (raw, acc, dmin, dmax, bmax, rmax) = ctx.level_stats(k);
                eprintln!(
                    "  k={k:>2} raw={raw:>7} acc={acc:>3} r_diag=[{dmin:.2e},{dmax:.2e}] bmax={bmax:.2e} resid_max={rmax:.2e}"
                );
            }
        }
    }

    /// Stage-by-stage timing at ε=1e-16 (the hang): reduction | per-level
    /// enumeration | norm equation.
    /// Run: `cargo test --release --lib gridsynth_q_hang_diag -- --ignored --nocapture`
    #[test]
    #[ignore = "diagnostic probe, print-only"]
    fn gridsynth_q_hang_diag() {
        let eps = 1e-16_f64;
        let prec = crate::synthesis::clifford_t::rz::ladder::prec_for_epsilon(eps);
        let theta = MpFloat::with_val(prec, 0.7_f64);
        let t0 = std::time::Instant::now();
        let ctx = GridCtx::new(&theta, eps, prec);
        eprintln!("ctx+reduction: {:?}", t0.elapsed());
        let mut rng = Rng::new(GSQ_SEED);
        for k in 0..=70u32 {
            let t1 = std::time::Instant::now();
            let mut cands: Vec<grid::Candidate> = Vec::new();
            ctx.visit_level(k, &mut |cand| {
                cands.push(cand);
                cands.len() >= 8
            });
            let t_enum = t1.elapsed();
            if cands.is_empty() {
                if t_enum > std::time::Duration::from_millis(50) {
                    eprintln!("k={k:>2} enum={t_enum:>10.2?} cands=0");
                }
                continue;
            }
            eprintln!("k={k:>2} enum={t_enum:>10.2?} cands={}", cands.len());
            for (i, cand) in cands.iter().take(3).enumerate() {
                let t2 = std::time::Instant::now();
                let mut budget = Budget::default();
                let res = solve_rel_norm(&cand.xi, &mut rng, &mut budget);
                eprintln!(
                    "    cand{i}: norm-eq {:?} solved={} norm_bits={}",
                    t2.elapsed(),
                    res.is_some(),
                    cand.xi.norm().significant_bits()
                );
            }
            if k > 46 {
                break;
            }
        }
    }

    /// Depth probe: where does the native √T path stop working?
    /// Run: `cargo test --release --lib gridsynth_q_depth -- --ignored --nocapture`
    #[test]
    #[ignore = "bench probe, print-only"]
    fn gridsynth_q_depth() {
        let mut bail = false;
        for decades in [6u32, 8, 10, 12, 14, 16, 18, 20, 24, 32, 40, 48] {
            if bail {
                break;
            }
            let eps = 10f64.powi(-i32::try_from(decades).expect("small"));
            let prec = crate::synthesis::clifford_t::rz::ladder::prec_for_epsilon(eps);
            for (name, th) in [("generic", 0.7_f64), ("near-id", 3.74507e-7)] {
                if th < eps * 10.0 {
                    continue;
                }
                let theta = MpFloat::with_val(prec, th);
                let t0 = std::time::Instant::now();
                let g = gridsynth_q_gates_native(&theta, eps, prec, 6);
                let dt = t0.elapsed();
                match g {
                    Some(g) => {
                        let u = crate::synthesis::near_clifford::eval_gates_q(&g)
                            .expect("evaluable");
                        let target = [
                            [
                                num_complex::Complex::from_polar(1.0, -th / 2.0),
                                num_complex::Complex::new(0.0, 0.0),
                            ],
                            [
                                num_complex::Complex::new(0.0, 0.0),
                                num_complex::Complex::from_polar(1.0, th / 2.0),
                            ],
                        ];
                        let _ = target;
                        let d = crate::synthesis::clifford_sqrt_t::rz::ladder::exact_rz_distance(
                            &u, &theta, prec,
                        );
                        let ok = if d < eps { "ok " } else { "BAD" };
                        let cost = crate::synthesis::clifford_sqrt_t::gates_cost(&g, 6);
                        eprintln!(
                            "eps=1e-{decades:<2} {name:8} cost={cost:<4} dist={d:9.2e} {ok} {dt:>9.2?}"
                        );
                    }
                    None => {
                        eprintln!("eps=1e-{decades:<2} {name:8} NONE {dt:>9.2?}");
                        bail = true;
                    }
                }
                if dt > std::time::Duration::from_secs(3) {
                    bail = true;
                }
            }
        }
    }

    /// Why NONE at 1e-5: per-level enumeration/acceptance/norm-eq stats.
    /// Run: `cargo test --release --lib gridsynth_q_none_diag -- --ignored --nocapture`
    #[test]
    #[ignore = "diagnostic probe, print-only"]
    fn gridsynth_q_none_diag() {
        let eps = 1e-5_f64;
        let prec = crate::synthesis::clifford_t::rz::ladder::prec_for_epsilon(eps);
        let theta = MpFloat::with_val(prec, 0.7_f64);
        let ctx = GridCtx::new(&theta, eps, prec);
        let mut rng = Rng::new(GSQ_SEED);
        for k in 0..46u32 {
            let mut cands = 0u32;
            let mut solved = 0u32;
            let rng = &mut rng;
            ctx.visit_level(k, &mut |cand| {
                cands += 1;
                let mut budget = Budget::default();
                if solve_rel_norm(&cand.xi, rng, &mut budget).is_some() {
                    solved += 1;
                }
                false
            });
            if cands > 0 || k > 38 {
                eprintln!("k={k:>2} exact-accepted={cands:>4} norm-eq-solved={solved}");
            }
        }
    }

    /// Validation battery: native gridsynth-√T vs the 16D pipeline on
    /// z-rotations — cost (T+3.5Q half-units), distance, wall time.
    /// Run: `cargo test --release --lib bench_gridsynth_q_battery -- --ignored --nocapture`
    #[test]
    #[ignore = "bench probe, print-only"]
    fn bench_gridsynth_q_battery() {
        use crate::synthesis::clifford_sqrt_t::{gates_cost, SynthesizerQ};
        let angles = [0.7_f64, 1.9, 0.3, std::f64::consts::PI / 16.0, 2.6];
        for eps in [1e-2_f64, 1e-3, 1e-4, 1e-5, 1e-6] {
            for &th in &angles {
                let target = rz_target(th);
                let theta = MpFloat::with_val(
                    crate::synthesis::clifford_t::rz::ladder::prec_for_epsilon(eps),
                    th,
                );
                let t0 = std::time::Instant::now();
                let native = crate::synthesis::clifford_sqrt_t::rz::ladder::synthesize_rz_q(
                    &theta, &target, eps, 6,
                );
                let t_native = t0.elapsed();
                let t1 = std::time::Instant::now();
                let pipe = SynthesizerQ::new(eps).synthesize(target);
                let t_pipe = t1.elapsed();
                let fmt = |gates: &Option<String>| -> String {
                    gates.as_ref().map_or("—".into(), |g| {
                        format!(
                            "cost={:<3} T={:<2} Q={:<2}",
                            gates_cost(g, 6),
                            g.chars().filter(|&c| c == 'T' || c == 't').count(),
                            g.chars().filter(|&c| c == 'Q' || c == 'q').count()
                        )
                    })
                };
                let ns = native.as_ref().map_or("NONE".into(), |r| {
                    format!("{} dist={:.1e} {:>8.1?}", fmt(&r.gates), r.distance, t_native)
                });
                let ps = pipe.as_ref().map_or("NONE".into(), |r| {
                    format!("{} dist={:.1e} {:>8.1?}", fmt(&r.gates), r.distance, t_pipe)
                });
                eprintln!("th={th:<6} eps={eps:<8.0e} native: {ns}   pipe: {ps}");
            }
        }
    }
}
