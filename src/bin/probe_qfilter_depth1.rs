//! Diagnostic: measure depth-1 shell-discriminant filter rejection rate.
//!
//! For each z[1] candidate that survives the existing partial_eucl prune
//! (i.e., that would recurse into depth 0), classify by:
//!   D < 0                    — analytical filter would prune (no real z[0])
//!   D ≥ 0, mod-16 says "no"  — D is not a perfect square (no integer z[0])
//!   D ≥ 0, mod-16 says "yes" — could be a perfect square (filter passes)
//!
//! High rejection rate (≥ 80%) means the path-2 filter is structurally
//! powerful and worth the phase-1 budget refactor.
//!
//! Usage: cargo run --release --bin probe_qfilter_depth1 -- <theta> <eps>

use cyclosynth::synthesis::clifford_sqrt_t::SynthesizerQ;
use cyclosynth::synthesis::diag;
use num_complex::Complex;
use std::sync::atomic::Ordering;
use std::time::Instant;

type C64 = Complex<f64>;
type Mat2 = [[C64; 2]; 2];

fn rz_f64(t: f64) -> Mat2 {
    [
        [C64::from_polar(1.0, -t / 2.0), C64::new(0.0, 0.0)],
        [C64::new(0.0, 0.0), C64::from_polar(1.0, t / 2.0)],
    ]
}

fn main() {
    std::env::set_var("CYCLOSYNTH_TRACE", "1");
    let theta: f64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.1);
    let eps: f64 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.5e-8);

    diag::reset_all();
    let target = rz_f64(theta);
    let synth = SynthesizerQ::new(eps).with_max_lde(35);
    let t0 = Instant::now();
    let r = synth.synthesize(target);
    let dt = t0.elapsed().as_secs_f64();

    let total = diag::N_QFILTER_TOTAL.load(Ordering::Relaxed);
    let d_neg = diag::N_QFILTER_D_NEG.load(Ordering::Relaxed);
    let mod16_bad = diag::N_QFILTER_D_GE0_MOD16_BAD.load(Ordering::Relaxed);
    let not_sq = diag::N_QFILTER_D_GE0_NOT_SQUARE.load(Ordering::Relaxed);
    let perfect = diag::N_QFILTER_PERFECT_SQUARE.load(Ordering::Relaxed);

    println!("=== depth-1 Q-filter: theta={} eps={:e} ===", theta, eps);
    match r {
        Some(r) => println!(
            "  FOUND lde={} dist={:.2e} time={:.2}s",
            r.lde, r.distance, dt
        ),
        None => println!("  NOT FOUND time={:.2}s", dt),
    }
    println!("  z[1] candidates measured (= depth-1 recurses to depth 0):");
    println!("    total                              {total:>12}");
    if total > 0 {
        let pct = |x: u64| 100.0 * x as f64 / total as f64;
        println!(
            "    [reject] D < 0                     {d_neg:>12} ({:>5.1}%)",
            pct(d_neg)
        );
        println!(
            "    [reject] D ≥ 0, mod-16 BAD         {mod16_bad:>12} ({:>5.1}%)",
            pct(mod16_bad)
        );
        println!(
            "    [reject] D ≥ 0, isqrt²≠D           {not_sq:>12} ({:>5.1}%)",
            pct(not_sq)
        );
        println!(
            "    [PASS]   D is a perfect square     {perfect:>12} ({:>5.1}%)",
            pct(perfect)
        );
        let reject = d_neg + mod16_bad + not_sq;
        println!(
            "    -- total filter rejections        {reject:>12} ({:>5.1}%)",
            pct(reject)
        );
    }

    // Per-depth profile: enter, prune-fires (f64 check), prune-actual (post
    // verify rescue). At ε ≤ 2e-8 with verify on, fires ≠ actual.
    println!();
    println!("  per-depth enters / prune fires / prune actual:");
    println!("    depth |  n_enter       |  n_fires        |  n_actual       | actual_rate");
    for d in (0..16).rev() {
        let n_e = diag::N_RECURSE_ENTER_AT_DEPTH[d].load(Ordering::Relaxed);
        let n_p = diag::N_PRUNE_FIRES_AT_DEPTH[d].load(Ordering::Relaxed);
        let n_a = diag::N_PRUNE_ACTUAL_AT_DEPTH[d].load(Ordering::Relaxed);
        if n_e == 0 && n_p == 0 && n_a == 0 {
            continue;
        }
        let rate = if n_e > 0 {
            100.0 * n_a as f64 / n_e as f64
        } else {
            0.0
        };
        println!("    {d:>5} | {n_e:>14} | {n_p:>15} | {n_a:>15} | {rate:>6.1}%");
    }

    let n_vf = diag::N_VERIFY_PRUNE_FIRES.load(Ordering::Relaxed);
    let n_vc = diag::N_VERIFY_PRUNE_CORRECTED.load(Ordering::Relaxed);
    if n_vf > 0 {
        println!(
            "    verify rescue rate: {n_vc}/{n_vf} = {:.1}%",
            100.0 * n_vc as f64 / n_vf as f64
        );
    }

    // Leaf-check ratio + mean per-leaf cost (A1)
    let n_d0 = diag::N_RECURSE_ENTER_AT_DEPTH[0].load(Ordering::Relaxed);
    let n_cb = diag::N_SE_CALLBACKS.load(Ordering::Relaxed);
    let n_norm = diag::N_NORM_REJECTED.load(Ordering::Relaxed);
    let n_bil = diag::N_BILINEAR_REJECTED.load(Ordering::Relaxed);
    let n_sols = diag::N_SOLS_RETURNED.load(Ordering::Relaxed);
    let t_leaf_ns = diag::T_LEAF_CHECK_NS.load(Ordering::Relaxed);
    let t_dd_ns = diag::T_VERIFY_DD_NS.load(Ordering::Relaxed);
    println!();
    println!("  depth-0 → leaf accounting:");
    println!("    depth-0 entries:           {n_d0:>14}");
    println!("    leaf_filter calls:         {n_cb:>14}");
    if n_d0 > 0 {
        println!(
            "    leaves per depth-0 entry:  {:>14.2}",
            n_cb as f64 / n_d0 as f64
        );
    }
    println!("    leaf norm-rejected:        {n_norm:>14}");
    println!("    leaf bilinear-rejected:    {n_bil:>14}");
    println!("    solutions returned:        {n_sols:>14}");
    if n_cb > 0 {
        println!(
            "    mean leaf_filter time:   {:>14.1} ns",
            t_leaf_ns as f64 / n_cb as f64
        );
    }
    if n_vf > 0 {
        println!(
            "    mean dd verify time:     {:>14.1} ns",
            t_dd_ns as f64 / n_vf as f64
        );
    }

    // A3: production filter cost
    let t_qpre = diag::T_QFILTER_PRECOMPUTE_NS.load(Ordering::Relaxed);
    let t_qcls = diag::T_QFILTER_CLASSIFY_NS.load(Ordering::Relaxed);
    let n_qpre = diag::N_QFILTER_PRECOMPUTE_CALLS.load(Ordering::Relaxed);
    let n_qcls = diag::N_QFILTER_TOTAL.load(Ordering::Relaxed);
    if n_qpre > 0 {
        println!();
        println!("  qfilter production timing (A3):");
        println!(
            "    mean precompute time:    {:>14.1} ns (n={n_qpre})",
            t_qpre as f64 / n_qpre as f64
        );
        if n_qcls > 0 {
            println!(
                "    mean classify time:      {:>14.1} ns (n={n_qcls})",
                t_qcls as f64 / n_qcls as f64
            );
        }
    }

    // Mechanism 3 discriminator: nodes consumed by first prefix's walker
    // at the moment it returned the solution. Compare filter-on vs filter-
    // off; similar ⇒ post-find drift; higher with filter ⇒ search-order
    // disruption; lower with filter but wall still higher ⇒ per-node cost
    // asymmetry dominates.
    let nodes_at_find = diag::N_NODES_AT_FIRST_SOLUTION.load(Ordering::Relaxed);
    println!();
    println!("  *** nodes at first solution (per-prefix):  {nodes_at_find} ***");
}
