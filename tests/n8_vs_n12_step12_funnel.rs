//! Steps 1+2 of PROMPT_n8_vs_n12_funnel.md.
//!
//! Side-by-side funnel comparison: identical counters in both rings at the
//! same (ε, k) on the same Haar seeds. Read-only — no algorithm or
//! parameter changes; bullet pruning OFF for n=12 (its pruning would
//! distort the funnel; one ON row reported separately for reference).
//!
//! Counter definitions (consistent across rings):
//!   - `n_nodes`   : SE recursion-tree nodes entered (= budget decrement total)
//!   - `n_leaves`  : leaves reaching the leaf-check callback
//!   - `n_shell`   : leaves passing the integer norm-shell check `‖x‖² = 2^k`
//!   - `n_bullet`  : of those, passing the bullet/bilinear vanishing check
//!                   (n=8: 3 bilinear forms; n=12: 3 bullets √2/√3/√6)
//!   - `n_align`   : of those, passing alignment `(y·x)² ≥ threshold`
//!
//! Asymmetry note: n=12's SE walker uses a depth-0 analytical norm-shell
//! filter (`schnorr_euchner_16d_norm_shell`), so emitted callback leaves
//! are already shell-passing → `n_leaves = n_shell`. n=8 emits all SE
//! leaves and filters at callback time → `n_leaves > n_shell`.
//!
//! GS profile: max/min ratio of the post-LLL r_bar diagonal (the
//! squared Gram-Schmidt norms). For an orthogonal lattice (n=8: ζ₁₆ per-
//! element Gram = 4I) this should be ≈1; for anisotropic (n=12: 4I+2C)
//! it should be ≫ 1 and grow with k/ε.

use cyclosynth::matrix::U2Q;
use cyclosynth::rings::Float;
use cyclosynth::synthesis::clifford_sqrt_t::unitary_to_uv_zeta;
use cyclosynth::synthesis::distance::{diamond_distance_u2q_float, Mat2};
use cyclosynth::synthesis::search_zeta::uv_to_xy_zeta;
use num_complex::Complex64;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::f64::consts::PI;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

fn haar_target(seed: u64) -> Mat2 {
    let mut rng = StdRng::seed_from_u64(seed);
    let theta = rng.random::<f64>() * (2.0 * PI);
    let phi = rng.random::<f64>() * (2.0 * PI);
    let lambda = rng.random::<f64>() * (2.0 * PI);
    let ct = (theta / 2.0).cos();
    let st = (theta / 2.0).sin();
    let global = Complex64::from_polar(1.0, -(phi + lambda) / 2.0);
    [
        [
            global * Complex64::new(ct, 0.0),
            global * (-Complex64::from_polar(st, lambda)),
        ],
        [
            global * Complex64::from_polar(st, phi),
            global * Complex64::from_polar(ct, phi + lambda),
        ],
    ]
}

#[derive(Debug, Clone, Copy)]
struct FunnelRow {
    ring: &'static str,
    eps: f64,
    k: u32,
    seed: u64,
    n_nodes: u64,
    n_leaves: u64,
    n_shell: u64,
    n_bullet: u64,
    n_align: u64,
    n_found: usize,
    budget_hit: bool,
    gs_max_over_min: f64,
    wall_s: f64,
    bound_sq: f64,
}

fn run_n8(eps: f64, k: u32, seed: u64, max_leaves: u64) -> FunnelRow {
    use cyclosynth::synthesis::diag;
    use cyclosynth::synthesis::lattice_zeta::{integer as l_int, scratch as l_scratch};

    // Reset trace counters at run start.
    diag::reset_all();

    let target = haar_target(seed);
    let v = unitary_to_uv_zeta(&target);
    let y = uv_to_xy_zeta(v, k);

    let mut scratch = l_scratch::IntScratch16::new(eps as Float);
    let budget_hit = AtomicBool::new(false);
    let consumed = AtomicU64::new(0);

    let t0 = Instant::now();
    let sols = l_int::phase1_with_stop(
        &mut scratch,
        &y,
        k,
        eps as Float,
        max_leaves,
        &budget_hit,
        |_| false, // collect all
        None,
        Some(&consumed),
    );
    let wall = t0.elapsed().as_secs_f64();

    let n_leaves = diag::N_SE_CALLBACKS.load(Ordering::Relaxed);
    let n_norm_rej = diag::N_NORM_REJECTED.load(Ordering::Relaxed);
    let n_bilin_rej = diag::N_BILINEAR_REJECTED.load(Ordering::Relaxed);
    let n_align_rej = diag::N_ALIGN_REJECTED.load(Ordering::Relaxed);
    let n_sols = diag::N_SOLS_RETURNED.load(Ordering::Relaxed);

    let n_shell = n_leaves.saturating_sub(n_norm_rej);
    let n_bullet = n_shell.saturating_sub(n_bilin_rej);
    let n_align = n_bullet.saturating_sub(n_align_rej);
    // n_align should equal n_sols when not aborted (and we never abort here).
    let _ = n_sols;

    // GS profile from MPFR r_bar diagonal (post-LLL). n=8 ring's bilinear
    // count: bilinear_forms — 3 forms (β_1, β_2, β_3), all must vanish.
    let mut gs = [0.0_f64; 16];
    for i in 0..16 {
        gs[i] = scratch.r_bar[i][i].to_f64();
    }
    let gs_min = gs
        .iter()
        .cloned()
        .filter(|x| *x > 0.0)
        .fold(f64::INFINITY, f64::min);
    let gs_max = gs.iter().cloned().fold(0.0_f64, f64::max);
    let gs_ratio = if gs_min > 0.0 { gs_max / gs_min } else { f64::INFINITY };

    FunnelRow {
        ring: "n=8",
        eps,
        k,
        seed,
        n_nodes: consumed.load(Ordering::Relaxed),
        n_leaves,
        n_shell,
        n_bullet,
        n_align,
        n_found: sols.len(),
        budget_hit: budget_hit.load(Ordering::Relaxed),
        gs_max_over_min: gs_ratio,
        wall_s: wall,
        bound_sq: f64::NAN, // n=8 doesn't expose bound_sq the same way; report NaN
    }
}

fn run_n12(eps: f64, k: u32, seed: u64, max_leaves: u64, prune_on: bool) -> FunnelRow {
    use cyclosynth::synthesis::lattice_upsilon::{integer as l_int, scratch as l_scratch};

    // SAFETY: serial test execution. The integer.rs path reads this env var
    // at phase1 entry and constructs (or skips) the bullet pruning context.
    unsafe {
        std::env::set_var(
            "CYCLOSYNTH_BULLET_PRUNE_N12",
            if prune_on { "1" } else { "0" },
        );
    }

    let target = haar_target(seed);
    let v = [
        target[0][0].re,
        target[0][0].im,
        target[1][0].re,
        target[1][0].im,
    ];

    let mut scratch = l_scratch::IntScratch16::new(eps as Float);
    let budget_hit = AtomicBool::new(false);
    let initial_budget = max_leaves;

    let t0 = Instant::now();
    let (sols, stats) = l_int::phase1_with_stop_stats(
        &mut scratch,
        v,
        k,
        eps as Float,
        max_leaves,
        &budget_hit,
        |_| false,
    );
    let wall = t0.elapsed().as_secs_f64();

    // n=12 SE uses a depth-0 analytical norm-shell filter; emitted callback
    // leaves are already shell-passing.
    let n_leaves = stats.se_leaves as u64;
    let n_shell = stats.pass_norm as u64;
    let n_bullet = stats.pass_bullets as u64;
    let n_align = stats.pass_align as u64;

    // Node count: initial budget - remaining. For norm-shell walker, every
    // recurse-entry decrements the budget. Compare to n=8's consumed counter.
    let n_nodes = initial_budget.saturating_sub(0); // budget is consumed atomically;
    // we can't read it back from `phase1_with_stop_stats` directly. Approximate
    // by leaves visited as a lower bound (better metric not exposed). We mark
    // this clearly in the output.
    let _ = n_nodes;
    // Actually: stats.se_leaves IS the leaf count from the walker; for the
    // node estimate we don't have a direct counter. Report leaves as a proxy.

    let mut gs = [0.0_f64; 16];
    for i in 0..16 {
        gs[i] = scratch.r_bar[i][i].to_f64();
    }
    let gs_min = gs
        .iter()
        .cloned()
        .filter(|x| *x > 0.0)
        .fold(f64::INFINITY, f64::min);
    let gs_max = gs.iter().cloned().fold(0.0_f64, f64::max);
    let gs_ratio = if gs_min > 0.0 { gs_max / gs_min } else { f64::INFINITY };

    // Optional independent distance check on the first found sol.
    let _independent_dist: Option<f64> = sols.first().and_then(|sol| {
        // Reconstruct via the n=12 best-phase path. Not needed for funnel
        // counters — we keep it for follow-on debugging.
        let _ = sol;
        None
    });

    FunnelRow {
        ring: if prune_on { "n=12+P" } else { "n=12" },
        eps,
        k,
        seed,
        n_nodes: n_leaves, // proxy: norm-shell-aware walker; nodes ≈ leaves+overhead
        n_leaves,
        n_shell,
        n_bullet,
        n_align,
        n_found: sols.len(),
        budget_hit: stats.budget_hit,
        gs_max_over_min: gs_ratio,
        wall_s: wall,
        bound_sq: 128.0, // n=12 default at ε ≤ 1e-4 per integer.rs
    }
}

fn fmt_header() -> String {
    format!(
        "{:>6} | {:>6} | {:>2} | {:>4} | {:>10} | {:>10} | {:>10} | {:>9} | {:>11} | {:>8} | {:>6} | {:>10} | {:>5}",
        "ring", "ε", "k", "seed", "n_nodes", "n_leaves", "n_shell", "n_bullet", "bullet-%", "n_align", "found", "GS_max/min", "wall"
    )
}

fn fmt_row(r: &FunnelRow) -> String {
    let bullet_pct = if r.n_shell > 0 {
        (r.n_bullet as f64) / (r.n_shell as f64) * 100.0
    } else {
        0.0
    };
    format!(
        "{:>6} | {:>6.0e} | {:>2} | {:>4} | {:>10} | {:>10} | {:>10} | {:>9} | {:>10.4}% | {:>8} | {:>6} | {:>10.2e} | {:>5.1}",
        r.ring,
        r.eps,
        r.k,
        r.seed,
        r.n_nodes,
        r.n_leaves,
        r.n_shell,
        r.n_bullet,
        bullet_pct,
        r.n_align,
        r.n_found,
        r.gs_max_over_min,
        r.wall_s,
    )
}

#[test]
#[ignore = "Steps 1+2 of PROMPT_n8_vs_n12_funnel: side-by-side funnel"]
fn step12_funnel_table() {
    // Tight budget so the comparison runs in minutes, not hours.
    let max_leaves: u64 = 30_000_000;

    eprintln!(
        "\n[Steps 1+2] n=8 vs n=12 side-by-side funnel — matched (ε, k, seed)\n\
         max_leaves={max_leaves} per call; n=12 bullet pruning OFF for baseline rows\n\
         (one *+P row added per (ε, k) for reference)\n"
    );
    eprintln!("Counter definitions:");
    eprintln!("  n_nodes  : SE recurse entries (n=8: consumed counter; n=12: ≈ leaves)");
    eprintln!("  n_leaves : leaves reaching leaf-callback");
    eprintln!("  n_shell  : leaves passing ‖x‖² = 2^k (n=12: pre-filtered → n_leaves = n_shell)");
    eprintln!("  n_bullet : leaves additionally passing bullet/bilinear vanishing");
    eprintln!("  n_align  : leaves additionally passing the alignment threshold");
    eprintln!("  GS_max/min: max/min ratio of post-LLL r_bar diag (conditioning proxy)\n");

    eprintln!("{}", fmt_header());
    eprintln!("{}", "-".repeat(140));

    let mut rows: Vec<FunnelRow> = Vec::new();
    for &eps in &[1e-4_f64, 1e-5_f64] {
        for &k in &[10_u32, 12, 14, 16] {
            for seed in 0_u64..3 {
                let r8 = run_n8(eps, k, seed, max_leaves);
                eprintln!("{}", fmt_row(&r8));
                rows.push(r8);

                let r12 = run_n12(eps, k, seed, max_leaves, false);
                eprintln!("{}", fmt_row(&r12));
                rows.push(r12);
            }
            // One pruning-ON row per (ε, k) at seed=0 for reference
            let r12p = run_n12(eps, k, 0, max_leaves, true);
            eprintln!("{}", fmt_row(&r12p));
            rows.push(r12p);
            eprintln!();
        }
    }

    // ── Summary ratios for the verdict (Step 3 inputs) ─────────────────
    eprintln!("\n[Steps 1+2] Summary ratios per (ε, k) — averaged over seeds 0..3");
    eprintln!("(bullet-pass fraction = n_bullet / n_shell; useful for the (G) vs (C) bucket)");
    for &eps in &[1e-4_f64, 1e-5_f64] {
        for &k in &[10_u32, 12, 14, 16] {
            let avg = |ring: &str| -> (f64, f64, f64) {
                let bucket: Vec<&FunnelRow> = rows
                    .iter()
                    .filter(|r| r.ring == ring && r.eps == eps && r.k == k)
                    .collect();
                if bucket.is_empty() {
                    return (f64::NAN, f64::NAN, f64::NAN);
                }
                let bp: f64 = bucket
                    .iter()
                    .map(|r| {
                        if r.n_shell > 0 {
                            r.n_bullet as f64 / r.n_shell as f64
                        } else {
                            0.0
                        }
                    })
                    .sum::<f64>()
                    / bucket.len() as f64;
                let gs: f64 = bucket.iter().map(|r| r.gs_max_over_min).sum::<f64>()
                    / bucket.len() as f64;
                let found: f64 = bucket.iter().map(|r| r.n_found as f64).sum::<f64>()
                    / bucket.len() as f64;
                (bp, gs, found)
            };
            let (bp8, gs8, f8) = avg("n=8");
            let (bp12, gs12, f12) = avg("n=12");
            let bp_ratio = bp8 / bp12.max(1e-30);
            let gs_ratio = gs12 / gs8.max(1e-30);
            eprintln!(
                "  ε={eps:.0e} k={k:>2}: n=8 bullet%={:.4e}  n=12 bullet%={:.4e}  ratio(8/12)={:.2e} | GS_n12/GS_n8={:.2e} | found(8)={:.2} found(12)={:.2}",
                bp8 * 100.0, bp12 * 100.0, bp_ratio, gs_ratio, f8, f12,
            );
        }
    }

    // Don't fail the test on any of these — measurement only. The verdict
    // is on the printed table and ratios.
}
