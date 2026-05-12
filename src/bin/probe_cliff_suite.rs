//! Cliff-fix correctness suite + performance profile. Runs targets serially
//! with CYCLOSYNTH_TRACE=1 enabled (sets the existing per-phase counters in
//! `synthesis::diag`) and prints a full timing breakdown: build_q, LLL,
//! Cholesky, LU, SE, leaf-check, and dd-verify. Used to identify remaining
//! bottlenecks after the rug-128 → inline-dd verify swap.

use cyclosynth::synthesis::clifford_sqrt_t::SynthesizerQ;
use cyclosynth::synthesis::diag;
use cyclosynth::synthesis::lattice_zeta::set_verify_prune_mpfr;
use num_complex::Complex;
use std::io::Write;
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
    // Per-phase tracing in the SE/LLL/Cholesky/LU stack. Default ON; set
    // CYCLOSYNTH_TRACE=0 in the environment to disable (for A/B perf tests).
    if std::env::var("CYCLOSYNTH_TRACE").is_err() {
        std::env::set_var("CYCLOSYNTH_TRACE", "1");
    }

    // Default to the fastest cliff target for a quick profile.  Override with
    // a comma-separated list of thetas via the first CLI arg, e.g.
    //   cargo run --bin probe_cliff_suite -- 0.3,0.7,1.1
    let args: Vec<String> = std::env::args().skip(1).collect();
    let thetas: Vec<f64> = if let Some(spec) = args.first() {
        spec.split(',').filter_map(|s| s.trim().parse().ok()).collect()
    } else {
        vec![1.1_f64]
    };
    let eps = 1.5e-8_f64;
    println!("=== Cliff-fix profile suite ===");
    println!(
        "ε = {:e}, verify=ON, max_lde=30, n_targets={}, thetas={:?}",
        eps, thetas.len(), thetas
    );
    println!("rayon workers = {}\n", rayon::current_num_threads());

    set_verify_prune_mpfr(true);

    let n_cores = rayon::current_num_threads() as f64;
    let t_total = Instant::now();
    for (i, &theta) in thetas.iter().enumerate() {
        diag::reset_all();

        println!("[{}/{}] theta={} starting...", i + 1, thetas.len(), theta);
        std::io::stdout().flush().ok();

        let target = rz_f64(theta);
        let synth = SynthesizerQ::new(eps).with_max_lde(30);
        let t0 = Instant::now();
        let result = synth.synthesize(target);
        let dt = t0.elapsed().as_secs_f64();

        // Snapshot counters.
        let t_build_ms   = diag::T_BUILD_NS.load(Ordering::Relaxed) as f64 / 1e6;
        let t_lll_ms     = diag::T_LLL_NS.load(Ordering::Relaxed) as f64 / 1e6;
        let t_chol_ms    = diag::T_CHOLESKY_NS.load(Ordering::Relaxed) as f64 / 1e6;
        let t_lu_ms      = diag::T_LU_NS.load(Ordering::Relaxed) as f64 / 1e6;
        let t_se_ms      = diag::T_SE_NS.load(Ordering::Relaxed) as f64 / 1e6;
        let t_leaf_ms    = diag::T_LEAF_CHECK_NS.load(Ordering::Relaxed) as f64 / 1e6;
        let t_verify_ms  = diag::T_VERIFY_DD_NS.load(Ordering::Relaxed) as f64 / 1e6;
        let fires        = diag::N_VERIFY_PRUNE_FIRES.load(Ordering::Relaxed);
        let corrected    = diag::N_VERIFY_PRUNE_CORRECTED.load(Ordering::Relaxed);
        let total_fires  = diag::N_PRUNE_FIRES.load(Ordering::Relaxed);
        let phase1_calls = diag::N_PHASE1_CALLS.load(Ordering::Relaxed);
        let se_leaves    = diag::N_SE_CALLBACKS.load(Ordering::Relaxed);
        let norm_rej     = diag::N_NORM_REJECTED.load(Ordering::Relaxed);
        let bilin_rej    = diag::N_BILINEAR_REJECTED.load(Ordering::Relaxed);
        let align_rej    = diag::N_ALIGN_REJECTED.load(Ordering::Relaxed);
        let sols_ret     = diag::N_SOLS_RETURNED.load(Ordering::Relaxed);

        match &result {
            Some(r) => println!(
                "  → FOUND lde={} dist={:.2e} time={:.1}s",
                r.lde, r.distance, dt
            ),
            None => println!("  → NOT FOUND time={:.1}s", dt),
        }

        // Per-target profile.
        let cpu_total_ms = dt * 1000.0 * n_cores;
        let pct = |x: f64| 100.0 * x / cpu_total_ms.max(1e-9);

        println!("\n  ─── CPU-summed timing (sum across cores) ─────────────────");
        println!("  total CPU-ms (≈ wall × n_cores): {:>10.0}", cpu_total_ms);
        println!("    build_q           {:>10.0} ms  ({:>5.1}%)", t_build_ms, pct(t_build_ms));
        println!("    LLL               {:>10.0} ms  ({:>5.1}%)", t_lll_ms, pct(t_lll_ms));
        println!("    Q-Cholesky        {:>10.0} ms  ({:>5.1}%)", t_chol_ms, pct(t_chol_ms));
        println!("    LU solve          {:>10.0} ms  ({:>5.1}%)", t_lu_ms, pct(t_lu_ms));
        println!("    SE walk           {:>10.0} ms  ({:>5.1}%)", t_se_ms, pct(t_se_ms));
        println!("      ↳ leaf check    {:>10.0} ms  ({:>5.1}%)", t_leaf_ms, pct(t_leaf_ms));
        println!("      ↳ dd verify     {:>10.0} ms  ({:>5.1}%)", t_verify_ms, pct(t_verify_ms));
        let t_se_other = (t_se_ms - t_leaf_ms - t_verify_ms).max(0.0);
        println!("      ↳ other (walk)  {:>10.0} ms  ({:>5.1}%)", t_se_other, pct(t_se_other));

        println!("\n  ─── SE walk counts ───────────────────────────────────────");
        println!("    phase1 calls:        {:>15}", phase1_calls);
        println!("    SE leaves visited:   {:>15}", se_leaves);
        println!("      norm rejected:     {:>15}", norm_rej);
        println!("      bilin rejected:    {:>15}", bilin_rej);
        println!("      align rejected:    {:>15}", align_rej);
        println!("      sols returned:     {:>15}", sols_ret);
        println!("    f64 prune fires:     {:>15}", total_fires);
        println!("    dd verify fires:     {:>15} ({:>5.1}% of f64)",
            fires, 100.0 * fires as f64 / total_fires.max(1) as f64);
        println!("    fn corrections:      {:>15} ({:>5.1}% of dd)",
            corrected, 100.0 * corrected as f64 / fires.max(1) as f64);
        if fires > 0 {
            let ns_per_v = t_verify_ms * 1e6 / fires as f64;
            println!("    avg ns/verify:       {:>15.1}", ns_per_v);
        }
        if total_fires > 0 {
            let ns_per_fire_cpu = t_se_ms * 1e6 / total_fires as f64;
            println!("    avg ns/SE-prune-fire (CPU): {:>10.1}", ns_per_fire_cpu);
        }

        // Per-depth survivorship histogram.
        println!("  ─── SE survivorship by depth ─────────────────────────────");
        println!("    depth | recurse_enter | prune_fires | prune_actual | actual/fires%");
        for d in 0..16 {
            let enter = diag::N_RECURSE_ENTER_AT_DEPTH[d].load(Ordering::Relaxed);
            let fires_d = diag::N_PRUNE_FIRES_AT_DEPTH[d].load(Ordering::Relaxed);
            let actual_d = diag::N_PRUNE_ACTUAL_AT_DEPTH[d].load(Ordering::Relaxed);
            let pct = if fires_d > 0 {
                100.0 * actual_d as f64 / fires_d as f64
            } else { 0.0 };
            if enter > 0 || fires_d > 0 {
                println!(
                    "    {:>5} | {:>13} | {:>11} | {:>12} | {:>11.2}%",
                    d, enter, fires_d, actual_d, pct
                );
            }
        }

        // Distance-to-shell histogram for visited leaves.
        println!("\n  ─── Leaves by shell ratio r = ‖x‖² / 2^k ────────────────");
        let ranges = [
            "r ≤ 0.50      (far below)",
            "0.50 < r ≤ 0.90",
            "0.90 < r ≤ 0.99",
            "0.99 < r < 1.00 (just below)",
            "r == 1.00       (exact shell)",
            "1.00 < r ≤ 1.01 (just above)",
            "1.01 < r ≤ 1.10",
            "r > 1.10        (far above)",
        ];
        let total_leaves: u64 = diag::N_LEAF_BY_SHELL_RATIO.iter()
            .map(|c| c.load(Ordering::Relaxed)).sum();
        for b in 0..diag::N_SHELL_BINS {
            let n = diag::N_LEAF_BY_SHELL_RATIO[b].load(Ordering::Relaxed);
            let pct = if total_leaves > 0 { 100.0 * n as f64 / total_leaves as f64 } else { 0.0 };
            println!("    bin {} {:32} | {:>14} ({:>5.2}%)", b, ranges[b], n, pct);
        }
        println!("    total leaves visited:       {:>14}", total_leaves);

        // 2-D conditioned histogram: depth-1 partial × leaf shell ratio.
        println!("\n  ─── Conditioned: depth-1 partial / T  vs  leaf r ─────────");
        println!("    Rows = partial_eucl_at_depth_0_entry / T  (= depth-1 outgoing partial)");
        println!("    Cols = shell-bins 0..7 (see above)");
        println!("    {:<14}  {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
            "depth-1 p/T",
            "b0 ≤.5", "b1 ≤.9", "b2 ≤.99", "b3 <1.0", "b4 ==1.0", "b5 ≤1.01", "b6 ≤1.10", "b7 >1.10");
        let d1_labels = ["< 0.5", "0.5–0.9", "0.9–0.99", "0.99–1.0"];
        for (i, label) in d1_labels.iter().enumerate() {
            let row: Vec<u64> = (0..8).map(|j| {
                diag::N_LEAF_BY_D1_AND_SHELL[i][j].load(Ordering::Relaxed)
            }).collect();
            print!("    {:<14}", label);
            for v in &row {
                print!("  {:>8}", v);
            }
            println!();
        }
        println!();
        std::io::stdout().flush().ok();
    }
    let total_time = t_total.elapsed().as_secs_f64();
    println!("  total wall: {:.1}s ({:.1} min)", total_time, total_time / 60.0);
}
