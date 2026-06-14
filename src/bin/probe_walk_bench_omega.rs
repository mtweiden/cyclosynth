//! 8D (Z[ω], Clifford+T) counterpart of `probe_walk_bench_zeta`: a shared
//! yardstick for exhaustive lattice-level walks.
//!
//! Runs ONE unbudgeted single-shell enumeration — the operation
//! `lll_aligned_search` performs per inner branch — and reports wall time,
//! SE nodes/leaves, solutions, min cost, CPU utilization, and the per-phase
//! (build/LLL/cholesky/LU/SE) breakdown from the `CYCLOSYNTH_TRACE=1` diag
//! counters.
//!
//! The walk is rebuilt from public APIs: v = normalize4(unitary_to_uv(target)),
//! y = uv_to_lattice_y(v, k), then `find_aligned_lattice_points(..., u64::MAX, ...)`,
//! and reconstruct/score every returned solution. (Supersedes the old
//! `w1_telemetry_8d` probe; the 16D analog is `probe_walk_bench_zeta`.)
//!
//! Args: probe_walk_bench_omega <theta> <eps> <k>

use cyclosynth::synthesis::clifford_t::{solution_to_u2t, unitary_to_uv, uv_to_lattice_y};
use cyclosynth::synthesis::lattice::omega::find_aligned_lattice_points;
use cyclosynth::synthesis::lattice::omega::scratch::IntScratch;
use cyclosynth::synthesis::lattice::omega::brute::normalize4;
use cyclosynth::synthesis::distance::{diamond_distance_float, Mat2};
use cyclosynth::synthesis::decomposer::BlochDecomposer;
use cyclosynth::synthesis::diag;
use num_complex::Complex64;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

// ─── process CPU time (no libc dep: direct libSystem call) ──────────────────

#[repr(C)]
struct Timespec {
    tv_sec: i64,
    tv_nsec: i64,
}

extern "C" {
    fn clock_gettime(clk_id: i32, tp: *mut Timespec) -> i32;
}

/// Darwin `_CLOCK_PROCESS_CPUTIME_ID` (sums CPU time of all threads).
const CLOCK_PROCESS_CPUTIME_ID: i32 = 12;

fn cpu_time_s() -> f64 {
    let mut ts = Timespec { tv_sec: 0, tv_nsec: 0 };
    let rc = unsafe { clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &mut ts) };
    assert_eq!(rc, 0, "clock_gettime(CLOCK_PROCESS_CPUTIME_ID) failed");
    ts.tv_sec as f64 + ts.tv_nsec as f64 * 1e-9
}

/// Half-unit cost 2·T + 7·Q of a decomposed gate string (the default
/// q_cost_x2 = 7); Clifford+T circuits carry no Q, so this is 2·T_count.
fn cost_half_units(gates: &str) -> usize {
    let t = gates.chars().filter(|&c| c == 'T').count();
    let q = gates.chars().filter(|&c| c == 'Q').count();
    2 * t + 7 * q
}

fn rz(theta: f64) -> Mat2 {
    let z = Complex64::new(0.0, 0.0);
    [
        [Complex64::from_polar(1.0, -theta / 2.0), z],
        [z, Complex64::from_polar(1.0, theta / 2.0)],
    ]
}

fn main() {
    // Trace must be on before the first trace_enabled() call (OnceLock).
    if std::env::var("CYCLOSYNTH_TRACE").is_err() {
        std::env::set_var("CYCLOSYNTH_TRACE", "1");
    }
    assert!(diag::trace_enabled(), "build with `--features trace` and run with CYCLOSYNTH_TRACE=1");

    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 3 {
        eprintln!("usage: probe_walk_bench_omega <theta> <eps> <k>");
        std::process::exit(2);
    }
    let theta: f64 = args[0].parse().expect("theta");
    let eps: f64 = args[1].parse().expect("eps");
    let k: u32 = args[2].parse().expect("k");

    let target = rz(theta);
    let v = normalize4(unitary_to_uv(&target)).unwrap_or([1.0, 0.0, 0.0, 0.0]);
    let y = uv_to_lattice_y(v, k);
    let mut scratch = Box::new(IntScratch::new(eps));

    println!(
        "probe_walk_bench_omega: rz({theta}) eps={eps:e} k={k} threads={}",
        rayon::current_num_threads()
    );

    diag::reset_all();
    let budget_hit = AtomicBool::new(false);
    let cpu0 = cpu_time_s();
    let t0 = Instant::now();
    let sols = find_aligned_lattice_points(
        scratch.as_mut(),
        &y,
        k,
        eps,
        usize::MAX,
        u64::MAX,
        u64::MAX,
        &budget_hit,
        None,
    );
    let wall_s = t0.elapsed().as_secs_f64();
    let cpu_s = cpu_time_s() - cpu0;
    let snap = diag::snapshot();
    assert!(
        !budget_hit.load(std::sync::atomic::Ordering::Relaxed),
        "budget hit with u64::MAX budget?!"
    );

    // Reconstruct + score every returned solution (post-walk; timed
    // separately so it doesn't pollute the walk numbers).
    let t1 = Instant::now();
    let mut sols_eps = 0usize;
    let mut min_cost: Option<usize> = None;
    for sol in &sols {
        let cand = solution_to_u2t(sol, k);
        if diamond_distance_float(&cand.to_float(), &target) < eps {
            sols_eps += 1;
            let c = cost_half_units(&BlochDecomposer.decompose(&cand));
            if min_cost.map_or(true, |m| c < m) {
                min_cost = Some(c);
            }
        }
    }
    let check_s = t1.elapsed().as_secs_f64();

    let stage_total =
        snap.t_build_ms + snap.t_lll_ms + snap.t_cholesky_ms + snap.t_lu_ms + snap.t_se_ms;
    let pct = |x: f64| if stage_total > 0.0 { 100.0 * x / stage_total } else { 0.0 };

    println!(
        "wall {:>9.3} s | cpu-util {:>5.2}x | nodes {:>13} | leaves {:>13} | sols {:>7} (eps-close {:>6}) | min cost {:>4} | check {:>6.2} s | phases b/lll/ch/lu/se = {:.1}/{:.1}/{:.1}/{:.1}/{:.1}% (stage cpu {:.0} ms)",
        wall_s,
        if wall_s > 0.0 { cpu_s / wall_s } else { 0.0 },
        snap.se_nodes,
        snap.se_callbacks,
        sols.len(),
        sols_eps,
        min_cost.map(|c| c.to_string()).unwrap_or_else(|| "-".into()),
        check_s,
        pct(snap.t_build_ms),
        pct(snap.t_lll_ms),
        pct(snap.t_cholesky_ms),
        pct(snap.t_lu_ms),
        pct(snap.t_se_ms),
        stage_total,
    );
}
