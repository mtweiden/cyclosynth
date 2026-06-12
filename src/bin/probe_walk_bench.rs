//! W0 (docs/plan_fast_exhaustive_walks.md): shared yardstick for
//! exhaustive lattice-level walks.
//!
//! Runs ONE unbudgeted m = 0 level enumeration — the operation
//! `SynthesizerQ::synthesize_exhaustive_certified` performs per parity branch —
//! and reports wall time, SE nodes/leaves, solutions, min cost, CPU
//! utilization, and the per-phase (build/LLL/cholesky/LU/SE) breakdown
//! from the `CYCLOSYNTH_TRACE=1` diag counters.
//!
//! The walk is rebuilt here from public APIs (the internal helper
//! `direct_lattice_search_at` is private and clifford_sqrt_t.rs is owned by
//! another workstream right now): project the target det onto the ζ₁₆
//! grid, take d = det_phase_of, v = unitary_to_uv_zeta, y =
//! uv_to_lattice_y_zeta(v, k), then `find_aligned_lattice_points_with_stop(..., u64::MAX, ...)`
//! with a no-op stop predicate (cost-min mode never early-exits), and
//! reconstruct/score every returned solution.
//!
//! Args: probe_walk_bench <theta> <eps> <k> [<parity: 0|1>] [<bound_sq>] [<dd: 0|1>]
//!   parity 0 = even branch (target as-is, after det projection)
//!   parity 1 = odd branch (target rotated by e^{iπ/16} first)
//!   omitted  = run both branches.
//!   bound_sq = optional SE bound override (sets CYCLOSYNTH_BOUND_SQ —
//!   convenience for retention sweeps; same effect as the env var).
//!   dd       = optional dd Q-bracket switch; 0 sets
//!   CYCLOSYNTH_QBRACKET_DD=0 (legacy deep-ε mode: f64 factor, no dd
//!   verification — pair with bound 3.0 for the pre-dd reference).

use cyclosynth::matrix::U2Q;
use cyclosynth::synthesis::clifford_sqrt_t::{
    det_phase_of, solution_to_u2q_with_det_phase, unitary_to_uv_zeta,
};
use cyclosynth::synthesis::decomposer::BlochDecomposer;
use cyclosynth::synthesis::diag;
use cyclosynth::synthesis::distance::{diamond_distance_u2q_float, Mat2};
use cyclosynth::synthesis::lattice_zeta::{find_aligned_lattice_points_with_stop, IntScratch16};
use cyclosynth::synthesis::search_zeta::uv_to_lattice_y_zeta;
use num_complex::Complex64;
use std::f64::consts::PI;
use std::sync::atomic::{AtomicBool, Ordering};
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

// ─── target construction (mirrors synthesize_exhaustive_certified, reimplemented
//     locally because project_det_to_zeta_coset is private) ─────────────────

fn rz(theta: f64) -> Mat2 {
    let z = Complex64::new(0.0, 0.0);
    [
        [Complex64::from_polar(1.0, -theta / 2.0), z],
        [z, Complex64::from_polar(1.0, theta / 2.0)],
    ]
}

fn scale(m: &Mat2, g: Complex64) -> Mat2 {
    [
        [m[0][0] * g, m[0][1] * g],
        [m[1][0] * g, m[1][1] * g],
    ]
}

/// Rotate `target` by a global phase so its det lands exactly on the
/// nearest ζ₁₆ power (local copy of the private
/// `clifford_sqrt_t::project_det_to_zeta_coset`; lossless for the
/// diamond distance).
fn project_det_to_zeta_coset(target: &Mat2) -> Mat2 {
    let det = target[0][0] * target[1][1] - target[0][1] * target[1][0];
    let d = det_phase_of(target) as f64;
    let mut residual = det.arg() - d * PI / 8.0;
    while residual > PI {
        residual -= 2.0 * PI;
    }
    while residual <= -PI {
        residual += 2.0 * PI;
    }
    scale(target, Complex64::from_polar(1.0, -residual / 2.0))
}

/// Half-unit cost 2·T + 7·Q of a decomposed gate string (local copy of
/// the private `gates_cost` with the default q_cost_x2 = 7).
fn cost_half_units(gates: &str) -> usize {
    let t = gates.chars().filter(|&c| c == 'T').count();
    let q = gates.chars().filter(|&c| c == 'Q').count();
    2 * t + 7 * q
}

fn nodes_total() -> u64 {
    diag::N_RECURSE_ENTER_AT_DEPTH
        .iter()
        .map(|c| c.load(Ordering::Relaxed))
        .sum()
}

struct RunReport {
    parity: u32,
    wall_s: f64,
    cpu_util: f64,
    nodes: u64,
    leaves: u64,
    sols_raw: usize,
    sols_eps: usize,
    min_cost: Option<usize>,
    check_s: f64,
    // phase shares in % of summed stage time
    build_pct: f64,
    lll_pct: f64,
    chol_pct: f64,
    lu_pct: f64,
    se_pct: f64,
    stage_total_ms: f64,
}

/// One unbudgeted m = 0 level enumeration on the given parity branch.
fn run_branch(target: &Mat2, eps: f64, k: u32, parity: u32, dump: bool) -> RunReport {
    let d = det_phase_of(target);
    let v = unitary_to_uv_zeta(target);
    let y = uv_to_lattice_y_zeta(v, k);

    // Scratch config mirrors SynthesizerQ::new defaults.
    let mut scratch = Box::new(IntScratch16::new(eps));
    scratch.use_f64_gs = eps > 1e-8;
    scratch.bkz_block_size = if eps <= 1e-7 { 4 } else { 0 };

    diag::reset_all();
    let budget_hit = AtomicBool::new(false);
    let cpu0 = cpu_time_s();
    let t0 = Instant::now();
    let sols = find_aligned_lattice_points_with_stop(
        scratch.as_mut(),
        &y,
        k,
        eps,
        u64::MAX,
        &budget_hit,
        |_| false, // cost-min mode: never early-exit — full level walk
        None,
        None,
    );
    let wall_s = t0.elapsed().as_secs_f64();
    let cpu_s = cpu_time_s() - cpu0;
    let snap = diag::snapshot();
    let nodes = nodes_total();
    assert!(
        !budget_hit.load(Ordering::Relaxed),
        "budget hit with u64::MAX budget?!"
    );

    // Reconstruct + score every returned solution (post-walk; timed
    // separately so it doesn't pollute the walk numbers).
    let t1 = Instant::now();
    let mut sols_eps = 0usize;
    let mut min_cost: Option<usize> = None;
    for sol in &sols {
        let cand: U2Q = solution_to_u2q_with_det_phase(sol, k, d).reduced();
        let dist = diamond_distance_u2q_float(&cand, target);
        if dist < eps {
            sols_eps += 1;
            let gates = BlochDecomposer.decompose(&cand);
            let c = cost_half_units(&gates);
            if min_cost.map_or(true, |m| c < m) {
                min_cost = Some(c);
            }
        }
    }
    let check_s = t1.elapsed().as_secs_f64();

    if dump {
        diag::dump_zeta(&snap, &format!("walk parity={parity} k={k} eps={eps:.0e}"));
    }

    let stage_total =
        snap.t_build_ms + snap.t_lll_ms + snap.t_cholesky_ms + snap.t_lu_ms + snap.t_se_ms;
    let pct = |x: f64| if stage_total > 0.0 { 100.0 * x / stage_total } else { 0.0 };

    RunReport {
        parity,
        wall_s,
        cpu_util: if wall_s > 0.0 { cpu_s / wall_s } else { 0.0 },
        nodes,
        leaves: snap.se_callbacks,
        sols_raw: sols.len(),
        sols_eps,
        min_cost,
        check_s,
        build_pct: pct(snap.t_build_ms),
        lll_pct: pct(snap.t_lll_ms),
        chol_pct: pct(snap.t_cholesky_ms),
        lu_pct: pct(snap.t_lu_ms),
        se_pct: pct(snap.t_se_ms),
        stage_total_ms: stage_total,
    }
}

fn print_row(r: &RunReport) {
    println!(
        "p={} | wall {:>9.3} s | cpu-util {:>5.2}x | nodes {:>13} | leaves {:>13} | sols {:>7} (eps-close {:>6}) | min cost {:>4} | check {:>6.2} s | phases b/lll/ch/lu/se = {:.1}/{:.1}/{:.1}/{:.1}/{:.1}% (stage cpu {:.0} ms)",
        r.parity,
        r.wall_s,
        r.cpu_util,
        r.nodes,
        r.leaves,
        r.sols_raw,
        r.sols_eps,
        r.min_cost.map(|c| c.to_string()).unwrap_or_else(|| "-".into()),
        r.check_s,
        r.build_pct,
        r.lll_pct,
        r.chol_pct,
        r.lu_pct,
        r.se_pct,
        r.stage_total_ms,
    );
}

fn main() {
    // Trace must be on before the first trace_enabled() call (OnceLock).
    if std::env::var("CYCLOSYNTH_TRACE").is_err() {
        std::env::set_var("CYCLOSYNTH_TRACE", "1");
    }
    assert!(diag::trace_enabled(), "run with CYCLOSYNTH_TRACE=1");

    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 3 {
        eprintln!("usage: probe_walk_bench <theta> <eps> <k> [<parity: 0|1>] [<bound_sq>]");
        std::process::exit(2);
    }
    let theta: f64 = args[0].parse().expect("theta");
    let eps: f64 = args[1].parse().expect("eps");
    let k: u32 = args[2].parse().expect("k");
    let parity: Option<u32> = args.get(3).map(|s| s.parse().expect("parity"));
    if let Some(bound) = args.get(4) {
        let _: f64 = bound.parse().expect("bound_sq");
        std::env::set_var("CYCLOSYNTH_BOUND_SQ", bound);
    }
    if args.get(5).map(|s| s.as_str()) == Some("0") {
        std::env::set_var("CYCLOSYNTH_QBRACKET_DD", "0");
    }

    let target = project_det_to_zeta_coset(&rz(theta));
    let target_odd = scale(&target, Complex64::from_polar(1.0, PI / 16.0));

    println!(
        "probe_walk_bench: rz({theta}) eps={eps:e} k={k} threads={}",
        rayon::current_num_threads()
    );
    let branches: Vec<u32> = match parity {
        Some(p) => vec![p],
        None => vec![0, 1],
    };
    for p in branches {
        let t = if p == 0 { &target } else { &target_odd };
        let r = run_branch(t, eps, k, p, true);
        print_row(&r);
    }
}
