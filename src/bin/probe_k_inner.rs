//! Probe the lattice path (LLL+SE) for n=6 at k=12, 15, 18 to locate the
//! SE performance cliff.
//!
//! For each k, constructs a deterministic (u, t) ∈ Z[ξ]² with
//! α(u)+α(t) = 2^k and β(u)+β(t) = 0, derives the alignment vector y,
//! and calls lattice_omicron::phase1 with no budget cap.
//!
//! Run with CYCLOSYNTH_TRACE=1 to get per-stage timing breakdowns AND the
//! [SE-PREP] partial[0] line (if integer.rs has the diagnostic enabled
//! for this k).
//!
//!   CYCLOSYNTH_TRACE=1 cargo run --release --bin probe_k_inner 2>&1 | tee /tmp/k_inner_probe.txt
//!
//! Expected (u, t) constructions:
//!   k=12: u=(32,0,32,0), t=(32,0,0,0)  →  α(u)=3·32²=3072, α(t)=1024, sum=4096=2^12
//!   k=15: u=(128,0,0,0), t=(128,0,0,0) →  α(u)=16384,       α(t)=16384, sum=32768=2^15
//!   k=18: u=(256,0,256,0),t=(256,0,0,0)→  α(u)=3·256²=196608,α(t)=65536,sum=262144=2^18

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use cyclosynth::synthesis::clifford_pi6::{
    check_bilinear, check_norm_eq, compute_y, solution_to_mat2, unitary_to_uv_n6,
};
use cyclosynth::synthesis::lattice_omicron;

// Hard-cap per-probe wall time.  If lattice_omicron::phase1 returns inside
// this window we report the real elapsed time; otherwise we'd need a separate
// thread (not done here — we rely on cargo timeout or manual Ctrl-C).
const TIMEOUT_MS: u128 = 60_000;

/// Deterministic expected_x for each probe k.  These are used BOTH here and
/// in the [SE-PREP] block inside lattice_omicron/integer.rs, so they must
/// match exactly.
fn expected_x_for_k(k: u32) -> [i64; 8] {
    match k {
        // α(u) = 32²+32²+32·32 = 3072, α(t) = 32² = 1024, sum = 4096 = 2^12
        12 => [32, 0, 32, 0, 32, 0, 0, 0],
        // α(u) = 128² = 16384,          α(t) = 128² = 16384, sum = 32768 = 2^15
        15 => [128, 0, 0, 0, 128, 0, 0, 0],
        // α(u) = 3·256² = 196608,       α(t) = 256² = 65536, sum = 262144 = 2^18
        18 => [256, 0, 256, 0, 256, 0, 0, 0],
        _ => panic!("no expected_x defined for k={k}"),
    }
}

struct ProbeResult {
    elapsed_ms: f64,
    n_sols: usize,
    found_expected: bool,
    budget_hit: bool,
}

fn run_probe(k: u32, eps: f64) -> ProbeResult {
    let expected_x = expected_x_for_k(k);

    // Sanity-check the construction (cheap, done once at startup).
    assert!(
        check_norm_eq(&expected_x, k),
        "expected_x={expected_x:?} fails norm check at k={k}"
    );
    assert!(
        check_bilinear(&expected_x),
        "expected_x={expected_x:?} fails bilinear check at k={k}"
    );

    // Derive the alignment vector y so the cap center points at expected_x.
    // Pipeline: expected_x → float SU(2) matrix → uv direction → y.
    let mat = solution_to_mat2(&expected_x, k);
    let uv = unitary_to_uv_n6(&mat);
    let y: [f64; 8] = compute_y(uv[0], uv[1], uv[2], uv[3]);

    let mut scratch = lattice_omicron::LatticeScratch::new(eps);
    let budget_hit_flag = AtomicBool::new(false);

    let t0 = Instant::now();
    // Call phase1 with no SE-node budget (u64::MAX) so we observe the real
    // SE cost rather than a 100 K node artificial cap.
    let sols = lattice_omicron::phase1(&mut scratch, &y, k, eps, u64::MAX, &budget_hit_flag);
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let found_expected = sols.iter().any(|s| *s == expected_x);

    // Extra: verify every returned solution satisfies the ring constraints.
    for s in &sols {
        assert!(
            check_norm_eq(s, k),
            "phase1 returned a solution that fails norm check: {s:?}"
        );
        assert!(
            check_bilinear(s),
            "phase1 returned a solution that fails bilinear check: {s:?}"
        );
    }

    ProbeResult {
        elapsed_ms,
        n_sols: sols.len(),
        found_expected,
        budget_hit: budget_hit_flag.load(Ordering::Relaxed),
    }
}

fn main() {
    let eps = 1e-3_f64;
    eprintln!("probe_k_inner: n=6 lattice path timing probe at eps={eps:.0e}");
    eprintln!("  Run with CYCLOSYNTH_TRACE=1 for [SE-PREP] partial[0] output.");
    eprintln!("  Hard timeout per probe: {TIMEOUT_MS}ms (manual Ctrl-C if hung).");
    eprintln!();

    // Reset diag counters so each probe's CYCLOSYNTH_TRACE output is clean.
    // (diag::reset_all() clears all atomic counters)
    cyclosynth::synthesis::diag::reset_all();

    for &k in &[12u32, 15, 18] {
        eprint!("k={k}: constructing target and running lattice phase1 ... ");

        cyclosynth::synthesis::diag::reset_all();

        let t_wall = Instant::now();
        let result = run_probe(k, eps);
        let wall_ms = t_wall.elapsed().as_millis();

        let status = if result.budget_hit {
            "budget-hit"
        } else if result.found_expected {
            "solved"
        } else if result.n_sols > 0 {
            "found-other-sols-not-expected"
        } else {
            "no-solution"
        };

        // Emit per-stage timings if tracing.
        if cyclosynth::synthesis::diag::trace_enabled() {
            let snap = cyclosynth::synthesis::diag::snapshot();
            eprintln!(
                "\n  [DIAG k={k}] lll={:.1}ms chol={:.1}ms lu={:.1}ms \
                 se={:.1}ms  se_callbacks={}  sols_returned={}",
                snap.t_lll_ms,
                snap.t_cholesky_ms,
                snap.t_lu_ms,
                snap.t_se_ms,
                snap.se_callbacks,
                snap.sols_returned,
            );
        }

        // Primary result line (the format the user asked for).
        println!(
            "k={k}: elapsed={wall_ms}ms, n_sols={}, status={status}",
            result.n_sols,
        );

        // Stop as soon as a probe exceeds the timeout — the cliff is here.
        if wall_ms > TIMEOUT_MS {
            eprintln!("k={k}: wall_ms={wall_ms} > {TIMEOUT_MS}ms TIMEOUT — stopping.");
            break;
        }
    }
}
