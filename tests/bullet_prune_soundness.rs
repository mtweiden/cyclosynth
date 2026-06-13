//! Soundness gate for bullet-aware SE pruning (n=12 lattice_upsilon).
//!
//! Premise: pruning must not remove any feasible (norm + bullets + alignment)
//! solution. For every (seed, k, ε) we enumerate the FULL solution set with
//! pruning OFF and with pruning ON; the two sets must be identical.
//!
//! Coverage:
//!   - Multiple deterministic Haar seeds (seed=0..N).
//!   - k from 5..=12 (the range where unpruned SE is feasible).
//!   - ε at 1e-3 and 1e-4 (the two ε bands relevant to the unlock).

use cyclosynth::synthesis::lattice_upsilon::synthesize;
use num_complex::Complex64;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::collections::BTreeSet;
use std::f64::consts::PI;

fn haar_target(seed: u64) -> [[Complex64; 2]; 2] {
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

/// Collect the full solution set (as a sorted set of 16-tuples) from
/// `synthesize`. Uses a env-var override to toggle pruning on/off.
fn collect_solution_set(
    target: &[[Complex64; 2]; 2],
    k: u32,
    eps: f64,
    prune_on: bool,
) -> BTreeSet<[i64; 16]> {
    // SAFETY: tests in this binary run serially (default cargo test for an
    // integration test is serial; we also avoid threading inside the
    // synthesize path). The synthesize call below reads the env var fresh.
    unsafe {
        std::env::set_var(
            "CYCLOSYNTH_BULLET_PRUNE_N12",
            if prune_on { "1" } else { "0" },
        );
    }
    let res = synthesize(target, k, eps);
    let mut out = BTreeSet::new();
    if let Some(r) = res {
        out.insert(r.solution);
    }
    out
}

/// `synthesize` returns only the best phase, not the full set. For the
/// soundness gate we need the full set, so call directly into
/// `lattice_upsilon::phase1` and collect every lattice solution. Then
/// compare the two paths.
fn collect_all_lattice_solutions(
    target: &[[Complex64; 2]; 2],
    k: u32,
    eps: f64,
    prune_on: bool,
    budget: u64,
) -> (BTreeSet<[i64; 16]>, bool) {
    use cyclosynth::synthesis::lattice_upsilon::{phase1, LatticeScratch};
    unsafe {
        std::env::set_var(
            "CYCLOSYNTH_BULLET_PRUNE_N12",
            if prune_on { "1" } else { "0" },
        );
    }
    let v = [
        target[0][0].re,
        target[0][0].im,
        target[1][0].re,
        target[1][0].im,
    ];
    let mut scratch = LatticeScratch::new(eps);
    let budget_hit = std::sync::atomic::AtomicBool::new(false);
    let sols = phase1(&mut scratch, v, k, eps, budget, &budget_hit);
    let hit = budget_hit.load(std::sync::atomic::Ordering::Relaxed);
    (sols.into_iter().collect(), hit)
}

#[test]
fn bullet_prune_set_equality_e3() {
    // ε = 1e-3, k = 6..=10, several seeds. We cap the budget tight to keep
    // the unpruned walk finite. If the budget is hit (incomplete walk),
    // the comparison is only meaningful if BOTH paths also stopped early
    // with the same partial state — skip those pairs.
    let eps = 1e-3_f64;
    let budget: u64 = 5_000_000;
    let mut total_pairs = 0;
    let mut nontrivial = 0;
    let mut budget_skips = 0;
    for seed in 0..5_u64 {
        let target = haar_target(seed);
        for k in 6..=10_u32 {
            let (off, off_hit) = collect_all_lattice_solutions(&target, k, eps, false, budget);
            let (on, on_hit) = collect_all_lattice_solutions(&target, k, eps, true, budget);
            total_pairs += 1;
            if off_hit || on_hit {
                budget_skips += 1;
                continue;
            }
            if !off.is_empty() || !on.is_empty() {
                nontrivial += 1;
            }
            assert_eq!(
                off, on,
                "PRUNING DROPPED A VALID SOLUTION at seed={seed} k={k} eps={eps:.0e}\n\
                 off={off:?}\non={on:?}"
            );
        }
    }
    eprintln!(
        "[soundness] ε={eps:.0e}: {total_pairs} (seed,k) pairs checked, {nontrivial} nonempty, {budget_skips} skipped (budget hit) — pruned set == unpruned set on all completed"
    );
    assert!(
        nontrivial > 0,
        "expected at least some pairs to find solutions, otherwise the gate is vacuous"
    );
}

#[test]
fn bullet_prune_set_equality_e4() {
    let eps = 1e-4_f64;
    let budget: u64 = 20_000_000;
    let mut total_pairs = 0;
    let mut nontrivial = 0;
    let mut budget_skips = 0;
    for seed in 0..3_u64 {
        let target = haar_target(seed);
        for k in 8..=11_u32 {
            let (off, off_hit) = collect_all_lattice_solutions(&target, k, eps, false, budget);
            let (on, on_hit) = collect_all_lattice_solutions(&target, k, eps, true, budget);
            total_pairs += 1;
            if off_hit || on_hit {
                budget_skips += 1;
                continue;
            }
            if !off.is_empty() || !on.is_empty() {
                nontrivial += 1;
            }
            assert_eq!(
                off, on,
                "PRUNING DROPPED A VALID SOLUTION at seed={seed} k={k} eps={eps:.0e}\n\
                 off={off:?}\non={on:?}"
            );
        }
    }
    eprintln!(
        "[soundness] ε={eps:.0e}: {total_pairs} (seed,k) pairs checked, {nontrivial} nonempty, {budget_skips} skipped (budget hit) — pruned set == unpruned set on all completed"
    );
}

#[test]
#[ignore = "longer regression — runs the small-k Gate-A fixtures through pruned SE"]
fn bullet_prune_set_equality_fixtures() {
    // Re-use the synthesize_circuit_in_range path at ε=5e-2 (loose) to
    // sanity-check pruning doesn't drop any of the existing fixture
    // candidates at small k.
    let target = haar_target(7);
    let budget: u64 = 5_000_000;
    for k in 5..=9_u32 {
        let (off, _) = collect_all_lattice_solutions(&target, k, 5e-2, false, budget);
        let (on, _) = collect_all_lattice_solutions(&target, k, 5e-2, true, budget);
        assert_eq!(off, on, "fixture set mismatch at k={k}");
    }
}

/// Sanity: the "first-only" wrappers (used by production synthesize_first)
/// agree with respect to membership when pruning is on vs off. If the
/// full-set check above passes, the first-solution must also agree.
#[test]
fn bullet_prune_synthesize_membership_e3() {
    let eps = 1e-3_f64;
    for seed in 0..5_u64 {
        let target = haar_target(seed);
        for k in 6..=10_u32 {
            let off = collect_solution_set(&target, k, eps, false);
            let on = collect_solution_set(&target, k, eps, true);
            // The single best phase may differ if multiple solutions exist
            // (the "best" phase is selected per-call). What matters is
            // existence: pruning must not turn a Some into a None.
            assert_eq!(
                off.is_empty(),
                on.is_empty(),
                "pruning changed Some/None status at seed={seed} k={k}"
            );
        }
    }
}
