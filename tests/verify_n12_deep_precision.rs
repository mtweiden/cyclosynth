//! Deep-precision verifier for the n=12 synthesizer.
//!
//! Per PROMPT_verify_n12_deep_precision.md:
//!  - Step 0: no pure ε-driven entry point exists in the project — we
//!    build the minimal wrapper here (loop k upward, return the first
//!    circuit whose *oracle-measured* distance to the target is ≤ ε).
//!    This is the ε-driven path under test.
//!  - Part 1: Haar-random U(2) targets, ε ∈ {1e-2, 1e-3, 1e-4}; classify
//!    met / over-tolerance / no-solution. Loose ε buckets assert 100% met;
//!    the tight ε=1e-4 bucket asserts native synthesis success after removal
//!    of the n=12 → pi6 fallback.
//!  - Part 2: 0-1 BFS over `{H, S, Sdg, X, Y, Z, P, Pdg}` (Clifford cost
//!    0, P/Pdg cost 1) using exact SO(3)-over-the-ring state keys.
//!    Compare BFS-min P-count to `decompose(U).t12_count` for every
//!    reachable state at minimal cost ≤ C.
//!
//! Oracle stays the verifier's: literal H/S/P matrices in f64 complex,
//! algebraic-Frobenius phase-invariant distance. The project's own
//! distance code is NEVER the source of truth in either part.

use cyclosynth::matrix::U2;
use cyclosynth::rings::ZUpsilon;
use cyclosynth::synthesis::clifford_pi12::{
    circuit_to_u2, decompose, synthesize_circuit_at_k, Gate, SO3Upsilon,
};
use num_complex::Complex64;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::time::Instant;

// ─── Oracle (re-stated here — copies, NOT imports of, prior verifier) ───────

type Mat2 = [[Complex64; 2]; 2];

fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

fn sqrt2_inv() -> f64 {
    1.0 / (0.5_f64).exp2()
}

fn oracle_h() -> Mat2 {
    let s = sqrt2_inv();
    [[c(s, 0.0), c(s, 0.0)], [c(s, 0.0), c(-s, 0.0)]]
}
fn oracle_s() -> Mat2 {
    [[c(1.0, 0.0), c(0.0, 0.0)], [c(0.0, 0.0), c(0.0, 1.0)]]
}
fn oracle_sdg() -> Mat2 {
    [[c(1.0, 0.0), c(0.0, 0.0)], [c(0.0, 0.0), c(0.0, -1.0)]]
}
fn oracle_p() -> Mat2 {
    let cos_p12 = ((6.0_f64).sqrt() + (2.0_f64).sqrt()) / 4.0;
    let sin_p12 = ((6.0_f64).sqrt() - (2.0_f64).sqrt()) / 4.0;
    [
        [c(1.0, 0.0), c(0.0, 0.0)],
        [c(0.0, 0.0), c(cos_p12, sin_p12)],
    ]
}
fn oracle_pdg() -> Mat2 {
    let cos_p12 = ((6.0_f64).sqrt() + (2.0_f64).sqrt()) / 4.0;
    let sin_p12 = ((6.0_f64).sqrt() - (2.0_f64).sqrt()) / 4.0;
    [
        [c(1.0, 0.0), c(0.0, 0.0)],
        [c(0.0, 0.0), c(cos_p12, -sin_p12)],
    ]
}
fn oracle_x() -> Mat2 {
    [[c(0.0, 0.0), c(1.0, 0.0)], [c(1.0, 0.0), c(0.0, 0.0)]]
}
fn oracle_y() -> Mat2 {
    [[c(0.0, 0.0), c(0.0, -1.0)], [c(0.0, 1.0), c(0.0, 0.0)]]
}
fn oracle_z() -> Mat2 {
    [[c(1.0, 0.0), c(0.0, 0.0)], [c(0.0, 0.0), c(-1.0, 0.0)]]
}
fn oracle_eye() -> Mat2 {
    [[c(1.0, 0.0), c(0.0, 0.0)], [c(0.0, 0.0), c(1.0, 0.0)]]
}

fn oracle_gate(g: Gate) -> Mat2 {
    match g {
        Gate::H => oracle_h(),
        Gate::S => oracle_s(),
        Gate::Sdg => oracle_sdg(),
        Gate::P => oracle_p(),
        Gate::Pdg => oracle_pdg(),
        Gate::X => oracle_x(),
        Gate::Y => oracle_y(),
        Gate::Z => oracle_z(),
    }
}

fn mat_mul(a: &Mat2, b: &Mat2) -> Mat2 {
    let mut out = [[c(0.0, 0.0); 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                out[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    out
}

fn mat_dagger(a: &Mat2) -> Mat2 {
    [
        [a[0][0].conj(), a[1][0].conj()],
        [a[0][1].conj(), a[1][1].conj()],
    ]
}

/// Compose `[g₀, g₁, …, gₙ]` as the matrix product `G₀·G₁·…·Gₙ` —
/// matches the project's `circuit_to_u2` (leftmost gate = leftmost
/// matrix factor).
fn oracle_circuit(circuit: &[Gate]) -> Mat2 {
    let mut u = oracle_eye();
    for &g in circuit {
        u = mat_mul(&u, &oracle_gate(g));
    }
    u
}

/// Phase-invariant distance via algebraic Frobenius (precision-stable
/// down to f64 epsilon; the naive `1 − |tr|²/4` shape has a ~2e-8
/// floor for identical inputs).
fn dist_oracle(a: &Mat2, b: &Mat2) -> f64 {
    let mut tr = c(0.0, 0.0);
    for i in 0..2 {
        for j in 0..2 {
            tr += a[i][j] * b[i][j].conj();
        }
    }
    let tr_abs = tr.norm();
    let phi = if tr_abs > 1e-300 {
        tr / tr_abs
    } else {
        c(1.0, 0.0)
    };
    let mut fro_sq = 0.0_f64;
    for i in 0..2 {
        for j in 0..2 {
            let diff = a[i][j] - phi * b[i][j];
            fro_sq += diff.norm_sqr();
        }
    }
    let d_sq = fro_sq * (8.0 - fro_sq) / 16.0;
    d_sq.max(0.0).sqrt()
}

fn unitary_defect(u: &Mat2) -> f64 {
    let p = mat_mul(u, &mat_dagger(u));
    let i = oracle_eye();
    let mut s = 0.0;
    for r in 0..2 {
        for cc in 0..2 {
            s += (p[r][cc] - i[r][cc]).norm_sqr();
        }
    }
    s.sqrt()
}

/// Count BOTH P and Pdg (each is a cost-1 π/12 rotation).
fn count_p_symbols(circuit: &[Gate]) -> usize {
    circuit
        .iter()
        .filter(|&&g| matches!(g, Gate::P | Gate::Pdg))
        .count()
}

fn assert_gate_set_clean(circuit: &[Gate], label: &str) {
    for &g in circuit {
        match g {
            Gate::H | Gate::S | Gate::Sdg | Gate::P | Gate::Pdg | Gate::X | Gate::Y | Gate::Z => {}
        }
    }
    let _ = label;
}

fn haar_target(seed: u64) -> Mat2 {
    let mut rng = StdRng::seed_from_u64(seed);
    loop {
        let raw: [f64; 4] = std::array::from_fn(|_| {
            let mut s = 0.0;
            for _ in 0..12 {
                s += rng.random::<f64>();
            }
            s - 6.0
        });
        let v00 = c(raw[0], raw[1]);
        let v10 = c(raw[2], raw[3]);
        let n = (v00.norm_sqr() + v10.norm_sqr()).sqrt();
        if n < 1e-6 {
            continue;
        }
        let v00 = v00 / n;
        let v10 = v10 / n;
        return [[v00, -v10.conj()], [v10, v00.conj()]];
    }
}

// ─── Step 0 — minimal ε-driven wrapper ──────────────────────────────────────

/// "Synthesize to ε": loop k from K_MIN upward, request the project's
/// `synthesize_circuit_at_k`, measure oracle distance, and return the
/// first circuit whose *oracle* distance is ≤ ε. This is the production
/// path under test: ask for ε, get a circuit within ε.
///
/// Returns `(circuit, k_reached, oracle_d, wall_time)` or `None` if no
/// `k ≤ K_MAX` produced an oracle-accepted circuit.
fn synthesize_to_eps(
    target: &Mat2,
    eps: f64,
    k_min: u32,
    k_max: u32,
) -> Option<(Vec<Gate>, u32, f64, std::time::Duration, usize)> {
    let t0 = Instant::now();
    for k in k_min..=k_max {
        // Avoid the slow brute-path k=3 regression (10s per call,
        // see prior verifier). Production-style ε-sweep uses SE at k≥5.
        if k < 5 {
            continue;
        }
        let r = synthesize_circuit_at_k(target, k, eps);
        if let Some(r) = r {
            let u = oracle_circuit(&r.circuit);
            let d = dist_oracle(&u, target);
            if d <= eps {
                let elapsed = t0.elapsed();
                return Some((r.circuit, k, d, elapsed, r.t12_count));
            }
        }
    }
    None
}

// ─── Part 1 — ε-driven Haar deep precision ──────────────────────────────────

#[derive(Default)]
struct EpsBucket {
    met: usize,
    over: Vec<(u64, f64, u32, usize, usize)>, // seed, d, k, p_emit, t12
    none: Vec<u64>,
    d_metrics: Vec<f64>,
    k_metrics: Vec<u32>,
    p_emit_metrics: Vec<usize>,
    t12_metrics: Vec<usize>,
    wall_us: Vec<u128>,
}

fn run_eps_bucket(eps: f64, n_seeds: u64, k_max: u32) -> EpsBucket {
    let mut b = EpsBucket::default();
    for seed in 0..n_seeds {
        let target = haar_target(seed);
        match synthesize_to_eps(&target, eps, 5, k_max) {
            Some((circuit, k, d, elapsed, t12)) => {
                // Unitarity + gate-set sanity.
                let u = oracle_circuit(&circuit);
                assert!(
                    unitary_defect(&u) < 1e-10,
                    "[Part 1] seed {seed} ε={eps}: oracle product not unitary"
                );
                assert_gate_set_clean(&circuit, &format!("seed {seed}"));
                let p_emit = count_p_symbols(&circuit);
                if d <= eps {
                    b.met += 1;
                    b.d_metrics.push(d);
                    b.k_metrics.push(k);
                    b.p_emit_metrics.push(p_emit);
                    b.t12_metrics.push(t12);
                    b.wall_us.push(elapsed.as_micros());
                } else {
                    // synthesize_to_eps filtered on oracle d ≤ ε, so this
                    // branch should be empty unless the filter logic
                    // changes.
                    b.over.push((seed, d, k, p_emit, t12));
                }
            }
            None => {
                b.none.push(seed);
            }
        }
    }
    b
}

#[test]
fn part1_synthesize_meets_epsilon() {
    let configs: &[(f64, u64, u32)] = &[
        // (ε, N seeds, k_max). Loose success buckets keep N≥30; the
        // tight bucket is smaller because the native n=12 SE search is
        // intentionally deeper there.
        (1e-2, 30, 9),
        (1e-3, 30, 12),
        (1e-4, 5, 12),
    ];
    eprintln!(
        "\n[Part 1] ε-driven Haar deep precision (independent oracle distance)\n\
         ε      | seeds |  met |  over |  none |  max d_proj |  med d_proj |  max k | med k | max P-emit | med P-emit | max t12 | med t12 | avg ms"
    );

    let mut all_clean = true;
    let mut details: Vec<String> = Vec::new();
    let mut p_emit_table: Vec<(f64, f64)> = Vec::new(); // (log2(1/ε), median P_emit)

    for &(eps, n_seeds, k_max) in configs {
        let mut b = run_eps_bucket(eps, n_seeds, k_max);
        let met = b.met;
        let over = b.over.len();
        let none = b.none.len();
        b.d_metrics.sort_by(|a, c| a.partial_cmp(c).unwrap());
        b.k_metrics.sort();
        b.p_emit_metrics.sort();
        b.t12_metrics.sort();
        b.wall_us.sort();
        let max_d = *b.d_metrics.last().unwrap_or(&f64::NAN);
        let med_d = if b.d_metrics.is_empty() {
            f64::NAN
        } else {
            b.d_metrics[b.d_metrics.len() / 2]
        };
        let max_k = b.k_metrics.last().copied().unwrap_or(0);
        let med_k = b.k_metrics.get(b.k_metrics.len() / 2).copied().unwrap_or(0);
        let max_pe = b.p_emit_metrics.last().copied().unwrap_or(0);
        let med_pe = b
            .p_emit_metrics
            .get(b.p_emit_metrics.len() / 2)
            .copied()
            .unwrap_or(0);
        let max_t12 = b.t12_metrics.last().copied().unwrap_or(0);
        let med_t12 = b
            .t12_metrics
            .get(b.t12_metrics.len() / 2)
            .copied()
            .unwrap_or(0);
        let avg_ms = if b.wall_us.is_empty() {
            f64::NAN
        } else {
            b.wall_us.iter().sum::<u128>() as f64 / b.wall_us.len() as f64 / 1000.0
        };

        eprintln!(
            "{:>6.0e} | {:>5} | {:>4} | {:>5} | {:>5} | {:>11.3e} | {:>11.3e} | {:>6} | {:>5} | {:>10} | {:>10} | {:>7} | {:>7} | {:>6.1}",
            eps, n_seeds, met, over, none, max_d, med_d, max_k, med_k, max_pe, med_pe, max_t12, med_t12, avg_ms
        );

        if over > 0 {
            all_clean = false;
            for (seed, d, k, pe, t12) in &b.over {
                details.push(format!(
                    "[Part 1 OVER] ε={eps} seed={seed}: d={d:.3e} > ε at k={k}, P_emit={pe}, t12={t12}"
                ));
            }
        }
        if none > 0 {
            all_clean = false;
            for seed in &b.none {
                details.push(format!(
                    "[Part 1 NONE] ε={eps} seed={seed}: no k ≤ {k_max} yielded an oracle-met circuit"
                ));
            }
        }

        // Track P-count vs log2(1/ε) (use median P_emit for stability).
        if met > 0 {
            p_emit_table.push(((1.0 / eps).log2(), med_pe as f64));
        }
    }

    // Brief P-count-vs-log2(1/ε) report.
    eprintln!("\n[Part 1] P-emit slope: median count grows with log₂(1/ε):");
    for (l, p) in &p_emit_table {
        eprintln!("   log₂(1/ε) = {l:.2}: median P_emit = {p:.0}");
    }
    if p_emit_table.len() >= 2 {
        // crude slope estimate
        let (lx0, py0) = p_emit_table.first().unwrap();
        let (lx1, py1) = p_emit_table.last().unwrap();
        let slope = (py1 - py0) / (lx1 - lx0);
        eprintln!("   slope ≈ {slope:.2} P-gates per log₂(1/ε)");
    }

    if !all_clean {
        for d in &details {
            eprintln!("{d}");
        }
        panic!(
            "[Part 1] FAILED: {} over/none case(s) — see stderr",
            details.len()
        );
    }
}

// ─── Part 2 — BFS-optimal vs decompose ──────────────────────────────────────

/// Single-qubit gate alphabet for the BFS. Cost is `1` for P/Pdg, `0`
/// for the Cliffords.
fn bfs_gates() -> &'static [(Gate, u32)] {
    &[
        (Gate::H, 0),
        (Gate::S, 0),
        (Gate::Sdg, 0),
        (Gate::X, 0),
        (Gate::Y, 0),
        (Gate::Z, 0),
        (Gate::P, 1),
        (Gate::Pdg, 1),
    ]
}

/// Vec-backed visited set; linear scan via `PartialEq` is fine at the
/// scale we reach (Clifford layer = 24 states; subsequent layers grow
/// by P/Pdg steps, doubling-then-Clifford-closing — at C_MAX=2 the
/// visited set is in the thousands, manageable).
fn visited_position(
    visited: &[(SO3Upsilon, u32, U2<ZUpsilon>)],
    target: &SO3Upsilon,
) -> Option<usize> {
    visited.iter().position(|(s, _, _)| s == target)
}

/// 0-cost Clifford closure of `seed`. Updates `visited` and returns
/// the set of NEW states added at this layer (for the next P/Pdg step).
fn clifford_closure(
    seed: Vec<(SO3Upsilon, U2<ZUpsilon>)>,
    visited: &mut Vec<(SO3Upsilon, u32, U2<ZUpsilon>)>,
    cost: u32,
) -> Vec<(SO3Upsilon, U2<ZUpsilon>)> {
    let cliffords: &[Gate] = &[Gate::H, Gate::S, Gate::Sdg, Gate::X, Gate::Y, Gate::Z];
    let mut layer = seed;
    let mut frontier = layer.clone();
    while !frontier.is_empty() {
        let mut new_states: Vec<(SO3Upsilon, U2<ZUpsilon>)> = Vec::new();
        for (_, u) in &frontier {
            for &g in cliffords {
                let u_next = *u * g.to_u2();
                let so3_next = SO3Upsilon::from_u2(&u_next);
                if visited_position(visited, &so3_next).is_none() {
                    visited.push((so3_next.clone(), cost, u_next));
                    new_states.push((so3_next, u_next));
                }
            }
        }
        layer.extend_from_slice(&new_states);
        frontier = new_states;
    }
    layer
}

#[test]
fn part2_decompose_matches_bfs_optimal() {
    // 0-1 BFS over the phase-invariant SO(3) state space. The key is
    // `SO3Upsilon` (exact-ring 3×3 Bloch matrix), which the project
    // already computes — it quotients out the global phase exactly.
    //
    // C_MAX = 3 reaches deep enough to find a=3,4,5 single-axis costs
    // (P, P², P³ at the z-axis and Clifford-conjugate copies on x/y).
    // C_MAX = 2 found 888 states with zero mismatch in 1 s.
    const C_MAX: u32 = 3;

    let mut visited: Vec<(SO3Upsilon, u32, U2<ZUpsilon>)> = Vec::new();
    let identity_u: U2<ZUpsilon> = U2::eye();
    let identity_so3 = SO3Upsilon::from_u2(&identity_u);
    visited.push((identity_so3.clone(), 0, identity_u));

    // Cost-0 layer: Clifford closure of identity.
    let layer0 = clifford_closure(vec![(identity_so3.clone(), identity_u)], &mut visited, 0);
    eprintln!(
        "[Part 2] BFS layer 0 (Clifford closure): {} states",
        layer0.len()
    );
    let mut layers: Vec<Vec<(SO3Upsilon, U2<ZUpsilon>)>> = vec![layer0];

    let p_steps: &[Gate] = &[Gate::P, Gate::Pdg];
    for cost in 1..=C_MAX {
        let mut seed: Vec<(SO3Upsilon, U2<ZUpsilon>)> = Vec::new();
        let prev = &layers[cost as usize - 1];
        for (_, u) in prev {
            for &g in p_steps {
                let u_next = *u * g.to_u2();
                let so3_next = SO3Upsilon::from_u2(&u_next);
                if visited_position(&visited, &so3_next).is_none() {
                    visited.push((so3_next.clone(), cost, u_next));
                    seed.push((so3_next, u_next));
                }
            }
        }
        let layer = clifford_closure(seed, &mut visited, cost);
        eprintln!(
            "[Part 2] BFS layer {cost} (after Clifford closure): {} new states",
            layer.len()
        );
        layers.push(layer);
    }

    let total = visited.len();
    eprintln!("[Part 2] BFS done: total reachable states with P-count ≤ {C_MAX} = {total}");

    let mut overcount: Vec<(u32, u32, U2<ZUpsilon>)> = Vec::new();
    let mut undercount: Vec<(u32, u32, U2<ZUpsilon>)> = Vec::new();
    let mut checked = 0usize;
    for (_so3, bfs_count, u) in &visited {
        let r = decompose(u);
        let dc = r.t12_count as u32;
        if dc > *bfs_count {
            overcount.push((*bfs_count, dc, *u));
        } else if dc < *bfs_count {
            undercount.push((*bfs_count, dc, *u));
            eprintln!(
                "[Part 2 UNDER] BFS-min={bfs_count} but decompose={dc} (re-multiplied via circuit_to_u2: {:?})",
                circuit_to_u2(&r.circuit).to_float()
            );
        }
        checked += 1;
    }
    eprintln!(
        "[Part 2] checked {checked} states; overcount = {}, undercount = {}",
        overcount.len(),
        undercount.len()
    );
    if !overcount.is_empty() {
        let mut hist: std::collections::BTreeMap<(u32, u32), usize> =
            std::collections::BTreeMap::new();
        for (b, d, _) in &overcount {
            *hist.entry((*b, *d)).or_insert(0) += 1;
        }
        eprintln!("[Part 2] overcount histogram (BFS_count, decompose_count) → frequency:");
        for ((b, d), n) in &hist {
            eprintln!("   ({b}, {d}) → {n}");
        }
        let example = &overcount[0];
        eprintln!(
            "[Part 2] one overcount witness: bfs={} decompose={} u11={} u22={} k={}",
            example.0, example.1, example.2.u11, example.2.u22, example.2.k
        );
    }
    assert!(
        undercount.is_empty(),
        "[Part 2] {} undercount cases (decompose < BFS) — BFS or decompose bug",
        undercount.len()
    );
    assert!(
        overcount.is_empty(),
        "[Part 2] decompose OVERCOUNTS on {} states — see stderr",
        overcount.len()
    );
}
