//! Step 0 of PROMPT_n8_vs_n12_funnel.md.
//!
//! Run n=8 (lattice_zeta / Clifford+√T) on deterministic Haar SU(2) targets
//! at ε ∈ {1e-4, 1e-5, 1e-6}, seeds 0..6. Verify success and confirm the
//! reported distance independently by reconstructing the gate string into a
//! U2Q exact unitary and computing the diamond distance.
//!
//! Outcomes:
//!   - n=8 succeeds at 1e-5/1e-6 → the side-by-side comparison is meaningful;
//!     proceed to Step 1+2.
//!   - n=8 ALSO fails past 1e-4 → major finding; report and stop.

use cyclosynth::matrix::U2Q;
use cyclosynth::synthesis::clifford_sqrt_t::SynthesizerQ;
use cyclosynth::synthesis::distance::{diamond_distance_u2q_float, Mat2};
use num_complex::Complex64;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::f64::consts::PI;
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

fn gates_to_u2q(gates: &str) -> U2Q {
    gates.chars().fold(U2Q::eye(), |acc, ch| {
        let gate = match ch {
            'H' => U2Q::h(),
            'S' => U2Q::s(),
            'T' => U2Q::t(),
            'Q' => U2Q::q(),
            'X' => U2Q::x(),
            'Y' => U2Q::y(),
            'Z' => U2Q::z(),
            'I' => U2Q::eye(),
            other => panic!("unexpected n=8 gate {other:?}"),
        };
        acc * gate
    })
}

#[derive(Debug, Clone, Copy)]
struct RunRow {
    eps: f64,
    seed: u64,
    found: bool,
    lde: u32,
    claimed: f64,
    actual: f64,
    wall_s: f64,
    meets_eps: bool,
}

fn run_one(eps: f64, seed: u64) -> RunRow {
    let target = haar_target(seed);
    let synth = SynthesizerQ::new(eps).with_max_lde(40);
    let t0 = Instant::now();
    let res = synth.synthesize(target);
    let wall = t0.elapsed().as_secs_f64();
    match res {
        Some(r) => {
            let gates = r.gates.as_deref().unwrap_or("");
            let u = gates_to_u2q(gates);
            let actual = diamond_distance_u2q_float(&u, &target);
            RunRow {
                eps,
                seed,
                found: true,
                lde: r.lde,
                claimed: r.distance,
                actual,
                wall_s: wall,
                meets_eps: actual <= eps,
            }
        }
        None => RunRow {
            eps,
            seed,
            found: false,
            lde: 0,
            claimed: f64::NAN,
            actual: f64::NAN,
            wall_s: wall,
            meets_eps: false,
        },
    }
}

#[test]
#[ignore = "Step 0 of PROMPT_n8_vs_n12_funnel: verify n=8 deep-ε reaches 1e-5/1e-6"]
fn step0_n8_eps_sweep() {
    eprintln!(
        "\n[Step 0] n=8 (ζ₁₆) ε-sweep on deterministic Haar targets\n\
         seeds 0..6, ε ∈ {{1e-4, 1e-5, 1e-6}}, ε-driven, max_lde=40\n\
         actual distance computed independently by gates→U2Q→diamond_distance\n"
    );
    eprintln!(
        " ε       | seed | found | lde | claimed     | actual      | meets_ε | wall_s"
    );
    eprintln!("{}", "-".repeat(80));

    let eps_grid: &[f64] = &[1e-4, 1e-5, 1e-6];
    let mut rows: Vec<RunRow> = Vec::new();
    for &eps in eps_grid {
        for seed in 0_u64..6 {
            let r = run_one(eps, seed);
            eprintln!(
                "{:>8.0e} | {:>4} | {:>5} | {:>3} | {:>11.3e} | {:>11.3e} | {:>7} | {:>6.2}",
                r.eps,
                r.seed,
                if r.found { "yes" } else { "no" },
                r.lde,
                r.claimed,
                r.actual,
                if r.meets_eps { "yes" } else { "no" },
                r.wall_s,
            );
            rows.push(r);
        }
        eprintln!();
    }

    // Summary verdict per ε
    eprintln!("[Step 0] Summary:");
    for &eps in eps_grid {
        let bucket: Vec<&RunRow> = rows.iter().filter(|r| r.eps == eps).collect();
        let met = bucket.iter().filter(|r| r.meets_eps).count();
        let total = bucket.len();
        let avg_wall = bucket.iter().map(|r| r.wall_s).sum::<f64>() / total as f64;
        eprintln!(
            "  ε={eps:.0e}: {met}/{total} oracle-confirmed met, avg wall {avg_wall:.1}s"
        );
    }

    // Verdict gate: n=8 should clear 1e-5 and (ideally) 1e-6.
    let met_at_e5 = rows
        .iter()
        .filter(|r| (r.eps - 1e-5).abs() < 1e-20 && r.meets_eps)
        .count();
    let met_at_e6 = rows
        .iter()
        .filter(|r| (r.eps - 1e-6).abs() < 1e-20 && r.meets_eps)
        .count();
    eprintln!(
        "\n[Step 0] Verdict: 1e-5 met by {met_at_e5}/6 seeds, 1e-6 met by {met_at_e6}/6 seeds."
    );
    if met_at_e5 == 0 && met_at_e6 == 0 {
        eprintln!(
            "[Step 0] MAJOR FINDING: n=8 ALSO fails past 1e-4. The comparison target is a mirage."
        );
    } else {
        eprintln!(
            "[Step 0] n=8 reaches 1e-5/1e-6 → side-by-side funnel comparison is meaningful. Proceed to Step 1+2."
        );
    }
}
