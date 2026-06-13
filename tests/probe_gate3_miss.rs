//! Diagnostic test: investigate the Gate-3 miss to localize whether it's a
//! per-half-region containment problem (math/derivation) or an LLL-SE walker
//! problem.

use cyclosynth::rings::types::Int;
use cyclosynth::rings::ZUpsilon;
use cyclosynth::synthesis::lattice_upsilon::mitm::{
    brute_mitm_norm_bullet_set, HalfSide, PerHalfRegion,
};
use cyclosynth::synthesis::lattice_upsilon::mitm_half_se::lll_se_mitm_norm_bullet_set;
use cyclosynth::synthesis::lattice_upsilon::synthesize::{best_phase, zeta_pow};
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

fn mul_zeta_pow(c: &[i64; 8], j: u32) -> [i64; 8] {
    let zu_in = ZUpsilon::new(
        Int::from_i64(c[0]),
        Int::from_i64(c[1]),
        Int::from_i64(c[2]),
        Int::from_i64(c[3]),
        Int::from_i64(c[4]),
        Int::from_i64(c[5]),
        Int::from_i64(c[6]),
        Int::from_i64(c[7]),
    );
    let zu_out = zu_in * zeta_pow(j);
    let coeffs = zu_out.coeffs();
    let mut out = [0i64; 8];
    for i in 0..8 {
        out[i] = coeffs[i].as_i128() as i64;
    }
    out
}

fn orbit_member(x: &[i64; 16], j: u32) -> [i64; 16] {
    let mut u1 = [0i64; 8];
    let mut u2 = [0i64; 8];
    u1.copy_from_slice(&x[..8]);
    u2.copy_from_slice(&x[8..]);
    let u1p = mul_zeta_pow(&u1, j);
    let u2p = mul_zeta_pow(&u2, j);
    let mut out = [0i64; 16];
    out[..8].copy_from_slice(&u1p);
    out[8..].copy_from_slice(&u2p);
    out
}

#[test]
#[ignore = "diagnostic; ε=0.2 k=5 — runs brute MITM (~30s) for comparison"]
fn probe_gate3_miss_seed0() {
    let target = haar_target(0);
    let k = 5_u32;
    let eps = 0.2_f64;
    let x_miss: [i64; 16] = [1, 4, 0, -1, -1, 1, -1, 2, -1, -2, -1, 0, 1, 1, 1, 1];

    eprintln!(
        "R = {}  V_11 = {:?}  V_21 = {:?}",
        (2.0_f64.powi(k as i32)).sqrt(),
        target[0][0],
        target[1][0],
    );

    let r1 = PerHalfRegion::new(HalfSide::U1, target[0][0], k, eps);
    let r2 = PerHalfRegion::new(HalfSide::U2, target[1][0], k, eps);

    eprintln!("\nOrbit containment check for x_miss (24 ζ^j shifts):");
    let mut hit = 0;
    for j in 0..24 {
        let xj = orbit_member(&x_miss, j);
        let mut u1j = [0i64; 8];
        let mut u2j = [0i64; 8];
        u1j.copy_from_slice(&xj[..8]);
        u2j.copy_from_slice(&xj[8..]);
        let c1 = r1.contains(&u1j);
        let c2 = r2.contains(&u2j);
        if c1 && c2 {
            hit += 1;
            eprintln!("  j={j}: BOTH contains() ✓  u1={u1j:?}  u2={u2j:?}");
        } else if c1 || c2 {
            eprintln!(
                "  j={j}: contains() partial: r1={c1} r2={c2}  u1={u1j:?}  u2={u2j:?}"
            );
        }
    }
    eprintln!("Total orbit members with both contains() ✓ : {hit}");

    let (_, _, d) = best_phase(&x_miss, k, &target);
    eprintln!("best_phase distance(x_miss → target) = {d}");

    eprintln!("\nComputing brute MITM for cross-check…");
    let brute_set: BTreeSet<[i64; 16]> = brute_mitm_norm_bullet_set(&target, k, eps)
        .into_iter()
        .collect();
    eprintln!("|brute_mitm| = {}", brute_set.len());

    let mut brute_orbit_hit = 0;
    for j in 0..24 {
        let xj = orbit_member(&x_miss, j);
        if brute_set.contains(&xj) {
            brute_orbit_hit += 1;
            eprintln!("  brute_mitm contains orbit j={j}");
        }
    }
    eprintln!("brute_mitm orbit hits: {brute_orbit_hit}");

    let se_set: BTreeSet<[i64; 16]> = lll_se_mitm_norm_bullet_set(&target, k, eps)
        .into_iter()
        .collect();
    eprintln!("|lll_se_mitm| = {}", se_set.len());
    let mut se_orbit_hit = 0;
    for j in 0..24 {
        let xj = orbit_member(&x_miss, j);
        if se_set.contains(&xj) {
            se_orbit_hit += 1;
            eprintln!("  lll_se_mitm contains orbit j={j}");
        }
    }
    eprintln!("lll_se_mitm orbit hits: {se_orbit_hit}");
}
