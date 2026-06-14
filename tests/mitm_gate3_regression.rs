//! Named-regression test for PROMPT_mitm_8d_completeness.md Lever-gate.
//!
//! This is the *fast* form of the Gate-3 probe: at (Haar seed=0, k=5,
//! ε=0.2), the joint-SE solution `x = [1,4,0,-1,-1,1,-1,2 | -1,-2,-1,0,
//! 1,1,1,1]` is verified to lie in the per-half regions for both halves
//! (j=0, j=1 orbit members), and the 273K-element brute MITM pool contains
//! both orbit members. The 8D LLL+SE backend MUST also emit them.
//!
//! Before Lever-1 (BKZ-8) was added, the LLL+SE pool was 18K and contained
//! NEITHER orbit member of x. After BKZ-8, this test must pass — that's the
//! gate that gates Part 4 / Part 5.

use cyclosynth::rings::types::Int;
use cyclosynth::rings::ZUpsilon;
use cyclosynth::synthesis::lattice_upsilon::mitm::{HalfSide, PerHalfRegion};
use cyclosynth::synthesis::lattice_upsilon::mitm_half_se::lll_se_mitm_norm_bullet_set;
use cyclosynth::synthesis::lattice_upsilon::synthesize::zeta_pow;
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

/// Diagnostic for the second Gate-3 miss (after Lever 1). Same target as
/// the named regression but a different joint-SE x. Find which canonical
/// orbit member lies in both per-half regions, then check whether the
/// LLL/BKZ-SE backend emits it.
#[test]
fn diagnose_gate3_miss_2_seed0_k5_eps2e1() {
    use cyclosynth::synthesis::lattice_upsilon::mitm_half_se::lll_se_enumerate_half_with_stats;

    let target = haar_target(0);
    let k = 5_u32;
    let eps = 0.2_f64;
    let x_miss: [i64; 16] = [2, 3, 1, -1, -1, -1, -1, 4, -1, -1, -2, 0, 0, 2, 1, 1];

    let r1 = PerHalfRegion::new(HalfSide::U1, target[0][0], k, eps);
    let r2 = PerHalfRegion::new(HalfSide::U2, target[1][0], k, eps);

    eprintln!("\nOrbit check for x_miss_2 = {x_miss:?}");
    let mut both_in: Vec<(u32, [i64; 8], [i64; 8])> = Vec::new();
    for j in 0..24 {
        let xj = orbit_member(&x_miss, j);
        let mut u1j = [0i64; 8];
        let mut u2j = [0i64; 8];
        u1j.copy_from_slice(&xj[..8]);
        u2j.copy_from_slice(&xj[8..]);
        let c1 = r1.contains(&u1j);
        let c2 = r2.contains(&u2j);
        if c1 && c2 {
            both_in.push((j, u1j, u2j));
            eprintln!("  j={j}: BOTH in region. u1={u1j:?} u2={u2j:?}");
        }
    }
    eprintln!("orbits with both u1, u2 in their regions: {}", both_in.len());

    let (pool1, stats1) = lll_se_enumerate_half_with_stats(&r1);
    let (pool2, stats2) = lll_se_enumerate_half_with_stats(&r2);
    eprintln!(
        "\npool1 size = {}, stats = {:?} | pool2 size = {}, stats = {:?}",
        pool1.len(),
        stats1,
        pool2.len(),
        stats2
    );

    let p1_set: BTreeSet<&[i64; 8]> = pool1.iter().collect();
    let p2_set: BTreeSet<&[i64; 8]> = pool2.iter().collect();
    for (j, u1j, u2j) in &both_in {
        eprintln!(
            "  j={j}: u1 in pool1? {}  u2 in pool2? {}",
            p1_set.contains(u1j),
            p2_set.contains(u2j),
        );
    }
}

/// The named regression: at (seed=0, k=5, ε=0.2), the LLL/BKZ+SE 8D
/// half-enumerator + MITM join MUST emit (at least) the j=0 and j=1
/// orbit members of the joint-SE solution that the previous LLL-only
/// build dropped.
#[test]
fn gate3_named_regression_seed0_k5_eps2e1() {
    let target = haar_target(0);
    let k = 5_u32;
    let eps = 0.2_f64;
    let x_miss: [i64; 16] = [1, 4, 0, -1, -1, 1, -1, 2, -1, -2, -1, 0, 1, 1, 1, 1];

    let r1 = PerHalfRegion::new(HalfSide::U1, target[0][0], k, eps);
    let r2 = PerHalfRegion::new(HalfSide::U2, target[1][0], k, eps);

    // Sanity: the canonical orbit members do lie in both regions (this is
    // the math-derivation contract; if it ever fails the region/derivation
    // is wrong, not the walker).
    let mut canonical_orbits = Vec::new();
    for j in 0..24 {
        let xj = orbit_member(&x_miss, j);
        let mut u1j = [0i64; 8];
        let mut u2j = [0i64; 8];
        u1j.copy_from_slice(&xj[..8]);
        u2j.copy_from_slice(&xj[8..]);
        if r1.contains(&u1j) && r2.contains(&u2j) {
            canonical_orbits.push((j, xj));
        }
    }
    assert!(
        !canonical_orbits.is_empty(),
        "math-derivation broken: no orbit member of x_miss lies in both per-half regions"
    );

    let mitm_set: BTreeSet<[i64; 16]> =
        lll_se_mitm_norm_bullet_set(&target, k, eps).into_iter().collect();
    let mut hits: Vec<u32> = Vec::new();
    for (j, xj) in &canonical_orbits {
        if mitm_set.contains(xj) {
            hits.push(*j);
        }
    }
    assert!(
        !hits.is_empty(),
        "Gate-3 named regression FAILED: |LLL/BKZ-SE-MITM|={}, contains 0 of {} canonical \
         orbit members of x_miss={x_miss:?}. (The walker is still mis-orienting the basis \
         and scattering its leaf budget over outer Q-shells before reaching the \
         in-region integers; Lever 2/3 of PROMPT_mitm_8d_completeness.md is needed.)",
        mitm_set.len(),
        canonical_orbits.len(),
    );
    eprintln!(
        "Gate-3 named regression PASS: |MITM|={}, hit {} of {} canonical orbits (j ∈ {:?})",
        mitm_set.len(),
        hits.len(),
        canonical_orbits.len(),
        hits,
    );
}
