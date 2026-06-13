//! Part 3 soundness gates for PROMPT_lattice_upsilon_mitm_8d_se.md.
//!
//! These run the **same gates** as `mitm_soundness.rs` but against the new
//! 8D LLL+SE backend (`lll_se_mitm_norm_bullet_set` / `lll_se_enumerate_half`)
//! plus a Part-3-(1) 8D-SE ≡ brute-half set equality check on fixture targets
//! AND several Haar V_i — the oracle-equality test demanded by the prompt.
//!
//! Each gate STOPs on the first miss and prints the offending point, its
//! halves, their keys, and which stage lost it.

use cyclosynth::matrix::U2;
use cyclosynth::rings::types::Int;
use cyclosynth::rings::ZUpsilon;
use cyclosynth::synthesis::lattice_upsilon::enumerate::phase1_brute;
use cyclosynth::synthesis::lattice_upsilon::mitm::{
    brute_enumerate_half, complement, key_of, HalfSide, PerHalfRegion,
};
use cyclosynth::synthesis::lattice_upsilon::mitm_half_se::{
    lll_se_enumerate_half, lll_se_mitm_norm_bullet_set,
};
use cyclosynth::synthesis::lattice_upsilon::synthesize::{best_phase, zeta_pow};
use num_complex::Complex64;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::collections::BTreeSet;
use std::f64::consts::PI;

type Mat2 = [[Complex64; 2]; 2];

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

fn split_x(x: &[i64; 16]) -> ([i64; 8], [i64; 8]) {
    let mut u1 = [0i64; 8];
    let mut u2 = [0i64; 8];
    u1.copy_from_slice(&x[..8]);
    u2.copy_from_slice(&x[8..]);
    (u1, u2)
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
    let (u1, u2) = split_x(x);
    let u1p = mul_zeta_pow(&u1, j);
    let u2p = mul_zeta_pow(&u2, j);
    let mut out = [0i64; 16];
    out[..8].copy_from_slice(&u1p);
    out[8..].copy_from_slice(&u2p);
    out
}

fn orbit_set(x: &[i64; 16]) -> BTreeSet<[i64; 16]> {
    (0..24).map(|j| orbit_member(x, j)).collect()
}

// ─── Part 3 step 1: 8D-SE ≡ brute-half set equality ─────────────────────────
//
// For k ∈ {2, 3, 4} on the fixture targets AND on several Haar V_i, the
// 8D SE half-pool must EQUAL the brute/box half-pool, exact i256 coordinates,
// full sets. This is the smart-≡-brute pattern already in the module, pointed
// at the new backend.

fn check_lll_se_eq_brute(target: &Mat2, k: u32, eps: f64, label: &str) {
    let r1 = PerHalfRegion::new(HalfSide::U1, target[0][0], k, eps);
    let r2 = PerHalfRegion::new(HalfSide::U2, target[1][0], k, eps);
    let brute1 = brute_enumerate_half(&r1);
    let brute2 = brute_enumerate_half(&r2);
    let se1 = lll_se_enumerate_half(&r1);
    let se2 = lll_se_enumerate_half(&r2);
    if brute1 != se1 {
        // Find first divergence.
        let brute_set: BTreeSet<&[i64; 8]> = brute1.iter().collect();
        let se_set: BTreeSet<&[i64; 8]> = se1.iter().collect();
        let missing_in_se: Vec<&&[i64; 8]> = brute_set.difference(&se_set).collect();
        let extra_in_se: Vec<&&[i64; 8]> = se_set.difference(&brute_set).collect();
        panic!(
            "Part-3-(1) STOP {label} u1 k={k} ε={eps:e}: brute={} se={} \
             missing_in_se={:?} extra_in_se={:?}",
            brute1.len(),
            se1.len(),
            missing_in_se,
            extra_in_se,
        );
    }
    if brute2 != se2 {
        let brute_set: BTreeSet<&[i64; 8]> = brute2.iter().collect();
        let se_set: BTreeSet<&[i64; 8]> = se2.iter().collect();
        let missing_in_se: Vec<&&[i64; 8]> = brute_set.difference(&se_set).collect();
        let extra_in_se: Vec<&&[i64; 8]> = se_set.difference(&brute_set).collect();
        panic!(
            "Part-3-(1) STOP {label} u2 k={k} ε={eps:e}: brute={} se={} \
             missing_in_se={:?} extra_in_se={:?}",
            brute2.len(),
            se2.len(),
            missing_in_se,
            extra_in_se,
        );
    }
    eprintln!(
        "Part-3-(1) {label} k={k} ε={eps:e}: |brute1|={} |se1|={} |brute2|={} |se2|={} ✓",
        brute1.len(),
        se1.len(),
        brute2.len(),
        se2.len()
    );
}

#[test]
fn part3_1_fixtures_k2_eps_1e_1() {
    let target_phpe = (U2::<ZUpsilon>::h() * U2::p() * U2::h()).to_float();
    let target_hpsh = (U2::<ZUpsilon>::h() * U2::p() * U2::s() * U2::h()).to_float();
    check_lll_se_eq_brute(&target_phpe, 2, 1e-1, "H·P·H");
    check_lll_se_eq_brute(&target_hpsh, 2, 1e-1, "H·P·S·H");
}

#[test]
fn part3_1_fixtures_k3_eps_1e_1() {
    let target = (U2::<ZUpsilon>::h() * U2::p() * U2::h() * U2::p() * U2::h()).to_float();
    check_lll_se_eq_brute(&target, 3, 1e-1, "H·P·H·P·H");
}

#[test]
fn part3_1_haar_k2_3_eps_2e_1() {
    // Haar targets at k ∈ {2, 3}, ε=0.2 (wide-disc regime where pools have
    // multiple elements).
    for seed in 0..4_u64 {
        let target = haar_target(seed);
        for k in 2..=3 {
            check_lll_se_eq_brute(&target, k, 2e-1, &format!("Haar seed={seed}"));
        }
    }
}

#[test]
#[ignore = "slow: k=4 brute box is 7^8 ≈ 5.7M, ~30s/region"]
fn part3_1_haar_k4_eps_2e_1() {
    // k=4 box ~ 7^8 ≈ 5.76M. Tractable but slow.
    for seed in 0..2_u64 {
        let target = haar_target(seed);
        check_lll_se_eq_brute(&target, 4, 2e-1, &format!("Haar seed={seed}"));
    }
}

// ─── Part 3 step 2: Gates 1-3 re-run on the LLL-SE backend ─────────────────

fn gate1_fixture(label: &str, u: U2<ZUpsilon>, k: u32, expected_x: [i64; 16], eps: f64) {
    let target = u.to_float();
    let (u1, u2) = split_x(&expected_x);

    let r1 = PerHalfRegion::new(HalfSide::U1, target[0][0], k, eps);
    let r2 = PerHalfRegion::new(HalfSide::U2, target[1][0], k, eps);
    assert!(
        r1.contains(&u1),
        "Gate 1 (LLL-SE, {label}) u1={u1:?} NOT in region (k={k}, ε={eps:e})"
    );
    assert!(
        r2.contains(&u2),
        "Gate 1 (LLL-SE, {label}) u2={u2:?} NOT in region (k={k}, ε={eps:e})"
    );
    let target_norm: i64 = 1i64 << k;
    assert_eq!(complement(target_norm, key_of(&u1)), key_of(&u2));

    let pool1 = lll_se_enumerate_half(&r1);
    let pool2 = lll_se_enumerate_half(&r2);
    assert!(
        pool1.iter().any(|u| *u == u1),
        "Gate 1 (LLL-SE, {label}): backend did NOT emit u1 = {u1:?} (pool size {})",
        pool1.len()
    );
    assert!(
        pool2.iter().any(|u| *u == u2),
        "Gate 1 (LLL-SE, {label}): backend did NOT emit u2 = {u2:?} (pool size {})",
        pool2.len()
    );
    eprintln!(
        "Gate 1 (LLL-SE, {label}): k={k} u1∈ ✓ u2∈ ✓ pool1={} pool2={}",
        pool1.len(),
        pool2.len()
    );
}

#[test]
fn gate1_lll_se_p_h() {
    gate1_fixture(
        "P·H",
        U2::p() * U2::h(),
        1,
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        1e-2,
    );
}

#[test]
fn gate1_lll_se_h_p_h() {
    gate1_fixture(
        "H·P·H",
        U2::h() * U2::p() * U2::h(),
        2,
        [1, 1, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0],
        1e-2,
    );
}

#[test]
fn gate1_lll_se_h_p_s_h() {
    gate1_fixture(
        "H·P·S·H",
        U2::h() * U2::p() * U2::s() * U2::h(),
        2,
        [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, -1],
        1e-2,
    );
}

#[test]
fn gate1_lll_se_h_p_h_p_h() {
    gate1_fixture(
        "H·P·H·P·H",
        U2::h() * U2::p() * U2::h() * U2::p() * U2::h(),
        3,
        [1, 2, -1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
        1e-2,
    );
}

// Gate 2: brute ⊆ LLL-SE-MITM (orbit-aware)
fn gate2_brute_subset_lll_se_mitm(label: &str, u: U2<ZUpsilon>, k: u32, eps: f64) {
    let target = u.to_float();
    let mitm_cands: Vec<[i64; 16]> = lll_se_mitm_norm_bullet_set(&target, k, eps);
    let brute_all = phase1_brute(k);
    let mitm_orbit_expanded: BTreeSet<[i64; 16]> = mitm_cands
        .iter()
        .flat_map(|x| orbit_set(x).into_iter())
        .collect();

    let mut total_brute_leaves = 0usize;
    let mut missing: Vec<[i64; 16]> = Vec::new();
    for x in &brute_all {
        let (_, _, d) = best_phase(x, k, &target);
        if d <= eps {
            total_brute_leaves += 1;
            if !mitm_orbit_expanded.contains(x) {
                missing.push(*x);
            }
        }
    }
    if !missing.is_empty() {
        let x = missing[0];
        let (u1, u2) = split_x(&x);
        panic!(
            "Gate 2 (LLL-SE, {label}) STOP: dropped {} of {total_brute_leaves} brute solutions.\n\
             First miss: x={x:?}\n  u1={u1:?} key {:?}\n  u2={u2:?} key {:?}",
            missing.len(),
            key_of(&u1),
            key_of(&u2),
        );
    }
    eprintln!(
        "Gate 2 (LLL-SE, {label}): brute={} pass-align={} LLL-SE-MITM emitted={} (orbit={}) ✓",
        brute_all.len(),
        total_brute_leaves,
        mitm_cands.len(),
        mitm_orbit_expanded.len()
    );
}

#[test]
fn gate2_lll_se_p_h() {
    gate2_brute_subset_lll_se_mitm("P·H", U2::p() * U2::h(), 1, 1e-2);
}

#[test]
fn gate2_lll_se_h_p_h() {
    gate2_brute_subset_lll_se_mitm("H·P·H", U2::h() * U2::p() * U2::h(), 2, 1e-2);
}

#[test]
fn gate2_lll_se_h_p_s_h() {
    gate2_brute_subset_lll_se_mitm(
        "H·P·S·H",
        U2::h() * U2::p() * U2::s() * U2::h(),
        2,
        1e-2,
    );
}

#[test]
fn gate2_lll_se_h_p_h_p_h() {
    gate2_brute_subset_lll_se_mitm(
        "H·P·H·P·H",
        U2::h() * U2::p() * U2::h() * U2::p() * U2::h(),
        3,
        1e-2,
    );
}

// Gate 3: joint-SE ⊆ LLL-SE-MITM
#[test]
fn gate3_lll_se_eps2e1_k5() {
    use cyclosynth::synthesis::lattice_upsilon::{phase1 as phase1_se, LatticeScratch};
    use std::sync::atomic::AtomicBool;
    let eps = 2e-1_f64;
    let k = 5u32;
    let n_seeds = 3_u64;
    let mut tested = 0_u64;
    let mut total_se_hits = 0_u64;
    let mut total_missing = 0_u64;

    for seed in 0..n_seeds {
        let target = haar_target(seed);
        let v = [
            target[0][0].re,
            target[0][0].im,
            target[1][0].re,
            target[1][0].im,
        ];
        let mut scratch = LatticeScratch::new(eps);
        let budget_hit = AtomicBool::new(false);
        unsafe {
            std::env::set_var("CYCLOSYNTH_BULLET_PRUNE_N12", "0");
        }
        let se_sols = phase1_se(&mut scratch, v, k, eps, 5_000_000, &budget_hit);
        let mut se_filtered: Vec<[i64; 16]> = Vec::new();
        for x in &se_sols {
            let (_, _, d) = best_phase(x, k, &target);
            if d <= eps {
                se_filtered.push(*x);
            }
        }
        if se_filtered.is_empty() {
            continue;
        }
        let mitm_set: BTreeSet<[i64; 16]> = lll_se_mitm_norm_bullet_set(&target, k, eps)
            .iter()
            .flat_map(|x| orbit_set(x).into_iter())
            .collect();
        for x in &se_filtered {
            total_se_hits += 1;
            if !mitm_set.contains(x) {
                total_missing += 1;
                let (u1, u2) = split_x(x);
                panic!(
                    "Gate 3 (LLL-SE) STOP at seed={seed}: SE x={x:?} NOT in LLL-SE-MITM (|set|={}).\n  u1={u1:?} key {:?}\n  u2={u2:?} key {:?}",
                    mitm_set.len(),
                    key_of(&u1),
                    key_of(&u2),
                );
            }
        }
        tested += 1;
    }
    eprintln!(
        "Gate 3 (LLL-SE) ε={eps:e} k={k} N={n_seeds}: {tested} seeds with ≥1 SE hit, \
         {total_se_hits} SE leaves, {total_missing} missing → all SE ⊆ LLL-SE-MITM ✓"
    );
}
