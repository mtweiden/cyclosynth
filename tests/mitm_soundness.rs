//! Part 3 soundness gates for PROMPT_lattice_upsilon_mitm.md.
//!
//! Gate 1 (fixture-half containment): for each Gate-A fixture, both halves
//!   of the known x16 are inside their per-half region AND emitted by the
//!   brute half-enumerator.
//!
//! Gate 2 (brute ⊆ MITM): at small k (≤ 3), every joint-brute solution
//!   `(x ∈ phase1_brute(k))` whose reconstructed unitary equals the chosen
//!   target (so this `x` is genuinely a leaf for THAT target) must appear
//!   in the MITM output.
//!
//! Gate 3 (joint-SE ⊆ MITM): every solution the existing 16D SE finds at
//!   small k for a deterministic target must also be in the MITM output.
//!
//! Any drop → STOP and print: the point, its halves, their keys, and which
//! gate step lost it.

use cyclosynth::matrix::U2;
use cyclosynth::rings::ZUpsilon;
use cyclosynth::synthesis::lattice_upsilon::enumerate::{
    bullets_zero, norm_sqr_total, phase1_brute,
};
use cyclosynth::synthesis::lattice_upsilon::mitm::{
    brute_enumerate_half, brute_mitm_norm_bullet_set, complement, key_of, HalfSide, PerHalfRegion,
};
use cyclosynth::rings::types::Int;
use cyclosynth::synthesis::lattice_upsilon::synthesize::{best_phase, solution_to_unitary, zeta_pow};
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

/// Multiply an 8D ZUpsilon coeff vector by `ζ^j` (j ∈ 0..24). Reduces with
/// the cyclotomic relation `ζ⁸ = ζ⁴ − 1`. Output coefficients fit in i64
/// because the input does (max scale ≤ 2 from the reduction).
fn mul_zeta_pow(c: &[i64; 8], j: u32) -> [i64; 8] {
    let zu_in = cyclosynth::rings::ZUpsilon::new(
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

/// Apply the ζ^j orbit action to the full 16-tuple. The
/// `solution_to_unitary` reconstruction is invariant under
/// `(u₁, u₂) → (ζ^j · u₁, ζ^j · u₂)` (modulo a global phase that
/// `solution_to_unitary` then absorbs via its `phase` parameter), so the
/// brute oracle's set of 16-tuples is closed under this action.
fn orbit_member(x: &[i64; 16], j: u32) -> [i64; 16] {
    let (u1, u2) = split_x(x);
    let u1p = mul_zeta_pow(&u1, j);
    let u2p = mul_zeta_pow(&u2, j);
    let mut out = [0i64; 16];
    out[..8].copy_from_slice(&u1p);
    out[8..].copy_from_slice(&u2p);
    out
}

/// All 24 orbit members of `x` (sorted, deduplicated).
fn orbit_set(x: &[i64; 16]) -> BTreeSet<[i64; 16]> {
    (0..24).map(|j| orbit_member(x, j)).collect()
}

// ─── Gate 1 — fixture-half containment + emission ───────────────────────────

fn gate1_fixture(
    label: &str,
    u: U2<ZUpsilon>,
    k: u32,
    expected_x: [i64; 16],
    eps: f64,
) {
    let target = u.to_float();
    let (u1, u2) = split_x(&expected_x);

    let r1 = PerHalfRegion::new(HalfSide::U1, target[0][0], k, eps);
    let r2 = PerHalfRegion::new(HalfSide::U2, target[1][0], k, eps);

    assert!(
        r1.contains(&u1),
        "Gate 1 ({label}): u1 = {u1:?} NOT inside region (k={k}, ε={eps:e})"
    );
    assert!(
        r2.contains(&u2),
        "Gate 1 ({label}): u2 = {u2:?} NOT inside region (k={k}, ε={eps:e})"
    );

    // Key complement check (key arithmetic is integer-exact).
    let target_norm: i64 = 1i64 << k;
    let need = complement(target_norm, key_of(&u1));
    assert_eq!(
        need,
        key_of(&u2),
        "Gate 1 ({label}): u2 key {:?} ≠ complement of u1 key {:?}",
        key_of(&u2),
        need,
    );

    // Brute half-enumerator must actually emit both halves.
    let pool1 = brute_enumerate_half(&r1);
    let pool2 = brute_enumerate_half(&r2);
    assert!(
        pool1.iter().any(|u| *u == u1),
        "Gate 1 ({label}): brute half-enumerator did NOT emit u1 = {u1:?} (pool size {})",
        pool1.len()
    );
    assert!(
        pool2.iter().any(|u| *u == u2),
        "Gate 1 ({label}): brute half-enumerator did NOT emit u2 = {u2:?} (pool size {})",
        pool2.len()
    );

    eprintln!(
        "Gate 1 ({label}): k={k} u1∈region ✓ u2∈region ✓ pool1={} pool2={}",
        pool1.len(),
        pool2.len()
    );
}

#[test]
fn gate1_fixture_p_h() {
    let u: U2<ZUpsilon> = U2::p() * U2::h();
    gate1_fixture(
        "P·H",
        u,
        1,
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        1e-2,
    );
}

#[test]
fn gate1_fixture_h_p_h() {
    let u: U2<ZUpsilon> = U2::h() * U2::p() * U2::h();
    gate1_fixture(
        "H·P·H",
        u,
        2,
        [1, 1, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0],
        1e-2,
    );
}

#[test]
fn gate1_fixture_h_p_s_h() {
    let u: U2<ZUpsilon> = U2::h() * U2::p() * U2::s() * U2::h();
    gate1_fixture(
        "H·P·S·H",
        u,
        2,
        [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, -1],
        1e-2,
    );
}

#[test]
fn gate1_fixture_h_p_h_p_h() {
    let u: U2<ZUpsilon> = U2::h() * U2::p() * U2::h() * U2::p() * U2::h();
    gate1_fixture(
        "H·P·H·P·H",
        u,
        3,
        [1, 2, -1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
        1e-2,
    );
}

// ─── Gate 2 — brute ⊆ MITM ──────────────────────────────────────────────────
//
// For a chosen target U at lde k, the brute oracle `phase1_brute(k)`
// enumerates ALL norm+bullet-valid 16-tuples. Only those that ALSO satisfy
// the alignment threshold for THIS target are leaves; among those, every
// one must appear in the MITM output (otherwise MITM dropped a valid
// solution). We don't enforce alignment on the brute side — the MITM
// candidate set is a SUPERSET of the alignment-passing brute set (MITM's
// per-half region is the soundness cover) — but we DO verify the
// known-fixture solution is among them.

fn gate2_brute_subset_mitm(label: &str, u: U2<ZUpsilon>, k: u32, eps: f64) {
    let target = u.to_float();
    let mitm_cands: Vec<[i64; 16]> = brute_mitm_norm_bullet_set(&target, k, eps);
    let brute_all = phase1_brute(k);

    // Soundness check (orbit-aware, per PROMPT §3-(1) "up to the ζ^ℓ × ±
    // orbit"): every brute solution whose best-phase reconstruction is
    // within ε of the target must have SOME ζ^j orbit member in the MITM
    // output. The MITM region is set by `target`'s V_11/V_21, so only the
    // canonically-aligned orbit member lies in the per-half region; the
    // others reach the same projective unitary via a global ζ phase that
    // the solution_to_unitary's `phase` parameter then absorbs.
    let mitm_orbit_expanded: BTreeSet<[i64; 16]> = mitm_cands
        .iter()
        .flat_map(|x| orbit_set(x).into_iter())
        .collect();

    let mut total_brute_leaves = 0usize;
    let mut missing: Vec<[i64; 16]> = Vec::new();
    for x in &brute_all {
        let (_u_ring, _phase, d) = best_phase(x, k, &target);
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
            "Gate 2 ({label}) STOP: MITM (orbit-expanded) dropped {} of {total_brute_leaves} brute solutions.\n\
             First miss: x = {x:?}\n  u1 = {u1:?} key {:?}\n  u2 = {u2:?} key {:?}\n  needed complement at k={k}: {:?}\n  MITM emitted {} candidates ({} after orbit expansion)",
            missing.len(),
            key_of(&u1),
            key_of(&u2),
            complement(1i64 << k, key_of(&u1)),
            mitm_cands.len(),
            mitm_orbit_expanded.len(),
        );
    }
    eprintln!(
        "Gate 2 ({label}): brute returned {} norm+bullet, {} pass ε={eps:e} alignment to U; all in MITM (orbit-expanded: {}/{} raw) ✓",
        brute_all.len(),
        total_brute_leaves,
        mitm_orbit_expanded.len(),
        mitm_cands.len(),
    );
}

#[test]
fn gate2_brute_subset_mitm_p_h() {
    let u: U2<ZUpsilon> = U2::p() * U2::h();
    gate2_brute_subset_mitm("P·H", u, 1, 1e-2);
}

#[test]
fn gate2_brute_subset_mitm_h_p_h() {
    let u: U2<ZUpsilon> = U2::h() * U2::p() * U2::h();
    gate2_brute_subset_mitm("H·P·H", u, 2, 1e-2);
}

#[test]
fn gate2_brute_subset_mitm_h_p_s_h() {
    let u: U2<ZUpsilon> = U2::h() * U2::p() * U2::s() * U2::h();
    gate2_brute_subset_mitm("H·P·S·H", u, 2, 1e-2);
}

#[test]
fn gate2_brute_subset_mitm_h_p_h_p_h() {
    let u: U2<ZUpsilon> = U2::h() * U2::p() * U2::h() * U2::p() * U2::h();
    gate2_brute_subset_mitm("H·P·H·P·H", u, 3, 1e-2);
}

// ─── Gate 3 — joint-SE ⊆ MITM ───────────────────────────────────────────────
//
// On deterministic Haar targets at moderate ε / small-ish k, every
// solution the existing 16D SE path emits must also be in the MITM output
// for the SAME target and ε. We pick (ε, k) so the joint SE actually
// finds solutions and brute MITM is still tractable.

fn gate3_se_subset_mitm(eps: f64, k: u32, n_seeds: u64) {
    use cyclosynth::synthesis::lattice_upsilon::{phase1 as phase1_se, LatticeScratch};
    use std::sync::atomic::AtomicBool;
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
        // Disable bullet pruning to keep the comparison clean.
        unsafe { std::env::set_var("CYCLOSYNTH_BULLET_PRUNE_N12", "0"); }
        let se_sols = phase1_se(&mut scratch, v, k, eps, 5_000_000, &budget_hit);

        // Filter SE solutions to those that also pass joint alignment to
        // this target (the SE leaf check applies alignment internally, so
        // every emitted sol is alignment-passing — we just confirm).
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
        let mitm_set: BTreeSet<[i64; 16]> = brute_mitm_norm_bullet_set(&target, k, eps)
            .iter()
            .flat_map(|x| orbit_set(x).into_iter())
            .collect();

        for x in &se_filtered {
            total_se_hits += 1;
            if !mitm_set.contains(x) {
                total_missing += 1;
                let (u1, u2) = split_x(x);
                panic!(
                    "Gate 3 STOP at seed={seed} eps={eps:e} k={k}: SE emitted x = {x:?} \
                     NOT in MITM (|MITM|={}).\n  u1 = {u1:?} key {:?}\n  \
                     u2 = {u2:?} key {:?}",
                    mitm_set.len(),
                    key_of(&u1),
                    key_of(&u2),
                );
            }
        }
        tested += 1;
    }
    eprintln!(
        "Gate 3 (ε={eps:e}, k={k}, N={n_seeds}): {tested} seeds with ≥1 SE hit, \
         {total_se_hits} SE leaves total, {total_missing} missing → all SE ⊆ MITM ✓"
    );
}

#[test]
fn gate3_se_subset_mitm_eps2e1_k5() {
    // ε wide so SE actually finds hits on Haar targets; k=5 just above
    // BRUTE_K_MAX so the lattice SE path engages (not brute). At k=5 the
    // per-half brute box is 11^8 ≈ 200M — slow but finite (~30s/seed).
    gate3_se_subset_mitm(2e-1, 5, 3);
}
