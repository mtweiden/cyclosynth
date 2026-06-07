//! Single-qubit exact synthesis over `Z[ζ₂₄]` (n=12 case).
//!
//! **Gate-set decision (resolved).** `ζ₂₄` is in the Forest–Gosset–Kliuchnikov–
//! McKinnon "golden set" `{2, 4, 6, 8, 12}` (J. Math. Phys. 56, 082201, 2015):
//! `G₁₂ = U₂(Z[ζ₂₄, 1/2])` is exactly the ancilla-free reachable group. Ring
//! membership ⟺ synthesizability, **no extra leaf check**. The denominator
//! generator is `√2` (forced because `√2 = ζ³ + ζ⁻³ ∈ Z[ζ₂₄]` and `Z[ζ₂₄, 1/2]
//! = Z[ζ₂₄, 1/√2]`). The synthesis radius is `2^k`. n=12 is the *largest* n in
//! the golden set; n≥16 fails and requires catalytic ancillas.
//!
//! ## Reconstruction (SPEC §6)
//!
//! Given an integer 16-vector `sol = [u₁-coeffs(8), u₂-coeffs(8)]` and a
//! phase index `ℓ ∈ 0..24`,
//!
//! ```text
//!   U_ℓ = [[ u₁,  −conj(u₂) · ζ^ℓ ],
//!          [ u₂,   conj(u₁) · ζ^ℓ ]]  /  √2^k
//! ```
//!
//! The 24 phase branches sweep the full `Z[ζ₂₄]` unit-of-determinant
//! coset. The synthesizer picks the `ℓ` minimizing the diamond distance
//! to the target unitary.

use crate::matrix::u2::U2;
use crate::rings::types::Int;
use crate::rings::ZUpsilon;
use crate::synthesis::lattice_upsilon::enumerate::{phase1_brute, phase1_brute_first};
use num_complex::Complex64;

/// Number of ζ₂₄ phase branches in the U reconstruction sweep (SPEC §6).
pub const NUM_PHASES: usize = 24;

/// Convert an `i64` lattice coordinate into the ring's `Int` type.
#[inline]
fn to_int(x: i64) -> Int {
    Int::from_i64(x)
}

/// Build a `ZUpsilon` from 8 i64 cyclotomic coords.
#[inline]
fn zu_from_coeffs(c: &[i64]) -> ZUpsilon {
    debug_assert_eq!(c.len(), 8);
    ZUpsilon::new(
        to_int(c[0]),
        to_int(c[1]),
        to_int(c[2]),
        to_int(c[3]),
        to_int(c[4]),
        to_int(c[5]),
        to_int(c[6]),
        to_int(c[7]),
    )
}

/// Compute `ζ^ℓ` for `ℓ ∈ 0..24`. Uses `ζ²⁴ = 1`.
pub fn zeta_pow(l: u32) -> ZUpsilon {
    let mut out = ZUpsilon::ONE;
    let mut z = ZUpsilon::ZETA;
    let mut e = l % 24;
    while e > 0 {
        if e & 1 == 1 {
            out = out * z;
        }
        z = z * z;
        e >>= 1;
    }
    out
}

/// Reconstruct the full unitary `U_ℓ` over `Z[ζ₂₄]` from a lattice
/// solution and a phase index. The denominator is `√2^k` (forced by
/// ζ₂₄'s golden-set membership; see module docs).
///
/// `phase ∈ 0..24` sweeps the SPEC §6 ζ^ℓ branch.
pub fn solution_to_unitary(sol: &[i64; 16], k: u32, phase: u32) -> U2<ZUpsilon> {
    let u1 = zu_from_coeffs(&sol[0..8]);
    let u2 = zu_from_coeffs(&sol[8..16]);
    let phase_factor = zeta_pow(phase);
    let u12 = -(u2.conj() * phase_factor);
    let u22 = u1.conj() * phase_factor;
    U2::new(u1, u12, u2, u22, k)
}

/// Search the 24 phase branches and return the one minimizing diamond
/// distance to `target`. Returns `(unitary, phase, distance)`.
pub fn best_phase(
    sol: &[i64; 16],
    k: u32,
    target: &[[Complex64; 2]; 2],
) -> (U2<ZUpsilon>, u32, f64) {
    use crate::synthesis::distance::diamond_distance_float;
    let mut best: Option<(U2<ZUpsilon>, u32, f64)> = None;
    for phase in 0..NUM_PHASES as u32 {
        let u = solution_to_unitary(sol, k, phase);
        let d = diamond_distance_float(&u.to_float(), target);
        match &best {
            None => best = Some((u, phase, d)),
            Some((_, _, db)) if d < *db => best = Some((u, phase, d)),
            _ => {}
        }
    }
    best.expect("phase loop ran at least once")
}

/// Output of [`synthesize`].
#[derive(Debug, Clone)]
pub struct SynthResult {
    /// Reconstructed unitary `U_ℓ` over Z[ζ₂₄] at denominator `√2^k`.
    pub u: U2<ZUpsilon>,
    /// Solution vector `[u₁-coeffs(8), u₂-coeffs(8)]`.
    pub solution: [i64; 16],
    /// Selected phase branch `ℓ ∈ 0..24`.
    pub phase: u32,
    /// Diamond distance from the reconstructed unitary to the target.
    pub distance: f64,
}

/// Dispatch threshold: `k ≤ BRUTE_K_MAX` uses brute force (which is
/// exhaustive and correctness-validated at small k); larger k goes
/// through LLL+SE so it terminates in finite time.
pub const BRUTE_K_MAX: u32 = 4;

/// Synthesize a single-qubit unitary at denominator `√2^k` to within
/// diamond distance `eps`.
///
/// **Dispatch.** For `k ≤ BRUTE_K_MAX` uses the exhaustive
/// [`phase1_brute`]; for larger `k` switches to LLL+SE
/// ([`super::phase1`]) so the call terminates in finite time even at
/// k=40 (where brute would enumerate ~2·2^40 Euclidean points and hang).
///
/// The constraint set is **norm shell + three bullets + alignment**, full
/// stop. ζ₂₄ being in the golden set (see module docs) means ring
/// membership is the entire reachability condition; no extra leaf check.
/// Returns the best (lowest-distance) reconstruction found across all 24
/// phase branches, or `None` if no candidates fall within `eps`.
pub fn synthesize(target: &[[Complex64; 2]; 2], k: u32, eps: f64) -> Option<SynthResult> {
    let sols: Vec<[i64; 16]> = if k <= BRUTE_K_MAX {
        phase1_brute(k)
    } else {
        // Pull v = (Re V₁₁, Im V₁₁, Re V₂₁, Im V₂₁) for the lattice search.
        let v = [
            target[0][0].re,
            target[0][0].im,
            target[1][0].re,
            target[1][0].im,
        ];
        let mut scratch = super::LatticeScratch::new(eps);
        let budget_hit = std::sync::atomic::AtomicBool::new(false);
        super::phase1(&mut scratch, v, k, eps, 100_000_000, &budget_hit)
    };
    let mut best: Option<SynthResult> = None;
    for sol in &sols {
        let (u, phase, d) = best_phase(sol, k, target);
        if d <= eps {
            match &best {
                None => {
                    best = Some(SynthResult {
                        u,
                        solution: *sol,
                        phase,
                        distance: d,
                    })
                }
                Some(b) if d < b.distance => {
                    best = Some(SynthResult {
                        u,
                        solution: *sol,
                        phase,
                        distance: d,
                    })
                }
                _ => {}
            }
        }
    }
    best
}

/// `max_solutions = 1` short-circuit synthesizer: returns on the first
/// candidate (`sol`, `phase`) that meets `eps`. Matches the bandb5/7.py
/// short-circuit semantics. Faster than [`synthesize`] when a "good
/// enough" answer is acceptable.
pub fn synthesize_first(target: &[[Complex64; 2]; 2], k: u32, eps: f64) -> Option<SynthResult> {
    // Walk the brute enumerator one sol at a time; on each, try all 24
    // phases; return the first (sol, phase) within eps.
    if k <= BRUTE_K_MAX {
        let sols = phase1_brute(k);
        for sol in &sols {
            let (u, phase, d) = best_phase(sol, k, target);
            if d <= eps {
                return Some(SynthResult {
                    u,
                    solution: *sol,
                    phase,
                    distance: d,
                });
            }
        }
        let _ = phase1_brute_first(k);
        return None;
    }

    let v = [
        target[0][0].re,
        target[0][0].im,
        target[1][0].re,
        target[1][0].im,
    ];
    let mut scratch = super::LatticeScratch::new(eps);
    let budget_hit = std::sync::atomic::AtomicBool::new(false);
    let mut hit: Option<SynthResult> = None;
    let max_leaves: u64 = std::env::var("CYCLOSYNTH_N12_MAX_LEAVES")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or_else(|| if eps <= 1e-5 { 2_000_000 } else { 100_000_000 });
    let _ = super::phase1_with_stop(&mut scratch, v, k, eps, max_leaves, &budget_hit, |sol| {
        let (u, phase, d) = best_phase(sol, k, target);
        if d <= eps {
            hit = Some(SynthResult {
                u,
                solution: *sol,
                phase,
                distance: d,
            });
            true
        } else {
            false
        }
    });
    hit
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeta_pow_basic() {
        assert_eq!(zeta_pow(0), ZUpsilon::ONE);
        assert_eq!(zeta_pow(1), ZUpsilon::ZETA);
        assert_eq!(zeta_pow(6), ZUpsilon::I);
        assert_eq!(zeta_pow(12), ZUpsilon::NEG_ONE);
        assert_eq!(zeta_pow(24), ZUpsilon::ONE);
        // ζ¹⁵ = ζ¹² · ζ³ = -ζ³.
        let zeta3 = ZUpsilon::ZETA * ZUpsilon::ZETA * ZUpsilon::ZETA;
        assert_eq!(zeta_pow(15), -zeta3);
    }

    /// Trivial unitary: `u₁=1, u₂=0, k=0`. With phase=0, `U = [[1,0],[0,1]] = I`.
    /// With phase=ℓ, `U = [[1, 0],[0, ζ^ℓ]]` — the diagonal R_z(ℓπ/12) phase
    /// gate.
    #[test]
    fn solution_to_unitary_identity_phase_zero() {
        let mut sol = [0i64; 16];
        sol[0] = 1; // u₁ = 1
        let u = solution_to_unitary(&sol, 0, 0);
        assert_eq!(u.u11, ZUpsilon::ONE);
        assert_eq!(u.u12, ZUpsilon::ZERO);
        assert_eq!(u.u21, ZUpsilon::ZERO);
        assert_eq!(u.u22, ZUpsilon::ONE);
        assert_eq!(u.k, 0);
    }

    #[test]
    fn solution_to_unitary_p_gate_phase_one() {
        let mut sol = [0i64; 16];
        sol[0] = 1; // u₁ = 1
                    // phase=1 ⇒ u₂₂ = conj(1) · ζ = ζ.
        let u = solution_to_unitary(&sol, 0, 1);
        assert_eq!(u.u11, ZUpsilon::ONE);
        assert_eq!(u.u12, ZUpsilon::ZERO);
        assert_eq!(u.u21, ZUpsilon::ZERO);
        assert_eq!(u.u22, ZUpsilon::ZETA);
    }

    /// At k=0 (denominator 1), `phase1_brute(0)` returns the trivial unit
    /// solutions; we can recover the identity matrix exactly.
    #[test]
    fn synthesize_identity_recovers_exact() {
        let id: [[Complex64; 2]; 2] = [
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        ];
        let result = synthesize(&id, 0, 1e-9).expect("identity should be recoverable at k=0");
        assert!(
            result.distance < 1e-9,
            "id recovery distance = {}",
            result.distance
        );
    }

    /// Diagonal R_z(π/12) phase gate `diag(1, ζ)` is reachable at k=0 via
    /// `u₁=1, u₂=0, phase=1`.
    #[test]
    fn synthesize_p_gate_recovers_exact() {
        let p: [[Complex64; 2]; 2] = [
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [
                Complex64::new(0.0, 0.0),
                Complex64::from_polar(1.0, std::f64::consts::PI / 12.0),
            ],
        ];
        let result = synthesize(&p, 0, 1e-9).expect("P gate should be recoverable at k=0");
        assert!(
            result.distance < 1e-9,
            "P recovery distance = {}",
            result.distance
        );
    }

    /// Round-trip a handful of `G₁₂ = U₂(Z[ζ₂₄, 1/2])` elements built as
    /// explicit products of `{H, S, P=R_z(π/12)}`. The golden-set theorem
    /// (FGKM 2015) guarantees recovery, so any failure here is a real
    /// bug, not a missing condition. Check passes only on **exact ring
    /// equality up to a single global phase** `ζ^j ∈ μ₂₄`.
    #[test]
    fn round_trip_clifford_level_unitaries() {
        fn round_trip(label: &str, target: U2<ZUpsilon>) {
            let target_float = target.to_float();
            let result = synthesize(&target_float, target.k, 1e-9)
                .unwrap_or_else(|| panic!("{label}: synthesize returned None at k={}", target.k));
            assert!(
                result.distance < 1e-10,
                "{label}: distance {} too large",
                result.distance
            );
            // The reconstructed U is congruent to target modulo a global
            // phase ζ^j (because the (u₁,u₂)/√2^k column representation
            // is identical up to multiplying the whole U by a unit of
            // Z[ζ₂₄]). Search for that j explicitly to assert *ring*
            // equality, not just float closeness.
            let exact = (0..24u32).any(|j| {
                let phase = zeta_pow(j);
                target.u11 * phase == result.u.u11
                    && target.u12 * phase == result.u.u12
                    && target.u21 * phase == result.u.u21
                    && target.u22 * phase == result.u.u22
            });
            assert!(
                exact,
                "{label}: no global phase ζ^j makes target == result \
                 entry-wise (k={}); u11_target={} u11_result={}",
                target.k, target.u11, result.u.u11
            );
        }

        let h: U2<ZUpsilon> = U2::h();
        let s: U2<ZUpsilon> = U2::s();
        let p: U2<ZUpsilon> = U2::p();

        // k=0 / k=1 cases (cheap)
        round_trip("I", U2::<ZUpsilon>::eye());
        round_trip("P", p);
        round_trip("S", s);
        round_trip("S·P", s * p);
        round_trip("P·P", p * p);
        round_trip("H", h);
        round_trip("H·P", h * p);
        round_trip("H·S", h * s);
        // One k=2 representative — exercises the brute force shell with
        // multiple solutions and a non-trivial bullet leaf-prune.
        round_trip("H·P·H", h * p * h);
    }

    /// 24 distinct phase branches give 24 distinct unitaries on a
    /// nontrivial solution.
    #[test]
    fn phase_sweep_covers_24_branches() {
        let mut sol = [0i64; 16];
        sol[0] = 1;
        let mut last_u22 = None;
        let mut seen = std::collections::HashSet::new();
        for phase in 0..NUM_PHASES as u32 {
            let u = solution_to_unitary(&sol, 0, phase);
            // u22 = ζ^phase, all distinct in 0..24.
            seen.insert(u.u22);
            last_u22 = Some(u.u22);
        }
        assert_eq!(
            seen.len(),
            24,
            "expected 24 distinct u22 values across phases"
        );
        assert!(last_u22.is_some());
    }
}
