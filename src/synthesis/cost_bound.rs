//! Certified cost-vs-lde lower bound for Clifford+√T circuits.
//!
//! `L(k) = cost_lb_half_units(k)` lower-bounds the weighted gate cost
//! (half-units `2·T_count + q_cost_x2·Q_count`, default `q_cost_x2 = 7`)
//! of any Clifford+√T unitary with reduced denominator exponent (lde)
//! `k`. It powers the certified search cutoff and the sound prefix
//! prune. Monotone non-decreasing in `k`, which the cutoff relies on.
//! The derivation lives on [`cost_lb_half_units`].

/// Maximum reduced lde over the single-qubit Clifford group in the
/// `U2Q` representation (⟨H, S⟩ closure, incl. phases): 1 under full
/// denominator reduction (H itself). `tests::clifford_lde_max_is_1`.
pub const CLIFFORD_LDE_MAX: u32 = 1;

/// Per-x/y-syllable lde contribution. `tests::syllable_lde_constants`.
pub const XY_SYLLABLE_LDE: u32 = 2;

/// Minimum half-unit cost over the 9 syllables (the T syllable costs 2;
/// Q costs 7; TQ costs 9).
const MIN_SYLLABLE_COST_HALF_UNITS: usize = 2;

/// Certified lower bound, in half-units, on the weighted cost of any
/// Clifford+√T unitary with reduced lde `k`:
///
///   c̃ = 2t + 7q ≥ 2·(t + 2q) ≥ 2·N ≥ 2·(2k − 3) = 4k − 6,
///
/// where N is the reduced Bloch/SO(3) denominator exponent:
///   * `t + 2q ≥ N` — Bloch-exponent subadditivity with per-gate
///     constants N(T) = 1, N(Q) = 2, N(Clifford) = 0, holding for every
///     circuit;
///   * `N ≥ 2k − 3` — adjugate argument + √2/λ conversion lemma
///     tight with deficit 3 at every k ≥ 3.
///
/// The syllable-count floor (≈ k − 1, the `max` arm) only binds at k ≤ 2.
pub fn cost_lb_half_units(k: u32) -> usize {
    let excess = k.saturating_sub(CLIFFORD_LDE_MAX);
    // n_xy ≥ ⌈excess / XY_SYLLABLE_LDE⌉, each costing ≥ 2 half-units.
    let n_xy = excess.div_ceil(XY_SYLLABLE_LDE) as usize;
    let syllable_floor = MIN_SYLLABLE_COST_HALF_UNITS * n_xy;
    let slope2_floor = (4 * k as usize).saturating_sub(6);
    syllable_floor.max(slope2_floor)
}

/// Per-det-phase-class cost lower bound, in half-units. Q syllables
/// contribute ζ₁₆ to det(U), T contributes ζ₁₆², Cliffords powers of
/// ζ₁₆⁴ — but a circuit matches a target only up to a global phase
/// ζ₁₆ʲ, which shifts the det class by 2j. Only the parity of `d`
/// survives, so an odd class forces ≥ 1 Q gate; even classes give nothing.
/// Callers add this to a prefix's own cost as a sound suffix lower bound. A
/// stronger mod-4 bound is NOT sound.
///
/// The `7` is one Q gate in half-units, i.e. the default `q_cost_x2` (see
/// [`crate::synthesis::clifford_sqrt_t`]). It is a sound lower bound only while
/// `q_cost_x2 ≥ 7`; if the Q weight is ever retuned below 3.5, this must track
/// `q_cost_x2` instead, or it would over-claim.
pub fn class_cost_lb_half_units(d: u32) -> usize {
    if d % 2 == 1 { 7 } else { 0 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::u2::U2Q;
    use std::collections::HashSet;

    fn rx(a: u32) -> U2Q {
        let mut d = U2Q::eye();
        for _ in 0..a {
            d = d * U2Q::q();
        }
        (U2Q::h() * d * U2Q::h()).reduced()
    }
    fn ry(a: u32) -> U2Q {
        let mut d = U2Q::eye();
        for _ in 0..a {
            d = d * U2Q::q();
        }
        (U2Q::s() * U2Q::h() * d * U2Q::h() * U2Q::s().dagger()).reduced()
    }
    fn rz(a: u32) -> U2Q {
        let mut d = U2Q::eye();
        for _ in 0..a {
            d = d * U2Q::q();
        }
        d
    }

    /// Exact per-syllable lde over all 9 syllable types: x/y syllables
    /// have lde exactly 2; z syllables 0.
    #[test]
    fn syllable_lde_constants() {
        for a in 1..=3u32 {
            assert_eq!(rz(a).k, 0, "R_z({a}·π/8) must have lde 0");
            assert_eq!(rx(a).k, XY_SYLLABLE_LDE, "R_x({a}·π/8) lde");
            assert_eq!(ry(a).k, XY_SYLLABLE_LDE, "R_y({a}·π/8) lde");
        }
    }

    /// Max lde over the ⟨H, S⟩ closure (the full single-qubit Clifford
    /// group in U2Q form, incl. phases).
    #[test]
    fn clifford_lde_max_is_1() {
        let gens = [U2Q::h(), U2Q::s()];
        let mut seen: HashSet<String> = HashSet::new();
        let mut frontier = vec![U2Q::eye()];
        seen.insert(format!("{:?}", U2Q::eye()));
        let mut max_lde = 0u32;
        while let Some(u) = frontier.pop() {
            max_lde = max_lde.max(u.k);
            for g in &gens {
                let v = (u * *g).reduced();
                let key = format!("{v:?}");
                if seen.insert(key) {
                    frontier.push(v);
                }
            }
            assert!(seen.len() <= 4000, "closure should be finite/small");
        }
        assert_eq!(max_lde, CLIFFORD_LDE_MAX,
            "Clifford closure max lde changed — update CLIFFORD_LDE_MAX \
             and the L(k) derivation");
    }

    /// Along the alternating R_x(T)·R_y(T) chain (s T gates, fully
    /// reduced), the staircase must stay at or below the realized cost,
    /// the submultiplicative premise lde ≤ 2s + c₀ must hold, and the
    /// reduced ladder is pinned so a change in reduction is caught.
    #[test]
    fn cost_bound_sound_on_alternating_chain() {
        let mut u = U2Q::eye();
        for s in 1..=20usize {
            u = (u * if s % 2 == 1 { rx(2) } else { ry(2) }).reduced();
            let realized_half_units = 2 * s; // s T gates
            assert!(
                u.k <= 2 * s as u32 + CLIFFORD_LDE_MAX,
                "chain lde {} exceeded the submultiplicative premise at s={s}", u.k
            );
            assert!(
                cost_lb_half_units(u.k) <= realized_half_units,
                "L({}) = {} exceeds a realized cost {}",
                u.k, cost_lb_half_units(u.k), realized_half_units
            );
        }
        // The reduced ladder is lde = s/2 + 1 for even s ≥ 4 (≈ 4
        // half-units of T per lde unit, 2× the provable staircase slope).
        assert_eq!(u.k, 11, "reduced lde of the s=20 alternating chain");
    }

    /// Monotonicity — required by the certified sweep cutoff
    /// (`stop at k once C* ≤ L(k+1)` assumes L never decreases).
    #[test]
    fn cost_bound_is_monotone() {
        for k in 0..200u32 {
            assert!(cost_lb_half_units(k) <= cost_lb_half_units(k + 1));
        }
    }

    /// Semantics check for [`class_cost_lb_half_units`]: every unitary
    /// on the brute shells must respect its det-phase class bound. A
    /// failure here means the det convention (`det_phase_of`) and the
    /// congruence derivation disagree.
    #[test]
    fn class_bound_holds_on_brute_shells() {
        use crate::synthesis::clifford_sqrt_t::{det_phase_of, solution_to_u2q};
        use crate::synthesis::decomposer::BlochDecomposer;
        use crate::synthesis::lattice::zeta::brute::enumerate_unitary_norm_shell;

        let mut checked = 0usize;
        for k in 0..=3u32 {
            for sol in &enumerate_unitary_norm_shell(k) {
                let u = solution_to_u2q(sol, k);
                let d = det_phase_of(&u.to_float());
                let gates = BlochDecomposer.decompose(&u);
                let t = gates.chars().filter(|&c| c == 'T').count();
                let q = gates.chars().filter(|&c| c == 'Q').count();
                assert_eq!(
                    q % 2,
                    (d as usize) % 2,
                    "Q-parity congruence violated at k={k}: d={d}, t={t}, q={q}, gates={gates}"
                );
                assert!(
                    2 * t + 7 * q >= class_cost_lb_half_units(d),
                    "class bound violated at k={k}: d={d}, cost={}",
                    2 * t + 7 * q
                );
                checked += 1;
            }
        }
        assert!(checked > 100, "expected to check many shell unitaries");
    }

    /// Brute-enumerate full shells at k ≤ 3, decompose every completion,
    /// and check the staircase never exceeds the cheapest realized cost.
    /// The brute minimum is an upper bound on the true L(k) (completions
    /// fix a convention), so this validates soundness, not exactness.
    #[test]
    fn cost_bound_below_brute_minimum_small_k() {
        use crate::synthesis::clifford_sqrt_t::solution_to_u2q;
        use crate::synthesis::decomposer::BlochDecomposer;
        use crate::synthesis::lattice::zeta::brute::enumerate_unitary_norm_shell;

        for k in 0..=3u32 {
            let sols = enumerate_unitary_norm_shell(k);
            let mut min_cost = usize::MAX;
            for sol in &sols {
                let u = solution_to_u2q(sol, k).reduced();
                if u.k != k {
                    continue; // completion reduced below the shell lde
                }
                let gates = BlochDecomposer.decompose(&u);
                let t = gates.chars().filter(|&c| c == 'T').count();
                let q = gates.chars().filter(|&c| c == 'Q').count();
                min_cost = min_cost.min(2 * t + 7 * q);
            }
            if min_cost != usize::MAX {
                assert!(
                    cost_lb_half_units(k) <= min_cost,
                    "L({k}) = {} exceeds brute minimum {min_cost}",
                    cost_lb_half_units(k)
                );
            }
        }
    }
}
