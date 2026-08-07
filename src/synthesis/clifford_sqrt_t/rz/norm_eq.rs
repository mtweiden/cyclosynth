//! Relative norm equation w·w̄ = ξ for the CM extension Z[ζ]/Z[g].
//!
//! Mirrors the Clifford+T Diophantine step (gridsynth/diophantine.rs)
//! with the prime constructions adapted to the ζ splitting law
//! (classes by p mod 16; the quadratic subfields Q(i) and Q(√−2) of
//! Q(ζ) supply the explicit generators):
//!   - the ramified part: (1−ζ)(1−ζ̄) = 2 − g,
//!   - p ≡ 1 (mod 4): t = gcd(h + i, π) with h² ≡ −1 (mod p), i = ζ⁴,
//!   - p ≡ 3 (mod 8): t = gcd(h + √−2, π) with h² ≡ −2, √−2 = ζ² + ζ⁶,
//!   - p ≡ 15 (mod 16), odd exponent: provably no solution (g-primes are
//!     inert in the CM step),
//!   - p ≡ 7 (mod 16), odd exponent: solvable in principle (residue-field
//!     F_p² construction) — UNSOLVED in v1; the candidate is skipped and
//!     the driver tries the next grid point.
//!
//! Every constructed factor is VERIFIED (t·t̄ ~ π) before use; failures
//! degrade to skips, never to wrong answers. The final unit is resolved
//! by a square root in Z[g] (relative norms of units are exactly the
//! squares of real units, since O*(Q(ζ)) = ⟨ζ⟩·O*(Q(g)) by Hasse's
//! unit index for prime-power cyclotomics).

use rug::{Complete, Integer};

use crate::synthesis::factor::{find_factor, is_prime, root_mod, sqrt_negative_one, Budget, Rng};
use crate::rings::real::ZRootTwoPlusRootTwo;
use crate::rings::zzeta::ZZetaBig;

const GCD_CAP: u32 = 128;

enum PrimeOutcome {
    Solved(ZZetaBig),
    NoSolution,
    Unsolved,
}

/// Associate test in Z[g].
fn sim_eta(a: &ZRootTwoPlusRootTwo, b: &ZRootTwoPlusRootTwo) -> bool {
    !a.is_zero() && !b.is_zero() && {
        let r1 = a.rem(b);
        let r2 = b.rem(a);
        r1.is_zero() && r2.is_zero()
    }
}

/// t with t·t̄ ~ π for a Z[g]-prime element π lying over the rational
/// prime p (any residue degree — the construction is verified, so a
/// wrong guess degrades to Unsolved).
fn solve_prime(pi: &ZRootTwoPlusRootTwo, p: &Integer, rng: &mut Rng) -> PrimeOutcome {
    let p_mod16 = p.mod_u(16);
    let pi_lift = ZZetaBig::from_real(pi);
    let verify = |t: &ZZetaBig| -> bool {
        let tt = t.rel_norm();
        sim_eta(&tt, pi)
    };
    if p_mod16 == 15 {
        return PrimeOutcome::NoSolution;
    }
    if p.mod_u(4) == 1 {
        // Z[i] route: i = ζ⁴.
        if let Some(h) = sqrt_negative_one(p, rng) {
            let i_elem = ZZetaBig::from_i64(0, 0, 0, 0, 1, 0, 0, 0); // i = ζ⁴
            let cand = ZZetaBig::from_int(h).add(&i_elem);
            if let Some(t) = ZZetaBig::gcd_capped(&cand, &pi_lift, GCD_CAP) {
                if verify(&t) {
                    return PrimeOutcome::Solved(t);
                }
            }
        }
        return PrimeOutcome::Unsolved;
    }
    if p.mod_u(8) == 3 {
        // Z[√−2] route: √−2 = ζ² + ζ⁶.
        if let Some(h) = root_mod(&Integer::from(-2), p, rng) {
            let s_elem = ZZetaBig::from_i64(0, 0, 1, 0, 0, 0, 1, 0); // √−2 = ζ² + ζ⁶
            let cand = ZZetaBig::from_int(h).add(&s_elem);
            if let Some(t) = ZZetaBig::gcd_capped(&cand, &pi_lift, GCD_CAP) {
                if verify(&t) {
                    return PrimeOutcome::Solved(t);
                }
            }
        }
        return PrimeOutcome::Unsolved;
    }
    if p_mod16 == 7 {
        // p ≡ 7 (mod 16): the g-prime π has residue field F_p² = F_p(g).
        // 2 is a QR (p ≡ −1 mod 8): s = √2 ∈ F_p, and g² ≡ 2+s with 2+s a
        // non-residue. The quadratic x² − gx + 1 has discriminant g² − 4 =
        // s − 2, whose square root in F_p² is either a ∈ F_p (a² = s−2) or
        // b·g (b² = (s−2)/(2+s)) — exactly one character works. Try both
        // signs of s and both root branches; each candidate is verified.
        let Some(s0) = root_mod(&Integer::from(2), p, rng) else {
            return PrimeOutcome::Unsolved;
        };
        let inv2 = {
            // 2⁻¹ mod p = (p+1)/2.
            ((p + 1u32).complete() >> 1u32).div_rem_euc(p.clone()).1
        };
        for s in [s0.clone(), (p - &s0).complete()] {
            let s_m2 = ((&s - 2u32).complete() + p).div_rem_euc(p.clone()).1;
            let s_p2 = ((&s + 2u32).complete()).div_rem_euc(p.clone()).1;
            // Branch 1: sqrt(s−2) = a ∈ F_p → x₀ = (a ± g)·2⁻¹.
            if let Some(a) = root_mod(&s_m2, p, rng) {
                for aa in [a.clone(), (p - &a).complete()] {
                    let alpha = ((&aa * &inv2).complete()).div_rem_euc(p.clone()).1;
                    let beta = inv2.clone();
                    if let Some(t) = try_fp2_root(&alpha, &beta, &pi_lift, pi) {
                        return PrimeOutcome::Solved(t);
                    }
                }
            }
            // Branch 2: sqrt(s−2) = b·g with b² = (s−2)/(2+s).
            if let Some(inv_sp2) = invert_mod(&s_p2, p) {
                let ratio = ((&s_m2 * &inv_sp2).complete()).div_rem_euc(p.clone()).1;
                if let Some(b) = root_mod(&ratio, p, rng) {
                    for bb in [b.clone(), (p - &b).complete()] {
                        // x₀ = (g + b·g)/2 = (1+b)·2⁻¹·g.
                        let beta =
                            ((Integer::from(1) + &bb) * &inv2).div_rem_euc(p.clone()).1;
                        let alpha = Integer::new();
                        if let Some(t) = try_fp2_root(&alpha, &beta, &pi_lift, pi) {
                            return PrimeOutcome::Solved(t);
                        }
                    }
                }
            }
        }
        return PrimeOutcome::Unsolved;
    }
    PrimeOutcome::Unsolved
}

/// Modular inverse via Fermat (p prime).
fn invert_mod(x: &Integer, p: &Integer) -> Option<Integer> {
    if x.is_divisible(p) {
        return None;
    }
    let e = (p - 2u32).complete();
    Some(x.clone().pow_mod(&e, p).expect("positive exponent"))
}

/// Try t = gcd(ζ − (α + β·g), π) as a relative-norm generator for π.
fn try_fp2_root(
    alpha: &Integer,
    beta: &Integer,
    pi_lift: &ZZetaBig,
    pi: &ZRootTwoPlusRootTwo,
) -> Option<ZZetaBig> {
    let g_lift = ZZetaBig::from_real(&ZRootTwoPlusRootTwo::generator());
    let x = ZZetaBig::from_int(alpha.clone()).add(&g_lift.scale(beta));
    let cand = ZZetaBig::zeta().sub(&x);
    let t = ZZetaBig::gcd_capped(&cand, pi_lift, GCD_CAP)?;
    let tt = t.rel_norm();
    if sim_eta(&tt, pi) {
        Some(t)
    } else {
        None
    }
}

/// Pairwise-coprime merge over Z[g] (port of the T-side stack; units
/// dropped — the final unit is fixed by the square-root step).
fn decompose_relatively(partial: Vec<(ZRootTwoPlusRootTwo, u32)>) -> Option<Vec<(ZRootTwoPlusRootTwo, u32)>> {
    let mut stack: Vec<(ZRootTwoPlusRootTwo, u32)> = partial.into_iter().rev().collect();
    let mut facs: Vec<(ZRootTwoPlusRootTwo, u32)> = Vec::new();
    let one = ZRootTwoPlusRootTwo::one();
    let mut guard = 0u32;
    while let Some((b, k_b)) = stack.pop() {
        guard += 1;
        if guard > 512 {
            return None;
        }
        let mut i = 0;
        loop {
            if i >= facs.len() {
                if !sim_eta(&b, &one) {
                    facs.push((b, k_b));
                }
                break;
            }
            let (a, k_a) = facs[i].clone();
            if sim_eta(&a, &b) {
                facs[i] = (a, k_a + k_b);
                break;
            }
            let g = ZRootTwoPlusRootTwo::gcd_capped(&a, &b, GCD_CAP)?;
            if sim_eta(&g, &one) {
                i += 1;
                continue;
            }
            let sub = vec![(a.divexact(&g), k_a), (g.clone(), k_a + k_b)];
            let facs_a = decompose_relatively(sub)?;
            if facs_a.is_empty() {
                facs.remove(i);
            } else {
                facs[i] = facs_a[0].clone();
                facs.extend_from_slice(&facs_a[1..]);
            }
            stack.push((b.divexact(&g), k_b));
            break;
        }
    }
    Some(facs)
}

/// Rational prime under a Z[g]-prime element: |N(π)| = p^f, f ∈ {1,2,4}.
fn rational_prime_under(pi: &ZRootTwoPlusRootTwo) -> Option<Integer> {
    let n = pi.norm().abs();
    if n <= 1 {
        return None;
    }
    for f in [1u32, 2, 4] {
        let (root, rem) = n.clone().root_rem(Integer::new(), f);
        if rem == 0 && is_prime(&root) {
            return Some(root);
        }
    }
    None
}

/// Solve t·t̄ ~ ξ up to a unit (the caller resolves the unit by sqrt).
fn adj_decompose(xi: &ZRootTwoPlusRootTwo, rng: &mut Rng, budget: &mut Budget) -> Option<ZZetaBig> {
    let mut facs: Vec<(ZRootTwoPlusRootTwo, u32)> = vec![(xi.clone(), 1)];
    let mut t = ZZetaBig::one();
    let mut guard = 0u32;
    while let Some((pi, e)) = facs.pop() {
        guard += 1;
        if guard > 256 {
            return None;
        }
        if sim_eta(&pi, &ZRootTwoPlusRootTwo::one()) {
            continue;
        }
        // Even exponent: π^e = (π^{e/2})·conj-free contribution.
        if e % 2 == 0 {
            let mut pw = ZZetaBig::one();
            let lift = ZZetaBig::from_real(&pi);
            for _ in 0..(e / 2) {
                pw = pw.mul(&lift);
            }
            t = t.mul(&pw);
            continue;
        }
        match rational_prime_under(&pi) {
            Some(p) => match solve_prime(&pi, &p, rng) {
                PrimeOutcome::Solved(tp) => {
                    let mut pw = ZZetaBig::one();
                    for _ in 0..e {
                        pw = pw.mul(&tp);
                    }
                    t = t.mul(&pw);
                }
                PrimeOutcome::NoSolution => return None,
                PrimeOutcome::Unsolved => return None,
            },
            None => {
                // Composite norm: split via a rational factor and re-merge.
                if budget.factor_attempts == 0 {
                    return None;
                }
                budget.factor_attempts -= 1;
                let n = pi.norm().abs();
                let Some(f) = find_factor(&n, rng) else {
                    facs.push((pi, e));
                    continue;
                };
                let g = ZRootTwoPlusRootTwo::gcd_capped(&pi, &ZRootTwoPlusRootTwo::from_int(f), GCD_CAP)?;
                if sim_eta(&g, &ZRootTwoPlusRootTwo::one()) || sim_eta(&g, &pi) {
                    // Unlucky split — retry with remaining budget.
                    facs.push((pi, e));
                    continue;
                }
                facs.push((pi.divexact(&g), e));
                facs.push((g, e));
                let merged = decompose_relatively(facs)?;
                facs = merged;
            }
        }
    }
    Some(t)
}

/// Solve w·w̄ = ξ exactly over Z[ζ], or None (no solution / gave up).
pub(crate) fn solve_rel_norm(xi: &ZRootTwoPlusRootTwo, rng: &mut Rng, budget: &mut Budget) -> Option<ZZetaBig> {
    if xi.is_zero() {
        return Some(ZZetaBig::zero());
    }
    if !xi.is_totally_nonneg() {
        return None;
    }
    // Ramified part: strip (2−g) factors, each contributing (1−ζ).
    let two_minus_eta = ZRootTwoPlusRootTwo::from_i64(2, -1, 0, 0);
    let one_minus_zeta = ZZetaBig::one().sub(&ZZetaBig::zeta());
    let mut rest = xi.clone();
    let mut t = ZZetaBig::one();
    loop {
        let (q, r) = rest.divmod(&two_minus_eta);
        if r.is_zero() {
            rest = q;
            t = t.mul(&one_minus_zeta);
        } else {
            break;
        }
    }
    let t2 = adj_decompose(&rest, rng, budget)?;
    let t = t.mul(&t2);
    // Unit fix: υ = ξ / (t·t̄) must be the square of a real unit.
    let tt = t.rel_norm();
    if tt.is_zero() {
        return None;
    }
    let (upsilon, rem) = xi.divmod(&tt);
    if !rem.is_zero() {
        return None;
    }
    let mu = upsilon.sqrt()?;
    let w = t.mul(&ZZetaBig::from_real(&mu));
    // Exact final verification — the only contract that matters.
    if w.rel_norm() == *xi {
        Some(w)
    } else {
        None
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn rng() -> Rng {
        Rng::new(0xFEED)
    }

    /// Round-trip: for random w, solve for ξ = w·w̄ and verify the
    /// returned solution has the same relative norm.
    #[test]
    fn test_round_trip_random() {
        let mut s = 0x1234_5678_u64;
        let mut next = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) % 5) as i64 - 2
        };
        let mut solved = 0u32;
        let mut tried = 0u32;
        for _ in 0..60 {
            let a = ZRootTwoPlusRootTwo::from_i64(next(), next(), next(), next());
            let b = ZRootTwoPlusRootTwo::from_i64(next(), next(), next(), next());
            let w = ZZetaBig::from_module(&a, &b);
            if w.is_zero() {
                continue;
            }
            let xi = w.rel_norm();
            tried += 1;
            let mut r = rng();
            let mut budget = Budget::default();
            if let Some(w2) = solve_rel_norm(&xi, &mut r, &mut budget) {
                assert_eq!(w2.rel_norm(), xi, "wrong solution for xi={xi:?}");
                solved += 1;
            }
        }
        // Every ξ here IS a relative norm by construction and all residue
        // classes are handled; only factoring-budget exhaustion may skip.
        assert!(
            solved * 10 >= tried * 9,
            "solved only {solved}/{tried} — construction too weak"
        );
    }

    /// The ramified generator: (1−ζ)(1−ζ̄) = 2 − g.
    #[test]
    fn test_ramified_generator() {
        let omz = ZZetaBig::one().sub(&ZZetaBig::zeta());
        assert_eq!(omz.rel_norm(), ZRootTwoPlusRootTwo::from_i64(2, -1, 0, 0));
    }

    /// Small hand cases: ξ = 1, 2, 2−g, and a known composite.
    #[test]
    fn test_small_cases() {
        let mut r = rng();
        let mut budget = Budget::default();
        let one = ZRootTwoPlusRootTwo::one();
        let w = solve_rel_norm(&one, &mut r, &mut budget).expect("1 solvable");
        assert_eq!(w.rel_norm(), one);
        let two = ZRootTwoPlusRootTwo::from_i64(2, 0, 0, 0);
        let w2 = solve_rel_norm(&two, &mut r, &mut budget).expect("2 solvable");
        assert_eq!(w2.rel_norm(), two);
        let tme = ZRootTwoPlusRootTwo::from_i64(2, -1, 0, 0);
        let w3 = solve_rel_norm(&tme, &mut r, &mut budget).expect("2−g solvable");
        assert_eq!(w3.rel_norm(), tme);
    }
}
