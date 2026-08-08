//! Norm-equation step: solve t†t = ξ in Z[ω] for ξ ∈ Z[√2]
//! (RS arXiv:1403.2975 §6). Needs integer
//! factoring (Brent–Pollard ρ) and square roots mod p — Las Vegas with a
//! deterministic seeded RNG; on budget exhaustion a candidate is
//! reported unsolvable and the caller moves on (worst case a few extra
//! T, same contract as every gridsynth implementation).

use rug::ops::Pow;
use rug::{Complete, Integer};

use crate::synthesis::factor::{find_factor, is_prime, root_mod, sqrt_negative_one, Budget, Rng};

use crate::rings::real::ZRootTwo;
use crate::rings::zomega::ZOmegaBig;

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Unsolved {
    NoSolution,
    Unsolved,
}

/// Solve t†t ~ p for an integer prime p.
fn adj_decompose_int_prime(p: &Integer, rng: &mut Rng) -> Result<ZOmegaBig, Unsolved> {
    let p = p.clone().abs();
    if p == 0 || p == 1 {
        return Ok(ZOmegaBig::from_int(p));
    }
    if p == 2 {
        return Ok(ZOmegaBig::from_i64(0, 1, 0, -1));
    }
    let m8 = p.mod_u(8);
    if is_prime(&p) {
        if p.mod_u(4) == 1 {
            let Some(h) = sqrt_negative_one(&p, rng) else {
                return Err(Unsolved::Unsolved);
            };
            let t = ZOmegaBig::gcd(
                &ZOmegaBig::from_int(h).add(&ZOmegaBig::from_i64(0, 0, 1, 0)),
                &ZOmegaBig::from_int(p.clone()),
            );
            let tt = t.conj().mul(&t);
            let pz = ZOmegaBig::from_int(p);
            if tt == pz || tt == pz.neg() {
                Ok(t)
            } else {
                Err(Unsolved::Unsolved)
            }
        } else if m8 == 3 {
            let Some(h) = root_mod(&Integer::from(-2), &p, rng) else {
                return Err(Unsolved::Unsolved);
            };
            let t = ZOmegaBig::gcd(
                &ZOmegaBig::from_int(h).add(&ZOmegaBig::from_i64(0, 1, 0, 1)),
                &ZOmegaBig::from_int(p.clone()),
            );
            let tt = t.conj().mul(&t);
            let pz = ZOmegaBig::from_int(p);
            if tt == pz || tt == pz.neg() {
                Ok(t)
            } else {
                Err(Unsolved::Unsolved)
            }
        } else if m8 == 7 {
            if root_mod(&Integer::from(2), &p, rng).is_some() {
                Err(Unsolved::NoSolution)
            } else {
                Err(Unsolved::Unsolved)
            }
        } else {
            Err(Unsolved::Unsolved)
        }
    } else if m8 == 7 {
        if root_mod(&Integer::from(2), &p, rng).is_some() {
            Err(Unsolved::NoSolution)
        } else {
            Err(Unsolved::Unsolved)
        }
    } else {
        Err(Unsolved::Unsolved)
    }
}

fn adj_decompose_int_prime_power(
    p: &Integer,
    k: u32,
    rng: &mut Rng,
) -> Result<ZOmegaBig, Unsolved> {
    if k.is_multiple_of(2) {
        Ok(ZOmegaBig::from_int(p.clone().pow(k / 2)))
    } else {
        let t = adj_decompose_int_prime(p, rng)?;
        Ok(t.pow(u64::from(k)))
    }
}

/// Merge partial factorizations into pairwise-coprime factors (the
/// unit is discarded, as at the call sites).
fn decompose_relatively_int_prime(partial: Vec<(Integer, u32)>) -> Vec<(Integer, u32)> {
    let mut stack: Vec<(Integer, u32)> = partial.into_iter().rev().collect();
    let mut facs: Vec<(Integer, u32)> = Vec::new();
    while let Some((b, k_b)) = stack.pop() {
        let mut i = 0;
        loop {
            if i >= facs.len() {
                if b != 1 && b != -1 {
                    facs.push((b, k_b));
                }
                break;
            }
            let (a, k_a) = facs[i].clone();
            if a == b || a == (-&b).complete() {
                facs[i] = (a, k_a + k_b);
                break;
            }
            let g = a.clone().gcd(&b);
            if g == 1 {
                i += 1;
                continue;
            }
            let sub = vec![((&a / &g).complete(), k_a), (g.clone(), k_a + k_b)];
            let facs_a = decompose_relatively_int_prime(sub);
            facs[i] = facs_a[0].clone();
            facs.extend_from_slice(&facs_a[1..]);
            stack.push(((&b / &g).complete(), k_b));
            break;
        }
    }
    facs
}

/// Solve t†t ~ n for an integer n.
fn adj_decompose_int(n: &Integer, rng: &mut Rng, budget: &mut Budget) -> Option<ZOmegaBig> {
    let n = n.clone().abs();
    let mut facs: Vec<(Integer, u32)> = vec![(n, 1)];
    let mut t = ZOmegaBig::one();
    while let Some((p, k)) = facs.pop() {
        match adj_decompose_int_prime_power(&p, k, rng) {
            Ok(tp) => t = t.mul(&tp),
            Err(Unsolved::NoSolution) => return None,
            Err(Unsolved::Unsolved) => {
                if budget.factor_attempts == 0 {
                    return None;
                }
                budget.factor_attempts -= 1;
                match find_factor(&p, rng) {
                    None => facs.push((p, k)),
                    Some(f) => {
                        facs.push(((&p / &f).complete(), k));
                        facs.push((f, k));
                        facs = decompose_relatively_int_prime(facs);
                    }
                }
            }
        }
    }
    Some(t)
}

/// ξ ~ ξ̄ case.
fn adj_decompose_selfassociate(
    xi: &ZRootTwo,
    rng: &mut Rng,
    budget: &mut Budget,
) -> Option<ZOmegaBig> {
    if xi.is_zero() {
        return Some(ZOmegaBig::zero());
    }
    let n = xi.a.clone().gcd(&xi.b);
    let r = xi.divexact(&ZRootTwo::from_int(n.clone()));
    let t1 = adj_decompose_int(&n, rng, budget)?;
    let sqrt2 = ZRootTwo::from_i64(0, 1);
    let t2 = if r.rem(&sqrt2).is_zero() {
        ZOmegaBig::from_i64(1, 1, 0, 0)
    } else {
        ZOmegaBig::one()
    };
    Some(t1.mul(&t2))
}

/// Pairwise-coprime merge over Z[√2] (unit discarded at call site).
fn decompose_relatively_zroottwo_prime(
    partial: Vec<(ZRootTwo, u32)>,
) -> Vec<(ZRootTwo, u32)> {
    let mut stack: Vec<(ZRootTwo, u32)> = partial.into_iter().rev().collect();
    let mut facs: Vec<(ZRootTwo, u32)> = Vec::new();
    let one = ZRootTwo::one();
    while let Some((b, k_b)) = stack.pop() {
        let mut i = 0;
        loop {
            if i >= facs.len() {
                if !ZRootTwo::sim(&b, &one) {
                    facs.push((b, k_b));
                }
                break;
            }
            let (a, k_a) = facs[i].clone();
            if ZRootTwo::sim(&a, &b) {
                facs[i] = (a, k_a + k_b);
                break;
            }
            let g = ZRootTwo::gcd(&a, &b);
            if ZRootTwo::sim(&g, &one) {
                i += 1;
                continue;
            }
            let sub = vec![(a.divexact(&g), k_a), (g.clone(), k_a + k_b)];
            let facs_a = decompose_relatively_zroottwo_prime(sub);
            facs[i] = facs_a[0].clone();
            facs.extend_from_slice(&facs_a[1..]);
            stack.push((b.divexact(&g), k_b));
            break;
        }
    }
    facs
}

/// t†t ~ η for η ∈ Z[√2] with prime norm handling.
fn adj_decompose_zroottwo_prime(eta: &ZRootTwo, rng: &mut Rng) -> Result<ZOmegaBig, Unsolved> {
    let p = eta.norm().abs();
    if p == 0 || p == 1 {
        return Ok(ZOmegaBig::from_int(p));
    }
    if p == 2 {
        return Ok(ZOmegaBig::from_i64(0, 1, 0, -1));
    }
    let m8 = p.mod_u(8);
    let eta_z = ZOmegaBig::from_zroottwo(eta);
    if is_prime(&p) {
        if p.mod_u(4) == 1 {
            let Some(h) = sqrt_negative_one(&p, rng) else {
                return Err(Unsolved::Unsolved);
            };
            let t = ZOmegaBig::gcd(
                &ZOmegaBig::from_int(h).add(&ZOmegaBig::from_i64(0, 0, 1, 0)),
                &eta_z,
            );
            if ZOmegaBig::sim(&t.conj().mul(&t), &eta_z) {
                Ok(t)
            } else {
                Err(Unsolved::Unsolved)
            }
        } else if m8 == 3 {
            let Some(h) = root_mod(&Integer::from(-2), &p, rng) else {
                return Err(Unsolved::Unsolved);
            };
            let t = ZOmegaBig::gcd(
                &ZOmegaBig::from_int(h).add(&ZOmegaBig::from_i64(0, 1, 0, 1)),
                &eta_z,
            );
            if ZOmegaBig::sim(&t.conj().mul(&t), &eta_z) {
                Ok(t)
            } else {
                Err(Unsolved::Unsolved)
            }
        } else if m8 == 7 {
            if root_mod(&Integer::from(2), &p, rng).is_some() {
                Err(Unsolved::NoSolution)
            } else {
                Err(Unsolved::Unsolved)
            }
        } else {
            Err(Unsolved::Unsolved)
        }
    } else if m8 == 7 {
        if root_mod(&Integer::from(2), &p, rng).is_some() {
            Err(Unsolved::NoSolution)
        } else {
            Err(Unsolved::Unsolved)
        }
    } else {
        Err(Unsolved::Unsolved)
    }
}

fn adj_decompose_zroottwo_prime_power(
    eta: &ZRootTwo,
    k: u32,
    rng: &mut Rng,
) -> Result<ZOmegaBig, Unsolved> {
    if k.is_multiple_of(2) {
        Ok(ZOmegaBig::from_zroottwo(&eta.pow(i64::from(k / 2))))
    } else {
        let t = adj_decompose_zroottwo_prime(eta, rng)?;
        Ok(t.pow(u64::from(k)))
    }
}

/// gcd(ξ, ξ̄) = 1 case.
fn adj_decompose_selfcoprime(
    xi: &ZRootTwo,
    rng: &mut Rng,
    budget: &mut Budget,
) -> Option<ZOmegaBig> {
    let mut facs: Vec<(ZRootTwo, u32)> = vec![(xi.clone(), 1)];
    let mut t = ZOmegaBig::one();
    while let Some((eta, k)) = facs.pop() {
        match adj_decompose_zroottwo_prime_power(&eta, k, rng) {
            Ok(te) => t = t.mul(&te),
            Err(Unsolved::NoSolution) => return None,
            Err(Unsolved::Unsolved) => {
                if budget.factor_attempts == 0 {
                    return None;
                }
                budget.factor_attempts -= 1;
                let n = eta.norm().abs();
                match find_factor(&n, rng) {
                    None => facs.push((eta, k)),
                    Some(f) => {
                        let fac = ZRootTwo::gcd(xi, &ZRootTwo::from_int(f));
                        facs.push((eta.divexact(&fac), k));
                        facs.push((fac, k));
                        facs = decompose_relatively_zroottwo_prime(facs);
                    }
                }
            }
        }
    }
    Some(t)
}

/// Solve t†t ~ ξ.
fn adj_decompose(xi: &ZRootTwo, rng: &mut Rng, budget: &mut Budget) -> Option<ZOmegaBig> {
    if xi.is_zero() {
        return Some(ZOmegaBig::zero());
    }
    let d = ZRootTwo::gcd(xi, &xi.conj_sq2());
    let eta = xi.divexact(&d);
    let t1 = adj_decompose_selfassociate(&d, rng, budget)?;
    let t2 = adj_decompose_selfcoprime(&eta, rng, budget)?;
    Some(t1.mul(&t2))
}

/// Solve t†t = ξ exactly, or None.
pub(crate) fn diophantine(xi: &ZRootTwo, rng: &mut Rng, budget: &mut Budget) -> Option<ZOmegaBig> {
    if xi.is_zero() {
        return Some(ZOmegaBig::zero());
    }
    if xi.is_negative() || xi.conj_sq2().is_negative() {
        return None;
    }
    let t = adj_decompose(xi, rng, budget)?;
    let tt = t.conj().mul(&t);
    let xi_assoc = ZRootTwo::from_zomega_big(&tt);
    let u = xi.divexact(&xi_assoc);
    let v = u.sqrt()?;
    Some(ZOmegaBig::from_zroottwo(&v).mul(&t))
}
