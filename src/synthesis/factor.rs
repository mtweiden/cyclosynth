//! Gate-set-agnostic number theory for the norm-equation solvers:
//! deterministic SplitMix64 randomness, attempt budgets, Brent–Pollard ρ
//! factoring, Miller–Rabin primality, and square roots mod p (Cipolla /
//! Tonelli-style). Both native z-rotation routes consume this — the ring
//! Diophantine layers stay per gate set in their `norm_eq` modules.

// Scaffolding: consumers land incrementally in later commits.
#![cfg_attr(not(test), allow(dead_code))]

// The ρ-cap arithmetic (digit-count heuristics) casts through f64;
// every value is bounded by the 150k iteration cap.
#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use rug::integer::IsPrime;
use rug::{Complete, Integer};

/// SplitMix64 — deterministic, no external dependency.
pub(crate) struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform-ish in [1, n] (modulo bias irrelevant for Las Vegas trials).
    fn below(&mut self, n: &Integer) -> Integer {
        let bits = n.significant_bits() + 64;
        let mut x = Integer::new();
        let mut got = 0;
        while got < bits {
            x <<= 64;
            x += self.next_u64();
            got += 64;
        }
        x.div_rem_floor(n.clone()).1 + 1u32
    }
}

/// Attempt budget — deterministic, unlike a wall-clock cutoff, so
/// results are reproducible (each attempt is a full Brent-ρ pass).
pub(crate) struct Budget {
    pub factor_attempts: u32,
}

impl Default for Budget {
    fn default() -> Self {
        // Tuned on the oracle battery: hard-to-factor ξ are best abandoned
        // quickly (the k-loop supplies more candidates; worst case is the
        // usual +2 T from skipping one). 64 attempts × 4M-iteration ρ
        // passes cost minutes on adversarial norms for no T gain.
        Self { factor_attempts: 12 }
    }
}

/// One Brent–Pollard ρ attempt.
pub(crate) fn find_factor(n: &Integer, rng: &mut Rng) -> Option<Integer> {
    if n.is_even() && *n > 2 {
        return Some(Integer::from(2));
    }
    let a = rng.below(n);
    let mut y = a.clone();
    let mut r: u64 = 1;
    let mut k: u64 = 0;
    // L ≈ 1.1774·n^(1/4) iterations, capped to keep worst-case bounded
    // (a full-length ρ pass on a 60-digit composite is ~100 ms; anything
    // longer is better spent on the next grid candidate).
    let digits = n.to_string().len() as f64;
    let l_f = 10f64.powf(digits / 4.0) * 1.1774 + 10.0;
    let l = if l_f > 150_000.0 { 150_000u64 } else { l_f as u64 };
    const M: u64 = 128;
    loop {
        let x = (&y + n).complete();
        while k < r {
            let mut q = Integer::from(1);
            let y0 = y.clone();
            for _ in 0..M {
                y = (&y * &y).complete() + &a;
                y = y.div_rem_floor(n.clone()).1;
                q *= (&x - &y).complete();
                q = q.div_rem_floor(n.clone()).1;
                k += 1;
                if k == r {
                    break;
                }
            }
            let g = q.gcd(n);
            if g != 1 {
                if g == *n {
                    y = y0;
                    for _ in 0..M {
                        y = (&y * &y).complete() + &a;
                        y = y.div_rem_floor(n.clone()).1;
                        let g2 = (&x - &y).complete().gcd(n);
                        if g2 != 1 {
                            return if g2 == *n { None } else { Some(g2) };
                        }
                    }
                    return None;
                }
                return Some(g);
            }
            if k >= l {
                return None;
            }
        }
        r <<= 1;
    }
}

pub(crate) fn is_prime(n: &Integer) -> bool {
    n.is_probably_prime(30) != IsPrime::No
}

fn pow_mod(b: &Integer, e: &Integer, m: &Integer) -> Integer {
    b.clone().pow_mod(e, m).expect("nonnegative exponent")
}

/// √(−1) mod p for p ≡ 1 (mod 4).
pub(crate) fn sqrt_negative_one(p: &Integer, rng: &mut Rng) -> Option<Integer> {
    for _ in 0..100 {
        let b = rng.below(&(p - 1u32).complete());
        let e = (p - 1u32).complete() >> 2u32;
        let h = pow_mod(&b, &e, p);
        let r = (&h * &h).complete().div_rem_floor(p.clone()).1;
        if r == (p - 1u32).complete() {
            return Some(h);
        } else if r != 1 {
            return None;
        }
    }
    None
}

/// Square root of x mod p via Cipolla in F_p².
pub(crate) fn root_mod(x: &Integer, p: &Integer, rng: &mut Rng) -> Option<Integer> {
    let x = x.div_rem_euc_ref(p).complete().1;
    if *p == 2 {
        return Some(x);
    }
    if x == 0 {
        return Some(Integer::new());
    }
    if p.is_even() {
        return None;
    }
    let half = (p - 1u32).complete() >> 1u32;
    if pow_mod(&x, &half, p) != 1 {
        return None;
    }
    for _ in 0..100 {
        let b = rng.below(&(p - 1u32).complete());
        if pow_mod(&b, &(p - 1u32).complete(), p) != 1 {
            return None; // p not prime after all
        }
        let base = ((&b * &b).complete() + p - &x).div_rem_euc(p.clone()).1;
        if pow_mod(&base, &half, p) != 1 {
            // Cipolla: (b + √base)^((p+1)/2) in F_p[√base].
            let e = (p + 1u32).complete() >> 1u32;
            let (mut ra, mut rb) = (Integer::from(1), Integer::new());
            let (mut ba, mut bb) = (b.clone(), Integer::from(1));
            let mut n = e;
            while n != 0 {
                if n.is_odd() {
                    let na = ((&ra * &ba).complete()
                        + ((&rb * &bb).complete() * &base).div_rem_euc_ref(p).complete().1)
                        .div_rem_euc(p.clone())
                        .1;
                    let nb = ((&ra * &bb).complete() + (&rb * &ba).complete())
                        .div_rem_euc(p.clone())
                        .1;
                    ra = na;
                    rb = nb;
                }
                let na = ((&ba * &ba).complete()
                    + ((&bb * &bb).complete() * &base).div_rem_euc_ref(p).complete().1)
                    .div_rem_euc(p.clone())
                    .1;
                let nb = ((&ba * &bb).complete() * 2u32).div_rem_euc(p.clone()).1;
                ba = na;
                bb = nb;
                n >>= 1;
            }
            return Some(ra);
        }
    }
    None
}

