//! Dyadic modules over the ring layer: elements α/√2^k with tracked
//! denominator exponents — [`DRootTwo`] over Z[√2] and [`DOmega`] over
//! Z[ω] (big width). The native z-rotation grid problems work on these
//! directly; the lattice pipelines instead keep one global √2^k outside
//! pure-integer numerators.

// Denominator-exponent deltas are bounded by the lde cap (≤ ~250),
// so the i64→u32 casts cannot truncate.
#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]

use rug::{Complete, Integer};

use crate::rings::real::{ntz, ZRootTwo};
use crate::rings::types::{cached_sqrt2, CompleteId, MpFloat};
use crate::rings::zomega::ZOmegaBig;


// ─── DRootTwo: α / √2^k ───────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub(crate) struct DRootTwo {
    pub alpha: ZRootTwo,
    pub k: u32,
}

impl DRootTwo {
    pub fn new(alpha: ZRootTwo, k: u32) -> Self {
        Self { alpha, k }
    }

    pub fn from_zroottwo(x: ZRootTwo) -> Self {
        Self::new(x, 0)
    }

    pub fn from_int_i64(x: i64) -> Self {
        Self::new(ZRootTwo::from_i64(x, 0), 0)
    }

    /// 1 / √2^k.
    pub fn power_of_inv_sqrt2(k: u32) -> Self {
        Self::new(ZRootTwo::one(), k)
    }

    /// From a DOmega whose value lies in D[√2] (b == 0, a == −c).
    pub fn from_domega(x: &DOmega) -> Self {
        assert!(x.u.c == 0 && x.u.d == (-&x.u.b).complete(), "DOmega not in D[√2]");
        Self::new(ZRootTwo::new(x.u.a.clone(), x.u.b.clone()), x.k)
    }

    /// Rewrite at denominator exponent `new_k ≥` current value's minimal k.
    pub fn renew_denomexp(&self, new_k: u32) -> Self {
        let d = i64::from(new_k) - i64::from(self.k);
        let out = self.mul_by_sqrt2_power(d);
        Self::new(out.alpha, new_k)
    }

    /// Multiply the *numerator* by √2^d (d may be negative when divisible).
    fn mul_by_sqrt2_power(&self, d: i64) -> Self {
        if d < 0 {
            let m = (-d) as u32;
            let d_div_2 = m >> 1;
            let d_mod_2 = m & 1;
            let a = &self.alpha.a;
            let b = &self.alpha.b;
            if d_mod_2 == 0 {
                assert!(
                    a.is_divisible_2pow(d_div_2) && b.is_divisible_2pow(d_div_2),
                    "not divisible by √2^{m}"
                );
                Self::new(
                    ZRootTwo::new((a >> d_div_2).complete(), (b >> d_div_2).complete()),
                    self.k,
                )
            } else {
                assert!(
                    a.is_divisible_2pow(d_div_2 + 1) && b.is_divisible_2pow(d_div_2),
                    "not divisible by √2^{m}"
                );
                Self::new(
                    ZRootTwo::new((b >> d_div_2).complete(), (a >> (d_div_2 + 1)).complete()),
                    self.k,
                )
            }
        } else {
            let d = d as u32;
            let d_div_2 = d >> 1;
            let d_mod_2 = d & 1;
            let mut alpha = self
                .alpha
                .scale(&(Integer::from(1) << d_div_2).complete_id());
            if d_mod_2 == 1 {
                alpha = alpha.mul(&ZRootTwo::from_i64(0, 1));
            }
            Self::new(alpha, self.k)
        }
    }

    /// Divide by √2^d without changing the numerator (k must allow it).
    pub fn mul_by_sqrt2_power_renewing_denomexp(&self, d: u32) -> Self {
        assert!(d <= self.k);
        Self::new(self.alpha.clone(), self.k - d)
    }

    pub fn parity_at_denomexp(&self, k: u32) -> u8 {
        self.renew_denomexp(k).alpha.parity()
    }

    pub fn add(&self, o: &Self) -> Self {
        let k = self.k.max(o.k);
        let a = self.renew_denomexp(k);
        let b = o.renew_denomexp(k);
        Self::new(a.alpha.add(&b.alpha), k)
    }

    pub fn sub(&self, o: &Self) -> Self {
        self.add(&o.neg())
    }

    pub fn neg(&self) -> Self {
        Self::new(self.alpha.neg(), self.k)
    }

    pub fn mul(&self, o: &Self) -> Self {
        Self::new(self.alpha.mul(&o.alpha), self.k + o.k)
    }

    pub fn conj_sq2(&self) -> Self {
        if self.k & 1 == 1 {
            Self::new(self.alpha.conj_sq2().neg(), self.k)
        } else {
            Self::new(self.alpha.conj_sq2(), self.k)
        }
    }

    /// self ≤ s exactly: s·√2^k − α ≥ 0 in the real embedding.
    pub fn le_zroottwo(&self, s: &ZRootTwo) -> bool {
        let scale = s.mul(&ZRootTwo::sqrt2_power(self.k));
        !scale.sub(&self.alpha).is_negative()
    }

    pub fn to_mpfloat(&self, prec: u32) -> MpFloat {
        let num = self.alpha.to_mpfloat(prec);
        let half_k = self.k / 2;
        let mut scale = MpFloat::with_val(prec, 1.0);
        scale <<= half_k;
        if self.k % 2 == 1 {
            scale *= cached_sqrt2(prec);
        }
        num / scale
    }
}
// ─── ZOmega (arbitrary precision): the unified ring, big instantiation ──────
// One ring, two widths — see `crate::rings::zomega`. Basis order is the
// crate convention a + bω + cω² + dω³.
// ─── DOmega: u / √2^k ─────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub(crate) struct DOmega {
    pub u: ZOmegaBig,
    pub k: u32,
}

impl DOmega {
    pub fn new(u: ZOmegaBig, k: u32) -> Self {
        Self { u, k }
    }

    pub fn from_droottwo(x: &DRootTwo) -> Self {
        Self::new(ZOmegaBig::from_zroottwo(&x.alpha), x.k)
    }

    /// (x + y·i) / √2^k from D[√2] components.
    pub fn from_droottwo_vector(x: &DRootTwo, y: &DRootTwo, k: u32) -> Self {
        let dx = Self::from_droottwo(x);
        let dy = Self::from_droottwo(y).mul_zomega(&ZOmegaBig::from_i64(0, 0, 1, 0)); // ·i = ω²
        dx.add(&dy).renew_denomexp(k)
    }

    pub fn add(&self, o: &Self) -> Self {
        let k = self.k.max(o.k);
        let a = self.renew_denomexp(k);
        let b = o.renew_denomexp(k);
        Self::new(a.u.add(&b.u), k)
    }

    pub fn mul(&self, o: &Self) -> Self {
        Self::new(self.u.mul(&o.u), self.k + o.k)
    }

    pub fn mul_zomega(&self, o: &ZOmegaBig) -> Self {
        Self::new(self.u.mul(o), self.k)
    }

    pub fn conj(&self) -> Self {
        Self::new(self.u.conj(), self.k)
    }

    pub fn conj_sq2(&self) -> Self {
        if self.k & 1 == 1 {
            Self::new(self.u.conj_sq2().neg(), self.k)
        } else {
            Self::new(self.u.conj_sq2(), self.k)
        }
    }

    pub fn mul_by_omega(&self) -> Self {
        Self::new(self.u.mul_by_omega(), self.k)
    }

    pub fn mul_by_omega_inv(&self) -> Self {
        Self::new(self.u.mul_by_omega_inv(), self.k)
    }

    pub fn residue(&self) -> u8 {
        self.u.residue()
    }

    pub fn renew_denomexp(&self, new_k: u32) -> Self {
        let d = i64::from(new_k) - i64::from(self.k);
        Self::new(self.mul_by_sqrt2_power(d), new_k)
    }

    /// Numerator × √2^d (negative d requires exact divisibility).
    fn mul_by_sqrt2_power(&self, d: i64) -> ZOmegaBig {
        if d >= 0 {
            let d = d as u32;
            let mut u = self
                .u
                .scale(&(Integer::from(1) << (d >> 1)).complete_id());
            if d & 1 == 1 {
                u = u.mul(&ZOmegaBig::from_i64(0, 1, 0, -1)); // √2 = ω − ω³
            }
            u
        } else {
            let m = (-d) as u32;
            let d_div_2 = m >> 1;
            let d_mod_2 = m & 1;
            if d_mod_2 == 0 {
                for x in [&self.u.a, &self.u.b, &self.u.c, &self.u.d] {
                    assert!(x.is_divisible_2pow(d_div_2), "not divisible by √2^{m}");
                }
                ZOmegaBig::from_parts(
                    (&self.u.a >> d_div_2).complete(),
                    (&self.u.b >> d_div_2).complete(),
                    (&self.u.c >> d_div_2).complete(),
                    (&self.u.d >> d_div_2).complete(),
                )
            } else {
                // x·√2 = (b−d) + (a+c)ω + (b+d)ω² + (c−a)ω³, then /2^{...}.
                let bmd = (&self.u.b - &self.u.d).complete();
                let apc = (&self.u.a + &self.u.c).complete();
                let bpd = (&self.u.b + &self.u.d).complete();
                let cma = (&self.u.c - &self.u.a).complete();
                for x in [&bmd, &apc, &bpd, &cma] {
                    assert!(x.is_divisible_2pow(d_div_2 + 1), "not divisible by √2^{m}");
                }
                ZOmegaBig::from_parts(
                    (bmd >> (d_div_2 + 1)).complete_id(),
                    (apc >> (d_div_2 + 1)).complete_id(),
                    (bpd >> (d_div_2 + 1)).complete_id(),
                    (cma >> (d_div_2 + 1)).complete_id(),
                )
            }
        }
    }

    /// Fully reduce the denominator exponent.
    pub fn reduce_denomexp(&self) -> Self {
        let kk = |x: &Integer| if *x == 0 { self.k } else { ntz(x).min(self.k) };
        let reduce_k = kk(&self.u.a).min(kk(&self.u.b)).min(kk(&self.u.c)).min(kk(&self.u.d));
        let mut new_k_signed = i64::from(self.k) - i64::from(reduce_k) * 2;
        let bit = (Integer::from(1) << (reduce_k + 1)).complete_id() - Integer::from(1);
        // The ω±ω³ / 1±ω² parity pairs in crate basis order.
        let ca = (&self.u.b + &self.u.d).complete();
        let bd = (&self.u.a + &self.u.c).complete();
        if (ca & bit.clone()).complete_id() == 0 && (bd & bit).complete_id() == 0 {
            new_k_signed -= 1;
        }
        let new_k = if new_k_signed < 0 { 0 } else { new_k_signed as u32 };
        self.renew_denomexp(new_k)
    }

    pub fn real(&self, prec: u32) -> MpFloat {
        self.u.real(prec) / sqrt2_pow_f(self.k, prec)
    }

    pub fn imag(&self, prec: u32) -> MpFloat {
        self.u.imag(prec) / sqrt2_pow_f(self.k, prec)
    }
}

/// √2^k as MPFR.
pub(crate) fn sqrt2_pow_f(k: u32, prec: u32) -> MpFloat {
    let mut s = MpFloat::with_val(prec, 1.0);
    s <<= k / 2;
    if k % 2 == 1 {
        s *= cached_sqrt2(prec);
    }
    s
}
