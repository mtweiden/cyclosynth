//! The real subfield rings of the synthesis cyclotomics, on
//! `rug::Integer` coefficients (arbitrary precision): Z[√2] = the real
//! subfield of Q(ω) ([`ZRootTwo`], Clifford+T) and Z[g], g = 2cos(π/8) =
//! √(2+√2) = the real subfield of Q(ζ₁₆) ([`ZRootTwoPlusRootTwo`],
//! Clifford+√T). Neither has a fixed-width counterpart — they exist for
//! the native z-rotation routes, whose norm-equation intermediates are
//! unbounded. Shared integer helpers (`rounddiv`, `ntz`, …) and the
//! bridges into the cyclotomic rings live here too.

// Scaffolding: consumers land incrementally in later commits.
#![cfg_attr(not(test), allow(dead_code))]

// Denominator-exponent deltas are bounded by the lde cap (≤ ~250),
// so the i64→u32 casts cannot truncate.
#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]

use rug::{Complete, Integer};

use crate::rings::types::{cached_sqrt2, CompleteId, MpFloat};
use crate::rings::zomega::ZOmegaBig;
use crate::rings::zzeta::ZZetaBig;

/// Number of trailing zero bits (0 for the zero element).
pub(crate) fn ntz(n: &Integer) -> u32 {
    n.find_one(0).unwrap_or(0)
}
/// Python-style floor division.
pub(crate) fn div_floor(x: &Integer, y: &Integer) -> Integer {
    x.div_rem_floor_ref(y).complete().0
}

/// Round-to-nearest division with floor-division tie semantics (the
/// convention the ODGP interval endpoints assume).
pub(crate) fn rounddiv(x: &Integer, y: &Integer) -> Integer {
    if *y > 0 {
        let num = x + (y / 2u32).complete();
        div_floor(&num, y)
    } else {
        let num = x - (-y).complete() / 2u32;
        div_floor(&num, y)
    }
}

/// Floor square root of a nonnegative integer.
pub(crate) fn floorsqrt(x: &Integer) -> Integer {
    x.clone().sqrt()
}
// ─── ZRootTwo: a + b√2 ────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ZRootTwo {
    pub a: Integer,
    pub b: Integer,
}

impl ZRootTwo {
    pub fn new(a: Integer, b: Integer) -> Self {
        Self { a, b }
    }

    pub fn from_i64(a: i64, b: i64) -> Self {
        Self::new(Integer::from(a), Integer::from(b))
    }

    pub fn from_int(x: Integer) -> Self {
        Self::new(x, Integer::new())
    }

    pub fn one() -> Self {
        Self::from_i64(1, 0)
    }

    /// λ = 1 + √2.
    pub fn lambda() -> Self {
        Self::from_i64(1, 1)
    }

    pub fn is_zero(&self) -> bool {
        self.a == 0 && self.b == 0
    }

    pub fn add(&self, o: &Self) -> Self {
        Self::new((&self.a + &o.a).complete(), (&self.b + &o.b).complete())
    }

    pub fn sub(&self, o: &Self) -> Self {
        Self::new((&self.a - &o.a).complete(), (&self.b - &o.b).complete())
    }

    pub fn neg(&self) -> Self {
        Self::new((-&self.a).complete(), (-&self.b).complete())
    }

    pub fn mul(&self, o: &Self) -> Self {
        let a = (&self.a * &o.a).complete() + 2u32 * (&self.b * &o.b).complete();
        let b = (&self.a * &o.b).complete() + (&self.b * &o.a).complete();
        Self::new(a, b)
    }

    pub fn scale(&self, o: &Integer) -> Self {
        Self::new((&self.a * o).complete(), (&self.b * o).complete())
    }

    /// a² − 2b².
    pub fn norm(&self) -> Integer {
        (&self.a * &self.a).complete() - 2u32 * (&self.b * &self.b).complete()
    }

    /// √2-conjugate: a − b√2.
    pub fn conj_sq2(&self) -> Self {
        Self::new(self.a.clone(), (-&self.b).complete())
    }

    pub fn parity(&self) -> u8 {
        // Low bit of a — well-defined for negative a too (two's
        // complement low bit == |a| mod 2).
        u8::from(self.a.is_odd())
    }

    /// Inverse of a unit (norm ±1); panics otherwise (matches the Python
    /// ZeroDivisionError contract — only ever called on λ powers).
    pub fn inv(&self) -> Self {
        let n = self.norm();
        if n == 1 {
            self.conj_sq2()
        } else if n == -1 {
            self.conj_sq2().neg()
        } else {
            panic!("ZRootTwo::inv of non-unit");
        }
    }

    pub fn pow(&self, e: i64) -> Self {
        if e < 0 {
            return self.inv().pow(-e);
        }
        let mut acc = Self::one();
        let mut base = self.clone();
        let mut n = e;
        while n > 0 {
            if n & 1 == 1 {
                acc = acc.mul(&base);
            }
            base = base.mul(&base);
            n >>= 1;
        }
        acc
    }

    /// Integer square root in Z[√2], or None.
    pub fn sqrt(&self) -> Option<Self> {
        let norm = self.norm();
        if norm < 0 || self.a < 0 {
            return None;
        }
        let r = floorsqrt(&norm);
        let two = Integer::from(2);
        let four = Integer::from(4);
        let a1 = floorsqrt(&div_floor(&(&self.a + &r).complete(), &two));
        let b1 = floorsqrt(&div_floor(&(&self.a - &r).complete(), &four));
        let a2 = floorsqrt(&div_floor(&(&self.a - &r).complete(), &two));
        let b2 = floorsqrt(&div_floor(&(&self.a + &r).complete(), &four));
        let same_sign = self.a.clone().signum() * self.b.clone().signum() >= 0;
        let (w1, w2) = if same_sign {
            (Self::new(a1, b1), Self::new(a2, b2))
        } else {
            (Self::new(a1, (-b1).complete_id()), Self::new(a2, (-b2).complete_id()))
        };
        if *self == w1.mul(&w1) {
            Some(w1)
        } else if *self == w2.mul(&w2) {
            Some(w2)
        } else {
            None
        }
    }

    /// Euclidean divmod via norm rounding.
    pub fn divmod(&self, o: &Self) -> (Self, Self) {
        let p = self.mul(&o.conj_sq2());
        let k = o.norm();
        let q = Self::new(rounddiv(&p.a, &k), rounddiv(&p.b, &k));
        let r = self.sub(&o.mul(&q));
        (q, r)
    }

    pub fn rem(&self, o: &Self) -> Self {
        self.divmod(o).1
    }

    pub fn divexact(&self, o: &Self) -> Self {
        self.divmod(o).0
    }

    /// a ~ b (associates): a | b and b | a.
    pub fn sim(a: &Self, b: &Self) -> bool {
        !b.is_zero() && !a.is_zero() && a.rem(b).is_zero() && b.rem(a).is_zero()
    }

    pub fn gcd(a: &Self, b: &Self) -> Self {
        let mut a = a.clone();
        let mut b = b.clone();
        while !b.is_zero() {
            let r = a.rem(&b);
            a = b;
            b = r;
        }
        a
    }

    /// self < 0 in the real embedding (a + b√2 < 0), exactly.
    pub fn is_negative(&self) -> bool {
        if self.a >= 0 && self.b >= 0 {
            return false;
        }
        if self.a <= 0 && self.b <= 0 {
            return !(self.a == 0 && self.b == 0);
        }
        // Mixed signs: a + b√2 < 0 ⟺ (a<0, b>0): a² > 2b²; (a>0, b<0): a² < 2b².
        let a2 = (&self.a * &self.a).complete();
        let b22 = 2u32 * (&self.b * &self.b).complete();
        if self.a < 0 {
            a2 > b22
        } else {
            a2 < b22
        }
    }

    pub fn to_mpfloat(&self, prec: u32) -> MpFloat {
        let sqrt2 = cached_sqrt2(prec);
        MpFloat::with_val(prec, &self.a) + MpFloat::with_val(prec, &self.b) * sqrt2
    }
}
impl ZRootTwo {
    /// √2^k as a ring element (k even: 2^{k/2}; k odd: 2^{(k−1)/2}·√2).
    pub fn sqrt2_power(k: u32) -> Self {
        let half = Integer::from(1) << (k / 2);
        if k.is_multiple_of(2) {
            Self::new(half.complete_id(), Integer::new())
        } else {
            Self::new(Integer::new(), half.complete_id())
        }
    }
}

impl ZOmegaBig {
    /// Z[√2] ↪ Z[ω]: a + b√2 = a + b(ω − ω³).
    pub(crate) fn from_zroottwo(x: &ZRootTwo) -> Self {
        Self::from_parts(x.a.clone(), x.b.clone(), Integer::new(), (-&x.b).complete())
    }
}


impl ZRootTwo {
    /// From a ZOmegaBig lying in Z[√2] (b == 0, a == −c); panics otherwise
    /// (internal invariant — t†t is always real for the Diophantine result).
    pub fn from_zomega_big(x: &ZOmegaBig) -> Self {
        assert!(x.c == 0 && x.d == (-&x.b).complete(), "ZOmegaBig not in Z[√2]");
        Self::new(x.a.clone(), x.b.clone())
    }
}

/// The four real embeddings' g values: 2cos(mπ/8) for m = 1, 3, 5, 7.
/// Index 0 is the "principal" embedding the synthesis targets.
pub(crate) fn generator_embeddings(prec: u32) -> [MpFloat; 4] {
    let pi = MpFloat::with_val(prec, rug::float::Constant::Pi);
    let mk = |m: u32| -> MpFloat {
        let ang = MpFloat::with_val(prec, &pi * m) / 8u32;
        MpFloat::with_val(prec, ang.cos() * 2u32)
    };
    [mk(1), mk(3), mk(5), mk(7)]
}
// ─── ZRootTwoPlusRootTwo: a + bg + cg² + dg³ ────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ZRootTwoPlusRootTwo {
    pub a: Integer,
    pub b: Integer,
    pub c: Integer,
    pub d: Integer,
}

impl ZRootTwoPlusRootTwo {
    pub fn new(a: Integer, b: Integer, c: Integer, d: Integer) -> Self {
        Self { a, b, c, d }
    }

    pub fn from_i64(a: i64, b: i64, c: i64, d: i64) -> Self {
        Self::new(Integer::from(a), Integer::from(b), Integer::from(c), Integer::from(d))
    }

    pub fn from_int(x: Integer) -> Self {
        Self::new(x, Integer::new(), Integer::new(), Integer::new())
    }

    pub fn zero() -> Self {
        Self::from_i64(0, 0, 0, 0)
    }

    pub fn one() -> Self {
        Self::from_i64(1, 0, 0, 0)
    }

    /// g.
    pub fn generator() -> Self {
        Self::from_i64(0, 1, 0, 0)
    }

    /// √2 = g² − 2.
    pub fn sqrt2() -> Self {
        Self::from_i64(-2, 0, 1, 0)
    }

    pub fn is_zero(&self) -> bool {
        self.a == 0 && self.b == 0 && self.c == 0 && self.d == 0
    }

    pub fn add(&self, o: &Self) -> Self {
        Self::new(
            (&self.a + &o.a).complete(),
            (&self.b + &o.b).complete(),
            (&self.c + &o.c).complete(),
            (&self.d + &o.d).complete(),
        )
    }

    pub fn sub(&self, o: &Self) -> Self {
        Self::new(
            (&self.a - &o.a).complete(),
            (&self.b - &o.b).complete(),
            (&self.c - &o.c).complete(),
            (&self.d - &o.d).complete(),
        )
    }

    #[cfg(test)]
    pub fn neg(&self) -> Self {
        Self::new(
            (-&self.a).complete(),
            (-&self.b).complete(),
            (-&self.c).complete(),
            (-&self.d).complete(),
        )
    }

    /// Product with g⁴ = 4g² − 2, g⁵ = 4g³ − 2g, g⁶ = 14g² − 8 reduction.
    pub fn mul(&self, o: &Self) -> Self {
        let s = [&self.a, &self.b, &self.c, &self.d];
        let t = [&o.a, &o.b, &o.c, &o.d];
        // Raw degree-6 convolution.
        let mut raw = [
            Integer::new(),
            Integer::new(),
            Integer::new(),
            Integer::new(),
            Integer::new(),
            Integer::new(),
            Integer::new(),
        ];
        for i in 0..4 {
            for j in 0..4 {
                raw[i + j] += (s[i] * t[j]).complete();
            }
        }
        let [mut r0, mut r1, mut r2, mut r3, r4, r5, r6] = raw;
        // g⁶ → 14g² − 8
        r2 += (14u32 * &r6).complete();
        r0 -= (8u32 * &r6).complete();
        // g⁵ → 4g³ − 2g
        r3 += (4u32 * &r5).complete();
        r1 -= (2u32 * &r5).complete();
        // g⁴ → 4g² − 2
        r2 += (4u32 * &r4).complete();
        r0 -= (2u32 * &r4).complete();
        Self::new(r0, r1, r2, r3)
    }

    #[cfg(test)]
    pub fn scale(&self, x: &Integer) -> Self {
        Self::new(
            (&self.a * x).complete(),
            (&self.b * x).complete(),
            (&self.c * x).complete(),
            (&self.d * x).complete(),
        )
    }

    /// Galois generator σ: g ↦ g³ − 3g. On coefficients:
    /// σ(a + bg + cg² + dg³) = (a+4c) + (−3b−10d)g − cg² + (b+3d)g³.
    pub fn sigma(&self) -> Self {
        let four_c = (4u32 * &self.c).complete();
        let ten_d = (10u32 * &self.d).complete();
        let three_b = (3u32 * &self.b).complete();
        let three_d = (3u32 * &self.d).complete();
        Self::new(
            &self.a + four_c,
            (-three_b).complete_id() - ten_d,
            (-&self.c).complete(),
            &self.b + three_d,
        )
    }

    /// σ²: g ↦ −g.
    pub fn sigma2(&self) -> Self {
        Self::new(
            self.a.clone(),
            (-&self.b).complete(),
            self.c.clone(),
            (-&self.d).complete(),
        )
    }

    /// Absolute norm N(x) = x·σx·σ²x·σ³x ∈ Z.
    pub fn norm(&self) -> Integer {
        let s1 = self.sigma();
        let s2 = self.sigma2();
        let s3 = s1.sigma2(); // σ³ = σ∘σ²
        let p = self.mul(&s1).mul(&s2).mul(&s3);
        debug_assert!(
            p.b == 0 && p.c == 0 && p.d == 0,
            "norm must be rational: {p:?}"
        );
        p.a
    }

    /// Value under embedding `idx` (g = gen[idx]).
    pub fn embed(&self, gen: &MpFloat, prec: u32) -> MpFloat {
        let e2 = MpFloat::with_val(prec, gen * gen);
        let e3 = MpFloat::with_val(prec, &e2 * gen);
        MpFloat::with_val(prec, &self.a)
            + MpFloat::with_val(prec, &self.b) * gen
            + MpFloat::with_val(prec, &self.c) * e2
            + MpFloat::with_val(prec, &self.d) * e3
    }

    /// Exactly negative in the principal embedding? (sign of a quartic
    /// algebraic number; MPFR with escalating precision — a true zero is
    /// only the zero element, checked first.)
    pub fn is_negative_embedding(&self, idx: usize) -> bool {
        if self.is_zero() {
            return false;
        }
        let mut prec = 128;
        loop {
            let gen = generator_embeddings(prec);
            let v = self.embed(&gen[idx], prec);
            // Trust the sign when |v| clears the rounding-noise floor.
            let bits = self.max_coeff_bits();
            let noise = MpFloat::with_val(prec, 1.0) << (i32::try_from(bits).expect("small") + 8 - i32::try_from(prec).expect("small"));
            if v.clone().abs() > noise {
                return v.is_sign_negative();
            }
            prec *= 2;
            assert!(prec <= 1 << 20, "sign undecidable: {self:?}");
        }
    }

    fn max_coeff_bits(&self) -> u32 {
        self.a
            .significant_bits()
            .max(self.b.significant_bits())
            .max(self.c.significant_bits())
            .max(self.d.significant_bits())
    }

    /// Totally non-negative: every embedding ≥ 0.
    pub fn is_totally_nonneg(&self) -> bool {
        (0..4).all(|i| !self.is_negative_embedding(i))
    }

    /// Euclidean-style divmod via conjugate multiplication and coefficient
    /// rounding (norm-Euclidean-style; callers cap gcd iterations).
    pub fn divmod(&self, o: &Self) -> (Self, Self) {
        let s1 = o.sigma();
        let s2 = o.sigma2();
        let s3 = s1.sigma2();
        let mult = s1.mul(&s2).mul(&s3);
        let num = self.mul(&mult);
        let n = o.norm();
        let q = Self::new(
            rounddiv(&num.a, &n),
            rounddiv(&num.b, &n),
            rounddiv(&num.c, &n),
            rounddiv(&num.d, &n),
        );
        let r = self.sub(&o.mul(&q));
        (q, r)
    }

    pub fn rem(&self, o: &Self) -> Self {
        self.divmod(o).1
    }

    pub fn divexact(&self, o: &Self) -> Self {
        self.divmod(o).0
    }

    /// gcd with an iteration cap; None = did not terminate (treated as
    /// UNSOLVED upstream — never produces a wrong answer).
    pub fn gcd_capped(a: &Self, b: &Self, cap: u32) -> Option<Self> {
        let mut a = a.clone();
        let mut b = b.clone();
        for _ in 0..cap {
            if b.is_zero() {
                return Some(a);
            }
            let r = a.rem(&b);
            a = b;
            b = r;
        }
        None
    }

    /// Square root in Z[g] via per-embedding numeric roots + exact verify.
    /// `self` must be totally non-negative; tries all sign patterns.
    pub fn sqrt(&self) -> Option<Self> {
        if self.is_zero() {
            return Some(Self::zero());
        }
        let bits = self.max_coeff_bits();
        let prec = bits / 2 + 96;
        let gen = generator_embeddings(prec);
        let roots: Vec<MpFloat> = (0..4)
            .map(|i| {
                let v = self.embed(&gen[i], prec);
                if v.is_sign_negative() {
                    MpFloat::with_val(prec, 0.0) // not a square; verify fails
                } else {
                    v.sqrt()
                }
            })
            .collect();
        // Solve the 4×4 Vandermonde-in-g system for each sign pattern and
        // round to integers; verify exactly.
        for pattern in 0u32..16 {
            let vals: Vec<MpFloat> = (0..4)
                .map(|i| {
                    if pattern >> i & 1 == 1 {
                        MpFloat::with_val(prec, -&roots[i])
                    } else {
                        roots[i].clone()
                    }
                })
                .collect();
            if let Some(cand) = solve_coeffs(&vals, &gen, prec) {
                if cand.mul(&cand) == *self {
                    return Some(cand);
                }
            }
        }
        None
    }
}

/// Given target values v_i at the four embeddings, solve for integer
/// coefficients (a,b,c,d) with a + bg_i + cg_i² + dg_i³ = v_i (rounded).
fn solve_coeffs(vals: &[MpFloat], gen: &[MpFloat; 4], prec: u32) -> Option<ZRootTwoPlusRootTwo> {
    // Gaussian elimination on the 4×4 system in MPFR.
    let n = 4usize;
    let mut m = vec![vec![MpFloat::with_val(prec, 0.0); n + 1]; n];
    for i in 0..n {
        let mut p = MpFloat::with_val(prec, 1.0);
        for j in 0..n {
            m[i][j] = p.clone();
            p = MpFloat::with_val(prec, &p * &gen[i]);
        }
        m[i][n] = vals[i].clone();
    }
    for col in 0..n {
        // Pivot.
        let piv = (col..n).max_by(|&r1, &r2| {
            m[r1][col].clone().abs().total_cmp(&m[r2][col].clone().abs())
        })?;
        m.swap(col, piv);
        if m[col][col].is_zero() {
            return None;
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let f = MpFloat::with_val(prec, &m[row][col] / &m[col][col]);
            for j in col..=n {
                let sub = MpFloat::with_val(prec, &f * &m[col][j]);
                m[row][j] -= sub;
            }
        }
    }
    let mut out = [Integer::new(), Integer::new(), Integer::new(), Integer::new()];
    for i in 0..n {
        let v = MpFloat::with_val(prec, &m[i][n] / &m[i][i]);
        let r = v.clone() - v.clone().round();
        if r.clone().abs() > 0.25 {
            return None; // not near-integer — wrong sign pattern
        }
        out[i] = v.round().to_integer()?;
    }
    let [a, b, c, d] = out;
    Some(ZRootTwoPlusRootTwo::new(a, b, c, d))
}
// ─── Real-subfield bridges for the big cyclotomic instantiations ─────────────

/// Real-subfield bridge: Z[g] has no fixed-width counterpart, so the
/// conversions and the Euclidean structure that route through it live
/// beside the big rings rather than on the generic type.
impl ZZetaBig {
    /// Lift a Z[g] element via g = ζ − ζ⁷ (g = ζ + ζ⁻¹, ζ⁻¹ = −ζ⁷).
    pub(crate) fn from_real(x: &ZRootTwoPlusRootTwo) -> Self {
        let gen = Self::from_i64(0, 1, 0, 0, 0, 0, 0, -1);
        let eta2 = gen.mul(&gen);
        let eta3 = eta2.mul(&gen);
        Self::from_int(x.a.clone())
            .add(&gen.scale(&x.b))
            .add(&eta2.scale(&x.c))
            .add(&eta3.scale(&x.d))
    }

    /// A + ζ·B for A, B ∈ Z[g].
    pub(crate) fn from_module(a: &ZRootTwoPlusRootTwo, b: &ZRootTwoPlusRootTwo) -> Self {
        Self::from_real(a).add(&Self::from_real(b).mul(&Self::zeta()))
    }

    /// Relative norm to Z[g]: x·x̄ (result lies in Z[g]; converted back).
    pub(crate) fn rel_norm(&self) -> ZRootTwoPlusRootTwo {
        let p = self.mul(&self.conj());
        p.to_real().expect("x·x̄ must be real")
    }

    /// Convert an element that lies in Z[g] (real subring) back to ZRootTwoPlusRootTwo
    /// coefficients; None if it isn't real.
    ///
    /// Basis bookkeeping: g = ζ − ζ⁷, g² = 2 + ζ² − ζ⁶, g³ = 3ζ + ζ³ − ζ⁵ − 3ζ⁷.
    /// So a real x = a + bg + cg² + dg³ has ζ-coefficients
    /// c0 = a + 2c; c1 = b + 3d; c2 = c; c3 = d; c4 = 0; c5 = −d; c6 = −c; c7 = −b − 3d.
    pub(crate) fn to_real(&self) -> Option<ZRootTwoPlusRootTwo> {
        let cs = self.coeffs();
        if *cs[4] != 0 {
            return None;
        }
        let c = cs[2].clone();
        let d = cs[3].clone();
        if *cs[6] != (-&c).complete() || *cs[5] != (-&d).complete() {
            return None;
        }
        let three_d = (3u32 * &d).complete();
        let b = (cs[1] - &three_d).complete();
        let neg_b_3d = -((&b + &three_d).complete());
        if *cs[7] != neg_b_3d {
            return None;
        }
        let a = (cs[0] - (2u32 * &c).complete()).complete_id();
        Some(ZRootTwoPlusRootTwo::new(a, b, c, d))
    }

    /// (A, B) with self = A + ζB, A,B ∈ Z[g] — inverse of `from_module`.
    /// Walk the power table ζⁱ = Aᵢ + ζBᵢ with (Aᵢ₊₁, Bᵢ₊₁) = (−Bᵢ, Aᵢ + gBᵢ).
    #[cfg(test)]
    pub(crate) fn to_module(&self) -> (ZRootTwoPlusRootTwo, ZRootTwoPlusRootTwo) {
        let gen = ZRootTwoPlusRootTwo::generator();
        let mut ai = ZRootTwoPlusRootTwo::one();
        let mut bi = ZRootTwoPlusRootTwo::zero();
        let mut a_out = ZRootTwoPlusRootTwo::zero();
        let mut b_out = ZRootTwoPlusRootTwo::zero();
        for x in self.coeffs() {
            a_out = a_out.add(&ai.scale(x));
            b_out = b_out.add(&bi.scale(x));
            let next_a = bi.neg();
            let next_b = ai.add(&gen.mul(&bi));
            ai = next_a;
            bi = next_b;
        }
        (a_out, b_out)
    }

    /// Absolute norm N(x) ∈ Z (via the relative norm's Z[g] norm).
    pub(crate) fn norm(&self) -> Integer {
        self.rel_norm().norm()
    }

    /// Euclidean-style divmod (rounded); callers cap gcd iterations.
    pub(crate) fn divmod(&self, o: &Self) -> (Self, Self) {
        // x/y = x·ȳ·lift(σr·σ²r·σ³r) / N(y), r = y·ȳ ∈ Z[g].
        let r = o.rel_norm();
        let s1 = r.sigma();
        let s2 = r.sigma2();
        let s3 = s1.sigma2();
        let mult = s1.mul(&s2).mul(&s3);
        let num = self.mul(&o.conj()).mul(&Self::from_real(&mult));
        let n = o.norm();
        let num_cs = num.coeffs();
        let q = Self::from_coeffs(std::array::from_fn(|i| rounddiv(num_cs[i], &n)));
        let rem = self.sub(&o.mul(&q));
        (q, rem)
    }

    pub(crate) fn rem(&self, o: &Self) -> Self {
        self.divmod(o).1
    }

    /// gcd with an iteration cap; None = non-termination (UNSOLVED upstream).
    pub(crate) fn gcd_capped(a: &Self, b: &Self, cap: u32) -> Option<Self> {
        let mut a = a.clone();
        let mut b = b.clone();
        for _ in 0..cap {
            if b.is_zero() {
                return Some(a);
            }
            let r = a.rem(&b);
            a = b;
            b = r;
        }
        None
    }
}


// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn f(x: &ZRootTwoPlusRootTwo) -> f64 {
        let gen = generator_embeddings(96);
        x.embed(&gen[0], 96).to_f64()
    }

    #[test]
    fn test_minimal_poly_and_sqrt2() {
        let gen = ZRootTwoPlusRootTwo::generator();
        let eta4 = gen.mul(&gen).mul(&gen).mul(&gen);
        let rhs = ZRootTwoPlusRootTwo::from_i64(-2, 0, 4, 0); // 4g² − 2
        assert_eq!(eta4, rhs);
        let s2 = ZRootTwoPlusRootTwo::sqrt2();
        assert!((f(&s2) - std::f64::consts::SQRT_2).abs() < 1e-12);
        assert_eq!(s2.mul(&s2), ZRootTwoPlusRootTwo::from_i64(2, 0, 0, 0));
    }

    #[test]
    fn test_galois() {
        let gen = ZRootTwoPlusRootTwo::generator();
        // σ has order 4; σ² = (g → −g); σ(√2) = −√2.
        assert_eq!(gen.sigma().sigma(), gen.sigma2());
        assert_eq!(gen.sigma2().sigma2(), gen);
        assert_eq!(ZRootTwoPlusRootTwo::sqrt2().sigma(), ZRootTwoPlusRootTwo::sqrt2().neg());
        // Numeric: σ(g) = 2cos(3π/8).
        assert!((f(&gen.sigma()) - 2.0 * (3.0 * std::f64::consts::PI / 8.0).cos()).abs() < 1e-12);
        // σ is a ring hom on a random-ish product.
        let x = ZRootTwoPlusRootTwo::from_i64(3, -2, 5, 1);
        let y = ZRootTwoPlusRootTwo::from_i64(-1, 4, 0, 2);
        assert_eq!(x.mul(&y).sigma(), x.sigma().mul(&y.sigma()));
    }

    #[test]
    fn test_norm_multiplicative() {
        let x = ZRootTwoPlusRootTwo::from_i64(2, 1, 0, -1);
        let y = ZRootTwoPlusRootTwo::from_i64(1, 1, 1, 0);
        assert_eq!(x.norm() * y.norm(), x.mul(&y).norm());
        // N(√2) = 4 (√2·(−√2)·√2·(−√2)).
        assert_eq!(ZRootTwoPlusRootTwo::sqrt2().norm(), 4);
        // N(g) = −2 (g·σ(g)·(−g)·(−σ(g)) = (g σ(g))²; gσ(g) = √2 → 2... )
        // Direct check against the resultant of x⁴−4x²+2 evaluated: N(g) = 2·(−1)⁴·? — just assert consistency numerically.
        let gen = ZRootTwoPlusRootTwo::generator();
        let n = gen.norm();
        let prod: f64 = {
            let e = generator_embeddings(96);
            (0..4).map(|i| gen.embed(&e[i], 96).to_f64()).product()
        };
        assert!((int_to_approx(&n) - prod).abs() < 1e-6);
    }

    #[test]
    fn test_zeta16_basics() {
        let z = ZZetaBig::zeta();
        // ζ⁸ = −1.
        let mut p = ZZetaBig::one();
        for _ in 0..8 {
            p = p.mul(&z);
        }
        assert_eq!(p, ZZetaBig::one().neg());
        // ζ·ζ̄ = 1.
        assert_eq!(z.mul(&z.conj()), ZZetaBig::one());
        // g lift round-trip: ζ + ζ⁻¹ = g.
        let gen_lift = ZZetaBig::from_real(&ZRootTwoPlusRootTwo::generator());
        assert_eq!(gen_lift.to_real(), Some(ZRootTwoPlusRootTwo::generator()));
        // ζ² = gζ − 1.
        let lhs = z.mul(&z);
        let rhs = gen_lift.mul(&z).sub(&ZZetaBig::one());
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn test_module_decomposition_and_rel_norm() {
        // x = A + ζB round-trips, and rel_norm(x) = A² + B² + gAB.
        let a = ZRootTwoPlusRootTwo::from_i64(1, -2, 3, 0);
        let b = ZRootTwoPlusRootTwo::from_i64(2, 1, 0, -1);
        let x = ZZetaBig::from_module(&a, &b);
        let (a2, b2) = x.to_module();
        assert_eq!(a2, a);
        assert_eq!(b2, b);
        let expect = a.mul(&a).add(&b.mul(&b)).add(&ZRootTwoPlusRootTwo::generator().mul(&a).mul(&b));
        assert_eq!(x.rel_norm(), expect);
    }

    #[test]
    fn test_zeta16_divmod_gcd() {
        let x = ZZetaBig::from_module(&ZRootTwoPlusRootTwo::from_i64(5, 1, -2, 0), &ZRootTwoPlusRootTwo::from_i64(0, 3, 1, 1));
        let y = ZZetaBig::from_module(&ZRootTwoPlusRootTwo::from_i64(1, 1, 0, 0), &ZRootTwoPlusRootTwo::from_i64(1, 0, 0, 0));
        let (q, r) = x.divmod(&y);
        assert_eq!(y.mul(&q).add(&r), x);
        // Remainder norm must shrink for the Euclidean loop to make sense.
        assert!(r.norm().clone().abs() < y.norm().clone().abs());
        // gcd(x·y, y) ~ y (associate): divides both ways.
        let g = ZZetaBig::gcd_capped(&x.mul(&y), &y, 64).expect("terminates");
        assert!(y.rem(&g).is_zero());
    }

    #[test]
    fn test_zeta_sqrt() {
        let x = ZRootTwoPlusRootTwo::from_i64(2, -1, 1, 0);
        let sq = x.mul(&x);
        let r = sq.sqrt().expect("perfect square");
        assert!(r == x || r == x.neg());
        // Non-square: 1 + g is not a square (norm check would catch most).
        assert!(ZRootTwoPlusRootTwo::from_i64(1, 1, 0, 0).sqrt().is_none());
    }

    #[test]
    fn test_totally_nonneg() {
        // 2 + g is positive in embeddings 1,3 and 2−1.84.. > 0 in all? g ranges ±1.85, ±0.77 → 2+g ∈ {3.85, 2.77, 1.23, 0.15} all > 0.
        assert!(ZRootTwoPlusRootTwo::from_i64(2, 1, 0, 0).is_totally_nonneg());
        // g itself is negative in two embeddings.
        assert!(!ZRootTwoPlusRootTwo::generator().is_totally_nonneg());
        // √2 is negative in embeddings 2 and 4.
        assert!(!ZRootTwoPlusRootTwo::sqrt2().is_totally_nonneg());
    }
}

/// Approximate Integer → f64 for tests.
#[cfg(test)]
pub(crate) fn int_to_approx(x: &Integer) -> f64 {
    x.to_f64()
}

