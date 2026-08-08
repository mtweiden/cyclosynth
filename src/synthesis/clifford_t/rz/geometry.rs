//! Geometry of the z-rotation grid problems: the ε-cap and unit-disk
//! regions (intervals, rectangles, ellipses on MPFR scalars), the
//! special grid operators of Z[ω], and the upright/skew reduction that
//! normalizes an ellipse pair before enumeration (RS arXiv:1403.2975
//! §5–6, "the enclosing upright rectangle").

use rug::{Complete, Integer};

use crate::rings::types::CompleteId;

use crate::rings::types::MpFloat;

use crate::rings::dyadic::DOmega;
use crate::rings::real::ZRootTwo;
use crate::rings::zomega::ZOmegaBig;


/// `MpFloat::with_val` shorthand — rug needs explicit precision on every
/// assignment from an arithmetic expression.
macro_rules! mpf {
    ($prec:expr, $e:expr) => {
        MpFloat::with_val($prec, $e)
    };
}
#[allow(unused_imports)]
pub(crate) use mpf;

#[derive(Clone, Debug)]
pub(crate) struct Interval {
    pub l: MpFloat,
    pub r: MpFloat,
}

impl Interval {
    pub fn new(l: MpFloat, r: MpFloat) -> Self {
        Self { l, r }
    }

    fn prec(&self) -> u32 {
        self.l.prec()
    }

    pub fn width(&self) -> MpFloat {
        mpf!(self.prec(), &self.r - &self.l)
    }

    pub fn sub_f(&self, x: &MpFloat) -> Self {
        let p = self.prec();
        Self::new(mpf!(p, &self.l - x), mpf!(p, &self.r - x))
    }

    pub fn mul_f(&self, x: &MpFloat) -> Self {
        let p = self.prec();
        if x.is_sign_negative() {
            Self::new(mpf!(p, &self.r * x), mpf!(p, &self.l * x))
        } else {
            Self::new(mpf!(p, &self.l * x), mpf!(p, &self.r * x))
        }
    }

    pub fn neg(&self) -> Self {
        let p = self.prec();
        Self::new(mpf!(p, -&self.r), mpf!(p, -&self.l))
    }

    pub fn fatten(&self, eps: &MpFloat) -> Self {
        let p = self.prec();
        Self::new(mpf!(p, &self.l - eps), mpf!(p, &self.r + eps))
    }

    pub fn within(&self, x: &MpFloat) -> bool {
        self.l <= *x && *x <= self.r
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Rectangle {
    pub ix: Interval,
    pub iy: Interval,
}

/// Symmetric 2×2 quadratic form D = [[a, b], [b, d]] with center p:
/// {v : (v−p)ᵀ D (v−p) ≤ 1}.
#[derive(Clone, Debug)]
pub(crate) struct Ellipse {
    pub a: MpFloat,
    pub b: MpFloat,
    pub d: MpFloat,
    pub px: MpFloat,
    pub py: MpFloat,
}

impl Ellipse {
    pub fn prec(&self) -> u32 {
        self.a.prec()
    }

    pub fn sqrt_det(&self) -> MpFloat {
        let p = self.prec();
        let det = mpf!(p, &self.a * &self.d) - mpf!(p, &self.b * &self.b);
        det.sqrt()
    }

    pub fn bbox(&self) -> Rectangle {
        let p = self.prec();
        let sqrt_det = self.sqrt_det();
        let w = mpf!(p, self.d.clone().sqrt() / &sqrt_det);
        let h = mpf!(p, self.a.clone().sqrt() / &sqrt_det);
        Rectangle {
            ix: Interval::new(mpf!(p, &self.px - &w), mpf!(p, &self.px + &w)),
            iy: Interval::new(mpf!(p, &self.py - &h), mpf!(p, &self.py + &h)),
        }
    }

    /// Normalize to det 1: D/√det, p·√(√det).
    pub fn normalize(&self) -> Self {
        let p = self.prec();
        let s = self.sqrt_det();
        let ps = s.clone().sqrt();
        Self {
            a: mpf!(p, &self.a / &s),
            b: mpf!(p, &self.b / &s),
            d: mpf!(p, &self.d / &s),
            px: mpf!(p, &self.px * &ps),
            py: mpf!(p, &self.py * &ps),
        }
    }

    pub fn skew(&self) -> MpFloat {
        mpf!(self.prec(), &self.b * &self.b)
    }

    pub fn bias(&self) -> MpFloat {
        mpf!(self.prec(), &self.d / &self.a)
    }

    /// Apply a grid operator: form transforms by G⁻¹ on coordinates, the
    /// center by G.
    pub fn transform(&self, g: &GridOp) -> Self {
        let p = self.prec();
        let gi = g.inv().expect("grid operator must be special");
        let [[m00, m01], [m10, m11]] = gi.to_matrix(p);
        let a = mpf!(p, &self.a * &m00) * &m00
            + mpf!(p, &self.b * &m00) * &m10 * 2u32
            + mpf!(p, &self.d * &m10) * &m10;
        let b = mpf!(p, &self.a * &m00) * &m01
            + mpf!(p, &self.b * &m00) * &m11
            + mpf!(p, &self.b * &m01) * &m10
            + mpf!(p, &self.d * &m10) * &m11;
        let d = mpf!(p, &self.a * &m01) * &m01
            + mpf!(p, &self.b * &m11) * &m01 * 2u32
            + mpf!(p, &self.d * &m11) * &m11;
        let [[n00, n01], [n10, n11]] = g.to_matrix(p);
        let px = mpf!(p, &n00 * &self.px) + mpf!(p, &n01 * &self.py);
        let py = mpf!(p, &n10 * &self.px) + mpf!(p, &n11 * &self.py);
        Self { a, b, d, px, py }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct EllipsePair {
    pub a: Ellipse,
    pub b: Ellipse,
}

impl EllipsePair {
    pub fn skew(&self) -> MpFloat {
        self.a.skew() + self.b.skew()
    }

    pub fn bias(&self) -> MpFloat {
        mpf!(self.a.prec(), self.b.bias() / self.a.bias())
    }

    /// G·pair: A by G, B by G's √2-conjugate.
    pub fn transform(&self, g: &GridOp) -> Self {
        Self {
            a: self.a.transform(g),
            b: self.b.transform(&g.conj_sq2()),
        }
    }
}

/// Solutions of a·t² + b·t + c ≤ 0:
/// the interval [t0, t1], or None if empty.
pub(crate) fn solve_quadratic(
    a: &MpFloat,
    b: &MpFloat,
    c: &MpFloat,
) -> Option<(MpFloat, MpFloat)> {
    let p = a.prec();
    let (a, b, c) = if a.is_sign_negative() && !a.is_zero() {
        (mpf!(p, -a), mpf!(p, -b), mpf!(p, -c))
    } else {
        (a.clone(), b.clone(), c.clone())
    };
    let disc = mpf!(p, &b * &b) - mpf!(p, &a * &c) * 4u32;
    if disc.is_sign_negative() && !disc.is_zero() {
        return None;
    }
    let sq = disc.sqrt();
    let s1 = mpf!(p, -&b) - &sq;
    let s2 = mpf!(p, -&b) + &sq;
    let two_a = mpf!(p, &a * 2u32);
    if !b.is_sign_negative() {
        Some((mpf!(p, &s1 / &two_a), mpf!(p, &s2 / &two_a)))
    } else if c.is_zero() {
        Some((MpFloat::with_val(p, 0.0), mpf!(p, mpf!(p, -&b) / &a)))
    } else {
        let two_c = mpf!(p, &c * 2u32);
        Some((mpf!(p, &two_c / &s2), mpf!(p, &two_c / &s1)))
    }
}


// ─── Grid operators ──────────────────────────────────────────────────────────



#[derive(Clone, Debug)]
pub(crate) struct GridOp {
    pub u0: ZOmegaBig,
    pub u1: ZOmegaBig,
}

impl GridOp {
    pub fn new(u0: ZOmegaBig, u1: ZOmegaBig) -> Self {
        Self { u0, u1 }
    }

    pub fn identity() -> Self {
        Self::new(ZOmegaBig::from_i64(1, 0, 0, 0), ZOmegaBig::from_i64(0, 0, 1, 0))
    }

    fn det_vec(&self) -> ZOmegaBig {
        self.u0.conj().mul(&self.u1)
    }

    pub fn is_special(&self) -> bool {
        // det ∈ {±ω²}: the ω/ω³ parts cancel and the ω² part is a unit.
        let v = self.det_vec();
        (&v.b + &v.d).complete() == 0 && (v.c == 1 || v.c == -1)
    }

    /// Real 2×2 matrix [[Re u0, Re u1], [Im u0, Im u1]].
    pub fn to_matrix(&self, prec: u32) -> [[MpFloat; 2]; 2] {
        [
            [self.u0.real(prec), self.u1.real(prec)],
            [self.u0.imag(prec), self.u1.imag(prec)],
        ]
    }

    /// Apply to a Z[ω] element viewed as a real 2-vector on the (1, i)
    /// grid basis; the integer half-sum combinations are exact for
    /// special operators.
    pub fn apply_zomega(&self, o: &ZOmegaBig) -> ZOmegaBig {
        let half = |x: Integer| -> Integer {
            debug_assert!(x.is_even(), "grid-op half-sum not even");
            x >> 1u32
        };
        // The transform formulas below are written in ω³..ω⁰ label
        // order; the ring stores (1, ω, ω², ω³), hence the reversed
        // field bindings.
        let (a0, b0, c0, d0) = (&self.u0.d, &self.u0.c, &self.u0.b, &self.u0.a);
        let (a1, b1, c1, d1) = (&self.u1.d, &self.u1.c, &self.u1.b, &self.u1.a);
        let ca0 = (c0 - a0).complete();
        let ca1 = (c1 - a1).complete();
        let cpa0 = (c0 + a0).complete();
        let cpa1 = (c1 + a1).complete();
        let bd0 = (b0 + d0).complete();
        let bd1 = (b1 + d1).complete();
        let bmd0 = (b0 - d0).complete();
        let bmd1 = (b1 - d1).complete();
        let new_d = (d0 * &o.a).complete()
            + (d1 * &o.c).complete()
            + half((&ca1 + &ca0).complete()) * &o.b
            + half((&ca1 - &ca0).complete()) * &o.d;
        let new_c = (c0 * &o.a).complete()
            + (c1 * &o.c).complete()
            + half((&bd1 + &bd0).complete()) * &o.b
            + half((&bd1 - &bd0).complete()) * &o.d;
        let new_b = (b0 * &o.a).complete()
            + (b1 * &o.c).complete()
            + half((&cpa1 + &cpa0).complete()) * &o.b
            + half((&cpa1 - &cpa0).complete()) * &o.d;
        let new_a = (a0 * &o.a).complete()
            + (a1 * &o.c).complete()
            + half((&bmd1 + &bmd0).complete()) * &o.b
            + half((&bmd1 - &bmd0).complete()) * &o.d;
        ZOmegaBig::from_parts(new_d, new_c, new_b, new_a)
    }

    pub fn apply_domega(&self, o: &DOmega) -> DOmega {
        DOmega::new(self.apply_zomega(&o.u), o.k)
    }

    pub fn compose(&self, o: &Self) -> Self {
        Self::new(self.apply_zomega(&o.u0), self.apply_zomega(&o.u1))
    }

    pub fn inv(&self) -> Option<Self> {
        if !self.is_special() {
            return None;
        }
        let (a0, b0, c0, d0) = (&self.u0.d, &self.u0.c, &self.u0.b, &self.u0.a);
        let (a1, b1, c1, d1) = (&self.u1.d, &self.u1.c, &self.u1.b, &self.u1.a);
        let two = |x: Integer| -> Integer { x >> 1u32 };
        let new_c0 = two(((c1 + a1).complete() - (c0 + a0).complete()).complete_id());
        let new_a0 = two(((-(c1 + a1).complete()).complete_id() - (c0 + a0).complete()).complete_id());
        let mut new_u0 = ZOmegaBig::from_parts(b1.clone(), new_c0, (-b0).complete(), new_a0);
        let new_c1 = two(((a1 - c1).complete() + (c0 - a0).complete()).complete_id());
        let new_a1 = two(((c1 - a1).complete() + (c0 - a0).complete()).complete_id());
        let mut new_u1 = ZOmegaBig::from_parts((-d1).complete(), new_c1, d0.clone(), new_a1);
        if self.det_vec().c == -1 {
            new_u0 = new_u0.neg();
            new_u1 = new_u1.neg();
        }
        Some(Self::new(new_u0, new_u1))
    }

    pub fn pow(&self, e: i64) -> Self {
        if e < 0 {
            return self.inv().expect("pow of non-special grid op").pow(-e);
        }
        let mut acc = Self::identity();
        let mut base = self.clone();
        let mut n = e;
        while n > 0 {
            if n & 1 == 1 {
                acc = acc.compose(&base);
            }
            base = base.compose(&base);
            n >>= 1;
        }
        acc
    }

    pub fn conj_sq2(&self) -> Self {
        Self::new(self.u0.conj_sq2(), self.u1.conj_sq2())
    }
}



// ─── Upright reduction ───────────────────────────────────────────────────────

// Mirrors GridOp(ZOmega(a0..d0), ZOmega(a1..d1)) literals from the paper.
#[allow(clippy::too_many_arguments)]
fn op(a0: i64, b0: i64, c0: i64, d0: i64, a1: i64, b1: i64, c1: i64, d1: i64) -> GridOp {
    // Call-site literals keep the paper's ω³..ω⁰ order; flip to crate order.
    GridOp::new(ZOmegaBig::from_i64(d0, c0, b0, a0), ZOmegaBig::from_i64(d1, c1, b1, a1))
}

/// λⁿ-shift of an upright-ish pair.
fn shift_pair(pair: &EllipsePair, n: i64, prec: u32) -> EllipsePair {
    let lambda = ZRootTwo::lambda();
    let ln = lambda.pow(n).to_mpfloat(prec);
    let lin = lambda.pow(-n).to_mpfloat(prec);
    let mut a = pair.a.clone();
    let mut b = pair.b.clone();
    let p = prec;
    a.a = mpf!(p, &a.a * &lin);
    a.d = mpf!(p, &a.d * &ln);
    b.a = mpf!(p, &b.a * &ln);
    b.d = mpf!(p, &b.d * &lin);
    if n & 1 == 1 {
        b.b = mpf!(p, -&b.b);
    }
    EllipsePair { a, b }
}

/// One reduction step; returns the updated (pair, left, right, done).
fn step_lemma(
    pair: EllipsePair,
    op_l: GridOp,
    op_r: GridOp,
    prec: u32,
) -> (EllipsePair, GridOp, GridOp, bool) {
    let reduce = |g: GridOp, pair: &EllipsePair, op_l: GridOp, op_r: GridOp| {
        let new_pair = pair.transform(&g);
        let new_r = g.compose(&op_r);
        (new_pair, op_l, new_r, false)
    };
    let a = &pair.a;
    let b = &pair.b;
    let lambda_ln = (1.0_f64 + std::f64::consts::SQRT_2).ln();
    if b.b.is_sign_negative() && !b.b.is_zero() {
        let g = op(0, 0, 0, 1, 0, -1, 0, 0);
        return reduce(g, &pair, op_l, op_r);
    }
    let bias_prod = (a.bias() * b.bias()).to_f64();
    if bias_prod < 1.0 {
        let g = op(0, 1, 0, 0, 0, 0, 0, 1);
        return reduce(g, &pair, op_l, op_r);
    }
    let pair_bias = pair.bias().to_f64();
    if !(0.029437..=33.971).contains(&pair_bias) {
        #[allow(clippy::cast_possible_truncation)]
        let n = (pair_bias.ln() / lambda_ln / 8.0).round() as i64;
        let s = op(-1, 0, 1, 1, 1, -1, 1, 0);
        return reduce(s.pow(n), &pair, op_l, op_r);
    }
    if pair.skew().to_f64() <= 15.0 {
        return (pair, op_l, op_r, true);
    }
    if !(0.17157..=5.8285).contains(&pair_bias) {
        #[allow(clippy::cast_possible_truncation)]
        let n = (pair_bias.ln() / lambda_ln / 4.0).round() as i64;
        let new_pair = shift_pair(&pair, n, prec);
        let (sig_l, sig_r) = if n >= 0 {
            (op(-1, 0, 1, 1, 0, 1, 0, 0).pow(n), op(0, 0, 0, 1, 1, -1, 1, 0).pow(n))
        } else {
            (op(-1, 0, 1, -1, 0, 1, 0, 0).pow(-n), op(0, 0, 0, 1, 1, 1, 1, 0).pow(-n))
        };
        let new_l = op_l.compose(&sig_l);
        let new_r = sig_r.compose(&op_r);
        return (new_pair, new_l, new_r, false);
    }
    let a_bias = a.bias().to_f64();
    let b_bias = b.bias().to_f64();
    if (0.24410..=4.0968).contains(&a_bias) && (0.24410..=4.0968).contains(&b_bias) {
        let g = op(0, 0, 1, 0, 1, 0, 0, 0);
        return reduce(g, &pair, op_l, op_r);
    }
    let a_b_nonneg = !a.b.is_sign_negative() || a.b.is_zero();
    if a_b_nonneg && a_bias <= 1.6969 {
        let g = op(-1, -1, 0, 0, 0, -1, 1, 0);
        return reduce(g, &pair, op_l, op_r);
    }
    if a_b_nonneg && b_bias <= 1.6969 {
        let g = op(1, -1, 0, 0, 0, -1, -1, 0);
        return reduce(g, &pair, op_l, op_r);
    }
    if a_b_nonneg {
        let n = 1.0_f64.max((a_bias.min(b_bias) / 4.0).sqrt().floor());
        #[allow(clippy::cast_possible_truncation)]
        let n = n as i64;
        let g = op(0, 0, 0, 1, 0, 1, 0, 2 * n);
        return reduce(g, &pair, op_l, op_r);
    }
    let n = 1.0_f64.max((a_bias.min(b_bias) / 2.0).sqrt().floor());
    #[allow(clippy::cast_possible_truncation)]
    let n = n as i64;
    let g = op(0, 0, 0, 1, n, 1, -n, 0);
    reduce(g, &pair, op_l, op_r)
}

/// Find the grid operator making (ellipseA, ellipseB) upright.
pub(crate) fn to_upright_ellipse_pair(ea: &Ellipse, eb: &Ellipse, prec: u32) -> GridOp {
    let pair = EllipsePair { a: ea.normalize(), b: eb.normalize() };
    let mut state = (pair, GridOp::identity(), GridOp::identity(), false);
    loop {
        state = step_lemma(state.0, state.1, state.2, prec);
        if state.3 {
            break;
        }
    }
    state.1.compose(&state.2)
}

/// Transform a set pair by its uprighting operator; returns
/// (G, upright A, upright B, bboxA, bboxB).
pub(crate) fn to_upright_set_pair(
    ea: &Ellipse,
    eb: &Ellipse,
    op_g: Option<GridOp>,
    prec: u32,
) -> (GridOp, Ellipse, Ellipse, Rectangle, Rectangle) {
    let g = op_g.unwrap_or_else(|| to_upright_ellipse_pair(ea, eb, prec));
    let pair = EllipsePair { a: ea.clone(), b: eb.clone() }.transform(&g);
    let bbox_a = pair.a.bbox();
    let bbox_b = pair.b.bbox();
    (g, pair.a, pair.b, bbox_a, bbox_b)
}
