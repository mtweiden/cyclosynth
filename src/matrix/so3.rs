//! Exact SO(3) matrices over Z[√2] (R2) or Z[γ] (R4 = Z[√(2+√2)]),
//! with per-entry rational representation: each entry = num / denom^exp.
//!
//! # R2 — Clifford+T
//!
//! Every SO(3) matrix arising from a Clifford+T unitary has entries in
//! Z[1/√2], i.e. of the form (a + b·√2) / √2^exp.
//!
//! # R4 — Clifford+√T
//!
//! Every SO(3) matrix arising from a Clifford+√T unitary has entries in
//! Z[1/γ] where γ = √(2+√2), i.e. (a + b√2 + cγ + dγ√2) / γ^exp.
//!
//! The basis of R4 is {1, √2, γ, γ√2} with γ² = 2+√2.
//!
//! # Convention
//!
//! Standard (column-major images): column j of the matrix is the image of
//! basis vector e_j under the rotation. Equivalently, M[i,j] is the
//! i-th component of R(e_j). Matrix–vector product R·v applies the rotation.
//!
//! # SO3<R2> from U2T
//!
//! Given U = [[u1, -ū2], [u2, ū1]] / √2^k with u1,u2 ∈ Z[ω]:
//!
//!   Let pp = u1², qq = u2², rr = u1·u2,
//!       cc = u1·conj(u2), nn1 = u1·conj(u1), nn2 = u2·conj(u2).
//!
//! The SO(3) matrix (standard convention, columns = images) entries:
//!
//!   [ax, ay, az]    ax = Re(pp-qq)   ay = Im(pp-qq)   az = 2Re(cc)
//!   [bx, by, bz]    bx = -Im(pp+qq)  by = Re(pp+qq)   bz = -2Im(cc)
//!   [cx, cy, cz]    cx = -2Re(rr)    cy = -2Im(rr)    cz = Re(nn1-nn2)
//!
//! all divided by 2^k = √2^{2k}.  With exp = 2k+1 (one extra √2),
//! the R2 numerators are:
//!   Re(z) as R2 with exp=1: R2(b-d, a)   for z = a+bω+cω²+dω³
//!   Im(z) as R2 with exp=1: R2(b+d, c)
//!   2Re(z) as R2 with exp=1: R2(2(b-d), 2a)
//!   2Im(z) as R2 with exp=1: R2(2(b+d), 2c)

// 3×3 matrix code reads more clearly with explicit (i, j) indexing than with
// iterator combinators threading multiple arrays in lockstep.
#![allow(clippy::needless_range_loop)]

use crate::matrix::u2::U2;
use crate::matrix::{U2Q, U2T};
use crate::rings::types::{int_to_f64, INT_ONE, INT_THREE, INT_TWO, INT_ZERO};
use crate::rings::{Int, ZOmega, ZOmicron, ZZeta};
#[cfg(test)]
use std::f64::consts::FRAC_1_SQRT_2;
use std::f64::consts::SQRT_2;
use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};

// ─── R2 ───────────────────────────────────────────────────────────────────────

/// An element of Z[√2]: `a + b·√2`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct R2(pub Int, pub Int);

impl R2 {
    pub const ZERO: Self = R2(INT_ZERO, INT_ZERO);
    pub const ONE: Self = R2(INT_ONE, INT_ZERO);

    /// Construct from small integer coefficients.
    #[inline]
    pub const fn from_i32(a: i32, b: i32) -> Self {
        R2(Int::from_i32(a), Int::from_i32(b))
    }

    /// Multiply by √2: (a + b√2)·√2 = 2b + a·√2.
    #[inline]
    pub fn mul_sqrt2(self) -> Self {
        R2(INT_TWO * self.1, self.0)
    }

    /// Divide by √2 (exact; panics in debug if self.0 is odd).
    #[inline]
    pub fn div_sqrt2(self) -> Self {
        debug_assert!(
            self.0 % INT_TWO == INT_ZERO,
            "R2::div_sqrt2: a must be even, got R2({},{})",
            self.0,
            self.1,
        );
        R2(self.1, self.0 / INT_TWO)
    }

    /// Largest n such that √2^n divides this element.
    pub fn sqrt2_valuation(self) -> u32 {
        if self.0 == INT_ZERO && self.1 == INT_ZERO {
            return u32::MAX;
        }
        let mut v = 0u32;
        let mut x = self;
        loop {
            if x.0 % INT_TWO != INT_ZERO {
                break;
            }
            x = R2(x.1, x.0 / INT_TWO);
            v += 1;
            if x.0 % INT_TWO != INT_ZERO {
                break;
            }
        }
        v
    }

    /// Convert to f64.
    pub fn to_f64(self) -> f64 {
        int_to_f64(self.0) + int_to_f64(self.1) * SQRT_2
    }
}

impl Add for R2 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        R2(self.0 + rhs.0, self.1 + rhs.1)
    }
}
impl Sub for R2 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        R2(self.0 - rhs.0, self.1 - rhs.1)
    }
}
impl Neg for R2 {
    type Output = Self;
    fn neg(self) -> Self {
        R2(-self.0, -self.1)
    }
}
impl Mul for R2 {
    type Output = Self;
    /// (a + b√2)(c + d√2) = (ac + 2bd) + (ad + bc)√2.
    fn mul(self, rhs: Self) -> Self {
        R2(
            self.0 * rhs.0 + INT_TWO * self.1 * rhs.1,
            self.0 * rhs.1 + self.1 * rhs.0,
        )
    }
}

// ─── R4 ───────────────────────────────────────────────────────────────────────

/// An element of Z[γ] where γ = √(2+√2):
///   `a + b·√2 + c·γ + d·γ√2`.
///
/// Basis: {1, √2, γ, γ√2} with γ² = 2+√2.
///
/// Multiplication rules derived from γ² = 2+√2, (√2)² = 2:
///   (γ√2)² = γ²·2 = (2+√2)·2 = 4+2√2
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct R4(pub Int, pub Int, pub Int, pub Int);

impl R4 {
    pub const ZERO: Self = R4(INT_ZERO, INT_ZERO, INT_ZERO, INT_ZERO);
    pub const ONE: Self = R4(INT_ONE, INT_ZERO, INT_ZERO, INT_ZERO);

    /// Construct from small integer coefficients.
    #[inline]
    pub const fn from_i32(a: i32, b: i32, c: i32, d: i32) -> Self {
        R4(
            Int::from_i32(a),
            Int::from_i32(b),
            Int::from_i32(c),
            Int::from_i32(d),
        )
    }

    /// Convert to f64.
    pub fn to_f64(self) -> f64 {
        let sqrt2 = SQRT_2;
        let gamma = (2.0f64 + sqrt2).sqrt(); // √(2+√2)
        let gamma_s = gamma * sqrt2; // γ·√2
        int_to_f64(self.0)
            + int_to_f64(self.1) * sqrt2
            + int_to_f64(self.2) * gamma
            + int_to_f64(self.3) * gamma_s
    }

    /// γ-adic valuation: largest n such that γ^n divides self.
    ///
    /// Uses the fact that γ · (2−√2)γ = 2, so testing divisibility by γ
    /// is equivalent to: `self * R4(0,0,2,-1)` has all even coefficients.
    pub fn gamma_valuation(self) -> u32 {
        if self == R4::ZERO {
            return u32::MAX;
        }
        let divisor = R4::from_i32(0, 0, 2, -1);
        let mut v = 0u32;
        let mut x = self;
        loop {
            let t = x * divisor;
            if t.0 % INT_TWO != INT_ZERO
                || t.1 % INT_TWO != INT_ZERO
                || t.2 % INT_TWO != INT_ZERO
                || t.3 % INT_TWO != INT_ZERO
            {
                break;
            }
            x = R4(t.0 / INT_TWO, t.1 / INT_TWO, t.2 / INT_TWO, t.3 / INT_TWO);
            v += 1;
        }
        v
    }

    /// Divide by γ (exact; panics in debug if not divisible).
    pub fn div_gamma(self) -> Self {
        let t = self * R4::from_i32(0, 0, 2, -1);
        debug_assert!(
            t.0 % INT_TWO == INT_ZERO
                && t.1 % INT_TWO == INT_ZERO
                && t.2 % INT_TWO == INT_ZERO
                && t.3 % INT_TWO == INT_ZERO,
            "R4::div_gamma: not divisible by γ, got R4({},{},{},{})",
            self.0,
            self.1,
            self.2,
            self.3
        );
        R4(t.0 / INT_TWO, t.1 / INT_TWO, t.2 / INT_TWO, t.3 / INT_TWO)
    }

    /// Multiply by γ: (a + b√2 + cγ + dγ√2)·γ = (2c+2d) + (c+2d)√2 + aγ + bγ√2.
    ///
    /// Derived from γ² = 2+√2:
    ///   aγ + b√2·γ + cγ² + dγ²√2
    ///   = aγ + bγ√2 + c(2+√2) + d(2+√2)√2
    ///   = (2c+2d) + (c+2d)√2 + aγ + bγ√2
    pub fn mul_gamma(self) -> Self {
        R4(
            INT_TWO * self.2 + INT_TWO * self.3,
            self.2 + INT_TWO * self.3,
            self.0,
            self.1,
        )
    }

    /// √2-adic valuation: largest n such that √2^n divides self.
    ///
    /// Iterated div_sqrt2 until impossible. R4(a, b, c, d) is divisible by √2
    /// iff a and c are even (since √2·R4(b, a/2, d, c/2) = R4(a, b, c, d)).
    pub fn sqrt2_valuation(self) -> u32 {
        if self == R4::ZERO {
            return u32::MAX;
        }
        let mut v = 0u32;
        let mut x = self;
        loop {
            if x.0 % INT_TWO != INT_ZERO || x.2 % INT_TWO != INT_ZERO {
                break;
            }
            x = R4(x.1, x.0 / INT_TWO, x.3, x.2 / INT_TWO);
            v += 1;
        }
        v
    }

    /// Divide by √2 (exact; panics in debug if not divisible).
    /// Requires self.0 and self.2 to be even.
    pub fn div_sqrt2(self) -> Self {
        debug_assert!(
            self.0 % INT_TWO == INT_ZERO && self.2 % INT_TWO == INT_ZERO,
            "R4::div_sqrt2: not divisible by √2, got R4({},{},{},{})",
            self.0,
            self.1,
            self.2,
            self.3
        );
        R4(self.1, self.0 / INT_TWO, self.3, self.2 / INT_TWO)
    }

    /// Multiply by √2: (a + b√2 + cγ + dγ√2)·√2 = 2b + a√2 + 2dγ + cγ√2.
    pub fn mul_sqrt2(self) -> Self {
        R4(INT_TWO * self.1, self.0, INT_TWO * self.3, self.2)
    }
}

impl Add for R4 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        R4(
            self.0 + rhs.0,
            self.1 + rhs.1,
            self.2 + rhs.2,
            self.3 + rhs.3,
        )
    }
}
impl Sub for R4 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        R4(
            self.0 - rhs.0,
            self.1 - rhs.1,
            self.2 - rhs.2,
            self.3 - rhs.3,
        )
    }
}
impl Neg for R4 {
    type Output = Self;
    fn neg(self) -> Self {
        R4(-self.0, -self.1, -self.2, -self.3)
    }
}

/// Multiplication in Z[γ].
///
/// (a + b√2 + cγ + dγ√2)(w + x√2 + yγ + zγ√2):
///   coefficient of 1:   aw + 2bx + 2cy + 2cz + 2dy + 4dz
///   coefficient of √2:  ax + bw  + cy  + 2cz + 2dy + 2dz
///   coefficient of γ:   ay + 2bz + cw  + 2dx
///   coefficient of γ√2: az + by  + cx  + dw
impl Mul for R4 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let (a, b, c, d) = (self.0, self.1, self.2, self.3);
        let (w, x, y, z) = (rhs.0, rhs.1, rhs.2, rhs.3);
        let t2 = INT_TWO;
        let t4 = crate::rings::types::INT_FOUR;
        R4(
            a * w + t2 * b * x + t2 * c * y + t2 * c * z + t2 * d * y + t4 * d * z,
            a * x + b * w + c * y + t2 * c * z + t2 * d * y + t2 * d * z,
            a * y + t2 * b * z + c * w + t2 * d * x,
            a * z + b * y + c * x + d * w,
        )
    }
}

// ─── Display helpers ─────────────────────────────────────────────────────────

fn fmt_poly(terms: &[(Int, &str)], f: &mut fmt::Formatter<'_>) -> fmt::Result {
    let mut first = true;
    for &(coeff, sym) in terms {
        if coeff == INT_ZERO {
            continue;
        }
        let neg = coeff < INT_ZERO;
        let abs = if neg { -coeff } else { coeff };
        if first {
            if sym.is_empty() {
                write!(f, "{coeff}")?;
            } else if abs == INT_ONE {
                write!(f, "{}{sym}", if neg { "-" } else { "" })?;
            } else {
                write!(f, "{coeff}{sym}")?;
            }
            first = false;
        } else {
            let sign = if neg { " - " } else { " + " };
            if sym.is_empty() {
                write!(f, "{sign}{abs}")?;
            } else if abs == INT_ONE {
                write!(f, "{sign}{sym}")?;
            } else {
                write!(f, "{sign}{abs}{sym}")?;
            }
        }
    }
    if first {
        write!(f, "0")?;
    }
    Ok(())
}

impl fmt::Display for R2 {
    /// Formats as a polynomial in √2, e.g. `3 - 2√2`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_poly(&[(self.0, ""), (self.1, "√2")], f)
    }
}

impl fmt::Display for R4 {
    /// Formats as a polynomial in {1, √2, γ, γ√2}, e.g. `1 + 2γ - γ√2`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_poly(
            &[(self.0, ""), (self.1, "√2"), (self.2, "γ"), (self.3, "γ√2")],
            f,
        )
    }
}

// ─── Ratio<R> ────────────────────────────────────────────────────────────────

/// A ring element divided by the ring-specific denominator unit to the power `exp`.
///
/// - For `Ratio<R2>`: actual value = `num / √2^exp`
/// - For `Ratio<R4>`: actual value = `num / √2^exp`
///
/// Note: `Ratio<R4>` uses √2 as the denominator base (same as `Ratio<R2>`),
/// not γ. This is because SO3 entries from a U2 with k denominator have
/// the form `Re(z)/2^k = Re(z)/√2^(2k)`. The R4 ring still includes
/// `mul_gamma`/`div_gamma`/`gamma_valuation` as algebraic operations on the
/// ring itself, but `Ratio<R4>` doesn't use them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ratio<R> {
    pub num: R,
    pub exp: u32,
}

// ─── Ratio<R2> ───────────────────────────────────────────────────────────────

impl Ratio<R2> {
    pub const ZERO: Self = Ratio {
        num: R2::ZERO,
        exp: 0,
    };
    pub const ONE: Self = Ratio {
        num: R2::ONE,
        exp: 0,
    };

    /// Cancel common √2 factors between numerator and denominator.
    pub fn simplify(&mut self) {
        if self.num == R2::ZERO {
            self.exp = 0;
            return;
        }
        let v = self.num.sqrt2_valuation().min(self.exp);
        for _ in 0..v {
            self.num = self.num.div_sqrt2();
        }
        self.exp -= v;
    }

    /// Multiply numerator by √2^n (used to align exponents before addition).
    fn lift_num(self, n: u32) -> R2 {
        let mut x = self.num;
        for _ in 0..n {
            x = x.mul_sqrt2();
        }
        x
    }

    pub fn to_f64(self) -> f64 {
        self.num.to_f64() / (self.exp as f64 / 2.0).exp2()
    }
}

impl Neg for Ratio<R2> {
    type Output = Self;
    fn neg(self) -> Self {
        Ratio {
            num: -self.num,
            exp: self.exp,
        }
    }
}

impl Mul for Ratio<R2> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let mut r = Ratio {
            num: self.num * rhs.num,
            exp: self.exp + rhs.exp,
        };
        r.simplify();
        r
    }
}

impl Add for Ratio<R2> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let max_e = self.exp.max(rhs.exp);
        let lhs_num = self.lift_num(max_e - self.exp);
        let rhs_num = rhs.lift_num(max_e - rhs.exp);
        let mut r = Ratio {
            num: lhs_num + rhs_num,
            exp: max_e,
        };
        r.simplify();
        r
    }
}

// ─── Ratio<R4> ───────────────────────────────────────────────────────────────

impl Ratio<R4> {
    pub const ZERO: Self = Ratio {
        num: R4::ZERO,
        exp: 0,
    };
    pub const ONE: Self = Ratio {
        num: R4::ONE,
        exp: 0,
    };

    /// Cancel common √2 factors between numerator and denominator.
    pub fn simplify(&mut self) {
        if self.num == R4::ZERO {
            self.exp = 0;
            return;
        }
        let v = self.num.sqrt2_valuation().min(self.exp);
        for _ in 0..v {
            self.num = self.num.div_sqrt2();
        }
        self.exp -= v;
    }

    /// Multiply numerator by √2^n (used to align exponents before addition).
    fn lift_num(self, n: u32) -> R4 {
        let mut x = self.num;
        for _ in 0..n {
            x = x.mul_sqrt2();
        }
        x
    }

    pub fn to_f64(self) -> f64 {
        self.num.to_f64() / SQRT_2.powi(self.exp as i32)
    }
}

impl Neg for Ratio<R4> {
    type Output = Self;
    fn neg(self) -> Self {
        Ratio {
            num: -self.num,
            exp: self.exp,
        }
    }
}

impl Mul for Ratio<R4> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let mut r = Ratio {
            num: self.num * rhs.num,
            exp: self.exp + rhs.exp,
        };
        r.simplify();
        r
    }
}

impl Add for Ratio<R4> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let max_e = self.exp.max(rhs.exp);
        let lhs_num = self.lift_num(max_e - self.exp);
        let rhs_num = rhs.lift_num(max_e - rhs.exp);
        let mut r = Ratio {
            num: lhs_num + rhs_num,
            exp: max_e,
        };
        r.simplify();
        r
    }
}

// ─── Generic SO3 ─────────────────────────────────────────────────────────────

/// A 3×3 SO(3) matrix with entries in ring R, stored as per-entry ratios.
///
/// Each entry `e[3*row + col]` is a `Ratio<R>`: the actual matrix value is
/// `e[i].num / denom^{e[i].exp}`, where `denom` is √2 for R2 or γ for R4.
///
/// This mirrors the Python `SO3Matrix` design where each entry is an
/// `AlgebraicIntegerOverRoot2` with its own `denominator_power`.
///
/// Standard convention: column j = image of basis vector e_j.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SO3<R> {
    /// 9 entries in row-major order: e[3*row + col].
    pub e: [Ratio<R>; 9],
}

// ─── SO3<R2> ──────────────────────────────────────────────────────────────────

impl SO3<R2> {
    /// Identity matrix (all entries have exp=0).
    pub fn identity() -> Self {
        let mut e = [Ratio::<R2>::ZERO; 9];
        e[0] = Ratio::<R2>::ONE;
        e[4] = Ratio::<R2>::ONE;
        e[8] = Ratio::<R2>::ONE;
        SO3 { e }
    }

    #[inline]
    pub fn get(&self, r: usize, c: usize) -> Ratio<R2> {
        self.e[3 * r + c]
    }

    /// Maximum denominator exponent across all non-zero entries.
    pub fn maximum_denominator_exponent(&self) -> u32 {
        self.e
            .iter()
            .filter(|r| r.num != R2::ZERO)
            .map(|r| r.exp)
            .max()
            .unwrap_or(0)
    }

    /// Simplify each entry individually (cancel √2 from numerator and denominator).
    pub fn reduce(&mut self) {
        for entry in self.e.iter_mut() {
            entry.simplify();
        }
    }

    /// Convert to 3×3 float matrix.
    pub fn to_float(&self) -> [[f64; 3]; 3] {
        let mut out = [[0.0f64; 3]; 3];
        for r in 0..3 {
            for c in 0..3 {
                out[r][c] = self.e[3 * r + c].to_f64();
            }
        }
        out
    }

    /// Build SO3<R2> from a U2T matrix using exact ZOmega ring arithmetic.
    ///
    /// Works for any unitary matrix (not just SU(2)) with entries in Z[ω].
    /// Column j of the result is the image of Bloch basis vector e_j under the rotation.
    ///
    /// Derivation: M_ij = (1/2)·Tr(σ_i · U·σ_j·U†) with
    ///   P = u11·u22† + u12·u21†,  Q = u11·u22† − u12·u21†,
    ///   R = u11·u12† − u21·u22†,  S = u11·u21† − u12·u22†,
    ///   N = u11·u11† − u12·u12† − u21·u21† + u22·u22†  (real, always even).
    ///
    ///   ax = Re(P),  bx = −Im(P),  cx = Re(R)
    ///   ay = Im(Q),  by =  Re(Q),  cy = Im(R)
    ///   az = Re(S),  bz = −Im(S),  cz = N/2
    pub fn from_u2(u: &U2T) -> Self {
        let a = u.u11;
        let b = u.u12;
        let c = u.u21;
        let d = u.u22;
        let k = u.k;

        let ad = a * d.conj();
        let bc = b * c.conj();
        let p = ad + bc; // → ax, bx
        let q = ad - bc; // → by, ay
        let r = a * b.conj() - c * d.conj(); // → cx, cy
        let s = a * c.conj() - b * d.conj(); // → az, bz
                                             // N is a real ZOmega element ZOmega(n, 0, 0, 0) with n always even.
        let n = a * a.conj() - b * b.conj() - c * c.conj() + d * d.conj();

        // Re(z)·√2 = R2(z.b − z.d, z.a),  Im(z)·√2 = R2(z.b + z.d, z.c).
        let re = |z: ZOmega| R2(z.b - z.d, z.a);
        let im = |z: ZOmega| R2(z.b + z.d, z.c);

        let init_exp = 2 * k + 1;
        let raw = [
            re(p),
            im(q),
            re(s),
            -im(p),
            re(q),
            -im(s),
            re(r),
            im(r),
            R2((n.b - n.d) / INT_TWO, n.a / INT_TWO),
        ];
        let e: [Ratio<R2>; 9] = std::array::from_fn(|i| {
            let mut entry = Ratio {
                num: raw[i],
                exp: init_exp,
            };
            entry.simplify();
            entry
        });
        SO3 { e }
    }
}

impl Mul for SO3<R2> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let mut e = [Ratio::<R2>::ZERO; 9];
        for r in 0..3 {
            for c in 0..3 {
                // Compute three products a_rk * b_kc
                let products: [Ratio<R2>; 3] =
                    std::array::from_fn(|k| self.e[3 * r + k] * rhs.e[3 * k + c]);
                // Find max exponent to align before summing
                let max_e = products.iter().map(|p| p.exp).max().unwrap();
                let sum = products
                    .iter()
                    .fold(R2::ZERO, |acc, p| acc + p.lift_num(max_e - p.exp));
                let mut entry = Ratio {
                    num: sum,
                    exp: max_e,
                };
                entry.simplify();
                e[3 * r + c] = entry;
            }
        }
        SO3 { e }
    }
}

// ─── SO3<R4> ──────────────────────────────────────────────────────────────────

impl SO3<R4> {
    /// Identity matrix (all entries have exp=0).
    pub fn identity() -> Self {
        let mut e = [Ratio::<R4>::ZERO; 9];
        e[0] = Ratio::<R4>::ONE;
        e[4] = Ratio::<R4>::ONE;
        e[8] = Ratio::<R4>::ONE;
        SO3 { e }
    }

    #[inline]
    pub fn get(&self, r: usize, c: usize) -> Ratio<R4> {
        self.e[3 * r + c]
    }

    /// Maximum denominator exponent across all non-zero entries.
    pub fn maximum_denominator_exponent(&self) -> u32 {
        self.e
            .iter()
            .filter(|r| r.num != R4::ZERO)
            .map(|r| r.exp)
            .max()
            .unwrap_or(0)
    }

    /// Simplify each entry individually (cancel γ from numerator and denominator).
    pub fn reduce(&mut self) {
        for entry in self.e.iter_mut() {
            entry.simplify();
        }
    }

    /// Convert to 3×3 float matrix.
    pub fn to_float(&self) -> [[f64; 3]; 3] {
        let mut out = [[0.0f64; 3]; 3];
        for r in 0..3 {
            for c in 0..3 {
                out[r][c] = self.e[3 * r + c].to_f64();
            }
        }
        out
    }

    /// Build SO3<R4> from a U2Q matrix using exact ZZeta ring arithmetic.
    ///
    /// Works for any unitary matrix (not just SU(2)) with entries in Z[ζ].
    ///
    /// Derivation: SO3 from U2. Mirrors the SO3<R2>::from_u2 structure but
    /// extracts Re/Im into R4 = Z[γ] = Z[1, √2, γ, γ√2].
    ///
    /// For z ∈ Z[ζ_16], 2·Re(z) and 2·Im(z) land in Z[γ] (the /2 factors in
    /// cos(jπ/8), sin(jπ/8) are absorbed by the leading 2). Using
    /// Ratio<R4>'s √2-denominator and `init_exp = 2k+2`, each entry value is
    ///   `entry = num / √2^(2k+2) = num / 2^(k+1)`
    /// Off-diagonal entries: num = 2·Re/Im(z), so entry = Re/Im(z)/2^k. ✓
    /// (3,3) entry: num = N (real Z[γ] element), so entry = N/2^(k+1). ✓
    ///
    ///   re3(z) = 2·Re(z) in R4 basis:
    ///     R4(2·z.a, z.c−z.g, z.b−z.h−z.d+z.f, z.d−z.f)
    ///   im3(z) = 2·Im(z) in R4 basis:
    ///     R4(2·z.e, z.c+z.g, z.d+z.f−z.b−z.h, z.b+z.h)
    ///   cz: N is real Z[ζ_16] (N.e=0, N.f=−N.d, N.g=−N.c, N.h=−N.b). In
    ///     Z[γ] basis {1, √2, γ, γ√2}, N = R4(N.a, N.c, N.b−N.d, N.d).
    pub fn from_u2(u: &U2Q) -> Self {
        let a = u.u11;
        let b = u.u12;
        let c = u.u21;
        let d = u.u22;
        let k = u.k;

        let ad = a * d.conj();
        let bc = b * c.conj();
        let p = ad + bc;
        let q = ad - bc;
        let r = a * b.conj() - c * d.conj();
        let s = a * c.conj() - b * d.conj();
        let n = a * a.conj() - b * b.conj() - c * c.conj() + d * d.conj();

        // 2·Re(z) for z = a + bζ + cζ² + dζ³ + eζ⁴ + fζ⁵ + gζ⁶ + hζ⁷.
        // Re(z) = a + (b−h)·γ/2 + (c−g)·√2/2 + (d−f)·(γ√2−γ)/2,
        // so 2·Re(z) = 2a + (c−g)·√2 + (b−h−d+f)·γ + (d−f)·γ√2.
        let re3 =
            |z: ZZeta| -> R4 { R4(INT_TWO * z.a, z.c - z.g, z.b - z.h - z.d + z.f, z.d - z.f) };
        // 2·Im(z) for z's coefficients (z.e is the imaginary-axis component).
        // Im(z) = e + (c+g)·√2/2 + (d+f−b−h)·γ/2... wait, Im of cos terms is 0,
        // and Im of i·ζ^k = sin terms. Recompute:
        //   Im(z) = b·sin(π/8) + c·sin(π/4) + d·sin(3π/8) + e·sin(π/2)
        //         + f·sin(5π/8) + g·sin(3π/4) + h·sin(7π/8)
        //         = e + (c+g)·√2/2 + (d+f)·γ/2 + (b+h)·(γ√2−γ)/2
        // So 2·Im(z) = 2e + (c+g)·√2 + (d+f−b−h)·γ + (b+h)·γ√2.
        let im3 =
            |z: ZZeta| -> R4 { R4(INT_TWO * z.e, z.c + z.g, z.d + z.f - z.b - z.h, z.b + z.h) };

        // N is a real Z[ζ_16] element: N.e = 0, N.f = -N.d, N.g = -N.c,
        // N.h = -N.b. In R4 basis it's R4(N.a, N.c, N.b−N.d, N.d).
        let cz = R4(n.a, n.c, n.b - n.d, n.d);

        let init_exp = 2 * k + 2;
        let raw: [R4; 9] = [
            re3(p),
            im3(q),
            re3(s),
            -im3(p),
            re3(q),
            -im3(s),
            re3(r),
            im3(r),
            cz,
        ];
        let e: [Ratio<R4>; 9] = std::array::from_fn(|i| {
            let mut entry = Ratio {
                num: raw[i],
                exp: init_exp,
            };
            entry.simplify();
            entry
        });
        SO3 { e }
    }
}

impl Mul for SO3<R4> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let mut e = [Ratio::<R4>::ZERO; 9];
        for r in 0..3 {
            for c in 0..3 {
                let products: [Ratio<R4>; 3] =
                    std::array::from_fn(|k| self.e[3 * r + k] * rhs.e[3 * k + c]);
                let max_e = products.iter().map(|p| p.exp).max().unwrap();
                let sum = products
                    .iter()
                    .fold(R4::ZERO, |acc, p| acc + p.lift_num(max_e - p.exp));
                let mut entry = Ratio {
                    num: sum,
                    exp: max_e,
                };
                entry.simplify();
                e[3 * r + c] = entry;
            }
        }
        SO3 { e }
    }
}

// ─── Type aliases ─────────────────────────────────────────────────────────────

/// SO3 matrix for Clifford+T unitaries (denominator in Z[√2]).
pub type SO3T = SO3<R2>;

/// SO3 matrix for Clifford+√T unitaries (denominator in Z[γ]).
pub type SO3Q = SO3<R4>;

// ─── Display for SO3 ─────────────────────────────────────────────────────────

/// Display SO3 with per-row denominators (matching Python's `exponents()` approach).
///
/// Each row is shown with its own `/ denom^row_exp`, where `row_exp` is the
/// maximum entry exponent in that row. Numerators are lifted to `row_exp` for display.
fn fmt_so3_rows_r2(e: &[Ratio<R2>; 9], f: &mut fmt::Formatter<'_>) -> fmt::Result {
    for row in 0..3 {
        let row_exp = (0..3).map(|c| e[3 * row + c].exp).max().unwrap_or(0);
        write!(f, "[")?;
        for col in 0..3 {
            if col > 0 {
                write!(f, ", ")?;
            }
            let lifted = e[3 * row + col].lift_num(row_exp - e[3 * row + col].exp);
            write!(f, "{lifted}")?;
        }
        write!(f, "]")?;
        if row_exp > 0 {
            write!(f, " / √2^{row_exp}")?;
        }
        if row < 2 {
            writeln!(f)?;
        }
    }
    Ok(())
}

fn fmt_so3_rows_r4(e: &[Ratio<R4>; 9], f: &mut fmt::Formatter<'_>) -> fmt::Result {
    for row in 0..3 {
        let row_exp = (0..3).map(|c| e[3 * row + c].exp).max().unwrap_or(0);
        write!(f, "[")?;
        for col in 0..3 {
            if col > 0 {
                write!(f, ", ")?;
            }
            let lifted = e[3 * row + col].lift_num(row_exp - e[3 * row + col].exp);
            write!(f, "{lifted}")?;
        }
        write!(f, "]")?;
        if row_exp > 0 {
            write!(f, " / γ^{row_exp}")?;
        }
        if row < 2 {
            writeln!(f)?;
        }
    }
    Ok(())
}

impl fmt::Display for SO3<R2> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_so3_rows_r2(&self.e, f)
    }
}

impl fmt::Display for SO3<R4> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_so3_rows_r4(&self.e, f)
    }
}

// ─── SO3Ops trait ─────────────────────────────────────────────────────────────

/// Minimal interface required by `BlochDecomposer` to work generically over ring.
///
/// Implemented by both `SO3<R2>` (Clifford+T) and `SO3<R4>` (Clifford+√T).
pub trait SO3Ops: Clone + Sized + Mul<Output = Self> {
    /// Maximum denominator exponent across all 9 entries.
    ///
    /// This correctly reflects the current "level" of the SO3 matrix — how
    /// many more gate peel-off steps the decomposer needs to take.
    fn max_exp(&self) -> u32;
    /// Left-multiply in place: `self ← rhs · self`.
    fn left_mul(&mut self, rhs: &Self);
}

impl SO3Ops for SO3<R2> {
    fn max_exp(&self) -> u32 {
        self.maximum_denominator_exponent()
    }
    fn left_mul(&mut self, rhs: &Self) {
        *self = rhs.clone() * self.clone();
    }
}

impl SO3Ops for SO3<R4> {
    fn max_exp(&self) -> u32 {
        self.maximum_denominator_exponent()
    }
    fn left_mul(&mut self, rhs: &Self) {
        *self = rhs.clone() * self.clone();
    }
}

// ─── Rotation factories for SO3<R2> (π/4 steps) ──────────────────────────────
//
// Standard Rz(π/4):  [[cos,-sin,0],[sin,cos,0],[0,0,1]]
//                   = [[1,-1,0],[1,1,0],[0,0,√2]] / √2
// with exp=1.

/// Rz(+π/4) as SO3<R2>.
pub fn rz_pos() -> SO3<R2> {
    let mut e = [Ratio::<R2>::ZERO; 9];
    e[0] = Ratio {
        num: R2::from_i32(1, 0),
        exp: 1,
    }; // cos(π/4) = 1/√2
    e[1] = Ratio {
        num: R2::from_i32(-1, 0),
        exp: 1,
    }; // -sin(π/4) = -1/√2
    e[3] = Ratio {
        num: R2::from_i32(1, 0),
        exp: 1,
    }; // sin(π/4) = 1/√2
    e[4] = Ratio {
        num: R2::from_i32(1, 0),
        exp: 1,
    }; // cos(π/4) = 1/√2
    e[8] = Ratio {
        num: R2::from_i32(0, 1),
        exp: 1,
    }; // 1 = √2/√2
    SO3 { e }
}

/// Rz(-π/4) = Rz(+π/4)ᵀ.
pub fn rz_neg() -> SO3<R2> {
    let mut m = rz_pos();
    m.e.swap(1, 3);
    m.e.swap(2, 6);
    m.e.swap(5, 7);
    m
}

/// Rx(+π/4) as SO3<R2>.
pub fn rx_pos() -> SO3<R2> {
    let mut e = [Ratio::<R2>::ZERO; 9];
    e[0] = Ratio {
        num: R2::from_i32(0, 1),
        exp: 1,
    };
    e[4] = Ratio {
        num: R2::from_i32(1, 0),
        exp: 1,
    };
    e[5] = Ratio {
        num: R2::from_i32(-1, 0),
        exp: 1,
    };
    e[7] = Ratio {
        num: R2::from_i32(1, 0),
        exp: 1,
    };
    e[8] = Ratio {
        num: R2::from_i32(1, 0),
        exp: 1,
    };
    SO3 { e }
}

/// Rx(-π/4) = Rx(+π/4)ᵀ.
pub fn rx_neg() -> SO3<R2> {
    let mut m = rx_pos();
    m.e.swap(1, 3);
    m.e.swap(2, 6);
    m.e.swap(5, 7);
    m
}

/// Ry(+π/4) as SO3<R2>.
pub fn ry_pos() -> SO3<R2> {
    let mut e = [Ratio::<R2>::ZERO; 9];
    e[0] = Ratio {
        num: R2::from_i32(1, 0),
        exp: 1,
    };
    e[2] = Ratio {
        num: R2::from_i32(1, 0),
        exp: 1,
    };
    e[4] = Ratio {
        num: R2::from_i32(0, 1),
        exp: 1,
    };
    e[6] = Ratio {
        num: R2::from_i32(-1, 0),
        exp: 1,
    };
    e[8] = Ratio {
        num: R2::from_i32(1, 0),
        exp: 1,
    };
    SO3 { e }
}

/// Ry(-π/4) = Ry(+π/4)ᵀ.
pub fn ry_neg() -> SO3<R2> {
    let mut m = ry_pos();
    m.e.swap(1, 3);
    m.e.swap(2, 6);
    m.e.swap(5, 7);
    m
}

// ─── Rotation factories for SO3<R4> (π/8 steps) ──────────────────────────────
//
// SO3 entries in the new √2-denominator convention:
//   cos(π/8) = γ/2 = R4(0,0,1,0) / √2²
//   sin(π/8) = (γ√2−γ)/2 = R4(0,0,−1,1) / √2²
//   −sin(π/8) = R4(0,0,1,−1) / √2²
//   1 = R4(2,0,0,0) / √2²  (held at exp=2 for uniform-exp matrix storage;
//                            simplify reduces to R4(1,0,0,0)/√2⁰)
//
// Standard Rz(π/8) = [[cos,-sin,0],[sin,cos,0],[0,0,1]], pre-simplify exp=2.

/// Rz(+π/8) as SO3<R4>.
pub fn rz_pos_q() -> SO3<R4> {
    let mut e = [Ratio::<R4>::ZERO; 9];
    e[0] = Ratio {
        num: R4::from_i32(0, 0, 1, 0),
        exp: 2,
    }; // cos(π/8)
    e[1] = Ratio {
        num: R4::from_i32(0, 0, 1, -1),
        exp: 2,
    }; // -sin(π/8)
    e[3] = Ratio {
        num: R4::from_i32(0, 0, -1, 1),
        exp: 2,
    }; // sin(π/8)
    e[4] = Ratio {
        num: R4::from_i32(0, 0, 1, 0),
        exp: 2,
    }; // cos(π/8)
    e[8] = Ratio {
        num: R4::from_i32(2, 0, 0, 0),
        exp: 2,
    }; // 1
    for entry in e.iter_mut() {
        entry.simplify();
    }
    SO3 { e }
}

/// Rz(-π/8) = Rz(+π/8)ᵀ.
pub fn rz_neg_q() -> SO3<R4> {
    let mut m = rz_pos_q();
    m.e.swap(1, 3);
    m.e.swap(2, 6);
    m.e.swap(5, 7);
    m
}

/// Rx(+π/8) as SO3<R4>.
pub fn rx_pos_q() -> SO3<R4> {
    let mut e = [Ratio::<R4>::ZERO; 9];
    e[0] = Ratio {
        num: R4::from_i32(2, 0, 0, 0),
        exp: 2,
    }; // 1
    e[4] = Ratio {
        num: R4::from_i32(0, 0, 1, 0),
        exp: 2,
    }; // cos(π/8)
    e[5] = Ratio {
        num: R4::from_i32(0, 0, 1, -1),
        exp: 2,
    }; // -sin(π/8)
    e[7] = Ratio {
        num: R4::from_i32(0, 0, -1, 1),
        exp: 2,
    }; // sin(π/8)
    e[8] = Ratio {
        num: R4::from_i32(0, 0, 1, 0),
        exp: 2,
    }; // cos(π/8)
    for entry in e.iter_mut() {
        entry.simplify();
    }
    SO3 { e }
}

/// Rx(-π/8) = Rx(+π/8)ᵀ.
pub fn rx_neg_q() -> SO3<R4> {
    let mut m = rx_pos_q();
    m.e.swap(1, 3);
    m.e.swap(2, 6);
    m.e.swap(5, 7);
    m
}

/// Ry(+π/8) as SO3<R4>.
pub fn ry_pos_q() -> SO3<R4> {
    let mut e = [Ratio::<R4>::ZERO; 9];
    e[0] = Ratio {
        num: R4::from_i32(0, 0, 1, 0),
        exp: 2,
    }; // cos(π/8)
    e[2] = Ratio {
        num: R4::from_i32(0, 0, -1, 1),
        exp: 2,
    }; // sin(π/8)
    e[4] = Ratio {
        num: R4::from_i32(2, 0, 0, 0),
        exp: 2,
    }; // 1
    e[6] = Ratio {
        num: R4::from_i32(0, 0, 1, -1),
        exp: 2,
    }; // -sin(π/8)
    e[8] = Ratio {
        num: R4::from_i32(0, 0, 1, 0),
        exp: 2,
    }; // cos(π/8)
    for entry in e.iter_mut() {
        entry.simplify();
    }
    SO3 { e }
}

/// Ry(-π/8) = Ry(+π/8)ᵀ.
pub fn ry_neg_q() -> SO3<R4> {
    let mut m = ry_pos_q();
    m.e.swap(1, 3);
    m.e.swap(2, 6);
    m.e.swap(5, 7);
    m
}

// ─── R3 ───────────────────────────────────────────────────────────────────────

const SQRT_3: f64 = 1.7320508075688772935_f64;

/// An element of Z[√3]: `a + b·√3`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct R3(pub Int, pub Int);

impl R3 {
    pub const ZERO: Self = R3(INT_ZERO, INT_ZERO);
    pub const ONE: Self = R3(INT_ONE, INT_ZERO);

    #[inline]
    pub const fn from_i32(a: i32, b: i32) -> Self {
        R3(Int::from_i32(a), Int::from_i32(b))
    }

    pub fn to_f64(self) -> f64 {
        int_to_f64(self.0) + int_to_f64(self.1) * SQRT_3
    }

    /// Number of times 2 divides both coefficients simultaneously.
    pub fn two_valuation(self) -> u32 {
        if self.0 == INT_ZERO && self.1 == INT_ZERO {
            return u32::MAX;
        }
        let mut v = 0u32;
        let mut p = self.0;
        let mut q = self.1;
        while p % INT_TWO == INT_ZERO && q % INT_TWO == INT_ZERO {
            p /= INT_TWO;
            q /= INT_TWO;
            v += 1;
        }
        v
    }
}

impl Add for R3 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        R3(self.0 + rhs.0, self.1 + rhs.1)
    }
}
impl Sub for R3 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        R3(self.0 - rhs.0, self.1 - rhs.1)
    }
}
impl Neg for R3 {
    type Output = Self;
    fn neg(self) -> Self {
        R3(-self.0, -self.1)
    }
}
/// (a + b√3)(c + d√3) = (ac + 3bd) + (ad + bc)√3.
impl Mul for R3 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        R3(
            self.0 * rhs.0 + INT_THREE * self.1 * rhs.1,
            self.0 * rhs.1 + self.1 * rhs.0,
        )
    }
}

impl fmt::Display for R3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.1 == INT_ZERO {
            write!(f, "{}", self.0)
        } else if self.0 == INT_ZERO {
            write!(f, "{}√3", self.1)
        } else {
            write!(f, "{} + {}√3", self.0, self.1)
        }
    }
}

// ─── Ratio3 ──────────────────────────────────────────────────────────────────

/// A Z[√3] element divided by 2^exp.
///
/// Denominator convention: 2^exp (integer powers of 2).
/// All SO(3) entries for Clifford+R_z(π/6) unitaries have denominators
/// that are powers of 2 since cos/sin(mπ/6) ∈ Z[√3]/2.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ratio3 {
    pub num: R3,
    pub exp: u32,
}

impl Ratio3 {
    pub const ZERO: Self = Ratio3 {
        num: R3::ZERO,
        exp: 0,
    };
    pub const ONE: Self = Ratio3 {
        num: R3::ONE,
        exp: 0,
    };

    /// Cancel common factors of 2 between numerator and denominator.
    pub fn simplify(&mut self) {
        if self.num == R3::ZERO {
            self.exp = 0;
            return;
        }
        let v = self.num.two_valuation().min(self.exp);
        let mut p = self.num.0;
        let mut q = self.num.1;
        for _ in 0..v {
            p /= INT_TWO;
            q /= INT_TWO;
        }
        self.num = R3(p, q);
        self.exp -= v;
    }

    /// Lift numerator by multiplying by 2^n (for exponent alignment in Add).
    fn lift_num(self, n: u32) -> R3 {
        let mut x = self.num;
        for _ in 0..n {
            x = R3(x.0 * INT_TWO, x.1 * INT_TWO);
        }
        x
    }

    pub fn to_f64(self) -> f64 {
        self.num.to_f64() / (1u64 << self.exp.min(63)) as f64
    }
}

impl Neg for Ratio3 {
    type Output = Self;
    fn neg(self) -> Self {
        Ratio3 {
            num: -self.num,
            exp: self.exp,
        }
    }
}

impl Add for Ratio3 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let max_e = self.exp.max(rhs.exp);
        let lhs_num = self.lift_num(max_e - self.exp);
        let rhs_num = rhs.lift_num(max_e - rhs.exp);
        let mut r = Ratio3 {
            num: lhs_num + rhs_num,
            exp: max_e,
        };
        r.simplify();
        r
    }
}

impl Mul for Ratio3 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let mut r = Ratio3 {
            num: self.num * rhs.num,
            exp: self.exp + rhs.exp,
        };
        r.simplify();
        r
    }
}

// ─── SO3Omicron ───────────────────────────────────────────────────────────────

/// A 3×3 SO(3) matrix with entries in Z[√3]/2^exp, for Clifford+R_z(π/6).
///
/// Entries stored row-major: e[3*row + col] = Ratio3 { num, exp }
/// where the actual value is num / 2^exp.
///
/// SO(3) entries for Clifford+R_z(π/6) unitaries live in Z[√3] with
/// denominators that are powers of 2 (since cos(mπ/6), sin(mπ/6) ∈ Z[√3]/2).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SO3Omicron {
    pub e: [Ratio3; 9],
}

/// Type alias for SO3 over Z[√3] (used for Clifford+R_z(π/6)).
pub type SO3O = SO3Omicron;

impl SO3Omicron {
    pub fn identity() -> Self {
        let mut e = [Ratio3::ZERO; 9];
        e[0] = Ratio3::ONE;
        e[4] = Ratio3::ONE;
        e[8] = Ratio3::ONE;
        SO3Omicron { e }
    }

    #[inline]
    pub fn get(&self, r: usize, c: usize) -> Ratio3 {
        self.e[3 * r + c]
    }

    /// Maximum denominator exponent across all non-zero entries.
    pub fn maximum_denominator_exponent(&self) -> u32 {
        self.e
            .iter()
            .filter(|r| r.num != R3::ZERO)
            .map(|r| r.exp)
            .max()
            .unwrap_or(0)
    }

    /// Convert to 3×3 float matrix for debugging.
    pub fn to_float(&self) -> [[f64; 3]; 3] {
        let mut out = [[0.0f64; 3]; 3];
        for r in 0..3 {
            for c in 0..3 {
                out[r][c] = self.e[3 * r + c].to_f64();
            }
        }
        out
    }

    /// Build SO3Omicron from a U2<ZOmicron> matrix using exact ring arithmetic.
    ///
    /// For U = [[u11,u12],[u21,u22]] / √2^k with entries in ZOmicron:
    ///   Re(z) = (2a+c)/2 + b√3/2 = R3(2a+c, b)/2  (for z = ZOmicron(a,b,c,d))
    ///   Im(z) = (b+2d)/2 + c√3/2 = R3(b+2d, c)/2
    ///
    /// All entries use init_exp = k+1 (denominator = 2^{k+1}), representing
    /// the SO(3) entry values Re(·)/2^k or Im(·)/2^k. The cz entry uses
    /// R3(n.a, n.b/2) where n = u11·ū11 - u12·ū12 - u21·ū21 + u22·ū22.
    pub fn from_u2(u: &U2<ZOmicron>) -> Self {
        let a = u.u11;
        let b = u.u12;
        let c = u.u21;
        let d = u.u22;
        let k = u.k;

        let ad = a * d.conj();
        let bc = b * c.conj();
        let p = ad + bc;
        let q = ad - bc;
        let r = a * b.conj() - c * d.conj();
        let s = a * c.conj() - b * d.conj();
        let n = a * a.conj() - b * b.conj() - c * c.conj() + d * d.conj();

        // re(z) = R3(2*z.a + z.c, z.b) represents 2·Re(z) as R3
        let re = |z: ZOmicron| R3(INT_TWO * z.a + z.c, z.b);
        // im(z) = R3(z.b + 2*z.d, z.c) represents 2·Im(z) as R3
        let im = |z: ZOmicron| R3(z.b + INT_TWO * z.d, z.c);

        // n is real: n.c == 0 and n.b is even (proven from unitarity constraints).
        // cz raw = R3(n.a, n.b/2) represents Re(n)/2^{k+1}.
        let cz_num = R3(n.a, n.b / INT_TWO);

        let init_exp = k + 1;
        let raw: [R3; 9] = [
            re(p),
            im(q),
            re(s),
            -im(p),
            re(q),
            -im(s),
            re(r),
            im(r),
            cz_num,
        ];
        let e: [Ratio3; 9] = std::array::from_fn(|i| {
            let mut entry = Ratio3 {
                num: raw[i],
                exp: init_exp,
            };
            entry.simplify();
            entry
        });
        SO3Omicron { e }
    }
}

impl Mul for SO3Omicron {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let mut e = [Ratio3::ZERO; 9];
        for r in 0..3 {
            for c in 0..3 {
                let products: [Ratio3; 3] =
                    std::array::from_fn(|k| self.e[3 * r + k] * rhs.e[3 * k + c]);
                let max_e = products.iter().map(|p| p.exp).max().unwrap_or(0);
                let sum = products
                    .iter()
                    .fold(R3::ZERO, |acc, p| acc + p.lift_num(max_e - p.exp));
                let mut entry = Ratio3 {
                    num: sum,
                    exp: max_e,
                };
                entry.simplify();
                e[3 * r + c] = entry;
            }
        }
        SO3Omicron { e }
    }
}

impl SO3Ops for SO3Omicron {
    fn max_exp(&self) -> u32 {
        self.maximum_denominator_exponent()
    }
    fn left_mul(&mut self, rhs: &Self) {
        *self = rhs.clone() * self.clone();
    }
}

// ─── Rotation factories for SO3Omicron (π/6 steps) ───────────────────────────
//
// SO3Omicron entries in 2-denominator convention:
//   cos(π/6) = √3/2  → R3(0,1) / 2^1
//   sin(π/6) = 1/2   → R3(1,0) / 2^1
//   1                → R3(1,0) / 2^0
//
// Rz(π/6):  [[cos,-sin,0],[sin,cos,0],[0,0,1]]
// Rx(π/6):  [[1,0,0],[0,cos,-sin],[0,sin,cos]]
// Ry(π/6):  [[cos,0,sin],[0,1,0],[-sin,0,cos]]

/// Rz(+π/6) as SO3Omicron.
pub fn rz_pos_o() -> SO3Omicron {
    let cos = Ratio3 {
        num: R3::from_i32(0, 1),
        exp: 1,
    };
    let sin = Ratio3 {
        num: R3::from_i32(1, 0),
        exp: 1,
    };
    let one = Ratio3::ONE;
    let mut e = [Ratio3::ZERO; 9];
    e[0] = cos;
    e[1] = -sin;
    e[3] = sin;
    e[4] = cos;
    e[8] = one;
    SO3Omicron { e }
}

/// Rz(-π/6) = Rz(+π/6)ᵀ.
pub fn rz_neg_o() -> SO3Omicron {
    let mut m = rz_pos_o();
    m.e.swap(1, 3);
    m.e.swap(2, 6);
    m.e.swap(5, 7);
    m
}

/// Rx(+π/6) as SO3Omicron.
pub fn rx_pos_o() -> SO3Omicron {
    let cos = Ratio3 {
        num: R3::from_i32(0, 1),
        exp: 1,
    };
    let sin = Ratio3 {
        num: R3::from_i32(1, 0),
        exp: 1,
    };
    let one = Ratio3::ONE;
    let mut e = [Ratio3::ZERO; 9];
    e[0] = one;
    e[4] = cos;
    e[5] = -sin;
    e[7] = sin;
    e[8] = cos;
    SO3Omicron { e }
}

/// Rx(-π/6) = Rx(+π/6)ᵀ.
pub fn rx_neg_o() -> SO3Omicron {
    let mut m = rx_pos_o();
    m.e.swap(1, 3);
    m.e.swap(2, 6);
    m.e.swap(5, 7);
    m
}

/// Ry(+π/6) as SO3Omicron.
pub fn ry_pos_o() -> SO3Omicron {
    let cos = Ratio3 {
        num: R3::from_i32(0, 1),
        exp: 1,
    };
    let sin = Ratio3 {
        num: R3::from_i32(1, 0),
        exp: 1,
    };
    let one = Ratio3::ONE;
    let mut e = [Ratio3::ZERO; 9];
    e[0] = cos;
    e[2] = sin;
    e[4] = one;
    e[6] = -sin;
    e[8] = cos;
    SO3Omicron { e }
}

/// Ry(-π/6) = Ry(+π/6)ᵀ.
pub fn ry_neg_o() -> SO3Omicron {
    let mut m = ry_pos_o();
    m.e.swap(1, 3);
    m.e.swap(2, 6);
    m.e.swap(5, 7);
    m
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rings::ZOmega;

    fn near3(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> bool {
        for r in 0..3 {
            for c in 0..3 {
                if (a[r][c] - b[r][c]).abs() > 1e-10 {
                    return false;
                }
            }
        }
        true
    }

    // ── SO3<R2> basic rotation tests ──────────────────────────────────────────

    #[test]
    fn test_rz_pos_float() {
        let c = FRAC_1_SQRT_2;
        let m = rz_pos().to_float();
        let expected = [[c, -c, 0.0], [c, c, 0.0], [0.0, 0.0, 1.0]];
        assert!(near3(m, expected), "rz_pos: {:?}", m);
    }

    #[test]
    fn test_rx_pos_float() {
        let c = FRAC_1_SQRT_2;
        let m = rx_pos().to_float();
        let expected = [[1.0, 0.0, 0.0], [0.0, c, -c], [0.0, c, c]];
        assert!(near3(m, expected), "rx_pos: {:?}", m);
    }

    #[test]
    fn test_ry_pos_float() {
        let c = FRAC_1_SQRT_2;
        let m = ry_pos().to_float();
        let expected = [[c, 0.0, c], [0.0, 1.0, 0.0], [-c, 0.0, c]];
        assert!(near3(m, expected), "ry_pos: {:?}", m);
    }

    #[test]
    fn test_rz_rz_dag_is_identity() {
        assert_eq!(rz_pos() * rz_neg(), SO3T::identity());
    }

    #[test]
    fn test_rx_rx_dag_is_identity() {
        assert_eq!(rx_pos() * rx_neg(), SO3T::identity());
    }

    #[test]
    fn test_ry_ry_dag_is_identity() {
        assert_eq!(ry_pos() * ry_neg(), SO3T::identity());
    }

    #[test]
    fn test_mul_associativity() {
        let a = rz_pos();
        let b = rx_pos();
        let c = ry_pos();
        assert_eq!((a.clone() * b.clone()) * c.clone(), a * (b * c));
    }

    // ── from_u2 tests ────────────────────────────────────────────────────────

    #[test]
    fn test_identity_from_u2() {
        let id = U2T::new(ZOmega::ONE, ZOmega::ZERO, ZOmega::ZERO, ZOmega::ONE, 0);
        assert_eq!(SO3T::from_u2(&id), SO3T::identity());
    }

    #[test]
    fn test_s_gate_from_u2() {
        // S gate SU(2): u1 = -ω³ = e^{-iπ/4}, u2=0, k=0.
        // SO3 = Rz(π/2) = [[0,-1,0],[1,0,0],[0,0,1]] (standard convention).
        let u1 = ZOmega::from_i32(0, 0, 0, -1);
        let id = U2T::new(
            u1,
            ZOmega::ZERO,
            ZOmega::ZERO,
            ZOmega::from_i32(0, 1, 0, 0),
            0,
        );
        let so3 = SO3T::from_u2(&id);
        let m = so3.to_float();
        let expected = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        assert!(near3(m, expected), "S gate SO3 mismatch: {:?}", m);
    }

    #[test]
    fn test_s_gate_matches_rz2() {
        // rz_pos() * rz_pos() should equal from_u2(S gate).
        let u1 = ZOmega::from_i32(0, 0, 0, -1);
        let s_u2t = U2T::new(
            u1,
            ZOmega::ZERO,
            ZOmega::ZERO,
            ZOmega::from_i32(0, 1, 0, 0),
            0,
        );
        let rz2 = rz_pos() * rz_pos();
        let from_s = SO3T::from_u2(&s_u2t);
        assert_eq!(rz2, from_s, "rz_pos()² ≠ from_u2(S)");
    }

    #[test]
    fn test_h_gate_from_u2() {
        // H gate SU(2): u1=i, u2=i, k=1 (the i·H matrix).
        // SO3 = [[0,0,1],[0,-1,0],[1,0,0]] (standard convention).
        // H maps: x→z, y→-y, z→x.
        let u1 = ZOmega::I;
        let u2 = ZOmega::I;
        let h_u2t = U2T::new(u1, u2, u2, ZOmega::from_i32(0, 0, -1, 0), 1);
        let so3 = SO3T::from_u2(&h_u2t);
        let m = so3.to_float();
        let expected = [[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]];
        assert!(near3(m, expected), "H gate SO3 mismatch: {:?}", m);
    }

    #[test]
    fn test_identity_max_exp_is_zero() {
        assert_eq!(SO3T::identity().max_exp(), 0);
    }

    // ── R4 ring arithmetic tests ───────────────────────────────────────────────

    #[test]
    fn test_r4_gamma_squared() {
        // γ² = 2+√2 = R4(2,1,0,0)
        let gamma = R4::from_i32(0, 0, 1, 0); // γ
        let expected = R4::from_i32(2, 1, 0, 0); // 2+√2
        assert_eq!(gamma * gamma, expected, "γ² ≠ 2+√2");
    }

    #[test]
    fn test_r4_gamma_sqrt2_squared() {
        // (γ√2)² = 4+2√2 = R4(4,2,0,0)
        let gs = R4::from_i32(0, 0, 0, 1); // γ√2
        let expected = R4::from_i32(4, 2, 0, 0);
        assert_eq!(gs * gs, expected, "(γ√2)² ≠ 4+2√2");
    }

    #[test]
    fn test_r4_gamma_valuation() {
        let gamma = R4::from_i32(0, 0, 1, 0);
        let gamma2 = gamma * gamma;
        let gamma3 = gamma2 * gamma;
        assert_eq!(R4::ONE.gamma_valuation(), 0);
        assert_eq!(gamma.gamma_valuation(), 1);
        assert_eq!(gamma2.gamma_valuation(), 2);
        assert_eq!(gamma3.gamma_valuation(), 3);
        assert_eq!(R4::ZERO.gamma_valuation(), u32::MAX);
    }

    #[test]
    fn test_r4_div_gamma() {
        // R4(0,0,2,1) = γ³ = 2γ+γ√2; div_gamma should give γ²=R4(2,1,0,0)
        let gamma3 = R4::from_i32(0, 0, 2, 1);
        assert_eq!(gamma3.div_gamma(), R4::from_i32(2, 1, 0, 0));
        assert_eq!(
            R4::from_i32(2, 1, 0, 0).div_gamma(),
            R4::from_i32(0, 0, 1, 0)
        );
        assert_eq!(
            R4::from_i32(0, 0, 1, 0).div_gamma(),
            R4::from_i32(1, 0, 0, 0)
        );
    }

    #[test]
    fn test_r4_mul_gamma() {
        // γ * γ = γ² = R4(2,1,0,0)
        let gamma = R4::from_i32(0, 0, 1, 0);
        assert_eq!(gamma.mul_gamma(), R4::from_i32(2, 1, 0, 0), "γ·γ ≠ γ²");
        // 1 * γ = γ
        assert_eq!(R4::ONE.mul_gamma(), R4::from_i32(0, 0, 1, 0), "1·γ ≠ γ");
        // (mul_gamma then div_gamma) = identity for non-zero
        let x = R4::from_i32(3, 2, 1, 0);
        assert_eq!(x.mul_gamma().div_gamma(), x);
    }

    #[test]
    fn test_r4_to_f64() {
        let gamma = (2.0f64 + SQRT_2).sqrt();
        // cos(π/8) = R4(3,2,0,0)/γ³
        let cos_pi8 = R4::from_i32(3, 2, 0, 0).to_f64() / gamma.powi(3);
        let expected = (std::f64::consts::PI / 8.0).cos();
        assert!(
            (cos_pi8 - expected).abs() < 1e-12,
            "cos(π/8) mismatch: {cos_pi8}"
        );
        // sin(π/8) = R4(1,1,0,0)/γ³
        let sin_pi8 = R4::from_i32(1, 1, 0, 0).to_f64() / gamma.powi(3);
        let expected = (std::f64::consts::PI / 8.0).sin();
        assert!(
            (sin_pi8 - expected).abs() < 1e-12,
            "sin(π/8) mismatch: {sin_pi8}"
        );
    }

    // ── SO3<R4> rotation tests ────────────────────────────────────────────────

    #[test]
    fn test_rz_pos_q_float() {
        let m = rz_pos_q().to_float();
        let c = (std::f64::consts::PI / 8.0).cos();
        let s = (std::f64::consts::PI / 8.0).sin();
        let expected = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]];
        assert!(near3(m, expected), "rz_pos_q: {:?}", m);
    }

    #[test]
    fn test_rx_pos_q_float() {
        let m = rx_pos_q().to_float();
        let c = (std::f64::consts::PI / 8.0).cos();
        let s = (std::f64::consts::PI / 8.0).sin();
        let expected = [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]];
        assert!(near3(m, expected), "rx_pos_q: {:?}", m);
    }

    #[test]
    fn test_ry_pos_q_float() {
        let m = ry_pos_q().to_float();
        let c = (std::f64::consts::PI / 8.0).cos();
        let s = (std::f64::consts::PI / 8.0).sin();
        let expected = [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]];
        assert!(near3(m, expected), "ry_pos_q: {:?}", m);
    }

    #[test]
    fn test_rz_rz_dag_is_identity_q() {
        assert_eq!(rz_pos_q() * rz_neg_q(), SO3::<R4>::identity());
    }

    #[test]
    fn test_rx_rx_dag_is_identity_q() {
        assert_eq!(rx_pos_q() * rx_neg_q(), SO3::<R4>::identity());
    }

    #[test]
    fn test_ry_ry_dag_is_identity_q() {
        assert_eq!(ry_pos_q() * ry_neg_q(), SO3::<R4>::identity());
    }

    #[test]
    fn test_rz_q_8_is_identity() {
        // Rz(π/8)^16 = Rz(2π) = identity (SO3 period is 2π, so 16 steps of π/8)
        let mut m = SO3::<R4>::identity();
        for _ in 0..16 {
            m = rz_pos_q() * m;
        }
        assert_eq!(m, SO3::<R4>::identity(), "Rz(π/8)^16 ≠ I");
    }

    #[test]
    fn test_rz_q_8_is_rz_pos() {
        // Rz(π/8)^2 = Rz(π/4), which should embed into SO3<R2>.
        // Verify numerically.
        let m = (rz_pos_q() * rz_pos_q()).to_float();
        let c = FRAC_1_SQRT_2;
        let expected = [[c, -c, 0.0], [c, c, 0.0], [0.0, 0.0, 1.0]];
        assert!(near3(m, expected), "Rz(π/8)^2 ≠ Rz(π/4): {:?}", m);
    }

    #[test]
    fn test_mul_associativity_q() {
        let a = rz_pos_q();
        let b = rx_pos_q();
        let c = ry_pos_q();
        assert_eq!((a.clone() * b.clone()) * c.clone(), a * (b * c));
    }

    #[test]
    fn test_maximum_denominator_exponent_r2_mixed_entries() {
        let mut e = [Ratio::<R2>::ZERO; 9];

        // non-zero entries
        e[0] = Ratio {
            num: R2::from_i32(1, 0),
            exp: 1,
        };
        e[4] = Ratio {
            num: R2::from_i32(3, 1),
            exp: 3,
        };
        e[8] = Ratio {
            num: R2::from_i32(-2, 5),
            exp: 2,
        };

        // zero entry with large exponent should be ignored
        e[1] = Ratio {
            num: R2::ZERO,
            exp: 99,
        };

        let m = SO3::<R2> { e };
        assert_eq!(m.maximum_denominator_exponent(), 3);
        assert_eq!(m.max_exp(), 3);
    }

    #[test]
    fn test_maximum_denominator_exponent_r2_all_zero_entries() {
        let e = [Ratio {
            num: R2::ZERO,
            exp: 7,
        }; 9];
        let m = SO3::<R2> { e };
        assert_eq!(m.maximum_denominator_exponent(), 0);
        assert_eq!(m.max_exp(), 0);
    }

    #[test]
    fn test_maximum_denominator_exponent_r4_mixed_entries() {
        let mut e = [Ratio::<R4>::ZERO; 9];

        // non-zero entries
        e[0] = Ratio {
            num: R4::from_i32(1, 0, 0, 0),
            exp: 2,
        };
        e[4] = Ratio {
            num: R4::from_i32(0, 0, 1, 0),
            exp: 5,
        };
        e[8] = Ratio {
            num: R4::from_i32(3, 2, 1, 1),
            exp: 4,
        };

        // zero entry with large exponent should be ignored
        e[2] = Ratio {
            num: R4::ZERO,
            exp: 77,
        };

        let m = SO3::<R4> { e };
        assert_eq!(m.maximum_denominator_exponent(), 5);
        assert_eq!(m.max_exp(), 5);
    }

    #[test]
    fn test_maximum_denominator_exponent_r4_all_zero_entries() {
        let e = [Ratio {
            num: R4::ZERO,
            exp: 11,
        }; 9];
        let m = SO3::<R4> { e };
        assert_eq!(m.maximum_denominator_exponent(), 0);
        assert_eq!(m.max_exp(), 0);
    }

    // ── SO3Omicron (n=6) tests ────────────────────────────────────────────────

    fn near_f64(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-10
    }

    fn near3_f64(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> bool {
        for r in 0..3 {
            for c in 0..3 {
                if (a[r][c] - b[r][c]).abs() > 1e-10 {
                    return false;
                }
            }
        }
        true
    }

    #[test]
    fn test_so3o_identity_max_exp_zero() {
        assert_eq!(SO3Omicron::identity().max_exp(), 0);
    }

    #[test]
    fn test_rz_pos_o_float() {
        let m = rz_pos_o().to_float();
        let c = (std::f64::consts::PI / 6.0).cos();
        let s = (std::f64::consts::PI / 6.0).sin();
        let expected = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]];
        assert!(near3_f64(m, expected), "rz_pos_o: {:?}", m);
    }

    #[test]
    fn test_rx_pos_o_float() {
        let m = rx_pos_o().to_float();
        let c = (std::f64::consts::PI / 6.0).cos();
        let s = (std::f64::consts::PI / 6.0).sin();
        let expected = [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]];
        assert!(near3_f64(m, expected), "rx_pos_o: {:?}", m);
    }

    #[test]
    fn test_ry_pos_o_float() {
        let m = ry_pos_o().to_float();
        let c = (std::f64::consts::PI / 6.0).cos();
        let s = (std::f64::consts::PI / 6.0).sin();
        let expected = [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]];
        assert!(near3_f64(m, expected), "ry_pos_o: {:?}", m);
    }

    #[test]
    fn test_rz_o_neg_is_transpose() {
        let pos = rz_pos_o();
        let neg = rz_neg_o();
        let prod = pos.clone() * neg.clone();
        assert_eq!(prod, SO3Omicron::identity(), "Rz_pos * Rz_neg ≠ I");
        let prod2 = neg * pos;
        assert_eq!(prod2, SO3Omicron::identity(), "Rz_neg * Rz_pos ≠ I");
    }

    #[test]
    fn test_rx_o_neg_is_transpose() {
        let prod = rx_pos_o() * rx_neg_o();
        assert_eq!(prod, SO3Omicron::identity(), "Rx_pos * Rx_neg ≠ I");
    }

    #[test]
    fn test_ry_o_neg_is_transpose() {
        let prod = ry_pos_o() * ry_neg_o();
        assert_eq!(prod, SO3Omicron::identity(), "Ry_pos * Ry_neg ≠ I");
    }

    #[test]
    fn test_so3o_from_u2_identity() {
        let id = U2::<ZOmicron>::eye();
        assert_eq!(SO3Omicron::from_u2(&id), SO3Omicron::identity());
    }

    #[test]
    fn test_so3o_from_u2_rz_pi6() {
        // R gate = diag(1, ξ), SO3 = Rz(π/6)
        let r = U2::<ZOmicron>::t(); // t() returns diag(1, omega()=ξ)
        let m = SO3Omicron::from_u2(&r).to_float();
        let expected = rz_pos_o().to_float();
        assert!(near3_f64(m, expected), "SO3O(R) ≠ Rz(π/6): {:?}", m);
    }

    #[test]
    fn test_so3o_from_u2_h_gate() {
        // H gate SO3 maps x→z, y→-y, z→x
        let h = U2::<ZOmicron>::h();
        let m = SO3Omicron::from_u2(&h).to_float();
        let expected = [[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]];
        assert!(near3_f64(m, expected), "SO3O(H): {:?}", m);
    }

    #[test]
    fn test_so3o_from_u2_rz12_equals_identity() {
        // R^12 = Rz(2π) = identity in SO3
        let r = U2::<ZOmicron>::t();
        let mut prod = U2::<ZOmicron>::eye();
        for _ in 0..12 {
            prod = prod * r;
        }
        let m = SO3Omicron::from_u2(&prod);
        assert_eq!(m.max_exp(), 0, "R^12 SO3 max_exp should be 0");
        assert!(
            near3_f64(m.to_float(), SO3Omicron::identity().to_float()),
            "R^12 ≠ identity in SO3: {:?}",
            m.to_float()
        );
    }

    #[test]
    fn test_so3o_product_consistency() {
        // SO3(U·V) == SO3(U) · SO3(V)
        let r = U2::<ZOmicron>::t();
        let h = U2::<ZOmicron>::h();
        let uv = r * h;
        let so3_uv = SO3Omicron::from_u2(&uv).to_float();
        let so3_u_so3_v = (SO3Omicron::from_u2(&r) * SO3Omicron::from_u2(&h)).to_float();
        assert!(
            near3_f64(so3_uv, so3_u_so3_v),
            "SO3(RH) ≠ SO3(R)·SO3(H):\n{:?}\nvs\n{:?}",
            so3_uv,
            so3_u_so3_v
        );
    }
}
