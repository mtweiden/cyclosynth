//! Exact synthesis of single-qubit `G₁₂ = U₂(Z[ζ₂₄, 1/2])` unitaries into
//! **optimal** Clifford + R_z(π/12) circuits.
//!
//! Background: ζ₂₄ is in the Forest–Gosset–Kliuchnikov–McKinnon golden set
//! `{2, 4, 6, 8, 12}` (J. Math. Phys. 56, 082201, 2015). Every ring-valued
//! unitary `U ∈ G₁₂` factors uniquely as
//!
//! ```text
//!   U = R̂_{p_1}(a_1·π/12) · R̂_{p_2}(a_2·π/12) · … · R̂_{p_N}(a_N·π/12) · D
//! ```
//!
//! with `p_i ∈ {x,y,z}`, `a_i ∈ {1..5}`, and `D` a Clifford. The total
//! R_z(π/12) count `T₁₂(U) = Σ min(a_i, 6 - a_i)` is provably minimal
//! (Forest Lemma 3.1 + canonical-form uniqueness).
//!
//! Role boundary: [`super::lattice_upsilon::synthesize`] does (approx float
//! target → exact ring `U`). This module does (exact ring `U → optimal
//! Clifford + R_z(π/12) circuit + minimal count). [`synthesize_circuit`]
//! chains them.
//!
//! Algorithm: trial-peel. At each step compute the Bloch SO(3) image,
//! find the unique `(p, a)` whose inverse strictly lowers the
//! √2-denominator exponent (`lde`), accumulate `min(a, 6 - a)`, and
//! recurse. When `lde = 0`, the residual is a signed-permutation
//! Clifford. The trial-peel is robust (no closed-form residue lookup
//! needed; see `TODO(fast-path)` below).

#![allow(clippy::too_many_arguments)]

use crate::matrix::U2;
use crate::rings::types::{Int, INT_ZERO};
use crate::rings::ZUpsilon;
use num_complex::Complex64;
use rayon::prelude::*;
use std::cmp::min;
use std::collections::HashMap;
use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};
use std::sync::{Arc, LazyLock, Mutex};

// ─── RealScalar = Z[ζ₂₄]_real / √2^m ─────────────────────────────────────────

/// A real element of Z[ζ₂₄] divided by `√2^m`.
///
/// `elem` is required to be real (i.e. `elem.conj() == elem`); this is
/// preserved by add/sub/neg/mul and checked in debug builds. We carry the
/// raw `ZUpsilon` rather than reducing into Z[√2, √3] because cleared
/// rotation entries like `(√6 ± √2)/2` lie in Z[ζ₂₄] but **not** in the
/// maximal order of `Z[√2, √3]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RealScalar {
    pub elem: ZUpsilon,
    pub m: u32,
}

impl RealScalar {
    pub const ZERO: Self = Self {
        elem: ZUpsilon::ZERO,
        m: 0,
    };
    pub const ONE: Self = Self {
        elem: ZUpsilon::ONE,
        m: 0,
    };

    #[inline]
    pub fn new(elem: ZUpsilon, m: u32) -> Self {
        debug_assert_eq!(
            elem.conj(),
            elem,
            "RealScalar: elem must be real (conj-fixed)"
        );
        let mut r = Self { elem, m };
        r.simplify();
        r
    }

    /// Cancel common √2 factors between numerator and denominator.
    pub fn simplify(&mut self) {
        if self.elem == ZUpsilon::ZERO {
            self.m = 0;
            return;
        }
        let v = self.elem.sqrt2_valuation().min(self.m);
        for _ in 0..v {
            // u / √2 = u · √2 / 2; divide all coefficients by 2 (guaranteed even
            // by sqrt2_valuation).
            self.elem = self.elem.mul_sqrt2().div2(1);
        }
        self.m -= v;
    }

    /// Multiply numerator by `√2^n` (used before addition to align denominators).
    fn lift_num(self, n: u32) -> ZUpsilon {
        let mut x = self.elem;
        for _ in 0..n {
            x = x.mul_sqrt2();
        }
        x
    }

    /// Numeric value as `f64` (real-valued; takes the real part of the
    /// `to_complex` since `elem` is real).
    pub fn to_f64(self) -> f64 {
        let c = self.elem.to_complex();
        debug_assert!(
            c.im.abs() < 1e-8,
            "RealScalar::to_f64: imaginary part {} should be 0",
            c.im
        );
        c.re / (self.m as f64 / 2.0).exp2()
    }

    /// Integer scalar `n` ∈ Z lifted into the ring.
    pub fn from_int(n: Int) -> Self {
        Self::new(
            ZUpsilon::new(
                n, INT_ZERO, INT_ZERO, INT_ZERO, INT_ZERO, INT_ZERO, INT_ZERO, INT_ZERO,
            ),
            0,
        )
    }
}

impl Neg for RealScalar {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            elem: -self.elem,
            m: self.m,
        }
    }
}

impl Add for RealScalar {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let max_m = self.m.max(rhs.m);
        let l = self.lift_num(max_m - self.m);
        let r = rhs.lift_num(max_m - rhs.m);
        let mut out = Self {
            elem: l + r,
            m: max_m,
        };
        out.simplify();
        out
    }
}

impl Sub for RealScalar {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        self + (-rhs)
    }
}

impl Mul for RealScalar {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let mut out = Self {
            elem: self.elem * rhs.elem,
            m: self.m + rhs.m,
        };
        out.simplify();
        out
    }
}

impl fmt::Display for RealScalar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.m == 0 {
            write!(f, "{}", self.elem)
        } else {
            write!(f, "({}) / √2^{}", self.elem, self.m)
        }
    }
}

// ─── Real parts of ZUpsilon as ZUpsilon-valued reals ─────────────────────────

/// Returns `z + conj(z) = 2·Re(z)` as a real ZUpsilon element.
#[inline]
fn re_doubled(z: ZUpsilon) -> ZUpsilon {
    z + z.conj()
}

/// Returns `(conj(z) - z) · i = 2·Im(z)` as a real ZUpsilon element
/// (note: multiplication by `i` flips the i-axis component to real).
#[inline]
fn im_doubled(z: ZUpsilon) -> ZUpsilon {
    (z.conj() - z) * ZUpsilon::I
}

// ─── SO3Upsilon: 3×3 SO(3) matrix with RealScalar entries ────────────────────

/// 3×3 SO(3) matrix with `RealScalar` entries (row-major).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SO3Upsilon {
    pub e: [RealScalar; 9],
}

impl SO3Upsilon {
    pub fn identity() -> Self {
        let mut e = [RealScalar::ZERO; 9];
        e[0] = RealScalar::ONE;
        e[4] = RealScalar::ONE;
        e[8] = RealScalar::ONE;
        Self { e }
    }

    #[inline]
    pub fn get(&self, r: usize, c: usize) -> RealScalar {
        self.e[3 * r + c]
    }

    /// `lde(Û) = max over non-zero entries of m` — the largest √2-power
    /// needed to clear the matrix.
    pub fn lde(&self) -> u32 {
        self.e
            .iter()
            .filter(|r| r.elem != ZUpsilon::ZERO)
            .map(|r| r.m)
            .max()
            .unwrap_or(0)
    }

    /// 3×3 float view (for debugging / numerical comparisons).
    pub fn to_float(&self) -> [[f64; 3]; 3] {
        let mut out = [[0.0f64; 3]; 3];
        for r in 0..3 {
            for c in 0..3 {
                out[r][c] = self.e[3 * r + c].to_f64();
            }
        }
        out
    }

    /// Build `SO3Upsilon` from `U2<ZUpsilon>` via the Bloch map
    /// `Û_{PP'} = ½·Tr(P · U · P' · U†)` for `P, P' ∈ {X,Y,Z}`.
    ///
    /// The closed form (matching `SO3<R2>::from_u2`):
    /// ```text
    ///   P = a·d̄ + b·c̄,    Q = a·d̄ − b·c̄,
    ///   R = a·b̄ − c·d̄,    S = a·c̄ − b·d̄,
    ///   N = a·ā − b·b̄ − c·c̄ + d·d̄        (real)
    ///
    ///   ax = Re(P),   ay = Im(Q),   az = Re(S)
    ///   bx = −Im(P),  by = Re(Q),   bz = −Im(S)
    ///   cx = Re(R),   cy = Im(R),   cz = N/2
    /// ```
    /// all divided by `2^k = √2^{2k}`. We store `2·Re/Im(·)` with denom
    /// `√2^{2k+2}` (so the actual entry value is `Re(·)/2^k`); `cz` stores
    /// raw `N` with the same denom, giving `N/2^{k+1} = (N/2)/2^k`. ✓
    pub fn from_u2(u: &U2<ZUpsilon>) -> Self {
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

        let m_init = 2 * k + 2;
        let raw = [
            re_doubled(p),
            im_doubled(q),
            re_doubled(s),
            -im_doubled(p),
            re_doubled(q),
            -im_doubled(s),
            re_doubled(r),
            im_doubled(r),
            n,
        ];
        let e: [RealScalar; 9] = std::array::from_fn(|i| RealScalar::new(raw[i], m_init));
        Self { e }
    }
}

impl Mul for SO3Upsilon {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let mut e = [RealScalar::ZERO; 9];
        for r in 0..3 {
            for c in 0..3 {
                e[3 * r + c] = self.e[3 * r] * rhs.e[c]
                    + self.e[3 * r + 1] * rhs.e[3 + c]
                    + self.e[3 * r + 2] * rhs.e[6 + c];
            }
        }
        Self { e }
    }
}

// ─── Rotation oracle (SPEC table) ────────────────────────────────────────────
//
// For a ∈ {1..5}, the cleared cos/sin numerators in Z[ζ₂₄] are (PROMPT
// §"Verified reference data"):
//
//   | a | q_a |  √2^q·cos          |  √2^q·sin          | cost min(a,6-a) |
//   |---|-----|--------------------|--------------------|-----------------|
//   | 1 |  2  | (√6+√2)/2 = ζ+ζ³−ζ⁷| (√6−√2)/2 = ζ⁵−ζ⁷  | 1               |
//   | 2 |  2  | √3 = 2ζ²−ζ⁶        | 1                  | 2               |
//   | 3 |  1  | 1                  | 1                  | 3               |
//   | 4 |  2  | 1                  | √3 = 2ζ²−ζ⁶        | 2               |
//   | 5 |  2  | (√6−√2)/2 = ζ⁵−ζ⁷  | (√6+√2)/2 = ζ+ζ³−ζ⁷| 1               |

/// `(2·cos(a·π/12), 2·sin(a·π/12))` as real-ZUpsilon numerators with denom
/// `√2^{q_a}` to clear (q_a per SPEC table). Returns `(cos_num, sin_num, q_a)`.
fn cos_sin_clear(a: u8) -> (ZUpsilon, ZUpsilon, u32) {
    match a {
        1 => (
            // 2·cos(π/12) = ζ + ζ³ - ζ⁷, then divide by √2² to get cos.
            // Stored RealScalar: elem = ζ+ζ³−ζ⁷, m = 2 → value = (√6+√2)/4 = cos(π/12) ✓
            ZUpsilon::from_i32(0, 1, 0, 1, 0, 0, 0, -1),
            // 2·sin(π/12) = ζ⁵ - ζ⁷.
            ZUpsilon::from_i32(0, 0, 0, 0, 0, 1, 0, -1),
            2,
        ),
        2 => (
            // 2·cos(π/6) = √3 = 2ζ² − ζ⁶.
            ZUpsilon::from_i32(0, 0, 2, 0, 0, 0, -1, 0),
            // 2·sin(π/6) = 1.
            ZUpsilon::from_i32(1, 0, 0, 0, 0, 0, 0, 0),
            2,
        ),
        3 => (
            // √2·cos(π/4) = √2·(√2/2) = 1.
            ZUpsilon::from_i32(1, 0, 0, 0, 0, 0, 0, 0),
            ZUpsilon::from_i32(1, 0, 0, 0, 0, 0, 0, 0),
            1,
        ),
        4 => (
            // 2·cos(π/3) = 1.
            ZUpsilon::from_i32(1, 0, 0, 0, 0, 0, 0, 0),
            // 2·sin(π/3) = √3 = 2ζ² − ζ⁶.
            ZUpsilon::from_i32(0, 0, 2, 0, 0, 0, -1, 0),
            2,
        ),
        5 => (
            // 2·cos(5π/12) = (√6−√2)/2 = ζ⁵ − ζ⁷ (same algebra as 2·sin(π/12)).
            ZUpsilon::from_i32(0, 0, 0, 0, 0, 1, 0, -1),
            // 2·sin(5π/12) = (√6+√2)/2 = ζ + ζ³ − ζ⁷ (same as 2·cos(π/12)).
            ZUpsilon::from_i32(0, 1, 0, 1, 0, 0, 0, -1),
            2,
        ),
        _ => panic!("cos_sin_clear: a must be in 1..=5, got {a}"),
    }
}

/// `(cos(aπ/12), sin(aπ/12))` as `RealScalar` pairs.
fn cos_sin_real(a: u8) -> (RealScalar, RealScalar) {
    let (c_num, s_num, q) = cos_sin_clear(a);
    (RealScalar::new(c_num, q), RealScalar::new(s_num, q))
}

/// `R̂_z(aπ/12)` as `SO3Upsilon`:
///   `[[c, -s, 0], [s, c, 0], [0, 0, 1]]`.
///
/// Counter-clockwise rotation about the z-axis. This matches the Bloch
/// correspondence used by [`SO3Upsilon::from_u2`] (and by the existing
/// `SO3<R2>::from_u2` for n=4 — same `(Re,Im,N)` formulas), so
/// `bloch(P) == rz_pos_u(1)` exactly.
///
/// NOTE: the prompt text gives `[[c, s, 0], [-s, c, 0], [0, 0, 1]]` which
/// is the *transpose* / inverse rotation. The convention chosen here is
/// the one consistent with the codebase's existing `from_u2` (verified by
/// the `bloch_so3_p_matches_rz_pos_u_1` test below).
pub fn rz_pos_u(a: u8) -> SO3Upsilon {
    let (c, s) = cos_sin_real(a);
    let mut e = [RealScalar::ZERO; 9];
    e[0] = c;
    e[1] = -s;
    e[3] = s;
    e[4] = c;
    e[8] = RealScalar::ONE;
    SO3Upsilon { e }
}

/// `R̂_x(aπ/12)` as `SO3Upsilon`:
///   `[[1, 0, 0], [0, c, -s], [0, s, c]]`.
pub fn rx_pos_u(a: u8) -> SO3Upsilon {
    let (c, s) = cos_sin_real(a);
    let mut e = [RealScalar::ZERO; 9];
    e[0] = RealScalar::ONE;
    e[4] = c;
    e[5] = -s;
    e[7] = s;
    e[8] = c;
    SO3Upsilon { e }
}

/// `R̂_y(aπ/12)` as `SO3Upsilon`:
///   `[[c, 0, s], [0, 1, 0], [-s, 0, c]]`.
pub fn ry_pos_u(a: u8) -> SO3Upsilon {
    let (c, s) = cos_sin_real(a);
    let mut e = [RealScalar::ZERO; 9];
    e[0] = c;
    e[2] = s;
    e[4] = RealScalar::ONE;
    e[6] = -s;
    e[8] = c;
    SO3Upsilon { e }
}

/// `R̂_p(-aπ/12) = R̂_p(aπ/12)^T` (orthogonal matrix transpose).
fn transpose(m: SO3Upsilon) -> SO3Upsilon {
    let mut e = [RealScalar::ZERO; 9];
    for r in 0..3 {
        for c in 0..3 {
            e[3 * r + c] = m.e[3 * c + r];
        }
    }
    SO3Upsilon { e }
}

/// `R̂_z(-aπ/12)`.
pub fn rz_neg_u(a: u8) -> SO3Upsilon {
    transpose(rz_pos_u(a))
}
/// `R̂_x(-aπ/12)`.
pub fn rx_neg_u(a: u8) -> SO3Upsilon {
    transpose(rx_pos_u(a))
}
/// `R̂_y(-aπ/12)`.
pub fn ry_neg_u(a: u8) -> SO3Upsilon {
    transpose(ry_pos_u(a))
}

// ─── U2 rotation generators ──────────────────────────────────────────────────

/// `R_z(π/12) = diag(1, ζ_24)`. Same as `U2::<ZUpsilon>::p()`.
pub fn p_gate() -> U2<ZUpsilon> {
    U2::p()
}

/// `R_z(aπ/12) = P^a` as `U2<ZUpsilon>`.
pub fn rz_pos_u2(a: u8) -> U2<ZUpsilon> {
    let mut u = U2::<ZUpsilon>::eye();
    for _ in 0..a {
        u = u * p_gate();
    }
    u
}

/// `R_x(aπ/12) = H · P^a · H`.
pub fn rx_pos_u2(a: u8) -> U2<ZUpsilon> {
    let h = U2::<ZUpsilon>::h();
    h * rz_pos_u2(a) * h
}

/// `R_y(aπ/12) = S · H · P^a · H · S†`.
pub fn ry_pos_u2(a: u8) -> U2<ZUpsilon> {
    let s = U2::<ZUpsilon>::s();
    let h = U2::<ZUpsilon>::h();
    s * h * rz_pos_u2(a) * h * s.dagger()
}

// ─── Gate enum and circuit ───────────────────────────────────────────────────

/// Single-qubit gate alphabet for Clifford + R_z(π/12) circuits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Gate {
    H,
    S,
    Sdg,
    /// `P = R_z(π/12)`.
    P,
    /// `P† = R_z(-π/12)`.
    Pdg,
    X,
    Y,
    Z,
}

impl Gate {
    /// `U2<ZUpsilon>` matrix for this gate.
    pub fn to_u2(self) -> U2<ZUpsilon> {
        match self {
            Gate::H => U2::h(),
            Gate::S => U2::s(),
            Gate::Sdg => U2::<ZUpsilon>::s().dagger(),
            Gate::P => U2::p(),
            Gate::Pdg => U2::<ZUpsilon>::p().dagger(),
            Gate::X => U2::x(),
            Gate::Y => U2::y(),
            Gate::Z => U2::z(),
        }
    }
}

impl fmt::Display for Gate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Gate::H => "H",
            Gate::S => "S",
            Gate::Sdg => "S†",
            Gate::P => "P",
            Gate::Pdg => "P†",
            Gate::X => "X",
            Gate::Y => "Y",
            Gate::Z => "Z",
        };
        write!(f, "{s}")
    }
}

/// Multiply out a gate sequence into a single `U2<ZUpsilon>` matrix.
/// Convention: leftmost gate = leftmost matrix factor (so applied LAST
/// to a state vector).
pub fn circuit_to_u2(circuit: &[Gate]) -> U2<ZUpsilon> {
    let mut u = U2::<ZUpsilon>::eye();
    for g in circuit {
        u = u * g.to_u2();
    }
    u
}

// ─── Clifford table over Z[ζ₂₄] ──────────────────────────────────────────────

/// Build all 24 single-qubit Cliffords as `U2<ZUpsilon>` with their gate
/// string `name → Vec<Gate>` decomposition. The matrices follow the
/// `CLIFFORD_TABLE_T` naming convention (e.g., "ZHSH" = Z·H·S·H).
fn clifford_table_u() -> Vec<(&'static str, U2<ZUpsilon>, Vec<Gate>)> {
    let names: [&str; 24] = [
        "I", "H", "S", "X", "Y", "Z", "XH", "YH", "ZH", "XS", "YS", "ZS", "SH", "XSH", "YSH",
        "ZSH", "HS", "XHS", "YHS", "ZHS", "HSH", "XHSH", "YHSH", "ZHSH",
    ];
    names
        .iter()
        .map(|&n| {
            let gates: Vec<Gate> = n
                .chars()
                .filter_map(|c| match c {
                    'I' => None,
                    'H' => Some(Gate::H),
                    'S' => Some(Gate::S),
                    'X' => Some(Gate::X),
                    'Y' => Some(Gate::Y),
                    'Z' => Some(Gate::Z),
                    _ => None,
                })
                .collect();
            let u = circuit_to_u2(&gates);
            (n, u, gates)
        })
        .collect()
}

/// Match a (small `k`) `U2<ZUpsilon>` against the Clifford table by
/// diamond distance. Returns the matching gate sequence.
fn identify_clifford(u: &U2<ZUpsilon>) -> Option<Vec<Gate>> {
    let table = clifford_table_u();
    let target_float = u.to_float();
    let mut best: Option<(f64, Vec<Gate>)> = None;
    for (_, cand, gates) in &table {
        let d = crate::synthesis::distance::diamond_distance_float(&cand.to_float(), &target_float);
        if let Some((db, _)) = &best {
            if d < *db {
                best = Some((d, gates.clone()));
            }
        } else {
            best = Some((d, gates.clone()));
        }
    }
    best.filter(|(d, _)| *d < 1e-6).map(|(_, g)| g)
}

// ─── Trial-peel decomposition ────────────────────────────────────────────────

/// One peel-step candidate.
struct PeelCandidate {
    axis: u8, // 0=x, 1=y, 2=z
    a: u8,    // 1..5
    so3_neg: SO3Upsilon,
    u2_pos: U2<ZUpsilon>,
}

fn peel_candidates() -> Vec<PeelCandidate> {
    let mut out = Vec::with_capacity(15);
    for axis in 0..3u8 {
        for a in 1..=5u8 {
            let (so3_neg, u2_pos) = match axis {
                0 => (rx_neg_u(a), rx_pos_u2(a)),
                1 => (ry_neg_u(a), ry_pos_u2(a)),
                2 => (rz_neg_u(a), rz_pos_u2(a)),
                _ => unreachable!(),
            };
            out.push(PeelCandidate {
                axis,
                a,
                so3_neg,
                u2_pos,
            });
        }
    }
    out
}

/// Emit the gate sequence for one peeled rotation `R_p(aπ/12)`.
fn rotation_gates(axis: u8, a: u8) -> Vec<Gate> {
    let p_run: Vec<Gate> = (0..a).map(|_| Gate::P).collect();
    match axis {
        0 => {
            // R_x = H · P^a · H
            let mut g = vec![Gate::H];
            g.extend(p_run);
            g.push(Gate::H);
            g
        }
        1 => {
            // R_y = S · H · P^a · H · S†
            let mut g = vec![Gate::S, Gate::H];
            g.extend(p_run);
            g.push(Gate::H);
            g.push(Gate::Sdg);
            g
        }
        2 => p_run, // R_z = P^a
        _ => unreachable!(),
    }
}

/// Output of [`decompose`].
#[derive(Debug, Clone)]
pub struct DecomposeResult {
    /// Gate sequence (leftmost gate = leftmost matrix factor → applied
    /// last in the time direction).
    pub circuit: Vec<Gate>,
    /// Minimal R_z(π/12) gate count (`T₁₂(U) = Σ min(a_i, 6 - a_i)`).
    pub t12_count: usize,
}

/// Output of native approximate synthesis followed by exact pi/12
/// decomposition.
#[derive(Debug, Clone)]
pub struct CircuitSynthResult {
    /// Gate sequence over Clifford + `P = R_z(pi/12)`.
    pub circuit: Vec<Gate>,
    /// Denominator exponent of the synthesized exact unitary.
    pub lde: u32,
    /// Selected `Z[ζ24]` phase branch.
    pub phase: u32,
    /// Diamond distance from the synthesized unitary to the float target.
    pub distance: f64,
    /// Minimal R_z(π/12) gate count.
    pub t12_count: usize,
}

/// Result of a successful n=12 synthesis.
pub struct SynthResultPi12 {
    /// Clifford+R_z(π/12) gate string (leftmost = leftmost matrix factor).
    pub gates: Option<String>,
    /// Denominator exponent of the synthesized unitary.
    pub lde: u32,
    /// Diamond distance from the synthesized unitary to the target.
    pub distance: f64,
    /// Selected `Z[ζ24]` phase branch.
    pub phase: u32,
    /// Minimal R_z(π/12) count for the exact synthesized unitary.
    pub t12_count: usize,
}

/// Decompose `U ∈ G₁₂` into an optimal Clifford + R_z(π/12) circuit.
///
/// Uses trial-peel (Forest §3 canonical form): at each step, pick the
/// unique `(p, a) ∈ {x,y,z} × {1..5}` such that `R̂_p(-a)·Û` has strictly
/// lower `lde`. Accumulate `min(a, 6 - a)`. Terminate when `lde = 0`
/// (residual is a signed-permutation Clifford), match it against the
/// 24-element table, append.
///
/// Panics if no candidate reduces `lde` while `lde > 0` — this would
/// indicate either a bug or input not in `G₁₂`.
///
/// TODO(fast-path): Forest §4 gives a one-multiplication-per-peel
/// `residue → a` lookup, but it requires the *normalized* residue
/// (quotient out the trailing unit). The lookup avoids the 15-way
/// linear scan. Trial-peel is shipped here.
pub fn decompose(u: &U2<ZUpsilon>) -> DecomposeResult {
    let candidates = peel_candidates();
    let mut so3 = SO3Upsilon::from_u2(u);
    let mut p_output_u2: U2<ZUpsilon> = U2::<ZUpsilon>::eye();
    let mut factors: Vec<(u8, u8)> = Vec::new();
    let mut t12_count: usize = 0;

    // Generous step bound: each peel strictly reduces lde, so an unbounded
    // loop would still terminate, but we cap to surface bugs.
    let max_steps = (so3.lde() as usize) * 16 + 64;
    for _ in 0..max_steps {
        let cur_lde = so3.lde();
        if cur_lde == 0 {
            break;
        }
        // Forest canonical-form algorithm: pick the (p, a) whose inverse
        // gives the SMALLEST resulting lde. Per Forest Theorem 4.1(c) the
        // minimizer is unique (no tie). Mirrors `decompose_so3_canonical_*`
        // for n=4/6/8. A "first that reduces" greedy is wrong: at lde=2,
        // both (z,1) and (z,4) may reduce, but only (z,4) is the canonical
        // factor and only it yields the minimum count (e.g., R_z(4π/12) → 2,
        // not 1+3=4).
        let mut best_idx: Option<usize> = None;
        let mut best_lde = cur_lde;
        for (i, cand) in candidates.iter().enumerate() {
            let trial = cand.so3_neg.clone() * so3.clone();
            let trial_lde = trial.lde();
            if trial_lde < best_lde {
                best_lde = trial_lde;
                best_idx = Some(i);
            }
        }
        let cand = &candidates[best_idx.unwrap_or_else(|| {
            panic!(
                "decompose: no (p,a) candidate reduces lde from {cur_lde}; \
                 input may not be in G₁₂"
            )
        })];
        so3 = cand.so3_neg.clone() * so3;
        p_output_u2 = p_output_u2 * cand.u2_pos;
        factors.push((cand.axis, cand.a));
        t12_count += min(cand.a, 6 - cand.a) as usize;
    }

    // Residual Clifford `D` so that `U = (Π R_p_i(a_iπ/12)) · D = p_output · D`.
    let d = p_output_u2.dagger() * *u;
    let clifford_gates = identify_clifford(&d).expect("residual not a Clifford after peel");

    // Circuit assembly: leftmost gates = leftmost matrix factor.
    // U = R_p_1(a_1) · R_p_2(a_2) · … · R_p_N(a_N) · D
    let mut circuit: Vec<Gate> = Vec::new();
    for &(axis, a) in &factors {
        circuit.extend(rotation_gates(axis, a));
    }
    circuit.extend(clifford_gates);

    // Debug verification: circuit re-multiplies to U up to global phase
    // (within numerical precision).
    debug_assert!(
        verify_circuit(&circuit, u),
        "decompose: emitted circuit does not re-multiply to U (factors={factors:?})"
    );

    DecomposeResult { circuit, t12_count }
}

/// Verify that `circuit_to_u2(circuit) ≈ u` within numerical precision
/// (allowing global phase difference).
pub fn verify_circuit(circuit: &[Gate], u: &U2<ZUpsilon>) -> bool {
    let cu = circuit_to_u2(circuit);
    let d = crate::synthesis::distance::diamond_distance_float(&cu.to_float(), &u.to_float());
    d < 1e-9
}

// ─── Prefix split for approximate synthesis ─────────────────────────────────

type Mat2 = [[Complex64; 2]; 2];

fn mat_mul_f64(a: Mat2, b: Mat2) -> Mat2 {
    let mut out = [[Complex64::new(0.0, 0.0); 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                out[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    out
}

fn mat_dag_f64(a: Mat2) -> Mat2 {
    [
        [a[0][0].conj(), a[1][0].conj()],
        [a[0][1].conj(), a[1][1].conj()],
    ]
}

fn canonical_key_f64(m: &Mat2) -> [i64; 8] {
    let flat = [m[0][0], m[0][1], m[1][0], m[1][1]];
    let (idx, _) = flat
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.norm_sqr().partial_cmp(&b.norm_sqr()).unwrap())
        .unwrap();
    let piv = flat[idx];
    let rot: Vec<f64> = if piv.norm() < 1e-12 {
        flat.iter().flat_map(|c| [c.re, c.im]).collect()
    } else {
        let phase = piv / piv.norm();
        flat.iter()
            .flat_map(|c| {
                let r = c / phase;
                [r.re, r.im]
            })
            .collect()
    };
    rot.iter()
        .map(|x| (x * 1_000_000.0).round() as i64)
        .collect::<Vec<_>>()
        .try_into()
        .unwrap()
}

fn default_dc_inner_k_pi12(eps: f64) -> u32 {
    if eps <= 1e-7 {
        22
    } else if eps <= 1e-5 {
        18
    } else if eps <= 1e-4 {
        12
    } else {
        8
    }
}

#[derive(Clone)]
struct PrefixPi12 {
    u: U2<ZUpsilon>,
    f: Mat2,
}

fn build_l_pi12(k_prefix: u32) -> Arc<Vec<PrefixPi12>> {
    static CACHE: LazyLock<Mutex<HashMap<u32, Arc<Vec<PrefixPi12>>>>> =
        LazyLock::new(|| Mutex::new(HashMap::new()));

    {
        let cache = CACHE.lock().unwrap();
        if let Some(v) = cache.get(&k_prefix) {
            return Arc::clone(v);
        }
    }

    let result = Arc::new(build_l_pi12_inner(k_prefix));
    CACHE.lock().unwrap().insert(k_prefix, Arc::clone(&result));
    result
}

fn push_prefix_pi12(
    out: &mut Vec<PrefixPi12>,
    seen: &mut std::collections::HashSet<[i64; 8]>,
    u: U2<ZUpsilon>,
) {
    let f = u.to_float();
    if seen.insert(canonical_key_f64(&f)) {
        out.push(PrefixPi12 { u, f });
    }
}

fn build_l_pi12_inner(k_prefix: u32) -> Vec<PrefixPi12> {
    if k_prefix == 0 {
        return vec![PrefixPi12 {
            u: U2::<ZUpsilon>::eye(),
            f: U2::<ZUpsilon>::eye().to_float(),
        }];
    }

    let h = U2::<ZUpsilon>::h();
    let s = U2::<ZUpsilon>::s();
    let p = U2::<ZUpsilon>::p();
    let hp = h * p;
    let hsp = h * s * p;
    let cliffords = clifford_table_u();

    let mut out = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let n_even = 1u64 << k_prefix.min(62);
    for bits in 0..n_even {
        let mut u = U2::<ZUpsilon>::eye();
        for i in 0..k_prefix {
            u = u * if (bits >> i) & 1 == 1 { hsp } else { hp };
        }
        for (_, clifford, _) in &cliffords {
            push_prefix_pi12(&mut out, &mut seen, u * *clifford);
        }
    }
    if k_prefix >= 1 {
        let n_odd = 1u64 << (k_prefix - 1).min(62);
        for bits in 0..n_odd {
            let mut u = p;
            for i in 0..(k_prefix - 1) {
                u = u * if (bits >> i) & 1 == 1 { hsp } else { hp };
            }
            for (_, clifford, _) in &cliffords {
                push_prefix_pi12(&mut out, &mut seen, u * *clifford);
            }
        }
    }
    if std::env::var_os("CYCLOSYNTH_TRACE_DEEP_EPS").is_some() {
        eprintln!(
            "[trace pi12 dc] built prefix_k={k_prefix} prefixes={}",
            out.len()
        );
    }
    out
}

fn synthesize_circuit_dc_at_k(
    target: &Mat2,
    k: u32,
    eps: f64,
    k_inner: u32,
) -> Option<CircuitSynthResult> {
    if k_inner >= k {
        return None;
    }
    let k_prefix = k - k_inner;
    let prefixes = build_l_pi12(k_prefix);
    if std::env::var_os("CYCLOSYNTH_TRACE_DEEP_EPS").is_some() {
        eprintln!(
            "[trace pi12 dc] try k={k} prefix_k={k_prefix} inner_k={k_inner} prefixes={}",
            prefixes.len()
        );
    }
    prefixes.par_iter().find_map_any(|prefix| {
        let inner_target = mat_mul_f64(mat_dag_f64(prefix.f), *target);
        let synth =
            crate::synthesis::lattice_upsilon::synthesize_first(&inner_target, k_inner, eps)?;
        let full_u = prefix.u * synth.u;
        let full_f = full_u.to_float();
        let d = crate::synthesis::distance::diamond_distance_float(&full_f, target);
        if d > eps {
            return None;
        }
        let decomposed = decompose(&full_u);
        Some(CircuitSynthResult {
            circuit: decomposed.circuit,
            lde: k,
            phase: synth.phase,
            distance: d,
            t12_count: decomposed.t12_count,
        })
    })
}

// ─── Integration with lattice_upsilon::synthesize ────────────────────────────

/// One-shot: approximate target → exact `U ∈ G₁₂` → optimal Clifford +
/// R_z(π/12) circuit.
///
/// Returns `None` if the lattice synthesizer found no ring-valued
/// unitary within `eps` of `target` at denominator `√2^k`.
pub fn synthesize_circuit(
    target: &[[Complex64; 2]; 2],
    k: u32,
    eps: f64,
) -> Option<DecomposeResult> {
    synthesize_circuit_at_k(target, k, eps).map(|r| DecomposeResult {
        circuit: r.circuit,
        t12_count: r.t12_count,
    })
}

/// Native one-shot approximate synthesis at denominator `√2^k`, followed
/// by exact Clifford + `P` decomposition.
pub fn synthesize_circuit_at_k(
    target: &[[Complex64; 2]; 2],
    k: u32,
    eps: f64,
) -> Option<CircuitSynthResult> {
    let synth = crate::synthesis::lattice_upsilon::synthesize_first(target, k, eps)?;
    let decomposed = decompose(&synth.u);
    Some(CircuitSynthResult {
        circuit: decomposed.circuit,
        lde: k,
        phase: synth.phase,
        distance: synth.distance,
        t12_count: decomposed.t12_count,
    })
}

/// Native approximate synthesis over a denominator range. This is the
/// pi/12 arbitrary-unitary entry point: every returned circuit is produced
/// by the `Z[ζ24]` lattice path and decomposed into `P`. There is no
/// cross-ring fallback; failure is reported honestly as `None`.
pub fn synthesize_circuit_in_range(
    target: &[[Complex64; 2]; 2],
    eps: f64,
    min_k: u32,
    max_k: u32,
) -> Option<CircuitSynthResult> {
    let enable_dc = std::env::var_os("CYCLOSYNTH_PI12_ENABLE_DC").is_some();
    for k in min_k..=max_k {
        if let Some(result) = synthesize_circuit_at_k(target, k, eps) {
            return Some(result);
        }
        if !enable_dc {
            continue;
        }
        let base_inner = default_dc_inner_k_pi12(eps).min(k);
        if k > base_inner {
            let mut inners = Vec::new();
            for inner in (base_inner.saturating_sub(2)..=base_inner + 2).rev() {
                if inner >= 5 && inner < k && !inners.contains(&inner) {
                    inners.push(inner);
                }
            }
            for inner in inners {
                if let Some(result) = synthesize_circuit_dc_at_k(target, k, eps, inner) {
                    return Some(result);
                }
            }
        }
    }
    None
}

// ─── Synthesizer ─────────────────────────────────────────────────────────────

/// Clifford + R_z(π/12) synthesis backend over ℤ[ζ₂₄].
pub struct SynthesizerPi12 {
    /// Approximation precision in diamond distance.
    pub epsilon: f64,
    /// Maximum denominator exponent to search before giving up.
    pub max_lde: u32,
    /// Minimum denominator exponent to start searching from.
    pub min_lde: u32,
}

impl SynthesizerPi12 {
    /// Create a synthesizer with defaults tuned for the native n=12 lattice path.
    pub fn new(epsilon: f64) -> Self {
        let (min_lde, max_lde) = default_pi12_lde_window(epsilon);
        Self {
            epsilon,
            max_lde,
            min_lde,
        }
    }

    pub fn with_max_lde(mut self, max_lde: u32) -> Self {
        self.max_lde = max_lde;
        self
    }

    pub fn with_min_lde(mut self, min_lde: u32) -> Self {
        self.min_lde = min_lde;
        self
    }

    /// Synthesize a Clifford+R_z(π/12) circuit approximating `target`.
    pub fn synthesize(&self, target: Mat2) -> Option<SynthResultPi12> {
        synthesize_circuit_in_range(&target, self.epsilon, self.min_lde, self.max_lde).map(|r| {
            SynthResultPi12 {
                gates: Some(gate_string(&r.circuit)),
                lde: r.lde,
                distance: r.distance,
                phase: r.phase,
                t12_count: r.t12_count,
            }
        })
    }
}

fn default_pi12_lde_window(eps: f64) -> (u32, u32) {
    if !(eps > 0.0 && eps < 1.0) {
        return (0, 20);
    }

    if eps >= 1e-3 {
        return (8, 18);
    }

    if eps <= 1e-5 {
        // Same frontier heuristic used by the native random-unitary regression:
        // 2^(4k) * eps^2 >= 2^14.
        let frontier_k = ((14.0_f64 - 2.0 * eps.log2()) / 4.0).ceil() as u32;
        return (frontier_k, (frontier_k + 4).max(14));
    }

    (5, 12)
}

fn gate_string(circuit: &[Gate]) -> String {
    circuit.iter().map(ToString::to_string).collect()
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rings::types::{INT_ONE, INT_ZERO};
    use std::f64::consts::PI;

    fn near_f64(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-10
    }

    #[test]
    fn gate_matrices_are_unitary() {
        let identity = U2::<ZUpsilon>::eye().to_float();
        for gate in [
            Gate::H,
            Gate::S,
            Gate::Sdg,
            Gate::P,
            Gate::Pdg,
            Gate::X,
            Gate::Y,
            Gate::Z,
        ] {
            let u = gate.to_u2();
            let product = (u * u.dagger()).to_float();
            let d = crate::synthesis::distance::diamond_distance_float(&product, &identity);
            assert!(d < 1e-12, "{gate}: U·U† is not identity (d={d:.3e})");
        }
    }

    // ── RealScalar arithmetic ────────────────────────────────────────────────

    #[test]
    fn realscalar_constants() {
        assert!(near_f64(RealScalar::ZERO.to_f64(), 0.0));
        assert!(near_f64(RealScalar::ONE.to_f64(), 1.0));
        assert_eq!(RealScalar::ZERO.m, 0);
        assert_eq!(RealScalar::ONE.m, 0);
    }

    #[test]
    fn realscalar_arithmetic_real_axis() {
        // 1 + 1 = 2
        let two = RealScalar::ONE + RealScalar::ONE;
        assert!(near_f64(two.to_f64(), 2.0));
        // 2 - 1 = 1
        let one = two - RealScalar::ONE;
        assert!(near_f64(one.to_f64(), 1.0));
        // 2 · 3 = 6
        let three = RealScalar::from_int(Int::from_i32(3));
        let six = two * three;
        assert!(near_f64(six.to_f64(), 6.0));
    }

    #[test]
    fn realscalar_simplifies_sqrt2() {
        // num = √2 (which has sqrt2_valuation = 1), m = 1 → simplifies to (1, 0).
        let s = RealScalar::new(ZUpsilon::sqrt2(), 1);
        assert!(near_f64(s.to_f64(), 1.0));
        assert_eq!(s.m, 0);
        assert_eq!(s.elem, ZUpsilon::ONE);
    }

    #[test]
    fn realscalar_cos_sin_match_spec_table() {
        // Verify cos_sin_real matches the SPEC oracle table numerically.
        let cases = [
            (1u8, (PI / 12.0).cos(), (PI / 12.0).sin()),
            (2, (PI / 6.0).cos(), (PI / 6.0).sin()),
            (3, (PI / 4.0).cos(), (PI / 4.0).sin()),
            (4, (PI / 3.0).cos(), (PI / 3.0).sin()),
            (5, (5.0 * PI / 12.0).cos(), (5.0 * PI / 12.0).sin()),
        ];
        for (a, c_expected, s_expected) in cases {
            let (c, s) = cos_sin_real(a);
            assert!(
                near_f64(c.to_f64(), c_expected),
                "cos({a}π/12): got {}, expected {}",
                c.to_f64(),
                c_expected
            );
            assert!(
                near_f64(s.to_f64(), s_expected),
                "sin({a}π/12): got {}, expected {}",
                s.to_f64(),
                s_expected
            );
        }
    }

    #[test]
    fn realscalar_lde_per_spec() {
        // q_a values from SPEC: a=3 → q=1, others → q=2.
        let expect_q = [(1u8, 2u32), (2, 2), (3, 1), (4, 2), (5, 2)];
        for (a, q) in expect_q {
            let (c, s) = cos_sin_real(a);
            let m_max = c.m.max(s.m);
            assert_eq!(
                m_max, q,
                "lde of cos/sin at a={a}: got {m_max}, expected {q}"
            );
        }
    }

    // ── SO(3) and bloch_so3 ───────────────────────────────────────────────────

    #[test]
    fn bloch_so3_identity() {
        let s = SO3Upsilon::from_u2(&U2::<ZUpsilon>::eye());
        let f = s.to_float();
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(near_f64(f[i][j], expected), "I[{i}][{j}] = {}", f[i][j]);
            }
        }
        assert_eq!(s.lde(), 0);
    }

    #[test]
    fn bloch_so3_h_is_clifford() {
        // H is a Clifford ⇒ its SO(3) image is a signed permutation ⇒ lde = 0.
        let s = SO3Upsilon::from_u2(&U2::<ZUpsilon>::h());
        assert_eq!(s.lde(), 0, "H should have lde=0");
    }

    #[test]
    fn bloch_so3_s_is_clifford() {
        let s = SO3Upsilon::from_u2(&U2::<ZUpsilon>::s());
        assert_eq!(s.lde(), 0, "S should have lde=0");
    }

    #[test]
    fn bloch_so3_p_matches_rz_pos_u_1() {
        // P = R_z(π/12) ⇒ bloch(P) = R̂_z(1).
        let bloch_p = SO3Upsilon::from_u2(&U2::<ZUpsilon>::p());
        let canonical = rz_pos_u(1);
        assert_eq!(bloch_p, canonical, "bloch(P) != R̂_z(π/12)");
    }

    #[test]
    fn rotation_matrices_are_so3() {
        // det of R̂_p(a) should be 1 (positive determinant), numerically.
        for a in 1..=5u8 {
            for &m in &[
                rz_pos_u(a).to_float(),
                rx_pos_u(a).to_float(),
                ry_pos_u(a).to_float(),
            ] {
                let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                    - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                    + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
                assert!(near_f64(det, 1.0), "det != 1: {det}");
            }
        }
    }

    // ── Oracle: each R_z(aπ/12) decomposes to a single peel of correct cost ──

    #[test]
    fn decompose_rz_oracle() {
        let expected = [(1u8, 1usize), (2, 2), (3, 3), (4, 2), (5, 1)];
        for (a, expected_count) in expected {
            let u = rz_pos_u2(a);
            let r = decompose(&u);
            assert_eq!(
                r.t12_count, expected_count,
                "R_z({a}π/12): count {} ≠ expected {}",
                r.t12_count, expected_count
            );
            assert!(
                verify_circuit(&r.circuit, &u),
                "R_z({a}π/12): circuit does not re-multiply to U"
            );
        }
    }

    /// **Cost-vs-lde divergence (PROMPT test #3).** T has the smallest lde
    /// (=1) but the largest cost (=3). A bug that wires the count to lde
    /// would give count=1 here.
    #[test]
    fn t_gate_costs_three_not_one() {
        // T = R_z(π/4) = R_z(3π/12) = P³.
        let t = rz_pos_u2(3);
        // sanity check lde
        let bloch_t = SO3Upsilon::from_u2(&t);
        assert_eq!(bloch_t.lde(), 1, "bloch(T) lde should be 1");

        let r = decompose(&t);
        assert_eq!(
            r.t12_count, 3,
            "T must cost 3 P-gates (got {}); cost-vs-lde wiring bug?",
            r.t12_count
        );
    }

    #[test]
    fn rz_pi_6_costs_two() {
        // R_z(π/6) = R_z(2π/12) = P² → cost 2.
        let u = rz_pos_u2(2);
        let r = decompose(&u);
        assert_eq!(r.t12_count, 2);
    }

    #[test]
    fn rz_pi_12_costs_one() {
        let u = rz_pos_u2(1);
        let r = decompose(&u);
        assert_eq!(r.t12_count, 1);
    }

    // ── Clifford passes through with cost 0 ──────────────────────────────────

    #[test]
    fn cliffords_have_zero_cost() {
        for u in [
            U2::<ZUpsilon>::eye(),
            U2::<ZUpsilon>::h(),
            U2::<ZUpsilon>::s(),
            U2::<ZUpsilon>::x(),
            U2::<ZUpsilon>::y(),
            U2::<ZUpsilon>::z(),
        ] {
            let r = decompose(&u);
            assert_eq!(r.t12_count, 0, "{u}: should cost 0");
            assert!(verify_circuit(&r.circuit, &u));
        }
    }

    // ── Composite circuits ───────────────────────────────────────────────────

    #[test]
    fn decompose_random_composite_circuits_re_multiply() {
        // Build U = product of K random {H, S, P} gates; decompose; verify.
        use rand::Rng;
        let mut rng = rand::rng();
        for trial in 0..15 {
            let k = rng.random_range(1..=6);
            let mut gates: Vec<Gate> = Vec::with_capacity(k);
            let mut p_count = 0;
            for _ in 0..k {
                let g = match rng.random_range(0..3) {
                    0 => Gate::H,
                    1 => Gate::S,
                    _ => {
                        p_count += 1;
                        Gate::P
                    }
                };
                gates.push(g);
            }
            let u = circuit_to_u2(&gates);
            let r = decompose(&u);
            assert!(
                r.t12_count <= p_count,
                "trial {trial}: count {} > P-count {p_count}",
                r.t12_count
            );
            assert!(
                verify_circuit(&r.circuit, &u),
                "trial {trial}: circuit ≠ u (gates={gates:?})"
            );
        }
    }

    /// Synthesize a deterministic Haar-style random SU(2) unitary using the
    /// native n=12 lattice path.
    ///
    /// At the default `eps = 1e-5`, this test is now routed through the 8D
    /// LLL+BKZ+SE MITM backend (see [`super::lattice_upsilon::synthesize_first`])
    /// which lands a within-ε circuit in ~5 s (vs the joint 16D SE that
    /// previously hung). The `#[ignore]` is removed because the test no
    /// longer takes minutes — it's a regular regression now.
    #[test]
    fn synthesize_random_unitary() {
        use rand::{rngs::StdRng, Rng, SeedableRng};

        let mut rng = StdRng::seed_from_u64(1);
        let eps = std::env::var("CYCLOSYNTH_PI12_RANDOM_EPS")
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(1e-6_f64);

        let theta = rng.random::<f64>() * (2.0 * PI);
        let phi = rng.random::<f64>() * (2.0 * PI);
        let lambda = rng.random::<f64>() * (2.0 * PI);
        let ct = (theta / 2.0).cos();
        let st = (theta / 2.0).sin();

        // U3(θ, φ, λ), normalized by a global phase so det(target)=1.
        let global_phase = Complex64::from_polar(1.0, -(phi + lambda) / 2.0);
        let target = [
            [
                global_phase * Complex64::new(ct, 0.0),
                global_phase * (-Complex64::from_polar(st, lambda)),
            ],
            [
                global_phase * Complex64::from_polar(st, phi),
                global_phase * Complex64::from_polar(ct, phi + lambda),
            ],
        ];

        // Frontier-k start: the smallest exhaustible k whose per-half
        // region is expected to be non-empty for `eps`. The pool size
        // estimate is ~2^{4k}·ε² ≥ 2^14 → k ≥ ⌈(14 − 2 log2 ε)/4⌉.
        // The constant 14 is tuned so 1e-5 → 12 (matches Part 4 successes)
        // and 1e-6 → 14, avoiding the wasted low-k attempts that were the
        // historical cause of the "test takes too long" symptom
        // (PROMPT_mitm_8d_completeness Part 5 step 2). Standalone timing
        // dropped from 124 s (k_min=10 scanning) to 2.6 s (k_min=12 jump).
        let frontier_k = ((14.0_f64 - 2.0 * eps.log2()) / 4.0).ceil() as u32;
        let min_k = std::env::var("CYCLOSYNTH_PI12_RANDOM_MIN_K")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or_else(|| {
                if eps >= 1e-3 {
                    8
                } else if eps <= 1e-5 {
                    frontier_k
                } else {
                    5
                }
            });
        let max_k = std::env::var("CYCLOSYNTH_PI12_RANDOM_MAX_K")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or_else(|| {
                if eps >= 1e-3 {
                    18
                } else if eps <= 1e-5 {
                    (frontier_k + 4).max(14)
                } else {
                    12
                }
            });

        let t0 = std::time::Instant::now();
        let result = synthesize_circuit_in_range(&target, eps, min_k, max_k).unwrap_or_else(|| {
            panic!(
                "n=12 failed to synthesize deterministic random unitary at eps={eps:.1e}, k={min_k}..={max_k}"
            )
        });
        let elapsed = t0.elapsed();
        let recovered = circuit_to_u2(&result.circuit);
        let actual =
            crate::synthesis::distance::diamond_distance_float(&recovered.to_float(), &target);
        let gates_pi12 = result
            .circuit
            .iter()
            .map(ToString::to_string)
            .collect::<String>();

        eprintln!(
            "n=12 pi12 eps={eps:.1e}: elapsed={}ms k={} phase={} claimed={:.6e} actual={:.6e} t12_count={} gates_len={} gates={}",
            elapsed.as_millis(),
            result.lde,
            result.phase,
            result.distance,
            actual,
            result.t12_count,
            result.circuit.len(),
            gates_pi12
        );
        assert!(
            result
                .circuit
                .iter()
                .any(|g| matches!(g, Gate::P | Gate::Pdg)),
            "pi/12 random synthesis produced no P gate: {gates_pi12}"
        );
        assert!(
            result.distance < eps && actual < eps,
            "n=12 pi12 failed: claimed={:.6e}, actual={actual:.6e}, epsilon={eps:.6e}, gates={gates_pi12}",
            result.distance
        );
    }

    /// Synthesize deterministic random exact G₁₂ unitaries through the full
    /// n=12 lattice → decomposition path.
    #[test]
    fn synthesize_random_exact_gate_unitaries() {
        use rand::{rngs::StdRng, Rng, SeedableRng};

        struct EnvGuard {
            key: &'static str,
            old: Option<std::ffi::OsString>,
        }
        impl Drop for EnvGuard {
            fn drop(&mut self) {
                unsafe {
                    match &self.old {
                        Some(v) => std::env::set_var(self.key, v),
                        None => std::env::remove_var(self.key),
                    }
                }
            }
        }

        let _bound_guard = EnvGuard {
            key: "CYCLOSYNTH_BOUND_SQ_N12",
            old: std::env::var_os("CYCLOSYNTH_BOUND_SQ_N12"),
        };
        unsafe {
            std::env::set_var("CYCLOSYNTH_BOUND_SQ_N12", "32");
        }

        let mut rng = StdRng::seed_from_u64(1);
        let eps = 1e-3_f64;

        for trial in 0..3 {
            let mut gates = Vec::new();
            for _ in 0..5 {
                match rng.random_range(0..4) {
                    0 => gates.push(Gate::S),
                    1 => gates.push(Gate::P),
                    2 => gates.push(Gate::Pdg),
                    _ => {}
                }
                gates.push(Gate::H);
                match rng.random_range(0..4) {
                    0 => gates.push(Gate::S),
                    1 => gates.push(Gate::P),
                    2 => gates.push(Gate::Pdg),
                    _ => {}
                }
            }
            match rng.random_range(0..4) {
                0 => gates.push(Gate::S),
                1 => gates.push(Gate::P),
                2 => gates.push(Gate::Pdg),
                _ => {}
            }

            let target_u = circuit_to_u2(&gates);
            assert_eq!(
                target_u.k, 5,
                "trial {trial}: random circuit should force k=5"
            );
            let target = target_u.to_float();

            let result = synthesize_circuit(&target, target_u.k, eps).unwrap_or_else(|| {
                panic!(
                    "trial {trial}: n=12 failed to synthesize random exact unitary at eps={eps:.1e}, k={}",
                    target_u.k
                )
            });
            let recovered = circuit_to_u2(&result.circuit);
            let actual =
                crate::synthesis::distance::diamond_distance_float(&recovered.to_float(), &target);
            assert!(
                actual < eps,
                "trial {trial}: n=12 exact gate synthesis failed: actual={actual:.6e}, epsilon={eps:.6e}"
            );
        }
    }

    // ── lattice_upsilon round-trip (cross-module test #6 of PROMPT) ─────────

    /// The lattice_upsilon round-trip (un-ignored in Part A.2) implicitly
    /// validates that the synthesizer recovers small G₁₂ elements
    /// exactly. Here we additionally compose that with decompose and
    /// check the gate count is well-formed.
    #[test]
    fn lattice_then_decompose_chain_works() {
        let p = U2::<ZUpsilon>::p();
        let target_float = p.to_float();
        let r = synthesize_circuit(&target_float, 0, 1e-9).expect("P at k=0 should synth");
        assert_eq!(r.t12_count, 1, "P cost = 1");
        assert!(verify_circuit(&r.circuit, &p));
    }

    /// Sanity: `cos_sin_clear` returns reals.
    #[test]
    fn cos_sin_clear_returns_real_zupsilon() {
        for a in 1..=5u8 {
            let (cn, sn, _) = cos_sin_clear(a);
            assert_eq!(cn.conj(), cn, "cos numerator at a={a} not real");
            assert_eq!(sn.conj(), sn, "sin numerator at a={a} not real");
        }
    }

    /// The 24-element Clifford table is closed under the 24-element Clifford group.
    #[test]
    fn clifford_table_24_elements() {
        let t = clifford_table_u();
        assert_eq!(t.len(), 24);
        // Identity should be the first entry.
        assert_eq!(t[0].0, "I");
        // Each Clifford should have lde = 0.
        for (name, u, _) in &t {
            let s = SO3Upsilon::from_u2(u);
            assert_eq!(s.lde(), 0, "Clifford {name}: lde should be 0");
        }
    }

    /// `identify_clifford` recognizes every Clifford by its U2 form.
    #[test]
    fn identify_clifford_round_trips_table() {
        let t = clifford_table_u();
        for (name, u, gates) in &t {
            let id = identify_clifford(u).unwrap_or_else(|| panic!("missing: {name}"));
            // The returned gates should re-multiply to u (modulo global phase).
            let cu = circuit_to_u2(&id);
            let d =
                crate::synthesis::distance::diamond_distance_float(&cu.to_float(), &u.to_float());
            assert!(d < 1e-9, "{name}: identified circuit ≠ u (d={d})");
            let _ = gates;
        }
    }

    // ── A sanity guard that we're computing INT_* correctly ─────────────────

    #[test]
    fn realscalar_int_consts_used() {
        assert_eq!(RealScalar::ZERO.elem.a, INT_ZERO);
        assert_eq!(RealScalar::ONE.elem.a, INT_ONE);
    }
}
