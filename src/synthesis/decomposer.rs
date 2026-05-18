//! Bloch sphere decomposition of exactly-implementable Clifford+T (or Clifford+√T) unitaries.
//!
//! The ring type `R` of the input `U2<R>` determines the gate set automatically:
//!   - `U2<ZOmega>` (= `U2T`) → Clifford+T,  SO3 over Z[√2], step = Rz(π/4)
//!   - `U2<ZZeta>`  (= `U2Q`) → Clifford+√T, SO3 over Z[γ],  step = Rz(π/8)
//!
//! Algorithm — common shape:
//!   1. Convert target unitary → SO(3) matrix (exact ring arithmetic, no floats).
//!   2. Peel rotations from the left until the residual is a Clifford SO(3).
//!   3. Identify the residual Clifford (24-element table).
//!   4. Translate the peel sequence + Clifford suffix into a gate string.
//!
//! Step 2 differs by ring:
//!   - **ZOmega**: `decompose_so3` does single-step greedy peeling. Each peel
//!     is one of `Rx/Ry/Rz(±π/4)`; cos(nπ/4) has √2-exp ∈ {0,1,0,1,…}, so
//!     `max_exp` is monotone-decreasing and the argmin among 3 candidates is
//!     always correct. Provably terminates in `max_exp` steps.
//!   - **ZZeta**: `decompose_so3_canonical_q` follows
//!     Forest–Gosset–Kliuchnikov–McKinnon 2015 (arXiv:1501.04944, Section 4).
//!     cos(nπ/8) has non-monotone √2-exp pattern (single π/8 peels can
//!     transiently *increase* `max_exp`), so we try all 9 candidate peels
//!     `R_p(a·π/8)` for `p ∈ {x,y,z}, a ∈ {1,2,3}` per step. By
//!     Theorem 4.1(c) the optimal `(p, a)` is unique while `max_exp > 0`.

use std::fmt::Debug;
use std::ops::{Mul, Sub};
use crate::rings::{ZOmega, ZZeta, ZOmicron};
use crate::rings::types::INT_ZERO;
use crate::matrix::so3::{SO3, SO3T, SO3Q, SO3Omicron, SO3Ops, R2, R4, Ratio};
use crate::matrix::u2::{U2, U2T, U2Q, RingElem};
#[cfg(feature = "python")]
use crate::matrix::u2::{PyU2, U2Variant};
use crate::matrix::{rz_pos, rx_pos, ry_pos, rz_pos_q, rx_pos_q, ry_pos_q};
use crate::matrix::{rz_neg, rx_neg, ry_neg, rz_neg_q, rx_neg_q, ry_neg_q};
use crate::matrix::{rz_pos_o, rx_pos_o, ry_pos_o, rz_neg_o, rx_neg_o, ry_neg_o};
use crate::synthesis::cliffords::CLIFFORD_TABLE_T;

// ─── GateRing trait ───────────────────────────────────────────────────────────

/// A ring type that carries its own gate-set context for Bloch decomposition.
///
/// Implemented by `ZOmega` (→ Clifford+T) and `ZZeta` (→ Clifford+√T).
/// Having this on the ring type means `U2<R>` automatically determines
/// the SO3 representation, rotation generators, and Clifford table lookup.
pub trait GateRing: RingElem + Mul<Output = Self> + Sub<Output = Self>{
    /// SO(3) matrix type for this gate set (`SO3T` or `SO3Q`).
    type SO3: SO3Ops + Debug;

    /// Convert a U2 matrix to its exact SO(3) representation.
    fn so3_from_u2(u: &U2<Self>) -> Self::SO3;

    /// Rz(+step) rotation generator (step = π/4 for T, π/8 for √T).
    fn rz_pos() -> Self::SO3;
    fn rx_pos() -> Self::SO3;
    fn ry_pos() -> Self::SO3;
    fn rz_neg() -> Self::SO3;
    fn rx_neg() -> Self::SO3;
    fn ry_neg() -> Self::SO3;

    /// Positive Rx/Ry/Rz rotation as U2 matrices (used to track output gate product).
    fn rx_pos_u2() -> U2<Self>;
    fn ry_pos_u2() -> U2<Self>;
    fn rz_pos_u2() -> U2<Self>;

    /// Find the Clifford gate whose SO3 equals `residual` exactly.
    /// (Used for debug; correctness in synthesis uses `identify_clifford_from_u2`.)
    fn identify_clifford(residual: &Self::SO3) -> Option<&'static str>;

    /// Find the Clifford gate label whose gate-primitive U2 matches `u` by diamond distance.
    /// This is the correct identification for synthesis (avoids S-convention mismatch).
    fn identify_clifford_from_u2(u: &U2<Self>) -> Option<&'static str>;

    /// Name of the magic gate: `"T"` or `"Q"`.
    fn magic_gate_name() -> &'static str;

    /// Ring-specific entry point for the Bloch-sphere decomposition.
    ///
    /// ZOmega routes through the single-step greedy peel
    /// (`decompose_so3`), which is correct for Clifford+T (rotation step
    /// = π/4). ZZeta routes through the canonical-form algorithm of
    /// Forest–Gosset–Kliuchnikov–McKinnon (`decompose_so3_canonical_q`),
    /// which tries all 9 candidate peels per step (axis × {1,2,3} π/8
    /// rotations) and is correct for Clifford+√T.
    fn decompose_target(target: &U2<Self>) -> String;
}

// ─── GateRing for ZOmega (Clifford+T) ────────────────────────────────────────

impl GateRing for ZOmega {
    type SO3 = SO3T;

    fn so3_from_u2(u: &U2<Self>) -> SO3T { SO3T::from_u2(u) }
    fn rz_pos() -> SO3T { rz_pos() }
    fn rx_pos() -> SO3T { rx_pos() }
    fn ry_pos() -> SO3T { ry_pos() }
    fn rz_neg() -> SO3T { rz_neg() }
    fn rx_neg() -> SO3T { rx_neg() }
    fn ry_neg() -> SO3T { ry_neg() }

    fn rx_pos_u2() -> U2T { U2T::h() * U2T::t() * U2T::h() }
    fn ry_pos_u2() -> U2T {
        // Ry(π/4) = S · Rx(π/4) · S†
        U2T::s() * U2T::h() * U2T::t() * U2T::h() * U2T::s().dagger()
    }
    fn rz_pos_u2() -> U2T { U2T::t() }

    fn identify_clifford(residual: &SO3T) -> Option<&'static str> {
        CLIFFORD_TABLE_T
            .iter()
            .find(|(_, u)| SO3T::from_u2(u) == *residual)
            .map(|(name, _)| *name)
    }

    fn identify_clifford_from_u2(u: &U2T) -> Option<&'static str> {
        // Pick the closest Clifford (argmin) rather than the first within
        // an absolute tolerance. Cliffords are pairwise distinct
        // (`test_cliffords_distinct`), so argmin is unambiguous.
        CLIFFORD_TABLE_T
            .iter()
            .map(|(name, _)| {
                let gate_u: U2T = name.chars().fold(U2T::eye(), |acc, ch| {
                    acc * match ch {
                        'H' => U2T::h(),
                        'S' => U2T::s(),
                        'X' => U2T::x(),
                        'Y' => U2T::y(),
                        'Z' => U2T::z(),
                        _ => U2T::eye(),
                    }
                });
                (*name, gate_u.diamond_distance(u))
            })
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .filter(|(_, d)| *d < 1e-3)
            .map(|(name, _)| name)
    }

    fn magic_gate_name() -> &'static str { "T" }

    fn decompose_target(target: &U2<Self>) -> String {
        decompose_so3::<Self>(target)
    }
}

// ─── GateRing for ZZeta (Clifford+√T) ────────────────────────────────────────

/// Embed a Clifford SO3T (entries in Z[√2]) into SO3Q (entries in Z[γ]).
///
/// Z[√2] ↪ Z[γ]:  R2(a, b) → R4(a, b, 0, 0).
fn embed_so3t_in_so3q(m: &SO3T) -> SO3Q {
    let e: [Ratio<R4>; 9] = std::array::from_fn(|i| {
        let R2(a, b) = m.e[i].num;
        Ratio { num: R4(a, b, INT_ZERO, INT_ZERO), exp: m.e[i].exp }
    });
    SO3 { e }
}

impl GateRing for ZZeta {
    type SO3 = SO3Q;

    fn so3_from_u2(u: &U2<Self>) -> SO3Q { SO3Q::from_u2(u) }
    fn rz_pos() -> SO3Q { rz_pos_q() }
    fn rx_pos() -> SO3Q { rx_pos_q() }
    fn ry_pos() -> SO3Q { ry_pos_q() }
    fn rz_neg() -> SO3Q { rz_neg_q() }
    fn rx_neg() -> SO3Q { rx_neg_q() }
    fn ry_neg() -> SO3Q { ry_neg_q() }

    fn rx_pos_u2() -> U2Q { U2Q::h() * U2Q::q() * U2Q::h() }
    fn ry_pos_u2() -> U2Q {
        // Ry(π/4) = S · Rx(π/4) · S†
        U2Q::s() * U2Q::h() * U2Q::q() * U2Q::h() * U2Q::s().dagger()
    }
    fn rz_pos_u2() -> U2Q { U2Q::q() }

    fn identify_clifford(residual: &SO3Q) -> Option<&'static str> {
        CLIFFORD_TABLE_T
            .iter()
            .find(|(_, u)| embed_so3t_in_so3q(&SO3T::from_u2(u)) == *residual)
            .map(|(name, _)| *name)
    }

    fn identify_clifford_from_u2(u: &U2Q) -> Option<&'static str> {
        // Pick the closest Clifford (argmin over the 24 entries) rather than
        // the first one within an absolute tolerance: the diamond_distance
        // computation goes through float scaling 1/√2^k where k grows with
        // the U2's denominator exponent, so the per-comparison noise floor
        // grows with k and a fixed-1e-9 threshold can fail to identify the
        // true Clifford for large-k inputs.
        CLIFFORD_TABLE_T
            .iter()
            .map(|(name, _)| {
                let gate_u: U2Q = name.chars().fold(U2Q::eye(), |acc, ch| {
                    acc * match ch {
                        'H' => U2Q::h(),
                        'S' => U2Q::s(),
                        'X' => U2Q::x(),
                        'Y' => U2Q::y(),
                        'Z' => U2Q::z(),
                        _ => U2Q::eye(),
                    }
                });
                (*name, gate_u.diamond_distance(u))
            })
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .filter(|(_, d)| *d < 1e-3)
            .map(|(name, _)| name)
    }

    fn magic_gate_name() -> &'static str { "Q" }

    fn decompose_target(target: &U2<Self>) -> String {
        decompose_so3_canonical_q(target)
    }
}

// ─── GateRing for ZOmicron (Clifford+R_z(π/6)) ───────────────────────────────

impl GateRing for ZOmicron {
    type SO3 = SO3Omicron;

    fn so3_from_u2(u: &U2<Self>) -> SO3Omicron { SO3Omicron::from_u2(u) }
    fn rz_pos() -> SO3Omicron { rz_pos_o() }
    fn rx_pos() -> SO3Omicron { rx_pos_o() }
    fn ry_pos() -> SO3Omicron { ry_pos_o() }
    fn rz_neg() -> SO3Omicron { rz_neg_o() }
    fn rx_neg() -> SO3Omicron { rx_neg_o() }
    fn ry_neg() -> SO3Omicron { ry_neg_o() }

    /// R gate = diag(1, ξ) = [[1,0],[0,ξ]], the Rz(π/6) magic gate.
    fn rz_pos_u2() -> U2<ZOmicron> {
        U2::new(ZOmicron::ONE, ZOmicron::ZERO, ZOmicron::ZERO, ZOmicron::XI, 0)
    }
    fn rx_pos_u2() -> U2<ZOmicron> {
        U2::<ZOmicron>::h() * Self::rz_pos_u2() * U2::<ZOmicron>::h()
    }
    fn ry_pos_u2() -> U2<ZOmicron> {
        let s = U2::<ZOmicron>::s();
        let h = U2::<ZOmicron>::h();
        let r = Self::rz_pos_u2();
        s * h * r * h * s.dagger()
    }

    fn identify_clifford(residual: &SO3Omicron) -> Option<&'static str> {
        // Compare SO3Omicron against all 24 Clifford SO3 matrices (computed via SO3T → SO3O).
        // A match exists iff all 9 entries are equal (exact ring comparison).
        CLIFFORD_TABLE_T
            .iter()
            .find(|(_, u)| {
                let u_o = clifford_u2t_to_u2_omicron(u);
                SO3Omicron::from_u2(&u_o) == *residual
            })
            .map(|(name, _)| *name)
    }

    fn identify_clifford_from_u2(u: &U2<ZOmicron>) -> Option<&'static str> {
        CLIFFORD_TABLE_T
            .iter()
            .map(|(name, _)| {
                let gate_u: U2<ZOmicron> = name.chars().fold(U2::eye(), |acc, ch| {
                    acc * match ch {
                        'H' => U2::<ZOmicron>::h(),
                        'S' => U2::<ZOmicron>::s(),
                        'X' => U2::<ZOmicron>::x(),
                        'Y' => U2::<ZOmicron>::y(),
                        'Z' => U2::<ZOmicron>::z(),
                        _   => U2::eye(),
                    }
                });
                (*name, gate_u.diamond_distance(u))
            })
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .filter(|(_, d)| *d < 1e-3)
            .map(|(name, _)| name)
    }

    fn magic_gate_name() -> &'static str { "R" }

    fn decompose_target(target: &U2<Self>) -> String {
        decompose_so3_canonical_n6(target)
    }
}

/// Convert a Clifford U2T (ZOmega, k≤1) to an equivalent U2<ZOmicron>.
///
/// All 24 Cliffords have k ∈ {0, 1}. For k=0: entries map directly.
/// For k=1: embed ZOmega entries into ZOmicron (ω = e^{iπ/4} not in ZOmicron,
/// so we can't convert arbitrary U2T). Fortunately, Clifford k=1 entries are
/// in {0, ±1, ±ω², ±ω³, ±ω⁶} ⊂ Z[i] ⊂ ZOmicron (since ω² = i).
///
/// Mapping: ZOmega(a,b,c,d) → a + b·e^{iπ/4} + c·e^{iπ/2} + d·e^{i3π/4}
///   = a + b·(1+i)/√2 + c·i + d·(-1+i)/√2    [not exact in ZOmicron in general]
///
/// But for Clifford entries ∈ {0, ±1} (k=1 matrices scaled by √2): the
/// raw ZOmega integer coords map to ZOmicron via ω^k → ZOmicron form.
/// We handle this by evaluating to complex float and matching to ZOmicron
/// 12th roots (which are all possible k=1 Clifford entry magnitudes).
fn clifford_u2t_to_u2_omicron(u: &U2T) -> U2<ZOmicron> {
    // Float conversion then scale by √2^k and round to ZOmicron integers.
    // Clifford entries are small (coefficients ≤ 2), so this is exact.
    let scale = (1u64 << u.k) as f64;  // 2^k (not √2^k — we want integer coords)
    let entries = [u.u11, u.u12, u.u21, u.u22];
    let convert = |z: crate::rings::ZOmega| -> ZOmicron {
        let c = z.to_complex() * scale.sqrt();  // this is the integer numerator as complex
        // Represent c as ZOmicron by matching to a 12th-root-of-unity linear combination.
        // c = a + b*xi + c2*xi^2 + d*xi^3, solve via:
        //   Re(c) = a + b*(√3/2) + c2*(1/2)
        //   Im(c) = b*(1/2) + c2*(√3/2) + d
        // with a,b,c2,d ∈ Z. For Clifford entries, all coords are small integers.
        let re = c.re;
        let im = c.im;
        // Try all combinations of (a,b,c2,d) ∈ {-2..=2}^4
        for a in -2i32..=2 {
            for b in -2i32..=2 {
                for c2 in -2i32..=2 {
                    let d_f = im - (b as f64) * 0.5 - (c2 as f64) * (3.0f64.sqrt()/2.0);
                    let d = d_f.round() as i32;
                    if (d_f - d as f64).abs() > 0.01 { continue; }
                    let z_test = ZOmicron::from_i32(a, b, c2, d);
                    let re_test = z_test.to_complex().re;
                    let im_test = z_test.to_complex().im;
                    if (re_test - re).abs() < 0.01 && (im_test - im).abs() < 0.01 {
                        return z_test;
                    }
                }
            }
        }
        ZOmicron::ZERO  // should never happen for valid Clifford entries
    };
    U2::new(convert(entries[0]), convert(entries[1]), convert(entries[2]), convert(entries[3]), u.k)
}

// ─── BlochDecomposer ─────────────────────────────────────────────────────────

/// Generic Bloch-sphere decomposer.
///
/// Stateless unit struct. `decompose` is generic over `R: GateRing`, so
/// `U2<ZOmega>` and `U2<ZZeta>` are handled by the same method.
#[derive(Default)]
pub struct BlochDecomposer;

impl BlochDecomposer {
    pub fn new() -> Self { Self }

    /// Decompose an exact unitary into a gate string.
    ///
    /// - `U2<ZOmega>` (`U2T`) → output in {H, S, T, X, Y, Z}
    /// - `U2<ZZeta>`  (`U2Q`) → output in {H, S, Q, X, Y, Z}
    pub fn decompose<R: GateRing>(&self, target: &U2<R>) -> String {
        R::decompose_target(target)
    }
}

// ─── Gate string translation ──────────────────────────────────────────────────

/// Translate a raw `{x,y,z,Clifford}` decomposition string into a gate string.
///
/// Rotation encoding (leftmost gate = leftmost matrix factor):
///   'x' → H·magic·H        (Rx step)
///   'y' → S·H·magic·H·S³   (Ry step: Ry(θ) = S · Rx(θ) · S†, with S† = S³)
///   'z' → magic             (Rz step)
///
/// `magic` is `"T"` for Clifford+T or `"Q"` for Clifford+√T.
/// After substitution, a fixpoint rewrite loop simplifies the result.
fn translate(raw: &str, magic: &str) -> String {
    let mut s = raw
        .replace('x', &format!("H{}H", magic))
        .replace('y', &format!("SH{}HSSS", magic))
        .replace('z', magic);

    let mut prev = String::new();
    while s != prev {
        prev = s.clone();
        // Commutations
        s = s.replace("SZ", "ZS");
        s = s.replace("TZ", "ZT");
        s = s.replace("QZ", "ZQ");
        s = s.replace("TS", "ST");
        s = s.replace("QS", "SQ");
        s = s.replace("QT", "TQ");
        // Combinations
        s = s.replace("QQ", "T");
        s = s.replace("TT", "S");
        s = s.replace("SS", "Z");
        // Cancellations
        s = s.replace("HH", "");
        s = s.replace("XX", "");
        s = s.replace("YY", "");
        s = s.replace("ZZ", "");
        s = s.replace('I', "");
    }
    s
}

/// Core peel-off loop and Clifford identification, generic over ring.
///
/// Uses negative rotation generators applied to SO3(target), matching the Python
/// reference implementation. This guarantees max_exp decreases by ≥1 each step.
///
/// Mathematical invariant after N steps:
///   rx_neg_N × … × rx_neg_1 × SO3(target) = C  (Clifford SO3)
/// ⟹ target = rx_pos_N × … × rx_pos_1 × gate_c
/// ⟹ p_output_u2 = rx_pos_N × … × rx_pos_1  (left-accumulated)
/// ⟹ gate_c = p_output_u2† × target
///
/// Gate string: raw_rev + clifford_suffix
///   raw_rev = "step_N…step_1" translates to p_output_u2;
///   appended Clifford makes the product equal target. ✓
///
/// This is the **single-step greedy** peel that works correctly for ZOmega
/// (Clifford+T): cos(nπ/4) has √2-exp ∈ {0, 1, 0, 1, …}, so each peel reduces
/// `max_exp` by exactly 1 and the optimal axis is determined by argmin among
/// the three single-step candidates.
///
/// For ZZeta (Clifford+√T) we instead use
/// [`decompose_so3_canonical_q`], which implements the
/// Forest–Gosset–Kliuchnikov–McKinnon (arXiv:1501.04944) canonical-form
/// algorithm: at each step it tries all 9 candidate peels
/// `R_p(a·π/8)` for `p ∈ {x,y,z}, a ∈ {1,2,3}` and picks the (unique, by
/// Theorem 4.1(c)) argmin in `max_exp`.
fn decompose_so3<R: GateRing>(target: &U2<R>) -> String {
    let rz = R::rz_neg();   // negative generators guarantee progress
    let rx = R::rx_neg();
    let ry = R::ry_neg();
    let rx_u2 = R::rx_pos_u2();  // positive U2 rotations (peeled steps)
    let ry_u2 = R::ry_pos_u2();
    let rz_u2 = R::rz_pos_u2();

    let mut raw = String::new();
    let mut so3 = R::so3_from_u2(target);  // start from SO3(target), not SO3(target†)
    let mut p_output_u2 = U2::<R>::eye();
    // For SO3T (Clifford+T) max_exp counts T-count, so each peel reduces by 1
    // and `max_steps = max_exp` is a tight bound. For SO3Q (Clifford+√T) the
    // √2-denom convention makes max_exp non-monotone per peel: SO3(T) has
    // max_exp=1 but T = QQ requires 2 Rz peels (T → Q → I, going 1 → 2 → 0).
    // Use a generous bound; the loop early-exits via `best == 0` when the
    // residual reaches a Clifford.
    // SO3T (Clifford+T) max_exp counts T-count, monotone-decreasing per peel:
    // `max_steps = max_exp` is tight. SO3Q (Clifford+√T) in √2-denom has
    // non-monotone max_exp per peel (e.g., SO3(T) max_exp=1 but T = QQ needs
    // 2 Rz peels: 1 → 2 → 0). Bound is generous; loop early-exits via
    // `best == 0` once a Clifford residual is reached.
    let max_steps = (so3.max_exp() as usize) * 4 + 32;

    for _ in 0..max_steps {
        if so3.max_exp() == 0 { break; }

        let mut rx_r = so3.clone(); rx_r.left_mul(&rx);
        let mut ry_r = so3.clone(); ry_r.left_mul(&ry);
        let mut rz_r = so3.clone(); rz_r.left_mul(&rz);

        let ex = rx_r.max_exp();
        let ey = ry_r.max_exp();
        let ez = rz_r.max_exp();
        let best = ex.min(ey).min(ez);

        if ex == best {
            so3 = rx_r;
            p_output_u2 = p_output_u2 * rx_u2;   // right-accumulate: p = rx_1*...*rx_k
            raw.push('x');
        } else if ey == best {
            so3 = ry_r;
            p_output_u2 = p_output_u2 * ry_u2;
            raw.push('y');
        } else {
            so3 = rz_r;
            p_output_u2 = p_output_u2 * rz_u2;
            raw.push('z');
        }

        if best == 0 { break; }
    }

    // After N steps: target = rx_pos_1 * ... * rx_pos_N * gate_c = p_output_u2 * gate_c
    // gate_c = p_output_u2† × target
    // Gate string: raw + clifford_suffix (no reversal — raw = "step_1…step_N" = p_output_u2)
    //   evaluates to p_output_u2 * gate_c = target. ✓
    let gate_c = p_output_u2.dagger() * *target;
    let clifford_suffix = R::identify_clifford_from_u2(&gate_c)
        .filter(|&name| name != "I")
        .unwrap_or("");
    let full_raw = format!("{raw}{clifford_suffix}");
    translate(&full_raw, R::magic_gate_name())
}

// ─── Canonical-form decomposition for ZOmicron (Clifford+R_z(π/6)) ──────────
//
// Analogous to the Forest–Gosset–Kliuchnikov–McKinnon (arXiv:1501.04944) algorithm
// for n=8, adapted for n=6. The same "try all candidates, pick argmin of max_exp"
// strategy is used with 15 candidates instead of 9:
//   axes ∈ {x, y, z}  ×  multiplicities ∈ {1, 2, 3, 4, 5}  = 15 candidates.
//
// The SO3 entries for Clifford+R_z(π/6) live in Z[√3] with denominator 2^exp.
// max_exp() counts the 2-adic denominator level; max_exp()==0 means a Clifford.

struct CanonicalCandidateO {
    axis: u8,
    a: u8,
    so3_neg: SO3Omicron,
    u2_pos: U2<ZOmicron>,
}

fn canonical_candidates_o() -> Vec<CanonicalCandidateO> {
    let mut out = Vec::with_capacity(15);
    let so3_axis_neg = [rx_neg_o(), ry_neg_o(), rz_neg_o()];
    let u2_axis_pos: [U2<ZOmicron>; 3] = [
        ZOmicron::rx_pos_u2(),
        ZOmicron::ry_pos_u2(),
        ZOmicron::rz_pos_u2(),
    ];
    for (axis_idx, (so3_step_neg, u2_step_pos)) in
        so3_axis_neg.iter().zip(u2_axis_pos.iter()).enumerate()
    {
        let mut cur_so3 = SO3Omicron::identity();
        let mut cur_u2  = U2::<ZOmicron>::eye();
        for a in 1..=5u8 {
            cur_so3 = so3_step_neg.clone() * cur_so3;
            cur_u2  = cur_u2 * *u2_step_pos;
            out.push(CanonicalCandidateO {
                axis: axis_idx as u8,
                a,
                so3_neg: cur_so3.clone(),
                u2_pos:  cur_u2,
            });
        }
    }
    out
}

/// Translate a single (axis, a) peel for n=6 into a literal gate-string fragment.
///
///   axis 0 (x): R_x(a·π/6) = H · R^a · H
///   axis 1 (y): R_y(a·π/6) = S · H · R^a · H · S†  (S† = S³ = ZS or SSS)
///   axis 2 (z): R_z(a·π/6) = R^a
///
/// The fragment is in the {H, S, R, Z} alphabet; simplify_n6 reduces further.
fn canonical_segment_string_n6(axis: u8, a: u8) -> String {
    let r_run: String = "R".repeat(a as usize);
    match axis {
        0 => format!("H{r_run}H"),
        1 => format!("SH{r_run}HSSS"),
        2 => r_run,
        _ => unreachable!(),
    }
}

/// Apply the n=6 gate-string simplification loop to a string in {H,S,R,X,Y,Z,I}.
pub fn simplify_gate_string_n6(input: &str) -> String {
    let mut s = input.to_string();
    let mut prev = String::new();
    while s != prev {
        prev = s.clone();
        s = s.replace("RRRRRR", "Z");
        s = s.replace("RRR", "S");
        s = s.replace("SS", "Z");
        s = s.replace("ZZ", "");
        s = s.replace("HH", "");
        s = s.replace("XX", "");
        s = s.replace("YY", "");
        s = s.replace("SZ", "ZS");
        s = s.replace("RZ", "ZR");
        s = s.replace('I', "");
    }
    s
}

/// Canonical-form decomposer for U2<ZOmicron> (Clifford+R_z(π/6)).
///
/// At each step, tries all 15 candidate peels (3 axes × multiplicities {1..5})
/// and picks the argmin of max_exp(). When max_exp()==0 the residual is a
/// Clifford; it's identified via diamond distance and appended as a suffix.
pub fn decompose_so3_canonical_n6(target: &U2<ZOmicron>) -> String {
    let candidates = canonical_candidates_o();

    let mut so3 = SO3Omicron::from_u2(target);
    let mut p_output_u2 = U2::<ZOmicron>::eye();
    let max_steps = (so3.max_exp() as usize) * 8 + 32;

    let mut raw_segments: Vec<String> = Vec::new();

    for _ in 0..max_steps {
        if so3.max_exp() == 0 { break; }

        let mut best_idx: usize = 0;
        let mut best_so3 = so3.clone();
        best_so3.left_mul(&candidates[0].so3_neg);
        let mut best_exp = best_so3.max_exp();

        for (idx, cand) in candidates.iter().enumerate().skip(1) {
            let mut trial = so3.clone();
            trial.left_mul(&cand.so3_neg);
            let trial_exp = trial.max_exp();
            if trial_exp < best_exp {
                best_exp = trial_exp;
                best_idx = idx;
                best_so3 = trial;
            }
        }

        let cand = &candidates[best_idx];
        so3 = best_so3;
        p_output_u2 = p_output_u2 * cand.u2_pos;
        raw_segments.push(canonical_segment_string_n6(cand.axis, cand.a));

        if best_exp == 0 { break; }
    }

    let gate_c = p_output_u2.dagger() * *target;
    let clifford_suffix = ZOmicron::identify_clifford_from_u2(&gate_c)
        .filter(|&name| name != "I")
        .unwrap_or("");

    let combined: String = raw_segments.join("");
    let full = format!("{combined}{clifford_suffix}");
    simplify_gate_string_n6(&full)
}

// ─── Canonical-form decomposition for ZZeta (Clifford+√T) ────────────────────
//
// Forest, Gosset, Kliuchnikov, McKinnon, *Exact synthesis of single-qubit
// unitaries over Clifford-cyclotomic gate sets* (arXiv:1501.04944), Section 4.
//
// For n=8 (Q = Rz(π/8)) the Bloch-sphere SO(3) representation `M̂` of any
// Clifford+√T unitary admits a unique factorisation
//
//     M̂ = R_{p_1}(a_1·π/8) · R_{p_2}(a_2·π/8) · … · R_{p_m}(a_m·π/8) · Ĉ
//
// with `p_i ∈ {x,y,z}`, `p_i ≠ p_{i+1}`, `a_i ∈ {1,2,3}`, and `Ĉ` a Clifford
// (Theorem 4.1 + Lemma 3.1). The greedy peel implemented in
// [`decompose_so3`] only considers single-step (a=1) candidates and so
// fails on `M̂ ∈ SO3<R4>`, where the √2-exponent of the entries of
// `R_p(a·π/8)` is non-monotone in a (a=1: 2, a=2: 1, a=3: 2 in our √2-denom
// convention). The algorithm below tries all 9 candidate peels at each step
// and picks the (unique, by Theorem 4.1(c)) argmin in `max_exp`.
//
// Termination: at each step the chosen peel strictly reduces `max_exp` on
// the *row* identified by Theorem 4.1(c); after finitely many steps the
// residual has `max_exp == 0` and is a Clifford.

/// Pre-computed candidate rotation generators for the canonical-form
/// algorithm: 9 (p, a) pairs `(p ∈ {x,y,z}, a ∈ {1,2,3})` with both the
/// negative SO3 generator (used to peel from the left of the SO3 residual)
/// and the positive U2 generator (right-accumulated to form `p_output_u2`).
struct CanonicalCandidate {
    /// Axis: 0=x, 1=y, 2=z.
    axis: u8,
    /// Rotation amount in units of π/8: 1, 2, or 3.
    a: u8,
    /// `R_p(-a·π/8)` as an SO3 matrix (left-multiplied to the residual).
    so3_neg: SO3Q,
    /// `R_p(+a·π/8)` as a U2 matrix (right-accumulated to track output).
    u2_pos: U2Q,
}

fn canonical_candidates() -> Vec<CanonicalCandidate> {
    let mut out = Vec::with_capacity(9);
    let so3_axis_neg = [rx_neg_q(), ry_neg_q(), rz_neg_q()];
    let u2_axis_pos: [U2Q; 3] = [
        ZZeta::rx_pos_u2(),
        ZZeta::ry_pos_u2(),
        ZZeta::rz_pos_u2(),
    ];
    for (axis_idx, (so3_step_neg, u2_step_pos)) in
        so3_axis_neg.iter().zip(u2_axis_pos.iter()).enumerate()
    {
        let mut cur_so3 = SO3Q::identity();
        let mut cur_u2 = U2Q::eye();
        for a in 1..=3u8 {
            cur_so3 = so3_step_neg.clone() * cur_so3;
            cur_u2 = cur_u2 * *u2_step_pos;
            out.push(CanonicalCandidate {
                axis: axis_idx as u8,
                a,
                so3_neg: cur_so3.clone(),
                u2_pos: cur_u2,
            });
        }
    }
    out
}

/// Forest et al. canonical-form decomposition for `U2<ZZeta>` (Clifford+√T).
///
/// At each step we left-multiply the SO3 residual by `R_p(-a·π/8)` for each
/// `(p, a) ∈ {x,y,z} × {1,2,3}` and pick the candidate that minimises
/// `max_exp`. The optimum is unique (Theorem 4.1(c)) once `max_exp > 0`.
/// Float-domain analog of [`canonical_form_axes_q`]. Operates on a 3×3
/// `f64` SO3 matrix so it can be applied to *continuous* targets (or to
/// inexact integer targets), returning the predicted FGKM syllable
/// sequence. The argmin proxy is `max(|entry|)` — the closest smooth
/// analog of integer-FGKM's `max_exp`.
///
/// Used by the Z1 prefix filter investigation: if this float prediction
/// matches the integer FGKM output for exact Clifford+√T targets, it's a
/// faithful predictor for arbitrary continuous targets too. `max_steps`
/// caps the iteration count for non-rounding-friendly inputs.
#[allow(dead_code)]
pub(crate) fn canonical_form_axes_q_float(
    so3: &mut [[f64; 3]; 3],
    max_steps: usize,
) -> Vec<(u8, u8)> {
    // Precompute the 9 candidate SO3 left-multipliers as 3×3 f64 matrices.
    // canonical_candidates() returns these for the (axis, a) ∈ {x,y,z} × {1,2,3} grid.
    let candidates = canonical_candidates();
    let cand_f: Vec<[[f64; 3]; 3]> = candidates.iter().map(|c| c.so3_neg.to_float()).collect();
    let cand_meta: Vec<(u8, u8)> = candidates.iter().map(|c| (c.axis, c.a)).collect();

    let max_abs = |m: &[[f64; 3]; 3]| -> f64 {
        let mut best: f64 = 0.0;
        for r in 0..3 {
            for c in 0..3 {
                let v = m[r][c].abs();
                if v > best { best = v; }
            }
        }
        best
    };

    let matmul = |a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]| -> [[f64; 3]; 3] {
        let mut out = [[0.0f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                let mut s = 0.0_f64;
                for k in 0..3 { s += a[i][k] * b[k][j]; }
                out[i][j] = s;
            }
        }
        out
    };

    // Convergence threshold: when max|entry| ≲ 1 we treat the residual as
    // a rotation by O(1) (i.e. effectively a Clifford in the float-domain
    // sense — there's no further reduction to pursue).
    const STOP_THRESHOLD: f64 = 1.0 + 1e-9;

    let mut out: Vec<(u8, u8)> = Vec::new();
    for _ in 0..max_steps {
        if max_abs(so3) <= STOP_THRESHOLD {
            break;
        }
        let mut best_idx: usize = 0;
        let first_trial = matmul(&cand_f[0], so3);
        let mut best_norm = max_abs(&first_trial);
        let mut best_so3 = first_trial;
        for (idx, cand) in cand_f.iter().enumerate().skip(1) {
            let trial = matmul(cand, so3);
            let n = max_abs(&trial);
            if n < best_norm {
                best_norm = n;
                best_idx = idx;
                best_so3 = trial;
            }
        }
        *so3 = best_so3;
        out.push(cand_meta[best_idx]);
        if best_norm <= STOP_THRESHOLD { break; }
    }
    out
}

/// Sister of [`decompose_so3_canonical_q`] that returns the FGKM
/// canonical-form `(axis, a)` syllable sequence directly, without
/// translating it to a gate string. Useful for the Z1 prefix-filter
/// investigation where we need the syllables, not their gate
/// representation. Same algorithm, same termination guarantee.
#[allow(dead_code)]
pub(crate) fn canonical_form_axes_q(target: &U2Q) -> Vec<(u8, u8)> {
    let candidates = canonical_candidates();
    let mut so3 = SO3Q::from_u2(target);
    let max_steps = (so3.max_exp() as usize) * 4 + 32;
    let mut out: Vec<(u8, u8)> = Vec::new();
    for _ in 0..max_steps {
        if so3.max_exp() == 0 {
            break;
        }
        let mut best_idx: usize = 0;
        let mut best_so3 = so3.clone();
        best_so3.left_mul(&candidates[0].so3_neg);
        let mut best_exp = best_so3.max_exp();
        for (idx, cand) in candidates.iter().enumerate().skip(1) {
            let mut trial = so3.clone();
            trial.left_mul(&cand.so3_neg);
            let trial_exp = trial.max_exp();
            if trial_exp < best_exp {
                best_exp = trial_exp;
                best_idx = idx;
                best_so3 = trial;
            }
        }
        let cand = &candidates[best_idx];
        so3 = best_so3;
        out.push((cand.axis, cand.a));
        if best_exp == 0 {
            break;
        }
    }
    out
}

fn decompose_so3_canonical_q(target: &U2Q) -> String {
    let candidates = canonical_candidates();

    let mut so3 = SO3Q::from_u2(target);
    let mut p_output_u2 = U2Q::eye();
    // Each peel reduces the maximum √2-exponent of the SO3 entries by at
    // least 1 (Theorem 4.1(b) gives an explicit closed form). The initial
    // exponent bounds the iteration count from above; the multiplicative
    // factor 4 + slack absorbs the difference between the paper's
    // β = 2cos(π/8) denominator base and our √2 denominator base.
    let max_steps = (so3.max_exp() as usize) * 4 + 32;

    let mut raw_segments: Vec<String> = Vec::new();

    for _ in 0..max_steps {
        if so3.max_exp() == 0 {
            break;
        }

        // Try all 9 candidates and pick the argmin in max_exp.
        let mut best_idx: usize = 0;
        let mut best_so3 = so3.clone();
        best_so3.left_mul(&candidates[0].so3_neg);
        let mut best_exp = best_so3.max_exp();
        for (idx, cand) in candidates.iter().enumerate().skip(1) {
            let mut trial = so3.clone();
            trial.left_mul(&cand.so3_neg);
            let trial_exp = trial.max_exp();
            if trial_exp < best_exp {
                best_exp = trial_exp;
                best_idx = idx;
                best_so3 = trial;
            }
        }

        let cand = &candidates[best_idx];
        so3 = best_so3;
        p_output_u2 = p_output_u2 * cand.u2_pos;
        raw_segments.push(canonical_segment_string(cand.axis, cand.a));

        if best_exp == 0 {
            break;
        }
    }

    // Identify the residual Clifford C such that target = p_output_u2 · C.
    let gate_c = p_output_u2.dagger() * *target;
    let clifford_suffix = ZZeta::identify_clifford_from_u2(&gate_c)
        .filter(|&name| name != "I")
        .unwrap_or("");

    let combined: String = raw_segments.join("");
    let full = format!("{combined}{clifford_suffix}");
    simplify_gate_string(&full)
}

/// Translate a single (axis, a) peel into a literal gate-string fragment
/// (without going through `translate`'s single-character substitution).
///
/// The fragment is in the {H, S, Q, X, Y, Z} alphabet; `simplify_gate_string`
/// applies the same combine/cancel rewrite loop as `translate`.
fn canonical_segment_string(axis: u8, a: u8) -> String {
    let q_run: String = "Q".repeat(a as usize);
    match axis {
        0 => format!("H{q_run}H"),       // x: R_x(a·π/8) = H · Q^a · H
        1 => format!("SH{q_run}HSSS"),   // y: R_y(a·π/8) = S · H · Q^a · H · S†
        2 => q_run,                       // z: R_z(a·π/8) = Q^a
        _ => unreachable!(),
    }
}

/// Apply the gate-string simplification rewrite loop to a free-form gate
/// string in the {H, S, Q, T, X, Y, Z, I} alphabet.
fn simplify_gate_string(input: &str) -> String {
    let mut s = input.to_string();
    let mut prev = String::new();
    while s != prev {
        prev = s.clone();
        // Commutations
        s = s.replace("SZ", "ZS");
        s = s.replace("TZ", "ZT");
        s = s.replace("QZ", "ZQ");
        s = s.replace("TS", "ST");
        s = s.replace("QS", "SQ");
        s = s.replace("QT", "TQ");
        // Combinations
        s = s.replace("QQ", "T");
        s = s.replace("TT", "S");
        s = s.replace("SS", "Z");
        // Cancellations
        s = s.replace("HH", "");
        s = s.replace("XX", "");
        s = s.replace("YY", "");
        s = s.replace("ZZ", "");
        s = s.replace('I', "");
    }
    s
}

// ─── PyO3 ─────────────────────────────────────────────────────────────────────

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Python-facing BlochDecomposer.
///
/// Stateless: construct once, call decompose() with the U2T parameters.
#[cfg(feature = "python")]
#[pyclass(name = "BlochDecomposer")]
pub struct PyBlochDecomposer;

#[cfg(feature = "python")]
#[pymethods]
impl PyBlochDecomposer {
    #[new]
    fn new() -> Self { Self }

    /// Decompose a unitary into a gate string.
    /// Accepts U2(ZOmega) → {H, S, T} or U2(ZZeta) → {H, S, T, Q}.
    fn decompose(&self, u: pyo3::PyRef<'_, PyU2>) -> String {
        match u.to_inner() {
            U2Variant::Omega(u2t) => BlochDecomposer.decompose(u2t),
            U2Variant::Zeta(u2q)  => BlochDecomposer.decompose(u2q),
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use crate::rings::ZOmega;
    use num_complex::Complex64;
    use std::f64::consts::FRAC_1_SQRT_2;

    // ── Exact SO3T construction from gate characters (test-only) ─────────────

    fn gate_to_u2t(ch: char) -> U2T {
        let u = match ch {
            'H' => { U2T::h() }
            'S' => { U2T::s() }
            'T' => { U2T::t() }
            'Z' => { U2T::z() }
            'X' => { U2T::x() }
            'Y' => { U2T::y() }
            _ => { panic!("Invalid gate character: {ch}"); }
        };
        u
    }

    /// Build exact SO3T for a gate string by left-multiplying each gate's SO3.
    fn gates_to_u2t(gate_str: &str) -> U2T {
        let mut u = U2T::eye();
        for ch in gate_str.chars() {
            let g = gate_to_u2t(ch);
            u = u * g;
        }
       u
    }

    /// Decompose a gate string via exact SO3T.
    fn decompose_from_gates_t(gates: &str) -> String {
        let u2 = gates_to_u2t(gates);
        decompose_so3::<ZOmega>(&u2)
    }

    // ── Same helpers, ZZeta variants (Clifford+√T) ────────────────────────────

    fn gate_to_u2q(ch: char) -> U2Q {
        match ch {
            'H' => U2Q::h(),
            'S' => U2Q::s(),
            'T' => U2Q::t(),
            'Q' => U2Q::q(),
            'Z' => U2Q::z(),
            'X' => U2Q::x(),
            'Y' => U2Q::y(),
            _ => panic!("Invalid gate character: {ch}"),
        }
    }

    fn gates_to_u2q(gate_str: &str) -> U2Q {
        let mut u = U2Q::eye();
        for ch in gate_str.chars() {
            u = u * gate_to_u2q(ch);
        }
        u
    }

    // ── Tests ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_cliffords() {
    }

    #[test]
    fn test_sdg_gate() {
        let s = U2T::s();
        let decomp = BlochDecomposer.decompose(&s);
        assert_eq!(decomp, "S");
    }

    #[test]
    fn test_identity() {
        let eye = U2T::eye();
        let decomp = BlochDecomposer.decompose(&eye);
        assert_eq!(decomp, "");
    }

    #[test]
    fn test_known_sequence_1() {
        let gates = "SSSHTHS";
        let decomp = decompose_from_gates_t(gates);
        let u_in = gates_to_u2t(gates);
        let u_out = gates_to_u2t(&decomp);
        // diamond_distance routes through to_complex()/f64; the sqrt(1 - x²)
        // branch where x ≈ 1 gives the sqrt-of-f64-epsilon noise floor (~1.5e-8).
        let dist = u_in.diamond_distance(&u_out);
        assert!(
            dist < 1e-7,
            "decomp \"{decomp}\" doesn't match input \"{gates}\": dist={dist:.3e}",
        );
    }

    #[test]
    fn test_known_sequence_2() {
        let gates = "HTH";
        let decomp = decompose_from_gates_t(gates);
        assert_eq!(decomp, gates);
    }

    #[test]
    fn test_known_sequence_3() {
        let gates = "HTHSTHTHSTHTH";
        let decomp = decompose_from_gates_t(gates);
        let u1 = gates_to_u2t(gates);
        let u2 = gates_to_u2t(&decomp);
        assert!(u1.diamond_distance(&u2) < 1e-10,
            "decomp: input=\"{gates}\" → \"{decomp}\", dist={:.3e}",
            u1.diamond_distance(&u2));
    }

    /// Debug: print gate_c details for the 5-T circuit.
    #[test]
    fn test_debug_gate_c() {
        let gates = "HTHSTHTHSTHTH";
        let target = gates_to_u2t(gates);

        let rz = rz_pos();
        let rx = rx_pos();
        let ry = ry_pos();
        let rx_u2 = ZOmega::rx_pos_u2();
        let ry_u2 = ZOmega::ry_pos_u2();
        let rz_u2 = ZOmega::rz_pos_u2();

        let mut so3 = SO3T::from_u2(&target.dagger());
        let mut p_output_u2 = U2T::eye();
        let max_steps = so3.max_exp() as usize;

        for _ in 0..max_steps {
            if so3.max_exp() == 0 { break; }
            let mut rx_r = so3.clone(); rx_r.left_mul(&rx);
            let mut ry_r = so3.clone(); ry_r.left_mul(&ry);
            let mut rz_r = so3.clone(); rz_r.left_mul(&rz);
            let ex = rx_r.max_exp();
            let ey = ry_r.max_exp();
            let ez = rz_r.max_exp();
            let best = ex.min(ey).min(ez);
            if ex == best { so3 = rx_r; p_output_u2 = p_output_u2 * rx_u2; }
            else if ey == best { so3 = ry_r; p_output_u2 = p_output_u2 * ry_u2; }
            else { so3 = rz_r; p_output_u2 = p_output_u2 * rz_u2; }
            if best == 0 { break; }
        }

        let gate_c_v1 = p_output_u2.dagger() * target;
        let gate_c_v2 = target * p_output_u2.dagger();

        eprintln!("p_output_u2: u11={:?} u12={:?} u21={:?} u22={:?} k={}",
            p_output_u2.u11, p_output_u2.u12, p_output_u2.u21, p_output_u2.u22, p_output_u2.k);
        eprintln!("gate_c (P†·target): u11={:?} u12={:?} u21={:?} u22={:?} k={}",
            gate_c_v1.u11, gate_c_v1.u12, gate_c_v1.u21, gate_c_v1.u22, gate_c_v1.k);
        eprintln!("gate_c (target·P†): u11={:?} u12={:?} u21={:?} u22={:?} k={}",
            gate_c_v2.u11, gate_c_v2.u12, gate_c_v2.u21, gate_c_v2.u22, gate_c_v2.k);

        eprintln!("\nDistances from P†·target to Clifford table entries (U2T table values):");
        for (name, u2t_entry) in CLIFFORD_TABLE_T {
            let d = gate_c_v1.diamond_distance(u2t_entry);
            if d < 0.1 { eprintln!("  {name}: dist={d:.6e}  <-- MATCH"); }
            else { eprintln!("  {name}: dist={d:.6e}"); }
        }

        eprintln!("\nDistances from P†·target to gate-primitive Cliffords:");
        for (name, _) in CLIFFORD_TABLE_T {
            let gate_u: U2T = name.chars().fold(U2T::eye(), |acc, ch| {
                acc * match ch {
                    'H' => U2T::h(), 'S' => U2T::s(), 'X' => U2T::x(),
                    'Y' => U2T::y(), 'Z' => U2T::z(), _ => U2T::eye(),
                }
            });
            let d = gate_c_v1.diamond_distance(&gate_u);
            if d < 0.1 { eprintln!("  {name}: dist={d:.6e}  <-- MATCH"); }
            else { eprintln!("  {name}: dist={d:.6e}"); }
        }

        eprintln!("\nDistances from target·P† to gate-primitive Cliffords:");
        for (name, _) in CLIFFORD_TABLE_T {
            let gate_u: U2T = name.chars().fold(U2T::eye(), |acc, ch| {
                acc * match ch {
                    'H' => U2T::h(), 'S' => U2T::s(), 'X' => U2T::x(),
                    'Y' => U2T::y(), 'Z' => U2T::z(), _ => U2T::eye(),
                }
            });
            let d = gate_c_v2.diamond_distance(&gate_u);
            if d < 0.1 { eprintln!("  {name}: dist={d:.6e}  <-- MATCH"); }
            else { eprintln!("  {name}: dist={d:.6e}"); }
        }
    }

    /// Debug: trace peel loop for the 5-T circuit.
    #[test]
    fn test_debug_peel_5t() {
        let gates = "HTHSTHTHSTHTH";
        let u2 = gates_to_u2t(gates);
        eprintln!("U2T k={}", u2.k);
        let so3 = SO3T::from_u2(&u2.dagger());
        eprintln!("Initial max_exp = {}", so3.max_exp());

        let rz = rz_pos();
        let rx = rx_pos();
        let ry = ry_pos();
        let max_steps = so3.max_exp() as usize;
        eprintln!("max_steps = {max_steps}");
        let mut so3 = so3;
        let mut raw = String::new();

        for step in 0..max_steps {
            let cur = so3.max_exp();
            if cur == 0 { eprintln!("step {step}: max_exp=0, breaking"); break; }

            let mut rx_r = so3.clone(); rx_r.left_mul(&rx);
            let mut ry_r = so3.clone(); ry_r.left_mul(&ry);
            let mut rz_r = so3.clone(); rz_r.left_mul(&rz);

            let ex = rx_r.max_exp();
            let ey = ry_r.max_exp();
            let ez = rz_r.max_exp();
            let best = ex.min(ey).min(ez);

            let ch = if ex == best { so3 = rx_r; 'x' }
                     else if ey == best { so3 = ry_r; 'y' }
                     else { so3 = rz_r; 'z' };
            raw.push(ch);
            eprintln!("step {step}: cur={cur}, ex={ex} ey={ey} ez={ez} → chose '{ch}', new max_exp={}", so3.max_exp());
            if best == 0 { eprintln!("best=0, breaking"); break; }
        }
        eprintln!("raw = \"{raw}\"");
        let cliff = ZOmega::identify_clifford(&so3);
        eprintln!("Clifford = {cliff:?}");
    }

    #[test]
    fn test_roundtrip_random() {
        let mut rng = rand::rng();
        for _ in 0..100 {
            let gates: String = (0..rng.random_range(100..=500))
                .map(|_| ['H', 'S', 'T'][rng.random_range(0..3)])
                .collect();
            let target     = gates_to_u2t(&gates);
            let decomp = BlochDecomposer.decompose(&target);
            let from_decomp = gates_to_u2t(&decomp);
            let dist = target.diamond_distance(&from_decomp);
            assert!(dist < 1e-6,
                "random roundtrip: input=\"{gates}\" → \"{decomp}\", dist={dist:.3e}");
        }
    }

    // ── Translation tests (moved from translation.rs) ─────────────────────────

    fn gate_to_mat(gates: &str) -> [[Complex64; 2]; 2] {
        use std::f64::consts::PI;
        let i  = Complex64::new(0.0, 1.0);
        let r2 = FRAC_1_SQRT_2;
        let h  = [[Complex64::new(r2, 0.0), Complex64::new(r2,  0.0)],
                  [Complex64::new(r2, 0.0), Complex64::new(-r2, 0.0)]];
        let s  = [[Complex64::new(1.0, 0.0), Complex64::ZERO],
                  [Complex64::ZERO, i]];
        let z  = [[Complex64::new(1.0, 0.0), Complex64::ZERO],
                  [Complex64::ZERO, Complex64::new(-1.0, 0.0)]];
        let x  = [[Complex64::ZERO, Complex64::new(1.0, 0.0)],
                  [Complex64::new(1.0, 0.0), Complex64::ZERO]];
        let t  = [[Complex64::from_polar(1.0, -PI/8.0), Complex64::ZERO],
                  [Complex64::ZERO, Complex64::from_polar(1.0, PI/8.0)]];
        let id = [[Complex64::new(1.0, 0.0), Complex64::ZERO],
                  [Complex64::ZERO, Complex64::new(1.0, 0.0)]];
        fn mmul(a: [[Complex64;2];2], b: [[Complex64;2];2]) -> [[Complex64;2];2] {
            let mut out = [[Complex64::ZERO; 2]; 2];
            for r in 0..2 { for c in 0..2 { for k in 0..2 { out[r][c] += a[r][k] * b[k][c]; } } }
            out
        }
        let mut mat = id;
        for ch in gates.chars() {
            // Right-multiply: leftmost gate is the leftmost matrix factor,
            // matching the synthesis-side `gates_to_u2t` convention.
            mat = mmul(mat, match ch { 'H'=>h, 'S'=>s, 'Z'=>z, 'X'=>x, 'T'=>t, _=>id });
        }
        mat
    }

    fn diamond_dist_mat(a: [[Complex64;2];2], b: [[Complex64;2];2]) -> f64 {
        let tr: Complex64 = (0..2).flat_map(|r| (0..2).map(move |c| a[r][c] * b[r][c].conj())).sum();
        (1.0 - tr.norm_sqr() / 4.0).max(0.0).sqrt()
    }

    #[test]
    fn test_x_translation_is_rx() {
        use std::f64::consts::PI;
        let c = (PI/8.0).cos(); let s = -(PI/8.0).sin();
        let rx = [[Complex64::new(c,0.0), Complex64::new(0.0,s)],
                  [Complex64::new(0.0,s), Complex64::new(c,0.0)]];
        assert!(diamond_dist_mat(gate_to_mat("HTH"), rx) < 1e-10);
    }

    #[test]
    fn test_y_translation_is_ry() {
        use std::f64::consts::PI;
        let translated = translate("y", "T");
        let c = (PI/8.0).cos(); let s = (PI/8.0).sin();
        let ry = [[Complex64::new(c,0.0), Complex64::new(-s,0.0)],
                  [Complex64::new(s,0.0), Complex64::new(c,0.0)]];
        assert!(diamond_dist_mat(gate_to_mat(&translated), ry) < 1e-10,
            "'y'→\"{translated}\"");
    }

    #[test]
    fn test_z_translation_is_rz() {
        use std::f64::consts::PI;
        let rz = [[Complex64::from_polar(1.0,-PI/8.0), Complex64::ZERO],
                  [Complex64::ZERO, Complex64::from_polar(1.0,PI/8.0)]];
        assert!(diamond_dist_mat(gate_to_mat("T"), rz) < 1e-10);
    }

    // ── ZZeta / Clifford+√T decomposer round-trip tests ───────────────────────
    //
    // Phase 0 of the Clifford+√T effort: the Bloch decomposer claims to be
    // generic over `R: GateRing`, but only ZOmega has been exercised. These
    // tests run the analogous coverage for U2<ZZeta>.

    #[test]
    fn test_zzeta_identity_decomposes_to_empty() {
        let eye = U2Q::eye();
        assert_eq!(BlochDecomposer.decompose(&eye), "");
    }

    #[test]
    fn test_zzeta_q_alone_decomposes_to_q() {
        // Q is the magic gate; should decompose to itself as a single character.
        let q = U2Q::q();
        assert_eq!(BlochDecomposer.decompose(&q), "Q");
    }

    #[test]
    fn test_zzeta_qq_simplifies_to_t() {
        // Q·Q = T; the translate() rewrite layer collapses "QQ" → "T".
        let qq = U2Q::q() * U2Q::q();
        assert_eq!(BlochDecomposer.decompose(&qq), "T");
    }

    #[test]
    fn test_zzeta_clifford_only_inputs() {
        // No magic gates → output should be a Clifford string with no Q or T.
        for input in ["S", "H", "Z", "X", "Y", "SH", "HS", "HSH", "SHSH"] {
            let u = gates_to_u2q(input);
            let decomp = BlochDecomposer.decompose(&u);
            assert!(
                !decomp.contains('Q') && !decomp.contains('T'),
                "Clifford-only input \"{input}\" produced non-Clifford decomp \"{decomp}\""
            );
            let recovered = gates_to_u2q(&decomp);
            let dist = u.diamond_distance(&recovered);
            assert!(dist < 1e-9,
                "Clifford-only round-trip: \"{input}\" → \"{decomp}\", dist={dist:.3e}");
        }
    }

    #[test]
    fn test_zzeta_known_sequence_qhq() {
        let gates = "QHQ";
        let u_in = gates_to_u2q(gates);
        let decomp = BlochDecomposer.decompose(&u_in);
        let u_out = gates_to_u2q(&decomp);
        let dist = u_in.diamond_distance(&u_out);
        assert!(dist < 1e-7,
            "decomp \"{decomp}\" doesn't match input \"{gates}\": dist={dist:.3e}");
    }

    // ── ZZeta decomposer history ──────────────────────────────────────────────
    //
    // Phase 0 of the Clifford+√T effort fixed a value-level bug in
    // `SO3Q::from_u2` (γ-denominator → √2-denominator), then surfaced an
    // *algorithmic* gap: the greedy single-step peeling that decomposes
    // SO3T correctly does not converge on SO3Q for non-trivial inputs.
    //
    //   - SO3T (Clifford+T): cos(nπ/4) has √2-exp ∈ {0, 1, 0, 1, …} —
    //     monotone-decreasing per single Rz peel. Greedy converges in
    //     `max_exp` steps.
    //   - SO3Q (Clifford+√T): cos(nπ/8) has √2-exp ∈ {0, 2, 1, 2, …}. A
    //     single Rz(π/8) peel can transiently *increase* max_exp, which
    //     makes single-step argmin go off-axis on non-trivial inputs.
    //
    // The fix lives in `decompose_so3_canonical_q` (above), which
    // implements the canonical-form algorithm of Forest, Gosset,
    // Kliuchnikov, McKinnon (arXiv:1501.04944, Section 4) for n=8: at each
    // step it tries all 9 candidate peels `R_p(a·π/8)` for
    // `p ∈ {x,y,z}, a ∈ {1,2,3}` and picks the (unique, by Theorem 4.1(c))
    // argmin in `max_exp`. `BlochDecomposer::decompose` dispatches via
    // `GateRing::decompose_target`: ZOmega keeps the original greedy peel
    // (correct for Clifford+T), ZZeta routes through the canonical-form
    // algorithm.

    #[test]
    fn test_zzeta_mixed_t_and_q() {
        // T and Q can mix freely in input — both produce ZZeta unitaries
        // (T = Q·Q algebraically, but the decomposer should not care which
        // is supplied).
        let gates = "HTHQHQH";
        let u_in = gates_to_u2q(gates);
        let decomp = BlochDecomposer.decompose(&u_in);
        let u_out = gates_to_u2q(&decomp);
        let dist = u_in.diamond_distance(&u_out);
        assert!(dist < 1e-7,
            "decomp \"{decomp}\" doesn't match input \"{gates}\": dist={dist:.3e}");
    }

    #[test]
    fn test_zzeta_roundtrip_random_hsq() {
        // Random Clifford+√T strings over the {H, S, Q} alphabet — Q alone
        // with Cliffords is sufficient to generate the full Clifford+√T
        // group.
        let mut rng = rand::rng();
        for _ in 0..50 {
            let gates: String = (0..rng.random_range(50..=200))
                .map(|_| ['H', 'S', 'Q'][rng.random_range(0..3)])
                .collect();
            let target = gates_to_u2q(&gates);
            let decomp = BlochDecomposer.decompose(&target);
            let recovered = gates_to_u2q(&decomp);
            let dist = target.diamond_distance(&recovered);
            assert!(dist < 1e-6,
                "{{H,S,Q}} roundtrip: input=\"{gates}\" → \"{decomp}\", dist={dist:.3e}");
        }
    }

    #[test]
    fn test_zzeta_roundtrip_random_hstq() {
        // Random strings over the full {H, S, T, Q} alphabet — verifies T
        // and Q can mix, and that the {QQ → T} rewrite in translate() doesn't
        // produce invalid intermediate states.
        let mut rng = rand::rng();
        for _ in 0..50 {
            let gates: String = (0..rng.random_range(50..=200))
                .map(|_| ['H', 'S', 'T', 'Q'][rng.random_range(0..4)])
                .collect();
            let target = gates_to_u2q(&gates);
            let decomp = BlochDecomposer.decompose(&target);
            let recovered = gates_to_u2q(&decomp);
            let dist = target.diamond_distance(&recovered);
            assert!(dist < 1e-6,
                "{{H,S,T,Q}} roundtrip: input=\"{gates}\" → \"{decomp}\", dist={dist:.3e}");
        }
    }

    // ── ZOmicron / Clifford+R_z(π/6) decomposer tests ────────────────────────

    type U2O = U2<ZOmicron>;

    fn gate_to_u2o(ch: char) -> U2O {
        match ch {
            'H' => U2O::h(),
            'S' => U2O::s(),
            'R' => ZOmicron::rz_pos_u2(),
            'Z' => U2O::z(),
            'X' => U2O::x(),
            'Y' => U2O::y(),
            _ => panic!("Invalid gate: {ch}"),
        }
    }

    fn gates_to_u2o(s: &str) -> U2O {
        s.chars().fold(U2O::eye(), |acc, ch| acc * gate_to_u2o(ch))
    }

    #[test]
    fn test_zomicron_identity_decomposes_to_empty() {
        let eye = U2O::eye();
        let decomp = BlochDecomposer.decompose(&eye);
        assert_eq!(decomp, "", "identity → \"{decomp}\"");
    }

    #[test]
    fn test_zomicron_r_alone_decomposes_to_r() {
        let r = ZOmicron::rz_pos_u2();
        let decomp = BlochDecomposer.decompose(&r);
        assert_eq!(decomp, "R", "R gate → \"{decomp}\"");
    }

    #[test]
    fn test_zomicron_rr_decomposes_correctly() {
        // RR = Rz(π/3); should round-trip
        let rr = ZOmicron::rz_pos_u2() * ZOmicron::rz_pos_u2();
        let decomp = BlochDecomposer.decompose(&rr);
        let recovered = gates_to_u2o(&decomp);
        let dist = rr.diamond_distance(&recovered);
        assert!(dist < 1e-7, "RR round-trip: \"{decomp}\", dist={dist:.3e}");
    }

    #[test]
    fn test_zomicron_hrh_decomposes_correctly() {
        // H·R·H = Rx(π/6); should produce a gate string that round-trips
        let hrh = U2O::h() * ZOmicron::rz_pos_u2() * U2O::h();
        let decomp = BlochDecomposer.decompose(&hrh);
        let recovered = gates_to_u2o(&decomp);
        let dist = hrh.diamond_distance(&recovered);
        assert!(dist < 1e-7, "HRH round-trip: \"{decomp}\", dist={dist:.3e}");
    }

    #[test]
    fn test_zomicron_clifford_only_inputs() {
        for input in ["H", "S", "X", "Y", "Z", "HH", "SH", "HS"] {
            let u = gates_to_u2o(input);
            let decomp = BlochDecomposer.decompose(&u);
            assert!(!decomp.contains('R'),
                "Clifford input \"{input}\" produced R-gate in \"{decomp}\"");
            let recovered = gates_to_u2o(&decomp);
            let dist = u.diamond_distance(&recovered);
            assert!(dist < 1e-9,
                "Clifford round-trip \"{input}\" → \"{decomp}\", dist={dist:.3e}");
        }
    }

    #[test]
    fn test_zomicron_roundtrip_random_hsr() {
        let mut rng = rand::rng();
        for _ in 0..30 {
            let gates: String = (0..rng.random_range(10..=60))
                .map(|_| ['H', 'S', 'R'][rng.random_range(0..3)])
                .collect();
            let target = gates_to_u2o(&gates);
            let decomp = BlochDecomposer.decompose(&target);
            let recovered = gates_to_u2o(&decomp);
            let dist = target.diamond_distance(&recovered);
            assert!(dist < 1e-6,
                "{{H,S,R}} roundtrip: \"{gates}\" → \"{decomp}\", dist={dist:.3e}");
        }
    }
}
