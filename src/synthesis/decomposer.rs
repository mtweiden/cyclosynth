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
use crate::rings::{ZOmega, ZZeta};
use crate::matrix::so3::{SO3T, SO3Q, SO3Ops};
#[cfg(test)]
use crate::matrix::so3::R4;
use crate::matrix::u2::{U2, U2T, U2Q, RingElem};
#[cfg(feature = "python")]
use crate::matrix::u2::{PyU2, U2Variant};
use crate::matrix::{rz_pos, rx_pos, ry_pos, rz_pos_q, rx_pos_q, ry_pos_q};
use crate::matrix::{rz_neg, rx_neg, ry_neg, rz_neg_q, rx_neg_q, ry_neg_q};
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
    let substituted = raw
        .replace('x', &format!("H{}H", magic))
        .replace('y', &format!("SH{}HSSS", magic))
        .replace('z', magic);
    simplify_gate_string(&substituted)
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
    // SO3T (Clifford+T) max_exp counts T-count, monotone-decreasing per
    // peel, so `max_steps = max_exp` would be tight. SO3Q (Clifford+√T)
    // in the √2-denom convention is non-monotone per peel (SO3(T) has
    // max_exp 1 but T = QQ needs 2 Rz peels: 1 → 2 → 0), so the bound is
    // generous; the loop early-exits via `best == 0` at a Clifford residual.
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

    // Why ZZeta needs `decompose_so3_canonical_q` and ZOmega doesn't: SO3T's
    // cos(nπ/4) has √2-exp ∈ {0,1,0,1,…} (monotone per Rz peel, so greedy
    // single-step argmin converges), but SO3Q's cos(nπ/8) has √2-exp ∈
    // {0,2,1,2,…} — a single Rz(π/8) peel can transiently *increase* max_exp,
    // sending greedy off-axis. The canonical-form algorithm tries all 9 peels
    // per step and picks the unique max_exp-reducing one instead.

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

    // ── P-a: per-peel Bloch-denominator drop ─
    //
    // Verifies the loop invariant of `decompose_so3_canonical_q` that the
    // slope-2 certificate floor (obligation
    // P-a) rests on. Two unit systems are in play:
    //
    //   * N  = `SO3Q::maximum_denominator_exponent()` — the √2-denominator
    //     exponent of the reduced Bloch matrix (the code's `max_exp`).
    //   * Nγ = the γ-denominator exponent, γ = √(2+√2) = 2cos(π/8) — the
    //     unit FGKM's Theorem 4.1 (arXiv:1501.04944) is stated in.
    //
    // Conversion: √2 = γ²·(√2−1) (unit), so a reduced entry w/√2^e has
    // γ-exponent 2e − min(v_γ(w), 1) and N = ⌈Nγ/2⌉ matrix-wide.
    //
    // FGKM Theorem 4.1 (n = 8): each canonical syllable R_p(a·π/8)
    // contributes exactly q_a to Nγ, with q = {a=1 (Q): 3, a=2 (T): 2,
    // a=3 (TQ): 3}. So the exact per-peel invariant is in γ-units:
    //
    //     Nγ drops by exactly q_a per peel.
    //
    // In the code's √2-units the drop is NOT a constant per syllable type:
    // a T-peel always drops N by 1, while a Q- or TQ-peel drops N by 2 when
    // the pre-peel Nγ is odd and by 1 when it is even (ceiling division).
    // The originally claimed table δ = {T:1, Q:2, TQ:3} is therefore false
    // for TQ (drop 3 is impossible) and only half-true for Q. Cost-per-peel
    // still dominates the drop: T = 1 ≥ 1, Q = 3 ≥ 2, TQ = 4 ≥ 2
    // (T-units), and the drops telescope to N(U), so cost ≥ N survives.

    /// γ-denominator exponent of a reduced SO3Q (max over non-zero entries).
    ///
    /// Entry w/√2^e (reduced: √2 ∤ w or e = 0) equals w·u^e/γ^{2e} with
    /// u = √2−1 a unit, so its γ-exponent is 2e − v_γ(w). After √2-reduction
    /// v_γ(w) ∈ {0, 1}: v_γ(w) ≥ 2 would mean γ² | w, i.e. √2 | w.
    fn so3q_gamma_exponent(m: &SO3Q) -> u32 {
        m.e.iter()
            .filter(|r| r.num != R4::ZERO)
            .map(|r| {
                if r.exp == 0 {
                    0
                } else {
                    let v = r.num.gamma_valuation();
                    assert!(v <= 1, "entry not √2-reduced: v_γ(num) = {v}, exp = {}", r.exp);
                    2 * r.exp - v
                }
            })
            .max()
            .unwrap_or(0)
    }

    /// One FGKM syllable R_p(a·π/8) as a reduced U2Q.
    fn fgkm_syllable(axis: usize, a: u32) -> U2Q {
        let mut d = U2Q::eye();
        for _ in 0..a {
            d = d * U2Q::q();
        }
        match axis {
            0 => (U2Q::h() * d * U2Q::h()).reduced(),
            1 => (U2Q::s() * U2Q::h() * d * U2Q::h() * U2Q::s().dagger()).reduced(),
            _ => d,
        }
    }

    /// Deterministic splitmix64 PRNG for reproducible random test inputs.
    struct Xs(u64);
    impl Xs {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        }
    }

    /// P-a companion: per-gate Bloch-denominator constants. With N
    /// subadditive under matrix products (Ratio exps add in Mul, Add takes
    /// max — same argument as the B2 U(2)-lde constants), N(T) = 1,
    /// N(Q) = 2, N(Clifford) = 0 give cost(W) = t + 3q ≥ t + 2q ≥ N(U)
    /// in T-units for EVERY gate word W, not just canonical ones. This is
    /// the exhaustive finite check those constants rest on.
    #[test]
    fn test_pa_per_gate_bloch_n_constants() {
        let n_of = |u: &U2Q| {
            let mut m = SO3Q::from_u2(u);
            m.reduce();
            m.maximum_denominator_exponent()
        };
        assert_eq!(n_of(&U2Q::t()), 1, "N(T) ≠ 1");
        assert_eq!(n_of(&U2Q::q()), 2, "N(Q) ≠ 2");
        // For the record: a whole TQ syllable also has N = 2, not 3 — the
        // single-syllable refutation of the claimed δ = 3.
        assert_eq!(n_of(&(U2Q::t() * U2Q::q())), 2, "N(TQ) ≠ 2");
        for (name, _) in CLIFFORD_TABLE_T {
            let g: U2Q = name.chars().fold(U2Q::eye(), |acc, ch| {
                acc * if ch == 'I' { U2Q::eye() } else { gate_to_u2q(ch) }
            });
            assert_eq!(n_of(&g), 0, "Clifford {name} has nonzero Bloch N");
        }
    }

    #[test]
    fn test_pa_peel_n_drop_exactness() {
        const N_WORDS: usize = 600;
        const M_MAX: u64 = 6;
        // FGKM Theorem 4.1 γ-exponent contributions per syllable amount a.
        const Q_GAMMA: [u32; 3] = [3, 2, 3];

        let candidates = canonical_candidates();
        let mut rng = Xs(0xBA5E_0FA1);
        // (a, √2-drop) → count, for the report table.
        let mut drop_hist: std::collections::BTreeMap<(u8, i64), usize> =
            std::collections::BTreeMap::new();
        let mut total_peels = 0usize;

        for word_idx in 0..N_WORDS {
            // Random reduced FGKM word: m syllables, consecutive axes differ.
            let m = 1 + (rng.next() % M_MAX) as u32;
            let mut u = U2Q::eye();
            let mut prev_axis = 3usize;
            for _ in 0..m {
                let mut axis = (rng.next() % 3) as usize;
                while axis == prev_axis {
                    axis = (rng.next() % 3) as usize;
                }
                prev_axis = axis;
                let a = 1 + (rng.next() % 3) as u32;
                u = (u * fgkm_syllable(axis, a)).reduced();
            }
            // Half the corpus: append a random Clifford word — must not
            // perturb N (Clifford Bloch matrices are signed permutations)
            // or any per-peel drop.
            if word_idx % 2 == 1 {
                for _ in 0..(rng.next() % 4) {
                    let g = match rng.next() % 2 {
                        0 => U2Q::h(),
                        _ => U2Q::s(),
                    };
                    u = (u * g).reduced();
                }
            }

            // Replicate decompose_so3_canonical_q's peel loop exactly
            // (same candidate order, same strict-< argmin), instrumented
            // with N and Nγ before/after each peel.
            let mut so3 = SO3Q::from_u2(&u);
            so3.reduce();
            let max_steps = (so3.max_exp() as usize) * 4 + 32;

            for _ in 0..max_steps {
                let n_before = so3.maximum_denominator_exponent();
                let ng_before = so3q_gamma_exponent(&so3);
                // Unit-system conversion invariant: N = ⌈Nγ/2⌉.
                assert_eq!(
                    n_before,
                    ng_before.div_ceil(2),
                    "N ≠ ⌈Nγ/2⌉ (N={n_before}, Nγ={ng_before})"
                );
                if n_before == 0 {
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
                let a = candidates[best_idx].a;
                so3 = best_so3;
                so3.reduce();

                let n_after = so3.maximum_denominator_exponent();
                let ng_after = so3q_gamma_exponent(&so3);
                total_peels += 1;

                // ── The exact invariant, γ-units: Nγ drops by exactly q_a. ──
                let gamma_drop = ng_before as i64 - ng_after as i64;
                assert_eq!(
                    gamma_drop,
                    Q_GAMMA[(a - 1) as usize] as i64,
                    "γ-drop ≠ q_a: a={a}, Nγ {ng_before}→{ng_after} (word {word_idx})"
                );

                // ── Derived √2-unit drop table. ──
                let drop = n_before as i64 - n_after as i64;
                let expected = match a {
                    2 => 1,                                    // T-peel: always 1
                    _ => if ng_before % 2 == 1 { 2 } else { 1 }, // Q/TQ: parity of Nγ
                };
                assert_eq!(
                    drop, expected,
                    "√2-drop off-table: a={a}, N {n_before}→{n_after}, Nγ_before={ng_before}"
                );

                // ── Cost-per-peel dominates the drop (T-units ×2 = half-units):
                //    cost(a=1)=7 HU, cost(a=2)=2 HU, cost(a=3)=9 HU ≥ 2·drop. ──
                let cost_hu = match a { 1 => 7, 2 => 2, _ => 9 };
                assert!(
                    2 * drop <= cost_hu,
                    "peel cost < N-drop: a={a}, drop={drop}"
                );

                *drop_hist.entry((a, drop)).or_insert(0) += 1;
                if so3.max_exp() == 0 {
                    break;
                }
            }
            assert_eq!(
                so3.max_exp(),
                0,
                "peel loop did not reach a Clifford residual (word {word_idx})"
            );
        }

        // Report the observed drop table (visible with --nocapture).
        eprintln!("P-a drop table over {total_peels} peels ({N_WORDS} words):");
        for ((a, drop), count) in &drop_hist {
            let name = match a { 1 => "Q ", 2 => "T ", _ => "TQ" };
            eprintln!("  syllable {name} (a={a}): N-drop {drop}  ×{count}");
        }
        // Every syllable type must actually occur for the table to be meaningful.
        for a in 1..=3u8 {
            assert!(
                drop_hist.keys().any(|(ka, _)| *ka == a),
                "corpus never peeled syllable a={a}"
            );
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
}
