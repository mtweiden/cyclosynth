//! Bloch sphere decomposition of exactly-implementable Clifford+T (or Clifford+√T) unitaries.
//!
//! The ring type `R` of the input `U2<R>` determines the gate set automatically:
//!   - `U2<ZOmega>` (= `U2T`) → Clifford+T, SO3 over Z[√2], step = Rz(π/4)
//!   - `U2<ZZeta>`  (= `U2Q`) → Clifford+√T, SO3 over Z[γ],  step = Rz(π/8)
//!
//! Algorithm:
//!   1. Convert target unitary → SO(3) matrix (exact ring arithmetic, no floats).
//!   2. Repeatedly left-multiply by Rx/Ry/Rz(+step) to reduce max denominator exponent.
//!   3. Identify the Clifford remainder by exact SO3 equality against the 24-element table.
//!   4. Translate the raw {x,y,z,Clifford} string to a concrete gate string.

use std::fmt::Debug;
use std::ops::{Mul, Sub};
use crate::rings::{ZOmega, ZZeta};
use crate::rings::types::INT_ZERO;
use crate::matrix::so3::{SO3, SO3T, SO3Q, SO3Ops, R2, R4, Ratio};
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

    /// Find the Clifford gate whose SO3 equals `residual` exactly.
    /// (Used for debug; correctness in synthesis uses `identify_clifford_from_u2`.)
    fn identify_clifford(residual: &Self::SO3) -> Option<&'static str>;

    /// Find the Clifford gate label whose gate-primitive U2 matches `u` by diamond distance.
    /// This is the correct identification for synthesis (avoids S-convention mismatch).
    fn identify_clifford_from_u2(u: &U2<Self>) -> Option<&'static str>;

    /// Name of the magic gate: `"T"` or `"Q"`.
    fn magic_gate_name() -> &'static str;
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
        CLIFFORD_TABLE_T.iter().find(|(name, _)| {
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
            gate_u.diamond_distance(u) < 1e-9
        }).map(|(name, _)| *name)
    }

    fn magic_gate_name() -> &'static str { "T" }
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
        CLIFFORD_TABLE_T.iter().find(|(name, _)| {
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
            gate_u.diamond_distance(u) < 1e-9
        }).map(|(name, _)| *name)
    }

    fn magic_gate_name() -> &'static str { "Q" }
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
        decompose_so3::<R>(target)
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
}
