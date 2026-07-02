//! Bloch-sphere decomposition of exact Clifford+T / Clifford+√T unitaries into gate strings.
//!
//! Peels rotations off SO3(target) until a Clifford residual remains, then translates.
//!   - `U2<ZOmega>` (`U2T`) → `decompose_so3` (greedy single-step peel, Clifford+T)
//!   - `U2<ZZeta>`  (`U2Q`) → `decompose_so3_canonical_q` (canonical form, arXiv:1501.04944 §4, Clifford+√T)

use std::fmt::Debug;
use std::ops::{Mul, Sub};
use crate::rings::{ZOmega, ZZeta};
use crate::matrix::so3::{SO3T, SO3Q, SO3Ops};
#[cfg(test)]
use crate::matrix::so3::R4;
use crate::matrix::u2::{U2, U2T, U2Q, RingElem};
use crate::matrix::{rz_neg, rx_neg, ry_neg, rz_neg_q, rx_neg_q, ry_neg_q};
use crate::synthesis::cliffords::CLIFFORD_TABLE_T;

// ─── GateRing trait ───────────────────────────────────────────────────────────

/// A ring type carrying its gate-set context (SO3 rep, generators, Clifford table)
/// for Bloch decomposition. `ZOmega` → Clifford+T, `ZZeta` → Clifford+√T.
pub trait GateRing: RingElem + Mul<Output = Self> + Sub<Output = Self>{
    /// SO(3) matrix type for this gate set (`SO3T` or `SO3Q`).
    type SO3: SO3Ops + Debug;

    /// Convert a U2 matrix to its exact SO(3) representation.
    fn so3_from_u2(u: &U2<Self>) -> Self::SO3;

    /// Negative Rz/Rx/Ry SO(3) rotation generators (step = π/4 for T, π/8 for √T).
    fn rz_neg() -> Self::SO3;
    fn rx_neg() -> Self::SO3;
    fn ry_neg() -> Self::SO3;

    /// Positive Rx/Ry/Rz rotation as U2 matrices (used to track output gate product).
    fn rx_pos_u2() -> U2<Self>;
    fn ry_pos_u2() -> U2<Self>;
    fn rz_pos_u2() -> U2<Self>;

    /// Find the Clifford label whose gate-primitive U2 matches `u`, by argmin
    /// diamond distance over the 24 entries (not first-within-tolerance: the
    /// noise floor grows with the U2's denominator exponent).
    fn identify_clifford_from_u2(u: &U2<Self>) -> Option<&'static str> {
        CLIFFORD_TABLE_T
            .iter()
            .map(|(name, _)| {
                let gate_u: U2<Self> = name.chars().fold(U2::<Self>::eye(), |acc, ch| {
                    acc * match ch {
                        'H' => U2::<Self>::h(),
                        'S' => U2::<Self>::s(),
                        'X' => U2::<Self>::x(),
                        'Y' => U2::<Self>::y(),
                        'Z' => U2::<Self>::z(),
                        _ => U2::<Self>::eye(),
                    }
                });
                (*name, gate_u.diamond_distance(u))
            })
            .min_by(|(_, a), (_, b)| a.total_cmp(b))
            .filter(|(_, d)| *d < 1e-3)
            .map(|(name, _)| name)
    }

    /// Name of the magic gate: `"T"` or `"Q"`.
    fn magic_gate_name() -> &'static str;

    /// Ring-specific entry point for the Bloch-sphere decomposition.
    fn decompose_target(target: &U2<Self>) -> String;
}

// ─── GateRing for ZOmega (Clifford+T) ────────────────────────────────────────

impl GateRing for ZOmega {
    type SO3 = SO3T;

    fn so3_from_u2(u: &U2<Self>) -> SO3T { SO3T::from_u2(u) }
    fn rz_neg() -> SO3T { rz_neg() }
    fn rx_neg() -> SO3T { rx_neg() }
    fn ry_neg() -> SO3T { ry_neg() }

    fn rx_pos_u2() -> U2T { U2T::h() * U2T::t() * U2T::h() }
    fn ry_pos_u2() -> U2T {
        // Ry(π/4) = S · Rx(π/4) · S†
        U2T::s() * U2T::h() * U2T::t() * U2T::h() * U2T::s().dagger()
    }
    fn rz_pos_u2() -> U2T { U2T::t() }

    fn magic_gate_name() -> &'static str { "T" }

    fn decompose_target(target: &U2<Self>) -> String {
        decompose_so3::<Self>(target)
    }
}

// ─── GateRing for ZZeta (Clifford+√T) ────────────────────────────────────────

impl GateRing for ZZeta {
    type SO3 = SO3Q;

    fn so3_from_u2(u: &U2<Self>) -> SO3Q { SO3Q::from_u2(u) }
    fn rz_neg() -> SO3Q { rz_neg_q() }
    fn rx_neg() -> SO3Q { rx_neg_q() }
    fn ry_neg() -> SO3Q { ry_neg_q() }

    fn rx_pos_u2() -> U2Q { U2Q::h() * U2Q::q() * U2Q::h() }
    fn ry_pos_u2() -> U2Q {
        // Ry(π/4) = S · Rx(π/4) · S†
        U2Q::s() * U2Q::h() * U2Q::q() * U2Q::h() * U2Q::s().dagger()
    }
    fn rz_pos_u2() -> U2Q { U2Q::q() }

    fn magic_gate_name() -> &'static str { "Q" }

    fn decompose_target(target: &U2<Self>) -> String {
        decompose_so3_canonical_q(target)
    }
}

// ─── BlochDecomposer ─────────────────────────────────────────────────────────

/// Generic (stateless) Bloch-sphere decomposer; `decompose` is generic over `R: GateRing`.
#[derive(Default)]
pub struct BlochDecomposer;

impl BlochDecomposer {
    /// Decompose an exact unitary into a gate string.
    ///
    /// - `U2<ZOmega>` (`U2T`) → output in {H, S, T, X, Y, Z}
    /// - `U2<ZZeta>`  (`U2Q`) → output in {H, S, Q, X, Y, Z}
    pub fn decompose<R: GateRing>(&self, target: &U2<R>) -> String {
        R::decompose_target(target)
    }
}

// ─── Gate string translation ──────────────────────────────────────────────────

/// Translate a raw `{x,y,z,Clifford}` decomposition string into a gate string,
/// substituting 'x'→H·magic·H, 'y'→S·H·magic·H·S³, 'z'→magic, then simplifying.
/// `magic` is `"T"` (Clifford+T) or `"Q"` (Clifford+√T).
fn translate(raw: &str, magic: &str) -> String {
    let substituted = raw
        .replace('x', &format!("H{}H", magic))
        .replace('y', &format!("SH{}HSSS", magic))
        .replace('z', magic);
    simplify_gate_string(&substituted)
}

/// Single-step greedy peel + Clifford identification (ZOmega / Clifford+T).
/// Each peel reduces `max_exp` by 1 (cos(nπ/4) has √2-exp ∈ {0,1,0,1,…}), so
/// argmin over the three axes is correct; output is canonicalized per syllable.
fn decompose_so3<R: GateRing>(target: &U2<R>) -> String {
    let rz = R::rz_neg();   // negative generators guarantee progress
    let rx = R::rx_neg();
    let ry = R::ry_neg();
    let rx_u2 = R::rx_pos_u2();
    let ry_u2 = R::ry_pos_u2();
    let rz_u2 = R::rz_pos_u2();

    let mut raw = String::new();
    let mut so3 = R::so3_from_u2(target);  // start from SO3(target), not SO3(target†)
    let mut p_output_u2 = U2::<R>::eye();
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
            p_output_u2 = p_output_u2 * rx_u2;
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

    // gate_c = p_output_u2† · target; raw + clifford_suffix evaluates to target.
    let gate_c = p_output_u2.dagger() * *target;
    let clifford_suffix = R::identify_clifford_from_u2(&gate_c)
        .filter(|&name| name != "I")
        .unwrap_or("");
    let full_raw = format!("{raw}{clifford_suffix}");
    canonicalize_syllables(&translate(&full_raw, R::magic_gate_name()))
}

// ─── Canonical-form decomposition for ZZeta (Clifford+√T) ────────────────────
//
// Canonical form from arXiv:1501.04944 §4. Greedy single-step peeling fails
// here (cos(nπ/8) √2-exp is non-monotone in a), so each step tries all 9 peels
// R_p(a·π/8), p ∈ {x,y,z}, a ∈ {1,2,3}, and takes the argmin in `max_exp`.

/// The 9 (p ∈ {x,y,z}, a ∈ {1,2,3}) candidate generators: negative SO3 (peeled
/// from the left of the residual) + positive U2 (right-accumulated into output).
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

// axis_idx enumerates the 3 rotation axes — fits u8.
#[allow(clippy::cast_possible_truncation)]
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
    canonicalize_syllables(&simplify_gate_string(&full))
}

/// Minimal gate string for a syllable by net √T-power `k` (mod 16), indexed
/// `SYLLABLE_FORMS[k]`. `Q=√T`, `T=Q²`, `S=Q⁴`, `Z=Q⁸`; lowercase `q,t,s` are
/// the adjoints. Matches the per-syllable cost charged by
/// [`crate::synthesis::clifford_sqrt_t::gates_cost`].
const SYLLABLE_FORMS: [&str; 16] = [
    "", "Q", "T", "qS", "S", "QS", "TS", "qZ", "Z", "QZ", "TZ", "qs", "s", "Qs", "t", "q",
];

/// Net √T-power of a single diagonal gate (`0` for non-diagonal / identity).
fn q_power(c: char) -> i32 {
    match c {
        'Q' => 1,
        'T' => 2,
        'S' => 4,
        'Z' => 8,
        'q' => -1,
        't' => -2,
        's' => -4,
        _ => 0,
    }
}

/// Rewrite each syllable (maximal run between off-diagonal `H`/`X`/`Y`) into its
/// minimal [`SYLLABLE_FORMS`] form. Net √T-power is preserved mod 16, so the
/// result is diamond-distance-identical to the input.
fn canonicalize_syllables(s: &str) -> String {
    let mut out = String::new();
    let mut k: i32 = 0;
    for c in s.chars() {
        if c == 'H' || c == 'X' || c == 'Y' {
            out.push_str(SYLLABLE_FORMS[k.rem_euclid(16) as usize]);
            out.push(c);
            k = 0;
        } else {
            k += q_power(c);
        }
    }
    out.push_str(SYLLABLE_FORMS[k.rem_euclid(16) as usize]);
    out
}

/// Translate a single (axis, a) peel into a literal `{H,S,Q,X,Y,Z}` gate-string fragment.
fn canonical_segment_string(axis: u8, a: u8) -> String {
    let q_run: String = "Q".repeat(a as usize);
    match axis {
        0 => format!("H{q_run}H"),       // x: R_x(a·π/8) = H · Q^a · H
        1 => format!("SH{q_run}HSSS"),   // y: R_y(a·π/8) = S · H · Q^a · H · S†
        2 => q_run,                       // z: R_z(a·π/8) = Q^a
        _ => unreachable!("axis is 0/1/2 by construction"),
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
            // Adjoints emitted by canonicalize_syllables (e.g. T·S·Z → t).
            's' => { U2T::s().dagger() }
            't' => { U2T::t().dagger() }
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
            // Adjoints emitted by canonicalize_syllables.
            's' => U2Q::s().dagger(),
            't' => U2Q::t().dagger(),
            'q' => U2Q::q().dagger(),
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

    /// Clifford+T syllables collapse to their minimal cost-class form, using
    /// the adjoints `t`/`s` when shorter.
    #[test]
    fn test_clifford_t_syllable_daggers() {
        // (input, expected minimal form).
        for (gates, want) in [("TSZ", "t"), ("TTTTTT", "s"), ("TTTTT", "TZ")] {
            let decomp = decompose_from_gates_t(gates);
            assert_eq!(decomp, want, "decompose(\"{gates}\")");
            // No `Q`/`q` ever leaks into a Clifford+T string.
            assert!(
                !decomp.contains('Q') && !decomp.contains('q'),
                "Clifford+T decomp \"{decomp}\" must not contain √T gates",
            );
            // And it is the same unitary.
            let dist = gates_to_u2t(gates).diamond_distance(&gates_to_u2t(&decomp));
            assert!(dist < 1e-7, "\"{gates}\" → \"{decomp}\": dist={dist:.3e}");
        }
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

    // ── Translation tests ─────────────────────────────────────────────────────

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

    // P-a: per-peel Bloch-denominator drop. N = code's √2-exp `max_exp`;
    // Nγ = γ-exp (γ = 2cos(π/8)) that FGKM Theorem 4.1 is stated in. Verifies
    // Nγ drops by exactly q_a per peel and cost-per-peel ≥ the derived N-drop.

    /// γ-denominator exponent of a reduced SO3Q (max over non-zero entries).
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

    /// P-a companion: per-gate Bloch-denominator constants N(T)=1, N(Q)=2,
    /// N(Clifford)=0 (the exhaustive finite check cost(W) ≥ N(U) rests on).
    #[test]
    fn test_pa_per_gate_bloch_n_constants() {
        let n_of = |u: &U2Q| {
            let mut m = SO3Q::from_u2(u);
            m.reduce();
            m.maximum_denominator_exponent()
        };
        assert_eq!(n_of(&U2Q::t()), 1, "N(T) ≠ 1");
        assert_eq!(n_of(&U2Q::q()), 2, "N(Q) ≠ 2");
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
            // Half the corpus: append a random Clifford word (must not perturb N).
            if word_idx % 2 == 1 {
                for _ in 0..(rng.next() % 4) {
                    let g = match rng.next() % 2 {
                        0 => U2Q::h(),
                        _ => U2Q::s(),
                    };
                    u = (u * g).reduced();
                }
            }

            // Replicate decompose_so3_canonical_q's peel loop, instrumented with N/Nγ.
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
                let gamma_drop = i64::from(ng_before) - i64::from(ng_after);
                assert_eq!(
                    gamma_drop,
                    i64::from(Q_GAMMA[(a - 1) as usize]),
                    "γ-drop ≠ q_a: a={a}, Nγ {ng_before}→{ng_after} (word {word_idx})"
                );

                // ── Derived √2-unit drop table. ──
                let drop = i64::from(n_before) - i64::from(n_after);
                let expected = match a {
                    2 => 1,                                    // T-peel: always 1
                    _ => if ng_before % 2 == 1 { 2 } else { 1 }, // Q/TQ: parity of Nγ
                };
                assert_eq!(
                    drop, expected,
                    "√2-drop off-table: a={a}, N {n_before}→{n_after}, Nγ_before={ng_before}"
                );

                // ── Cost-per-peel (half-units) dominates the drop: 2·drop ≤ cost. ──
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
