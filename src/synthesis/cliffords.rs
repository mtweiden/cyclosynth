//! Clifford gate table for Clifford+T synthesis.
//!
//! All 24 single-qubit Cliffords are represented as SU(2) matrices
//! in the U2T parameterization (ZOmega numerators with denominator √2^k).
//!
//! Convention for gate strings: leftmost character = first gate applied
//! (rightmost factor in matrix product).

use crate::matrix::{U2T, U2Q};
use crate::rings::{ZOmega, ZZeta};

/// All 24 single-qubit Clifford gates as (gate_string, U2T) pairs.
///
/// These are SU(2) representatives (det = 1 up to global phase — the diamond
/// distance is phase-invariant so global phases are irrelevant for synthesis).
// Conversion from old (u1,u2,k) form: u11=u1, u12=-conj(u2), u21=u2, u22=conj(u1).
// For ZOmega: conj(a,b,c,d)=(a,-d,-c,-b), so -conj(a,b,c,d)=(-a,d,c,b).
pub static CLIFFORD_TABLE_T: &[(&str, U2T)] = &[
    //         u11                     u12                     u21                     u22                     k
    ("I",    U2T::new(ZOmega::from_i32( 1, 0, 0, 0), ZOmega::from_i32( 0, 0, 0, 0), ZOmega::from_i32( 0, 0, 0, 0), ZOmega::from_i32( 1, 0, 0, 0), 0)),
    ("H",    U2T::new(ZOmega::from_i32( 0, 0,-1, 0), ZOmega::from_i32( 0, 0,-1, 0), ZOmega::from_i32( 0, 0,-1, 0), ZOmega::from_i32( 0, 0, 1, 0), 1)),
    ("S",    U2T::new(ZOmega::from_i32( 0, 0, 0,-1), ZOmega::from_i32( 0, 0, 0, 0), ZOmega::from_i32( 0, 0, 0, 0), ZOmega::from_i32( 0, 1, 0, 0), 0)),
    ("X",    U2T::new(ZOmega::from_i32( 0, 0, 0, 0), ZOmega::from_i32( 0, 0,-1, 0), ZOmega::from_i32( 0, 0,-1, 0), ZOmega::from_i32( 0, 0, 0, 0), 0)),
    ("Y",    U2T::new(ZOmega::from_i32( 0, 0, 0, 0), ZOmega::from_i32(-1, 0, 0, 0), ZOmega::from_i32( 1, 0, 0, 0), ZOmega::from_i32( 0, 0, 0, 0), 0)),
    ("Z",    U2T::new(ZOmega::from_i32( 0, 0,-1, 0), ZOmega::from_i32( 0, 0, 0, 0), ZOmega::from_i32( 0, 0, 0, 0), ZOmega::from_i32( 0, 0, 1, 0), 0)),
    ("XH",   U2T::new(ZOmega::from_i32( 1, 0, 0, 0), ZOmega::from_i32( 1, 0, 0, 0), ZOmega::from_i32(-1, 0, 0, 0), ZOmega::from_i32( 1, 0, 0, 0), 1)),
    ("YH",   U2T::new(ZOmega::from_i32( 0, 0, 1, 0), ZOmega::from_i32( 0, 0,-1, 0), ZOmega::from_i32( 0, 0,-1, 0), ZOmega::from_i32( 0, 0,-1, 0), 1)),
    ("ZH",   U2T::new(ZOmega::from_i32( 1, 0, 0, 0), ZOmega::from_i32(-1, 0, 0, 0), ZOmega::from_i32( 1, 0, 0, 0), ZOmega::from_i32( 1, 0, 0, 0), 1)),
    ("XS",   U2T::new(ZOmega::from_i32( 0, 0, 0, 0), ZOmega::from_i32( 0, 1, 0, 0), ZOmega::from_i32( 0, 0, 0, 1), ZOmega::from_i32( 0, 0, 0, 0), 0)),
    ("YS",   U2T::new(ZOmega::from_i32( 0, 0, 0, 0), ZOmega::from_i32( 0, 0, 0,-1), ZOmega::from_i32( 0,-1, 0, 0), ZOmega::from_i32( 0, 0, 0, 0), 0)),
    ("ZS",   U2T::new(ZOmega::from_i32( 0, 1, 0, 0), ZOmega::from_i32( 0, 0, 0, 0), ZOmega::from_i32( 0, 0, 0, 0), ZOmega::from_i32( 0, 0, 0,-1), 0)),
    ("SH",   U2T::new(ZOmega::from_i32( 0, 1, 0, 0), ZOmega::from_i32( 0, 0, 0, 1), ZOmega::from_i32( 0, 1, 0, 0), ZOmega::from_i32( 0, 0, 0,-1), 1)),
    ("XSH",  U2T::new(ZOmega::from_i32( 0, 1, 0, 0), ZOmega::from_i32( 0, 0, 0,-1), ZOmega::from_i32( 0,-1, 0, 0), ZOmega::from_i32( 0, 0, 0,-1), 1)),
    ("YSH",  U2T::new(ZOmega::from_i32( 0, 0, 0, 1), ZOmega::from_i32( 0,-1, 0, 0), ZOmega::from_i32( 0, 0, 0,-1), ZOmega::from_i32( 0,-1, 0, 0), 1)),
    ("ZSH",  U2T::new(ZOmega::from_i32( 0, 0, 0,-1), ZOmega::from_i32( 0,-1, 0, 0), ZOmega::from_i32( 0, 0, 0,-1), ZOmega::from_i32( 0, 1, 0, 0), 1)),
    ("HS",   U2T::new(ZOmega::from_i32( 0, 1, 0, 0), ZOmega::from_i32( 0, 1, 0, 0), ZOmega::from_i32( 0, 0, 0, 1), ZOmega::from_i32( 0, 0, 0,-1), 1)),
    ("XHS",  U2T::new(ZOmega::from_i32( 0, 0, 0,-1), ZOmega::from_i32( 0, 0, 0,-1), ZOmega::from_i32( 0,-1, 0, 0), ZOmega::from_i32( 0, 1, 0, 0), 1)),
    ("YHS",  U2T::new(ZOmega::from_i32( 0, 1, 0, 0), ZOmega::from_i32( 0,-1, 0, 0), ZOmega::from_i32( 0, 0, 0,-1), ZOmega::from_i32( 0, 0, 0,-1), 1)),
    ("ZHS",  U2T::new(ZOmega::from_i32( 0, 0, 0,-1), ZOmega::from_i32( 0, 0, 0, 1), ZOmega::from_i32( 0, 1, 0, 0), ZOmega::from_i32( 0, 1, 0, 0), 1)),
    ("HSH",  U2T::new(ZOmega::from_i32( 1, 0, 0, 0), ZOmega::from_i32( 0, 0,-1, 0), ZOmega::from_i32( 0, 0,-1, 0), ZOmega::from_i32( 1, 0, 0, 0), 1)),
    ("XHSH", U2T::new(ZOmega::from_i32( 1, 0, 0, 0), ZOmega::from_i32( 0, 0, 1, 0), ZOmega::from_i32( 0, 0, 1, 0), ZOmega::from_i32( 1, 0, 0, 0), 1)),
    ("YHSH", U2T::new(ZOmega::from_i32( 0, 0, 1, 0), ZOmega::from_i32( 1, 0, 0, 0), ZOmega::from_i32(-1, 0, 0, 0), ZOmega::from_i32( 0, 0,-1, 0), 1)),
    ("ZHSH", U2T::new(ZOmega::from_i32( 0, 0, 1, 0), ZOmega::from_i32(-1, 0, 0, 0), ZOmega::from_i32( 1, 0, 0, 0), ZOmega::from_i32( 0, 0,-1, 0), 1)),
];

/// Indices into [`CLIFFORD_TABLE_T`] of the 8 Cliffords with lde 0 — the
/// subgroup ⟨S, X⟩ mod global phase: I, S, X, Y, Z, XS, YS, ZS. Exactly
/// the Cliffords whose U2T denominator exponent `k` is 0 (entries are
/// units of Z[ω], no 1/√2 factor).
///
/// Used by `build_l`'s right-coset dedup (stage 1 of
/// docs/plan_8d_prefix_rework.md): for lde-0 `C`, the D&C subproblems of
/// prefixes `U_L` and `U_L·C` are Q-isometric bijections with identical
/// total unitaries (`U_L·C·U_R = U_L·(C·U_R)`, and `C·U_R` lies on the
/// same norm shell with the same lde), so one representative per right
/// coset `U_L·⟨S,X⟩` suffices. The 24 Cliffords fall into 3 right cosets
/// of this subgroup, which is why `build_l`'s 24-fold Clifford
/// postmultiplication is ~2/3+ duplicated work that plain phase-dedup
/// misses.
pub static CLIFFORD_LDE0_IDX: [usize; 8] = [0, 2, 3, 4, 5, 9, 10, 11];

// Note on the T gate: T = Rz(π/4) = diag(e^{−iπ/8}, e^{iπ/8}). The phase
// e^{iπ/8} is *not* in Z[ω], so T as an exact unitary needs the larger Z[ζ]
// ring (or a base-8 DyadicComplex). For Clifford+T synthesis we never need T
// as a U2T object: the "T branch" applies T as a right-factor on the
// alignment vector rather than constructing an exact U2T value. In the
// output, the literal 'T' in a gate string denotes Rz(π/4) up to the global
// phase, which is the convention the synthesizer's tests and decomposer
// agree on.

/// Left-multiply a U2T by the Clifford C†, returning C†·target.
/// Used in the C-phase of synthesis to search over all 24 Clifford left-prefixes.
pub fn apply_clifford_dagger(clifford: &U2T, target: &U2T) -> U2T {
    clifford.dagger() * *target
}

/// Find the Clifford (by index into CLIFFORD_TABLE_T) that best matches target.
/// Returns the index and the diamond distance.
pub fn match_clifford(target: &U2T) -> Option<usize> {
    CLIFFORD_TABLE_T
        .iter()
        .enumerate()
        .min_by(|(_, (_, a)), (_, (_, b))| {
            let da = a.diamond_distance(target);
            let db = b.diamond_distance(target);
            da.partial_cmp(&db).unwrap()
        })
        .filter(|(_, (_, c))| c.diamond_distance(target) < 1e-6)
        .map(|(i, _)| i)
}

/// Placeholder: Clifford+√T table would live here.
/// For now, we embed T-gate Cliffords into ZZeta space.
pub fn clifford_table_q() -> Vec<(&'static str, U2Q)> {
    CLIFFORD_TABLE_T
        .iter()
        .map(|(name, u2t)| {
            let u11 = ZZeta::from_zomega(u2t.u11.a, u2t.u11.b, u2t.u11.c, u2t.u11.d);
            let u12 = ZZeta::from_zomega(u2t.u12.a, u2t.u12.b, u2t.u12.c, u2t.u12.d);
            let u21 = ZZeta::from_zomega(u2t.u21.a, u2t.u21.b, u2t.u21.c, u2t.u21.d);
            let u22 = ZZeta::from_zomega(u2t.u22.a, u2t.u22.b, u2t.u22.c, u2t.u22.d);
            (*name, U2Q::new(u11, u12, u21, u22, u2t.k))
        })
        .collect()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Every Clifford should be unitary: C·C† ≈ I.
    #[test]
    fn test_cliffords_are_unitary() {
        let id = U2T::new(ZOmega::ONE, ZOmega::ZERO, ZOmega::ZERO, ZOmega::ONE, 0);
        for (name, c) in CLIFFORD_TABLE_T {
            let cc_dag = *c * c.dagger();
            let dist = cc_dag.diamond_distance(&id);
            assert!(
                dist < 1e-9,
                "Clifford {name}: C·C† not identity, dist={dist}"
            );
        }
    }

    /// All 24 Cliffords must be distinct (pairwise distance > 0).
    #[test]
    fn test_cliffords_distinct() {
        for (i, (ni, ci)) in CLIFFORD_TABLE_T.iter().enumerate() {
            for (j, (nj, cj)) in CLIFFORD_TABLE_T.iter().enumerate() {
                if i == j {
                    continue;
                }
                let d = ci.diamond_distance(cj);
                assert!(d > 1e-6, "Clifford {ni} and {nj} are identical (dist={d})");
            }
        }
    }

    /// The lde-0 subgroup table: all 8 entries have k == 0, no other
    /// Clifford does, and the set is closed under multiplication mod
    /// global phase (i.e. it really is the subgroup ⟨S, X⟩).
    #[test]
    fn test_lde0_subgroup_table() {
        use crate::synthesis::cliffords::CLIFFORD_LDE0_IDX;
        // Exactness: the listed entries are the k == 0 entries.
        for (i, (name, c)) in CLIFFORD_TABLE_T.iter().enumerate() {
            let in_table = CLIFFORD_LDE0_IDX.contains(&i);
            assert_eq!(
                c.k == 0,
                in_table,
                "Clifford {name} (idx {i}): k={} but lde0-table membership={in_table}",
                c.k
            );
        }
        // Closure mod phase: a·b matches some subgroup element.
        for &i in &CLIFFORD_LDE0_IDX {
            for &j in &CLIFFORD_LDE0_IDX {
                let prod = CLIFFORD_TABLE_T[i].1 * CLIFFORD_TABLE_T[j].1;
                let closed = CLIFFORD_LDE0_IDX.iter().any(|&m| {
                    prod.diamond_distance(&CLIFFORD_TABLE_T[m].1) < 1e-9
                });
                assert!(
                    closed,
                    "subgroup not closed: {} * {}",
                    CLIFFORD_TABLE_T[i].0, CLIFFORD_TABLE_T[j].0
                );
            }
        }
    }

    /// match_clifford should recover each Clifford by its index.
    #[test]
    fn test_match_clifford_round_trip() {
        for (i, (name, c)) in CLIFFORD_TABLE_T.iter().enumerate() {
            let idx = match_clifford(c);
            assert_eq!(
                idx,
                Some(i),
                "match_clifford failed for {name}: got {idx:?}, expected Some({i})"
            );
        }
    }
}
