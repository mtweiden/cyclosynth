//! Clifford gate table for Clifford+T synthesis.
//!
//! All 24 single-qubit Cliffords are represented as SU(2) matrices
//! in the U2T parameterization (ZOmega numerators with denominator √2^k).
//!
//! Convention for gate strings: leftmost character = first gate applied
//! (rightmost factor in matrix product).

use crate::matrix::U2T;
use crate::rings::ZOmega;

/// All 24 single-qubit Clifford gates as (gate_string, U2T) pairs, SU(2)
/// representatives (global phase is irrelevant — diamond distance is
/// phase-invariant).
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

/// Indices into [`CLIFFORD_TABLE_T`] of the 8 lde-0 Cliffords — the subgroup
/// ⟨S, X⟩ mod phase (units of Z[ω], no 1/√2). `build_ma_prefix_set`'s
/// right-coset dedup keeps one rep per coset `U_L·⟨S,X⟩` (the 24 Cliffords fall
/// into 3 cosets), since `U_L·C·U_R = U_L·(C·U_R)` shares shell and lde.
pub static CLIFFORD_LDE0_IDX: [usize; 8] = [0, 2, 3, 4, 5, 9, 10, 11];

// T = Rz(π/4) has phase e^{iπ/8} ∉ Z[ω], so it can't be a U2T value. We
// never need it as one: the "T branch" applies T as a right-factor on the
// alignment vector, and the literal 'T' in a gate string denotes Rz(π/4)
// up to global phase.

/// Find the Clifford (by index into CLIFFORD_TABLE_T) within 1e-6 diamond
/// distance of `target`. Returns its index, or `None` if none match.
#[cfg(test)]
pub(crate) fn match_clifford(target: &U2T) -> Option<usize> {
    CLIFFORD_TABLE_T
        .iter()
        .enumerate()
        .min_by(|(_, (_, a)), (_, (_, b))| {
            let da = a.diamond_distance(target);
            let db = b.diamond_distance(target);
            da.total_cmp(&db)
        })
        .filter(|(_, (_, c))| c.diamond_distance(target) < 1e-6)
        .map(|(i, _)| i)
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
