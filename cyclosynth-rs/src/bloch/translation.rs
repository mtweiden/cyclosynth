//! Gate string simplification: translate discrete rotations into Clifford+T/Q.

use pyo3::prelude::*;

/// Translate a decomposition in {x, y, z} into Clifford+T or Clifford+Q.
#[pyfunction]
pub fn translate_decomposition(gates: &str, magic_gate: &str) -> String {
    let mut translation = gates
        .replace('x', &format!("H{magic_gate}H"))
        .replace('y', &format!("SH{magic_gate}HZS"))
        .replace('z', magic_gate);

    let mut last = String::new();
    while translation != last {
        last = translation.clone();
        // Commutations
        translation = translation.replace("SZ", "ZS");
        translation = translation.replace("TZ", "ZT");
        translation = translation.replace("QZ", "ZQ");
        translation = translation.replace("TS", "ST");
        translation = translation.replace("QS", "SQ");
        translation = translation.replace("QT", "TQ");
        // Combinations
        translation = translation.replace("QQ", "T");
        translation = translation.replace("TT", "S");
        translation = translation.replace("SS", "Z");
        // Cancellations
        translation = translation.replace("HH", "");
        translation = translation.replace("XX", "");
        translation = translation.replace("YY", "");
        translation = translation.replace("ZZ", "");
        translation = translation.replace("I", "");
    }
    translation
}
