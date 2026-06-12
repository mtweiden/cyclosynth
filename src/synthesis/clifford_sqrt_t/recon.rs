//! Lattice solution → U2Q reconstruction and det-phase handling.

use super::*;

// ─── Solution → U2Q reconstruction (Z[ζ_16] analog of solution_to_u2t) ───────

/// Build `U2Q` from a 16-element solution and denominator exponent.
///
/// Convention: `sol = [u_1.a, …, u_1.h, u_2.a, …, u_2.h]` with
/// `U = [[u_1, −u_2*], [u_2, u_1*]] / √(2^k)` (SU(2) form, det = 1).
pub fn solution_to_u2q(sol: &[i64; 16], k: u32) -> U2Q {
    solution_to_u2q_with_det_phase(sol, k, 0)
}

/// `ζ_16^d` as a `ZZeta` element, for `d` in `0..16`. `ζ_16^8 = −1`, so
/// `ζ_16^(d+8) = −ζ_16^d`.
pub(crate) fn zeta_16_pow(d: u32) -> ZZeta {
    let d = d % 16;
    if d < 8 {
        let mut c = [0i32; 8];
        c[d as usize] = 1;
        ZZeta::from_i32(c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7])
    } else {
        -zeta_16_pow(d - 8)
    }
}

/// Build a Clifford+√T `U2Q` from a 16-element solution at lde `k` with
/// **det-phase parameter** `det_phase` in `0..16`.
///
/// The reconstructed `U2Q` has determinant `ζ_16^det_phase`. Convention:
///
/// ```text
/// U = [[u_1, ζ_16^d · (−u_2*)], [u_2, ζ_16^d · u_1*]] / √(2^k)
/// ```
///
/// For `d = 0` this matches [`solution_to_u2q`] (SU(2) form). For `d ≠ 0`
/// the second column is rotated by `ζ_16^d`, making `U` reach Clifford+√T
/// products with non-unit determinant (e.g. circuits containing an odd
/// number of Q gates).
pub fn solution_to_u2q_with_det_phase(sol: &[i64; 16], k: u32, det_phase: u32) -> U2Q {
    let mk = |s: &[i64]| ZZeta::new(
        Int::from_i64(s[0]), Int::from_i64(s[1]), Int::from_i64(s[2]), Int::from_i64(s[3]),
        Int::from_i64(s[4]), Int::from_i64(s[5]), Int::from_i64(s[6]), Int::from_i64(s[7]),
    );
    let u1 = mk(&sol[0..8]);
    let u2 = mk(&sol[8..16]);
    let phase = zeta_16_pow(det_phase);
    U2Q::new(u1, phase * (-u2.conj()), u2, phase * u1.conj(), k)
}

/// Rotate `target` by a global phase so its det lands exactly on the
/// nearest ζ₁₆ power. Lossless (diamond distance is phase-invariant) —
/// but without it a U(2) input whose det is not a 16th root carries a
/// residual phase no completion can absorb, and the search burns to
/// max_lde finding nothing.
pub fn project_det_to_zeta_coset(target: &Mat2) -> Mat2 {
    let det = target[0][0] * target[1][1] - target[0][1] * target[1][0];
    let d = det_phase_of(target) as f64;
    let mut residual = det.arg() - d * PI / 8.0;
    while residual > PI {
        residual -= 2.0 * PI;
    }
    while residual <= -PI {
        residual += 2.0 * PI;
    }
    let g = Complex64::from_polar(1.0, -residual / 2.0);
    [
        [target[0][0] * g, target[0][1] * g],
        [target[1][0] * g, target[1][1] * g],
    ]
}

/// The det-phase `d ∈ {0..15}` of V: the integer with `ζ_16^d` closest
/// to `det(V)` on the unit circle (16-valued analog of Z[ω]'s
/// `det_zeta_parity`).
pub fn det_phase_of(target: &Mat2) -> u32 {
    let det = target[0][0] * target[1][1] - target[0][1] * target[1][0];
    let arg = det.arg();
    let d_float = arg * 16.0 / (2.0 * PI);
    let d_int = d_float.round() as i32;
    (((d_int % 16) + 16) % 16) as u32
}

/// Column-1 of `target` as a 4-element real vector
/// `(Re V_{00}, Im V_{00}, Re V_{10}, Im V_{10})`. Used as the SU(2)-style
/// alignment direction `v` for the lattice search.
///
/// **Differs from 8D's `unitary_to_uv`**: that function divides by `√det`
/// to project to SU(2) because `solution_to_u2t` produces a fixed SU(2)
/// form. Here we leave the column unprojected and absorb the det-phase
/// mismatch via [`solution_to_u2q_with_det_phase`]'s `d` parameter (set to
/// [`det_phase_of`]`(target)` at the call site). Column 1 of any 2×2
/// unitary is unit-norm by construction, so no further normalization is
/// needed.
pub fn unitary_to_uv_zeta(target: &Mat2) -> [f64; 4] {
    [target[0][0].re, target[0][0].im, target[1][0].re, target[1][0].im]
}

