//! Brute-force search and y-vector helpers for the Z[ζ_16] / Clifford+√T
//! flow. Mirrors the role of [`super::search`] for the Z[ω] / Clifford+T
//! flow: y-vector construction (`compute_align_vec_zeta`, `uv_to_xy_zeta`)
//! plus a brute-force enumerator (`phase1_brute`) used as a correctness
//! oracle for the lattice pipeline in [`super::lattice_zeta`].
//!
//! Cost of `phase1_brute` is exponential in `k` (the shell at k=4 has
//! ~5·10⁸ points); useful for `k ≤ 4` for full enumeration. The L²-LLL +
//! Schnorr-Euchner port is in [`super::lattice_zeta`].

use std::f64::consts::PI;

use super::lattice_zeta::se::bilinear_forms;

// ─── y-vector helpers ────────────────────────────────────────────────────────

/// Convert a 4-element direction `v = (Re V_{11}, Im V_{11}, Re V_{21},
/// Im V_{21})` (extracted from the SU(2) form of the target) into a 16D
/// lattice-coord y vector. Analog of [`super::search::compute_align_vec`]
/// for Z[ω].
///
/// Construction: `y_lattice = Σ_full^T · v_padded`, where `v_padded` has
/// `v` placed at the σ_1 indices `{0, 1, 8, 9}` of the per-element layout
/// and zero elsewhere. By the orthogonality of Σ rows
/// (`Σ_full Σ_full^T = 4·I_16`), `Σ_full · y_lattice = 4 · v_padded`
/// — i.e. the Σ-image of y is `4·target` on σ_1, zero on σ_5/σ_9/σ_13.
///
/// Components: for `j ∈ {0..7}`,
///   `y[j]   = cos(jπ/8)·v[0] + sin(jπ/8)·v[1]`  (u_1 block)
///   `y[8+j] = cos(jπ/8)·v[2] + sin(jπ/8)·v[3]`  (u_2 block)
pub fn compute_align_vec_zeta(v: [f64; 4]) -> [f64; 16] {
    let mut y = [0.0f64; 16];
    for j in 0..8 {
        let theta = (j as f64) * PI / 8.0;
        let c = theta.cos();
        let s = theta.sin();
        y[j] = c * v[0] + s * v[1];
        y[8 + j] = c * v[2] + s * v[3];
    }
    y
}

/// Scale a 4-element alignment direction `v` to the 16-element y vector
/// used by the Z[ζ_16] lattice pipeline. Convention chosen so that
/// `Σ_full · y = √(2^k) · v_padded` (target × √(2^k) on σ_1, zero on
/// σ_5/9/13), consistent with the Z[ω] flow's scale convention.
pub fn uv_to_xy_zeta(v: [f64; 4], k: u32) -> [f64; 16] {
    let scale = 2.0f64.powf(k as f64 / 2.0) / 4.0;
    let raw = compute_align_vec_zeta(v);
    std::array::from_fn(|i| raw[i] * scale)
}

/// MPFR-precision variant of [`uv_to_xy_zeta`]. Caller provides an
/// MPFR `v` (of any precision) and gets back y at the same precision.
/// The `prec` argument matches the precision of the returned RFloats.
///
/// **Deep-ε radial-norm contract** (docs/w_precision_audit_notes.md):
/// the SE cap's radial window is keyed multiplicatively to ‖y‖, and at
/// ε = 1e-8 the window is only ε²/2 = 5e-17 wide RELATIVE — below one
/// f64 ulp. Two ~1e-16 norm-error channels used to live here and
/// displaced the cap by up to ±1.5 window-widths (the 1e-8 find/miss
/// flicker):
///   1. f64 `theta.cos()/sin()` lifted into MPFR (≤1 ulp each — the
///      "single-rounding is exact" comment was true of the conversion
///      and irrelevant to the chain);
///   2. `v` carries the target column's own f64 quantization defect
///      |v| = 1 + ν, ν ~ 1e-16, which the true acceptance window
///      cancels out (threshold and |t| scale together) but the cap
///      construction did not.
/// Fix: MPFR cos/sin tables, then rescale so ‖y‖ = 2^(k/2)/2 = ρ
/// EXACTLY (to `prec`). Norm errors of any upstream origin become pure
/// direction errors, which enter the cap radius only via the ρε
/// tangential arm (~1e-24·ρ — harmless). Residual η ~ 2^−prec.
pub fn uv_to_xy_zeta_mpfr(v: &[rug::Float; 4], k: u32, prec: u32) -> [rug::Float; 16] {
    use rug::Float as RFloat;
    // cos/sin(jπ/8) tables at `prec` (MPFR Pi — no f64 trig roundings).
    let pi = RFloat::with_val(prec, rug::float::Constant::Pi);
    let mut raw: [RFloat; 16] = std::array::from_fn(|_| RFloat::with_val(prec, 0.0));
    let mut raw_norm_sq = RFloat::with_val(prec, 0.0);
    for j in 0..8 {
        let theta = RFloat::with_val(prec, &pi * (j as u32)) / 8u32;
        let c = theta.clone().cos();
        let s = theta.sin();
        // raw[j] = c·v[0] + s·v[1]
        let cv0 = RFloat::with_val(prec, &c * &v[0]);
        let sv1 = RFloat::with_val(prec, &s * &v[1]);
        raw[j] = RFloat::with_val(prec, &cv0 + &sv1);
        // raw[8+j] = c·v[2] + s·v[3]
        let cv2 = RFloat::with_val(prec, &c * &v[2]);
        let sv3 = RFloat::with_val(prec, &s * &v[3]);
        raw[8 + j] = RFloat::with_val(prec, &cv2 + &sv3);
        raw_norm_sq += RFloat::with_val(prec, &raw[j] * &raw[j]);
        raw_norm_sq += RFloat::with_val(prec, &raw[8 + j] * &raw[8 + j]);
    }
    if raw_norm_sq.is_zero() {
        // Degenerate v: keep the legacy convention (zero y → SE walk
        // returns empty downstream).
        return raw;
    }
    // ρ = 2^(k/2) / 2 in MPFR (exact shift; √2 at prec for odd k).
    let rho = {
        let mut s = RFloat::with_val(prec, 1.0);
        s <<= (k / 2) as i32;
        if k % 2 == 1 {
            let sqrt2 = RFloat::with_val(prec, 2.0).sqrt();
            s *= &sqrt2;
        }
        s >>= 1;
        s
    };
    // y = raw · ρ / ‖raw‖ — exact radial-norm anchoring (‖y‖ ≡ ρ).
    let raw_norm = raw_norm_sq.sqrt();
    let scale = RFloat::with_val(prec, &rho / &raw_norm);
    std::array::from_fn(|i| RFloat::with_val(prec, &raw[i] * &scale))
}

// ─── Brute-force phase1 ──────────────────────────────────────────────────────

/// Recursive enumerator: walks integer 16-vectors with `‖x‖² = remaining`
/// at the current recursion depth.
fn enumerate<F: FnMut(&[i64; 16])>(
    x: &mut [i64; 16],
    pos: usize,
    remaining: i64,
    cb: &mut F,
) {
    if pos == 16 {
        if remaining == 0 {
            cb(x);
        }
        return;
    }
    let bound = (remaining as f64).sqrt().floor() as i64;
    for v in -bound..=bound {
        let v2 = v * v;
        if v2 > remaining {
            continue;
        }
        x[pos] = v;
        enumerate(x, pos + 1, remaining - v2, cb);
    }
}

/// Brute-force phase1 for Z[ζ_16]: enumerate all `(u_1, u_2) ∈ Z[ζ_16]²`
/// with `‖u_1‖² + ‖u_2‖² = 2^k` and `B_1 = B_2 = B_3 = 0`.
///
/// Returns 16-element integer solutions. Cost is exponential in `k`.
pub fn phase1_brute(k: u32) -> Vec<[i64; 16]> {
    assert!(k < 31, "k too large for i64 norm shell (would overflow)");
    let target_norm_sq = 1i64 << k;
    let mut x = [0i64; 16];
    let mut results = Vec::new();
    enumerate(&mut x, 0, target_norm_sq, &mut |x| {
        let (b1, b2, b3) = bilinear_forms(x);
        if b1 == 0 && b2 == 0 && b3 == 0 {
            results.push(*x);
        }
    });
    results
}
