//! Near-shallow-point detection and the escape-circuit route (issue #2).
//!
//! Targets a distance δ with ε ≤ δ ≲ 10³ε from a *shallow* circuit point
//! (a Clifford, or a T-count-1 / √T-count-1 unitary) provably need
//! ~3·log₂(δ₀/δ) more non-Clifford gates than generic targets (verified
//! against the gridsynth oracle): lattice points repel low-height points.
//! Those solutions sit above the ε-tuned lde band, and the prefix-split
//! search would grow a 2^t' prefix set trying to reach them — the
//! hang/`None`/OOM of issue #2.
//!
//! Instead: multiply the target by `W† = ((HT)^m)†` — an exact circuit far
//! from every shallow point — synthesize the now-generic `U·W†` with the
//! normal pipeline, and append `W`'s gates. Diamond distance is exactly
//! preserved under composition with the exact `W`, and the total
//! non-Clifford count lands within a few gates of the true optimum
//! (empirically T=91 vs oracle 91 at δ=37ε, T=99 vs 99 at δ=2ε).
//!
//! Below ε the shallow point itself is the (optimal) answer and is
//! returned directly from the catalog.

use std::sync::OnceLock;

use num_complex::Complex;

use crate::matrix::{U2Q, U2T};
use crate::rings::types::{int_to_f64, MpFloat};
use crate::rings::ZOmega;
use crate::synthesis::cliffords::CLIFFORD_TABLE_T;
use crate::synthesis::distance::{diamond_distance_float, to_su2, Mat2};

/// Targets farther than `RING_FACTOR`·ε from every shallow point take the
/// normal pipeline. Calibrated on the oracle sweep: the T-count climb
/// starts around δ ≈ 10³ε, but in-band solutions persist down to ~10²ε;
/// 256 trades ≤ ~6 extra T for escaping the slow empty-level scan.
pub(crate) const RING_FACTOR: f64 = 256.0;

/// The escape only runs at deep ε, where the lde band binds. Shallower,
/// `max_lde`'s `.max(50)` floor leaves ~20 levels of headroom over the
/// generic find-lde plus the near-Clifford climb, the direct search
/// handles the ring fine, and escaping would only add ~m T-gates (at
/// ε=1e-2 the ring would even swallow most of SU(2)).
pub(crate) const ESCAPE_EPS_MAX: f64 = 1e-4;

/// Escape-circuit length: `(HT)^m`. Deeper in the ring the escape point
/// itself must be "less shallow", or the inner search inherits the
/// repulsion (δ=2ε with m=12 took 219 s; m=20 took 0.65 s).
const M_FAR: usize = 12; // δ/ε ≥ 32
const M_NEAR: usize = 20; // δ/ε < 32

/// Evaluate a Clifford+T gate string (result convention: leftmost char =
/// leftmost matrix factor) to an exact reduced U2T. `W`/`I` (gridsynth
/// global phase / identity) are skipped; lowercase `s`/`t` are adjoints.
pub(crate) fn eval_gates_t(gates: &str) -> Option<U2T> {
    let mut u = U2T::eye();
    for ch in gates.chars() {
        let g = match ch {
            'H' => U2T::h(),
            'S' => U2T::s(),
            'T' => U2T::t(),
            'X' => U2T::x(),
            'Y' => U2T::y(),
            'Z' => U2T::z(),
            's' => U2T::s().dagger(),
            't' => U2T::t().dagger(),
            'W' | 'I' => continue,
            _ => return None,
        };
        u = (u * g).reduced();
    }
    Some(u)
}

/// Clifford+√T variant (adds `Q` = √T).
pub(crate) fn eval_gates_q(gates: &str) -> Option<U2Q> {
    let mut u = U2Q::eye();
    for ch in gates.chars() {
        let g = match ch {
            'H' => U2Q::h(),
            'S' => U2Q::s(),
            'T' => U2Q::t(),
            'Q' => U2Q::q(),
            'X' => U2Q::x(),
            'Y' => U2Q::y(),
            'Z' => U2Q::z(),
            's' => U2Q::s().dagger(),
            't' => U2Q::t().dagger(),
            'q' => U2Q::q().dagger(),
            'W' | 'I' => continue,
            _ => return None,
        };
        u = (u * g).reduced();
    }
    Some(u)
}

// ─── Shallow-point catalog ────────────────────────────────────────────────────

pub(crate) struct ShallowPoint {
    /// Result-convention gate string.
    pub gates: String,
    /// f64 unitary for detection distances.
    pub mat: Mat2,
}

static CATALOG_T: OnceLock<Vec<ShallowPoint>> = OnceLock::new();
static CATALOG_Q: OnceLock<Vec<ShallowPoint>> = OnceLock::new();

/// CLIFFORD_TABLE_T names are in apply order (leftmost = first applied =
/// rightmost matrix factor); result strings are in matrix order. Reverse.
fn clifford_result_strings() -> Vec<String> {
    CLIFFORD_TABLE_T
        .iter()
        .map(|(name, _)| name.chars().rev().collect())
        .collect()
}

fn push_dedup(cat: &mut Vec<ShallowPoint>, gates: String) {
    let Some(u) = eval_gates_q(&gates) else { return };
    let mat = u.to_float();
    if cat.iter().any(|p| diamond_distance_float(&p.mat, &mat) < 1e-9) {
        return;
    }
    cat.push(ShallowPoint { gates, mat });
}

/// Cliffords + all distinct T-count-1 points, cheapest first. ~120 entries.
fn catalog_t() -> &'static [ShallowPoint] {
    CATALOG_T.get_or_init(|| {
        let cliffords = clifford_result_strings();
        let mut cat: Vec<ShallowPoint> = Vec::new();
        push_dedup(&mut cat, String::new()); // identity
        for c in &cliffords {
            push_dedup(&mut cat, c.clone());
        }
        for c1 in &cliffords {
            for c2 in &cliffords {
                push_dedup(&mut cat, format!("{c1}T{c2}"));
            }
        }
        cat
    })
}

/// √T-count-1 additions for the Clifford+√T backend (√T is shallow in
/// Z[ζ] but not in Z[ω]). Deduped against `catalog_t` entries too.
fn catalog_q_extra() -> &'static [ShallowPoint] {
    CATALOG_Q.get_or_init(|| {
        let cliffords = clifford_result_strings();
        let base = catalog_t();
        let mut cat: Vec<ShallowPoint> = Vec::new();
        for c1 in &cliffords {
            for c2 in &cliffords {
                let gates = format!("{c1}Q{c2}");
                let Some(u) = eval_gates_q(&gates) else { continue };
                let mat = u.to_float();
                if base.iter().chain(cat.iter())
                    .any(|p| diamond_distance_float(&p.mat, &mat) < 1e-9)
                {
                    continue;
                }
                cat.push(ShallowPoint { gates, mat });
            }
        }
        cat
    })
}

/// Nearest shallow point: `(δ, point)`. Catalog order is cheapest-first,
/// and strict `<` keeps the cheapest among ties (e.g. identity vs S·S†
/// duplicates are already deduped).
pub(crate) fn nearest_shallow(target: &Mat2, include_q: bool) -> (f64, &'static ShallowPoint) {
    let mut best_d = f64::INFINITY;
    let mut best: Option<&'static ShallowPoint> = None;
    let extra: &[ShallowPoint] = if include_q { catalog_q_extra() } else { &[] };
    for p in catalog_t().iter().chain(extra.iter()) {
        let d = diamond_distance_float(&p.mat, target);
        if d < best_d {
            best_d = d;
            best = Some(p);
        }
    }
    (best_d, best.expect("catalog is non-empty"))
}

// ─── Escape circuit ───────────────────────────────────────────────────────────

pub(crate) struct Escape {
    /// `W`'s gates (appended to the inner result).
    pub w_gates: String,
    /// f64 target for the inner synthesis: `target·W†`.
    pub target: Mat2,
    /// √det-normalized first column of `su2(target·W†)` at the input
    /// column's precision — the exact cap center for the deep-ε path.
    pub col: [MpFloat; 4],
}

/// ZOmega entry over √2^k → (re, im) MPFR pair.
/// ω = (1+i)/√2: re = (a + (b−d)/√2)/√2^k, im = (c + (b+d)/√2)/√2^k.
fn zomega_to_mpfr(z: &ZOmega, k: u32, prec: u32) -> (MpFloat, MpFloat) {
    let inv_sqrt2 = MpFloat::with_val(prec, 1.0) / MpFloat::with_val(prec, 2.0).sqrt();
    let half_k = k / 2;
    let mut inv_scale = MpFloat::with_val(prec, 1.0);
    inv_scale >>= half_k;
    if k % 2 == 1 {
        inv_scale *= &inv_sqrt2;
    }
    // int_to_f64 is exact here: W's coefficients are ≤ 2^k ≤ 2^20 ≪ 2^53
    // (same conversion diamond_distance_u2t_float uses).
    let a = MpFloat::with_val(prec, int_to_f64(z.a));
    let b = MpFloat::with_val(prec, int_to_f64(z.b));
    let c = MpFloat::with_val(prec, int_to_f64(z.c));
    let d = MpFloat::with_val(prec, int_to_f64(z.d));
    let bd_diff = MpFloat::with_val(prec, &b - &d);
    let bd_sum = MpFloat::with_val(prec, &b + &d);
    let re = (a + bd_diff * &inv_sqrt2) * &inv_scale;
    let im = (c + bd_sum * &inv_sqrt2) * &inv_scale;
    (re, im)
}

/// `det(w) = ω^j` for a unitary-up-to-phase U2T: exactly one ZOmega
/// coefficient of the numerator determinant equals ±2^k, the rest 0.
/// Returns `j ∈ 0..8`, or `None` if `w` isn't unit-determinant (a bug).
fn det_unit_exponent(w: &U2T) -> Option<u32> {
    let det = w.u11 * w.u22 - w.u12 * w.u21;
    // Exact in f64: |coeff| = 2^k ≤ 2^20 for escape circuits.
    #[allow(clippy::cast_precision_loss)]
    let scale = (1i64 << w.k) as f64;
    let coeffs = [
        int_to_f64(det.a),
        int_to_f64(det.b),
        int_to_f64(det.c),
        int_to_f64(det.d),
    ];
    let mut j = None;
    for (i, &v) in coeffs.iter().enumerate() {
        // basis (1, ω, i, ω³) ↔ ω^{0,1,2,3}; negated ↔ ω^{4,5,6,7}
        // i ≤ 3 and u32 conversion of a 0..8 index is lossless.
        #[allow(clippy::cast_possible_truncation)]
        let idx = i as u32;
        if v == scale {
            if j.is_some() {
                return None;
            }
            j = Some(idx);
        } else if v == -scale {
            if j.is_some() {
                return None;
            }
            j = Some(idx + 4);
        } else if v != 0.0 {
            return None;
        }
    }
    j
}

/// Build the escape data for `target` (δ from its nearest shallow point).
/// Returns `None` only on internal inconsistency — callers then fall back
/// to the direct pipeline.
pub(crate) fn build_escape(target: &Mat2, col: &[MpFloat; 4], delta_over_eps: f64) -> Option<Escape> {
    let m = if delta_over_eps >= 32.0 { M_FAR } else { M_NEAR };
    let w_gates = "HT".repeat(m);
    let w = eval_gates_t(&w_gates)?;
    let wd = w.dagger();

    // f64 inner target: target · W†.
    let wd_f = wd.to_float();
    let mut t_new = [[Complex::new(0.0, 0.0); 2]; 2];
    for i in 0..2 {
        for jj in 0..2 {
            t_new[i][jj] = target[i][0] * wd_f[0][jj] + target[i][1] * wd_f[1][jj];
        }
    }

    // Exact column: su2(target·W†) = su2(target)·su2(W†), and the first
    // column of A·B needs only A's columns — both recoverable from `col`
    // (SU(2): second column = (−conj u10, conj u00)).
    let prec = col[0].prec();
    let j = det_unit_exponent(&wd)?;
    // su2(W†) = W† / √(ω^j) = W† · e^{−ijπ/8} (sign branch fixed below).
    let pi = MpFloat::with_val(prec, rug::float::Constant::Pi);
    let ang = MpFloat::with_val(prec, &pi * i64::from(j)) / -8i32;
    let (ph_re, ph_im) = (ang.clone().cos(), ang.sin());
    let (w00r, w00i) = zomega_to_mpfr(&wd.u11, wd.k, prec);
    let (w10r, w10i) = zomega_to_mpfr(&wd.u21, wd.k, prec);
    // a = phase·W†₀₀, b = phase·W†₁₀ (first column of su2(W†)).
    let a_re = MpFloat::with_val(prec, &ph_re * &w00r) - MpFloat::with_val(prec, &ph_im * &w00i);
    let a_im = MpFloat::with_val(prec, &ph_re * &w00i) + MpFloat::with_val(prec, &ph_im * &w00r);
    let b_re = MpFloat::with_val(prec, &ph_re * &w10r) - MpFloat::with_val(prec, &ph_im * &w10i);
    let b_im = MpFloat::with_val(prec, &ph_re * &w10i) + MpFloat::with_val(prec, &ph_im * &w10r);

    // col' = a·(u00, u10) + b·(−conj u10, conj u00), complex ops by hand.
    let (u00r, u00i, u10r, u10i) = (&col[0], &col[1], &col[2], &col[3]);
    let mul = |pr: &MpFloat, pi_: &MpFloat, qr: &MpFloat, qi: &MpFloat| {
        (
            MpFloat::with_val(prec, pr * qr) - MpFloat::with_val(prec, pi_ * qi),
            MpFloat::with_val(prec, pr * qi) + MpFloat::with_val(prec, pi_ * qr),
        )
    };
    let (au00r, au00i) = mul(&a_re, &a_im, u00r, u00i);
    let (au10r, au10i) = mul(&a_re, &a_im, u10r, u10i);
    // −conj(u10) = (−u10r, u10i); conj(u00) = (u00r, −u00i).
    let neg_u10r = MpFloat::with_val(prec, -u10r.clone());
    let neg_u00i = MpFloat::with_val(prec, -u00i.clone());
    let (bv0r, bv0i) = mul(&b_re, &b_im, &neg_u10r, u10i);
    let (bv1r, bv1i) = mul(&b_re, &b_im, u00r, &neg_u00i);
    let mut col_new = [
        MpFloat::with_val(prec, &au00r + &bv0r),
        MpFloat::with_val(prec, &au00i + &bv0i),
        MpFloat::with_val(prec, &au10r + &bv1r),
        MpFloat::with_val(prec, &au10i + &bv1i),
    ];

    // √(ω^j) branch: align with the f64 convention (`to_su2(t_new)`), which
    // the 8D pipeline derives its uv frame from. A mismatch would put the
    // exact cap center at the antipode.
    let su = to_su2(&t_new);
    let expected = [su[0][0].re, su[0][0].im, su[1][0].re, su[1][0].im];
    let dot: f64 = expected
        .iter()
        .zip(col_new.iter())
        .map(|(e, c)| e * c.to_f64())
        .sum();
    if dot < 0.0 {
        for c in &mut col_new {
            *c = MpFloat::with_val(prec, -c.clone());
        }
    }

    Some(Escape { w_gates, target: t_new, col: col_new })
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthesis::angle::Angle;
    use crate::synthesis::synthesizer::Synthesizer;

    #[test]
    fn test_catalog_sizes_sane() {
        let n_t = catalog_t().len();
        let n_q = catalog_q_extra().len();
        // 1 identity + 24 Cliffords + deduped T-count-1 set.
        assert!(n_t > 25 && n_t < 601, "catalog_t size {n_t}");
        assert!(n_q > 0 && n_q < 577, "catalog_q_extra size {n_q}");
        // Every entry's string must round-trip through the evaluator.
        for p in catalog_t() {
            assert!(eval_gates_q(&p.gates).is_some(), "bad entry {:?}", p.gates);
        }
    }

    /// Issue #2, u3 form: a general (β ≠ 0, so the native-Rz route does
    /// not apply) target in the pathological ring must return quickly via
    /// the escape circuit instead of hanging or returning None.
    #[test]
    fn test_u3_near_identity_ring() {
        let synth = Synthesizer::new(1e-8, false);
        let r = synth
            .synthesize_zyz_joint(Angle::Rad(0.0), Angle::Rad(3.74507e-7), Angle::Rad(0.0))
            .expect("near-identity u3 should synthesize");
        assert!(r.distance < 1e-8, "distance {}", r.distance);
        let gates = r.gates.expect("gates");
        // The escape suffix is (HT)^m — the result must end with it.
        assert!(gates.ends_with("HT"), "unexpected tail: {gates}");
    }

    /// Same, near H (a non-diagonal Clifford).
    #[test]
    fn test_u3_near_h_ring() {
        let synth = Synthesizer::new(1e-8, false);
        let r = synth
            .synthesize_zyz_joint(
                Angle::Rad(0.0),
                Angle::Rad(std::f64::consts::FRAC_PI_2 + 3.74507e-7),
                Angle::PiRatio(1, 1),
            )
            .expect("near-H u3 should synthesize");
        assert!(r.distance < 1e-8, "distance {}", r.distance);
    }

    /// Below ε the catalog answers directly with the shallow circuit.
    #[test]
    fn test_u3_below_eps_catalog() {
        let synth = Synthesizer::new(1e-8, false);
        let r = synth
            .synthesize_zyz_joint(Angle::Rad(0.0), Angle::Rad(5e-9), Angle::Rad(0.0))
            .expect("below-eps u3 should synthesize");
        assert_eq!(r.gates.as_deref(), Some(""), "expected the identity");
        assert!(r.distance < 1e-8);
    }

    /// Clifford+√T backend takes the same escape (ε ≤ ESCAPE_EPS_MAX).
    #[test]
    fn test_q_backend_ring_escape() {
        let synth = Synthesizer::new(1e-5, true);
        let r = synth
            .synthesize_zyz_joint(Angle::Rad(0.0), Angle::Rad(5e-5), Angle::Rad(0.0))
            .expect("near-identity √T target should synthesize");
        assert!(r.distance < 1e-5, "distance {}", r.distance);
    }

    /// Shallow ε: the ring escape must NOT trigger (direct path handles
    /// near-Clifford fine there and cheaper).
    #[test]
    fn test_shallow_eps_no_escape() {
        let synth = Synthesizer::new(1e-2, false);
        let r = synth
            .synthesize_zyz_joint(Angle::Rad(0.0), Angle::Rad(0.05), Angle::Rad(0.0))
            .expect("shallow-eps target should synthesize");
        assert!(r.distance < 1e-2);
        let gates = r.gates.expect("gates");
        // The escape appends (HT)^m with m ≥ 12; a direct-path result
        // ending in that exact suffix is astronomically unlikely.
        assert!(
            !gates.ends_with(&"HT".repeat(12)),
            "unexpected escape at shallow ε: {gates}"
        );
    }
}
