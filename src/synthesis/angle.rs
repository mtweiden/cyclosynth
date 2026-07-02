//! Angle inputs for single-qubit synthesis.
//!
//! An [`Angle`] is either a plain f64 in radians or an exact rational multiple
//! of π, `PiRatio(p, q) = (p/q)·π`. The rational form evaluates to any MPFR
//! precision, which is what lets a rational-π target be built exactly rather
//! than at the f64 wall (the cap half-width ≈ ε² reaches the f64 ULP at
//! ε≈1e-8). [`su2_from_zyz`] builds the f64 target; [`su2_col_mpfr`] builds the
//! exact target column at a given precision; [`angle_target`] builds BOTH from
//! the same angles and is the only sanctioned way to obtain the pair.

use num_complex::Complex;

use crate::rings::types::MpFloat;
use crate::synthesis::distance::Mat2;

/// A parsed angle: `PiRatio(p, q)` is exactly `(p/q)·π`; `Rad(x)` is f64 radians.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Angle {
    Rad(f64),
    PiRatio(i64, i64),
}

impl Angle {
    /// Evaluate to f64 radians.
    // f64 evaluation is the approximate path by contract (exact path: to_radians_mpfr).
    #[allow(clippy::cast_precision_loss)]
    pub(crate) fn to_radians_f64(self) -> f64 {
        match self {
            Angle::Rad(x) => x,
            Angle::PiRatio(p, q) => (p as f64) / (q as f64) * std::f64::consts::PI,
        }
    }

    /// Evaluate to radians as an MPFR float at `prec` bits. For `PiRatio` the
    /// result is the correctly-rounded `(p/q)·π` (π carries full `prec` bits,
    /// p and q are exact), the source of exactness below the f64 ULP.
    pub(crate) fn to_radians_mpfr(self, prec: u32) -> MpFloat {
        match self {
            Angle::Rad(x) => MpFloat::with_val(prec, x),
            Angle::PiRatio(p, q) => {
                let pi = MpFloat::with_val(prec, rug::float::Constant::Pi);
                // Multiply/divide by i64 directly — an f64 round-trip would
                // silently lose exactness for |p| or |q| > 2^53.
                MpFloat::with_val(prec, &pi * p) / q
            }
        }
    }
}

fn gcd_i64(a: i64, b: i64) -> i64 {
    let (mut a, mut b) = (a.abs(), b.abs());
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a.max(1)
}

/// Parse a signed integer-or-decimal literal as an exact rational `(num, den)`,
/// e.g. `"3" -> (3, 1)`, `"0.25" -> (1, 4)`, `"-1.5" -> (-3, 2)`.
fn parse_decimal_ratio(s: &str) -> Option<(i64, i64)> {
    let (sign, body) = match s.strip_prefix('-') {
        Some(rest) => (-1i64, rest),
        None => (1i64, s.strip_prefix('+').unwrap_or(s)),
    };
    if body.is_empty() {
        return None;
    }
    let (num, den) = match body.split_once('.') {
        None => (body.parse::<i64>().ok()?, 1i64),
        Some((int_part, frac)) => {
            // >18 fractional digits overflows the i64 denominator (10^19 > i64::MAX).
            if frac.is_empty() || frac.len() > 18 {
                return None;
            }
            let den = 10i64.checked_pow(u32::try_from(frac.len()).ok()?)?;
            let int_v: i64 = if int_part.is_empty() { 0 } else { int_part.parse().ok()? };
            let frac_v: i64 = frac.parse().ok()?;
            (int_v.checked_mul(den)?.checked_add(frac_v)?, den)
        }
    };
    let g = gcd_i64(num, den);
    Some((sign * num / g, den / g))
}

/// Parse an angle string. A string containing `pi` (whitespace ignored,
/// optional `*`) is a rational multiple of π — `[coeff][*]pi[/denom]`, e.g.
/// `"pi"`, `"3pi"`, `"3*pi"`, `"pi/8"`, `"3*pi/4"`, `"-2pi/3"`, `"0.25pi"` —
/// returned as a reduced [`Angle::PiRatio`]. A string with no `pi` parses as
/// [`Angle::Rad`]. The `Err` is a human-readable message.
#[cfg_attr(not(feature = "python"), allow(dead_code))] // consumed by the PySynthesizer angle-string parser (and tests)
pub(crate) fn parse_angle_str(raw: &str) -> Result<Angle, String> {
    let s: String = raw.chars().filter(|c| !c.is_whitespace()).collect();
    let lower = s.to_lowercase();
    let Some(pos) = lower.find("pi") else {
        return lower
            .parse::<f64>()
            .map(Angle::Rad)
            .map_err(|_| format!("could not parse angle '{raw}'"));
    };
    let before = lower[..pos].trim_end_matches('*');
    let after = &lower[pos + 2..];
    let (cnum, cden): (i64, i64) = match before {
        "" | "+" => (1, 1),
        "-" => (-1, 1),
        c => parse_decimal_ratio(c).ok_or_else(|| format!("bad π coefficient in angle '{raw}'"))?,
    };
    let denom: i64 = if after.is_empty() {
        1
    } else if let Some(d) = after.strip_prefix('/') {
        d.parse::<i64>().map_err(|_| format!("bad π denominator in angle '{raw}'"))?
    } else {
        return Err(format!("unexpected '{after}' after π in angle '{raw}'"));
    };
    if denom == 0 {
        return Err(format!("zero π denominator in angle '{raw}'"));
    }
    let den = cden.checked_mul(denom).ok_or_else(|| format!("π denominator overflow in angle '{raw}'"))?;
    let g = gcd_i64(cnum, den);
    let sign = if den < 0 { -1 } else { 1 };
    Ok(Angle::PiRatio(sign * cnum / g, den.abs() / g))
}

/// `Rz(alpha)·Ry(beta)·Rz(gamma)` as an SU(2) `Mat2` (det = 1 by construction).
/// Convention: `Rz(t) = diag(e^{-it/2}, e^{it/2})`, `Ry(t) = [[c,-s],[s,c]]`.
pub(crate) fn su2_from_zyz(alpha: f64, beta: f64, gamma: f64) -> Mat2 {
    let cb = (beta * 0.5).cos();
    let sb = (beta * 0.5).sin();
    let pag = (alpha + gamma) * 0.5;
    let pamg = (alpha - gamma) * 0.5;
    let polar = |r: f64, theta: f64| Complex::new(r * theta.cos(), r * theta.sin());
    [
        [polar(cb, -pag), -polar(sb, -pamg)],
        [polar(sb, pamg), polar(cb, pag)],
    ]
}

/// Default MPFR precision for the exact target column: 384 bits covers the
/// search precision (≈6·log₂(1/ε)) for any ε ≳ 1e-19.
pub(crate) const DEFAULT_COL_PREC: u32 = 384;

/// Build BOTH synthesis artifacts from the SAME ZYZ angles: the f64 SU(2)
/// target matrix (acceptance check) and the MPFR target column at `prec`
/// bits (deep-ε box center; exact for `PiRatio` angles).
///
/// This is the ONE place the (target, column) pair is constructed — every
/// angle entry point (`Synthesizer`/`SynthesizerT`/`SynthesizerQ`
/// `synthesize_zyz`/`synthesize_u3`) routes through it. Hand-assembling the
/// pair risks a convention mismatch (e.g. U3 angles passed positionally into
/// the ZYZ signature), which centers the deep-ε box on a different unitary
/// than the acceptance target and produces a silent-miss ladder.
pub(crate) fn angle_target(alpha: Angle, beta: Angle, gamma: Angle, prec: u32) -> (Mat2, [MpFloat; 4]) {
    let mat = su2_from_zyz(alpha.to_radians_f64(), beta.to_radians_f64(), gamma.to_radians_f64());
    let col = su2_col_mpfr(alpha, beta, gamma, prec);
    debug_assert!(
        col_target_mismatch(&col, &mat) < 1e-9,
        "MPFR column disagrees with f64 target (err {:.3e}) — angle-convention \
         mismatch (U3 vs ZYZ arg order into su2_col_mpfr)",
        col_target_mismatch(&col, &mat)
    );
    (mat, col)
}

/// Max phase-aligned error between the MPFR column and column 1 of the f64
/// target. Debug-build guard against angle-convention drift: the column must
/// be (up to global phase) column 1 of the target — else the deep-ε search
/// centers on a different unitary than the acceptance check.
fn col_target_mismatch(col: &[MpFloat; 4], target: &Mat2) -> f64 {
    let c0 = Complex::new(col[0].to_f64(), col[1].to_f64());
    let c1 = Complex::new(col[2].to_f64(), col[3].to_f64());
    let (t00, t10) = (target[0][0], target[1][0]);
    // Phase-align on the larger component, then compare both.
    let (ref_c, ref_t) = if c0.norm() >= c1.norm() { (c0, t00) } else { (c1, t10) };
    let phase = if ref_c.norm() > 1e-12 { ref_t / ref_c } else { Complex::new(1.0, 0.0) };
    (c0 * phase - t00).norm().max((c1 * phase - t10).norm())
}

/// Column 1 of `Rz(alpha)·Ry(beta)·Rz(gamma)` in MPFR at `prec` bits, as
/// `[Re u00, Im u00, Re u10, Im u10]`. Exact for `PiRatio` angles. The column
/// is unit-norm by construction, so no `√det` normalization is needed.
pub(crate) fn su2_col_mpfr(alpha: Angle, beta: Angle, gamma: Angle, prec: u32) -> [MpFloat; 4] {
    let a = alpha.to_radians_mpfr(prec);
    let b = beta.to_radians_mpfr(prec);
    let g = gamma.to_radians_mpfr(prec);
    let hb = MpFloat::with_val(prec, &b / 2.0);
    let pag = MpFloat::with_val(prec, MpFloat::with_val(prec, &a + &g) / 2.0);
    let pamg = MpFloat::with_val(prec, MpFloat::with_val(prec, &a - &g) / 2.0);
    let cb = hb.clone().cos();
    let sb = hb.sin();
    let cpag = pag.clone().cos();
    let spag = pag.sin();
    let cpamg = pamg.clone().cos();
    let spamg = pamg.sin();
    let re00 = MpFloat::with_val(prec, &cb * &cpag);
    let im00 = MpFloat::with_val(prec, &cb * &spag);
    let re10 = MpFloat::with_val(prec, &sb * &cpamg);
    let im10 = MpFloat::with_val(prec, &sb * &spamg);
    [re00, -im00, re10, im10]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_pi_forms() {
        assert_eq!(parse_angle_str("pi/2").unwrap(), Angle::PiRatio(1, 2));
        assert_eq!(parse_angle_str("0.25pi").unwrap(), Angle::PiRatio(1, 4));
        assert_eq!(parse_angle_str("3*pi/4").unwrap(), Angle::PiRatio(3, 4));
        assert_eq!(parse_angle_str("3pi/4").unwrap(), Angle::PiRatio(3, 4));
        assert_eq!(parse_angle_str("-2pi/3").unwrap(), Angle::PiRatio(-2, 3));
        assert_eq!(parse_angle_str("2pi/4").unwrap(), Angle::PiRatio(1, 2)); // reduced
        assert_eq!(parse_angle_str(" - pi / 2 ").unwrap(), Angle::PiRatio(-1, 2));
        assert_eq!(parse_angle_str("pi").unwrap(), Angle::PiRatio(1, 1));
        assert!(matches!(parse_angle_str("0.3").unwrap(), Angle::Rad(_)));
        assert!(parse_angle_str("pi/0").is_err());
        assert!(parse_angle_str("xyz").is_err());
    }

    #[test]
    fn mpfr_column_matches_f64() {
        let prec = 128;
        // Mixed rational-π and plain-radian angles.
        let cases = [
            (Angle::PiRatio(1, 4), Angle::PiRatio(1, 2), Angle::PiRatio(-1, 3)),
            (Angle::Rad(0.7), Angle::Rad(1.1), Angle::Rad(0.4)),
            (Angle::PiRatio(3, 7), Angle::Rad(0.9), Angle::PiRatio(2, 5)),
        ];
        for (a, b, g) in cases {
            let col = su2_col_mpfr(a, b, g, prec);
            let m = su2_from_zyz(a.to_radians_f64(), b.to_radians_f64(), g.to_radians_f64());
            let got = [
                col[0].to_f64(),
                col[1].to_f64(),
                col[2].to_f64(),
                col[3].to_f64(),
            ];
            let want = [m[0][0].re, m[0][0].im, m[1][0].re, m[1][0].im];
            for i in 0..4 {
                assert!(
                    (got[i] - want[i]).abs() < 1e-12,
                    "entry {i}: mpfr {} vs f64 {}",
                    got[i],
                    want[i]
                );
            }
        }
    }
}
