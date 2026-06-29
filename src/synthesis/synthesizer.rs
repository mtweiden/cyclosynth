//! Unified user-facing `Synthesizer` API.
//!
//! Wraps the two ring-specific synthesis backends —
//! [`SynthesizerT`](crate::synthesis::clifford_t::SynthesizerT) for
//! Clifford+T over Z[ω] and
//! [`SynthesizerQ`](crate::synthesis::clifford_sqrt_t::SynthesizerQ) for
//! Clifford+√T over Z[ζ_16] — behind a single struct. Pick the backend at
//! construction with the `sqrt_t: bool` flag (default false → Clifford+T).
//!
//! ## Why two backends behind one type
//!
//! The two flows currently use *different algorithms* (Z[ω]: 8D MA-prefix
//! divide-and-conquer; Z[ζ_16]: 16D LLL+SE with a brute-force small-k mode
//! and an FGKM-prefix divide-and-conquer mode for deep k), so they can't be
//! expressed cleanly as a single generic `Synthesizer<R: GateRing>`. This
//! wrapper gives users a single API while the internals keep their own
//! optimised code paths.

use crate::synthesis::distance::Mat2;
use crate::synthesis::clifford_t::SynthesizerT;
use crate::synthesis::clifford_sqrt_t::SynthesizerQ;

/// Result of a successful synthesis call. Same shape regardless of the
/// underlying gate set.
#[derive(Debug, Clone)]
pub struct SynthResult {
    /// Gate string (leftmost = first gate applied). Alphabet is
    /// `{H, S, T, X, Y, Z}` for Clifford+T and `{H, S, T, Q, X, Y, Z}` for
    /// Clifford+√T (`Q = √T`). `None` if extraction failed.
    pub gates: Option<String>,
    /// Denominator exponent of the synthesized unitary.
    pub lde: u32,
    /// Diamond distance from synthesized unitary to target.
    pub distance: f64,
}

/// Single-qubit unitary synthesizer over either Clifford+T (Z[ω]) or
/// Clifford+√T (Z[ζ_16]).
///
/// ```rust,ignore
/// // Clifford+T (default).
/// let synth = Synthesizer::new(1e-3, false);
/// // Clifford+√T (denser gate set, generally fewer gates).
/// let synth = Synthesizer::new(1e-3, true);
/// let result = synth.synthesize(target);
/// ```
pub struct Synthesizer {
    inner: Backend,
}

// A `Synthesizer` is created once per session and never held in bulk, so the
// T-vs-Q size gap is irrelevant; boxing would only fight the consuming
// `with_*` builder methods.
#[allow(clippy::large_enum_variant)]
enum Backend {
    T(SynthesizerT),
    Q(SynthesizerQ),
}

impl Synthesizer {
    /// Create a synthesizer with the given precision target and gate set.
    /// `sqrt_t = false` (the default in user code) selects Clifford+T;
    /// `true` selects Clifford+√T.
    pub fn new(epsilon: f64, sqrt_t: bool) -> Self {
        let inner = if sqrt_t {
            Backend::Q(SynthesizerQ::new(epsilon))
        } else {
            Backend::T(SynthesizerT::new(epsilon))
        };
        Self { inner }
    }

    /// Override the maximum lde the search will probe.
    pub fn with_max_lde(mut self, max_lde: u32) -> Self {
        match &mut self.inner {
            Backend::T(s) => s.max_lde = max_lde,
            Backend::Q(s) => s.max_lde = max_lde,
        }
        self
    }

    /// Override the minimum lde the search will probe.
    pub fn with_min_lde(mut self, min_lde: u32) -> Self {
        match &mut self.inner {
            Backend::T(s) => s.min_lde = min_lde,
            Backend::Q(s) => s.min_lde = min_lde,
        }
        self
    }

    /// Clifford+√T only: enumerate all ε-close candidates at the found
    /// lde and return the one minimising the weighted gate cost (see
    /// [`SynthesizerQ::with_optimize_cost`]). Ignored for Clifford+T,
    /// where the search is already T-count-minimal by construction.
    pub fn with_optimize_cost(mut self, on: bool) -> Self {
        if let Backend::Q(s) = self.inner {
            self.inner = Backend::Q(s.with_optimize_cost(on));
        }
        self
    }

    /// Clifford+√T only: Q-gate weight in T units for the optimize-cost
    /// model (default 3). Quantized to the nearest half-unit (the model
    /// compares `2·T + round(2·weight)·Q`), so e.g. 3.6 → 3.5. Ignored for
    /// Clifford+T.
    pub fn with_q_cost(mut self, weight: f64) -> Self {
        if let Backend::Q(s) = self.inner {
            self.inner = Backend::Q(s.with_q_cost(weight));
        }
        self
    }

    /// Clifford+√T only: also search `window` lde levels above the first
    /// hit and return the global min-cost candidate (see
    /// [`SynthesizerQ::with_optimal_lde_window`]). Ignored for Clifford+T.
    pub fn with_optimal_lde_window(mut self, window: u32) -> Self {
        if let Backend::Q(s) = self.inner {
            self.inner = Backend::Q(s.with_optimal_lde_window(window));
        }
        self
    }

    /// Clifford+√T only: wall-clock budget in ms for the min-cost
    /// enumeration (see [`SynthesizerQ::with_optimal_deadline_ms`]);
    /// `None` removes the deadline. Ignored for Clifford+T.
    pub fn with_optimal_deadline_ms(mut self, ms: Option<u64>) -> Self {
        if let Backend::Q(s) = self.inner {
            self.inner = Backend::Q(s.with_optimal_deadline_ms(ms));
        }
        self
    }

    /// Clifford+√T only: override the deep-ε sequential-parity schedule
    /// (see [`SynthesizerQ::with_seq_parity`]); `Some(false)` forces the
    /// concurrent (lower-wall) branches below the 2.5e-8 sequential
    /// threshold. Ignored for Clifford+T.
    pub fn with_seq_parity(mut self, seq: Option<bool>) -> Self {
        if let Backend::Q(s) = self.inner {
            self.inner = Backend::Q(s.with_seq_parity(seq));
        }
        self
    }

    /// Synthesize `target` (a 2×2 unitary). Returns `None` if no circuit
    /// in the chosen gate set within `max_lde` reaches diamond distance
    /// below `epsilon`.
    pub fn synthesize(&self, target: Mat2) -> Option<SynthResult> {
        match &self.inner {
            Backend::T(s) => s.synthesize(target).map(|r| SynthResult {
                gates: r.gates,
                lde: r.lde,
                distance: r.distance,
            }),
            Backend::Q(s) => s.synthesize(target).map(|r| SynthResult {
                gates: r.gates,
                lde: r.lde,
                distance: r.distance,
            }),
        }
    }

    /// Target diamond distance the synthesized circuit must come within.
    pub fn epsilon(&self) -> f64 {
        match &self.inner {
            Backend::T(s) => s.epsilon,
            Backend::Q(s) => s.epsilon,
        }
    }

    /// Largest lde (search depth) the synthesizer will try before giving up.
    pub fn max_lde(&self) -> u32 {
        match &self.inner {
            Backend::T(s) => s.max_lde,
            Backend::Q(s) => s.max_lde,
        }
    }

    /// Smallest lde the synthesizer starts from (skips guaranteed-empty
    /// shallow depths at deep ε).
    pub fn min_lde(&self) -> u32 {
        match &self.inner {
            Backend::T(s) => s.min_lde,
            Backend::Q(s) => s.min_lde,
        }
    }

    /// `true` if this is a Clifford+√T synthesizer, `false` for Clifford+T.
    pub fn sqrt_t(&self) -> bool {
        matches!(&self.inner, Backend::Q(_))
    }

    /// Cost in `T` states of one √T-class syllable in the syllable cost model
    /// (a T-class syllable costs 1). Canonical 3; reflects a custom
    /// `with_q_cost` on the √T backend.
    pub fn q_weight(&self) -> f64 {
        match &self.inner {
            Backend::T(_) => 3.0,
            Backend::Q(s) => s.q_cost_x2 as f64 / 2.0,
        }
    }
}

// ─── PyO3 bindings ────────────────────────────────────────────────────────────

#[cfg(feature = "python")]
use num_complex::Complex;
#[cfg(feature = "python")]
use numpy::{Complex64 as PyComplex64, PyReadonlyArray2};
#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Python-facing result of a synthesis run. Same shape for both gate sets.
#[cfg(feature = "python")]
#[pyclass(name = "SynthResult", frozen)]
pub struct PySynthResult {
    /// Gate string (leftmost = first gate applied), or `None` if extraction
    /// failed. Alphabet `{H, S, T, X, Y, Z}` for Clifford+T,
    /// `{H, S, T, Q, X, Y, Z}` for Clifford+√T.
    #[pyo3(get)]
    pub gates: Option<String>,
    /// Denominator exponent (lde, the search depth) of the synthesized unitary.
    #[pyo3(get)]
    pub lde: u32,
    /// Diamond distance from the synthesized unitary to the target (< epsilon).
    #[pyo3(get)]
    pub distance: f64,
    /// Q-gate weight used for `cost` (3 unless overridden on the √T backend).
    q_weight: f64,
}

#[cfg(feature = "python")]
#[pymethods]
impl PySynthResult {
    /// Number of T-class gates (`T` or its adjoint `t`=T†) in the circuit
    /// (0 if synthesis failed).
    #[getter]
    fn t_count(&self) -> usize {
        self.gates.as_deref().map_or(0, |g| {
            g.chars().filter(|&c| c == 'T' || c == 't').count()
        })
    }

    /// Number of √T-class gates (`Q`=√T or its adjoint `q`=Q†) in the circuit
    /// (0 for Clifford+T, or on failure).
    #[getter]
    fn q_count(&self) -> usize {
        self.gates.as_deref().map_or(0, |g| {
            g.chars().filter(|&c| c == 'Q' || c == 'q').count()
        })
    }

    /// The minimized resource cost, in `T` states. This is the syllable-model
    /// cost the optimizer minimizes: gates are charged per diagonal syllable
    /// by their net √T-power class (a √T-class syllable costs `q_weight`, a
    /// T-class syllable 1, Cliffords 0), so a `T` that composes with a `√T`
    /// into `T^{3/2}=√T†S` is one √T-class injection. It can therefore be
    /// *below* `t_count + q_weight·q_count`. (`q_weight` 3 default.)
    #[getter]
    fn cost(&self) -> f64 {
        let q_cost_x2 = (2.0 * self.q_weight).round() as usize;
        self.gates.as_deref().map_or(0.0, |g| {
            crate::synthesis::clifford_sqrt_t::gates_cost(g, q_cost_x2) as f64 / 2.0
        })
    }

    /// `True` if synthesis produced a circuit, so `if result:` works.
    fn __bool__(&self) -> bool {
        self.gates.is_some()
    }

    fn __repr__(&self) -> String {
        let gates_repr = self
            .gates
            .as_deref()
            .map(|g| format!("{g:?}"))
            .unwrap_or_else(|| "None".to_string());
        format!(
            "SynthResult(gates={gates_repr}, lde={}, distance={:.3e})",
            self.lde, self.distance
        )
    }
}

/// Unified Python-facing single-qubit unitary synthesizer.
///
/// ```python
/// import numpy as np, cyclosynth
/// theta = 0.3
/// target = np.array([[np.exp(-1j * theta / 2), 0],
///                    [0, np.exp(1j * theta / 2)]], dtype=np.complex128)
///
/// # Clifford+T (default).
/// synth = cyclosynth.Synthesizer(epsilon=1e-5)
/// # Clifford+√T (denser, generally fewer gates).
/// synth = cyclosynth.Synthesizer(epsilon=1e-5, sqrt_t=True)
///
/// result = synth.synthesize(target)
/// print(result.gates, result.lde, result.distance)
/// ```
#[cfg(feature = "python")]
#[pyclass(name = "Synthesizer", frozen)]
pub struct PySynthesizer {
    inner: Synthesizer,
}

#[cfg(feature = "python")]
#[pymethods]
impl PySynthesizer {
    /// Build a synthesizer for target diamond distance `epsilon`.
    ///
    /// `sqrt_t=False` gives Clifford+T; `sqrt_t=True` gives Clifford+√T
    /// (a denser gate set, usually fewer gates). `max_lde`/`min_lde` bound the
    /// search depth. The remaining kwargs (`optimize_cost`, `q_cost`,
    /// `lde_window`, `deadline_ms`, `seq_parity`) are **Clifford+√T-only**
    /// cost-optimizer tuning and raise `ValueError` if passed with
    /// `sqrt_t=False`.
    #[new]
    #[pyo3(signature = (epsilon, *, sqrt_t=false, max_lde=None, min_lde=None,
                        optimize_cost=None, q_cost=None, lde_window=None,
                        deadline_ms=None, seq_parity=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        epsilon: f64,
        sqrt_t: bool,
        max_lde: Option<u32>,
        min_lde: Option<u32>,
        optimize_cost: Option<bool>,
        q_cost: Option<f64>,
        lde_window: Option<u32>,
        deadline_ms: Option<u64>,
        seq_parity: Option<bool>,
    ) -> PyResult<Self> {
        // The cost-optimizer kwargs only affect the √T backend; silently
        // ignoring them for Clifford+T is a footgun, so reject up front.
        if !sqrt_t
            && (optimize_cost.is_some() || q_cost.is_some() || lde_window.is_some()
                || deadline_ms.is_some() || seq_parity.is_some())
        {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "optimize_cost / q_cost / lde_window / deadline_ms / seq_parity \
                 are Clifford+√T-only; pass sqrt_t=True to use them",
            ));
        }
        let mut s = Synthesizer::new(epsilon, sqrt_t);
        if let Some(v) = max_lde {
            s = s.with_max_lde(v);
        }
        if let Some(v) = min_lde {
            s = s.with_min_lde(v);
        }
        // None = keep the backend default (on for √T at every ε);
        // Some(b) = explicit override in either direction.
        if let Some(b) = optimize_cost {
            s = s.with_optimize_cost(b);
        }
        if let Some(w) = q_cost {
            s = s.with_q_cost(w);
        }
        if let Some(w) = lde_window {
            s = s.with_optimal_lde_window(w);
        }
        // None = keep the ε-based default deadline; an explicit value
        // overrides it (there is no kwarg form for "no deadline").
        if let Some(ms) = deadline_ms {
            s = s.with_optimal_deadline_ms(Some(ms));
        }
        if seq_parity.is_some() {
            s = s.with_seq_parity(seq_parity);
        }
        Ok(Self { inner: s })
    }

    /// Synthesize `target` (a 2×2 `np.complex128` unitary). Returns a
    /// `SynthResult`, or `None` if no circuit within `epsilon` was found at the
    /// allowed lde range. Raises `ValueError` if `target` isn't a 2×2 unitary.
    fn synthesize(
        &self,
        target: PyReadonlyArray2<PyComplex64>,
    ) -> PyResult<Option<PySynthResult>> {
        let view = target.as_array();
        if view.shape() != [2, 2] {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "target must be a 2×2 matrix, got shape {:?}",
                view.shape()
            )));
        }
        let mat: Mat2 = [
            [
                Complex::new(view[[0, 0]].re, view[[0, 0]].im),
                Complex::new(view[[0, 1]].re, view[[0, 1]].im),
            ],
            [
                Complex::new(view[[1, 0]].re, view[[1, 0]].im),
                Complex::new(view[[1, 1]].re, view[[1, 1]].im),
            ],
        ];
        // Reject non-unitary input up front: ‖U†U − I‖_F must be ~0. Loose
        // tolerance so f64-quantized unitaries pass; a clear error beats a
        // meaningless distance downstream.
        let mut off = 0.0_f64;
        for i in 0..2 {
            for j in 0..2 {
                let dot = mat[0][i].conj() * mat[0][j] + mat[1][i].conj() * mat[1][j];
                let want = if i == j { 1.0 } else { 0.0 };
                off += (dot.re - want).powi(2) + dot.im.powi(2);
            }
        }
        if off.sqrt() > 1e-6 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "target is not unitary (‖U†U − I‖_F = {:.2e})",
                off.sqrt()
            )));
        }
        let q_weight = self.inner.q_weight();
        Ok(self.inner.synthesize(mat).map(|r| PySynthResult {
            gates: r.gates,
            lde: r.lde,
            distance: r.distance,
            q_weight,
        }))
    }

    /// Synthesize a `U3(theta, phi, lambda)` gate (qiskit/bqskit convention)
    /// from its angles — the entry point for bqskit `U3Gate` inputs (pass
    /// `op.params`).
    ///
    /// `U3` is `e^{i(phi+lambda)/2}·Rz(phi)·Ry(theta)·Rz(lambda)`; the global
    /// phase is unobservable, so the SU(2) rotation `Rz(phi)·Ry(theta)·Rz(lambda)`
    /// is built directly as the target.
    ///
    /// Each angle is a float (radians) or a string. A string containing `pi`
    /// (whitespace ignored, optional `*`) is a rational multiple of π —
    /// `"pi"`, `"3pi"`, `"3*pi"`, `"pi/8"`, `"3*pi/4"`, `"-2pi/3"`, `"0.25pi"`;
    /// any other string parses as a float in radians.
    #[pyo3(signature = (theta, phi, lam))]
    fn synthesize_u3(
        &self,
        theta: &Bound<'_, PyAny>,
        phi: &Bound<'_, PyAny>,
        lam: &Bound<'_, PyAny>,
    ) -> PyResult<Option<PySynthResult>> {
        Ok(self.run_zyz(
            parse_angle(phi)?.to_radians_f64(),
            parse_angle(theta)?.to_radians_f64(),
            parse_angle(lam)?.to_radians_f64(),
        ))
    }

    /// Synthesize the SU(2) rotation `Rz(alpha)·Ry(beta)·Rz(gamma)` from its
    /// ZYZ Euler angles. Each angle accepts the same float/`pi`-string forms
    /// as [`Self::synthesize_u3`].
    #[pyo3(signature = (alpha, beta, gamma))]
    fn synthesize_zyz(
        &self,
        alpha: &Bound<'_, PyAny>,
        beta: &Bound<'_, PyAny>,
        gamma: &Bound<'_, PyAny>,
    ) -> PyResult<Option<PySynthResult>> {
        Ok(self.run_zyz(
            parse_angle(alpha)?.to_radians_f64(),
            parse_angle(beta)?.to_radians_f64(),
            parse_angle(gamma)?.to_radians_f64(),
        ))
    }

    #[getter]
    fn epsilon(&self) -> f64 {
        self.inner.epsilon()
    }

    #[getter]
    fn max_lde(&self) -> u32 {
        self.inner.max_lde()
    }

    #[getter]
    fn min_lde(&self) -> u32 {
        self.inner.min_lde()
    }

    #[getter]
    fn sqrt_t(&self) -> bool {
        self.inner.sqrt_t()
    }

    fn __repr__(&self) -> String {
        format!(
            "Synthesizer(epsilon={:.3e}, sqrt_t={}, min_lde={}, max_lde={})",
            self.inner.epsilon(),
            self.inner.sqrt_t(),
            self.inner.min_lde(),
            self.inner.max_lde(),
        )
    }
}

#[cfg(feature = "python")]
impl PySynthesizer {
    /// Build the SU(2) target from ZYZ angles (radians) and run the search.
    fn run_zyz(&self, alpha: f64, beta: f64, gamma: f64) -> Option<PySynthResult> {
        let mat = su2_from_zyz(alpha, beta, gamma);
        let q_weight = self.inner.q_weight();
        self.inner.synthesize(mat).map(|r| PySynthResult {
            gates: r.gates,
            lde: r.lde,
            distance: r.distance,
            q_weight,
        })
    }
}

/// `Rz(alpha)·Ry(beta)·Rz(gamma)` as an SU(2) `Mat2` (det = 1 by construction).
/// Convention: `Rz(t) = diag(e^{-it/2}, e^{it/2})`, `Ry(t) = [[c,-s],[s,c]]`.
#[cfg(feature = "python")]
fn su2_from_zyz(alpha: f64, beta: f64, gamma: f64) -> Mat2 {
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

/// A parsed angle: `PiRatio(p, q)` is exactly `(p/q)·π`; `Rad(x)` is f64
/// radians.
#[cfg(feature = "python")]
#[derive(Clone, Copy, Debug)]
enum Angle {
    Rad(f64),
    PiRatio(i64, i64),
}

#[cfg(feature = "python")]
impl Angle {
    /// Evaluate to f64 radians.
    fn to_radians_f64(self) -> f64 {
        match self {
            Angle::Rad(x) => x,
            Angle::PiRatio(p, q) => (p as f64) / (q as f64) * std::f64::consts::PI,
        }
    }
}

#[cfg(feature = "python")]
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
#[cfg(feature = "python")]
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
            let den = 10i64.checked_pow(frac.len() as u32)?;
            let int_v: i64 = if int_part.is_empty() { 0 } else { int_part.parse().ok()? };
            let frac_v: i64 = frac.parse().ok()?;
            (int_v.checked_mul(den)?.checked_add(frac_v)?, den)
        }
    };
    let g = gcd_i64(num, den);
    Some((sign * num / g, den / g))
}

/// Parse one angle argument — a Python float/int or a string.
#[cfg(feature = "python")]
fn parse_angle(obj: &Bound<'_, PyAny>) -> PyResult<Angle> {
    if let Ok(x) = obj.extract::<f64>() {
        return Ok(Angle::Rad(x));
    }
    let s: String = obj.extract().map_err(|_| {
        PyErr::new::<pyo3::exceptions::PyTypeError, _>("angle must be a float or string")
    })?;
    parse_angle_str(&s)
}

/// Parse an angle string. A string containing `pi` (whitespace ignored,
/// optional `*`) is a rational multiple of π — `[coeff][*]pi[/denom]`, e.g.
/// `"pi"`, `"3pi"`, `"3*pi"`, `"pi/8"`, `"3*pi/4"`, `"-2pi/3"`, `"0.25pi"` —
/// returned as an exact `PiRatio`. A string with no `pi` parses as `Rad`.
#[cfg(feature = "python")]
fn parse_angle_str(raw: &str) -> PyResult<Angle> {
    let bad = |m: String| PyErr::new::<pyo3::exceptions::PyValueError, _>(m);
    let s: String = raw.chars().filter(|c| !c.is_whitespace()).collect();
    let lower = s.to_lowercase();
    let Some(pos) = lower.find("pi") else {
        return lower
            .parse::<f64>()
            .map(Angle::Rad)
            .map_err(|_| bad(format!("could not parse angle '{raw}'")));
    };
    let before = lower[..pos].trim_end_matches('*');
    let after = &lower[pos + 2..];
    let (cnum, cden): (i64, i64) = match before {
        "" | "+" => (1, 1),
        "-" => (-1, 1),
        c => parse_decimal_ratio(c)
            .ok_or_else(|| bad(format!("bad π coefficient in angle '{raw}'")))?,
    };
    let denom: i64 = if after.is_empty() {
        1
    } else if let Some(d) = after.strip_prefix('/') {
        d.parse::<i64>()
            .map_err(|_| bad(format!("bad π denominator in angle '{raw}'")))?
    } else {
        return Err(bad(format!("unexpected '{after}' after π in angle '{raw}'")));
    };
    if denom == 0 {
        return Err(bad(format!("zero π denominator in angle '{raw}'")));
    }
    let den = cden
        .checked_mul(denom)
        .ok_or_else(|| bad(format!("π denominator overflow in angle '{raw}'")))?;
    let g = gcd_i64(cnum, den);
    let sign = if den < 0 { -1 } else { 1 };
    Ok(Angle::PiRatio(sign * cnum / g, den.abs() / g))
}
