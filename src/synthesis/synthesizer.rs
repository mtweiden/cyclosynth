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
//! The two flows use *different algorithms* (Z[ω]: 8D MA-prefix
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

    /// Synthesize with a higher-precision target column `exact_col` — the
    /// √det-normalized first column `[Re u00, Im u00, Re u10, Im u10]` of the
    /// SU(2) target (e.g. from exact rational-π angles via
    /// [`crate::synthesis::angle::su2_col_mpfr`]). Clifford+T aligns to it on
    /// the deep-ε MPFR path; Clifford+√T uses the f64 `target` for now.
    pub fn synthesize_su2_col(
        &self,
        target: Mat2,
        exact_col: &[crate::rings::MpFloat; 4],
    ) -> Option<SynthResult> {
        match &self.inner {
            Backend::T(s) => s.synthesize_with_exact_col(target, exact_col).map(|r| SynthResult {
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
            // q_cost_x2 is a small user knob (default 7; set from 2·weight).
            #[allow(clippy::cast_precision_loss)]
            Backend::Q(s) => s.q_cost_x2 as f64 / 2.0,
        }
    }
}

// ─── PyO3 bindings ────────────────────────────────────────────────────────────

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
/// import cyclosynth
/// # Clifford+T (default); sqrt_t=True selects Clifford+√T (denser, often fewer gates).
/// synth = cyclosynth.Synthesizer(epsilon=1e-5)
///
/// # Angles only — U3 (theta, phi, lambda) or ZYZ Euler (alpha, beta, gamma);
/// # each a float or an exact-pi string like "pi/64".
/// result = synth.synthesize_u3("pi/64", 0, 0)
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
        Ok(self.run_zyz(parse_angle(phi)?, parse_angle(theta)?, parse_angle(lam)?))
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
        Ok(self.run_zyz(parse_angle(alpha)?, parse_angle(beta)?, parse_angle(gamma)?))
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
    /// Build the SU(2) target from ZYZ angles and run the search, passing the
    /// exact MPFR target column so the deep-ε path can align below the f64 ULP
    /// (exact for rational-π angles).
    fn run_zyz(
        &self,
        alpha: crate::synthesis::angle::Angle,
        beta: crate::synthesis::angle::Angle,
        gamma: crate::synthesis::angle::Angle,
    ) -> Option<PySynthResult> {
        use crate::synthesis::angle::{su2_col_mpfr, su2_from_zyz};
        let mat = su2_from_zyz(alpha.to_radians_f64(), beta.to_radians_f64(), gamma.to_radians_f64());
        // 384 bits covers the search precision (≈6·log₂(1/ε)) for any ε ≳ 1e-19.
        let col = su2_col_mpfr(alpha, beta, gamma, 384);
        let q_weight = self.inner.q_weight();
        self.inner.synthesize_su2_col(mat, &col).map(|r| PySynthResult {
            gates: r.gates,
            lde: r.lde,
            distance: r.distance,
            q_weight,
        })
    }
}

/// Parse one angle argument — a Python float/int, or a string (the `pi`
/// rational forms of [`crate::synthesis::angle::parse_angle_str`]).
#[cfg(feature = "python")]
fn parse_angle(obj: &Bound<'_, PyAny>) -> PyResult<crate::synthesis::angle::Angle> {
    use crate::synthesis::angle::{parse_angle_str, Angle};
    if let Ok(x) = obj.extract::<f64>() {
        return Ok(Angle::Rad(x));
    }
    let s: String = obj.extract().map_err(|_| {
        PyErr::new::<pyo3::exceptions::PyTypeError, _>("angle must be a float or string")
    })?;
    parse_angle_str(&s).map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)
}
