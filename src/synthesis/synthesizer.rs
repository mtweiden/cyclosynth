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
//! The two flows currently use *different algorithms* (8D MA-prefix
//! decomposition vs single-shot 16D LLL+SE), so they can't be expressed
//! cleanly as a single generic `Synthesizer<R: GateRing>`. This wrapper
//! gives users a single API today while the internals keep their own
//! optimised code paths. Once the algorithmic structure converges (see
//! the `project_synthesizer_generic_followup` memory), the wrapper will
//! be replaced with monomorphised generic instantiations and `sqrt_t`
//! will keep working as a public-API parameter.

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

    pub fn epsilon(&self) -> f64 {
        match &self.inner {
            Backend::T(s) => s.epsilon,
            Backend::Q(s) => s.epsilon,
        }
    }

    pub fn max_lde(&self) -> u32 {
        match &self.inner {
            Backend::T(s) => s.max_lde,
            Backend::Q(s) => s.max_lde,
        }
    }

    pub fn min_lde(&self) -> u32 {
        match &self.inner {
            Backend::T(s) => s.min_lde,
            Backend::Q(s) => s.min_lde,
        }
    }

    pub fn sqrt_t(&self) -> bool {
        matches!(&self.inner, Backend::Q(_))
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
    /// Denominator exponent of the synthesized unitary.
    #[pyo3(get)]
    pub lde: u32,
    /// Diamond distance from the synthesized unitary to the target.
    #[pyo3(get)]
    pub distance: f64,
}

#[cfg(feature = "python")]
#[pymethods]
impl PySynthResult {
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
    #[new]
    #[pyo3(signature = (epsilon, *, sqrt_t=false, max_lde=None, min_lde=None))]
    fn new(
        epsilon: f64,
        sqrt_t: bool,
        max_lde: Option<u32>,
        min_lde: Option<u32>,
    ) -> Self {
        let mut s = Synthesizer::new(epsilon, sqrt_t);
        if let Some(v) = max_lde {
            s = s.with_max_lde(v);
        }
        if let Some(v) = min_lde {
            s = s.with_min_lde(v);
        }
        Self { inner: s }
    }

    /// Synthesize `target` (a 2×2 `np.complex128` array).
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
        Ok(self.inner.synthesize(mat).map(|r| PySynthResult {
            gates: r.gates,
            lde: r.lde,
            distance: r.distance,
        }))
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
