//! cyclosynth — single-qubit gate synthesis: approximate a target 2×2 unitary
//! by a Clifford+T or Clifford+√T circuit. See the [`synthesis`] module for
//! the domain glossary and the algorithm overview.

pub mod rings;
pub mod matrix;
pub mod synthesis;

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Python extension module. Built when the `python` Cargo feature is enabled
/// (via `maturin develop` / `maturin build`); see `pyproject.toml`.
#[cfg(feature = "python")]
#[pymodule]
fn cyclosynth(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // The public surface is exactly the synthesizer and its result. The ring /
    // U2 / decomposer types are internal plumbing and are intentionally not
    // exported (they have no standalone Python use).
    m.add_class::<synthesis::synthesizer::PySynthesizer>()?;
    m.add_class::<synthesis::synthesizer::PySynthResult>()?;
    // D&C / inner-cap diagnostics (trace-only).
    #[cfg(feature = "trace")]
    {
        m.add_function(wrap_pyfunction!(crate::synthesis::diag::decompose_gates_t, m)?)?;
        m.add_function(wrap_pyfunction!(crate::synthesis::diag::trace_inner, m)?)?;
        m.add_function(wrap_pyfunction!(crate::synthesis::diag::diag_inner_cap, m)?)?;
    }
    Ok(())
}
