//! cyclosynth — single-qubit gate synthesis: approximate a target 2×2 unitary
//! by a Clifford+T or Clifford+√T circuit, minimizing `T_count + 3.5·Q_count`
//! (arXiv:2510.05816). The entry point is [`synthesis::Synthesizer`]; see the
//! [`synthesis`] module for the domain glossary and the algorithm overview.

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
    m.add_class::<rings::zomega::PyZOmega>()?;
    m.add_class::<rings::zzeta::PyZZeta>()?;
    m.add_class::<matrix::u2::PyU2>()?;
    m.add_class::<synthesis::decomposer::PyBlochDecomposer>()?;
    m.add_class::<synthesis::synthesizer::PySynthesizer>()?;
    m.add_class::<synthesis::synthesizer::PySynthResult>()?;
    Ok(())
}
