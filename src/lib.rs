pub mod matrix;
pub mod rings;
pub mod synthesis;

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Python extension module. Built when the `python` Cargo feature is enabled
/// (via `maturin develop` / `maturin build`); see `pyproject.toml`.
#[cfg(feature = "python")]
#[pymodule]
fn cyclosynth(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<rings::zomega::PyZOmega>()?;
    m.add_class::<rings::zomicron::PyZOmicron>()?;
    m.add_class::<rings::zzeta::PyZZeta>()?;
    m.add_class::<matrix::u2::PyU2>()?;
    m.add_class::<synthesis::decomposer::PyBlochDecomposer>()?;
    m.add_class::<synthesis::synthesizer::PySynthesizer>()?;
    m.add_class::<synthesis::synthesizer::PySynthResult>()?;
    Ok(())
}
