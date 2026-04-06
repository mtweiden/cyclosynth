pub mod algebra;
pub mod ratio;
pub mod matrix;
pub mod bloch;

use pyo3::prelude::*;

#[pymodule]
fn cyclosynth(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Algebra types
    m.add_class::<algebra::RingRoot2>()?;
    m.add_class::<algebra::RingRootRoot2Plus2>()?;
    m.add_class::<algebra::DyadicComplexNumber>()?;
    m.add_class::<algebra::DOmega>()?;

    // Ratio types
    m.add_class::<ratio::IntegerRatio>()?;
    m.add_class::<ratio::AlgebraicIntegerOverRoot2>()?;
    m.add_class::<ratio::AlgebraicIntegerOverRootRoot2Plus2>()?;

    // Matrix types
    m.add_class::<matrix::U2Matrix>()?;
    m.add_class::<matrix::SO3Matrix>()?;

    // Bloch decomposer
    m.add_class::<bloch::BlochDecomposer>()?;

    // Functions
    m.add_function(wrap_pyfunction!(bloch::translation::translate_decomposition, m)?)?;

    Ok(())
}
