pub mod types;
pub mod zomega;
pub mod zzeta;

pub use types::{Int, Float};
pub use zomega::ZOmega;
pub use zzeta::ZZeta;


#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn tilers(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<zomega::PyZOmega>()?;
    m.add_class::<zzeta::PyZZeta>()?;
    Ok(())
}
