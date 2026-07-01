//! Exact matrix types over the synthesis rings: 2×2 unitaries ([`U2`]) and
//! SO(3) rotations ([`SO3`]), with their Clifford+T (`*T`) and Clifford+√T
//! (`*Q`) ring instantiations.

pub mod so3;
pub mod u2;

pub use so3::{
    rz_neg, rx_neg, ry_neg,
    rz_neg_q, rx_neg_q, ry_neg_q,
};
pub use u2::{U2T, U2Q};