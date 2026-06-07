mod scratch {
    pub(super) use crate::synthesis::lattice_upsilon::scratch::*;
}

mod lll {
    pub(super) use crate::synthesis::lattice_upsilon::lll::{
        gram_update_size_reduce, gram_update_swap,
    };
}

mod lll_f64 {
    pub(super) use crate::synthesis::lattice_upsilon::lll_f64::cfa_row_f64;
}

#[cfg(test)]
mod integer {
    use crate::synthesis::lattice_upsilon::scratch::IntScratch16;
    use std::sync::atomic::AtomicBool;

    #[allow(clippy::too_many_arguments)]
    pub(super) fn phase1_with_stop<F>(
        scratch: &mut IntScratch16,
        y: &[f64; 16],
        k: u32,
        eps: f64,
        max_leaves: u64,
        budget_hit: &AtomicBool,
        should_stop: F,
        _euclid_chol: Option<()>,
        _target_norm_sq_i64: Option<i64>,
    ) -> Vec<[i64; 16]>
    where
        F: FnMut(&[i64; 16]) -> bool,
    {
        let v = [y[0], y[1], y[2], y[3]];
        crate::synthesis::lattice_upsilon::integer::phase1_with_stop(
            scratch,
            v,
            k,
            eps,
            max_leaves,
            budget_hit,
            should_stop,
        )
    }
}

#[cfg(test)]
mod se {
    pub(super) use crate::synthesis::lattice_upsilon::se::det16_exact;
}

#[path = "../lattice_zeta/bkz.rs"]
mod zeta_bkz;

pub use zeta_bkz::*;
