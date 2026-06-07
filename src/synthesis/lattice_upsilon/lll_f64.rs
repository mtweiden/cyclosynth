use super::scratch::IntScratch16;

#[inline]
pub fn cfa_row_f64(scratch: &mut IntScratch16, i: usize) {
    for j in 0..i {
        let mut r = crate::synthesis::lattice::lll::i256_to_f64(scratch.gram[i][j]);
        for k in 0..j {
            r -= scratch.mu_bar_f64[j][k] * scratch.r_bar_f64[i][k];
        }
        scratch.r_bar_f64[i][j] = r;
        let r_jj = scratch.r_bar_f64[j][j];
        scratch.mu_bar_f64[i][j] = if r_jj.abs() < 1e-300 { 0.0 } else { r / r_jj };
    }
    scratch.s_bar_f64[i][0] = crate::synthesis::lattice::lll::i256_to_f64(scratch.gram[i][i]);
    for j in 1..=i {
        scratch.s_bar_f64[i][j] = scratch.s_bar_f64[i][j - 1]
            - scratch.mu_bar_f64[i][j - 1] * scratch.r_bar_f64[i][j - 1];
    }
    scratch.r_bar_f64[i][i] = scratch.s_bar_f64[i][i];
}
