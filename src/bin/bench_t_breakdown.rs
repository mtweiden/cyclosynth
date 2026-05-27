//! Print the LLL vs SE time breakdown for Clifford+T synthesis.
//! Used to decide whether BKZ would help: if LLL >> SE, BKZ adds
//! more LLL work without recouping anything from SE.

use cyclosynth::synthesis::clifford_t::SynthesizerT;
use cyclosynth::synthesis::diag;
use num_complex::Complex;
use std::f64::consts::PI;
use std::time::Instant;

type C64 = Complex<f64>;
type Mat2 = [[C64; 2]; 2];

fn mat_mul(a: Mat2, b: Mat2) -> Mat2 {
    [
        [
            a[0][0] * b[0][0] + a[0][1] * b[1][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1],
        ],
        [
            a[1][0] * b[0][0] + a[1][1] * b[1][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1],
        ],
    ]
}
fn rz(t: f64) -> Mat2 {
    [
        [C64::from_polar(1.0, -t / 2.0), C64::new(0.0, 0.0)],
        [C64::new(0.0, 0.0), C64::from_polar(1.0, t / 2.0)],
    ]
}
fn ry(t: f64) -> Mat2 {
    let c = (t / 2.0).cos();
    let s = (t / 2.0).sin();
    [
        [C64::new(c, 0.0), C64::new(-s, 0.0)],
        [C64::new(s, 0.0), C64::new(c, 0.0)],
    ]
}
fn u3(a: f64, b: f64, c: f64) -> Mat2 {
    mat_mul(mat_mul(rz(a), ry(b)), rz(c))
}
fn xorshift64(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}
fn rand_angle(s: &mut u64) -> f64 {
    let b = xorshift64(s) >> 11;
    (b as f64) / ((1u64 << 53) as f64) * 2.0 * PI
}

fn main() {
    std::env::set_var("CYCLOSYNTH_TRACE", "1");
    let mut state: u64 = 0xC0FFEEBAADD0E;
    let n = 4;
    let targets: Vec<Mat2> = (0..n)
        .map(|_| {
            u3(
                rand_angle(&mut state),
                rand_angle(&mut state),
                rand_angle(&mut state),
            )
        })
        .collect();

    for &eps in &[1e-4, 1e-5, 1e-6, 1e-7, 1e-8_f64] {
        diag::reset_all();
        let t = Instant::now();
        let synth = SynthesizerT::new(eps);
        for target in &targets {
            let _ = synth.synthesize(*target);
        }
        let wall_ms = t.elapsed().as_secs_f64() * 1000.0;
        let s = diag::snapshot();
        let stage = s.t_build_ms + s.t_lll_ms + s.t_cholesky_ms + s.t_lu_ms + s.t_se_ms;
        let pct = |x: f64| if stage > 0.0 { 100.0 * x / stage } else { 0.0 };
        println!(
            "ε={:e}  wall {:>8.1} ms  | lll {:>7.1} ms ({:>4.1}%)  se {:>7.1} ms ({:>4.1}%)  others {:>5.1} ms",
            eps, wall_ms,
            s.t_lll_ms, pct(s.t_lll_ms),
            s.t_se_ms, pct(s.t_se_ms),
            stage - s.t_lll_ms - s.t_se_ms,
        );
    }
}
