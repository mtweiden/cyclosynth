//! Quick driver for `synthesize_certified` — fast iteration + trace
//! visibility without the cargo-test rebuild cycle.
//! Args: <which: t|q|rz> [<k_max> [<eps>]]
use cyclosynth::matrix::u2::U2Q;
use cyclosynth::synthesis::clifford_sqrt_t::SynthesizerQ;
use cyclosynth::synthesis::distance::Mat2;
use num_complex::Complex;
use std::f64::consts::PI;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let which = args.first().map(|s| s.as_str()).unwrap_or("t").to_string();
    let k_max: u32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(2);
    let eps: f64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1e-3);

    let scale = |m: [[Complex<f64>; 2]; 2], phi: f64| -> Mat2 {
        let g = Complex::from_polar(1.0, phi);
        [[m[0][0]*g, m[0][1]*g], [m[1][0]*g, m[1][1]*g]]
    };
    let target: Mat2 = match which.as_str() {
        "q" => scale((U2Q::h()*U2Q::q()*U2Q::h()).reduced().to_float(), -PI/16.0),
        "rz" => [
            [Complex::from_polar(1.0, -0.35), Complex::new(0.0, 0.0)],
            [Complex::new(0.0, 0.0), Complex::from_polar(1.0, 0.35)],
        ],
        _ => scale(U2Q::t().to_float(), -PI/16.0),
    };

    let t0 = Instant::now();
    match SynthesizerQ::new(eps).synthesize_certified(target, k_max) {
        Some((r, c)) => println!(
            "{which} k={k_max} eps={eps:e} → cost={} HU in [{}, {}] certified={} gates={:?} dist={:.2e} t={:.2}s",
            c.upper_half_units, c.lower_half_units, c.upper_half_units,
            c.certified_optimal, r.gates, r.distance, t0.elapsed().as_secs_f64()
        ),
        None => println!("{which} k={k_max} → None t={:.2}s", t0.elapsed().as_secs_f64()),
    }
}
