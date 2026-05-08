//! Probe: synthesize Rz(0.3) at ε=1e-8 in Clifford+T, embed U2T → U2Q,
//! and verify the MPFR-direct distance fn says the embedded result is
//! still within ε.
//!
//! Two questions:
//!   (a) Does our new `diamond_distance_u2q_float` correctly recognize
//!       a known-good solution (the CT one, embedded)? If not, the fn
//!       has a bug.
//!   (b) Establishes an upper bound on lde Q-search must reach — Q's
//!       lde should be ≤ CT's lde because every CT gate sequence is
//!       valid in Q (ω = ζ²).

use cyclosynth::matrix::{U2T, U2Q};
use cyclosynth::rings::ZZeta;
use cyclosynth::synthesis::clifford_t::SynthesizerT;
use cyclosynth::synthesis::diamond_distance_u2q_float;
use num_complex::Complex;
use std::time::Instant;

type C64 = Complex<f64>;
type Mat2 = [[C64; 2]; 2];

fn rz(t: f64) -> Mat2 {
    [[C64::from_polar(1.0, -t / 2.0), C64::new(0.0, 0.0)],
     [C64::new(0.0, 0.0), C64::from_polar(1.0, t / 2.0)]]
}

fn embed_u2t_in_u2q(u: &U2T) -> U2Q {
    U2Q::new(
        ZZeta::from_zomega(u.u11.a, u.u11.b, u.u11.c, u.u11.d),
        ZZeta::from_zomega(u.u12.a, u.u12.b, u.u12.c, u.u12.d),
        ZZeta::from_zomega(u.u21.a, u.u21.b, u.u21.c, u.u21.d),
        ZZeta::from_zomega(u.u22.a, u.u22.b, u.u22.c, u.u22.d),
        u.k,
    )
}

fn main() {
    let target = rz(0.3);
    let eps = 1e-8_f64;

    eprintln!("─── Step 1: Clifford+T at ε=1e-8 ───");
    let t = Instant::now();
    let synth_t = SynthesizerT::new(eps);
    let r_t = synth_t.synthesize(target).expect("CT must succeed");
    let dt = t.elapsed().as_secs_f64();
    eprintln!("  CT: lde={}, dist={:.3e}, took {:.2} s", r_t.lde, r_t.distance, dt);

    eprintln!("─── Step 2: rebuild U2T from gate string and embed → U2Q ───");
    // We don't expose the U2T directly from SynthResult; let's rebuild from gates.
    let gates = r_t.gates.expect("CT result has gates");
    let mut u2t = U2T::eye();
    for ch in gates.chars() {
        u2t = match ch {
            'T' => u2t * U2T::t(),
            'H' => u2t * U2T::h(),
            'S' => u2t * U2T::s(),
            'X' => u2t * U2T::x(),
            'Y' => u2t * U2T::y(),
            'Z' => u2t * U2T::z(),
            _ => panic!("unexpected gate {ch}"),
        };
    }
    let u2q: U2Q = embed_u2t_in_u2q(&u2t);
    eprintln!("  U2Q: k={}", u2q.k);

    eprintln!("─── Step 3: distance check via new MPFR-direct fn ───");
    let dist_q = diamond_distance_u2q_float(&u2q, &target);
    eprintln!("  diamond_distance_u2q_float = {:.6e}", dist_q);
    if dist_q < eps {
        eprintln!("  ✓ recognized as ε-good (matches CT result)");
    } else {
        eprintln!("  ✗ FAILED: distance {} >= ε={}", dist_q, eps);
    }
}
