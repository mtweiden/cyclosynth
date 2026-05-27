//! Fixed-target release timing check for n=4 (ZOmega) and n=8 (ZZeta).
//!
//! Usage:
//!   cargo run --release --bin deep_eps_check -- t
//!   cargo run --release --bin deep_eps_check -- q

use cyclosynth::matrix::{U2Q, U2T};
use cyclosynth::synthesis::clifford_sqrt_t::SynthesizerQ;
use cyclosynth::synthesis::clifford_t::SynthesizerT;
use cyclosynth::synthesis::distance::{diamond_distance_float, diamond_distance_u2q_float, Mat2};
use num_complex::Complex64;
use std::time::Instant;

fn rz(theta: f64) -> Mat2 {
    [
        [
            Complex64::from_polar(1.0, -theta / 2.0),
            Complex64::new(0.0, 0.0),
        ],
        [
            Complex64::new(0.0, 0.0),
            Complex64::from_polar(1.0, theta / 2.0),
        ],
    ]
}

fn gates_to_u2t(gates: &str) -> U2T {
    gates.chars().fold(U2T::eye(), |acc, ch| {
        let gate = match ch {
            'H' => U2T::h(),
            'S' => U2T::s(),
            'T' => U2T::t(),
            'X' => U2T::x(),
            'Y' => U2T::y(),
            'Z' => U2T::z(),
            'I' => U2T::eye(),
            other => panic!("unexpected n=4 gate {other:?}"),
        };
        acc * gate
    })
}

fn gates_to_u2q(gates: &str) -> U2Q {
    gates.chars().fold(U2Q::eye(), |acc, ch| {
        let gate = match ch {
            'H' => U2Q::h(),
            'S' => U2Q::s(),
            'T' => U2Q::t(),
            'Q' => U2Q::q(),
            'X' => U2Q::x(),
            'Y' => U2Q::y(),
            'Z' => U2Q::z(),
            'I' => U2Q::eye(),
            other => panic!("unexpected n=8 gate {other:?}"),
        };
        acc * gate
    })
}

fn run_t() {
    let target = rz(0.3);
    for &eps in &[1e-3_f64, 1e-4, 1e-5] {
        let synth = SynthesizerT::new(eps).with_max_lde(80);
        let t = Instant::now();
        let result = synth.synthesize(target);
        let elapsed = t.elapsed();
        match result {
            Some(r) => {
                let gates = r.gates.as_deref().unwrap_or("");
                let actual = diamond_distance_float(&gates_to_u2t(gates).to_float(), &target);
                println!(
                    "n=4 eps={eps:.0e}: lde={} claimed={:.3e} actual={:.3e} t={:.2}s",
                    r.lde,
                    r.distance,
                    actual,
                    elapsed.as_secs_f64()
                );
            }
            None => println!("n=4 eps={eps:.0e}: FAILED t={:.2}s", elapsed.as_secs_f64()),
        }
    }
}

fn run_q() {
    let target = rz(0.3);
    for &eps in &[1e-3_f64, 1e-4, 1e-5] {
        let synth = SynthesizerQ::new(eps).with_max_lde(30);
        let t = Instant::now();
        let result = synth.synthesize(target);
        let elapsed = t.elapsed();
        match result {
            Some(r) => {
                let gates = r.gates.as_deref().unwrap_or("");
                let actual = diamond_distance_u2q_float(&gates_to_u2q(gates), &target);
                println!(
                    "n=8 eps={eps:.0e}: lde={} claimed={:.3e} actual={:.3e} t={:.2}s",
                    r.lde,
                    r.distance,
                    actual,
                    elapsed.as_secs_f64()
                );
            }
            None => println!("n=8 eps={eps:.0e}: FAILED t={:.2}s", elapsed.as_secs_f64()),
        }
    }
}

fn main() {
    match std::env::args().nth(1).as_deref() {
        Some("t") | Some("n4") | Some("4") => run_t(),
        Some("q") | Some("n8") | Some("8") => run_q(),
        _ => {
            eprintln!("usage: deep_eps_check <t|q>");
            std::process::exit(2);
        }
    }
}
