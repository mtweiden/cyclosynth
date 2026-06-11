//! Gate-2 identity harness for the Z[ω] fix package: synthesize the same
//! 12 U3 targets `probe_t_vs_qt 1e-5 12 0xC0FFEE` uses, through the
//! Clifford+T backend only, and print the per-target (lde, T-count,
//! distance) columns. Run against the baseline and the fixed build and
//! diff the output — the T side is deterministic first-hit, so the
//! columns must be identical (an abort-race tie may change the gate
//! string/distance at unchanged lde; that must be reported).
//!
//! Uses ONLY the public API so the same file compiles against both trees.
//! Run: `cargo test --release --test t_identity_1e5 -- --ignored --nocapture`

use cyclosynth::synthesis::clifford_t::SynthesizerT;
use cyclosynth::synthesis::distance::Mat2;
use num_complex::Complex;

/// Deterministic SplitMix64 — identical to probe_t_vs_qt's generator.
struct Xs(u64);
impl Xs {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }
}

type C64 = Complex<f64>;

fn u3(theta: f64, phi: f64, lam: f64) -> Mat2 {
    let (c, s) = ((theta / 2.0).cos(), (theta / 2.0).sin());
    let eilam = C64::from_polar(1.0, lam);
    let eiphi = C64::from_polar(1.0, phi);
    let m = [
        [C64::new(c, 0.0), -eilam * s],
        [eiphi * s, eiphi * eilam * c],
    ];
    let g = C64::from_polar(1.0, -(phi + lam) / 2.0);
    [
        [m[0][0] * g, m[0][1] * g],
        [m[1][0] * g, m[1][1] * g],
    ]
}

#[test]
#[ignore]
fn t_identity_12_targets_1e5() {
    let eps = 1e-5_f64;
    let mut rng = Xs(0xC0FFEE);
    use std::f64::consts::PI;
    let targets: Vec<(f64, f64, f64)> = (0..12)
        .map(|_| {
            (
                rng.range(0.2, PI - 0.2),
                rng.range(0.1, 2.0 * PI - 0.1),
                rng.range(0.1, 2.0 * PI - 0.1),
            )
        })
        .collect();

    for (i, &(th, ph, la)) in targets.iter().enumerate() {
        let target = u3(th, ph, la);
        let r = SynthesizerT::new(eps).synthesize(target);
        match r {
            Some(r) => {
                let t_count = r
                    .gates
                    .as_deref()
                    .map(|g| g.chars().filter(|&c| c == 'T').count())
                    .unwrap_or(0);
                println!(
                    "IDENT {i:>2} lde={:>2} T={t_count:>3} dist={:.6e}",
                    r.lde, r.distance
                );
            }
            None => println!("IDENT {i:>2} FAILED"),
        }
    }
}
