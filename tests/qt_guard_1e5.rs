//! Gate-6 √T integration guard for the Z[ω] fix package: replicates
//! `probe_t_vs_qt 1e-5 8 12648430 optimal 2` (which could not be run
//! directly while the box was serialized) through the public API and
//! prints the per-target √T cost columns + totals. Expected from the
//! pre-fix reference: √T total cost = 308.5, wall ≈ 5.3 s clean.
//!
//! Run: `cargo test --release --test qt_guard_1e5 -- --ignored --nocapture`

use cyclosynth::synthesis::clifford_sqrt_t::SynthesizerQ;
use cyclosynth::synthesis::distance::Mat2;
use num_complex::Complex;
use std::time::Instant;

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
fn qt_guard_8_targets_optimal_w2() {
    let eps = 1e-5_f64;
    let mut rng = Xs(12648430);
    use std::f64::consts::PI;
    let targets: Vec<(f64, f64, f64)> = (0..8)
        .map(|_| {
            (
                rng.range(0.2, PI - 0.2),
                rng.range(0.1, 2.0 * PI - 0.1),
                rng.range(0.1, 2.0 * PI - 0.1),
            )
        })
        .collect();

    let mut total_cost = 0.0_f64;
    let t0 = Instant::now();
    for (i, &(th, ph, la)) in targets.iter().enumerate() {
        let target = u3(th, ph, la);
        let r = SynthesizerQ::new(eps)
            .with_optimize_cost(true)
            .with_optimal_lde_window(2)
            .synthesize(target);
        let (t, q, lde) = r
            .as_ref()
            .map(|r| {
                let g = r.gates.as_deref().unwrap_or("");
                (
                    g.chars().filter(|&c| c == 'T').count(),
                    g.chars().filter(|&c| c == 'Q').count(),
                    r.lde,
                )
            })
            .unwrap_or((0, 0, 0));
        let cost = t as f64 + 3.5 * q as f64;
        total_cost += cost;
        println!("QT {i} lde={lde:>2} T={t:>2} Q={q:>2} cost={cost:>5.1}");
    }
    let wall = t0.elapsed().as_secs_f64();
    println!("QT TOTAL cost={total_cost:.1} wall={wall:.2}s (reference: 308.5, ~5.3s clean)");
}
