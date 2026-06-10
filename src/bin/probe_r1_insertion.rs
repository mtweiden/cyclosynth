//! Experiment E5 (docs/design_certified_optimal_cost.md, Tier 1.5):
//! anchored r=1 recursive synthesis. The r=1 stratum is exactly
//! { A · Q · B : A, B Clifford+T }. Anchor a *short, exact* MA prefix A
//! (or mirrored suffix), solve the other side with the fast 8D
//! Clifford+T backend at full ε (the anchor is exact, so no ε split):
//!
//!     candidate cost = T(A) + 3.5 + T(B),   B ≈ su2(Q† · A† · V)
//!
//! Compares, per random target: t₀ (pure Clifford+T), the default √T
//! hybrid, and the best anchored-r1 candidate.
//!
//! Args: <eps> [<n_targets> [<seed> [<j_max> [--skip-hybrid]]]]
//! Defaults: eps=1e-5, n=8, seed=0xC0FFEE, j_max=2.

use cyclosynth::synthesis::clifford_t::{build_l, SynthesizerT};
use cyclosynth::synthesis::clifford_sqrt_t::SynthesizerQ;
use cyclosynth::synthesis::distance::Mat2;
use num_complex::Complex;
use rayon::prelude::*;
use std::f64::consts::PI;
use std::time::Instant;

type C64 = Complex<f64>;

/// SplitMix64 (matches probe_t_vs_qt).
struct Xs(u64);
impl Xs {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 { (self.next() >> 11) as f64 * (1.0 / (1u64 << 53) as f64) }
    fn range(&mut self, lo: f64, hi: f64) -> f64 { lo + (hi - lo) * self.unit() }
}

fn u3(theta: f64, phi: f64, lam: f64) -> Mat2 {
    let (c, s) = ((theta / 2.0).cos(), (theta / 2.0).sin());
    let eilam = C64::from_polar(1.0, lam);
    let eiphi = C64::from_polar(1.0, phi);
    let m = [
        [C64::new(c, 0.0), -eilam * s],
        [eiphi * s, eiphi * eilam * c],
    ];
    let g = C64::from_polar(1.0, -(phi + lam) / 2.0);
    [[m[0][0] * g, m[0][1] * g], [m[1][0] * g, m[1][1] * g]]
}

fn matmul(a: &Mat2, b: &Mat2) -> Mat2 {
    [
        [a[0][0]*b[0][0] + a[0][1]*b[1][0], a[0][0]*b[0][1] + a[0][1]*b[1][1]],
        [a[1][0]*b[0][0] + a[1][1]*b[1][0], a[1][0]*b[0][1] + a[1][1]*b[1][1]],
    ]
}

fn dagger(m: &Mat2) -> Mat2 {
    [
        [m[0][0].conj(), m[1][0].conj()],
        [m[0][1].conj(), m[1][1].conj()],
    ]
}

/// Normalize the global phase so det = 1 (diamond distance is
/// phase-invariant; the T backend needs the SU(2) det coset).
fn su2(m: &Mat2) -> Mat2 {
    let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    // |det| = 1 for a unitary, so 1/√det = conj(√det).
    let g = det.sqrt().conj();
    [[m[0][0] * g, m[0][1] * g], [m[1][0] * g, m[1][1] * g]]
}

fn t_count(gates: &Option<String>) -> usize {
    gates.as_deref().map(|g| g.chars().filter(|&c| c == 'T').count()).unwrap_or(usize::MAX)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let eps: f64 = args.first().and_then(|s| s.parse().ok()).unwrap_or(1e-5);
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(8);
    let seed: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0xC0FFEE);
    let j_max: u32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(2);
    let skip_hybrid = args.iter().any(|a| a == "--skip-hybrid");

    // Q = diag(1, e^{iπ/8}) as a float matrix (exact circuit element).
    let qf: Mat2 = [
        [C64::new(1.0, 0.0), C64::new(0.0, 0.0)],
        [C64::new(0.0, 0.0), C64::from_polar(1.0, PI / 8.0)],
    ];
    let qf_dag = dagger(&qf);

    let mut rng = Xs(seed);
    let targets: Vec<(f64, f64, f64)> = (0..n).map(|_| (
        rng.range(0.2, PI - 0.2),
        rng.range(0.1, 2.0 * PI - 0.1),
        rng.range(0.1, 2.0 * PI - 0.1),
    )).collect();

    println!("E5 anchored-r1: ε={eps:e}, n={n}, seed=0x{seed:X}, j_max={j_max}");
    println!("{:>3} | {:>5} | {:>6} | {:>6} {:>9} | winner", "#", "t0", "hybrid", "r1", "(j,side)");

    let (mut r1_beats_t0, mut r1_beats_hybrid) = (0usize, 0usize);
    let mut sum_t0 = 0.0; let mut sum_hyb = 0.0; let mut sum_best = 0.0;
    for (i, &(th, ph, la)) in targets.iter().enumerate() {
        let v = u3(th, ph, la);

        let rt = SynthesizerT::new(eps).synthesize(v);
        let t0 = t_count(&rt.as_ref().map(|r| r.gates.clone()).unwrap_or(None)) as f64;

        let hybrid_cost = if skip_hybrid { f64::NAN } else {
            SynthesizerQ::new(eps)
                .synthesize(v)
                .and_then(|r| r.gates)
                .map(|g| {
                    let t = g.chars().filter(|&c| c == 'T').count() as f64;
                    let q = g.chars().filter(|&c| c == 'Q').count() as f64;
                    t + 3.5 * q
                })
                .unwrap_or(f64::NAN)
        };

        // Anchored r=1: front anchors A (U = A·Q·B) and mirrored back
        // anchors (U = B·Q·A  ⟸ anchor the rightmost syllables).
        let t_r1 = Instant::now();
        let mut best: (f64, u32, &str) = (f64::INFINITY, 0, "-");
        for j in 0..=j_max {
            let prefixes = build_l(j);
            let front: Option<(f64, f64)> = prefixes
                .par_iter()
                .map(|a| {
                    let af = a.to_float();
                    // target' = Q† · A† · V
                    let tgt = su2(&matmul(&qf_dag, &matmul(&dagger(&af), &v)));
                    SynthesizerT::new(eps).synthesize(tgt).map(|r| {
                        (j as f64 + 3.5 + t_count(&r.gates) as f64, r.distance)
                    })
                })
                .flatten()
                .min_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
            let back: Option<(f64, f64)> = prefixes
                .par_iter()
                .map(|a| {
                    let af = a.to_float();
                    // U = B · Q · A  ⟹  B ≈ V · A† · Q†
                    let tgt = su2(&matmul(&matmul(&v, &dagger(&af)), &qf_dag));
                    SynthesizerT::new(eps).synthesize(tgt).map(|r| {
                        (j as f64 + 3.5 + t_count(&r.gates) as f64, r.distance)
                    })
                })
                .flatten()
                .min_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
            if let Some((c, _)) = front {
                if c < best.0 { best = (c, j, "front"); }
            }
            if let Some((c, _)) = back {
                if c < best.0 { best = (c, j, "back"); }
            }
        }
        let r1_wall = t_r1.elapsed().as_secs_f64();

        if best.0 < t0 { r1_beats_t0 += 1; }
        if !hybrid_cost.is_nan() && best.0 < hybrid_cost { r1_beats_hybrid += 1; }
        sum_t0 += t0; sum_hyb += hybrid_cost; sum_best += best.0.min(hybrid_cost.min(t0));
        let winner = if best.0 < hybrid_cost.min(t0) { "r1" }
                     else if hybrid_cost < t0 { "hybrid" } else { "T" };
        println!("{i:>3} | {t0:>5.1} | {hybrid_cost:>6.1} | {:>6.1} ({},{})  {winner}  [r1 {r1_wall:.1}s]",
            best.0, best.1, best.2);
    }
    println!("\nmean: t0={:.1} hybrid={:.1} min(all)={:.1}",
        sum_t0 / n as f64, sum_hyb / n as f64, sum_best / n as f64);
    println!("r1 beats t0 on {r1_beats_t0}/{n}, beats hybrid on {r1_beats_hybrid}/{n}");
}
