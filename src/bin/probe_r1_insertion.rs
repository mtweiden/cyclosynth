//! Experiment E5/E5b (docs/design_certified_optimal_cost.md, Tier 1.5):
//! anchored **recursive** r-stratified synthesis.
//!
//! The r ≥ 1 strata are { A₁·Q·A₂·Q·…·B } with Aᵢ, B Clifford+T.
//! Anchor a short, *exact* MA prefix (or mirrored suffix) plus one Q,
//! and recurse on the residual; the base case solves the residual with
//! the fast 8D Clifford+T backend. The anchors are exact circuit
//! elements, so at every depth there is exactly ONE approximation
//! problem (the leaf T-synthesis) at the FULL ε — recursion adds no
//! ε-splitting penalty, only gate cost:
//!
//!     synth(V, budget) = min( minT(V),
//!         min over anchors A·Q: j + 3.5 + synth(Q†A†V, budget−j−3.5) )
//!
//! Branching control: at each node every anchor is scored by its
//! residual's r=0 T-count (that *is* the r=1 evaluation); recursion
//! descends only into the `top_k` cheapest residuals, and only while
//! the remaining budget can pay for a Q plus at least one T.
//!
//! Args: <eps> [<n> [<seed> [<j_max> [<depth> [<top_k> [--skip-hybrid]]]]]]
//! Defaults: eps=1e-5, n=8, seed=0xC0FFEE, j_max=2, depth=2, top_k=8.

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

const Q_COST: f64 = 3.5;

struct Cfg {
    eps: f64,
    j_max: u32,
    top_k: usize,
}

/// One anchored step: all (anchor, side) residuals of `v`, each scored
/// by its r=0 T-count. Returns (residual target, anchor gate cost,
/// r1 candidate total cost, label).
fn anchored_children(v: &Mat2, qf_dag: &Mat2, cfg: &Cfg) -> Vec<(Mat2, f64, f64, String)> {
    let mut out = Vec::new();
    for j in 0..=cfg.j_max {
        let prefixes = build_l(j);
        let mut batch: Vec<(Mat2, f64, f64, String)> = prefixes
            .par_iter()
            .flat_map_iter(|a| {
                let af = a.to_float();
                // Front: U = A·Q·B  ⟹  B ≈ Q†·A†·V
                let tf = su2(&matmul(qf_dag, &matmul(&dagger(&af), v)));
                // Back:  U = B·Q·A  ⟹  B ≈ V·A†·Q†
                let tb = su2(&matmul(&matmul(v, &dagger(&af)), qf_dag));
                [(tf, "f"), (tb, "b")].into_iter().map(move |(tgt, side)| {
                    let anchor_cost = j as f64 + Q_COST;
                    (tgt, anchor_cost, side)
                })
            })
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(tgt, anchor_cost, side)| {
                let r0 = SynthesizerT::new(cfg.eps)
                    .synthesize(tgt)
                    .map(|r| t_count(&r.gates) as f64)
                    .unwrap_or(f64::INFINITY);
                (tgt, anchor_cost, anchor_cost + r0, format!("Q{side}{j}"))
            })
            .collect();
        out.append(&mut batch);
    }
    out
}

/// Recursive anchored synthesis: best cost ≤ `budget` reachable with at
/// most `depth` more Q syllables. Returns (cost, structure label).
fn synth_anchored(
    v: &Mat2,
    qf_dag: &Mat2,
    cfg: &Cfg,
    budget: f64,
    depth: u32,
) -> (f64, String) {
    // r = 0 leaf: one full-ε T-synthesis.
    let leaf = SynthesizerT::new(cfg.eps)
        .synthesize(*v)
        .map(|r| t_count(&r.gates) as f64)
        .unwrap_or(f64::INFINITY);
    let mut best = (leaf, "T".to_string());

    // No room for a Q plus at least one T → leaf only.
    if depth == 0 || budget < Q_COST + 1.0 {
        return best;
    }

    let mut children = anchored_children(v, qf_dag, cfg);
    // r1 candidates directly (anchor + leaf of residual).
    for (_, _, r1_cost, label) in &children {
        if *r1_cost < best.0 {
            best = (*r1_cost, format!("{label}+T"));
        }
    }
    if depth == 1 {
        return best;
    }
    // Descend into the top_k cheapest residuals only.
    children.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
    for (tgt, anchor_cost, _, label) in children.into_iter().take(cfg.top_k) {
        let sub_budget = best.0.min(budget) - anchor_cost;
        if sub_budget < 1.0 {
            continue;
        }
        let (sub_cost, sub_label) = synth_anchored(&tgt, qf_dag, cfg, sub_budget, depth - 1);
        let total = anchor_cost + sub_cost;
        if total < best.0 {
            best = (total, format!("{label}+{sub_label}"));
        }
    }
    best
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let eps: f64 = args.first().and_then(|s| s.parse().ok()).unwrap_or(1e-5);
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(8);
    let seed: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0xC0FFEE);
    let j_max: u32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(2);
    let depth: u32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(2);
    let top_k: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(8);
    let skip_hybrid = args.iter().any(|a| a == "--skip-hybrid");
    let cfg = Cfg { eps, j_max, top_k };

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

    println!("E5b anchored recursion: ε={eps:e}, n={n}, seed=0x{seed:X}, j_max={j_max}, depth={depth}, top_k={top_k}");
    println!("{:>3} | {:>5} | {:>6} | {:>6}  structure | winner", "#", "t0", "hybrid", "rec");

    let (mut rec_beats_t0, mut rec_beats_hybrid) = (0usize, 0usize);
    let (mut sum_t0, mut sum_hyb, mut sum_min) = (0.0, 0.0, 0.0);
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

        let t_rec = Instant::now();
        let (rec_cost, structure) = synth_anchored(&v, &qf_dag, &cfg, t0, depth);
        let rec_wall = t_rec.elapsed().as_secs_f64();

        if rec_cost < t0 { rec_beats_t0 += 1; }
        if !hybrid_cost.is_nan() && rec_cost < hybrid_cost { rec_beats_hybrid += 1; }
        sum_t0 += t0; sum_hyb += hybrid_cost;
        sum_min += rec_cost.min(hybrid_cost.min(t0));
        let winner = if rec_cost < hybrid_cost.min(t0) { "rec" }
                     else if hybrid_cost < t0 { "hybrid" } else { "T" };
        println!("{i:>3} | {t0:>5.1} | {hybrid_cost:>6.1} | {rec_cost:>6.1}  {structure:<12} {winner}  [{rec_wall:.1}s]");
    }
    println!("\nmean: t0={:.1} hybrid={:.1} min(all)={:.1}",
        sum_t0 / n as f64, sum_hyb / n as f64, sum_min / n as f64);
    println!("recursion beats t0 on {rec_beats_t0}/{n}, beats hybrid on {rec_beats_hybrid}/{n}");
}
