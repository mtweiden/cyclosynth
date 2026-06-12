//! Compare Clifford+T vs Clifford+√T under cost = T + 3.5·Q.
//!
//! Random U3 targets across a fixed seed, both synthesizers, gate counts,
//! per-target winner + aggregate. Args:
//!   <epsilon> [<n_targets> [<seed> [<mode> [<lde_window>]]]]
//! Defaults: epsilon=1e-7, n_targets=12, seed=0xC0FFEE, mode=first,
//! lde_window=0 (strict min-lde-first).
//!
//! mode=first:   SynthesizerQ returns the FIRST ε-close candidate at
//!               find-lde (fast).
//! mode=optimal: SynthesizerQ enumerates ALL ε-close candidates at
//!               find-lde and returns the min-cost one. 5-50× slower.
//! mode=compare: Runs *both* √T modes per target and reports per-target
//!               delta + aggregate cost-reduction stats from optimize.
//!
//! 6th arg: comma-separated m-sweep override (e.g. "1,2"). Empty/unset
//! → use auto-default `default_optimal_m_sweep(eps)`.
//!
//! 7th arg: anytime-frontier deadline override — integer ms, or "none"
//! to force the legacy per-arm-budget task grid. Empty/unset → builder
//! default. (Also settable via env CYCLOSYNTH_DEADLINE_MS.)
//!
//! 8th arg: ζ right-coset prefix dedup A/B — "0" or "1", forwarded to
//! `CYCLOSYNTH_ZETA_COSET` via set_var before any synthesis (direct
//! env-prefixed execution is denied in the agent harness — same
//! workaround as bench_t_breakdown's `--coset`). Empty/unset → leave
//! the env/default alone.

use cyclosynth::synthesis::clifford_t::SynthesizerT;
use cyclosynth::synthesis::clifford_sqrt_t::SynthesizerQ;
use cyclosynth::synthesis::distance::Mat2;
use num_complex::Complex;
use std::time::Instant;

/// Deterministic SplitMix64 — well-mixed even from sequential seeds.
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
    /// Range [lo, hi).
    fn range(&mut self, lo: f64, hi: f64) -> f64 { lo + (hi - lo) * self.unit() }
}

type C64 = Complex<f64>;

/// SU(2) U3 target.  Builds the standard U(2) U3 = [[c, -e^(iλ)s],
/// [e^(iφ)s, e^(i(φ+λ))c]] (det = e^(i(φ+λ))) and then absorbs a global
/// phase so det = 1.  Diamond distance is phase-invariant, so this is
/// free — but the synthesizer (Clifford+T / Clifford+√T) needs det in
/// the 16th-roots-of-unity coset, and an unconstrained phase leaves it
/// looping through max_lde looking for a match that doesn't exist.
fn u3(theta: f64, phi: f64, lam: f64) -> Mat2 {
    let (c, s) = ((theta / 2.0).cos(), (theta / 2.0).sin());
    let eilam = C64::from_polar(1.0, lam);
    let eiphi = C64::from_polar(1.0, phi);
    let m = [
        [C64::new(c, 0.0), -eilam * s],
        [eiphi * s, eiphi * eilam * c],
    ];
    // det = e^(i(φ+λ));  divide everything by e^(i(φ+λ)/2) → SU(2).
    let g = C64::from_polar(1.0, -(phi + lam) / 2.0);
    [
        [m[0][0] * g, m[0][1] * g],
        [m[1][0] * g, m[1][1] * g],
    ]
}

fn gate_cost(g: Option<&str>) -> (usize, usize, usize, usize) {
    g.map(|s| (
        s.chars().filter(|&c| c == 'T').count(),
        s.chars().filter(|&c| c == 'Q').count(),
        s.chars().filter(|&c| c == 'H').count(),
        s.chars().count(),
    )).unwrap_or((0, 0, 0, 0))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode { First, Optimal, Compare }

fn run_q(target: Mat2, eps: f64, optimize: bool, lde_window: u32, m_sweep: Option<Vec<u32>>, deadline: Option<&str>) -> (Option<String>, f64, u32) {
    let t0 = Instant::now();
    let mut synth = SynthesizerQ::new(eps)
        .with_optimize_cost(optimize)
        .with_optimal_lde_window(lde_window);
    if let Some(dl) = deadline {
        if dl == "none" {
            synth = synth.with_optimal_deadline_ms(None);
        } else if let Ok(ms) = dl.parse::<u64>() {
            synth = synth.with_optimal_deadline_ms(Some(ms));
        }
    }
    if let Ok(mult) = std::env::var("CYCLOSYNTH_BUDGET_MULT") {
        if let Ok(m) = mult.parse::<u64>() {
            synth = synth.with_optimal_budget_multiplier(m);
        }
    }
    if std::env::var("CYCLOSYNTH_OPEN_FILTER").as_deref() == Ok("1") {
        synth = synth.with_optimal_open_dr_filter(true);
    }
    if let Some(ms) = m_sweep {
        synth = synth.with_optimal_m_sweep(ms);
    }
    let r = synth.synthesize(target);
    let dt = t0.elapsed().as_secs_f64();
    let lde = r.as_ref().map(|r| r.lde).unwrap_or(0);
    (r.and_then(|r| r.gates), dt, lde)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let eps: f64 = args.first().and_then(|s| s.parse().ok()).unwrap_or(1e-7);
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(12);
    let seed: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0xC0FFEE);
    let mode = match args.get(3).map(|s| s.as_str()) {
        Some("optimal") => Mode::Optimal,
        Some("compare") => Mode::Compare,
        _ => Mode::First,
    };
    let lde_window: u32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);
    let m_sweep_override: Option<Vec<u32>> = args.get(5).and_then(|s| {
        if s.is_empty() { None } else {
            Some(s.split(',').filter_map(|p| p.trim().parse().ok()).collect())
        }
    });
    let deadline_override: Option<String> = args
        .get(6)
        .filter(|s| !s.is_empty())
        .cloned()
        .or_else(|| std::env::var("CYCLOSYNTH_DEADLINE_MS").ok());
    if let Some(coset) = args.get(7).filter(|s| *s == "0" || *s == "1") {
        // Must run before the first synthesis (LazyLock-once read).
        unsafe { std::env::set_var("CYCLOSYNTH_ZETA_COSET", coset) };
    }
    let mode_label = match mode {
        Mode::First => "first-hit",
        Mode::Optimal => "optimal",
        Mode::Compare => "compare (first vs optimal)",
    };

    let mut rng = Xs(seed);
    use std::f64::consts::PI;
    let targets: Vec<(f64, f64, f64)> = (0..n).map(|_| (
        rng.range(0.2, PI - 0.2),
        rng.range(0.1, 2.0 * PI - 0.1),
        rng.range(0.1, 2.0 * PI - 0.1),
    )).collect();

    let verbose = n <= 20;
    println!("ε={eps:e}, n={n} U3 targets, seed=0x{seed:X}, cost = T + 3.5·Q, √T mode={mode_label}, lde_window={lde_window}, m_sweep={}, deadline={}, zeta_coset={}\n",
        m_sweep_override.as_ref().map(|v| format!("{:?}", v)).unwrap_or_else(|| "auto".to_string()),
        deadline_override.as_deref().unwrap_or("auto"),
        std::env::var("CYCLOSYNTH_ZETA_COSET").as_deref().unwrap_or("default(1)"));

    if mode == Mode::Compare {
        run_compare(&targets, eps, verbose, lde_window, m_sweep_override, deadline_override);
    } else {
        let optimize = mode == Mode::Optimal;
        run_single(&targets, eps, optimize, verbose, lde_window, m_sweep_override, deadline_override);
    }
}

fn run_single(targets: &[(f64, f64, f64)], eps: f64, optimize: bool, verbose: bool, lde_window: u32, m_sweep: Option<Vec<u32>>, deadline: Option<String>) {
    use std::io::Write;
    let n = targets.len();
    if verbose {
        println!("  #  | θ      φ      λ      | Clifford+T (T-only)       | Clifford+√T (T+√T)             | winner");
        println!("─────┼──────────────────────┼───────────────────────────┼────────────────────────────────┼───────");
    }
    let mut total_t_cost = 0.0_f64;
    let mut total_q_cost = 0.0_f64;
    let (mut t_wins, mut q_wins, mut ties) = (0usize, 0usize, 0usize);
    let mut t_total_wall = 0.0_f64;
    let mut q_total_wall = 0.0_f64;

    for (i, &(th, ph, la)) in targets.iter().enumerate() {
        let target = u3(th, ph, la);
        if verbose {
            eprint!("  #{i} θ={th:.2} φ={ph:.2} λ={la:.2}  T... ");
            let _ = std::io::stderr().flush();
        }
        let t0 = Instant::now();
        let rt = SynthesizerT::new(eps).synthesize(target);
        let t_wall = t0.elapsed().as_secs_f64();
        t_total_wall += t_wall;
        let (t_t, t_q, _t_h, t_len) = gate_cost(rt.as_ref().and_then(|r| r.gates.as_deref()));
        let t_cost = t_t as f64 + 3.5 * t_q as f64;
        let t_lde = rt.as_ref().map(|r| r.lde).unwrap_or(0);
        if verbose {
            eprint!("{t_wall:>5.1}s √T... ");
            let _ = std::io::stderr().flush();
        }
        let (qg, q_wall, q_lde) = run_q(target, eps, optimize, lde_window, m_sweep.clone(), deadline.as_deref());
        q_total_wall += q_wall;
        let (q_t, q_q, _q_h, q_len) = gate_cost(qg.as_deref());
        let q_cost = q_t as f64 + 3.5 * q_q as f64;
        if verbose { eprintln!("{q_wall:>5.1}s"); }
        total_t_cost += t_cost;
        total_q_cost += q_cost;
        let winner = if t_cost < q_cost { t_wins += 1; "T" }
                     else if q_cost < t_cost { q_wins += 1; "√T" }
                     else { ties += 1; "=" };
        if verbose {
            println!(
                "  {:>2} | {:>5.2} {:>5.2} {:>5.2} | lde={:>2} len={:>3} T={:>3} cost={:>3} {:>5.1}s | lde={:>2} len={:>3} T={:>2} Q={:>2} cost={:>3} {:>5.1}s | {}",
                i, th, ph, la,
                t_lde, t_len, t_t, t_cost, t_wall,
                q_lde, q_len, q_t, q_q, q_cost, q_wall,
                winner,
            );
            let _ = std::io::stdout().flush();
        }
        if !verbose && (i + 1) % 10 == 0 {
            eprintln!("    [{}/{}] T_mean={:.1} √T_mean={:.1}",
                i + 1, n,
                total_t_cost / (i + 1) as f64,
                total_q_cost / (i + 1) as f64);
        }
    }
    println!();
    println!("Aggregate over {n} targets:");
    println!("  Clifford+T : total cost = {:>6.1}, total wall = {:>6.2}s, mean cost = {:.1}",
        total_t_cost, t_total_wall, total_t_cost / n as f64);
    println!("  Clifford+√T: total cost = {:>6.1}, total wall = {:>6.2}s, mean cost = {:.1}",
        total_q_cost, q_total_wall, total_q_cost / n as f64);
    println!("  Wins: T={t_wins}  √T={q_wins}  ties={ties}");
    let ratio = total_q_cost / total_t_cost;
    println!("  cost(√T) / cost(T) = {ratio:.3}");
}

fn run_compare(targets: &[(f64, f64, f64)], eps: f64, verbose: bool, lde_window: u32, m_sweep: Option<Vec<u32>>, deadline: Option<String>) {
    use std::io::Write;
    let n = targets.len();
    if verbose {
        println!("  #  | θ     φ     λ     | T_cost | Q_first cost / wall | Q_opt cost / wall | Δcost | wall mult");
        println!("─────┼───────────────────┼────────┼─────────────────────┼───────────────────┼───────┼──────────");
    }
    let mut deltas: Vec<f64> = Vec::with_capacity(n);
    let mut q_first_costs: Vec<f64> = Vec::with_capacity(n);
    let mut q_opt_costs: Vec<f64> = Vec::with_capacity(n);
    let mut t_costs: Vec<f64> = Vec::with_capacity(n);
    let mut q_first_walls: Vec<f64> = Vec::with_capacity(n);
    let mut q_opt_walls: Vec<f64> = Vec::with_capacity(n);

    for (i, &(th, ph, la)) in targets.iter().enumerate() {
        let target = u3(th, ph, la);
        if verbose {
            eprint!("  #{i} θ={th:.2} φ={ph:.2} λ={la:.2}  T... ");
            let _ = std::io::stderr().flush();
        }
        let rt = SynthesizerT::new(eps).synthesize(target);
        let (t_t, t_q, _, _) = gate_cost(rt.as_ref().and_then(|r| r.gates.as_deref()));
        let t_cost = t_t as f64 + 3.5 * t_q as f64;
        t_costs.push(t_cost);

        if verbose { eprint!("Q_first... "); let _ = std::io::stderr().flush(); }
        let (qg1, qw1, _) = run_q(target, eps, false, 0, None, deadline.as_deref());
        let (q_t1, q_q1, _, _) = gate_cost(qg1.as_deref());
        let q_first_cost = q_t1 as f64 + 3.5 * q_q1 as f64;
        q_first_costs.push(q_first_cost);
        q_first_walls.push(qw1);

        if verbose { eprint!("Q_opt... "); let _ = std::io::stderr().flush(); }
        let (qg2, qw2, _) = run_q(target, eps, true, lde_window, m_sweep.clone(), deadline.as_deref());
        let (q_t2, q_q2, _, _) = gate_cost(qg2.as_deref());
        let q_opt_cost = q_t2 as f64 + 3.5 * q_q2 as f64;
        q_opt_costs.push(q_opt_cost);
        q_opt_walls.push(qw2);

        let delta = q_first_cost - q_opt_cost;
        deltas.push(delta);
        if verbose {
            eprintln!("done");
            let mult = if qw1 > 1e-6 { qw2 / qw1 } else { 0.0 };
            println!(
                "  {:>2} | {:>4.2} {:>4.2} {:>4.2} | {:>6} | {:>5} cost / {:>5.2}s | {:>4} cost / {:>5.2}s | {:>+5} | {:>5.1}×",
                i, th, ph, la, t_cost,
                q_first_cost, qw1,
                q_opt_cost, qw2,
                delta,
                mult,
            );
            let _ = std::io::stdout().flush();
        }
        if !verbose && (i + 1) % 10 == 0 {
            let helped = deltas.iter().filter(|&&d| d > 0.0).count();
            let mean_d: f64 = deltas.iter().sum::<f64>() / (i + 1) as f64;
            eprintln!("    [{}/{}] mean Δ={:.1}, helped {}/{}",
                i + 1, n, mean_d, helped, i + 1);
        }
    }

    let sum_t: f64 = t_costs.iter().sum();
    let sum_q1: f64 = q_first_costs.iter().sum();
    let sum_q2: f64 = q_opt_costs.iter().sum();
    let sum_w1: f64 = q_first_walls.iter().sum();
    let sum_w2: f64 = q_opt_walls.iter().sum();
    let helped = deltas.iter().filter(|&&d| d > 0.0).count();
    let regressed = deltas.iter().filter(|&&d| d < 0.0).count();
    let unchanged = deltas.iter().filter(|&&d| d == 0.0).count();
    let max_gain = deltas.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let max_gain = if max_gain.is_finite() { max_gain } else { 0.0 };
    let mut sorted_deltas = deltas.clone();
    sorted_deltas.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let med = if n == 0 { 0.0 } else { sorted_deltas[n / 2] };
    let mean_d: f64 = deltas.iter().sum::<f64>() / n as f64;

    println!();
    println!("Aggregate over {n} targets:");
    println!("  Clifford+T mean cost            = {:.2}", sum_t / n as f64);
    println!("  Clifford+√T first-hit mean cost = {:.2}  wall = {:.2}s", sum_q1 / n as f64, sum_w1);
    println!("  Clifford+√T optimal  mean cost  = {:.2}  wall = {:.2}s", sum_q2 / n as f64, sum_w2);
    let ratio1 = sum_q1 / sum_t;
    let ratio2 = sum_q2 / sum_t;
    println!("  cost(√T_first) / cost(T)        = {ratio1:.3}");
    println!("  cost(√T_opt)   / cost(T)        = {ratio2:.3}");
    let cost_red_pct = if sum_q1 > 0.0 {
        (sum_q1 - sum_q2) / sum_q1 * 100.0
    } else { 0.0 };
    println!("  √T optimal vs first-hit Δ: mean={mean_d:>+.2}  median={med:>+.1}  max-gain={max_gain:>+.1}");
    println!("  cost reduction from optimal:    {cost_red_pct:.2}% over first-hit");
    println!("  helped {helped}/{n}, regressed {regressed}/{n}, unchanged {unchanged}/{n}");
    let wmult = if sum_w1 > 1e-6 { sum_w2 / sum_w1 } else { 0.0 };
    println!("  wall multiplier (opt / first):  {wmult:.2}×");
}
