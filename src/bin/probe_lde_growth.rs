//! Experiment E1 from docs/design_certified_optimal_cost.md: how fast
//! does reduced lde grow along cheap syllable chains? Decides whether
//! the L(k) ≥ k/2 (T-units) staircase is tight — i.e., whether the
//! certified-sweep horizon really is ≈ 2·C*.
//!
//! Chains probed (s = syllable count, all syllables cost 1 T-unit):
//!   A: alternating R_x(T)·R_y(T)·R_x(T)·…   (max conceivable growth)
//!   B: alternating R_x(T)·R_z(T)·…           (z contributes lde 0)
//!   C: alternating R_x(Q)·R_y(Q)·…           (Q chains, cost 3.5/syll)
//!
//! Each product is fully reduced via `U2Q::reduced()` (Mul alone only
//! accumulates an upper bound on lde).
//!
//! Prints s, reduced lde k, realized cost, and cost/lde — the minimum
//! observed cost/lde across chains upper-bounds the true L(k)/k slope.

use cyclosynth::matrix::u2::U2Q;

fn main() {
    let rx_t: U2Q = U2Q::h() * U2Q::t() * U2Q::h();
    let ry_t: U2Q = U2Q::s() * U2Q::h() * U2Q::t() * U2Q::h() * U2Q::s().dagger();
    let rz_t: U2Q = U2Q::t();
    let rx_q: U2Q = U2Q::h() * U2Q::q() * U2Q::h();
    let ry_q: U2Q = U2Q::s() * U2Q::h() * U2Q::q() * U2Q::h() * U2Q::s().dagger();

    let chains: [(&str, [&U2Q; 2], f64); 3] = [
        ("A: Rx(T)/Ry(T)", [&rx_t, &ry_t], 1.0),
        ("B: Rx(T)/Rz(T)", [&rx_t, &rz_t], 1.0),
        ("C: Rx(Q)/Ry(Q)", [&rx_q, &ry_q], 3.5),
    ];

    let max_s: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(40);

    for (label, pair, cost_per_syll) in &chains {
        println!("--- chain {label} (cost/syllable = {cost_per_syll}) ---");
        let mut u = U2Q::eye();
        for s in 1..=max_s {
            u = (u * *pair[(s - 1) % 2]).reduced();
            let k = u.k;
            let cost = *cost_per_syll * s as f64;
            if s % 4 == 0 || s == 1 || s == max_s {
                println!(
                    "  s={s:>3}  lde={k:>3}  cost={cost:>6.1}  cost/lde={:>5.2}",
                    if k > 0 { cost / k as f64 } else { f64::NAN }
                );
            }
        }
    }
}
