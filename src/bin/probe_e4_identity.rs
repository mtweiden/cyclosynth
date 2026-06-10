//! Experiment E4 (docs/design_certified_optimal_cost.md): verify the
//! FGKM-derived cost identity in THIS codebase's conventions.
//!
//! Claim (from FGKM Theorem 4.1 via the literature pull): for canonical
//! Clifford+√T circuits, the Bloch (SO(3)) denominator exponent N
//! satisfies  N = 2·T_count + 3·Q_count,  equivalently
//! W = T + 3.5Q = N/2 + 2r. If it holds for our decomposer's outputs,
//! N becomes an exactly-costed enumeration coordinate (the (N, r) grid
//! walk) and N-vs-k data feeds the P1' staircase slope.
//!
//! Method: random reduced FGKM words → decompose (t, q) → SO3Q
//! denominator exponent N (per-entry reduced) → compare. Also tabulate
//! N against the U(2) matrix lde k.
//!
//! Args: [<n_words> [<m_max> [<seed>]]]   (defaults 200, 6, 0xE4)

use cyclosynth::matrix::so3::SO3Q;
use cyclosynth::matrix::u2::U2Q;
use cyclosynth::synthesis::decomposer::BlochDecomposer;

struct Xs(u64);
impl Xs {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
}

fn syllable(axis: usize, a: u32) -> U2Q {
    let mut d = U2Q::eye();
    for _ in 0..a {
        d = d * U2Q::q();
    }
    match axis {
        0 => (U2Q::h() * d * U2Q::h()).reduced(),
        1 => (U2Q::s() * U2Q::h() * d * U2Q::h() * U2Q::s().dagger()).reduced(),
        _ => d,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let n_words: usize = args.first().and_then(|s| s.parse().ok()).unwrap_or(200);
    let m_max: u32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(6);
    let seed: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0xE4);

    let mut rng = Xs(seed);
    let (mut holds, mut fails, mut shown) = (0usize, 0usize, 0usize);
    // N vs k slope data: min N seen per matrix-lde k.
    let mut min_n_at_k: std::collections::BTreeMap<u32, u32> = std::collections::BTreeMap::new();

    for _ in 0..n_words {
        let m = 1 + (rng.next() % m_max as u64) as u32;
        let mut u = U2Q::eye();
        let mut prev_axis = 3usize;
        for _ in 0..m {
            let mut axis = (rng.next() % 3) as usize;
            while axis == prev_axis {
                axis = (rng.next() % 3) as usize;
            }
            prev_axis = axis;
            let a = 1 + (rng.next() % 3) as u32;
            u = (u * syllable(axis, a)).reduced();
        }

        let gates = BlochDecomposer.decompose(&u);
        let t = gates.chars().filter(|&c| c == 'T').count() as u32;
        let q = gates.chars().filter(|&c| c == 'Q').count() as u32;

        let mut so3 = SO3Q::from_u2(&u);
        so3.reduce();
        let n = so3.maximum_denominator_exponent();

        let predicted = 2 * t + 3 * q;
        if n == predicted {
            holds += 1;
        } else {
            fails += 1;
        }
        // The two inequalities that matter for the slope-2 staircase:
        // (a) cost ≥ N (in T-units; half-units: 2t+7q ≥ 2N), from
        //     N-exact peeling with cost/drop ≥ 1 per syllable type;
        // (b) N ≥ 2k − 3 (Bloch valuation vs matrix lde).
        if 2 * t + 7 * q < 2 * n {
            if shown < 8 {
                println!("COST<N VIOLATION: cost={} HU < 2N={} (t={t},q={q},k={},gates={gates})",
                    2*t+7*q, 2*n, u.k);
                shown += 1;
            }
        }
        if n + 3 < 2 * u.k {
            if shown < 8 {
                println!("N<2k-3 VIOLATION: N={n} vs 2k-3={} (k={}, gates={gates})",
                    2*u.k as i64 - 3, u.k);
                shown += 1;
            }
        }
        min_n_at_k
            .entry(u.k)
            .and_modify(|v| *v = (*v).min(n))
            .or_insert(n);
    }

    println!("\nidentity N == 2t+3q: holds {holds}/{}, fails {fails}", holds + fails);
    println!("\nmin Bloch-N per matrix-lde k (slope data for P1'):");
    for (k, n) in &min_n_at_k {
        println!("  k={k:>2}  min N={n:>3}  N/k={:.2}", *n as f64 / (*k).max(1) as f64);
    }
}
