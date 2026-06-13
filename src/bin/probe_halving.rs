//! Halving go/no-go probe (survey idea #6 — norm-equation dimension halving).
//!
//! Two unknowns decide whether enumerating only the 8D u-column and
//! completing t via |t|²=1−|u|² is viable:
//!
//!  (1) SOLVABILITY DENSITY: a t exists only if β=1−|u|² is totally
//!      positive, i.e. ALL FOUR Galois conjugates σ_j(u) (j=1,3,5,7,
//!      ζ→e^{iπj/8}) have |σ_j(u)|² < 2^k. The target only constrains the
//!      MAIN embedding (σ_1) — the other three roam free over the lattice.
//!      Density = fraction of main-in-disk u that are all-four-in-disk.
//!      If this decays like 2^{-c·k} the halving over-generates
//!      exponentially and is dead; if it is a mild constant it is alive.
//!
//!  (2) FACTORING COST: for the totally-positive u, solving the norm
//!      equation factors Norm(β)=∏_j(2^k−|σ_j(u_num)|²), a rational
//!      integer. We report its bit-size and largest prime factor (trial
//!      division proxy) — the Ross-Selinger factoring-oracle dependency.
//!
//! Monte-Carlo over integer coefficient vectors at the post-LLL scale
//! (small coeffs). First-order signal, not a faithful cap enumeration —
//! see docs/plan_halving_2026_06_13.md for what it does/doesn't show.
//!
//! Args: [k_csv] [samples]   e.g.  `probe_halving 8,12,16,20,24 2000000`

use num_complex::Complex64;
use std::f64::consts::PI;

/// σ_j(u_num): evaluate Σ c[m]·ζ^m at ζ = e^{iπ·j/8}.
fn embed(c: &[i64; 8], j: u32) -> Complex64 {
    let mut s = Complex64::new(0.0, 0.0);
    for (m, &cm) in c.iter().enumerate() {
        s += (cm as f64) * Complex64::from_polar(1.0, PI * (j * m as u32) as f64 / 8.0);
    }
    s
}

/// Trial-division largest prime factor of |n| (proxy for factoring cost).
/// Caps work at 1e7 — returns (largest_found, fully_factored?).
fn largest_prime_factor(mut n: u128) -> (u128, bool) {
    if n < 2 {
        return (1, true);
    }
    let mut largest = 1u128;
    let mut d = 2u128;
    while d * d <= n && d <= 10_000_000 {
        while n % d == 0 {
            largest = d;
            n /= d;
        }
        d += if d == 2 { 1 } else { 2 };
    }
    if n > 1 {
        // Remaining cofactor is prime or a product of primes > 1e7.
        (n.max(largest), n <= 10_000_000)
    } else {
        (largest, true)
    }
}

/// SplitMix64.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    /// integer in [-r, r]
    fn coeff(&mut self, r: i64) -> i64 {
        (self.next() % (2 * r as u64 + 1)) as i64 - r
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let ks: Vec<u32> = args
        .first()
        .map(|s| s.split(',').filter_map(|x| x.parse().ok()).collect())
        .unwrap_or_else(|| vec![8, 12, 16, 20, 24]);
    let samples: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(2_000_000);

    println!("# halving go/no-go: density of totally-positive u + Norm(β) factoring");
    println!("# k  main_in_disk  all4_in_disk  density   1/density   med_normbits  hard_factor_frac");
    let mut rng = Rng(0xC0FFEE);

    for &k in &ks {
        let disk = (1u128 << k) as f64; // 2^k threshold on |σ_j(u_num)|²
        // Coefficient spread: post-LLL lattice points near the cap have
        // |σ(u)|² ~ 2^k spread over 8 coeffs → per-coeff scale ~2^(k/2)/√8.
        // Sample a bit wider so the main-in-disk shell is well populated.
        let r = ((disk.sqrt() / 2.0).round() as i64).max(2);

        let mut main_in = 0u64;
        let mut all4_in = 0u64;
        let mut norm_bits: Vec<u32> = Vec::new();
        let mut hard = 0u64;
        let mut pos_count = 0u64;

        for _ in 0..samples {
            let c: [i64; 8] = std::array::from_fn(|_| rng.coeff(r));
            let m1 = embed(&c, 1).norm_sqr();
            if m1 > disk {
                continue;
            }
            main_in += 1;
            let m3 = embed(&c, 3).norm_sqr();
            let m5 = embed(&c, 5).norm_sqr();
            let m7 = embed(&c, 7).norm_sqr();
            if m3 < disk && m5 < disk && m7 < disk {
                all4_in += 1;
                // Norm(β)·2^{4k} = ∏_j (2^k − |σ_j|²), integer.
                let f = |m: f64| (disk - m).round().max(1.0) as u128;
                let nrm = f(m1).saturating_mul(f(m3)).saturating_mul(f(m5)).saturating_mul(f(m7));
                norm_bits.push(128 - nrm.leading_zeros());
                let (_lpf, ok) = largest_prime_factor(nrm);
                if !ok {
                    hard += 1;
                }
                pos_count += 1;
            }
        }

        norm_bits.sort_unstable();
        let med_bits = norm_bits.get(norm_bits.len() / 2).copied().unwrap_or(0);
        let density = all4_in as f64 / main_in.max(1) as f64;
        let hard_frac = hard as f64 / pos_count.max(1) as f64;
        println!(
            "{:>3}  {:>11}  {:>11}  {:.3e}  {:>9.1}  {:>11}  {:.4}",
            k, main_in, all4_in, density,
            if density > 0.0 { 1.0 / density } else { f64::INFINITY },
            med_bits, hard_frac
        );
    }
}
