//! Halving faithful-confirm: does the totally-positive density (~1/3 in
//! the uniform-coeff MC) survive conditioning on the main embedding
//! being near a REAL, asymmetric target u? That tests the load-bearing
//! assumption behind the MC verdict — Galois independence of the
//! conjugate embeddings from the main-embedding constraint. If the
//! conditioned density ≈ unconditioned ~1/3, the gate is faithfully
//! closed; if it collapses, the real search's near-target u population
//! is much harder to extend to a valid t than the MC implied.
//!
//! For each random SU(2) target and k: rejection-sample integer coeff
//! vectors u_num, keep those whose σ_1(u_num)/√2^k lands within a ball
//! of target_u (the alignment the real search demands), and measure the
//! fraction whose other three conjugates are also in the 2^k disk.
//!
//! Args: [k_csv] [samples] [n_targets]   e.g. `probe_halving_target 8,12,16,20 8000000 8`

use num_complex::Complex64;
use std::f64::consts::PI;

fn embed(c: &[i64; 8], j: u32) -> Complex64 {
    let mut s = Complex64::new(0.0, 0.0);
    for (m, &cm) in c.iter().enumerate() {
        s += (cm as f64) * Complex64::from_polar(1.0, PI * (j * m as u32) as f64 / 8.0);
    }
    s
}

struct Rng(u64);
impl Rng {
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
    fn coeff(&mut self, r: i64) -> i64 {
        (self.next() % (2 * r as u64 + 1)) as i64 - r
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let ks: Vec<u32> = args
        .first()
        .map(|s| s.split(',').filter_map(|x| x.parse().ok()).collect())
        .unwrap_or_else(|| vec![8, 12, 16, 20]);
    let samples: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(8_000_000);
    let n_targets: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(8);

    println!("# halving faithful confirm: totally-positive density CONDITIONED on");
    println!("# main embedding near a real SU(2) target u (vs unconditioned MC ~1/3)");
    println!("# k  targets  near_target  all4_in_disk  cond_density");
    let mut rng = Rng(0xC0FFEE);

    for &k in &ks {
        let disk = (1u128 << k) as f64;
        let scale = disk.sqrt(); // √2^k
        let r = ((scale / 2.0).round() as i64).max(2);
        // Main-embedding ball radius: a modest fraction of the unit scale.
        // The real search demands ε-closeness; the conjugate marginal is
        // ρ-independent under Galois independence, so a generous ρ that
        // yields samples is the right probe of the assumption.
        let rho = 0.15 * scale;
        let rho_sq = rho * rho;

        let mut near = 0u64;
        let mut all4 = 0u64;

        for _ in 0..n_targets {
            // Random SU(2) u-entry: |target_u| ≤ 1, generic phase.
            let mag = (rng.unit()).sqrt(); // |u|, area-uniform in disk
            let ph = 2.0 * PI * rng.unit();
            let tu = Complex64::from_polar(mag, ph) * scale; // √2^k · target_u

            for _ in 0..(samples / n_targets as u64) {
                let c: [i64; 8] = std::array::from_fn(|_| rng.coeff(r));
                let s1 = embed(&c, 1);
                if (s1 - tu).norm_sqr() > rho_sq {
                    continue;
                }
                if s1.norm_sqr() > disk {
                    continue; // |u| ≤ 1 required for a valid unitary
                }
                near += 1;
                if embed(&c, 3).norm_sqr() < disk
                    && embed(&c, 5).norm_sqr() < disk
                    && embed(&c, 7).norm_sqr() < disk
                {
                    all4 += 1;
                }
            }
        }

        let dens = all4 as f64 / near.max(1) as f64;
        println!(
            "{:>3}  {:>7}  {:>11}  {:>12}  {:.3e}",
            k, n_targets, near, all4, dens
        );
    }
}
