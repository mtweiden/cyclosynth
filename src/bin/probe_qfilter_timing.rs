//! Microbench the depth-1 Q-filter components.
//!
//! Generates synthetic-but-realistic depth-1 state (cliff-ish magnitudes) and
//! times: (a) `qfilter_depth1_state` precompute, (b) per-candidate
//! discriminant evaluation in 3 paths (D<0 short-circuit / mod-16 reject /
//! full isqrt). Tells us where the per-node cost actually lives.

use cyclosynth::synthesis::lattice_zeta::se::{
    isqrt_i256_pub as isqrt_i256, qfilter_depth1_state_pub as qfilter_depth1_state,
    qfilter_discriminant_class_pub as qfilter_discriminant_class,
};
use i256::i256;
use std::time::Instant;

fn main() {
    // Cliff-ish magnitudes: basis entries up to ~2^41, z[2..15] up to ~2^43.
    let mut basis = [[0_i64; 16]; 16];
    for i in 0..16 {
        for j in 0..16 {
            let v: i64 = (((i as i64) * 17 + (j as i64) * 23 + 7) % 2_000_000_000) - 1_000_000_000;
            basis[i][j] = v;
        }
        basis[i][i] += 100_000_000_000;
    }
    let z: [i64; 16] = [
        12, 7, 3_000_000_000, -1_500_000_000, 4_200_000_000, -800_000_000,
        2_100_000_000, -3_500_000_000, 1_900_000_000, -2_700_000_000,
        4_800_000_000, -1_100_000_000, 3_300_000_000, -2_500_000_000,
        1_700_000_000, -3_900_000_000,
    ];
    let mut x = [0_i64; 16];
    for i in 0..16 {
        for j in 0..16 {
            x[j] = x[j].wrapping_add(z[i].wrapping_mul(basis[i][j]));
        }
    }
    // Pick T to land in the "interesting" range. Without this most candidates
    // are way off-shell and immediately D<0. We want some mix of D<0, mod-16
    // reject, and isqrt paths to match production distribution.
    let mut x_norm_sq: i128 = 0;
    for i in 0..16 { x_norm_sq += (x[i] as i128) * (x[i] as i128); }
    let target_norm_sq_i64: i64 = (x_norm_sq / 100) as i64;
    println!("calibration: ‖x‖² ≈ {}, T = {} (target_norm_sq_i64)", x_norm_sq, target_norm_sq_i64);

    // === Precompute timing ===
    let n_pre = 1_000_000;
    let t0 = Instant::now();
    let mut acc = i256::from_i64(0);
    for _ in 0..n_pre {
        let (g00, g01, g11, a, v0, v1) =
            qfilter_depth1_state(&basis, &x, z[0], z[1]);
        acc = acc
            .wrapping_add(g00).wrapping_add(g01).wrapping_add(g11)
            .wrapping_add(a).wrapping_add(v0).wrapping_add(v1);
    }
    let dt_pre = t0.elapsed();
    let ns_pre = dt_pre.as_nanos() as f64 / n_pre as f64;
    println!("precompute (qfilter_depth1_state):  {ns_pre:>7.0} ns/call   ({n_pre} iters, acc={})", acc != i256::from_i64(0));

    // Need a state for per-candidate test.
    let (g00, g01, g11, a, v0, v1) =
        qfilter_depth1_state(&basis, &x, z[0], z[1]);

    // === Per-candidate timing across a sweep of zd values ===
    let n_per = 10_000_000;
    let t0 = Instant::now();
    let mut bin = [0u64; 4];
    for i in 0..n_per {
        let zd = (i as i64).wrapping_mul(1_009_937);
        let c = qfilter_discriminant_class(
            g00, g01, g11, a, v0, v1, target_norm_sq_i64, zd,
        );
        bin[c as usize % 4] += 1;
    }
    let dt_per = t0.elapsed();
    let ns_per = dt_per.as_nanos() as f64 / n_per as f64;
    println!("per-candidate (full filter):        {ns_per:>7.0} ns/call   ({n_per} iters)");
    println!("  class 0 (D<0):       {} ({:.1}%)", bin[0], 100.0 * bin[0] as f64 / n_per as f64);
    println!("  class 1 (mod-16 bad): {} ({:.1}%)", bin[1], 100.0 * bin[1] as f64 / n_per as f64);
    println!("  class 2 (isqrt≠):    {} ({:.1}%)", bin[2], 100.0 * bin[2] as f64 / n_per as f64);
    println!("  class 3 (perfect):   {} ({:.4}%)", bin[3], 100.0 * bin[3] as f64 / n_per as f64);

    // === isqrt_i256 timing in isolation ===
    let test_vals: Vec<i256> = (0..1000).map(|i| {
        let v = i256::from_i128((i as i128) * 100_000_000_000_000_000);
        v.wrapping_mul(v).wrapping_add(i256::from_i64(12345))
    }).collect();
    let n_sqrt = 1_000_000;
    let t0 = Instant::now();
    let mut sum_sqrt = i256::from_i64(0);
    for i in 0..n_sqrt {
        let v = test_vals[i % test_vals.len()];
        let s = isqrt_i256(v);
        sum_sqrt = sum_sqrt.wrapping_add(s);
    }
    let dt_sqrt = t0.elapsed();
    let ns_sqrt = dt_sqrt.as_nanos() as f64 / n_sqrt as f64;
    println!("isqrt_i256 (isolated):              {ns_sqrt:>7.0} ns/call   ({n_sqrt} iters, sum_sqrt!=0={})", sum_sqrt != i256::from_i64(0));
}
