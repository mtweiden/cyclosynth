//! Phase 3 — empirical leaf-check rejection rate for the proposed
//! Clifford+√T 16D Lenstra search.
//!
//! Architecture under test: rank-16 lattice over `Z[ζ_16]²`, Schnorr-
//! Euchner enumeration over a stub Q-metric (= identity here), with 4
//! quadratic constraints applied as leaf checks:
//!   - norm shell `‖x‖² = 2^k` (already enforced by enumeration)
//!   - 3 bilinear forms `B_1 = B_2 = B_3 = 0`
//!
//! The decision point: if the *additional* rejection rate from the 3
//! bilinear forms is too aggressive (say > 10^9× per shell), the natural
//! "all-leaf-check" architecture is intractable and we'd need to absorb
//! constraints into the lattice basis (research-level math). Otherwise
//! the architecture is fine and we proceed to Phase 4 (Q-metric).
//!
//! Method: brute-force enumerate all `x ∈ ℤ¹⁶` with `‖x‖² = N` for small
//! `N = 2^k`, k ∈ {1..6}. For each, compute B_1, B_2, B_3 and tally the
//! sequential survival counts.
//!
//! The bilinear forms come from the 4-constraint decomposition of
//! `u_1·u_1* + u_2·u_2* = 2^k` derived in `clifford_sqrt_t_research.md`.

use std::env;
use std::time::Instant;

/// Per-element bilinear form `β_k(u)` for u = (u_0, …, u_7) ∈ Z[ζ_16]
/// in the standard ζ-basis.
#[inline]
fn beta_1(u: &[i64; 8]) -> i64 {
    u[0] * u[1] + u[1] * u[2] + u[2] * u[3] + u[3] * u[4] + u[4] * u[5] + u[5] * u[6] + u[6] * u[7]
        - u[0] * u[7]
}

#[inline]
fn beta_2(u: &[i64; 8]) -> i64 {
    u[0] * u[2] + u[1] * u[3] + u[2] * u[4] + u[3] * u[5] + u[4] * u[6] + u[5] * u[7]
        - u[0] * u[6]
        - u[1] * u[7]
}

#[inline]
fn beta_3(u: &[i64; 8]) -> i64 {
    u[0] * u[3] + u[1] * u[4] + u[2] * u[5] + u[3] * u[6] + u[4] * u[7]
        - u[0] * u[5]
        - u[1] * u[6]
        - u[2] * u[7]
}

/// Joint forms on the 16D pair x = (u_1's 8 coords, u_2's 8 coords).
fn forms(x: &[i64; 16]) -> (i64, i64, i64) {
    let u1: [i64; 8] = x[0..8].try_into().unwrap();
    let u2: [i64; 8] = x[8..16].try_into().unwrap();
    (
        beta_1(&u1) + beta_1(&u2),
        beta_2(&u1) + beta_2(&u2),
        beta_3(&u1) + beta_3(&u2),
    )
}

/// Z[ω] bilinear form (algo.md line 20, paper eq 3.10) on x ∈ ℤ⁸ where
/// (a_1, b_1, c_1, d_1) = (x_0, x_1, x_2, x_3) for u_1 and similarly for u_2:
///
///   B(x) = a_1·b_1 − a_1·d_1 + b_1·c_1 + c_1·d_1
///        + a_2·b_2 − a_2·d_2 + b_2·c_2 + c_2·d_2
#[inline]
fn b_zomega(x: &[i64; 8]) -> i64 {
    x[0] * x[1] - x[0] * x[3] + x[1] * x[2] + x[2] * x[3] + x[4] * x[5] - x[4] * x[7]
        + x[5] * x[6]
        + x[6] * x[7]
}

/// Recursively enumerate D-dim integer points x with ‖x‖² = target.
/// `pos` is the next coordinate to set; `remaining` is the squared-norm
/// budget left for x[pos..D]. At pos=D, remaining must be 0.
fn enumerate_norm_shell<const D: usize>(
    x: &mut [i64; D],
    pos: usize,
    remaining: i64,
    cb: &mut impl FnMut(&[i64; D]),
) {
    if pos == D {
        if remaining == 0 {
            cb(x);
        }
        return;
    }
    let bound = (remaining as f64).sqrt().floor() as i64;
    for v in -bound..=bound {
        let v2 = v * v;
        if v2 > remaining {
            continue;
        }
        x[pos] = v;
        enumerate_norm_shell(x, pos + 1, remaining - v2, cb);
    }
}

#[derive(Default, Debug, Clone, Copy)]
struct ZetaTally {
    shell: u64,
    pass_b123: u64,
}

#[derive(Default, Debug, Clone, Copy)]
struct OmegaTally {
    shell: u64,
    pass_b: u64,
}

fn measure_zeta(k: u32) -> ZetaTally {
    let target = 1i64 << k;
    let mut x = [0i64; 16];
    let mut t = ZetaTally::default();
    enumerate_norm_shell::<16>(&mut x, 0, target, &mut |x| {
        t.shell += 1;
        let (b1, b2, b3) = forms(x);
        if b1 == 0 && b2 == 0 && b3 == 0 {
            t.pass_b123 += 1;
        }
    });
    t
}

fn measure_omega(k: u32) -> OmegaTally {
    let target = 1i64 << k;
    let mut x = [0i64; 8];
    let mut t = OmegaTally::default();
    enumerate_norm_shell::<8>(&mut x, 0, target, &mut |x| {
        t.shell += 1;
        if b_zomega(x) == 0 {
            t.pass_b += 1;
        }
    });
    t
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let k_min: u32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1);
    let k_max: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5);

    println!(
        "{:>3} {:>5}  {:>14} {:>14}  {:>14} {:>14}  {:>10}",
        "k", "2^k", "ω-shell", "ω-valid", "ζ-shell", "ζ-valid", "ζ:ω ratio",
    );
    println!("{}", "-".repeat(98));

    for k in k_min..=k_max {
        let t0 = Instant::now();
        let omega = measure_omega(k);
        let elapsed_o = t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        let zeta = measure_zeta(k);
        let elapsed_z = t1.elapsed().as_secs_f64();

        // The ratio of "valid (= passes all bilinears) per norm shell" is
        // the empirical density advantage of Z[ζ] over Z[ω] at this lde.
        let ratio = if omega.pass_b > 0 {
            zeta.pass_b123 as f64 / omega.pass_b as f64
        } else {
            0.0
        };

        println!(
            "{:>3} {:>5}  {:>14} {:>14}  {:>14} {:>14}  {:>10.3e}    [ω: {:.2}s, ζ: {:.2}s]",
            k,
            1u64 << k,
            omega.shell,
            omega.pass_b,
            zeta.shell,
            zeta.pass_b123,
            ratio,
            elapsed_o,
            elapsed_z,
        );
    }
}
