//! Diagnostic harness for the ε=1e-4 no-solution failure.
//!
//! Per PROMPT_diagnose_n12_1e4_failure.md: diagnostic only — change no
//! algorithm. Allowed: `CYCLOSYNTH_BOUND_SQ_N12` env override and
//! READ-ONLY computations alongside the project's calls.
//!
//! The probes use 2 fixed Haar seeds (`StdRng::seed_from_u64(0)` and
//! seed=1) from the failing run; both produced `None` at every k ∈
//! [5, 20] in the prior verifier.

use cyclosynth::synthesis::clifford_pi12::{synthesize_circuit_at_k, Gate};
use cyclosynth::synthesis::lattice_upsilon::enumerate::{
    bullets_total_twice, compute_align_vec, norm_sqr_total, uv_to_xy,
};
use num_complex::Complex64;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::f64::consts::PI;
use std::time::Instant;

// ─── Oracle gates / distance (copies, NOT imports) ──────────────────────────

type Mat2 = [[Complex64; 2]; 2];
fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}
fn sqrt2_inv() -> f64 {
    1.0 / (0.5_f64).exp2()
}
fn oracle_h() -> Mat2 {
    let s = sqrt2_inv();
    [[c(s, 0.0), c(s, 0.0)], [c(s, 0.0), c(-s, 0.0)]]
}
fn oracle_s() -> Mat2 {
    [[c(1.0, 0.0), c(0.0, 0.0)], [c(0.0, 0.0), c(0.0, 1.0)]]
}
fn oracle_sdg() -> Mat2 {
    [[c(1.0, 0.0), c(0.0, 0.0)], [c(0.0, 0.0), c(0.0, -1.0)]]
}
fn oracle_p() -> Mat2 {
    let cs = ((6.0_f64).sqrt() + (2.0_f64).sqrt()) / 4.0;
    let sn = ((6.0_f64).sqrt() - (2.0_f64).sqrt()) / 4.0;
    [[c(1.0, 0.0), c(0.0, 0.0)], [c(0.0, 0.0), c(cs, sn)]]
}
fn oracle_pdg() -> Mat2 {
    let cs = ((6.0_f64).sqrt() + (2.0_f64).sqrt()) / 4.0;
    let sn = ((6.0_f64).sqrt() - (2.0_f64).sqrt()) / 4.0;
    [[c(1.0, 0.0), c(0.0, 0.0)], [c(0.0, 0.0), c(cs, -sn)]]
}
fn oracle_x() -> Mat2 {
    [[c(0.0, 0.0), c(1.0, 0.0)], [c(1.0, 0.0), c(0.0, 0.0)]]
}
fn oracle_y() -> Mat2 {
    [[c(0.0, 0.0), c(0.0, -1.0)], [c(0.0, 1.0), c(0.0, 0.0)]]
}
fn oracle_z() -> Mat2 {
    [[c(1.0, 0.0), c(0.0, 0.0)], [c(0.0, 0.0), c(-1.0, 0.0)]]
}
fn oracle_eye() -> Mat2 {
    [[c(1.0, 0.0), c(0.0, 0.0)], [c(0.0, 0.0), c(1.0, 0.0)]]
}
fn oracle_gate(g: Gate) -> Mat2 {
    match g {
        Gate::H => oracle_h(),
        Gate::S => oracle_s(),
        Gate::Sdg => oracle_sdg(),
        Gate::P => oracle_p(),
        Gate::Pdg => oracle_pdg(),
        Gate::X => oracle_x(),
        Gate::Y => oracle_y(),
        Gate::Z => oracle_z(),
    }
}
fn mat_mul(a: &Mat2, b: &Mat2) -> Mat2 {
    let mut out = [[c(0.0, 0.0); 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                out[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    out
}
fn oracle_circuit(circuit: &[Gate]) -> Mat2 {
    let mut u = oracle_eye();
    for &g in circuit {
        u = mat_mul(&u, &oracle_gate(g));
    }
    u
}
fn dist_oracle(a: &Mat2, b: &Mat2) -> f64 {
    let mut tr = c(0.0, 0.0);
    for i in 0..2 {
        for j in 0..2 {
            tr += a[i][j] * b[i][j].conj();
        }
    }
    let tab = tr.norm();
    let phi = if tab > 1e-300 { tr / tab } else { c(1.0, 0.0) };
    let mut fro = 0.0_f64;
    for i in 0..2 {
        for j in 0..2 {
            let d = a[i][j] - phi * b[i][j];
            fro += d.norm_sqr();
        }
    }
    let dsq = fro * (8.0 - fro) / 16.0;
    dsq.max(0.0).sqrt()
}
fn haar_target(seed: u64) -> Mat2 {
    let mut rng = StdRng::seed_from_u64(seed);
    loop {
        let raw: [f64; 4] = std::array::from_fn(|_| {
            let mut s = 0.0;
            for _ in 0..12 {
                s += rng.random::<f64>();
            }
            s - 6.0
        });
        let v00 = c(raw[0], raw[1]);
        let v10 = c(raw[2], raw[3]);
        let n = (v00.norm_sqr() + v10.norm_sqr()).sqrt();
        if n < 1e-6 {
            continue;
        }
        let v00 = v00 / n;
        let v10 = v10 / n;
        return [[v00, -v10.conj()], [v10, v00.conj()]];
    }
}
fn v_of(target: &Mat2) -> [f64; 4] {
    [
        target[0][0].re,
        target[0][0].im,
        target[1][0].re,
        target[1][0].im,
    ]
}

// ─── Probe 1 — bound_sq sweep ───────────────────────────────────────────────

#[test]
#[ignore = "diagnostic-only; run with `cargo test --release --test diagnose_n12_1e4 \
            probe1_bound_sweep -- --ignored --nocapture`"]
fn probe1_bound_sweep() {
    let eps = 1e-4_f64;
    // PROMPT-prescribed bound sweep PLUS extended k range to resolve
    // "higher-k need" vs "(T) constant mismatch" — PROMPT specifically
    // asks: "If it's genuinely a higher-k need ... the bar should
    // become reachable as k→16,20 with a wide bound; report whether
    // it does."
    let bounds = [2.0_f64, 8.0, 32.0, 128.0, 512.0, 2048.0];
    let ks = [12u32, 16u32, 20u32, 24u32, 28u32];
    let seeds = [0u64, 1u64];
    eprintln!(
        "\n[Probe 1] bound_sq sweep at ε={eps:.0e}\n\
         seed | k  | bound  | found | t (ms) | oracle d  (if Some)"
    );
    let mut any_found = false;
    for seed in seeds {
        for k in ks {
            let target = haar_target(seed);
            for &b in &bounds {
                // SAFETY: serial test execution.
                unsafe {
                    std::env::set_var("CYCLOSYNTH_BOUND_SQ_N12", format!("{b}"));
                }
                let t0 = Instant::now();
                let r = synthesize_circuit_at_k(&target, k, eps);
                let dt = t0.elapsed();
                let (found_str, d_str) = match &r {
                    Some(r) => {
                        let u = oracle_circuit(&r.circuit);
                        let d = dist_oracle(&u, &target);
                        any_found = true;
                        ("Y", format!("{d:.3e}"))
                    }
                    None => ("N", "—".to_string()),
                };
                eprintln!(
                    "{seed:>4} | {k:>2} | {b:>6} | {found_str:>5} | {:>6.1} | {d_str}",
                    dt.as_secs_f64() * 1000.0
                );
            }
        }
    }
    unsafe {
        std::env::remove_var("CYCLOSYNTH_BOUND_SQ_N12");
    }
    if any_found {
        eprintln!("\n[Probe 1] → at least one bound found a candidate → bucket (B) candidate");
    } else {
        eprintln!("\n[Probe 1] → none across 2 seeds × 2 k × 6 bounds → NOT bound width; go to Probes 2–4");
    }
}

// ─── Probe 2/3/4 — funnel + alignment ceiling + cap-center sanity ──────────

/// Read-only enumeration of the LLL-style alignment funnel, computed by
/// the test harness (mirrors the project's leaf check exactly but in a
/// separate enumeration so we can step through and count).
///
/// For diagnostic purposes only: enumerate Σx² ≤ 2·2^k (the project's
/// brute-force pruning bound, which is what the SE walker stays within
/// at bound_sq large), then count the funnel.
fn funnel_for_seed(target: &Mat2, k: u32, eps: f64) -> FunnelStats {
    // We can't realistically enumerate Σx² ≤ 2·2^12 = 8192 here — that's
    // ~100M points in 16D. Instead we run the project's brute-force
    // enumerator (`phase1_brute`) at modest k and count via the project's
    // exposed functions. At k=12 that's intractable; report a HARD CAP.
    let v = v_of(target);
    let y_lat = uv_to_xy(v, k);
    let target_norm = 1i64 << k;
    let threshold = 2.0_f64.powi(k as i32) * (1.0 - eps * eps);

    // For k=12 enumerate is infeasible; report None enumeration and only
    // do the alignment-direction "ceiling" computations downstream.
    FunnelStats {
        n_enumerated: 0,
        n_pass_norm: 0,
        n_pass_bullets: 0,
        n_pass_align: 0,
        threshold,
        target_norm,
        y: y_lat,
    }
}

#[allow(dead_code)]
struct FunnelStats {
    n_enumerated: u64,
    n_pass_norm: u64,
    n_pass_bullets: u64,
    n_pass_align: u64,
    threshold: f64,
    target_norm: i64,
    y: [f64; 16],
}

// ─── Σ helpers used by Probes 3/4 (independent of the project's q_metric) ──

const COSET_REPS: [u32; 4] = [1, 17, 13, 5];

fn sigma_el() -> [[f64; 8]; 8] {
    let mut m = [[0.0f64; 8]; 8];
    for (k, &rep) in COSET_REPS.iter().enumerate() {
        for j in 0..8 {
            let theta = (rep as f64) * (j as f64) * PI / 12.0;
            m[2 * k][j] = theta.cos();
            m[2 * k + 1][j] = theta.sin();
        }
    }
    m
}

fn sigma_16() -> [[f64; 16]; 16] {
    let el = sigma_el();
    let mut m = [[0.0f64; 16]; 16];
    for i in 0..8 {
        for j in 0..8 {
            m[i][j] = el[i][j];
            m[8 + i][8 + j] = el[i][j];
        }
    }
    m
}

/// Solve `(ΣᵀΣ) z = Σᵀ · v_pad` (independent f64 LU). Returns
/// `z = Σ⁻¹·v_pad`. v_pad puts `v` on cap rows {0,1,8,9}.
fn sigma_inv_v_pad(v: [f64; 4]) -> [f64; 16] {
    let sigma = sigma_16();
    let mut v_pad = [0.0f64; 16];
    v_pad[0] = v[0];
    v_pad[1] = v[1];
    v_pad[8] = v[2];
    v_pad[9] = v[3];
    let mut rhs = [0.0f64; 16];
    for j in 0..16 {
        for i in 0..16 {
            rhs[j] += sigma[i][j] * v_pad[i];
        }
    }
    let mut g = [[0.0f64; 16]; 16];
    for i in 0..16 {
        for j in 0..16 {
            let mut s = 0.0f64;
            for r in 0..16 {
                s += sigma[r][i] * sigma[r][j];
            }
            g[i][j] = s;
        }
    }
    lu_solve_16(&mut g, &mut rhs)
}

fn lu_solve_16(a: &mut [[f64; 16]; 16], b: &mut [f64; 16]) -> [f64; 16] {
    let n = 16;
    for k in 0..n {
        let mut p = k;
        let mut max = a[k][k].abs();
        for i in (k + 1)..n {
            if a[i][k].abs() > max {
                max = a[i][k].abs();
                p = i;
            }
        }
        if p != k {
            a.swap(p, k);
            b.swap(p, k);
        }
        if a[k][k].abs() < 1e-18 {
            return [0.0; 16];
        }
        for i in (k + 1)..n {
            let f = a[i][k] / a[k][k];
            a[i][k] = f;
            for j in (k + 1)..n {
                a[i][j] -= f * a[k][j];
            }
            b[i] -= f * b[k];
        }
    }
    let mut x = [0.0f64; 16];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n {
            s -= a[i][j] * x[j];
        }
        x[i] = s / a[i][i];
    }
    x
}

// ─── Probe 3 / 4 combined ───────────────────────────────────────────────────

#[test]
#[ignore = "diagnostic-only; run with `cargo test --release --test diagnose_n12_1e4 \
            probe34_alignment_ceiling_and_cap -- --ignored --nocapture`"]
fn probe34_alignment_ceiling_and_cap() {
    let eps = 1e-4_f64;
    let k = 12u32;
    let seeds = [0u64, 1u64];

    for seed in seeds {
        let target = haar_target(seed);
        let v = v_of(&target);
        let r = 2.0_f64.powf(k as f64 / 2.0);

        // ── ‖ỹ‖² and threshold sanity ─────────────────────────────
        // ỹ = compute_align_vec(v) (raw, no R factor — see uv_to_xy docs).
        let y = compute_align_vec(v);
        let y_norm_sq: f64 = y.iter().map(|x| x * x).sum();
        let threshold = 2.0_f64.powi(k as i32) * (1.0 - eps * eps);

        // Theoretical max (x·ỹ)² for x on the norm shell: by
        // Cauchy-Schwarz with the cyclotomic Gram norm,
        //   (x·ỹ) ≤ ‖x‖_eucl · ‖ỹ‖_eucl
        // and ‖x‖²_eucl can range up to 2·2^k (the brute Euclidean bound).
        // Tighter: project ỹ on the σ_1 subspace and use the lattice norm.
        // Here we just report the analytical pieces.
        eprintln!("\n=== seed {seed}  k={k}  ε={eps:.0e} ===");
        eprintln!(
            "|v|²              = {:.6}",
            v.iter().map(|x| x * x).sum::<f64>()
        );
        eprintln!("R = √(2^k)        = {r:.3e}");
        eprintln!("‖ỹ‖²              = {y_norm_sq:.6}");
        eprintln!("threshold (2^k·(1-ε²)) = {threshold:.3e}");
        eprintln!(
            "Cauchy-Schwarz ceiling (‖x‖²_eucl_max · ‖ỹ‖²) = (2·2^k) · ‖ỹ‖² = {:.3e}",
            (2.0 * 2.0_f64.powi(k as i32)) * y_norm_sq
        );
        // Target (y · x_target)² = 2^k for ideal x_target on cap.
        eprintln!(
            "Target ideal (y · x_target)² = 2^k = {:.3e}",
            2.0_f64.powi(k as i32)
        );

        // ── Probe 4 — cap center vs target ray ────────────────────
        let c_lat = sigma_inv_v_pad(v);
        let c_scaled: [f64; 16] = std::array::from_fn(|i| c_lat[i] * r);
        let target_emb = {
            // R · Σᵀ · (v, 0). v_pad on cap rows; Σᵀ · v_pad is the
            // pullback-target embedding.
            let sigma = sigma_16();
            let mut v_pad = [0.0f64; 16];
            v_pad[0] = v[0];
            v_pad[1] = v[1];
            v_pad[8] = v[2];
            v_pad[9] = v[3];
            let mut t = [0.0f64; 16];
            for j in 0..16 {
                let mut s = 0.0f64;
                for i in 0..16 {
                    s += sigma[i][j] * v_pad[i];
                }
                t[j] = r * s;
            }
            t
        };
        let mut diff = [0.0f64; 16];
        for i in 0..16 {
            diff[i] = c_scaled[i] - target_emb[i];
        }
        let diff_norm_sq: f64 = diff.iter().map(|x| x * x).sum();
        let c_norm_sq: f64 = c_scaled.iter().map(|x| x * x).sum();
        let t_norm_sq: f64 = target_emb.iter().map(|x| x * x).sum();
        eprintln!("cap-center c = R·Σ⁻¹·(v,0) ‖c‖² = {c_norm_sq:.4}");
        eprintln!("target ray R·Σᵀ·(v,0)      ‖t‖² = {t_norm_sq:.4}");
        eprintln!("‖c - t‖²                   = {diff_norm_sq:.4e}");

        // ── Probe 3 — Babai nearest-plane on the embedded target ─
        // We bypass the project's LLL-reduced basis (since the test
        // can't read it from outside the SE call), and Babai onto the
        // IDENTITY basis of Z^16 instead — i.e. round each coord of
        // the embedded target to the nearest integer. This is a loose
        // upper bound on the achievable cap alignment, NOT the LLL-
        // optimal Babai. Reading the project's reduced basis would
        // require an instrumentation change which the PROMPT
        // forbids.
        let x_babai_id: [i64; 16] = std::array::from_fn(|i| target_emb[i].round() as i64);
        // Score this point.
        let dot: f64 = (0..16).map(|i| (x_babai_id[i] as f64) * y[i]).sum();
        let dot_sq = dot * dot;
        let norm_sq = norm_sqr_total(&x_babai_id);
        let bullets = bullets_total_twice(&x_babai_id);
        let target_norm = 1i64 << k;
        eprintln!(
            "Babai (identity-basis, loose) x_id ‖x‖²_cyc = {norm_sq} (target 2^k = {target_norm})"
        );
        eprintln!("  bullets = {:?} (need (0,0,0))", bullets);
        eprintln!(
            "  (y·x)² = {dot_sq:.3e}, threshold = {threshold:.3e} → {}",
            if dot_sq >= threshold { "PASS" } else { "FAIL" }
        );

        // Also: drop the rounding and use the f64 target embedding to
        // compute an upper bound on the alignment if we could choose
        // any integer lattice point inside the shell.
        // The COS distance of x_id to the target embedding tells us
        // how close we get.
        let _ = (c_lat, threshold);
    }
}

// ─── Probe 2 — for k=4..6 where brute is feasible — funnel ─────────────────

#[test]
#[ignore = "diagnostic-only; run with `cargo test --release --test diagnose_n12_1e4 \
            probe2_funnel_small_k -- --ignored --nocapture`"]
fn probe2_funnel_small_k() {
    // At k=12, full leaf enumeration is intractable from outside the
    // SE walker (the bound ball alone holds millions of points). The
    // PROMPT's "instrument the SE call" path requires changing the
    // synthesizer, which the diagnostic discipline forbids. We
    // substitute: run the project's brute-force enumerator at smaller
    // k where it's feasible, and report the funnel pattern. If even
    // brute at k≤5 with ε=1e-4 returns no n_pass_align, the threshold
    // is unreachable at that k for these targets — and the pattern
    // extrapolates.
    use cyclosynth::synthesis::lattice_upsilon::phase1_brute;
    let eps = 1e-4_f64;
    // phase1_brute(5) at ε=1e-4 enumerates ~10^6 candidates and the
    // post-filter is slow; k=4 is the largest tractable here.
    let ks = [3u32, 4];
    let seeds = [0u64, 1u64];
    eprintln!(
        "\n[Probe 2] brute-force funnel at small k (k=12 is intractable from outside the SE walker)\n\
         seed | k | n_pass_norm_bullets | n_pass_align | threshold     | max_dot²"
    );
    for seed in seeds {
        for k in ks {
            let target = haar_target(seed);
            let v = v_of(&target);
            let y = uv_to_xy(v, k);
            let threshold = 2.0_f64.powi(k as i32) * (1.0 - eps * eps);
            let pool = phase1_brute(k);
            let n_pass_nb = pool.len();
            let mut max_dot_sq: f64 = 0.0;
            let mut n_pass_align = 0usize;
            for x in &pool {
                let dot: f64 = (0..16).map(|i| (x[i] as f64) * y[i]).sum();
                let d2 = dot * dot;
                if d2 > max_dot_sq {
                    max_dot_sq = d2;
                }
                if d2 >= threshold {
                    n_pass_align += 1;
                }
            }
            eprintln!(
                "{seed:>4} | {k} | {:>19} | {:>12} | {threshold:.3e} | {max_dot_sq:.3e}",
                n_pass_nb, n_pass_align
            );
        }
    }
}
