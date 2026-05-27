//! verify_n6_lattice — integration test wiring the standalone lattice-math
//! verifier against the crate's production Σ matrix and Gram matrix.
//!
//! All algebra helpers, invariant checks, and structural checks are preserved
//! verbatim from src/verify_n6_lattice.rs.  The only changes are:
//!
//!   • build_sigma()  is REMOVED; instead we call
//!     cyclosynth::synthesis::clifford_pi6::sigma_matrix() and permute its
//!     rows into the [σ₁(u), σ₁(t), σ₅(u), σ₅(t)] order that
//!     check_invariants() expects.
//!
//!   • gate_h()/gate_s()/gate_x()/gate_y()/gate_z() are REMOVED; the crate's
//!     CLIFFORD_TABLE_T is used via to_float() instead.
//!
//!   • gate_r() (R = T₆ = diag(1, ξ)) is kept as a local helper because the
//!     crate's rz_pi6_mat() uses the global-phase convention diag(e^{−iπ/12},
//!     e^{iπ/12}), whose u entry is NOT in ℤ[ξ], so find_su2_form() would
//!     fail on it.
//!
//! The four invariants and six structural checks are unchanged.

use cyclosynth::rings::zomicron::SIGMA_GRAM_U;
use cyclosynth::synthesis::clifford_pi6::sigma_matrix;
use cyclosynth::synthesis::cliffords::CLIFFORD_TABLE_T;
use num_complex::Complex64;
use std::f64::consts::PI;

const SQRT2: f64 = std::f64::consts::SQRT_2;
const SQRT3: f64 = 1.7320508075688772_f64;
const TOL: f64 = 1e-9;

// ────────────────────────────────────────────────────────────────────────────
// Algebra of ℤ[ξ], ξ = e^{iπ/6}.  Matches zomicron.rs conventions.
// ────────────────────────────────────────────────────────────────────────────

type ZXi = [i64; 4];

fn xi() -> Complex64 {
    Complex64::from_polar(1.0, PI / 6.0)
}

fn zxi_to_c(p: ZXi) -> Complex64 {
    let x = xi();
    Complex64::new(p[0] as f64, 0.0)
        + Complex64::new(p[1] as f64, 0.0) * x
        + Complex64::new(p[2] as f64, 0.0) * x * x
        + Complex64::new(p[3] as f64, 0.0) * x * x * x
}

/// Complex conjugate in ℤ[ξ]: conj(a + bξ + cξ² + dξ³) = (a+c) + bξ − cξ² + (−b−d)ξ³.
fn zxi_conj(p: ZXi) -> ZXi {
    [p[0] + p[2], p[1], -p[2], -p[1] - p[3]]
}

/// α(p) = a² + b² + c² + d² + ac + bd
fn alpha(p: ZXi) -> i64 {
    p[0] * p[0] + p[1] * p[1] + p[2] * p[2] + p[3] * p[3] + p[0] * p[2] + p[1] * p[3]
}

/// β(p) = ab + bc + cd
fn beta(p: ZXi) -> i64 {
    p[0] * p[1] + p[1] * p[2] + p[2] * p[3]
}

fn xi_pow(k: i32) -> ZXi {
    let k = ((k % 12) + 12) % 12;
    match k {
        0 => [1, 0, 0, 0],
        1 => [0, 1, 0, 0],
        2 => [0, 0, 1, 0],
        3 => [0, 0, 0, 1],
        4 => [-1, 0, 1, 0],
        5 => [0, -1, 0, 1],
        6 => [-1, 0, 0, 0],
        7 => [0, -1, 0, 0],
        8 => [0, 0, -1, 0],
        9 => [0, 0, 0, -1],
        10 => [1, 0, -1, 0],
        11 => [0, 1, 0, -1],
        _ => unreachable!(),
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Sigma adapter: crate → standalone row order
//
// Crate's sigma_matrix() row order: [σ₁(u), σ₅(u), σ₁(t), σ₅(t)]
//   rows 0,1 = Re/Im σ₁(u)
//   rows 2,3 = Re/Im σ₅(u)   (labelled "u•")
//   rows 4,5 = Re/Im σ₁(t)
//   rows 6,7 = Re/Im σ₅(t)   (labelled "t•")
//
// Standalone check_invariants() expects row order: [σ₁(u), σ₁(t), σ₅(u), σ₅(t)]
//   rows 0,1 = Re/Im σ₁(u)   ← Σ_top, used for invariant (1)
//   rows 2,3 = Re/Im σ₁(t)   ← Σ_top
//   rows 4,5 = Re/Im σ₅(u)
//   rows 6,7 = Re/Im σ₅(t)
//
// Permutation: standalone[i] = crate[PERM[i]]
const SIGMA_ROW_PERM: [usize; 8] = [0, 1, 4, 5, 2, 3, 6, 7];

fn crate_sigma_reordered() -> [[f64; 8]; 8] {
    let crate_s = sigma_matrix();
    let mut out = [[0.0_f64; 8]; 8];
    for (new_row, &old_row) in SIGMA_ROW_PERM.iter().enumerate() {
        out[new_row] = crate_s[old_row];
    }
    out
}

/// Build the 8×8 Gram matrix G = ΣᵀΣ from the reordered sigma.
/// (Row permutations don't change ΣᵀΣ, so this equals what the crate computes.)
fn build_gram(sigma: &[[f64; 8]; 8]) -> [[f64; 8]; 8] {
    let mut g = [[0.0_f64; 8]; 8];
    for j in 0..8 {
        for k in 0..8 {
            let mut s = 0.0_f64;
            for i in 0..8 {
                s += sigma[i][j] * sigma[i][k];
            }
            g[j][k] = s;
        }
    }
    g
}

// ────────────────────────────────────────────────────────────────────────────
// Matrix helpers (verbatim from standalone)
// ────────────────────────────────────────────────────────────────────────────

fn mat_vec_8(m: &[[f64; 8]; 8], v: &[f64; 8]) -> [f64; 8] {
    let mut out = [0.0_f64; 8];
    for i in 0..8 {
        for j in 0..8 {
            out[i] += m[i][j] * v[j];
        }
    }
    out
}

fn mat_mat_8(a: &[[f64; 8]; 8], b: &[[f64; 8]; 8]) -> [[f64; 8]; 8] {
    let mut out = [[0.0_f64; 8]; 8];
    for i in 0..8 {
        for j in 0..8 {
            let mut s = 0.0;
            for k in 0..8 {
                s += a[i][k] * b[k][j];
            }
            out[i][j] = s;
        }
    }
    out
}

fn transpose_8(a: &[[f64; 8]; 8]) -> [[f64; 8]; 8] {
    let mut out = [[0.0_f64; 8]; 8];
    for i in 0..8 {
        for j in 0..8 {
            out[i][j] = a[j][i];
        }
    }
    out
}

// ────────────────────────────────────────────────────────────────────────────
// SU(2)-form finder (verbatim from standalone)
// ────────────────────────────────────────────────────────────────────────────

fn round_to_zxi(z: Complex64, k: u32, tol: f64) -> Option<ZXi> {
    const RANGE: i64 = 50;
    let scale = SQRT2.powi(k as i32);
    let tr = scale * z.re;
    let ti = scale * z.im;
    for a in -RANGE..=RANGE {
        for d in -RANGE..=RANGE {
            let r = tr - (a as f64);
            let s = ti - (d as f64);
            let b = 2.0 * (SQRT3 / 2.0 * r - 0.5 * s);
            let c = 2.0 * (-0.5 * r + SQRT3 / 2.0 * s);
            let br = b.round() as i64;
            let cr = c.round() as i64;
            if (b - br as f64).abs() > tol || (c - cr as f64).abs() > tol {
                continue;
            }
            let cand: ZXi = [a, br, cr, d];
            let err = zxi_to_c(cand) - Complex64::new(tr, ti);
            if err.norm() < tol {
                return Some(cand);
            }
        }
    }
    None
}

fn find_su2_form(
    u_mat: &[[Complex64; 2]; 2],
    max_k: u32,
    tol: f64,
) -> Option<(ZXi, ZXi, u32, u32, u32)> {
    let x = xi();
    for k in 0..=max_k {
        let scale = SQRT2.powi(k as i32);
        for ph_idx in 0..12 {
            let phase = x.powi(ph_idx as i32);
            let up = [
                [phase * u_mat[0][0], phase * u_mat[0][1]],
                [phase * u_mat[1][0], phase * u_mat[1][1]],
            ];
            let u_round = round_to_zxi(up[0][0], k, tol);
            let t_round = round_to_zxi(up[1][0], k, tol);
            let (u, t) = match (u_round, t_round) {
                (Some(u), Some(t)) => (u, t),
                _ => continue,
            };
            let u_bar = zxi_to_c(zxi_conj(u));
            let t_bar = zxi_to_c(zxi_conj(t));
            let target_01 = Complex64::new(scale, 0.0) * up[0][1];
            let target_11 = Complex64::new(scale, 0.0) * up[1][1];
            for l in 0..12u32 {
                let xil = x.powi(l as i32);
                let ok11 = (u_bar * xil - target_11).norm() < tol;
                let ok01 = (-t_bar * xil - target_01).norm() < tol;
                if ok11 && ok01 {
                    return Some((u, t, k, l, ph_idx as u32));
                }
            }
        }
    }
    None
}

// ────────────────────────────────────────────────────────────────────────────
// Gate helpers
//
// Clifford gates come from the crate's CLIFFORD_TABLE_T via to_float().
// R = diag(1, ξ) is kept as a local definition because the crate's
// rz_pi6_mat() = diag(e^{−iπ/12}, e^{iπ/12}) is NOT in ℤ[ξ] and would
// cause find_su2_form() to fail.
// ────────────────────────────────────────────────────────────────────────────

type Mat2 = [[Complex64; 2]; 2];

fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

/// Fetch a named gate from CLIFFORD_TABLE_T as a float Mat2.
fn clifford(name: &str) -> Mat2 {
    CLIFFORD_TABLE_T
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, u2t)| u2t.to_float())
        .unwrap_or_else(|| panic!("gate '{}' not found in CLIFFORD_TABLE_T", name))
}

/// R = T₆ = diag(1, ξ) — the generator of the n=6 gate set.
/// Uses the T₆ convention (determinant = ξ, NOT ±1) so that u = [1,0,0,0] ∈ ℤ[ξ]
/// is an exact integer and find_su2_form() succeeds at k=0.
fn gate_r() -> Mat2 {
    [[c(1.0, 0.0), c(0.0, 0.0)], [c(0.0, 0.0), xi()]]
}

fn mat2_mul(a: Mat2, b: Mat2) -> Mat2 {
    let mut out = [[c(0.0, 0.0); 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                out[i][j] = out[i][j] + a[i][k] * b[k][j];
            }
        }
    }
    out
}

fn mat2_prod(ms: &[Mat2]) -> Mat2 {
    let mut out = [[c(1.0, 0.0), c(0.0, 0.0)], [c(0.0, 0.0), c(1.0, 0.0)]];
    for m in ms {
        out = mat2_mul(out, *m);
    }
    out
}

// ────────────────────────────────────────────────────────────────────────────
// Four invariants (verbatim from standalone)
// ────────────────────────────────────────────────────────────────────────────

struct GateResult {
    name: &'static str,
    ok: bool,
    msg: String,
}

fn check_invariants(
    name: &'static str,
    u_mat: Mat2,
    sigma: &[[f64; 8]; 8],
    gram: &[[f64; 8]; 8],
) -> GateResult {
    let mut msg = format!("\n=== {} ===\n", name);
    let mut ok = true;

    let res = find_su2_form(&u_mat, 5, TOL);
    let (u, t, k, l, ph) = match res {
        Some(r) => r,
        None => {
            msg.push_str("  ✗ could not place into SU(2) form within max_k=5\n");
            return GateResult {
                name,
                ok: false,
                msg,
            };
        }
    };
    msg.push_str(&format!(
        "  u = {:?}, t = {:?}, k = {}, ξˡ phase l = {}, pre-phase η = ξ^{}\n",
        u, t, k, l, ph
    ));

    let x: [f64; 8] = [
        u[0] as f64,
        u[1] as f64,
        u[2] as f64,
        u[3] as f64,
        t[0] as f64,
        t[1] as f64,
        t[2] as f64,
        t[3] as f64,
    ];

    // (1) Σ_top · x / √2^k == column 0 of (ξ^ph · U), in (Re, Im, Re, Im) form
    let sx = mat_vec_8(sigma, &x);
    let scale = SQRT2.powi(k as i32);
    let got = [sx[0] / scale, sx[1] / scale, sx[2] / scale, sx[3] / scale];
    let phase = xi().powi(ph as i32);
    let col0 = [phase * u_mat[0][0], phase * u_mat[1][0]];
    let expected = [col0[0].re, col0[0].im, col0[1].re, col0[1].im];
    let err1: f64 = got
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt();
    let ok1 = err1 < 1e-9;
    msg.push_str(&format!(
        "  (1) Σ_top·x / √2^k matches column 0:        err = {:.2e}  {}\n",
        err1,
        if ok1 { "✓" } else { "✗" }
    ));
    if !ok1 {
        ok = false;
    }

    // (2) α(u) + α(t) == 2^k
    let asum = alpha(u) + alpha(t);
    let expected_pow = 1_i64 << k;
    let ok2 = asum == expected_pow;
    msg.push_str(&format!(
        "  (2) α(u)+α(t) = {}, expected 2^{} = {}:    {}\n",
        asum,
        k,
        expected_pow,
        if ok2 { "✓" } else { "✗" }
    ));
    if !ok2 {
        ok = false;
    }

    // (3) β(u) + β(t) == 0
    let bsum = beta(u) + beta(t);
    let ok3 = bsum == 0;
    msg.push_str(&format!(
        "  (3) β(u)+β(t) = {}, expected 0:              {}\n",
        bsum,
        if ok3 { "✓" } else { "✗" }
    ));
    if !ok3 {
        ok = false;
    }

    // (4) ‖Σx‖²_E == 2^{k+1}    and    xᵀG x == 2^{k+1}
    let nsq: f64 = sx.iter().map(|v| v * v).sum();
    let xt_g_x: f64 = {
        let gx = mat_vec_8(gram, &x);
        x.iter().zip(gx.iter()).map(|(a, b)| a * b).sum()
    };
    let expected_norm = (1_i64 << (k + 1)) as f64;
    let ok4a = (nsq - expected_norm).abs() < 1e-9;
    let ok4b = (xt_g_x - expected_norm).abs() < 1e-9;
    msg.push_str(&format!(
        "  (4) ‖Σx‖²_E = {:.6}, expected 2^{} = {:.0}: {}\n",
        nsq,
        k + 1,
        expected_norm,
        if ok4a { "✓" } else { "✗" }
    ));
    msg.push_str(&format!(
        "      xᵀG x   = {:.6}  (cross-check via G):   {}\n",
        xt_g_x,
        if ok4b { "✓" } else { "✗" }
    ));
    if !ok4a || !ok4b {
        ok = false;
    }

    GateResult { name, ok, msg }
}

// ────────────────────────────────────────────────────────────────────────────
// Six structural checks (verbatim from standalone)
// ────────────────────────────────────────────────────────────────────────────

fn structural_checks(sigma: &[[f64; 8]; 8], gram: &[[f64; 8]; 8]) -> bool {
    println!("========================================================================");
    println!("STRUCTURAL CHECKS on Σ and G");
    println!("========================================================================");
    let mut ok = true;

    // Σ_top Σ_topᵀ should match [[2, √3/2, 0, 0], …]
    let mut sigma_top = [[0.0_f64; 8]; 4];
    for i in 0..4 {
        for j in 0..8 {
            sigma_top[i][j] = sigma[i][j];
        }
    }
    let mut sttt = [[0.0_f64; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            let mut s = 0.0;
            for k in 0..8 {
                s += sigma_top[i][k] * sigma_top[j][k];
            }
            sttt[i][j] = s;
        }
    }
    let expected_sttt = [
        [2.0, SQRT3 / 2.0, 0.0, 0.0],
        [SQRT3 / 2.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 2.0, SQRT3 / 2.0],
        [0.0, 0.0, SQRT3 / 2.0, 2.0],
    ];
    let err: f64 = (0..4)
        .flat_map(|i| (0..4).map(move |j| (i, j)))
        .map(|(i, j)| (sttt[i][j] - expected_sttt[i][j]).powi(2))
        .sum::<f64>()
        .sqrt();
    let ok1 = err < 1e-12;
    println!(
        "  Σ_top Σ_topᵀ matches expected (with √3/2 off-diag):  err = {:.2e}  {}",
        err,
        if ok1 { "✓" } else { "✗" }
    );
    if !ok1 {
        ok = false;
    }

    // ΣᵀΣ per 4-block should match [[2,0,1,0],[0,2,0,1],[1,0,2,0],[0,1,0,2]]
    let expected_block = [
        [2.0, 0.0, 1.0, 0.0],
        [0.0, 2.0, 0.0, 1.0],
        [1.0, 0.0, 2.0, 0.0],
        [0.0, 1.0, 0.0, 2.0],
    ];
    let err: f64 = (0..4)
        .flat_map(|i| (0..4).map(move |j| (i, j)))
        .map(|(i, j)| (gram[i][j] - expected_block[i][j]).powi(2))
        .sum::<f64>()
        .sqrt();
    let ok2 = err < 1e-12;
    println!(
        "  ΣᵀΣ per 4-block matches [[2,0,1,0],[0,2,0,1],…]:    err = {:.2e}  {}",
        err,
        if ok2 { "✓" } else { "✗" }
    );
    if !ok2 {
        ok = false;
    }

    // ΣᵀΣ block-diagonal (no u-t coupling)
    let off: f64 = (0..4)
        .flat_map(|i| (4..8).map(move |j| (i, j)))
        .map(|(i, j)| gram[i][j].powi(2))
        .sum::<f64>()
        .sqrt();
    let ok3 = off < 1e-12;
    println!(
        "  ΣᵀΣ is block-diagonal (u ⊥ t):                       off = {:.2e}  {}",
        off,
        if ok3 { "✓" } else { "✗" }
    );
    if !ok3 {
        ok = false;
    }

    // det(G_block) == 9
    fn det4(m: &[[f64; 4]; 4]) -> f64 {
        let mut s = 0.0;
        for j in 0..4 {
            let mut minor = [[0.0_f64; 3]; 3];
            for i in 1..4 {
                let mut col = 0;
                for jj in 0..4 {
                    if jj == j {
                        continue;
                    }
                    minor[i - 1][col] = m[i][jj];
                    col += 1;
                }
            }
            let sub = minor[0][0] * (minor[1][1] * minor[2][2] - minor[1][2] * minor[2][1])
                - minor[0][1] * (minor[1][0] * minor[2][2] - minor[1][2] * minor[2][0])
                + minor[0][2] * (minor[1][0] * minor[2][1] - minor[1][1] * minor[2][0]);
            let sign = if j % 2 == 0 { 1.0 } else { -1.0 };
            s += sign * m[0][j] * sub;
        }
        s
    }
    let det = det4(&expected_block);
    let ok4 = (det - 9.0).abs() < 1e-9;
    println!(
        "  det(G_block) = {:.6}, expected 9:                   {}",
        det,
        if ok4 { "✓" } else { "✗" }
    );
    if !ok4 {
        ok = false;
    }

    // σ₅(√3) == −√3
    let x = xi();
    let sqrt3_bullet = Complex64::new(2.0, 0.0) * x.powi(5) - x.powi(15);
    let err = (sqrt3_bullet - Complex64::new(-SQRT3, 0.0)).norm();
    let ok5 = err < 1e-9;
    println!(
        "  σ₅(√3) = −√3 (numeric):                              err = {:.2e}  {}",
        err,
        if ok5 { "✓" } else { "✗" }
    );
    if !ok5 {
        ok = false;
    }

    // σ₅ fixes i: ξ^15 == ξ^3
    let err = (x.powi(15) - x.powi(3)).norm();
    let ok6 = err < 1e-9;
    println!(
        "  σ₅(i) = i (σ₅ is NOT complex conj):                   err = {:.2e}  {}",
        err,
        if ok6 { "✓" } else { "✗" }
    );
    if !ok6 {
        ok = false;
    }

    // zxi_conj formula matches numerical conjugate
    let mut conj_ok = true;
    let test_elements: [ZXi; 5] = [
        [1, 2, -1, 3],
        [-2, 1, 3, 0],
        [3, -4, 2, -1],
        [0, 1, 0, -1],
        [5, -3, 1, 2],
    ];
    for p in test_elements {
        let p_conj = zxi_conj(p);
        let num_conj = zxi_to_c(p).conj();
        let err = (zxi_to_c(p_conj) - num_conj).norm();
        if err >= 1e-9 {
            conj_ok = false;
            println!(
                "  ✗ conj formula failed on {:?}: got {:?}, err={:.2e}",
                p, p_conj, err
            );
        }
    }
    println!(
        "  zxi_conj formula matches numeric conjugate:                              {}",
        if conj_ok { "✓" } else { "✗" }
    );
    if !conj_ok {
        ok = false;
    }

    // Cross-check: G from ΣᵀΣ matches SIGMA_GRAM_U (u-block)
    let crate_sigma_t = transpose_8(sigma);
    let derived_gram = mat_mat_8(&crate_sigma_t, sigma);
    let mut gram_err = 0.0_f64;
    for i in 0..4 {
        for j in 0..4 {
            gram_err += (derived_gram[i][j] - SIGMA_GRAM_U[i][j] as f64).powi(2);
        }
    }
    let gram_err = gram_err.sqrt();
    let gram_ok = gram_err < 1e-12;
    println!(
        "  G from ΣᵀΣ matches crate SIGMA_GRAM_U (u-block):      err = {:.2e}  {}",
        gram_err,
        if gram_ok { "✓" } else { "✗" }
    );
    if !gram_ok {
        ok = false;
    }

    ok
}

// ────────────────────────────────────────────────────────────────────────────
// Test entry point
// ────────────────────────────────────────────────────────────────────────────

#[test]
fn n6_lattice_invariants() {
    println!("Verifying n=6 (Clifford+R_z(π/6)) lattice math.");
    println!("  Using crate's sigma_matrix() (rows reordered to standalone convention).");
    println!("  Using crate's CLIFFORD_TABLE_T for H, S, X, Y, Z.");
    println!("  Using local gate_r() = diag(1, ξ) for R (T₆ convention).\n");

    let sigma = crate_sigma_reordered();
    let gram = build_gram(&sigma);

    let struct_ok = structural_checks(&sigma, &gram);

    println!("\n========================================================================");
    println!("GATE-BY-GATE INVARIANT CHECKS");
    println!("========================================================================");

    let h = clifford("H");
    let s = clifford("S");
    let z = clifford("Z");
    let x = clifford("X");
    let r = gate_r();

    let zoo: &[(&'static str, Mat2)] = &[
        ("I", clifford("I")),
        ("H", h),
        ("S", s),
        ("Z", z),
        ("X", x),
        ("R^3 = S?", mat2_prod(&[r, r, r])),
        ("R^6 = -I?", mat2_prod(&[r, r, r, r, r, r])),
        ("R", r),
        ("R^2", mat2_prod(&[r, r])),
        ("R^4", mat2_prod(&[r, r, r, r])),
        ("R^5", mat2_prod(&[r, r, r, r, r])),
        ("H·R", mat2_prod(&[h, r])),
        ("H·S·R", mat2_prod(&[h, s, r])),
        ("H·R^2", mat2_prod(&[h, r, r])),
        ("H·S·R^2", mat2_prod(&[h, s, r, r])),
        ("R·H·R", mat2_prod(&[r, h, r])),
        ("R^2·H·R^2", mat2_prod(&[r, r, h, r, r])),
        ("H·R·H", mat2_prod(&[h, r, h])),
        ("H·R·H·R·H", mat2_prod(&[h, r, h, r, h])),
    ];

    let mut results: Vec<GateResult> = Vec::new();
    for &(name, u_mat) in zoo {
        let gr = check_invariants(name, u_mat, &sigma, &gram);
        print!("{}", gr.msg);
        results.push(gr);
    }

    println!("\n========================================================================");
    println!("SUMMARY");
    println!("========================================================================");
    println!(
        "  structural checks: {}",
        if struct_ok { "PASS" } else { "FAIL" }
    );
    let n_pass = results.iter().filter(|r| r.ok).count();
    println!("  gate invariants:   {}/{} pass", n_pass, results.len());
    for r in &results {
        println!("    {}  {}", if r.ok { "✓" } else { "✗" }, r.name);
    }
    let all_ok = struct_ok && results.iter().all(|r| r.ok);

    assert!(
        all_ok,
        "n=6 lattice invariant check FAILED — see output above"
    );
}
