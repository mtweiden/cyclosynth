//! verify_n6_lattice.rs — first-principles verification of the n=6 lattice math.
//!
//! Self-contained: depends only on the standard library plus `num_complex`
//! (which the project already pulls in for `zomicron.rs`).
//!
//! Drop this into the crate as a test file (e.g. `tests/verify_n6_lattice.rs`)
//! or run it as a `cargo test --release` target.  Every gate in the zoo
//! verifies the four invariants below.  Plus six structural checks on Σ and G.
//!
//! The invariants any correct n=6 lattice implementation MUST satisfy
//! (for U  =  (1/√2^k) · [[u, −t̄·ξˡ], [t, ū·ξˡ]],  u,t ∈ ℤ[ξ]):
//!
//!   (1) Σ_top · x / √2^k  reproduces (Re u₁, Im u₁, Re u₂, Im u₂) of η·U[:,0]
//!   (2) α(u) + α(t) = 2^k                                (rational part)
//!   (3) β(u) + β(t) = 0                                  (√3 part)
//!   (4) ‖Σx‖²_E = xᵀG x = 2^{k+1}
//!
//! where  α(u) = a²+b²+c²+d² + ac + bd     and     β(u) = ab + bc + cd
//! for u = a + bξ + cξ² + dξ³.

use num_complex::Complex64;
use std::f64::consts::PI;

const SQRT2: f64 = std::f64::consts::SQRT_2;
const SQRT3: f64 = 1.7320508075688772_f64;
const TOL:   f64 = 1e-9;

// ────────────────────────────────────────────────────────────────────────────
// Algebra of ℤ[ξ], ξ = e^{iπ/6}.  Matches zomicron.rs conventions.
// ────────────────────────────────────────────────────────────────────────────

type ZXi = [i64; 4];                  // (a, b, c, d) in basis {1, ξ, ξ², ξ³}

fn xi() -> Complex64 {
    Complex64::from_polar(1.0, PI / 6.0)
}

/// (a,b,c,d) ↦ a + bξ + cξ² + dξ³ as a Complex64.
fn zxi_to_c(p: ZXi) -> Complex64 {
    let x = xi();
    Complex64::new(p[0] as f64, 0.0)
        + Complex64::new(p[1] as f64, 0.0) * x
        + Complex64::new(p[2] as f64, 0.0) * x * x
        + Complex64::new(p[3] as f64, 0.0) * x * x * x
}

/// Complex conjugate in ℤ[ξ] (closed form, from zomicron.rs):
/// conj(a + bξ + cξ² + dξ³) = (a+c) + bξ − cξ² + (−b−d)ξ³.
fn zxi_conj(p: ZXi) -> ZXi {
    [p[0] + p[2], p[1], -p[2], -p[1] - p[3]]
}

/// α(p) = a² + b² + c² + d² + ac + bd       (rational part of |p|² in ℤ[√3])
fn alpha(p: ZXi) -> i64 {
    p[0]*p[0] + p[1]*p[1] + p[2]*p[2] + p[3]*p[3] + p[0]*p[2] + p[1]*p[3]
}

/// β(p) = ab + bc + cd                       (√3-coefficient of |p|²)
fn beta(p: ZXi) -> i64 {
    p[0]*p[1] + p[1]*p[2] + p[2]*p[3]
}

/// ξ^k as a ZXi tuple, for any k.  Uses ξ⁴ = ξ² − 1 to reduce.
fn xi_pow(k: i32) -> ZXi {
    let k = ((k % 12) + 12) % 12;
    match k {
        0  => [ 1, 0, 0, 0],
        1  => [ 0, 1, 0, 0],
        2  => [ 0, 0, 1, 0],
        3  => [ 0, 0, 0, 1],
        4  => [-1, 0, 1, 0],         // ξ⁴ = ξ² − 1
        5  => [ 0,-1, 0, 1],         // ξ⁵ = ξ³ − ξ
        6  => [-1, 0, 0, 0],         // ξ⁶ = −1
        7  => [ 0,-1, 0, 0],
        8  => [ 0, 0,-1, 0],
        9  => [ 0, 0, 0,-1],
        10 => [ 1, 0,-1, 0],
        11 => [ 0, 1, 0,-1],
        _ => unreachable!(),
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Σ matrix (8×8) — cap σ₁ on top, bullet σ₅ (ξ ↦ ξ⁵, √3 ↦ −√3) on bottom.
// ────────────────────────────────────────────────────────────────────────────

fn build_sigma() -> [[f64; 8]; 8] {
    let mut s = [[0.0_f64; 8]; 8];
    let x = xi();
    for j in 0..4 {
        let bj_cap = x.powi(j as i32);
        let bj_bul = x.powi(5 * j as i32);
        s[0][j  ]   = bj_cap.re;        // Re σ₁(u)
        s[1][j  ]   = bj_cap.im;        // Im σ₁(u)
        s[2][j+4]   = bj_cap.re;        // Re σ₁(t)
        s[3][j+4]   = bj_cap.im;        // Im σ₁(t)
        s[4][j  ]   = bj_bul.re;        // Re σ₅(u)
        s[5][j  ]   = bj_bul.im;        // Im σ₅(u)
        s[6][j+4]   = bj_bul.re;        // Re σ₅(t)
        s[7][j+4]   = bj_bul.im;        // Im σ₅(t)
    }
    s
}

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
    for i in 0..8 { for j in 0..8 { out[i][j] = a[j][i]; }}
    out
}

// ────────────────────────────────────────────────────────────────────────────
// SU(2)-form finder: given a 2×2 complex unitary U, find integers
// (u, t, k, l, η_idx) such that
//      √2^k · (ξ^η_idx · U)  =  [[u, −t̄·ξˡ], [t, ū·ξˡ]]
// where u, t ∈ ℤ[ξ].
// ────────────────────────────────────────────────────────────────────────────

/// Try to round `z ∈ ℂ` to (a,b,c,d) ∈ ℤ⁴ such that  √2^k · z  ≈  a + bξ + cξ² + dξ³.
/// Brute-forces over a, d ∈ [-RANGE, RANGE] and solves the 2×2 system for (b, c).
fn round_to_zxi(z: Complex64, k: u32, tol: f64) -> Option<ZXi> {
    const RANGE: i64 = 50;
    let scale = SQRT2.powi(k as i32);
    let tr = scale * z.re;
    let ti = scale * z.im;
    // Re(target) = a + b·(√3/2) + c·(1/2)
    // Im(target) = d + b·(1/2)  + c·(√3/2)
    // Inverse of [[√3/2, 1/2], [1/2, √3/2]] is 2·[[√3/2, -1/2], [-1/2, √3/2]]
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

/// Find (u, t, k, l, η_idx) such that
///   √2^k · (ξ^η_idx · U)  =  [[u, −t̄·ξˡ], [t, ū·ξˡ]],   u,t ∈ ℤ[ξ].
fn find_su2_form(u_mat: &[[Complex64; 2]; 2], max_k: u32, tol: f64)
    -> Option<(ZXi, ZXi, u32, u32, u32)>
{
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
// Gate zoo and matrix helpers
// ────────────────────────────────────────────────────────────────────────────

type Mat2 = [[Complex64; 2]; 2];

fn c(re: f64, im: f64) -> Complex64 { Complex64::new(re, im) }

fn mat2_mul(a: Mat2, b: Mat2) -> Mat2 {
    let mut out = [[c(0.0, 0.0); 2]; 2];
    for i in 0..2 { for j in 0..2 { for k in 0..2 {
        out[i][j] = out[i][j] + a[i][k] * b[k][j];
    }}}
    out
}

fn mat2_prod(ms: &[Mat2]) -> Mat2 {
    let mut out = [[c(1.0, 0.0), c(0.0, 0.0)],
                   [c(0.0, 0.0), c(1.0, 0.0)]];
    for m in ms {
        out = mat2_mul(out, *m);
    }
    out
}

fn gate_h() -> Mat2 {
    let s = 1.0 / SQRT2;
    [[c(s, 0.0), c( s, 0.0)],
     [c(s, 0.0), c(-s, 0.0)]]
}
fn gate_s() -> Mat2 { [[c(1.0, 0.0), c(0.0, 0.0)], [c(0.0, 0.0), c(0.0, 1.0)]] }
fn gate_x() -> Mat2 { [[c(0.0, 0.0), c(1.0, 0.0)], [c(1.0, 0.0), c(0.0, 0.0)]] }
fn gate_y() -> Mat2 { [[c(0.0, 0.0), c(0.0,-1.0)], [c(0.0, 1.0), c(0.0, 0.0)]] }
fn gate_z() -> Mat2 { [[c(1.0, 0.0), c(0.0, 0.0)], [c(0.0, 0.0), c(-1.0,0.0)]] }
fn gate_i() -> Mat2 { [[c(1.0, 0.0), c(0.0, 0.0)], [c(0.0, 0.0), c(1.0, 0.0)]] }

/// R = T_6 = diag(1, e^{iπ/6}) = diag(1, ξ)
fn gate_r() -> Mat2 {
    [[c(1.0, 0.0), c(0.0, 0.0)],
     [c(0.0, 0.0), xi()]]
}

// ────────────────────────────────────────────────────────────────────────────
// The four invariants — the heart of the check
// ────────────────────────────────────────────────────────────────────────────

struct GateResult {
    name: &'static str,
    ok:   bool,
    msg:  String,
}

fn check_invariants(name: &'static str, u_mat: Mat2, sigma: &[[f64; 8]; 8],
                    gram: &[[f64; 8]; 8]) -> GateResult {
    let mut msg = format!("\n=== {} ===\n", name);
    let mut ok = true;

    let res = find_su2_form(&u_mat, 5, TOL);
    let (u, t, k, l, ph) = match res {
        Some(r) => r,
        None => {
            msg.push_str("  ✗ could not place into SU(2) form within max_k=5\n");
            return GateResult { name, ok: false, msg };
        }
    };
    msg.push_str(&format!(
        "  u = {:?}, t = {:?}, k = {}, ξˡ phase l = {}, pre-phase η = ξ^{}\n",
        u, t, k, l, ph
    ));

    // x = (u || t) as f64 vector
    let x: [f64; 8] = [
        u[0] as f64, u[1] as f64, u[2] as f64, u[3] as f64,
        t[0] as f64, t[1] as f64, t[2] as f64, t[3] as f64,
    ];

    // (1) Σ_top · x / √2^k == column 0 of (ξ^ph · U), in (Re, Im, Re, Im) form
    let sx = mat_vec_8(sigma, &x);
    let scale = SQRT2.powi(k as i32);
    let got = [sx[0] / scale, sx[1] / scale, sx[2] / scale, sx[3] / scale];
    let phase = xi().powi(ph as i32);
    let col0 = [phase * u_mat[0][0], phase * u_mat[1][0]];
    let expected = [col0[0].re, col0[0].im, col0[1].re, col0[1].im];
    let err1: f64 = got.iter().zip(expected.iter())
        .map(|(a,b)| (a-b)*(a-b)).sum::<f64>().sqrt();
    let ok1 = err1 < 1e-9;
    msg.push_str(&format!(
        "  (1) Σ_top·x / √2^k matches column 0:        err = {:.2e}  {}\n",
        err1, if ok1 { "✓" } else { "✗" }
    ));
    if !ok1 { ok = false; }

    // (2) α(u) + α(t) == 2^k
    let asum = alpha(u) + alpha(t);
    let expected_pow = 1_i64 << k;
    let ok2 = asum == expected_pow;
    msg.push_str(&format!(
        "  (2) α(u)+α(t) = {}, expected 2^{} = {}:    {}\n",
        asum, k, expected_pow, if ok2 { "✓" } else { "✗" }
    ));
    if !ok2 { ok = false; }

    // (3) β(u) + β(t) == 0
    let bsum = beta(u) + beta(t);
    let ok3 = bsum == 0;
    msg.push_str(&format!(
        "  (3) β(u)+β(t) = {}, expected 0:              {}\n",
        bsum, if ok3 { "✓" } else { "✗" }
    ));
    if !ok3 { ok = false; }

    // (4) ‖Σx‖²_E == 2^{k+1}    and    xᵀG x == 2^{k+1}
    let nsq: f64 = sx.iter().map(|v| v*v).sum();
    let xt_g_x: f64 = {
        let gx = mat_vec_8(gram, &x);
        x.iter().zip(gx.iter()).map(|(a,b)| a*b).sum()
    };
    let expected_norm = (1_i64 << (k+1)) as f64;
    let ok4a = (nsq - expected_norm).abs() < 1e-9;
    let ok4b = (xt_g_x - expected_norm).abs() < 1e-9;
    msg.push_str(&format!(
        "  (4) ‖Σx‖²_E = {:.6}, expected 2^{} = {:.0}: {}\n",
        nsq, k+1, expected_norm, if ok4a { "✓" } else { "✗" }
    ));
    msg.push_str(&format!(
        "      xᵀG x   = {:.6}  (cross-check via G):   {}\n",
        xt_g_x, if ok4b { "✓" } else { "✗" }
    ));
    if !ok4a || !ok4b { ok = false; }

    GateResult { name, ok, msg }
}

// ────────────────────────────────────────────────────────────────────────────
// Structural checks (independent of any particular unitary)
// ────────────────────────────────────────────────────────────────────────────

fn structural_checks(sigma: &[[f64; 8]; 8], gram: &[[f64; 8]; 8]) -> bool {
    println!("========================================================================");
    println!("STRUCTURAL CHECKS on Σ and G");
    println!("========================================================================");
    let mut ok = true;

    // Σ_top Σ_topᵀ should be [[2, √3/2, 0, 0], [√3/2, 2, 0, 0], [0, 0, 2, √3/2], [0, 0, √3/2, 2]]
    let mut sigma_top = [[0.0_f64; 8]; 4];
    for i in 0..4 { for j in 0..8 { sigma_top[i][j] = sigma[i][j]; }}
    let mut sttt = [[0.0_f64; 4]; 4];
    for i in 0..4 { for j in 0..4 {
        let mut s = 0.0;
        for k in 0..8 { s += sigma_top[i][k] * sigma_top[j][k]; }
        sttt[i][j] = s;
    }}
    let expected_sttt = [
        [2.0,      SQRT3/2.0, 0.0,      0.0     ],
        [SQRT3/2.0, 2.0,      0.0,      0.0     ],
        [0.0,      0.0,       2.0,      SQRT3/2.0],
        [0.0,      0.0,       SQRT3/2.0, 2.0    ],
    ];
    let err: f64 = (0..4).flat_map(|i| (0..4).map(move |j| (i,j)))
        .map(|(i,j)| (sttt[i][j] - expected_sttt[i][j]).powi(2))
        .sum::<f64>().sqrt();
    let ok1 = err < 1e-12;
    println!("  Σ_top Σ_topᵀ matches expected (with √3/2 off-diag):  err = {:.2e}  {}",
             err, if ok1 { "✓" } else { "✗" });
    if !ok1 { ok = false; }

    // ΣᵀΣ per 4-block should match [[2,0,1,0],[0,2,0,1],[1,0,2,0],[0,1,0,2]]
    let expected_block = [
        [2.0, 0.0, 1.0, 0.0],
        [0.0, 2.0, 0.0, 1.0],
        [1.0, 0.0, 2.0, 0.0],
        [0.0, 1.0, 0.0, 2.0],
    ];
    let err: f64 = (0..4).flat_map(|i| (0..4).map(move |j| (i,j)))
        .map(|(i,j)| (gram[i][j] - expected_block[i][j]).powi(2))
        .sum::<f64>().sqrt();
    let ok2 = err < 1e-12;
    println!("  ΣᵀΣ per 4-block matches [[2,0,1,0],[0,2,0,1],…]:    err = {:.2e}  {}",
             err, if ok2 { "✓" } else { "✗" });
    if !ok2 { ok = false; }

    // ΣᵀΣ block-diagonal (no u-t coupling)
    let off: f64 = (0..4).flat_map(|i| (4..8).map(move |j| (i,j)))
        .map(|(i,j)| gram[i][j].powi(2))
        .sum::<f64>().sqrt();
    let ok3 = off < 1e-12;
    println!("  ΣᵀΣ is block-diagonal (u ⊥ t):                       off = {:.2e}  {}",
             off, if ok3 { "✓" } else { "✗" });
    if !ok3 { ok = false; }

    // Symbolic eigenvalues of G_block: {1,1,3,3}, det = 9.
    // Use the closed form rather than rolling a real eigensolver:
    //   G_block = 2I + P  where P = [[0,0,1,0],[0,0,0,1],[1,0,0,0],[0,1,0,0]],
    //   P is the swap (0,2)(1,3), eigenvalues ±1, multiplicities 2 each.
    //   So G_block has eigenvalues 2±1 = {1, 1, 3, 3}.
    //   det(G_block) = 9.
    // Cross-check det via 4x4 determinant of expected_block:
    fn det4(m: &[[f64; 4]; 4]) -> f64 {
        // expand along first row
        let mut s = 0.0;
        for j in 0..4 {
            let mut minor = [[0.0_f64; 3]; 3];
            for i in 1..4 {
                let mut col = 0;
                for jj in 0..4 {
                    if jj == j { continue; }
                    minor[i-1][col] = m[i][jj];
                    col += 1;
                }
            }
            let sub = minor[0][0]*(minor[1][1]*minor[2][2] - minor[1][2]*minor[2][1])
                    - minor[0][1]*(minor[1][0]*minor[2][2] - minor[1][2]*minor[2][0])
                    + minor[0][2]*(minor[1][0]*minor[2][1] - minor[1][1]*minor[2][0]);
            let sign = if j % 2 == 0 { 1.0 } else { -1.0 };
            s += sign * m[0][j] * sub;
        }
        s
    }
    let det = det4(&expected_block);
    let ok4 = (det - 9.0).abs() < 1e-9;
    println!("  det(G_block) = {:.6}, expected 9:                   {}",
             det, if ok4 { "✓" } else { "✗" });
    if !ok4 { ok = false; }

    // σ₅(√3) = −√3 numerically.  √3 = 2ξ − ξ³.  σ₅(2ξ − ξ³) = 2ξ⁵ − ξ¹⁵ = 2ξ⁵ − ξ³.
    let x = xi();
    let sqrt3_bullet = Complex64::new(2.0, 0.0) * x.powi(5) - x.powi(15);
    let err = (sqrt3_bullet - Complex64::new(-SQRT3, 0.0)).norm();
    let ok5 = err < 1e-9;
    println!("  σ₅(√3) = −√3 (numeric):                              err = {:.2e}  {}",
             err, if ok5 { "✓" } else { "✗" });
    if !ok5 { ok = false; }

    // σ₅ fixes i: ξ¹⁵ = ξ³ = i
    let err = (x.powi(15) - x.powi(3)).norm();
    let ok6 = err < 1e-9;
    println!("  σ₅(i) = i (σ₅ is NOT complex conj):                   err = {:.2e}  {}",
             err, if ok6 { "✓" } else { "✗" });
    if !ok6 { ok = false; }

    // zxi_conj formula matches numerical conjugate on a few elements
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
            println!("  ✗ conj formula failed on {:?}: got {:?}, err={:.2e}", p, p_conj, err);
        }
    }
    println!("  zxi_conj formula matches numeric conjugate:                              {}",
             if conj_ok { "✓" } else { "✗" });
    if !conj_ok { ok = false; }

    // Suppress the unused-import warning for mat_mat_8 / transpose_8 — they're here
    // for callers that want to derive G = ΣᵀΣ themselves rather than trusting the
    // pre-computed `gram`.
    let _ = mat_mat_8(&transpose_8(sigma), sigma);

    ok
}

// ────────────────────────────────────────────────────────────────────────────
// Entry point
// ────────────────────────────────────────────────────────────────────────────

fn main() {
    println!("Verifying n=6 (Clifford+R_z(π/6)) lattice math from first principles.");
    println!("  ξ = e^(iπ/6) = {}", xi());
    println!("  basis: {{1, ξ, ξ², ξ³}}");
    println!("  bullet automorphism: σ₅ (ξ ↦ ξ⁵, √3 ↦ −√3, fixes i)\n");

    let sigma = build_sigma();
    let sigma_t = transpose_8(&sigma);
    let gram = mat_mat_8(&sigma_t, &sigma);

    let struct_ok = structural_checks(&sigma, &gram);

    println!("\n========================================================================");
    println!("GATE-BY-GATE INVARIANT CHECKS");
    println!("========================================================================");

    let h = gate_h(); let s = gate_s(); let r = gate_r();
    let i_g = gate_i(); let x_g = gate_x(); let y_g = gate_y(); let z_g = gate_z();

    let zoo: Vec<(&'static str, Mat2)> = vec![
        ("I",          i_g),
        ("H",          h),
        ("S",          s),
        ("Z",          z_g),
        ("X",          x_g),
        ("Y",          y_g),
        ("R^3 = S",    mat2_prod(&[r, r, r])),
        ("R^6 = -I",   mat2_prod(&[r, r, r, r, r, r])),
        ("R",          r),
        ("R^2",        mat2_prod(&[r, r])),
        ("R^4",        mat2_prod(&[r, r, r, r])),
        ("R^5",        mat2_prod(&[r, r, r, r, r])),
        ("H·R",        mat2_prod(&[h, r])),
        ("H·S·R",      mat2_prod(&[h, s, r])),
        ("H·R^2",      mat2_prod(&[h, r, r])),
        ("H·S·R^2",    mat2_prod(&[h, s, r, r])),
        ("R·H·R",      mat2_prod(&[r, h, r])),
        ("R^2·H·R^2",  mat2_prod(&[r, r, h, r, r])),
        ("H·R·H",      mat2_prod(&[h, r, h])),
        ("H·R·H·R·H",  mat2_prod(&[h, r, h, r, h])),
    ];

    let mut results = Vec::with_capacity(zoo.len());
    for (name, u_mat) in zoo {
        let r = check_invariants(name, u_mat, &sigma, &gram);
        print!("{}", r.msg);
        results.push(r);
    }

    println!("\n========================================================================");
    println!("SUMMARY");
    println!("========================================================================");
    println!("  structural checks: {}", if struct_ok { "PASS" } else { "FAIL" });
    let n_pass = results.iter().filter(|r| r.ok).count();
    println!("  gate invariants:   {}/{} pass", n_pass, results.len());
    for r in &results {
        println!("    {}  {}", if r.ok { "✓" } else { "✗" }, r.name);
    }
    let all_ok = struct_ok && results.iter().all(|r| r.ok);
    println!("\n  OVERALL: {}",
             if all_ok { "PASS — n=6 lattice math is correct" } else { "FAIL — see above" });

    if !all_ok {
        std::process::exit(1);
    }
}

// ────────────────────────────────────────────────────────────────────────────
// As a #[test], so `cargo test` picks it up too.
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn n6_lattice_invariants() {
        let sigma = build_sigma();
        let sigma_t = transpose_8(&sigma);
        let gram = mat_mat_8(&sigma_t, &sigma);

        assert!(structural_checks(&sigma, &gram), "structural checks failed");

        let h = gate_h(); let s = gate_s(); let r = gate_r();
        let i_g = gate_i(); let x_g = gate_x(); let z_g = gate_z();

        let zoo: Vec<(&'static str, Mat2)> = vec![
            ("I", i_g), ("H", h), ("S", s), ("Z", z_g), ("X", x_g),
            ("R^3 = S",   mat2_prod(&[r, r, r])),
            ("R",         r),
            ("R^2",       mat2_prod(&[r, r])),
            ("R^4",       mat2_prod(&[r, r, r, r])),
            ("R^5",       mat2_prod(&[r, r, r, r, r])),
            ("H·R",       mat2_prod(&[h, r])),
            ("H·S·R",     mat2_prod(&[h, s, r])),
            ("H·R^2",     mat2_prod(&[h, r, r])),
            ("H·S·R^2",   mat2_prod(&[h, s, r, r])),
            ("R·H·R",     mat2_prod(&[r, h, r])),
            ("R^2·H·R^2", mat2_prod(&[r, r, h, r, r])),
            ("H·R·H",     mat2_prod(&[h, r, h])),
            ("H·R·H·R·H", mat2_prod(&[h, r, h, r, h])),
        ];

        for (name, u_mat) in zoo {
            let r = check_invariants(name, u_mat, &sigma, &gram);
            assert!(r.ok, "gate {} failed:\n{}", name, r.msg);
        }
    }
}
