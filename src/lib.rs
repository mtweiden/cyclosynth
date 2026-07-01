//! cyclosynth — single-qubit gate synthesis: approximate a target 2×2 unitary
//! by a Clifford+T or Clifford+√T circuit. See the [`synthesis`] module for
//! the domain glossary and the algorithm overview.

pub mod rings;
pub mod matrix;
pub mod synthesis;

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Diagnostic: take a Clifford+T gate word (H/S/T/X/Y/Z) and return our canonical
/// (MA normal-form) decomposition. Used to check our exact-synthesis T-count
/// VERIFY: I don't think this actually does MA form.
#[cfg(all(feature = "python", feature = "trace"))]
#[pyfunction]
fn decompose_gates_t(gates: &str) -> String {
    use matrix::U2T;
    let mut u = U2T::eye();
    for ch in gates.chars() {
        let g = match ch {
            'H' => U2T::h(),
            'S' => U2T::s(),
            'T' => U2T::t(),
            'X' => U2T::x(),
            'Y' => U2T::y(),
            'Z' => U2T::z(),
            _ => continue,
        };
        u = u * g;
    }
    synthesis::decomposer::BlochDecomposer.decompose(&u)
}

/// Diagnostic: trace the D&C decomposition of a gate word `x*` at split `t'`.
/// Builds x* (W dropped), takes its MA-canonical t'-T prefix as U_L, then the
/// inner factor for both branches: even U_R = U_L†·x* and odd U_R' = U_R·T†.
/// Returns (T(U_L), k(x*), T(U_R_even), k(U_R_even), T(U_R_odd), k(U_R_odd)).
/// Compare k(U_R) to the search's inner shell lde_inner to find an off-by-one.
#[cfg(all(feature = "python", feature = "trace"))]
#[pyfunction]
fn trace_inner(gates: &str, t_prime: u32) -> (u32, u32, u32, u32, u32, u32) {
    use matrix::U2T;
    let g_of = |ch: char| -> Option<U2T> {
        match ch {
            'H' => Some(U2T::h()), 'S' => Some(U2T::s()), 'T' => Some(U2T::t()),
            'X' => Some(U2T::x()), 'Y' => Some(U2T::y()), 'Z' => Some(U2T::z()),
            's' => Some(U2T::s().dagger()), 't' => Some(U2T::t().dagger()),
            _ => None,
        }
    };
    let count_t = |s: &str| s.chars().filter(|&c| c == 'T' || c == 't').count() as u32;

    let mut x = U2T::eye();
    for ch in gates.chars() {
        if let Some(g) = g_of(ch) { x = x * g; }
    }
    let x = x.reduced();
    let canon = synthesis::decomposer::BlochDecomposer.decompose(&x);

    // U_L = MA-canonical prefix up to and including the t'-th T-gate.
    let mut u_l = U2T::eye();
    let mut tc = 0u32;
    for ch in canon.chars() {
        if let Some(g) = g_of(ch) {
            u_l = u_l * g;
            if ch == 'T' || ch == 't' {
                tc += 1;
                if tc >= t_prime { break; }
            }
        }
    }
    let u_r_even = (u_l.dagger() * x).reduced();
    let u_r_odd = (u_r_even * U2T::t().dagger()).reduced();
    let t_re = count_t(&synthesis::decomposer::BlochDecomposer.decompose(&u_r_even));
    let t_ro = count_t(&synthesis::decomposer::BlochDecomposer.decompose(&u_r_odd));
    (tc, x.k, t_re, u_r_even.k, t_ro, u_r_odd.k)
}

/// Diagnostic: for a KNOWN D&C solution (gate word `x*`, split `t_prime`,
/// inner shell `k`, tolerance `eps`, target Rz((a_num/a_den)·π)), build the
/// inner factor's lattice vector `x_R` and the inner cap EXACTLY as the
/// production inner search builds them for that one prefix, and print the
/// decisive numbers so we can pin down why the SE walk emits no leaf.
///
/// Returns a human-readable multi-line report.
#[cfg(all(feature = "python", feature = "trace"))]
#[pyfunction]
fn diag_inner_cap(
    gates: &str,
    t_prime: u32,
    k: u32,
    eps: f64,
    a_num: i64,
    a_den: i64,
) -> String {
    use matrix::U2T;
    use rings::MpFloat;
    use rug::Assign;
    use synthesis::angle::{su2_col_mpfr, Angle};
    use synthesis::clifford_t::solution_to_u2t;
    use synthesis::decomposer::BlochDecomposer;
    use synthesis::lattice::omega::brute::apply_u2t_dag_to_uv_mpfr;
    use synthesis::lattice::omega::cholesky_lu::{
        cholesky_f64, cholesky_int, lu_solve_int_inplace, snapshot_gram_to_mpfr,
    };
    use synthesis::lattice::omega::lll::lll_l2;
    use synthesis::lattice::omega::q_metric::{build_q_int, build_q_mpfr_y, uv_to_lattice_y_mpfr};
    use synthesis::lattice::omega::scratch::IntScratch;
    use synthesis::lattice::omega::se::{bilinear_b, reconstruct_x, SE_PREC};

    let mut out = String::new();
    macro_rules! p { ($($t:tt)*) => {{ out.push_str(&format!($($t)*)); out.push('\n'); }} }

    let g_of = |ch: char| -> Option<U2T> {
        match ch {
            'H' => Some(U2T::h()), 'S' => Some(U2T::s()), 'T' => Some(U2T::t()),
            'X' => Some(U2T::x()), 'Y' => Some(U2T::y()), 'Z' => Some(U2T::z()),
            's' => Some(U2T::s().dagger()), 't' => Some(U2T::t().dagger()),
            _ => None,
        }
    };

    // ── x* and MA-canonical t'-prefix U_L (mirror of trace_inner) ──────────
    let mut x = U2T::eye();
    for ch in gates.chars() {
        if let Some(g) = g_of(ch) { x = x * g; }
    }
    let x = x.reduced();
    let canon = BlochDecomposer.decompose(&x);
    let mut u_l = U2T::eye();
    let mut tc = 0u32;
    for ch in canon.chars() {
        if let Some(g) = g_of(ch) {
            u_l = u_l * g;
            if ch == 'T' || ch == 't' {
                tc += 1;
                if tc >= t_prime { break; }
            }
        }
    }
    let u_r = (u_l.dagger() * x).reduced();
    p!("== diag_inner_cap  t'={t_prime} k={k} eps={eps:e}  target=Rz({a_num}/{a_den}·π) ==");
    p!("x*.k = {}   U_L.T = {}   U_R.k = {}", x.k, tc, u_r.k);

    // ── x_R = 8 integer coeffs (same ordering solution_to_u2t/reconstruct_x use) ──
    let x_r: [i64; 8] = [
        u_r.u11.a.as_i64(), u_r.u11.b.as_i64(), u_r.u11.c.as_i64(), u_r.u11.d.as_i64(),
        u_r.u21.a.as_i64(), u_r.u21.b.as_i64(), u_r.u21.c.as_i64(), u_r.u21.d.as_i64(),
    ];
    let roundtrip = solution_to_u2t(&x_r, k) == u_r;
    p!("(1) x_R = {x_r:?}");
    p!("    round-trip solution_to_u2t(x_R,{k}) == U_R : {roundtrip}   (U_R.k=={k}: {})", u_r.k == k);

    // ── (2) norm shell, (3) bilinear form ──────────────────────────────────
    let norm_sq: i128 = x_r.iter().map(|&v| (v as i128) * (v as i128)).sum();
    let target_norm: i128 = 1i128 << k;
    let bil = bilinear_b(&x_r);
    p!("(2) ‖x_R‖² = {norm_sq}   target 2^{k} = {target_norm}   on-shell: {}", norm_sq == target_norm);
    p!("(3) bilinear_b(x_R) = {bil}   (must be 0: {})", bil == 0);

    // ── alignment vector (even branch), exactly as production ───────────────
    let col = su2_col_mpfr(
        Angle::PiRatio(a_num, a_den), Angle::PiRatio(0, 1), Angle::PiRatio(0, 1), 384,
    );
    let mut s = IntScratch::new(eps);
    s.reset_basis();
    let prec_q = s.prec_q;
    let v_inner_mpfr = apply_u2t_dag_to_uv_mpfr(&u_l, &col, prec_q);
    let y_q = uv_to_lattice_y_mpfr(&v_inner_mpfr, k, prec_q);

    // ── (4) alignment acceptance test (MPFR-128, matches production) ────────
    let prec = SE_PREC;
    let y_mpfr: [MpFloat; 8] = std::array::from_fn(|i| MpFloat::with_val(prec, &y_q[i]));
    let two_to_2k = MpFloat::with_val(prec, 1.0) << (2 * k);
    let eps_rf = MpFloat::with_val(prec, eps);
    let one_minus_eps_sq = MpFloat::with_val(prec, 1.0) - eps_rf.clone() * &eps_rf;
    let threshold_xy = MpFloat::with_val(prec, &two_to_2k * &one_minus_eps_sq) / 4u32;
    let mut dot = MpFloat::with_val(prec, 0.0);
    for i in 0..8 {
        let mut t = MpFloat::with_val(prec, x_r[i]);
        t *= &y_mpfr[i];
        dot += &t;
    }
    let dot_sq = MpFloat::with_val(prec, &dot * &dot);
    let dot_ratio = MpFloat::with_val(prec, &dot_sq / &threshold_xy).to_f64();
    p!("(4) dot = Σx_R·y_q = {:.9e}   dot² = {:.9e}", dot.to_f64(), dot_sq.to_f64());
    p!("    threshold_xy = 2^(2k)(1−ε²)/4 = {:.9e}   dot²/thresh = {:.12}  (aligned: {})",
        threshold_xy.to_f64(), dot_ratio, dot_sq >= threshold_xy);

    // ── ALIGNED REPRESENTATIVE ─────────────────────────────────────────────
    // The alignment TEST is on (x·y)² so it accepts ±x, but the cap CENTER is
    // c = +cap_mid·y (the +y hemisphere). If dot<0 the SE-reachable in-cap
    // representative is −x_R (its global-phase −1 partner, an equally valid
    // U2T column, same shell & bilinear). Evaluate the cap on that one.
    let aligned_is_neg = dot < MpFloat::with_val(prec, 0.0);
    let x_a: [i64; 8] = if aligned_is_neg { x_r.map(|v| -v) } else { x_r };
    p!("    dot sign: {}  → SE-reachable representative x_a = {}x_R",
        if aligned_is_neg { "NEGATIVE (x_R anti-aligned with cap center +y)" } else { "positive" },
        if aligned_is_neg { "−" } else { "+" });

    // helper: TRUE Q-dist (x−c)ᵀ Q (x−c) via MPFR q_mpfr/c (built below).
    let true_q_of = |s: &IntScratch, x: &[i64; 8]| -> MpFloat {
        let mut acc = MpFloat::with_val(prec_q, 0.0);
        for a in 0..8 {
            for b in 0..8 {
                let da = MpFloat::with_val(prec_q, x[a]) - &s.c[a];
                let db = MpFloat::with_val(prec_q, x[b]) - &s.c[b];
                acc += MpFloat::with_val(prec_q, da * db) * &s.q_mpfr[a][b];
            }
        }
        acc
    };

    // ── (5) TRUE Q-distance for BOTH signs via MPFR q_mpfr/c ────────────────
    build_q_mpfr_y(&mut s, &y_q, k, eps);
    let true_q_xr = true_q_of(&s, &x_r);
    let true_q = true_q_of(&s, &x_a); // the reachable representative
    p!("(5) TRUE Q-dist (MPFR q_mpfr/c):  +x_R = {:.6e}   −x_R = {:.6e}",
        true_q_xr.to_f64(), true_q_of(&s, &x_r.map(|v| -v)).to_f64());
    p!("    → reachable representative x_a: TRUE Q-dist = {:.9}", true_q.to_f64());

    // ── production LLL + f64 Cholesky + LU z_c ─────────────────────────────
    build_q_int(&mut s);
    let lll_res = lll_l2(&mut s);
    let basis = s.basis;
    let chol_ok = cholesky_f64(&mut s);
    for i in 0..8 {
        for j in 0..8 {
            s.lu_a[i][j].assign(basis[j][i] as f64);
        }
        let ci = s.c[i].clone();
        s.lu_rhs[i].assign(&ci);
    }
    let lu_ok = lu_solve_int_inplace(&mut s);
    let z_c: [MpFloat; 8] = std::array::from_fn(|i| MpFloat::with_val(SE_PREC, &s.lu_x[i]));
    p!("    LLL={lll_res:?} scale_bits={} chol_f64={chol_ok} lu={lu_ok}", s.scale_bits);
    let zc_max_bits = z_c.iter()
        .map(|v| { let a = v.clone().abs().to_f64().max(1.0); a.log2() })
        .fold(0.0f64, f64::max);
    let zc_fits_i64 = z_c.iter().all(|v| v.clone().abs().to_f64() < 9.2e18);
    p!("    cap-center z_c log2|max| = {:.1} bits  (fits i64 {}, fits f64-exact-int {})",
        zc_max_bits, zc_fits_i64, zc_max_bits < 53.0);

    // z_a : exact solve Bᵀ z_a = x_a (fraction-free Bareiss, det ±1), kept in
    // rug integers (z can exceed i64 when x is far from the reduced frame).
    use rug::Integer as RInt;
    let z_big: Vec<RInt> = {
        let aij = |i: usize, j: usize| RInt::from(basis[j][i]);
        let mut m: Vec<Vec<RInt>> = (0..8)
            .map(|i| {
                let mut row: Vec<RInt> = (0..8).map(|j| aij(i, j)).collect();
                row.push(RInt::from(x_a[i]));
                row
            })
            .collect();
        let mut sign = 1i32;
        let mut prev = RInt::from(1);
        for col in 0..8 {
            if m[col][col] == 0 {
                if let Some(pr) = (col + 1..8).find(|&r1| m[r1][col] != 0) {
                    m.swap(col, pr);
                    sign = -sign;
                }
            }
            for r2 in (col + 1)..8 {
                for cc in (col + 1)..9 {
                    let t1 = RInt::from(&m[col][col] * &m[r2][cc]);
                    let t2 = RInt::from(&m[r2][col] * &m[col][cc]);
                    let num = t1 - t2;
                    let (q, rem) = num.div_rem(prev.clone());
                    debug_assert!(rem == 0);
                    m[r2][cc] = q;
                }
                m[r2][col] = RInt::from(0);
            }
            prev = m[col][col].clone();
        }
        let det = RInt::from(&m[7][7] * sign);
        p!("    det(Bᵀ) = {det}");
        let mut z_big: Vec<RInt> = vec![RInt::from(0); 8];
        for r2 in (0..8).rev() {
            let mut v = m[r2][8].clone();
            for cc in (r2 + 1)..8 {
                v -= RInt::from(&m[r2][cc] * &z_big[cc]);
            }
            let (q, _rem) = v.div_rem(m[r2][r2].clone());
            z_big[r2] = q;
        }
        z_big
    };
    // max |z_a| (as f64 magnitude, for reporting) and reconstruct check via rug.
    let zmax = z_big.iter().map(|z| z.significant_bits()).max().unwrap_or(0);
    p!("    z_a bit-width max = {zmax}");

    // diff = z_a − z_c, lifted exactly to MPFR from rug integers.
    let diff: [MpFloat; 8] = std::array::from_fn(|j| {
        let mut d = MpFloat::with_val(SE_PREC, &z_big[j]);
        d -= &z_c[j];
        d
    });

    // ── (6) f64-Cholesky-computed ‖R(z_a − z_c)‖² (R = l_f64ᵀ, lifted to 128) ──
    let mut f64_q = MpFloat::with_val(SE_PREC, 0.0);
    for d in 0..8 {
        let mut lvl = MpFloat::with_val(SE_PREC, 0.0);
        for j in d..8 {
            let r = MpFloat::with_val(SE_PREC, s.l_f64[j][d]); // R[d][j] = l_f64[j][d]
            lvl += MpFloat::with_val(SE_PREC, &r * &diff[j]);
        }
        f64_q += MpFloat::with_val(SE_PREC, &lvl * &lvl);
    }
    p!("(6) f64-Cholesky ‖R(z_a−z_c)‖² = {:.9}   (production SE bound test uses this)", f64_q.to_f64());

    // MPFR-oracle cross-check: cholesky_int on the snapshotted post-LLL Gram.
    snapshot_gram_to_mpfr(&mut s);
    let oracle_ok = cholesky_int(&mut s);
    let mut oracle_q = MpFloat::with_val(prec_q, 0.0);
    for d in 0..8 {
        let mut lvl = MpFloat::with_val(prec_q, 0.0);
        for j in d..8 {
            let r = MpFloat::with_val(prec_q, &s.l[j][d]);
            let mut dd = MpFloat::with_val(prec_q, &z_big[j]);
            dd -= &z_c[j];
            lvl += MpFloat::with_val(prec_q, &r * &dd);
        }
        oracle_q += MpFloat::with_val(prec_q, &lvl * &lvl);
    }
    p!("    MPFR-oracle ‖R(z_a−z_c)‖² (cholesky_int, ok={oracle_ok}) = {:.9}  (≈ TRUE cross-check)", oracle_q.to_f64());
    let _ = reconstruct_x; // (kept import; rug path used for z)

    // ── (7) se_bound + worst-diagonal f64-vs-MPFR R discrepancy ─────────────
    let bound = std::env::var("CYCLOSYNTH_SE_BOUND_8D").ok()
        .and_then(|v| v.parse::<f64>().ok()).unwrap_or(1.51);
    p!("(7) se_bound() = {bound}");
    let mut worst_rel = 0.0f64;
    let mut worst_at = (0usize, 0usize, 0.0f64, 0.0f64);
    for i in 0..8 {
        for j in 0..=i {
            let f = s.l_f64[i][j];
            let o = s.l[i][j].to_f64();
            let rel = (f - o).abs() / (1e-300 + o.abs());
            if rel > worst_rel { worst_rel = rel; worst_at = (i, j, f, o); }
        }
    }
    p!("    worst f64-vs-MPFR Cholesky L rel-err = {:.3e} at L[{}][{}] (f64={:.6e} mpfr={:.6e})",
        worst_rel, worst_at.0, worst_at.1, worst_at.2, worst_at.3);

    // ── verdict ────────────────────────────────────────────────────────────
    let on_shell = norm_sq == target_norm && bil == 0 && roundtrip;
    let tq = true_q.to_f64();
    let fq = f64_q.to_f64();
    let verdict = if !on_shell {
        "CONVENTION bug — x_R is not on the enumerated shell / does not round-trip"
    } else if tq > bound {
        "T2 — cap/bound miscalibrated: TRUE Q-dist exceeds se_bound at the inner shell"
    } else if fq > bound {
        "T1 — f64 Cholesky distorts the box: TRUE ≤ bound but f64-computed > bound"
    } else {
        "NEITHER — x_R is on-shell AND in-cap by both metrics (bug is elsewhere: alignment/threshold or upstream)"
    };
    p!("VERDICT: {verdict}");
    p!("  on_shell={on_shell}  TRUE_Q={tq:.6}  f64_Q={fq:.6}  bound={bound}  aligned={}",
        dot_sq >= threshold_xy);

    out
}

/// Python extension module. Built when the `python` Cargo feature is enabled
/// (via `maturin develop` / `maturin build`); see `pyproject.toml`.
#[cfg(feature = "python")]
#[pymodule]
fn cyclosynth(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // The public surface is exactly the synthesizer and its result. The ring /
    // U2 / decomposer types are internal plumbing and are intentionally not
    // exported (they have no standalone Python use).
    m.add_class::<synthesis::synthesizer::PySynthesizer>()?;
    m.add_class::<synthesis::synthesizer::PySynthResult>()?;
    // Gridsynth-oracle probes, `trace`-only to keep the default surface clean.
    #[cfg(feature = "trace")]
    {
        m.add_function(wrap_pyfunction!(decompose_gates_t, m)?)?;
        m.add_function(wrap_pyfunction!(trace_inner, m)?)?;
        m.add_function(wrap_pyfunction!(diag_inner_cap, m)?)?;
    }
    Ok(())
}
