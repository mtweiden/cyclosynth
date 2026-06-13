//! Research probes (all #[ignore]): forensic and telemetry harnesses
//! kept runnable but out of the unit-test file. Run individually, e.g.
//! `cargo test --release --lib l_coset_census -- --ignored --nocapture`.

#![allow(unused_imports)]
use super::*; // the tests module: shared helpers (u3, rz, …)
use super::super::*; // clifford_t internals
use crate::rings::Float;
use crate::synthesis::distance::{diamond_distance_float, Mat2};
use num_complex::Complex;
use std::f64::consts::PI;

    /// Diagnostic probe (ignored): the t_identity target-2 @1e-5 FOUND→none
    /// flip under coset dedup, reproduced at level t=47 (t'=6). Finds every
    /// PLAIN prefix that yields an ε-valid solution, maps each winner to its
    /// kept coset representative, reruns the rep's two branches, and checks
    /// whether the image solution c·U_R appears — pinpointing where the
    /// Q-isometric-bijection argument breaks in practice.
    /// Run: `cargo test --release --lib probe_coset_flip_t47 -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn probe_coset_flip_t47() {
        // SplitMix64(0xC0FFEE) — t_identity_1e5's generator; target idx 2.
        struct Xs(u64);
        impl Xs {
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
            fn range(&mut self, lo: f64, hi: f64) -> f64 {
                lo + (hi - lo) * self.unit()
            }
        }
        // Widen the SE walk bound for the WHOLE probe (LazyLock-once; must
        // precede the first find_aligned_lattice_points call). If the rep's missing image shows
        // up at bound 4.0, its Q-norm in that frame is in (1.51, 4.0] and
        // the Q-band model is frame-fragile; if it stays missing, the f64
        // partial-eucl norm prune (the known 1.5e-8-cliff mechanism) is
        // killing the branch.
        std::env::set_var("CYCLOSYNTH_SE_BOUND_8D", "4.0");
        let mut rng = Xs(0xC0FFEE);
        let mut tri = (0.0, 0.0, 0.0);
        for _ in 0..3 {
            tri = (
                rng.range(0.2, PI - 0.2),
                rng.range(0.1, 2.0 * PI - 0.1),
                rng.range(0.1, 2.0 * PI - 0.1),
            );
        }
        let (th, ph, la) = tri;
        // u3 with the t_identity convention (global-phase normalized).
        let (c, s) = ((th / 2.0).cos(), (th / 2.0).sin());
        let eilam = Complex::from_polar(1.0, la);
        let eiphi = Complex::from_polar(1.0, ph);
        let g = Complex::from_polar(1.0, -(ph + la) / 2.0);
        let target: Mat2 = [
            [Complex::new(c, 0.0) * g, -eilam * s * g],
            [eiphi * s * g, eiphi * eilam * Complex::new(c, 0.0) * g],
        ];

        coset_flip_probe(target, 1e-5, 47);
    }

    /// Same forensic probe at the bench-suite 1e-8 flip: time_synthesis
    /// target_00 (xorshift64, seed 0xC0FFEEBAADD0E|1), lde 78 (t'=12),
    /// which still drifts to 80 under coset dedup after the
    /// euclidean_cholesky trust guards.
    /// Run: `cargo test --release --lib probe_coset_flip_t78 -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn probe_coset_flip_t78() {
        std::env::set_var("CYCLOSYNTH_SE_BOUND_8D", "4.0");
        fn xorshift64(s: &mut u64) -> u64 {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
            *s
        }
        fn rand_angle(s: &mut u64) -> f64 {
            let b = xorshift64(s) >> 11;
            (b as f64) / ((1u64 << 53) as f64) * 2.0 * PI
        }
        let mut state: u64 = 0xC0FFEE_BAADD0E_u64 | 1;
        let a = rand_angle(&mut state);
        let b = rand_angle(&mut state);
        let c = rand_angle(&mut state);
        let target = u3(a, b, c);
        coset_flip_probe(target, 1e-8, 78);
    }

    #[allow(clippy::needless_range_loop)]
    fn coset_flip_probe(target: Mat2, eps: Float, t: u32) {
        use std::sync::atomic::AtomicBool;
        let t_prime = optimal_t_prime(t, eps);
        let t_inner = t - t_prime;
        let lde_inner: u32 = if t_inner % 2 == 1 { (t_inner - 1) / 2 + 1 } else { t_inner / 2 + 1 };
        eprintln!("t={t} t'={t_prime} t_inner={t_inner} lde_inner={lde_inner}");

        let plain = build_l_inner_with(t_prime, false);
        let coset = build_l_inner_with(t_prime, true);
        eprintln!("|plain|={} |coset|={}", plain.len(), coset.len());

        let run_prefix = |u_l: &U2T, max_sols: usize| -> Vec<(bool, [i64; 8], f64)> {
            let mut out = Vec::new();
            let m_inner = prefix_dag_times_target(u_l, &target);
            let Some(v_inner) = try_unitary_to_uv(&m_inner) else { return out };
            let mut scratch = crate::synthesis::lattice::scratch::IntScratch::new(eps);
            for odd in [false, true] {
                let v_b = if odd { apply_t_dag_to_uv(v_inner) } else { v_inner };
                let hit = AtomicBool::new(false);
                for sol in lll_aligned_search(
                    &mut scratch, v_b, lde_inner, eps, max_sols, u64::MAX,
                    50_000_000, &hit, None,
                ) {
                    let u2t = if odd {
                        *u_l * solution_to_u2t(&sol, lde_inner) * U2T::t()
                    } else {
                        *u_l * solution_to_u2t(&sol, lde_inner)
                    };
                    let dist = diamond_distance_u2t_float(&u2t, &target);
                    out.push((odd, sol, dist));
                }
            }
            out
        };

        // 1) all plain winners.
        let winners: Vec<(usize, bool, [i64; 8], f64)> = plain
            .par_iter()
            .enumerate()
            .flat_map_iter(|(i, u_l)| {
                run_prefix(u_l, 16)
                    .into_iter()
                    .filter(|&(_, _, d)| d < eps)
                    .map(move |(odd, sol, d)| (i, odd, sol, d))
                    .collect::<Vec<_>>()
            })
            .collect();
        eprintln!("plain winners: {}", winners.len());
        for &(i, odd, sol, d) in winners.iter().take(8) {
            eprintln!("  plain[{i}] odd={odd} sol={sol:?} dist={d:.3e}");
        }

        // 2) orbit-key → rep map for the coset set.
        let mut rep_of: HashMap<[i64; 8], usize> = HashMap::new();
        for (ri, r) in coset.iter().enumerate() {
            for &ci in CLIFFORD_LDE0_IDX.iter() {
                rep_of.entry(canonical_key(&(*r * CLIFFORD_TABLE_T[ci].1))).or_insert(ri);
            }
        }

        for &(i, _odd, sol, d) in winners.iter().take(4) {
            let w = &plain[i];
            let Some(&ri) = rep_of.get(&canonical_key(w)) else {
                eprintln!("plain[{i}]: NO REP FOUND (coverage hole!)");
                continue;
            };
            let r = &coset[ri];
            // which c maps rep -> winner? r·c ≡ w (up to phase).
            let c_idx = CLIFFORD_LDE0_IDX.iter().copied().find(|&ci| {
                canonical_key(&(*r * CLIFFORD_TABLE_T[ci].1)) == canonical_key(w)
            });
            eprintln!(
                "plain[{i}] (dist {d:.3e}) -> rep coset[{ri}] via c={:?} (rep==winner: {})",
                c_idx.map(|ci| CLIFFORD_TABLE_T[ci].0),
                ri_eq(r, w),
            );
            // 3) rerun the rep with a deep candidate budget.
            let rsols = run_prefix(r, 4096);
            let n_close = rsols.iter().filter(|&&(_, _, d)| d < eps).count();
            eprintln!(
                "  rep sols={} eps-close={} dists(first 6)={:?}",
                rsols.len(),
                n_close,
                rsols.iter().take(6).map(|&(o, _, d)| (o, d)).collect::<Vec<_>>()
            );
            // image solution: x_img = c · x_w (matrix-vector in the ring).
            if let Some(ci) = c_idx {
                let c_mat = &CLIFFORD_TABLE_T[ci].1;
                // w ≈ r·c  ⇒  w·U(sol) = r·(c·U(sol)); image x = first col
                // of c·U(sol). Winner was ODD branch: total = r·img·T.
                let img_u2t = *c_mat * solution_to_u2t(&sol, lde_inner);
                eprintln!(
                    "  image k={} (lde_inner={lde_inner}); in rep sols: {}",
                    img_u2t.k,
                    rsols.iter().any(|(_, s, _)| solution_to_u2t(s, lde_inner).diamond_distance(&img_u2t) < 1e-9),
                );
                let img_total = *r * img_u2t * U2T::t();
                eprintln!(
                    "  dist(r·img·T, target) = {:.3e}",
                    diamond_distance_u2t_float(&img_total, &target)
                );
                // Geometry of x_img in the rep's ODD frame.
                let m_inner_r = prefix_dag_times_target(r, &target);
                let v_inner_r = try_unitary_to_uv(&m_inner_r).expect("rep try_unitary_to_uv");
                let v_odd_r = apply_t_dag_to_uv(v_inner_r);
                let y = uv_to_lattice_y(v_odd_r, lde_inner);
                // x_img integer coords: (u1, u2) coefficients of img_u2t.
                let gi = |z: &crate::rings::ZOmega| -> [f64; 4] {
                    use crate::rings::types::int_to_f64;
                    [
                        int_to_f64(z.a),
                        int_to_f64(z.b),
                        int_to_f64(z.c),
                        int_to_f64(z.d),
                    ]
                };
                let (i1, i2) = (gi(&img_u2t.u11), gi(&img_u2t.u21));
                let x_img: [f64; 8] =
                    [i1[0], i1[1], i1[2], i1[3], i2[0], i2[1], i2[2], i2[3]];
                let dot: f64 = (0..8).map(|j| y[j] * x_img[j]).sum();
                let norm_sq: f64 = x_img.iter().map(|v| v * v).sum();
                let thresh = (1.0 - eps * eps) * 2f64.powi(2 * lde_inner as i32) / 4.0;
                eprintln!(
                    "  x_img: |x|^2/2^k = {:.6}  dot^2/thresh - 1 = {:+.6e}",
                    norm_sq / 2f64.powi(lde_inner as i32),
                    dot * dot / thresh - 1.0,
                );
                // Q-norm of x_img from the rep's odd frame, evaluated in
                // MPFR at the scratch precision (an f64 eval of this form
                // is garbage: Q eigenvalues reach 1/Δ_y² ~ 1e14 at 1e-5 and
                // the form only stays O(1) through cancellation).
                use crate::synthesis::lattice::{q_metric::build_q_mpfr, scratch::IntScratch};
                use rug::Float as RFloat;
                let mut qs = IntScratch::new(eps);
                build_q_mpfr(&mut qs, &y, lde_inner, eps);
                let prec = qs.q_mpfr[0][0].prec();
                let mut qn = RFloat::with_val(prec, 0.0);
                for a in 0..8 {
                    for b in 0..8 {
                        let da = RFloat::with_val(prec, x_img[a]) - &qs.c[a];
                        let db = RFloat::with_val(prec, x_img[b]) - &qs.c[b];
                        qn += da * db * &qs.q_mpfr[a][b];
                    }
                }
                eprintln!(
                    "  x_img Q-norm in rep odd frame = {:.6} (walk bound 1.51; probe bound 4.0)",
                    qn.to_f64()
                );
                // Call integer::find_aligned_lattice_points_outcome directly to expose should_escalate
                // (mod.rs's wrapper silently drops it).
                {
                    use std::sync::atomic::AtomicBool;
                    let mut s2 = IntScratch::new(eps);
                    s2.reset_basis();
                    let hit = AtomicBool::new(false);
                    let out = crate::synthesis::lattice::integer::find_aligned_lattice_points_outcome(
                        &mut s2, &y, lde_inner, eps, usize::MAX, u64::MAX,
                        50_000_000, &hit, None,
                    );
                    eprintln!(
                        "  rep odd frame direct find_aligned_lattice_points: sols={} should_escalate={} budget_hit={}",
                        out.solutions.len(),
                        out.should_escalate,
                        hit.load(std::sync::atomic::Ordering::Relaxed),
                    );
                }
                // SE-walk replay: reproduce find_aligned_lattice_points's setup, locate x_img's
                // z-path, and print the walker's own per-depth partials to
                // find which level excludes it.
                {
                    use crate::synthesis::lattice::{
                        cholesky_lu::{cholesky_f64_8, lu_solve_int_inplace},
                        lll::lll_l2_8,
                        q_metric::build_q_int,
                        cholesky_lu::euclidean_cholesky,
                        se::{bilinear_b, reconstruct_x},
                    };
                    use rug::Assign;
                    let mut s3 = IntScratch::new(eps);
                    s3.reset_basis();
                    build_q_mpfr(&mut s3, &y, lde_inner, eps);
                    build_q_int(&mut s3);
                    let lll_res = lll_l2_8(&mut s3);
                    eprintln!("  replay: lll={lll_res:?} scale_bits={}", s3.scale_bits);
                    let basis = s3.basis;
                    let chol_ok = cholesky_f64_8(&mut s3);
                    for i in 0..8 {
                        for j in 0..8 {
                            let v = basis[j][i] as f64;
                            s3.lu_a[i][j].assign(v);
                        }
                        let ci = s3.c[i].clone();
                        s3.lu_rhs[i].assign(&ci);
                    }
                    let lu_ok = lu_solve_int_inplace(&mut s3);
                    eprintln!("  replay: chol_ok={chol_ok} lu_ok={lu_ok}");
                    let z_c: [f64; 8] = std::array::from_fn(|i| s3.lu_x[i].to_f64());
                    // Solve B^T z = x_img EXACTLY (det ±1) with rug::Integer
                    // adjugate (an f64 solve fails here — basis dynamic range
                    // is huge; scale_bits=132). z = adj(A)·x / det(A).
                    use rug::Integer as RInt;
                    let aij = |i: usize, j: usize| RInt::from(basis[j][i]);
                    // det via cofactor expansion is fine at 8x8 with exact
                    // ints? Too slow (8!). Use fraction-free Bareiss.
                    let mut m: Vec<Vec<RInt>> = (0..8)
                        .map(|i| {
                            let mut row: Vec<RInt> =
                                (0..8).map(|j| aij(i, j)).collect();
                            row.push(RInt::from(x_img[i] as i64));
                            row
                        })
                        .collect();
                    let mut sign = 1i32;
                    let mut prev = RInt::from(1);
                    for col in 0..8 {
                        if m[col][col] == 0 {
                            let p = (col + 1..8).find(|&r1| m[r1][col] != 0).unwrap();
                            m.swap(col, p);
                            sign = -sign;
                        }
                        for r2 in (col + 1)..8 {
                            for cc in (col + 1)..9 {
                                let t1 = RInt::from(&m[col][col] * &m[r2][cc]);
                                let t2 = RInt::from(&m[r2][col] * &m[col][cc]);
                                let num = t1 - t2;
                                let (q, rem) = num.div_rem(prev.clone());
                                assert!(rem == 0, "Bareiss exact division failed");
                                m[r2][cc] = q;
                            }
                            m[r2][col] = RInt::from(0);
                        }
                        prev = m[col][col].clone();
                    }
                    // After Bareiss, m[7][7] = det·sign' and back-substitution
                    // on the triangular system is exact.
                    let det = RInt::from(&m[7][7] * sign);
                    eprintln!("  replay: det(B^T) = {det}");
                    let mut z_big: Vec<RInt> = vec![RInt::from(0); 8];
                    for r2 in (0..8).rev() {
                        let mut v = m[r2][8].clone();
                        for cc in (r2 + 1)..8 {
                            v -= RInt::from(&m[r2][cc] * &z_big[cc]);
                        }
                        let (q, rem) = v.div_rem(m[r2][r2].clone());
                        assert!(rem == 0, "back-substitution not integral at {r2}");
                        z_big[r2] = q;
                    }
                    let mut z_img = [0i64; 8];
                    for (zi, zb) in z_img.iter_mut().zip(z_big.iter()) {
                        *zi = zb.to_i64().expect("z fits i64");
                    }
                    let x_chk = reconstruct_x(&basis, &z_img);
                    let x_int: [i64; 8] = std::array::from_fn(|i| x_img[i] as i64);
                    eprintln!(
                        "  replay: z_img={z_img:?} reconstruct==x_img: {}  bilinear_b={}",
                        x_chk == x_int,
                        bilinear_b(&x_int)
                    );
                    // Walker partials: R = l_f64^T (Q-metric), per depth d:
                    // partial_d = sum_{i>=d} (sum_{j>=i} R[i][j] (z[j]-z_c[j]))^2.
                    let mut rq = [[0.0f64; 8]; 8];
                    for i in 0..8 {
                        for j in 0..8 {
                            rq[i][j] = s3.l_f64[j][i];
                        }
                    }
                    let mut pq = [0.0f64; 9]; // pq[d] = partial entering depth d-1
                    for d in (0..8).rev() {
                        let mut lvl = 0.0;
                        for j in d..8 {
                            lvl += rq[d][j] * (z_img[j] as f64 - z_c[j]);
                        }
                        pq[d] = pq[d + 1] + lvl * lvl;
                    }
                    eprintln!("  replay: Q-partials by depth (7..0): {:?}",
                        (0..8).rev().map(|d| (d, pq[d])).collect::<Vec<_>>());
                    if let Some(re) = euclidean_cholesky(&basis) {
                        let mut pe = [0.0f64; 9];
                        for d in (0..8).rev() {
                            let mut lvl = 0.0;
                            for j in d..8 {
                                lvl += re[d][j] * z_img[j] as f64;
                            }
                            pe[d] = pe[d + 1] + lvl * lvl;
                        }
                        let tgt = 2f64.powi(lde_inner as i32);
                        eprintln!(
                            "  replay: eucl partials/2^k by depth (7..0): {:?} (cut if > 1 + 2^-k)",
                            (0..8).rev().map(|d| (d, pe[d] / tgt)).collect::<Vec<_>>()
                        );
                    } else {
                        eprintln!("  replay: euclidean_cholesky FAILED (prune disabled)");
                    }
                }
            }
        }

        fn ri_eq(a: &U2T, b: &U2T) -> bool {
            canonical_key(a) == canonical_key(b)
        }
    }

    /// M1 census probe (stage 0 of docs/plan_8d_prefix_rework.md):
    /// |L_{t'}| with vs without right-coset dedup, t' = 1..13. Lever B1
    /// predicts 4.5-8×; kill if < 2×.
    /// Run: `cargo test --release --lib l_coset_census -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn l_coset_census() {
        eprintln!("\nM1 census: build_ma_prefix_set_reference plain phase-dedup vs right-coset dedup");
        eprintln!("  t'   |L| plain   |L| coset   ratio");
        for tp in 1..=13u32 {
            let t0 = std::time::Instant::now();
            let plain = build_l_inner_with(tp, false).len();
            let t_plain = t0.elapsed().as_secs_f64() * 1000.0;
            let t0 = std::time::Instant::now();
            let coset = build_l_inner_with(tp, true).len();
            let t_coset = t0.elapsed().as_secs_f64() * 1000.0;
            eprintln!(
                "  {tp:>2}   {plain:>9}   {coset:>9}   {:>5.2}x   (build {t_plain:.0} / {t_coset:.0} ms)",
                plain as f64 / coset as f64
            );
        }
    }

    /// Telemetry (ignored): geometric Q-norm² distribution of ε-close 8D
    /// solutions, the Z[ω] mirror of the 16D `q_telemetry_sweep` that
    /// found the ζ₁₆ band [0.875, 1.25] and dropped that bound 8 → 1.5.
    /// The 8D SE bound is the empirical 1.51 (lattice/integer.rs); this
    /// measures where ε-close solutions actually sit, from the TRUE cap
    /// center (the 8D walk already uses a fractional center, so measured
    /// ≈ geometric — no rounding-inflation step needed). If the max pins
    /// well below 1.51, a tightened bound buys (1.51/max)⁴ fewer nodes.
    ///
    /// 2026-06-11: collects ALL in-region solutions per level
    /// (`max_solutions = usize::MAX`) — the earlier first-hit numbers
    /// ([0.75, 0.94]) were maximally center-biased because find_aligned_lattice_points stopped
    /// at the first hit of a distance-ordered walk. Walks are bounded by
    /// the new node budget (T8_NODES, default 50M per branch walk), which
    /// is what makes ε=1e-3 runnable at all (empty/slow branches used to
    /// walk unbudgeted for tens of minutes). Optionally widen the walk
    /// region via CYCLOSYNTH_SE_BOUND_8D (e.g. 2.5) to check for
    /// solutions ABOVE the production bound.
    /// Run: `cargo test --release --lib q_telemetry_sweep_8d -- --ignored --nocapture`
    /// Env: T8_EPS (default sweeps 3e-2 and 1e-3), T8_BUDGET (default 20M),
    /// T8_NODES (default 50M).
    #[test]
    #[ignore]
    fn q_telemetry_sweep_8d() {
        use crate::synthesis::lattice::{integer::find_aligned_lattice_points_outcome as find_aligned_lattice_points, q_metric::build_q_mpfr};
        use crate::synthesis::lattice::scratch::IntScratch;
        use std::sync::atomic::AtomicBool;

        let budget: u64 = std::env::var("T8_BUDGET").ok()
            .and_then(|s| s.parse().ok()).unwrap_or(20_000_000);
        let nodes: u64 = std::env::var("T8_NODES").ok()
            .and_then(|s| s.parse().ok()).unwrap_or(50_000_000);
        let mut global_max_close = 0.0f64;
        let mut global_min_close = f64::INFINITY;
        let mut total_close = 0usize;

        // t (lde) scan ranges per ε. CAUTION (learned the 46-minute way,
        // twice): `max_leaf_checks` caps CANDIDATE COMPLETIONS, not raw
        // nodes — on a no-solution level almost nothing reaches
        // candidacy, so the walk runs effectively unbudgeted and a
        // single below-first-hit level burns tens of minutes on one
        // core. Per-θ first-hit levels can't be reliably guessed, so
        // scan DOWNWARD from t_hi: every level at-or-above first-hit is
        // solution-dense and returns fast, and the two-level early-stop
        // fires before the scan can descend into empty territory.
        // Optional deep entry (T8_DEEP=1): ε=1e-5, scanned down from
        // t_hi=46 (typical first-hit lde ≈ 40-44 across these θ).
        let deep = std::env::var("T8_DEEP").as_deref() == Ok("1");
        let mut grid: Vec<(f64, u32, u32)> =
            vec![(3e-2f64, 8u32, 14u32), (1e-3, 27, 34)];
        if deep {
            grid.push((1e-5, 38, 46));
        }
        for &theta in &[0.3f64, 0.55, 0.8, 1.05, 1.3] {
            let target = rz(theta);
            let raw_uv = unitary_to_uv(&target);
            let v = normalize4(raw_uv).unwrap_or([1.0, 0.0, 0.0, 0.0]);
            for &(eps, t_lo, t_hi) in &grid {
                let mut levels_with_sols = 0;
                'levels: for t in (t_lo..=t_hi).rev() {
                    if levels_with_sols >= 2 {
                        break;
                    }
                    // Probe the PRODUCTION geometry for this level. With
                    // t' = optimal_t_prime == 0 that's the three direct
                    // branches at k = t; with t' > 0 it's the MA-prefix
                    // inner frames at lde_inner — the frames whose walks the
                    // 1.51 bound actually governs. (Direct full-lde
                    // probing at ε ≤ 1e-3 is hopeless: the t=27..34 region
                    // is so large that a 50M-node budgeted walk finds
                    // nothing — the pre-fix 46-minute deadlock geometry.)
                    // Q/c are built in each frame's own (y, k); find_aligned_lattice_points
                    // sols have already passed the alignment-cap leaf
                    // check, which is exactly the in-cap criterion the
                    // bound governs.
                    let t_level = std::time::Instant::now();
                    let t_prime = optimal_t_prime(t, eps);
                    let mut frames: Vec<([Float; 4], u32)> = Vec::new();
                    if t_prime == 0 || t_prime > t {
                        for v_s in [v, apply_t_dag_to_uv(v), apply_t_to_uv(v)] {
                            frames.push((v_s, t));
                        }
                    } else {
                        let t_inner = t - t_prime;
                        let lde_inner: u32 = if t_inner % 2 == 1 {
                            (t_inner - 1) / 2 + 1
                        } else {
                            t_inner / 2 + 1
                        };
                        let target_parity = det_zeta_parity(&target);
                        for u_l in build_ma_prefix_set_reference(t_prime).iter() {
                            if frames.len() >= 64 {
                                break;
                            }
                            if let Some(tp) = target_parity {
                                if det_zeta_parity(&u_l.to_float()) != Some(tp) {
                                    continue;
                                }
                            }
                            let m_inner = prefix_dag_times_target(u_l, &target);
                            let Some(v_inner) = try_unitary_to_uv(&m_inner) else { continue };
                            frames.push((v_inner, lde_inner));
                            if t_inner > 0 {
                                frames.push((apply_t_dag_to_uv(v_inner), lde_inner));
                            }
                        }
                    }

                    let mut min_close = f64::INFINITY;
                    let mut max_close = 0.0f64;
                    let mut n_close = 0usize;
                    let mut sol_frames = 0usize;
                    let mut any_trunc = false;
                    let mut k_probed = t;
                    let mut breaker = false;
                    for &(v_s, k_f) in &frames {
                        // Sample at most 6 solution-bearing frames per level.
                        if sol_frames >= 6 {
                            break;
                        }
                        k_probed = k_f;
                        let y = uv_to_lattice_y(v_s, k_f);
                        let mut s = IntScratch::new(eps);
                        let hit = AtomicBool::new(false);
                        let out = find_aligned_lattice_points(
                            &mut s, &y, k_f, eps, usize::MAX, budget, nodes, &hit, None,
                        );
                        // Circuit breaker: this level is expensive territory.
                        if t_level.elapsed().as_secs() > 60 {
                            breaker = true;
                            break;
                        }
                        if out.solutions.is_empty() {
                            continue;
                        }
                        sol_frames += 1;
                        any_trunc |= hit.load(std::sync::atomic::Ordering::Relaxed);
                        // Fresh scratch for Q + cap center in THIS frame:
                        // find_aligned_lattice_points's LLL may have mutated downstream state;
                        // build_q alone is cheap and sets q_mpfr and c.
                        let mut qs = IntScratch::new(eps);
                        build_q_mpfr(&mut qs, &y, k_f, eps);
                        let q: [[f64; 8]; 8] = std::array::from_fn(|i| {
                            std::array::from_fn(|j| qs.q_mpfr[i][j].to_f64())
                        });
                        let c: [f64; 8] = std::array::from_fn(|i| qs.c[i].to_f64());
                        for sol in &out.solutions {
                            let dvec: [f64; 8] =
                                std::array::from_fn(|i| sol[i] as f64 - c[i]);
                            let mut qn = 0.0;
                            for i in 0..8 {
                                for j in 0..8 {
                                    qn += dvec[i] * q[i][j] * dvec[j];
                                }
                            }
                            max_close = max_close.max(qn);
                            min_close = min_close.min(qn);
                            n_close += 1;
                        }
                    }
                    if n_close > 0 {
                        levels_with_sols += 1;
                        eprintln!(
                            "θ={theta:<4} ε={eps:.0e} t={t:<2} k={k_probed:<2} frames={sol_frames} close={n_close:<5} Q∈[{min_close:.4}, {max_close:.4}]{}",
                            if any_trunc { "  (TRUNCATED walk)" } else { "" }
                        );
                        global_max_close = global_max_close.max(max_close);
                        global_min_close = global_min_close.min(min_close);
                        total_close += n_close;
                    } else if breaker {
                        break 'levels;
                    }
                }
            }
        }
        eprintln!(
            "GLOBAL 8D: eps-close sols={total_close}  Q∈[{global_min_close:.4}, {global_max_close:.4}]  (walk bound: 1.51)"
        );
        assert!(total_close > 0, "telemetry collected no eps-close solutions");
    }

    /// Telemetry (ignored): W0-style yardstick for ONE 8D level walk —
    /// wall, CPU utilization (process cpu-time / wall), solutions. The
    /// 16D version of this measurement (util 1.08× on 14 threads)
    /// motivated the W1 flat-frontier parallelization (~10×). Whether
    /// the port pays here depends on the T-baseline's wall at deep ε.
    /// Run: `cargo test --release --lib w1_telemetry_8d -- --ignored --nocapture`
    /// Env: T8_THETA (0.7), T8_EPS (1e-3), T8_LDE (30), T8_BUDGET (500M),
    /// T8_NODES (node budget, default 200M ≈ a few minutes single-core —
    /// a full-lde frame at ε=1e-3 is the 46-minute-runaway geometry, so
    /// an unbounded default is a footgun; raise it explicitly for a pure
    /// yardstick). 2026-06-11: with the node budget landed, an EMPTY
    /// level is also safe to measure — that's the budgeted-empty
    /// yardstick configuration.
    #[test]
    #[ignore]
    fn w1_telemetry_8d() {
        use std::sync::atomic::AtomicBool;

        fn envf(name: &str, default: f64) -> f64 {
            std::env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
        }
        let theta = envf("T8_THETA", 0.7);
        let eps = envf("T8_EPS", 1e-3);
        let t = envf("T8_LDE", 30.0) as u32;
        let budget = envf("T8_BUDGET", 500_000_000.0) as u64;
        let nodes: u64 = std::env::var("T8_NODES").ok()
            .and_then(|s| s.parse().ok()).unwrap_or(200_000_000);

        // CLOCK_PROCESS_CPUTIME_ID = 12 on macOS (same constant the 16D
        // w1_walk_bench uses).
        #[repr(C)]
        struct Timespec { tv_sec: i64, tv_nsec: i64 }
        extern "C" {
            fn clock_gettime(clk_id: i32, tp: *mut Timespec) -> i32;
        }
        fn cpu_time_s() -> f64 {
            let mut ts = Timespec { tv_sec: 0, tv_nsec: 0 };
            let rc = unsafe { clock_gettime(12, &mut ts) };
            if rc != 0 { return f64::NAN; }
            ts.tv_sec as f64 + ts.tv_nsec as f64 * 1e-9
        }

        let target = rz(theta);
        let raw_uv = unitary_to_uv(&target);
        let v = normalize4(raw_uv).unwrap_or([1.0, 0.0, 0.0, 0.0]);
        let mut scratch = crate::synthesis::lattice::scratch::IntScratch::new(eps);
        let hit = AtomicBool::new(false);

        crate::synthesis::diag::N_SE_NODES
            .store(0, std::sync::atomic::Ordering::Relaxed);
        let cpu0 = cpu_time_s();
        let t0 = std::time::Instant::now();
        let sols = lll_aligned_search(
            &mut scratch, v, t, eps, usize::MAX, budget, nodes, &hit, None,
        );
        let wall = t0.elapsed().as_secs_f64();
        let cpu = cpu_time_s() - cpu0;
        let n_nodes = crate::synthesis::diag::N_SE_NODES
            .load(std::sync::atomic::Ordering::Relaxed);

        let n_close = sols.iter().filter(|sol| {
            diamond_distance_float(&solution_to_u2t(sol, t).to_float(), &target) <= eps
        }).count();
        eprintln!(
            "8D walk: rz({theta}) ε={eps:e} t={t} | wall {wall:.3} s | cpu-util {:.2}x | nodes {n_nodes} ({:.2} Mnode/s) | sols {} (eps-close {n_close}) | budget_hit={}",
            cpu / wall.max(1e-9),
            n_nodes as f64 / wall.max(1e-9) / 1e6,
            sols.len(),
            hit.load(std::sync::atomic::Ordering::Relaxed),
        );
    }

    /// Stage-4 warm-LLL gate experiment (docs/plan_8d_prefix_rework.md
    /// lever C): on a captured set of production prefixes (bench
    /// target_00, found/empty levels at 1e-7 and 1e-8), compare
    /// `lll_l2_8` iteration counts between the identity start and a seed
    /// = the LLL-reduced basis of the prefix-independent Q_base(k, ε).
    /// Adoption gate: ≥25% total iteration reduction, else kill (16D
    /// precedent).
    /// Env: WARM_N (prefix cap per level, default 400).
    /// Run: `cargo test --release --lib warm_lll_gate -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn warm_lll_gate() {
        use crate::synthesis::lattice::lll::{lll_l2_8_seeded, LllResult};
        use crate::synthesis::lattice::q_metric::{build_q_int, build_q_mpfr};
        use crate::synthesis::lattice::scratch::IntScratch;
        use rug::Assign;

        fn xorshift64(s: &mut u64) -> u64 {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
            *s
        }
        fn rand_angle(s: &mut u64) -> f64 {
            let b = xorshift64(s) >> 11;
            (b as f64) / ((1u64 << 53) as f64) * 2.0 * PI
        }
        let mut state: u64 = 0xC0FFEE_BAADD0E_u64 | 1;
        let a = rand_angle(&mut state);
        let b = rand_angle(&mut state);
        let c = rand_angle(&mut state);
        let target = u3(a, b, c);
        let n_cap: usize = std::env::var("WARM_N")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(400);

        // (eps, t): found levels 66@1e-7 / 76@1e-8 plus the expensive
        // empty level 74@1e-8 (M0-refresh structure, fixed pipeline).
        for &(eps, t) in &[(1e-7f64, 66u32), (1e-8, 74), (1e-8, 76)] {
            let t_prime = optimal_t_prime(t, eps);
            let t_inner = t - t_prime;
            let lde_inner: u32 = if t_inner % 2 == 1 {
                (t_inner - 1) / 2 + 1
            } else {
                t_inner / 2 + 1
            };
            let prefixes = build_ma_prefix_set(t_prime, coset_mode_for(eps));
            let target_parity = det_zeta_parity(&target);

            // Capture surviving prefixes' y vectors (both inner branches,
            // like production).
            let mut ys: Vec<[Float; 8]> = Vec::new();
            for u_l in prefixes.iter() {
                if ys.len() >= n_cap {
                    break;
                }
                if let Some(tp) = target_parity {
                    if det_zeta_parity(&u_l.to_float()) != Some(tp) {
                        continue;
                    }
                }
                let m_inner = prefix_dag_times_target(u_l, &target);
                let Some(v_inner) = try_unitary_to_uv(&m_inner) else { continue };
                ys.push(uv_to_lattice_y(v_inner, lde_inner));
                if t_inner > 0 && ys.len() < n_cap {
                    ys.push(uv_to_lattice_y(apply_t_dag_to_uv(v_inner), lde_inner));
                }
            }
            if ys.is_empty() {
                eprintln!("eps={eps:e} t={t}: no surviving prefixes (parity-dead level), skipping");
                continue;
            }

            let mut s = IntScratch::new(eps);
            // Warm seed: LLL-reduce Q_base itself. Populate q_base via one
            // build_q_mpfr call, copy it into q_mpfr, snapshot, reduce.
            build_q_mpfr(&mut s, &ys[0], lde_inner, eps);
            for i in 0..8 {
                for j in 0..8 {
                    s.q_mpfr[i][j].assign(&s.q_base[i][j]);
                }
            }
            build_q_int(&mut s);
            let (res_base, it_base) = lll_l2_8_seeded(&mut s, None);
            let warm = s.basis;
            eprintln!(
                "eps={eps:e} t={t} t'={t_prime} lde_inner={lde_inner} captured={} \
                 | q_base LLL: {res_base:?} iters={it_base}",
                ys.len()
            );

            let (mut tot_cold, mut tot_warm) = (0u64, 0u64);
            let mut nonconv = 0usize;
            for y in &ys {
                build_q_mpfr(&mut s, y, lde_inner, eps);
                build_q_int(&mut s);
                let (rc, ic) = lll_l2_8_seeded(&mut s, None);
                let (rw, iw) = lll_l2_8_seeded(&mut s, Some(&warm));
                if !matches!(rc, LllResult::Converged)
                    || !matches!(rw, LllResult::Converged)
                {
                    nonconv += 1;
                }
                tot_cold += ic as u64;
                tot_warm += iw as u64;
            }
            eprintln!(
                "  cold_iters={tot_cold} warm_iters={tot_warm} \
                 warm/cold={:.3} (gate: <=0.75) nonconverged={nonconv}",
                tot_warm as f64 / tot_cold.max(1) as f64
            );
        }
    }

    /// Census (ignored): `|L_{t'}|` sizes, `k_prefix` histogram, and the
    /// `S(t', α) = Σ_k count(t', k)/α^k` D&C cost-ratio for Clifford+T's
    /// `build_ma_prefix_set_reference` — the empirical sister of
    /// `clifford_sqrt_t`'s `fgkm_prefix_split_cost_ratio`.
    /// Run: `cargo test --release --lib build_l_size_and_cost_ratio -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn build_l_size_and_cost_ratio() {
        eprintln!("\n|L_{{t'}}| sizes:");
        for t_prime in 0..=10 {
            let l = build_ma_prefix_set_reference(t_prime);
            eprintln!("  t'={t_prime:>2}  |L_{{t'}}|={:>8}", l.len());
        }

        eprintln!("\nk_prefix histogram (Clifford+T, build_ma_prefix_set_reference):");
        for t_prime in 1..=8 {
            let l = build_ma_prefix_set_reference(t_prime);
            let mut k_min = u32::MAX; let mut k_max = 0;
            for u in l.iter() { k_min = k_min.min(u.k); k_max = k_max.max(u.k); }
            eprintln!(
                "  t'={t_prime:>2}  total={:>8}  k range [{k_min}, {k_max}]",
                l.len()
            );
        }

        eprintln!("\nS(t', α) = Σ_k count(t', k) / α^k  (D&C cost ratio):");
        eprintln!("  t'  total      α=2.0    α=2.5    α=3.0    α=3.5    α=4.0");
        for t_prime in 1..=10 {
            let l = build_ma_prefix_set_reference(t_prime);
            let mut counts: Vec<u64> = vec![0; 64];
            for u in l.iter() {
                let k = u.k as usize;
                if k < counts.len() { counts[k] += 1; }
            }
            eprint!("  {t_prime:>2}  {:>8}", l.len());
            for &alpha in &[2.0_f64, 2.5, 3.0, 3.5, 4.0] {
                let s: f64 = counts
                    .iter()
                    .enumerate()
                    .map(|(k, &c)| (c as f64) / alpha.powi(k as i32))
                    .sum();
                eprint!("   {s:>8.2}");
            }
            eprintln!();
        }
    }


