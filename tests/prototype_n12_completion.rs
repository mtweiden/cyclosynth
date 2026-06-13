use cyclosynth::matrix::U2;
use cyclosynth::rings::types::Int;
use cyclosynth::rings::ZUpsilon;
use cyclosynth::synthesis::clifford_pi12::decompose;
use cyclosynth::synthesis::lattice_upsilon::sigma::sigma_el;
use num_complex::Complex64;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::collections::HashMap;

type Mat2 = [[Complex64; 2]; 2];

fn haar_u3_seed1() -> Mat2 {
    let mut rng = StdRng::seed_from_u64(1);
    let theta = rng.random::<f64>() * (2.0 * std::f64::consts::PI);
    let phi = rng.random::<f64>() * (2.0 * std::f64::consts::PI);
    let lambda = rng.random::<f64>() * (2.0 * std::f64::consts::PI);
    let ct = (theta / 2.0).cos();
    let st = (theta / 2.0).sin();
    let global_phase = Complex64::from_polar(1.0, -(phi + lambda) / 2.0);
    [
        [
            global_phase * Complex64::new(ct, 0.0),
            global_phase * (-Complex64::from_polar(st, lambda)),
        ],
        [
            global_phase * Complex64::from_polar(st, phi),
            global_phase * Complex64::from_polar(ct, phi + lambda),
        ],
    ]
}

fn z_from_i64(c: &[i64; 8]) -> ZUpsilon {
    ZUpsilon::new(
        Int::from_i64(c[0]),
        Int::from_i64(c[1]),
        Int::from_i64(c[2]),
        Int::from_i64(c[3]),
        Int::from_i64(c[4]),
        Int::from_i64(c[5]),
        Int::from_i64(c[6]),
        Int::from_i64(c[7]),
    )
}

fn norm_key(z: ZUpsilon) -> (Int, Int, Int, Int) {
    z.complex_norm_sqr_components_twice()
}

fn zeta_pow(l: u32) -> ZUpsilon {
    cyclosynth::synthesis::lattice_upsilon::zeta_pow(l)
}

fn build_u(u1: ZUpsilon, u2: ZUpsilon, k: u32, phase: u32) -> U2<ZUpsilon> {
    let p = zeta_pow(phase);
    U2::new(u1, -(u2.conj() * p), u2, u1.conj() * p, k)
}

fn enumerate_norm_table(euclid_sq_bound: i64) -> HashMap<(Int, Int, Int, Int), Vec<[i64; 8]>> {
    fn walk(
        pos: usize,
        rem_sq: i64,
        c: &mut [i64; 8],
        out: &mut HashMap<(Int, Int, Int, Int), Vec<[i64; 8]>>,
    ) {
        if pos == 8 {
            let z = z_from_i64(c);
            out.entry(norm_key(z)).or_default().push(*c);
            return;
        }
        let b = (rem_sq as f64).sqrt().floor() as i64;
        for x in -b..=b {
            c[pos] = x;
            walk(pos + 1, rem_sq - x * x, c, out);
        }
    }

    let mut out = HashMap::new();
    let mut c = [0i64; 8];
    walk(0, euclid_sq_bound, &mut c, &mut out);
    out
}

fn solve_8(mut a: [[f64; 8]; 8], mut b: [f64; 8]) -> [f64; 8] {
    for k in 0..8 {
        let mut piv = k;
        let mut best = a[k][k].abs();
        for i in (k + 1)..8 {
            if a[i][k].abs() > best {
                best = a[i][k].abs();
                piv = i;
            }
        }
        if piv != k {
            a.swap(k, piv);
            b.swap(k, piv);
        }
        let akk = a[k][k];
        for i in (k + 1)..8 {
            let f = a[i][k] / akk;
            a[i][k] = 0.0;
            for j in (k + 1)..8 {
                a[i][j] -= f * a[k][j];
            }
            b[i] -= f * b[k];
        }
    }
    let mut x = [0.0; 8];
    for i in (0..8).rev() {
        let mut s = b[i];
        for j in (i + 1)..8 {
            s -= a[i][j] * x[j];
        }
        x[i] = s / a[i][i];
    }
    x
}

fn u1_cap_center(target: &Mat2, k: u32) -> [f64; 8] {
    let sigma = sigma_el();
    let scale = 2.0_f64.powf(k as f64 / 2.0);
    let v = [target[0][0].re * scale, target[0][0].im * scale];

    let mut gram = [[0.0; 8]; 8];
    for i in 0..8 {
        for j in 0..8 {
            for row in 0..8 {
                gram[i][j] += sigma[row][i] * sigma[row][j];
            }
        }
    }
    let mut rhs = [0.0; 8];
    for j in 0..8 {
        rhs[j] = sigma[0][j] * v[0] + sigma[1][j] * v[1];
    }
    solve_8(gram, rhs)
}

#[test]
#[ignore = "prototype: u1 cap enumeration plus brute u2 norm completion"]
fn prototype_u1_completion_seed1() {
    let target = haar_u3_seed1();
    let eps = std::env::var("CYCLOSYNTH_COMPLETION_EPS")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(1e-5);
    let k = std::env::var("CYCLOSYNTH_COMPLETION_K")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(20);
    let u1_radius = std::env::var("CYCLOSYNTH_COMPLETION_U1_RADIUS")
        .ok()
        .and_then(|s| s.parse::<i64>().ok())
        .unwrap_or(2);
    let u2_euclid_sq = std::env::var("CYCLOSYNTH_COMPLETION_U2_EUCLID_SQ")
        .ok()
        .and_then(|s| s.parse::<i64>().ok())
        .unwrap_or(1i64 << (k + 1));

    let skip_u2 = std::env::var_os("CYCLOSYNTH_COMPLETION_SKIP_U2").is_some();
    let table = if skip_u2 {
        HashMap::new()
    } else {
        enumerate_norm_table(u2_euclid_sq)
    };
    eprintln!(
        "completion prototype k={k} eps={eps:.1e} u1_radius={u1_radius} u2_euclid_sq={u2_euclid_sq} skip_u2={skip_u2} norm_keys={} u2_words={}",
        table.len(),
        table.values().map(Vec::len).sum::<usize>()
    );

    let scale = 2.0_f64.powf(k as f64 / 2.0);
    let wanted = target[0][0] * scale;
    let center = u1_cap_center(&target, k);
    let center_i: [i64; 8] = std::array::from_fn(|i| center[i].round() as i64);
    eprintln!("u1 center = {center:?}");
    eprintln!("u1 rounded center = {center_i:?}");

    let mut u1_seen = 0usize;
    let mut u1_tested = 0usize;
    let mut alpha_in_table = 0usize;
    let mut completions = 0usize;
    let mut alpha_samples: Vec<String> = Vec::new();
    let mut best_d = f64::INFINITY;
    let mut best: Option<(u32, usize, String)> = None;

    fn walk_u1<F: FnMut(&[i64; 8])>(
        pos: usize,
        radius: i64,
        center: &[i64; 8],
        c: &mut [i64; 8],
        cb: &mut F,
    ) {
        if pos == 8 {
            cb(c);
            return;
        }
        for dx in -radius..=radius {
            c[pos] = center[pos] + dx;
            walk_u1(pos + 1, radius, center, c, cb);
        }
    }

    let mut c1 = [0i64; 8];
    walk_u1(0, u1_radius, &center_i, &mut c1, &mut |c1| {
        u1_tested += 1;
        let u1 = z_from_i64(c1);
        let cap_d = (u1.to_complex() - wanted).norm() / scale;
        if cap_d > eps {
            return;
        }
        u1_seen += 1;
        let (r, s2, s3, s6) = norm_key(u1);
        let alpha = (Int::from_i64(1i64 << k) - r, -s2, -s3, -s6);
        if alpha_samples.len() < 8 {
            alpha_samples.push(format!(
                "({}, {}, {}, {})",
                alpha.0, alpha.1, alpha.2, alpha.3
            ));
        }
        if skip_u2 {
            return;
        }
        let Some(u2s) = table.get(&alpha) else {
            return;
        };
        alpha_in_table += 1;
        for c2 in u2s {
            completions += 1;
            let u2 = z_from_i64(c2);
            for phase in 0..24 {
                let u = build_u(u1, u2, k, phase);
                let d =
                    cyclosynth::synthesis::distance::diamond_distance_float(&u.to_float(), &target);
                if d < best_d {
                    best_d = d;
                    best = Some((
                        phase,
                        decompose(&u).t12_count,
                        format!("u1={c1:?} u2={c2:?}"),
                    ));
                }
            }
        }
    });

    eprintln!(
        "u1_tested={u1_tested} u1_cap_hits={u1_seen} alpha_samples={alpha_samples:?} alpha_in_table={alpha_in_table} completions={completions} best_d={best_d:.6e} best={best:?}"
    );
}
