//! The 8D grid problem for Clifford+√T z-rotations.
//!
//! Enumerate u = A + ζB (A, B ∈ Z[g], 8 integer coordinates) with, at
//! level k:
//!   - principal embedding u₁/√2^k inside the ε-cap around e^{−iθ/2},
//!   - |u_m|² ≤ 2^k in the three conjugate embeddings (⟺ the norm-equation
//!     right-hand side ξ = 2^k − u·ū is totally non-negative).
//!
//! Geometry is f64 (whitened per-plane, with a completeness margin);
//! every candidate is re-checked EXACTLY (ring arithmetic + MPFR) before
//! it reaches the visitor, so float error can only cost completeness at
//! the margin, never correctness — the same layering as the rest of the
//! crate.

use rug::Integer;

use crate::rings::types::MpFloat;

use crate::rings::real::{generator_embeddings, ZRootTwoPlusRootTwo};
use crate::rings::zzeta::ZZetaBig;

/// Safety inflation on the enumeration radius (completeness margin for
/// the f64 geometry).
const RADIUS_MARGIN: f64 = 1.02;

/// One grid candidate: u = A + ζB and its precomputed ξ = 2^k − u·ū ≥ 0.
pub(crate) struct Candidate {
    pub a: ZRootTwoPlusRootTwo,
    pub b: ZRootTwoPlusRootTwo,
    pub xi: ZRootTwoPlusRootTwo,
}

/// θ-dependent context reused across levels.
///
/// The whitened constraint matrix is k-independent up to a uniform √2^k
/// scale (every axis is proportional to R = √2^k while the center is
/// fixed), so the expensive part — LLL + Cholesky of the badly
/// conditioned (~ε⁻²) system — happens ONCE per context, in MPFR. Each
/// level then Babai-rounds the (astronomically distant) center in MPFR
/// and hands an O(1)-magnitude residual problem to a plain f64
/// Schnorr–Euchner.
pub(crate) struct GridCtx {
    prec: u32,
    /// cos/sin of −θ/2 (cap direction) at full precision.
    z_x: MpFloat,
    z_y: MpFloat,
    /// √(1 − ε²/4) — the cap's cosine threshold.
    d: MpFloat,
    epsilon: f64,
    /// One-time reduction of the unit-scale whitened lattice.
    red: Option<Reduction>,
}

struct Reduction {
    /// Integer coordinates of reduced vector j (column j), i128 —
    /// unimodular entries can reach the conditioning scale.
    basis: [[i128; 8]; 8],
    /// Upper-triangular Cholesky factor of the reduced Gram (MPFR).
    r_mp: Vec<Vec<MpFloat>>,
    /// Same, rounded to f64 for the SE stage.
    r_f: [[f64; 8]; 8],
    /// Center of the unit-scale system in the reduced frame (MPFR).
    t_unit: Vec<MpFloat>,
}

impl GridCtx {
    pub fn new(theta: &MpFloat, epsilon: f64, prec: u32) -> Self {
        let half = MpFloat::with_val(prec, theta / 2u32);
        let z_x = half.clone().cos();
        let z_y = MpFloat::with_val(prec, -half.sin());
        let eps_f = MpFloat::with_val(prec, epsilon);
        let one = MpFloat::with_val(prec, 1.0);
        let d = MpFloat::with_val(prec, &one - MpFloat::with_val(prec, &eps_f * &eps_f) / 4u32)
            .sqrt();
        let mut ctx = Self { prec, z_x, z_y, d, epsilon, red: None };
        ctx.red = ctx.reduce_once();
        ctx
    }

    /// Geometry precision: the Gram spans ~ε⁻⁴.
    fn prec_geo(&self) -> u32 {
        let decades = (1.0 / self.epsilon).log10().ceil().max(1.0);
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let bits = (14.0 * decades) as u32;
        bits + 128
    }

    /// Unit-scale (R = 1) whitened system in MPFR.
    fn whitened_system_mp(&self) -> (Vec<Vec<MpFloat>>, Vec<MpFloat>) {
        let p = self.prec_geo();
        let gen = generator_embeddings(p);
        let pi = MpFloat::with_val(p, rug::float::Constant::Pi);
        let ms = [1u32, 3, 5, 7];
        let mut w = vec![vec![MpFloat::with_val(p, 0.0); 8]; 8];
        let mut c = vec![MpFloat::with_val(p, 0.0); 8];
        let eps = MpFloat::with_val(p, self.epsilon);
        for (m, &mm) in ms.iter().enumerate() {
            let g = MpFloat::with_val(p, &gen[m]);
            let ang = MpFloat::with_val(p, &pi * mm) / 8u32;
            let cz = ang.clone().cos();
            let sz = ang.sin();
            let pow = [
                MpFloat::with_val(p, 1.0),
                g.clone(),
                MpFloat::with_val(p, &g * &g),
                MpFloat::with_val(p, MpFloat::with_val(p, &g * &g) * &g),
            ];
            let mut re_row = vec![MpFloat::with_val(p, 0.0); 8];
            let mut im_row = vec![MpFloat::with_val(p, 0.0); 8];
            for i in 0..4 {
                re_row[i] = pow[i].clone();
                re_row[4 + i] = MpFloat::with_val(p, &cz * &pow[i]);
                im_row[4 + i] = MpFloat::with_val(p, &sz * &pow[i]);
            }
            if m == 0 {
                let zx = MpFloat::with_val(p, &self.z_x);
                let zy = MpFloat::with_val(p, &self.z_y);
                let axis_r = MpFloat::with_val(p, &eps * &eps) / 4u32;
                let axis_t = eps.clone();
                let dc = MpFloat::with_val(p, &self.d);
                for i in 0..8 {
                    let radial = MpFloat::with_val(p, &zx * &re_row[i])
                        + MpFloat::with_val(p, &zy * &im_row[i]);
                    let tangent = MpFloat::with_val(p, &zx * &im_row[i])
                        - MpFloat::with_val(p, &zy * &re_row[i]);
                    w[0][i] = MpFloat::with_val(p, &radial / &axis_r);
                    w[1][i] = MpFloat::with_val(p, &tangent / &axis_t);
                }
                // Center p = d·ẑ: radial component d, tangential 0.
                c[0] = MpFloat::with_val(p, &dc / &axis_r);
                c[1] = MpFloat::with_val(p, 0.0);
            } else {
                w[2 * m].clone_from_slice(&re_row);
                w[2 * m + 1].clone_from_slice(&im_row);
            }
        }
        (w, c)
    }

    /// One-time MPFR LLL + Cholesky + center projection.
    fn reduce_once(&self) -> Option<Reduction> {
        let p = self.prec_geo();
        let (w, c) = self.whitened_system_mp();
        let (basis, bcols) = lll_reduce_mp(&w, p)?;
        let r_mp = cholesky_upper_mp(&bcols, p)?;
        let t_unit = center_in_frame_mp(&r_mp, &bcols, &c, p);
        let mut r_f = [[0f64; 8]; 8];
        for i in 0..8 {
            for j in 0..8 {
                r_f[i][j] = r_mp[i][j].to_f64();
            }
        }
        Some(Reduction { basis, r_mp, r_f, t_unit })
    }

    /// Exact acceptance: ξ totally non-negative and the cap cosine
    /// condition at full precision. Returns ξ on success.
    fn exact_accept(&self, a: &ZRootTwoPlusRootTwo, b: &ZRootTwoPlusRootTwo, k: u32) -> Option<ZRootTwoPlusRootTwo> {
        // ξ = 2^k − (A² + B² + gAB), totally non-negative.
        let rel = a.mul(a).add(&b.mul(b)).add(&ZRootTwoPlusRootTwo::generator().mul(a).mul(b));
        let two_k = ZRootTwoPlusRootTwo::from_int(Integer::from(1) << k);
        let xi = two_k.sub(&rel);
        if !xi.is_totally_nonneg() {
            return None;
        }
        // Cap: Re(u·e^{iθ/2}) ≥ d·√2^k, u = e(A) + e^{iπ/8}e(B) at prec.
        let prec = self.prec;
        let gen = generator_embeddings(prec);
        let ea = a.embed(&gen[0], prec);
        let eb = b.embed(&gen[0], prec);
        let pi = MpFloat::with_val(prec, rug::float::Constant::Pi);
        let ang = MpFloat::with_val(prec, &pi / 8u32);
        let (c1, s1) = (ang.clone().cos(), ang.sin());
        let u_re = MpFloat::with_val(prec, &ea + &MpFloat::with_val(prec, &eb * &c1));
        let u_im = MpFloat::with_val(prec, &eb * &s1);
        let cos_sim =
            MpFloat::with_val(prec, &self.z_x * &u_re) + MpFloat::with_val(prec, &self.z_y * &u_im);
        let mut rhs = MpFloat::with_val(prec, &self.d);
        rhs <<= k / 2;
        if k % 2 == 1 {
            rhs *= MpFloat::with_val(prec, 2.0).sqrt();
        }
        if cos_sim >= rhs {
            Some(xi)
        } else {
            None
        }
    }

    /// Is u divisible by √2 in Z[ζ]? Those candidates already appeared
    /// at level k−1 — skip.
    fn divisible_by_sqrt2(a: &ZRootTwoPlusRootTwo, b: &ZRootTwoPlusRootTwo) -> bool {
        let u = ZZetaBig::from_module(a, b);
        // u/√2 = u·√2/2: in-ring ⟺ all coefficients of u·√2 are even.
        let s2 = ZZetaBig::from_real(&ZRootTwoPlusRootTwo::sqrt2());
        let prod = u.mul(&s2);
        prod.coeffs().iter().all(|x| x.is_even())
    }

    /// Enumerate level-k candidates into `visit` (return true to stop);
    /// candidates stream in ascending whitened distance from the cap
    /// center — a quality-first order.
    pub fn visit_level(&self, k: u32, visit: &mut dyn FnMut(Candidate) -> bool) {
        let Some(red) = &self.red else { return };
        let p = self.prec_geo();
        // Level scale R = √2^{k}: the level-k system is
        // |W_unit·x − R·c|² ≤ (2R)², i.e. |R_unit·n − R·t_unit|² ≤ (2R)².
        let mut r_scale = MpFloat::with_val(p, 1.0);
        r_scale <<= k / 2;
        if k % 2 == 1 {
            r_scale *= MpFloat::with_val(p, 2.0).sqrt();
        }
        // Babai round the scaled center: n0 = round(R⁻¹·(R·t_unit)) via
        // back-substitution on the MPFR factor; the residual is O(1) in
        // lattice units and safely f64. n0 entries are ~ε⁻² — kept as
        // exact Integers (they pass i128 around ε=1e-16).
        let mut n0: [Integer; 8] = Default::default();
        let mut resid = [0f64; 8]; // t_k − R_mp·n0, computed exactly in MPFR
        {
            let mut t_k: Vec<MpFloat> = red
                .t_unit
                .iter()
                .map(|t| MpFloat::with_val(p, t * &r_scale))
                .collect();
            for i in (0..8).rev() {
                let mut v = t_k[i].clone();
                for j in (i + 1)..8 {
                    v -= MpFloat::with_val(p, &red.r_mp[i][j] * &MpFloat::with_val(p, &n0[j]));
                }
                v /= &red.r_mp[i][i];
                n0[i] = v.round().to_integer().unwrap_or_default();
            }
            // Residual target in the reduced frame: t_k − R·n0 per row.
            for i in 0..8 {
                let mut v = t_k[i].clone();
                for j in i..8 {
                    v -= MpFloat::with_val(p, &red.r_mp[i][j] * &MpFloat::with_val(p, &n0[j]));
                }
                resid[i] = v.to_f64();
            }
            let _ = &mut t_k;
        }
        // Anchor lattice point x0 = basis·n0, computed exactly: the
        // intermediate products basis[i][j]·n0[j] overflow i128 at deep ε
        // even though x0 itself is a lattice point of size O(R).
        let mut x0 = [0i128; 8];
        for i in 0..8 {
            let mut acc = Integer::new();
            for j in 0..8 {
                acc += Integer::from(red.basis[i][j]) * &n0[j];
            }
            let Some(v) = acc.to_i128() else { return };
            x0[i] = v;
        }
        let radius = 2.0 * r_scale.to_f64() * RADIUS_MARGIN;
        let radius_sq = radius * radius;
        let mut hits: Vec<([i64; 8], f64)> = Vec::new();
        se_enumerate_resid(&red.r_f, &resid, radius_sq, &mut |m, dist| {
            hits.push((*m, dist));
            // Levels far above the first feasible one hold ~16× more
            // points per level; the driver only consumes the first few.
            hits.len() >= 50_000
        });
        hits.sort_by(|x, y| x.1.total_cmp(&y.1));
        #[cfg(test)]
        let mut skip_stats = [0u64; 6]; // ovf, i64, zero, div2, reject, accept
        for (m, _) in hits {
            // x = x0 + basis·m; the SE offset m is O(1) so every product
            // stays far from the i128 edge.
            let mut x = x0;
            let mut ok = true;
            for i in 0..8 {
                for (j, mj) in m.iter().enumerate() {
                    let Some(prod) = red.basis[i][j].checked_mul(i128::from(*mj)) else {
                        ok = false;
                        break;
                    };
                    let Some(sum) = x[i].checked_add(prod) else {
                        ok = false;
                        break;
                    };
                    x[i] = sum;
                }
                if !ok {
                    break;
                }
            }
            if !ok {
                #[cfg(test)]
                {
                    skip_stats[0] += 1;
                }
                continue;
            }
            let Some(xs) = x
                .iter()
                .map(|v| i64::try_from(*v).ok())
                .collect::<Option<Vec<i64>>>()
            else {
                #[cfg(test)]
                {
                    skip_stats[1] += 1;
                }
                continue;
            };
            let a = ZRootTwoPlusRootTwo::from_i64(xs[0], xs[1], xs[2], xs[3]);
            let b = ZRootTwoPlusRootTwo::from_i64(xs[4], xs[5], xs[6], xs[7]);
            if a.is_zero() && b.is_zero() {
                #[cfg(test)]
                {
                    skip_stats[2] += 1;
                }
                continue;
            }
            if Self::divisible_by_sqrt2(&a, &b) {
                #[cfg(test)]
                {
                    skip_stats[3] += 1;
                }
                continue;
            }
            if let Some(xi) = self.exact_accept(&a, &b, k) {
                #[cfg(test)]
                {
                    skip_stats[5] += 1;
                }
                if visit(Candidate { a, b, xi }) {
                    return;
                }
            } else {
                #[cfg(test)]
                {
                    skip_stats[4] += 1;
                }
            }
        }
        #[cfg(test)]
        if std::env::var_os("GSQ_SKIP_STATS").is_some() {
            eprintln!(
                "      [visit_level k={k}] ovf={} i64={} zero={} div2={} reject={} accept={}",
                skip_stats[0], skip_stats[1], skip_stats[2], skip_stats[3], skip_stats[4], skip_stats[5]
            );
        }
    }
}

#[cfg(test)]
impl GridCtx {
    /// For the first `n` raw hits at level k: (cap cosine margin in units
    /// of ε²·√2^k, ξ totally-non-negative?).
    pub(crate) fn reject_anatomy(&self, k: u32, n: usize) -> Vec<(f64, bool)> {
        let mut out = Vec::new();
        let Some(red) = &self.red else { return out };
        let p = self.prec_geo();
        let mut r_scale = MpFloat::with_val(p, 1.0);
        r_scale <<= k / 2;
        if k % 2 == 1 {
            r_scale *= MpFloat::with_val(p, 2.0).sqrt();
        }
        let mut n0 = [0i128; 8];
        let mut resid = [0f64; 8];
        {
            let t_k: Vec<MpFloat> = red
                .t_unit
                .iter()
                .map(|t| MpFloat::with_val(p, t * &r_scale))
                .collect();
            for i in (0..8).rev() {
                let mut v = t_k[i].clone();
                for j in (i + 1)..8 {
                    v -= MpFloat::with_val(p, &red.r_mp[i][j] * &MpFloat::with_val(p, n0[j]));
                }
                v /= &red.r_mp[i][i];
                n0[i] = v.round().to_integer().and_then(|x| x.to_i128()).unwrap_or(0);
            }
            for i in 0..8 {
                let mut v = t_k[i].clone();
                for j in i..8 {
                    v -= MpFloat::with_val(p, &red.r_mp[i][j] * &MpFloat::with_val(p, n0[j]));
                }
                resid[i] = v.to_f64();
            }
        }
        let radius = 2.0 * r_scale.to_f64() * RADIUS_MARGIN;
        let mut hits: Vec<[i64; 8]> = Vec::new();
        se_enumerate_resid(&red.r_f, &resid, radius * radius, &mut |m, _d| {
            hits.push(*m);
            hits.len() >= n
        });
        for m in hits {
            let mut x = [0i128; 8];
            for i in 0..8 {
                for (j, mj) in m.iter().enumerate() {
                    x[i] += red.basis[i][j] * (n0[j] + i128::from(*mj));
                }
            }
            let a = ZRootTwoPlusRootTwo::from_i64(
                x[0] as i64, x[1] as i64, x[2] as i64, x[3] as i64,
            );
            let b = ZRootTwoPlusRootTwo::from_i64(
                x[4] as i64, x[5] as i64, x[6] as i64, x[7] as i64,
            );
            // Exact cap margin: (cos_sim − d·√2^k) / (ε²·√2^k).
            let prec = self.prec;
            let gen = generator_embeddings(prec);
            let ea = a.embed(&gen[0], prec);
            let eb = b.embed(&gen[0], prec);
            let pi = MpFloat::with_val(prec, rug::float::Constant::Pi);
            let ang = MpFloat::with_val(prec, &pi / 8u32);
            let (c1, s1) = (ang.clone().cos(), ang.sin());
            let u_re = MpFloat::with_val(prec, &ea + &MpFloat::with_val(prec, &eb * &c1));
            let u_im = MpFloat::with_val(prec, &eb * &s1);
            let cos_sim = MpFloat::with_val(prec, &self.z_x * &u_re)
                + MpFloat::with_val(prec, &self.z_y * &u_im);
            let mut rhs = MpFloat::with_val(prec, &self.d);
            rhs <<= k / 2;
            if k % 2 == 1 {
                rhs *= MpFloat::with_val(prec, 2.0).sqrt();
            }
            let margin = MpFloat::with_val(prec, &cos_sim - &rhs);
            let mut scale = MpFloat::with_val(prec, self.epsilon);
            scale.square_mut();
            scale <<= k / 2;
            if k % 2 == 1 {
                scale *= MpFloat::with_val(prec, 2.0).sqrt();
            }
            let rel = MpFloat::with_val(prec, &margin / &scale).to_f64();
            let rel_norm = a.mul(&a).add(&b.mul(&b)).add(
                &ZRootTwoPlusRootTwo::generator().mul(&a).mul(&b),
            );
            let two_k = ZRootTwoPlusRootTwo::from_int(Integer::from(1) << k);
            let xi = two_k.sub(&rel_norm);
            let div2 = Self::divisible_by_sqrt2(&a, &b);
            let acc = self.exact_accept(&a, &b, k).is_some();
            eprintln!(
                "      [direct] div_sqrt2={div2} exact_accept={acc}"
            );
            out.push((rel, xi.is_totally_nonneg()));
        }
        out
    }

    /// Diagnostics: (raw SE hits, exact-accepted, r_f diag min/max, max |basis|, |resid| max).
    pub(crate) fn level_stats(&self, k: u32) -> (u64, u64, f64, f64, f64, f64) {
        let Some(red) = &self.red else {
            return (0, 0, 0.0, 0.0, 0.0, 0.0);
        };
        let p = self.prec_geo();
        let mut r_scale = MpFloat::with_val(p, 1.0);
        r_scale <<= k / 2;
        if k % 2 == 1 {
            r_scale *= MpFloat::with_val(p, 2.0).sqrt();
        }
        let mut n0 = [0i128; 8];
        let mut resid = [0f64; 8];
        {
            let t_k: Vec<MpFloat> = red
                .t_unit
                .iter()
                .map(|t| MpFloat::with_val(p, t * &r_scale))
                .collect();
            for i in (0..8).rev() {
                let mut v = t_k[i].clone();
                for j in (i + 1)..8 {
                    v -= MpFloat::with_val(p, &red.r_mp[i][j] * &MpFloat::with_val(p, n0[j]));
                }
                v /= &red.r_mp[i][i];
                n0[i] = v.round().to_integer().and_then(|x| x.to_i128()).unwrap_or(0);
            }
            for i in 0..8 {
                let mut v = t_k[i].clone();
                for j in i..8 {
                    v -= MpFloat::with_val(p, &red.r_mp[i][j] * &MpFloat::with_val(p, n0[j]));
                }
                resid[i] = v.to_f64();
            }
        }
        let radius = 2.0 * r_scale.to_f64() * RADIUS_MARGIN;
        let mut raw = 0u64;
        se_enumerate_resid(&red.r_f, &resid, radius * radius, &mut |_m, _d| {
            raw += 1;
            raw >= 200_000
        });
        let mut accepted = 0u64;
        self.visit_level(k, &mut |_c| {
            accepted += 1;
            false
        });
        let diag_min = (0..8).map(|i| red.r_f[i][i].abs()).fold(f64::INFINITY, f64::min);
        let diag_max = (0..8).map(|i| red.r_f[i][i].abs()).fold(0.0, f64::max);
        let bmax = red
            .basis
            .iter()
            .flatten()
            .map(|x| x.unsigned_abs())
            .max()
            .unwrap_or(0);
        #[allow(clippy::cast_precision_loss)]
        let bmax_f = bmax as f64;
        let rmax = resid.iter().fold(0.0f64, |m, x| m.max(x.abs()));
        (raw, accepted, diag_min, diag_max, bmax_f, rmax)
    }
}

// ─── One-time MPFR reduction + per-level f64 Schnorr–Euchner ─────────────────

/// MPFR LLL on the columns of W (δ = 0.99, incremental-k). Returns the
/// unimodular transform (exact during reduction, i128 on output — the
/// final reduced entries are small) and the reduced basis columns in MPFR.
#[allow(clippy::type_complexity)]
fn lll_reduce_mp(
    w: &[Vec<MpFloat>],
    p: u32,
) -> Option<([[i128; 8]; 8], Vec<Vec<MpFloat>>)> {
    // Columns as vectors.
    let mut b: Vec<Vec<MpFloat>> = (0..8)
        .map(|j| (0..8).map(|i| w[i][j].clone()).collect())
        .collect();
    // Unimodular transform in exact Integers: size-reduction
    // intermediates scale with the conditioning (~ε⁻²) and overflow
    // i128 below ε ≈ 1e-19, while the FINAL reduced entries are small —
    // conversion happens once at the end.
    let mut u: Vec<Vec<Integer>> = (0..8)
        .map(|j| (0..8).map(|i| Integer::from(i32::from(i == j))).collect())
        .collect();
    let dot = |x: &[MpFloat], y: &[MpFloat]| -> MpFloat {
        let mut s = MpFloat::with_val(p, 0.0);
        for i in 0..8 {
            s += MpFloat::with_val(p, &x[i] * &y[i]);
        }
        s
    };
    let gram_schmidt =
        |b: &Vec<Vec<MpFloat>>| -> Option<(Vec<MpFloat>, Vec<Vec<MpFloat>>)> {
            let mut gs: Vec<Vec<MpFloat>> = Vec::with_capacity(8);
            let mut mu = vec![vec![MpFloat::with_val(p, 0.0); 8]; 8];
            let mut norms = vec![MpFloat::with_val(p, 0.0); 8];
            for i in 0..8 {
                let mut v = b[i].clone();
                for j in 0..i {
                    if norms[j].is_zero() || norms[j].is_sign_negative() {
                        return None;
                    }
                    mu[i][j] = MpFloat::with_val(p, dot(&b[i], &gs[j]) / &norms[j]);
                    for t in 0..8 {
                        let sub = MpFloat::with_val(p, &mu[i][j] * &gs[j][t]);
                        v[t] -= sub;
                    }
                }
                norms[i] = dot(&v, &v);
                if norms[i].is_zero() || norms[i].is_sign_negative() {
                    return None;
                }
                gs.push(v);
            }
            Some((norms, mu))
        };
    let (mut norms, mut mu) = gram_schmidt(&b)?;
    let mut k = 1usize;
    let mut iters = 0u64;
    while k < 8 {
        iters += 1;
        if iters > 200_000 {
            return None;
        }
        for j in (0..k).rev() {
            let r = mu[k][j].clone().round();
            if r.clone().abs() >= 0.5 {
                let ri = r.to_integer()?;
                for t in 0..8 {
                    let sub = MpFloat::with_val(p, &r * &b[j][t]);
                    b[k][t] -= sub;
                    let prod = Integer::from(&ri * &u[j][t]);
                    u[k][t] -= prod;
                }
                for j2 in 0..j {
                    let sub = MpFloat::with_val(p, &r * &mu[j][j2]);
                    mu[k][j2] -= sub;
                }
                mu[k][j] -= r;
            }
        }
        let lhs = norms[k].clone();
        let thresh = MpFloat::with_val(p, 0.99)
            - MpFloat::with_val(p, &mu[k][k - 1] * &mu[k][k - 1]);
        let rhs = MpFloat::with_val(p, &thresh * &norms[k - 1]);
        if lhs >= rhs {
            k += 1;
        } else {
            b.swap(k, k - 1);
            u.swap(k, k - 1);
            let refreshed = gram_schmidt(&b)?;
            norms = refreshed.0;
            mu = refreshed.1;
            k = k.max(2) - 1;
        }
    }
    let mut basis = [[0i128; 8]; 8];
    for (j, row) in u.iter().enumerate() {
        for i in 0..8 {
            basis[i][j] = row[i].to_i128()?;
        }
    }
    Some((basis, b))
}

/// Upper-triangular MPFR Cholesky of the reduced Gram.
fn cholesky_upper_mp(bcols: &[Vec<MpFloat>], p: u32) -> Option<Vec<Vec<MpFloat>>> {
    let dot = |x: &[MpFloat], y: &[MpFloat]| -> MpFloat {
        let mut s = MpFloat::with_val(p, 0.0);
        for i in 0..8 {
            s += MpFloat::with_val(p, &x[i] * &y[i]);
        }
        s
    };
    let mut g = vec![vec![MpFloat::with_val(p, 0.0); 8]; 8];
    for i in 0..8 {
        for j in 0..8 {
            g[i][j] = dot(&bcols[i], &bcols[j]);
        }
    }
    let mut r = vec![vec![MpFloat::with_val(p, 0.0); 8]; 8];
    for i in 0..8 {
        for j in i..8 {
            let mut s = g[i][j].clone();
            for t in 0..i {
                s -= MpFloat::with_val(p, &r[t][i] * &r[t][j]);
            }
            if i == j {
                if s.is_sign_negative() || s.is_zero() {
                    return None;
                }
                r[i][i] = s.sqrt();
            } else {
                r[i][j] = MpFloat::with_val(p, &s / &r[i][i]);
            }
        }
    }
    Some(r)
}

/// t with |B·n − c|² = |R·n − t|² (Rᵀt = Bᵀc, forward substitution).
fn center_in_frame_mp(
    r: &[Vec<MpFloat>],
    bcols: &[Vec<MpFloat>],
    c: &[MpFloat],
    p: u32,
) -> Vec<MpFloat> {
    let mut rhs = vec![MpFloat::with_val(p, 0.0); 8];
    for j in 0..8 {
        let mut s = MpFloat::with_val(p, 0.0);
        for i in 0..8 {
            s += MpFloat::with_val(p, &bcols[j][i] * &c[i]);
        }
        rhs[j] = s;
    }
    let mut t = vec![MpFloat::with_val(p, 0.0); 8];
    for i in 0..8 {
        let mut s = rhs[i].clone();
        for j in 0..i {
            s -= MpFloat::with_val(p, &r[j][i] * &t[j]);
        }
        t[i] = MpFloat::with_val(p, &s / &r[i][i]);
    }
    t
}

/// Schnorr–Euchner: all m ∈ Z⁸ with |R·m − t|² ≤ r², upper-triangular R
/// and O(1)-magnitude residual target t (the Babai shift guarantees this).
fn se_enumerate_resid(
    r: &[[f64; 8]; 8],
    t: &[f64; 8],
    radius_sq: f64,
    visit: &mut dyn FnMut(&[i64; 8], f64) -> bool,
) {
    let mut n = [0i64; 8];
    se_rec(7, 0.0, r, t, radius_sq, &mut n, visit);
}

#[allow(clippy::cast_precision_loss)]
fn se_rec(
    lvl: usize,
    acc: f64,
    r: &[[f64; 8]; 8],
    t: &[f64; 8],
    radius_sq: f64,
    n: &mut [i64; 8],
    visit: &mut dyn FnMut(&[i64; 8], f64) -> bool,
) -> bool {
    let mut target = t[lvl];
    for j in (lvl + 1)..8 {
        target -= r[lvl][j] * (n[j] as f64);
    }
    target /= r[lvl][lvl];
    #[allow(clippy::cast_possible_truncation)]
    let center = target.round() as i64;
    let rii_sq = r[lvl][lvl] * r[lvl][lvl];
    let mut up = center;
    let mut down = center - 1;
    let mut up_alive = true;
    let mut down_alive = true;
    let mut next_up = true;
    while up_alive || down_alive {
        let cand = if next_up && up_alive {
            let c = up;
            up += 1;
            c
        } else if down_alive {
            let c = down;
            down -= 1;
            c
        } else {
            let c = up;
            up += 1;
            c
        };
        next_up = !next_up;
        let delta = (cand as f64) - target;
        let cost = acc + delta * delta * rii_sq;
        if cost > radius_sq {
            if cand >= center {
                up_alive = false;
            } else {
                down_alive = false;
            }
            continue;
        }
        n[lvl] = cand;
        if lvl == 0 {
            if visit(n, cost) {
                return true;
            }
        } else if se_rec(lvl - 1, cost, r, t, radius_sq, n, visit) {
            return true;
        }
    }
    false
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Definitive completeness check at small k: brute-force every
    /// coefficient vector in a box and compare against visit_level's set.
    /// Coefficient bound: all four |embeddings| ≤ √2^k caps the coords
    /// well inside ±3 for k ≤ 3.
    #[test]
    #[ignore = "thorough probe (~2 min); the default suite keeps test_level_scan_sane"]
    fn test_enumerator_completeness_bruteforce() {
        let prec = 96;
        let theta = MpFloat::with_val(prec, 0.7_f64);
        let ctx = GridCtx::new(&theta, 0.8, prec);
        for k in [2u32, 3] {
            // Collect the enumerator's answer set.
            let mut got: std::collections::HashSet<[i64; 8]> = std::collections::HashSet::new();
            ctx.visit_level(k, &mut |cand| {
                got.insert([
                    cand.a.a.to_i64().unwrap(),
                    cand.a.b.to_i64().unwrap(),
                    cand.a.c.to_i64().unwrap(),
                    cand.a.d.to_i64().unwrap(),
                    cand.b.a.to_i64().unwrap(),
                    cand.b.b.to_i64().unwrap(),
                    cand.b.c.to_i64().unwrap(),
                    cand.b.d.to_i64().unwrap(),
                ]);
                false
            });
            // Brute force with a cheap exact integer prefilter (relative
            // norm's rational coefficient must satisfy the trace bound
            // |a-coeff of ξ| ≤ 4·2^k) before the full exact check.
            let mut x = [0i64; 8];
            let mut missed = 0u32;
            let mut planted = 0u32;
            brute(&mut x, 0, &mut |x| {
                let a = ZRootTwoPlusRootTwo::from_i64(x[0], x[1], x[2], x[3]);
                let b = ZRootTwoPlusRootTwo::from_i64(x[4], x[5], x[6], x[7]);
                if a.is_zero() && b.is_zero() {
                    return;
                }
                if GridCtx::divisible_by_sqrt2(&a, &b) {
                    return;
                }
                if ctx.exact_accept(&a, &b, k).is_some() {
                    planted += 1;
                    if !got.contains(x) {
                        missed += 1;
                    }
                }
            });
            assert_eq!(missed, 0, "k={k}: enumerator missed {missed} of {planted}");
            assert!(planted > 0, "k={k}: brute force found nothing — test broken");
        }
    }

    fn brute(x: &mut [i64; 8], i: usize, f: &mut dyn FnMut(&[i64; 8])) {
        if i == 8 {
            f(x);
            return;
        }
        for v in -3..=3i64 {
            x[i] = v;
            brute(x, i + 1, f);
        }
    }

    /// Level counts behave sanely: empty at tiny k, populated later, and
    /// every emitted candidate has totally-non-negative ξ.
    #[test]
    fn test_level_scan_sane() {
        let prec = 96;
        let theta = MpFloat::with_val(prec, 1.1_f64);
        let ctx = GridCtx::new(&theta, 0.2, prec);
        let mut total = 0u32;
        for k in 0..=8u32 {
            let mut per_level = 0u32;
            ctx.visit_level(k, &mut |cand| {
                assert!(cand.xi.is_totally_nonneg());
                total += 1;
                per_level += 1;
                per_level >= 50 // cap the per-level walk
            });
        }
        assert!(total > 0, "no candidates found up to k=8");
    }
}

