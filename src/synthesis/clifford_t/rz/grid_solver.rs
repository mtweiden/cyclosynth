//! Grid-point enumeration for the z-rotation synthesis: the convex
//! target sets and the one- and two-dimensional grid problems (ODGP /
//! TDGP in RS arXiv:1403.2975 §5): stream all u ∈ Z[ω]/√2^k whose
//! principal embedding lies in the ε-cap and whose √2-conjugate lies in
//! the unit disk. Visitor-style streaming throughout — levels hold
//! astronomically many points and callers stop at the first few.

use rug::Integer;

use crate::rings::types::MpFloat;

use super::geometry::{mpf, solve_quadratic, Ellipse, GridOp, Interval, Rectangle};
use crate::rings::dyadic::{sqrt2_pow_f, DOmega, DRootTwo};
use crate::rings::real::ZRootTwo;
use crate::rings::types::cached_sqrt2;

pub(crate) trait ConvexSet {
    fn ellipse(&self) -> &Ellipse;
    fn inside(&self, u: &DOmega, prec: u32) -> bool;
    /// Parameter range t where u0 + t·v stays inside, or None.
    fn intersect(&self, u0: &DOmega, v: &DOmega, prec: u32) -> Option<(MpFloat, MpFloat)>;
}

pub(crate) struct EpsilonRegion {
    scale: ZRootTwo,
    d: MpFloat,
    z_x: MpFloat,
    z_y: MpFloat,
    ellipse: Ellipse,
}

impl EpsilonRegion {
    /// Cap of half-angle-metric ε around e^{−iθ/2}, scaled by `scale`
    /// (the up-to-phase branch uses scale = 2+√2 and disk scale 2−√2).
    pub fn new(theta: &MpFloat, epsilon: &MpFloat, scale: ZRootTwo, prec: u32) -> Self {
        let scale_f = scale.to_mpfloat(prec);
        let one = MpFloat::with_val(prec, 1.0);
        let eps_sq_4 = mpf!(prec, epsilon * epsilon) / 4u32;
        let d = mpf!(prec, &one - &eps_sq_4).sqrt() * scale_f.clone().sqrt();
        let half_theta = mpf!(prec, theta / 2u32) * -1i32;
        let z_x = half_theta.clone().cos();
        let z_y = half_theta.sin();
        // D = R(z) · diag(64/ε⁴, 4/ε²)/scale · R(z)ᵀ, p = d·z.
        let inv_eps = mpf!(prec, &one / epsilon);
        let e2 = mpf!(prec, &inv_eps * &inv_eps);
        let e4 = mpf!(prec, &e2 * &e2);
        let d1 = mpf!(prec, &e4 * 64u32) / &scale_f;
        let d2 = mpf!(prec, &e2 * 4u32) / &scale_f;
        // Conjugate the diagonal by the rotation [[zx, −zy], [zy, zx]].
        let a = mpf!(prec, &z_x * &z_x) * &d1 + mpf!(prec, &z_y * &z_y) * &d2;
        let b = mpf!(prec, &z_x * &z_y) * mpf!(prec, &d1 - &d2);
        let dd = mpf!(prec, &z_y * &z_y) * &d1 + mpf!(prec, &z_x * &z_x) * &d2;
        let px = mpf!(prec, &d * &z_x);
        let py = mpf!(prec, &d * &z_y);
        let ellipse = Ellipse { a, b, d: dd, px, py };
        Self { scale, d, z_x, z_y, ellipse }
    }
}

impl ConvexSet for EpsilonRegion {
    fn ellipse(&self) -> &Ellipse {
        &self.ellipse
    }

    fn inside(&self, u: &DOmega, prec: u32) -> bool {
        let uu = DRootTwo::from_domega(&u.conj().mul(u));
        let cos_sim =
            mpf!(prec, &self.z_x * &u.real(prec)) + mpf!(prec, &self.z_y * &u.imag(prec));
        uu.le_zroottwo(&self.scale) && cos_sim >= self.d
    }

    fn intersect(&self, u0: &DOmega, v: &DOmega, prec: u32) -> Option<(MpFloat, MpFloat)> {
        let a = v.conj().mul(v).real(prec);
        let b = mpf!(prec, &v.conj().mul(u0).real(prec) * 2u32);
        let c = u0.conj().mul(u0).real(prec) - self.scale.to_mpfloat(prec);
        let vz = mpf!(prec, &self.z_x * &v.real(prec)) + mpf!(prec, &self.z_y * &v.imag(prec));
        let rhs = mpf!(prec, &self.d - &mpf!(prec, &self.z_x * &u0.real(prec)))
            - mpf!(prec, &self.z_y * &u0.imag(prec));
        let (t0, t1) = solve_quadratic(&a, &b, &c)?;
        if vz.is_sign_positive() && !vz.is_zero() {
            let t2 = mpf!(prec, &rhs / &vz);
            Some(if t0 > t2 { (t0, t1) } else { (t2, t1) })
        } else if vz.is_sign_negative() && !vz.is_zero() {
            let t2 = mpf!(prec, &rhs / &vz);
            Some(if t1 < t2 { (t0, t1) } else { (t0, t2) })
        } else if rhs.is_sign_negative() || rhs.is_zero() {
            Some((t0, t1))
        } else {
            None
        }
    }
}

pub(crate) struct UnitDisk {
    scale: ZRootTwo,
    ellipse: Ellipse,
}

impl UnitDisk {
    pub fn new(scale: ZRootTwo, prec: u32) -> Self {
        let one = MpFloat::with_val(prec, 1.0);
        let s_inv = mpf!(prec, &one / &scale.to_mpfloat(prec));
        let zero = MpFloat::with_val(prec, 0.0);
        let ellipse = Ellipse {
            a: s_inv.clone(),
            b: zero.clone(),
            d: s_inv,
            px: zero.clone(),
            py: zero,
        };
        Self { scale, ellipse }
    }
}

impl ConvexSet for UnitDisk {
    fn ellipse(&self) -> &Ellipse {
        &self.ellipse
    }

    fn inside(&self, u: &DOmega, _prec: u32) -> bool {
        DRootTwo::from_domega(&u.conj().mul(u)).le_zroottwo(&self.scale)
    }

    fn intersect(&self, u0: &DOmega, v: &DOmega, prec: u32) -> Option<(MpFloat, MpFloat)> {
        let a = v.conj().mul(v).real(prec);
        let b = mpf!(prec, &v.conj().mul(u0).real(prec) * 2u32);
        let c = u0.conj().mul(u0).real(prec) - self.scale.to_mpfloat(prec);
        solve_quadratic(&a, &b, &c)
    }
}


// ─── One-dimensional grid problems ───────────────────────────────────────────



/// floor(x) as a rug Integer (x finite).
fn floor_int(x: &MpFloat) -> Integer {
    x.clone().floor().to_integer().expect("finite float")
}

/// ceil(x) as a rug Integer.
fn ceil_int(x: &MpFloat) -> Integer {
    x.clone().ceil().to_integer().expect("finite float")
}

/// (n, _) with λⁿ ≤ x < λⁿ⁺¹ for x > 0; only n is consumed by the
/// ODGP.
fn floorlog_lambda(x: &MpFloat, prec: u32) -> i64 {
    let lambda = MpFloat::with_val(prec, 1.0) + cached_sqrt2(prec);
    let ln_l = lambda.clone().ln();
    let mut n = mpf!(prec, x.clone().ln() / &ln_l).floor().to_f64() as i64;
    // λⁿ by binary exponentiation, then guard the float rounding with
    // incremental steps: enforce λⁿ ≤ x < λⁿ⁺¹ exactly in MPFR.
    let mut p = {
        let mut acc = MpFloat::with_val(prec, 1.0);
        let mut base = lambda.clone();
        let neg = n < 0;
        let mut e = n.unsigned_abs();
        while e > 0 {
            if e & 1 == 1 {
                acc *= &base;
            }
            base = mpf!(prec, &base * &base);
            e >>= 1;
        }
        if neg {
            acc.recip()
        } else {
            acc
        }
    };
    while p > *x {
        n -= 1;
        p /= &lambda;
    }
    loop {
        let next = mpf!(prec, &p * &lambda);
        if next <= *x {
            n += 1;
            p = next;
        } else {
            break;
        }
    }
    n
}

/// Stream solutions into `visit` (return `true` to stop); returns whether
/// the enumeration was stopped. Streaming matters: callers like the TDGP
/// anchor need only the FIRST solution of an enumeration that can hold
/// 10⁷+ points at deep k; materializing them is ~10⁴× the streamed cost.
fn visit_odgp_internal(
    i: &Interval,
    j: &Interval,
    prec: u32,
    visit: &mut dyn FnMut(ZRootTwo) -> bool,
) -> bool {
    let zero = MpFloat::with_val(prec, 0.0);
    if i.width() < zero || j.width() < zero {
        return false;
    }
    if i.width() > zero && j.width() <= zero {
        // Swap roles; solutions conjugate back.
        return visit_odgp_internal(j, i, prec, &mut |beta| visit(beta.conj_sq2()));
    }
    let n = if j.width() <= zero {
        0
    } else {
        floorlog_lambda(&j.width(), prec)
    };
    if n == 0 {
        let sqrt2 = cached_sqrt2(prec);
        let a_min = ceil_int(&(mpf!(prec, &i.l + &j.l) / 2u32));
        let a_max = floor_int(&(mpf!(prec, &i.r + &j.r) / 2u32));
        let mut a = a_min;
        while a <= a_max {
            let af = MpFloat::with_val(prec, &a);
            let b_min = ceil_int(&(mpf!(prec, &af - &j.r) * &sqrt2 / 2u32));
            let b_max = floor_int(&(mpf!(prec, &af - &j.l) * &sqrt2 / 2u32));
            let mut b = b_min;
            while b <= b_max {
                if visit(ZRootTwo::new(a.clone(), b.clone())) {
                    return true;
                }
                b += 1u32;
            }
            a += 1u32;
        }
        false
    } else {
        let lambda = ZRootTwo::lambda();
        let lambda_n = lambda.pow(n);
        let lambda_inv_n = lambda.pow(-n);
        let lambda_conj_n = lambda.conj_sq2().pow(n);
        let i2 = i.mul_f(&lambda_n.to_mpfloat(prec));
        let j2 = j.mul_f(&lambda_conj_n.to_mpfloat(prec));
        visit_odgp_internal(&i2, &j2, prec, &mut |beta| visit(beta.mul(&lambda_inv_n)))
    }
}

/// Stream ODGP solutions (offset + exact filter applied) into `visit`.
fn visit_odgp(
    i: &Interval,
    j: &Interval,
    prec: u32,
    visit: &mut dyn FnMut(ZRootTwo) -> bool,
) {
    let zero = MpFloat::with_val(prec, 0.0);
    if i.width() < zero || j.width() < zero {
        return;
    }
    let sqrt2 = cached_sqrt2(prec);
    let a = floor_int(&(mpf!(prec, &i.l + &j.l) / 2u32));
    let b = floor_int(&(mpf!(prec, &i.l - &j.l) * &sqrt2 / 4u32));
    let alpha = ZRootTwo::new(a, b);
    let alpha_f = alpha.to_mpfloat(prec);
    let alpha_conj_f = alpha.conj_sq2().to_mpfloat(prec);
    let i2 = i.sub_f(&alpha_f);
    let j2 = j.sub_f(&alpha_conj_f);
    visit_odgp_internal(&i2, &j2, prec, &mut |beta| {
        let beta = beta.add(&alpha);
        if i.within(&beta.to_mpfloat(prec)) && j.within(&beta.conj_sq2().to_mpfloat(prec)) {
            visit(beta)
        } else {
            false
        }
    });
}

pub(crate) fn solve_odgp(i: &Interval, j: &Interval, prec: u32) -> Vec<ZRootTwo> {
    let mut out = Vec::new();
    visit_odgp(i, j, prec, &mut |beta| {
        out.push(beta);
        false
    });
    out
}

fn solve_odgp_with_parity(
    i: &Interval,
    j: &Interval,
    beta: &ZRootTwo,
    prec: u32,
) -> Vec<ZRootTwo> {
    let p = i64::from(beta.parity());
    let pf = MpFloat::with_val(prec, p);
    let sqrt2 = cached_sqrt2(prec);
    let half_sqrt2 = mpf!(prec, &sqrt2 / 2u32);
    let neg_half_sqrt2 = mpf!(prec, -&half_sqrt2);
    let i2 = i.sub_f(&pf).mul_f(&half_sqrt2);
    let j2 = j.sub_f(&pf).mul_f(&neg_half_sqrt2);
    solve_odgp(&i2, &j2, prec)
        .into_iter()
        .map(|alpha| {
            alpha
                .mul(&ZRootTwo::from_i64(0, 1))
                .add(&ZRootTwo::from_i64(p, 0))
        })
        .collect()
}

pub(crate) fn solve_scaled_odgp(
    i: &Interval,
    j: &Interval,
    k: u32,
    prec: u32,
) -> Vec<DRootTwo> {
    let scale = sqrt2_pow_f(k, prec);
    let i2 = i.mul_f(&scale);
    let j2 = if k & 1 == 1 {
        j.neg().mul_f(&scale)
    } else {
        j.mul_f(&scale)
    };
    solve_odgp(&i2, &j2, prec)
        .into_iter()
        .map(|alpha| DRootTwo::new(alpha, k))
        .collect()
}

/// First solution only — the TDGP anchor `alpha0`. The full enumeration
/// can hold 10⁷+ points at deep k; any single element serves as anchor.
pub(crate) fn solve_scaled_odgp_first(
    i: &Interval,
    j: &Interval,
    k: u32,
    prec: u32,
) -> Option<DRootTwo> {
    let scale = sqrt2_pow_f(k, prec);
    let i2 = i.mul_f(&scale);
    let j2 = if k & 1 == 1 {
        j.neg().mul_f(&scale)
    } else {
        j.mul_f(&scale)
    };
    let mut first = None;
    visit_odgp(&i2, &j2, prec, &mut |alpha| {
        first = Some(DRootTwo::new(alpha, k));
        true
    });
    first
}

/// Streaming variant of the parity-refined scaled ODGP (`visit` returns
/// `true` to stop): the TDGP consumes the first few of a possibly-10⁴
/// stream at the solution level.
pub(crate) fn visit_scaled_odgp_with_parity(
    i: &Interval,
    j: &Interval,
    k: u32,
    beta: &DRootTwo,
    prec: u32,
    visit: &mut dyn FnMut(DRootTwo) -> bool,
) {
    if k == 0 {
        let beta0 = beta.renew_denomexp(0).alpha;
        for alpha in solve_odgp_with_parity(i, j, &beta0, prec) {
            if visit(DRootTwo::from_zroottwo(alpha)) {
                return;
            }
        }
    } else {
        let p = beta.parity_at_denomexp(k);
        let offset = if p == 0 {
            DRootTwo::from_int_i64(0)
        } else {
            DRootTwo::power_of_inv_sqrt2(k)
        };
        let off_f = offset.to_mpfloat(prec);
        let off_conj_f = offset.conj_sq2().to_mpfloat(prec);
        let i2 = i.sub_f(&off_f);
        let j2 = j.sub_f(&off_conj_f);
        let scale = sqrt2_pow_f(k - 1, prec);
        let i3 = i2.mul_f(&scale);
        let j3 = if (k - 1) & 1 == 1 {
            j2.neg().mul_f(&scale)
        } else {
            j2.mul_f(&scale)
        };
        visit_odgp(&i3, &j3, prec, &mut |alpha| {
            visit(DRootTwo::new(alpha, k - 1).add(&offset))
        });
    }
}


// ─── Two-dimensional grid problems ───────────────────────────────────────────


/// Stream the TDGP solutions at level `k` into `visit`, in enumeration
/// order; `visit` returns `true` to stop early — a level can hold 10⁴
/// candidates when only the first few are ever consumed.
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_tdgp_visit(
    set_a: &dyn ConvexSet,
    set_b: &dyn ConvexSet,
    op_g: &GridOp,
    bbox_a: &Rectangle,
    bbox_b: &Rectangle,
    k: u32,
    prec: u32,
    visit: &mut dyn FnMut(DOmega) -> bool,
) {
    let Some(op_g_inv) = op_g.inv() else {
        return;
    };

    let Some(alpha0) = solve_scaled_odgp_first(&bbox_a.ix, &bbox_b.ix, k + 1, prec) else {
        return;
    };

    let fatten_a = mpf!(prec, &bbox_a.iy.width() * &MpFloat::with_val(prec, 1e-4));
    let fatten_b = mpf!(prec, &bbox_b.iy.width() * &MpFloat::with_val(prec, 1e-4));
    let sol_y = solve_scaled_odgp(
        &bbox_a.iy.fatten(&fatten_a),
        &bbox_b.iy.fatten(&fatten_b),
        k + 1,
        prec,
    );

    let dx = DRootTwo::power_of_inv_sqrt2(k);
    // v is β-invariant: the x-step direction of the k-grid.
    let v = op_g_inv.apply_domega(&DOmega::from_droottwo_vector(
        &dx,
        &DRootTwo::from_int_i64(0),
        k,
    ));
    for beta in sol_y {
        let z0 = op_g_inv.apply_domega(&DOmega::from_droottwo_vector(&alpha0, &beta, k + 1));
        let Some(t_a) = set_a.intersect(&z0, &v, prec) else { continue };
        let Some(t_b) = set_b.intersect(&z0.conj_sq2(), &v.conj_sq2(), prec) else { continue };

        let parity = beta.sub(&alpha0).mul_by_sqrt2_power_renewing_denomexp(k);
        let int_a = Interval::new(t_a.0, t_a.1);
        let int_b = Interval::new(t_b.0, t_b.1);
        // Fatten by 10/max(10, 2^k · other width): guards against a
        // near-degenerate interval dropping boundary solutions.
        let pow2k = MpFloat::with_val(prec, 1.0) << k;
        let ten = MpFloat::with_val(prec, 10.0);
        let da_den = mpf!(prec, &pow2k * &int_b.width());
        let db_den = mpf!(prec, &pow2k * &int_a.width());
        let dt_a = mpf!(prec, &ten / &da_den.clone().max(&ten));
        let dt_b = mpf!(prec, &ten / &db_den.clone().max(&ten));
        let int_a = int_a.fatten(&dt_a);
        let int_b = int_b.fatten(&dt_b);

        let mut stopped = false;
        visit_scaled_odgp_with_parity(&int_a, &int_b, 1, &parity, prec, &mut |alpha_t| {
            let alpha = alpha_t.mul(&dx).add(&alpha0);
            let cand = DOmega::from_droottwo_vector(&alpha, &beta, k);
            let z = op_g_inv.apply_domega(&cand);
            if set_a.inside(&z, prec) && set_b.inside(&z.conj_sq2(), prec) && visit(z) {
                stopped = true;
                return true;
            }
            false
        });
        if stopped {
            return;
        }
    }
}
