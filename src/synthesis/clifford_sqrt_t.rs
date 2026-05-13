//! Clifford+√T synthesis backend over Z[ζ_16].
//!
//! [`SynthesizerQ`] is one of two backends behind the unified user-facing
//! [`crate::synthesis::Synthesizer`]; the other is
//! [`crate::synthesis::clifford_t::SynthesizerT`] (Clifford+T, Z[ω]). Code
//! shouldn't construct `SynthesizerQ` directly — use `Synthesizer` with
//! `sqrt_t = true`. The struct stays public so the test suite can poke at
//! it (`pub` instead of `pub(crate)`).
//!
//! ## Backend (hybrid)
//!
//! For `k ≤ BRUTE_LIMIT` (=4): brute-force enumeration via
//! [`crate::synthesis::search_zeta::phase1_brute`] — cheap exact-find
//! for small Clifford+√T targets.
//!
//! For `k > BRUTE_LIMIT`: 16D L²-LLL + Schnorr-Euchner via
//! [`crate::synthesis::lenstra_zeta::phase1`] with adaptive leaf budget
//! scaling exponentially in `k`. Reaches ε ≲ 1e-5 at k ≈ 30.
//!
//! ## Reconstruction
//!
//! Single det-phase reconstruction: `d = det_phase_of(target)` chosen
//! once, then `solution_to_u2q_d(sol, k, d)` per candidate. Column-1
//! direction extracted directly from the target (no `/√det`
//! normalization — unlike 8D's `unitary_to_uv` — because our `d` parameter
//! in the reconstruction already absorbs the det-phase mismatch).

use crate::matrix::u2::U2Q;
use crate::rings::ZZeta;
use crate::rings::types::Int;
use crate::synthesis::decomposer::BlochDecomposer;
use crate::synthesis::distance::{diamond_distance_float, Mat2};
use crate::synthesis::lenstra_zeta::{phase1_with_stop, IntScratch16};
use crate::synthesis::search_zeta::{phase1_brute, uv_to_xy_zeta};
use num_complex::Complex64;
use std::f64::consts::PI;
use std::sync::atomic::AtomicBool;

// ─── Solution → U2Q reconstruction (Z[ζ_16] analog of solution_to_u2t) ───────

/// Build `U2Q` from a 16-element solution and denominator exponent.
///
/// Convention: `sol = [u_1.a, …, u_1.h, u_2.a, …, u_2.h]` with
/// `U = [[u_1, −u_2*], [u_2, u_1*]] / √(2^k)` (SU(2) form, det = 1).
pub fn solution_to_u2q(sol: &[i64; 16], k: u32) -> U2Q {
    solution_to_u2q_d(sol, k, 0)
}

/// `ζ_16^d` as a `ZZeta` element, for `d` in `0..16`. `ζ_16^8 = −1`, so
/// `ζ_16^(d+8) = −ζ_16^d`.
fn zeta_16_pow(d: u32) -> ZZeta {
    let d = d % 16;
    if d < 8 {
        let mut c = [0i32; 8];
        c[d as usize] = 1;
        ZZeta::from_i32(c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7])
    } else {
        -zeta_16_pow(d - 8)
    }
}

/// Build a Clifford+√T `U2Q` from a 16-element solution at lde `k` with
/// **det-phase parameter** `det_phase` in `0..16`.
///
/// The reconstructed `U2Q` has determinant `ζ_16^det_phase`. Convention:
///
/// ```text
/// U = [[u_1, ζ_16^d · (−u_2*)], [u_2, ζ_16^d · u_1*]] / √(2^k)
/// ```
///
/// For `d = 0` this matches [`solution_to_u2q`] (SU(2) form). For `d ≠ 0`
/// the second column is rotated by `ζ_16^d`, making `U` reach Clifford+√T
/// products with non-unit determinant (e.g. circuits containing an odd
/// number of Q gates).
pub fn solution_to_u2q_d(sol: &[i64; 16], k: u32, det_phase: u32) -> U2Q {
    let mk = |s: &[i64]| ZZeta::new(
        Int::from_i64(s[0]), Int::from_i64(s[1]), Int::from_i64(s[2]), Int::from_i64(s[3]),
        Int::from_i64(s[4]), Int::from_i64(s[5]), Int::from_i64(s[6]), Int::from_i64(s[7]),
    );
    let u1 = mk(&sol[0..8]);
    let u2 = mk(&sol[8..16]);
    let phase = zeta_16_pow(det_phase);
    U2Q::new(u1, phase * (-u2.conj()), u2, phase * u1.conj(), k)
}

/// Determine the det-phase `d ∈ {0..15}` of a target matrix V — the
/// integer such that `ζ_16^d` is closest to `det(V)` on the unit circle.
///
/// Z[ζ_16] analog of [`super::synthesizer`]'s `det_zeta_parity` (which
/// returns just a parity bit for Z[ω]).
pub fn det_phase_of(target: &Mat2) -> u32 {
    let det = target[0][0] * target[1][1] - target[0][1] * target[1][0];
    let arg = det.arg();
    let d_float = arg * 16.0 / (2.0 * PI);
    let d_int = d_float.round() as i32;
    (((d_int % 16) + 16) % 16) as u32
}


/// Result of a synthesis call: the gate string, its lde, and the diamond
/// distance achieved.
///
/// Field shape matches `crate::synthesis::clifford_t::SynthResultT` so
/// callers can swap implementations transparently after the merge.
#[derive(Debug, Clone)]
pub struct SynthResultQ {
    /// Clifford+√T gate string in the alphabet `{H, S, T, Q, X, Y, Z}`
    /// (leftmost gate = first applied; matching the rest of cyclosynth's
    /// composition convention). `None` if the gate string couldn't be
    /// extracted.
    pub gates: Option<String>,
    /// Denominator exponent of the synthesized unitary.
    pub lde: u32,
    /// Diamond distance to the target.
    pub distance: f64,
}

/// Clifford+√T synthesizer over `Z[ζ_16]`.
///
/// Field names match `crate::synthesis::clifford_t::SynthesizerT`'s
/// for the future merge. The brute-force backend caps `max_lde` at a
/// small value; LLL+SE (Phase 5b M3+) will lift this.
pub struct SynthesizerQ {
    /// Approximation precision in diamond distance.
    pub epsilon: f64,
    /// Maximum lde to search before giving up. Brute-force backend
    /// limits this to ~4 in practice.
    pub max_lde: u32,
    /// Minimum lde to start searching from.
    pub min_lde: u32,
}

/// k cutoff: brute-force handles `k ≤ BRUTE_LIMIT`, the 16D LLL+SE
/// backend handles larger k.
///
/// **Was 4** until profiling found that `phase1_brute(4)` (~5·10⁸ shell
/// points, ~10 s) was wasted on every approximation target at moderate-
/// or-deep ε, since the actual answer lives in the lattice regime at k≥5.
/// At BRUTE_LIMIT=3, brute tops out at ~10⁷ shell points (~100 ms) and
/// the lattice walker handles k=4 efficiently when needed.
const BRUTE_LIMIT: u32 = 3;

/// Estimate the smallest lde at which a generic SU(2) target is reachable
/// within ε. Empirical from the ε-1e-3 / ε-1e-4 / ε-1e-5 benches: lde lands
/// at roughly `⌈-log₂(ε)⌉ - 3`, with a per-target jitter of ±2. We start
/// the lattice search 2 below the estimate so that easy targets land
/// without an extra full-shell sweep, and harder ones advance into deeper k.
fn lattice_lde_estimate(epsilon: f64) -> u32 {
    if !(epsilon > 0.0 && epsilon < 1.0) {
        return 0;
    }
    let raw = (-epsilon.log2()).ceil() as i32 - 3;
    raw.max(0) as u32
}

/// Two-pass leaf-budget strategy (mirrors 8D `dc_search`):
///   - **Pass 1** is the aggressive cap: small enough that doomed k
///     (where no sol exists in this ε regime) bail quickly. At each k
///     in the lattice range we try Pass 1 first.
///   - If Pass 1 finds a sol, return.
///   - If Pass 1 exhausts the SE region (no budget hit, no sol), the
///     search was complete at this k — advance to k+1.
///   - If Pass 1 budget-hits without finding a sol, mark this k for
///     Pass 2 retry and continue to k+1.
///   - **Pass 2** is the unbounded cap: only run on k's that Pass 1
///     budget-hit, after the Pass-1 sweep finishes without finding a
///     sol elsewhere. Guarantees no completeness loss.
///
/// Empirically: at ε=1e-5 target_01 lands at lde=13 but k=12 has no
/// sol — single-pass with 4G budget burns ~30 s on k=12 before
/// advancing. Pass 1 at 100 M lets k=12 bail in ~7 s, k=13 finds
/// quickly.
const PASS1_CAP: u64 = 100_000_000;
const PASS2_CAP: u64 = 4_000_000_000;

/// Column-1 of `target` as a 4-element real vector
/// `(Re V_{00}, Im V_{00}, Re V_{10}, Im V_{10})`. Used as the SU(2)-style
/// alignment direction `v` for the lattice search.
///
/// **Differs from 8D's `unitary_to_uv`**: that function divides by `√det`
/// to project to SU(2) because `solution_to_u2t` produces a fixed SU(2)
/// form. Here we leave the column unprojected and absorb the det-phase
/// mismatch via [`solution_to_u2q_d`]'s `d` parameter (set to
/// [`det_phase_of`]`(target)` at the call site). Column 1 of any 2×2
/// unitary is unit-norm by construction, so no further normalization is
/// needed.
pub fn unitary_to_uv_zeta(target: &Mat2) -> [f64; 4] {
    [target[0][0].re, target[0][0].im, target[1][0].re, target[1][0].im]
}

impl SynthesizerQ {
    /// Create a synthesizer with the given precision and sensible defaults.
    ///
    /// `min_lde = 0`: start from the trivial shell so exact small-T
    /// Clifford+√T targets (e.g. Q itself) are found immediately.
    /// `max_lde = 30`: high enough to reach ε ≲ 1e-5 via the LLL backend.
    /// Override via [`with_max_lde`] for a tighter (faster) ceiling, e.g.
    /// `with_max_lde(4)` to stay in the brute regime.
    pub fn new(epsilon: f64) -> Self {
        Self { epsilon, min_lde: 0, max_lde: 30 }
    }

    pub fn with_max_lde(mut self, max_lde: u32) -> Self {
        self.max_lde = max_lde;
        self
    }

    pub fn with_min_lde(mut self, min_lde: u32) -> Self {
        self.min_lde = min_lde;
        self
    }

    /// Find a minimum-lde Clifford+√T circuit approximating `target`.
    ///
    /// Returns `None` if no circuit within `max_lde` achieves diamond
    /// distance < `epsilon`. Returns the FIRST candidate found at the
    /// smallest k that works (not necessarily √T-count optimal).
    ///
    /// **Backend**: hybrid — brute-force `phase1_brute` for `k ≤ BRUTE_LIMIT`,
    /// 16D L²-LLL + Schnorr-Euchner `phase1` for larger k.
    pub fn synthesize(&self, target: Mat2) -> Option<SynthResultQ> {
        use crate::synthesis::diag;
        let trace = diag::trace_enabled();
        if trace {
            diag::reset_all();
        }

        let d = det_phase_of(&target);
        let v = unitary_to_uv_zeta(&target);

        // Lattice scratch is allocated lazily on first lattice call.
        let mut scratch: Option<Box<IntScratch16>> = None;

        // k schedule:
        //   - Brute regime [min_lde .. BRUTE_LIMIT]: cheap exact-find for
        //     small Clifford+√T targets (Q, T, H, …).
        //   - Lattice regime: skip k that the empirical ε→lde fit
        //     `lde ≈ ⌈-log₂(ε)⌉ - 3` says are too small. Start 2 below the
        //     estimate to absorb per-target jitter, then advance to
        //     `max_lde` with two-pass budgeting.
        let lattice_start = lattice_lde_estimate(self.epsilon)
            .saturating_sub(2)
            .max(BRUTE_LIMIT + 1)
            .max(self.min_lde);

        // Lattice search with early-exit. The `should_stop` predicate
        // runs only on leaves that pass the integer-exact filter (norm
        // shell, bilinear, alignment) — typically a handful per call —
        // and short-circuits the walker once we find a candidate whose
        // diamond distance to `target` is already below ε. At deep ε this
        // can cut the walk by orders of magnitude.
        let epsilon = self.epsilon;
        let try_lattice_k = |k: u32,
                             budget: u64,
                             scratch: &mut Option<Box<IntScratch16>>|
         -> (Vec<[i64; 16]>, bool) {
            let s = scratch
                .get_or_insert_with(|| Box::new(IntScratch16::new(epsilon)));
            let y = uv_to_xy_zeta(v, k);
            let budget_hit = AtomicBool::new(false);
            let should_stop = |x: &[i64; 16]| -> bool {
                let cand = solution_to_u2q_d(x, k, d);
                diamond_distance_float(&cand.to_float(), &target) < epsilon
            };
            let sols = phase1_with_stop(
                s.as_mut(), &y, k, epsilon, budget, &budget_hit, should_stop,
            );
            (sols, budget_hit.load(std::sync::atomic::Ordering::Relaxed))
        };

        let check_sols = |sols: &[[i64; 16]], k: u32| -> Option<SynthResultQ> {
            for sol in sols {
                let cand: U2Q = solution_to_u2q_d(sol, k, d);
                let dist = diamond_distance_float(&cand.to_float(), &target);
                if dist < self.epsilon {
                    let gates = BlochDecomposer.decompose(&cand);
                    return Some(SynthResultQ {
                        gates: Some(gates),
                        lde: k,
                        distance: dist,
                    });
                }
            }
            None
        };

        // Brute regime: iterate every k for exact small-T Clifford+√T finds.
        for k in self.min_lde..=BRUTE_LIMIT.min(self.max_lde) {
            let sols = phase1_brute(k);
            if let Some(r) = check_sols(&sols, k) {
                if trace {
                    diag::dump_zeta(&diag::snapshot(),
                        &format!("synthesize ε={:.0e} k={k}", self.epsilon));
                }
                return Some(r);
            }
        }

        // Lattice regime, Pass 1: aggressive budget cap. k's that hit the
        // budget without finding a sol get queued for Pass 2.
        let mut pass2_queue: Vec<u32> = Vec::new();
        for k in lattice_start..=self.max_lde {
            let (sols, budget_was_hit) = try_lattice_k(k, PASS1_CAP, &mut scratch);
            if let Some(r) = check_sols(&sols, k) {
                if trace {
                    diag::dump_zeta(&diag::snapshot(),
                        &format!("synthesize ε={:.0e} k={k} (pass1)", self.epsilon));
                }
                return Some(r);
            }
            if budget_was_hit {
                pass2_queue.push(k);
            }
        }

        // Lattice regime, Pass 2: only retry the k's that Pass 1
        // budget-hit. Guarantees no completeness loss vs single-pass-at-
        // PASS2_CAP, while skipping k's where Pass 1 was already
        // exhaustive.
        for k in pass2_queue {
            let (sols, _) = try_lattice_k(k, PASS2_CAP, &mut scratch);
            if let Some(r) = check_sols(&sols, k) {
                if trace {
                    diag::dump_zeta(&diag::snapshot(),
                        &format!("synthesize ε={:.0e} k={k} (pass2)", self.epsilon));
                }
                return Some(r);
            }
        }

        if trace {
            diag::dump_zeta(&diag::snapshot(),
                &format!("synthesize ε={:.0e} (no sol)", self.epsilon));
        }
        None
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;
    use std::f64::consts::PI;

    fn complex_target(matrix: [[Complex64; 2]; 2]) -> Mat2 {
        matrix
    }

    #[test]
    fn synthesize_identity_at_k_0() {
        let one = Complex64::new(1.0, 0.0);
        let zero = Complex64::new(0.0, 0.0);
        let target = complex_target([[one, zero], [zero, one]]);
        let synth = SynthesizerQ::new(1e-9);
        let result = synth.synthesize(target).expect("identity should synthesize");
        assert_eq!(result.lde, 0, "identity should be at k=0");
        assert!(result.distance < 1e-9);
    }

    #[test]
    fn synthesize_q_gate() {
        let q = U2Q::q();
        let target = q.to_float();
        let synth = SynthesizerQ::new(1e-9);
        let result = synth.synthesize(target).expect("Q should synthesize");
        assert_eq!(result.lde, 0, "Q should be found at k=0");
        assert!(result.distance < 1e-9);
        // The synthesized gate string, when applied, should give back Q.
        assert!(result.gates.is_some());
    }

    #[test]
    fn synthesize_t_gate() {
        let t = U2Q::t();
        let target = t.to_float();
        let synth = SynthesizerQ::new(1e-9);
        let result = synth.synthesize(target).expect("T should synthesize");
        assert_eq!(result.lde, 0, "T should be found at k=0");
        assert!(result.distance < 1e-9);
    }

    #[test]
    fn synthesize_hqh() {
        let hqh: U2Q = U2Q::h() * U2Q::q() * U2Q::h();
        let target = hqh.to_float();
        let synth = SynthesizerQ::new(1e-9);
        let result = synth.synthesize(target).expect("HQH should synthesize");
        // HQH has k=2 (1 from each H).
        assert_eq!(result.lde, 2);
        assert!(result.distance < 1e-9);
    }

    #[test]
    fn synthesize_qhq() {
        let qhq: U2Q = U2Q::q() * U2Q::h() * U2Q::q();
        let target = qhq.to_float();
        let synth = SynthesizerQ::new(1e-9);
        let result = synth.synthesize(target).expect("QHQ should synthesize");
        assert_eq!(result.lde, 1);
        assert!(result.distance < 1e-9);
    }

    #[test]
    fn synthesize_h_gate() {
        // H has k=1 (one H gate). Should be found at k=1.
        let h = U2Q::h();
        let target = h.to_float();
        let synth = SynthesizerQ::new(1e-9);
        let result = synth.synthesize(target).expect("H should synthesize");
        assert_eq!(result.lde, 1);
        assert!(result.distance < 1e-9);
    }

    #[test]
    fn synthesize_returns_none_when_unreachable() {
        // Target Rx(π/16) — angle isn't a multiple of π/8, so the closest
        // Clifford+√T circuit at any small k is bounded away from it. With
        // ε=1e-9 (tight) and max_lde=2 (so the test stays under a second),
        // should return None.
        let theta = PI / 16.0;
        let c = (theta / 2.0).cos();
        let s = (theta / 2.0).sin();
        let i = Complex64::new(0.0, 1.0);
        let target: Mat2 = [
            [Complex64::new(c, 0.0), -i * s],
            [-i * s, Complex64::new(c, 0.0)],
        ];
        let synth = SynthesizerQ::new(1e-9).with_max_lde(2);
        let result = synth.synthesize(target);
        assert!(result.is_none(),
            "Rx(π/16) should not be reachable in Clifford+√T at k≤2 with ε=1e-9");
    }

    #[test]
    fn synthesize_approximation_with_loose_epsilon() {
        // For Rx(π/16) at LOOSE ε, the synthesizer should find a closeby
        // approximation at small k. Tests the "approximate synthesis" path.
        let theta = PI / 16.0;
        let c = (theta / 2.0).cos();
        let s = (theta / 2.0).sin();
        let i = Complex64::new(0.0, 1.0);
        let target: Mat2 = [
            [Complex64::new(c, 0.0), -i * s],
            [-i * s, Complex64::new(c, 0.0)],
        ];
        let synth = SynthesizerQ::new(0.3).with_max_lde(2);  // very loose
        let result = synth.synthesize(target);
        assert!(result.is_some(), "loose ε should find an approximation");
        let r = result.unwrap();
        assert!(r.distance < 0.3);
    }

    #[test]
    fn synthesized_gate_string_roundtrip() {
        // For each of several Clifford+√T targets, the gate string from
        // the synthesizer should reconstruct (via gates_to_u2q) to a
        // U2Q close to the target.
        use crate::matrix::u2::U2Q;
        let targets: Vec<U2Q> = vec![
            U2Q::q(),
            U2Q::t(),
            U2Q::q() * U2Q::q(),  // = T
            U2Q::h() * U2Q::q() * U2Q::h(),
            U2Q::q() * U2Q::h() * U2Q::q(),
        ];
        let synth = SynthesizerQ::new(1e-9);
        for u in targets {
            let target = u.to_float();
            let result = synth.synthesize(target).expect("should synthesize");
            let gates = result.gates.expect("should have gate string");
            // Reconstruct via gates_to_u2q.
            let mut rebuilt = U2Q::eye();
            for c in gates.chars() {
                rebuilt = rebuilt * match c {
                    'H' => U2Q::h(),
                    'S' => U2Q::s(),
                    'T' => U2Q::t(),
                    'Q' => U2Q::q(),
                    'X' => U2Q::x(),
                    'Y' => U2Q::y(),
                    'Z' => U2Q::z(),
                    _ => panic!("unexpected gate {c}"),
                };
            }
            let dist = diamond_distance_float(&rebuilt.to_float(), &target);
            assert!(dist < 1e-7,
                "round-trip dist for gate string \"{gates}\" = {dist:.3e}");
        }
    }

    /// End-to-end deep-ε test: Rz(0.3) at ε=1e-3. Behind `#[ignore]` because
    /// it can take minutes — the lattice search at k=10 needs ~1G SE leaves.
    /// Run with `cargo test --release --lib synthesize_rz_eps_1e_3 --
    /// --ignored --nocapture`.
    #[test]
    #[ignore]
    fn synthesize_rz_eps_1e_3() {
        let theta = 0.3_f64;
        let target: Mat2 = [
            [Complex64::from_polar(1.0, -theta / 2.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::from_polar(1.0, theta / 2.0)],
        ];
        let synth = SynthesizerQ::new(1e-3).with_max_lde(15);
        let t0 = std::time::Instant::now();
        let result = synth.synthesize(target).expect("Rz(0.3) at ε=1e-3 should land");
        eprintln!(
            "Rz(0.3) at ε=1e-3: lde={} dist={:.3e} t={:?}",
            result.lde, result.distance, t0.elapsed()
        );
        assert!(result.distance < 1e-3);
        // Upper bound from 8D Clifford+T: lde=28. Z[ζ_16] should land much
        // smaller (~10) since `T = QQ` doubles the effective denominator
        // factor in the 8D path.
        assert!(result.lde <= 14,
            "expected lde ≤ 14 (8D Clifford+T is 28); got {}", result.lde);
    }

    #[test]
    fn synthesize_rz_via_lattice_backend() {
        // Rz(0.3) at ε=0.05 is unreachable at k ≤ 4 (brute regime), so
        // forcing min_lde > BRUTE_LIMIT exercises the LLL+SE lattice path.
        let theta = 0.3_f64;
        let target: Mat2 = [
            [Complex64::from_polar(1.0, -theta / 2.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::from_polar(1.0, theta / 2.0)],
        ];
        let synth = SynthesizerQ::new(0.05)
            .with_min_lde(BRUTE_LIMIT + 1)
            .with_max_lde(12);
        let result = synth.synthesize(target).expect("Rz(0.3) at ε=0.05 should land");
        assert!(result.lde > BRUTE_LIMIT,
            "expected lattice backend (k > {BRUTE_LIMIT}), got k={}", result.lde);
        assert!(result.distance < 0.05,
            "diamond distance {:.3e} exceeds ε=0.05", result.distance);
        assert!(result.gates.is_some());
    }
}
