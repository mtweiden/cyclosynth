//! Unified user-facing `Synthesizer` API.
//!
//! Wraps the two ring-specific synthesis backends —
//! [`SynthesizerT`](crate::synthesis::clifford_t::SynthesizerT) for
//! Clifford+T over Z[ω] and
//! [`SynthesizerQ`](crate::synthesis::clifford_sqrt_t::SynthesizerQ) for
//! Clifford+√T over Z[ζ] — behind a single struct. Pick the backend at
//! construction with the `sqrt_t: bool` flag (default false → Clifford+T).
//!
//! ## Why two backends behind one type
//!
//! The two flows use *different algorithms* (Z[ω]: 8D MA-prefix
//! divide-and-conquer; Z[ζ]: 16D LLL+SE with a brute-force small-k mode
//! and an FGKM-prefix divide-and-conquer mode for deep k), so they can't be
//! expressed cleanly as a single generic `Synthesizer<R: GateRing>`. This
//! wrapper gives users a single API while the internals keep their own
//! optimised code paths.

use crate::synthesis::angle::Angle;
use crate::synthesis::clifford_sqrt_t::rz::synthesize_rz_q;
use crate::synthesis::clifford_t::rz::synthesize_rz as synthesize_rz_native;
use crate::synthesis::distance::Mat2;
use crate::synthesis::clifford_t::SynthesizerT;
use crate::synthesis::clifford_sqrt_t::SynthesizerQ;

/// Result of a successful synthesis call. Same shape regardless of the
/// underlying gate set.
#[derive(Debug, Clone)]
pub struct SynthResult {
    /// Gate string (leftmost = first gate applied). Alphabet is
    /// `{H, S, T, X, Y, Z}` for Clifford+T and `{H, S, T, Q, X, Y, Z}` for
    /// Clifford+√T (`Q = √T`). `None` if extraction failed.
    pub gates: Option<String>,
    /// Denominator exponent of the synthesized unitary.
    pub lde: u32,
    /// Diamond distance from synthesized unitary to target. For
    /// [`Synthesizer::synthesize_zyz`] this is the sum of the verified
    /// per-rotation distances — a sound upper bound rather than the
    /// measured composite distance.
    pub distance: f64,
}

/// Single-qubit unitary synthesizer over either Clifford+T (Z[ω]) or
/// Clifford+√T (Z[ζ]).
///
/// ```rust,ignore
/// // Clifford+T (default).
/// let synth = Synthesizer::new(1e-3, false);
/// // Clifford+√T (denser gate set, generally fewer gates).
/// let synth = Synthesizer::new(1e-3, true);
/// let result = synth.synthesize_u3(theta, phi, lam);
/// ```
pub struct Synthesizer {
    inner: Backend,
    /// Route diagonal (β = 0) targets through the native Ross–Selinger
    /// path (`clifford_t::rz` for Clifford+T, `clifford_sqrt_t::rz` for
    /// Clifford+√T) instead of the lattice pipelines. On by default: it
    /// runs in milliseconds, reaches ε = 1e-48 on both gate sets, and
    /// covers near-Clifford angles whose solutions lie above the ε-tuned
    /// lde band (issue #2).
    native_rz: bool,
}


/// If `theta` is within `epsilon` of a diagonal gate power — k·π/4
/// (Clifford+T: powers of T) or k·π/8 (Clifford+√T: powers of √T) —
/// return that exact circuit at lde 0.
///
/// Catches both exactly-Clifford-looking angles (π/2 multiples: I, S,
/// Z, S†) and the exact T/√T powers between them, at any ε: the
/// distance to the snapped power is the phase-invariant trace formula
/// D² = q(8−q)/16, q = 4 − 4|cos(δ/2)|, evaluated in MPFR at the
/// angle's precision — no f64 saturation, so a `PiRatio` input lands on
/// the exact circuit even at the 1e-48 floor. Gate strings stay in the
/// diagonal alphabet ({"", T, S, ST, Z, ZT, S†, T†} · √T^{0,1}), which
/// the syllable cost model already prices minimally.
fn snap_rz_to_diag_power(theta: &crate::rings::MpFloat, epsilon: f64, sqrt_t: bool) -> Option<SynthResult> {
    use crate::rings::MpFloat;
    let prec = theta.prec();
    let pi = MpFloat::with_val(prec, rug::float::Constant::Pi);
    let m = if sqrt_t { 8u32 } else { 4 };
    let step = MpFloat::with_val(prec, &pi / m);
    let k = MpFloat::with_val(prec, theta / &step).round();
    let k_int = k.to_integer()?.to_i64()?;
    let delta = MpFloat::with_val(prec, theta - &MpFloat::with_val(prec, &k * &step));
    let cos_half = MpFloat::with_val(prec, &delta / 2u32).cos().abs();
    let q = MpFloat::with_val(prec, 4.0) - MpFloat::with_val(prec, &cos_half * 4u32);
    let d = (MpFloat::with_val(prec, &q * &(MpFloat::with_val(prec, 8.0) - &q)) / 16u32).sqrt();
    // NaN-safe accept: only a definite d < ε snaps.
    if d.partial_cmp(&epsilon) != Some(std::cmp::Ordering::Less) {
        return None;
    }
    const DIAG: [&str; 8] = ["", "T", "S", "ST", "Z", "ZT", "s", "t"];
    let gates = if sqrt_t {
        let steps = usize::try_from(k_int.rem_euclid(16)).expect("0..16");
        format!("{}{}", DIAG[(steps / 2) % 8], if steps % 2 == 1 { "Q" } else { "" })
    } else {
        let steps = usize::try_from(k_int.rem_euclid(8)).expect("0..8");
        DIAG[steps].to_string()
    };
    Some(SynthResult { gates: Some(gates), lde: 0, distance: d.to_f64() })
}

// A `Synthesizer` is created once per session and never held in bulk, so the
// T-vs-Q size gap is irrelevant; boxing would only fight the consuming
// `with_*` builder methods.
#[allow(clippy::large_enum_variant)]
enum Backend {
    T(SynthesizerT),
    Q(SynthesizerQ),
}

impl Synthesizer {
    /// Create a synthesizer with the given precision target and gate set.
    /// `sqrt_t = false` (the default in user code) selects Clifford+T;
    /// `true` selects Clifford+√T.
    pub fn new(epsilon: f64, sqrt_t: bool) -> Self {
        let inner = if sqrt_t {
            Backend::Q(SynthesizerQ::new(epsilon))
        } else {
            Backend::T(SynthesizerT::new(epsilon))
        };
        Self { inner, native_rz: true }
    }

    /// Disable the native Ross–Selinger route for diagonal targets and
    /// force the lattice pipeline (mainly for A/B testing).
    pub fn with_native_rz(mut self, on: bool) -> Self {
        self.native_rz = on;
        self
    }

    /// Override the maximum lde the search will probe.
    pub fn with_max_lde(mut self, max_lde: u32) -> Self {
        match &mut self.inner {
            Backend::T(s) => s.max_lde = max_lde,
            Backend::Q(s) => s.max_lde = max_lde,
        }
        self
    }

    /// Override the minimum lde the search will probe.
    pub fn with_min_lde(mut self, min_lde: u32) -> Self {
        match &mut self.inner {
            Backend::T(s) => s.min_lde = min_lde,
            Backend::Q(s) => s.min_lde = min_lde,
        }
        self
    }

    /// Clifford+√T only: enumerate all ε-close candidates at the found
    /// lde and return the one minimising the weighted gate cost (see
    /// [`SynthesizerQ::with_optimize_cost`]). Ignored for Clifford+T,
    /// where the search is already T-count-minimal by construction.
    pub fn with_optimize_cost(mut self, on: bool) -> Self {
        if let Backend::Q(s) = self.inner {
            self.inner = Backend::Q(s.with_optimize_cost(on));
        }
        self
    }

    /// Clifford+√T only: Q-gate weight in T units for the optimize-cost
    /// model (default 3). Quantized to the nearest half-unit (the model
    /// compares `2·T + round(2·weight)·Q`), so e.g. 3.6 → 3.5. Ignored for
    /// Clifford+T.
    #[cfg_attr(not(feature = "python"), allow(dead_code))] // consumed by the PySynthesizer binding
    pub(crate) fn with_q_cost(mut self, weight: f64) -> Self {
        if let Backend::Q(s) = self.inner {
            self.inner = Backend::Q(s.with_q_cost(weight));
        }
        self
    }

    /// Clifford+√T only: also search `window` lde levels above the first
    /// hit and return the global min-cost candidate (see
    /// [`SynthesizerQ::with_optimal_lde_window`]). Ignored for Clifford+T.
    #[cfg_attr(not(feature = "python"), allow(dead_code))] // consumed by the PySynthesizer binding
    pub(crate) fn with_optimal_lde_window(mut self, window: u32) -> Self {
        if let Backend::Q(s) = self.inner {
            self.inner = Backend::Q(s.with_optimal_lde_window(window));
        }
        self
    }

    /// Clifford+√T only: wall-clock budget in ms for the min-cost
    /// enumeration (see [`SynthesizerQ::with_optimal_deadline_ms`]);
    /// `None` removes the deadline. Ignored for Clifford+T.
    #[cfg_attr(not(feature = "python"), allow(dead_code))] // consumed by the PySynthesizer binding
    pub(crate) fn with_optimal_deadline_ms(mut self, ms: Option<u64>) -> Self {
        if let Backend::Q(s) = self.inner {
            self.inner = Backend::Q(s.with_optimal_deadline_ms(ms));
        }
        self
    }

    /// Clifford+√T only: override the deep-ε sequential-parity schedule
    /// (see [`SynthesizerQ::with_seq_parity`]); `Some(false)` forces the
    /// concurrent (lower-wall) branches below the 2.5e-8 sequential
    /// threshold. Ignored for Clifford+T.
    #[cfg_attr(not(feature = "python"), allow(dead_code))] // consumed by the PySynthesizer binding
    pub(crate) fn with_seq_parity(mut self, seq: Option<bool>) -> Self {
        if let Backend::Q(s) = self.inner {
            self.inner = Backend::Q(s.with_seq_parity(seq));
        }
        self
    }

    /// Jointly-optimized synthesis of `Rz(alpha)·Ry(beta)·Rz(gamma)` —
    /// the engine behind the u-gate family ([`Self::synthesize_u3`] and
    /// friends). For the rotation-by-rotation construction with the full
    /// native ε range, see [`Self::synthesize_zyz`].
    ///
    /// Builds BOTH the f64 acceptance target and the exact MPFR target
    /// column from the SAME angles (one construction, via
    /// [`crate::synthesis::angle::angle_target`]), so the deep-ε box center
    /// and the acceptance check can never disagree, then routes to
    /// [`Self::synthesize_su2_col`]. Exact below the f64 ULP for
    /// [`Angle::PiRatio`] angles; [`Angle::Rad`] covers plain-f64 callers.
    pub(crate) fn synthesize_zyz_joint(&self, alpha: Angle, beta: Angle, gamma: Angle) -> Option<SynthResult> {
        use crate::synthesis::angle::{angle_target, DEFAULT_COL_PREC};
        let (target, col) = angle_target(alpha, beta, gamma, DEFAULT_COL_PREC);
        // β = 0 → diagonal target Rz(α+γ): take the native Ross–Selinger
        // route (see the `native_rz` field). Falls through to the lattice
        // pipeline if it declines (it never hangs).
        if self.native_rz && beta.is_zero() {
            let theta = alpha.to_radians_mpfr(DEFAULT_COL_PREC)
                + gamma.to_radians_mpfr(DEFAULT_COL_PREC);
            // Diagonal gate powers (Clifford-looking angles and exact
            // T/√T powers) answer directly at lde 0.
            if let Some(r) = snap_rz_to_diag_power(&theta, self.epsilon(), self.sqrt_t()) {
                return Some(r);
            }
            let r = match &self.inner {
                Backend::T(s) => synthesize_rz_native(&theta, &target, s.epsilon),
                Backend::Q(s) => {
                    synthesize_rz_q(&theta, &target, s.epsilon(), s.q_cost_x2)
                }
            };
            if r.is_some() {
                return r;
            }
        }
        self.synthesize_su2_col(target, &col)
    }

    /// Synthesize a `U3(theta, phi, lambda)` gate (qiskit/bqskit convention)
    /// from its angles; the global phase is unobservable and dropped.
    pub fn synthesize_u3(&self, theta: Angle, phi: Angle, lam: Angle) -> Option<SynthResult> {
        // U3(θ,φ,λ) ≡ ZYZ(α=φ, β=θ, γ=λ)
        self.synthesize_zyz_joint(phi, theta, lam)
    }

    /// Synthesize a `U1(lambda)` gate (qiskit convention), ≅ `Rz(lambda)` up
    /// to global phase.
    pub fn synthesize_u1(&self, lam: Angle) -> Option<SynthResult> {
        // U1(λ) ≡ ZYZ(α=λ, β=0, γ=0); β/γ exactly zero (PiRatio, not Rad)
        self.synthesize_zyz_joint(lam, Angle::PiRatio(0, 1), Angle::PiRatio(0, 1))
    }

    /// Synthesize a `U2(phi, lambda)` gate (qiskit convention),
    /// `U2(φ,λ) = U3(π/2, φ, λ)`.
    pub fn synthesize_u2(&self, phi: Angle, lam: Angle) -> Option<SynthResult> {
        // U2(φ,λ) ≡ ZYZ(α=φ, β=π/2, γ=λ); β exactly π/2 (PiRatio, not Rad)
        self.synthesize_zyz_joint(phi, Angle::PiRatio(1, 2), lam)
    }

    /// Synthesize `Rz(theta)` (≅ `U1(theta)` up to the dropped global
    /// phase) — the native Ross–Selinger route, ε down to 1e-48 on both
    /// gate sets.
    pub fn synthesize_rz(&self, theta: Angle) -> Option<SynthResult> {
        self.synthesize_zyz_joint(theta, Angle::PiRatio(0, 1), Angle::PiRatio(0, 1))
    }

    /// Synthesize `Rx(theta) = H·Rz(theta)·H` — the Rz circuit wrapped in
    /// exact Cliffords, so it inherits the native route's full ε range and
    /// its distance bound (diamond distance is conjugation-invariant).
    pub fn synthesize_rx(&self, theta: Angle) -> Option<SynthResult> {
        self.wrap_rz(theta, "H", "H")
    }

    /// Synthesize `Ry(theta) = S·H·Rz(theta)·H·S†` (Y = S·X·S†), with the
    /// same range and distance guarantees as [`Self::synthesize_rx`].
    pub fn synthesize_ry(&self, theta: Angle) -> Option<SynthResult> {
        self.wrap_rz(theta, "SH", "Hs")
    }

    /// Synthesize `Rz(alpha)·Ry(beta)·Rz(gamma)` rotation-by-rotation
    /// through the native gridsynth infrastructure: three z-axis
    /// rotations at `epsilon/3` each (Ry via its exact Clifford wrap);
    /// β = 0 collapses to a single full-ε z-rotation.
    ///
    /// Full native ε range (down to ~3e-48) on both gate sets and
    /// seconds-fast at any depth; the u-gate family
    /// ([`Self::synthesize_u3`] and friends) is jointly optimized
    /// instead (~2.5–3× cheaper circuits, narrower validated range).
    /// The returned `distance` is the sum of the three verified
    /// component distances — a sound upper bound on the composite
    /// diamond distance.
    pub fn synthesize_zyz(
        &self,
        alpha: Angle,
        beta: Angle,
        gamma: Angle,
    ) -> Option<SynthResult> {
        use crate::synthesis::near_clifford::{eval_gates_q, eval_gates_t};
        // A zero middle rotation collapses to a single z-rotation — run it
        // at full ε (exact distance, ~3× cheaper than splitting).
        if beta.is_zero() {
            return self.synthesize_zyz_joint(alpha, beta, gamma);
        }
        let eps_each = self.epsilon() / 3.0;
        if eps_each < 1e-48 {
            return None;
        }
        // The sub-synthesizer inherits the caller's routing and cost-model
        // knobs; only ε changes.
        let sub = Synthesizer::new(eps_each, self.sqrt_t())
            .with_native_rz(self.native_rz)
            .with_q_cost(self.q_weight());
        let ra = sub.synthesize_rz(alpha)?;
        let rb = sub.synthesize_ry(beta)?;
        let rc = sub.synthesize_rz(gamma)?;
        let gates = format!("{}{}{}", ra.gates?, rb.gates?, rc.gates?);
        let lde = match &self.inner {
            Backend::T(_) => eval_gates_t(&gates)?.k,
            Backend::Q(_) => eval_gates_q(&gates)?.k,
        };
        Some(SynthResult {
            gates: Some(gates),
            lde,
            distance: ra.distance + rb.distance + rc.distance,
        })
    }

    /// Rz circuit conjugated by an exact Clifford word: `pre·C·post` where
    /// `C ≈ Rz(theta)`. The diamond distance carries over unchanged; the
    /// lde is re-derived from the exact evaluation of the wrapped word
    /// (Clifford factors can shift the denominator exponent).
    fn wrap_rz(&self, theta: Angle, pre: &str, post: &str) -> Option<SynthResult> {
        use crate::synthesis::near_clifford::{eval_gates_q, eval_gates_t};
        let r = self.synthesize_rz(theta)?;
        let gates = format!("{pre}{}{post}", r.gates?);
        let lde = match &self.inner {
            Backend::T(_) => eval_gates_t(&gates)?.k,
            Backend::Q(_) => eval_gates_q(&gates)?.k,
        };
        Some(SynthResult { gates: Some(gates), lde, distance: r.distance })
    }

    /// Synthesize with a higher-precision target column `exact_col` — the
    /// √det-normalized first column `[Re u00, Im u00, Re u10, Im u10]` of the
    /// SU(2) target (e.g. from exact rational-π angles via
    /// [`crate::synthesis::angle::su2_col_mpfr`]). Clifford+T aligns to it on
    /// the deep-ε MPFR path; Clifford+√T uses the f64 `target` for now.
    ///
    /// Prefer [`Self::synthesize_zyz`]/[`Self::synthesize_u3`] — building the
    /// target and exact column separately risks a convention mismatch.
    pub(crate) fn synthesize_su2_col(
        &self,
        target: Mat2,
        exact_col: &[crate::rings::MpFloat; 4],
    ) -> Option<SynthResult> {
        if let Some(r) = self.try_near_shallow(&target, exact_col) {
            return Some(r);
        }
        self.synthesize_su2_col_direct(target, exact_col)
    }

    /// Near-shallow-point handling (issue #2): targets within ε of a
    /// catalog point return it directly (optimal); targets in the
    /// pathological ring ε ≤ δ < RING_FACTOR·ε synthesize `target·W†`
    /// (generic) and append the exact escape circuit `W`. `None` = not
    /// near a shallow point, or the escape declined — take the direct path.
    fn try_near_shallow(
        &self,
        target: &Mat2,
        exact_col: &[crate::rings::MpFloat; 4],
    ) -> Option<SynthResult> {
        use crate::synthesis::distance::diamond_distance_u2t_float;
        use crate::synthesis::near_clifford as nc;
        let eps = self.epsilon();
        let is_q = matches!(self.inner, Backend::Q(_));
        let (delta, point) = nc::nearest_shallow(target, is_q);
        if delta < eps {
            // The shallow point itself is the minimal-cost answer. Verify
            // with the backend's exact acceptance check before returning.
            let hit = if is_q {
                nc::eval_gates_q(&point.gates).map(|u| {
                    let d = crate::synthesis::distance::diamond_distance_float(
                        &u.to_float(),
                        target,
                    );
                    (u.k, d)
                })
            } else {
                nc::eval_gates_t(&point.gates).map(|u| (u.k, diamond_distance_u2t_float(&u, target)))
            };
            if let Some((lde, dist)) = hit {
                if dist < eps {
                    return Some(SynthResult {
                        gates: Some(point.gates.clone()),
                        lde,
                        distance: dist,
                    });
                }
            }
            // δ ≈ ε boundary disagreement — fall through to the escape.
        }
        if eps > nc::ESCAPE_EPS_MAX || delta >= nc::RING_FACTOR * eps {
            return None;
        }
        let esc = nc::build_escape(target, exact_col, delta / eps)?;
        let inner = self.synthesize_su2_col_direct(esc.target, &esc.col)?;
        let inner_gates = inner.gates?;
        let gates = format!("{inner_gates}{}", esc.w_gates);
        // Composition with the exact W preserves diamond distance; re-verify
        // against the ORIGINAL target with the exact evaluators anyway.
        let (lde, dist) = if is_q {
            let u = nc::eval_gates_q(&gates)?;
            (u.k, crate::synthesis::distance::diamond_distance_float(&u.to_float(), target))
        } else {
            let u = nc::eval_gates_t(&gates)?;
            (u.k, diamond_distance_u2t_float(&u, target))
        };
        if dist < eps {
            Some(SynthResult { gates: Some(gates), lde, distance: dist })
        } else {
            None
        }
    }

    /// Backend dispatch without the near-shallow front half — the escape
    /// path calls this for its inner (generic) synthesis.
    fn synthesize_su2_col_direct(
        &self,
        target: Mat2,
        exact_col: &[crate::rings::MpFloat; 4],
    ) -> Option<SynthResult> {
        match &self.inner {
            Backend::T(s) => s.synthesize_with_exact_col(target, exact_col).map(|r| SynthResult {
                gates: r.gates,
                lde: r.lde,
                distance: r.distance,
            }),
            Backend::Q(s) => s.synthesize(target).map(|r| SynthResult {
                gates: r.gates,
                lde: r.lde,
                distance: r.distance,
            }),
        }
    }

    /// Target diamond distance the synthesized circuit must come within.
    #[cfg_attr(not(feature = "python"), allow(dead_code))] // consumed by the PySynthesizer binding
    pub(crate) fn epsilon(&self) -> f64 {
        match &self.inner {
            Backend::T(s) => s.epsilon,
            Backend::Q(s) => s.epsilon(),
        }
    }

    /// Largest lde (search depth) the synthesizer will try before giving up.
    pub fn max_lde(&self) -> u32 {
        match &self.inner {
            Backend::T(s) => s.max_lde,
            Backend::Q(s) => s.max_lde,
        }
    }

    /// Smallest lde the synthesizer starts from (skips guaranteed-empty
    /// shallow depths at deep ε).
    #[cfg_attr(not(feature = "python"), allow(dead_code))] // consumed by the PySynthesizer binding
    pub(crate) fn min_lde(&self) -> u32 {
        match &self.inner {
            Backend::T(s) => s.min_lde,
            Backend::Q(s) => s.min_lde,
        }
    }

    /// `true` if this is a Clifford+√T synthesizer, `false` for Clifford+T.
    #[cfg_attr(not(feature = "python"), allow(dead_code))] // consumed by the PySynthesizer binding
    pub(crate) fn sqrt_t(&self) -> bool {
        matches!(&self.inner, Backend::Q(_))
    }

    /// Cost in `T` states of one √T-class syllable in the syllable cost model
    /// (a T-class syllable costs 1). Canonical 3; reflects a custom
    /// `with_q_cost` on the √T backend.
    #[cfg_attr(not(feature = "python"), allow(dead_code))] // consumed by the PySynthesizer binding
    pub(crate) fn q_weight(&self) -> f64 {
        match &self.inner {
            Backend::T(_) => 3.0,
            // q_cost_x2 is a small user knob (default 6 = 3·T; set from 2·weight).
            #[allow(clippy::cast_precision_loss)]
            Backend::Q(s) => s.q_cost_x2 as f64 / 2.0,
        }
    }
}

// ─── PyO3 bindings ────────────────────────────────────────────────────────────

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Python-facing result of a synthesis run. Same shape for both gate sets.
#[cfg(feature = "python")]
#[pyclass(name = "SynthResult", frozen)]
pub(crate) struct PySynthResult {
    /// Gate string (leftmost = first gate applied), or `None` if extraction
    /// failed. Alphabet `{H, S, T, X, Y, Z}` for Clifford+T,
    /// `{H, S, T, Q, X, Y, Z}` for Clifford+√T.
    #[pyo3(get)]
    pub(crate) gates: Option<String>,
    /// Denominator exponent (lde, the search depth) of the synthesized unitary.
    #[pyo3(get)]
    pub(crate) lde: u32,
    /// Diamond distance from the synthesized unitary to the target (< epsilon).
    #[pyo3(get)]
    pub(crate) distance: f64,
    /// Q-gate weight used for `cost` (3 unless overridden on the √T backend).
    q_weight: f64,
}

#[cfg(feature = "python")]
#[pymethods]
impl PySynthResult {
    /// Number of T-class gates (`T` or its adjoint `t`=T†) in the circuit
    /// (0 if synthesis failed).
    #[getter]
    fn t_count(&self) -> usize {
        self.gates.as_deref().map_or(0, |g| {
            g.chars().filter(|&c| c == 'T' || c == 't').count()
        })
    }

    /// Number of √T-class gates (`Q`=√T or its adjoint `q`=Q†) in the circuit
    /// (0 for Clifford+T, or on failure).
    #[getter]
    fn q_count(&self) -> usize {
        self.gates.as_deref().map_or(0, |g| {
            g.chars().filter(|&c| c == 'Q' || c == 'q').count()
        })
    }

    /// The minimized resource cost, in `T` states. This is the syllable-model
    /// cost the optimizer minimizes: gates are charged per diagonal syllable
    /// by their net √T-power class (a √T-class syllable costs `q_weight`, a
    /// T-class syllable 1, Cliffords 0), so a `T` that composes with a `√T`
    /// into `T^{3/2}=√T†S` is one √T-class injection. It can therefore be
    /// *below* `t_count + q_weight·q_count`. (`q_weight` 3 default.)
    #[getter]
    fn cost(&self) -> f64 {
        let q_cost_x2 = (2.0 * self.q_weight).round() as usize;
        self.gates.as_deref().map_or(0.0, |g| {
            crate::synthesis::clifford_sqrt_t::gates_cost(g, q_cost_x2) as f64 / 2.0
        })
    }

    /// `True` if synthesis produced a circuit, so `if result:` works.
    fn __bool__(&self) -> bool {
        self.gates.is_some()
    }

    fn __repr__(&self) -> String {
        let gates_repr = self
            .gates
            .as_deref()
            .map(|g| format!("{g:?}"))
            .unwrap_or_else(|| "None".to_string());
        format!(
            "SynthResult(gates={gates_repr}, lde={}, distance={:.3e})",
            self.lde, self.distance
        )
    }
}

/// Unified Python-facing single-qubit unitary synthesizer.
///
/// ```python
/// import cyclosynth
/// # Clifford+T (default); sqrt_t=True selects Clifford+√T (denser, often fewer gates).
/// synth = cyclosynth.Synthesizer(epsilon=1e-5)
///
/// # Angles only — U3 (theta, phi, lambda) or ZYZ Euler (alpha, beta, gamma);
/// # each a float or an exact-pi string like "pi/64".
/// result = synth.synthesize_u3("pi/64", 0, 0)
/// print(result.gates, result.lde, result.distance)
/// ```
#[cfg(feature = "python")]
#[pyclass(name = "Synthesizer", frozen)]
pub(crate) struct PySynthesizer {
    inner: Synthesizer,
}

#[cfg(feature = "python")]
#[pymethods]
impl PySynthesizer {
    /// Build a synthesizer for target diamond distance `epsilon`.
    ///
    /// `sqrt_t=False` gives Clifford+T; `sqrt_t=True` gives Clifford+√T
    /// (a denser gate set, usually fewer gates). `max_lde`/`min_lde` bound the
    /// search depth. The remaining kwargs (`optimize_cost`, `q_cost`,
    /// `lde_window`, `deadline_ms`, `seq_parity`) are **Clifford+√T-only**
    /// cost-optimizer tuning and raise `ValueError` if passed with
    /// `sqrt_t=False`.
    ///
    /// ε-range policy (see [`check_epsilon_policy`]): ε ≥ 1e-48 always
    /// (`ValueError` below); the range checks happen per call.
    /// Z-rotation targets (u1/rz/rx/ry) and `synthesize_zyz` (the
    /// rotation-by-rotation route) take the native gridsynth
    /// infrastructure down to the 1e-48-class floor on both gate sets.
    /// Jointly-optimized general targets: `sqrt_t=True` (u2/u3) raises
    /// `ValueError` below 1e-8; `sqrt_t=False` warns below 1e-10.
    ///
    /// `native_rz` (default on) routes z-rotation targets through the
    /// native Ross–Selinger path — milliseconds at any supported ε,
    /// including near-Clifford angles; pass `False` to force the lattice
    /// pipeline.
    #[new]
    #[pyo3(signature = (epsilon, *, sqrt_t=false, max_lde=None, min_lde=None,
                        optimize_cost=None, q_cost=None, lde_window=None,
                        deadline_ms=None, seq_parity=None, native_rz=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        epsilon: f64,
        sqrt_t: bool,
        max_lde: Option<u32>,
        min_lde: Option<u32>,
        optimize_cost: Option<bool>,
        q_cost: Option<f64>,
        lde_window: Option<u32>,
        deadline_ms: Option<u64>,
        seq_parity: Option<bool>,
        native_rz: Option<bool>,
    ) -> PyResult<Self> {
        // Only the hard floor here: whether the narrower general-target
        // ranges apply depends on what gets synthesized, so those are
        // checked per call.
        let _ = py;
        check_epsilon_floor(epsilon)?;
        // The cost-optimizer kwargs only affect the √T backend; silently
        // ignoring them for Clifford+T is a footgun, so reject up front.
        if !sqrt_t
            && (optimize_cost.is_some() || q_cost.is_some() || lde_window.is_some()
                || deadline_ms.is_some() || seq_parity.is_some())
        {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "optimize_cost / q_cost / lde_window / deadline_ms / seq_parity \
                 are Clifford+√T-only; pass sqrt_t=True to use them",
            ));
        }
        let mut s = Synthesizer::new(epsilon, sqrt_t);
        if let Some(v) = max_lde {
            s = s.with_max_lde(v);
        }
        if let Some(v) = min_lde {
            s = s.with_min_lde(v);
        }
        // None = keep the backend default (on for √T at every ε);
        // Some(b) = explicit override in either direction.
        if let Some(b) = optimize_cost {
            s = s.with_optimize_cost(b);
        }
        if let Some(w) = q_cost {
            s = s.with_q_cost(w);
        }
        if let Some(w) = lde_window {
            s = s.with_optimal_lde_window(w);
        }
        // None = keep the ε-based default deadline; an explicit value
        // overrides it (there is no kwarg form for "no deadline").
        if let Some(ms) = deadline_ms {
            s = s.with_optimal_deadline_ms(Some(ms));
        }
        if seq_parity.is_some() {
            s = s.with_seq_parity(seq_parity);
        }
        // None = default (on for both gate sets).
        if let Some(b) = native_rz {
            s = s.with_native_rz(b);
        }
        Ok(Self { inner: s })
    }

    /// Synthesize a `U3(theta, phi, lambda)` gate (qiskit/bqskit convention)
    /// from its angles — the entry point for bqskit `U3Gate` inputs (pass
    /// `op.params`).
    ///
    /// `U3` is `e^{i(phi+lambda)/2}·Rz(phi)·Ry(theta)·Rz(lambda)`; the global
    /// phase is unobservable, so the SU(2) rotation `Rz(phi)·Ry(theta)·Rz(lambda)`
    /// is built directly as the target.
    ///
    /// Each angle is a float (radians) or a string. A string containing `pi`
    /// (whitespace ignored, optional `*`) is a rational multiple of π —
    /// `"pi"`, `"3pi"`, `"3*pi"`, `"pi/8"`, `"3*pi/4"`, `"-2pi/3"`, `"0.25pi"`;
    /// any other string parses as a float in radians.
    #[pyo3(signature = (theta, phi, lam))]
    fn synthesize_u3(
        &self,
        theta: &Bound<'_, PyAny>,
        phi: &Bound<'_, PyAny>,
        lam: &Bound<'_, PyAny>,
    ) -> PyResult<Option<PySynthResult>> {
        let py = theta.py();
        let (theta, phi, lam) = (parse_angle(theta)?, parse_angle(phi)?, parse_angle(lam)?);
        self.check_policy(py, theta.is_zero())?;
        Ok(self.wrap(self.inner.synthesize_u3(theta, phi, lam)))
    }

    /// Synthesize `Rz(alpha)·Ry(beta)·Rz(gamma)` rotation-by-rotation
    /// through the native gridsynth routes (each rotation at
    /// `epsilon/3`): the full ε range (floor ~3e-48) on both gate sets,
    /// seconds-fast at any depth, at ~2.5–3× the jointly-optimized cost
    /// of [`Self::synthesize_u3`]. The reported distance is the sum of
    /// the three verified component distances (a sound upper bound).
    /// Each angle accepts the same float/`pi`-string forms as
    /// [`Self::synthesize_u3`].
    #[pyo3(signature = (alpha, beta, gamma))]
    fn synthesize_zyz(
        &self,
        alpha: &Bound<'_, PyAny>,
        beta: &Bound<'_, PyAny>,
        gamma: &Bound<'_, PyAny>,
    ) -> PyResult<Option<PySynthResult>> {
        let py = alpha.py();
        let (alpha, beta, gamma) =
            (parse_angle(alpha)?, parse_angle(beta)?, parse_angle(gamma)?);
        self.check_policy(py, true)?;
        Ok(self.wrap(self.inner.synthesize_zyz(alpha, beta, gamma)))
    }

    /// Synthesize a `U1(lambda)` gate (qiskit convention) — a phase gate,
    /// ≅ `Rz(lambda)` up to global phase. The angle accepts the same
    /// float/`pi`-string forms as [`Self::synthesize_u3`]. Native route,
    /// ε down to 1e-48 on both gate sets.
    #[pyo3(signature = (lam))]
    fn synthesize_u1(&self, lam: &Bound<'_, PyAny>) -> PyResult<Option<PySynthResult>> {
        let py = lam.py();
        let lam = parse_angle(lam)?;
        self.check_policy(py, true)?;
        Ok(self.wrap(self.inner.synthesize_u1(lam)))
    }

    /// Synthesize a `U2(phi, lambda)` gate (qiskit convention),
    /// `U2(phi, lambda) = U3(pi/2, phi, lambda)`. Each angle accepts the same
    /// float/`pi`-string forms as [`Self::synthesize_u3`].
    #[pyo3(signature = (phi, lam))]
    fn synthesize_u2(
        &self,
        phi: &Bound<'_, PyAny>,
        lam: &Bound<'_, PyAny>,
    ) -> PyResult<Option<PySynthResult>> {
        let py = phi.py();
        let (phi, lam) = (parse_angle(phi)?, parse_angle(lam)?);
        self.check_policy(py, false)?;
        Ok(self.wrap(self.inner.synthesize_u2(phi, lam)))
    }

    /// Synthesize `Rz(theta)` (≅ `U1(theta)` up to global phase). Native
    /// route, ε down to 1e-48 on both gate sets. The angle accepts the
    /// same float/`pi`-string forms as [`Self::synthesize_u3`].
    #[pyo3(signature = (theta))]
    fn synthesize_rz(&self, theta: &Bound<'_, PyAny>) -> PyResult<Option<PySynthResult>> {
        let py = theta.py();
        let theta = parse_angle(theta)?;
        self.check_policy(py, true)?;
        Ok(self.wrap(self.inner.synthesize_rz(theta)))
    }

    /// Synthesize `Rx(theta) = H·Rz(theta)·H` — the native Rz circuit
    /// wrapped in exact Cliffords (same ε range and distance bound).
    #[pyo3(signature = (theta))]
    fn synthesize_rx(&self, theta: &Bound<'_, PyAny>) -> PyResult<Option<PySynthResult>> {
        let py = theta.py();
        let theta = parse_angle(theta)?;
        self.check_policy(py, true)?;
        Ok(self.wrap(self.inner.synthesize_rx(theta)))
    }

    /// Synthesize `Ry(theta) = S·H·Rz(theta)·H·S†` — the native Rz circuit
    /// wrapped in exact Cliffords (same ε range and distance bound).
    #[pyo3(signature = (theta))]
    fn synthesize_ry(&self, theta: &Bound<'_, PyAny>) -> PyResult<Option<PySynthResult>> {
        let py = theta.py();
        let theta = parse_angle(theta)?;
        self.check_policy(py, true)?;
        Ok(self.wrap(self.inner.synthesize_ry(theta)))
    }

    #[getter]
    fn epsilon(&self) -> f64 {
        self.inner.epsilon()
    }

    #[getter]
    fn max_lde(&self) -> u32 {
        self.inner.max_lde()
    }

    #[getter]
    fn min_lde(&self) -> u32 {
        self.inner.min_lde()
    }

    #[getter]
    fn sqrt_t(&self) -> bool {
        self.inner.sqrt_t()
    }

    fn __repr__(&self) -> String {
        format!(
            "Synthesizer(epsilon={:.3e}, sqrt_t={}, min_lde={}, max_lde={})",
            self.inner.epsilon(),
            self.inner.sqrt_t(),
            self.inner.min_lde(),
            self.inner.max_lde(),
        )
    }
}

#[cfg(feature = "python")]
impl PySynthesizer {
    /// Attach the backend's q-weight to a library result for Python. All
    /// target/column construction lives in [`Synthesizer::synthesize_zyz`].
    fn wrap(&self, r: Option<SynthResult>) -> Option<PySynthResult> {
        let q_weight = self.inner.q_weight();
        r.map(|r| PySynthResult {
            gates: r.gates,
            lde: r.lde,
            distance: r.distance,
            q_weight,
        })
    }

    /// Per-call ε policy (see [`check_epsilon_policy`]): z-rotation calls
    /// are valid at any constructed ε; general targets enforce the
    /// lattice pipelines' narrower ranges.
    fn check_policy(&self, py: Python<'_>, rz_class: bool) -> PyResult<()> {
        check_epsilon_policy(py, self.inner.epsilon(), self.inner.sqrt_t(), rz_class)
    }
}

/// Parse one angle argument — a Python float/int, or a string (the `pi`
/// rational forms of [`crate::synthesis::angle::parse_angle_str`]).
#[cfg(feature = "python")]
fn parse_angle(obj: &Bound<'_, PyAny>) -> PyResult<Angle> {
    use crate::synthesis::angle::parse_angle_str;
    if let Ok(x) = obj.extract::<f64>() {
        return Ok(Angle::Rad(x));
    }
    let s: String = obj.extract().map_err(|_| {
        PyErr::new::<pyo3::exceptions::PyTypeError, _>("angle must be a float or string")
    })?;
    parse_angle_str(&s).map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)
}

/// ε-range policy of the Python API (the Rust core is unrestricted). Enforced
/// at every Python entry point: `Synthesizer.__init__` and the module-level
/// `synthesize_u1/u2/u3` functions.
///
/// - Clifford+√T (`sqrt_t=True`): ε < 1e-8 raises `ValueError` — the Z[ζ]
///   backend's f64 SE walk loses the cap below that (cap half-width ε² hits
///   the f64 ULP; see the ε=1.5e-8 cliff analysis).
/// - Clifford+T (`sqrt_t=False`): ε < 1e-10 emits a `UserWarning` and
///   proceeds — below the oracle-validated range, runtime grows steeply and
///   T-optimality is unverified.
#[cfg(feature = "python")]
/// Hard floor shared by both gate sets: the native z-rotation routes are
/// exact-verified to 1e-48; below that the exact-integer widths of the
/// gate decomposition overflow.
#[cfg(feature = "python")]
fn check_epsilon_floor(epsilon: f64) -> PyResult<()> {
    if epsilon < 1e-48 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "synthesis is supported for epsilon >= 1e-48 (the exact-integer \
             width of the gate decomposition; verified to that depth for \
             z-rotations on both gate sets); requested {epsilon:e}."
        )));
    }
    Ok(())
}

/// Per-call ε policy. `rz_class` marks z-rotation targets (u1/rz/rx/ry,
/// or β = 0): those take the native Ross–Selinger routes, valid to 1e-48
/// on BOTH gate sets. General targets go to the lattice pipelines, whose
/// validated ranges are narrower.
#[cfg(feature = "python")]
fn check_epsilon_policy(
    py: Python<'_>,
    epsilon: f64,
    sqrt_t: bool,
    rz_class: bool,
) -> PyResult<()> {
    check_epsilon_floor(epsilon)?;
    if rz_class {
        return Ok(());
    }
    if sqrt_t {
        if epsilon < 1e-8 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "jointly-optimized Clifford+√T synthesis (u2/u3) is supported \
                 for epsilon >= 1e-8 (the Z[ζ] lattice backend's validated \
                 range); requested {epsilon:e}. Use synthesize_zyz — the \
                 rotation-by-rotation route through the native gridsynth \
                 infrastructure — for the full range at ~2.5-3x the cost, or \
                 sqrt_t=False for general targets."
            )));
        }
    } else if epsilon < 1e-10 {
        PyErr::warn_bound(
            py,
            &py.get_type_bound::<pyo3::exceptions::PyUserWarning>(),
            &format!(
                "epsilon {epsilon:e} is below the oracle-validated range \
                 (1e-10). Z-rotations (u1 / rz / rx / ry, beta=0) stay \
                 milliseconds via the native Ross-Selinger route down to \
                 1e-48; general u3 targets use the lattice pipeline, whose \
                 runtime grows steeply (~10x per decade, minutes-scale tails \
                 at 1e-12) and whose T-optimality is unverified at this depth."
            ),
            2,
        )?;
    }
    Ok(())
}

/// One-shot driver behind the module-level functions: policy check, build a
/// default synthesizer, run `f` on it, wrap for Python. All target/column
/// construction stays inside the library `Synthesizer` angle adapters.
#[cfg(feature = "python")]
fn synthesize_oneshot(
    py: Python<'_>,
    epsilon: f64,
    sqrt_t: bool,
    rz_class: bool,
    f: impl FnOnce(&Synthesizer) -> Option<SynthResult>,
) -> PyResult<Option<PySynthResult>> {
    check_epsilon_policy(py, epsilon, sqrt_t, rz_class)?;
    let synth = Synthesizer::new(epsilon, sqrt_t);
    let q_weight = synth.q_weight();
    Ok(f(&synth).map(|r| PySynthResult {
        gates: r.gates,
        lde: r.lde,
        distance: r.distance,
        q_weight,
    }))
}

/// Synthesize a `U1(lam)` gate (qiskit convention; ≅ `Rz(lam)` up to global
/// phase) to diamond distance `epsilon` with default settings.
///
/// The angle is a float (radians) or an exact-π string like `"pi/64"`;
/// `sqrt_t=True` selects Clifford+√T. Z-rotation targets support ε down
/// to 1e-48 on both gate sets (native route). Returns `None` if no
/// circuit was found. For repeated calls or tuning knobs, use
/// [`PySynthesizer`] (`cyclosynth.Synthesizer`).
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (lam, epsilon, *, sqrt_t=false))]
pub(crate) fn synthesize_u1(
    py: Python<'_>,
    lam: &Bound<'_, PyAny>,
    epsilon: f64,
    sqrt_t: bool,
) -> PyResult<Option<PySynthResult>> {
    let lam = parse_angle(lam)?;
    synthesize_oneshot(py, epsilon, sqrt_t, true, |s| s.synthesize_u1(lam))
}

/// Synthesize `Rz(theta)` (≅ `U1(theta)` up to global phase) to diamond
/// distance `epsilon` with default settings. Same angle forms / `sqrt_t`
/// semantics as [`synthesize_u1`]; ε down to 1e-48 on both gate sets.
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (theta, epsilon, *, sqrt_t=false))]
pub(crate) fn synthesize_rz(
    py: Python<'_>,
    theta: &Bound<'_, PyAny>,
    epsilon: f64,
    sqrt_t: bool,
) -> PyResult<Option<PySynthResult>> {
    let theta = parse_angle(theta)?;
    synthesize_oneshot(py, epsilon, sqrt_t, true, |s| s.synthesize_rz(theta))
}

/// Synthesize `Rx(theta) = H·Rz(theta)·H` to diamond distance `epsilon`
/// with default settings — the native Rz circuit wrapped in exact
/// Cliffords, with the same ε range and distance bound as
/// [`synthesize_rz`].
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (theta, epsilon, *, sqrt_t=false))]
pub(crate) fn synthesize_rx(
    py: Python<'_>,
    theta: &Bound<'_, PyAny>,
    epsilon: f64,
    sqrt_t: bool,
) -> PyResult<Option<PySynthResult>> {
    let theta = parse_angle(theta)?;
    synthesize_oneshot(py, epsilon, sqrt_t, true, |s| s.synthesize_rx(theta))
}

/// Synthesize `Ry(theta) = S·H·Rz(theta)·H·S†` to diamond distance
/// `epsilon` with default settings — same construction and guarantees as
/// [`synthesize_rx`].
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (theta, epsilon, *, sqrt_t=false))]
pub(crate) fn synthesize_ry(
    py: Python<'_>,
    theta: &Bound<'_, PyAny>,
    epsilon: f64,
    sqrt_t: bool,
) -> PyResult<Option<PySynthResult>> {
    let theta = parse_angle(theta)?;
    synthesize_oneshot(py, epsilon, sqrt_t, true, |s| s.synthesize_ry(theta))
}

/// Synthesize a `U2(phi, lam)` gate (qiskit convention;
/// `U2(φ,λ) = U3(π/2, φ, λ)`) to diamond distance `epsilon` with default
/// settings. Same angle forms / `sqrt_t` semantics as [`synthesize_u1`].
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (phi, lam, epsilon, *, sqrt_t=false))]
pub(crate) fn synthesize_u2(
    py: Python<'_>,
    phi: &Bound<'_, PyAny>,
    lam: &Bound<'_, PyAny>,
    epsilon: f64,
    sqrt_t: bool,
) -> PyResult<Option<PySynthResult>> {
    let (phi, lam) = (parse_angle(phi)?, parse_angle(lam)?);
    synthesize_oneshot(py, epsilon, sqrt_t, false, |s| s.synthesize_u2(phi, lam))
}

/// Synthesize `Rz(alpha)·Ry(beta)·Rz(gamma)` rotation-by-rotation through
/// the native gridsynth routes (each rotation at `epsilon/3`) — the full
/// ε range on both gate sets at ~2.5–3× the jointly-optimized cost of
/// [`synthesize_u3`]. Same angle forms as [`synthesize_u1`].
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (alpha, beta, gamma, epsilon, *, sqrt_t=false))]
pub(crate) fn synthesize_zyz(
    py: Python<'_>,
    alpha: &Bound<'_, PyAny>,
    beta: &Bound<'_, PyAny>,
    gamma: &Bound<'_, PyAny>,
    epsilon: f64,
    sqrt_t: bool,
) -> PyResult<Option<PySynthResult>> {
    let (alpha, beta, gamma) = (parse_angle(alpha)?, parse_angle(beta)?, parse_angle(gamma)?);
    synthesize_oneshot(py, epsilon, sqrt_t, true, |s| s.synthesize_zyz(alpha, beta, gamma))
}

/// Synthesize a `U3(theta, phi, lam)` gate (qiskit/bqskit convention) to
/// diamond distance `epsilon` with default settings. Same angle forms /
/// `sqrt_t` semantics as [`synthesize_u1`].
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (theta, phi, lam, epsilon, *, sqrt_t=false))]
pub(crate) fn synthesize_u3(
    py: Python<'_>,
    theta: &Bound<'_, PyAny>,
    phi: &Bound<'_, PyAny>,
    lam: &Bound<'_, PyAny>,
    epsilon: f64,
    sqrt_t: bool,
) -> PyResult<Option<PySynthResult>> {
    let (theta, phi, lam) = (parse_angle(theta)?, parse_angle(phi)?, parse_angle(lam)?);
    synthesize_oneshot(py, epsilon, sqrt_t, theta.is_zero(), |s| s.synthesize_u3(theta, phi, lam))
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthesis::distance::{diamond_distance_float, diamond_distance_u2t_float};
    use crate::synthesis::near_clifford::{eval_gates_q, eval_gates_t};
    use num_complex::Complex;

    fn rx_target(t: f64) -> Mat2 {
        let (c, s) = ((t / 2.0).cos(), (t / 2.0).sin());
        [
            [Complex::new(c, 0.0), Complex::new(0.0, -s)],
            [Complex::new(0.0, -s), Complex::new(c, 0.0)],
        ]
    }

    fn ry_target(t: f64) -> Mat2 {
        let (c, s) = ((t / 2.0).cos(), (t / 2.0).sin());
        [
            [Complex::new(c, 0.0), Complex::new(-s, 0.0)],
            [Complex::new(s, 0.0), Complex::new(c, 0.0)],
        ]
    }

    fn rz_target(t: f64) -> Mat2 {
        [
            [Complex::from_polar(1.0, -t / 2.0), Complex::new(0.0, 0.0)],
            [Complex::new(0.0, 0.0), Complex::from_polar(1.0, t / 2.0)],
        ]
    }

    /// The Clifford-wrap algebra behind rx/ry (X = H·Z·H, Y = S·X·S†):
    /// re-verify each circuit against a numerically built Rx/Ry/Rz target
    /// on BOTH gate sets.
    #[test]
    fn test_rx_ry_rz_against_targets() {
        let eps = 1e-3;
        for sqrt_t in [false, true] {
            let synth = Synthesizer::new(eps, sqrt_t);
            for theta in [0.7_f64, 1.9] {
                let cases: [(&str, Option<SynthResult>, Mat2); 3] = [
                    ("rz", synth.synthesize_rz(Angle::Rad(theta)), rz_target(theta)),
                    ("rx", synth.synthesize_rx(Angle::Rad(theta)), rx_target(theta)),
                    ("ry", synth.synthesize_ry(Angle::Rad(theta)), ry_target(theta)),
                ];
                for (name, r, target) in cases {
                    let r = r.unwrap_or_else(|| panic!("{name}({theta}) sqrt_t={sqrt_t}"));
                    let gates = r.gates.expect("gates");
                    let dist = if sqrt_t {
                        let u = eval_gates_q(&gates).expect("eval q");
                        diamond_distance_float(&u.to_float(), &target)
                    } else {
                        let u = eval_gates_t(&gates).expect("eval t");
                        diamond_distance_u2t_float(&u, &target)
                    };
                    assert!(
                        dist < eps,
                        "{name}({theta}) sqrt_t={sqrt_t}: re-verified dist {dist:.3e} >= {eps}"
                    );
                    assert!((dist - r.distance).abs() < 1e-12, "{name} distance mismatch");
                }
            }
        }
    }

    /// u3 (general-target) pipeline benchmark through the front door,
    /// both gate sets. Deterministic targets; per-ε minimum over trials.
    /// Run: `cargo test --release --lib bench_u3_pipeline -- --ignored --nocapture`
    #[test]
    #[ignore = "bench probe, print-only"]
    fn bench_u3_pipeline() {
        let mut state = 0xC0FF_EE00_D00D_5EEDu64;
        let mut rnd = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64 * std::f64::consts::TAU
        };
        let targets: Vec<[f64; 3]> = (0..5).map(|_| [rnd(), rnd(), rnd()]).collect();
        for (sqrt_t, levels) in [
                        (false, vec![1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]),
            (true, vec![1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]),
        ] {
            let set = if sqrt_t { "sqrt_t" } else { "t     " };
            for eps in levels {
                let synth = Synthesizer::new(eps, sqrt_t);
                let mut total_ms = 0.0;
                let mut costs = Vec::new();
                for a in &targets {
                    let mut best = f64::INFINITY;
                    let mut lde = 0;
                    for _ in 0..2 {
                        let t0 = std::time::Instant::now();
                        let r = synth
                            .synthesize_u3(Angle::Rad(a[0]), Angle::Rad(a[1]), Angle::Rad(a[2]))
                            .expect("solves");
                        best = best.min(t0.elapsed().as_secs_f64() * 1e3);
                        lde = r.lde;
                    }
                    total_ms += best;
                    costs.push(lde);
                }
                eprintln!(
                    "{set} eps={eps:.0e}  sum(min) {total_ms:9.1} ms across 5 targets  lde={costs:?}"
                );
            }
        }
    }

    #[test]
    #[ignore = "bench probe, print-only"]
    fn bench_u3_deep_q() {
        let mut state = 0xC0FF_EE00_D00D_5EEDu64;
        let mut rnd = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64 * std::f64::consts::TAU
        };
        let targets: Vec<[f64; 3]> = (0..5).map(|_| [rnd(), rnd(), rnd()]).collect();
        let synth = Synthesizer::new(1e-8, true);
        for (i, a) in targets.iter().enumerate() {
            let t0 = std::time::Instant::now();
            let r = synth
                .synthesize_u3(Angle::Rad(a[0]), Angle::Rad(a[1]), Angle::Rad(a[2]))
                .expect("solves");
            eprintln!(
                "deep_q target_{i}  {:8.2?}  lde={} dist={:.2e}",
                t0.elapsed(), r.lde, r.distance
            );
        }
    }

    /// Where do the u3 pipelines actually stop? Direct pipeline probes
    /// below the policy floors, plus the rotation route
    /// ([`Synthesizer::synthesize_zyz`]) at the same depths.
    /// Run: `cargo test --release --lib probe_u3_deep_floors -- --ignored --nocapture`
    #[test]
    #[ignore = "diagnostic probe, print-only"]
    fn probe_u3_deep_floors() {
        use crate::synthesis::distance::diamond_distance_float;
        use crate::synthesis::near_clifford::{eval_gates_q, eval_gates_t};
        let (a, b, g) = (0.7_f64, 1.9, 0.3);
        // Direct pipelines below their floors, wall-clock reported.
        // The 16D pipeline below its 1e-8 floor grinds without terminating
        // (>17 min on one target at 1e-9, with or without the reduced-
        // basis cache — the SE enumeration volume is the wall). Probe T
        // only; use the synthesize_zyz rows for deep sqrt-T u3.
        for (sqrt_t, eps_list) in [(false, vec![1e-10_f64, 1e-11])] {
            for eps in eps_list {
                let synth = Synthesizer::new(eps, sqrt_t);
                let t0 = std::time::Instant::now();
                let r = synth.synthesize_u3(Angle::Rad(b), Angle::Rad(a), Angle::Rad(g));
                match r {
                    Some(r) => eprintln!(
                        "direct sqrt_t={sqrt_t} eps={eps:.0e}: lde={} dist={:.2e} {:?}",
                        r.lde, r.distance, t0.elapsed()
                    ),
                    None => eprintln!("direct sqrt_t={sqrt_t} eps={eps:.0e}: NONE {:?}", t0.elapsed()),
                }
            }
        }
        // Rotation route (synthesize_zyz) at deep ε, both gate sets,
        // verified here against an independently composed target.
        for sqrt_t in [false, true] {
            for eps in [1e-10_f64, 1e-12] {
                let synth = Synthesizer::new(eps, sqrt_t);
                let t0 = std::time::Instant::now();
                let r = synth
                    .synthesize_zyz(Angle::Rad(a), Angle::Rad(b), Angle::Rad(g))
                    .expect("zyz");
                let gates = r.gates.expect("gates");
                let dt = t0.elapsed();
                let target = {
                    use num_complex::Complex;
                    let rz = |t: f64| [
                        [Complex::from_polar(1.0, -t / 2.0), Complex::new(0.0, 0.0)],
                        [Complex::new(0.0, 0.0), Complex::from_polar(1.0, t / 2.0)],
                    ];
                    let ry = |t: f64| [
                        [Complex::new((t / 2.0).cos(), 0.0), Complex::new(-(t / 2.0).sin(), 0.0)],
                        [Complex::new((t / 2.0).sin(), 0.0), Complex::new((t / 2.0).cos(), 0.0)],
                    ];
                    let mm = |x: [[Complex<f64>; 2]; 2], y: [[Complex<f64>; 2]; 2]| {
                        let mut o = [[Complex::new(0.0, 0.0); 2]; 2];
                        for i in 0..2 {
                            for j in 0..2 {
                                o[i][j] = x[i][0] * y[0][j] + x[i][1] * y[1][j];
                            }
                        }
                        o
                    };
                    mm(mm(rz(a), ry(b)), rz(g))
                };
                let (dist, tq) = if sqrt_t {
                    let u = eval_gates_q(&gates).expect("eval q");
                    (diamond_distance_float(&u.to_float(), &target),
                     crate::synthesis::clifford_sqrt_t::gates_cost(&gates, 6))
                } else {
                    let u = eval_gates_t(&gates).expect("eval t");
                    (diamond_distance_float(&u.to_float(), &target),
                     crate::synthesis::clifford_sqrt_t::gates_cost(&gates, 6))
                };
                eprintln!(
                    "zyz    sqrt_t={sqrt_t} eps={eps:.0e}: dist={dist:.2e} cost_x2={tq} {dt:?}"
                );
            }
        }
    }

    /// `synthesize_zyz` (the rotation-by-rotation gridsynth route)
    /// matches its ZYZ target on both gate sets, at coarse ε and below
    /// the √T joint pipeline's 1e-8 floor.
    #[test]
    fn test_zyz_rotation_route() {
        use crate::synthesis::distance::{diamond_distance_float, diamond_distance_u2t_float};
        use crate::synthesis::near_clifford::{eval_gates_q, eval_gates_t};
        use num_complex::Complex;
        let (a, b, g) = (0.7_f64, 1.9, 0.3);
        let target = {
            let rz = |t: f64| [
                [Complex::from_polar(1.0, -t / 2.0), Complex::new(0.0, 0.0)],
                [Complex::new(0.0, 0.0), Complex::from_polar(1.0, t / 2.0)],
            ];
            let ry = |t: f64| [
                [Complex::new((t / 2.0).cos(), 0.0), Complex::new(-(t / 2.0).sin(), 0.0)],
                [Complex::new((t / 2.0).sin(), 0.0), Complex::new((t / 2.0).cos(), 0.0)],
            ];
            let mm = |x: [[Complex<f64>; 2]; 2], y: [[Complex<f64>; 2]; 2]| {
                let mut o = [[Complex::new(0.0, 0.0); 2]; 2];
                for i in 0..2 {
                    for j in 0..2 {
                        o[i][j] = x[i][0] * y[0][j] + x[i][1] * y[1][j];
                    }
                }
                o
            };
            mm(mm(rz(a), ry(b)), rz(g))
        };
        for (sqrt_t, eps) in [(false, 1e-4), (true, 1e-4), (true, 1e-10)] {
            let r = Synthesizer::new(eps, sqrt_t)
                .synthesize_zyz(Angle::Rad(a), Angle::Rad(b), Angle::Rad(g))
                .unwrap_or_else(|| panic!("zyz sqrt_t={sqrt_t} eps={eps}"));
            assert!(r.distance < eps, "bound {} >= {eps}", r.distance);
            let gates = r.gates.expect("gates");
            let d = if sqrt_t {
                diamond_distance_float(&eval_gates_q(&gates).expect("eval").to_float(), &target)
            } else {
                diamond_distance_u2t_float(&eval_gates_t(&gates).expect("eval"), &target)
            };
            assert!(d < eps, "measured {d:.3e} >= {eps} (sqrt_t={sqrt_t})");
        }
    }

    /// Clifford-looking and exact-T/√T-power angles answer directly from
    /// the diagonal snap: exact strings at lde 0, at any ε (including a
    /// PiRatio input at the 1e-48 floor), and near-misses within ε snap
    /// to the power while genuine non-powers still synthesize.
    #[test]
    fn test_rz_diag_power_snap() {
        use std::f64::consts::{FRAC_PI_2, FRAC_PI_4};
        // Exact powers, Clifford+T.
        let t = Synthesizer::new(1e-10, false);
        for (theta, want) in [
            (Angle::PiRatio(1, 4), "T"),
            (Angle::PiRatio(1, 2), "S"),
            (Angle::PiRatio(1, 1), "Z"),
            (Angle::PiRatio(-1, 4), "t"),
            (Angle::Rad(FRAC_PI_2), "S"),
        ] {
            let r = t.synthesize_rz(theta).expect("snap");
            assert_eq!(r.gates.as_deref(), Some(want));
            assert_eq!(r.lde, 0);
            assert!(r.distance < 1e-12, "dist {}", r.distance);
        }
        // Exact powers, Clifford+√T — including the √T-only π/8 class.
        let q = Synthesizer::new(1e-10, true);
        for (theta, want) in [
            (Angle::PiRatio(1, 8), "Q"),
            (Angle::PiRatio(3, 8), "TQ"),
            (Angle::PiRatio(1, 2), "S"),
        ] {
            let r = q.synthesize_rz(theta).expect("snap");
            assert_eq!(r.gates.as_deref(), Some(want));
            assert_eq!(r.lde, 0);
        }
        // Deep ε: PiRatio stays exact where f64 catalog distances saturate.
        let deep = Synthesizer::new(1e-30, false);
        let r = deep.synthesize_rz(Angle::PiRatio(1, 4)).expect("deep snap");
        assert_eq!(r.gates.as_deref(), Some("T"));
        assert!(r.distance < 1e-30);
        // Near-miss within ε snaps; outside ε synthesizes normally.
        let t8 = Synthesizer::new(1e-8, false);
        let r = t8.synthesize_rz(Angle::Rad(FRAC_PI_4 + 1e-9)).expect("near snap");
        assert_eq!(r.gates.as_deref(), Some("T"));
        assert!(r.distance < 1e-8);
        let r = t8.synthesize_rz(Angle::Rad(0.7)).expect("non-power");
        assert!(r.gates.as_deref().is_some_and(|g| g.len() > 8), "0.7 must synthesize");
        // The wrap routes inherit the snap: Rx(π/2) = H·S·H.
        let r = t.synthesize_rx(Angle::PiRatio(1, 2)).expect("rx snap");
        assert_eq!(r.gates.as_deref(), Some("HSH"));
    }

    /// Issue #2 repro, exactly as reported: u1 at ε=1e-8 near Cliffords.
    /// Run: `cargo test --release --lib probe_issue2_repro -- --ignored --nocapture`
    #[test]
    #[ignore = "diagnostic probe, print-only"]
    fn probe_issue2_repro() {
        let synth = Synthesizer::new(1e-8, false);
        let cases: Vec<(&str, f64)> = vec![
            ("generic 1.0472", 1.0472),
            ("pi/2^23 near id", 3.74507e-07),
            ("near S", std::f64::consts::FRAC_PI_2 + 3.74507e-07),
            ("delta 1e-2", 1e-2),
            ("delta 1e-4", 1e-4),
            ("delta 5e-6", 5e-6),
            ("delta 2e-6", 2e-6),
            ("delta 1e-6", 1e-6),
            ("delta 1e-7", 1e-7),
        ];
        for (name, th) in cases {
            let t0 = std::time::Instant::now();
            let r = synth.synthesize_u1(Angle::Rad(th));
            match r {
                Some(r) => {
                    let tc = r.gates.as_deref().map_or(0, |g| g.matches('T').count() + g.matches('t').count());
                    eprintln!("{name:18} ok  T={tc:<4} dist={:.2e} {:?}", r.distance, t0.elapsed());
                }
                None => eprintln!("{name:18} NONE {:?}", t0.elapsed()),
            }
        }
    }

    /// √T + deep ε through the front door proves the native-Q routing:
    /// the 16D lattice pipeline is not validated below 1e-8, so only the
    /// native route can produce this result.
    #[test]
    fn test_u1_sqrt_t_native_deep() {
        let synth = Synthesizer::new(1e-12, true);
        let r = synth.synthesize_u1(Angle::Rad(0.7)).expect("deep sqrt_t u1");
        assert!(r.distance < 1e-12, "dist {:.3e}", r.distance);
        let gates = r.gates.expect("gates");
        let u = eval_gates_q(&gates).expect("eval");
        let d = diamond_distance_float(&u.to_float(), &rz_target(0.7));
        assert!(d < 1e-11, "re-verified dist {d:.3e}");
    }
}
