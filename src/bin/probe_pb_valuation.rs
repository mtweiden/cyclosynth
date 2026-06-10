//! Probe P-b (docs/certificate_validity.md §4): machine checks for the
//! Bloch-valuation bound  N ≥ 2k − 3, where k is the REDUCED √2-denominator
//! exponent of a Z[ζ₁₆] unitary U and N is the reduced √2-denominator
//! exponent of its SO(3)/Bloch image (`SO3Q::from_u2` + per-entry reduce).
//!
//! The full analytic proof is in docs/proof_pb_valuation.md. It needs no
//! residue enumeration: the adjugate identity collapses the nine Bloch
//! quadratics so that the z-column pair (az, bz) = (2·Re S, −2·Im S) with
//! S = 2·u11·conj(u21) already forces a numerator of λ-adic valuation
//! ≤ 16 + 2m₀ ≤ 22 < 24, i.e. √2-valuation ≤ 5, i.e. exp ≥ 2k − 3.
//!
//! This probe machine-verifies every step the proof uses:
//!   (0) unit pinning: v_λ(λ)=1, v_λ(γ)=2, v_λ(√2)=4, v_λ(2)=8, and the
//!       embedding R4 = Z[γ] ↪ Z[ζ₁₆] is numerically correct;
//!   (1) EXHAUSTIVE finite check (all 4095 nonzero R4 residues mod 8):
//!       R4::sqrt2_valuation == ⌊v_λ(embed)/4⌋, clamped at 6 (= the √2^6
//!       threshold the proof uses; stable mod 8 since 8 = √2^6);
//!   (2) corpus (default 50 000 reduced words, syllable count ≤ 12, with
//!       random Clifford left/right multipliers and random ζ^p global
//!       phases — well beyond E4's 3000 words at k ≤ 7):
//!         L1  adjugate/det-coset form: u12 = −ζ^j·conj(u21),
//!             u22 = ζ^j·conj(u11) for some j;
//!         L2  S := u11·c̄21 − u12·c̄22 == 2·u11·conj(u21) exactly;
//!         L3  k ≥ 1 ⟹ v_λ(u11) = v_λ(u21) = m₀ ≤ 3;
//!         L4  re3/im3 numerators really are S+S̄ and −i(S−S̄) (embedding
//!             identity), and min(v_λ(S+S̄), v_λ(S−S̄)) ≤ 16 + 2m₀ ≤ 22;
//!         L5  the reduced SO3Q exponents of entries (0,2) and (1,2) equal
//!             2k+2 − min(⌊v_λ/4⌋, 2k+2) (pins exp = √2-power count), and
//!             max of the two ≥ 2k−3 for k ≥ 2;
//!         L6  N ≥ 2k − 3 (the P-b claim), deficit statistics per k.
//!
//! Args: [<n_words> [<m_max> [<seed>]]]   (defaults 50000, 12, 0xB)

use std::collections::BTreeMap;

use cyclosynth::matrix::so3::{R4, SO3Q};
use cyclosynth::matrix::u2::U2Q;
use cyclosynth::rings::types::{INT_TWO, INT_ZERO};
use cyclosynth::rings::ZZeta;

// ─── RNG (same splitmix as probe_e4_identity) ────────────────────────────────

struct Xs(u64);
impl Xs {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
}

// ─── Ring helpers ────────────────────────────────────────────────────────────

fn zeta_pow(j: u32) -> ZZeta {
    let mut z = ZZeta::ONE;
    for _ in 0..(j % 16) {
        z = z * ZZeta::ZETA;
    }
    z
}

/// √2 = ζ² − ζ⁶.
fn sqrt2_z() -> ZZeta {
    ZZeta::from_i32(0, 0, 1, 0, 0, 0, -1, 0)
}

/// γ = 2cos(π/8) = ζ + ζ̄ = ζ − ζ⁷.
fn gamma_z() -> ZZeta {
    ZZeta::from_i32(0, 1, 0, 0, 0, 0, 0, -1)
}

/// 2/λ = ∏_{j ∈ {3,5,…,15}} (1 − ζ^j)  (since ∏_{j odd} (1−ζ^j) = Φ₁₆(1) = 2).
fn two_over_lambda() -> ZZeta {
    let mut k = ZZeta::ONE;
    for j in [3u32, 5, 7, 9, 11, 13, 15] {
        k = k * (ZZeta::ONE - zeta_pow(j));
    }
    k
}

/// λ | x  ⟺  x ↦ 0 in the residue field F₂ (ζ ↦ 1)  ⟺  Σ coords even.
fn divisible_by_lambda(x: ZZeta) -> bool {
    let s = x.a + x.b + x.c + x.d + x.e + x.f + x.g + x.h;
    s % INT_TWO == INT_ZERO
}

/// λ-adic valuation of a NONZERO x ∈ Z[ζ₁₆]: x/λ = (x · 2/λ)/2 exactly.
fn v_lambda(mut x: ZZeta, two_over_l: ZZeta) -> u32 {
    assert!(x != ZZeta::ZERO, "v_lambda of zero");
    let mut v = 0u32;
    while divisible_by_lambda(x) {
        x = (x * two_over_l).div2(1);
        v += 1;
    }
    v
}

/// Embed R4 = Z[γ] (basis {1, √2, γ, γ√2}) into Z[ζ₁₆].
fn embed_r4(x: R4) -> ZZeta {
    let s2 = sqrt2_z();
    let g = gamma_z();
    ZZeta::ONE.scale(x.0) + s2.scale(x.1) + g.scale(x.2) + (g * s2).scale(x.3)
}

/// 2·Re(z) in R4 coordinates — identical to the closure in SO3Q::from_u2.
fn re3(z: ZZeta) -> R4 {
    R4(INT_TWO * z.a, z.c - z.g, z.b - z.h - z.d + z.f, z.d - z.f)
}

/// 2·Im(z) in R4 coordinates — identical to the closure in SO3Q::from_u2.
fn im3(z: ZZeta) -> R4 {
    R4(INT_TWO * z.e, z.c + z.g, z.d + z.f - z.b - z.h, z.b + z.h)
}

// ─── Word generation (FGKM syllables, as in probe_e4_identity) ───────────────

fn syllable(axis: usize, a: u32) -> U2Q {
    let mut d = U2Q::eye();
    for _ in 0..a {
        d = d * U2Q::q();
    }
    match axis {
        0 => (U2Q::h() * d * U2Q::h()).reduced(),
        1 => (U2Q::s() * U2Q::h() * d * U2Q::h() * U2Q::s().dagger()).reduced(),
        _ => d,
    }
}

fn random_word(rng: &mut Xs, m_max: u32) -> U2Q {
    let m = 1 + (rng.next() % m_max as u64) as u32;
    let mut u = U2Q::eye();
    let mut prev_axis = 3usize;
    for _ in 0..m {
        let mut axis = (rng.next() % 3) as usize;
        while axis == prev_axis {
            axis = (rng.next() % 3) as usize;
        }
        prev_axis = axis;
        let a = 1 + (rng.next() % 3) as u32;
        u = (u * syllable(axis, a)).reduced();
    }
    u
}

fn random_clifford(rng: &mut Xs) -> U2Q {
    let len = (rng.next() % 7) as usize;
    let mut c = U2Q::eye();
    for _ in 0..len {
        c = if rng.next() % 2 == 0 { c * U2Q::h() } else { c * U2Q::s() };
    }
    c
}

fn apply_phase(u: U2Q, p: u32) -> U2Q {
    let z = zeta_pow(p);
    U2Q::new(z * u.u11, z * u.u12, z * u.u21, z * u.u22, u.k)
}

// ─── Main ────────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let n_words: usize = args.first().and_then(|s| s.parse().ok()).unwrap_or(50_000);
    let m_max: u32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(12);
    let seed: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0xB);

    let tol = two_over_lambda();
    let lambda = ZZeta::ONE - ZZeta::ZETA;

    // ── Check 0: unit pinning ────────────────────────────────────────────────
    assert_eq!(lambda * tol, ZZeta::ONE.scale(INT_TWO), "2/λ identity");
    assert_eq!(v_lambda(lambda, tol), 1, "v_λ(λ) = 1");
    assert_eq!(v_lambda(gamma_z(), tol), 2, "v_λ(γ) = 2");
    assert_eq!(v_lambda(sqrt2_z(), tol), 4, "v_λ(√2) = 4");
    assert_eq!(v_lambda(ZZeta::ONE.scale(INT_TWO), tol), 8, "v_λ(2) = 8");
    assert_eq!(v_lambda(ZZeta::ONE, tol), 0, "v_λ(1) = 0");
    for j in 0..16 {
        assert_eq!(v_lambda(zeta_pow(j), tol), 0, "v_λ(ζ^j) = 0");
    }
    // Embedding sanity: γ ↦ the positive real 2cos(π/8); √2 ↦ √2.
    let gf = gamma_z().to_complex();
    assert!((gf.re - 2.0 * (std::f64::consts::PI / 8.0).cos()).abs() < 1e-12 && gf.im.abs() < 1e-12);
    let r4g = R4::from_i32(0, 0, 1, 0).to_f64();
    assert!((gf.re - r4g).abs() < 1e-12, "R4 γ and ZZeta γ agree numerically");
    let sf = sqrt2_z().to_complex();
    assert!((sf.re - std::f64::consts::SQRT_2).abs() < 1e-12 && sf.im.abs() < 1e-12);
    println!("check 0 (units): v_λ(λ,γ,√2,2) = (1,2,4,8); embeddings agree.  OK");

    // ── Check 1: exhaustive conversion check on R4 residues mod 8 ───────────
    // 8 = √2^6, so min(v_√2, 6) is well-defined on residues mod 8: this
    // exhaustively certifies v_√2(x) = ⌊v_λ(embed x)/4⌋ at every level ≤ 6,
    // which is exactly the range the P-b proof uses (threshold v_√2 ≤ 5).
    let mut n_classes = 0usize;
    for a in 0..8 {
        for b in 0..8 {
            for c in 0..8 {
                for d in 0..8 {
                    if a == 0 && b == 0 && c == 0 && d == 0 {
                        continue;
                    }
                    let x = R4::from_i32(a, b, c, d);
                    let v_r4 = x.sqrt2_valuation().min(6);
                    let v_emb = (v_lambda(embed_r4(x), tol) / 4).min(6);
                    assert_eq!(
                        v_r4, v_emb,
                        "conversion mismatch at R4({a},{b},{c},{d}): R4 {v_r4} vs λ {v_emb}"
                    );
                    n_classes += 1;
                }
            }
        }
    }
    println!("check 1 (finite, exhaustive): v_√2 = ⌊v_λ/4⌋ on all {n_classes} nonzero R4 residues mod 8 (clamped at 6).  OK");

    // ── Check 2: corpus ──────────────────────────────────────────────────────
    let mut rng = Xs(seed);
    let mut stats: BTreeMap<u32, (usize, u32, usize)> = BTreeMap::new(); // k → (count, min N, #equality)
    let mut worst_d: i64 = i64::MIN; // D = 2k − N, proof says ≤ 3
    let mut max_k = 0u32;

    for i in 0..n_words {
        let w = random_word(&mut rng, m_max);
        // Variants: bare word / Clifford-conjugated / phase-multiplied.
        let u = match i % 3 {
            0 => w,
            1 => (random_clifford(&mut rng) * w * random_clifford(&mut rng)).reduced(),
            _ => apply_phase(
                (random_clifford(&mut rng) * w).reduced(),
                (rng.next() % 16) as u32,
            )
            .reduced(),
        };
        let k = u.k;
        max_k = max_k.max(k);
        let (a, b, c, d) = (u.u11, u.u12, u.u21, u.u22);

        // L1: adjugate / det-coset form.
        let mut e_unit = None;
        for j in 0..16 {
            let e = zeta_pow(j);
            if b == -(e * c.conj()) && d == e * a.conj() {
                e_unit = Some(j);
                break;
            }
        }
        assert!(e_unit.is_some(), "L1 FAIL: no ζ^j adjugate form at word {i} (k={k})");

        // L2: S = 2·a·c̄ exactly.
        let s = a * c.conj() - b * d.conj();
        assert_eq!(s, (a * c.conj()).scale(INT_TWO), "L2 FAIL: S ≠ 2ac̄ at word {i}");

        // L4 (embedding identities, all k): re3/im3 are 2Re, 2Im.
        let s_plus = s + s.conj(); // 2 Re S
        let s_minus = (s - s.conj()) * ZZeta::NEG_I; // 2 Im S
        assert_eq!(embed_r4(re3(s)), s_plus, "L4 FAIL: embed(re3 S) ≠ S+S̄ at word {i}");
        assert_eq!(embed_r4(im3(s)), s_minus, "L4 FAIL: embed(im3 S) ≠ −i(S−S̄) at word {i}");

        if k >= 1 {
            // L3: balanced column valuations.
            assert!(a != ZZeta::ZERO && c != ZZeta::ZERO, "L3 FAIL: zero entry at k≥1, word {i}");
            let va = v_lambda(a, tol);
            let vc = v_lambda(c, tol);
            assert!(
                va == vc && va <= 3,
                "L3 FAIL: v(a)={va}, v(c)={vc} at word {i} (k={k})"
            );
            // L4 (valuation chain): min(v(S+S̄), v(S−S̄)) ≤ 16 + 2m₀ ≤ 22.
            let vp = if s_plus == ZZeta::ZERO { u32::MAX } else { v_lambda(s_plus, tol) };
            let vm = if s_minus == ZZeta::ZERO { u32::MAX } else { v_lambda(s_minus, tol) };
            let vmin = vp.min(vm);
            assert!(
                vmin <= 16 + 2 * va && vmin <= 22,
                "L4 FAIL: min(v(S±S̄))={vmin} > 16+2m₀={} at word {i} (k={k})",
                16 + 2 * va
            );
        }

        // L5/L6: SO3Q exponents.
        let mut so3 = SO3Q::from_u2(&u);
        so3.reduce();
        let n = so3.maximum_denominator_exponent();
        let exp_az = so3.e[2].exp; // (0,2) numerator = re3(S)
        let exp_bz = so3.e[5].exp; // (1,2) numerator = −im3(S)

        // L5: exp really is 2k+2 − min(⌊v_λ(num)/4⌋, 2k+2)  (pins the units).
        for (idx, num_z) in [(2usize, s_plus), (5usize, s_minus)] {
            let expect = if num_z == ZZeta::ZERO {
                0
            } else {
                (2 * k + 2).saturating_sub((v_lambda(num_z, tol) / 4).min(2 * k + 2))
            };
            assert_eq!(
                so3.e[idx].exp, expect,
                "L5 FAIL: entry {idx} exp {} ≠ predicted {expect} at word {i} (k={k})",
                so3.e[idx].exp
            );
        }
        if k >= 2 {
            assert!(
                exp_az.max(exp_bz) as i64 >= 2 * k as i64 - 3,
                "L5 FAIL: z-column max exp {} < 2k−3={} at word {i}",
                exp_az.max(exp_bz),
                2 * k as i64 - 3
            );
        }

        // L6: the P-b claim itself.
        assert!(
            n as i64 >= 2 * k as i64 - 3,
            "L6 FAIL (P-b violated): N={n} < 2k−3={} at word {i} (k={k})",
            2 * k as i64 - 3
        );
        let d_def = 2 * k as i64 - n as i64;
        worst_d = worst_d.max(d_def);
        let ent = stats.entry(k).or_insert((0, u32::MAX, 0));
        ent.0 += 1;
        ent.1 = ent.1.min(n);
        if n as i64 == 2 * k as i64 - 3 {
            ent.2 += 1;
        }
    }

    println!(
        "check 2 (corpus): {n_words} words (m ≤ {m_max}, Clifford/phase variants, seed {seed:#x}), max k = {max_k}.  ALL LEMMAS OK"
    );
    println!("\n  k   count   min N   2k−3   #tight(N=2k−3)");
    for (k, (cnt, min_n, tight)) in &stats {
        println!(
            "  {k:>2}  {cnt:>6}  {min_n:>6}  {:>5}  {tight:>6}",
            2 * *k as i64 - 3
        );
    }
    println!("\nworst observed deficit D = 2k − N: {worst_d} (proof bound: ≤ 3)");
    println!("\nALL CHECKS PASSED — N ≥ 2k − 3 holds; see docs/proof_pb_valuation.md");
}
