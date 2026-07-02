"""Compare Clifford+T vs Clifford+√T synthesis cost on the same targets."""
import numpy as np
import cyclosynth

rng = np.random.default_rng(42)


def u3(alpha, beta, gamma):
    rz = lambda t: np.array([[np.exp(-1j * t / 2), 0],
                             [0, np.exp(1j * t / 2)]], dtype=np.complex128)
    c, s = np.cos(beta / 2), np.sin(beta / 2)
    ry = np.array([[c, -s], [s, c]], dtype=np.complex128)
    return rz(alpha) @ ry @ rz(gamma)


def haar_u3(rng):
    """Draw a Haar-random SU(2) in ZYZ form.

    The Haar measure on SU(2) is ∝ sin(β) dα dβ dγ, so α and γ are uniform on
    [0, 2π) but β must be sine-weighted on [0, π]. Drawing β uniformly is *not*
    Haar — it over-samples near-diagonal unitaries.
    """
    alpha, gamma = 2 * np.pi * rng.random(2)
    beta = np.arccos(1.0 - 2.0 * rng.random())
    return u3(alpha, beta, gamma), (alpha, beta, gamma)


def main():
    epsilon, n_targets = 1e-8, 10
    synth_t = cyclosynth.Synthesizer(epsilon)
    synth_q = cyclosynth.Synthesizer(epsilon, sqrt_t=True, optimize_cost=True)

    print(f"ε = {epsilon:.0e}, {n_targets} Haar-random U3 targets, cost = T + 3·Q\n")
    header = (f"{'#':>3}  {'T':>3} {'Q':>3}  {'cost_T':>7} {'cost_√T':>8}  "
              f"{'dist_T':>9} {'dist_√T':>9}  {'win':>4}")
    print(header)
    print("-" * len(header))

    costs_t, costs_q, dists_t, dists_q = [], [], [], []
    wins = {"T": 0, "√T": 0, "tie": 0}
    for i in range(n_targets):
        target, angles = haar_u3(rng)

        # ========================================
        # DOING SYNTHESIS HERE
        # ========================================
        r_t, r_q = synth_t.synthesize_zyz(*angles), synth_q.synthesize_zyz(*angles)
        # ========================================

        if not (r_t and r_q):
            print(f"{i:>3}  (no circuit within ε)")
            continue
        win = "√T" if r_q.cost < r_t.cost else ("T" if r_t.cost < r_q.cost else "tie")
        wins[win] += 1
        costs_t.append(r_t.cost)
        costs_q.append(r_q.cost)
        dists_t.append(r_t.distance)
        dists_q.append(r_q.distance)
        # r_q.t_count / r_q.q_count are the √T circuit's gate split.
        print(f"{i:>3}  {r_q.t_count:>3} {r_q.q_count:>3}  "
            f"{r_t.cost:>7.1f} {r_q.cost:>8.1f}  "
            f"{r_t.distance:>9.2e} {r_q.distance:>9.2e}  {win:>4}")

    print(f"\navg cost:  T={np.mean(costs_t):.1f}  √T={np.mean(costs_q):.1f}  "
          f"({(np.mean(costs_q) / np.mean(costs_t) - 1) * 100:+.1f}%)")
    print(f"avg dist:  T={np.mean(dists_t):.2e}  √T={np.mean(dists_q):.2e}  (ε={epsilon:.0e})")
    print(f"wins:  T={wins['T']}  √T={wins['√T']}  tie={wins['tie']}")


if __name__ == "__main__":
    main()
