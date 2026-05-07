"""Compare Clifford+T vs Clifford+√T synthesis cost on the same targets.

Cost model:
    cost(T)  = 1.0
    cost(Q)  = 2.5    # √T gates cost more than T gates in this model
    cost(H/S/X/Y/Z) = 0  (Clifford gates are free in fault-tolerant accounting)

Per target:
    cost_T    = T_count                       (Clifford+T circuit)
    cost_sqrtT = T_count + 2.5 · Q_count       (Clifford+√T circuit)

Lower wins. The interesting question is whether Clifford+√T's denser gate
set (fewer total non-Clifford gates) makes up for Q being 2.5× as expensive
as T. Clifford+√T wins when (1.5·Q + ΔT) < 0, i.e. when the Q gates buy
enough T-count savings.
"""

import numpy as np
import cyclosynth
from random import random, seed


# --- Target generator ------------------------------------------------------

def rz(t):
    return np.array([[np.exp(-1j * t / 2), 0],
                     [0, np.exp(1j * t / 2)]], dtype=np.complex128)


def ry(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def u3(alpha, beta, gamma):
    return rz(alpha) @ ry(beta) @ rz(gamma)


# --- Cost model ------------------------------------------------------------

T_COST = 1.0
Q_COST = 2.5


def cost(gates: str) -> float:
    return T_COST * gates.count("T") + Q_COST * gates.count("Q")


# --- Comparison run --------------------------------------------------------

def main():
    epsilon = 1e-5
    n_targets = 10
    seed(42)

    synth_t = cyclosynth.Synthesizer(epsilon=epsilon)
    synth_q = cyclosynth.Synthesizer(epsilon=epsilon, sqrt_t=True)

    print(f"ε = {epsilon:.0e}, {n_targets} random U3 targets, "
          f"cost(T)={T_COST}, cost(Q)={Q_COST}")
    print()
    header = (
        f"{'#':>3}  "
        f"{'lde':>4}/{'lde':>3}  "
        f"{'T':>3} {'Q':>3}  "
        f"{'cost_T':>7}  {'cost_√T':>8}  "
        f"{'Δcost':>7}  {'win':>5}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for i in range(n_targets):
        alpha, beta, gamma = [2 * np.pi * random() for _ in range(3)]
        target = u3(alpha, beta, gamma)

        r_t = synth_t.synthesize(target)
        r_q = synth_q.synthesize(target)

        t_count_t = r_t.gates.count("T")
        q_count_t = r_t.gates.count("Q")     # always 0 for Clifford+T
        t_count_q = r_q.gates.count("T")
        q_count_q = r_q.gates.count("Q")

        cost_t = cost(r_t.gates)
        cost_q = cost(r_q.gates)
        delta = cost_q - cost_t
        winner = "√T" if cost_q < cost_t else ("T" if cost_t < cost_q else "tie")

        rows.append({
            "i": i,
            "lde_t": r_t.lde, "lde_q": r_q.lde,
            "t_count_t": t_count_t, "q_count_t": q_count_t,
            "t_count_q": t_count_q, "q_count_q": q_count_q,
            "cost_t": cost_t, "cost_q": cost_q,
            "delta": delta, "winner": winner,
        })

        print(
            f"{i:>3}  "
            f"{r_t.lde:>4}/{r_q.lde:>3}  "
            f"{t_count_q:>3} {q_count_q:>3}  "  # √T circuit's T/Q split
            f"{cost_t:>7.1f}  {cost_q:>8.1f}  "
            f"{delta:>+7.1f}  {winner:>5}"
        )

    # Aggregate.
    avg_cost_t = np.mean([r["cost_t"] for r in rows])
    avg_cost_q = np.mean([r["cost_q"] for r in rows])
    n_t_wins = sum(1 for r in rows if r["winner"] == "T")
    n_q_wins = sum(1 for r in rows if r["winner"] == "√T")
    n_tie = sum(1 for r in rows if r["winner"] == "tie")

    print()
    print(f"avg cost (T):  {avg_cost_t:.1f}")
    print(f"avg cost (√T): {avg_cost_q:.1f}")
    print(f"avg Δcost:     {avg_cost_q - avg_cost_t:+.1f}  "
          f"({(avg_cost_q - avg_cost_t) / avg_cost_t * 100:+.1f}%)")
    print(f"wins:  T={n_t_wins}  √T={n_q_wins}  tie={n_tie}")


if __name__ == "__main__":
    main()
