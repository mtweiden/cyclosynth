"""Z[ω]-lifted Clifford+√T baseline.

Phase 0 piece 2: cheap correctness baseline for the Clifford+√T effort.

Pipeline:
  1. Synthesize a target U via the existing Clifford+T `cyclosynth.Synthesizer`
     to precision ε. This yields a gate string over {H, S, T, X, Y, Z}.
  2. Lift to Clifford+√T by string substitution `T → QQ`, where Q = √T is
     defined (in the same convention as cyclosynth's S/T) as
         Q = diag(1, exp(i·π/8))
     so Q·Q = diag(1, exp(i·π/4)) = T exactly.
     The substituted string is therefore a faithful rewrite — no extra
     synthesis error is introduced.
  3. Verify at 60-decimal mpmath precision that the rebuilt Clifford+√T
     circuit is within ε of the target (Frobenius reformulation, same as
     `examples/verify.py`).

This is a *baseline*: the √T-count is exactly 2·T-count by construction.
A future native Clifford+√T synthesizer should beat it.
"""
import os
import sys

import mpmath as mp
import numpy as np
import cyclosynth
from random import random, seed

# Pull in verify.py's helpers so distances are computed via the exact same
# Frobenius path as the rest of the verification stack.
sys.path.insert(0, os.path.dirname(__file__))
from verify import (  # noqa: E402
    _gates_mp,
    diamond_mp,
    rebuild_mp,
    target_to_mp,
    rz,
    ry,
)


def _gates_mp_with_q():
    """`_gates_mp()` plus Q = diag(1, exp(i·π/8)) = √T at current mp.dps.

    Q is half a T: Q² = diag(1, exp(i·π/4)) = T. Note `mp.pi / 8`,
    NOT `mp.pi / 4` (that would be T itself).
    """
    gates = _gates_mp()
    j = mp.mpc(0, 1)
    one = mp.mpf(1)
    gates["Q"] = mp.matrix([[one, 0], [0, mp.exp(j * mp.pi / 8)]])
    return gates


def lift_t_to_qq(gates: str) -> str:
    """Rewrite a Clifford+T string into a Clifford+√T string by T → QQ.

    Q·Q = T algebraically, so this preserves the unitary exactly (no extra
    approximation). The √T-count of the lifted string is `2 · T-count` of
    the input, plus the unchanged Clifford gate count.
    """
    return gates.replace("T", "QQ")


def verify_lifted(gates_q: str, target: np.ndarray, gates_mp: dict) -> float:
    """mpmath diamond distance between the Clifford+√T rebuild and target."""
    rebuilt = rebuild_mp(gates_q, gates_mp)
    target_mp_ = target_to_mp(target)
    return float(diamond_mp(rebuilt, target_mp_))


def main():
    seed(0)
    epsilons = [1e-3, 1e-5, 1e-7]
    n_trials = 5

    # 60 decimals: ~200 mantissa bits. Even at ε=1e-7 the gate string is a
    # few hundred symbols (lifted ≈ 2× longer for the Q-doubling), so
    # accumulated mpmath round-off stays ~10⁻⁵⁸ — many orders below ε.
    mp.mp.dps = 60
    GATES_MP = _gates_mp_with_q()

    print(
        f"{'epsilon':>9}  {'trial':>5}  {'T-count':>7}  "
        f"{'sqrtT-count':>11}  {'distance':>11}  {'success':>7}"
    )
    print("-" * 64)

    all_ok = True
    for epsilon in epsilons:
        synth = cyclosynth.Synthesizer(epsilon=epsilon)
        for i in range(n_trials):
            alpha, beta, gamma = [2 * np.pi * random() for _ in range(3)]
            target = rz(alpha) @ ry(beta) @ rz(gamma)

            result = synth.synthesize(target)
            if result is None or result.gates is None:
                print(
                    f"{epsilon:>9.0e}  {i:>5d}  {'-':>7}  {'-':>11}  "
                    f"{'-':>11}  {'SYNTH-FAIL':>7}"
                )
                all_ok = False
                continue

            gates_t = result.gates
            t_count = gates_t.count("T")
            gates_q = lift_t_to_qq(gates_t)
            q_count = gates_q.count("Q")
            assert q_count == 2 * t_count, (
                f"sqrtT-count {q_count} != 2 * T-count {t_count}"
            )

            dist = verify_lifted(gates_q, target, GATES_MP)
            success = dist < epsilon
            all_ok = all_ok and success

            print(
                f"{epsilon:>9.0e}  {i:>5d}  {t_count:>7d}  {q_count:>11d}  "
                f"{dist:>11.3e}  {str(success):>7}"
            )

    print("-" * 64)
    print(f"all trials succeeded: {all_ok}")
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
