"""Interactive Clifford+T vs Clifford+√T comparison that prints the gate strings.

Run: python examples/compare_t_vs_sqrtt_gates.py

At the prompt, enter U3 angles (in radians) as 1, 2, or 3 numbers:

    1 number  -> U3(α, 0, 0)
    2 numbers -> U3(α, β, 0)
    3 numbers -> U3(α, β, γ)

Or press 'r' for a Haar-random target, 'd' to change the target distance,
'q' to quit.
"""
import numpy as np
import numpy.typing as npt
import cyclosynth

rng = np.random.default_rng()


def u3(alpha: float, beta: float, gamma: float) -> npt.NDArray[np.complex128]:
    """U3(α, β, γ) = Rz(α)·Ry(β)·Rz(γ)."""
    rz = lambda t: np.array([[np.exp(-1j * t / 2), 0],
                             [0, np.exp(1j * t / 2)]], dtype=np.complex128)
    c, s = np.cos(beta / 2), np.sin(beta / 2)
    ry = np.array([[c, -s], [s, c]], dtype=np.complex128)
    return rz(alpha) @ ry @ rz(gamma)


def haar_u3(rng: np.random.Generator) -> tuple[float, float, float]:
    """Draw a Haar-random SU(2) in ZYZ form.

    The Haar measure on SU(2) is ∝ sin(β) dα dβ dγ, so α and γ are uniform on
    [0, 2π) but β must be sine-weighted on [0, π].
    """
    alpha, gamma = 2 * np.pi * rng.random(2)
    beta = np.arccos(1.0 - 2.0 * rng.random())
    return (alpha, beta, gamma)


def format_gates(gates: object, indent: int = 4, width: int = 80) -> str:
    """Render a gate sequence wrapped at `width` with no spaces between gates.

    The synthesizer returns gates as one unspaced string (each char is a gate);
    we chunk that string into lines so it wraps without inserting separators.
    """
    seq = gates if isinstance(gates, str) else "".join(str(g) for g in gates)
    if not seq:
        return " " * indent + "(empty)"
    chunk = max(1, width - indent)
    lines = [" " * indent + seq[i:i + chunk] for i in range(0, len(seq), chunk)]
    return "\n".join(lines)


def report_stats(label: str, result: cyclosynth.SynthResult | None,
                 epsilon: float) -> None:
    if not result:
        print(f"  {label}: no circuit within ε={epsilon:.0e}")
        return
    print(f"  {label}: T={result.t_count} Q={result.q_count} "
          f"cost={result.cost:.1f}  distance={result.distance:.2e}")


def report_gates(label: str, result: cyclosynth.SynthResult | None) -> None:
    if not result:
        return
    print(f"  {label} ({len(result.gates)} gates):")
    print(format_gates(result.gates))


def build_synths(epsilon: float):
    synth_t = cyclosynth.Synthesizer(epsilon)
    synth_q = cyclosynth.Synthesizer(epsilon, sqrt_t=True, optimize_cost=True)
    return synth_t, synth_q


def prompt_distance(current: float) -> float:
    """Ask for an integer n in [1, 8]; the new distance is ε = 10^-n.

    The cap of n = 8 keeps ε ≥ 1e-8. Returns the unchanged distance on a
    blank/invalid entry.
    """
    n_current = round(-np.log10(current))
    raw = input(f"  distance exponent n (ε = 1e-n, 1-8) [{n_current}]> ").strip()
    if not raw:
        return current
    try:
        n = int(raw)
    except ValueError:
        print("  enter a single integer between 1 and 8\n")
        return current
    if not (1 <= n <= 9):
        print("  n must be between 1 and 8 (ε no smaller than 1e-8)\n")
        return current
    return 10.0 ** (-n)


def main() -> None:
    epsilon = 1e-5
    synth_t, synth_q = build_synths(epsilon)

    print(f"ε = {epsilon:.0e}, cost = T + 3·Q")
    print("Enter 1-3 angles in radians (α [β [γ]]), 'r' for random, "
          "'d' to set distance, 'q' to quit.\n")

    while True:
        try:
            line = input("angles> ").strip()
        except EOFError:
            break
        if not line:
            continue
        if line.lower() == "q":
            break
        if line.lower() == "d":
            new_eps = prompt_distance(epsilon)
            if new_eps != epsilon:
                epsilon = new_eps
                synth_t, synth_q = build_synths(epsilon)
                print(f"  ε = {epsilon:.0e}\n")
            continue

        angles: tuple[float, ...]
        if line.lower() == "r":
            angles = haar_u3(rng)
        else:
            try:
                nums = [float(x) for x in line.replace(",", " ").split()]
            except ValueError:
                print("  could not parse angles; enter 1-3 numbers, 'r', or 'q'\n")
                continue
            if not (1 <= len(nums) <= 3):
                print("  enter 1, 2, or 3 angles\n")
                continue
            angles = tuple(nums) + (0.0,) * (3 - len(nums))

        target = u3(*angles)
        print(f"  α={angles[0]:.6f}  β={angles[1]:.6f}  γ={angles[2]:.6f}")

        # ========================================
        # DOING SYNTHESIS HERE
        # ========================================
        r_t, r_q = synth_t.synthesize_zyz(*angles), synth_q.synthesize_zyz(*angles)
        # ========================================

        report_stats("Clifford+T ", r_t, epsilon)
        report_stats("Clifford+√T", r_q, epsilon)
        if r_t and r_q:
            win = "√T" if r_q.cost < r_t.cost else ("T" if r_t.cost < r_q.cost else "tie")
            print(f"  cost: T={r_t.cost:.1f}  √T={r_q.cost:.1f}  win={win}")

        if r_t or r_q:
            print("\n  gate sequences:")
            report_gates("Clifford+T ", r_t)
            report_gates("Clifford+√T", r_q)
        print()


if __name__ == "__main__":
    main()
