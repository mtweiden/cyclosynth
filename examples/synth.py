import numpy as np
import cyclosynth
from random import random

# Build a single-qubit unitary as U3(α, β, γ) = Rz(α) · Ry(β) · Rz(γ).
# Angles fixed for reproducibility (originally drawn from uniform(0, 2π)).
def rz(t):
    return np.array([[np.exp(-1j * t / 2), 0],
                     [0,                    np.exp(1j * t / 2)]],
                    dtype=np.complex128)

def ry(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -s],
                     [s,  c]], dtype=np.complex128)

epsilon = 1e-6
synth = cyclosynth.Synthesizer(epsilon=epsilon)

for _ in range(10):
    alpha, beta, gamma = [2 * np.pi * random() for _ in range(3)]
    target = rz(alpha) @ ry(beta) @ rz(gamma)

    # Approximate to within ε = 1e-5 in diamond distance.
    result = synth.synthesize(target)
    t_count = result.gates.count("T") if result.gates else 0

    print("=" * 60)
    print(f"U3({alpha:.3f}, {beta:.3f}, {gamma:.3f})") # target unitary
    print(f"  gates    = {result.gates}")      # Clifford+T sequence over {H, S, T, X, Y, Z}
    print(f"  T-count  = {t_count}")
    print(f"  distance = {result.distance:e}") # < epsilon
    print("=" * 60)

    assert result.distance < epsilon
