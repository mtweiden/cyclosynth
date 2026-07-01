"""Basic synthesis: approximate a unitary with Clifford+T and Clifford+√T.

Run: python examples/synth.py
"""
import numpy as np
import cyclosynth

rng = np.random.default_rng(0)  # seeded for reproducibility


def u3(alpha, beta, gamma):
    """U3(α, β, γ) = Rz(α)·Ry(β)·Rz(γ)."""
    rz = lambda t: np.array([[np.exp(-1j * t / 2), 0],
                             [0, np.exp(1j * t / 2)]], dtype=np.complex128)
    c, s = np.cos(beta / 2), np.sin(beta / 2)
    ry = np.array([[c, -s], [s, c]], dtype=np.complex128)
    return rz(alpha) @ ry @ rz(gamma)


epsilon = 1e-5
alpha, beta, gamma = 2 * np.pi * rng.random(3)
target = u3(alpha, beta, gamma)

for label, synth in [
    ("Clifford+T ", cyclosynth.Synthesizer(epsilon)),
    ("Clifford+√T", cyclosynth.Synthesizer(epsilon, sqrt_t=True)),
]:
    result = synth.synthesize_zyz(alpha, beta, gamma)
    if not result:                       # None, or no gates extracted
        print(f"{label}: no circuit within ε={epsilon:.0e}")
        continue
    print(f"{label}: T={result.t_count} Q={result.q_count} "
          f"cost={result.cost:.1f} lde={result.lde} "
          f"distance={result.distance:.2e}")
    print(f"            gates = {result.gates}")
    assert result.distance < epsilon
