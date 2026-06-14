"""Choosing ε: tighter accuracy costs more gates.

Synthesizes one target at a range of ε and prints the T-count and achieved
distance, so you can see the accuracy/cost trade-off.

Run: python examples/choosing_epsilon.py
"""
import numpy as np
import cyclosynth

# A fixed Rz(0.3) target.
theta = 0.3
target = np.array([[np.exp(-1j * theta / 2), 0],
                   [0, np.exp(1j * theta / 2)]], dtype=np.complex128)

print(f"{'epsilon':>9}  {'T-count':>7}  {'distance':>10}")
print("-" * 30)
for epsilon in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
    result = cyclosynth.Synthesizer(epsilon).synthesize(target)
    if result:
        print(f"{epsilon:>9.0e}  {result.t_count:>7}  {result.distance:>10.2e}")
    else:
        print(f"{epsilon:>9.0e}  (no circuit found)")
