"""Clifford+√T cost optimization.

By default the Clifford+√T synthesizer returns the first circuit it finds
within ε. With `optimize_cost=True` it instead minimizes the weighted cost
T_count + q_cost·Q_count (q_cost defaults to 3.5), trading Q gates for T gates
where that lowers the total. This is the headline √T knob.

Run: python examples/optimize_cost.py
"""
import numpy as np
import cyclosynth

theta = 0.3
target = np.array([[np.exp(-1j * theta / 2), 0],
                   [0, np.exp(1j * theta / 2)]], dtype=np.complex128)
epsilon = 1e-6

first_hit = cyclosynth.Synthesizer(epsilon, sqrt_t=True)
optimized = cyclosynth.Synthesizer(epsilon, sqrt_t=True, optimize_cost=True, q_cost=3.5)

for label, synth in [("first-hit ", first_hit), ("optimized ", optimized)]:
    r = synth.synthesize(target)
    if not r:
        print(f"{label}: no circuit within ε={epsilon:.0e}")
        continue
    print(f"{label}: T={r.t_count} Q={r.q_count} cost={r.cost:.1f} (T + 3.5·Q)")
