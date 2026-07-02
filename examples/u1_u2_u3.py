"""The one-shot API: synthesize_u1 / synthesize_u2 / synthesize_u3.

Each function approximates its gate by a Clifford+T circuit (Clifford+sqrt(T)
with sqrt_t=True) within diamond distance epsilon. Angles are floats (radians)
or exact-pi strings like "pi/64" — strings stay exact below f64 precision,
which is what deep epsilon needs.
"""
import cyclosynth


def show(label, r):
    print(f"{label}\n  gates    = {r.gates}")
    print(f"  T={r.t_count} Q={r.q_count} lde={r.lde} distance={r.distance:.3e}\n")


# U1(lam) — phase gate, equals Rz(lam) up to global phase.
show('synthesize_u1("pi/64", 1e-6)',
     cyclosynth.synthesize_u1("pi/64", 1e-6))

# U2(phi, lam) = U3(pi/2, phi, lam); mix float and pi-string angles.
show('synthesize_u2(0.3, "pi/8", 1e-5)',
     cyclosynth.synthesize_u2(0.3, "pi/8", 1e-5))

# U3(theta, phi, lam) — general single-qubit gate (qiskit/bqskit convention).
show('synthesize_u3(1.0472, "3*pi/4", 2.5, 1e-5)',
     cyclosynth.synthesize_u3(1.0472, "3*pi/4", 2.5, 1e-5))

# Same target over Clifford+sqrt(T): Q (= sqrt(T)) gates appear, total cost
# T + 3*Q is minimized and never exceeds the Clifford+T cost.
show('synthesize_u3(1.0472, "3*pi/4", 2.5, 1e-5, sqrt_t=True)',
     cyclosynth.synthesize_u3(1.0472, "3*pi/4", 2.5, 1e-5, sqrt_t=True))

# ── epsilon-range policy ─────────────────────────────────────────────────────
# Clifford+sqrt(T) requires epsilon >= 1e-8 (the backend's validated range):
try:
    cyclosynth.synthesize_u1(0.5, 1e-9, sqrt_t=True)
except ValueError as e:
    print(f"sqrt_t=True at 1e-9 -> ValueError:\n  {e}\n")

# Clifford+T warns below the oracle-validated 1e-10 but proceeds. The warning
# fires at call/construction time; the synthesis itself would run for minutes
# at e.g. 1e-12, so it is not run here:
#   cyclosynth.synthesize_u1("pi/64", 1e-12)   # UserWarning, then a long run
import warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    cyclosynth.Synthesizer(5e-11)              # warns; nothing synthesized yet
print(f"Synthesizer(5e-11) -> UserWarning:\n  {w[0].message}")
