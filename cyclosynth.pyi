# Type stubs for the `cyclosynth` extension module.
#
# To ship these for mypy/IDE autocomplete, maturin needs a mixed layout:
# move this file to `python/cyclosynth/cyclosynth.pyi`, add an empty
# `python/cyclosynth/py.typed`, and set `python-source = "python"` under
# `[tool.maturin]`. Kept at the repo root as documentation until then.

class SynthResult:
    """Result of a synthesis run (same shape for Clifford+T and Clifford+√T)."""

    gates: str | None
    """Gate string, leftmost = first gate applied; None if extraction failed."""
    lde: int
    """Denominator exponent (search depth) of the synthesized unitary."""
    distance: float
    """Diamond distance to the target (< epsilon)."""
    @property
    def t_count(self) -> int: ...
    @property
    def q_count(self) -> int: ...
    @property
    def cost(self) -> float: ...
    def __bool__(self) -> bool: ...

class Synthesizer:
    """Single-qubit unitary synthesizer; minimizes T_count + 3·Q_count."""

    def __init__(
        self,
        epsilon: float,
        *,
        sqrt_t: bool = False,
        max_lde: int | None = None,
        min_lde: int | None = None,
        # Clifford+√T-only (raise if passed with sqrt_t=False):
        optimize_cost: bool | None = None,
        q_cost: float | None = None,
        lde_window: int | None = None,
        deadline_ms: int | None = None,
        seq_parity: bool | None = None,
    ) -> None: ...
    def synthesize_u3(
        self,
        theta: float | str,
        phi: float | str,
        lam: float | str,
    ) -> SynthResult | None:
        """Synthesize a ``U3(theta, phi, lambda)`` gate (qiskit/bqskit
        convention) from its angles, building the SU(2) target
        ``Rz(phi)·Ry(theta)·Rz(lambda)`` directly.

        Each angle is a float in radians, or a string. A string containing
        ``pi`` (whitespace ignored, optional ``*``) is a rational multiple of
        π: ``"pi"``, ``"3pi"``, ``"3*pi"``, ``"pi/8"``, ``"3*pi/4"``,
        ``"-2pi/3"``, ``"0.25pi"``. Any other string parses as a float.
        """
        ...
    def synthesize_zyz(
        self,
        alpha: float | str,
        beta: float | str,
        gamma: float | str,
    ) -> SynthResult | None:
        """Synthesize ``Rz(alpha)·Ry(beta)·Rz(gamma)`` from ZYZ Euler angles.

        Each angle accepts the same float / ``pi``-string forms as
        :meth:`synthesize_u3`.
        """
        ...
    @property
    def epsilon(self) -> float: ...
    @property
    def max_lde(self) -> int: ...
    @property
    def min_lde(self) -> int: ...
    @property
    def sqrt_t(self) -> bool: ...
