# Type stubs for the `cyclosynth` extension module.
#
# To ship these for mypy/IDE autocomplete, maturin needs a mixed layout:
# move this file to `python/cyclosynth/cyclosynth.pyi`, add an empty
# `python/cyclosynth/py.typed`, and set `python-source = "python"` under
# `[tool.maturin]`. Kept at the repo root as documentation until then.

def synthesize_u1(
    lam: float | str,
    epsilon: float,
    *,
    sqrt_t: bool = False,
) -> SynthResult | None:
    """Synthesize a ``U1(lam)`` gate (qiskit convention; ``Rz(lam)`` up to
    global phase) to diamond distance ``epsilon`` with default settings.

    The angle is a float in radians or an exact-π string like ``"pi/64"``;
    ``sqrt_t=True`` selects Clifford+√T (requires ``epsilon >= 1e-8``, else
    ``ValueError``). With ``sqrt_t=False``, ``epsilon < 1e-10`` emits a
    ``UserWarning`` (below the oracle-validated range) and proceeds. Returns
    ``None`` if no circuit was found. For repeated calls or tuning knobs, use
    :class:`Synthesizer`.
    """
    ...

def synthesize_u2(
    phi: float | str,
    lam: float | str,
    epsilon: float,
    *,
    sqrt_t: bool = False,
) -> SynthResult | None:
    """Synthesize a ``U2(phi, lam)`` gate (qiskit convention;
    ``U2(φ,λ) = U3(π/2, φ, λ)``) to diamond distance ``epsilon`` with default
    settings. Same angle forms / ``sqrt_t`` semantics as
    :func:`synthesize_u1`.
    """
    ...

def synthesize_u3(
    theta: float | str,
    phi: float | str,
    lam: float | str,
    epsilon: float,
    *,
    sqrt_t: bool = False,
) -> SynthResult | None:
    """Synthesize a ``U3(theta, phi, lam)`` gate (qiskit/bqskit convention)
    to diamond distance ``epsilon`` with default settings. Same angle forms /
    ``sqrt_t`` semantics as :func:`synthesize_u1`.
    """
    ...

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
    ) -> None:
        """``sqrt_t=True`` requires ``epsilon >= 1e-8`` (``ValueError``
        below); ``sqrt_t=False`` emits a ``UserWarning`` for
        ``epsilon < 1e-10`` (below the oracle-validated range).
        """
        ...
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
    def synthesize_u1(self, lam: float | str) -> SynthResult | None:
        """Synthesize a ``U1(lam)`` gate (``Rz(lam)`` up to global phase).
        The angle accepts the same float / ``pi``-string forms as
        :meth:`synthesize_u3`.
        """
        ...
    def synthesize_u2(
        self,
        phi: float | str,
        lam: float | str,
    ) -> SynthResult | None:
        """Synthesize a ``U2(phi, lam)`` gate (``U2(φ,λ) = U3(π/2, φ, λ)``).
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
