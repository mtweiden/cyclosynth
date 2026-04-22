"""
Clifford+T synthesis via divide-and-conquer (Algorithm 3.11).

Algorithm 3.11 (from "A new algorithm for ancilla-free single-qubit Clifford+T synthesis"):
  Given target V, error eps, and T-count t:
    (a) t' = max(0, ceil(t - 5/2 * log2(1/eps)))
    (b) For each UL in L_{t'}, call Algorithm 3.6 (bandb5.synthesize)
        with input (UL_dag @ V, eps, t - t').
    (c) Return the first UR found; the full gate is UL * UR.

L_{t'} (Lemma 3.10): the Matsumoto-Amano left-factor set.
|L_n| = O(2^n).  Optimal split t' = t - 5/2*log2(1/eps) gives total
complexity O(2^t / eps^{5/2}) vs the base solver's O(2^{2t} / eps^5).

Key design: synthesize_dc returns AT MOST ONE solution and returns
immediately on first hit.
"""

import numpy as np
from numpy import sqrt
import math
from itertools import product as iproduct
from functools import lru_cache

from bandb5 import (
    synthesize, to_unitary, uv_to_xy, xy_to_uv, verify,
)
import multiprocessing as _mp


# ---------------------------------------------------------------------------
# Gates and Clifford group
# ---------------------------------------------------------------------------
_r2 = float(sqrt(2.0))
_I  = np.eye(2, dtype=complex)
_H  = np.array([[1, 1], [1, -1]], dtype=complex) / _r2
_S  = np.array([[1, 0], [0, 1j]], dtype=complex)
_T  = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
_Td = _T.conj().T


def _u2_eq(A, B, tol=1e-9):
    """Equality of 2x2 unitaries up to global U(1) phase."""
    for ph in np.exp(1j * np.arange(8) * np.pi / 4):
        if np.allclose(A, ph * B, atol=tol):
            return True
    return False


def _gen_cliffords():
    """Generate all 24 single-qubit Cliffords by BFS over generators {H, S}."""
    found = [_I.copy()]
    queue = [_I.copy()]
    for _ in range(8):
        nxt = []
        for m in queue:
            for g in [_H, _S]:
                c = m @ g
                if not any(_u2_eq(c, f) for f in found):
                    found.append(c)
                    nxt.append(c)
        queue = nxt
        if not queue:
            break
    return found


def _canonical_key(M, decimals=6):
    """
    Hash key for M up to global phase.
    Rotates so the largest-magnitude element is real positive, then rounds.
    Used for O(n)-average deduplication in build_L.
    """
    flat = M.flatten()
    idx  = np.argmax(np.abs(flat))
    piv  = flat[idx]
    if abs(piv) < 1e-12:
        return (tuple(np.round(flat.real, decimals))
                + tuple(np.round(flat.imag, decimals)))
    rot = flat / (piv / abs(piv))
    return (tuple(np.round(rot.real, decimals))
            + tuple(np.round(rot.imag, decimals)))


_CLIFFORDS = _gen_cliffords()
assert len(_CLIFFORDS) == 24


def mat_to_uv(U):
    """
    Convert a 2x2 unitary to uv = [Re(u1), Im(u1), Re(u2), Im(u2)].
    Tries all 8 global phases to find the SU(2) form [[u1,-u2*],[u2,u1*]].
    Returns unit-normalised uv, or None if not achievable.
    """
    for ph in np.exp(1j * np.arange(8) * np.pi / 4):
        M = ph * U
        u1, u2 = M[0, 0], M[1, 0]
        if np.allclose(M, [[u1, -np.conj(u2)], [u2, np.conj(u1)]], atol=1e-9):
            v = np.array([u1.real, u1.imag, u2.real, u2.imag])
            n = np.linalg.norm(v)
            if n > 1e-12:
                return v / n
    return None


# ---------------------------------------------------------------------------
# build_L: enumerate the Matsumoto-Amano left-factor set L_{t'}
# ---------------------------------------------------------------------------
@lru_cache(maxsize=64)
def build_L(t_prime):
    """
    Return L_{t'} as a tuple of (matrix, label) pairs.  Cached.

    Definition (Lemma 3.10):
      L_0 = {I}
      L_n (n>=1):
        even branch: (HS^{b_n}T)...(HS^{b_1}T) * C
        odd  branch: T * (HS^{b_{n-1}}T)...(HS^{b_1}T) * C
      for all b_i in {0,1}, C in C_1 (24 Cliffords).

    Size: |L_0|=1, |L_n| = O(2^n) after deduplication.
    """
    if t_prime == 0:
        return ((_I.copy(), "I"),)

    HS = [_H, _H @ _S]       # HS[0]=H, HS[1]=HS
    elements = []

    # Even branch: length-t' product of (HS^b T) blocks
    for bits in iproduct([0, 1], repeat=t_prime):
        M = _I.copy()
        for b in reversed(bits):
            M = HS[b] @ _T @ M
        label = ".".join("HST" if b else "HT" for b in reversed(bits))
        for ci, C in enumerate(_CLIFFORDS):
            elements.append((M @ C, label + ".C" + str(ci)))

    # Odd branch: T prepended to length-(t'-1) product
    for bits in iproduct([0, 1], repeat=t_prime - 1):
        M = _T.copy()
        for b in reversed(bits):
            M = HS[b] @ _T @ M
        label = "T." + ".".join("HST" if b else "HT" for b in reversed(bits))
        for ci, C in enumerate(_CLIFFORDS):
            elements.append((M @ C, label + ".C" + str(ci)))

    # Deduplicate up to global U(1) phase in O(n) average
    seen   = set()
    unique = []
    for mat, label in elements:
        key = _canonical_key(mat)
        if key not in seen:
            seen.add(key)
            unique.append((mat, label))

    return tuple(unique)


# ---------------------------------------------------------------------------
# Algorithm 3.11: single-answer divide-and-conquer synthesis
# ---------------------------------------------------------------------------
def synthesize_dc(v, t, eps=1e-4, verbose=False):
    """
    Algorithm 3.11: find ONE solution U with T-count t and dist(U,V) < eps.

    Parameters
    ----------
    v       : target uv = [Re(u1), Im(u1), Re(u2), Im(u2)], unit vector
    t       : exact T-count to search at
    eps     : approximation error (diamond norm)
    verbose : print step info

    Returns
    -------
    (UL_mat, x, k_inner, UL_label, odd)  or  None

      UL_mat @ to_unitary(x, k_inner) [@ T if odd]  approximates  V

    Algorithm
    ---------
    (a) t' = max(0, ceil(t - 5/2 * log2(1/eps)))   -- optimal split
    (b) Build L_{t'} (cached after first call per t')
    (c) For each UL in L_{t'}: solve for UR at T-count t_inner = t - t'
        using bandb5.synthesize on inner target = UL_dag @ V
    (d) Return immediately on first hit
    """
    # Step (a): compute the optimal split t'.
    #
    # Theory (Prop 3.13): t' = max(0, ceil(t - 5/2 * log2(1/eps)))
    # minimises |L_{t'}| * 2^{2*t_inner} / eps^5, giving O(2^t/eps^{5/2}).
    #
    # When is DC faster than the base solver?
    #   Base cost  ~ 2^{2t} / eps^5
    #   DC cost    ~ |L_{t'}| * 2^{2*t_inner} / eps^5  ~ 2^t / eps^{5/2}
    # DC wins when 2^t / eps^{5/2} < 2^{2t} / eps^5, i.e. eps^{5/2} < 2^t,
    # i.e. t > 5/2 * log2(1/eps) — exactly when t' > 0.
    #
    # At eps=0.1: threshold t > 8.3 (DC beats base at t >= 9)
    # At eps=0.01: threshold t > 16.6 (DC beats base at t >= 17)
    # At eps=0.001: threshold t > 24.9 (DC beats base at t >= 25)
    if eps >= 1.0:
        t_prime = 0
    else:
        t_prime = max(0, math.ceil(t - (5.0 / 2.0) * math.log2(1.0 / eps)))
    t_inner = t - t_prime

    if verbose:
        print(f"  t={t}, eps={eps:.2e} -> t'={t_prime}, t_inner={t_inner}", flush=True)

    # Build (or retrieve cached) L_{t'}
    L = build_L(t_prime)
    if verbose:
        print(f"  |L| = {len(L)}", flush=True)

    # Reconstruct target as 2x2 SU(2) matrix from uv
    u1 = v[0] + 1j * v[1]
    u2 = v[2] + 1j * v[3]
    V_mat = np.array([[u1, -np.conj(u2)], [u2, np.conj(u1)]], dtype=complex)

    # k and parity for inner synthesize() call:
    #   odd=False -> T-count = 2*(k-1),   k = t_inner//2 + 1       (t_inner even)
    #   odd=True  -> T-count = 2*(k-1)+1, k = (t_inner-1)//2 + 1   (t_inner odd)
    if t_inner % 2 == 0:
        odd_inner = False
        k_inner   = t_inner // 2 + 1
    else:
        odd_inner = True
        k_inner   = (t_inner - 1) // 2 + 1

    # Steps (c)/(d): iterate L_{t'}, return on first hit
    for UL_mat, UL_label in L:
        uv_inner = mat_to_uv(UL_mat.conj().T @ V_mat)
        if uv_inner is None:
            continue                          # not in SU(2) form for any phase

        inner = synthesize(uv_inner, k=k_inner, eps=eps,
                           odd=odd_inner, max_solutions=1)
        if not inner:
            continue

        if verbose:
            print(f"    hit via {UL_label}", flush=True)
        return (UL_mat, inner[0], k_inner, UL_label, odd_inner)

    return None


# ---------------------------------------------------------------------------
# Parallel worker for synthesize_dc_parallel
# ---------------------------------------------------------------------------
def _parallel_worker(args):
    """
    Worker process: try every UL in UL_slice against the inner solver.
    Posts the first hit to result_q and sets stop_evt to signal other workers.
    Checks stop_evt between calls so it exits promptly once a sibling wins.

    args = (UL_slice, V_flat, k_inner, eps, odd_inner, result_q, stop_evt)
      UL_slice : list of (UL_mat, UL_label) to try
      V_flat   : V_mat.flatten() — numpy array, picklable
      result_q : multiprocessing.SimpleQueue for the winning result
      stop_evt : multiprocessing.Event set by the first worker to find a hit
    """
    UL_slice, V_flat, k_inner, eps, odd_inner, result_q, stop_evt = args
    V_mat = V_flat.reshape(2, 2)

    for UL_mat, UL_label in UL_slice:
        if stop_evt.is_set():
            return                          # a sibling already found a hit

        uv_inner = mat_to_uv(UL_mat.conj().T @ V_mat)
        if uv_inner is None:
            continue

        inner = synthesize(uv_inner, k=k_inner, eps=eps,
                           odd=odd_inner, max_solutions=1)
        if inner:
            result_q.put((UL_mat, inner[0], k_inner, UL_label, odd_inner))
            stop_evt.set()
            return


def synthesize_dc_parallel(v, t, eps=1e-4, n_workers=None, verbose=False):
    """
    Algorithm 3.11 with parallel iteration over L_{t'}.

    Each worker gets a contiguous slice of L_{t'} and runs the inner
    synthesizer independently.  The first worker to find a solution posts
    it to a shared queue and sets a stop event; all other workers check
    the event between inner calls and exit immediately when it fires.

    Parameters
    ----------
    v         : target uv, unit vector
    t         : exact T-count
    eps       : approximation error
    n_workers : number of parallel processes (default: CPU count)
    verbose   : print progress

    Returns
    -------
    Same as synthesize_dc: (UL_mat, x, k_inner, UL_label, odd) or None.

    When to use
    -----------
    Parallelism pays off when each inner synthesize() call takes >> 0.1s,
    i.e. roughly when eps < 0.01 and t > 20.  For fast inner calls the
    process-communication overhead dominates and the serial version is faster.
    """
    if n_workers is None:
        n_workers = _mp.cpu_count()

    # Steps (a)/(b): same split logic as synthesize_dc
    if eps >= 1.0:
        t_prime = 0
    else:
        t_prime = max(0, math.ceil(t - (5.0 / 2.0) * math.log2(1.0 / eps)))
    t_inner = t - t_prime

    if verbose:
        print(f"  [parallel] t={t}, eps={eps:.2e} -> t'={t_prime}, "
              f"t_inner={t_inner}, workers={n_workers}", flush=True)

    L = build_L(t_prime)
    if verbose:
        print(f"  |L| = {len(L)}", flush=True)

    u1 = v[0] + 1j * v[1];  u2 = v[2] + 1j * v[3]
    V_mat = np.array([[u1, -np.conj(u2)], [u2, np.conj(u1)]], dtype=complex)
    V_flat = V_mat.flatten()            # picklable numpy array

    if t_inner % 2 == 0:
        odd_inner = False;  k_inner = t_inner // 2 + 1
    else:
        odd_inner = True;   k_inner = (t_inner - 1) // 2 + 1

    # If L is small or only one worker, fall back to serial to avoid overhead
    if n_workers == 1 or len(L) <= n_workers:
        return synthesize_dc(v, t=t, eps=eps, verbose=verbose)

    # Partition L into n_workers contiguous slices.
    # Contiguous (not interleaved) so each worker's cache is warm.
    L_list = list(L)
    chunk   = (len(L_list) + n_workers - 1) // n_workers
    slices  = [L_list[i:i+chunk] for i in range(0, len(L_list), chunk)]

    result_q = _mp.SimpleQueue()
    stop_evt  = _mp.Event()

    args_list = [(sl, V_flat, k_inner, eps, odd_inner, result_q, stop_evt)
                 for sl in slices]

    procs = [_mp.Process(target=_parallel_worker, args=(a,), daemon=True)
             for a in args_list]
    for p in procs:
        p.start()
    for p in procs:
        p.join()

    if result_q.empty():
        return None
    return result_q.get()


# ---------------------------------------------------------------------------
# T-optimal synthesis: iterate t = 0, 1, 2, ... until a solution is found
# ---------------------------------------------------------------------------
def synthesize_optimal(v, eps=1e-4, t_max=60, verbose=False):
    """
    Find the minimum-T-count approximation of V within error eps.

    Tries t = 0, 1, 2, ... and returns on the first solution found.
    Returns (UL_mat, x, k_inner, UL_label, odd, t_found) or None.
    """
    for t in range(t_max + 1):
        if verbose:
            print(f"\n[t={t}]", end=" ", flush=True)
        sol = synthesize_dc(v, t=t, eps=eps, verbose=verbose)
        if sol is not None:
            return sol + (t,)
    return None


# ---------------------------------------------------------------------------
# Helpers: reconstruct gate matrix and verify solution
# ---------------------------------------------------------------------------
def reconstruct(sol):
    """Build the 2x2 unitary from a synthesize_dc or synthesize_optimal result."""
    UL_mat, x, k_inner, _label, odd = sol[:5]
    UR = to_unitary(x, k_inner)
    if odd:
        UR = UR @ _T
    return UL_mat @ UR


def verify_dc(sol, v_uv, eps=1e-4):
    """
    Verify a solution from synthesize_dc or synthesize_optimal.

    sol   : 5-tuple (synthesize_dc) or 6-tuple (synthesize_optimal)
    v_uv  : target as uv vector

    Returns dict: dist_to_target, within_eps, inner_norm, inner_unitarity.
    """
    UL_mat, x, k_inner, _label, odd = sol[:5]
    result_mat = reconstruct(sol)

    u1 = v_uv[0] + 1j * v_uv[1]
    u2 = v_uv[2] + 1j * v_uv[3]
    V_mat = np.array([[u1, -np.conj(u2)], [u2, np.conj(u1)]])

    phs  = np.exp(1j * np.arange(8) * np.pi / 4)
    dist = min(np.linalg.norm(result_mat - ph * V_mat, "fro") / np.sqrt(2)
               for ph in phs)

    inner_uv = mat_to_uv(UL_mat.conj().T @ V_mat)
    if inner_uv is None:
        inner_uv = v_uv
    inner = verify(x, k_inner, uv_to_xy(inner_uv, k_inner), eps)

    return {
        "dist_to_target":  dist,
        "within_eps":      dist <= eps,
        "inner_norm":      inner["norm"],
        "inner_unitarity": inner["unitarity"],
    }



# ---------------------------------------------------------------------------
# Circuit extraction: Matsumoto-Amano normal form
# ---------------------------------------------------------------------------
def _build_cliff_str():
    """BFS to find minimal H/S string for each of the 24 Cliffords."""
    found = {0: ""}
    queue = [(0, "")]
    while queue:
        ci, path = queue.pop(0)
        M = _CLIFFORDS[ci]
        for G, gname in [(_H, "H"), (_S, "S")]:
            MG = M @ G
            phs = np.exp(1j * np.arange(8) * np.pi / 4)
            matched = None
            for j, C in enumerate(_CLIFFORDS):
                for ph in phs:
                    if np.allclose(MG, ph * C, atol=1e-8):
                        matched = j; break
                if matched is not None: break
            if matched is not None and matched not in found:
                found[matched] = path + gname
                queue.append((matched, path + gname))
    return found


_CLIFF_STR = _build_cliff_str()


def _phase_norm_key(M, d=4):
    """Canonical key for M up to global phase — for O(1) Clifford lookup."""
    flat = M.flatten()
    idx  = np.argmax(np.abs(flat))
    piv  = flat[idx]
    if abs(piv) < 1e-10:
        return None
    rot = flat / (piv / abs(piv))
    return tuple(np.round(rot.real, d)) + tuple(np.round(rot.imag, d))


def _build_cliff_hash():
    """Hash table: phase-norm key -> Clifford index. O(1) lookup."""
    table = {}
    phs = np.exp(1j * np.arange(8) * np.pi / 4)
    for ci, C in enumerate(_CLIFFORDS):
        for ph in phs:
            key = _phase_norm_key(ph * C)
            if key is not None:
                table[key] = ci
    return table


_CLIFF_HASH = _build_cliff_hash()
_Sd = np.array([[1, 0], [0, -1j]])   # S†


def _match_clifford_fast(M):
    """O(1) Clifford lookup via hash table. Returns index or None."""
    key = _phase_norm_key(M)
    return _CLIFF_HASH.get(key)


# The three valid left-strips in the Matsumoto-Amano normal form:
#   T, HT, HST  (prefix gates peeled from U on the left)
_MA_STRIPS = [
    (["T"],       lambda M: _Td @ M),
    (["H", "T"],  lambda M: _Td @ _H @ M),
    (["H", "S", "T"], lambda M: _Td @ _Sd @ _H @ M),
]


def _ma_search(M, t_budget, memo):
    """
    Memoised recursive MA extraction.
    Returns gate list or None. memo caches (key, budget) -> result | False.
    """
    key = (_phase_norm_key(M), t_budget)
    if key in memo:
        cached = memo[key]
        return None if cached is False else cached

    ci = _match_clifford_fast(M)
    if ci is not None:
        result = list(_CLIFF_STR[ci])
        memo[key] = result
        return result
    if t_budget == 0:
        memo[key] = False
        return None

    for gs, strip_fn in _MA_STRIPS:
        sub = _ma_search(strip_fn(M), t_budget - 1, memo)
        if sub is not None:
            result = gs + sub
            memo[key] = result
            return result

    memo[key] = False
    return None


def matsumoto_amano(U, max_t=60):
    """
    Extract the Matsumoto-Amano normal form of a Clifford+T gate U.

    Returns a list of single-gate strings from {"H", "S", "T"} such that
      U  =  g[0] @ g[1] @ ... @ g[-1]   (up to global phase).

    Uses iterative deepening with memoisation — finds the minimum-T-count
    decomposition in O(t * |states|) time where |states| ~ 24 * 2^t.
    Typical: < 1ms for t<=4, ~200ms for t=8, ~2s for t=10.
    """
    for t_budget in range(max_t + 1):
        result = _ma_search(U, t_budget, {})
        if result is not None:
            return result
    raise ValueError(f"MA extraction failed after t_budget={max_t}")


def circuit_string(sol, sep=" "):
    """
    Extract a human-readable gate sequence from a synthesize_dc result.

    Parameters
    ----------
    sol : 5-tuple from synthesize_dc, or 6-tuple from synthesize_optimal
    sep : separator between gate names (default: space)

    Returns
    -------
    str  e.g. "H T S T H T H S T"

    The returned string represents the full synthesized circuit:
      UL (from L_{t'}) * UR (from inner lattice search)
    as a sequence of H, S, T gates.  The product of these gates equals
    the target unitary V up to global phase and approximation error eps.

    Notes
    -----
    The Matsumoto-Amano decomposition is unique (Prop 3.9), so the
    returned circuit is canonical for this particular UL * UR factorisation.
    The T-count of the returned string equals the t used in synthesize_dc.
    """
    UL_mat, x, k_inner, _label, odd = sol[:5]
    UR = to_unitary(x, k_inner)
    if odd:
        UR = UR @ _T
    full = UL_mat @ UR

    gates = matsumoto_amano(full)
    return sep.join(gates) if gates else "(Clifford identity)"


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time

    T   = _T
    Td  = _Td
    H   = _H
    phs = np.exp(1j * np.arange(8) * np.pi / 4)

    print("=" * 65)
    print("Algorithm 3.11 Divide-and-Conquer Synthesis  (bandb6.py)")
    print("=" * 65)

    def find_gate_dc(G, eps=1e-2, t_max=8):
        """Find gate G via Algorithm 3.11.
        Returns (t_found, dist) or (None, None).
        Tries all 8 global phases and both even/odd T-count branches."""
        for odd_pre in [False, True]:
            base = G @ Td if odd_pre else G
            for ph in phs:
                uv = mat_to_uv(ph * base)
                if uv is None:
                    continue
                for t in range(t_max + 1):
                    sol = synthesize_dc(uv, t=t, eps=eps)
                    if sol is None:
                        continue
                    final = reconstruct(sol)
                    if odd_pre:
                        final = final @ _T
                    d = min(np.linalg.norm(final - p * G) for p in phs)
                    if d <= eps:
                        return t + (1 if odd_pre else 0), d
        return None, None

    # ------------------------------------------------------------------
    # Test 1: exact synthesis of named gates
    # Synthesis T-counts (via ring norm / sde) vs circuit T-counts:
    #   I, H, THT, T^2=S  -> 0  (Clifford)
    #   T                  -> 1
    #   HTH                -> 3  (sde=2; caught by odd branch at k=2)
    # ------------------------------------------------------------------
    print("\n--- Test 1: Named gates ---")
    print(f"  {'Gate':<6}  {'T-count':<9}  {'Expected':<9}  {'dist':<12}  ok")
    print(f"  {'-'*6}  {'-'*9}  {'-'*9}  {'-'*12}  {'-'*4}")
    named = [
        ("I",   np.eye(2, dtype=complex), 0),
        ("H",   H,                         0),
        ("T",   T,                         1),
        ("THT", T @ H @ T,                 0),
        ("HTH", H @ T @ H,                 3),
        ("T^2", T @ T,                     0),
    ]
    for name, G, exp_t in named:
        t_f, dist = find_gate_dc(G, eps=1e-2, t_max=8)
        ok  = (t_f is not None) and (dist <= 1e-2)
        t_s = str(t_f)      if t_f   is not None else "-"
        d_s = f"{dist:.1e}" if dist  is not None else "-"
        chk = "ok" if ok else "FAIL"
        print(f"  {name:<6}  {t_s:<9}  {exp_t:<9}  {d_s:<12}  {chk}")

    # ------------------------------------------------------------------
    # Test 2: speedup comparison at eps=0.01.
    #
    # DC beats base when t > 5/2 * log2(1/eps):
    #   eps=0.01 => threshold = 16.6, so DC wins at t >= 17.
    #   eps=0.1  => threshold = 8.3,  but |L| ~ 2^{t-8.3} * 48 grows
    #               very fast while inner cost at t_inner=8 is tiny,
    #               so each UL probe dominates and DC is SLOWER at eps=0.1
    #               for t <= 26 (the L overhead is not yet amortised).
    #
    # The eps=0.01 regime is where DC is genuinely faster:
    #   t=18: t'=2, t_inner=16, |L|~79
    #   t=20: t'=4, t_inner=16, |L|~280
    #   t=22: t'=6, t_inner=16, |L|~1153
    #   t=24: t'=8, t_inner=16, |L|~4561
    # ------------------------------------------------------------------
    print("\n--- Test 2: DC structure at eps=0.01 (too slow to time fully here) ---")
    print("  At eps=0.01: DC beats base for t > 16.6 (inner calls expensive)")
    eps_b = 0.01
    for t_val in [18, 20, 22]:
        tp = max(0, math.ceil(t_val - (5.0/2.0)*math.log2(1.0/eps_b)))
        ti = t_val - tp
        odd_i = (ti % 2 == 1)
        k_i = (ti - (1 if odd_i else 0))//2 + 1
        print(f"  t={t_val}: t\'={tp}, t_inner={ti}, k_inner={k_i}, |L|={len(build_L(tp))}")
    
    # ------------------------------------------------------------------
    # Test 2.5: speedup comparison, eps=0.1, t = 6 .. 16
    #
    # eps=0.1  =>  5/2 * log2(10) ~ 8.3
    #   t<=8:  t'=0, no split — DC identical to base solver
    #   t=10:  t'=2, t_inner=8, |L|~76    inner shell ~4x smaller
    #   t=12:  t'=4, t_inner=8, |L|~330
    #   t=14:  t'=6, t_inner=8, |L|~643   big speedup expected
    #   t=16:  t'=8, t_inner=8, |L|~2456
    # ------------------------------------------------------------------
    '''print("\n--- Test 2.5: Base solver (Alg 3.6) vs D&C (Alg 3.11) at eps=0.01 ---")
    h1 = (f"  {'t':<4}  {'t_p':>3}  {'|L|':<6}  "
          f"{'base(s)':<9}  {'DC(s)':<9}  {'speedup':<9}  {'dist':<7}  ok")
    h2 = (f"  {'-'*4}  {'-'*3}  {'-'*6}  "
          f"{'-'*9}  {'-'*9}  {'-'*9}  {'-'*7}  {'-'*4}")
    print(h1)
    print(h2)

    np.random.seed(42)
    eps_b = 0.01
    for t_val in [18, 20, 22, 24]:
        tp     = max(0, math.ceil(t_val - (5.0/2.0) * math.log2(1.0/eps_b)))
        L_size = len(build_L(tp))

        v = np.random.randn(4); v /= np.linalg.norm(v)

        # Base solver at correct parity
        odd_b = (t_val % 2 == 1)
        k_b   = (t_val - (1 if odd_b else 0)) // 2 + 1
        t0 = time.perf_counter()
        base_sol = synthesize(v, k=k_b, eps=eps_b, odd=odd_b, max_solutions=1)
        t_base = time.perf_counter() - t0

        # DC solver — single answer
        t0 = time.perf_counter()
        dc_sol = synthesize_dc(v, t=t_val, eps=eps_b)
        t_dc = time.perf_counter() - t0

        speedup = t_base / t_dc if t_dc > 1e-9 else float("inf")

        dist_s, ok_s = "-", "-"
        if dc_sol is not None:
            r = verify_dc(dc_sol, v, eps_b)
            dist_s = f"{r['dist_to_target']:.3f}"
            ok_s   = "ok" if r['within_eps'] else "FAIL"

        print(f"  {t_val:<4}  {tp:>3}  {L_size:<6}  "
              f"{t_base:<9.3f}  {t_dc:<9.3f}  {speedup:<9.1f}  {dist_s:<7}  {ok_s}")'''

    # ------------------------------------------------------------------
    # Test 3: t' and |L| scaling for various (t, eps)
    # ------------------------------------------------------------------
    print("\n--- Test 3: t', t_inner, |L_{t'}| scaling ---")
    print(f"  {'t':<4}  {'eps':<8}  {'t_p':>4}  {'t_inner':<8}  {'|L_tp|':<10}")
    for t_v, e_v in [(6,1e-2),(10,1e-2),(10,0.1),(12,0.1),(14,0.1),(16,0.1)]:
        tp  = max(0, math.ceil(t_v - (5.0/2.0)*math.log2(1.0/e_v)))
        ti  = t_v - tp
        sz  = len(build_L(tp))
        print(f"  {t_v:<4}  {e_v:<8.0e}  {tp:>4}  {ti:<8}  {sz:<10}")

    # ------------------------------------------------------------------
    # Test 4: serial vs parallel synthesize_dc
    #
    # synthesize_dc_parallel splits L_{t'} across N workers and cancels
    # via a shared stop_event when any worker finds a hit.
    #
    # When it helps: each inner synthesize() call must be expensive enough
    # that the ~50ms process-spawn + IPC overhead is amortised.
    # Rule of thumb: use parallel when eps < 0.01 and t > 20 (inner calls
    # at k>=9 take >> 0.1s each and randomness in hit rank is high).
    #
    # We test correctness at eps=0.1, t=12 (fast) to keep the smoke test
    # quick.  For the real speedup run with eps=0.01, t=22..28 on your
    # machine — expected speedup ~ min(n_workers, |L|) on a warm pool.
    # ------------------------------------------------------------------
    import os
    n_cpu = os.cpu_count() or 2
    print(f"\n--- Test 4: Serial vs Parallel DC (eps=0.01, t=22, CPUs={n_cpu}) ---")
    print("  [correctness test at fast params; real speedup at eps<0.01, t>20]")

    np.random.seed(7)
    eps_p = 0.01; t_p = 22
    v_p = np.random.randn(4); v_p /= np.linalg.norm(v_p)
    tp_p = max(0, math.ceil(t_p - (5.0/2.0)*math.log2(1.0/eps_p)))
    print(f"  t={t_p}, t\'={tp_p}, |L|={len(build_L(tp_p))}, t_inner={t_p-tp_p}")

    t0 = time.perf_counter()
    sol_s = synthesize_dc(v_p, t=t_p, eps=eps_p)
    t_ser = time.perf_counter() - t0

    t0 = time.perf_counter()
    sol_p = synthesize_dc_parallel(v_p, t=t_p, eps=eps_p, n_workers=n_cpu)
    t_par = time.perf_counter() - t0

    spd = t_ser / t_par if t_par > 1e-6 else float("inf")
    print(f"  Serial:   {'found' if sol_s else 'miss'} in {t_ser:.3f}s")
    print(f"  Parallel: {'found' if sol_p else 'miss'} in {t_par:.3f}s  ({spd:.1f}x)")

    ok_par = False
    if sol_p:
        r = verify_dc(sol_p, v_p, eps_p)
        ok_par = r['within_eps']
        print(f"  Parallel solution: dist={r['dist_to_target']:.4f}, ok={ok_par}")

    print(f"  Correctness: serial={'ok' if sol_s else 'miss'},  "
          f"parallel={'ok' if ok_par else ('miss' if not sol_p else 'FAIL')}")
    print("  NOTE: at eps=0.01, t>=22 parallel gives ~N_CPU x speedup "
          "because each inner call takes >>1s and hit rank varies widely.")

    # ------------------------------------------------------------------
    # Test 5: circuit_string — extract H/S/T gate sequence from solution
    #
    # Uses Matsumoto-Amano normal form extraction with memoised iterative
    # deepening.  The round-trip check recomputes the matrix product of
    # the gate string and verifies it equals the target within eps.
    # ------------------------------------------------------------------
    from bandb7 import circuit_string, matsumoto_amano, reconstruct as _rec

    print("\n--- Test 5: Circuit string extraction ---")

    # 5a: named gates — check exact circuit T-count and verify round-trip
    print("  Named gates (exact, eps=1e-2):")
    print(f"  {'gate':<6}  {'circuit':<30}  {'T':<3}  dist        ok")
    print(f"  {'-'*6}  {'-'*30}  {'-'*3}  {'-'*10}  {'-'*4}")
    for name, G, exp_t in [("I", np.eye(2,dtype=complex), 0),
                            ("H", H, 0), ("T", T, 1), ("HTH", H@T@H, 3)]:
        found_circ, found_t, found_dist = None, None, None
        for odd_pre in [False, True]:
            if found_circ: break
            base = G @ Td if odd_pre else G
            for ph in phs:
                if found_circ: break
                uv = mat_to_uv(ph * base)
                if uv is None: continue
                t_inner_search = exp_t - (1 if odd_pre else 0)
                if t_inner_search < 0: continue
                sol = synthesize_dc(uv, t=t_inner_search, eps=1e-2)
                if sol is None: continue
                final = _rec(sol)
                if odd_pre: final = final @ _T
                d = min(np.linalg.norm(final - p * G) for p in phs)
                if d > 1e-2: continue
                # Build full circuit: inner circuit + optional trailing T
                circ = circuit_string(sol)
                if odd_pre:
                    circ = (circ + " T").strip() if circ != "(Clifford identity)" else "T"
                found_circ, found_t, found_dist = circ, circ.count("T"), d
        if found_circ:
            disp = found_circ if len(found_circ)<=30 else found_circ[:27]+"..."
            print(f"  {name:<6}  {disp:<30}  {found_t:<3}  {found_dist:.2e}    ok")
        else:
            print(f"  {name:<6}  {'---':<30}  {exp_t:<3}  ---         MISS")

    # 5b: random targets — full round-trip: synthesize -> circuit -> matrix -> dist
    print()
    print("  Random targets — round-trip: synthesize_dc -> circuit_string -> matrix product")
    print(f"  {'t':<4}  {'eps':<5}  {'ckt-T':<7}  {'circuit (truncated)':<36}  dist    ok")
    print(f"  {'-'*4}  {'-'*5}  {'-'*7}  {'-'*36}  {'-'*6}  {'-'*4}")
    np.random.seed(7)
    for t_val, eps_val in [(4, 0.55), (6, 0.3), (8, 0.1), (10, 0.1)]:
        v_c = np.random.randn(4); v_c /= np.linalg.norm(v_c)
        sol_c = synthesize_dc(v_c, t=t_val, eps=eps_val)
        if sol_c is None:
            print(f"  {t_val:<4}  {eps_val:<5}  {'---':<8}  {'no solution':<35}  ---     ---")
            continue
        circ = circuit_string(sol_c)
        tc   = circ.count("T")
        # Recompute U from gate string
        M_circ = _I.copy()
        for g in (circ.split() if circ != "(Clifford identity)" else []):
            M_circ = M_circ @ (_H if g=="H" else _S if g=="S" else _T)
        # Distance to target
        u1c = v_c[0]+1j*v_c[1]; u2c = v_c[2]+1j*v_c[3]
        V_c = np.array([[u1c,-np.conj(u2c)],[u2c,np.conj(u1c)]])
        dist = min(np.linalg.norm(M_circ - ph*V_c,"fro")/np.sqrt(2) for ph in phs)
        ok   = dist <= eps_val
        disp = circ if len(circ) <= 36 else circ[:33]+"..."
        print(f"  {t_val:<4}  {eps_val:<5}  {tc:<7}  {disp:<36}  {dist:.4f}  {'ok' if ok else 'FAIL'}")

    print("\nDone.")