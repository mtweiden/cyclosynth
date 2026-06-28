"""Syllable-model resource cost of a Clifford+sqrt(T) gate string, in T states.

Mirrors the Rust `gates_cost` in src/synthesis/clifford_sqrt_t/mod.rs. Cost is
charged per *syllable* -- a maximal run of gates with no off-diagonal
H/X/Y between them -- not per gate. A syllable's gates commute and compose to one
net Z-rotation Q^k (Q = sqrt(T)); its cost depends only on k mod 4 (each Q adds
1, T adds 2, while S, Z add multiples of 4 and never change the class):

    k odd        -> sqrt(T)-class, one sqrt(T) injection  -> c  (default 3)
    k == 2 (mod 4) -> T-class, one T injection            -> 1
    k == 0 (mod 4) -> Clifford                            -> 0

So TQ = QT = Q^3 = T^{3/2} = Q-dagger S costs 3 (not 4), and QQ = T costs 1.

Lowercase q, t, s are the adjoints Q-dagger, T-dagger, S-dagger (powers -1, -2,
-4) the decomposer emits in canonical syllable form; Python's `%` is already
floor/euclidean, so negative powers classify correctly.
"""
import re

_P = {"Q": 1, "T": 2, "S": 4, "Z": 8, "q": -1, "t": -2, "s": -4}


def syllable_classes(gates):
    """(n_Tclass, n_sqrtTclass): counts of T-class and sqrt(T)-class syllables.

    These are the cost-bearing gate counts under the syllable model: total cost in
    T states is `n_Tclass + c * n_sqrtTclass`, with c the sqrt(T) price.
    """
    n_t = n_r = 0
    for blk in re.split("[HXY]", gates):
        k = sum(_P.get(ch, 0) for ch in blk) % 4
        if k in (1, 3):
            n_r += 1
        elif k == 2:
            n_t += 1
    return n_t, n_r


def syllable_cost(gates, c=3):
    """Resource cost in T states: T-class syllable = 1, sqrt(T)-class syllable = c."""
    n_t, n_r = syllable_classes(gates)
    return n_t + c * n_r
