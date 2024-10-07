"""A module for translating decomposition sequences into Clifford+T/Q gates."""
from re import sub


def translate_decomposition(gates: str, magic_gate: str) -> str:
    """
    Translate a decomposition into Clifford+T or Clifford+Q (sqrt(T)). 

    Args:
        gates (str): The decomposition of a unitary matrix into discrete
            gate rotations. The string is expected to consist of elements
            in {x, y, z}, where each represents a discrete rotations about
            the corresponding axis by `pi / self.base`. Capital letters
            indicate Clifford rotations.
        
        magic_gate (str): The magic gate to use for non-Clifford rotations.
            This must be either 'T' or 'Q'.
        
    Returns:
        str: The decomposition translated into Clifford+T or Clifford+Q
            notation.
    
    Raises:
        ValueError: If decomposition contains an invalid gate.

        ValueError: If magic_gate is neither 'T' nor 'Q'.
    
    Note:
        rz -> magic_gate
        rx -> H rz H
        ry -> SH rz HSSS
    """
    translation = sub(r'x', f'H{magic_gate}H', gates)
    translation = sub(r'y', f'SH{magic_gate}HZS', translation)
    translation = sub(r'z', f'{magic_gate}', translation)

    last_translation = None
    while translation != last_translation:
        last_translation = translation
        # Commutations
        # Commute Z gates to the left of S, T, Q
        translation = sub(r'SZ', 'ZS', translation)
        translation = sub(r'TZ', 'ZT', translation)
        translation = sub(r'QZ', 'ZQ', translation)
        # Commute S gates to the left of T, Q
        translation = sub(r'TS', 'ST', translation)
        translation = sub(r'QS', 'SQ', translation)
        # Commute T gates to the left of Q
        translation = sub(r'QT', 'TQ', translation)

        # Combinations
        # Combine pairs of neighboring Q gates (QQ -> T)
        translation = sub(r'QQ', 'T', translation)
        # Combine pairs of neighboring T gates (TT -> S)
        translation = sub(r'TT', 'S', translation)
        # Combine pairs of neighboring S gates (SS -> Z)
        translation = sub(r'SS', 'Z', translation)

        # Cancelations
        # Cancel pairs of neighboring Clifford gates
        translation = sub('HH', '', translation)
        translation = sub('XX', '', translation)
        translation = sub('YY', '', translation)
        translation = sub('ZZ', '', translation)
        translation = sub('I', '', translation)

    return translation
