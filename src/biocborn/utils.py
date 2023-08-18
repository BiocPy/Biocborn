from collections import namedtuple
from typing import Sequence

from numpy import int32, zeros

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"

FactorizedArray = namedtuple("FactorizedArray", ["levels", "indices"])
FactorizedArray.__doc__ = """Named tuple of a Factorized Array.

levels (:py:class:`numpy.ndarray`): Levels in the array.

indices (:py:class:`numpy.ndarray`): Indices.
"""


def factorize(x: Sequence) -> FactorizedArray:
    """Factorize an array.

    Args:
        x (Sequence): Any array.

    Returns:
        FactorizedArray: A factorized tuple.
    """

    if not isinstance(x, list):
        raise TypeError("x is not a list")

    levels = []
    mapping = {}
    output = zeros((len(x),), dtype=int32)

    for i in range(len(x)):
        lev = x[i]

        if lev not in mapping:
            mapping[lev] = len(levels)
            levels.append(lev)

        output[i] = mapping[lev]

    return FactorizedArray(levels=levels, indices=output)
