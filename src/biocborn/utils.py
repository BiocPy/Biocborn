from collections import namedtuple
from typing import Literal, Sequence, Tuple

from biocframe import BiocFrame
from numpy import int32, zeros
from pandas import DataFrame
from scipy import sparse
from singlecellexperiment import SingleCellExperiment

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


def _is_unique(x: Sequence):
    """Check if ``x`` contains a unique list of elements."""
    return len(set(x)) == len(x)


def _to_list(x):
    if sparse.issparse(x):
        return x.toarray()[0].tolist()

    return x.tolist()


def _extract_variable_from_sce(
    x: SingleCellExperiment,
    var_key: str,
    var_value: str,
    assay: str,
    check_col_data: bool = True,
    check_row_data: bool = True,
) -> Tuple[Sequence, Literal["annotation", "gene"]]:
    """Extract a variable from :py:class:`~singlecellexperiment.SingleCellExperiment.SingleCellExperiment`.

    Variable ``var_value`` can either be a column in the
    :py:attr:`singlecellexperiment.SingleCellExperiment.col_data`
    and/or a row in the :py:attr:`singlecellexperiment.SingleCellExperiment.row_data`.

    Raises:
        ValueError: If ``var_value`` is not found in col_data or row_data of the SCE.

    Returns:
        Tuple[Sequence, Literal["annotation", "gene"]]:
        A list containing the values and where it was found (annotation or gene).
    """
    _variable = None
    # first check if the variable we are looking for in col_data
    if check_col_data is True and x.col_data is not None:
        _cdata = x.col_data
        if isinstance(_cdata, BiocFrame) and _cdata.has_column(var_value):
            _variable = x.col_data.column(var_value)
        elif isinstance(_cdata, DataFrame) and var_value in _cdata.columns:
            _variable = _cdata[var_value]
        _where = "annotation"

    # if it isn't, then it might be a feature (row_data)
    if check_row_data is True and _variable is None and x.row_data is not None:
        _rdata = x.row_data
        _var_idx = None
        if isinstance(_rdata, BiocFrame) and _rdata.row_names is not None:
            if _rdata.row_names is not None and var_value in _rdata.row_names:
                _var_idx = x.row_data.row_names.index(var_value)
                _variable = _to_list(x.assay(assay)[_var_idx, :])
        elif isinstance(_rdata, DataFrame) and var_value in _rdata.index:
            _var_idx = _rdata.index.get_loc(var_value)
            _variable = _to_list(x.assay(assay)[_var_idx, :])
        _where = "gene"

    # if we can't find it, throw an error
    if _variable is None:
        raise ValueError(
            f"`{var_key}` is neither a cell annotation column nor a gene symbol."
        )

    return (list(_variable), _where)


def _extract_variable_from_frame(frame, var_key):
    _vec = None
    if isinstance(frame, BiocFrame) and frame.has_column(var_key):
        _vec = frame.column(var_key)
    elif isinstance(frame, DataFrame) and var_key in frame.columns:
        _vec = _to_list(frame[var_key])

    return _vec


def _extract_index_from_frame(frame):
    _index = None
    if isinstance(frame, BiocFrame):
        _index = frame.row_names
    elif isinstance(frame, DataFrame):
        _index = _to_list(frame.index)

    return _index
