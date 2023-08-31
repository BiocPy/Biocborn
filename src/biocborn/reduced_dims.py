from functools import singledispatch
from typing import Literal, Optional, Sequence, Tuple, Union
from warnings import warn

from biocframe import BiocFrame
from numpy import ndarray
from pandas import DataFrame
from scipy import sparse
from seaborn import FacetGrid, cubehelix_palette, relplot
from singlecellexperiment import SingleCellExperiment

from ._checks import is_list_of_type
from .types import ArrayLike

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


def _to_list(x):
    if sparse.issparse(x):
        return x.toarray()[0].tolist()

    return x.tolist()


def _extract_variable_from_sce(
    x: SingleCellExperiment, var_key: str, var_value: str, assay: str
) -> Tuple[Sequence, Literal["annotation", "gene"]]:
    """Extract a variable from :py:class:`~singlecellexperiment.SingleCellExperiment.SingleCellExperiment`.

    Variable ``var_value`` can either be a column in the
    :py:meth:`singlecellexperiment.SingleCellExperiment.col_data`
    or a row in the :py:meth:`singlecellexperiment.SingleCellExperiment.row_data`.

    Raises:
        ValueError: If ``var_value`` is not found in col_data or row_data of the SCE.

    Returns:
        Tuple[Sequence, Literal["annotation", "gene"]]:
        A list containing the values and where it was found (annotation or gene).
    """
    _variable = None
    # first check if the variable we are looking for in col_data
    if x.col_data is not None:
        _cdata = x.col_data
        if isinstance(_cdata, BiocFrame) and _cdata.has_column(var_value):
            _variable = x.col_data.column(var_value)
        elif isinstance(_cdata, DataFrame) and var_value in _cdata.columns:
            _variable = _cdata[var_value]
        _where = "annotation"

    # if it isn't, then it might be a feature (row_data)
    if _variable is None and x.row_data is not None:
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


def _dim_plot(x: ArrayLike, y: ArrayLike, **kwargs) -> FacetGrid:
    """Function to create the seaborn plot from the parameters."""
    g = relplot(x=x, y=y, **kwargs)
    g.ax.xaxis.grid(True, "minor", linewidth=0.25)
    g.ax.yaxis.grid(True, "minor", linewidth=0.25)
    g.despine()

    return g


@singledispatch
def plot_reduced_dim(
    x,
    dimred: Optional[str] = None,
    color_by: Optional[Union[str, Sequence]] = None,
    size_by: Optional[Union[str, Sequence]] = None,
    shape_by: Optional[Union[str, Sequence]] = None,
    assay_name: Optional[Union[str, Sequence]] = None,
) -> FacetGrid:
    """Plot cell-level reduced dimensions.

    The first two components are used to plot along the `x` and `y` axis respectively.

    Args:
        x : Object containing the embeddings to plot.

            This may be a 2-dimensional :py:class:`numpy.ndarray` containing the
            per-cell coordinates, where rows are cells and columns are components.

            Alternatively, ``x`` may be a
            :py:class:`~singlecellexperiment.SingleCellExperiment.SingleCellExperiment`,
            in which case
            a ``dimred`` name must be provided to extract the embeddings using
            :py:meth:`~singlecellexperiment.SingleCellExperiment.SingleCellExperiment.reducedDim`.

        dimred (str): Reduced dimension to plot, only used if ``x`` is
            a :py:class:`~singlecellexperiment.SingleCellExperiment.SingleCellExperiment`.

        color_by (Union[str, Sequence], optional): Variable that specifies `colors` per-cell.

            This maps to the ``hue`` parameter of
            :py:func:`~seaborn.relplot`. This may be a categorical or numerical.

            Alternatively, if ``x`` is a
            :py:class:`~singlecellexperiment.SingleCellExperiment.SingleCellExperiment`, you can specify
            the column annotation field or feature to color by.

            Defaults to None, all cells have the same color.

        size_by (Union[str, Sequence], optional): Variable to specify the `size` of the dots per-cell.

            This maps to the ``size`` parameter of
            :py:func:`~seaborn.relplot`. This may be a categorical or numerical.

            Alternatively, if ``x`` is a
            :py:class:`~singlecellexperiment.SingleCellExperiment.SingleCellExperiment`, you may specify
            the column annotation field or feature to render the size of the dots.

            Defaults to None, all cells have the same size.

        shape_by (Union[str, Sequence], optional): Variable that specifies `shape` of the dots per-cell.

            This maps to the ``markers`` parameter of
            :py:func:`~seaborn.relplot`. This must be a categorical variable.

            Alternatively, if ``x`` is a
            :py:class:`~singlecellexperiment.SingleCellExperiment.SingleCellExperiment`, you may specify
            the column annotation field to specify markers.

            Defaults to None, all cells have the same marker.

        assay_name (str, optional): Assay to extract feature information from,
            only used if ``x`` is a :py:class:`~singlecellexperiment.SingleCellExperiment.SingleCellExperiment`.

            This is used when the parameters, ``color_by``, ``size_by`` or ``shape_by`` map
            to a feature in the :py:class:`~singlecellexperiment.SingleCellExperiment.SingleCellExperiment`.

            Defaults to None, all cells have the same default color.. Defaults to None.

    Raises:
        NotImplementedError: When ``x`` is not an expected type.

    Returns:
        FacetGrid: A seaborn plot object.
    """
    raise NotImplementedError(
        f"`plot_reduced_dim` is not implemented for objects of class: '{type(x)}'."
    )


@plot_reduced_dim.register
def _plot_reduced_dim_numpy(
    x: ndarray,
    dimred: Optional[str] = None,
    color_by: Optional[Sequence] = None,
    size_by: Optional[Sequence] = None,
    shape_by: Optional[Sequence] = None,
    assay_name: Literal[None] = None,
) -> FacetGrid:
    NCELLS = x.shape[0]
    params = {}

    if color_by is not None:
        if isinstance(color_by, list):
            if len(color_by) != NCELLS:
                raise ValueError(
                    "Length of `color_by` and number of cells do not match. "
                    f" {len(color_by)} != {NCELLS}"
                )
            params["hue"] = color_by
        else:
            raise TypeError(f"`color_by` must be a list. provided {type(color_by)}")

    if size_by is not None:
        if isinstance(size_by, list):
            if len(size_by) != NCELLS:
                raise ValueError(
                    "Length of `size_by` and number of cells do not match. "
                    f" {len(size_by)} != {NCELLS}"
                )
            params["size"] = size_by
        else:
            raise TypeError(f"`size_by` must be a list. provided {type(size_by)}")

        params["sizes"] = (min(params["size"]), max(params["size"]))

    if shape_by is not None:
        if isinstance(shape_by, list):
            if len(shape_by) != NCELLS:
                raise ValueError(
                    "Length of `shape_by` and number of cells do not match. "
                    f" {len(shape_by)} != {NCELLS}"
                )
            params["markers"] = shape_by
        else:
            raise TypeError(f"`shape_by` must be a list. provided {type(shape_by)}")

    return _dim_plot(x=x[:, 0].tolist(), y=x[:, 1].tolist(), **params)


@plot_reduced_dim.register
def _plot_reduced_dim_sce(
    x: SingleCellExperiment,
    dimred: Optional[str] = None,
    color_by: Optional[Union[str, Sequence]] = None,
    size_by: Optional[Union[str, Sequence]] = None,
    shape_by: Optional[Union[str, Sequence]] = None,
    assay_name: Optional[Union[str, Sequence]] = None,
) -> FacetGrid:
    _rdims = x.reduced_dim(dimred)
    NCELLS = _rdims.shape[0]

    params = {}

    if color_by is not None:
        if isinstance(color_by, str):
            params["hue"], _where = _extract_variable_from_sce(
                x, var_key="color_by", var_value=color_by, assay=assay_name
            )

            if _where == "gene":
                params["palette"] = cubehelix_palette(as_cmap=True)
            else:
                if len(set(params["hue"])) > len(params["hue"]) / 2:
                    warn("`color_by` contains too many unique values.")
        elif isinstance(color_by, list):
            if len(color_by) != NCELLS:
                raise ValueError(
                    "Length of `color_by` and number of cells do not match. "
                    f" {len(color_by)} != {NCELLS}"
                )
            params["hue"] = color_by
        else:
            raise TypeError(
                f"`color_by` must be a list or a column in col_data. provided {type(color_by)}"
            )

    if size_by is not None:
        if isinstance(size_by, str):
            params["size"], _where = _extract_variable_from_sce(
                x, var_key="size_by", var_value=size_by, assay=assay_name
            )

            if not (
                is_list_of_type(params["size"], float)
                or is_list_of_type(params["size"], int)
            ):
                raise ValueError("All values in `size_by` must only contain numbers.")
        elif isinstance(size_by, list):
            if len(size_by) != NCELLS:
                raise ValueError(
                    "Length of `size_by` and number of cells do not match. "
                    f" {len(size_by)} != {NCELLS}"
                )
            params["size"] = size_by
        else:
            raise TypeError(
                f"`size_by` must be a list or a column in col_data. provided {type(size_by)}"
            )
        # params["sizes"] = (min(params["size"]), max(params["size"]))

    if shape_by is not None:
        if isinstance(shape_by, str):
            params["markers"], _where = _extract_variable_from_sce(
                x, var_key="shape_by", var_value=shape_by, assay=assay_name
            )

            if _where == "gene":
                raise ValueError("`shape_by` must be a categorical variable.")
            else:
                if len(set(params["markers"])) > len(params["markers"]) / 2:
                    warn("`shape_by` contains too many unique values.")
        elif isinstance(shape_by, list):
            if len(shape_by) != NCELLS:
                raise ValueError(
                    "Length of `shape_by` and number of cells do not match. "
                    f" {len(shape_by)} != {NCELLS}"
                )
            params["markers"] = shape_by
        else:
            raise TypeError(
                f"`shape_by` must be a list or a column in col_data. provided {type(shape_by)}"
            )

    return _dim_plot(x=_rdims[:, 0].tolist(), y=_rdims[:, 1].tolist(), **params)
