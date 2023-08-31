from functools import singledispatch
from typing import Optional, Sequence, Union

from biocframe import BiocFrame
from matplotlib.axes import Axes
from numpy import ndarray, repeat
from pandas import DataFrame
from scipy.sparse import issparse
from seaborn import heatmap
from singlecellexperiment import SingleCellExperiment

from .utils import _extract_index_from_frame, _extract_variable_from_frame, _is_unique

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


def _heatmap_plot(x, kwargs) -> Axes:
    """Function to create the seaborn heatmap plot from the parameters."""
    g = heatmap(x, **kwargs)

    return g


@singledispatch
def plot_heatmap(
    x,
    features: Optional[Union[str, Sequence]] = None,
    annotations: Optional[Union[str, Sequence]] = None,
    assay_name: Optional[str] = None,
    **kwargs,
):
    """Plot a heatmap. A wrapper around seaborn's
    `heatmap <https://seaborn.pydata.org/generated/seaborn.heatmap.html>`_ visualization for
    biocpy representations.

    Args:
        x (Any): `Matrix`-like object to visualize.

            This may be a 2-dimensional :py:class:`numpy.ndarray` containing the values to
            visualize.

            Alternatively, ``x`` may be a data frame representation, either a :py:class:`biocframe.BiocFrame.BiocFrame`
            or a  :py:class:`pandas.DataFrame` object. Each column in the frame must contain a numerical
            values. If the data frame is in long format, you might want to perform the pivot operation, e.g.

            .. code-block:: python

                my_df = pd.DataFrame({...})
                pivot_df = my_df.pivot(index=<COL_NAME>, columns=<COL_NAME>, values=<COL_NAME>)

            Checkout documentation for
            `pivot <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.pivot.html>`_.

            Alternatively, ``x`` may be a
            :py:class:`~singlecellexperiment.SingleCellExperiment.SingleCellExperiment` to plot the expression
            values from an assay matrix.

        assay_name (str, optional): Assay to plot expression values from,
            only used if ``x`` is a :py:class:`~singlecellexperiment.SingleCellExperiment.SingleCellExperiment`.

            Defaults to None, in that case the first assay is used.

        features (Union[str, Sequence], optional): A vector of unique feature names.
            only used if ``x`` is a :py:class:`~singlecellexperiment.SingleCellExperiment.SingleCellExperiment`.

        annotations (Union[str, Sequence], optional): A vector of unique annotation names.
            only used if ``x`` is a :py:class:`~singlecellexperiment.SingleCellExperiment.SingleCellExperiment`.

        **kwargs: Additional keyword arguments to forward to :py:func:`~seaborn.heatmap`.

    Raises:
        NotImplementedError: When ``x`` is not an expected type.
    """
    raise NotImplementedError(
        f"`plot_heatmap` is not implemented for objects of class: '{type(x)}'."
    )


@plot_heatmap.register
def _plot_heatmap_ndarray(x: ndarray, **kwargs):
    return _heatmap_plot(x, kwargs=kwargs)


@plot_heatmap.register
def _plot_heatmap_bframe(x: BiocFrame, **kwargs):
    return _heatmap_plot(x.to_pandas(), kwargs=kwargs)


@plot_heatmap.register
def _plot_heatmap_df(x: DataFrame, **kwargs):
    return _heatmap_plot(x, kwargs=kwargs)


@plot_heatmap.register
def _plot_heatmap_sce(
    x: SingleCellExperiment,
    features: Optional[Union[str, Sequence]] = None,
    annotations: Optional[Union[str, Sequence]] = None,
    assay_name: Optional[str] = None,
    **kwargs,
):
    if assay_name is None:
        assay_name = x.assay_names[0]

    if assay_name not in x.assay_names:
        raise ValueError(
            f"SingleCellExperment does not contain '{assay_name}' in assays."
        )

    _features_vec = None
    if features is not None:
        if isinstance(features, str):
            _features_vec = _extract_variable_from_frame(x.row_data, features)

            if _features_vec is None:
                raise ValueError(
                    f"SingleCellExperiment does not contain '{features}' in `row_data`."
                )
        elif isinstance(features, list):
            _features_vec = features
        else:
            raise TypeError(
                f"`features` must be a list or a column in `row_data`. provided {type(features)}."
            )
    else:
        if x.row_data is not None:
            _features_vec = _extract_index_from_frame(x.row_data)

    if _features_vec is None:
        _features_vec = [f"row_{i}" for i in range(x.shape[0])]

    if not _is_unique(_features_vec):
        raise ValueError("`features` must be unique.")

    _annotation_vec = None
    if annotations is not None:
        if isinstance(annotations, str):
            _annotation_vec = _extract_variable_from_frame(x.col_data, annotations)

            if _annotation_vec is None:
                raise ValueError(
                    f"SingleCellExperiment does not contain '{annotations}' in `col_data`."
                )
        elif isinstance(annotations, list):
            _annotation_vec = annotations
        else:
            raise TypeError(
                f"`annotations` must be a list or a column in `col_data`. provided {type(annotations)}."
            )
    else:
        if x.col_data is not None:
            _annotation_vec = _extract_index_from_frame(x.col_data)

    if _annotation_vec is None:
        _annotation_vec = [f"column_{i}" for i in range(x.shape[1])]

    if not _is_unique(_annotation_vec):
        raise ValueError("`annotations` must be unique.")

    _frame = None

    _assay = x.assay(assay_name)
    if isinstance(_assay, ndarray):
        _assay_flatten = _assay.flatten()
        _columns = repeat(_annotation_vec, _assay.shape[0])
        _rows = _features_vec * _assay.shape[1]

        _frame = DataFrame(
            {"features": _rows, "annotations": _columns, "values": _assay_flatten}
        )
    elif issparse(_assay):
        _coo_mat = _assay.tocoo()
        _frame = DataFrame(
            {
                "features": [_features_vec[i] for i in _coo_mat.row],
                "annotations": [_annotation_vec[i] for i in _coo_mat.col],
                "values": _coo_mat.data,
            }
        )

    _frame = _frame.pivot(index="features", columns="annotations", values="values")

    return _heatmap_plot(_frame, kwargs=kwargs)
