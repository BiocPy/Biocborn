from functools import singledispatch
from typing import Optional, Sequence, Union

from biocframe import BiocFrame
from matplotlib.axes import Axes
from numpy import ndarray, repeat
from pandas import DataFrame
from scipy.sparse import sparray
from seaborn import heatmap
from singlecellexperiment import SingleCellExperiment

from .utils import _extract_variable_from_frame

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


def _heatmap_plot(x, **kwargs) -> Axes:
    """Function to create the seaborn heatmap plot from the parameters."""
    g = heatmap(x, annot=True, linewidths=0.5, kwargs=kwargs)

    return g


@singledispatch
def plot_heatmap(x, assay_name: Optional[str] = None, **kwargs):
    raise NotImplementedError(
        f"`plot_reduced_dim` is not implemented for objects of class: '{type(x)}'."
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
                f"`features` must be a list or a column in row_data. provided {type(features)}."
            )
    else:
        _features_vec = [f"row_{i}" for i in range(x.shape[0])]

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
                f"`annotations` must be a list or a column in col_data. provided {type(annotations)}."
            )
    else:
        _annotation_vec = [f"column_{i}" for i in range(x.shape[1])]

    _frame = None

    _assay = x.assay(assay_name)
    if isinstance(_assay, ndarray):
        _assay_flatten = _assay.flatten()
        _columns = repeat(_annotation_vec, _assay.shape[0])
        _rows = _features_vec * _assay.shape[1]

        _frame = DataFrame(
            {"features": _rows, "annotations": _columns, "value": _assay_flatten}
        )
    elif isinstance(_assay, sparray):
        _coo_mat = _assay.tocoo()
        _frame = DataFrame(
            {
                "features": [_features_vec[i] for i in _coo_mat.row],
                "annotations": [_annotation_vec[i] for i in _coo_mat.col],
                "value": _coo_mat.data,
            }
        )

    print(_frame.pivot(index="features", columns="annotations", values="value"))

    return _heatmap_plot(_frame, kwargs=kwargs)
