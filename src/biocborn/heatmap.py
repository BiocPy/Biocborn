from functools import singledispatch
from typing import Optional, Sequence, Union

from biocframe import BiocFrame
from pandas import DataFrame
from seaborn import heatmap
from matplotlib.axes import Axes
from singlecellexperiment import SingleCellExperiment


__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


def _heatmap_plot(x: DataFrame, **kwargs) -> Axes:
    """Function to create the seaborn heatmap plot from the parameters."""
    g = heatmap(x, annot=True, linewidths=0.5)

    return g


@singledispatch
def plot_heatmap(x, assay_name: Optional[str] = None, **kwargs):
    raise NotImplementedError(
        f"`plot_reduced_dim` is not implemented for objects of class: '{type(x)}'."
    )


@plot_heatmap.register
def plot_heatmap(x: BiocFrame, assay_name: Optional[str] = None, **kwargs):
    pass


@plot_heatmap.register
def plot_heatmap(x: DataFrame, assay_name: Optional[str] = None, **kwargs):
    pass


@plot_heatmap.register
def plot_heatmap(
    x: SingleCellExperiment,
    features: Optional[Union[str, Sequence]] = None,
    annotations: Optional[Union[str, Sequence]] = None,
    assay_name: Optional[str] = None,
    **kwargs,
):
    pass
