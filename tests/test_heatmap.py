from biocborn.heatmap import plot_heatmap
from matplotlib.axes import Axes

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


def test_plot_heatmap(mock_data):
    sce = mock_data.sce
    g = plot_heatmap(sce, annotations="treatment")

    assert isinstance(g, Axes)
