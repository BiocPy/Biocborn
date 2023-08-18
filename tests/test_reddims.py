from biocborn.reduced_dims import plot_reduced_dim
from seaborn import FacetGrid

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


def test_plot_reduced_dim(mock_data):
    sce = mock_data.sce
    g = plot_reduced_dim(sce, dimred="tsne")

    assert isinstance(g, FacetGrid)
