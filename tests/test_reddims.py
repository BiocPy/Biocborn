from biocborn.reduced_dims import plot_reduced_dim

def test_plot_reduced_dim(mock_data):

    sce = mock_data.sce

    plot_reduced_dim(sce, dimred="tsne")