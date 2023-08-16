from functools import singledispatch
from typing import Optional, Sequence

import seaborn as sns
from singlecellexperiment import SingleCellExperiment

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


def _extract_variable(
    x: SingleCellExperiment, var_key: str, var_value: str, assay: str
) -> Sequence:
    _variable = None
    if var_value is not None:
        if var_value in x.colData.colNames:
            _variable = x.colData.column(var_value)
        elif x.rowData.rowNames is not None and var_value in x.rowData.rowNames:
            _var_idx = x.rowData.rowNames.index(var_value)
            _variable = x.assay(assay)[_var_idx, :]
        else:
            raise ValueError(
                f"`{var_key}`:'{var_value}' is neither a cell annotation column nor a gene symbol."
            )

    return _variable


@singledispatch
def plot_reduced_dim(
    x,
    dimred: str,
    color_by: Optional[str] = None,
    size_by: Optional[str] = None,
    ncomponents: int = 2,
):
    raise NotImplementedError(
        f"`plot_reduced_dim` is not implemented for objects of class: '{type(x)}'."
    )


@plot_reduced_dim.register
def plot_reduced_dim_sce(
    x: SingleCellExperiment,
    dimred: str,
    color_by: Optional[str] = None,
    size_by: Optional[str] = None,
    shape_by: Optional[str] = None,
    assay_name: Optional[str] = None,
    ncomponents: int = 2,
):
    _rdims = x.reducedDim(dimred)

    params = {}

    if color_by is not None:
        params["hue"] = _extract_variable(
            x, var_key="color_by", var_value=color_by, assay=assay_name
        )
        params["palette"] = sns.cubehelix_palette(as_cmap=True)

    if size_by is not None:
        params["size"] = _extract_variable(
            x, var_key="size_by", var_value=size_by, assay=assay_name
        )

        params["sizes"] = (min(_sizeby), max(_sizeby))

    if shape_by is not None:
        params["markers"] = _extract_variable(
            x, var_key="shape_by", var_value=shape_by, assay=assay_name
        )

    g = sns.relplot(x=_rdims[:, 0].tolist(), y=_rdims[:, 1].tolist(), **params)
    g.ax.xaxis.grid(True, "minor", linewidth=0.25)
    g.ax.yaxis.grid(True, "minor", linewidth=0.25)
    g.despine(left=True, bottom=True)
