import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import anndata as ad

from sc_workflows.visualization import (
    plot_qc_violin,
    plot_qc_scatter,
)


def test_qc_plots():
    adata = ad.AnnData(
        X=np.ones((10, 3)),
        obs=pd.DataFrame({
            "total_counts": np.arange(10) + 1,
            "n_genes_by_counts": np.arange(10) + 2,
            "pct_counts_mt": np.arange(10) * 2,
        }),
    )

    plot_qc_violin(adata, show=False)
    plot_qc_scatter(adata, show=False)

    plt.close("all")