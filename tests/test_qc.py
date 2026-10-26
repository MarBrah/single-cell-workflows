import numpy as np
import pandas as pd
import anndata as ad

from sc_workflows.qc import calculate_qc_metrics


def test_calculate_qc_metrics():

    counts = np.array([
        [1, 2, 0],
        [0, 3, 4],
        [5, 0, 1],
    ])

    adata = ad.AnnData(
        X=counts,
        var=pd.DataFrame(
            index=["MT-CO1", "RPS3", "GENE1"]
        ),
    )

    result = calculate_qc_metrics(adata)

    assert "pct_counts_mt" in result.obs
    assert "pct_counts_ribo" in result.obs
    assert "total_counts" in result.obs

    # Original object should remain unchanged
    assert "total_counts" not in adata.obs

    # First cell has 1 mitochondrial count out of 3
    assert np.isclose(
        result.obs["pct_counts_mt"].iloc[0],
        100 / 3,
    )