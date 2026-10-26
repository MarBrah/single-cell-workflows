import numpy as np
import pandas as pd
import anndata as ad
import pytest

from sc_workflows.qc import filter_qc


@pytest.fixture
def example_adata():
    counts = np.array([
        [1, 2, 0],
        [0, 3, 4],
        [5, 0, 1],
        [1, 1, 1],
    ])

    obs = pd.DataFrame({
        "n_genes_by_counts": [2, 2, 2, 3],
        "pct_counts_mt": [10, 30, 5, 15],
    })

    return ad.AnnData(X=counts, obs=obs)


def test_filter_cells(example_adata):
    result = filter_qc(
        example_adata,
        min_thresholds={"n_genes_by_counts": 2},
        max_thresholds={"pct_counts_mt": 15},
    )

    assert result.n_obs == 3
    assert example_adata.n_obs == 4


def test_missing_metric(example_adata):
    with pytest.raises(ValueError):
        filter_qc(
            example_adata,
            max_thresholds={"unknown_metric": 10},
        )


def test_filter_genes(example_adata):
    result = filter_qc(
        example_adata,
        min_cells_per_gene=3,
    )

    assert result.n_vars == 3