import numpy as np
import anndata as ad

from sc_workflows.preprocessing import normalize_counts


def test_normalize_counts_preserves_raw():
    counts = np.array([
        [10, 20, 0],
        [5, 0, 15],
        [0, 10, 10],
    ])

    adata = ad.AnnData(X=counts)

    result = normalize_counts(adata)

    np.testing.assert_array_equal(
        result.layers["counts"],
        counts,
    )

    # Original data remains unchanged
    np.testing.assert_array_equal(
        adata.X,
        counts,
    )


def test_normalize_counts_target_sum():
    counts = np.array([
        [10, 20, 0],
        [5, 0, 15],
    ])

    adata = ad.AnnData(X=counts)

    result = normalize_counts(
        adata,
        target_sum=100,
        log1p=False,
    )

    np.testing.assert_allclose(
        result.X.sum(axis=1),
        [100, 100],
    )