import anndata as ad
import scanpy as sc


def normalize_counts(
    adata: ad.AnnData,
    target_sum: float = 1e4,
    log1p: bool = True,
    counts_layer: str = "counts",
    **kwargs,
) -> ad.AnnData:
    """
    Library-size normalize single-cell RNA-seq counts.

    Parameters
    ----------
    adata
        AnnData containing raw counts in .X.
    target_sum
        Target total counts per cell after normalization.
    log1p
        Whether to apply log(1 + x) transformation.
    counts_layer
        Layer used to preserve original raw counts.
    **kwargs
        Additional arguments passed to sc.pp.normalize_total.

    Returns
    -------
    AnnData
        New AnnData with normalized expression in .X
        and raw counts preserved in .layers[counts_layer].
    """

    result = adata.copy()

    if counts_layer in result.layers:
        raise ValueError(
            f"Layer '{counts_layer}' already exists. "
            "Refusing to overwrite existing data."
        )

    # Preserve original raw counts
    result.layers[counts_layer] = result.X.copy()

    # Normalize library sizes
    sc.pp.normalize_total(
        result,
        target_sum=target_sum,
        inplace=True,
        **kwargs,
    )

    if log1p:
        sc.pp.log1p(result)

    return result