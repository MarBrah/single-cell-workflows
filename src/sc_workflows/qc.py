import anndata as ad
import scanpy as sc
import numpy as np

def calculate_qc_metrics(
    adata: ad.AnnData,
    mitochondrial_prefix: str = "MT-",
    ribosomal_prefixes: tuple[str, ...] = ("RPS", "RPL"),
    inplace: bool = False,
    **kwargs,
) -> ad.AnnData:
    """
    Calculate standard single-cell RNA-seq QC metrics.

    Parameters
    ----------
    adata
        AnnData object containing raw gene counts in .X.
    mitochondrial_prefix
        Prefix identifying mitochondrial genes.
        Human: "MT-", mouse: "mt-".
    ribosomal_prefixes
        Prefixes identifying ribosomal protein genes.
    inplace
        If True, modify the input AnnData object.
        Otherwise, return a copy.
    **kwargs
        Additional arguments passed to sc.pp.calculate_qc_metrics.

    Returns
    -------
    AnnData
        AnnData with QC metrics added to .obs and .var.
    """

    if not inplace:
        adata = adata.copy()

    # Identify gene categories
    adata.var["mt"] = adata.var_names.str.startswith(
        mitochondrial_prefix
    )

    adata.var["ribo"] = adata.var_names.str.startswith(
        ribosomal_prefixes
    )

    # Calculate QC metrics
    
    # Defaults can be overridden through kwargs
    qc_kwargs = {
        "qc_vars": ["mt", "ribo"],
        "percent_top": None,
        "log1p": True,
        **kwargs,
    }

    sc.pp.calculate_qc_metrics(
        adata,
        inplace = True,
        **qc_kwargs,
    )

    return adata

def filter_qc(
    adata: ad.AnnData,
    min_thresholds: dict[str, float] | None = None,
    max_thresholds: dict[str, float] | None = None,
    min_cells_per_gene: int | None = None,
    inplace: bool = False,
) -> ad.AnnData:
    """
    Filter cells using QC metrics and optionally filter genes.

    Parameters
    ----------
    adata
        AnnData containing calculated QC metrics in .obs.
    min_thresholds
        Minimum acceptable values for QC metrics (inclusive).
    max_thresholds
        Maximum acceptable values for QC metrics (inclusive).
    min_cells_per_gene
        Minimum number of cells in which a gene must be detected.
    inplace
        Whether to modify the input AnnData.

    Returns
    -------
    AnnData
        Filtered AnnData object.
    """

    min_thresholds = min_thresholds or {}
    max_thresholds = max_thresholds or {}

    # Check that requested metrics exist
    metrics = set(min_thresholds) | set(max_thresholds)
    missing = metrics - set(adata.obs.columns)

    if missing:
        raise ValueError(
            f"QC metrics missing from adata.obs: {sorted(missing)}"
        )

    # Build a combined cell filtering mask
    mask = np.ones(adata.n_obs, dtype=bool)

    for metric, threshold in min_thresholds.items():
        mask &= adata.obs[metric].to_numpy() >= threshold

    for metric, threshold in max_thresholds.items():
        mask &= adata.obs[metric].to_numpy() <= threshold

    # Filter cells
    result = adata[mask, :].copy()

    # Optionally filter genes
    if min_cells_per_gene is not None:
        sc.pp.filter_genes(
            result,
            min_cells=min_cells_per_gene,
        )

    return result