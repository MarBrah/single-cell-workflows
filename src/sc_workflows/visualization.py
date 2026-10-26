import anndata as ad
import scanpy as sc


def plot_qc_violin(
    adata: ad.AnnData,
    keys: tuple[str, ...] = (
        "n_genes_by_counts",
        "total_counts",
        "pct_counts_mt",
    ),
    **kwargs,
):
    """
    Visualize distributions of cell-level QC metrics.

    Parameters
    ----------
    adata
        AnnData object with QC metrics in .obs.
    keys
        QC metrics to visualize.
    **kwargs
        Additional arguments passed to sc.pl.violin.
    """

    plot_kwargs = {
        "multi_panel": True,
        "jitter": 0.4,
        **kwargs,
    }

    return sc.pl.violin(
        adata,
        keys=list(keys),
        **plot_kwargs,
    )


def plot_qc_scatter(
    adata: ad.AnnData,
    x: str = "total_counts",
    y: str = "n_genes_by_counts",
    color: str = "pct_counts_mt",
    **kwargs,
):
    """
    Visualize relationships between QC metrics.

    Additional arguments are passed to sc.pl.scatter.
    """

    return sc.pl.scatter(
        adata,
        x=x,
        y=y,
        color=color,
        **kwargs,
    )