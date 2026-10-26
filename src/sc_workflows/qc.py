import anndata as ad
import scanpy as sc


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