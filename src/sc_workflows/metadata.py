import anndata as ad
import pandas as pd


def summarize_metadata(
    adata: ad.AnnData,
    axis: str = "obs",
    max_examples: int = 5,
) -> pd.DataFrame:
    """
    Summarize metadata columns in an AnnData object.

    Parameters
    ----------
    adata
        AnnData object.
    axis
        'obs' for cell metadata, 'var' for gene metadata.
    max_examples
        Maximum number of example values per column.

    Returns
    -------
    pd.DataFrame
        Metadata summary with types, missingness,
        unique counts, and example values.
    """

    if axis not in ("obs", "var"):
        raise ValueError("axis must be 'obs' or 'var'")

    df = adata.obs if axis == "obs" else adata.var

    records = []

    for col in df.columns:
        series = df[col]
        examples = series.dropna().unique()[:max_examples]

        records.append({
            "column": col,
            "dtype": str(series.dtype),
            "n_unique": series.nunique(dropna=True),
            "n_missing": series.isna().sum(),
            "pct_missing": round(
                series.isna().mean() * 100, 2
            ),
            "examples": list(examples),
        })

    return pd.DataFrame(records)

def summarize_categories(
    adata: ad.AnnData,
    column: str,
    normalize: bool = False,
    dropna: bool = False,
) -> pd.DataFrame:
    """
    Count observations belonging to each category.

    Parameters
    ----------
    column
        Column in adata.obs.
    normalize
        If True, return proportions instead of counts.
    dropna
        Whether to exclude missing values.
    """

    if column not in adata.obs.columns:
        raise ValueError(
            f"Column '{column}' not found in adata.obs"
        )

    counts = adata.obs[column].value_counts(
        normalize=normalize,
        dropna=dropna,
    )

    return counts.rename(
        "proportion" if normalize else "count"
    ).reset_index()

def summarize_cohort(
    adata: ad.AnnData,
    donor_col: str,
    sample_col: str | None = None,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Summarize cell counts and sample coverage per donor.

    Parameters
    ----------
    donor_col
        Column identifying patients/donors.
    sample_col
        Optional column identifying biological samples.
    group_cols
        Additional metadata columns to summarize per donor.
    """

    obs = adata.obs

    required = [donor_col]

    if sample_col is not None:
        required.append(sample_col)

    if group_cols:
        required.extend(group_cols)

    missing = set(required) - set(obs.columns)

    if missing:
        raise ValueError(
            f"Missing metadata columns: {sorted(missing)}"
        )

    grouped = obs.groupby(
        donor_col,
        observed=True,
        dropna=False,
    )

    result = grouped.size().rename("n_cells").to_frame()

    if sample_col is not None:
        result["n_samples"] = grouped[
            sample_col
        ].nunique()

    for col in group_cols or []:
        result[col] = grouped[col].agg(
            lambda x: list(pd.unique(x.dropna()))
        )

    return result.reset_index()

def assess_longitudinal_coverage(
    adata: ad.AnnData,
    donor_col: str,
    time_col: str,
    sample_col: str | None = None,
) -> pd.DataFrame:
    """
    Assess longitudinal sampling coverage per donor.

    Parameters
    ----------
    donor_col
        Patient identifier column.
    time_col
        Timepoint or visit identifier column.
    sample_col
        Optional sample identifier column.

    Returns
    -------
    pd.DataFrame
        One row per donor with timepoint and sample coverage.
    """

    obs = adata.obs

    required = [donor_col, time_col]

    if sample_col is not None:
        required.append(sample_col)

    missing = set(required) - set(obs.columns)

    if missing:
        raise ValueError(
            f"Missing metadata columns: {sorted(missing)}"
        )

    grouped = obs.groupby(
        donor_col,
        observed=True,
        dropna=False,
    )

    result = grouped.agg(
        n_cells=(time_col, "size"),
        n_timepoints=(time_col, "nunique"),
        timepoints=(
            time_col,
            lambda x: list(pd.unique(x.dropna())),
        ),
    )

    if sample_col is not None:
        result["n_samples"] = grouped[
            sample_col
        ].nunique()

    result["is_longitudinal"] = (
        result["n_timepoints"] > 1
    )

    return result.reset_index()