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

def assess_dataset_suitability(
    metadata: pd.DataFrame | ad.AnnData,
    donor_col: str,
    time_col: str | None = None,
    sample_col: str | None = None,
    *,
    min_timepoints: int = 2,
) -> dict:
    """Screen observed sampling structure, not biological suitability.

    Accepts one row per sample or cell. Time labels represent observed
    visits; disease stages or inferred pseudotime must not be passed
    as longitudinal visits.
    """
    if min_timepoints < 2:
        raise ValueError("min_timepoints must be at least 2")
    obs = metadata.obs if isinstance(metadata, ad.AnnData) else metadata
    if not isinstance(obs, pd.DataFrame):
        raise TypeError("metadata must be a DataFrame or AnnData")
    required = [donor_col] + ([time_col] if time_col else []) + ([sample_col] if sample_col else [])
    missing = sorted(set(required).difference(obs.columns))
    if missing:
        raise ValueError(f"Missing metadata columns: {missing}")
    if len(set(required)) != len(required):
        raise ValueError("donor, time, and sample columns must be distinct")

    # Each observed sample must belong to exactly one donor and visit.
    conflicts = []
    if sample_col:
        sample_rows = obs.loc[obs[sample_col].notna()]
        for col in [donor_col] + ([time_col] if time_col else []):
            inconsistent = sample_rows.groupby(sample_col, observed=True)[col].nunique(dropna=True)
            n_conflicts = int((inconsistent > 1).sum())
            if n_conflicts:
                conflicts.append(f"{n_conflicts} sample IDs map to multiple {col} values")
        # Same sample ID cannot simultaneously have missing and present identifiers.
        for col in [donor_col] + ([time_col] if time_col else []):
            mixed = sample_rows.groupby(sample_col, observed=True)[col].agg(
                lambda s: bool(s.isna().any() and s.notna().any())
            )
            if mixed.any():
                conflicts.append(f"{int(mixed.sum())} sample IDs mix missing and present {col} values")

    usable = obs.loc[obs[donor_col].notna()]
    donor_count = int(usable[donor_col].nunique())
    n_samples = int(obs[sample_col].nunique()) if sample_col else None
    n_timepoints = None
    n_repeated = None
    repeated_fraction = None
    coverage = None
    if time_col:
        donor_time = usable.groupby(donor_col, observed=True)[time_col].nunique(dropna=True)
        n_repeated = int((donor_time >= min_timepoints).sum())
        repeated_fraction = n_repeated / donor_count if donor_count else None
        n_timepoints = int(usable[time_col].nunique(dropna=True))
        coverage = donor_time.rename("n_timepoints").reset_index()
        coverage["has_repeated_visits"] = coverage["n_timepoints"] >= min_timepoints

    limitations = []
    if time_col is None:
        limitations.append("No visit/time column specified; longitudinal sampling cannot be assessed")
    elif n_repeated == 0:
        limitations.append("No donors with repeated observed visits")
    if obs[donor_col].isna().any():
        limitations.append("Missing donor identifiers")
    if time_col and obs[time_col].isna().any():
        limitations.append("Missing visit/time labels")
    limitations.extend(conflicts)
    limitations.append("Visit labels alone do not establish elapsed time, clinical progression, or temporal ordering")
    limitations.append("Expression data, per-sample cell counts, covariates, and statistical power were not assessed")

    return {
        "n_rows": len(obs),
        "n_donors": donor_count,
        "n_samples": n_samples,
        "n_distinct_time_labels": n_timepoints,
        "n_donors_with_repeated_visits": n_repeated,
        "fraction_donors_with_repeated_visits": repeated_fraction,
        "missingness_pct": {
            col: round(float(obs[col].isna().mean() * 100), 2) if len(obs) else None
            for col in required
        },
        "sample_id_conflicts": conflicts,
        "donor_coverage": coverage,
        "limitations": limitations,
    }