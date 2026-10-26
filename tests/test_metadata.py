import numpy as np
import pandas as pd
import anndata as ad

from sc_workflows.metadata import (
    summarize_metadata,
    summarize_categories,
    summarize_cohort,
    assess_longitudinal_coverage,
)


def make_adata():
    obs = pd.DataFrame({
        "donor": ["P1", "P1", "P1", "P2", "P2"],
        "sample": ["S1", "S1", "S2", "S3", "S3"],
        "visit": ["V1", "V1", "V2", "V1", "V1"],
        "diagnosis": ["PD", "PD", "PD", "Control", "Control"],
        "cell_type": ["Neuron", "Astro", "Neuron", "Neuron", "Astro"],
    })

    return ad.AnnData(
        X=np.ones((5, 3)),
        obs=obs,
    )


def test_summarize_metadata():
    result = summarize_metadata(make_adata())
    assert "donor" in result["column"].values
    assert "pct_missing" in result.columns


def test_summarize_categories():
    result = summarize_categories(
        make_adata(), "diagnosis"
    )
    assert result["count"].sum() == 5


def test_summarize_cohort():
    result = summarize_cohort(
        make_adata(),
        donor_col="donor",
        sample_col="sample",
    )
    assert result.loc[
        result["donor"] == "P1", "n_samples"
    ].iloc[0] == 2


def test_longitudinal_coverage():
    result = assess_longitudinal_coverage(
        make_adata(),
        donor_col="donor",
        time_col="visit",
    )

    p1 = result[result["donor"] == "P1"].iloc[0]
    p2 = result[result["donor"] == "P2"].iloc[0]

    assert p1["n_timepoints"] == 2
    assert p1["is_longitudinal"]
    assert not p2["is_longitudinal"]