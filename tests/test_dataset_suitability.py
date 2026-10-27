
import numpy as np
import pandas as pd
import pytest
from sc_workflows.metadata import assess_dataset_suitability


def test_repeated_donors_not_cells():
    obs = pd.DataFrame({
        "donor": ["A", "A", "A", "B", "B"],
        "sample": ["s1", "s1", "s2", "s3", "s3"],
        "visit": ["v1", "v1", "v2", "v1", "v1"],
    })
    result = assess_dataset_suitability(obs, "donor", "visit", "sample")
    assert result["n_donors"] == 2
    assert result["n_samples"] == 3
    assert result["n_donors_with_repeated_visits"] == 1
    assert result["fraction_donors_with_repeated_visits"] == 0.5
    assert result["sample_id_conflicts"] == []


def test_conflicting_sample_assignments():
    obs = pd.DataFrame({
        "donor": ["A", "B"],
        "sample": ["s1", "s1"],
        "visit": ["v1", "v2"],
    })
    result = assess_dataset_suitability(obs, "donor", "visit", "sample")
    assert len(result["sample_id_conflicts"]) == 2


def test_no_time_column():
    obs = pd.DataFrame({"donor": ["A", "B"]})
    result = assess_dataset_suitability(obs, "donor")
    assert result["n_donors_with_repeated_visits"] is None


def test_missing_columns():
    with pytest.raises(ValueError, match="Missing metadata"):
        assess_dataset_suitability(pd.DataFrame({"donor": ["A"]}), "donor", "visit")


def test_min_timepoints_validation():
    with pytest.raises(ValueError, match="min_timepoints"):
        assess_dataset_suitability(pd.DataFrame({"donor": ["A"]}), "donor", min_timepoints=1)
