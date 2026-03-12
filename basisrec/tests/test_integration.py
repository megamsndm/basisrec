"""End-to-end integration tests.

These tests run the full pipeline from XYZ file → ORCA input.
They are marked 'integration' so they can be skipped in fast CI::

    pytest -m "not integration"
"""

import pytest
from unittest.mock import patch

from basisrec import recommend
from basisrec.models import RecommendationResult


@pytest.mark.integration
def test_full_pipeline_water(water_xyz):
    with patch("basisrec.orca_writer.fetch_basis_block_orca", return_value="S 1 1.00\n 1.0 1.0"):
        result = recommend(water_xyz, use_gnn=False)

    assert isinstance(result, RecommendationResult)
    assert result.molecule.n_atoms == 3
    assert len(result.recommendations) == 3
    assert len(result.orca_input) > 200
    assert "* xyz" in result.orca_input
    assert "%basis" in result.orca_input
    assert len(result.report) > 50


@pytest.mark.integration
def test_full_pipeline_caffeine(caffeine_xyz):
    with patch("basisrec.orca_writer.fetch_basis_block_orca", return_value="S 1 1.00\n 1.0 1.0"):
        result = recommend(caffeine_xyz, use_gnn=False)

    assert result.molecule.n_atoms == 24
    assert len(result.recommendations) == 24
    elements = {r.symbol for r in result.recommendations}
    assert "C" in elements
    assert "N" in elements
    assert "O" in elements
    assert "H" in elements


@pytest.mark.integration
def test_output_file_written(water_xyz, tmp_path):
    out = tmp_path / "water.inp"
    with patch("basisrec.orca_writer.fetch_basis_block_orca", return_value="S 1 1.00\n 1.0 1.0"):
        result = recommend(water_xyz, use_gnn=False, output_path=out)
    assert out.exists()
    content = out.read_text()
    assert "B3LYP" in content or "RKS" in content


@pytest.mark.integration
def test_custom_method_line(water_xyz):
    with patch("basisrec.orca_writer.fetch_basis_block_orca", return_value="S 1 1.00\n 1.0 1.0"):
        result = recommend(water_xyz, method_line="! RKS PBE0 TightSCF", use_gnn=False)
    assert "PBE0" in result.orca_input


@pytest.mark.integration
def test_charged_molecule(water_xyz):
    with patch("basisrec.orca_writer.fetch_basis_block_orca", return_value="S 1 1.00\n 1.0 1.0"):
        result = recommend(water_xyz, charge=-1, multiplicity=2, use_gnn=False)
    assert "* xyz -1 2" in result.orca_input


@pytest.mark.integration
def test_unique_basis_sets_non_empty(water_xyz):
    with patch("basisrec.orca_writer.fetch_basis_block_orca", return_value="S 1 1.00\n 1.0 1.0"):
        result = recommend(water_xyz, use_gnn=False)
    assert len(result.unique_basis_sets()) >= 1


@pytest.mark.integration
def test_report_contains_reasoning(water_xyz):
    with patch("basisrec.orca_writer.fetch_basis_block_orca", return_value="S 1 1.00\n 1.0 1.0"):
        result = recommend(water_xyz, use_gnn=False)
    assert "Reasoning" in result.report
    assert "O" in result.report
