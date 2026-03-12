"""Unit tests for the public api.py surface.

Verifies that recommend() returns the correct types, handles
all keyword arguments, and surfaces warnings appropriately.
"""

import pytest
from pathlib import Path
from unittest.mock import patch

from basisrec import recommend
from basisrec.models import RecommendationResult, BasisRecommendation
from basisrec.exceptions import ParseError


def test_recommend_returns_result_type(water_xyz):
    with patch("basisrec.orca_writer.fetch_basis_block_orca", return_value="S 1 1.00\n 1.0 1.0"):
        result = recommend(water_xyz, use_gnn=False)
    assert isinstance(result, RecommendationResult)


def test_recommend_molecule_title(water_xyz):
    with patch("basisrec.orca_writer.fetch_basis_block_orca", return_value="S 1 1.00\n 1.0 1.0"):
        result = recommend(water_xyz, use_gnn=False)
    assert result.molecule.title == "water"


def test_recommend_n_recommendations_equals_n_atoms(water_xyz):
    with patch("basisrec.orca_writer.fetch_basis_block_orca", return_value="S 1 1.00\n 1.0 1.0"):
        result = recommend(water_xyz, use_gnn=False)
    assert len(result.recommendations) == result.molecule.n_atoms


def test_recommend_orca_input_is_nonempty(water_xyz):
    with patch("basisrec.orca_writer.fetch_basis_block_orca", return_value="S 1 1.00\n 1.0 1.0"):
        result = recommend(water_xyz, use_gnn=False)
    assert len(result.orca_input) > 100


def test_recommend_report_is_nonempty(water_xyz):
    with patch("basisrec.orca_writer.fetch_basis_block_orca", return_value="S 1 1.00\n 1.0 1.0"):
        result = recommend(water_xyz, use_gnn=False)
    assert len(result.report) > 50


def test_recommend_charge_propagates(water_xyz):
    with patch("basisrec.orca_writer.fetch_basis_block_orca", return_value="S 1 1.00\n 1.0 1.0"):
        result = recommend(water_xyz, charge=-1, multiplicity=2, use_gnn=False)
    assert result.molecule.charge == -1
    assert result.molecule.multiplicity == 2
    assert "* xyz -1 2" in result.orca_input


def test_recommend_custom_method_line(water_xyz):
    with patch("basisrec.orca_writer.fetch_basis_block_orca", return_value="S 1 1.00\n 1.0 1.0"):
        result = recommend(water_xyz, method_line="! RKS PBE0 TightSCF", use_gnn=False)
    assert "PBE0" in result.orca_input


def test_recommend_writes_output_file(water_xyz, tmp_path):
    out = tmp_path / "water_test.inp"
    with patch("basisrec.orca_writer.fetch_basis_block_orca", return_value="S 1 1.00\n 1.0 1.0"):
        recommend(water_xyz, use_gnn=False, output_path=out)
    assert out.exists()
    assert out.read_text().startswith("#")


def test_recommend_basis_per_atom_dict(water_xyz):
    with patch("basisrec.orca_writer.fetch_basis_block_orca", return_value="S 1 1.00\n 1.0 1.0"):
        result = recommend(water_xyz, use_gnn=False)
    d = result.basis_per_atom()
    assert isinstance(d, dict)
    assert 0 in d
    assert isinstance(d[0], str)


def test_recommend_file_not_found():
    with pytest.raises(FileNotFoundError):
        recommend("/nonexistent/path/mol.xyz")


def test_recommend_warnings_is_list(water_xyz):
    with patch("basisrec.orca_writer.fetch_basis_block_orca", return_value="S 1 1.00\n 1.0 1.0"):
        result = recommend(water_xyz, use_gnn=False)
    assert isinstance(result.warnings, list)
