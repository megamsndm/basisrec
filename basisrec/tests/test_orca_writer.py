"""Unit tests for orca_writer.py."""

from unittest.mock import patch

import pytest

from basisrec.models import BasisRecommendation, BasisTier
from basisrec.orca_writer import _geometry_block, _header_comment, write_orca_input


@pytest.fixture
def simple_recs():
    return [
        BasisRecommendation(
            atom_index=0, symbol="O", basis_name="aug-cc-pVDZ",
            tier=BasisTier.AUGMENTED, confidence=0.92,
            reason="O: lone pairs", source="rule",
        ),
        BasisRecommendation(
            atom_index=1, symbol="H", basis_name="6-31G*",
            tier=BasisTier.DOUBLE_ZETA, confidence=0.88,
            reason="H: light", source="rule",
        ),
        BasisRecommendation(
            atom_index=2, symbol="H", basis_name="6-31G*",
            tier=BasisTier.DOUBLE_ZETA, confidence=0.88,
            reason="H: light", source="rule",
        ),
    ]


def test_geometry_block_contains_all_atoms(water_molecule):
    block = _geometry_block(water_molecule)
    assert "* xyz 0 1" in block
    assert block.count("O") >= 1
    assert block.count("H") >= 2
    assert block.endswith("*")


def test_geometry_block_charge_multiplicity(water_molecule):
    water_molecule = water_molecule.model_copy(update={"charge": -1, "multiplicity": 2})
    block = _geometry_block(water_molecule)
    assert "* xyz -1 2" in block


def test_header_comment_contains_all_atoms(water_molecule, simple_recs):
    header = _header_comment(water_molecule, simple_recs)
    assert "O" in header
    assert "H" in header
    assert "aug-cc-pVDZ" in header
    assert "6-31G*" in header


def test_write_orca_contains_method_line(water_molecule, simple_recs):
    with patch("basisrec.orca_writer.fetch_basis_block_orca", return_value="S 1 1.00\n 1.0 1.0"):
        content = write_orca_input(water_molecule, simple_recs)
    assert "B3LYP" in content or "RKS" in content


def test_write_orca_contains_newgto(water_molecule, simple_recs):
    with patch("basisrec.orca_writer.fetch_basis_block_orca", return_value="S 1 1.00\n 1.0 1.0"):
        content = write_orca_input(water_molecule, simple_recs)
    assert "newgto" in content


def test_write_orca_contains_geometry(water_molecule, simple_recs):
    with patch("basisrec.orca_writer.fetch_basis_block_orca", return_value="S 1 1.00\n 1.0 1.0"):
        content = write_orca_input(water_molecule, simple_recs)
    assert "* xyz" in content
    assert "O" in content


def test_write_orca_saves_file(tmp_path, water_molecule, simple_recs):
    out = tmp_path / "test.inp"
    with patch("basisrec.orca_writer.fetch_basis_block_orca", return_value="S 1 1.00\n 1.0 1.0"):
        write_orca_input(water_molecule, simple_recs, output_path=str(out))
    assert out.exists()
    assert out.stat().st_size > 0


def test_maxcore_nprocs_in_output(water_molecule, simple_recs):
    with patch("basisrec.orca_writer.fetch_basis_block_orca", return_value="S 1 1.00\n 1.0 1.0"):
        content = write_orca_input(water_molecule, simple_recs, maxcore_mb=4000, nprocs=8)
    assert "%maxcore 4000" in content
    assert "nprocs 8" in content
