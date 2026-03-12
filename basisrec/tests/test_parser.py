"""Unit tests for xyz_parser.py."""

import pytest

from basisrec.exceptions import ParseError
from basisrec.xyz_parser import parse_xyz, parse_xyz_string


def test_parse_water(water_xyz):
    mol = parse_xyz(water_xyz)
    assert mol.n_atoms == 3
    assert mol.title == "water"
    assert mol.atoms[0].symbol == "O"
    assert mol.atoms[1].symbol == "H"
    assert mol.atoms[2].symbol == "H"


def test_parse_coordinates(water_xyz):
    mol = parse_xyz(water_xyz)
    assert abs(mol.atoms[0].z - 0.119) < 1e-6
    assert abs(mol.atoms[1].y - 0.757) < 1e-6


def test_parse_string():
    xyz = "2\nethanol fragment\nC  0.0 0.0 0.0\nO  1.4 0.0 0.0\n"
    mol = parse_xyz_string(xyz)
    assert mol.n_atoms == 2
    assert mol.atoms[0].symbol == "C"
    assert mol.atoms[1].symbol == "O"


def test_unique_elements(water_xyz):
    mol = parse_xyz(water_xyz)
    assert mol.unique_elements == ["H", "O"]


def test_file_not_found():
    with pytest.raises(FileNotFoundError):
        parse_xyz("/nonexistent/path/molecule.xyz")


def test_malformed_atom_count(tmp_path):
    f = tmp_path / "bad.xyz"
    f.write_text("not_a_number\ntitle\nO 0 0 0\n")
    with pytest.raises(ParseError, match="integer atom count"):
        parse_xyz(f)


def test_too_few_coordinate_lines(tmp_path):
    f = tmp_path / "short.xyz"
    f.write_text("5\ntitle\nO 0.0 0.0 0.0\n")
    with pytest.raises(ParseError, match="5 atoms but only"):
        parse_xyz(f)


def test_invalid_element_symbol(tmp_path):
    f = tmp_path / "invalid.xyz"
    f.write_text("1\ntitle\n123 0.0 0.0 0.0\n")
    with pytest.raises(ParseError, match="valid element symbol"):
        parse_xyz(f)


def test_malformed_coordinates(tmp_path):
    f = tmp_path / "badcoords.xyz"
    f.write_text("1\ntitle\nO notanumber 0.0 0.0\n")
    with pytest.raises(ParseError, match="floats"):
        parse_xyz(f)


def test_empty_title_allowed(tmp_path):
    f = tmp_path / "notitle.xyz"
    f.write_text("1\n\nO 0.0 0.0 0.0\n")
    mol = parse_xyz(f)
    assert mol.title == ""
    assert mol.n_atoms == 1


def test_symbol_normalisation(tmp_path):
    """Lower-case symbols should be normalised to title case."""
    f = tmp_path / "lower.xyz"
    f.write_text("2\ntest\nfe 0.0 0.0 0.0\nn 1.0 0.0 0.0\n")
    mol = parse_xyz(f)
    assert mol.atoms[0].symbol == "Fe"
    assert mol.atoms[1].symbol == "N"
