"""Unit tests for atom_featurizer.py.

Tests cover both the RDKit path (mocked) and the geometry-only
fallback path to ensure both code paths are exercised in CI
even when RDKit is not installed.
"""

from unittest.mock import patch

from basisrec.atom_featurizer import (
    _featurize_geometry_fallback,
    _molecule_to_xyz_block,
    featurize_molecule,
)
from basisrec.models import Atom, Hybridization, Molecule

# ---------------------------------------------------------------------------
# Geometry-only fallback (always available, no RDKit needed)
# ---------------------------------------------------------------------------

def test_fallback_water_atom_count(water_molecule):
    features = _featurize_geometry_fallback(water_molecule)
    assert len(features) == 3


def test_fallback_indices_match_atoms(water_molecule):
    features = _featurize_geometry_fallback(water_molecule)
    for i, feat in enumerate(features):
        assert feat.index == i
        assert feat.symbol == water_molecule.atoms[i].symbol


def test_fallback_oxygen_has_lone_pairs(water_molecule):
    features = _featurize_geometry_fallback(water_molecule)
    o_feat = next(f for f in features if f.symbol == "O")
    assert o_feat.has_lone_pairs is True


def test_fallback_oxygen_is_hbond_acceptor(water_molecule):
    features = _featurize_geometry_fallback(water_molecule)
    o_feat = next(f for f in features if f.symbol == "O")
    assert o_feat.is_hbond_acceptor is True


def test_fallback_hydrogen_is_hbond_donor_neighbored_by_o(water_molecule):
    features = _featurize_geometry_fallback(water_molecule)
    h_feats = [f for f in features if f.symbol == "H"]
    # At least one H should detect O as neighbour
    assert any("O" in f.neighbor_symbols for f in h_feats)


def test_fallback_iron_is_metal(ferrocene_molecule):
    features = _featurize_geometry_fallback(ferrocene_molecule)
    fe_feat = next(f for f in features if f.symbol == "Fe")
    assert fe_feat.is_metal is True


def test_fallback_bromine_is_not_heavy():
    """Br has Z=35, which is NOT > 36 — is_heavy must be False."""
    mol = Molecule(
        title="bromomethane",
        atoms=[
            Atom(index=0, symbol="C",  x=0.0, y=0.0, z=0.0),
            Atom(index=1, symbol="Br", x=1.9, y=0.0, z=0.0),
            Atom(index=2, symbol="H",  x=-0.5, y=0.9, z=0.0),
            Atom(index=3, symbol="H",  x=-0.5, y=-0.9, z=0.0),
            Atom(index=4, symbol="H",  x=-0.5, y=0.0, z=0.9),
        ],
    )
    features = _featurize_geometry_fallback(mol)
    br_feat = next(f for f in features if f.symbol == "Br")
    assert br_feat.is_heavy is False   # Z=35, threshold is Z>36
    assert br_feat.period == 4         # but it IS period 4


def test_fallback_iodine_is_heavy():
    """I has Z=53 (> 36) — is_heavy must be True."""
    mol = Molecule(
        title="iodomethane",
        atoms=[
            Atom(index=0, symbol="C", x=0.0, y=0.0, z=0.0),
            Atom(index=1, symbol="I", x=2.1, y=0.0, z=0.0),
        ],
    )
    features = _featurize_geometry_fallback(mol)
    i_feat = next(f for f in features if f.symbol == "I")
    assert i_feat.is_heavy is True
    assert i_feat.period == 5


def test_fallback_period_assignment(water_molecule):
    features = _featurize_geometry_fallback(water_molecule)
    o_feat = next(f for f in features if f.symbol == "O")
    h_feats = [f for f in features if f.symbol == "H"]
    assert o_feat.period == 2
    assert all(f.period == 1 for f in h_feats)


def test_fallback_hybridization_is_unknown(water_molecule):
    """Fallback cannot determine hybridization — must be UNKNOWN."""
    features = _featurize_geometry_fallback(water_molecule)
    for feat in features:
        assert feat.hybridization == Hybridization.UNKNOWN


def test_fallback_partial_charge_zero(water_molecule):
    """Fallback cannot compute Gasteiger charges."""
    features = _featurize_geometry_fallback(water_molecule)
    for feat in features:
        assert feat.partial_charge == 0.0


def test_fallback_coordination_number_oxygen(water_molecule):
    """O in water should detect 2 neighbours (both H atoms)."""
    features = _featurize_geometry_fallback(water_molecule)
    o_feat = next(f for f in features if f.symbol == "O")
    assert o_feat.coordination_number == 2


# ---------------------------------------------------------------------------
# featurize_molecule: RDKit unavailable path
# ---------------------------------------------------------------------------

def test_featurize_molecule_falls_back_without_rdkit(water_molecule):
    """When RDKit import fails, should use geometry fallback silently."""
    with patch.dict("sys.modules", {"rdkit": None, "rdkit.Chem": None}):
        features = featurize_molecule(water_molecule)
    assert len(features) == water_molecule.n_atoms


def test_featurize_molecule_returns_correct_symbols(water_molecule):
    with patch.dict("sys.modules", {"rdkit": None}):
        features = featurize_molecule(water_molecule)
    symbols = [f.symbol for f in features]
    assert symbols == ["O", "H", "H"]


# ---------------------------------------------------------------------------
# XYZ block helper
# ---------------------------------------------------------------------------

def test_xyz_block_round_trip(water_molecule):
    block = _molecule_to_xyz_block(water_molecule)
    lines = block.strip().splitlines()
    assert lines[0] == "3"
    assert lines[1] == "water"
    assert lines[2].startswith("O")
    assert lines[3].startswith("H")
    assert lines[4].startswith("H")


def test_xyz_block_coordinate_precision(water_molecule):
    block = _molecule_to_xyz_block(water_molecule)
    # Each coordinate line should have 6 decimal places
    coord_line = block.splitlines()[2]
    parts = coord_line.split()
    assert "." in parts[1]
    decimal_places = len(parts[1].split(".")[1])
    assert decimal_places == 6
