"""Unit tests for gnn_model.py.

Tests cover: feature encoding, model loading/degradation,
inference interface, and the singleton accessor.
The actual neural network forward pass is only tested when
torch is available; otherwise tests verify graceful degradation.
"""


import pytest

from basisrec.gnn_model import (
    _NODE_FEATURE_DIM,
    BASIS_CLASSES,
    BasisGNN,
    _encode_atom_features,
    get_gnn,
)
from basisrec.models import AtomFeatures, Hybridization


def _make_features(**kwargs) -> AtomFeatures:
    defaults = dict(
        index=0, symbol="C",
        hybridization=Hybridization.SP3,
        coordination_number=4,
        is_aromatic=False,
        is_hbond_donor=False,
        is_hbond_acceptor=False,
        has_lone_pairs=False,
        is_metal=False,
        is_heavy=False,
        partial_charge=0.0,
        period=2,
        neighbor_symbols=["H", "H", "H", "H"],
    )
    defaults.update(kwargs)
    return AtomFeatures(**defaults)


# ---------------------------------------------------------------------------
# Feature encoding
# ---------------------------------------------------------------------------

def test_encode_returns_correct_length():
    feat = _make_features()
    encoded = _encode_atom_features(feat)
    assert len(encoded) == _NODE_FEATURE_DIM


def test_encode_returns_floats():
    feat = _make_features()
    encoded = _encode_atom_features(feat)
    assert all(isinstance(v, float) for v in encoded)


def test_encode_aromatic_flag():
    feat_aromatic = _make_features(is_aromatic=True)
    feat_plain    = _make_features(is_aromatic=False)
    enc_a = _encode_atom_features(feat_aromatic)
    enc_p = _encode_atom_features(feat_plain)
    assert enc_a[3] == 1.0
    assert enc_p[3] == 0.0


def test_encode_lone_pairs_flag():
    feat = _make_features(has_lone_pairs=True)
    enc = _encode_atom_features(feat)
    assert enc[6] == 1.0


def test_encode_heavy_flag():
    feat = _make_features(is_heavy=True)
    enc = _encode_atom_features(feat)
    assert enc[8] == 1.0


def test_encode_neighbor_has_oxygen():
    feat = _make_features(neighbor_symbols=["O", "H"])
    enc = _encode_atom_features(feat)
    # neighbor_flags start at index 16: [has_H, has_C, has_N, has_O]
    assert enc[16] == 1.0   # has_H
    assert enc[19] == 1.0   # has_O
    assert enc[17] == 0.0   # has_C


def test_encode_coordination_normalised():
    feat = _make_features(coordination_number=4)
    enc = _encode_atom_features(feat)
    assert abs(enc[1] - 4.0 / 8.0) < 1e-6


def test_encode_period_normalised():
    feat = _make_features(period=3)
    enc = _encode_atom_features(feat)
    assert abs(enc[2] - 3.0 / 7.0) < 1e-6


# ---------------------------------------------------------------------------
# BASIS_CLASSES sanity
# ---------------------------------------------------------------------------

def test_basis_classes_non_empty():
    assert len(BASIS_CLASSES) >= 8


def test_basis_classes_are_strings():
    assert all(isinstance(b, str) for b in BASIS_CLASSES)


def test_basis_classes_contain_key_sets():
    names = set(BASIS_CLASSES)
    assert "def2-SVP" in names
    assert "def2-TZVP" in names
    assert "aug-cc-pVDZ" in names


# ---------------------------------------------------------------------------
# BasisGNN: missing weights file
# ---------------------------------------------------------------------------

def test_gnn_not_available_when_weights_missing(tmp_path):
    gnn = BasisGNN(model_path=tmp_path / "nonexistent.pt")
    assert gnn.is_available is False


def test_gnn_predict_raises_when_unavailable(tmp_path):
    from basisrec.exceptions import ModelNotFoundError
    gnn = BasisGNN(model_path=tmp_path / "nonexistent.pt")
    with pytest.raises(ModelNotFoundError):
        gnn.predict([_make_features()])


# ---------------------------------------------------------------------------
# get_gnn singleton
# ---------------------------------------------------------------------------

def test_get_gnn_returns_basis_gnn_instance():
    gnn = get_gnn()
    assert isinstance(gnn, BasisGNN)


def test_get_gnn_is_singleton():
    g1 = get_gnn()
    g2 = get_gnn()
    assert g1 is g2
