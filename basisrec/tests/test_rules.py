"""Unit tests for rule_engine.py.

Each test targets a specific chemical rule, verifying that the
correct basis set is selected and the reasoning is appropriate.
"""

import pytest
from basisrec.rule_engine import apply_rules, is_hard_rule
from basisrec.models import AtomFeatures, Hybridization, BasisTier


def _make_features(**kwargs) -> AtomFeatures:
    """Build a minimal AtomFeatures with sensible defaults."""
    defaults = dict(
        index=0,
        symbol="C",
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


# --- Hard ECP rules ---

def test_heavy_atom_gets_ecp():
    """Z > 36 must always get an ECP basis."""
    feat = _make_features(symbol="Br", is_heavy=True, period=4)
    rec = apply_rules(feat)
    assert "def2" in rec.basis_name.lower() or "lanl" in rec.basis_name.lower()
    assert rec.confidence == 1.0
    assert is_hard_rule(feat) is True


def test_transition_metal_gets_ecp():
    feat = _make_features(symbol="Fe", is_metal=True, is_heavy=False, period=4)
    rec = apply_rules(feat)
    assert "def2" in rec.basis_name.lower() or "lanl" in rec.basis_name.lower()
    assert is_hard_rule(feat) is True


def test_hard_rule_cannot_be_overridden():
    """is_hard_rule must return True for metals and heavy atoms."""
    for symbol, heavy, metal in [("Au", True, True), ("U", True, False), ("Fe", False, True)]:
        feat = _make_features(symbol=symbol, is_heavy=heavy, is_metal=metal, period=5)
        assert is_hard_rule(feat) is True


# --- Diffuse function rules ---

def test_oxygen_with_lone_pairs_gets_diffuse():
    feat = _make_features(
        symbol="O",
        has_lone_pairs=True,
        is_hbond_acceptor=True,
        hybridization=Hybridization.SP3,
        period=2,
    )
    rec = apply_rules(feat)
    assert "aug" in rec.basis_name.lower()
    assert is_hard_rule(feat) is False


def test_nitrogen_lone_pairs_gets_diffuse():
    feat = _make_features(
        symbol="N",
        has_lone_pairs=True,
        is_hbond_acceptor=True,
        hybridization=Hybridization.SP3,
        period=2,
    )
    rec = apply_rules(feat)
    assert "aug" in rec.basis_name.lower()


# --- Aromatic rules ---

def test_aromatic_carbon_gets_larger_basis():
    feat = _make_features(
        symbol="C",
        is_aromatic=True,
        hybridization=Hybridization.SP2,
        period=2,
    )
    rec = apply_rules(feat)
    # Should be at least double-zeta (not minimal)
    assert rec.tier != BasisTier.MINIMAL


# --- sp2 carbon ---

def test_sp2_carbon_gets_polarization():
    feat = _make_features(
        symbol="C",
        hybridization=Hybridization.SP2,
        is_aromatic=False,
        period=2,
    )
    rec = apply_rules(feat)
    assert rec.tier in (BasisTier.DOUBLE_ZETA, BasisTier.TRIPLE_ZETA, BasisTier.AUGMENTED)


# --- Hydrogen rules ---

def test_hbond_hydrogen_gets_polarization():
    feat = _make_features(
        symbol="H",
        neighbor_symbols=["O"],
        is_hbond_donor=False,
        period=1,
    )
    rec = apply_rules(feat)
    assert "**" in rec.basis_name or "aug" in rec.basis_name.lower()


def test_plain_ch_hydrogen_gets_small_basis():
    feat = _make_features(
        symbol="H",
        neighbor_symbols=["C"],
        period=1,
    )
    rec = apply_rules(feat)
    # Should be small — 6-31G* or similar, not aug-cc-pVTZ
    assert rec.tier in (BasisTier.MINIMAL, BasisTier.DOUBLE_ZETA)


# --- Confidence ---

def test_confidence_is_normalised():
    for symbol in ["C", "N", "O", "H", "Fe", "Br"]:
        feat = _make_features(symbol=symbol, is_heavy=symbol in ("Br",),
                               is_metal=symbol in ("Fe",), period=2)
        rec = apply_rules(feat)
        assert 0.0 <= rec.confidence <= 1.0


# --- Reason text ---

def test_reason_contains_element():
    feat = _make_features(symbol="S", has_lone_pairs=True, period=3)
    rec = apply_rules(feat)
    assert "S" in rec.reason


def test_source_is_rule():
    feat = _make_features()
    rec = apply_rules(feat)
    assert rec.source == "rule"
