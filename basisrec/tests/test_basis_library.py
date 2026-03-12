"""Unit tests for basis_library.py.

Tests cover BSE coverage lookups, fallback chains, tier metadata,
and the fetch functions. BSE network calls are NOT mocked here —
these are real integration tests against the local BSE database,
which is bundled with the basis-set-exchange package and requires
no internet connection.
"""

import pytest

from basisrec.basis_library import (
    _TIER_ORDER_MAP,
    BASIS_TIERS,
    basis_covers_element,
    fetch_basis_block_orca,
    get_basis_for_element,
    get_suitability_tags,
    get_tier,
)
from basisrec.exceptions import UnsupportedElementError
from basisrec.models import BasisTier

# ---------------------------------------------------------------------------
# BSE coverage
# ---------------------------------------------------------------------------

def test_def2svp_covers_carbon():
    assert basis_covers_element("def2-SVP", "C") is True


def test_def2svp_covers_oxygen():
    assert basis_covers_element("def2-SVP", "O") is True


def test_def2svp_covers_iron():
    assert basis_covers_element("def2-SVP", "Fe") is True


def test_def2tzvp_covers_bromine():
    assert basis_covers_element("def2-TZVP", "Br") is True


def test_augccpvdz_covers_nitrogen():
    assert basis_covers_element("aug-cc-pVDZ", "N") is True


def test_nonexistent_basis_returns_false():
    assert basis_covers_element("totally-made-up-basis-xyz", "C") is False


def test_coverage_result_is_bool():
    result = basis_covers_element("6-31G*", "H")
    assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Fallback chain
# ---------------------------------------------------------------------------

def test_get_basis_for_element_returns_same_when_coverage_exists():
    result = get_basis_for_element("def2-SVP", "C")
    assert result == "def2-SVP"


def test_get_basis_for_element_falls_back_when_needed():
    """cc-pVDZ does not cover Fe — should fall back to def2-SVP or LANL2DZ."""
    result = get_basis_for_element("cc-pVDZ", "Fe")
    # Should not raise, and result should cover Fe
    assert basis_covers_element(result, "Fe") is True


def test_unsupported_element_raises():
    """If nothing in the fallback chain covers an element, raise."""
    with pytest.raises(UnsupportedElementError):
        # Use a real element but a basis with empty fallback chain
        # that definitely can't cover it — simulate by patching
        from unittest.mock import patch
        with patch("basisrec.basis_library.basis_covers_element", return_value=False):
            get_basis_for_element("STO-3G", "C")


# ---------------------------------------------------------------------------
# Tier metadata
# ---------------------------------------------------------------------------

def test_tier_augmented_for_aug_cc_pvdz():
    assert get_tier("aug-cc-pVDZ") == BasisTier.AUGMENTED


def test_tier_triple_zeta_for_def2tzvp():
    assert get_tier("def2-TZVP") == BasisTier.TRIPLE_ZETA


def test_tier_double_zeta_for_def2svp():
    assert get_tier("def2-SVP") == BasisTier.DOUBLE_ZETA


def test_tier_minimal_for_sto3g():
    assert get_tier("STO-3G") == BasisTier.MINIMAL


def test_tier_unknown_defaults_to_double_zeta():
    assert get_tier("completely-unknown-basis") == BasisTier.DOUBLE_ZETA


def test_all_tiers_have_at_least_one_basis():
    for tier, names in BASIS_TIERS.items():
        assert len(names) >= 1, f"Tier {tier} is empty"


# ---------------------------------------------------------------------------
# Suitability tags
# ---------------------------------------------------------------------------

def test_def2tzvp_has_transition_metal_tag():
    tags = get_suitability_tags("def2-TZVP")
    assert any("metal" in t.lower() or "tm" in t.lower() for t in tags)


def test_augccpvdz_has_diffuse_tag():
    tags = get_suitability_tags("aug-cc-pVDZ")
    assert any("diffuse" in t.lower() or "anion" in t.lower() for t in tags)


def test_unknown_basis_returns_empty_list():
    tags = get_suitability_tags("nonexistent-basis-abc")
    assert tags == []


# ---------------------------------------------------------------------------
# Tier order map
# ---------------------------------------------------------------------------

def test_augmented_outranks_triple_zeta():
    assert _TIER_ORDER_MAP[BasisTier.AUGMENTED] > _TIER_ORDER_MAP[BasisTier.TRIPLE_ZETA]


def test_triple_zeta_outranks_double_zeta():
    assert _TIER_ORDER_MAP[BasisTier.TRIPLE_ZETA] > _TIER_ORDER_MAP[BasisTier.DOUBLE_ZETA]


def test_double_zeta_outranks_minimal():
    assert _TIER_ORDER_MAP[BasisTier.DOUBLE_ZETA] > _TIER_ORDER_MAP[BasisTier.MINIMAL]


# ---------------------------------------------------------------------------
# fetch_basis_block_orca
# ---------------------------------------------------------------------------

def test_fetch_returns_string():
    block = fetch_basis_block_orca("def2-SVP", "C")
    assert isinstance(block, str)
    assert len(block) > 50


def test_fetch_contains_s_or_p_shell():
    block = fetch_basis_block_orca("def2-SVP", "C")
    # ORCA format uses S, P, D shell labels
    assert any(shell in block for shell in ["S ", "P ", "D ", "s ", "p "])


def test_fetch_aug_cc_pvdz_oxygen():
    block = fetch_basis_block_orca("aug-cc-pVDZ", "O")
    assert len(block) > 100
