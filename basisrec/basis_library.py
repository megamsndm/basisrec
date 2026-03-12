"""Basis set tier metadata and BSE vocabulary.

This module does NOT store Gaussian exponents or contraction
coefficients — those are always fetched live from the Basis Set
Exchange (BSE) Python API.

What this module provides:
    - A tiered vocabulary of basis set names the decision engine uses.
    - Suitability tags per basis set (when to prefer it).
    - A BSE coverage checker so the decision engine never recommends
      a basis set that doesn't cover the target element.
    - A fallback chain: if the preferred basis set doesn't cover an
      element, walk down the fallback list until one does.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Dict, List

import basis_set_exchange as bse

from .exceptions import UnsupportedElementError
from .models import BasisTier

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tiered vocabulary
# ---------------------------------------------------------------------------

BASIS_TIERS: Dict[BasisTier, List[str]] = {
    BasisTier.MINIMAL: [
        "STO-3G",
        "STO-6G",
        "3-21G",
    ],
    BasisTier.DOUBLE_ZETA: [
        "6-31G*",
        "6-31G**",
        "def2-SVP",
        "cc-pVDZ",
        "pc-1",
    ],
    BasisTier.TRIPLE_ZETA: [
        "def2-TZVP",
        "def2-TZVPP",
        "cc-pVTZ",
        "6-311G**",
        "pc-2",
    ],
    BasisTier.AUGMENTED: [
        "aug-cc-pVDZ",
        "aug-cc-pVTZ",
        "def2-TZVPD",
        "def2-SVPD",
        "aug-pc-1",
    ],
    BasisTier.ECP: [
        "def2-TZVP",       # includes ECP for Z > 36 in def2 family
        "def2-SVP",
        "LANL2DZ",
        "Stuttgart RSC 1997",
        "cc-pVDZ-PP",
        "cc-pVTZ-PP",
        "aug-cc-pVDZ-PP",
    ],
}

# Suitability tags: which chemical situations each basis is best for.
# Used by the rule engine and report generator for human-readable reasoning.
BASIS_SUITABILITY: Dict[str, List[str]] = {
    "STO-3G":             ["minimal", "exploratory", "large systems"],
    "3-21G":              ["minimal", "qualitative geometry"],
    "6-31G*":             ["organic", "ground state", "geometry optimisation"],
    "6-31G**":            ["organic", "hydrogen explicit", "geometry optimisation"],
    "def2-SVP":           ["general purpose", "fast", "transition metals"],
    "def2-TZVP":          ["accurate", "transition metals", "properties", "ECP Z>36"],
    "def2-TZVPP":         ["high accuracy", "polarizable H", "NMR"],
    "def2-TZVPD":         ["anions", "excited states", "diffuse needed"],
    "def2-SVPD":          ["anions fast", "diffuse needed"],
    "cc-pVDZ":            ["correlated methods", "benchmark", "sp atoms"],
    "cc-pVTZ":            ["high accuracy correlated", "benchmark"],
    "aug-cc-pVDZ":        ["anions", "diffuse required", "H-bond", "lone pairs"],
    "aug-cc-pVTZ":        ["high accuracy anions", "polarizability", "NMR"],
    "LANL2DZ":            ["heavy metals", "ECP", "qualitative TM"],
    "Stuttgart RSC 1997": ["heavy metals", "ECP", "accurate TM"],
    "cc-pVDZ-PP":         ["post-Ar correlated", "ECP"],
    "cc-pVTZ-PP":         ["post-Ar high accuracy", "ECP"],
    "aug-cc-pVDZ-PP":     ["post-Ar diffuse", "ECP", "anions"],
}

# Fallback chains: if preferred basis doesn't cover element, try these in order.
FALLBACK_CHAIN: Dict[str, List[str]] = {
    "aug-cc-pVDZ":    ["cc-pVDZ",    "def2-SVP",   "STO-3G"],
    "aug-cc-pVTZ":    ["cc-pVTZ",    "def2-TZVP",  "def2-SVP"],
    "cc-pVDZ":        ["def2-SVP",   "6-31G*",     "STO-3G"],
    "cc-pVTZ":        ["def2-TZVP",  "def2-SVP",   "6-31G*"],
    "def2-TZVP":      ["def2-SVP",   "LANL2DZ",    "STO-3G"],
    "def2-SVP":       ["LANL2DZ",    "STO-3G"],
    "6-31G*":         ["def2-SVP",   "STO-3G"],
    "LANL2DZ":        ["Stuttgart RSC 1997", "STO-3G"],
}


# ---------------------------------------------------------------------------
# BSE coverage helpers
# ---------------------------------------------------------------------------

@lru_cache(maxsize=512)
def basis_covers_element(basis_name: str, element_symbol: str) -> bool:
    """Return True if *basis_name* covers *element_symbol* in BSE.

    Results are cached so repeated lookups (e.g. same basis for many
    atoms of the same element) hit memory rather than BSE.
    """
    try:
        result = bse.get_basis(basis_name, elements=[element_symbol])
        return element_symbol.title() in [
            bse.lut.element_sym_from_Z(int(z), normalize=True)
            for z in result.get("elements", {}).keys()
        ]
    except Exception as exc:
        logger.debug("BSE coverage check failed for %s/%s: %s", basis_name, element_symbol, exc)
        return False


def get_basis_for_element(basis_name: str, element_symbol: str) -> str:
    """Return the best available basis for *element_symbol*, starting
    from *basis_name* and walking the fallback chain if needed.

    Raises
    ------
    UnsupportedElementError
        If no basis in the fallback chain covers the element.
    """
    candidates = [basis_name] + FALLBACK_CHAIN.get(basis_name, [])
    for candidate in candidates:
        if basis_covers_element(candidate, element_symbol):
            if candidate != basis_name:
                logger.warning(
                    "Basis '%s' does not cover element '%s' — "
                    "falling back to '%s'.",
                    basis_name, element_symbol, candidate,
                )
            return candidate
    raise UnsupportedElementError(element_symbol, basis_name)


def fetch_basis_block_orca(basis_name: str, element_symbol: str) -> str:
    """Fetch the full basis set definition for *element_symbol* in ORCA format.

    Uses the BSE Python API. The returned string contains the raw
    Gaussian exponents and contraction coefficients, ready to be
    embedded in an ORCA ``%basis ... end`` block.

    Parameters
    ----------
    basis_name:
        BSE basis set name, e.g. ``'def2-TZVP'``.
    element_symbol:
        Element symbol, e.g. ``'Fe'``.

    Returns
    -------
    str
        ORCA-format basis block string.
    """
    actual_basis = get_basis_for_element(basis_name, element_symbol)
    basis_str = bse.get_basis(
        actual_basis,
        elements=[element_symbol],
        fmt="orca",
        header=False,
    )
    return basis_str.strip()


def get_tier(basis_name: str) -> BasisTier:
    """Return the :class:`~basisrec.models.BasisTier` for *basis_name*."""
    for tier, names in BASIS_TIERS.items():
        if basis_name in names:
            return tier
    return BasisTier.DOUBLE_ZETA  # sensible default


def get_suitability_tags(basis_name: str) -> List[str]:
    """Return the suitability tag list for *basis_name*."""
    return BASIS_SUITABILITY.get(basis_name, [])


# Numeric tier rank used by orca_writer and decision_engine for comparison
_TIER_ORDER_MAP: dict = {
    BasisTier.MINIMAL: 0,
    BasisTier.DOUBLE_ZETA: 1,
    BasisTier.TRIPLE_ZETA: 2,
    BasisTier.AUGMENTED: 3,
    BasisTier.ECP: 2,
}
