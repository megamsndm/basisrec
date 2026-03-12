"""Physics-aware rule engine for basis set selection.

Each rule encodes a concrete piece of chemical knowledge about
when a specific type of basis function is necessary. Rules are
evaluated in priority order — higher priority rules override lower
ones. Hard rules (e.g. ECP for heavy atoms) are never overridden
by the GNN.

Rule priority (highest first):
    1. HARD_ECP      — Z > 36 → ECP mandatory, no exceptions
    2. HARD_METAL    — transition metal → relativistic ECP
    3. HARD_PERIOD4  — period 4+ → scalar relativistic effects
    4. DIFFUSE       — lone pairs / anion / H-bond acceptor → aug-
    5. POLARIZATION  — sp2/sp3 carbon, H-bond donor H
    6. AROMATIC      — aromatic ring → larger valence
    7. DEFAULT       — everything else → def2-SVP
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

from .basis_library import get_basis_for_element, get_tier
from .models import AtomFeatures, BasisRecommendation, Hybridization

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Rule:
    """A single basis set selection rule."""

    name: str
    priority: int          # lower number = higher priority (evaluated first)
    is_hard: bool          # True → GNN cannot override this rule
    basis_name: str
    reason_template: str   # '{symbol}' placeholder filled at evaluation time


# ---------------------------------------------------------------------------
# Rule registry
# ---------------------------------------------------------------------------

_RULES: List[Rule] = [
    # --- Hard rules (always win) ---
    Rule(
        name="HARD_ECP",
        priority=10,
        is_hard=True,
        basis_name="def2-TZVP",
        reason_template=(
            "{symbol}: Z > 36 — core electrons must be replaced by ECP. "
            "def2-TZVP includes the appropriate Stuttgart ECP for this element."
        ),
    ),
    Rule(
        name="HARD_METAL_LANTHA",
        priority=12,
        is_hard=True,
        basis_name="def2-TZVP",
        reason_template=(
            "{symbol}: lanthanide/actinide — relativistic ECP and large valence "
            "space required. def2-TZVP (ECP variant) selected."
        ),
    ),
    Rule(
        name="HARD_TRANSITION_METAL",
        priority=15,
        is_hard=True,
        basis_name="def2-TZVP",
        reason_template=(
            "{symbol}: transition metal — d-orbital polarization and relativistic "
            "effects require def2-TZVP as minimum viable basis."
        ),
    ),
    # --- Soft rules (GNN can upgrade but not downgrade) ---
    Rule(
        name="DIFFUSE_ANION",
        priority=20,
        is_hard=False,
        basis_name="aug-cc-pVDZ",
        reason_template=(
            "{symbol}: lone pairs present and acts as H-bond acceptor — "
            "diffuse functions required for accurate electron density tail. "
            "aug-cc-pVDZ selected."
        ),
    ),
    Rule(
        name="DIFFUSE_LONE_PAIRS",
        priority=25,
        is_hard=False,
        basis_name="aug-cc-pVDZ",
        reason_template=(
            "{symbol}: lone pairs detected — diffuse functions (aug-) "
            "needed to describe non-bonding electron density. aug-cc-pVDZ selected."
        ),
    ),
    Rule(
        name="CARBONYL_OXYGEN",
        priority=28,
        is_hard=False,
        basis_name="aug-cc-pVDZ",
        reason_template=(
            "{symbol}: carbonyl or sp2 oxygen — partial negative charge "
            "and lone pairs require diffuse functions. aug-cc-pVDZ selected."
        ),
    ),
    Rule(
        name="AROMATIC",
        priority=30,
        is_hard=False,
        basis_name="cc-pVDZ",
        reason_template=(
            "{symbol}: aromatic ring — delocalized pi system requires "
            "larger valence space than minimal basis. cc-pVDZ selected."
        ),
    ),
    Rule(
        name="SP2_CARBON",
        priority=35,
        is_hard=False,
        basis_name="cc-pVDZ",
        reason_template=(
            "{symbol}: sp2 hybridization — pi bonding requires p-polarization "
            "functions. cc-pVDZ selected."
        ),
    ),
    Rule(
        name="HBOND_DONOR_O_N",
        priority=40,
        is_hard=False,
        basis_name="aug-cc-pVDZ",
        reason_template=(
            "{symbol}: H-bond donor (O/N bonded to H) — polarization and "
            "slight diffuse character needed for H-bond geometry. aug-cc-pVDZ selected."
        ),
    ),
    Rule(
        name="HBOND_HYDROGEN",
        priority=45,
        is_hard=False,
        basis_name="6-31G**",
        reason_template=(
            "{symbol}: hydrogen involved in H-bonding — p-polarization "
            "on H required (6-31G** adds p functions to H). 6-31G** selected."
        ),
    ),
    Rule(
        name="LIGHT_HYDROGEN",
        priority=80,
        is_hard=False,
        basis_name="6-31G*",
        reason_template=(
            "{symbol}: light hydrogen bonded to C/N/O — 6-31G* provides "
            "adequate description at low computational cost."
        ),
    ),
    Rule(
        name="DEFAULT_ORGANIC",
        priority=90,
        is_hard=False,
        basis_name="def2-SVP",
        reason_template=(
            "{symbol}: general organic/main-group element — def2-SVP "
            "provides balanced accuracy and efficiency."
        ),
    ),
    Rule(
        name="FALLBACK",
        priority=99,
        is_hard=False,
        basis_name="def2-SVP",
        reason_template="{symbol}: no specific rule matched — def2-SVP selected as default.",
    ),
]

# Sorted once at module load time
_RULES_SORTED = sorted(_RULES, key=lambda r: r.priority)

# Lanthanides and actinides
_LANTHA_ACTINIDES: frozenset[str] = frozenset({
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
    "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Ac", "Th", "Pa", "U", "Np", "Pu", "Am",
})

_TRANSITION_METALS: frozenset[str] = frozenset({
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
})


def apply_rules(features: AtomFeatures) -> BasisRecommendation:
    """Apply the rule engine to a single atom's features.

    Evaluates rules in priority order and returns the first match.

    Parameters
    ----------
    features:
        Per-atom chemical features from the featurizer.

    Returns
    -------
    BasisRecommendation
        The recommended basis set with reasoning.
    """
    matching_rule = _find_matching_rule(features)

    # Verify BSE coverage and fall back if needed
    basis_name = get_basis_for_element(matching_rule.basis_name, features.symbol)

    return BasisRecommendation(
        atom_index=features.index,
        symbol=features.symbol,
        basis_name=basis_name,
        tier=get_tier(basis_name),
        confidence=_rule_confidence(matching_rule, features),
        reason=matching_rule.reason_template.format(symbol=features.symbol),
        source="rule",
    )


def is_hard_rule(features: AtomFeatures) -> bool:
    """Return True if the atom is governed by a hard rule that the GNN
    must not override."""
    rule = _find_matching_rule(features)
    return rule.is_hard


def _find_matching_rule(features: AtomFeatures) -> Rule:
    """Return the highest-priority matching rule for *features*."""
    sym = features.symbol

    for rule in _RULES_SORTED:
        if rule.name == "HARD_ECP" and features.is_heavy:
            return rule
        if rule.name == "HARD_METAL_LANTHA" and sym in _LANTHA_ACTINIDES:
            return rule
        if rule.name == "HARD_TRANSITION_METAL" and sym in _TRANSITION_METALS:
            return rule
        if rule.name == "DIFFUSE_ANION" and features.has_lone_pairs and features.is_hbond_acceptor:
            return rule
        if rule.name == "DIFFUSE_LONE_PAIRS" and features.has_lone_pairs:
            return rule
        if rule.name == "CARBONYL_OXYGEN" and sym == "O" and features.hybridization == Hybridization.SP2:
            return rule
        if rule.name == "AROMATIC" and features.is_aromatic:
            return rule
        if rule.name == "SP2_CARBON" and sym == "C" and features.hybridization == Hybridization.SP2:
            return rule
        if rule.name == "HBOND_DONOR_O_N" and features.is_hbond_donor and sym in {"O", "N"}:
            return rule
        if rule.name == "HBOND_HYDROGEN" and sym == "H" and "O" in features.neighbor_symbols:
            return rule
        if rule.name == "LIGHT_HYDROGEN" and sym == "H":
            return rule
        if rule.name == "DEFAULT_ORGANIC" and features.period <= 3:
            return rule
        if rule.name == "FALLBACK":
            return rule

    # Should never reach here, but be safe
    return _RULES_SORTED[-1]


def _rule_confidence(rule: Rule, features: AtomFeatures) -> float:
    """Assign a confidence score to a rule-based recommendation.

    Hard rules get 1.0 by definition. Soft rules get confidence
    based on how many supporting features the atom has.
    """
    if rule.is_hard:
        return 1.0

    supporting = 0
    total = 4

    if features.has_lone_pairs:
        supporting += 1
    if features.is_hbond_donor or features.is_hbond_acceptor:
        supporting += 1
    if features.is_aromatic:
        supporting += 1
    if features.hybridization not in (Hybridization.UNKNOWN, Hybridization.S):
        supporting += 1

    base = 0.65
    return round(min(1.0, base + (supporting / total) * 0.30), 2)
