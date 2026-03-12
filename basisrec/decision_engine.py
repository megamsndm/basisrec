"""Decision engine — fuses the rule engine and GNN into final recommendations.

Fusion logic:
    1. Rule engine runs first for every atom.
    2. If the rule is HARD (ECP, transition metal), the rule wins unconditionally.
    3. If GNN is available and not a hard rule:
        - If GNN confidence > GNN_OVERRIDE_THRESHOLD and recommends a
          higher-quality basis tier, GNN recommendation is used.
        - Otherwise, rule recommendation is kept.
    4. Final basis is verified against BSE coverage; fallback applied if needed.
    5. A confidence score and source label are attached to every atom.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from .atom_featurizer import featurize_molecule
from .basis_library import get_basis_for_element, get_tier
from .exceptions import BasisRecError
from .gnn_model import BasisGNN, get_gnn
from .models import (
    AtomFeatures,
    BasisRecommendation,
    BasisTier,
    Molecule,
)
from .rule_engine import apply_rules, is_hard_rule

logger = logging.getLogger(__name__)

# GNN confidence threshold above which the GNN may override a soft rule.
GNN_OVERRIDE_THRESHOLD = 0.75

# Tier ordering for comparison (higher index = better quality)
_TIER_ORDER = {
    BasisTier.MINIMAL: 0,
    BasisTier.DOUBLE_ZETA: 1,
    BasisTier.TRIPLE_ZETA: 2,
    BasisTier.AUGMENTED: 3,
    BasisTier.ECP: 2,  # ECP is mandatory not upgradeable via quality
}


def recommend(
    molecule: Molecule,
    features: Optional[List[AtomFeatures]] = None,
    model_path: Optional[Path] = None,
    use_gnn: bool = True,
) -> List[BasisRecommendation]:
    """Run the full recommendation pipeline for *molecule*.

    Parameters
    ----------
    molecule:
        Parsed molecule.
    features:
        Pre-computed atom features (optional — computed if not provided).
    model_path:
        Path to GNN weights file. Uses default if None.
    use_gnn:
        Set to False to disable the GNN entirely and use only the rule engine.
        Useful for fast/offline use.

    Returns
    -------
    list[BasisRecommendation]
        One recommendation per atom, in atom index order.
    """
    # Step 1: featurize if not provided
    if features is None:
        features = featurize_molecule(molecule)

    # Step 2: rule engine — one recommendation per atom
    rule_recs = [apply_rules(feat) for feat in features]

    # Step 3: GNN predictions (if available and requested)
    gnn_recs: Optional[List[BasisRecommendation]] = None
    if use_gnn:
        gnn = get_gnn(model_path)
        if gnn.is_available:
            try:
                gnn_recs = gnn.predict(features)
                logger.debug("GNN predictions obtained for %d atoms.", len(gnn_recs))
            except Exception as exc:
                logger.warning("GNN inference failed (%s) — using rule engine only.", exc)

    # Step 4: fusion
    final_recs = []
    for i, (feat, rule_rec) in enumerate(zip(features, rule_recs)):
        if is_hard_rule(feat):
            # Hard rule always wins
            final_recs.append(rule_rec)
            logger.debug("Atom %d (%s): hard rule → %s", i, feat.symbol, rule_rec.basis_name)
            continue

        if gnn_recs is not None:
            gnn_rec = gnn_recs[i]
            chosen = _fuse(rule_rec, gnn_rec, feat)
        else:
            chosen = rule_rec

        # Verify BSE coverage
        verified_basis = get_basis_for_element(chosen.basis_name, feat.symbol)
        if verified_basis != chosen.basis_name:
            chosen = chosen.model_copy(update={
                "basis_name": verified_basis,
                "tier": get_tier(verified_basis),
                "reason": chosen.reason + f" [BSE fallback: {verified_basis}]",
            })

        final_recs.append(chosen)

    return final_recs


def _fuse(
    rule_rec: BasisRecommendation,
    gnn_rec: BasisRecommendation,
    feat: AtomFeatures,
) -> BasisRecommendation:
    """Decide between rule and GNN recommendation for a single soft-rule atom."""

    rule_tier_rank = _TIER_ORDER.get(rule_rec.tier, 1)
    gnn_tier_rank = _TIER_ORDER.get(gnn_rec.tier, 1)

    # GNN overrides only if: high confidence AND recommends equal/better tier
    if (
        gnn_rec.confidence >= GNN_OVERRIDE_THRESHOLD
        and gnn_tier_rank >= rule_tier_rank
    ):
        logger.debug(
            "Atom %d (%s): GNN overrides rule — %s → %s (conf %.2f)",
            feat.index, feat.symbol,
            rule_rec.basis_name, gnn_rec.basis_name, gnn_rec.confidence,
        )
        return gnn_rec.model_copy(update={
            "reason": (
                f"{gnn_rec.reason} "
                f"[GNN override of rule '{rule_rec.basis_name}']"
            )
        })

    # Rule wins
    logger.debug(
        "Atom %d (%s): rule retained — %s (GNN: %s conf %.2f)",
        feat.index, feat.symbol,
        rule_rec.basis_name, gnn_rec.basis_name, gnn_rec.confidence,
    )
    return rule_rec
