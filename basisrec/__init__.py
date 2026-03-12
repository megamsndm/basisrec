"""BasisRec — automatic per-atom basis set recommendation.

Upload an XYZ file, receive a complete ORCA input file with
per-atom basis sets selected from chemistry and physics.

Quick start::

    from basisrec import recommend

    result = recommend("water.xyz")
    print(result.orca_input)

    # Save directly
    result = recommend("molecule.xyz", output_path="molecule.inp")
"""

from .api import recommend
from .exceptions import (
    BasisRecError,
    FeaturizationError,
    ModelNotFoundError,
    ORCAWriterError,
    ParseError,
    UnsupportedElementError,
)
from .method_recommender import MethodRecommendation, recommend_method
from .models import (
    Atom,
    AtomFeatures,
    BasisRecommendation,
    BasisTier,
    Molecule,
    RecommendationResult,
)

__version__ = "1.1.0"
__all__ = [
    "recommend",
    "recommend_method",
    "BasisRecError",
    "ParseError",
    "FeaturizationError",
    "UnsupportedElementError",
    "ModelNotFoundError",
    "ORCAWriterError",
    "Atom",
    "AtomFeatures",
    "BasisRecommendation",
    "BasisTier",
    "MethodRecommendation",
    "Molecule",
    "RecommendationResult",
]
