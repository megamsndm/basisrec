"""Shared data models for BasisRec.

All inter-module data transfer uses these typed models,
ensuring that every layer of the pipeline is type-safe
and self-documenting.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class Hybridization(str, Enum):
    S = "S"
    SP = "SP"
    SP2 = "SP2"
    SP3 = "SP3"
    SP3D = "SP3D"
    SP3D2 = "SP3D2"
    OTHER = "OTHER"
    UNKNOWN = "UNKNOWN"


class BasisTier(str, Enum):
    MINIMAL = "minimal"
    DOUBLE_ZETA = "double-zeta"
    TRIPLE_ZETA = "triple-zeta"
    AUGMENTED = "augmented"
    ECP = "ecp"


class Atom(BaseModel):
    """A single atom parsed from an XYZ file."""

    index: int = Field(..., description="0-based atom index in the molecule")
    symbol: str = Field(..., description="Element symbol, e.g. 'O'")
    x: float
    y: float
    z: float

    @field_validator("symbol")
    @classmethod
    def symbol_must_be_valid(cls, v: str) -> str:
        return v.strip().capitalize()


class AtomFeatures(BaseModel):
    """Chemical environment features computed for a single atom."""

    index: int
    symbol: str
    hybridization: Hybridization
    coordination_number: int
    is_aromatic: bool
    is_hbond_donor: bool
    is_hbond_acceptor: bool
    has_lone_pairs: bool
    is_metal: bool
    is_heavy: bool = Field(
        ..., description="True if atomic number > 36 (requires ECP)"
    )
    partial_charge: float
    period: int = Field(..., description="Row in periodic table")
    neighbor_symbols: List[str]


class BasisRecommendation(BaseModel):
    """Basis set recommendation for a single atom."""

    atom_index: int
    symbol: str
    basis_name: str
    tier: BasisTier
    confidence: float = Field(..., ge=0.0, le=1.0)
    reason: str
    source: str = Field(
        ..., description="'rule' if from rule engine, 'gnn' if from ML model"
    )
    bse_family: Optional[str] = None


class Molecule(BaseModel):
    """A parsed molecule with all atoms."""

    title: str = ""
    atoms: List[Atom]
    charge: int = 0
    multiplicity: int = 1

    @property
    def n_atoms(self) -> int:
        return len(self.atoms)

    @property
    def unique_elements(self) -> List[str]:
        return sorted(set(a.symbol for a in self.atoms))


class RecommendationResult(BaseModel):
    """Full output of the recommendation pipeline for one molecule."""

    molecule: Molecule
    recommendations: List[BasisRecommendation]
    orca_input: str
    report: str
    warnings: List[str] = Field(default_factory=list)
    method_summary: str = ""

    def basis_per_atom(self) -> Dict[int, str]:
        return {r.atom_index: r.basis_name for r in self.recommendations}

    def unique_basis_sets(self) -> List[str]:
        return sorted(set(r.basis_name for r in self.recommendations))
