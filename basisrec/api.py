"""Public Python API for BasisRec."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

from .atom_featurizer import featurize_molecule
from .decision_engine import recommend as _engine_recommend
from .method_recommender import MethodRecommendation, recommend_method
from .models import RecommendationResult
from .orca_writer import DEFAULT_NPROCS, write_orca_input
from .report_generator import generate_report
from .xyz_parser import parse_xyz

logger = logging.getLogger(__name__)


def recommend(
    xyz_path: Union[str, Path],
    accuracy: str = "balanced",
    excited_states: bool = False,
    dispersion_dominated: bool = False,
    solvent: str = "",
    nprocs: int = DEFAULT_NPROCS,
    memory_gb: int = 8,
    charge: int = 0,
    multiplicity: int = 1,
    use_gnn: bool = True,
    model_path: Optional[Union[str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
    method_line: Optional[str] = None,
) -> RecommendationResult:
    """Recommend per-atom basis sets AND optimal DFT method for *xyz_path*.

    Parameters
    ----------
    xyz_path : path to the .xyz file
    accuracy : 'fast' | 'balanced' | 'accurate' | 'benchmark'
    excited_states : True → CAM-B3LYP for TD-DFT
    dispersion_dominated : True → r2SCAN-D4 or PBE-D4
    solvent : 'water', 'dmso', etc. — empty = gas phase
    nprocs : CPU cores available
    memory_gb : total RAM in GB
    charge : molecular charge
    multiplicity : spin multiplicity
    use_gnn : use GNN in basis selection
    model_path : custom GNN weights path
    output_path : write ORCA input to this file
    method_line : manual override for the ORCA ! line
    """
    xyz_path = Path(xyz_path)
    if model_path is not None:
        model_path = Path(model_path)

    molecule = parse_xyz(xyz_path)
    molecule = molecule.model_copy(update={"charge": charge, "multiplicity": multiplicity})

    features = featurize_molecule(molecule)

    recs = _engine_recommend(molecule, features=features, model_path=model_path, use_gnn=use_gnn)

    method_rec: MethodRecommendation = recommend_method(
        molecule=molecule,
        features=features,
        multiplicity=multiplicity,
        charge=charge,
        accuracy=accuracy,
        solvent=solvent,
        nprocs=nprocs,
        memory_gb=memory_gb,
        excited_states=excited_states,
        dispersion_dominated=dispersion_dominated,
    )

    final_method_line = method_line if method_line else method_rec.keywords_line()

    warnings = []
    low_conf = [r for r in recs if r.confidence < 0.65]
    if low_conf:
        syms = ", ".join(f"{r.symbol}({r.atom_index})" for r in low_conf)
        warnings.append(
            f"Low confidence (<65%) for atoms: {syms}. "
            "Consider reviewing these basis set recommendations manually."
        )
    if accuracy == "benchmark":
        warnings.append(
            "DLPNO-CCSD(T) is very expensive. "
            "Run a geometry optimisation at a lower level first."
        )

    orca_str = write_orca_input(
        molecule=molecule,
        recommendations=recs,
        method_line=final_method_line,
        maxcore_mb=method_rec.memory_per_core_mb,
        nprocs=method_rec.nprocs,
        method_rec=method_rec,
        output_path=str(output_path) if output_path else None,
    )

    report_str = generate_report(molecule, recs)
    method_summary = method_rec.summary()

    return RecommendationResult(
        molecule=molecule,
        recommendations=recs,
        orca_input=orca_str,
        report=report_str,
        method_summary=method_summary,
        warnings=warnings,
    )
