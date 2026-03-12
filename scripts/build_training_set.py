"""Build the GNN training set from QCArchive.

Downloads small-molecule geometries and energies from QCArchive,
runs calculations at reference (cc-pVTZ) and candidate basis sets,
and labels each atom with the minimum basis set that recovers 99%
of the cc-pVTZ energy contribution for that chemical environment.

Output: data/training_set.pkl
    A list of (feature_vector: List[float], label_index: int) tuples,
    one per atom across all molecules in the dataset.

Requirements:
    pip install qcportal pyscf basis-set-exchange

Usage:
    python scripts/build_training_set.py
    python scripts/build_training_set.py --n-molecules 500 --output data/my_set.pkl
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Candidate basis sets evaluated per atom.
# Label = index into this list for whichever is the minimum adequate basis.
CANDIDATE_BASIS = [
    "STO-3G",
    "6-31G*",
    "6-31G**",
    "def2-SVP",
    "def2-TZVP",
    "cc-pVDZ",
    "cc-pVTZ",
    "aug-cc-pVDZ",
    "aug-cc-pVTZ",
    "LANL2DZ",
]

REFERENCE_BASIS = "cc-pVTZ"
ACCURACY_THRESHOLD = 0.99  # Fraction of reference energy to consider "adequate"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build BasisRec GNN training set.")
    p.add_argument(
        "--n-molecules", type=int, default=200,
        help="Number of molecules to fetch from QCArchive (default: 200)",
    )
    p.add_argument(
        "--dataset", default="OpenFF Optimization Set 1",
        help="QCArchive dataset name",
    )
    p.add_argument(
        "--output", type=Path, default=Path("data/training_set.pkl"),
        help="Output path for the training set pickle",
    )
    p.add_argument(
        "--max-atoms", type=int, default=20,
        help="Skip molecules with more than this many atoms",
    )
    return p.parse_args()


def check_dependencies() -> None:
    missing = []
    for pkg in ("qcportal", "pyscf"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        logger.error("Missing required packages: %s", ", ".join(missing))
        logger.error("Install with: pip install %s", " ".join(missing))
        sys.exit(1)


def fetch_molecules_from_qcarchive(
    dataset_name: str,
    n: int,
    max_atoms: int,
) -> List[Dict]:
    """Fetch molecule geometries from QCArchive.

    Returns a list of dicts with keys: 'symbols', 'geometry' (bohr),
    'charge', 'multiplicity'.
    """
    import qcportal as ptl

    logger.info("Connecting to QCFractal public server…")
    client = ptl.FractalClient()

    logger.info("Fetching dataset '%s'…", dataset_name)
    try:
        ds = client.get_collection("OptimizationDataset", dataset_name)
    except Exception as exc:
        logger.error("Could not fetch dataset: %s", exc)
        logger.info("Falling back to QM9-style local molecules if available.")
        return []

    molecules = []
    for entry in ds.get_molecules().values():
        if len(entry.symbols) > max_atoms:
            continue
        molecules.append({
            "symbols":      list(entry.symbols),
            "geometry":     entry.geometry,   # in bohr
            "charge":       entry.molecular_charge,
            "multiplicity": entry.molecular_multiplicity,
        })
        if len(molecules) >= n:
            break

    logger.info("Fetched %d molecules.", len(molecules))
    return molecules


def run_pyscf_single_point(
    symbols: List[str],
    geometry_bohr,
    basis_name: str,
    charge: int,
    multiplicity: int,
) -> Optional[float]:
    """Run a PySCF HF single-point and return the total energy (Hartree).

    Returns None if the calculation fails.
    """
    import pyscf
    import numpy as np

    # Convert geometry to Angstrom for PySCF atom string
    bohr_to_ang = 0.529177
    geom_ang = geometry_bohr * bohr_to_ang if hasattr(geometry_bohr, "__mul__") else \
               [c * bohr_to_ang for c in geometry_bohr]

    atom_str = ""
    coords = list(geom_ang)
    for i, sym in enumerate(symbols):
        x, y, z = coords[i * 3], coords[i * 3 + 1], coords[i * 3 + 2]
        atom_str += f"{sym} {x:.6f} {y:.6f} {z:.6f}; "

    try:
        mol = pyscf.gto.M(
            atom=atom_str,
            basis=basis_name,
            charge=charge,
            spin=multiplicity - 1,
            verbose=0,
        )
        mf = pyscf.scf.RHF(mol) if multiplicity == 1 else pyscf.scf.ROHF(mol)
        mf.max_cycle = 200
        energy = mf.kernel()
        if not mf.converged:
            return None
        return float(energy)
    except Exception as exc:
        logger.debug("PySCF failed for %s with basis %s: %s", symbols, basis_name, exc)
        return None


def label_atoms(
    symbols: List[str],
    ref_energy: float,
    candidate_energies: Dict[str, Optional[float]],
) -> List[int]:
    """Assign a basis set label to each atom in the molecule.

    Strategy: for each candidate basis (in order from smallest to largest),
    check if its energy is within ACCURACY_THRESHOLD of the reference.
    Assign the label of the smallest adequate basis to all atoms.

    Per-atom labelling (advanced) would require energy decomposition —
    for the initial training set we use the whole-molecule criterion
    and assign the same label to all atoms. Future work can refine this
    with DFTB-based per-atom energy decomposition.
    """
    for i, basis in enumerate(CANDIDATE_BASIS):
        e = candidate_energies.get(basis)
        if e is None:
            continue
        ratio = abs(e / ref_energy) if ref_energy != 0 else 0.0
        if ratio >= ACCURACY_THRESHOLD:
            return [i] * len(symbols)

    # If nothing meets the threshold, use the best available
    best_idx = len(CANDIDATE_BASIS) - 1
    return [best_idx] * len(symbols)


def main() -> None:
    args = parse_args()
    check_dependencies()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # --- Fetch molecules ---
    molecules = fetch_molecules_from_qcarchive(
        args.dataset, args.n_molecules, args.max_atoms
    )

    if not molecules:
        logger.warning(
            "No molecules fetched. Using built-in example molecules for demonstration."
        )
        # Fall back to example XYZ files
        example_dir = Path(__file__).parent.parent / "data" / "example_molecules"
        from basisrec.xyz_parser import parse_xyz
        from basisrec.atom_featurizer import _featurize_geometry_fallback
        molecules = []
        for xyz_file in example_dir.glob("*.xyz"):
            mol = parse_xyz(xyz_file)
            molecules.append({
                "symbols":      [a.symbol for a in mol.atoms],
                "geometry":     [(a.x, a.y, a.z) for a in mol.atoms],
                "charge":       0,
                "multiplicity": 1,
                "_mol_obj":     mol,
            })

    logger.info("Processing %d molecules…", len(molecules))

    # --- Build dataset ---
    from basisrec.atom_featurizer import featurize_molecule, _featurize_geometry_fallback
    from basisrec.gnn_model import _encode_atom_features
    from basisrec.xyz_parser import parse_xyz_string

    dataset: List[Tuple[List[float], int]] = []
    skipped = 0

    for mol_idx, mol_data in enumerate(molecules):
        symbols = mol_data["symbols"]
        charge = mol_data.get("charge", 0)
        mult = mol_data.get("multiplicity", 1)
        geometry = mol_data.get("geometry", [])

        logger.info(
            "[%d/%d] %s (%d atoms)",
            mol_idx + 1, len(molecules),
            "".join(symbols[:6]), len(symbols),
        )

        # Build xyz string for featurizer
        if "_mol_obj" in mol_data:
            mol_obj = mol_data["_mol_obj"]
        else:
            import numpy as np
            geom = list(geometry)
            bohr_to_ang = 0.529177
            xyz_lines = [str(len(symbols)), "from QCArchive"]
            for i, sym in enumerate(symbols):
                if hasattr(geom[0], "__len__"):
                    x, y, z = geom[i]
                else:
                    x = geom[i*3] * bohr_to_ang
                    y = geom[i*3+1] * bohr_to_ang
                    z = geom[i*3+2] * bohr_to_ang
                xyz_lines.append(f"{sym} {x:.6f} {y:.6f} {z:.6f}")
            from basisrec.xyz_parser import parse_xyz_string
            mol_obj = parse_xyz_string("\n".join(xyz_lines))

        features_list = _featurize_geometry_fallback(mol_obj)

        # Reference energy
        ref_energy = run_pyscf_single_point(
            symbols, geometry, REFERENCE_BASIS, charge, mult
        )
        if ref_energy is None:
            logger.warning("Reference calc failed for mol %d — skipping.", mol_idx)
            skipped += 1
            continue

        # Candidate energies
        candidate_energies: Dict[str, Optional[float]] = {}
        for basis in CANDIDATE_BASIS:
            e = run_pyscf_single_point(symbols, geometry, basis, charge, mult)
            candidate_energies[basis] = e

        # Labels
        labels = label_atoms(symbols, ref_energy, candidate_energies)

        # Encode features and add to dataset
        for feat, label in zip(features_list, labels):
            encoded = _encode_atom_features(feat)
            dataset.append((encoded, label))

    logger.info(
        "Dataset built: %d atom samples from %d molecules (%d skipped).",
        len(dataset), len(molecules) - skipped, skipped,
    )

    with open(args.output, "wb") as fh:
        pickle.dump(dataset, fh)

    logger.info("Saved training set to '%s'.", args.output)
    logger.info("Run training with: python scripts/train_gnn.py")


if __name__ == "__main__":
    main()
