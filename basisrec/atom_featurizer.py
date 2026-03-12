"""Per-atom chemical feature extraction.

Converts a parsed :class:`~basisrec.models.Molecule` into a list of
:class:`~basisrec.models.AtomFeatures` using RDKit for bond perception,
hybridization assignment, aromaticity detection, and Gasteiger charge
calculation.

Each feature directly maps to a physically meaningful basis set need:

    hybridization SP2/SP3     → polarization functions required
    has_lone_pairs True       → diffuse functions (aug-) required
    is_hbond_donor/acceptor   → polarization on H; diffuse on acceptor
    is_heavy True             → ECP basis required (Z > 36)
    is_metal True             → relativistic ECP, larger valence space
    period >= 4               → scalar relativistic effects non-negligible
"""

from __future__ import annotations

import logging
from typing import List

from .exceptions import FeaturizationError
from .models import Atom, AtomFeatures, Hybridization, Molecule

logger = logging.getLogger(__name__)

# Periodic table data: symbol → (atomic_number, period)
_ELEMENT_DATA: dict[str, tuple[int, int]] = {
    "H": (1, 1), "He": (2, 1),
    "Li": (3, 2), "Be": (4, 2), "B": (5, 2), "C": (6, 2),
    "N": (7, 2), "O": (8, 2), "F": (9, 2), "Ne": (10, 2),
    "Na": (11, 3), "Mg": (12, 3), "Al": (13, 3), "Si": (14, 3),
    "P": (15, 3), "S": (16, 3), "Cl": (17, 3), "Ar": (18, 3),
    "K": (19, 4), "Ca": (20, 4), "Sc": (21, 4), "Ti": (22, 4),
    "V": (23, 4), "Cr": (24, 4), "Mn": (25, 4), "Fe": (26, 4),
    "Co": (27, 4), "Ni": (28, 4), "Cu": (29, 4), "Zn": (30, 4),
    "Ga": (31, 4), "Ge": (32, 4), "As": (33, 4), "Se": (34, 4),
    "Br": (35, 4), "Kr": (36, 4),
    "Rb": (37, 5), "Sr": (38, 5), "Y": (39, 5), "Zr": (40, 5),
    "Nb": (41, 5), "Mo": (42, 5), "Tc": (43, 5), "Ru": (44, 5),
    "Rh": (45, 5), "Pd": (46, 5), "Ag": (47, 5), "Cd": (48, 5),
    "In": (49, 5), "Sn": (50, 5), "Sb": (51, 5), "Te": (52, 5),
    "I": (53, 5), "Xe": (54, 5),
    "Cs": (55, 6), "Ba": (56, 6), "La": (57, 6), "Hf": (72, 6),
    "Ta": (73, 6), "W": (74, 6), "Re": (75, 6), "Os": (76, 6),
    "Ir": (77, 6), "Pt": (78, 6), "Au": (79, 6), "Hg": (80, 6),
    "Tl": (81, 6), "Pb": (82, 6), "Bi": (83, 6),
    "U": (92, 7), "Th": (90, 7),
}

# d-block elements (transition metals)
_TRANSITION_METALS: frozenset[str] = frozenset({
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
})

# Elements that always have lone pairs (relevant for diffuse function need)
_LONE_PAIR_ELEMENTS: frozenset[str] = frozenset({
    "N", "O", "F", "P", "S", "Cl", "Se", "Br", "I",
    "As", "Te", "At",
})

# H-bond donor elements (when bonded to H)
_HBOND_DONOR_ELEMENTS: frozenset[str] = frozenset({"N", "O", "F"})

# H-bond acceptor elements
_HBOND_ACCEPTOR_ELEMENTS: frozenset[str] = frozenset({"N", "O", "F", "S", "Cl"})


def featurize_molecule(molecule: Molecule) -> List[AtomFeatures]:
    """Compute chemical features for every atom in *molecule*.

    Attempts to use RDKit for full bond perception and hybridization.
    Falls back to a geometry-only heuristic if RDKit is unavailable
    or bond perception fails.

    Parameters
    ----------
    molecule:
        Parsed molecule from :func:`~basisrec.xyz_parser.parse_xyz`.

    Returns
    -------
    list[AtomFeatures]
        One feature object per atom, in the same order as
        ``molecule.atoms``.

    Raises
    ------
    FeaturizationError
        If feature extraction fails and no fallback is possible.
    """
    try:
        return _featurize_with_rdkit(molecule)
    except ImportError:
        logger.warning(
            "RDKit not available — falling back to geometry-only featurizer. "
            "Install rdkit for more accurate basis set recommendations."
        )
        return _featurize_geometry_fallback(molecule)
    except Exception as exc:
        logger.warning(
            "RDKit featurization failed (%s) — falling back to geometry-only.", exc
        )
        return _featurize_geometry_fallback(molecule)


def _featurize_with_rdkit(molecule: Molecule) -> List[AtomFeatures]:
    """Full featurization via RDKit bond perception."""
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds, rdPartialCharges

    # Build RDKit mol from xyz block
    xyz_block = _molecule_to_xyz_block(molecule)
    raw_mol = Chem.MolFromXYZBlock(xyz_block)
    if raw_mol is None:
        raise FeaturizationError("RDKit could not parse the molecule geometry.")

    # Determine bonds from 3D coordinates
    try:
        rdDetermineBonds.DetermineConnectivity(raw_mol)
        rdDetermineBonds.DetermineBondOrders(raw_mol, charge=molecule.charge)
    except Exception as exc:
        logger.debug("Bond order determination failed: %s — using connectivity only", exc)
        try:
            rdDetermineBonds.DetermineConnectivity(raw_mol)
        except Exception:
            pass

    # Compute Gasteiger partial charges
    try:
        rdPartialCharges.ComputeGasteigerCharges(raw_mol)
    except Exception:
        logger.debug("Gasteiger charge computation failed — using 0.0 fallback")

    features: list[AtomFeatures] = []
    for atom in molecule.atoms:
        rdatom = raw_mol.GetAtomWithIdx(atom.index)
        feat = _extract_rdkit_atom_features(atom, rdatom, molecule)
        features.append(feat)

    return features


def _extract_rdkit_atom_features(
    atom: Atom, rdatom, molecule: Molecule
) -> AtomFeatures:
    """Extract features from a single RDKit atom object."""
    from rdkit.Chem import rdchem

    symbol = atom.symbol
    z, period = _ELEMENT_DATA.get(symbol, (0, 1))

    # Hybridization
    hyb_map = {
        rdchem.HybridizationType.S: Hybridization.S,
        rdchem.HybridizationType.SP: Hybridization.SP,
        rdchem.HybridizationType.SP2: Hybridization.SP2,
        rdchem.HybridizationType.SP3: Hybridization.SP3,
        rdchem.HybridizationType.SP3D: Hybridization.SP3D,
        rdchem.HybridizationType.SP3D2: Hybridization.SP3D2,
    }
    hyb = hyb_map.get(rdatom.GetHybridization(), Hybridization.UNKNOWN)

    # Neighbor symbols
    neighbor_symbols = [
        n.GetSymbol() for n in rdatom.GetNeighbors()
    ]

    # Partial charge (Gasteiger stored as property)
    try:
        charge = float(rdatom.GetDoubleProp("_GasteigerCharge"))
        if charge != charge:  # NaN check
            charge = 0.0
    except KeyError:
        charge = 0.0

    # H-bond roles
    is_hbond_donor = (
        symbol in _HBOND_DONOR_ELEMENTS
        and "H" in neighbor_symbols
    )
    is_hbond_acceptor = symbol in _HBOND_ACCEPTOR_ELEMENTS

    return AtomFeatures(
        index=atom.index,
        symbol=symbol,
        hybridization=hyb,
        coordination_number=rdatom.GetDegree(),
        is_aromatic=rdatom.GetIsAromatic(),
        is_hbond_donor=is_hbond_donor,
        is_hbond_acceptor=is_hbond_acceptor,
        has_lone_pairs=symbol in _LONE_PAIR_ELEMENTS,
        is_metal=symbol in _TRANSITION_METALS,
        is_heavy=z > 36,
        partial_charge=charge,
        period=period,
        neighbor_symbols=neighbor_symbols,
    )


def _featurize_geometry_fallback(molecule: Molecule) -> List[AtomFeatures]:
    """Geometry-only fallback: estimate coordination from distance matrix."""
    import math

    coords = [(a.x, a.y, a.z) for a in molecule.atoms]
    # Rough covalent radii (Angstrom) for coordination number estimation
    covalent_r: dict[str, float] = {
        "H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
        "P": 1.07, "S": 1.05, "Cl": 1.02, "Br": 1.20, "I": 1.39,
        "Fe": 1.32, "Cu": 1.32, "Zn": 1.22,
    }
    default_r = 1.0
    bond_tolerance = 1.3  # Angstrom tolerance factor

    features: list[AtomFeatures] = []
    for i, atom in enumerate(molecule.atoms):
        neighbors = []
        ri = covalent_r.get(atom.symbol, default_r)
        for j, other in enumerate(molecule.atoms):
            if i == j:
                continue
            rj = covalent_r.get(other.symbol, default_r)
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            dz = coords[i][2] - coords[j][2]
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            if dist < (ri + rj) * bond_tolerance:
                neighbors.append(other.symbol)

        symbol = atom.symbol
        z, period = _ELEMENT_DATA.get(symbol, (0, 1))

        is_hbond_donor = symbol in _HBOND_DONOR_ELEMENTS and "H" in neighbors
        is_hbond_acceptor = symbol in _HBOND_ACCEPTOR_ELEMENTS

        features.append(AtomFeatures(
            index=atom.index,
            symbol=symbol,
            hybridization=Hybridization.UNKNOWN,
            coordination_number=len(neighbors),
            is_aromatic=False,
            is_hbond_donor=is_hbond_donor,
            is_hbond_acceptor=is_hbond_acceptor,
            has_lone_pairs=symbol in _LONE_PAIR_ELEMENTS,
            is_metal=symbol in _TRANSITION_METALS,
            is_heavy=z > 36,
            partial_charge=0.0,
            period=period,
            neighbor_symbols=neighbors,
        ))

    return features


def _molecule_to_xyz_block(molecule: Molecule) -> str:
    """Render a Molecule back to an XYZ string for RDKit ingestion."""
    lines = [str(molecule.n_atoms), molecule.title]
    for atom in molecule.atoms:
        lines.append(f"{atom.symbol}  {atom.x:.6f}  {atom.y:.6f}  {atom.z:.6f}")
    return "\n".join(lines)
