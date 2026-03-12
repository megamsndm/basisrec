"""Method recommender — DFT functional, auxiliary basis, approximations,
memory, and CPU thread recommendations for ORCA.

Given a molecule and its per-atom features, this module outputs a
:class:`MethodRecommendation` that contains every ORCA keyword needed
for a production-quality calculation:

    - DFT functional (+ Jacob's ladder tier)
    - Dispersion correction (D3BJ, D4, …)
    - Auxiliary basis (def2/J, def2/JK, …)
    - Fitting approximation (RIJCOSX, RIJK, RI-MP2, …)
    - SCF convergence threshold (NormalSCF / TightSCF / VeryTightSCF)
    - Integration grid (Grid3 … Grid6)
    - Relativistic treatment (ZORA, DKH, or none)
    - Solvent model (CPCM / SMD if requested)
    - Memory per core (MB)
    - Number of MPI processes
    - Extra keywords (e.g. UKS for open-shell, NORI for small systems)

Decision logic is encoded as a priority-ordered rule tree that mirrors
how an experienced computational chemist would choose settings:

    1. Open-shell?            → UKS/ROHF, looser convergence
    2. Transition metal?      → B3LYP or TPSS + D3BJ, larger grid, ZORA
    3. Dispersion-dominated?  → PBE-D4 or r²SCAN-D4
    4. Excited states needed? → CAM-B3LYP or wB97X-D3
    5. Anions / lone pairs?   → extra diffuse, aug- basis
    6. Large molecule (>50)?  → PBE-D3BJ + RIJCOSX, coarser grid
    7. High accuracy needed?  → DLPNO-CCSD(T) suggestion
    8. Default organic        → B3LYP-D3BJ / def2-SVP or def2-TZVP
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional

from .models import AtomFeatures, BasisRecommendation, Molecule

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output data model
# ---------------------------------------------------------------------------

@dataclass
class MethodRecommendation:
    """Complete method recommendation for an ORCA calculation."""

    # --- Functional & dispersion ---
    functional: str
    dispersion: str                     # e.g. 'D3BJ', 'D4', '' for none
    jacobs_ladder: str                  # 'GGA', 'meta-GGA', 'hybrid', 'double-hybrid'

    # --- Auxiliary basis & fitting ---
    aux_basis: str                      # e.g. 'def2/J', 'def2/JK', 'cc-pVTZ/C'
    fitting_approx: str                 # e.g. 'RIJCOSX', 'RIJK', 'RI', ''

    # --- SCF & grid ---
    scf_convergence: str                # 'NormalSCF' | 'TightSCF' | 'VeryTightSCF'
    grid: str                           # 'Grid3' | 'Grid4' | 'Grid5' | 'Grid6'
    final_grid: str                     # 'FinalGrid5' | 'FinalGrid6'

    # --- Relativistic ---
    relativistic: str                   # 'ZORA' | 'DKH2' | ''

    # --- Spin ---
    scf_type: str                       # 'RKS' | 'UKS' | 'ROHF'

    # --- Resources ---
    memory_per_core_mb: int
    nprocs: int

    # --- Solvent (optional) ---
    solvent_model: str                  # 'CPCM' | 'SMD' | ''
    solvent: str                        # e.g. 'Water', 'DMSO', ''

    # --- Extra keywords ---
    extra_keywords: List[str] = field(default_factory=list)

    # --- Human-readable reasoning ---
    reasoning: List[str] = field(default_factory=list)

    def keywords_line(self) -> str:
        """Return the full ORCA '!' keywords line."""
        parts = [f"! {self.scf_type}", self.functional]
        if self.dispersion:
            parts.append(self.dispersion)
        if self.aux_basis:
            parts.append(self.aux_basis)
        if self.fitting_approx:
            parts.append(self.fitting_approx)
        parts.append(self.scf_convergence)
        parts.append(self.grid)
        parts.append(self.final_grid)
        if self.relativistic:
            parts.append(self.relativistic)
        parts.extend(self.extra_keywords)
        return " ".join(parts)

    def pal_block(self) -> str:
        return f"%pal nprocs {self.nprocs} end"

    def maxcore_block(self) -> str:
        return f"%maxcore {self.memory_per_core_mb}"

    def scf_block(self) -> str:
        lines = [
            "%scf",
            "  MaxIter 500",
        ]
        if self.scf_type == "UKS":
            lines.append("  STABPerform true")
        lines.append("end")
        return "\n".join(lines)

    def solvent_block(self) -> str:
        if not self.solvent_model:
            return ""
        if self.solvent_model == "CPCM":
            return f"%cpcm\n  smd false\n  epsilon 80.0\nend"
        if self.solvent_model == "SMD":
            return f'%cpcm\n  smd true\n  SMDsolvent "{self.solvent}"\nend'
        return ""

    def reasoning_block(self) -> str:
        lines = ["# Method selection reasoning:"]
        for r in self.reasoning:
            lines.append(f"#   • {r}")
        return "\n".join(lines)

    def summary(self) -> str:
        lines = [
            "Method Recommendation",
            "=" * 50,
            f"Functional    : {self.functional} ({self.jacobs_ladder})",
            f"Dispersion    : {self.dispersion or 'none'}",
            f"Auxiliary     : {self.aux_basis}  [{self.fitting_approx}]",
            f"SCF type      : {self.scf_type}",
            f"Convergence   : {self.scf_convergence}",
            f"Grid          : {self.grid} / {self.final_grid}",
            f"Relativistic  : {self.relativistic or 'none'}",
            f"Memory/core   : {self.memory_per_core_mb} MB",
            f"MPI procs     : {self.nprocs}",
            f"Solvent       : {self.solvent_model or 'none'}",
            "",
            "Keywords line :",
            f"  {self.keywords_line()}",
            "",
            "Reasoning:",
        ]
        for r in self.reasoning:
            lines.append(f"  • {r}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# System profiler
# ---------------------------------------------------------------------------

@dataclass
class MoleculeProfile:
    """Derived properties used by the decision tree."""

    n_atoms: int
    n_electrons: int
    multiplicity: int
    charge: int
    has_transition_metal: bool
    has_heavy_atom: bool          # Z > 36
    has_lanthanide: bool
    has_lone_pairs: bool
    has_aromatic: bool
    has_hbond: bool
    has_halogen: bool
    is_open_shell: bool
    is_large: bool                # n_atoms > 50
    is_very_large: bool           # n_atoms > 150
    unique_elements: List[str]
    metal_symbols: List[str]


_ATOMIC_NUMBERS: dict[str, int] = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8,
    "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
    "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Sc": 21, "Ti": 22,
    "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29,
    "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36,
    "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Ru": 44,
    "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50, "Sb": 51,
    "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56, "La": 57, "Hf": 72,
    "Ta": 73, "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79,
    "Hg": 80, "Tl": 81, "Pb": 82, "Bi": 83, "U": 92, "Th": 90,
}

_TRANSITION_METALS = frozenset({
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
})

_LANTHANIDES = frozenset({
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
    "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
})

_HALOGENS = frozenset({"F", "Cl", "Br", "I", "At"})


def profile_molecule(
    molecule: Molecule,
    features: List[AtomFeatures],
    multiplicity: int = 1,
    charge: int = 0,
) -> MoleculeProfile:
    """Extract derived properties from molecule + features."""

    n_electrons = sum(
        _ATOMIC_NUMBERS.get(a.symbol, 0) for a in molecule.atoms
    ) - charge

    metal_symbols = sorted({
        f.symbol for f in features if f.symbol in _TRANSITION_METALS
    })

    return MoleculeProfile(
        n_atoms=molecule.n_atoms,
        n_electrons=n_electrons,
        multiplicity=multiplicity,
        charge=charge,
        has_transition_metal=any(f.symbol in _TRANSITION_METALS for f in features),
        has_heavy_atom=any(f.is_heavy for f in features),
        has_lanthanide=any(f.symbol in _LANTHANIDES for f in features),
        has_lone_pairs=any(f.has_lone_pairs for f in features),
        has_aromatic=any(f.is_aromatic for f in features),
        has_hbond=any(f.is_hbond_donor or f.is_hbond_acceptor for f in features),
        has_halogen=any(f.symbol in _HALOGENS for f in features),
        is_open_shell=(multiplicity != 1),
        is_large=(molecule.n_atoms > 50),
        is_very_large=(molecule.n_atoms > 150),
        unique_elements=molecule.unique_elements,
        metal_symbols=metal_symbols,
    )


# ---------------------------------------------------------------------------
# Decision tree
# ---------------------------------------------------------------------------

def recommend_method(
    molecule: Molecule,
    features: List[AtomFeatures],
    multiplicity: int = 1,
    charge: int = 0,
    accuracy: str = "balanced",      # 'fast' | 'balanced' | 'accurate' | 'benchmark'
    solvent: str = "",               # e.g. 'water', 'dmso', '' for gas phase
    nprocs: int = 4,
    memory_gb: int = 8,
    excited_states: bool = False,
    dispersion_dominated: bool = False,
) -> MethodRecommendation:
    """Recommend a complete ORCA method for *molecule*.

    Parameters
    ----------
    molecule:
        Parsed molecule.
    features:
        Per-atom features from the featurizer.
    multiplicity:
        Spin multiplicity (1 = singlet, 2 = doublet, …).
    charge:
        Total molecular charge.
    accuracy:
        'fast'      → PBE-D3BJ, Grid3, NormalSCF  (screening/geometry)
        'balanced'  → B3LYP-D3BJ, Grid4, TightSCF  (default, production)
        'accurate'  → wB97X-D3 or TPSS0-D3BJ, Grid5, TightSCF
        'benchmark' → DLPNO-CCSD(T) suggestion flagged in keywords
    solvent:
        Solvent name for implicit solvation (CPCM/SMD). '' = gas phase.
    nprocs:
        Number of CPU cores available.
    memory_gb:
        Total RAM available in GB. Memory per core is computed from this.
    excited_states:
        If True, recommend a range-separated or long-range-corrected functional.
    dispersion_dominated:
        If True, prefer a functional with excellent dispersion description.
    """
    prof = profile_molecule(molecule, features, multiplicity, charge)
    reasoning: List[str] = []
    extra_keywords: List[str] = []

    memory_per_core = max(512, int((memory_gb * 1024 * 0.85) / nprocs))

    # ------------------------------------------------------------------ #
    # 1. SCF type
    # ------------------------------------------------------------------ #
    if prof.is_open_shell:
        scf_type = "UKS"
        reasoning.append(
            f"Multiplicity = {multiplicity} → unrestricted KS (UKS). "
            "Stability analysis enabled (%scf STABPerform true)."
        )
    else:
        scf_type = "RKS"

    # ------------------------------------------------------------------ #
    # 2. Relativistic treatment
    # ------------------------------------------------------------------ #
    if prof.has_lanthanide:
        relativistic = "ZORA"
        reasoning.append(
            "Lanthanide detected → ZORA scalar relativistic treatment required."
        )
    elif prof.has_heavy_atom:
        relativistic = "ZORA"
        reasoning.append(
            "Heavy atom (Z > 36) detected → ZORA scalar relativistic treatment."
        )
    elif prof.has_transition_metal and any(
        _ATOMIC_NUMBERS.get(s, 0) > 36 for s in prof.metal_symbols
    ):
        relativistic = "ZORA"
        reasoning.append(
            f"5d transition metal ({', '.join(prof.metal_symbols)}) → ZORA recommended."
        )
    else:
        relativistic = ""

    # ------------------------------------------------------------------ #
    # 3. Functional selection
    # ------------------------------------------------------------------ #
    if accuracy == "benchmark":
        functional = "DLPNO-CCSD(T)"
        dispersion = ""
        jacobs_ladder = "coupled-cluster"
        aux_basis = "cc-pVTZ/C"
        fitting_approx = "RI-C"
        scf_convergence = "VeryTightSCF"
        grid = "Grid5"
        final_grid = "FinalGrid6"
        reasoning.append(
            "Benchmark accuracy requested → DLPNO-CCSD(T)/cc-pVTZ with tight PNO settings. "
            "Note: run an RKS/B3LYP geometry optimisation first."
        )
        extra_keywords += ["TightPNO", "NormalPNO"]

    elif excited_states:
        functional = "CAM-B3LYP"
        dispersion = "D3BJ"
        jacobs_ladder = "hybrid"
        aux_basis = "def2/J"
        fitting_approx = "RIJCOSX"
        scf_convergence = "TightSCF"
        grid = "Grid4"
        final_grid = "FinalGrid5"
        reasoning.append(
            "Excited states requested → CAM-B3LYP-D3BJ (range-separated hybrid). "
            "Accurate charge-transfer and Rydberg excitations. "
            "Follow up with TD-DFT: add '! TD-DFT NRoots 10' to your run."
        )

    elif prof.has_transition_metal or prof.has_lanthanide:
        metals_str = ", ".join(prof.metal_symbols) if prof.metal_symbols else "TM"
        if accuracy == "fast":
            functional = "PBE"
            dispersion = "D3BJ"
            jacobs_ladder = "GGA"
            aux_basis = "def2/J"
            fitting_approx = "RIJ"
            scf_convergence = "NormalSCF"
            grid = "Grid4"
            final_grid = "FinalGrid5"
            reasoning.append(
                f"Transition metal ({metals_str}) + fast mode → "
                "PBE-D3BJ/def2-SVP. Good for geometry screening."
            )
        else:
            functional = "TPSSh"
            dispersion = "D3BJ"
            jacobs_ladder = "hybrid meta-GGA"
            aux_basis = "def2/J"
            fitting_approx = "RIJCOSX"
            scf_convergence = "TightSCF"
            grid = "Grid5"
            final_grid = "FinalGrid6"
            reasoning.append(
                f"Transition metal ({metals_str}) → TPSSh-D3BJ. "
                "Meta-GGA hybrid; well-validated for 3d/4d/5d spin states, "
                "bond energies and geometries. Grid5 required for TM accuracy."
            )
            if prof.is_open_shell:
                reasoning.append(
                    f"Open-shell TM → check spin contamination: "
                    "add '! PrintMOs' and inspect <S²>."
                )

    elif dispersion_dominated:
        if accuracy in ("accurate", "benchmark"):
            functional = "r2SCAN"
            dispersion = "D4"
            jacobs_ladder = "meta-GGA"
            aux_basis = "def2/J"
            fitting_approx = "RIJ"
        else:
            functional = "PBE"
            dispersion = "D4"
            jacobs_ladder = "GGA"
            aux_basis = "def2/J"
            fitting_approx = "RIJ"
        scf_convergence = "TightSCF"
        grid = "Grid4"
        final_grid = "FinalGrid5"
        reasoning.append(
            "Dispersion-dominated system → "
            f"{functional}-D4. D4 dispersion is more accurate than D3BJ "
            "for stacking, van-der-Waals complexes and non-covalent interactions."
        )

    elif prof.is_very_large:
        functional = "PBE"
        dispersion = "D3BJ"
        jacobs_ladder = "GGA"
        aux_basis = "def2/J"
        fitting_approx = "RIJ"
        scf_convergence = "NormalSCF"
        grid = "Grid3"
        final_grid = "FinalGrid4"
        reasoning.append(
            f"Very large molecule ({prof.n_atoms} atoms) → "
            "PBE-D3BJ/def2-SVP with RIJ approximation. "
            "Fastest viable DFT for systems > 150 atoms."
        )

    elif prof.is_large:
        functional = "PBE0"
        dispersion = "D3BJ"
        jacobs_ladder = "hybrid"
        aux_basis = "def2/J"
        fitting_approx = "RIJCOSX"
        scf_convergence = "NormalSCF"
        grid = "Grid3"
        final_grid = "FinalGrid4"
        reasoning.append(
            f"Large molecule ({prof.n_atoms} atoms) → "
            "PBE0-D3BJ with RIJCOSX. Hybrid DFT remains feasible "
            "with density fitting up to ~150 atoms."
        )

    elif accuracy == "fast":
        functional = "PBE"
        dispersion = "D3BJ"
        jacobs_ladder = "GGA"
        aux_basis = "def2/J"
        fitting_approx = "RIJ"
        scf_convergence = "NormalSCF"
        grid = "Grid3"
        final_grid = "FinalGrid4"
        reasoning.append(
            "Fast mode → PBE-D3BJ/def2-SVP. "
            "Good for geometry optimisation and screening."
        )

    elif accuracy == "accurate":
        functional = "wB97X-D3"
        dispersion = ""          # D3 is built into wB97X-D3
        jacobs_ladder = "range-separated hybrid"
        aux_basis = "def2/JK"
        fitting_approx = "RIJCOSX"
        scf_convergence = "TightSCF"
        grid = "Grid5"
        final_grid = "FinalGrid6"
        reasoning.append(
            "Accurate mode → wB97X-D3. Range-separated hybrid with built-in D3 "
            "dispersion. Excellent for thermochemistry, barriers, and non-covalent "
            "interactions. One of the best overall functionals."
        )

    else:  # balanced (default)
        functional = "B3LYP"
        dispersion = "D3BJ"
        jacobs_ladder = "hybrid"
        aux_basis = "def2/J"
        fitting_approx = "RIJCOSX"
        scf_convergence = "TightSCF"
        grid = "Grid4"
        final_grid = "FinalGrid5"
        reasoning.append(
            "Balanced mode → B3LYP-D3BJ. "
            "Most widely validated functional for organic and bio-organic chemistry. "
            "D3BJ dispersion correction essential for geometries and energies."
        )

    # ------------------------------------------------------------------ #
    # 4. Grid upgrade for anions and lone-pair-rich systems
    # ------------------------------------------------------------------ #
    if prof.has_lone_pairs and grid in ("Grid3", "Grid4") and not prof.is_very_large:
        grid = "Grid4"
        final_grid = "FinalGrid5"
        reasoning.append(
            "Lone pairs / electronegative atoms detected → grid upgraded to Grid4 "
            "for accurate integration of diffuse density."
        )

    # ------------------------------------------------------------------ #
    # 5. Convergence tightening for charged / anions
    # ------------------------------------------------------------------ #
    if charge < 0 and scf_convergence == "NormalSCF":
        scf_convergence = "TightSCF"
        reasoning.append(
            f"Anion (charge={charge}) → SCF convergence tightened to TightSCF."
        )

    # ------------------------------------------------------------------ #
    # 6. Solvent model
    # ------------------------------------------------------------------ #
    solvent_model = ""
    solvent_name = ""
    if solvent:
        solvent_lower = solvent.lower()
        # SMD for absolute solvation energies; CPCM for geometry/spectra
        solvent_model = "SMD"
        solvent_map = {
            "water": "Water", "h2o": "Water",
            "dmso": "DMSO", "methanol": "Methanol",
            "ethanol": "Ethanol", "acetonitrile": "Acetonitrile",
            "chloroform": "Chloroform", "thf": "THF",
            "dichloromethane": "Dichloromethane", "dcm": "Dichloromethane",
            "toluene": "Toluene", "hexane": "Hexane",
        }
        solvent_name = solvent_map.get(solvent_lower, solvent.capitalize())
        extra_keywords.append(f"CPCM({solvent_name})")
        reasoning.append(
            f"Solvent '{solvent_name}' requested → SMD implicit solvation model. "
            "SMD preferred over plain CPCM for solvation free energies."
        )

    # ------------------------------------------------------------------ #
    # 7. Parallelism advice
    # ------------------------------------------------------------------ #
    recommended_nprocs = _recommend_nprocs(prof, nprocs, fitting_approx)
    if recommended_nprocs != nprocs:
        reasoning.append(
            f"Parallelism: {nprocs} cores requested but {recommended_nprocs} recommended "
            f"for this system size with {fitting_approx}. "
            "RIJCOSX scales well up to ~16 cores; beyond that returns diminish."
        )

    return MethodRecommendation(
        functional=functional,
        dispersion=dispersion,
        jacobs_ladder=jacobs_ladder,
        aux_basis=aux_basis,
        fitting_approx=fitting_approx,
        scf_convergence=scf_convergence,
        grid=grid,
        final_grid=final_grid,
        relativistic=relativistic,
        scf_type=scf_type,
        memory_per_core_mb=memory_per_core,
        nprocs=recommended_nprocs,
        solvent_model=solvent_model,
        solvent=solvent_name,
        extra_keywords=extra_keywords,
        reasoning=reasoning,
    )


def _recommend_nprocs(
    prof: MoleculeProfile,
    available: int,
    fitting_approx: str,
) -> int:
    """Suggest the optimal number of cores for this molecule and method."""
    if fitting_approx in ("RIJ", "RI"):
        # RI scales well but has communication overhead
        optimal = min(available, max(1, prof.n_atoms // 3))
    elif fitting_approx == "RIJCOSX":
        # RIJCOSX scales excellently up to ~16 cores
        optimal = min(available, 16, max(1, prof.n_atoms // 2))
    else:
        optimal = min(available, 8)

    # Always use at least 1, at most what's available
    return max(1, min(available, optimal))
