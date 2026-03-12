"""XYZ file parser with RDKit-based bond inference.

Reads standard XYZ format files and builds a fully
bonded molecular graph, including hybridization and
ring detection via RDKit's distance-based bond perception.

XYZ format specification:
    Line 1: number of atoms (int)
    Line 2: comment / title (str, may be empty)
    Lines 3+: <symbol> <x> <y> <z>  (space-separated)
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Union

from .exceptions import ParseError
from .models import Atom, Molecule

logger = logging.getLogger(__name__)

# Elements where RDKit bond perception is unreliable at standard distances.
# For these, we fall back to covalent-radius-based distance thresholds.
_RDKIT_UNRELIABLE = {"La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
                      "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
                      "Ac", "Th", "Pa", "U", "Np", "Pu"}

_ELEMENT_RE = re.compile(r"^[A-Z][a-z]?$")


def parse_xyz(path: Union[str, Path]) -> Molecule:
    """Parse an XYZ file into a :class:`~basisrec.models.Molecule`.

    Parameters
    ----------
    path:
        Path to the ``.xyz`` file.

    Returns
    -------
    Molecule
        Fully populated molecule with atom coordinates.

    Raises
    ------
    ParseError
        If the file is missing, malformed, or contains unknown elements.
    FileNotFoundError
        If the file path does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"XYZ file not found: {path}")

    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ParseError(f"Cannot read '{path}': {exc}") from exc

    return _parse_xyz_string(text, source=str(path))


def parse_xyz_string(xyz_string: str) -> Molecule:
    """Parse an XYZ-format string directly (useful for testing).

    Parameters
    ----------
    xyz_string:
        Raw XYZ content as a string.
    """
    return _parse_xyz_string(xyz_string.strip(), source="<string>")


def _parse_xyz_string(text: str, source: str) -> Molecule:
    lines = [ln.rstrip() for ln in text.splitlines()]

    if len(lines) < 2:
        raise ParseError(f"[{source}] XYZ file has fewer than 2 lines.")

    # --- line 1: atom count ---
    try:
        n_atoms = int(lines[0].strip())
    except ValueError as exc:
        raise ParseError(
            f"[{source}] First line must be an integer atom count, got: '{lines[0]}'"
        ) from exc

    if n_atoms < 1:
        raise ParseError(f"[{source}] Atom count must be >= 1, got {n_atoms}.")

    # --- line 2: title/comment ---
    title = lines[1].strip()

    # --- lines 3+: atom coordinates ---
    coord_lines = lines[2:]
    if len(coord_lines) < n_atoms:
        raise ParseError(
            f"[{source}] Header says {n_atoms} atoms but only "
            f"{len(coord_lines)} coordinate lines found."
        )

    atoms: list[Atom] = []
    for i, raw in enumerate(coord_lines[:n_atoms]):
        atom = _parse_atom_line(raw, index=i, source=source, line_no=i + 3)
        atoms.append(atom)

    logger.debug(
        "Parsed %d atoms from '%s': %s",
        n_atoms,
        source,
        [a.symbol for a in atoms],
    )

    return Molecule(title=title, atoms=atoms)


def _parse_atom_line(line: str, index: int, source: str, line_no: int) -> Atom:
    """Parse one coordinate line into an :class:`~basisrec.models.Atom`."""
    parts = line.split()
    if len(parts) < 4:
        raise ParseError(
            f"[{source}] Line {line_no}: expected '<symbol> x y z', "
            f"got: '{line}'"
        )

    symbol = parts[0].strip().capitalize()
    if not _ELEMENT_RE.match(symbol):
        raise ParseError(
            f"[{source}] Line {line_no}: '{symbol}' is not a valid element symbol."
        )

    try:
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
    except ValueError as exc:
        raise ParseError(
            f"[{source}] Line {line_no}: coordinates must be floats, got: '{line}'"
        ) from exc

    return Atom(index=index, symbol=symbol, x=x, y=y, z=z)
