"""Shared pytest fixtures for the BasisRec test suite."""

import pytest
from basisrec.models import Atom, Molecule


@pytest.fixture
def water_xyz(tmp_path):
    f = tmp_path / "water.xyz"
    f.write_text("3\nwater\nO  0.000  0.000  0.119\nH  0.000  0.757 -0.476\nH  0.000 -0.757 -0.476\n")
    return f


@pytest.fixture
def water_molecule():
    return Molecule(
        title="water",
        atoms=[
            Atom(index=0, symbol="O", x=0.0,   y=0.0,    z=0.119),
            Atom(index=1, symbol="H", x=0.0,   y=0.757,  z=-0.476),
            Atom(index=2, symbol="H", x=0.0,   y=-0.757, z=-0.476),
        ],
    )


@pytest.fixture
def ferrocene_molecule():
    """Fe with two cyclopentadienyl rings (simplified geometry)."""
    atoms = [Atom(index=0, symbol="Fe", x=0.0, y=0.0, z=0.0)]
    for i, (x, y, z) in enumerate([
        (1.2, 0.0, 1.65), (0.37, 1.14, 1.65), (-0.97, 0.70, 1.65),
        (-0.97, -0.70, 1.65), (0.37, -1.14, 1.65),
        (1.2, 0.0, -1.65), (0.37, 1.14, -1.65), (-0.97, 0.70, -1.65),
        (-0.97, -0.70, -1.65), (0.37, -1.14, -1.65),
    ], 1):
        atoms.append(Atom(index=i, symbol="C", x=x, y=y, z=z))
    return Molecule(title="ferrocene", atoms=atoms)


@pytest.fixture
def caffeine_xyz(tmp_path):
    content = """24
caffeine
 C  -1.2144  1.3191 -0.0000
 C  -0.3604  2.3866 -0.0000
 N   1.0200  2.1998 -0.0000
 C   1.4706  0.9294 -0.0000
 N   0.4972 -0.0000 -0.0000
 C  -0.8567  0.0631 -0.0000
 N  -1.6767 -1.0822 -0.0000
 C  -0.8832 -2.1266 -0.0000
 N   0.4392 -1.9755 -0.0000
 C   2.9506  0.7458 -0.0000
 C  -2.6644  1.5551 -0.0000
 C   1.9067  3.2543 -0.0000
 O  -1.2024 -3.3135 -0.0000
 O   2.5244 -0.3447 -0.0000
 H  -0.7067  3.4014 -0.0000
 H   3.3498  1.7618 -0.0000
 H   3.4328  0.1700  0.8810
 H   3.4328  0.1700 -0.8810
 H  -3.0818  0.5499 -0.0000
 H  -2.9604  2.1256  0.8843
 H  -2.9604  2.1256 -0.8843
 H   1.5266  4.2665 -0.0000
 H   2.5390  3.1698  0.8843
 H   2.5390  3.1698 -0.8843
"""
    f = tmp_path / "caffeine.xyz"
    f.write_text(content)
    return f
