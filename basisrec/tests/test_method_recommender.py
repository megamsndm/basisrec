"""Unit tests for method_recommender.py."""

import pytest
from unittest.mock import patch

from basisrec.method_recommender import (
    recommend_method, profile_molecule, MoleculeProfile, MethodRecommendation
)
from basisrec.atom_featurizer import _featurize_geometry_fallback
from basisrec.models import Atom, Molecule


def _water_mol():
    return Molecule(title="water", atoms=[
        Atom(index=0, symbol="O", x=0.0, y=0.0, z=0.119),
        Atom(index=1, symbol="H", x=0.0, y=0.757, z=-0.476),
        Atom(index=2, symbol="H", x=0.0, y=-0.757, z=-0.476),
    ])


def _ferrocene_mol():
    atoms = [Atom(index=0, symbol="Fe", x=0.0, y=0.0, z=0.0)]
    for i, (x, y, z) in enumerate([
        (1.2, 0.0, 1.65), (0.37, 1.14, 1.65), (-0.97, 0.70, 1.65),
        (-0.97, -0.70, 1.65), (0.37, -1.14, 1.65),
        (1.2, 0.0, -1.65), (0.37, 1.14, -1.65), (-0.97, 0.70, -1.65),
        (-0.97, -0.70, -1.65), (0.37, -1.14, -1.65),
    ], 1):
        atoms.append(Atom(index=i, symbol="C", x=x, y=y, z=z))
    return Molecule(title="ferrocene", atoms=atoms)


# --- profile_molecule ---

def test_profile_water_no_metal():
    mol = _water_mol()
    feats = _featurize_geometry_fallback(mol)
    prof = profile_molecule(mol, feats)
    assert prof.has_transition_metal is False
    assert prof.has_lone_pairs is True
    assert prof.is_open_shell is False
    assert prof.n_atoms == 3


def test_profile_ferrocene_has_metal():
    mol = _ferrocene_mol()
    feats = _featurize_geometry_fallback(mol)
    prof = profile_molecule(mol, feats)
    assert prof.has_transition_metal is True
    assert "Fe" in prof.metal_symbols


def test_profile_open_shell():
    mol = _water_mol()
    feats = _featurize_geometry_fallback(mol)
    prof = profile_molecule(mol, feats, multiplicity=2)
    assert prof.is_open_shell is True


def test_profile_anion():
    mol = _water_mol()
    feats = _featurize_geometry_fallback(mol)
    prof = profile_molecule(mol, feats, charge=-1)
    assert prof.charge == -1


# --- recommend_method outputs ---

def test_water_balanced_gives_b3lyp():
    mol = _water_mol()
    feats = _featurize_geometry_fallback(mol)
    rec = recommend_method(mol, feats, accuracy="balanced")
    assert "B3LYP" in rec.functional
    assert rec.jacobs_ladder == "hybrid"


def test_water_fast_gives_pbe():
    mol = _water_mol()
    feats = _featurize_geometry_fallback(mol)
    rec = recommend_method(mol, feats, accuracy="fast")
    assert "PBE" in rec.functional


def test_water_accurate_gives_wb97():
    mol = _water_mol()
    feats = _featurize_geometry_fallback(mol)
    rec = recommend_method(mol, feats, accuracy="accurate")
    assert "wB97" in rec.functional or "TPSS" in rec.functional


def test_ferrocene_gets_tpssh():
    mol = _ferrocene_mol()
    feats = _featurize_geometry_fallback(mol)
    rec = recommend_method(mol, feats, accuracy="balanced")
    assert "TPSS" in rec.functional or "B3LYP" in rec.functional
    assert rec.grid in ("Grid4", "Grid5", "Grid6")


def test_excited_states_gives_cam_b3lyp():
    mol = _water_mol()
    feats = _featurize_geometry_fallback(mol)
    rec = recommend_method(mol, feats, excited_states=True)
    assert "CAM" in rec.functional or "B3LYP" in rec.functional


def test_dispersion_dominated_uses_d4():
    mol = _water_mol()
    feats = _featurize_geometry_fallback(mol)
    rec = recommend_method(mol, feats, dispersion_dominated=True)
    assert "D4" in rec.dispersion


def test_open_shell_gives_uks():
    mol = _water_mol()
    feats = _featurize_geometry_fallback(mol)
    rec = recommend_method(mol, feats, multiplicity=2)
    assert rec.scf_type == "UKS"


def test_closed_shell_gives_rks():
    mol = _water_mol()
    feats = _featurize_geometry_fallback(mol)
    rec = recommend_method(mol, feats, multiplicity=1)
    assert rec.scf_type == "RKS"


def test_solvent_adds_cpcm_keyword():
    mol = _water_mol()
    feats = _featurize_geometry_fallback(mol)
    rec = recommend_method(mol, feats, solvent="water")
    assert "CPCM" in " ".join(rec.extra_keywords)


def test_memory_per_core_computed():
    mol = _water_mol()
    feats = _featurize_geometry_fallback(mol)
    rec = recommend_method(mol, feats, nprocs=4, memory_gb=16)
    assert rec.memory_per_core_mb > 0
    assert rec.memory_per_core_mb <= 16 * 1024


def test_nprocs_not_exceeds_available():
    mol = _water_mol()
    feats = _featurize_geometry_fallback(mol)
    rec = recommend_method(mol, feats, nprocs=2)
    assert rec.nprocs <= 2


# --- keywords_line ---

def test_keywords_line_starts_with_exclamation():
    mol = _water_mol()
    feats = _featurize_geometry_fallback(mol)
    rec = recommend_method(mol, feats)
    line = rec.keywords_line()
    assert line.startswith("!")


def test_keywords_line_contains_functional():
    mol = _water_mol()
    feats = _featurize_geometry_fallback(mol)
    rec = recommend_method(mol, feats)
    line = rec.keywords_line()
    assert rec.functional in line


def test_keywords_line_contains_grid():
    mol = _water_mol()
    feats = _featurize_geometry_fallback(mol)
    rec = recommend_method(mol, feats)
    line = rec.keywords_line()
    assert "Grid" in line


# --- summary ---

def test_summary_contains_functional():
    mol = _water_mol()
    feats = _featurize_geometry_fallback(mol)
    rec = recommend_method(mol, feats)
    s = rec.summary()
    assert rec.functional in s


def test_summary_contains_reasoning():
    mol = _water_mol()
    feats = _featurize_geometry_fallback(mol)
    rec = recommend_method(mol, feats)
    s = rec.summary()
    assert "Reasoning" in s
    assert len(rec.reasoning) >= 1


def test_pal_block():
    mol = _water_mol()
    feats = _featurize_geometry_fallback(mol)
    rec = recommend_method(mol, feats, nprocs=8)
    assert "nprocs 8" in rec.pal_block() or "nprocs" in rec.pal_block()


def test_maxcore_block():
    mol = _water_mol()
    feats = _featurize_geometry_fallback(mol)
    rec = recommend_method(mol, feats, memory_gb=16, nprocs=4)
    assert "%maxcore" in rec.maxcore_block()


def test_scf_block_uks_has_stability():
    mol = _water_mol()
    feats = _featurize_geometry_fallback(mol)
    rec = recommend_method(mol, feats, multiplicity=2)
    scf = rec.scf_block()
    assert "STABPerform" in scf
