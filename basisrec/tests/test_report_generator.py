"""Unit tests for report_generator.py."""

import pytest
from basisrec.report_generator import generate_report
from basisrec.models import BasisRecommendation, BasisTier, Molecule, Atom


@pytest.fixture
def simple_molecule():
    return Molecule(
        title="ethanol",
        atoms=[
            Atom(index=0, symbol="C",  x=0.0, y=0.0, z=0.0),
            Atom(index=1, symbol="O",  x=1.4, y=0.0, z=0.0),
            Atom(index=2, symbol="H",  x=-0.5, y=0.9, z=0.0),
        ],
    )


@pytest.fixture
def simple_recs():
    return [
        BasisRecommendation(
            atom_index=0, symbol="C", basis_name="def2-SVP",
            tier=BasisTier.DOUBLE_ZETA, confidence=0.80,
            reason="C: sp3 carbon", source="rule",
        ),
        BasisRecommendation(
            atom_index=1, symbol="O", basis_name="aug-cc-pVDZ",
            tier=BasisTier.AUGMENTED, confidence=0.92,
            reason="O: lone pairs", source="rule",
        ),
        BasisRecommendation(
            atom_index=2, symbol="H", basis_name="6-31G*",
            tier=BasisTier.DOUBLE_ZETA, confidence=0.85,
            reason="H: light", source="rule",
        ),
    ]


def test_report_contains_title(simple_molecule, simple_recs):
    report = generate_report(simple_molecule, simple_recs)
    assert "ethanol" in report


def test_report_contains_all_symbols(simple_molecule, simple_recs):
    report = generate_report(simple_molecule, simple_recs)
    assert "C" in report
    assert "O" in report
    assert "H" in report


def test_report_contains_all_basis_names(simple_molecule, simple_recs):
    report = generate_report(simple_molecule, simple_recs)
    assert "def2-SVP" in report
    assert "aug-cc-pVDZ" in report
    assert "6-31G*" in report


def test_report_contains_reasoning_section(simple_molecule, simple_recs):
    report = generate_report(simple_molecule, simple_recs)
    assert "Reasoning" in report


def test_report_contains_unique_basis_section(simple_molecule, simple_recs):
    report = generate_report(simple_molecule, simple_recs)
    assert "Unique basis sets" in report


def test_report_contains_bse_note(simple_molecule, simple_recs):
    report = generate_report(simple_molecule, simple_recs)
    assert "BSE" in report or "Basis Set Exchange" in report


def test_report_contains_atom_count(simple_molecule, simple_recs):
    report = generate_report(simple_molecule, simple_recs)
    assert "3" in report


def test_report_is_string(simple_molecule, simple_recs):
    report = generate_report(simple_molecule, simple_recs)
    assert isinstance(report, str)
    assert len(report) > 100
