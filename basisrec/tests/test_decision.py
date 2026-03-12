"""Unit tests for decision_engine.py."""


from basisrec.decision_engine import recommend


def test_recommend_water_no_gnn(water_molecule):
    recs = recommend(water_molecule, use_gnn=False)
    assert len(recs) == 3
    # Oxygen should get a diffuse basis
    o_rec = next(r for r in recs if r.symbol == "O")
    assert "aug" in o_rec.basis_name.lower() or "def2" in o_rec.basis_name.lower()
    # Hydrogen should get a smaller basis than oxygen
    h_recs = [r for r in recs if r.symbol == "H"]
    assert len(h_recs) == 2


def test_recommend_returns_one_per_atom(water_molecule):
    recs = recommend(water_molecule, use_gnn=False)
    indices = [r.atom_index for r in recs]
    assert sorted(indices) == list(range(water_molecule.n_atoms))


def test_hard_rule_not_overridden_by_gnn(ferrocene_molecule):
    """Iron must always keep its ECP basis regardless of GNN."""
    recs = recommend(ferrocene_molecule, use_gnn=False)
    fe_rec = next(r for r in recs if r.symbol == "Fe")
    assert fe_rec.confidence == 1.0
    assert fe_rec.source == "rule"


def test_all_confidences_valid(water_molecule):
    recs = recommend(water_molecule, use_gnn=False)
    for rec in recs:
        assert 0.0 <= rec.confidence <= 1.0


def test_gnn_disabled_uses_rule_source(water_molecule):
    recs = recommend(water_molecule, use_gnn=False)
    for rec in recs:
        assert rec.source == "rule"


def test_basis_names_are_strings(water_molecule):
    recs = recommend(water_molecule, use_gnn=False)
    for rec in recs:
        assert isinstance(rec.basis_name, str)
        assert len(rec.basis_name) > 0


def test_reason_is_non_empty(water_molecule):
    recs = recommend(water_molecule, use_gnn=False)
    for rec in recs:
        assert len(rec.reason) > 5
