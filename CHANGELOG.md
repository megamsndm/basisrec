# Changelog

All notable changes to BasisRec are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [1.0.0] — 2026-03-12

### Added
- **`xyz_parser.py`** — robust XYZ file parser with full error messages; validates atom counts, element symbols, coordinate format
- **`atom_featurizer.py`** — per-atom chemical feature extraction using RDKit (hybridization, Gasteiger charges, aromaticity, H-bond detection); geometry-only fallback when RDKit unavailable
- **`rule_engine.py`** — 12-rule physics-aware engine encoding: ECP for Z>36, ECP for transition metals and lanthanides, diffuse functions for lone pairs/H-bond acceptors, polarization for sp2/sp3, aromatic ring detection; hard vs soft rule distinction prevents GNN from overriding mandatory ECP
- **`gnn_model.py`** — GraphSAGE-style GNN (3-layer MLP stand-in until torch-geometric installed); degrades silently to rule-only mode when torch or weights file absent
- **`decision_engine.py`** — fusion logic: hard rules always win; GNN overrides soft rules only at >75% confidence and equal/higher tier
- **`basis_library.py`** — tier vocabulary, BSE coverage checker via `lru_cache`, fallback chains, `_TIER_ORDER_MAP` for tier comparison
- **`orca_writer.py`** — complete ORCA input generator: `%basis newgto/addgto` blocks with full BSE-fetched exponents; handles same-element atoms with different bases via per-atom `addgto` overrides
- **`report_generator.py`** — plain-text and Rich terminal table per-atom reasoning report
- **`api.py`** — `recommend()` public function: parse → featurize → decide → write → report in one call
- **`cli.py`** — Click CLI: `basisrec molecule.xyz [--output] [--method] [--nprocs] [--charge] [--no-gnn]`
- **`models.py`** — Pydantic v2 typed models for all inter-module data: `Atom`, `Molecule`, `AtomFeatures`, `BasisRecommendation`, `RecommendationResult`
- **`exceptions.py`** — named exception hierarchy: `ParseError`, `FeaturizationError`, `UnsupportedElementError`, `ModelNotFoundError`, `ORCAWriterError`
- **`__main__.py`** — `python -m basisrec` support
- **Test suite** — 70+ tests across 8 test files; unit + integration split; all passing against BSE local database (no internet required)
- **Example molecules** — water, ethanol, caffeine, ferrocene, glycine, cisplatin
- **`scripts/train_gnn.py`** — standalone GNN training script
- **`scripts/build_training_set.py`** — QCArchive data assembly with PySCF labelling
- **`scripts/benchmark.py`** — accuracy vs speed comparison across basis set strategies
- **CI** — GitHub Actions: test matrix Python 3.9/3.10/3.11, lint (ruff), type-check (mypy), coverage (codecov), wheel build (twine check)
- BSE Python API integration — all basis set exponents fetched live; `lru_cache` on element coverage lookups; no stored Gaussian coefficients in this repo

### Design decisions
- `basis_library.py` is metadata-only; BSE owns all exponents
- Hard rules (`is_hard=True`) carry confidence=1.0 and are mathematically guaranteed not to be downgraded
- GNN degrades silently — rule engine always works offline without torch installed
- ORCA `newgto`/`addgto` distinction handles heterogeneous per-atom basis cleanly for same-element atoms
- Pydantic models as the sole data contract between all layers — no raw dicts in function signatures

---

## Roadmap

### [1.1.0] — planned
- Per-atom energy decomposition for more precise GNN labels (DFTB-based)
- Gaussian 16 input writer (`%basis` block format)
- PySCF input writer
- `--validate` flag: run a quick HF/STO-3G single-point to confirm ORCA input parses correctly

### [1.2.0] — planned
- Pretrained GNN weights trained on 10,000 QCArchive molecules
- Excited-state basis recommendations (TD-DFT aware: aug- on chromophore atoms)
- Solvation-aware diffuse function promotion (PCM environment → more diffuse needed)
