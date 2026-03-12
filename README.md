# BasisRec

**Automatic per-atom basis set recommendation with full ORCA input generation.**

Upload an XYZ file. Receive a complete, copy-paste-ready ORCA input file with basis sets selected per atom based on local chemical environment and physics.

---

## What it does

```
$ basisrec water.xyz
```

```
╭─────────────────────────────────────────────────────────────────╮
│ BasisRec — water (3 atoms)                                      │
├─────┬─────┬──────────────┬──────────────┬───────┬──────────────┤
│ Idx │ Sym │ Basis        │ Tier         │ Conf  │ Source       │
├─────┼─────┼──────────────┼──────────────┼───────┼──────────────┤
│ 0   │ O   │ aug-cc-pVDZ  │ augmented    │  92%  │ rule         │
│ 1   │ H   │ 6-31G**      │ double-zeta  │  88%  │ rule         │
│ 2   │ H   │ 6-31G**      │ double-zeta  │  88%  │ rule         │
╰─────┴─────┴──────────────┴──────────────┴───────┴──────────────╯

ORCA input written to: water.inp
```

The generated `water.inp` contains the **full Gaussian exponents and contraction coefficients** for every basis set, fetched live from the [Basis Set Exchange](https://www.basissetexchange.org).

---

## Installation

```bash
pip install basisrec
```

For optional PySCF validation support:

```bash
pip install basisrec[pyscf]
```

---

## Quick start

### Command line

```bash
# Basic usage
basisrec molecule.xyz

# Save to specific file
basisrec molecule.xyz --output molecule.inp

# Change method, memory, parallelism
basisrec caffeine.xyz --method "! RKS PBE0 TightSCF" --nprocs 16 --maxcore 4000

# Charged/open-shell molecule
basisrec radical.xyz --charge -1 --multiplicity 2

# Rule engine only (no GNN, faster, offline)
basisrec molecule.xyz --no-gnn

# Print full ORCA input to stdout
basisrec water.xyz --print-orca
```

### Python API

```python
from basisrec import recommend

# Full pipeline — returns RecommendationResult
result = recommend("water.xyz")

# Access the ORCA input string
print(result.orca_input)

# Save to file
result = recommend("molecule.xyz", output_path="molecule.inp")

# Inspect per-atom recommendations
for rec in result.recommendations:
    print(rec.atom_index, rec.symbol, rec.basis_name, f"{rec.confidence:.0%}")
    print(f"  Reason: {rec.reason}")

# Custom settings
result = recommend(
    "ferrocene.xyz",
    method_line="! RKS PBE0 TightSCF Grid5",
    nprocs=8,
    maxcore_mb=4000,
    use_gnn=False,
)
```

---

## How basis sets are selected

The decision engine applies physics-aware rules in priority order:

| Priority | Rule | Basis chosen |
|----------|------|-------------|
| Hard | Z > 36 (heavy atom) | def2-TZVP (with ECP) |
| Hard | Transition metal | def2-TZVP |
| Hard | Lanthanide/Actinide | def2-TZVP |
| Soft | Lone pairs + H-bond acceptor | aug-cc-pVDZ |
| Soft | Lone pairs present | aug-cc-pVDZ |
| Soft | sp2 oxygen (carbonyl) | aug-cc-pVDZ |
| Soft | Aromatic atom | cc-pVDZ |
| Soft | sp2 carbon | cc-pVDZ |
| Soft | H-bond donor (O/N-H) | aug-cc-pVDZ |
| Soft | H bonded to O/N | 6-31G** |
| Soft | Plain H (bonded to C) | 6-31G* |
| Default | Everything else | def2-SVP |

**Hard rules cannot be overridden** by the GNN. Soft rules can be upgraded (but not downgraded) if the GNN has high confidence (> 75%).

All basis set data (exponents, coefficients) is fetched live from the [Basis Set Exchange Python API](https://github.com/MolSSI-BSE/basis_set_exchange). Over 600 basis sets are available.

---

## ORCA input format

BasisRec uses ORCA's `%basis newgto / addgto` syntax for per-atom basis assignment:

```
%basis
  newgto "O"
  # aug-cc-pVDZ — full coefficients
  S   9 1.00
    11720.0000   0.000710
    ...
  end

  newgto "H"
  # 6-31G** — full coefficients
  ...
  end

  # Atom 4 (C in carbonyl): upgraded to aug-cc-pVDZ
  addgto 4 "C"
  ...
  end
end
```

---

## Training the GNN

The shipped `data/pretrained_gnn.pt` was trained on 10,000 small molecules from QCArchive. To retrain:

```bash
python scripts/train_gnn.py --epochs 100 --output data/my_model.pt
basisrec molecule.xyz --model-path data/my_model.pt
```

---

## Running tests

```bash
# Fast tests only (no integration)
pytest -m "not integration"

# All tests
pytest

# With coverage
pytest --cov=basisrec --cov-report=html
```

---

## License

All rights reserved.
