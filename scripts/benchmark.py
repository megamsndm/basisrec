"""Benchmark BasisRec recommendations against standard basis set strategies.

Runs three strategies on each test molecule and reports:
    - Mean absolute energy error (MAE) vs cc-pVTZ reference
    - Wall-clock time per molecule
    - Basis set augmentation rate

Strategies compared:
    1. BasisRec (rule engine only)         — this package, no GNN
    2. BasisRec (rule engine + GNN)        — this package, full pipeline
    3. Uniform def2-SVP                    — standard practice baseline
    4. Uniform 6-31G*                      — common fast baseline
    5. Uniform def2-TZVP                   — high-accuracy baseline

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --molecules data/example_molecules/ --n 50
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class BenchmarkResult(NamedTuple):
    strategy:        str
    molecule:        str
    n_atoms:         int
    energy_hartree:  Optional[float]
    ref_energy:      Optional[float]
    mae_kcal:        Optional[float]
    wall_time_s:     float
    unique_bases:    List[str]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark BasisRec vs fixed basis sets.")
    p.add_argument(
        "--molecules", type=Path,
        default=Path(__file__).parent.parent / "data" / "example_molecules",
        help="Directory of .xyz files to benchmark on.",
    )
    p.add_argument(
        "--n", type=int, default=None,
        help="Max number of molecules to benchmark (default: all).",
    )
    p.add_argument(
        "--output", type=Path, default=Path("benchmark_results.csv"),
        help="CSV output path.",
    )
    p.add_argument(
        "--no-pyscf", action="store_true",
        help="Skip energy calculations; report timing and basis stats only.",
    )
    return p.parse_args()


HARTREE_TO_KCAL = 627.509


def run_pyscf_energy(
    xyz_path: Path,
    basis: str | Dict[str, str],
    charge: int = 0,
    multiplicity: int = 1,
) -> Optional[float]:
    """Run PySCF HF and return total energy in Hartree."""
    try:
        import pyscf
        from basisrec.xyz_parser import parse_xyz

        mol_data = parse_xyz(xyz_path)
        atom_str = "; ".join(
            f"{a.symbol} {a.x:.6f} {a.y:.6f} {a.z:.6f}"
            for a in mol_data.atoms
        )
        mol = pyscf.gto.M(
            atom=atom_str,
            basis=basis,
            charge=charge,
            spin=multiplicity - 1,
            verbose=0,
        )
        mf = pyscf.scf.RHF(mol)
        mf.max_cycle = 200
        e = mf.kernel()
        return float(e) if mf.converged else None
    except Exception as exc:
        logger.debug("PySCF failed: %s", exc)
        return None


def benchmark_molecule(
    xyz_path: Path,
    strategy_name: str,
    basis_or_recs,
    ref_energy: Optional[float],
    run_pyscf: bool,
) -> BenchmarkResult:
    from basisrec.xyz_parser import parse_xyz

    mol = parse_xyz(xyz_path)
    t0 = time.perf_counter()

    if run_pyscf and isinstance(basis_or_recs, str):
        energy = run_pyscf_energy(xyz_path, basis_or_recs)
        unique_bases = [basis_or_recs]
    elif run_pyscf and isinstance(basis_or_recs, dict):
        energy = run_pyscf_energy(xyz_path, basis_or_recs)
        unique_bases = sorted(set(basis_or_recs.values()))
    else:
        energy = None
        if isinstance(basis_or_recs, dict):
            unique_bases = sorted(set(basis_or_recs.values()))
        elif isinstance(basis_or_recs, list):
            unique_bases = sorted(set(r.basis_name for r in basis_or_recs))
        else:
            unique_bases = [str(basis_or_recs)]

    wall_time = time.perf_counter() - t0

    mae = None
    if energy is not None and ref_energy is not None:
        mae = abs(energy - ref_energy) * HARTREE_TO_KCAL

    return BenchmarkResult(
        strategy=strategy_name,
        molecule=xyz_path.stem,
        n_atoms=mol.n_atoms,
        energy_hartree=energy,
        ref_energy=ref_energy,
        mae_kcal=mae,
        wall_time_s=wall_time,
        unique_bases=unique_bases,
    )


def main() -> None:
    args = parse_args()

    try:
        import pyscf
        has_pyscf = True
    except ImportError:
        has_pyscf = False
        logger.warning("pyscf not installed — running timing/basis stats only.")

    run_energy = has_pyscf and not args.no_pyscf

    xyz_files = sorted(args.molecules.glob("*.xyz"))
    if args.n:
        xyz_files = xyz_files[: args.n]

    if not xyz_files:
        logger.error("No .xyz files found in '%s'.", args.molecules)
        sys.exit(1)

    logger.info("Benchmarking %d molecules…", len(xyz_files))

    from basisrec import recommend

    all_results: List[BenchmarkResult] = []

    for xyz_path in xyz_files:
        logger.info("  %s", xyz_path.name)

        # Reference energy (cc-pVTZ)
        ref_energy = run_pyscf_energy(xyz_path, "cc-pVTZ") if run_energy else None

        # Strategy 1: BasisRec rule-only
        from unittest.mock import patch
        with patch("basisrec.orca_writer.fetch_basis_block_orca", return_value="S 1 1.00\n 1.0 1.0"):
            t0 = time.perf_counter()
            result_rules = recommend(str(xyz_path), use_gnn=False)
            t_rules = time.perf_counter() - t0

        basis_map_rules = {
            a.symbol: r.basis_name
            for a, r in zip(result_rules.molecule.atoms, result_rules.recommendations)
        }
        all_results.append(BenchmarkResult(
            strategy="BasisRec (rules)",
            molecule=xyz_path.stem,
            n_atoms=result_rules.molecule.n_atoms,
            energy_hartree=run_pyscf_energy(xyz_path, basis_map_rules) if run_energy else None,
            ref_energy=ref_energy,
            mae_kcal=None,
            wall_time_s=t_rules,
            unique_bases=result_rules.unique_basis_sets(),
        ))

        # Strategy 2-4: uniform basis sets
        for strat_name, basis in [
            ("Uniform def2-SVP",  "def2-SVP"),
            ("Uniform 6-31G*",   "6-31G*"),
            ("Uniform def2-TZVP", "def2-TZVP"),
        ]:
            t0 = time.perf_counter()
            energy = run_pyscf_energy(xyz_path, basis) if run_energy else None
            t_strat = time.perf_counter() - t0
            mae = abs(energy - ref_energy) * HARTREE_TO_KCAL if (energy and ref_energy) else None
            all_results.append(BenchmarkResult(
                strategy=strat_name,
                molecule=xyz_path.stem,
                n_atoms=result_rules.molecule.n_atoms,
                energy_hartree=energy,
                ref_energy=ref_energy,
                mae_kcal=mae,
                wall_time_s=t_strat,
                unique_bases=[basis],
            ))

    # Write CSV
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as fh:
        fh.write("strategy,molecule,n_atoms,energy_hartree,ref_energy,mae_kcal,"
                 "wall_time_s,unique_bases\n")
        for r in all_results:
            fh.write(
                f"{r.strategy},{r.molecule},{r.n_atoms},"
                f"{r.energy_hartree or ''},"
                f"{r.ref_energy or ''},"
                f"{r.mae_kcal or ''},"
                f"{r.wall_time_s:.4f},"
                f"{';'.join(r.unique_bases)}\n"
            )

    logger.info("Benchmark results written to '%s'.", args.output)

    # Print summary table
    strategies = list(dict.fromkeys(r.strategy for r in all_results))
    print("\n--- Summary ---")
    print(f"{'Strategy':<25} {'Molecules':<10} {'Mean MAE (kcal/mol)':<22} {'Mean time (s)'}")
    print("-" * 72)
    for strat in strategies:
        rows = [r for r in all_results if r.strategy == strat]
        maes = [r.mae_kcal for r in rows if r.mae_kcal is not None]
        times = [r.wall_time_s for r in rows]
        mean_mae = sum(maes) / len(maes) if maes else float("nan")
        mean_time = sum(times) / len(times) if times else float("nan")
        print(f"{strat:<25} {len(rows):<10} {mean_mae:<22.3f} {mean_time:.4f}")


if __name__ == "__main__":
    main()
