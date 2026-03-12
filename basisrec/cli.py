"""Command-line interface for BasisRec."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from .api import recommend
from .exceptions import BasisRecError
from .orca_writer import DEFAULT_MAXCORE, DEFAULT_NPROCS
from .report_generator import print_rich_table


@click.command()
@click.argument("xyz_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--output", "-o", type=click.Path(dir_okay=False, path_type=Path), default=None,
              help="Write ORCA input to this file (default: <stem>.inp).")
@click.option("--accuracy", "-a",
              type=click.Choice(["fast", "balanced", "accurate", "benchmark"], case_sensitive=False),
              default="balanced", show_default=True,
              help="Accuracy level: fast=PBE screening, balanced=B3LYP production, "
                   "accurate=wB97X-D3, benchmark=DLPNO-CCSD(T).")
@click.option("--nprocs", "-n", default=DEFAULT_NPROCS, show_default=True, type=int,
              help="Number of CPU cores.")
@click.option("--memory", "-m", default=8, show_default=True, type=int,
              help="Total RAM in GB (memory per core is computed automatically).")
@click.option("--charge", "-c", default=0, show_default=True, type=int,
              help="Total molecular charge.")
@click.option("--multiplicity", "-s", default=1, show_default=True, type=int,
              help="Spin multiplicity.")
@click.option("--solvent", default="", show_default=False,
              help="Implicit solvent (SMD): water, dmso, methanol, etc. Default: gas phase.")
@click.option("--excited-states", is_flag=True, default=False,
              help="Recommend a functional suitable for TD-DFT excited states (CAM-B3LYP).")
@click.option("--dispersion", is_flag=True, default=False,
              help="Dispersion-dominated system: use D4 correction (stacking, vdW complexes).")
@click.option("--no-gnn", is_flag=True, default=False,
              help="Disable GNN; use rule engine only for basis selection.")
@click.option("--model-path", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default=None, help="Custom GNN weights file.")
@click.option("--print-orca", is_flag=True, default=False,
              help="Print the full ORCA input to stdout.")
@click.option("--print-method", is_flag=True, default=False,
              help="Print the method recommendation reasoning.")
@click.version_option(package_name="basisrec")
def main(
    xyz_file, output, accuracy, nprocs, memory, charge, multiplicity,
    solvent, excited_states, dispersion, no_gnn, model_path, print_orca, print_method,
):
    """Recommend per-atom basis sets and optimal DFT method for XYZ_FILE.

    Generates a complete, copy-paste-ready ORCA input with full Gaussian
    exponents, method keywords, memory/parallelism settings, and reasoning.

    \b
    Examples:
        basisrec water.xyz
        basisrec ferrocene.xyz --accuracy accurate --nprocs 16 --memory 32
        basisrec drug.xyz --solvent water --output drug.inp
        basisrec radical.xyz --multiplicity 2 --charge -1
        basisrec pi_stack.xyz --dispersion
        basisrec chromophore.xyz --excited-states
    """
    if output is None:
        output = xyz_file.with_suffix(".inp")

    try:
        result = recommend(
            xyz_path=xyz_file,
            accuracy=accuracy,
            nprocs=nprocs,
            memory_gb=memory,
            charge=charge,
            multiplicity=multiplicity,
            solvent=solvent,
            excited_states=excited_states,
            dispersion_dominated=dispersion,
            use_gnn=not no_gnn,
            model_path=model_path,
            output_path=output,
        )
    except BasisRecError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    except FileNotFoundError as exc:
        click.echo(f"File not found: {exc}", err=True)
        sys.exit(1)

    # Print basis table
    print_rich_table(result.molecule, result.recommendations)

    # Method summary
    if print_method:
        click.echo("\n" + result.method_summary)
    else:
        # Always show the one-line method summary
        for rec in result.recommendations[:1]:
            pass
        from .method_recommender import recommend_method
        click.echo(f"\nMethod : {result.orca_input.splitlines()[len([l for l in result.orca_input.splitlines() if l.startswith('#')])]}")

    for warning in result.warnings:
        click.echo(f"\n⚠  {warning}", err=True)

    click.echo(f"\nORCA input written to: {output}")

    if print_orca:
        click.echo("\n" + "=" * 60)
        click.echo(result.orca_input)


if __name__ == "__main__":
    main()
