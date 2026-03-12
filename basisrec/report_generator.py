"""Human-readable per-atom basis set reasoning report generator.

Produces both a plain-text report and a Rich-formatted terminal
table. The report is included in the ORCA input file header and
printed to stdout when using the CLI.
"""

from __future__ import annotations

from typing import List

from .models import BasisRecommendation, Molecule


def generate_report(
    molecule: Molecule,
    recommendations: List[BasisRecommendation],
) -> str:
    """Generate a plain-text reasoning report.

    Parameters
    ----------
    molecule:
        The parsed molecule.
    recommendations:
        Per-atom recommendations from the decision engine.

    Returns
    -------
    str
        Multi-line report string, suitable for embedding in
        the ORCA input header or saving to a ``.txt`` file.
    """
    lines = [
        "BasisRec — Per-Atom Basis Set Recommendation Report",
        "=" * 54,
        f"Molecule : {molecule.title or 'untitled'}",
        f"Atoms    : {molecule.n_atoms}",
        f"Elements : {', '.join(molecule.unique_elements)}",
        "",
        f"{'Idx':<5} {'Sym':<4} {'Basis':<22} {'Tier':<14} {'Conf':<8} {'Src':<5}",
        "-" * 62,
    ]

    for rec in recommendations:
        lines.append(
            f"{rec.atom_index:<5} {rec.symbol:<4} {rec.basis_name:<22} "
            f"{rec.tier.value:<14} {rec.confidence:<8.0%} {rec.source}"
        )

    lines += ["", "Reasoning:", "-" * 54]
    for rec in recommendations:
        lines.append(f"[{rec.atom_index}] {rec.reason}")

    unique_bases = sorted(set(r.basis_name for r in recommendations))
    lines += [
        "",
        "Unique basis sets used:",
        "  " + ", ".join(unique_bases),
        "",
        "Note: Full Gaussian exponents embedded in ORCA input file.",
        "      Fetched from the Basis Set Exchange (BSE).",
    ]

    return "\n".join(lines)


def print_rich_table(
    molecule: Molecule,
    recommendations: List[BasisRecommendation],
) -> None:
    """Print a Rich-formatted table to stdout.

    Degrades gracefully to plain print if Rich is not installed.
    """
    try:
        from rich import box
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(
            title=f"BasisRec — {molecule.title or 'untitled'} ({molecule.n_atoms} atoms)",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold",
        )
        table.add_column("Idx",    style="dim",      width=5)
        table.add_column("Sym",    style="bold cyan", width=4)
        table.add_column("Basis",                    width=22)
        table.add_column("Tier",   style="dim",      width=14)
        table.add_column("Conf",                     width=7)
        table.add_column("Src",    style="dim",      width=5)
        table.add_column("Reason", style="dim",      min_width=30)

        for rec in recommendations:
            conf_str = f"{rec.confidence:.0%}"
            conf_style = (
                "green" if rec.confidence >= 0.85
                else "yellow" if rec.confidence >= 0.65
                else "red"
            )
            table.add_row(
                str(rec.atom_index),
                rec.symbol,
                rec.basis_name,
                rec.tier.value,
                f"[{conf_style}]{conf_str}[/{conf_style}]",
                rec.source,
                rec.reason[:60] + ("…" if len(rec.reason) > 60 else ""),
            )

        console.print(table)

    except ImportError:
        # Plain fallback
        print(generate_report(molecule, recommendations))
