"""Generate reliability plots for backtest diagnostics."""
from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import typer

app = typer.Typer(help="Calibration plotting utilities")


@app.command()
def reliability(
    inputs: List[Path] = typer.Argument(..., help="CSV files with predicted vs realised columns."),
    output_dir: Path = typer.Option(
        Path("diagnostics/plots"),
        help="Directory where plots will be saved.",
    ),
    bins: int = typer.Option(10, min=2, help="Number of quantile bins used for calibration."),
) -> None:
    """Plot predicted vs realised log returns for each diagnostics file."""

    output_dir.mkdir(parents=True, exist_ok=True)

    for csv_path in inputs:
        df = pd.read_csv(csv_path)
        if {"predicted_edge", "realized_log_return"} - set(df.columns):
            typer.echo(f"Skipping {csv_path}: required columns missing.")
            continue

        clean = df[["predicted_edge", "realized_log_return"]].dropna()
        if clean.empty:
            typer.echo(f"Skipping {csv_path}: no rows after dropping NaNs.")
            continue

        try:
            labels = pd.qcut(clean["predicted_edge"], bins, labels=False, duplicates="drop")
        except ValueError:
            typer.echo(f"Skipping {csv_path}: unable to form {bins} quantile bins.")
            continue

        grouped = clean.groupby(labels).agg(
            predicted_mean=("predicted_edge", "mean"),
            realized_mean=("realized_log_return", "mean"),
            count=("predicted_edge", "size"),
        )
        if grouped.empty:
            typer.echo(f"Skipping {csv_path}: grouping produced no data.")
            continue

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(grouped["predicted_mean"], grouped["realized_mean"], marker="o", label="Observed")
        min_edge = grouped["predicted_mean"].min()
        max_edge = grouped["predicted_mean"].max()
        ax.plot([min_edge, max_edge], [min_edge, max_edge], linestyle="--", color="gray", label="Ideal")
        ax.set_title(csv_path.name)
        ax.set_xlabel("Predicted log return")
        ax.set_ylabel("Realized log return")
        ax.grid(True, alpha=0.3)
        ax.legend()
        for _, row in grouped.iterrows():
            ax.annotate(int(row["count"]), (row["predicted_mean"], row["realized_mean"]))

        output_path = output_dir / f"{csv_path.stem}_reliability.png"
        fig.tight_layout()
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        typer.echo(f"Saved {output_path}")


if __name__ == "__main__":  # pragma: no cover
    app()
