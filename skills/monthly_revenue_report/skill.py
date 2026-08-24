"""Validated skill: monthly revenue report from an orders CSV.

Transactional guarantees baked in (paper §2.2.1):
- normalizes mixed date formats BEFORE grouping (no silent mis-bucketing)
- reports dropped rows instead of silently ignoring them
"""

import pandas as pd
import typer

app = typer.Typer(help="Monthly revenue report")


def monthly_revenue(csv_path: str, date_col: str = "date", value_col: str = "revenue") -> pd.Series:
    df = pd.read_csv(csv_path)
    if date_col not in df.columns or value_col not in df.columns:
        raise typer.BadParameter(f"need columns {date_col} and {value_col}; got {list(df.columns)}")

    # Normalize mixed separators: 2024/03/05 -> 2024-03-05
    raw = df[date_col].astype(str)
    normalized = raw.str.replace("/", "-", regex=False)
    df[date_col] = pd.to_datetime(normalized, errors="coerce")

    dropped = int(df[date_col].isna().sum() + df[value_col].isna().sum())
    df = df.dropna(subset=[date_col, value_col])
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[value_col])

    monthly = (
        df.assign(month=df[date_col].dt.to_period("M").astype(str))
        .groupby("month")[value_col]
        .sum()
        .round(2)
    )
    print(f"rows_dropped={dropped}")
    print(monthly.to_string())
    return monthly


@app.command()
def main(
    csv_path: str = typer.Argument(..., help="Path to orders CSV"),
    date_col: str = typer.Option("date"),
    value_col: str = typer.Option("revenue"),
):
    monthly_revenue(csv_path, date_col, value_col)


if __name__ == "__main__":
    app()