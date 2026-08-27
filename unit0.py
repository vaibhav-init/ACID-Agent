#!/usr/bin/env python3
"""
unit0.py -- Load orders.csv, verify schema + data expectations, report revenue by region.

Path resolution is anchored to the REPO ROOT (the directory that contains
``data/orders.csv``), discovered by walking upward from this script's own
location. This keeps the load independent of the current working directory and
guarantees we read the canonical 10-row file rather than the 200-row decoy at
``workspaces/eval_acid_revenue_region/orders.csv``.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- #
# Expectations (canonical data/orders.csv)
# --------------------------------------------------------------------------- #

EXPECTED_COLUMNS = {"order_id", "region", "product", "qty", "unit_price"}

# Column order as stated in the unit goal. The file on disk uses a DIFFERENT
# order (order_id, region, product, qty, unit_price), so schema verification
# uses set equality / reindex -- never positional comparison.
GOAL_COLUMN_ORDER = ["order_id", "region", "qty", "unit_price", "product"]

EXPECTED_DTYPES = {
    "order_id": "int64",
    "region": "str",
    "product": "str",
    "qty": "int64",
    "unit_price": "float64",
}

EXPECTED_ROWS = 10
EXPECTED_REGIONS = {"north", "east", "south", "west"}
EXPECTED_PRODUCTS = {"doohickey", "gadget", "widget"}
CANONICAL_MD5 = "7b33ff6777f1ef3e9344b8d9889afaa6"

RELATIVE_DATA_PATH = Path("data") / "orders.csv"


# --------------------------------------------------------------------------- #
# Path resolution -- anchored to repo root, not to cwd
# --------------------------------------------------------------------------- #

def resolve_orders_path() -> Path:
    """Return the absolute path to the canonical ``data/orders.csv``.

    Starts at this script's directory and walks upward through its parents,
    returning the first ``<ancestor>/data/orders.csv`` that exists. Anchoring on
    ``__file__`` (never on ``Path.cwd()`` and never on a bare filename) means the
    script loads the same file no matter where it is invoked from, and walking up
    to the repo root means it still works if the script itself lives in a
    subdirectory such as ``workspaces/<name>/``.
    """
    script_dir = Path(__file__).resolve().parent

    # script_dir first, then each ancestor up to the filesystem root.
    for anchor in (script_dir, *script_dir.parents):
        candidate = anchor / RELATIVE_DATA_PATH
        if candidate.is_file():
            return candidate.resolve()

    searched = "\n".join(
        f"  - {anchor / RELATIVE_DATA_PATH}"
        for anchor in (script_dir, *script_dir.parents)
    )
    raise FileNotFoundError(
        f"Could not locate {RELATIVE_DATA_PATH} anchored at the repo root.\n"
        f"Searched (from script location {script_dir} upward):\n{searched}"
    )


def md5_of(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# Reporting helpers
# --------------------------------------------------------------------------- #

CHECKS: list[tuple[str, bool, str]] = []


def check(label: str, passed: bool, detail: str) -> None:
    """Record a check, print it, and hard-fail on violation."""
    status = "PASS" if passed else "FAIL"
    CHECKS.append((label, passed, detail))
    print(f"  [{status}] {label}: {detail}")
    if not passed:
        raise AssertionError(f"{label} -> {detail}")


def section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    print("=" * 72)
    print("UNIT 0 -- LOAD orders.csv, VERIFY SCHEMA & DATA, REVENUE BY REGION")
    print("=" * 72)

    # ---------------------------------------------------------------- load
    section("1. FILE RESOLUTION (absolute path anchored at repo root)")
    script_path = Path(__file__).resolve()
    csv_path = resolve_orders_path()
    repo_root = csv_path.parent.parent

    print(f"  script         : {script_path}")
    print(f"  repo root      : {repo_root}")
    print(f"  resolved CSV   : {csv_path}")
    print(f"  absolute?      : {csv_path.is_absolute()}")
    print(f"  exists?        : {csv_path.is_file()}")
    print(f"  size (bytes)   : {csv_path.stat().st_size}")

    digest = md5_of(csv_path)
    check(
        "canonical file identity (md5)",
        digest == CANONICAL_MD5,
        f"{digest} == {CANONICAL_MD5} (not the 200-row decoy)",
    )

    df = pd.read_csv(csv_path)
    print(f"\n  loaded shape   : {df.shape}")

    # -------------------------------------------------------------- schema
    section("2. SCHEMA VERIFICATION (set equality, order-independent)")
    actual_columns = set(df.columns)
    print(f"  actual column order   : {list(df.columns)}")
    print(f"  goal-stated order     : {GOAL_COLUMN_ORDER}")
    print(f"  actual column set     : {sorted(actual_columns)}")
    print(f"  expected column set   : {sorted(EXPECTED_COLUMNS)}")
    print(f"  column ORDER matches? : {list(df.columns) == GOAL_COLUMN_ORDER}"
          "  <- order differs; irrelevant to schema validity")

    check(
        "column set equality",
        actual_columns == EXPECTED_COLUMNS,
        f"{sorted(actual_columns)} == {sorted(EXPECTED_COLUMNS)}",
    )
    check(
        "no missing / no extra columns",
        not (EXPECTED_COLUMNS - actual_columns) and not (actual_columns - EXPECTED_COLUMNS),
        f"missing={sorted(EXPECTED_COLUMNS - actual_columns)} "
        f"extra={sorted(actual_columns - EXPECTED_COLUMNS)}",
    )

    # Reindex to the goal-stated order to prove order-independent access.
    df_goal_order = df.reindex(columns=GOAL_COLUMN_ORDER)
    check(
        "reindex to goal column order",
        list(df_goal_order.columns) == GOAL_COLUMN_ORDER
        and not df_goal_order.isna().any().any(),
        f"{list(df_goal_order.columns)} with 0 NaN introduced",
    )

    # Bracket notation only: df.product is DataFrame.product (a bound method).
    product_series = df["product"]
    check(
        "bracket access df['product'] returns a Series",
        isinstance(product_series, pd.Series),
        f"type(df['product'])={type(product_series).__name__}, "
        f"type(df.product)={type(df.product).__name__} (bound method -> never used)",
    )

    # -------------------------------------------------------------- dtypes
    section("3. DTYPE ASSERTIONS (df.dtypes.astype(str).to_dict())")
    actual_dtypes = df.dtypes.astype(str).to_dict()
    for col in GOAL_COLUMN_ORDER:
        print(f"  {col:<11}: {actual_dtypes[col]:<9} (repr: {df[col].dtype!r})")
    check(
        "dtype mapping exact match",
        actual_dtypes == EXPECTED_DTYPES,
        f"{actual_dtypes} == {EXPECTED_DTYPES}",
    )
    print("  note: region/product are StringDtype(storage='python') -> "
          "'str', so `dtype == object` would be False.")

    # ---------------------------------------------------------------- data
    section("4. DATA VALIDATION")
    regions = set(df["region"].unique())
    products = set(product_series.unique())
    order_ids = df["order_id"]

    check("row count", len(df) == EXPECTED_ROWS, f"{len(df)} == {EXPECTED_ROWS}")
    check("column count", df.shape[1] == len(EXPECTED_COLUMNS),
          f"{df.shape[1]} == {len(EXPECTED_COLUMNS)}")
    check("unique region count", len(regions) == 4, f"{len(regions)} == 4")
    check("region values", regions == EXPECTED_REGIONS,
          f"{sorted(regions)} == {sorted(EXPECTED_REGIONS)}")
    check("unique product count", len(products) == 3, f"{len(products)} == 3")
    check("product values", products == EXPECTED_PRODUCTS,
          f"{sorted(products)} == {sorted(EXPECTED_PRODUCTS)}")
    check("order_id contiguous 1-10",
          order_ids.tolist() == list(range(1, EXPECTED_ROWS + 1)),
          f"min={order_ids.min()} max={order_ids.max()} -> {order_ids.tolist()}")
    check("order_id unique", order_ids.is_unique and order_ids.duplicated().sum() == 0,
          f"{order_ids.nunique()} unique / {len(order_ids)} rows, 0 duplicates")
    check("zero nulls", int(df.isna().sum().sum()) == 0,
          f"total nulls = {int(df.isna().sum().sum())} "
          f"(per column: {df.isna().sum().to_dict()})")
    check("zero duplicate rows", int(df.duplicated().sum()) == 0,
          f"duplicated rows = {int(df.duplicated().sum())}")
    check("qty positive int range",
          bool(df['qty'].min() >= 1) and bool(df['qty'].max() <= 10),
          f"qty range = {df['qty'].min()}..{df['qty'].max()}")
    check("unit_price float range",
          bool(df['unit_price'].min() >= 5.0) and bool(df['unit_price'].max() <= 50.0),
          f"unit_price range = {df['unit_price'].min()}..{df['unit_price'].max()}")

    # -------------------------------------------------------------- content
    section("5. LOADED DATA (reindexed to goal column order)")
    print(df_goal_order.to_string(index=False))

    section("6. REGION DISTRIBUTION (row counts)")
    region_counts = df["region"].value_counts()
    for region, count in region_counts.items():
        print(f"  {region:<7}: {count}")

    section("7. PRODUCT DISTRIBUTION (row counts)")
    for product, count in product_series.value_counts().items():
        print(f"  {product:<10}: {count}")

    # -------------------------------------------------------------- revenue
    section("8. REVENUE BY REGION  (revenue = qty * unit_price)")
    df = df.assign(revenue=df["qty"] * df["unit_price"])
    revenue_by_region = (
        df.groupby("region", observed=True)["revenue"].sum().sort_values(ascending=False)
    )
    for region, revenue in revenue_by_region.items():
        print(f"  {region:<7}: {revenue:>8.1f}")
    grand_total = float(revenue_by_region.sum())
    print(f"  {'-' * 18}")
    print(f"  {'TOTAL':<7}: {grand_total:>8.1f}")

    check("grand total revenue", abs(grand_total - 466.0) < 1e-9,
          f"{grand_total} == 466.0")
    check("revenue by region matches expected",
          {k: round(float(v), 2) for k, v in revenue_by_region.items()}
          == {"north": 280.0, "south": 80.0, "east": 70.0, "west": 36.0},
          f"{ {k: float(v) for k, v in revenue_by_region.items()} }")

    # --------------------------------------------------------------- answer
    top_region = str(revenue_by_region.index[0])
    top_revenue = float(revenue_by_region.iloc[0])
    north_revenue = float(revenue_by_region.loc["north"])

    print()
    print("=" * 72)
    print("ANSWER")
    print("=" * 72)
    print(f"  Top region by revenue : {top_region}")
    print(f"  north revenue         : {north_revenue}")
    print(f"  Revenue by region     : "
          f"north={revenue_by_region.loc['north']:.1f}, "
          f"south={revenue_by_region.loc['south']:.1f}, "
          f"east={revenue_by_region.loc['east']:.1f}, "
          f"west={revenue_by_region.loc['west']:.1f}")
    print(f"  Grand total revenue   : {grand_total}")
    print(f"  >>> {top_region.upper()} leads with {top_revenue} of {grand_total} total revenue.")

    print()
    print("=" * 72)
    passed = sum(1 for _, ok, _ in CHECKS if ok)
    print(f"SUMMARY: {passed}/{len(CHECKS)} checks passed")
    print(f"  source file : {csv_path}")
    print(f"  shape       : {df.shape[0]} rows x {len(EXPECTED_COLUMNS)} columns")
    print(f"  regions     : {len(regions)} unique -> {sorted(regions)}")
    print(f"  products    : {len(products)} unique -> {sorted(products)}")
    print(f"  north revenue = {north_revenue}  |  total = {grand_total}")
    print("UNIT 0 RESULT: ALL SCHEMA AND DATA CHECKS PASSED")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (AssertionError, FileNotFoundError) as exc:
        print(f"\nUNIT 0 RESULT: FAILED\n  {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
