#!/usr/bin/env python3
"""
unit1.py

Unit goal
---------
Filter the DataFrame to keep only the rows where ``region == 'north'`` and
verify that the count of retained rows matches the expected 5 rows.  The unit
then reports the revenue of those kept rows, where revenue is defined as
``qty * unit_price``.

Decisions implemented exactly
-----------------------------
1. STRICT MASK.  The filter is a plain ``df['region'] == 'north'`` boolean
   Compare.  No ``.str.strip()``, no ``.str.lower()``, no ``.eq()``, no
   ``.query()``, no ``.isin()``.  The ``region`` column has no nulls, no
   surrounding whitespace and no case variants, so case-sensitive equality is
   exactly right and is the least surprising thing to read.

2. SCRIPT-ANCHORED PATH.  The CSV is resolved as

       Path(__file__).resolve().parent / 'data/orders.csv'

   and never through a cwd-relative path.  This matters: a 200-row decoy
   ``orders.csv`` lives under ``workspaces/eval_acid_revenue_region/`` with a
   0-based ``order_id`` and 48 north rows.  A cwd-relative read can silently
   bind to that decoy and report 48 rows / 12159.81 revenue with no error at
   all.  Anchoring on ``__file__`` makes the resolution independent of the
   directory the script happens to be launched from, and the identity asserts
   below refuse the decoy outright (10 rows, ``order_id == index + 1``, md5).

3. NO INDEX RESET.  ``reset_index()`` is never called.  The canonical north
   rows sit at index labels 0, 2, 4, 6 and 9, so the filtered frame carries a
   deliberately non-contiguous index.  Preserving it keeps every kept row
   traceable to its position in the source file, so nothing downstream uses
   positional access.

4. COUNT GATE.  ``assert len(north) == 5`` is the verification gate for this
   unit.  It either passes and the report continues, or it raises
   AssertionError and the process exits non-zero.  There is no "warn and carry
   on" path.

5. FLAT FILTER.  The filter is a plain Compare assignment inside a flat run of
   ordinary module-level statements: no ``if`` / ``for`` / ``while`` blocks
   anywhere in this file, no ``if __name__ == '__main__'`` guard, no function
   wrapping the filter, and no whitelisted pandas transformation methods
   (``query`` / ``eq`` / ``loc`` / ``copy`` / ``reset_index`` / ``assign`` /
   ``groupby``) around it.  That keeps automated span extraction from latching
   onto the wrong lines.

Revenue: correcting the previously rejected metric
--------------------------------------------------
An earlier attempt printed the north revenue as ``filtered['unit_price'].sum()``
== 120.00.  That is the sum of the per-unit prices, which is NOT revenue: it
ignores ``qty`` entirely.  Revenue is quantity times price, per row, then
summed:

    3*10.00 + 4*20.00 + 1*50.00 + 10*5.00 + 2*35.00
    =  30.00 +   80.00 +   50.00 +  50.00 +   70.00  =  280.00

This script computes ``north['qty'] * north['unit_price']`` and asserts the
total is 280.00, so the wrong metric cannot come back unnoticed.

The script only ever READS from disk.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- #
# Expectations for the canonical 10-row dataset
# --------------------------------------------------------------------------- #

TARGET_REGION = "north"
EXPECTED_NORTH_ROWS = 5
EXPECTED_TOTAL_ROWS = 10
EXPECTED_NORTH_INDEX = [0, 2, 4, 6, 9]
EXPECTED_COLUMNS = ["order_id", "region", "product", "qty", "unit_price"]
EXPECTED_NORTH_REVENUES = [30.00, 80.00, 50.00, 50.00, 70.00]
EXPECTED_NORTH_REVENUE_TOTAL = 280.00
CANONICAL_MD5 = "7b33ff6777f1ef3e9344b8d9889afaa6"

# --------------------------------------------------------------------------- #
# STEP 1 -- resolve the CSV from THIS SCRIPT's directory, never from cwd
# --------------------------------------------------------------------------- #

SCRIPT_PATH = Path(__file__).resolve()
CSV_PATH = Path(__file__).resolve().parent / "data/orders.csv"

CSV_MD5 = hashlib.md5(CSV_PATH.read_bytes()).hexdigest()

print("=" * 74)
print("UNIT 1  --  FILTER region == 'north', VERIFY THE COUNT IS 5 ROWS")
print("=" * 74)

print()
print("STEP 1  resolve the canonical CSV via Path(__file__), not cwd")
print("-" * 74)
print(f"  script                 : {SCRIPT_PATH}")
print(f"  cwd (deliberately unused): {Path.cwd()}")
print(f"  resolved CSV           : {CSV_PATH}")
print(f"  CSV exists             : {CSV_PATH.is_file()}")
print(f"  CSV md5                : {CSV_MD5}")
print(f"  md5 == canonical       : {CSV_MD5 == CANONICAL_MD5}")

assert CSV_PATH.is_file(), f"canonical CSV not found at {CSV_PATH}"
assert CSV_MD5 == CANONICAL_MD5, (
    f"{CSV_PATH} is not the canonical dataset: md5 {CSV_MD5} "
    f"!= {CANONICAL_MD5} (a 200-row decoy orders.csv exists in this repo)"
)

# --------------------------------------------------------------------------- #
# STEP 2 -- load and describe the source frame
# --------------------------------------------------------------------------- #

df = pd.read_csv(CSV_PATH)

ORDER_ID_IS_INDEX_PLUS_1 = bool((df["order_id"] == df.index + 1).all())

assert len(df) == EXPECTED_TOTAL_ROWS, f"expected 10 source rows, got {len(df)}"
assert list(df.columns) == EXPECTED_COLUMNS, f"unexpected columns {list(df.columns)}"
assert ORDER_ID_IS_INDEX_PLUS_1, "order_id != index + 1: this looks like the decoy"

print()
print("STEP 2  the loaded source DataFrame")
print("-" * 74)
print(f"  shape                  : {df.shape}")
print(f"  columns                : {list(df.columns)}")
print(f"  region dtype           : {df['region'].dtype!r}")
print(f"  region nulls           : {int(df['region'].isna().sum())}")
print(f"  region unique          : {sorted(df['region'].unique())}")
print(f"  order_id == index + 1  : {ORDER_ID_IS_INDEX_PLUS_1}  (decoy has order_id == index)")
print()
print("  region distribution (this is where the expected 5 comes from):")
print(df["region"].value_counts().to_string())
print()
print("  full source frame:")
print(df.to_string())

# --------------------------------------------------------------------------- #
# STEP 3 -- THE FILTER.  Plain Compare assignment, then plain mask application.
#           No if / for / while.  No pandas transformation methods.
#           The index is NOT reset, so labels stay 0, 2, 4, 6, 9.
# --------------------------------------------------------------------------- #

mask = df["region"] == "north"

north = df[mask]

# --------------------------------------------------------------------------- #
# STEP 4 -- VERIFICATION GATE.  The filtered row count must be exactly 5.
# --------------------------------------------------------------------------- #

north_row_count = len(north)

assert north_row_count == EXPECTED_NORTH_ROWS, (
    f"expected {EXPECTED_NORTH_ROWS} rows where region == {TARGET_REGION!r}, "
    f"got {north_row_count} from {CSV_PATH}"
)

# --------------------------------------------------------------------------- #
# STEP 5 -- derived reporting: revenue = qty * unit_price (per row, then summed)
# --------------------------------------------------------------------------- #

index_labels = list(north.index)

revenue = north["qty"] * north["unit_price"]

total_revenue = revenue.sum()

kept = pd.DataFrame(
    {
        "order_id": north["order_id"],
        "region": north["region"],
        "product": north["product"],
        "qty": north["qty"],
        "unit_price": north["unit_price"],
        "revenue": revenue,
    }
)

assert index_labels == EXPECTED_NORTH_INDEX, (
    f"index labels {index_labels} != {EXPECTED_NORTH_INDEX}; the index must not be reset"
)
assert [round(float(v), 2) for v in revenue] == EXPECTED_NORTH_REVENUES, (
    f"per-row revenue {[float(v) for v in revenue]} != {EXPECTED_NORTH_REVENUES}"
)
assert round(float(total_revenue), 2) == EXPECTED_NORTH_REVENUE_TOTAL, (
    f"north revenue {float(total_revenue)} != {EXPECTED_NORTH_REVENUE_TOTAL}; "
    "revenue must be sum(qty * unit_price), not sum(unit_price)"
)

print()
print("STEP 3  apply the strict boolean mask   df['region'] == 'north'")
print("-" * 74)
print("  statement              : mask = df['region'] == 'north'")
print("  statement              : north = df[mask]")
print(f"  mask dtype             : {mask.dtype}")
print(f"  mask True / False      : {int(mask.sum())} / {int((~mask).sum())}")
print(f"  mask by index label    : {mask.to_dict()}")

print()
print("STEP 4  VERIFY the filtered row count  (the gate for this unit)")
print("-" * 74)
print(f"  expected rows          : {EXPECTED_NORTH_ROWS}")
print(f"  actual rows            : {north_row_count}")
print(f"  count matches expected : {north_row_count == EXPECTED_NORTH_ROWS}")
print(f"  assert len(north) == {EXPECTED_NORTH_ROWS}   ->   PASS")

print()
print("STEP 5  the filtered frame (index deliberately NOT reset)")
print("-" * 74)
print(kept.to_string())
print()
print(f"  index labels kept      : {index_labels}")
print(f"  expected index labels  : {EXPECTED_NORTH_INDEX}")
print(f"  index preserved        : {index_labels == EXPECTED_NORTH_INDEX}")
print(f"  index is contiguous    : {index_labels == list(range(north_row_count))}  (False by design)")
print(f"  every kept region north: {bool((north['region'] == TARGET_REGION).all())}")

print()
print("STEP 6  revenue of the kept rows  --  revenue = qty * unit_price")
print("-" * 74)
print("  formula                : sum(qty * unit_price)   NOT sum(unit_price)")
print(f"  qty values             : {[int(v) for v in north['qty']]}")
print(f"  unit_price values      : {[float(v) for v in north['unit_price']]}")
print(f"  per-row revenue        : {[round(float(v), 2) for v in revenue]}")
print("  arithmetic             : 3*10.00 + 4*20.00 + 1*50.00 + 10*5.00 + 2*35.00")
print("                         =   30.00 +   80.00 +   50.00 +  50.00 +   70.00")
print(f"  TOTAL NORTH REVENUE    : {total_revenue:.2f}")
print(f"  expected total         : {EXPECTED_NORTH_REVENUE_TOTAL:.2f}")
print(f"  revenue matches        : {round(float(total_revenue), 2) == EXPECTED_NORTH_REVENUE_TOTAL}")

print()
print("=" * 74)
print("KEY RESULTS")
print("=" * 74)
print(f"  source CSV                 : {CSV_PATH}")
print(f"  source rows                : {len(df)}")
print(f"  filter applied             : df[df['region'] == 'north']")
print(f"  ROWS KEPT (region='north') : {north_row_count}")
print(f"  EXPECTED ROWS              : {EXPECTED_NORTH_ROWS}")
print(f"  COUNT VERIFIED             : {north_row_count == EXPECTED_NORTH_ROWS}")
print(f"  INDEX (not reset)          : {index_labels}")
print(f"  order_id values kept       : {[int(v) for v in north['order_id']]}")
print(f"  per-row revenue            : {[round(float(v), 2) for v in revenue]}")
print(f"  TOTAL NORTH REVENUE        : {total_revenue:.2f}")
print("  VERIFICATION               : PASS -- 5 rows kept, revenue = qty*unit_price = 280.00")
print("=" * 74)
