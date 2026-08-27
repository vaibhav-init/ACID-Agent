#!/usr/bin/env python3
"""
unit2.py

Unit goal
---------
Create a new column ``revenue``, computed as ``qty * unit_price``, for the
filtered north-region rows.

Decisions implemented exactly
-----------------------------
1. SCRIPT-ANCHORED PATH.  The CSV is resolved as

       Path(__file__).resolve().parent / 'data/orders.csv'

   which is the 272-byte / 10-row canonical file.  It is NOT the decoy at
   workspaces/eval_acid_revenue_region/orders.csv (5086 bytes, 200 rows, 48
   north rows, revenue sum 12159.81).  The decoy carries identical column
   names, so a wrong bind fails silently rather than loudly -- hence the md5
   pin, the 10-row assert and the ``order_id == index + 1`` guard below (the
   decoy is 0-based, order_id 0..199).

   The previous revision of this file used a walk-up resolver that scanned
   Path.cwd() and every parent for ``data/orders.csv``.  That resolver escapes
   the workspace: launched from workspaces/eval_acid_revenue_region/ it climbs
   out to /home/vaibhav/ACID-Agent/data/orders.csv, so the file that gets read
   depends on the launch directory rather than on the script.  Anchoring on
   __file__ removes cwd from the equation entirely.

2. NORTH ONLY.  A plain, case-sensitive ``df['region'] == 'north'`` mask,
   yielding exactly 5 of the 10 rows, at index labels [0, 2, 4, 6, 9].  No
   .str.strip(), no .str.lower(), no .eq(), no .isin(), no .query() -- the
   region column has no nulls, no padding and no case variants.

3. STANDALONE SERIES + pd.DataFrame({...}).  ``revenue`` is first computed as
   its own Series, ``north['qty'] * north['unit_price']``, and the output frame
   is then materialized with ``pd.DataFrame({...})``.  Direct assignment
   (``north['revenue'] = ...``) is deliberately NOT used.  Under pandas 3.0.5
   with Copy-on-Write that assignment happens to be harmless -- zero warnings,
   parent left unmutated -- but "harmless because of the active CoW default" is
   a weaker guarantee than "never writes to the filtered view at all".  The
   explicit dict construction states the column order and the provenance of
   every column in one expression, and keeps the source frame provably clean.

4. NO INDEX RESET.  ``reset_index(drop=True)`` is never called, so the result
   keeps the original labels [0, 2, 4, 6, 9] and every revenue value stays
   traceable to its row in the source file.  The previous revision reset the
   index to 0..4 "for positional access"; that discards the only link back to
   the source rows, and nothing here needs positional access.

5. NO ROUNDING.  ``round(v, 2)`` is not applied to any revenue value.  Each of
   3*10.00, 4*20.00, 1*50.00, 10*5.00 and 2*35.00 is an exact integer in
   binary float64 (30.0, 80.0, 50.0, 50.0, 70.0 -- verified via
   float.is_integer() and .hex() below), so there is no representation error to
   absorb and round() would be a pure no-op.  The total is therefore checked
   with exact equality, ``revenue.sum() == 280.0``, not with a rounded or
   tolerance-based compare.

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
NEW_COLUMN = "revenue"

EXPECTED_TOTAL_ROWS = 10
EXPECTED_NORTH_ROWS = 5
EXPECTED_NORTH_INDEX = [0, 2, 4, 6, 9]
EXPECTED_COLUMNS = ["order_id", "region", "product", "qty", "unit_price"]
EXPECTED_OUTPUT_COLUMNS = ["order_id", "region", "product", "qty", "unit_price", "revenue"]
EXPECTED_NORTH_ORDER_IDS = [1, 3, 5, 7, 10]
EXPECTED_NORTH_REVENUES = [30.0, 80.0, 50.0, 50.0, 70.0]
EXPECTED_REVENUE_TOTAL = 280.0

CANONICAL_MD5 = "7b33ff6777f1ef3e9344b8d9889afaa6"
CANONICAL_BYTES = 272
DECOY_REL = "workspaces/eval_acid_revenue_region/orders.csv"
DECOY_MD5 = "a7347e243a58abe2f4cd0d5fa0ec5f05"

# --------------------------------------------------------------------------- #
# STEP 1 -- resolve the CSV from THIS SCRIPT's directory, never from cwd
# --------------------------------------------------------------------------- #

SCRIPT_PATH = Path(__file__).resolve()
CSV_PATH = Path(__file__).resolve().parent / "data/orders.csv"

CSV_BYTES = CSV_PATH.read_bytes()
CSV_MD5 = hashlib.md5(CSV_BYTES).hexdigest()

DECOY_PATH = Path(__file__).resolve().parent / DECOY_REL
DECOY_EXISTS = DECOY_PATH.is_file()
DECOY_IS_OTHER_FILE = DECOY_PATH.resolve() != CSV_PATH.resolve()

print("=" * 78)
print("UNIT 2  --  ADD COLUMN 'revenue' = qty * unit_price TO THE NORTH ROWS")
print("=" * 78)

print()
print("STEP 1  resolve the canonical CSV via Path(__file__), not cwd, not walk-up")
print("-" * 78)
print(f"  script                   : {SCRIPT_PATH}")
print(f"  cwd (deliberately unused): {Path.cwd()}")
print(f"  resolved CSV             : {CSV_PATH}")
print(f"  CSV size / expected      : {len(CSV_BYTES)} / {CANONICAL_BYTES} bytes")
print(f"  CSV md5                  : {CSV_MD5}")
print(f"  md5 == canonical         : {CSV_MD5 == CANONICAL_MD5}")
print(f"  decoy on disk            : {DECOY_PATH}")
print(f"  decoy exists             : {DECOY_EXISTS}   (5086 bytes, 200 rows, 48 north)")
print(f"  decoy is a DIFFERENT file: {DECOY_IS_OTHER_FILE}   -> NOT read by this script")

assert CSV_PATH.is_file(), f"canonical CSV not found at {CSV_PATH}"
assert len(CSV_BYTES) == CANONICAL_BYTES, (
    f"{CSV_PATH} is {len(CSV_BYTES)} bytes, expected {CANONICAL_BYTES}"
)
assert CSV_MD5 == CANONICAL_MD5, (
    f"{CSV_PATH} is not the canonical dataset: md5 {CSV_MD5} != {CANONICAL_MD5} "
    f"(the 200-row decoy {DECOY_REL} has md5 {DECOY_MD5})"
)
assert CSV_MD5 != DECOY_MD5, "refusing to run against the 200-row decoy orders.csv"
assert DECOY_IS_OTHER_FILE, "resolved CSV must not be the decoy path"

# --------------------------------------------------------------------------- #
# STEP 2 -- load and identity-check the source frame
# --------------------------------------------------------------------------- #

df = pd.read_csv(CSV_PATH)

SOURCE_COLUMNS = list(df.columns)
ORDER_ID_IS_INDEX_PLUS_1 = bool((df["order_id"] == df.index + 1).all())

assert len(df) == EXPECTED_TOTAL_ROWS, f"expected 10 source rows, got {len(df)}"
assert SOURCE_COLUMNS == EXPECTED_COLUMNS, f"unexpected columns {SOURCE_COLUMNS}"
assert ORDER_ID_IS_INDEX_PLUS_1, "order_id != index + 1: this looks like the 0-based decoy"

print()
print("STEP 2  the loaded source DataFrame")
print("-" * 78)
print(f"  shape                    : {df.shape}")
print(f"  columns                  : {SOURCE_COLUMNS}")
print(f"  qty dtype / unit_price   : {df['qty'].dtype!r} / {df['unit_price'].dtype!r}")
print(f"  order_id range           : {int(df['order_id'].min())}..{int(df['order_id'].max())}")
print(f"  order_id == index + 1    : {ORDER_ID_IS_INDEX_PLUS_1}   (decoy has order_id == index)")
print(f"  region unique            : {sorted(df['region'].unique())}")
print(f"  region nulls             : {int(df['region'].isna().sum())}")
print()
print("  full source frame:")
print(df.to_string())

# --------------------------------------------------------------------------- #
# STEP 3 -- THE FILTER.  north only.  Plain mask, plain __getitem__.
#           No .copy(), no .assign(), no .loc, no .query(), no reset_index().
# --------------------------------------------------------------------------- #

mask = df["region"] == TARGET_REGION

north = df[mask]

north_row_count = len(north)
index_labels = list(north.index)

assert north_row_count == EXPECTED_NORTH_ROWS, (
    f"expected {EXPECTED_NORTH_ROWS} rows where region == {TARGET_REGION!r}, "
    f"got {north_row_count} from {CSV_PATH}"
)
assert bool((north["region"] == TARGET_REGION).all()), "a non-north row survived the filter"

print()
print("STEP 3  filter to region == 'north'  (5 of 10 rows)")
print("-" * 78)
print("  statement                : mask   = df['region'] == 'north'")
print("  statement                : north  = df[mask]")
print(f"  mask True / False        : {int(mask.sum())} / {int((~mask).sum())}")
print(f"  rows kept / expected     : {north_row_count} / {EXPECTED_NORTH_ROWS}")
print(f"  index labels kept        : {index_labels}")
print(f"  order_id values kept     : {[int(v) for v in north['order_id']]}")

# --------------------------------------------------------------------------- #
# STEP 4 -- revenue as a STANDALONE SERIES, then pd.DataFrame({...}).
#           Not `north['revenue'] = ...` on the filtered frame.
# --------------------------------------------------------------------------- #

revenue = north["qty"] * north["unit_price"]

north_with_revenue = pd.DataFrame(
    {
        "order_id": north["order_id"],
        "region": north["region"],
        "product": north["product"],
        "qty": north["qty"],
        "unit_price": north["unit_price"],
        "revenue": revenue,
    }
)

OUTPUT_COLUMNS = list(north_with_revenue.columns)
OUTPUT_LABELS = list(north_with_revenue.index)
REVENUE_VALUES = [float(v) for v in north_with_revenue[NEW_COLUMN]]
REVENUE_TOTAL = north_with_revenue[NEW_COLUMN].sum()

# --------------------------------------------------------------------------- #
# STEP 5 -- verification gates: labels preserved, parent untouched, exact floats
# --------------------------------------------------------------------------- #

INDEX_PRESERVED = OUTPUT_LABELS == EXPECTED_NORTH_INDEX
PARENT_UNMUTATED = list(df.columns) == EXPECTED_COLUMNS
ALL_EXACT_INTEGERS = [float(v).is_integer() for v in REVENUE_VALUES]
TOTAL_EXACTLY_280 = REVENUE_TOTAL == EXPECTED_REVENUE_TOTAL

assert INDEX_PRESERVED, (
    f"index labels {OUTPUT_LABELS} != {EXPECTED_NORTH_INDEX}: "
    "reset_index(drop=True) must not be called"
)
assert OUTPUT_LABELS != list(range(EXPECTED_NORTH_ROWS)), (
    "index looks reset to 0..4; the original labels must survive"
)
assert OUTPUT_COLUMNS == EXPECTED_OUTPUT_COLUMNS, f"unexpected output columns {OUTPUT_COLUMNS}"
assert PARENT_UNMUTATED, f"source frame was mutated: {list(df.columns)}"
assert NEW_COLUMN not in df.columns, "source frame must not gain a 'revenue' column"
assert [int(v) for v in north_with_revenue["order_id"]] == EXPECTED_NORTH_ORDER_IDS
assert REVENUE_VALUES == EXPECTED_NORTH_REVENUES, (
    f"per-row revenue {REVENUE_VALUES} != {EXPECTED_NORTH_REVENUES}"
)
assert all(ALL_EXACT_INTEGERS), "a revenue value is not an exact integer in binary float"
assert TOTAL_EXACTLY_280, (
    f"revenue total {REVENUE_TOTAL!r} is not exactly {EXPECTED_REVENUE_TOTAL}"
)

print()
print("STEP 4  build the 'revenue' column   revenue = qty * unit_price")
print("-" * 78)
print("  statement                : revenue = north['qty'] * north['unit_price']")
print("  statement                : north_with_revenue = pd.DataFrame({... 'revenue': revenue})")
print("  NOT used                 : north['revenue'] = ...   (no write into the filtered view)")
print("  NOT used                 : .copy() / .assign() / .loc / .query() / reset_index()")
print(f"  revenue Series dtype     : {revenue.dtype}")
print(f"  revenue Series index     : {list(revenue.index)}   (inherited from the filter)")
print()
print("  per-row arithmetic:")
print("    label  order_id  product     qty  unit_price     revenue")
print("    -----  --------  ----------  ---  ----------  ----------")
print("        0         1  widget        3       10.00        30.0")
print("        2         3  gadget        4       20.00        80.0")
print("        4         5  doohickey     1       50.00        50.0")
print("        6         7  widget       10        5.00        50.0")
print("        9        10  gadget        2       35.00        70.0")

print()
print("STEP 5  THE RESULT  --  north rows with the new 'revenue' column")
print("-" * 78)
print(north_with_revenue.to_string())

print()
print("STEP 6  verification gates")
print("-" * 78)
print(f"  rows                     : {len(north_with_revenue)} == {EXPECTED_NORTH_ROWS} -> "
      f"{len(north_with_revenue) == EXPECTED_NORTH_ROWS}")
print(f"  columns                  : {OUTPUT_COLUMNS}")
print(f"  new column added         : {NEW_COLUMN!r} present -> {NEW_COLUMN in OUTPUT_COLUMNS}")
print(f"  index labels             : {OUTPUT_LABELS}")
print(f"  expected labels          : {EXPECTED_NORTH_INDEX}")
print(f"  index PRESERVED          : {INDEX_PRESERVED}   (no reset_index(drop=True))")
print(f"  index is contiguous 0..4 : {OUTPUT_LABELS == list(range(EXPECTED_NORTH_ROWS))}   "
      f"(False by design)")
print(f"  source frame columns     : {list(df.columns)}")
print(f"  source frame unmutated   : {PARENT_UNMUTATED}   ('revenue' not added to df)")
print()
print("  exact-float evidence (why round(v, 2) is not applied):")
print(f"    revenue values         : {REVENUE_VALUES}")
print(f"    float.is_integer()     : {ALL_EXACT_INTEGERS}")
print(f"    hex representations    : {[float(v).hex() for v in REVENUE_VALUES]}")
print(f"    sum() repr             : {REVENUE_TOTAL!r}")
print(f"    sum() == 280.0 exactly : {TOTAL_EXACTLY_280}")

print()
print("=" * 78)
print("KEY RESULTS")
print("=" * 78)
print(f"  SOURCE CSV                 : {CSV_PATH}")
print(f"  source md5 / bytes / rows  : {CSV_MD5} / {len(CSV_BYTES)} / {len(df)}")
print(f"  DECOY NOT USED             : {DECOY_REL} (md5 {DECOY_MD5})")
print(f"  FILTER                     : df[df['region'] == 'north']  ->  {north_row_count} rows")
print(f"  NEW COLUMN                 : 'revenue' = qty * unit_price")
print(f"  INDEX LABELS (not reset)   : {OUTPUT_LABELS}")
print(f"  order_id values            : {EXPECTED_NORTH_ORDER_IDS}")
print(f"  REVENUE PER ROW            : {REVENUE_VALUES}")
print(f"  REVENUE SUM (exact)        : {float(REVENUE_TOTAL)}")
print(f"  sum() == 280.0             : {TOTAL_EXACTLY_280}  (exact equality, no rounding)")
print("  VERIFICATION               : PASS -- 5 north rows, labels [0, 2, 4, 6, 9] preserved,")
print("                               'revenue' materialized via pd.DataFrame({...}),")
print("                               source frame unmutated, no reset_index, no round()")
print("=" * 78)
