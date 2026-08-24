import pandas as pd

df = pd.read_csv("data/orders.csv")

print("Shape:", df.shape)
print("Columns:", list(df.columns))
print("\nDtypes:")
print(df.dtypes)
print("\nNull counts:", df.isnull().sum().sum())
print("order_id unique:", sorted(df["order_id"].unique()))

TARGET_FIELDS = ["region", "qty", "unit_price"]
assert all(c in df.columns for c in TARGET_FIELDS), "target fields missing"

region_field = "region"
qty_field = "qty"
unit_price_field = "unit_price"

print(f"\nTarget fields identified: region='{region_field}', qty='{qty_field}', unit_price='{unit_price_field}'")
print(f"region uniques: {sorted(df[region_field].unique())}")
print(f"qty dtype={df[qty_field].dtype}, range {df[qty_field].min()}-{df[qty_field].max()}")
print(f"unit_price dtype={df[unit_price_field].dtype}, range {df[unit_price_field].min()}-{df[unit_price_field].max()}")
print(f"product categories: {sorted(df['product'].unique())}")

print("\nRegion distribution:")
print(df[region_field].value_counts())

revenue = df[qty_field] * df[unit_price_field]
print("\nRevenue proxy (qty * unit_price) by region:")
print(revenue.groupby(df[region_field]).sum())
print("Total revenue proxy:", revenue.sum())
