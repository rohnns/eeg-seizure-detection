from pathlib import Path
import pandas as pd
import hashlib

shard_dir = Path(r"D:\CHBMIT\features\shards")

feature_files = sorted(shard_dir.glob("*_features.parquet"))

reference_cols = None
reference_hash = None

for f in feature_files[:]:
    df = pd.read_parquet(f)

    cols = list(df.columns)
    h = hashlib.md5(",".join(cols).encode()).hexdigest()[:8]

    print(f"{f.name}")
    print(f"  rows={len(df):5d} cols={len(cols)} hash={h}")

    if reference_cols is None:
        reference_cols = cols
        reference_hash = h
    elif cols != reference_cols:
        print("  ❌ COLUMN MISMATCH")