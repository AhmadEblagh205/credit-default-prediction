"""Download the UCI Taiwan credit-default dataset and recreate the coursework split."""
from __future__ import annotations
import io
import zipfile
from pathlib import Path
from urllib.request import urlopen
import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def validate(df: pd.DataFrame):
    if len(df) != 30_000:
        raise ValueError(f"Expected 30,000 rows, got {len(df)}")
    target = "default payment next month"
    if target not in df.columns:
        raise ValueError("Unexpected UCI schema: target column missing")
    if not set(df[target].unique()).issubset({0, 1}):
        raise ValueError("Unexpected UCI target values")


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    print("Downloading the UCI dataset...")
    raw = urlopen(URL, timeout=30).read()
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        xls_name = next(n for n in zf.namelist() if n.lower().endswith(".xls"))
        with zf.open(xls_name) as fh:
            df = pd.read_excel(fh, header=1)
    validate(df)
    # The original coursework used the first 24k rows as development and the
    # final 6k as the held-out file. Preserve that split for comparability.
    df.iloc[:24_000].to_csv(RAW / "CreditCard_train.csv", index=False)
    df.iloc[24_000:].to_csv(RAW / "CreditCard_test.csv", index=False)
    print("Wrote 24,000 development + 6,000 held-out rows to data/raw/")


if __name__ == "__main__":
    main()
