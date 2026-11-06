"""Read and write the parquet files in data/raw/ without creating duplicates."""

from pathlib import Path

import pandas as pd


def upsert(new: pd.DataFrame, path: Path, key: str | list[str]) -> pd.DataFrame:
    """Merge `new` into the parquet file at `path`, keeping the newest row per key.

    Backfills and the daily job both call this, so re-running an ingest for
    dates you already have is always safe.
    """
    keys = [key] if isinstance(key, str) else key
    if path.exists():
        old = pd.read_parquet(path)
        combined = pd.concat([old, new], ignore_index=True)
    else:
        combined = new.copy()
    combined = (
        combined.drop_duplicates(subset=keys, keep="last")  # `new` rows come last, so they win
        .sort_values(keys)
        .reset_index(drop=True)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(path, index=False)
    return combined


def load(path: Path) -> pd.DataFrame:
    """Read a parquet file, with a helpful message if ingest hasn't been run."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `uv run gridforecast ingest` first.")
    return pd.read_parquet(path)
