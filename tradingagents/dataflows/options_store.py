"""Saved option-chain snapshots: the history a live-only vendor cannot give back.

Deribit serves today's chain only, so every day without a saved copy is history
lost for good. A snapshot is one gzipped CSV of a normalized
:class:`~tradingagents.dataflows.options_metrics.OptionChain`, filed as
``<root>/<venue>/<underlying>/<YYYY-MM-DDTHHMMZ>.csv.gz``. Its rows are the
chain's quotes plus the snapshot's time and spot, so a file reads back on its own.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from tradingagents.dataflows.options_metrics import CHAIN_COLUMNS, OptionChain

SUFFIX = ".csv.gz"


def snapshot_path(root: str | os.PathLike, chain: OptionChain) -> Path:
    """Where a chain's snapshot is filed, named by the minute it was taken (UTC)."""
    stamp = pd.Timestamp(chain.as_of).tz_convert("UTC").strftime("%Y-%m-%dT%H%MZ")
    return Path(root) / chain.venue.lower() / chain.underlying.upper() / f"{stamp}{SUFFIX}"


def save_chain(chain: OptionChain, root: str | os.PathLike) -> Path:
    """Write a chain's snapshot and return its path; a half-written file is never left behind."""
    path = snapshot_path(root, chain)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = chain.quotes.loc[:, list(CHAIN_COLUMNS)].copy()
    frame.insert(0, "as_of", pd.Timestamp(chain.as_of).isoformat())
    frame.insert(1, "spot", chain.spot)
    frame.insert(2, "underlying", chain.underlying)
    frame.insert(3, "venue", chain.venue)
    partial = path.with_name(path.name + ".partial")
    frame.to_csv(partial, index=False, compression="gzip")
    partial.replace(path)
    return path


def load_chain(path: str | os.PathLike) -> OptionChain:
    """Read a saved snapshot back into the chain it was taken from."""
    frame = pd.read_csv(path, compression="gzip")
    first = frame.iloc[0]
    quotes = frame.loc[:, list(CHAIN_COLUMNS)].copy()
    quotes["expiry"] = pd.to_datetime(quotes["expiry"], utc=True)
    return OptionChain(
        underlying=str(first["underlying"]),
        venue=str(first["venue"]),
        as_of=pd.Timestamp(first["as_of"]).to_pydatetime(),
        spot=float(first["spot"]),
        quotes=quotes,
    )


def list_snapshots(root: str | os.PathLike, venue: str, underlying: str) -> list[Path]:
    """An underlying's saved snapshots, oldest first."""
    folder = Path(root) / venue.lower() / underlying.upper()
    return sorted(folder.glob(f"*{SUFFIX}")) if folder.is_dir() else []
