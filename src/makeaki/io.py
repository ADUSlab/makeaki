"""Robust table readers for MIMIC-IV and eICU.

Tables may be stored as ``name.csv`` or ``name.csv.gz``; ``resolve_table`` finds
whichever exists. Large tables (labevents, chartevents) are read in chunks with
an item-identifier filter so that memory use stays bounded.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def resolve_table(directory: Path, name: str) -> Path:
    """Return the path to ``name`` accepting either .csv or .csv.gz.

    ``name`` is given without extension, for example ``"patients"``.
    """
    for suffix in (".csv", ".csv.gz"):
        candidate = directory / f"{name}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find {name}.csv or {name}.csv.gz in {directory}"
    )


def read_table(
    directory: Path,
    name: str,
    usecols: Sequence[str] | None = None,
    dtype: dict[str, object] | None = None,
    parse_dates: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Read a full table into memory (use only for small or moderate tables)."""
    path = resolve_table(directory, name)
    logger.info("Reading %s", path)
    return pd.read_csv(
        path,
        usecols=usecols,
        dtype=dtype,
        parse_dates=list(parse_dates) if parse_dates else None,
        low_memory=False,
    )


def read_filtered_by_itemid(
    directory: Path,
    name: str,
    itemids: Iterable[int],
    itemid_col: str = "itemid",
    usecols: Sequence[str] | None = None,
    parse_dates: Sequence[str] | None = None,
    chunksize: int = 2_000_000,
) -> pd.DataFrame:
    """Stream a large table in chunks, keeping only the requested item ids.

    Suitable for labevents and chartevents, which are too large to hold in
    memory in full.
    """
    path = resolve_table(directory, name)
    wanted = set(int(i) for i in itemids)
    logger.info("Streaming %s filtered to %d item ids", path, len(wanted))

    parts: list[pd.DataFrame] = []
    reader = pd.read_csv(
        path,
        usecols=usecols,
        parse_dates=list(parse_dates) if parse_dates else None,
        chunksize=chunksize,
        low_memory=False,
    )
    for chunk in reader:
        keep = chunk[chunk[itemid_col].isin(wanted)]
        if not keep.empty:
            parts.append(keep)
    if not parts:
        return pd.DataFrame(columns=list(usecols) if usecols else None)
    return pd.concat(parts, ignore_index=True)
