"""IO helpers: parquet, JSONL, zstandard JSONL (Pushshift dumps)."""
from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import orjson
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# --------------------------------------------------------------------------- #
# Parquet
# --------------------------------------------------------------------------- #


def write_parquet(df: pd.DataFrame, path: str | Path, compression: str = "zstd") -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, p, compression=compression)


def read_parquet(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(path, columns=columns)


def read_parquet_dataset(paths: list[str | Path]) -> pd.DataFrame:
    """Concat-read a list of parquet files."""
    return pd.concat([read_parquet(p) for p in paths], ignore_index=True)


# --------------------------------------------------------------------------- #
# JSONL
# --------------------------------------------------------------------------- #


def write_jsonl(records: list[dict[str, Any]], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        for r in records:
            f.write(orjson.dumps(r))
            f.write(b"\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    out: list[dict[str, Any]] = []
    with p.open("rb") as f:
        for line in f:
            if not line.strip():
                continue
            out.append(orjson.loads(line))
    return out


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Streaming JSONL reader — for files that don't fit in memory."""
    p = Path(path)
    with p.open("rb") as f:
        for line in f:
            if not line.strip():
                continue
            yield orjson.loads(line)


# --------------------------------------------------------------------------- #
# Pushshift / Reddit dumps (.zst-compressed JSONL)
# --------------------------------------------------------------------------- #


def iter_zst_jsonl(path: str | Path, chunk_size: int = 2**24) -> Iterator[dict[str, Any]]:
    """Stream records from a `.zst`-compressed JSONL file (Pushshift / arctic_shift dumps).

    Reddit dump files for academic research come compressed this way. They can
    be tens of GB; we never load the whole thing into memory.
    """
    import zstandard as zstd

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    dctx = zstd.ZstdDecompressor(max_window_size=2**31)
    with p.open("rb") as fh, dctx.stream_reader(fh) as reader:
        buf = b""
        while True:
            chunk = reader.read(chunk_size)
            if not chunk:
                break
            buf += chunk
            *lines, buf = buf.split(b"\n")
            for line in lines:
                if not line.strip():
                    continue
                try:
                    yield orjson.loads(line)
                except orjson.JSONDecodeError:
                    # Some Pushshift records have stray bytes; skip silently.
                    continue
        if buf.strip():
            try:
                yield orjson.loads(buf)
            except orjson.JSONDecodeError:
                pass


# --------------------------------------------------------------------------- #
# Generic dump (json + jsonl auto-detect by extension)
# --------------------------------------------------------------------------- #


def dump_json(obj: Any, path: str | Path, indent: int = 2) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, default=str)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)
