from __future__ import annotations

import pandas as pd

from src.collection.author_history import recover_author_usernames
from src.preprocessing.anonymize import _hash_username
from src.utils.io import write_parquet


def test_recover_usernames_maps_hashes_and_skips_deleted(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    # Two raw shards with real authors + a deleted one.
    write_parquet(
        pd.DataFrame({"id": ["1", "2"], "author": ["alice", "[deleted]"]}),
        raw_dir / "SubA.parquet",
    )
    write_parquet(
        pd.DataFrame({"id": ["3"], "author": ["bob"]}),
        raw_dir / "SubB.parquet",
    )
    users = pd.DataFrame({"author_hash": [_hash_username("alice"), _hash_username("bob")]})
    mapping = recover_author_usernames(users, raw_dir=raw_dir)
    assert mapping == {_hash_username("alice"): "alice", _hash_username("bob"): "bob"}
    # Deleted author never appears.
    assert "[deleted]" not in mapping.values()
