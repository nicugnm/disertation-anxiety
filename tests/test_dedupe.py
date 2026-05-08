import pandas as pd

from src.preprocessing.dedupe import deduplicate


def test_drops_exact_duplicates():
    df = pd.DataFrame({
        "id": ["a", "b", "c"],
        "subreddit": ["x", "x", "y"],
        "clean_text": ["hello world", "hello world", "totally different"],
    })
    out = deduplicate(df)
    assert len(out) == 2


def test_handles_empty():
    df = pd.DataFrame(columns=["id", "subreddit", "clean_text"])
    out = deduplicate(df)
    assert out.empty
