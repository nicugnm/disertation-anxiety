from src.collection.synthetic import SyntheticCollector
from src.utils.config import load_subreddits


def test_synthetic_collector_yields():
    cfg = load_subreddits("configs/subreddits.yaml")
    coll = SyntheticCollector(cfg, n_per_subreddit=10, seed=42)
    out = list(coll.collect_subreddit("Anxiety"))
    assert len(out) > 0
    p = out[0]
    assert p.subreddit == "Anxiety"
    assert p.is_self
    assert p.body
    assert p.source == "synthetic"


def test_synthetic_reproducible():
    cfg = load_subreddits("configs/subreddits.yaml")
    a = list(SyntheticCollector(cfg, n_per_subreddit=5, seed=42).collect_subreddit("Anxiety"))
    b = list(SyntheticCollector(cfg, n_per_subreddit=5, seed=42).collect_subreddit("Anxiety"))
    assert [p.body for p in a] == [p.body for p in b]
