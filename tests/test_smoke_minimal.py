"""Minimal end-to-end smoke: synthetic -> preprocess -> weak label -> tfidf train -> evaluate.

This is the test that proves the wiring works without external creds.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.collection.synthetic import SyntheticCollector
from src.evaluation.metrics import full_report
from src.labeling.aggregate import aggregate_labels
from src.labeling.weak import apply_weak_labels
from src.models.registry import build_model
from src.models.splits import split
from src.preprocessing.pipeline import preprocess_dataframe
from src.utils.config import load_labeling, load_model_config, load_subreddits


@pytest.mark.timeout(180)  # safety net; harness may not enforce
def test_smoke_pipeline_end_to_end():
    cfg_subs = load_subreddits("configs/subreddits.yaml")
    cfg_lab = load_labeling("configs/labeling.yaml")

    # 1) Collect (synthetic — small)
    collector = SyntheticCollector(cfg_subs, n_per_subreddit=80, seed=42)
    rows = []
    for sub in cfg_subs.subreddits:
        rows.extend(p.to_dict() for p in collector.collect_subreddit(sub.name))
    df = pd.DataFrame(rows)
    assert len(df) > 0

    # 2) Preprocess (skip NER for speed/no-deps)
    df = preprocess_dataframe(df, use_ner=False, keep_only_english=False)
    assert len(df) > 0
    assert "clean_text" in df.columns

    # 3) Weak label + aggregate
    df = apply_weak_labels(df, cfg_subs, cfg_lab)
    df = aggregate_labels(df, cfg_lab)
    assert "label_anxiety" in df.columns

    # 4) Train baseline TF-IDF
    cfg = load_model_config("configs/models/baseline.yaml")
    train, val, test = split(df, cfg.target, test_size=0.2, val_size=0.1)
    if len(test) == 0:
        pytest.skip("Dataset too small after split")

    model = build_model(cfg)
    model.fit(train, val=val if not val.empty else None)

    # 5) Evaluate
    proba = model.predict_proba(test)
    y = (test[f"label_{cfg.target}"].astype(float).fillna(0.0) >= 0.5).astype(int).values
    if len(set(y)) > 1:
        report = full_report(y, proba, bootstrap=False)
        assert "f1" in report
        # sanity: synthetic Anxiety subreddit should be very learnable
        assert report["f1"] >= 0.4
