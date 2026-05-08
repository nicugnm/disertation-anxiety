"""TF-IDF + Logistic Regression baseline."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.models.base import BaseModel


class TfidfLogRegModel(BaseModel):
    def __init__(self, config) -> None:  # noqa: ANN001
        super().__init__(config)
        e = config.extra
        vec_kwargs = e.get("vectorizer", {})
        clf_kwargs = e.get("classifier", {})
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=tuple(vec_kwargs.get("ngram_range", [1, 2])),
                min_df=vec_kwargs.get("min_df", 5),
                max_df=vec_kwargs.get("max_df", 0.95),
                max_features=vec_kwargs.get("max_features", 100000),
                sublinear_tf=vec_kwargs.get("sublinear_tf", True),
                strip_accents="unicode",
                lowercase=True,
            )),
            ("clf", LogisticRegression(
                C=clf_kwargs.get("C", 1.0),
                class_weight=clf_kwargs.get("class_weight", "balanced"),
                max_iter=clf_kwargs.get("max_iter", 1000),
                solver=clf_kwargs.get("solver", "liblinear"),
                random_state=clf_kwargs.get("random_state", 42),
            )),
        ])

    def fit(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> TfidfLogRegModel:
        X = self.x_from_df(train)
        y = self.y_from_df(train)
        if sample_weight is None and f"label_{self.target}_weight" in train.columns:
            sample_weight = train[f"label_{self.target}_weight"].astype(float).fillna(1.0).values
        # Pipeline param-passing for sample_weight goes to the final step
        if sample_weight is not None:
            self.pipeline.fit(X, y, clf__sample_weight=sample_weight)
        else:
            self.pipeline.fit(X, y)
        self._fitted = True
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X = self.x_from_df(df)
        return self.pipeline.predict_proba(X)[:, 1]

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as f:
            pickle.dump({"pipeline": self.pipeline, "target": self.target}, f)

    def load(self, path: str | Path) -> TfidfLogRegModel:
        with Path(path).open("rb") as f:
            obj = pickle.load(f)
        self.pipeline = obj["pipeline"]
        self._fitted = True
        return self
