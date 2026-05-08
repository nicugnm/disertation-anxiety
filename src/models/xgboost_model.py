"""XGBoost on hand-crafted linguistic features.

This model is the bridge to the linguistic-analysis chapter: SHAP values
on its features tell us which markers drive predictions. Easier to defend
in a thesis than "the transformer learned something."
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.features.linguistic import extract_dataframe, feature_columns
from src.models.base import BaseModel


class XgboostLinguisticModel(BaseModel):
    def __init__(self, config) -> None:  # noqa: ANN001
        super().__init__(config)
        e = config.extra
        cls_kwargs = e.get("classifier", {})
        scale = cls_kwargs.get("scale_pos_weight", "auto")
        self._scale_pos_weight = scale  # may be "auto"
        self._kwargs = cls_kwargs
        self.model: XGBClassifier | None = None
        self._feature_cols: list[str] | None = None

    def _build(self, scale_pos_weight: float) -> XGBClassifier:
        c = self._kwargs
        return XGBClassifier(
            n_estimators=c.get("n_estimators", 500),
            max_depth=c.get("max_depth", 6),
            learning_rate=c.get("learning_rate", 0.05),
            subsample=c.get("subsample", 0.8),
            colsample_bytree=c.get("colsample_bytree", 0.8),
            scale_pos_weight=scale_pos_weight,
            random_state=c.get("random_state", 42),
            eval_metric="logloss",
            tree_method="hist",
        )

    def _features_df(self, df: pd.DataFrame) -> pd.DataFrame:
        feat_df = extract_dataframe(df, text_col=self.config.text_field)
        cols = feature_columns(feat_df)
        if self._feature_cols is None:
            self._feature_cols = cols
        return feat_df[self._feature_cols]

    def fit(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> XgboostLinguisticModel:
        X = self._features_df(train).values
        y = self.y_from_df(train)
        if self._scale_pos_weight == "auto":
            pos = max(1, int(y.sum()))
            spw = (len(y) - pos) / pos
        else:
            spw = float(self._scale_pos_weight)
        self.model = self._build(spw)

        early = self._kwargs.get("early_stopping_rounds")
        fit_kwargs: dict = {}
        if val is not None and not val.empty:
            X_val = self._features_df(val).values
            y_val = self.y_from_df(val)
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            if early:
                self.model.set_params(early_stopping_rounds=early)
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        elif f"label_{self.target}_weight" in train.columns:
            fit_kwargs["sample_weight"] = (
                train[f"label_{self.target}_weight"].astype(float).fillna(1.0).values
            )

        self.model.fit(X, y, **fit_kwargs)
        self._fitted = True
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X = self._features_df(df).values
        return self.model.predict_proba(X)[:, 1]  # type: ignore[union-attr]

    def feature_importance(self) -> pd.Series:
        if self.model is None or self._feature_cols is None:
            return pd.Series(dtype=float)
        return pd.Series(self.model.feature_importances_, index=self._feature_cols).sort_values(
            ascending=False
        )

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as f:
            pickle.dump(
                {"model": self.model, "feature_cols": self._feature_cols, "target": self.target},
                f,
            )

    def load(self, path: str | Path) -> XgboostLinguisticModel:
        with Path(path).open("rb") as f:
            obj = pickle.load(f)
        self.model = obj["model"]
        self._feature_cols = obj["feature_cols"]
        self._fitted = True
        return self
