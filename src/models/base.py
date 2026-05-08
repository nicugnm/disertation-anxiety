"""Common Model interface. All models implement fit / predict / predict_proba / save / load.

Adding a new model = subclass `BaseModel`, register it in `registry.py`. The
CLI, evaluation, and analysis code touch only the interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import ModelConfig


class BaseModel(ABC):
    """All anxiety-detection models implement this interface."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self._fitted = False

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def target(self) -> str:
        if self.config.target is None:
            raise ValueError(f"{self.name} is multi-target, use `targets` instead.")
        return self.config.target

    @property
    def targets(self) -> list[str]:
        return self.config.targets or [self.target]

    @abstractmethod
    def fit(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> "BaseModel":
        ...

    @abstractmethod
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return shape (n,) for single-target, (n, n_targets) for multi-target."""

    def predict(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(df) >= threshold).astype(int)

    @abstractmethod
    def save(self, path: str | Path) -> None:
        ...

    @abstractmethod
    def load(self, path: str | Path) -> "BaseModel":
        ...

    # ------------------------------------------------------------------ #
    # Conveniences
    # ------------------------------------------------------------------ #

    def y_from_df(self, df: pd.DataFrame) -> np.ndarray:
        col = f"label_{self.target}"
        if col not in df.columns:
            raise KeyError(f"Missing column {col} in dataframe")
        y = df[col].astype(float).fillna(0.0).values
        # Threshold soft labels at 0.5 for training
        return (y >= 0.5).astype(int)

    def x_from_df(self, df: pd.DataFrame):  # type-loose: depends on subclass
        return df[self.config.text_field].astype(str).fillna("").tolist()
