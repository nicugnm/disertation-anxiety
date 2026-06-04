"""Factory mapping config.model_type -> concrete BaseModel subclass."""
from __future__ import annotations

from src.models.base import BaseModel
from src.utils.config import ModelConfig


def build_model(config: ModelConfig) -> BaseModel:
    if config.model_type == "tfidf":
        from src.models.tfidf import TfidfLogRegModel

        return TfidfLogRegModel(config)
    if config.model_type == "xgboost":
        from src.models.xgboost_model import XgboostLinguisticModel

        return XgboostLinguisticModel(config)
    if config.model_type == "transformer":
        from src.models.transformer import TransformerModel

        return TransformerModel(config)
    if config.model_type == "multitask_transformer":
        from src.models.multitask import MultiTaskTransformer

        return MultiTaskTransformer(config)
    if config.model_type == "dann_multitask":
        from src.models.dann import DannMultiTaskModel

        return DannMultiTaskModel(config)
    if config.model_type == "fusion_multitask":
        from src.models.fusion import FusionMultiTaskModel

        return FusionMultiTaskModel(config)
    if config.model_type == "hier_user":
        from src.models.hier import HierUserModel

        return HierUserModel(config)
    if config.model_type == "llm_zero_shot":
        from src.models.llm_zero_shot import LlmZeroShotModel

        return LlmZeroShotModel(config)
    raise ValueError(f"Unknown model_type: {config.model_type}")
