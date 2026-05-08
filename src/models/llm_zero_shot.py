"""Claude zero-/few-shot baseline. Tests prompting vs fine-tuning."""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.base import BaseModel
from src.utils.cache import SqliteCache
from src.utils.logging import get_logger

log = get_logger(__name__)


SYSTEM_PROMPT = """You are an expert NLP annotator for a mental-health research project. \
For each post, decide whether the labeled phenomenon is present. Respond with a single \
word: "yes" or "no". Do not include any explanation."""

PROMPT_TEMPLATE = """Label: {target_description}

Post:
\"\"\"
{post}
\"\"\"

Is the label present? Answer "yes" or "no" only."""

TARGET_DESCRIPTIONS = {
    "anxiety": "the author expresses present-day anxious affect, anxious cognition, or anxious physiological experience",
    "health_anxiety": "the author expresses anxiety specifically about their own (or a loved one's) physical health, illness, or fear of disease",
    "depression": "the author expresses depressive symptoms (anhedonia, hopelessness, persistent low mood, worthlessness)",
    "suicidality": "the author expresses suicidal ideation, intent, plan, or recent attempt",
}


class LlmZeroShotModel(BaseModel):
    """No fitting — runs the prompt at predict time. `fit` is a no-op."""

    def __init__(self, config) -> None:  # noqa: ANN001
        super().__init__(config)
        e = config.extra
        self._model = e.get("model", "claude-sonnet-4-6")
        self._max_tokens = e.get("max_tokens", 64)
        self._temperature = e.get("temperature", 0.0)
        self._cache_path = e.get("cache_path", ".cache/llm_zero_shot.sqlite")
        self._rpm = e.get("rpm", 30)
        self._n_few_shot = int(e.get("n_few_shot", 0))
        self._few_shot_path = e.get("few_shot_path", "data/processed/few_shot_examples.jsonl")
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as e:
                raise RuntimeError("Install `anthropic` to use the zero-shot baseline.") from e
            if not os.getenv("ANTHROPIC_API_KEY"):
                raise RuntimeError("ANTHROPIC_API_KEY not set.")
            self._client = anthropic.Anthropic()
        return self._client

    def _few_shot_block(self) -> str:
        if self._n_few_shot <= 0:
            return ""
        from src.utils.io import read_jsonl

        path = Path(self._few_shot_path)
        if not path.exists():
            log.warning("llm_zero_shot.few_shot_missing", path=str(path))
            return ""
        examples = read_jsonl(path)[: self._n_few_shot]
        out = []
        for ex in examples:
            ans = "yes" if ex.get(self.target, 0) else "no"
            out.append(f"Post: \"\"\"{ex.get('clean_text', '')}\"\"\"\nAnswer: {ans}\n")
        return "Examples:\n" + "\n".join(out) + "\n---\n"

    def fit(self, train, val=None, sample_weight=None) -> "LlmZeroShotModel":  # noqa: ANN001
        # No fitting; few-shot examples are loaded at predict time.
        self._fitted = True
        return self

    def _label_one(self, text: str, cache: SqliteCache) -> float:
        target_desc = TARGET_DESCRIPTIONS.get(self.target, self.target)
        prompt = self._few_shot_block() + PROMPT_TEMPLATE.format(
            target_description=target_desc, post=text[:4000]
        )
        key = SqliteCache.make_key(self._model, SYSTEM_PROMPT, prompt)
        cached = cache.get(key)
        if cached is not None:
            return float(cached)

        client = self._get_client()
        msg = client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        text_resp = msg.content[0].text.strip().lower()  # type: ignore[attr-defined]
        score = 1.0 if text_resp.startswith("yes") else 0.0
        cache.set(key, score)
        return score

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        cache = SqliteCache(self._cache_path)
        interval = 60.0 / max(1, self._rpm)
        last = 0.0
        scores: list[float] = []
        for text in df[self.config.text_field].astype(str).fillna(""):
            key = SqliteCache.make_key(self._model, SYSTEM_PROMPT, text)
            if key not in cache:
                elapsed = time.time() - last
                if elapsed < interval:
                    time.sleep(interval - elapsed)
                last = time.time()
            scores.append(self._label_one(text, cache))
        cache.close()
        return np.array(scores)

    def save(self, path: str | Path) -> None:
        # Nothing to save beyond config; touch a sentinel file.
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "llm_zero_shot.txt").write_text(self._model)

    def load(self, path: str | Path) -> "LlmZeroShotModel":
        self._fitted = True
        return self
