"""Train the multi-task transformer on a FULL-corpus sample (all subreddits) so the
external-validation comparison against TF-IDF is like-for-like.

The checkpoint used earlier was trained on the narrower DANN-transfer split (panic +
baseline subs excluded); this retrains on a ~200k sample spanning the whole corpus,
matching the TF-IDF training distribution. Saves to
experiments/runs/multitask_fullcorpus/model (gitignored; regenerate as needed).

GPU. Run:
  python scripts/train_multitask_fullcorpus.py --n 200000 --epochs 3
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.models.registry import build_model
from src.utils.config import load_model_config
from src.utils.io import read_parquet

DATA = "data/processed/labeled.parquet"
SEED = 42
OUT = Path("experiments/runs/multitask_fullcorpus/model")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--val-size", type=int, default=5000)
    args = ap.parse_args()

    df = read_parquet(DATA)
    df = df[df["clean_text"].astype(str).str.len() >= 30].reset_index(drop=True)
    take = min(len(df), args.n + args.val_size)
    df = df.sample(take, random_state=SEED).reset_index(drop=True)
    val = df.iloc[: args.val_size].reset_index(drop=True)
    train = df.iloc[args.val_size :].reset_index(drop=True)
    print(f"full-corpus train={len(train):,}  val={len(val):,}  (subreddits={df['subreddit'].nunique()})")

    cfg = load_model_config("configs/models/multitask.yaml")
    cfg.extra["train"]["num_train_epochs"] = args.epochs
    model = build_model(cfg).fit(train, val=val)
    OUT.mkdir(parents=True, exist_ok=True)
    model.save(OUT)
    print(f"saved full-corpus multitask checkpoint -> {OUT}")


if __name__ == "__main__":
    main()
