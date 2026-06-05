"""Phase 2 (b1) — domain-adaptive pretraining (DAPT) of RoBERTa on our corpus.

Continue masked-LM pretraining of a base encoder on the 744k-post Reddit
mental-health corpus (disclosure-test authors excluded — no leakage), then
fine-tune the adapted encoder on the headline r/HealthAnxiety-vs-r/Anxiety task
and compare to the un-adapted base. Question: does in-domain MLM pretraining of a
generic ``roberta-base`` close the gap to the domain-pretrained ``mental-roberta``?

Two stages (``--stage pretrain|evaluate|all``):
  * pretrain : MLM continue-pretraining -> checkpoints/roberta-dapt
  * evaluate : fine-tune {base, dapt} on HA (submissions, author-disjoint),
               weighted-F1 vs Low 2020 (0.851); writes docs/dapt_mlm.md + figure.

GPU. Run:
  python scripts/exp_dapt_mlm.py                          # both stages
  python scripts/exp_dapt_mlm.py --stage pretrain --max-docs 200000
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_ha_vs_anxiety import author_disjoint_split, load_head_to_head  # noqa: E402
from exp_stronger_models import train_eval  # noqa: E402

from src.utils.io import read_parquet  # noqa: E402

DATA = "data/processed/labeled.parquet"
DISC = "data/processed/disclosure_testset.parquet"
SEED = 42
LOW_2020 = 0.851
CKPT = "checkpoints/roberta-dapt"
DOC = Path("docs/dapt_mlm.md")
OUTCSV = Path("experiments/dapt_mlm.csv")
FIG = Path("docs/figures/dapt_mlm.png")


def pretrain(base: str, out_dir: str, max_docs: int, block_size: int, epochs: float,
             batch_size: int) -> None:
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForMaskedLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    # corpus: clean_text, excluding disclosure-test authors (no leakage)
    df = read_parquet(DATA)
    df = df[df["clean_text"].astype(str).str.len() >= 30]
    try:
        disc_authors = set(read_parquet(DISC)["author_hash"].dropna())
        df = df[~df["author_hash"].isin(disc_authors)]
    except Exception as ex:  # noqa: BLE001
        print(f"  (no disclosure set to exclude: {ex})")
    texts = df["clean_text"].astype(str).tolist()
    if max_docs and len(texts) > max_docs:
        texts = pd.Series(texts).sample(n=max_docs, random_state=SEED).tolist()
    print(f"DAPT corpus: {len(texts):,} docs from base '{base}' (block_size={block_size})", flush=True)

    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForMaskedLM.from_pretrained(base)
    ds = Dataset.from_dict({"text": texts})

    def tok_fn(b):
        return tok(b["text"], truncation=True, max_length=block_size,
                   return_special_tokens_mask=True)

    ds = ds.map(tok_fn, batched=True, remove_columns=["text"], desc="tokenize")
    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=True, mlm_probability=0.15)

    ta_kwargs = dict(
        output_dir=out_dir, overwrite_output_dir=True,
        per_device_train_batch_size=batch_size, num_train_epochs=epochs,
        learning_rate=5e-5, weight_decay=0.01, warmup_ratio=0.06,
        fp16=torch.cuda.is_available(), save_strategy="no", report_to=[], logging_steps=50,
        seed=SEED,
    )
    try:
        args = TrainingArguments(**ta_kwargs)
    except TypeError:
        args = TrainingArguments(**ta_kwargs)  # no eval here, nothing to rename
    trainer_kwargs = dict(model=model, args=args, train_dataset=ds, data_collator=collator)
    try:
        trainer = Trainer(processing_class=tok, **trainer_kwargs)
    except TypeError:
        trainer = Trainer(tokenizer=tok, **trainer_kwargs)
    trainer.train()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)
    print(f"DAPT checkpoint -> {out_dir}", flush=True)


def evaluate(base: str, ckpt: str, test_size: float) -> None:
    d, stats = load_head_to_head(submissions_only=True)
    print(f"[submissions-only] head-to-head: {stats}", flush=True)
    train, test = author_disjoint_split(d, test_size=test_size)
    tr2, val = author_disjoint_split(train, test_size=0.15)
    print(f"train={len(tr2):,} val={len(val):,} test={len(test):,}", flush=True)

    encoders = [
        ("roberta-base (no DAPT)", base),
        ("roberta-DAPT (ours)", ckpt),
        ("MentalRoBERTa (ref)", "mental/mental-roberta-base"),
    ]
    rows = [{"model": "Low 2020 (SGD-L1)", "pretrained": "-", "weighted_f1": LOW_2020,
             "auroc": None, "note": "published baseline"}]
    for label, hf in encoders:
        print(f"\n=== fine-tuning {label} ({hf}) ===", flush=True)
        try:
            r = train_eval(hf, tr2, test, val)
            rows.append({"model": label, "pretrained": hf, **r, "note": "this run"})
            print(f"  {label}: weighted-F1 {r['weighted_f1']}  AUROC {r['auroc']}", flush=True)
        except Exception as ex:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            rows.append({"model": label, "pretrained": hf, "weighted_f1": None, "auroc": None,
                         "f1": None, "note": f"FAILED: {str(ex)[:80]}"})
        pd.DataFrame(rows).to_csv(OUTCSV, index=False)

    out = pd.DataFrame(rows)
    OUTCSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTCSV, index=False)

    plot = out[out["weighted_f1"].notna()].copy()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(len(plot)), plot["weighted_f1"],
           color=["#7f7f7f", "#C44E52", "#55A868", "#4C72B0"][: len(plot)])
    for i, v in enumerate(plot["weighted_f1"]):
        ax.text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=9)
    ax.axhline(LOW_2020, ls="--", color="k", lw=1, label=f"Low 2020 = {LOW_2020}")
    ax.set_xticks(range(len(plot))); ax.set_xticklabels(plot["model"], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("weighted-F1 (submissions)"); ax.set_ylim(0.78, 0.95)
    ax.set_title("Phase 2 (b1) — domain-adaptive MLM pretraining on r/HealthAnxiety vs r/Anxiety")
    ax.legend()
    fig.tight_layout(); FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=130); plt.close(fig)

    cols = ["model", "pretrained", "weighted_f1", "auroc", "f1", "note"]
    md = [
        "# Phase 2 (b1) — domain-adaptive MLM pretraining (DAPT)",
        "",
        "Continue masked-LM pretraining of `roberta-base` on the 744k-post corpus "
        "(disclosure-test authors excluded), then fine-tune on the submissions-only, "
        "author-disjoint r/HealthAnxiety-vs-r/Anxiety task. Does in-domain MLM close the "
        f"gap from generic RoBERTa to the domain-pretrained MentalRoBERTa? Baseline: Low 2020 = {LOW_2020}. "
        "`scripts/exp_dapt_mlm.py`.",
        "",
        "_Regenerate: `python scripts/exp_dapt_mlm.py`_",
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for r in rows:
        md.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    md += ["", "![dapt mlm](figures/dapt_mlm.png)"]
    DOC.write_text("\n".join(md), encoding="utf-8")
    print("\n", out[cols].to_string(index=False))
    print(f"\nWrote {OUTCSV}, {DOC}, {FIG}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["pretrain", "evaluate", "all"], default="all")
    ap.add_argument("--base", default="roberta-base")
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--max-docs", type=int, default=200000)
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--mlm-epochs", type=float, default=1.0)
    ap.add_argument("--mlm-batch-size", type=int, default=32)
    ap.add_argument("--test-size", type=float, default=0.2)
    args = ap.parse_args()

    if args.stage in ("pretrain", "all"):
        pretrain(args.base, args.checkpoint, args.max_docs, args.block_size,
                 args.mlm_epochs, args.mlm_batch_size)
    if args.stage in ("evaluate", "all"):
        evaluate(args.base, args.checkpoint, args.test_size)


if __name__ == "__main__":
    main()
