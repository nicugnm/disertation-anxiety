"""Pilot LLM-labeling run on a small stratified sample.

Goal: prove the tier-2 pipeline works end-to-end with the real API key,
print a few labels and rationales, and estimate the cost of the full run
before we spend real money.
"""
from __future__ import annotations

import json
import time

import pandas as pd
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

load_dotenv()  # pick up ANTHROPIC_API_KEY from .env

from src.labeling.llm import USER_PROMPT_TEMPLATE, _make_client, label_one  # noqa: E402
from src.utils.cache import SqliteCache
from src.utils.config import load_labeling, load_subreddits

console = Console()

# 5 posts per subreddit_group → ~30 posts total — enough to see signal
PER_GROUP = 5

cfg_subs = load_subreddits()
cfg_lab = load_labeling()
df = pd.read_parquet("data/processed/labeled.parquet")

name_to_group = {s.name.lower(): s.group for s in cfg_subs.subreddits}
df["subreddit_group"] = df["subreddit"].str.lower().map(name_to_group).fillna("baseline")

sample = (
    df.groupby("subreddit_group", group_keys=False)
      .apply(lambda g: g.sample(n=min(PER_GROUP, len(g)), random_state=7))
      .reset_index(drop=True)
)
console.print(f"[bold]Pilot sample size:[/bold] {len(sample)}")
console.print(sample.groupby("subreddit_group").size().to_string())
console.print()

client = _make_client()
cache = SqliteCache(".cache/llm_labels_pilot.sqlite")

t0 = time.time()
rows = []
for _, row in sample.iterrows():
    parsed = label_one(
        client,
        row["clean_text"],
        model=cfg_lab.tier2_llm.model,
        max_tokens=cfg_lab.tier2_llm.max_tokens,
        temperature=cfg_lab.tier2_llm.temperature,
        cache=cache,
    )
    rows.append({
        "id": row["id"],
        "subreddit": row["subreddit"],
        "weak_anxiety_bin": int(row.get("weak_anxiety_bin", 0)),
        "weak_health_anxiety_bin": int(row.get("weak_health_anxiety_bin", 0)),
        **{f"llm_{k}": parsed.get(k, 0) for k in ("anxiety", "health_anxiety", "depression", "suicidality")},
        "rationale": parsed.get("rationale", "")[:120],
        "preview": (row["clean_text"][:140] + "…") if len(row["clean_text"]) > 140 else row["clean_text"],
    })
    time.sleep(60.0 / cfg_lab.tier2_llm.rpm)

elapsed = time.time() - t0
result = pd.DataFrame(rows)

# ---- Display
console.print(f"\n[green]Pilot done in {elapsed:.1f}s ({elapsed/len(result):.1f}s/post)[/green]\n")

t = Table(title="LLM labels — pilot")
t.add_column("subreddit", style="dim")
t.add_column("weak\nA / HA")
t.add_column("LLM\nA / HA / D / S", style="cyan")
t.add_column("rationale", overflow="fold")
t.add_column("preview", overflow="fold")
for _, r in result.iterrows():
    t.add_row(
        r["subreddit"],
        f"{r['weak_anxiety_bin']} / {r['weak_health_anxiety_bin']}",
        f"{r['llm_anxiety']} / {r['llm_health_anxiety']} / {r['llm_depression']} / {r['llm_suicidality']}",
        r["rationale"],
        r["preview"],
    )
console.print(t)

# ---- Agreement vs weak
agree_anx = (result["llm_anxiety"] == result["weak_anxiety_bin"]).mean()
agree_ha  = (result["llm_health_anxiety"] == result["weak_health_anxiety_bin"]).mean()
console.print(f"\n[bold]Agreement on this pilot (LLM vs weak):[/bold]")
console.print(f"  anxiety:        {agree_anx:.1%}")
console.print(f"  health_anxiety: {agree_ha:.1%}")
console.print(f"\nLLM positive rates on the pilot:")
console.print(result[["llm_anxiety", "llm_health_anxiety", "llm_depression", "llm_suicidality"]].mean().to_string())

# ---- Cost estimate
# Sonnet 4.6 pricing: $3/M input, $15/M output (approx; verify on dashboard)
INPUT_PER_REQ = 1100   # system + codebook + post (~700 tokens for 1500-char post + boilerplate)
OUTPUT_PER_REQ = 90    # JSON response ~80–120 tokens
N_FULL = sum(cfg_lab.tier2_llm.per_group_sample.values())  # 8000 by default
in_cost = N_FULL * INPUT_PER_REQ / 1_000_000 * 3.0
out_cost = N_FULL * OUTPUT_PER_REQ / 1_000_000 * 15.0
seconds = N_FULL * 60.0 / cfg_lab.tier2_llm.rpm
console.print(f"\n[bold]Cost estimate for FULL tier-2 run ({N_FULL} posts, {cfg_lab.tier2_llm.model}):[/bold]")
console.print(f"  est. input tokens   : {N_FULL * INPUT_PER_REQ / 1_000_000:.2f} M  →  ${in_cost:.2f}")
console.print(f"  est. output tokens  : {N_FULL * OUTPUT_PER_REQ / 1_000_000:.2f} M  →  ${out_cost:.2f}")
console.print(f"  est. total          : [bold]${in_cost + out_cost:.2f}[/bold]")
console.print(f"  est. wall time      : ~{seconds/60:.0f} min at rpm={cfg_lab.tier2_llm.rpm}")
console.print("\n(Cache makes re-runs free for already-labeled posts.)")
