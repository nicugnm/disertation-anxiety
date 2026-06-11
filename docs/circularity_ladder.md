# The circularity ladder

A single TF-IDF + LogReg model, evaluated against test labels of decreasing dependence on the researcher's own heuristic. `scripts/exp_circularity_ladder.py`.

| rung | test label | independence | anxiety AUROC |
|---|---|---|---:|
| 1. weak label, in-domain | subreddit prior + our lexicon | most circular | 0.9898 |
| 2. subreddit proxy (HA vs Anxiety) | subreddit membership | researcher-chosen proxy | 0.9444 |
| 3. self-disclosure, masked | self-reported diagnosis (post hidden) | independent label | 0.736 |
| 4. expert ANGST | 3 expert psychologists | fully independent | 0.816 |

**Circularity tax.** In-domain weak-label AUROC is 0.990, but against labels the lexicon cannot have produced it falls to 0.736--0.816 (a drop of up to 0.254 AUROC). That gap is the share of the headline number that reflects the model recovering our labelling heuristic rather than a clinical construct.

![circularity ladder](figures/circularity_ladder.png)