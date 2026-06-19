# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A single mechanistic-interpretability experiment. It trains the **same** tiny 2-layer
transformer two different ways and contrasts the attention heads that emerge:

- **Task A — copy/repeat** `[BOS, s, s]`: produces an **induction head** whose attention
  forms an offset-diagonal stripe (offset `-(K-1)`).
- **Task B — Dyck-2 bracket strings** `( ) [ ]`: produces a **matching head** where each
  closer token attends back to its matching opener (a stack/arc pattern).

The only thing that differs between the two runs is the training data — that contrast is
the entire point of the experiment.

## Layout

- `Induction_vs_Dyck-2.ipynb` — the experiment (the real artifact). 12 cells, run top to
  bottom. Cell 0's docstring references a `contrast.py`, but no such file exists; the
  notebook is the source of truth.
- `Claude-Running in-context learning with gpt2-small locally.md` — exported Claude chat
  transcript that motivated the project. Background reading, not code.
- `README.md` — one-line description.

## Running

Notebook only; no build/lint/test tooling exists. Kernel is `python3`. Use `python3`
(there is no `python` on this machine).

```
pip install torch transformer_lens matplotlib
```

Run cells in order. Outputs: console training logs + accuracy table, and `contrast.png`
(the side-by-side attention plots) written by cell 9.

Targets CPU (`DEVICE = "cpu"`); each model trains in a couple of minutes. Training step
counts are env-overridable: `IND_STEPS` (default 3000) and `DYCK_STEPS` (default 6000).

## Key facts when editing

- Both models share architecture via `make_model(d_vocab)`: 2 layers, `d_model=128`,
  4 heads, `d_head=32`, `n_ctx = 1 + 2*K = 41`. Vocab differs per task — induction uses
  `V_IND=30`, Dyck-2 uses 5 (`0=BOS, 1='(' 2=')' 3='[' 4=']'`).
- `K=20` is the repeat length and also fixes `N_CTX`; changing it changes both tasks and
  the induction stripe offset.
- Head selection is automatic: the induction head is scored by mean attention on the
  `offset=-(K-1)` diagonal; the matching head is scored by mean closer→opener attention
  over the ground-truth bracket pairs from `bracket_pairs()`. Don't hardcode head
  indices — they're found at runtime.
- Seeds are pinned (`torch.manual_seed(0)`, `random.seed(0)`, `cfg.seed=0`) for
  reproducibility; preserve this when adding randomness.
- The final cell holds the `dialog_content` conversation transcript (the experiment's
  narrative notes), not executable analysis.
