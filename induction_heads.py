"""
Find and visualize induction heads in gpt2-small.

Feeds the model a random sequence repeated twice, then:
  1. scores every attention head for how strongly it attends along the
     induction "stripe" (key = query - seq_len), and
  2. plots the score grid plus the top head's raw attention pattern.

Run:  python induction_heads.py
Needs: pip install torch transformer_lens matplotlib
"""

import torch
from transformer_lens import HookedTransformer
import matplotlib.pyplot as plt

# CPU is fine -- gpt2-small is 124M params.
model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads

# --- build BOS + [random seq] + [same random seq] ---------------------------
torch.manual_seed(0)
seq_len = 50
rand = torch.randint(0, model.cfg.d_vocab, (1, seq_len))
bos = torch.tensor([[model.tokenizer.bos_token_id]])
tokens = torch.cat([bos, rand, rand], dim=1)            # length 2*seq_len + 1

# --- run once, keeping all activations (incl. attention patterns) -----------
with torch.no_grad():
    logits, cache = model.run_with_cache(tokens)

# --- induction score for every head ----------------------------------------
# A second-copy token at query position p should attend to the token that
# FOLLOWED the first occurrence of the current token, i.e. key = p - (seq_len-1).
# That is the diagonal at offset -(seq_len-1) from the main diagonal.
induction_score = torch.zeros(n_layers, n_heads)
for layer in range(n_layers):
    pattern = cache["pattern", layer][0]                # [head, query, key]
    stripe = pattern.diagonal(offset=-(seq_len - 1), dim1=-2, dim2=-1)
    induction_score[layer] = stripe.mean(dim=-1)

# --- report the strongest heads --------------------------------------------
vals, idx = induction_score.flatten().topk(5)
print("Top induction heads (gpt2-small):")
for v, i in zip(vals, idx):
    layer, head = divmod(i.item(), n_heads)
    print(f"  L{layer}H{head}:  score {v:.3f}")

# --- plot 1: induction score per head --------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(induction_score.numpy(), cmap="viridis", aspect="auto")
ax.set_xlabel("head"); ax.set_ylabel("layer")
ax.set_title("Induction score per head — gpt2-small")
ax.set_xticks(range(n_heads)); ax.set_yticks(range(n_layers))
fig.colorbar(im, label="mean attention on induction stripe")
fig.savefig("induction_scores.png", dpi=120, bbox_inches="tight")

# --- plot 2: attention pattern of the single best induction head -----------
best = induction_score.flatten().argmax().item()
L, H = divmod(best, n_heads)
attn = cache["pattern", L][0, H].numpy()                # [query, key]
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(attn, cmap="viridis")
ax.set_xlabel("key position"); ax.set_ylabel("query position")
ax.set_title(f"L{L}H{H} attention on a repeated sequence")
fig.colorbar(im, label="attention weight")
fig.savefig("induction_pattern.png", dpi=120, bbox_inches="tight")

print("\nWrote induction_scores.png and induction_pattern.png")
plt.show()
