"""
Two circuits, one architecture.

Trains the SAME tiny 2-layer transformer two ways:
  (A) on a copy/repeat task  -> forms an INDUCTION head (offset-diagonal stripe)
  (B) on Dyck-2 bracket text  -> forms a MATCHING head (stack/arc pattern)

Then plots both heads' attention side by side. Only the training data
differs, so the difference in the learned attention is the point.

Run:  python contrast.py
Needs: pip install torch transformer_lens matplotlib
CPU is fine; each model trains in a couple of minutes.
"""

import os
import random
import torch
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer, HookedTransformerConfig

torch.manual_seed(0)
random.seed(0)
DEVICE = "cpu"

# --- shared architecture ----------------------------------------------------
K = 20                       # repeat length for the induction task
N_CTX = 1 + 2 * K            # = 41 ; Dyck strings use the same length
IND_STEPS = int(os.environ.get("IND_STEPS", 3000))
DYCK_STEPS = int(os.environ.get("DYCK_STEPS", 6000))
BATCH = 64

def make_model(d_vocab):
    cfg = HookedTransformerConfig(
        n_layers=2, d_model=128, n_heads=4, d_head=32,
        n_ctx=N_CTX, d_vocab=d_vocab, act_fn="relu",
        normalization_type="LN", device=DEVICE, seed=0,
    )
    return HookedTransformer(cfg)

def train(model, sampler, steps, tag):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for step in range(steps):
        tokens = sampler(BATCH).to(DEVICE)
        loss = model(tokens, return_type="loss")
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0 or step == steps - 1:
            print(f"[{tag}] step {step:5d}  loss {loss.item():.4f}")
    return model

# --- task A: copy / repeat (vocab 0=BOS, 1..V-1 random) ---------------------
V_IND = 30
def induction_sampler(b):
    seq = torch.randint(1, V_IND, (b, K))
    bos = torch.zeros(b, 1, dtype=torch.long)
    return torch.cat([bos, seq, seq], dim=1)        # [BOS, s, s]

# --- task B: Dyck-2 (0=BOS, 1='(' 2=')' 3='[' 4=']') ------------------------
OPEN = {"(": 1, "[": 3}
CLOSE = {"(": 2, "[": 4}
def random_dyck2(L):
    seq, stack = [], []
    for i in range(L):
        gap = (L - i) - len(stack)               # remaining - stack size (always even)
        if len(stack) == 0 or (gap >= 2 and random.random() < 0.5):
            t = "(" if random.random() < 0.5 else "["
            stack.append(t); seq.append(OPEN[t])
        else:
            t = stack.pop(); seq.append(CLOSE[t])
    return seq
def dyck_sampler(b):
    rows = [[0] + random_dyck2(N_CTX - 1) for _ in range(b)]
    return torch.tensor(rows, dtype=torch.long)

def bracket_pairs(row):
    """Return (closer_pos, opener_pos) pairs for a token row (BOS at 0)."""
    pairs, stack = [], []
    for i, t in enumerate(row):
        if t in (1, 3):
            stack.append(i)
        elif t in (2, 4):
            pairs.append((i, stack.pop()))
    return pairs

# --- train both -------------------------------------------------------------
print("Training induction model...")
ind_model = train(make_model(V_IND), induction_sampler, IND_STEPS, "ind")
print("Training Dyck-2 model...")
dyck_model = train(make_model(5), dyck_sampler, DYCK_STEPS, "dyck")

# --- find the induction head (offset-diagonal stripe) -----------------------
ind_sample = induction_sampler(1)
with torch.no_grad():
    _, ind_cache = ind_model.run_with_cache(ind_sample)
ind_score = torch.zeros(2, 4)
for layer in range(2):
    p = ind_cache["pattern", layer][0]
    ind_score[layer] = p.diagonal(offset=-(K - 1), dim1=-2, dim2=-1).mean(-1)
iL, iH = divmod(ind_score.argmax().item(), 4)
ind_attn = ind_cache["pattern", iL][0, iH].numpy()
print(f"Induction head: L{iL}H{iH}  score {ind_score.max().item():.3f}")

# --- find the matching head (attends closer -> its opener) ------------------
dyck_sample = dyck_sampler(1)
pairs = bracket_pairs(dyck_sample[0].tolist())
with torch.no_grad():
    _, dyck_cache = dyck_model.run_with_cache(dyck_sample)
match_score = torch.zeros(2, 4)
for layer in range(2):
    p = dyck_cache["pattern", layer][0]               # [head, q, k]
    s = torch.stack([p[:, c, o] for c, o in pairs], dim=0).mean(0)  # [head]
    match_score[layer] = s
mL, mH = divmod(match_score.argmax().item(), 4)
dyck_attn = dyck_cache["pattern", mL][0, mH].numpy()
print(f"Matching head:  L{mL}H{mH}  score {match_score.max().item():.3f}")

# --- plot side by side ------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].imshow(ind_attn, cmap="viridis")
axes[0].set_title(f"Induction model — L{iL}H{iH}\n(copy task: offset-diagonal stripe)")
axes[0].set_xlabel("key position"); axes[0].set_ylabel("query position")

axes[1].imshow(dyck_attn, cmap="viridis")
# overlay ground-truth matches: closer (query, y) -> opener (key, x)
xs = [o for _, o in pairs]; ys = [c for c, _ in pairs]
axes[1].scatter(xs, ys, s=40, facecolors="none", edgecolors="red", linewidths=1.2,
                label="true match (closer→opener)")
axes[1].legend(loc="upper right", fontsize=8)
axes[1].set_title(f"Dyck-2 model — L{mL}H{mH}\n(matching: closers attend to their opener)")
axes[1].set_xlabel("key position"); axes[1].set_ylabel("query position")

fig.tight_layout()
fig.savefig("contrast.png", dpi=120, bbox_inches="tight")
print("Wrote contrast.png")
