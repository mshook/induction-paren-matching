"""
The counting half of the stack mechanism.

contrast.py showed the MATCHING head (closers attend to their opener, in arcs).
Those arcs are only findable if the model already knows the nesting DEPTH at
each position. This script shows the model builds exactly that:

  1. a linear probe recovers ground-truth bracket depth from the residual
     stream, with R^2 jumping after the first block (cf. Yao et al. 2021, where
     layer 1 computes depth);
  2. decoded depth tracks the true stack height along a sample string;
  3. the "counting" head (diffuse, near-uniform attention) is a DIFFERENT head
     from the sharp matching head -- the two halves of the stack mechanism.

Run:  python depth_probe.py
Needs: pip install torch transformer_lens matplotlib numpy
"""

import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer, HookedTransformerConfig

torch.manual_seed(0)
random.seed(0)
DEVICE = "cpu"
N_CTX = 41                                  # BOS + 40 bracket tokens
DYCK_STEPS = int(os.environ.get("DYCK_STEPS", 6000))
BATCH = 64

# --- model + Dyck-2 data (same setup as contrast.py) ------------------------
def make_model(d_vocab=5):
    cfg = HookedTransformerConfig(
        n_layers=2, d_model=128, n_heads=4, d_head=32,
        n_ctx=N_CTX, d_vocab=d_vocab, act_fn="relu",
        normalization_type="LN", device=DEVICE, seed=0,
    )
    return HookedTransformer(cfg)

OPEN = {"(": 1, "[": 3}
CLOSE = {"(": 2, "[": 4}
def random_dyck2(L):
    seq, stack = [], []
    for i in range(L):
        gap = (L - i) - len(stack)
        if len(stack) == 0 or (gap >= 2 and random.random() < 0.5):
            t = "(" if random.random() < 0.5 else "["
            stack.append(t); seq.append(OPEN[t])
        else:
            t = stack.pop(); seq.append(CLOSE[t])
    return seq
def dyck_sampler(b):
    return torch.tensor([[0] + random_dyck2(N_CTX - 1) for _ in range(b)],
                        dtype=torch.long)

def depth_of(row):
    """Nesting depth (open brackets) after each token; BOS=0."""
    d, out = 0, []
    for t in row:
        d += (t in (1, 3)) - (t in (2, 4))
        out.append(d)
    return out

def bracket_pairs(row):
    pairs, stack = [], []
    for i, t in enumerate(row):
        if t in (1, 3): stack.append(i)
        elif t in (2, 4): pairs.append((i, stack.pop()))
    return pairs

# --- train ------------------------------------------------------------------
print("Training Dyck-2 model...")
model = make_model()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
for step in range(DYCK_STEPS):
    tokens = dyck_sampler(BATCH).to(DEVICE)
    loss = model(tokens, return_type="loss")
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 500 == 0 or step == DYCK_STEPS - 1:
        print(f"  step {step:5d}  loss {loss.item():.4f}")

# --- collect residual stream + ground-truth depth ---------------------------
def collect(n):
    rows = [[0] + random_dyck2(N_CTX - 1) for _ in range(n)]
    toks = torch.tensor(rows, dtype=torch.long)
    with torch.no_grad():
        _, cache = model.run_with_cache(toks)
    dep = np.array([depth_of(r) for r in rows], dtype=np.float32)     # [n, pos]
    sites = {
        "embeddings":    cache["resid_pre", 0].numpy(),
        "after block 0": cache["resid_post", 0].numpy(),
        "after block 1": cache["resid_post", 1].numpy(),
    }
    return rows, dep, sites, cache

train_rows, train_dep, train_sites, _ = collect(256)
test_rows,  test_dep,  test_sites,  test_cache = collect(64)

# --- linear depth probe per site --------------------------------------------
def fit(X, y):
    Xa = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    w, *_ = np.linalg.lstsq(Xa, y, rcond=None)
    return w
def predict(X, w):
    return np.concatenate([X, np.ones((len(X), 1))], axis=1) @ w
def r2(y, pred):
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()

results, probes = {}, {}
for name in train_sites:
    Xtr = train_sites[name].reshape(-1, 128); ytr = train_dep.reshape(-1)
    Xte = test_sites[name].reshape(-1, 128);  yte = test_dep.reshape(-1)
    w = fit(Xtr, ytr)
    results[name] = r2(yte, predict(Xte, w))
    probes[name] = w
    print(f"depth-probe R^2  [{name:>13}] = {results[name]:.3f}")

best = max(results, key=results.get)

# --- counting head (diffuse) vs matching head (sharp) -----------------------
sample = test_cache
def attn_entropy(p):                       # p: [head, q, k], mean over queries
    p = p.clamp_min(1e-9)
    ent = -(p * p.log()).sum(-1)           # [head, q]
    return ent.mean(-1)                    # [head]
ent = torch.stack([attn_entropy(test_cache["pattern", l][0]) for l in range(2)])
cL, cH = divmod(ent.argmax().item(), 4)    # most uniform = counting candidate

pairs0 = bracket_pairs(test_rows[0])
mscore = torch.zeros(2, 4)
for l in range(2):
    p = test_cache["pattern", l][0]
    mscore[l] = torch.stack([p[:, c, o] for c, o in pairs0], 0).mean(0)
mL, mH = divmod(mscore.argmax().item(), 4)
print(f"counting head (diffuse): L{cL}H{cH}   matching head (sharp): L{mL}H{mH}")

# --- plot -------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

names = list(results); vals = [results[n] for n in names]
axes[0].bar(names, vals, color=["#bbb", "#4c78a8", "#3a5e8c"])
for i, v in enumerate(vals):
    axes[0].text(i, v + 0.01, f"{v:.2f}", ha="center")
axes[0].set_ylim(0, 1.05)
axes[0].set_ylabel("depth-probe R$^2$ (test)")
axes[0].set_title("Bracket depth becomes linearly decodable\nafter the first block")

row0 = test_rows[0]
true_d = depth_of(row0)
pred_d = predict(test_sites[best][0], probes[best])
axes[1].step(range(len(true_d)), true_d, where="mid", label="true depth", linewidth=2)
axes[1].plot(range(len(pred_d)), pred_d, "o-", ms=3, color="crimson",
             label=f"decoded ({best})")
axes[1].set_xlabel("position"); axes[1].set_ylabel("nesting depth")
axes[1].set_title("Decoded depth tracks the stack height")
axes[1].legend(loc="upper right", fontsize=9)

fig.tight_layout()
fig.savefig("depth_probe.png", dpi=120, bbox_inches="tight")
print("Wrote depth_probe.png")
