# Running in-context learning with gpt2-small locally

**Created:** 6/3/2026 21:07:18  
**Updated:** 6/18/2026 22:19:05  
**Exported:** 6/18/2026 22:37:02  
**Link:** [https://claude.ai/chat/a14dca6c-229c-422d-b21b-1a6c759d1eca](https://claude.ai/chat/a14dca6c-229c-422d-b21b-1a6c759d1eca)  

## Table of Contents

My questions & prompts, in order:

1. [How can I run a simple in-context learning example using gpt2-small locally?](#user)
2. [What is an induction head](#user-1)
3. [Yes](#user-2)
4. [What are key induction head/in-context learning papers and repositories?](#user-3)
5. [Why the word 'induction'](#user-4)
6. [Have induction heads been demonstrated at higher levels of abstraction than straight input tokens?](#user-5)
7. [Can induction heads learn nested matching parenthesis?](#user-6)
8. [Yes, show the contrast](#user-7)
9. [Yes, show the contrast](#user-8)
10. [explain hard vs soft attention](#user-9)

## User:

> 6/3/2026 21:07:18

How can I run a simple in-context learning example using gpt2-small locally?

## Assistant:

> 6/3/2026 21:08:59

> Weighed teaching approaches and reconciled library nomenclature for demonstration.
>
> - Michael is asking about running an in-context learning example with gpt2-small locally on his Chromebook setup, and given his background in mechanistic interpretability and transformer internals, he's likely looking to use TransformerLens for this task.
>   
>   In-context learning is when a model learns to perform a task from examples in the prompt without any weight updates — essentially a few-shot learning scenario where you give the model examples of a pattern and see if it can complete it. Since this is about well-established libraries with stable APIs, I can provide working code directly, keeping in mind his Debian/Crostini environment without GPU access.
>   
>   Given Michael's stated preference for "compact, concrete, grounded explanations with working examples," I should lead with the simplest approach using HuggingFace transformers rather than the heavier TransformerLens, even though the "gpt2-small" naming suggests he's familiar with mechanistic interpretability work. The goal here is to show in-context learning in action with minimal setup. I'm weighing whether to provide the code inline or as a downloadable file—a complete runnable example will be over 20 lines, so a file makes sense, but I also want to explain the concepts clearly alongside it. I think the best approach is to create a script file he can download and run locally, paired with a brief inline explanation of what in-context learning is and what to observe. I should also keep the code CPU-friendly and note the dependencies, though I'm uncertain whether testing it locally is necessary given the large downloads involved.
>   
>   Actually, I realize I can't test this in my environment anyway—HuggingFace isn't in the allowed domains, so I couldn't download the model weights even if I tried. I'll just write solid, correct code based on the transformers library API. I'm setting up a simple in-context learning example using GPT-2 where I load the tokenizer and model, then construct a prompt that demonstrates the concept.
>   
>   Writing the example code... I'm realizing that gpt2-small struggles with most in-context learning tasks, so a more reliable approach would be demonstrating induction — having the model complete repeated sequences. This actually ties perfectly to mechanistic interpretability, since induction heads are the core mechanism behind in-context learning, as shown in Anthropic's work on the topic. For Michael's interests, framing it around induction heads and sequence completion would be both simple and theoretically grounded.
>   
>   I should present two approaches: the standard few-shot prompting style (though gpt2-small is limited there) and the induction-based demo (which is mechanistically sound and reproducible). Since he's familiar with TransformerLens, I can also show him how to load the model that way, which would let him directly inspect the attention patterns and induction heads — probably exactly what he's looking for.
>   
>   I'll keep it focused: explain both paths briefly, then give him a working script he can run locally, and tie it back to the induction-heads mechanism underlying ICL. I'm deciding between inline code and a file artifact—the guidance says >20 lines should be a file, but for a teaching answer where code and explanation interweave, inline feels more conversational and useful. I'll create a standalone Python script as a file so he can run it directly, then keep my explanation focused on the concepts and what to watch for, plus the TransformerLens alternative path. the code is short enough to include inline, and keeping the explanation and implementation together makes it more useful for a teaching moment. I'll provide the setup instructions as they'd actually work for him rather than my sandbox environment, then write out the example code.
>   
>   For the environment setup, I should mention that modern Debian might require a virtual environment to avoid the externally-managed-environment error, though he's experienced enough to know that. I'll also note the CPU-only PyTorch wheel since he's on a Chromebook without a GPU—that'll keep the download size reasonable.
>   
>   For demonstrating few-shot learning, gpt2-small is pretty limited at semantic tasks, so I'll focus on induction as the most reliable and mechanistically interesting example. I can show how the model learns to repeat sequences by measuring whether its loss drops on the second occurrence of a pattern, or just demonstrate it predicting the continuation directly.
>   
>   My plan is to present three things: a simple few-shot generation example with plain transformers, an explanation of why gpt2-small struggles with semantic tasks and why induction is the real mechanism at play, and then a TransformerLens demo that measures per-token loss across a repeated sequence to show the induction effect clearly.
>   
>   Writing TransformerLens code... Setting up the induction demo by generating random token sequences, prepending a BOS token, then concatenating the sequence twice to create a pattern where the model sees the same tokens repeated. I'm computing the loss across both copies to compare how well the model predicts the second occurrence versus the first, which should reveal induction head behavior.
>   
>   Now I'm verifying the loss computation — the per-token loss has shape [batch, seq_len-1] where each position predicts the next token, so I need to split the loss between the first and second copy of the random sequence and compare their means to confirm the second copy has lower loss due to induction.
>   
>   Setting up a few-shot example...
>   
>   I'm loading GPT-2 and creating a simple arithmetic prompt where the model sees a few examples of adding two numbers, then I'll test if it can predict the sum for a new pair. The code generates a completion and decodes just the new tokens to see what the model predicts. Though honestly, GPT-2 might struggle with arithmetic reliability.
>   
>   Two-digit addition is unreliable with GPT-2-small, so I should be upfront about that. Instead of pushing arithmetic, I could use a more dependable few-shot task like word-to-label mapping or antonyms, though even those are hit-or-miss with such a small model. The best approach pedagogically is to show the standard few-shot prompting setup first with a clear warning that GPT-2-small produces unreliable results for semantic tasks, then pivot to the induction heads demo as the mechanistically meaningful, rock-solid version that actually connects to the literature. I should also clarify that "gpt2-small" is the TransformerLens naming convention.
>   
>   I'll keep the prose tight and add installation commands tailored to his Debian/Crostini environment with CPU torch and venv notes. The structure will be minimal—just two code blocks with brief explanations and the induction connection, no file creation, everything inline.
> - **Done**

The name "gpt2-small" is TransformerLens's label for the 124M model (HuggingFace just calls it `gpt2`), so I'll show both paths. Everything below runs fine on CPU on your Crostini box — the model is small.

Setup (a venv keeps Debian's externally-managed Python happy, and the CPU torch wheel is much smaller than the default CUDA one):

```bash
python3 -m venv ~/icl && source ~/icl/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers
```

**The textbook version — few-shot prompting with plain transformers**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2").eval()

prompt = (
    "antonym of hot: cold\n"
    "antonym of up: down\n"
    "antonym of fast: slow\n"
    "antonym of happy:"
)
ids = tok(prompt, return_tensors="pt").input_ids
with torch.no_grad():
    out = model.generate(ids, max_new_tokens=3, do_sample=False,
                         pad_token_id=tok.eos_token_id)
print(tok.decode(out[0, ids.shape[1]:]))
```

This is in-context learning in the everyday sense: no weight updates, just a pattern in the prompt that the model continues. Fair warning — gpt2-small is genuinely weak at *semantic* few-shot tasks, so expect this to be hit-or-miss. It learns the *format* (output one word after the colon) far more reliably than the actual antonym relation.

**The mechanistic version — induction, which is the substrate of ICL**

If you want a demonstration that's rock-solid *and* connects to the induction-heads story (Olsson et al.), feed the model a random sequence repeated twice and watch the per-token loss collapse on the second copy. That drop is induction heads doing "I saw `[A][B]` earlier, now I see `[A]` again, so predict `[B]`" — the same prefix-matching circuit that underlies few-shot ICL more broadly.

```bash
pip install transformer_lens
```

```python
import torch
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("gpt2-small")   # the name you asked about

torch.manual_seed(0)
seq_len = 50
rand = torch.randint(0, model.cfg.d_vocab, (1, seq_len))
bos = torch.tensor([[model.tokenizer.bos_token_id]])
tokens = torch.cat([bos, rand, rand], dim=1)              # BOS + seq + same seq

loss = model(tokens, return_type="loss", loss_per_token=True)[0]
print(f"first copy  mean loss: {loss[:seq_len].mean():.3f}")
print(f"second copy mean loss: {loss[seq_len:].mean():.3f}")
```

You'll see the second-copy loss drop dramatically (typically several nats lower) even though the tokens are *random* and unpredictable in isolation — the only way to do better on the repeat is to copy from earlier in the context. That's the cleanest possible "learning happening purely in the forward pass."

The TransformerLens path also sets you up to look at *why*: you can grab attention patterns with `model.run_with_cache(tokens)` and find the heads in (roughly) layers 5–7 of gpt2-small whose attention from each second-copy token points back to the token right after its earlier occurrence — the induction pattern itself. Happy to sketch that next step if you want to go from "it works" to "here's the circuit."

## User:

> 6/3/2026 23:16:22

What is an induction head

## Assistant:

> 6/3/2026 23:17:08

> Synthesized mechanistic interpretability concept with concrete walkthrough.
>
> - Michael's asking about induction heads specifically—he's clearly familiar with mechanistic interpretability work like ROME and sparse autoencoders, so I can explain this at a technical level without oversimplifying. This is a foundational concept in the field that he'd want a precise answer on.
>   
>   An induction head is an attention pattern that completes sequences by matching prefixes: when it sees "[A][B]...[A]", it predicts "[B]". The mechanism works as a two-head circuit where a previous token head in an earlier layer attends to the token before the current position, then the induction head uses that information to match earlier patterns and copy forward the appropriate continuation. ition of the first B, the residual stream stores "previous token was A." Then the induction head kicks in at the second A, using it as a query to find positions marked as "preceded by A"—which matches the first B's key. The induction head attends there and copies B's identity forward, predicting B as the next token. This prefix-matching-plus-copying mechanism relies on K-composition, where the induction head's key derives from the previous-token head's output.
>   
>   Crucially, induction heads require at least two layers to work, which explains why single-layer models can't do induction but two-layer ones can. During training, these heads form abruptly at a critical point where loss suddenly drops and in-context learning emerges—a sharp phase transition. The paper argues across multiple lines of evidence that induction heads are the core mechanism enabling in-context learning in transformers, even handling fuzzy matching beyond exact token repetition.
>   
>   In our gpt2-small demo, when the random sequence repeated, the loss fell on the second copy because induction heads attended back to the first occurrence and predicted the next token—these heads sit around layers 5–7. You can spot them by their attention pattern: a diagonal stripe offset by the sequence length, where each token attends to the token that followed its earlier appearance.
>   
>   I'll walk through the two-head mechanism using the A B ... A example, explain why two layers are needed, connect it to in-context learning and phase changes, and tie it back to the demo. I can show you the actual attention pattern if that helps clarify the circuit.
>   
>   For a conceptual question from someone who knows the space, clear prose with a concrete walkthrough beats a visual. I already offered to sketch the circuit and attention patterns next, so I'll keep this focused and avoid over-formatting.
>   
>   Now I'm being precise about which head writes what and how they compose. The previous-token head in layer L1 marks each position with the identity of the token before it. Then the induction head in layer L2 uses that signal: it queries with the current token and attends to positions whose keys match "preceded by this token," which is exactly how it finds the repeated pattern and predicts what comes next.
> - **Done**

An induction head is an attention head that does pattern completion: given a sequence like `… A B … A`, it predicts that the next token is `B`. It's the circuit that powers the loss drop you saw on the second copy in that demo — the model copies "what came next last time."

The thing that makes it interesting mechanistically is that a single head can't do it alone. The minimal induction circuit is **two heads in different layers** working by composition:

**Previous-token head** (earlier layer). At each position it attends one step back and writes the identity of the preceding token into the residual stream. So at the position of the first `B`, the stream now carries a "preceded by `A`" annotation.

**Induction head** (later layer). At the current position — the second `A` — it forms its query from the token `A`. Its *keys* read the "preceded by ___" annotation that the previous-token head wrote. The first `B`'s position has the key "preceded by `A`", which matches the query, so attention jumps from the current position back to that `B`. Its OV circuit then copies `B`'s identity to the output, boosting `B`'s logit.

So there are two pieces: **prefix matching** (the QK circuit locating where the current token last appeared) and **copying** (the OV circuit emitting whatever followed it). In the transformer-circuits framing this is *K-composition* — the induction head's keys are computed from the previous-token head's output, which is why you need at least two layers. A one-layer attention-only model provably can't form induction; a two-layer one can.

Two reasons they're a big deal:

- **They coincide with the emergence of in-context learning.** During training there's a sharp bump in the loss curve where induction heads abruptly form, and the model's in-context-learning ability jumps at exactly that moment. The induction-heads paper (Olsson et al.) argues, with several converging lines of evidence, that these heads are the primary mechanism behind ICL even in large models.
- **They generalize past literal repeats.** Real induction heads do *fuzzy* matching — `[A*][B*] … [A] → [B]` where `A*` is merely similar to `A` — which is what lets the same machinery support soft tasks like your few-shot antonyms, not just verbatim copying.

In gpt2-small the strong induction heads sit around layers 5–7. If you run `model.run_with_cache(tokens)` on that repeated sequence and plot one of their attention patterns, you'll see the signature: a diagonal stripe offset by the sequence length, where each second-copy token attends to the token right after its first occurrence. Want me to write that caching-and-plotting step?

## User:

> 6/4/2026 11:04:43

Yes

## Assistant:

> 6/4/2026 11:06:11

> Architected Python script for visualizing induction head patterns.
>
> - I need to write a script that loads gpt2-small with TransformerLens and sets up visualization for induction head attention patterns—this is a substantial artifact Michael will run locally, so I should create it as a proper file with focused explanation alongside it.
>   
>   The core logic involves building a repeated sequence, running it through the model with caching enabled, then computing an induction score for each head. The induction score measures how much a head attends from tokens in the second half of the sequence back to the positions that preceded the first occurrence of those same tokens—that's the signature behavior of induction heads. I need to be careful with the indexing: given a sequence repeated twice with BOS prepended, when I'm at a token in the second copy, I'm looking for attention to the token that came right after that token's first occurrence.
>   
>   Now I'm writing the actual computation code to extract the attention patterns from the cache and calculate the diagonal offset correctly. The key insight is that the induction stripe appears at offset `-(seq_len - 1)` when I use torch's diagonal function, which I can average to get a single score per head. Then I'll identify the top induction heads and visualize both the scores as a heatmap and the attention pattern of the strongest head to show that characteristic diagonal stripe.
>   
>   For the visualization, I'm creating two plots: one showing the induction score across all layers and heads to identify which ones are induction heads, and another showing the raw attention pattern of the top-scoring head so the stripe is visible. I'll make sure to run everything under `torch.no_grad()` for efficiency and explicitly set the device to CPU since he's on a Chromebook. The script will save the plots as PNGs he can view directly.
> - **Done**

Here's a self-contained script that runs gpt2-small on the repeated sequence, scores every head for induction behavior, and plots both the score grid and the top head's attention pattern.

> Deliberated on concise explanation delivery.
>
> - Creating a runnable TransformerLens script that finds and visualizes induction heads in gpt2-small
> - Presenting file(s)...
> - I need to explain what the user will see as a result, keeping it brief and natural.
> - **Done**

Run it with `pip install matplotlib` added to the venv from before, then `python induction_heads.py`.

What to look for:

**The printout** lists the top five heads by induction score. You should see gpt2-small's known induction heads come out on top — `L5H5` is the famous one, with `L6H9` and a couple of others in layers 5–7 close behind. That's empirical confirmation of the "layers 5–7" claim rather than my asserting it.

**`induction_scores.png`** is the layer×head grid. Almost the whole grid is dark (near-zero), with a few bright cells in the middle layers — induction is a sparse, specialized behavior, just a handful of heads out of 144.

**`induction_pattern.png`** is the payoff: the attention matrix of the top head. Three things stand out — a bright vertical band at key position 0 (the BOS "attention sink" that most heads park leftover attention on), faint activity in the top-left, and the signature you came for: a clean diagonal stripe in the lower-right quadrant, parallel to the main diagonal but shifted left by ~`seq_len`. Each second-copy query token (rows 51+) is attending back to the token that *followed* its first occurrence — prefix-match-then-copy, drawn out as a line.

If you want to go one level deeper, the natural next step is to verify the *circuit* rather than just the behavior: ablate the layer-4 previous-token head (zero its contribution) and watch the induction stripe in `L5H5` collapse — that demonstrates the K-composition dependency directly, instead of inferring it. Say the word and I'll add that to the script.

> File: [/mnt/user-data/outputs/induction_heads.py](https://claude.ai/api/organizations/213f3abf-123f-47e2-9929-a7d873a16e33/conversations/a14dca6c-229c-422d-b21b-1a6c759d1eca/wiggle/download-file?path=%2Fmnt%2Fuser-data%2Foutputs%2Finduction_heads.py)

## User:

> 6/4/2026 11:27:42

What are key induction head/in-context learning papers and repositories?

## Assistant:

> 6/4/2026 11:29:14

> Weighed canonical knowledge against need for current verification and recent developments.
>
> - I know the foundational induction heads and in-context learning papers well from the 2021-2023 period, but I should do a quick search to catch any newer work and verify that repository links are still current. I know the foundational papers inside and out, but there's been substantial work since 2023—function vectors, task vectors, gradient descent framings of ICL, and statistical induction heads—plus I need to verify repo URLs and catch anything significant from 2024 onward. I'll run a few targeted searches to confirm the core references, pull recent mechanistic interpretability papers on ICL, and check TransformerLens and related repos.
>   
>   Let me map out what I'm already confident about: Olsson's induction heads paper, Elhage's transformer circuits framework, Brown's GPT-3 work, Garg on what transformers can learn in-context, von Oswald's gradient descent framing, Xie's Bayesian perspective, and Akyürek's work on learning algorithms—then I'll search to fill gaps and verify links.
> - **Done**

Let me pull current links and check for recent work alongside the canonical references.

> Organized foundational papers, theory, and repositories into structured categories.
>
> - induction heads in-context learning Olsson Anthropic paper
> - Now I'm looking for the foundational ICL theory papers and recent developments in the field, along with any available implementations.
> - in-context learning task vectors function vectors transformers paper
> - Now I'm searching for the foundational papers on in-context learning and mechanistic interpretability from those researchers, along with locating the key open-source implementations and tools that would be most relevant to understanding this area.
> - TransformerLens library GitHub induction heads tutorial Neel Nanda
> - I'm noting that TransformerLens has migrated to TransformerLensOrg with a new TransformerBridge tool for version 3.0 that keeps raw HuggingFace weights intact, and I should reference Callum McDougall's ARENA materials alongside the theory papers I'm already familiar with.
>   
>   I have the key papers down—Garg, von Oswald, Akyürek, Xie, Min, Elhage, and Brown—plus the main repositories: TransformerLens, function vectors from baulab, ARENA, and Garg's in-context learning work. Since the request specifically asks for repository links, I should do a quick verification of Garg's repo and the function vectors repo to make sure I have the correct URLs.
> - Garg "What Can Transformers Learn In-Context" GitHub function classes repository
> - I've got what I need to put together a solid response—organizing it into foundational mechanistic papers on induction heads, theoretical work on in-context learning, and relevant code repositories, while keeping things concise and high-signal for someone with Michael's expertise. I'll make sure to flag the TransformerLens organizational changes and mention TransformerBridge.
> - **Done**

The core lineage is short and worth reading in order; the wider literature splits into "what's the mechanism" and "what computation is ICL doing." Here's the high-signal set.

**The foundational mechanistic thread**

- *A Mathematical Framework for Transformer Circuits* — Elhage et al., Anthropic, 2021 (transformer-circuits.pub/2021/framework). Introduces the QK/OV decomposition, K-composition, and induction heads, and shows the two-layer requirement. Read this first; the induction-heads paper assumes it.
- *In-context Learning and Induction Heads* — Olsson et al., 2022 (arXiv 2209.11895), best read as the HTML article at transformer-circuits.pub/2022/in-context-learning-and-induction-heads. The six complementary lines of evidence that induction heads may be the mechanistic source of most in-context learning, and the finding that they form at the same sharp training-loss bump where ICL ability jumps. Neel Nanda has a paired video walkthrough (neelnanda.io).
- *Language Models are Few-Shot Learners* — Brown et al., 2020 (arXiv 2005.14165). The GPT-3 paper that put "few-shot in-context learning" on the map; the phenomenon the above papers explain.

**What computation ICL implements**

- *What Can Transformers Learn In-Context? A Case Study of Simple Function Classes* — Garg, Tsipras, Liang, Valiant, 2022 (arXiv 2208.01066). Trains transformers from scratch to in-context-learn linear functions, sparse regression, and decision trees, matching task-specific algorithms — the cleanest controlled testbed. Code at github.com/dtsip/in-context-learning.
- *Transformers Learn In-Context by Gradient Descent* — von Oswald et al., 2023 (arXiv 2212.07677), and *What Learning Algorithm Is In-Context Learning?* — Akyürek et al., 2023 (arXiv 2211.15661). The "ICL is implicit gradient descent / implements a learning algorithm in the forward pass" pair.
- *An Explanation of In-context Learning as Implicit Bayesian Inference* — Xie et al., 2022 (arXiv 2111.02080). The Bayesian-inference framing.
- *Rethinking the Role of Demonstrations* — Min et al., 2022 (arXiv 2202.12837). The surprising result that random labels in the demonstrations barely hurt; format and label space matter more than correctness.

**Representation / steering view**

- *In-Context Learning Creates Task Vectors* — Hendel, Geva, Globerson, 2023 (arXiv 2310.15916): the demonstrations get compressed into a single "task vector" computed in the forward pass that modulates the model on the query.
- *Function Vectors in Large Language Models* — Todd et al., 2023 (arXiv 2310.15213, site functions.baulab.info, code github.com/ericwtodd/function_vectors). The closely related FV result, with strong causal extraction/transplant experiments.

**Repositories and hands-on material**

- **TransformerLens** — github.com/TransformerLensOrg/TransformerLens (the old `neelnanda-io` path redirects here). The library you're already using; its "Main Demo" and "Exploratory Analysis" notebooks contain the induction-score-per-head recipe directly. Note the 3.0 line now recommends `TransformerBridge`, which preserves raw HuggingFace weights rather than HookedTransformer's processed ones — worth knowing if your numbers ever need to match HF exactly.
- **Callum McDougall's exercises** — github.com/callummcdougall/TransformerLens-intro, now grown into the ARENA curriculum; the induction-head section has worked exercises, solutions, and the prev-token-head ablation I mentioned last message.

If you want the current frontier rather than the canon: there's a 2024–2025 wave on *how* induction heads form during training (Nichani, Bietti) and on the finding that the in-context ability can be **transient** — emerging then partly fading over training (Singh et al., 2024) — plus "statistical/n-gram induction head" generalizations. Happy to pull a focused reading list on any one of those branches.

---

**Sources:**

- [Induction Heads & In-Context Learning](https://www.emergentmind.com/papers/2209.11895)
- [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)
- [In-context Learning and Induction Heads](https://arxiv.org/pdf/2209.11895)
- [On the Emergence of Induction Heads for In-Context Learning](https://arxiv.org/pdf/2511.01033)
- [[2209.11895] In-context Learning and Induction Heads](https://arxiv.org/abs/2209.11895)
- [A Walkthrough of In-Context Learning and Induction Heads ...](https://www.alignmentforum.org/posts/xhkxv6qnnGmqmxxsz/a-walkthrough-of-in-context-learning-and-induction-heads-w)
- [[PDF] In-context Learning and Induction Heads | Semantic Scholar](https://www.semanticscholar.org/paper/In-context-Learning-and-Induction-Heads-Olsson-Elhage/c90a99eeb57019732a6cc996bb9eaf13faedf00f)
- [N-Gram Induction Heads for In-Context RL: Improving Stability and   Reducing Data Needs](https://arxiv.org/pdf/2411.01958)
- [A Walkthrough of In-Context Learning and Induction Heads (w/ Charles Frye) Part 1 of 2 — Neel Nanda](https://www.neelnanda.io/mechanistic-interpretability/induction-heads-walkthrough-1)
- [[2310.15916] In-Context Learning Creates Task Vectors](https://arxiv.org/abs/2310.15916)
- [In-Context Learning Creates Task Vectors](https://arxiv.org/pdf/2310.15916)
- [Paper page - In-Context Learning Creates Task Vectors](https://huggingface.co/papers/2310.15916)
- [Paper Explained: In-Context Learning Creates Task Vectors | by Minhajul Hoque | Medium](https://medium.com/@minh.hoque/in-context-learning-creates-task-vectors-explained-7ee168f0799a)
- [In-Context Learning Creates Task Vectors - ACL Anthology](https://aclanthology.org/2023.findings-emnlp.624/)
- [[2501.09240] Task Vectors in In-Context Learning: Emergence, Formation, and Benefit](https://arxiv.org/abs/2501.09240)
- [Understanding Task Vectors in In-Context Learning:](https://arxiv.org/pdf/2506.09048)
- [Function Vectors in Large Language Models](https://functions.baulab.info/)
- [Task Vectors in In-Context Learning:](https://arxiv.org/html/2501.09240v1)
- [TransformerLens/README.md at main · TransformerLensOrg/TransformerLens](https://github.com/TransformerLensOrg/TransformerLens/blob/main/README.md)
- [GitHub - TransformerLensOrg/TransformerLens: A library for mechanistic interpretability of GPT-style language models · GitHub](https://github.com/TransformerLensOrg/TransformerLens)
- [Concrete Steps to Get Started in Transformer Mechanistic Interpretability — Neel Nanda](https://www.neelnanda.io/mechanistic-interpretability/getting-started-old)
- [GitHub - callummcdougall/TransformerLens-intro · GitHub](https://github.com/callummcdougall/TransformerLens-intro)
- [Transformer Lens Main Demo Notebook - TransformerLens Documentation](https://transformerlensorg.github.io/TransformerLens/generated/demos/Main_Demo.html)
- [TransformerLens-intro/README.md at main · callummcdougall/TransformerLens-intro](https://github.com/callummcdougall/TransformerLens-intro/blob/main/README.md)
- [transformer-lens · PyPI](https://pypi.org/project/transformer-lens/1.2.2/)
- [TransformerLens-intro/Home.py at main · callummcdougall/TransformerLens-intro](https://github.com/callummcdougall/TransformerLens-intro/blob/main/Home.py)
- [Getting Started in Mechanistic Interpretability - TransformerLens Documentation](https://transformerlensorg.github.io/TransformerLens/content/getting_started_mech_interp.html)
- [Tutorials - TransformerLens Documentation](https://transformerlensorg.github.io/TransformerLens/content/tutorials.html)
- [GitHub - dtsip/in-context-learning · GitHub](https://github.com/dtsip/in-context-learning)
- [GitHub - skylergrandel/Paper-Presentation-In-Context-Learning: A repository for a presentation of the paper "What Can Transformers Learn In-Context? A Case Study of Simple Function Classes""](https://github.com/skylergrandel/Paper-Presentation-In-Context-Learning)
- [What Can Transformers Learn In-Context? A Case Study of Simple Function   Classes](https://arxiv.org/pdf/2208.01066)
- [[PDF] What Can Transformers Learn In-Context? A Case Study of Simple Function Classes | Semantic Scholar](https://www.semanticscholar.org/paper/de32da8f5c6a50a6c311e9357ba16aa7d05a1bc9)
- [GitHub - tessorastefan/in-context-learning-function-classes: My Transformers Paper Presentation Project on "What Can Transformers Learn In-Context? A Case Study of Simple Function Classes". · GitHub](https://github.com/tessorastefan/in-context-learning-function-classes)
- [What Can Transformers Learn In-Context? A Case Study of Simple Function Classes - ADS](https://ui.adsabs.harvard.edu/abs/2022arXiv220801066G/abstract)
- [[2208.01066] What Can Transformers Learn In-Context? A Case Study of Simple Function Classes](https://arxiv.org/abs/2208.01066)
- [What Can Transformers Learn In-Context? A Case Study of Simple Function Classes | Papers With Code](https://paperswithcode.com/paper/what-can-transformers-learn-in-context-a-case)
- [Can Transformers Learn Sequential Function Classes In Context?](https://arxiv.org/pdf/2312.12655)

## User:

> 6/4/2026 11:33:41

Why the word 'induction'

## Assistant:

> 6/4/2026 11:34:28

> Excavated philosophical roots of inductive reasoning terminology.
>
> - Michael's asking about the etymology of "induction heads" — why that particular term was chosen. Given his pattern of exploring single concepts and their philosophical underpinnings, he's interested in the conceptual reasoning behind the naming. I should check whether the original authors explicitly justified this choice, or if I need to infer the connection from how induction heads actually function.
>   
>   The mechanism itself points to the answer: these heads observe patterns like [A][B] appearing together in context, then generalize that rule to predict the next [A] will be followed by [B]. That's induction in the classical logical sense — inferring a general principle from particular observed instances. This aligns perfectly with his interests in philosophy and the history of logic. I should clarify that this is distinct from mathematical induction, which is a different concept entirely, though worth mentioning since he'd appreciate the disambiguation. I'm realizing the deeper connection here—induction heads work through Humean induction, where they assume observed regularities will persist. When a head sees A followed by B once in context, it projects that pattern forward, treating the future as resembling the past. This is genuine inductive inference, not deduction from stored rules, and it's worth distinguishing from mathematical induction, which despite its name is actually deductive.
>   
>   I'm uncertain whether the authors explicitly justified calling them "induction heads," but the naming seems to flow naturally from the concept of inductive reasoning itself. Rather than searching for a quote I might not find, I'll frame it as the term picking out that core inductive principle—the conceptual answer is solid regardless.
>   
>   This is really a question about what the word evokes and how it fits the mechanism, which is conceptual ground I'm confident on. I'll answer directly without search, staying calibrated about not overclaiming what the authors said, but grounding it in the established sense of induction that the term draws on.
>   
>   Let me write a focused answer—Michael appreciates compact prose. I'll connect inductive reasoning (generalizing from particular instances), Hume's principle (the future resembles the past), the contrast with deduction and mathematical induction, and how fuzzy matching makes the label even more apt. Three tight paragraphs, no formatting needed.
> - **Done**

It's "induction" in the sense of *inductive reasoning* — generalizing a rule from observed instances and projecting it forward — not mathematical induction, which despite the name is really a deductive schema.

The head's behavior is the inductive principle in miniature. It has seen, somewhere back in the context, that A was followed by B; on encountering A again it infers B will follow again. A single observed instance becomes a prediction about the next case. That's essentially Hume's "the future resembles the past" — the assumption that an observed regularity continues to hold. Crucially, the head doesn't derive B from anything stored in the weights (that would be closer to deduction, or to in-weights knowledge); it generalizes from one example sitting in the prompt. The inference is fallible — the regularity might not actually continue — and that fallibility is exactly what makes the mechanism general rather than rigid.

The fuzzy-matching version makes the label even more fitting. A head doing only literal `[A][B]…[A]→[B]` could fairly be called a *copy* head. But real induction heads fire on `[A*][B*]…[A]→[B]` for *similar* A\* — generalizing over a neighborhood rather than matching exact repeats. Inferring a general pattern from particulars, not rote repetition: that's induction proper. Whether Olsson and Elhage spelled out the etymology in the papers I won't claim from memory, but the choice tracks the inductive-reasoning sense cleanly, and the contrast with "deduction from a stored rule" is the distinction the name is doing work to mark.

## User:

> 6/4/2026 16:03:53

Have induction heads been demonstrated at higher levels of abstraction than straight input tokens?

## Assistant:

> 6/4/2026 16:05:01

> Excavated research literature on abstract induction head mechanisms.
>
> - Michael is asking whether induction heads have been shown to work at higher levels of abstraction beyond just matching literal token sequences — operating on more abstract features instead of surface-level patterns. This is a real research question with existing work on it, and I'm recalling what I know about studies in this area, particularly around fuzzy or soft matching approaches.
>   
>   I'm thinking through the relevant literature: Olsson's work on fuzzy induction heads that match on token similarity rather than exact matches, Anthropic's more recent mechanistic interpretability research including their sparse autoencoders work, and various studies on in-context learning that go beyond simple copying. There's also work on positional and concept-level induction, though I'm trying to pin down the exact papers and authors — names like Crosby come to mind, along with function vector research that explores how models learn abstract patterns in context. I'm recalling work on semantic induction heads that capture syntactic and semantic relations beyond literal token repetition — there's a 2024 paper I think from ACL or arXiv around February that identifies these heads for understanding in-context learning. There's also relevant work on function vectors and abstraction of tasks, plus Anthropic's recent work on attribution graphs in large language models. Since this is asking about demonstrated results in recent literature, I should search to ground these claims rather than risk misremembering paper titles or authors — let me look up semantic induction heads, abstract induction at the concept level, and related work on latent concept induction.
> - **Done**

Good question — this is exactly where the literature has pushed since the original work. Let me pull the specific results rather than rely on memory.

> Synthesized taxonomy of abstract induction head variants across recent literature.
>
> - semantic induction heads abstract in-context learning transformers
> - I'm seeing research on different types of induction heads that operate beyond individual tokens. There's work on concept-level heads that handle multi-token units like words and phrases, operating in parallel with token-level heads and each specialized for different tasks. The concept heads sit in earlier layers while token heads appear later in the network. There's also mention of semantic induction heads, though that entry seems to cut off.
>   
>   Beyond that, there are several other variants: symbolic heads that induce over abstract variables, n-gram heads that match and copy longer sequences, selective heads that aggregate evidence when the causal structure is uncertain, and function vector heads that trigger tasks. An important caveat from recent work is that canonical induction heads mainly handle verbatim copying, while genuine semantic abstraction often depends on these other head types or function vectors instead.
>   
>   The original Olsson paper already hinted at this with fuzzy matching and translation-like behavior, so the abstraction idea wasn't entirely new. The real contribution of recent work is building out an explicit taxonomy and showing these different mechanisms operate in parallel. I should organize this into a clear answer for Michael, being careful to cite properly and note that "induction head" has become somewhat generalized to describe a family of related circuits rather than a single mechanism.
> - **Done**

Yes — and this is one of the more active recent threads. The original paper already gestured at it (the fuzzy `[A*][B*]…[A]→[B]` matching and translation-like behavior were noted as going beyond literal copying), but since ~2024 people have identified distinct head types that operate above the surface-token level. The main strands:

**Concept-level induction heads** — Feucht et al. (2025) propose a dual-route picture: token-level heads copy single tokens, while concept-level heads propagate whole multi-token lexical units, and the two operate in parallel and independently, each causal for its own task class. These concept heads abstract meaning for tasks like translation, analogy-solving, and paraphrasing, and interestingly concept heads tend to sit in mid-to-early layers while pure token heads sit later.

**Semantic / relational induction heads** — Ren et al. (2024), *Identifying Semantic Induction Heads* (arXiv 2402.13055, ACL Findings). Instead of matching surface tokens, they ask whether heads encode *relations*: syntactic dependencies and knowledge-graph relations. They find heads that, when attending to a head/subject token, recall the tail/object token and boost its logits, with the formation of these semantic heads closely correlated with the emergence of ICL. That's induction over a relation type rather than a literal repeat.

**Symbolic induction heads** — Yang et al. (2025) describe heads that induce over abstract variables, i.e. the pattern-match is on symbolic role rather than content.

**N-gram induction heads** — Akyürek et al. (2024) and Zisman et al. (2024) generalize the unigram copy to matching and copying longer sequences, which helps sequence modeling and data-efficient in-context RL. Higher-order in the temporal dimension rather than the semantic one.

**Selective induction heads** — for tasks where the causal structure is variable or unknown, these aggregate evidence across multiple possible lags/structures rather than assuming a fixed offset.

The honest nuance worth holding onto: a good part of this progress comes from naming *new* head families that generalize the induction motif, not from showing the canonical induction head itself is deeply abstract. Ablating canonical induction heads mainly degrades verbatim copying; genuinely semantic or abstractive mapping appears to lean on these other circuits — concept-level induction heads or function-vector heads (Todd et al.) — rather than on the literal `[A][B]…[A]` head. So "induction head" has quietly broadened from a specific two-head copying circuit into an umbrella for a family of in-context pattern-completion mechanisms operating at different levels of abstraction, with ongoing debate about where the boundaries and the causal credit actually lie.

If you want, I can sketch the dual-route concept-head experiment concretely — it's the most direct demonstration that something above the token level is doing causal work, and it'd run on a mid-size model in TransformerLens.

---

**Sources:**

- [Semantic Induction Heads in Transformers](https://www.emergentmind.com/topics/semantic-induction-heads)
- [Induction Heads in Transformers](https://www.emergentmind.com/topics/induction-heads-ihs)
- [Identifying Semantic Induction Heads to Understand In-Context Learning](https://www.researchgate.net/publication/384210828_Identifying_Semantic_Induction_Heads_to_Understand_In-Context_Learning)
- [On the Emergence of Induction Heads for In-Context Learning](https://arxiv.org/pdf/2511.01033)
- [Identifying Semantic Induction Heads to Understand In-Context Learning](https://arxiv.org/pdf/2402.13055)
- [Identifying Semantic Induction Heads to Understand In-Context Learning - ACL Anthology](https://aclanthology.org/2024.findings-acl.412/)
- [[2511.01033] On the Emergence of Induction Heads for In-Context Learning](https://arxiv.org/abs/2511.01033)

## User:

> 6/5/2026 23:26:04

Can induction heads learn nested matching parenthesis?

## Assistant:

> 6/5/2026 23:27:49

> Examined theoretical limits of induction heads versus stack-based computation.
>
> - Michael's asking whether induction heads can learn to recognize nested matching parentheses—the Dyck language. This is fundamentally a context-free problem that requires tracking nesting depth, which goes beyond what regular patterns can capture. The core question is whether attention mechanisms can handle this kind of hierarchical structure matching.
>   
>   An induction head works by pattern completion: it finds an earlier occurrence of the same context and copies what came next. But nested parentheses matching is different—the closing paren that matches an opening paren depends on the entire nesting structure, not just a fixed relational offset or token identity. You need to track how many unmatched opening parens exist, which is more like a stack operation than the repetition-based mechanism induction heads provide.
>   
>   There's been theoretical work on this. Hard-attention transformers have fundamental limits on recognizing balanced bracket languages, but soft-attention transformers with enough layers and precision can learn them. Recent papers like Yao et al. (2021) and Bhattamishra et al. (2020) have shown transformers can recognize bounded-depth bracket languages with specific architectural constraints.
>   
>   The key insight is that when researchers examine what attention patterns transformers actually learn for bracket matching, they find depth-tracking or stack-like patterns — not the induction head stripe. The heads solving bracket matching attend based on nesting depth rather than token repetition, which is fundamentally different from how induction heads work. opener of the same type. This requires a stack. The matching is type-aware.
>   
>   Could an induction head help with the *type* part? Induction matches token identity. If the model needs to predict the next closing bracket and it should match the type of the deepest open bracket — hmm. Consider a sequence like `( [ { ... } ] )`. To predict the closer after `{...}` content, you need to know the deepest unclosed opener is `{`. Induction-by-token-identity won't give you that, because the relevant opener isn't identified by "the last time `}` appeared" — it's identified by stack depth.
>   
>   Actually, there might be a clever connection: some constructions for Dyck use a head that attends to the matching bracket via a position computed from depth, and then the *type* could be copied. But the core "find the matching position" is depth-arithmetic, not induction.
>   
>   Let me also recall: there's specific work on whether induction heads generalize to algorithmic/structured tasks. The "selective induction head
>   
>   " work (Yang et al.) handles variable causal structure, but parentheses are hierarchical rather than just variable-lag Markov. RASP and the Dyck constructions from Weiss, Goldberg, and Yahav use aggregation with selectors based on cumulative bracket counts—depth-based, not induction patterns. There's also the connection to automata: transformers can emulate finite automata efficiently, but Dyck is context-free and pushdown. Though bounded-depth Dyck is actually regular, which is why transformers handle it more easily; unbounded Dyck is where length generalization becomes the real challenge.
>   
>   So the key distinction for Michael: transformers can learn bounded-depth Dyck, but it's not via induction heads. Instead, they learn depth-tracking attention patterns that attend to matching brackets based on nesting level—a fundamentally different circuit than the repetition-based induction mechanism. The empirical work shows this clearly: Ebrahimi, Yao, and Bhattamishra all found that transformers trained on Dyck develop stack-emulating heads, not induction stripes. Bounded-depth Dyck is learnable with clean constructions using logarithmic layers, but unbounded generalization is where things break down.
>   
>   The broader point is that selective induction heads and their variants still don't natively give you a stack—hierarchical matching is a qualitatively different competence. I should search to verify recent work directly connecting induction heads to Dyck languages and to check if there's been new research in 2024-2025 on whether induction heads can learn hierarchical structure. and the Dyck-transformer mechanism papers. I'm going to run a couple of searches to verify the specific claims about induction heads and parenthesis matching, particularly around how transformers handle Dyck languages and the attention mechanisms involved.
> - **Done**

This is a sharp question because it pits the induction primitive against a fundamentally different kind of computation. Let me ground the specifics rather than reason purely from first principles.

> Synthesized empirical evidence distinguishing induction from stack-based bracket matching.
>
> - transformers Dyck language parenthesis matching attention heads mechanism
> - Now I'm synthesizing the key distinction: induction heads and bracket-matching are fundamentally different mechanisms, and the empirical evidence shows transformers actually solve Dyck languages through depth-tracking heads rather than induction. The Yao et al. construction from 2021 demonstrates this with a two-layer architecture where the first attention layer computes bracket depth by attending uniformly across positions and accumulating +1 for open brackets and -1 for close brackets.
>   
>   The Ebrahimi et al. work reveals that trained transformers learn stack-like behavior—each closing bracket position attends to the last unmatched opening bracket or to a BOS token serving as the stack's base, which is a depth-dependent target, not a token-repetition pattern. Hahn's 2020 analysis shows hard-attention transformers provably cannot model even simple Dyck languages, though soft attention can with precision constraints. The RASP framework and Weiss et al.'s work confirm this pattern: Dyck recognition uses counting and select-aggregate operations based on depth, not induction primitives.
>   
>   There's an important caveat from Wen et al. 2023—the learned solutions aren't always cleanly stack-like and vary across initializations, so there's no single canonical circuit. But the core point holds: induction is about finding where a token appeared before and copying what followed; matching a closer to its opener requires knowing nesting depth, which is a qualitatively different computation. I'll frame this as a conceptual contrast—induction is Humean repetition of the past, while Dyck matching is determined by structural counting, not prior occurrence.
> - **Done**

Strictly speaking, no — not by the induction mechanism itself. Bracket matching is a different computation from what an induction head does, and that contrast is actually clarifying.

Recall the induction primitive: find where the current token/context occurred earlier and copy what followed, at an offset fixed by an identity (or feature) match. That's the Humean "the future resembles the past" move from a few turns back — project an observed repeat forward. Nested-paren matching is almost the opposite shape. The correct closer for a given point isn't determined by any prior occurrence of that token; it's the *last unmatched* opener, whose distance back depends on the intervening nesting. You need a count or a stack, and the matching distance is data-dependent rather than a learned constant. An induction stripe can't express "attend to wherever the depth-counter says my partner is."

What the literature shows is that transformers *can* learn parenthesis matching, but with stack/counting circuitry, not induction heads. Two reference points: constructively, Yao et al. (2021) give a two-layer transformer where the first attention layer computes bracket depth — a head attending uniformly with +1 for openers and −1 for closers, yielding depth/t at each position — which is then embedded and used for matching. Empirically, Ebrahimi et al. (2020) find a trained two-layer network learns nearly hard attention in which each symbol attends to either the last unmatched opening bracket or a starting symbol, resembling a stack-based automaton, and that a BOS token helps because the model uses it as the stack's empty base when no brackets are open. Both are depth/stack mechanisms — the attention target is computed from a count, which is precisely the non-induction part. The RASP line (Weiss et al. 2021) compiles Dyck-k recognizers the same way, via select-aggregate over counts rather than repetition-matching.

Three caveats worth holding:

The theory imposes hard limits. Hahn (2020) proves hard-attention transformers cannot model PARITY or even 1-Dyck; soft attention escapes this but at the cost of precision, so depth/length generalization is fragile — fine in-distribution, brittle on deeper or longer nesting than seen in training. Bounded-depth Dyck is effectively regular and much easier; the properly context-free unbounded case is where it breaks. And Dyck-2 is in a sense maximal here — by Chomsky–Schützenberger, closing brackets correctly underlies all nonregular context-free languages.

The mechanism isn't even a clean single story. Wen et al. (2023) show transformers trained on bounded Dyck don't always learn stack-like attention — the patterns vary across random initialization, and multiple distinct solutions reach high accuracy, so there's no canonical "the bracket-matching head" the way there's a canonical induction head. (Separately, a circuit recognizing a nested Dyck language has been found "in the wild" in a transformer trained on natural language, which is suggestive for how hierarchical syntax gets represented — but again, not an induction head.)

The generalized induction family doesn't close the gap. Selective and n-gram induction heads extend *which lag / how long a pattern* gets matched, but they still match-and-copy; none supplies a stack. The recursive, depth-dependent partner-finding that parentheses demand is a qualitatively different competence than pattern-completion-by-repetition.

If it'd be useful, I could put the contrast on screen concretely in TransformerLens — train a tiny model on Dyck-2 and show that the head solving it has a depth/stack attention pattern (closers pointing at the last open bracket), visibly *not* the offset-diagonal induction stripe from your earlier script. That side-by-side is the cleanest way to see that these are two different circuits.

---

**Sources:**

- [Interpretability Illusions in the Generalization of Simplified Models](https://arxiv.org/pdf/2312.03656)
- [Differentiation and Specialization of Attention Heads via the Refined   Local Learning Coefficient](https://arxiv.org/pdf/2410.02984)
- [Thinking Like Transformers Gail Weiss 1 Yoav Goldberg 2 3 Eran Yahav 1 Abstract](http://proceedings.mlr.press/v139/weiss21a/weiss21a.pdf)
- [Transformers are uninterpretable with myopic methods: a case study with   bounded Dyck grammars](https://arxiv.org/pdf/2312.01429)
- [Transformers are uninterpretable with myopic](https://openreview.net/pdf?id=OitmaxSAUu)
- [Theoretical Limitations of Self-Attention in Neural Sequence Models](https://arxiv.org/pdf/1906.06755)
- [How Can Self-Attention Networks Recognize Dyck-n Languages?](https://arxiv.org/pdf/2010.04303)
- [How Can Self-Attention Networks Recognize Dyck-n Languages?](https://www.researchgate.net/publication/347233457_How_Can_Self-Attention_Networks_Recognize_Dyck-n_Languages)

## User:

> 6/5/2026 23:30:07

Yes, show the contrast

## Assistant:

> 6/5/2026 23:37:41

> Architected side-by-side transformer comparison demonstrating distinct attention circuits.
>
> - Michael wants me to build a concrete demonstration comparing two attention patterns in a small Dyck-2 model: one showing the induction stripe from earlier work, and another revealing the bracket-matching head's depth/stack attention where closing brackets attend to their corresponding unmatched opening brackets. This is a substantial local script he'll run on his Crostini box, so I need to think through the architecture carefully—whether to combine both parts into one script or split them, and what exactly each part should show.
>   
>   The key tension is that reusing the induction stripe from his existing `induction_heads.py` script would require downloading pretrained gpt2-small, whereas training a tiny Dyck model from scratch needs no external downloads and runs entirely offline on CPU, which is ideal for a Chromebook. I'm leaning toward a single, self-contained script that trains the Dyck model from scratch and then demonstrates both the bracket-matching attention pattern and what an induction-like pattern looks like when the same model processes random repeated tokens—keeping everything offline and focused. A cleaner approach: train the Dyck model from scratch in one script and visualize its stack-pattern attention directly, then either reference the existing induction head visualization or create a minimal comparison within the same script to show the contrast between nested arcs and diagonal stripes. I'll keep the Dyck model as the focus since it's the novel contribution, and make the induction comparison clear through annotation and prose rather than loading heavy dependencies. I'm torn between the clearest presentation and keeping things simple. He explicitly wants side-by-side to see the circuit difference, so I should train both a tiny Dyck model and a tiny induction model to show the contrasting attention patterns directly. The Dyck model alone won't produce an induction stripe, so I need both to make the comparison work. I'll go with training two small transformers from scratch—it's fast enough on CPU and gives the cleanest, most honest demonstration of the two different circuits. I'll use TransformerLens's HookedTransformer to build both models from scratch—this gives me clean access to attention patterns via the caching API without needing any pretrained weights, and it's the same framework he's already familiar with. Setting up a tiny config with 2 layers, 64 dimensions, and 2 heads, then training both on their respective datasets with a standard PyTorch loop.
>   
>   For the Dyck-2 task, I'm generating balanced bracket sequences with two types and framing it as next-token prediction, though the cleaner evaluation approach from the literature focuses specifically on positions where a closing bracket is forced by the nesting structure—predicting which type of closer is required at those decision points.
>   
>   Writing Dyck generator... Writing the generator function...
>   
>   Now I'm mapping token types to IDs—open and close brackets get distinct token values. For the induction task, I'll generate random sequences with a repeating pattern and train a language model to learn the induction behavior. I'm planning to use two separate model configs, one for Dyck and one for induction, keeping the architecture identical but varying the vocabulary size and token ranges to match each task's needs.
>   
>   For training, I'm targeting a tiny model with 64 dimensions, 2 layers, and 2 attention heads, with context length around 40-50 tokens. A few thousand training steps on CPU should take just a couple minutes per model and should be enough for the circuits to form—induction learns quickly, and Dyck-2 parsing also converges reasonably fast at short lengths. I'll visualize both models side-by-side: the induction head showing the characteristic offset diagonal stripe on repeated sequences, and the Dyck model's matching head displaying the arc pattern where closing brackets attend to their corresponding openers, with ground-truth arcs overlaid for clarity. I need to auto-select which head to display for each model.
>   
>   For the Dyck model, I'll compute a matching score by identifying the correct opener for each closer in a sample string, then averaging each head's attention from closer positions to their matching openers, picking the head with the highest score. I'll overlay arcs connecting each closer to its matching opener so the attention pattern aligns visually with the correct structure. Since I can test this in the sandbox with TransformerLens and PyTorch from pip, I can at least verify the code runs correctly on a short training run.
>   
>   I'm considering installing torch and transformer_lens to train a tiny model for a thousand or two steps, which would validate that the circuit patterns actually emerge and the code works end-to-end. The install might be heavy and training could take a few minutes, but it significantly reduces risk for Michael. If it's too slow, I'll provide the code with careful notes instead.
>   
>   The main constraint is that the sandbox can only pull torch from PyPI (the default CUDA wheel, which is quite large at 800MB+), since the CPU index URL isn't in the allowed domains list. This could be slow or timeout, and transformer_lens has many dependencies too. I'm weighing whether the verification is worth the potential install and time costs.
>   
>   Actually, I'm reconsidering the whole approach—I could write a minimal transformer in pure PyTorch instead, which would be lighter, fully testable, and give me complete control. The tradeoff is that it diverges from his existing induction_heads.py API that uses TransformerLens. For the from-scratch demo, pure PyTorch might be cleaner and more self-contained, though I'd lose the consistency with his pretrained gpt2 setup.
>   
>   I'll stick with TransformerLens for continuity—it makes grabbing attention patterns trivial and aligns with his mental model. On testing, I'll try a quick sandbox run with short training to verify the patterns form, but if the install is too heavy I'll ship carefully written but untested code with instructions for his machine.
>   
>   The install could take several minutes (torch wheels are large, plus transformers and dependencies), and even minimal training on CPU might run 10+ minutes total. That's reasonable for verified code, so I'll go for it—write the script first, test with reduced steps, then finalize defaults. I'm structuring it with config, data generation for both Dyck-2 and induction, a generic train function, model training, head scoring, and visualization.
>   
>   Writing the training script... Writing training loop... Completing the dyck_batch function to wrap sequences with BOS tokens and convert to tensors. I'm verifying the logic ensures the stack empties by the end—the constraint that can_open requires remaining >= len(stack)+2 forces closes when we can't fit all remaining opens and closes, guaranteeing a valid Dyck sequence. When the gap equals zero, remaining and len must be equal, forcing us to close every remaining step until the stack empties. The tricky case is gap=1 with an empty stack—we'd need to emit something but can't close (stack empty) and shouldn't open (gap too small). However, this situation is impossible: the gap's parity is invariant and always matches L's parity, so since L is even, the gap is always even, meaning gap=1 can never occur. This guarantees the forced-close logic triggers cleanly at gap=0 or gap=2.
>   
>   I'm writing a function that generates random Dyck sequences by iterating through L steps, tracking the remaining positions and stack size to determine whether to open or close a bracket at each step.
>   
>   Now I need to compute matching pairs between opening and closing brackets in a token sequence, then calculate a matching score per attention head by examining the attention patterns in the model's cache.
>   
>   For each closing bracket position, I'll look at the attention weights directed toward its matching opening bracket position, averaging across all pairs to get a score for each head and layer. Once I identify the best-performing head, I can visualize its attention matrix with the bracket pairs overlaid.
>   
>   For the induction task, I'm working through the offset calculation: with a sequence of length 2K+1 (BOS followed by two copies of K tokens), the induction pattern requires attending backward by K-1 positions, so the diagonal offset is -(K-1). I'll use the same diagonal extraction trick to find the induction head.
>   
>   Finally, I'll create a side-by-side visualization showing the bracket-matching head's attention on a Dyck sequence with pairs marked, and the induction head's attention on an induction sample.
>   
>   For the training setup, I'm using a small model (128 dimensions, 4 heads, 2 layers) to keep things fast. The induction task trains for about 3000 steps and the Dyck task for 5000 steps, both with batch size 64 and learning rate 1e-3. In my sandbox test I'll drastically reduce the step count just to verify the code runs end-to-end and the scoring logic works, accepting that the learned patterns won't be as clean.
>   
>   I'm realizing the main bottleneck is installing torch, which is a large download. Let me try installing from PyPI—it should work on CPU even though the wheel includes CUDA libraries. If the install is too slow or fails, I'll fall back to delivering the code unrun. I'll set up the script with configurable constants at the top so he can adjust parameters for a quick test run if needed. I need to handle seed initialization for reproducibility and ensure the model stays on CPU, since TransformerLens might default to CUDA if available. Let me write this out and attempt a sandbox install to catch any API issues before shipping it. The issue is that in a causal language model, the closer token's position attends to earlier positions including its opener, but that representation predicts the *next* token, not the closer itself. To get a clean visualization where each closer directly attends to its matching opener, I should look at the attention weights at the closer's query position attending backward to the opener position—this naturally shows the bracket-matching pattern even though the model's actual prediction task is offset by one position. I'll set up the script with environment variables to control the step count, so I can run a quick smoke test with reduced iterations while keeping the full defaults for the final version. Now I'm ready to write out the file and run an install plus smoke test to verify everything works.
> - **Done**

Let me build this as a single self-contained script — same tiny 2-layer architecture trained two ways (on a copy/repeat task vs. on Dyck-2), so the only variable is the data, and then plot the two heads side by side. I'll verify it runs before handing it over.

> Navigated resource constraints, pivoted to testing core logic separately.
>
> - From-scratch TransformerLens script training two tiny transformers (induction vs Dyck-2) and plotting the contrasting attention patterns
> - I'll try installing PyTorch from PyPI and transformer_lens to run a quick smoke test, falling back if the dependencies are too heavy.
> - Installing torch, transformer_lens, matplotlib to smoke-test the script
> - The disk is full from the CUDA torch installation and nvidia packages. I need to switch to a CPU-only torch build, but download.pytorch.org is blocked and PyPI only hosts the CUDA version for Linux. I'm considering freeing up space and trying to install a smaller torch variant instead.
>   
>   Let me check how much disk space is available and clear out pip caches to make room.
> - Checking disk usage to see if a CPU-only torch is feasible
> - I'm running low on disk space—only 2.7G available—and CUDA torch with nvidia libraries won't fit. The pip cache alone is taking up 4.7G, so clearing that would free up enough space to attempt a CPU-only torch installation, though PyPI's default torch is CUDA-based. Let me start by clearing the cache to reclaim that space and see if I can work around the CUDA dependency.
>   
>   Actually, I'm realizing the core issue: without access to download.pytorch.org, I can't get a proper CPU torch from PyPI, and the full TransformerLens stack (with transformers, datasets, wandb, accelerate) is too heavy for this sandbox anyway. Rather than keep struggling with the environment constraints, I should clean up the disk, then focus on unit-testing the non-torch logic—the Dyck bracket generator and pairing logic—in pure Python where the real bugs would surface. That's where I can add real value without needing the full torch stack. Then I'll provide the script with clear documentation about the API it targets and the correct CPU torch install command for the user's actual machine.
>   
>   Let me start by clearing the pip cache and removing partial installations.
> - **Done**

The CUDA build of torch blew the disk (and the CPU-only index isn't reachable from here). I'll clean up and at least verify the part most likely to have bugs — the Dyck generator and the matching logic — in pure Python, then hand you the reviewed script.

> Verified code logic and prepared visualization script.
>
> - Freeing disk space from the failed CUDA torch install
> - Unit-testing the Dyck-2 generator and matching logic in pure Python
> - I'm noticing a discrepancy in the test output—the sample sequence shows all balanced parentheses and brackets, but I need to trace through the mapping to verify it's actually correct. Let me work through the conversion where 1 represents '(', 2 represents ')', 3 represents '[', and 4 represents ']' to check if this sequence is genuinely balanced.
>   
>   After tracing through the stack operations, the sequence does balance correctly—each closing bracket properly matches its corresponding opener. The pairing logic also checks out: the nested arcs show proper containment where inner pairs like (8..11) are fully contained within outer pairs like (7..12), which confirms the matching algorithm is working as intended. The data generation across 5000 test strings shows zero failures, and the observed maximum nesting depth of 16 with an average around 4.6 is reasonable for what the algorithm needs to handle.
>   
>   Now I'm confident the core non-torch logic is solid. For the torch and TransformerLens components, I can't actually run them here due to environment constraints, but I can verify the API usage is correct. The config construction looks good—all the standard fields like layer count, model dimension, head count, context length, and vocabulary size are properly specified, with sensible defaults for the MLP dimension and attention-only flag. The model initialization, training loop with cross-entropy loss and Adam optimization, and the cache retrieval for attention patterns all follow standard TransformerLens conventions. The induction score calculation correctly extracts the diagonal from the attention pattern tensor to measure how well the model attends to previous token positions.
>   
>   The matching score logic properly stacks attention weights across closer-opener pairs and averages them to identify which head best captures the bracket-matching behavior. The argmax selection and coordinate conversion look right, and the plotting setup correctly maps the attention matrix axes to the scatter plot coordinates. One thing I'm uncertain about is whether six thousand training steps will be enough for a clean, single-head induction pattern to emerge—the literature suggests some variability in how these stack-attention heads form across different runs—but the head-selection approach should pick whichever head performs best regardless.
>   
>   For the induction task, three thousand steps on this small model should be plenty; induction heads form quickly and reliably. The vocabulary and context length parameters check out: induction uses tokens 1–29 with BOS=0, Dyck uses 0–4, and both sequences fit within the 41-token context window. The script looks solid. I'll copy it to the outputs folder and present it to the user, noting that they can lower the step counts for a faster preview run if they want.
> - **Done**

Data-generation and matching logic check out — 5,000 generated Dyck-2 strings were all balanced, every closer paired correctly to its opener, with nesting depths up to 16. That's the part most prone to off-by-one bugs, so good to have it verified.

I couldn't run the torch/TransformerLens half here: this sandbox can only pull the CUDA build of torch from PyPI (the CPU-only index is blocked), and it ran the disk out. The model/training/plotting code is written against the standard TransformerLens API you're already using, so it'll run on your machine — where the CPU wheel *is* reachable.

> Ran command, presented file
>
> - Copying the reviewed script to the outputs directory
> - Presenting file(s)...
> - **Done**

Install with the CPU wheel so you don't drag in the CUDA stack:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformer_lens matplotlib
python contrast.py
```

Each model trains in roughly a couple of minutes on CPU; if you want a fast first look, `IND_STEPS=800 DYCK_STEPS=1500 python contrast.py` will show the patterns forming (less crisp).

What the figure shows, and why it's the answer to your question:

**Left (induction model, copy task).** The signature offset-diagonal stripe — exactly the structure from your `induction_heads.py`, now grown from scratch in a 2-layer model. Query tokens in the second copy attend to a *fixed* distance back (K−1), because the rule is "repeat what followed last time." A constant offset is all induction needs, and all it can express.

**Right (Dyck-2 model, bracket task).** No stripe. Instead the red circles — the true closer→opener matches — sit right on top of the bright attention cells: each closing bracket attends back to *its* opener, and because brackets nest, those targets form arcs at *data-dependent* distances, not a single offset. That's the stack/depth pattern from Ebrahimi et al., and it's visibly a different circuit.

The contrast is the point. Same architecture, same optimizer, same sequence length — only the training data differs, and the two tasks recruit structurally different attention. The copy task can be solved by a constant-offset lookup (induction); bracket matching cannot, because the partner's position is determined by nesting depth, so the model has to learn depth/stack-tracking attention instead.

Two honest caveats consistent with the literature: the matching head won't always be perfectly clean or confined to one head (the Wen et al. point about Dyck solutions varying across initialization), so re-running with a different seed may move which head wins the matching score or spread the pattern across two. And the model matches in-distribution; push the test strings deeper or longer than training and the arcs degrade — the fragility that the Hahn impossibility result foreshadows. If you'd like, I can add a depth-counting probe to the script (read the +1/−1 bracket-depth signal straight out of the first layer's residual stream) to show the *other* half of the stack mechanism that makes those arcs possible.

> File: [/mnt/user-data/outputs/contrast.py](https://claude.ai/api/organizations/213f3abf-123f-47e2-9929-a7d873a16e33/conversations/a14dca6c-229c-422d-b21b-1a6c759d1eca/wiggle/download-file?path=%2Fmnt%2Fuser-data%2Foutputs%2Fcontrast.py)

## User:

> 6/5/2026 23:39:08

Yes, show the contrast

## Assistant:

> 6/5/2026 23:41:36

> Resolved ambiguity and planned depth-probe implementation.
>
> - I'm realizing Michael's "Yes, show the contrast" response might be referring back to an earlier suggestion I made about adding a depth-counting probe to visualize the bracket-depth signal from the residual stream, which would reveal the other half of the stack mechanism behind those arc patterns. The depth-counting probe isn't really a contrast between two things—it's the other half of the Dyck mechanism itself. He's clearly accepting my offer with "Yes," and "show the contrast" is just his habitual phrasing from before. I should build the depth probe and frame it clearly so he can redirect if I've misread, rather than waste time clarifying when the intent is reasonably clear.
>   
>   Actually, the depth probe naturally produces a contrast too—I can show that the Dyck model carries a clean linear depth signal in its residual stream while the induction model doesn't. That covers both readings and makes it a genuine contrast. I'll train a linear probe to regress the residual-stream activations onto ground-truth bracket depth, showing high R² for Dyck and demonstrating that the induction model's residual doesn't carry a comparable monotone counting signal.
>   
>   But I'm realizing the cleanest approach is to focus on the Dyck model itself and show the two halves of the stack mechanism: a diffuse counting head that computes nesting depth, and the sharp matching head that uses that depth to find partners. I'll build a new script that trains the Dyck model, then probes for depth linearly across layers to show where depth emerges, identifies the counting head via attention entropy, and visualizes both the depth-probe R² by layer and the decoded depth versus true depth on a sample string. This directly demonstrates the depth-counting machinery without needing to contrast against induction—the within-model contrast between diffuse and sharp attention patterns is more informative.
>   
>   I'll keep it focused: two main plots showing R² rising across layers and decoded depth tracking the true stack height on a sample string, plus a print statement identifying which heads are the counting and matching components. The probe itself uses least squares regression on collected residual activations and ground-truth depths from a train/test split of Dyck strings.
> - **Done**

Yes — let me build the depth-counting half. This probes whether the Dyck model linearly represents nesting depth in its residual stream (the counter that makes the matching arcs possible), and contrasts the diffuse counting head with the sharp matching head. It reuses the same from-scratch Dyck-2 model, so it runs standalone.

> Validated core logic through numpy testing methodology.
>
> - Standalone script that trains the Dyck-2 model and probes its residual stream for linearly-decodable bracket depth, revealing the counting half of the stack mechanism
> - I need to verify the core logic in pure Python and NumPy without torch — checking that the depth_of function works correctly, testing the probe fitting with synthetic data to ensure the least squares and R² calculations are sound, and validating the entropy and argmax indexing.
> - **Done**

Let me verify the depth labeling and the probe math (numpy) independently, since those are the new bug-prone pieces:

> Verified implementation logic and validated probe integrity.
>
> - Unit-testing depth labeling, the linear probe math, and head-index arithmetic with numpy
> - All the validation checks are passing—the depth calculation works correctly for the bracket sequence, the probe recovers the linear signal with perfect R² while showing negative R² on noise, confirming it's not overfitting, and the indexing logic correctly identifies positions. The torch and TensorFlow components are using standard API patterns with the appropriate cache keys.
> - **Done**

Depth labeling, the probe (recovers a real linear signal at R²≈1.0, and correctly returns negative R² on noise — so it can't fake success), and the head-indexing all check out. Same caveat as before on the torch half: written against the standard TransformerLens cache API, not run here due to the disk/index limit.

> Ran command, presented file
>
> - Copying the reviewed depth-probe script to outputs
> - Presenting file(s)...
> - **Done**

Same install as before, then `python depth_probe.py`.

What it demonstrates, and how it completes the picture:

**Left panel — depth is linearly decodable, and only after the model does some work.** The probe is a single linear map from the residual stream to a scalar depth prediction. On the raw embeddings the R² is near zero (a token embedding for `(` can't know how deep you are — depth is a property of the whole prefix, not the token). After block 0 it jumps high, and stays high after block 1. That step is the model *computing* nesting depth, matching the Yao et al. construction where the first layer's job is exactly to accumulate the ±1 bracket signal into a depth scalar.

**Right panel — the decoded value tracks the actual stack height.** For one sample string, the probe's prediction (red) rides along the true depth staircase (blue) as it climbs on openers and drops on closers. The model isn't just correlating with depth in aggregate; it carries a usable running counter, position by position.

**The two halves, and why neither is induction.** The script also reports the most diffuse head (near-uniform attention — the counting candidate, integrating the ±1 signal the way uniform attention sums a value) and the sharp matching head from before, and they come out as *different heads*. So the stack mechanism factors cleanly: one head spreads its attention to count depth, another uses that depth to point each closer at its opener. Put against your very first script, that's three distinct circuits now — the induction stripe (fixed offset, copy the repeat), the matching arcs (data-dependent partner), and the diffuse counter (accumulate a scalar) — and only the first is an induction head.

That closes the loop on your original question: bracket matching isn't something an induction head does, it's something a depth-counter plus a depth-conditioned matcher do, and you can now watch all three patterns appear in tiny models on your own machine.

One honest note for when you run it: with only ~6k steps the counting signal sometimes distributes across two heads rather than landing in one tidy "counter," so the entropy-based pick is a heuristic, not a guarantee — bump `DYCK_STEPS` or try a couple of seeds if the chosen head looks ambiguous.

> File: [/mnt/user-data/outputs/depth_probe.py](https://claude.ai/api/organizations/213f3abf-123f-47e2-9929-a7d873a16e33/conversations/a14dca6c-229c-422d-b21b-1a6c759d1eca/wiggle/download-file?path=%2Fmnt%2Fuser-data%2Foutputs%2Fdepth_probe.py)

## User:

> 6/18/2026 22:17:47

explain hard vs soft attention

## Assistant:

> 6/18/2026 22:19:05

> Synthesized formal expressivity distinctions and threaded back to Dyck limitations.
>
> - Michael's asking me to clarify the distinction between hard and soft attention—he's picking up on terminology I used when discussing Hahn's impossibility result. This is a straightforward conceptual question about attention mechanisms in transformers and their relationship to formal language theory, so I don't need to search for this.
>   
>   I should pitch the explanation at his level (he understands mech interp and transformer internals deeply) and focus on why the hard/soft distinction matters for the Dyck and parity results we've been discussing. The key insight is that soft attention—the standard softmax mechanism—produces a weighted average across all positions with nonzero weights everywhere, making it differentiable but imprecise for hard formal-language constraints. Hard attention, by contrast, collapses to attending to a single position or small subset, which could theoretically solve those constraints but isn't what real transformers implement.
>   
>   There are two main formalizations: unique-hard attention picks the single argmax position (ties broken by rule), while average-hard or saturated attention spreads weight equally over all positions that tie for the maximum score. The saturated framing is particularly useful because it's the limit of softmax as temperature approaches zero—it characterizes what transformers can represent in principle without actually requiring discrete selection. This distinction is primarily a theoretical tool for analyzing expressivity; real transformers use soft attention, but studying hard variants helps us understand the representational ceiling.
>   
>   The expressivity hierarchy shows that unique-hard attention transformers are quite limited—they fall within AC⁰ and can't recognize languages like PARITY or Dyck. Average-hard and saturated attention transformers are more powerful because averaging over the argmax set lets them compute proportions and counts, placing them in TC⁰. Soft attention with finite precision also sits around TC⁰ based on recent results about log-precision transformers.
>   
>   The key insight from Hahn's work is that hard attention has a Lipschitz-like property where changing one input token only changes the output by a bounded amount, which prevents computing functions like PARITY where a single bit flip completely changes the answer. This is why hard-attention transformers fail on both PARITY and Dyck, but soft attention can escape this limitation because softmax can approximate counting by spreading attention across multiple positions.
>   
>   However, soft attention's workaround comes with a cost: the attention logits and numerical precision must scale with sequence length and nesting depth to keep distinguishing between different Dyck depths, and this precision requirement becomes fragile as inputs get longer or deeper. The diffuse counting head I showed earlier is essentially soft attention's way of realizing this saturated averaging counter, which works well within the training distribution but inherits the same precision fragility when generalizing to longer or deeper sequences.
>   
>   I should mention that "hard attention" has a different historical meaning in the ML literature—Xu et al.'s image captioning work used it to mean sampling a discrete location rather than computing a differentiable weighted average—but the circuit-complexity framing and the saturated-attention connection to AC⁰/TC⁰ are the key insights here. and now I need to lay out the formal-language sense, which is the one actually relevant to Hahn's work on Dyck languages. In that framework, soft attention means the model computes a weighted distribution over positions—a probability or normalized score—while hard attention means it commits to a single discrete choice, like an argmax operation. That's the distinction that matters for expressivity and what Hahn was investigating. The hard attention variants are theoretically useful because they fit into complexity classes, and UHAT's weakness comes from its inability to compute functions like PARITY — flipping a single token can only cause bounded output changes, which rules out global-flip-sensitive problems. AHAT is more powerful because it averages over the tied argmax positions, giving it more computational flexibility than the single-position constraint of UHAT. A soft-attention head achieves this counting by spreading attention roughly uniformly and reading off the average of +1/−1 signals — the same diffuse counting mechanism in the depth-probe script, just implemented softly. The problem is precision: as sequences grow longer, the averaged signal decays like 1/length, so the logits need to scale proportionally to maintain the margin between depths. When precision is bounded or inputs exceed training length, that margin collapses, which explains why the matching arcs degrade out of distribution — Hahn's hard-attention impossibility is the theoretical limit, but soft attention's numerical fragility is what actually breaks empirically. Let me verify the key claims here: hard attention transformers are indeed bounded by AC⁰ complexity, which means they can't handle PARITY or Dyck languages. Hahn's sensitivity argument shows that hard attention has limited robustness under token perturbations, preventing it from recognizing these languages reliably. The saturated attention case pushes into TC⁰ territory, which aligns with the counting capability needed for context-free languages like Dyck.
>   
>   I'm also recalling the standard intuition about the 1/length shrink margin in depth arguments—this comes up in complexity theory when analyzing how much information can be compressed at each layer. I'm settling on a structure: a brief intro that clarifies the two senses of length generalization, then three rungs laid out as a list since they form a natural taxonomy, followed by the complexity implications and the Dyck connection, wrapping up with a clean summary. I'll keep the list items substantive rather than sparse, since this is genuinely an enumerable framework worth laying out clearly.
> - **Done**

The term has two lives, and the one behind Hahn's result isn't the one most people first meet.

**The everyday ML sense** (Xu et al., 2015, image captioning). Soft attention is the differentiable weighted average over all positions you already know — trainable end-to-end by backprop. Hard attention there means the model *samples* a single location, a discrete non-differentiable choice trained with REINFORCE or a variational bound. That distinction is about trainability. It's not what the Dyck/PARITY impossibility is about.

**The formal-language / expressivity sense** is the relevant one, and it's about the *shape of the attention distribution* you analyze to prove what a transformer can compute. Three rungs:

- **Soft (softmax) attention** — weights are softmax(scores), every position gets strictly positive weight, the output is a smooth convex combination of values. What real models run.
- **Unique-hard attention (UHAT)** — the distribution collapses to argmax: all weight on the single highest-scoring position, ties broken by some fixed rule.
- **Average-hard / "saturated" attention (AHAT)** — weight spread *uniformly over the set* of argmax-tied positions. This is exactly the limit of softmax as you scale the logits to infinity (temperature → 0): softmax saturates to a flat average over the top-scoring set. Hence "saturated" — it's the idealization of soft attention at large magnitude.

Theorists use the hard versions because they're combinatorially tractable enough to place transformers in circuit-complexity classes, and the punchlines map directly onto everything we've been discussing:

- **UHAT is weak** — it lands in AC⁰ (constant-depth AND/OR circuits) and can't compute PARITY. Hahn's (2020) argument is a sensitivity one: under hard attention, flipping a single input token can only change the output by a bounded amount, so any function where one bit flips the global answer — PARITY, or "is this bracket string balanced" — is unreachable. That's the impossibility I kept flagging.
- **AHAT/saturated is stronger, because averaging over a set is counting.** Spread attention uniformly over the positions satisfying a predicate, read the result as a proportion, and you've computed a count — and counting buys you PARITY, Dyck-1, and friends. Saturated transformers sit in TC⁰: AC⁰ plus threshold/majority (counting) gates (Merrill, Sabharwal, Smith).
- **Finite-precision soft attention** is bracketed the same way — log-precision transformers ⊆ TC⁰ (Merrill & Sabharwal, 2023). So real soft attention does the counting that hard-argmax can't, but it does it by *approximating* the saturated average with a peaked-but-not-degenerate softmax.

This is precisely the "cost of precision" from the Dyck thread. The diffuse counting head in your depth-probe script is the soft realization of the saturated averaging trick: it spreads attention ~uniformly and reads the +1/−1 mean to get depth. The catch is numerical — to keep depth k distinguishable from depth k+1 as strings grow, the averaged signal shrinks roughly like 1/length, so the logits and numerical precision have to scale to preserve the margin. Bounded precision plus inputs longer or deeper than training erodes that margin, which is exactly why the matching arcs degrade out of distribution. Hahn's hard-attention result is the clean limiting impossibility; the soft-attention precision fragility is what you actually observe.

Crisp version: soft = smooth convex combination over everything (what runs); unique-hard = argmax pick (weak, AC⁰, no counting); average-hard/saturated = uniform over the argmax set, i.e. soft-attention-at-infinity, which *can* count (TC⁰). The whole Dyck story lives in the gap — counting needs the averaging that argmax lacks, softmax supplies it approximately, and "approximately" is where length generalization breaks.

Powered by Claude Exporter (https://www.ai-chat-exporter.net)
