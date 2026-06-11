# Modernizing nanoGPT: GPT-2-style → Llama-style, with ablations

Karpathy's [nanoGPT](https://github.com/karpathy/ng-video-lecture) (video-lecture version, ~10.7M params),
upgraded component-by-component to the modern Llama-style stack — **RMSNorm**, **SwiGLU**, and **RoPE** —
with a controlled ablation of each change on character-level Shakespeare.

**Headline result:** the fully modernized model reaches a better validation loss than the baseline's
best in roughly **half the training steps** (step ~1500 vs. ~3000). The most interesting single finding
is negative: **SwiGLU optimizes faster but overfits the 1MB dataset**, ending with the worst validation
loss of all variants — a small-scale reproduction of why frontier-scale architecture wins don't
automatically transfer downward.

## Results

5000 iters, batch 64, context 256, lr 3e-4, single seed (1337), identical batch order across runs.
Hardware: RTX 5060 (8GB, Blackwell — requires PyTorch ≥ 2.7 / CUDA 12.8).

| variant  | params (M) | final train | final val | **best val** | tok/s   | train min |
|----------|------------|-------------|-----------|--------------|---------|-----------|
| baseline | 10.789     | 0.8595      | 1.5669    | 1.4831       | 119,196 | 11.5      |
| rmsnorm  | 10.784     | 0.8655      | 1.5690    | 1.4817       | 120,176 | 11.4      |
| swiglu   | 10.777     | 0.6212      | 1.7707    | 1.4927       | 117,827 | 11.6      |
| rope     | 10.691     | 0.8403      | 1.5424    | **1.4625**   | 117,265 | 11.6      |
| all      | 10.674     | 0.6358      | 1.7256    | 1.4676       | 116,622 | 11.7      |

![validation loss curves](results/val_curves.png)

Ranking by **best val** (the early-stopping value — all configs overfit by iter 5000):

- **RoPE** is the clear winner, and dominates the *entire trajectory*, not just the endpoint
  (val 1.67 vs. 1.91 at step 500). This trajectory-wide dominance is an indicator that **RoPE**'s performance is better (and not a random seed effect) due to its embedding strategy which takes into account the relative positions from the very start. 
- **RMSNorm** is a quality wash (Δ 0.0014 = noise) with the best throughput — exactly its sales pitch.
- **SwiGLU** has the lowest train loss at every checkpoint from step 500 onward, and converts that
  expressivity into memorization: val climbs from step 2500. This indicates that **SwiGLU** fits to the data faster, and at this scale, it starts to overfit. 
- Throughput is flat across variants (~3% spread), empirically confirming the parameter/FLOP
  matching (see below).
- Caveat: single seed; differences under ~0.01–0.02 val loss are noise. The RoPE result survives
  this bar via trajectory dominance; rmsnorm-vs-baseline does not.

## What changed and why

### RMSNorm (replaces LayerNorm) — [Zhang & Sennrich 2019](https://arxiv.org/abs/1910.07467)
Drops mean-centering and the bias; keeps only RMS scaling and a learnable gain.
Same normalization axes as LayerNorm — the difference is the *statistic*, not the geometry. The computation overhead is decreased, leading to a marginal throughput gain (~1% here; the effect matters more at scale), and the stabilization of the gradients is maintained.


### SwiGLU (replaces the ReLU MLP) — [Shazeer 2020](https://arxiv.org/abs/2002.05202)
Replaces `W2·ReLU(W1·x)` with `W_down·(SiLU(W_gate·x) ⊙ W_up·x)`. Hidden dim is set to
`2/3 · 4 · n_embd` so total FFN weights exactly match the original (`3·d·h = 8d²` — exact here
because `3 | 384`). Biases dropped throughout, Llama-style; that removal is a *separate* design
choice bundled with the gating, worth ~0.1% of parameters. Assuming that the feed-forward network takes $x$ inputs, and $y$ is the original FFN's hidden width (SwiGLU's weight-matched hidden is $2y/3$), the bias dropping **SwiGLUFFN** has $y+x$ parameters less per FFN, whereas with bias parameters, it has $y/3$ more parameters per FFN.  

### RoPE (replaces learned positional embeddings) — [Su et al. 2021](https://arxiv.org/abs/2104.09864)
Rotates q and k per-head, per-layer (never v), by position-dependent angles so attention scores
depend only on *relative* position: `(R_m q)·(R_n k) = qᵀ R_{n−m} k`. Implemented via the complex formulation (`torch.polar` / `view_as_complex`). Deletes the `block_size × n_embd` PE table (−98K params — the bulk of the parameter drop in the table above). Attention scores decay with relative distance ('long-term decay'), biasing the model toward nearby context — structure that learned absolute embeddings would have to discover from data. Although the number of parameters is smaller, the per-layer rotation adds a small elementwise overhead (~2% throughput in the table).


## Reproduce

```bash
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
pip install -r requirements.txt
python train_ablations.py        # trains all 5 variants serially, ~80 min on an RTX 5060
```

Results stream into `results.md` (crash-safe: the summary table is rewritten after every
completed run). Checkpoints for all five variants are on [Hugging Face](https://huggingface.co/Ridwan-1642/nanogpt-modernization-ckpts).

Windows note: `torch.compile` may fail (Triton support); set `USE_COMPILE = False` for eager mode.

## Bugs I hit (and how I found them)
1. BatchNorm fossils in RMSNorm: As I created my own RMSnorm class, I made mistakes such as using a momentum parameter, a running mean, a bias parameter, using plain tensors instead of the nn.Parameter. After going through the actual paper, the paper made clear why momentum, running statistics, and bias don't apply: RMSNorm computes per-sample statistics, so there's nothing to track across batches. Wrapping in nn.Parameter is necessary for tracking the gradients together.
2. **SwiGLU parameter counts:** Since **SwiGLUFFN** uses 3 weight matrices instead of the usual 2, a silent issue that I ran into was keeping the layer-sizes the same, resulting in more parameters for **SwiGLUFFN**. Comparing models with a variation in parameter-size is faulty, hence I precisely calculated the required size of the hidden layer.
3. Bias-dimension swap in my parameter-count derivation: Linear bias lives in the OUTPUT space; initially I made the mistake by assuming the bias lives in the input space, which flipped my conclusion. This was caught when I checked the actual parameter counts.
4. RoPE frequency buffer not sliced to sequence length: This bug is invisible at T == block_size, but crashes at generation time. Caught by forwarding a T=7 batch.
5. eps placed outside the sqrt in RMSNorm: This runs fine, but is a different function than every reference implementation. This is a silent bug, caught only by line-by-line comparison against the reference implementation.

## Future work

- Dense (per-100-step) loss logging to locate the induction-head phase transition
  ([Olsson et al. 2022](https://arxiv.org/abs/2209.11895)) per variant — do positional
  encodings shift *when* induction heads form? Checkpoints already saved for this.
- Trainable RoPE frequencies (θ as parameters) as a sixth ablation row.
- Fused multi-head attention via `F.scaled_dot_product_attention` + before/after kernel profile.

## Acknowledgements

Built on Andrej Karpathy's [ng-video-lecture](https://github.com/karpathy/ng-video-lecture) code
(including, faithfully, the `FeedFoward` typo — preserved here as an easter egg, then fixed).
