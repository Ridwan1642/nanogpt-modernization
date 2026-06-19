"""
Serial ablation runner: trains 5 nanoGPT variants back-to-back and logs results.

Variants:
  1. baseline  : LayerNorm + ReLU FFN + learned positional embeddings
  2. rmsnorm   : RMSNorm  + ReLU FFN + learned positional embeddings
  3. swiglu    : LayerNorm + SwiGLU  + learned positional embeddings
  4. rope      : LayerNorm + ReLU FFN + RoPE (no learned PE)
  5. all       : RMSNorm  + SwiGLU  + RoPE

Outputs:
  results.md        - markdown table (paste straight into README) + per-run logs
  ckpt_<name>.pt    - final weights per variant (kept for later MI experiments)
"""

import time
import torch
import torch.nn as nn
from torch.nn import functional as F

# ----------------- hyperparameters (identical across all runs) -----------------
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
SEED = 1337
USE_COMPILE = False
RESULTS_FILE = 'results.md'
# -------------------------------------------------------------------------------

ABLATIONS = [
    ('baseline', dict(rms=False, swiglu=False, rope=False)),
    ('rmsnorm',  dict(rms=True,  swiglu=False, rope=False)),
    ('swiglu',   dict(rms=False, swiglu=True,  rope=False)),
    ('rope',     dict(rms=False, swiglu=False, rope=True)),
    ('all',      dict(rms=True,  swiglu=True,  rope=True)),
]

# ----------------------------- data -----------------------------
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split, g):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,), generator=g)
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, g):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, g)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# ----------------------------- components -----------------------------
def precompute_freqs_cis(head_dim, max_seq_len, base=10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, inv_freq)                    
    return torch.polar(torch.ones_like(freqs), freqs)   


def apply_rope(x, freqs_cis):
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    out = torch.view_as_real(xc * freqs_cis)
    return out.flatten(-2).type_as(x)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.gamma


class Head(nn.Module):
    """one head of self-attention, optionally with RoPE"""

    def __init__(self, head_size, use_rope):
        super().__init__()
        self.use_rope = use_rope
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        if use_rope:
            self.register_buffer('freq', precompute_freqs_cis(head_size, block_size),
                                 persistent=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        if self.use_rope:
            q = apply_rope(q, self.freq[:T])
            k = apply_rope(k, self.freq[:T])
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, use_rope):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, use_rope) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """original: up-project, ReLU, down-project"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class SwiGLUFFN(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        hidden = int(2 / 3 * 4 * n_embd)  
        self.w_gate = nn.Linear(n_embd, hidden, bias=False)
        self.w_up = nn.Linear(n_embd, hidden, bias=False)
        self.w_down = nn.Linear(hidden, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class Block(nn.Module):
    def __init__(self, n_embd, n_head, cfg):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, cfg['rope'])
        self.ffwd = SwiGLUFFN(n_embd) if cfg['swiglu'] else FeedForward(n_embd)
        norm = RMSNorm if cfg['rms'] else nn.LayerNorm
        self.ln1 = norm(n_embd)
        self.ln2 = norm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        if not cfg['rope']:
            self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, cfg) for _ in range(n_layer)])
        self.ln_f = RMSNorm(n_embd) if cfg['rms'] else nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_embedding_table(idx)
        if not self.cfg['rope']:
            pos = torch.arange(T, device=idx.device)
            x = x + self.position_embedding_table(pos)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss


# ----------------------------- runner -----------------------------
def log(msg):
    print(msg, flush=True)
    with open(RESULTS_FILE, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')


def train_one(name, cfg):
    torch.manual_seed(SEED)
    data_gen=torch.Generator()
    data_gen.manual_seed(SEED)
    if USE_COMPILE:
        torch._dynamo.reset()

    raw_model = GPTLanguageModel(cfg).to(device)
    n_params = sum(p.numel() for p in raw_model.parameters())

    model = raw_model
    if USE_COMPILE:
        try:
            model = torch.compile(raw_model)
        except Exception as e:
            log(f'  [warn] torch.compile failed ({e}); running eager')
            model = raw_model

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, fused=(device == 'cuda'))

    log(f'\n### run: {name} | cfg={cfg} | params={n_params/1e6:.3f}M')

    best_val = float('inf')
    train_time = 0.0
    for it in range(max_iters):
        if it % eval_interval == 0 or it == max_iters - 1:
            losses = estimate_loss(model, data_gen)
            best_val = min(best_val, losses['val'])
            log(f"  step {it}: train {losses['train']:.4f}, val {losses['val']:.4f}")

        xb, yb = get_batch('train', data_gen)
        if device == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if device == 'cuda':
            torch.cuda.synchronize()
        train_time += time.perf_counter() - t0

    final = estimate_loss(model, data_gen)
    best_val = min(best_val, final['val'])
    tok_per_s = max_iters * batch_size * block_size / train_time

    torch.save(raw_model.state_dict(), f'ckpt_{name}.pt')  # keep for MI experiments later

    return dict(name=name, params=n_params, train=final['train'], val=final['val'],
                best_val=best_val, tok_s=tok_per_s, minutes=train_time / 60)


def main():
    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        f.write(f'# nanoGPT modernization ablations\n')
        f.write(f'device={device}, seed={SEED}, iters={max_iters}, '
                f'batch={batch_size}, block={block_size}, lr={learning_rate}\n')

    summaries = []
    t_start = time.time()
    for name, cfg in ABLATIONS:
        summaries.append(train_one(name, cfg))
        table = ['\n## Summary (so far)\n',
                 '| variant | params (M) | final train | final val | best val | tok/s | train min |',
                 '|---|---|---|---|---|---|---|']
        for s in summaries:
            table.append(f"| {s['name']} | {s['params']/1e6:.3f} | {s['train']:.4f} "
                         f"| {s['val']:.4f} | {s['best_val']:.4f} "
                         f"| {s['tok_s']:,.0f} | {s['minutes']:.1f} |")
        with open(RESULTS_FILE, 'a', encoding='utf-8') as f:
            f.write('\n'.join(table) + '\n')

    log(f'\nAll runs done in {(time.time()-t_start)/60:.1f} min total.')


if __name__ == '__main__':
    main()
