# dlm-mini 🗿⚡

A minimal, readable **Mixture-of-Depths (MoD)** language model — dynamic compute
for tokens, built from scratch in ~350 lines of PyTorch.

Most "dynamic compute" demos compute every token and mask the output. That saves
nothing at inference. **dlm-mini actually skips work**: at MoD layers a learned
router gathers only the top-k tokens into the heavy sub-block, computes them, and
scatters the results back. Skipped tokens keep their residual state and cost zero
FLOPs in the sub-block.

```
dense block:   every token  ->  attention + MLP   (T tokens computed)
MoD block:     top-k tokens ->  attention + MLP   (k = capacity × T)
```

## Why this exists

Transformers spend the same compute on the token `"the"` as on the token that
actually needs reasoning. Mixture-of-Depths asks: *why?* Give every layer a
router, let it decide per token whether the token is "hard enough", and save the
difference. This repo is the smallest honest implementation of that idea, plus
a dense baseline to measure it against.

Based on:

- **Mixture-of-Depths: Dynamically allocating compute in transformer-based language models** — Raposo et al., 2024 ([arXiv:2404.02258](https://arxiv.org/abs/2404.02258))
- **FlexiDepth** — retrofit layer-skipping into pretrained models ([arXiv:2503.23798](https://arxiv.org/abs/2503.23798))

## Quick start

```bash
pip install -r requirements.txt

# dense baseline
python train.py --mode dense --steps 300

# dynamic (Mixture-of-Depths)
python train.py --mode mod --steps 300 --capacity 0.5

# tests
pytest -q
```

Runs on CPU in seconds. On first run it downloads tiny-shakespeare; with no
network it falls back to a built-in synthetic corpus, so it always runs.

## Architecture

```
input ids ─► token embedding
               │
        ┌──────┴───────────────────────────────┐
        │ layer 0   : DenseBlock               │  every token computed
        │ layer 1   : MoDBlock  (capacity 0.5) │  50% of tokens computed
        │ layer 2   : DenseBlock               │
        │ layer 3   : MoDBlock                 │
        │ ...                                  │  mod_every=2
        └──────┬───────────────────────────────┘
               │
          RMSNorm ─► LM head (tied embeddings)
```

Every block: pre-norm residual + causal **GQA attention with RoPE**, then
SwiGLU MLP. Nothing exotic, so the dynamic part stands out.

### The MoD router

```python
scores = router(x)                    # one scalar per token
idx    = topk(scores, k=T*capacity)   # which tokens are worth computing
soft   = sigmoid(scores)
hard   = one_hot(idx)
gate   = hard + soft - soft.detach()  # straight-through: hard forward, soft grad

sel    = x[batch_idx, idx]            # gather only those tokens
out    = sub_block(sel)               # compute only those tokens
x      = x + gate * scatter(out)      # skipped tokens pass through untouched
```

Two details that matter:

1. **Straight-through gate** — the forward pass routes hard (exact top-k), but
   gradients flow through the sigmoid so the router actually learns.
2. **Load-balance auxiliary loss** — `(soft.mean() - capacity)²` nudges the
   router's probability mass toward the capacity instead of collapsing onto a
   few "favourite" positions.

## Measured results (CPU, tiny config)

Config: `dim=128, layers=4, heads=4, block=64`, 300 steps, char-level
tiny-shakespeare, single CPU core. One seed. Treat as a smoke benchmark, not a
paper result.

```
mode    params     val loss   steps/s
dense   787,584    2.3384     14.8
mod     787,712    2.1007     16.4     capacity=0.5
```

MoD matched or beat dense at ~half the sub-block compute in this run. The
parameter count barely moves (+128 router weights) — the win is FLOPs, not size.

## Repo layout

```
dlm_mini/model.py   RMSNorm, RoPE, GQA attention, SwiGLU, DenseBlock, MoDBlock, DLM
dlm_mini/data.py    char-level dataset (tiny-shakespeare with offline fallback)
train.py            training loop: AdamW, cosine schedule, grad clip, eval, sampling
tests/test_model.py shape tests, skip-count test, router-gradient test
```

## What this is not

- Not a production LLM — it is a reference implementation of dynamic compute.
- Causal masking inside a MoD layer is applied to the *selected subsequence*, an
  approximation shared by most practical MoD implementations.
- No KV cache yet (see roadmap).

## Roadmap

- [ ] KV cache for fast generation
- [ ] Per-token telemetry: which tokens get skipped, by layer and type
- [ ] Drop-in comparison: MoD vs dense at equal *wall-clock* budget, multi-seed
- [ ] Hybrid: attention layers + selective SSM (Mamba) layers in the same trunk
- [ ] Quantized CPU export (ONNX int8)

## License

**AGPL-3.0** — see [LICENSE](LICENSE). Use it, fork it, ship it; if you run a
modified version as a network service, publish your changes.
