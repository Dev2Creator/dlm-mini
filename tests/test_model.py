"""Sanity tests for dlm-mini. Run: pytest -q"""

import torch

from dlm_mini.model import DLM, DLMConfig, MoDBlock


def tiny_cfg(**kw):
    base = dict(vocab_size=32, dim=64, n_layers=4, n_heads=4, n_kv_heads=2, max_len=64, capacity=0.5)
    base.update(kw)
    return DLMConfig(**base)


def test_forward_shapes():
    model = DLM(tiny_cfg())
    x = torch.randint(0, 32, (2, 16))
    logits, loss, aux = model(x)
    assert logits.shape == (2, 16, 32)
    assert loss is None

    _, loss, aux = model(x, torch.randint(0, 32, (2, 16)))
    assert loss.item() > 0
    assert aux.item() >= 0


def test_dense_matches_when_mod_disabled():
    model = DLM(tiny_cfg(mod_every=0))
    assert all(not isinstance(l, MoDBlock) for l in model.layers)


def test_mod_skips_tokens():
    block = MoDBlock(dim=32, n_heads=4, n_kv_heads=2, max_len=64, dropout=0.0, capacity=0.5)
    x = torch.randn(2, 20, 32)
    pos = torch.arange(20).unsqueeze(0).expand(2, 20)
    out, aux = block(x, pos)
    assert out.shape == x.shape
    # at least some tokens must be untouched by the sub-block
    changed = (out != x).any(-1).sum().item()
    assert changed <= 2 * 10, f"too many tokens computed: {changed}"


def test_router_gets_gradients():
    block = MoDBlock(dim=32, n_heads=4, n_kv_heads=2, max_len=64, dropout=0.0, capacity=0.5)
    x = torch.randn(2, 20, 32)
    pos = torch.arange(20).unsqueeze(0).expand(2, 20)
    out, aux = block(x, pos)
    (out.sum() + aux).backward()
    assert block.router.weight.grad is not None
    assert block.router.weight.grad.abs().sum().item() > 0


def test_generate():
    model = DLM(tiny_cfg(max_len=32))
    out = model.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=8)
    assert out.shape == (1, 9)
