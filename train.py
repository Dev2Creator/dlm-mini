"""Train dlm-mini from scratch on character-level text.

Examples
--------
Dense baseline (no dynamic compute):
    python train.py --mode dense --steps 500

Mixture-of-Depths:
    python train.py --mode mod --capacity 0.5 --steps 500
"""

from __future__ import annotations

import argparse
import math
import time

import torch

from dlm_mini.data import CharDataset, load_text
from dlm_mini.model import DLM, DLMConfig


def get_lr(step: int, warmup: int, total: int, max_lr: float, min_lr: float) -> float:
    if step < warmup:
        return max_lr * (step + 1) / warmup
    if step >= total:
        return min_lr
    ratio = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * ratio))


@torch.no_grad()
def estimate_loss(model: DLM, ds: CharDataset, batch_size: int, device: str, evals: int = 10) -> dict:
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(evals)
        for i in range(evals):
            x, y = ds.get_batch(split, batch_size, device)
            _, loss, _ = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["dense", "mod"], default="mod")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--dim", type=int, default=192)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--kv-heads", type=int, default=2)
    p.add_argument("--capacity", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    text = load_text()
    ds = CharDataset(text, block_size=args.block_size)
    print(f"[data] chars={len(text):,} vocab={ds.vocab_size}")

    cfg = DLMConfig(
        vocab_size=ds.vocab_size,
        dim=args.dim,
        n_layers=args.layers,
        n_heads=args.heads,
        n_kv_heads=args.kv_heads,
        max_len=args.block_size,
        capacity=args.capacity,
        mod_every=2 if args.mode == "mod" else 0,
    )
    model = DLM(cfg).to(args.device)
    print(f"[model] mode={args.mode} params={model.num_params():,}")

    decay, no_decay = [], []
    for name, param in model.named_parameters():
        (decay if param.dim() >= 2 else no_decay).append(param)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr,
        betas=(0.9, 0.95),
    )

    warmup = max(10, args.steps // 20)
    t0 = time.time()
    for step in range(args.steps):
        lr = get_lr(step, warmup, args.steps, args.lr, args.lr / 10)
        for g in opt.param_groups:
            g["lr"] = lr

        x, y = ds.get_batch("train", args.batch_size, args.device)
        _, loss, aux = model(x, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 20 == 0:
            print(f"step {step:5d} | loss {loss.item():.4f} | aux {aux.item():.5f} | lr {lr:.2e}")

        if args.steps > 0 and (step + 1) % args.eval_every == 0:
            m = estimate_loss(model, ds, args.batch_size, args.device)
            print(f"  eval @ {step + 1}: train {m['train']:.4f} | val {m['val']:.4f}")

    dt = time.time() - t0
    print(f"[done] {args.steps} steps in {dt:.1f}s ({args.steps / dt:.2f} steps/s)")

    sample = model.generate(ds.encode("\n").unsqueeze(0), max_new_tokens=200, temperature=0.9, top_k=40)
    print("[sample]")
    print(ds.decode(sample[0]))

    if args.out:
        torch.save({"model": model.state_dict(), "config": cfg.__dict__, "stoi": ds.stoi}, args.out)
        print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
