"""Character-level data loading for dlm-mini.

Tries tiny-shakespeare first, falls back to a small synthetic corpus so the
project is runnable with zero network access.
"""

from __future__ import annotations

import os
import urllib.request

import torch

TINY_SHAKESPEARE = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)

SYNTHETIC = """the quick brown fox jumps over the lazy dog.
a mamba is a snake that moves with a smooth steady motion.
attention mixes every token with every other token in the window.
a dynamic model spends compute only where it is needed.
small models can still write sentences when trained on clean data.
the router decides which tokens deserve the deeper path.
"""


def load_text(cache_dir: str = "data", allow_download: bool = True) -> str:
    path = os.path.join(cache_dir, "input.txt")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    if allow_download:
        try:
            os.makedirs(cache_dir, exist_ok=True)
            with urllib.request.urlopen(TINY_SHAKESPEARE, timeout=10) as r:
                text = r.read().decode("utf-8")
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            return text
        except Exception as exc:  # pragma: no cover - offline fallback
            print(f"[data] download failed ({exc}); using synthetic corpus")

    return SYNTHETIC * 200


class CharDataset:
    def __init__(self, text: str, block_size: int = 128, split: float = 0.9):
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}

        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        n = int(len(data) * split)
        self.train = data[:n]
        self.val = data[n:]
        self.block_size = block_size

    def encode(self, s: str) -> torch.Tensor:
        return torch.tensor([self.stoi[c] for c in s if c in self.stoi], dtype=torch.long)

    def decode(self, t) -> str:
        if isinstance(t, torch.Tensor):
            t = t.tolist()
        return "".join(self.itos[i] for i in t)

    def get_batch(self, split: str = "train", batch_size: int = 16, device: str = "cpu"):
        data = self.train if split == "train" else self.val
        ix = torch.randint(len(data) - self.block_size - 1, (batch_size,))
        x = torch.stack([data[i : i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + self.block_size + 1] for i in ix])
        return x.to(device), y.to(device)
