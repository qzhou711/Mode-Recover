"""Low-rank linear operators that preserve Transformer residual coordinates."""
import torch
import torch.nn as nn


class LowRankLinear(nn.Module):
    def __init__(self, in_features, out_features, rank, bias=True):
        super().__init__()
        if not 0 < rank < min(in_features, out_features):
            raise ValueError(f"rank {rank} must be below min({in_features}, {out_features})")
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=bias)

    def forward(self, value):
        return self.up(self.down(value))


def get_submodule(root, path):
    module = root
    for part in path.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def set_submodule(root, path, value):
    parent_path, name = path.rsplit(".", 1)
    parent = get_submodule(root, parent_path)
    if name.isdigit():
        parent[int(name)] = value
    else:
        setattr(parent, name, value)


def apply_operator_compression(network, ranks):
    """Replace named Linear modules with low-rank factors."""
    for name, rank in ranks.items():
        original = get_submodule(network, name)
        if not isinstance(original, nn.Linear):
            raise TypeError(f"{name} is not Linear: {type(original)}")
        replacement = LowRankLinear(
            original.in_features, original.out_features, int(rank),
            bias=original.bias is not None,
        ).to(original.weight.device, dtype=original.weight.dtype)
        set_submodule(network, name, replacement)


def operator_config(variant):
    ranks = {}
    if variant in {"uniform_svd", "uniform_activation"}:
        for layer in range(3):
            for name in ("key", "query", "value", "proj"):
                ranks[f"blocks.{layer}.attn.{name}"] = 32
            ranks[f"blocks.{layer}.mlp.0"] = 8
            ranks[f"blocks.{layer}.mlp.2"] = 8
        return {"variant": variant, "ffn_dim": None, "ranks": ranks}
    if variant == "routing_aware":
        for layer in range(3):
            ranks[f"blocks.{layer}.attn.key"] = 40
            ranks[f"blocks.{layer}.attn.query"] = 40
            ranks[f"blocks.{layer}.attn.value"] = 24
            ranks[f"blocks.{layer}.attn.proj"] = 24
            ranks[f"blocks.{layer}.mlp.0"] = 6
            ranks[f"blocks.{layer}.mlp.2"] = 6
        return {"variant": variant, "ffn_dim": None, "ranks": ranks}
    if variant == "hybrid_balanced":
        for layer in range(3):
            ranks[f"blocks.{layer}.attn.key"] = 32
            ranks[f"blocks.{layer}.attn.query"] = 32
            ranks[f"blocks.{layer}.attn.value"] = 16
            ranks[f"blocks.{layer}.attn.proj"] = 16
        return {"variant": variant, "ffn_dim": 96, "ranks": ranks}
    raise ValueError(variant)
