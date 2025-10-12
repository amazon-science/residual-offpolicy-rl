# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0

import einops
import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.init import trunc_normal_


class PatchEmbed1(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.conv = nn.Conv2d(3, embed_dim, kernel_size=8, stride=8)

        self.num_patch = 144  # if input image is 96x96, then num_patch = 144
        self.patch_dim = embed_dim

    def forward(self, x: torch.Tensor):
        y = self.conv(x)
        y = einops.rearrange(y, "b c h w -> b (h  w) c")
        return y  # noqa: RET504


class PatchEmbed2(nn.Module):
    def __init__(self, embed_dim, use_norm):
        super().__init__()
        layers = [
            nn.Conv2d(3, embed_dim, kernel_size=8, stride=4),
            nn.GroupNorm(embed_dim, embed_dim) if use_norm else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2),
        ]
        self.embed = nn.Sequential(*layers)

        # self.num_patch = 121  # if input image is 96x96, then num_patch = 121
        self.num_patch = 81  # if input image is 84x84, then num_patch = 81
        self.patch_dim = embed_dim

    def forward(self, x: torch.Tensor):
        y = self.embed(x)
        y = einops.rearrange(y, "b c h w -> b (h  w) c")
        return y  # noqa: RET504


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_head):
        super().__init__()
        assert embed_dim % num_head == 0

        self.num_head = num_head
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, attn_mask):
        """
        x: [batch, seq, embed_dim]
        """
        qkv = self.qkv_proj(x)
        q, k, v = einops.rearrange(qkv, "b t (k h d) -> b k h t d", k=3, h=self.num_head).unbind(1)
        # force flash/mem-eff attention, it will raise error if flash cannot be applied
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            attn_v = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, attn_mask=attn_mask)
        attn_v = einops.rearrange(attn_v, "b h t d -> b t (h d)")
        return self.out_proj(attn_v)


class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_head, dropout):
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.mha = MultiHeadAttention(embed_dim, num_head)

        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.linear1 = nn.Linear(embed_dim, 4 * embed_dim)
        self.linear2 = nn.Linear(4 * embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        x = x + self.dropout(self.mha(self.layer_norm1(x), attn_mask))
        x = x + self.dropout(self._ff_block(self.layer_norm2(x)))
        return x  # noqa: RET504

    def _ff_block(self, x):
        x = self.linear2(nn.functional.gelu(self.linear1(x)))
        return x  # noqa: RET504


class MinVit(nn.Module):
    def __init__(self, embed_style, embed_dim, embed_norm, num_head, depth):
        super().__init__()

        if embed_style == "embed1":
            raise NotImplementedError("embed1 is not tested")
            # self.patch_embed = PatchEmbed1(embed_dim)
        if embed_style == "embed2":
            self.patch_embed = PatchEmbed2(embed_dim, use_norm=embed_norm)
        else:
            raise NotImplementedError(f"Unknown embed style {embed_style}")

        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patch, embed_dim))
        layers = [TransformerLayer(embed_dim, num_head, 0) for _ in range(depth)]

        self.net = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.num_patches = self.patch_embed.num_patch

        # weight init
        trunc_normal_(self.pos_embed, std=0.02)
        named_apply(init_weights_vit_timm, self)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.net(x)
        return self.norm(x)


def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def named_apply(fn, module: nn.Module, name="", depth_first=True, include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        full_child_name = f"{name}.{child_name}" if name else child_name
        named_apply(fn=fn, module=child_module, name=full_child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


def test_patch_embed():
    print("embed 1")
    embed = PatchEmbed1(128)
    x = torch.rand(10, 3, 84, 84)
    y = embed(x)
    print(y.size())

    print("embed 2")
    embed = PatchEmbed2(128, True)
    x = torch.rand(10, 3, 84, 84)
    y = embed(x)
    print(y.size())


def test_transformer_layer():
    embed = PatchEmbed1(128)
    x = torch.rand(10, 3, 84, 84)
    y = embed(x)
    print(y.size())

    transformer = TransformerLayer(
        num_head=4,
        dropout=0,
        embed_dim=128,
    )
    z = transformer(y)
    print(z.size())


if __name__ == "__main__":
    test_patch_embed()
    test_transformer_layer()
