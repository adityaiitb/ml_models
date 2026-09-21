import torch
import torch.nn as nn
import torch.functional as F

from llm import config as cfg


class MHA(nn.Module):
    def __init__(self, config: cfg.ModelConfig):
        super().__init__()
        self.D = config.emb_dim
        self.N = config.heads
        self.H = config.head_dim
        self.q_proj = nn.Linear(self.D, self.N * self.H, bias=config.qkv_bias)
        self.k_proj = nn.Linear(self.D, self.N * self.H, bias=config.qkv_bias)
        self.v_proj = nn.Linear(self.D, self.N * self.H, bias=config.qkv_bias)
        self.o_proj = nn.Linear(self.N * self.H, self.D)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        assert D == self.D

        query = self.q_proj(x)  # (B, T, NH)
        keys = self.k_proj(x)
        value = self.v_proj(x)

        query = query.view(B, T, self.N, self.H).transpose(1, 2)  # (B, N, T, H)
        keys = keys.view(B, T, self.N, self.H).transpose(1, 2)
        value = value.view(B, T, self.N, self.H).transpose(1, 2)

        logits = query @ keys.transpose(2, 3)  # (B, N, T, T)

        probs = torch.softmax(logits, dim=-1)
        probs = self.dropout(probs)
        context = probs @ value  # (B, N, T, H)
        context = context.transpose(1, 2)
        context = context.reshape(B, T, -1)  # (B, T, NH)
        context = self.o_proj(context)  # (B, T, D)
        return context


class FFN(nn.Module):
    def __init__(self, config: cfg.ModelConfig):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(config.emb_dim, config.ffn_hidden_dim),
            nn.GELU(),
            nn.Linear(config.ffn_hidden_dim, config.emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: cfg.ModelConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.emb_dim)
        self.mha = MHA(config)
        self.norm2 = nn.LayerNorm(config.emb_dim)
        self.ffn = FFN(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.norm1(x)
        x = self.mha(x)
        x = self.dropout(x)
        x += identity

        identity = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x += identity
        return x


class ViT(nn.Module):
    def __init__(self, config: cfg.ModelConfig):
        super().__init__()

        C, H, W = config.image_shape
        R, S = config.patch_shape
        assert (H == W) and (R == S)
        P, Q = H // R, W // S
        num_patches = P * Q

        self.conv = nn.Conv2d(C, config.emb_dim, kernel_size=R, stride=R)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.emb_dim))
        self.pos_emb = nn.Parameter(torch.zeros(1, 1 + num_patches, config.emb_dim))
        self.emb_dropout = nn.Dropout(config.dropout)
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config.layers)]
        )
        self.final_norm = nn.LayerNorm(config.emb_dim)
        self.out_head = nn.Linear(config.emb_dim, config.num_classes, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.conv(x)  # (B, emb_dim, P, Q)
        x = x.flatten(2, 3)  # (B, emb_dim, PQ)
        x = x.transpose(1, 2)  # (B, PQ, emb_dim)
        # x is the image encoding. PQ patches each emb_dim wide.
        # Append the CLS token at the front.
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        x += self.pos_emb
        x = self.emb_dropout(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)

        # Use only the CLS token for classification
        x = x[:, 0, :]
        x = self.out_head(x)
        return x
