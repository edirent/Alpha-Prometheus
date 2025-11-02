# -*- coding: utf-8 -*-
# research/gen_dit.py
import math
from typing import Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------
# utils
# ----------------------
def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Sinusoidal time embedding (t in [0,1]).
    """
    device = t.device
    half = dim // 2
    freqs = torch.exp(
        torch.arange(half, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / (half - 1))
    )
    # map t->[0,1000] for more spread
    args = t[:, None] * 1000.0 * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb


class AdaLayerNorm(nn.Module):
    """
    Simple AdaLN: y = (x * scale + shift), where [shift, scale] from cond.
    """
    def __init__(self, hidden: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden, eps=1e-6)
        self.fc = nn.Linear(cond_dim, hidden * 2)
        nn.init.zeros_(self.fc.weight); nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        shift_scale = self.fc(cond)  # (B, 2H)
        shift, scale = shift_scale.chunk(2, dim=-1)
        return h * (1 + scale[..., None, :]) + shift[..., None, :]


# ----------------------
# tiny DiT-style encoder for context
# ----------------------
class PatchEmbed2D(nn.Module):
    def __init__(self, in_ch: int, hidden: int, patch: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, hidden, kernel_size=patch, stride=patch, padding=0)
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity="relu")
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W) -> (B,HW',D)
        x = self.conv(x)  # (B,D,H',W')
        x = x.flatten(2).transpose(1, 2)
        return x


class MLP(nn.Module):
    def __init__(self, hidden: int, mlp_ratio: float = 4.0):
        super().__init__()
        inner = int(hidden * mlp_ratio)
        self.fc1 = nn.Linear(hidden, inner)
        self.fc2 = nn.Linear(inner, hidden)
        nn.init.xavier_uniform_(self.fc1.weight); nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight); nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, hidden: int, heads: int, mlp_ratio: float, cond_dim: int):
        super().__init__()
        self.attn_norm = AdaLayerNorm(hidden, cond_dim)
        self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.mlp_norm = AdaLayerNorm(hidden, cond_dim)
        self.mlp = MLP(hidden, mlp_ratio)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B,N,D), cond: (B,cond_dim)
        h = self.attn_norm(x, cond)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        h = self.mlp_norm(x, cond)
        x = x + self.mlp(h)
        return x


class ContextDiT(nn.Module):
    """
    Encode historical window X -> context vector c (B,D).
    """
    def __init__(self, in_ch: int, hidden: int = 384, depth: int = 6, heads: int = 6, patch: int = 2, mlp_ratio: float = 4.0, cond_dim: int = 256):
        super().__init__()
        self.hidden = hidden
        self.patch = PatchEmbed2D(in_ch, hidden, patch)
        self.pos = nn.Parameter(torch.zeros(1, 1024, hidden))  # big enough; will slice
        nn.init.normal_(self.pos, std=0.02)

        self.t_proj = nn.Linear(128, cond_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden, heads, mlp_ratio, cond_dim=cond_dim) for _ in range(depth)
        ])
        self.final = nn.LayerNorm(hidden, eps=1e-6)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W), t: (B,) in [0,1]
        tokens = self.patch(x)  # (B,N,D)
        if tokens.size(1) > self.pos.size(1):
            raise ValueError("Increase positional buffer for longer token sequences.")
        tokens = tokens + self.pos[:, :tokens.size(1), :]

        t_emb = timestep_embedding(t, 128)
        cond = self.t_proj(t_emb)  # (B,cond_dim)

        h = tokens
        for blk in self.blocks:
            h = blk(h, cond)
        h = self.final(h)  # (B,N,D)
        # global average pool tokens -> context
        c = h.mean(dim=1)  # (B,D)
        return c  # context vector


# ----------------------
# future increment encoder (Conv1d) to latent z, and decoder back
# ----------------------
class FutureEncoder(nn.Module):
    def __init__(self, horizon: int, in_ch: int = 1, z_dim: int = 128):
        super().__init__()
        # r: (B, H) -> (B,1,H)
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 64, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(64, 128, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(128, z_dim, 3, padding=1),
        )
        self.horizon = horizon
        self.z_dim = z_dim

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        # r: (B,H) or (B,1,H)
        if r.dim() == 2:
            r = r.unsqueeze(1)
        z = self.net(r)  # (B,z_dim,H)
        # flatten over time -> vector
        z = z.mean(dim=-1)  # (B,z_dim)
        return z


class FutureDecoder(nn.Module):
    def __init__(self, horizon: int, z_dim: int = 128):
        super().__init__()
        self.horizon = horizon
        self.net = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.GELU(),
            nn.Linear(128, horizon),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # (B,z_dim) -> (B,H)
        return self.net(z)


# ----------------------
# MarketDiTGen: conditional diffusion + aux heads
# ----------------------
class MarketDiTGen(nn.Module):
    """
    Inputs:
      - X: (B,C,H,W) historical window (features-as-image)
      - r: (B,H) future log-return vector (train only)
      - t: (B,) diffusion time in [0,1]
    Outputs:
      - eps_pred: (B,z_dim) predicted noise in latent z space
      - q_pred:  (B,3) quantiles of cum-return (q10,q50,q90)
      - bar_pred:(B,3) triple-barrier logits (SL/None/TP)
    """
    def __init__(self,
                 in_ch: int,
                 horizon: int = 16,
                 hidden: int = 384,
                 depth: int = 6,
                 heads: int = 6,
                 z_dim: int = 128,
                 mlp_ratio: float = 4.0):
        super().__init__()
        self.ctx = ContextDiT(in_ch, hidden=hidden, depth=depth, heads=heads, mlp_ratio=mlp_ratio, cond_dim=256)
        self.f_enc = FutureEncoder(horizon=horizon, z_dim=z_dim)
        self.f_dec = FutureDecoder(horizon=horizon, z_dim=z_dim)

        # eps prediction head: f( [z_t, t_emb, c] )
        self.t_proj = nn.Linear(128, 256)
        self.eps_mlp = nn.Sequential(
            nn.Linear(z_dim + 256 + hidden, 512),
            nn.GELU(),
            nn.Linear(512, z_dim),
        )
        nn.init.xavier_uniform_(self.eps_mlp[0].weight); nn.init.zeros_(self.eps_mlp[0].bias)
        nn.init.xavier_uniform_(self.eps_mlp[2].weight); nn.init.zeros_(self.eps_mlp[2].bias)

        # auxiliary heads on context c
        self.q_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 3)  # q10,q50,q90
        )
        self.bar_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 3)  # [SL, None, TP]
        )

    # noise schedule helpers
    @staticmethod
    def alpha_sigma(t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cosine schedule-ish mapping t->[alpha,sigma], both (B,1)
        """
        # s-curve between [0.01, 0.999]
        t = t.clamp(0, 1)
        eps = 1e-4
        f = torch.cos((t + eps) / (1 + eps) * math.pi * 0.5)  # in (0,1]
        alpha = f[:, None]
        sigma = torch.sqrt(torch.clamp(1 - alpha**2, 1e-5, 1.0))
        return alpha, sigma

    def forward(self, X: torch.Tensor, r: Optional[torch.Tensor], t: torch.Tensor):
        """
        Train path: X,r,t -> eps_pred,q_pred,bar_pred, z, z_t, eps
        Inference (no r): you can pass r=None and only use context heads.
        """
        B = X.size(0)
        # context with DiT
        c = self.ctx(X, t)  # (B,hidden)

        # time emb for eps head
        t_emb = timestep_embedding(t, 128)
        t_feat = self.t_proj(t_emb)  # (B,256)

        if r is not None:
            # encode future increments -> z
            z = self.f_enc(r)  # (B,z_dim)
            # sample noise & diffuse
            eps = torch.randn_like(z)
            alpha, sigma = self.alpha_sigma(t)  # (B,1)
            z_t = alpha * z + sigma * eps  # (B,z_dim)
            # predict noise
            h = torch.cat([z_t, t_feat, c], dim=-1)
            eps_pred = self.eps_mlp(h)
        else:
            z = z_t = eps = None
            eps_pred = None

        # aux heads
        q_pred = self.q_head(c)
        bar_pred = self.bar_head(c)
        return eps_pred, q_pred, bar_pred, z, z_t, eps

    # convenience
    def sample_paths(self, X: torch.Tensor, H: int, steps: int = 20, w: float = 0.0):
        """
        Optional: simple ancestral sampling of future increments (no CFG by default).
        You can extend with unconditional branch for CFG if需要.
        """
        device = X.device
        B = X.size(0)
        # we maintain z variable in latent space
        z = torch.randn(B, self.f_dec.net[0].in_features, device=device)
        # naive schedule from t=1->0
        for s in range(steps, 0, -1):
            t = torch.full((B,), s / steps, device=device)
            c = self.ctx(X, t)
            t_emb = timestep_embedding(t, 128)
            t_feat = self.t_proj(t_emb)
            alpha, sigma = self.alpha_sigma(t)
            # predict noise at current z (treat z as z_t)
            h = torch.cat([z, t_feat, c], dim=-1)
            eps_pred = self.eps_mlp(h)
            # simple DDIM-like update
            z = (z - sigma * eps_pred) / (alpha + 1e-6)
        r_hat = self.f_dec(z)  # (B,H)
        return r_hat
