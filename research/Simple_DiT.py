# models/market_dit.py
import torch, torch.nn as nn
import torch.nn.functional as F

class MSA(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.qkv = nn.Linear(dim, dim*3, bias=True)
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):  # (B, N, D)
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, D//self.heads).permute(2,0,3,1,4)
        q,k,v = qkv[0], qkv[1], qkv[2]  # (B, heads, N, D_head)
        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1,2).reshape(B, N, D)
        return self.proj(out)

class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0):
        super().__init__()
        hidden = int(dim*mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
    def forward(self, x):
        return self.fc2(F.silu(self.fc1(x)))

class DiTBlockLite(nn.Module):
    def __init__(self, dim, heads=8, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = MSA(dim, heads=heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp  = MLP(dim, mlp_ratio=mlp_ratio)
    def forward(self, x):                 # (B, N, D)
        x = x + self.attn(self.norm1(x)) # 残差
        x = x + self.mlp(self.norm2(x))  # 残差
        return x

class MarketDiT(nn.Module):
    def __init__(self, in_ch=8, H=16, W=16, dim=256, depth=6, heads=8, num_classes=2):
        super().__init__()
        self.H, self.W = H, W
        self.patch_embed = nn.Conv2d(in_ch, dim, kernel_size=3, padding=1)  # C'→D
        self.pos = nn.Parameter(torch.zeros(1, H*W, dim))
        self.blocks = nn.ModuleList([DiTBlockLite(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):  # x: (B, C, H, W)
        h = self.patch_embed(x)          # (B, D, H, W)
        h = h.flatten(2).transpose(1,2)  # (B, N=H*W, D)
        h = h + self.pos
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h).mean(dim=1)     # token 平均
        logits = self.head(h)            # (B, 2)
        return logits
