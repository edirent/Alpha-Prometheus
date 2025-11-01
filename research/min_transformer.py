import math
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- 多尺度位置编码：同时注入不同“时间尺度”的正余弦 ----
class MultiScalePositionalEncoding(nn.Module):
    """
    把不同scale的正余弦PE叠加（或拼接后线性投影）。
    例如 scales=[1,4,16,64] 大致对应捕捉 1、4、16、64 步长的周期/趋势。
    """
    def __init__(self, d_model: int, max_len: int = 8192, scales: Optional[List[int]] = None, mode: str = "sum"):
        super().__init__()
        assert d_model % 2 == 0, "d_model 必须为偶数以便一半sin一半cos"
        self.d_model = d_model
        self.max_len = max_len
        self.scales = scales or [1, 4, 16, 64]
        self.mode = mode  # "sum" 更简洁；"concat" 会把各尺度拼接后线性投影到 d_model

        # 为每个尺度预计算一个基本的正余弦表
        # pe_s[scale] 的形状: (1, max_len, d_model)
        self.tables = nn.ParameterList()
        self.register_buffer("_dummy", torch.empty(0), persistent=False)  # 为了设备对齐

        if self.mode == "sum":
            # 每个尺度各自用 d_model 维
            for s in self.scales:
                self.tables.append(nn.Parameter(self._make_table(self.d_model, s), requires_grad=False))
        elif self.mode == "concat":
            # 拼接后再线性映射回 d_model
            self.concat_dim = self.d_model * len(self.scales)
            for s in self.scales:
                self.tables.append(nn.Parameter(self._make_table(self.d_model, s), requires_grad=False))
            self.proj = nn.Linear(self.concat_dim, self.d_model, bias=False)
        else:
            raise ValueError("mode 必须为 'sum' 或 'concat'")

    def _make_table(self, d_model: int, scale: int) -> torch.Tensor:
        pe = torch.zeros(self.max_len, d_model)
        position = torch.arange(0, self.max_len, dtype=torch.float32).unsqueeze(1)  # (T,1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        # 缩放 “位置” 以改变有效频率（scale 越大，周期越长，更偏长周期）
        pos_scaled = position / float(scale)
        pe[:, 0::2] = torch.sin(pos_scaled * div_term)
        pe[:, 1::2] = torch.cos(pos_scaled * div_term)
        return pe.unsqueeze(0)  # (1, T, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, d_model)
        return: (B, T, d_model) 注入多尺度位置信息
        """
        B, T, D = x.shape
        device = x.device

        if self.mode == "sum":
            out = x
            for table in self.tables:
                pe = table.to(device)[:, :T, :]
                out = out + pe
            return out
        else:
            # concat 模式，把各尺度 pe 拼起来 → 线性映射回 d_model
            pes = []
            for table in self.tables:
                pes.append(table.to(device)[:, :T, :])  # (1,T,D)
            pe_cat = torch.cat(pes, dim=-1)  # (1,T,D*len(scales))
            pe_cat = pe_cat.expand(B, -1, -1)  # (B,T,ConcatD)
            return x + self.proj(pe_cat)

# ---- 最小时间序列 Transformer（多头注意力） ----
class TinyTimeTransformer(nn.Module):
    """
    最小可用：
    - 输入: (B, T, F) 特征序列
    - 线性投影 -> d_model
    - 多尺度PE
    - TransformerEncoder (多头注意力)
    - 取最后时间步表示 -> 线性头 -> 回归/分类输出
    """
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        out_dim: int = 1,
        pe_scales: Optional[List[int]] = None,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = MultiScalePositionalEncoding(d_model, scales=pe_scales or [1, 4, 16, 64], mode="sum")

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # 直接使用 (B,T,E) 形状
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, out_dim)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, T, F)
        attn_mask: 可选 (T,T) 或 (B*nhead, T, T)，不传就是全局双向注意力（适合“看全历史”的判别/信号）
        """
        h = self.input_proj(x)              # (B,T,d_model)
        h = self.pos_enc(h)                 # 注入多尺度时序信息
        h = self.encoder(h, mask=attn_mask) # (B,T,d_model)
        h_last = h[:, -1, :]                # 取最后一步（预测"下一步"或窗口整体信号）
        y = self.head(h_last)               # (B,out_dim)
        return y

# ---- 一个可直接运行的小 demo ----
def _demo():
    torch.manual_seed(0)
    B, T, F = 16, 256, 8   # batch / 序列长度 / 特征维
    x = torch.randn(B, T, F)

    model = TinyTimeTransformer(input_dim=F, d_model=128, nhead=4, num_layers=2, out_dim=1)
    y = model(x)  # (B,1)
    print("output shape:", y.shape)

    # 简单的“下一步收益”回归 toy 训练（10步）
    target = torch.randn(B, 1)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in range(10):
        pred = model(x)
        loss = F.mse_loss(pred, target)
        optim.zero_grad()
        loss.backward()
        optim.step()
        if (step+1) % 2 == 0:
            print(f"step {step+1:02d}  loss={loss.item():.4f}")

if __name__ == "__main__":
    _demo()
