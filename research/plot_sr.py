import os, json, gzip, math
import numpy as np, pandas as pd
import torch, torch.nn as nn
import matplotlib.pyplot as plt

# ======== 与之前相同的工具函数（read_ohlcv, make_features, make_windows, SRTransformer 等） ========
# ... 这里保持和你已有的版本一致，我只修改 main() 的后半部分 ...

# ---------- 模型定义 ----------
class PosEnc(nn.Module):
    def __init__(self, d_model, max_len=8192):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()*(-math.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(pos*div); pe[:,1::2] = torch.cos(pos*div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1), :]

class SRTransformer(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=4, layers=2, ffn=256, dropout=0.1, num_classes=3):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pe = PosEnc(d_model)
        enc = nn.TransformerEncoderLayer(d_model, nhead, ffn, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(d_model, num_classes)
    def forward(self, x):
        h = self.proj(x); h = self.pe(h); h = self.enc(h); return self.head(h[:,-1,:])

# ---------- 画K线 ----------
def plot_candles(ax, df: pd.DataFrame):
    xs = np.arange(len(df))
    for x, hi, lo in zip(xs, df["high"], df["low"]):
        ax.vlines(x, lo, hi, linewidth=1)
    w = 0.3
    for x, o, c in zip(xs, df["open"], df["close"]):
        ax.hlines(o, x-w, x, linewidth=2)
        ax.hlines(c, x, x+w, linewidth=2)
    ax.set_xlim(-1, len(df))

# ======== 简化版 pivot+zone 计算函数 ========
def atr(df, n=14):
    h,l,c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h-l).abs(), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def find_pivots(df, left=3, right=3):
    close = df["close"].values
    lo = np.zeros(len(df), dtype=bool); hi = np.zeros(len(df), dtype=bool)
    for i in range(left, len(df)-right):
        win = close[i-left:i+right+1]
        if close[i] == win.min(): lo[i]=True
        if close[i] == win.max(): hi[i]=True
    return lo, hi

def build_zones(df, lookback=800, merge_k=1.2, min_touches=3):
    closes = df["close"].values
    atr14 = atr(df,14).fillna(method="bfill")
    piv_lo, piv_hi = find_pivots(df)
    def cluster(vals, thr):
        if len(vals)==0: return []
        vals = np.sort(vals); groups=[[vals[0]]]
        for v in vals[1:]:
            if abs(v - np.mean(groups[-1])) <= thr: groups[-1].append(v)
            else: groups.append([v])
        zones=[]
        for g in groups:
            if len(g) >= min_touches:
                zones.append(np.mean(g))
        return zones
    thr = merge_k * float(np.nanmedian(atr14))
    lows  = closes[piv_lo]
    highs = closes[piv_hi]
    return cluster(lows,thr), cluster(highs,thr)

# ======== 主函数 ========
def main():
    data_path = os.path.expanduser("~/Alpha-Prometheus/user_data/data/binance/BTC_USDT-15m.json")
    meta_path = os.path.expanduser("~/Alpha-Prometheus/models/sr_meta.json")
    model_path = os.path.expanduser("~/Alpha-Prometheus/models/sr_transformer.pth")

    # 读数据
    import pandas as pd, gzip, json
    df_raw = read_ohlcv(data_path)

    # 只取最近一段画图
    df_plot = df_raw.iloc[-240:].copy()

    # --- 画K线 ---
    fig, ax = plt.subplots(figsize=(12,6))
    plot_candles(ax, df_plot)

    # --- 叠加 zones ---
    lows, highs = build_zones(df_raw, lookback=1200, merge_k=1.2, min_touches=3)
    xs = np.arange(len(df_plot))
    for z in lows:
        ax.hlines(z, xs[0], xs[-1], linewidth=3, color="red")
    for z in highs:
        ax.hlines(z, xs[0], xs[-1], linewidth=3, color="red")

    ax.set_title("BTC/USDT 15m — 多次确认的支撑/阻力水平")
    ax.set_xlabel("bars"); ax.set_ylabel("price")
    plt.tight_layout()
    plt.show()

def read_ohlcv(path: str):
    import gzip, json, pandas as pd
    if path.endswith(".gz"):
        with gzip.open(path, "rt") as f:
            raw = json.load(f)
    else:
        with open(path, "r") as f:
            raw = json.load(f)
    df = pd.DataFrame(raw)

    # 如果是list-of-lists (freqtrade/ccxt 格式)
    if 0 in df.columns:
        df.columns = ["time","open","high","low","close","volume"]

    # 转换时间戳
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    return df


if __name__ == "__main__":
    main()
