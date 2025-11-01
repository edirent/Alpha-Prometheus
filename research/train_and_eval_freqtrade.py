import argparse, os, glob, math, json, gzip
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
import torch, torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import joblib

# ========= Tiny Transformer（同前） =========
import math
class MultiScalePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 8192, scales: Optional[List[int]] = None, mode: str = "sum"):
        super().__init__()
        assert d_model % 2 == 0
        self.d_model = d_model
        self.max_len = max_len
        self.scales = scales or [1, 4, 16, 64]
        self.mode = mode
        self.tables = nn.ParameterList()
        if self.mode == "sum":
            for s in self.scales:
                self.tables.append(nn.Parameter(self._make_table(self.d_model, s), requires_grad=False))
        elif self.mode == "concat":
            for s in self.scales:
                self.tables.append(nn.Parameter(self._make_table(self.d_model, s), requires_grad=False))
            self.proj = nn.Linear(self.d_model * len(self.scales), self.d_model, bias=False)
        else:
            raise ValueError("mode must be 'sum' or 'concat'")
    def _make_table(self, d_model: int, scale: int) -> torch.Tensor:
        pe = torch.zeros(self.max_len, d_model)
        position = torch.arange(0, self.max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pos_scaled = position / float(scale)
        pe[:, 0::2] = torch.sin(pos_scaled * div_term)
        pe[:, 1::2] = torch.cos(pos_scaled * div_term)
        return pe.unsqueeze(0)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        if self.mode == "sum":
            out = x
            for table in self.tables:
                out = out + table[:, :T, :].to(x.device)
            return out
        else:
            pes = [table[:, :T, :].to(x.device) for table in self.tables]
            pe_cat = torch.cat(pes, dim=-1).expand(B, -1, -1)
            return x + self.proj(pe_cat)

class TinyTimeTransformer(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 128, nhead: int = 4, num_layers: int = 2,
                 dim_feedforward: int = 256, dropout: float = 0.1, out_dim: int = 1,
                 pe_scales: Optional[List[int]] = None):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = MultiScalePositionalEncoding(d_model, scales=pe_scales or [1,4,16,64], mode="sum")
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                               dim_feedforward=dim_feedforward,
                                               dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, out_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.pos_enc(h)
        h = self.encoder(h)
        y = self.head(h[:, -1, :])
        return y

# ========= 数据加载（兼容多种 Freqtrade/CCXT JSON 格式） =========
def find_freqtrade_file(data_root: str, exchange: str, pair: str, timeframe: str) -> str:
    pair_fn = pair.replace('/', '_')
    # 兼容 data_root 传到 exchange 层或其上一层的两种用法
    candidates = []
    base1 = os.path.join(data_root, exchange)
    base2 = data_root  # 直接就是 /.../data/binance
    for base in [base1, base2]:
        for pat in [
            os.path.join(base, f"{pair_fn}-{timeframe}.json"),
            os.path.join(base, f"{pair_fn}-{timeframe}.json.gz"),
            os.path.join(base, "spot", f"{pair_fn}-{timeframe}.json"),
            os.path.join(base, "spot", f"{pair_fn}-{timeframe}.json.gz"),
            os.path.join(base, f"{pair_fn}-{timeframe}.ohlcv.json"),
            os.path.join(base, f"{pair_fn}-{timeframe}.ohlcv.json.gz"),
        ]:
            if os.path.isfile(pat):
                candidates.append(pat)
    if candidates:
        return sorted(candidates)[0]
    # 最后兜底：通配递归搜
    globs = []
    for root, dirs, files in os.walk(data_root):
        for fn in files:
            if fn.startswith(f"{pair_fn}-{timeframe}") and fn.endswith(".json") or fn.endswith(".json.gz"):
                globs.append(os.path.join(root, fn))
    if not globs:
        raise FileNotFoundError(f"未找到数据文件：在 {data_root} 下未匹配到 {pair_fn}-{timeframe}.json(.gz)")
    return sorted(globs)[0]

def _to_datetime(ts):
    # 自动判断秒/毫秒
    ts = int(ts)
    if ts > 10_000_000_000:  # > ~2001年（毫秒）
        return pd.to_datetime(ts, unit="ms", utc=True)
    return pd.to_datetime(ts, unit="s", utc=True)

def read_freqtrade_ohlcv(path: str) -> pd.DataFrame:
    # 支持：
    # 1) [[ts, o, h, l, c, v], ...]
    # 2) [{"date": "...", "open":...}, ...] 或 {"timestamp": ...}
    # 3) {"data":[...]} 或 {"ohlcv":[...]} 嵌套
    if path.endswith(".gz"):
        with gzip.open(path, 'rt') as f:
            raw = json.load(f)
    else:
        with open(path, 'r') as f:
            raw = json.load(f)

    # 嵌套解包
    if isinstance(raw, dict):
        for k in ["data", "ohlcv", "result"]:
            if k in raw:
                raw = raw[k]
                break

    if isinstance(raw, list) and len(raw) > 0:
        first = raw[0]
        # 1) list-of-lists
        if isinstance(first, (list, tuple)) and len(first) >= 6:
            arr = np.array(raw, dtype=object)
            df = pd.DataFrame(arr[:, :6], columns=["ts","open","high","low","close","volume"])
            df["date"] = pd.to_datetime([_to_datetime(x) for x in df["ts"]], utc=True)
            df = df.drop(columns=["ts"])
        # 2) list-of-dicts
        elif isinstance(first, dict):
            # 常见键名集合
            time_keys = ["date", "timestamp", "time", "ts"]
            ohlc_map = {
                "open": ["open","o"],
                "high": ["high","h"],
                "low":  ["low","l"],
                "close":["close","c"],
                "volume":["volume","v"]
            }
            recs = []
            for d in raw:
                # 时间
                tval = None
                for k in time_keys:
                    if k in d:
                        tval = d[k]; break
                if tval is None:
                    continue
                # ohlcv
                row = {"date": pd.to_datetime(tval, utc=True) if not isinstance(tval,(int,float)) else _to_datetime(tval)}
                ok = True
                for std, aliases in ohlc_map.items():
                    val = None
                    for a in aliases:
                        if a in d:
                            val = d[a]; break
                    if val is None:
                        ok = False; break
                    row[std] = float(val)
                if ok:
                    recs.append(row)
            if not recs:
                raise ValueError("无法从字典列表解析出 OHLCV 字段。")
            df = pd.DataFrame(recs)
        else:
            raise ValueError("未识别的 OHLCV 列表元素结构。")
    else:
        raise ValueError("JSON 不是列表或为空，无法解析。")

    df = df.sort_values("date").set_index("date")
    # 转为 float
    for col in ["open","high","low","close","volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna()
    return df

# ========= 特征工程 =========
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ret"] = df["close"].pct_change()
    df["log_r"] = np.log(df["close"]).diff()
    df["hl_spread"] = (df["high"] - df["low"]) / df["close"]
    df["oc_spread"] = (df["open"] - df["close"]) / df["close"]
    df["sma20"] = df["close"].rolling(20).mean()
    df["ema12"] = df["close"].ewm(span=12, adjust=False).mean()
    delta = df["close"].diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = -delta.clip(upper=0).rolling(14).mean()
    df["rsi14"] = 100 - 100 / (1 + (up / (down + 1e-12)))
    df["vol_chg"] = df["volume"].pct_change()
    return df

def make_windows(df: pd.DataFrame, feature_cols: List[str], seq_len: int, horizon: int,
                 scaler: Optional[StandardScaler] = None, fit_scaler: bool = True):
    df = df.dropna().copy()
    feats = df[feature_cols].values
    if scaler is None:
        scaler = StandardScaler()
    feats = scaler.fit_transform(feats) if fit_scaler else scaler.transform(feats)
    X, y, idx = [], [], []
    close = df["close"].values
    dates = df.index
    for i in range(len(df) - seq_len - horizon + 1):
        X.append(feats[i:i+seq_len])
        fut_ret = (close[i+seq_len+horizon-1] / close[i+seq_len-1]) - 1.0
        y.append(fut_ret); idx.append(dates[i+seq_len-1])
    return np.asarray(X, np.float32), np.asarray(y, np.float32), pd.to_datetime(idx), scaler

# ========= 简易评估 =========
def directional_acc(y_true, y_pred):
    return float((np.sign(y_true) == np.sign(y_pred)).mean())

def annualized_sharpe(pnl_series: np.ndarray, bars_per_year: float):
    mean = float(np.nanmean(pnl_series)); std = float(np.nanstd(pnl_series))
    return 0.0 if std == 0 else float((mean / std) * math.sqrt(bars_per_year))

# ========= 主流程 =========
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default="user_data/data", help="Freqtrade 数据根目录（可以指到 data/binance）")
    ap.add_argument("--exchange", type=str, default="binance")
    ap.add_argument("--pair", type=str, default="BTC/USDT")
    ap.add_argument("--timeframe", type=str, default="15m")
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--ffn", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--save-dir", type=str, default="models")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    path = find_freqtrade_file(args.data_root, args.exchange, args.pair, args.timeframe)
    print(f"[INFO] 使用数据文件: {path}")
    df = read_freqtrade_ohlcv(path)

    tf_map = {"1m":1,"3m":3,"5m":5,"15m":15,"30m":30,"1h":60,"4h":240,"1d":1440}
    bars_per_year = (24*60 // tf_map.get(args.timeframe, 15)) * 365

    df_feat = add_features(df)
    feature_cols = ["ret","log_r","hl_spread","oc_spread","sma20","ema12","rsi14","vol_chg"]

    cut = int(len(df_feat) * 0.8)
    df_tr = df_feat.iloc[:cut].copy()
    df_te = df_feat.iloc[max(0, cut - (args.seq_len + args.horizon)):].copy()

    Xtr, ytr, idx_tr, scaler = make_windows(df_tr, feature_cols, args.seq_len, args.horizon, scaler=None, fit_scaler=True)
    Xte, yte, idx_te, _      = make_windows(df_te, feature_cols, args.seq_len, args.horizon, scaler=scaler, fit_scaler=False)

    print(f"[INFO] 训练样本: {Xtr.shape}, 测试样本: {Xte.shape}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyTimeTransformer(input_dim=len(feature_cols),
                                d_model=args.d_model, nhead=args.nhead,
                                num_layers=args.layers, dim_feedforward=args.ffn,
                                dropout=args.dropout, out_dim=1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    crit = nn.MSELoss()

    def batches(X, y, bs):
        n = len(X)
        for i in range(0, n, bs):
            yield X[i:i+bs], y[i:i+bs]

    model.train()
    for ep in range(1, args.epochs+1):
        tot = 0.0
        for xb, yb in batches(Xtr, ytr, args.batch_size):
            xb_t = torch.from_numpy(xb).to(device)
            yb_t = torch.from_numpy(yb).unsqueeze(-1).to(device)
            with torch.autocast(device_type=("cuda" if device=="cuda" else "cpu"),
                                dtype=torch.bfloat16, enabled=(device=="cuda")):
                pred = model(xb_t)
                loss = crit(pred, yb_t)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.item()) * len(xb)
        tr_loss = tot / len(Xtr)

        with torch.no_grad():
            xb = torch.from_numpy(Xte[:1024]).to(device)
            yb = torch.from_numpy(yte[:1024]).unsqueeze(-1).to(device)
            pred = model(xb); te_loss = float(crit(pred, yb).item())

        print(f"[EPOCH {ep:03d}] train_loss={tr_loss:.6f}  test_loss(sample)={te_loss:.6f}")

    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in batches(Xte, yte, 1024):
            xb_t = torch.from_numpy(xb).to(device)
            preds.append(model(xb_t).squeeze(-1).cpu().numpy())
    ypred = np.concatenate(preds, axis=0)

    mse = float(np.mean((ypred - yte)**2))
    dacc = float((np.sign(ypred) == np.sign(yte)).mean())
    pnl = yte * (ypred > 0.0)
    sharpe = annualized_sharpe(pnl, bars_per_year)

    print("\n======== 测试集指标 ========")
    print(f"MSE: {mse:.8f}")
    print(f"方向准确率: {dacc*100:.2f}%")
    print(f"年化夏普(粗略, thr=0): {sharpe:.3f}")
    print(f"测试样本区间: {idx_te.min()} ~ {idx_te.max()}  | 样本数={len(yte)}")

    torch.save(model.state_dict(), os.path.join(args.save_dir, "transformer_ts.pth"))
    joblib.dump(scaler, os.path.join(args.save_dir, "scaler.pkl"))
    meta = {
        "feature_cols": feature_cols,
        "seq_len": args.seq_len,
        "horizon": args.horizon,
        "timeframe": args.timeframe,
        "pair": args.pair,
        "exchange": args.exchange,
    }
    with open(os.path.join(args.save_dir, "meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] 已保存: models/transformer_ts.pth, models/scaler.pkl, models/meta.json")

if __name__ == "__main__":
    main()
