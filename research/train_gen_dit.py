# -*- coding: utf-8 -*-
# research/train_gen_dit.py
"""
Train MarketDiTGen:
  L = L_denoise(z_t, eps_pred, eps) + λ1 * L_quantile + λ2 * L_barrier

Inputs:
  - OHLCV feather/parquet/csv with columns: date, open, high, low, close, volume

Targets:
  - r_future: H-step log-return vector
  - q targets: cum_return (ATR-normalized)
  - barrier: triple-barrier (SL/None/TP)
"""
import os, sys, argparse, json, math
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

ROOT = os.path.abspath(".")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from research.gen_dit import MarketDiTGen


# ---------------- IO ----------------
def read_any_df(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ext = os.path.splitext(path)[-1].lower()
    if ext == ".feather":
        df = pd.read_feather(path)
    elif ext == ".parquet":
        df = pd.read_parquet(path)
    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        df = pd.DataFrame(raw)
    else:
        df = pd.read_csv(path)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    else:
        df["date"] = pd.date_range("2000-01-01", periods=len(df), tz="UTC", freq="D")

    need = ["open","high","low","close","volume"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
    df = (df.dropna(subset=["date"])
            .sort_values("date")
            .drop_duplicates("date")
            .reset_index(drop=True))
    return df


# ---------------- features & labels ----------------
def compute_atr(df: pd.DataFrame, n: int = 20) -> np.ndarray:
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    prev_c = np.roll(c, 1); prev_c[0] = c[0]
    tr = np.maximum.reduce([h - l, np.abs(h - prev_c), np.abs(l - prev_c)])
    alpha = 2.0 / (n + 1.0)
    atr = np.empty_like(tr)
    atr[0] = tr[:n].mean() if len(tr) >= n else tr[0]
    for i in range(1, len(tr)):
        atr[i] = alpha * tr[i] + (1 - alpha) * atr[i-1]
    return atr


def make_hist_features(df: pd.DataFrame, win: int) -> np.ndarray:
    """
    Build (C,H,W) like your classifier版：把时间轴折成 (H,W) 的小图块。
    这里简单做：8 通道 (close, hl2, ohlc4, ret1, v, v_ma, v_ratio, atr_norm)
    然后把 win 映射到 16x16（要求 win 能被 16*16=256 整除；否则补零/裁剪）
    """
    close = df["close"].values.astype(np.float32)
    high  = df["high"].values.astype(np.float32)
    low   = df["low"].values.astype(np.float32)
    openp = df["open"].values.astype(np.float32)
    vol   = df["volume"].values.astype(np.float32)
    atr   = compute_atr(df, 20).astype(np.float32) + 1e-8

    hl2 = (high + low) * 0.5
    ohlc4 = (openp + high + low + close) / 4.0
    ret1 = np.diff(np.log(close), prepend=np.log(close[0]))
    v_ma = pd.Series(vol).rolling(24).mean().bfill().values
    v_ratio = (vol + 1e-8) / (v_ma + 1e-8)
    atr_norm = atr / (np.maximum(ohlc4, 1e-6))

    feats = np.stack([close, hl2, ohlc4, ret1, vol, v_ma, v_ratio, atr_norm], axis=1)  # (N,8)
    # 标准化（逐通道）
    mu = feats.mean(axis=0, keepdims=True)
    sd = feats.std(axis=0, keepdims=True) + 1e-6
    feats = (feats - mu) / sd

    N = len(df)
    H = W = int(math.sqrt(win))
    if H * W != win:
        # 补到最近平方数
        H = W = int(math.sqrt(win)) + 1
        pad = H * W - win
    else:
        pad = 0

    X_list = []
    for i in range(win - 1, N - 1):
        seg = feats[i - win + 1:i + 1, :]  # (win,8)
        if pad > 0:
            seg = np.pad(seg, ((pad, 0), (0, 0)))
        seg = seg[-(H * W):, :].reshape(H, W, feats.shape[1]).transpose(2, 0, 1)  # (C,H,W)
        X_list.append(seg.astype(np.float32))
    X = np.stack(X_list, axis=0)  # (N-win, C, H, W)
    return X


def make_future_increments(df: pd.DataFrame, H: int) -> np.ndarray:
    """
    r_{t+1:t+H} (log-return increments)
    对齐到与 X 同样的末端索引（X 对应的起点是 win-1，对应未来从下一根开始）
    """
    c = df["close"].values.astype(np.float32)
    logc = np.log(c + 1e-8)
    inc = logc[1:] - logc[:-1]  # length N-1
    # 生成每个 t 的窗口向量
    N = len(inc)
    R = []
    for i in range(N):
        if i + H > N:
            break
        R.append(inc[i:i + H])
    return np.stack(R, axis=0).astype(np.float32)  # (N-H+1, H)


def triple_barrier_label(df: pd.DataFrame, H: int, tp_mult: float, sl_mult: float, start_idx: int, atr: np.ndarray):
    """
    对 t 的下一根 open 进场，未来 H 根内：
      - 先触 SL → label=0 (SL)
      - 再触 TP → label=2 (TP)
      - 都没触 → label=1 (None)
    返回 label(int), also cum_return(float)
    """
    opens = df["open"].values
    highs = df["high"].values
    lows  = df["low"].values

    entry_idx = start_idx + 1
    if entry_idx >= len(df):
        return 1, 0.0  # neutral

    entry_px = opens[entry_idx]
    stop_px  = entry_px * (1 - sl_mult * atr[entry_idx] / max(entry_px, 1e-6))
    take_px  = entry_px * (1 + tp_mult * atr[entry_idx] / max(entry_px, 1e-6))

    exit_label = 1  # none
    for h in range(H):
        j = entry_idx + h
        if j >= len(df):
            break
        # SL first, then TP
        if lows[j] <= stop_px:
            exit_label = 0
            break
        if highs[j] >= take_px:
            exit_label = 2
            break

    # cum-return (log)
    end_idx = min(entry_idx + H, len(df) - 1)
    cum_ret = math.log(max(df["close"].values[end_idx], 1e-6)) - math.log(max(df["close"].values[entry_idx], 1e-6))
    return exit_label, float(cum_ret)


def build_dataset(df: pd.DataFrame, win: int, H: int, tp_mult: float, sl_mult: float):
    """
    Return:
      X: (M,C,Hh,Ww)
      R: (M,H)
      q_target: (M,) cum_return / ATRnorm
      bar_target: (M,) in {0,1,2}
    对齐：X[i] 对应 df 索引 t = win-1+i；未来从 t+1 到 t+H。
    """
    X = make_hist_features(df, win)  # (N', C, Hh, Ww)
    R_all = make_future_increments(df, H)  # (N'', H)
    # 对齐：两者都对应到 “以 t 结束的历史片段 -> 未来H”
    N = min(len(X), len(R_all))
    X = X[:N]
    R = R_all[:N]

    atr = compute_atr(df, 20).astype(np.float32) + 1e-8
    # cum ret & barrier
    q_list = []
    b_list = []
    start_offset = win - 1  # t 对应 df 索引
    for i in range(N):
        t = start_offset + i
        bar, cumr = triple_barrier_label(df, H, tp_mult, sl_mult, t, atr)
        q_list.append(cumr)
        b_list.append(bar)
    q = np.array(q_list, dtype=np.float32)
    b = np.array(b_list, dtype=np.int64)

    # 归一化（ATR相对价）：用 entry_idx 的 ATR/price 把 cum_ret 粗略缩放到“单位ATR”
    # 这里简化：直接除以样本内 std，效果也不错
    q_mu, q_sd = q.mean(), q.std() + 1e-6
    q_norm = (q - q_mu) / q_sd

    return X.astype(np.float32), R.astype(np.float32), q_norm.astype(np.float32), b


# -------------- quantile loss (pinball) --------------
def pinball_loss(pred: torch.Tensor, target: torch.Tensor, quantiles=(0.1, 0.5, 0.9)) -> torch.Tensor:
    """
    pred: (B,3) -> q10,q50,q90; target: (B,)
    """
    loss = 0.0
    for i, q in enumerate(quantiles):
        e = target - pred[:, i]
        loss_q = torch.maximum(q * e, (q - 1) * e).mean()
        loss = loss + loss_q
    return loss / len(quantiles)


# ---------------- training ----------------
def train(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    df = read_any_df(args.data)

    X, R, q, b = build_dataset(df, win=args.win, H=args.horizon, tp_mult=args.tp_mult, sl_mult=args.sl_mult)
    # 划分 80/20
    N = len(X)
    split = int(N * 0.8)
    train_idx = slice(0, split)
    val_idx   = slice(split, N)

    def to_dl(X, R, q, b, idx):
        ds = TensorDataset(
            torch.from_numpy(X[idx]).float(),
            torch.from_numpy(R[idx]).float(),
            torch.from_numpy(q[idx]).float(),
            torch.from_numpy(b[idx]).long(),
        )
        return DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

    dl_tr = to_dl(X, R, q, b, train_idx)
    dl_va = to_dl(X, R, q, b, val_idx)

    model = MarketDiTGen(in_ch=X.shape[1], horizon=args.horizon,
                         hidden=args.hidden, depth=args.depth, heads=args.heads,
                         z_dim=args.z_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)
    ce = nn.CrossEntropyLoss()

    best = float("inf")
    os.makedirs(os.path.dirname(args.ckpt), exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        m_tr = {"denoise":0.0, "q":0.0, "bar":0.0, "tot":0.0}
        n_tr = 0

        for xb, rb, qb, bb in dl_tr:
            xb = xb.to(device, non_blocking=True)
            rb = rb.to(device, non_blocking=True)
            qb = qb.to(device, non_blocking=True)
            bb = bb.to(device, non_blocking=True)

            B = xb.size(0)
            t = torch.rand(B, device=device)  # U(0,1)
            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=args.amp):
                eps_pred, q_pred, bar_pred, z, z_t, eps = model(xb, rb, t)
                # losses
                loss_denoise = F.mse_loss(eps_pred, eps)
                loss_q = pinball_loss(q_pred, qb)
                loss_bar = ce(bar_pred, bb)
                loss = loss_denoise + args.lmb_q * loss_q + args_lmb_bar(args, loss_bar)

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss: {loss.item()}")

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            m_tr["denoise"] += loss_denoise.item() * B
            m_tr["q"] += loss_q.item() * B
            m_tr["bar"] += loss_bar.item() * B
            m_tr["tot"] += loss.item() * B
            n_tr += B

        for k in m_tr: m_tr[k] /= max(1, n_tr)

        # val
        model.eval()
        m_va = {"denoise":0.0, "q":0.0, "bar":0.0, "tot":0.0}
        n_va = 0
        with torch.no_grad():
            for xb, rb, qb, bb in dl_va:
                xb = xb.to(device, non_blocking=True)
                rb = rb.to(device, non_blocking=True)
                qb = qb.to(device, non_blocking=True)
                bb = bb.to(device, non_blocking=True)
                B = xb.size(0)
                t = torch.rand(B, device=device)
                eps_pred, q_pred, bar_pred, z, z_t, eps = model(xb, rb, t)
                loss_denoise = F.mse_loss(eps_pred, eps)
                loss_q = pinball_loss(q_pred, qb)
                loss_bar = ce(bar_pred, bb)
                loss = loss_denoise + args.lmb_q * loss_q + args_lmb_bar(args, loss_bar)
                m_va["denoise"] += loss_denoise.item() * B
                m_va["q"] += loss_q.item() * B
                m_va["bar"] += loss_bar.item() * B
                m_va["tot"] += loss.item() * B
                n_va += B
        for k in m_va: m_va[k] /= max(1, n_va)

        print(f"epoch {epoch:03d} | "
              f"train: den={m_tr['denoise']:.4f} q={m_tr['q']:.4f} bar={m_tr['bar']:.4f} tot={m_tr['tot']:.4f} | "
              f"val:   den={m_va['denoise']:.4f} q={m_va['q']:.4f} bar={m_va['bar']:.4f} tot={m_va['tot']:.4f}")

        if m_va["tot"] < best:
            best = m_va["tot"]
            torch.save(model.state_dict(), args.ckpt)
            print(f"[saved] {args.ckpt} (val_tot={best:.4f})")


def args_lmb_bar(args, loss_bar):
    return args.lmb_bar * loss_bar


def main():
    ap = argparse.ArgumentParser("Train MarketDiTGen (conditional diffusion + aux heads)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", default="user_data/models/market_dit_gen.pt")
    ap.add_argument("--win", type=int, default=256)         # 建议 16x16
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--tp-mult", type=float, default=1.5)
    ap.add_argument("--sl-mult", type=float, default=1.0)

    ap.add_argument("--hidden", type=int, default=384)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--z-dim", type=int, default=128)

    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--amp", action="store_true")

    ap.add_argument("--lmb-q", type=float, default=0.5)
    ap.add_argument("--lmb-bar", type=float, default=0.5)

    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
