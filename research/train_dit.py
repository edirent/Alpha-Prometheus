# -*- coding: utf-8 -*-
"""
Train Market DiT - research/train_dit.py

特性：
- 支持 CSV / Parquet / Feather / Freqtrade-JSON 自动加载
- AMP 混合精度 (--amp) + unscale 后梯度裁剪
- 固定随机种子 (--seed)
- 断点续训 (--resume)
- 输入标准化 + NaN/Inf 安全处理 + 截断，避免 loss=nan
"""
import os
import sys
import json
import time
import argparse
import logging
import traceback
from datetime import datetime

# 允许以模块方式运行时导入 research 包
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils as nn_utils
from torch.utils.data import TensorDataset, DataLoader
from torch.amp import GradScaler, autocast

# 你的实现文件
from research.Simple_DiT import MarketDiT
from research.prepare_dataset import make_market_tensor


# ---------------------- Utils ----------------------
def setup_logger(log_path: str = None, level=logging.INFO):
    fmt = "[%(asctime)s] %(levelname)s: %(message)s"
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    logging.basicConfig(level=level, format=fmt, handlers=handlers)


def set_seed(seed: int | None):
    if seed is None:
        return
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 追求确定性（稍慢）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_any_df(path: str) -> pd.DataFrame:
    """
    自动识别常见格式：
    - .csv / .parquet / .feather
    - Freqtrade JSON: 至少包含 date 或 timestamp；以及 open/high/low/close/volume
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    ext = os.path.splitext(path)[-1].lower()
    if ext == ".parquet":
        df = pd.read_parquet(path)
    elif ext == ".feather":
        df = pd.read_feather(path)
    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        df = pd.DataFrame(raw)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        elif "timestamp" in df.columns:
            ts = pd.to_numeric(df["timestamp"], errors="coerce")
            if ts.dropna().iloc[0] > 1e13:
                df["date"] = pd.to_datetime(ts, unit="ns", utc=True)
            elif ts.dropna().iloc[0] > 1e10:
                df["date"] = pd.to_datetime(ts, unit="ms", utc=True)
            else:
                df["date"] = pd.to_datetime(ts, unit="s", utc=True)
        else:
            raise ValueError("Freqtrade JSON 缺少 'date' 或 'timestamp'。")
    else:
        df = pd.read_csv(path)

    lower = {c.lower(): c for c in df.columns}
    need = ["open", "high", "low", "close", "volume"]
    miss = [c for c in need if c not in lower]
    if miss:
        raise ValueError(f"缺少必须列: {miss}；现有列: {list(df.columns)}")

    # 确保存在 date 列（UTC）
    if "date" not in df.columns:
        if "timestamp" in lower:
            c = lower["timestamp"]
            ts = pd.to_numeric(df[c], errors="coerce")
            if ts.dropna().iloc[0] > 1e13:
                df["date"] = pd.to_datetime(ts, unit="ns", utc=True, errors="coerce")
            elif ts.dropna().iloc[0] > 1e10:
                df["date"] = pd.to_datetime(ts, unit="ms", utc=True, errors="coerce")
            else:
                df["date"] = pd.to_datetime(ts, unit="s", utc=True, errors="coerce")
        else:
            logging.warning("未找到 date/timestamp，将生成虚拟日期索引。")
            df["date"] = pd.date_range("2000-01-01", periods=len(df), freq="D", tz="UTC")

    df = df.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
    return df


def save_checkpoint(model: nn.Module, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(model.state_dict(), path)
    logging.info(f"Checkpoint saved to: {path}")


def load_checkpoint(model: nn.Module, path: str, device: torch.device):
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    logging.info(f"Loaded checkpoint from: {path}")


# ---------------------- Train Core ----------------------
def train_one(df_path: str,
              out_ckpt: str,
              win: int = 128,
              batch_size: int = 64,
              epochs: int = 15,
              lr: float = 1e-4,            # 默认更稳：1e-4
              weight_decay: float = 1e-4,
              num_workers: int = 4,
              amp: bool = False,
              resume: str | None = None):

    t0 = time.time()
    logging.info(f"Loading data from: {df_path}")
    df = read_any_df(df_path)
    logging.info(f"Data loaded. shape={df.shape} head={df.head(2).to_dict(orient='records')}")

    logging.info(f"Building market tensors (win={win}) ...")
    X, Y = make_market_tensor(df, win=win)  # X: (N,C,H,W), Y: (N,)
    if len(X) == 0:
        raise RuntimeError("make_market_tensor 返回空数据，请检查窗口长度与数据量。")

    # ★★★ 数值稳定关键：强制 float32 ★★★
    X = X.astype(np.float32)

    pos = int(Y.sum())
    neg = int((Y == 0).sum())
    logging.info(f"Tensors ready. X={X.shape} Y={Y.shape} (pos={pos} neg={neg})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device = {device}")

    # 允许 TF32（新 API 可按需迁移；此设置对 Ampere+ 有加速）
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

    ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                       torch.tensor(Y).long())
    dl = DataLoader(ds,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    pin_memory=True,
                    drop_last=True)

    model = MarketDiT(in_ch=X.shape[1]).to(device)
    opt   = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit  = nn.CrossEntropyLoss()
    scaler = GradScaler('cuda', enabled=amp)

    # 断点续训
    if resume and os.path.exists(resume):
        load_checkpoint(model, resume, device)

    logging.info(f"Start training: epochs={epochs} bs={batch_size} lr={lr} amp={amp}")
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss, n = 0.0, 0
        for xb, yb in dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            if amp:
                with autocast('cuda'):
                    logits = model(xb)
                    loss = crit(logits, yb)
                # 非有限 loss 直接报错并打印统计
                if not torch.isfinite(loss):
                    stats = dict(
                        xb_mean=float(xb.mean().item()),
                        xb_std=float(xb.std().item()),
                        logits_mean=float(logits.mean().item()),
                        logits_std=float(logits.std().item()),
                    )
                    raise RuntimeError(f"Non-finite loss under AMP. Stats: {stats}")

                scaler.scale(loss).backward()
                # ★ unscale 后裁剪，抑制梯度爆炸
                scaler.unscale_(opt)
                nn_utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(opt)
                scaler.update()
            else:
                logits = model(xb)
                loss = crit(logits, yb)
                if not torch.isfinite(loss):
                    stats = dict(
                        xb_mean=float(xb.mean().item()),
                        xb_std=float(xb.std().item()),
                        logits_mean=float(logits.mean().item()),
                        logits_std=float(logits.std().item()),
                    )
                    raise RuntimeError(f"Non-finite loss in fp32. Stats: {stats}")
                loss.backward()
                nn_utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()

            bs = xb.size(0)
            epoch_loss += loss.item() * bs
            n += bs

        logging.info(f"Epoch {epoch:03d}/{epochs} | loss={epoch_loss / max(1, n):.6f}")

    save_checkpoint(model, out_ckpt)
    logging.info(f"Done. elapsed={time.time() - t0:.2f}s")


# ---------------------- Main ----------------------
def main():
    parser = argparse.ArgumentParser("Train Market DiT")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to CSV/Parquet/Feather/Freqtrade-JSON")
    parser.add_argument("--win", type=int, default=128, help="历史窗口长度")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)          # 更稳默认
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--ckpt", type=str, default="user_data/models/market_dit.pt")
    parser.add_argument("--log", type=str, default=f"logs/train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    parser.add_argument("--amp", action="store_true", help="启用混合精度")
    parser.add_argument("--seed", type=int, default=42, help="随机种子 (None 则不固定)")
    parser.add_argument("--resume", type=str, default=None, help="从已保存权重继续训练")
    args = parser.parse_args()

    setup_logger(log_path=args.log)
    set_seed(args.seed)

    logging.info("===== Train Market DiT (start) =====")
    logging.info(f"Args: {vars(args)}")

    try:
        train_one(df_path=args.data,
                  out_ckpt=args.ckpt,
                  win=args.win,
                  batch_size=args.batch_size,
                  epochs=args.epochs,
                  lr=args.lr,
                  weight_decay=args.wd,
                  num_workers=args.workers,
                  amp=args.amp,
                  resume=args.resume)
    except Exception:
        logging.error("Training failed:")
        logging.error(traceback.format_exc())
        sys.exit(1)

    logging.info("===== Train Market DiT (end) =====")


if __name__ == "__main__":
    main()
