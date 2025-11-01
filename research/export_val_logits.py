# -*- coding: utf-8 -*-
import os, sys, numpy as np, pandas as pd, torch
from torch.utils.data import TensorDataset, DataLoader

ROOT = os.path.abspath(".")
if ROOT not in sys.path: sys.path.insert(0, ROOT)

from research.Simple_DiT import MarketDiT
from research.prepare_dataset import make_market_tensor

def read_any(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".feather": return pd.read_feather(path)
    elif ext == ".parquet": return pd.read_parquet(path)
    else: return pd.read_csv(path)

def main():
    import argparse; ap=argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--win", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--out", type=str, default="logs/val_logits.npz")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    df = read_any(args.data)
    if "date" in df: df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").drop_duplicates("date").reset_index(drop=True)

    X, Y = make_market_tensor(df, win=args.win)
    X = X.astype(np.float32)
    N = len(X); split = int(N*0.8)
    Xte, Yte = X[split:], Y[split:]
    dates = df["date"].values[args.win:args.win+len(Y)]
    date_te = dates[split:]

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = MarketDiT(in_ch=X.shape[1]).to(dev).eval()
    model.load_state_dict(torch.load(args.ckpt, map_location=dev))

    dl = DataLoader(TensorDataset(torch.tensor(Xte), torch.tensor(Yte).long()),
                    batch_size=args.batch_size, shuffle=False)

    logits_list, y_list = [], []
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(dev, non_blocking=True)
            lg = model(xb)                       # raw logits (B,2)
            logits_list.append(lg.detach().cpu().numpy())
            y_list.append(yb.numpy())
    logits = np.concatenate(logits_list, 0)
    y = np.concatenate(y_list, 0).astype(np.int64)

    np.savez(args.out, logits=logits, y=y, date=date_te.astype("datetime64[ns]"))
    print(f"Saved: {args.out}  logits={logits.shape}  y={y.shape}")

if __name__ == "__main__":
    main()
