# -*- coding: utf-8 -*-
"""
research/infer_paths_and_signal.py

用途：
1) 载入 MarketDiTGen（扩散去噪版）
2) 对数据后 20%（或 --test-max 指定的数量）逐 bar 采样，生成 K 条未来路径
3) 计算：
   - P(TP), P(SL), PnL 的期望 E[r]
   - 一致性 Cons = 正收益路径比例
4) 生成交易信号：满足 (P(TP)-P(SL) >= p_edge) 且 Cons >= cons_min 且 E[r] >= cost_buffer

输出：{--out}/paths_signals.csv

命令示例：
python -u -m research.infer_paths_and_signal \
  --data user_data/data/binance/BTC_USDT-15m.feather \
  --ckpt user_data/models/market_dit_gen.pt \
  --win 256 --horizon 16 --K 64 --steps 20 \
  --tp-mult 1.5 --sl-mult 1.0 \
  --p-edge 0.08 --cons-min 0.62 --cost-buffer 0.0008 \
  --test-max 5000 \
  --device cuda --out logs
"""
import os, sys, argparse
import numpy as np
import pandas as pd
import torch

ROOT = os.path.abspath(".")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from research.gen_dit import MarketDiTGen
from research.train_gen_dit import read_any_df, make_hist_features, compute_atr


@torch.no_grad()
def sample_K_paths(model: "MarketDiTGen", Xb: torch.Tensor, H: int, K: int, steps: int = 20) -> torch.Tensor:
    """
    Xb: (1,C,Hh,Ww) 单个时刻
    return: (K,H) torch.float32
    """
    Xb = Xb.repeat(K, 1, 1, 1).contiguous()
    r_hat = model.sample_paths(Xb, H=H, steps=steps)  # (K,H) 累计用对数收益增量
    return r_hat


def barrier_stats_for_paths(df: pd.DataFrame, start_idx: int, r_paths: np.ndarray,
                            tp_mult: float, sl_mult: float, ret_cap: float = 0.20):
    """
    使用 triple-barrier（真实 high/low 的保守触发）评估路径：
    - 不需要把路径还原成价格（避免 exp 溢出），直接用累计对数收益 cum
    - P_TP / P_SL：仍按真实 K 线判断（先 SL 后 TP）
    - E_r：取路径末端累计对数收益的均值（可 clip 防极值）
    - Cons：正收益路径比例
    """
    opens = df["open"].values
    highs = df["high"].values
    lows  = df["low"].values
    H = r_paths.shape[1]

    entry_idx = start_idx + 1
    if entry_idx >= len(df):
        return 0.0, 0.0, 0.0, 0.0

    entry_px = float(opens[entry_idx])

    # ATR -> 相对比例（与训练一致）
    atr = compute_atr(df, n=20)
    alpha_tp = float(tp_mult * atr[entry_idx] / max(entry_px, 1e-6))
    alpha_sl = float(sl_mult * atr[entry_idx] / max(entry_px, 1e-6))

    stop_px = entry_px * (1 - alpha_sl)
    take_px = entry_px * (1 + alpha_tp)

    P_TP = 0.0
    P_SL = 0.0
    rets = []

    for k in range(r_paths.shape[0]):
        cum = np.cumsum(r_paths[k])  # (H,)

        # 触发判断基于真实 K 线（先 SL 再 TP）
        hit = 1  # 1: none, 0: SL, 2: TP
        for h in range(H):
            j = entry_idx + h
            if j >= len(df):
                break
            if lows[j] <= stop_px:
                hit = 0
                break
            if highs[j] >= take_px:
                hit = 2
                break
        if hit == 0:
            P_SL += 1.0
        elif hit == 2:
            P_TP += 1.0

        r_end = float(cum[min(H - 1, len(df) - 1 - entry_idx)])
        if ret_cap is not None:
            r_end = float(np.clip(r_end, -ret_cap, ret_cap))
        rets.append(r_end)

    K = float(r_paths.shape[0])
    if K <= 0:
        return 0.0, 0.0, 0.0, 0.0

    P_TP /= K
    P_SL /= K
    E_r   = float(np.mean(rets))
    Cons  = float(np.mean(np.sign(rets) > 0))
    return P_TP, P_SL, E_r, Cons


def main():
    ap = argparse.ArgumentParser("Sample K future paths & form consistency signals")
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--win", type=int, default=256)
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--tp-mult", type=float, default=1.5)
    ap.add_argument("--sl-mult", type=float, default=1.0)
    ap.add_argument("--p-edge", type=float, default=0.08, help="min P(TP)-P(SL)")
    ap.add_argument("--cons-min", type=float, default=0.62, help="min positive-path ratio")
    ap.add_argument("--cost-buffer", type=float, default=0.0008, help="E[r] must > this")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="logs")

    # ★ 新增：限制测试段 bar 数量（只用最后 test-max 条作为测试集）
    ap.add_argument("--test-max", type=int, default=None, help="max number of test bars")

    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    # 读数据 & 特征
    df = read_any_df(args.data)
    X = make_hist_features(df, win=args.win)  # (N',C,Hh,Ww)
    N = len(X)

    # 默认后 20% 作为测试
    split = int(N * 0.8)

    # ★ 若指定 --test-max，则只用最后 test-max 条作为测试集
    if args.test_max is not None:
        split = max(0, N - int(args.test_max))

    # 模型
    model = MarketDiTGen(in_ch=X.shape[1], horizon=args.horizon).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()

    rows = []
    for i in range(split, N):
        df_idx = (args.win - 1) + i
        if df_idx + 1 >= len(df):
            break

        xb = torch.from_numpy(X[i:i+1]).float().to(device)
        r_paths = sample_K_paths(model, xb, H=args.horizon, K=args.K, steps=args.steps)  # (K,H)
        P_TP, P_SL, E_r, Cons = barrier_stats_for_paths(
            df, df_idx, r_paths.cpu().numpy(),
            tp_mult=args.tp_mult, sl_mult=args.sl_mult
        )
        edge = P_TP - P_SL
        go_long = (edge >= args.p_edge) and (Cons >= args.cons_min) and (E_r >= args.cost_buffer)

        rows.append(dict(
            date=str(df["date"].iloc[df_idx]),
            df_idx=int(df_idx),
            P_TP=float(P_TP), P_SL=float(P_SL), edge=float(edge),
            E_r=float(E_r), Cons=float(Cons),
            long=int(go_long)
        ))

        if (len(rows) % 500) == 0:
            print(f"... processed {len(rows)} test bars")

    out_csv = os.path.join(args.out, "paths_signals.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
