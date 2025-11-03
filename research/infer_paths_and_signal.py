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

# --- 替换 / 新增在 infer_paths_and_signal.py 中 ---

def ev_from_paths_logdom(r_paths: np.ndarray,
                         alpha_tp: float, alpha_sl: float,
                         fee_roundtrip: float = 0.0) -> tuple[float, float, float, float]:
    """
    仅用模拟路径决定 TP/SL/EV/Cons（不看真实K线）。
    路径为对数收益增量，alpha_tp/sl 为对数域阈值。
    返回: P_TP, P_SL, E_r, Cons
    E_r 为扣除双边手续费/滑点后的“期望对数收益”（近似）。
    """
    K, H = r_paths.shape
    tp = sl = 0.0
    rets = []

    for k in range(K):
        cum = np.cumsum(r_paths[k])
        hit = 0  # +1 TP, -1 SL, 0 none
        e = cum[-1]
        # 先触发者为准
        for h in range(H):
            if cum[h] <= -alpha_sl:
                hit = -1; e = -alpha_sl  # 触发即按阈值收益记
                break
            if cum[h] >= +alpha_tp:
                hit = +1; e = +alpha_tp
                break
        if hit == +1: tp += 1.0
        elif hit == -1: sl += 1.0

        # 简单把双边cost近似为对数收益的常数惩罚
        e -= fee_roundtrip
        # 可选截断，避免极端路径
        e = float(np.clip(e, -0.25, 0.25))
        rets.append(e)

    if K == 0:
        return 0.0, 0.0, 0.0, 0.0
    P_TP = tp / K
    P_SL = sl / K
    E_r  = float(np.mean(rets))
    Cons = float((np.array(rets) > 0.0).mean())
    return P_TP, P_SL, E_r, Cons

def barrier_stats_for_paths_logdomain(r_paths: np.ndarray,
                                      alpha_tp: float,
                                      alpha_sl: float,
                                      ret_cap: float | None = 0.20):
    """
    仅使用“采样得到的对数收益路径”来计算：
      - P_TP:  有路径在 H 内 cum >= log(1+alpha_tp)
      - P_SL:  有路径在 H 内 cum <= log(1-alpha_sl)  (先 SL 后 TP 的保守规则在这里用min/max顺序近似)
      - E_r :  路径末端累计对数收益的均值（可选 clip）
      - Cons: 末端累计收益为正的路径比例
    注意：这里不依赖真实 high/low，避免 ETH 上 16 根内“不碰阈值”的极端零概率。
    """
    if r_paths.size == 0:
        return 0.0, 0.0, 0.0, 0.0

    # 对数域阈值
    tp_log = np.log1p(max(alpha_tp, 1e-9))        # log(1+alpha_tp)
    sl_log = np.log(max(1.0 - alpha_sl, 1e-9))    # log(1-alpha_sl) <= 0

    K, H = r_paths.shape
    cum = np.cumsum(r_paths, axis=1)              # (K,H)

    # 命中：先判 SL，再判 TP（保守）
    hit_sl = (cum <= sl_log).any(axis=1)
    # 只有在没先触 SL 的路径里才看 TP
    hit_tp = (~hit_sl) & (cum >= tp_log).any(axis=1)

    P_SL = float(hit_sl.mean())
    P_TP = float(hit_tp.mean())

    # 末端对数收益
    r_end = cum[:, -1].astype(np.float64)
    if ret_cap is not None:
        r_end = np.clip(r_end, -ret_cap, ret_cap)
    E_r  = float(r_end.mean())
    Cons = float((r_end > 0).mean())
    return P_TP, P_SL, E_r, Cons


def main():
    ap = argparse.ArgumentParser("Sample K future paths & form consistency signals")
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--win", type=int, default=256)
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--steps", type=int, default=20)

    # TP/SL 幅度（相对 ATR/price）
    ap.add_argument("--tp-mult", type=float, default=1.5)
    ap.add_argument("--sl-mult", type=float, default=1.0)

    # 采样路径缩放（放大对数收益，避免全 0）
    ap.add_argument("--path-scale", type=float, default=1.0,
                    help="multiply sampled log-returns by this factor before stats")

    # 发单门槛（基于采样分布统计）
    ap.add_argument("--p-edge", type=float, default=0.08,  help="min P(TP)-P(SL)")
    ap.add_argument("--ev-min", type=float, default=0.0002, help="min expected log-return E[r]")
    ap.add_argument("--cons-min", type=float, default=0.56, help="min positive-path ratio")
    ap.add_argument("--cost-buffer", type=float, default=0.0006, help="E[r] must exceed costs")

    # 波动/趋势门控
    ap.add_argument("--vol-pctl", type=float, default=0.85,     help="upper vol percentile gate (keep if vol>=pctl)")
    ap.add_argument("--vol-pctl-low", type=float, default=0.02, help="lower vol percentile gate (keep if vol<=pctl_low)")
    ap.add_argument("--trend-min", type=float, default=0.001,   help="min |EMA log-price slope| over recent window")

    # 测试集大小限制（只用最后 test-max 条作为测试集）
    ap.add_argument("--test-max", type=int, default=None, help="max number of test bars")

    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="logs")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    # ---------- 数据与特征 ----------
    df = read_any_df(args.data)
    X  = make_hist_features(df, win=args.win)  # (N',C,Hh,Ww)
    N  = len(X)

    # 默认 80%/20%，若指定 test-max，则只用最后 test-max 条为测试
    split = int(N * 0.8)
    if args.test_max is not None:
        split = max(0, N - int(args.test_max))

    # 预计算：ATR、简单波动与趋势指标
    atr = compute_atr(df, n=20)                                 # (len(df),)
    log_close = np.log(np.clip(df["close"].values, 1e-9, None)) # 对数价
    # 用 ATR/price 作为波动 proxy，并做分位阈值
    px = df["open"].values
    vol_proxy = atr / np.clip(px, 1e-6, None)
    vol_hi = np.quantile(vol_proxy, np.clip(args.vol_pctl, 0.0, 1.0))
    vol_lo = np.quantile(vol_proxy, np.clip(args.vol_pctl_low, 0.0, 1.0))
    # 趋势：最近 32 根对数价线性回归斜率（近似 EMA 斜率）
    trend_W = 32
    def recent_slope(i_df):
        j0 = max(0, i_df - trend_W + 1)
        seg = log_close[j0:i_df+1]
        if len(seg) < 4:
            return 0.0
        x = np.arange(len(seg), dtype=np.float64)
        x = x - x.mean()
        y = seg - seg.mean()
        denom = (x**2).sum() + 1e-12
        return float((x*y).sum() / denom)

    # ---------- 模型 ----------
    model = MarketDiTGen(in_ch=X.shape[1], horizon=args.horizon).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()

    rows = []
    for i in range(split, N):
        df_idx = (args.win - 1) + i
        if df_idx + 1 >= len(df):
            break

        # 波动/趋势门控：高波或极低波；同时趋势绝对值需超过阈值
        v = vol_proxy[df_idx]
        keep_vol = (v >= vol_hi) or (v <= vol_lo)
        slope = recent_slope(df_idx)
        keep_trend = (abs(slope) >= args.trend_min)

        # 采样 K 条路径
        xb = torch.from_numpy(X[i:i+1]).float().to(device)
        r_paths = sample_K_paths(model, xb, H=args.horizon, K=args.K, steps=args.steps).cpu().numpy()
        if args.path_scale != 1.0:
            r_paths = r_paths * float(args.path_scale)

        # 用 ATR/price 转为对数域阈值的 alpha（不再用真实 high/low 判定）
        entry_i = df_idx + 1
        entry_px = float(df["open"].iloc[entry_i])
        alpha_tp = float(args.tp_mult * atr[entry_i] / max(entry_px, 1e-6))
        alpha_sl = float(args.sl_mult * atr[entry_i] / max(entry_px, 1e-6))

        # 对数域统计：P_TP、P_SL、E_r、Cons（先 SL 再 TP 的保守逻辑在函数里）
        P_TP, P_SL, E_r, Cons = barrier_stats_for_paths_logdomain(
            r_paths=r_paths,
            alpha_tp=alpha_tp,
            alpha_sl=alpha_sl,
            ret_cap=0.20
        )
        edge = P_TP - P_SL

        # 组合门槛：概率边际 + 一致性 + 期望收益覆盖成本 + 外部门控
        pass_edge = (edge >= args.p_edge)
        pass_cons = (Cons >= args.cons_min)
        pass_ev   = (E_r  >= max(args.ev_min, args.cost_buffer))
        side = 1 if (pass_edge and pass_cons and pass_ev and keep_vol and keep_trend) else 0

        rows.append(dict(
            date=str(df["date"].iloc[df_idx]),
            df_idx=int(df_idx),
            P_TP=float(P_TP), P_SL=float(P_SL), edge=float(edge),
            E_r=float(E_r), Cons=float(Cons),
            side=int(side),
            gated_vol=int(keep_vol), gated_trend=int(keep_trend)
        ))

        # 进度提示
        if (len(rows) % 500) == 0:
            print(f"... processed {len(rows)} test bars")

    out_csv = os.path.join(args.out, "paths_signals.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")
