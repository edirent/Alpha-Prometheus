# -*- coding: utf-8 -*-
"""
research/backtest_paths_signal.py

从 infer_paths_and_signal.py 导出的信号进行回测：
- 支持双边 side ∈ {-1, 0, +1}（若只给 long∈{0,1} 也自动兼容）
- ATR 自适应止损/止盈（价格域），先 SL 后 TP 的保守执行
- 进出各一次成本 cost、方向性滑点 slip
- 杠杆 leverage，且对单笔收益设置下限 leverage_floor（防止爆仓以外的数值爆炸）
- 冷却期 cooldown，避免信号过密
- 收益曲线 **从首笔建仓时刻开始绘制**，并输出 trades.csv / equity.csv / equity.png

用法示例：
python -u -m research.backtest_paths_signal \
  --data user_data/data/binance/BTC_USDT-15m.feather \
  --signals logs/paths_signals.csv \
  --hold 16 --atr-n 20 --atr-sl-mult 1.0 --atr-tp-mult 1.5 \
  --cost 0.0006 --slip 0.0002 --cooldown 8 \
  --leverage 10 --leverage-floor -0.95 \
  --bars-per-day 96 --outdir logs
"""
import os, sys, json, math, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- IO ----------------
ROOT = os.path.abspath(".")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def read_any_df(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ext = os.path.splitext(path)[-1].lower()
    if ext == ".feather":
        df = pd.read_feather(path)
    elif ext == ".parquet":
        df = pd.read_parquet(path)
    elif ext in [".csv", ".txt"]:
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(path)

    # 统一时间列
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    elif "timestamp" in df.columns:
        ts = pd.to_numeric(df["timestamp"], errors="coerce")
        v = ts.dropna().iloc[0] if len(ts.dropna()) else 0
        unit = "ns" if v > 1e13 else ("ms" if v > 1e10 else "s")
        df["date"] = pd.to_datetime(ts, unit=unit, utc=True, errors="coerce")
    else:
        df["date"] = pd.date_range("2000-01-01", periods=len(df), tz="UTC", freq="D")

    need = ["open", "high", "low", "close", "volume"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}. Columns={df.columns.tolist()}")

    df = (
        df.dropna(subset=["date"])
          .sort_values("date")
          .drop_duplicates("date")
          .reset_index(drop=True)
    )
    return df


# ---------------- Technicals ----------------
def compute_atr(df: pd.DataFrame, n: int = 20) -> np.ndarray:
    """EMA ATR."""
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    prev_c = np.roll(c, 1)
    prev_c[0] = c[0]
    tr = np.maximum.reduce([h - l, np.abs(h - prev_c), np.abs(l - prev_c)])
    alpha = 2.0 / (n + 1.0)
    atr = np.empty_like(tr)
    atr[0] = tr[:n].mean() if len(tr) >= n else tr[0]
    for i in range(1, len(tr)):
        atr[i] = alpha * tr[i] + (1 - alpha) * atr[i-1]
    return atr


# ---------------- Metrics ----------------
def sharpe_from_equity(equity_df: pd.DataFrame, bars_per_day: int = 96) -> float:
    if equity_df.empty or len(equity_df) < 3:
        return 0.0
    eq = equity_df["equity"].values
    r = np.diff(np.log(eq), prepend=np.log(eq[0]))
    ann = r.mean() * (bars_per_day * 252)
    vol = r.std(ddof=1) * np.sqrt(bars_per_day * 252)
    return float(ann / (vol + 1e-12))


def max_drawdown(equity_df: pd.DataFrame) -> float:
    if equity_df.empty:
        return 0.0
    eq = equity_df["equity"].values
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / (peak + 1e-12)
    return float(dd.min())


# ---------------- Backtest Core ----------------
def _price_with_slip(px_open: float, slip: float, side: int) -> float:
    """下根 open 成交 + 方向性滑点：多→加价，空→减价。"""
    if side > 0:   # long
        return float(px_open * (1 + slip))
    elif side < 0: # short
        return float(px_open * (1 - slip))
    return float(px_open)


def backtest_from_signals(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    hold: int = 16,
    atr_n: int = 20,
    atr_sl_mult: float = 1.0,
    atr_tp_mult: float = 1.5,
    cost: float = 0.0006,
    slip: float = 0.0002,
    leverage: float = 10.0,
    leverage_floor: float = -0.95,
    cooldown: int = 0
):
    """
    执行规则：
    - 按 signals 的 df_idx 升序遍历，遇到 side∈{-1, +1} 即在下一根 open 入场
    - 止损/止盈：以 ATR * mult 的绝对价格距离设置（long: entry-ATR*sl / entry+ATR*tp；short 反向）
    - 先判 SL 后判 TP（同 K）
    - 每次进出各扣一次成本（乘 (1 - cost)）
    - 单笔收益 ret 经杠杆放大：ret_levered = clip(leverage * ret, leverage_floor, +∞)
    - 退出后进入 cooldown 根内不再开新仓
    - 不允许同时持有多单与空单；每笔交易不重叠
    """
    opens = df["open"].values
    highs = df["high"].values
    lows  = df["low"].values
    dates = df["date"].values

    atr = compute_atr(df, n=atr_n)

    # 支持两种输入：side 或 long
    if "side" in signals.columns:
        side_series = signals["side"].astype(int).values
    elif "long" in signals.columns:
        side_series = np.where(signals["long"].astype(int).values > 0, 1, 0)
    else:
        raise ValueError("signals must contain 'side' (-1/0/1) or 'long' (0/1).")

    idxs = signals["df_idx"].astype(int).values
    order = np.argsort(idxs)
    idxs = idxs[order]
    side_series = side_series[order]

    trades = []
    equity_pts = []
    eq = 1.0

    in_trade = False
    next_allowed_idx = 0
    first_entry_time = None

    i = 0
    while i < len(idxs):
        k = idxs[i]
        side = side_series[i]

        # 跳过无信号或冷却未到
        if (side == 0) or (k < next_allowed_idx):
            i += 1
            continue

        # 下一根开仓
        entry_idx = k + 1
        if entry_idx >= len(df):
            break

        # 记录首笔建仓时间，用于收益曲线起点
        if first_entry_time is None:
            first_entry_time = dates[entry_idx]
            equity_pts.append((first_entry_time, eq))

        entry_open = opens[entry_idx]
        entry_px = _price_with_slip(entry_open, slip, side)
        entry_dt = dates[entry_idx]

        # 入场费用（乘法复利下处理为折损）
        eq *= (1 - cost)

        # ATR 绝对距离
        atr_now = atr[entry_idx]
        if side > 0:  # long
            stop_px = entry_px - atr_sl_mult * atr_now
            take_px = entry_px + atr_tp_mult * atr_now
        else:         # short
            stop_px = entry_px + atr_sl_mult * atr_now  # 价格上涨触发空单止损
            take_px = entry_px - atr_tp_mult * atr_now  # 下跌触发空单止盈

        exit_idx = None
        exit_px = None
        exit_reason = "timeout"

        for h in range(hold):
            j = entry_idx + h
            if j >= len(df):
                exit_idx = len(df) - 1
                exit_px = opens[exit_idx] * (1 - slip if side > 0 else 1 + slip)
                exit_reason = "eod"
                break

            # 先判 SL 后判 TP（同 K 保守）
            if side > 0:
                # 多：先看 low 触 stop，再看 high 触 take
                if lows[j] <= stop_px:
                    exit_idx = j
                    exit_px = max(stop_px, opens[j] * (1 - slip))
                    exit_reason = "SL"
                    break
                if highs[j] >= take_px:
                    exit_idx = j
                    exit_px = min(take_px, opens[j] * (1 + slip))
                    exit_reason = "TP"
                    break
            else:
                # 空：先看 high 触 stop，再看 low 触 take
                if highs[j] >= stop_px:
                    exit_idx = j
                    exit_px = min(stop_px, opens[j] * (1 + slip))
                    exit_reason = "SL"
                    break
                if lows[j] <= take_px:
                    exit_idx = j
                    exit_px = max(take_px, opens[j] * (1 - slip))
                    exit_reason = "TP"
                    break

        if exit_idx is None:
            exit_idx = min(entry_idx + hold, len(df) - 1)
            exit_px = opens[exit_idx] * (1 - slip if side > 0 else 1 + slip)
            exit_reason = "timeout"

        # 出场费用
        eq *= (1 - cost)

        # 原始收益（价格域）
        if side > 0:
            ret = (exit_px - entry_px) / entry_px
        else:
            ret = (entry_px - exit_px) / entry_px

        # 杠杆与下限（防数值爆炸 / 爆仓以外的错误）
        ret_levered = leverage * ret
        if ret_levered < leverage_floor:
            ret_levered = leverage_floor

        eq *= (1 + ret_levered)

        trades.append(dict(
            entry_date=str(entry_dt), entry_idx=int(entry_idx), entry=float(entry_px),
            exit_date=str(dates[exit_idx]), exit_idx=int(exit_idx), exit=float(exit_px),
            side=int(side), ret=float(ret), ret_levered=float(ret_levered),
            reason=exit_reason
        ))
        equity_pts.append((dates[exit_idx], eq))

        # 冷却
        next_allowed_idx = exit_idx + cooldown

        # 前进到下一个（保证不重叠）
        # 找到下一个 idx >= next_allowed_idx 的位置
        while i < len(idxs) and idxs[i] < next_allowed_idx:
            i += 1

    trades_df = pd.DataFrame(trades)

    # 收益曲线：只从首笔建仓时刻开始
    if len(equity_pts) == 0:
        equity_df = pd.DataFrame(columns=["date", "equity"])
    else:
        equity_df = pd.DataFrame(equity_pts, columns=["date", "equity"]).drop_duplicates("date")

    return trades_df, equity_df


# ---------------- Plotting ----------------
def plot_equity_from_first_entry(equity_df: pd.DataFrame, out_png: str):
    if equity_df.empty:
        # 也生成一张空图，避免流水线报错
        plt.figure(figsize=(9, 4.5), dpi=160)
        plt.title("Equity (no trades)")
        plt.savefig(out_png, bbox_inches="tight")
        plt.close()
        return

    # 美化但不指定颜色（让 matplotlib 自选）
    plt.figure(figsize=(11, 5), dpi=160)
    plt.plot(pd.to_datetime(equity_df["date"]), equity_df["equity"], linewidth=1.8)
    plt.title("Equity curve (from first entry)")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()


# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser("Backtest from exported path-consensus signals")
    ap.add_argument("--data", required=True, help="OHLCV file (.feather/.parquet/.csv)")
    ap.add_argument("--signals", required=True, help="CSV from infer_paths_and_signal.py")
    ap.add_argument("--hold", type=int, default=16)
    ap.add_argument("--atr-n", type=int, default=20)
    ap.add_argument("--atr-sl-mult", type=float, default=1.0)
    ap.add_argument("--atr-tp-mult", type=float, default=1.5)
    ap.add_argument("--cost", type=float, default=0.0006)
    ap.add_argument("--slip", type=float, default=0.0002)
    ap.add_argument("--cooldown", type=int, default=8)
    ap.add_argument("--leverage", type=float, default=10.0)
    ap.add_argument("--leverage-floor", type=float, default=-0.95)
    ap.add_argument("--bars-per-day", type=int, default=96, help="15m=96, 1h=24")
    ap.add_argument("--outdir", default="logs")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = read_any_df(args.data)
    sig = pd.read_csv(args.signals)

    # 防呆：确保 df_idx 在范围内
    if "df_idx" not in sig.columns:
        raise ValueError("signals CSV must contain 'df_idx' column.")
    sig = sig[(sig["df_idx"] >= 0) & (sig["df_idx"] < len(df))].copy()
    if sig.empty:
        print("No valid signals after filtering by df bounds.")
        trades_df = pd.DataFrame(columns=[
            "entry_date","entry_idx","entry","exit_date","exit_idx","exit","side","ret","ret_levered","reason"
        ])
        equity_df = pd.DataFrame(columns=["date","equity"])
    else:
        trades_df, equity_df = backtest_from_signals(
            df, sig,
            hold=args.hold, atr_n=args.atr_n,
            atr_sl_mult=args.atr_sl_mult, atr_tp_mult=args.atr_tp_mult,
            cost=args.cost, slip=args.slip,
            leverage=args.leverage, leverage_floor=args.leverage_floor,
            cooldown=args.cooldown
        )

    # 指标
    sharpe = sharpe_from_equity(equity_df, bars_per_day=args.bars_per_day)
    mdd = max_drawdown(equity_df)
    total_ret = float(equity_df["equity"].iloc[-1] - 1.0) if not equity_df.empty else 0.0
    n_trades = len(trades_df)
    win_rate = float((trades_df["ret_levered"] > 0).mean()) if n_trades else 0.0

    print("\n===== BACKTEST (FROM SIGNALS) =====")
    print(f"trades={n_trades}  win_rate={win_rate:.3f}")
    print(f"total_return={total_ret:.3f}  Sharpe={sharpe:.2f}  MDD={mdd:.3f}")

    # 导出
    trades_path = os.path.join(args.outdir, "trades.csv")
    equity_path = os.path.join(args.outdir, "equity.csv")
    png_path = os.path.join(args.outdir, "equity.png")
    trades_df.to_csv(trades_path, index=False)
    equity_df.to_csv(equity_path, index=False)
    plot_equity_from_first_entry(equity_df, png_path)
    print(f"Saved: {trades_path}, {equity_path}, {png_path}")


if __name__ == "__main__":
    main()
