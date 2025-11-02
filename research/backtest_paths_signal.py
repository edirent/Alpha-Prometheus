# -*- coding: utf-8 -*-
"""
research/backtest_paths_signal.py

用途：
- 读取 OHLCV 数据与由 infer_paths_and_signal.py 生成的 paths_signals.csv
- 对 long==1 的位置：按照“下一根开盘”入场，ATR 自适应 SL/TP，支持持仓上限(hold)、冷却、手续费/滑点
- 输出交易明细 trades.csv 与权益曲线 equity.csv，并打印绩效指标

命令示例：
python -u -m research.backtest_paths_signal \
  --data user_data/data/binance/BTC_USDT-15m.feather \
  --signals logs/paths_signals.csv \
  --hold 16 --atr-n 20 --atr-sl-mult 1.0 --atr-tp-mult 1.5 \
  --cost 0.0006 --slip 0.0002 --cooldown 8 \
  --bars-per-day 96 --outdir logs
"""
import os, sys, argparse
import numpy as np
import pandas as pd

ROOT = os.path.abspath(".")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# 复用前面训练里同款 ATR（指数平滑）
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

def read_any_df(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ext = os.path.splitext(path)[-1].lower()
    if ext == ".feather":
        df = pd.read_feather(path)
    elif ext == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    # 规范时间列
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    elif "timestamp" in df.columns:
        ts = pd.to_numeric(df["timestamp"], errors="coerce")
        unit = "ns" if ts.dropna().iloc[0] > 1e13 else ("ms" if ts.dropna().iloc[0] > 1e10 else "s")
        df["date"] = pd.to_datetime(ts, unit=unit, utc=True, errors="coerce")
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

def sharpe_from_equity(equity_df: pd.DataFrame, bars_per_day: int = 96) -> float:
    if equity_df.empty: return 0.0
    eq = equity_df["equity"].values
    r = np.diff(np.log(eq), prepend=np.log(eq[0]))
    ann = r.mean() * (bars_per_day * 252)
    vol = r.std(ddof=1) * np.sqrt(bars_per_day * 252)
    return float(ann / (vol + 1e-12))

def max_drawdown(equity_df: pd.DataFrame) -> float:
    if equity_df.empty: return 0.0
    eq = equity_df["equity"].values
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / (peak + 1e-12)
    return float(dd.min())

def backtest_long_from_signals(
    df: pd.DataFrame,
    signals_df: pd.DataFrame,
    *,
    atr_n: int = 20,
    atr_sl_mult: float = 1.0,
    atr_tp_mult: float = 1.5,
    hold: int = 16,
    cost: float = 0.0006,
    slip: float = 0.0002,
    cooldown: int = 0
):
    opens = df["open"].values
    highs = df["high"].values
    lows  = df["low"].values
    dates = df["date"].values
    atr = compute_atr(df, n=atr_n)

    # 只保留 long==1 的信号，并去重/排序
    sig = (signals_df.copy()
           .sort_values("df_idx")
           .drop_duplicates("df_idx"))
    sig = sig[sig["long"] == 1]
    sig_idx = sig["df_idx"].astype(int).values

    trades = []
    equity = []
    eq = 1.0
    next_allowed_idx = 0

    for k in sig_idx:
        if k < next_allowed_idx:
            continue
        entry_idx = k + 1
        if entry_idx >= len(df):
            break

        entry_px = float(opens[entry_idx]) * (1 + slip)
        entry_dt = dates[entry_idx]

        atr_now  = float(atr[entry_idx])
        stop_px  = entry_px - atr_sl_mult * atr_now
        take_px  = entry_px + atr_tp_mult * atr_now

        # 入场成本
        eq *= (1 - cost)

        exit_idx = None
        exit_px  = None
        reason   = "timeout"

        for h in range(hold):
            j = entry_idx + h
            if j >= len(df):
                exit_idx = len(df) - 1
                exit_px  = float(opens[exit_idx]) * (1 - slip)
                reason   = "eod"
                break
            # 先 SL 后 TP （保守）
            if lows[j] <= stop_px:
                exit_idx = j
                exit_px  = max(stop_px, float(opens[j]) * (1 - slip))
                reason   = "SL"
                break
            if highs[j] >= take_px:
                exit_idx = j
                exit_px  = min(take_px, float(opens[j]) * (1 + slip))
                reason   = "TP"
                break

        if exit_idx is None:
            exit_idx = min(entry_idx + hold, len(df) - 1)
            exit_px  = float(opens[exit_idx]) * (1 - slip)
            reason   = "timeout"

        # 出场成本
        eq *= (1 - cost)

        ret = (exit_px - entry_px) / entry_px
        eq *= (1 + ret)

        trades.append(dict(
            entry_date=str(entry_dt), entry_idx=int(entry_idx), entry=float(entry_px),
            exit_date=str(dates[exit_idx]), exit_idx=int(exit_idx), exit=float(exit_px),
            ret=float(ret), reason=reason
        ))
        equity.append((dates[exit_idx], eq))

        # 冷却：阻止过于密集交易
        next_allowed_idx = exit_idx + cooldown

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity, columns=["date","equity"]).drop_duplicates("date")
    return trades_df, equity_df

def main():
    ap = argparse.ArgumentParser("Backtest from path-based signals")
    ap.add_argument("--data", required=True)
    ap.add_argument("--signals", required=True)
    ap.add_argument("--hold", type=int, default=16)
    ap.add_argument("--atr-n", type=int, default=20)
    ap.add_argument("--atr-sl-mult", type=float, default=1.0)
    ap.add_argument("--atr-tp-mult", type=float, default=1.5)
    ap.add_argument("--cost", type=float, default=0.0006)
    ap.add_argument("--slip", type=float, default=0.0002)
    ap.add_argument("--cooldown", type=int, default=8)
    ap.add_argument("--bars-per-day", type=int, default=96)
    ap.add_argument("--outdir", default="logs")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = read_any_df(args.data)
    sig = pd.read_csv(args.signals)

    trades_df, equity_df = backtest_long_from_signals(
        df, sig,
        atr_n=args.atr_n,
        atr_sl_mult=args.atr_sl_mult,
        atr_tp_mult=args.atr_tp_mult,
        hold=args.hold,
        cost=args.cost,
        slip=args.slip,
        cooldown=args.cooldown
    )

    sharpe = sharpe_from_equity(equity_df, bars_per_day=args.bars_per_day)
    mdd    = max_drawdown(equity_df)
    total_ret = float(equity_df["equity"].iloc[-1] - 1.0) if not equity_df.empty else 0.0
    n_trades = len(trades_df)
    win_rate = float((trades_df["ret"] > 0).mean()) if n_trades else 0.0

    print("\n===== BACKTEST (PATH SIGNALS) =====")
    print(f"trades={n_trades}  win_rate={win_rate:.3f}")
    print(f"total_return={total_ret:.3f}  Sharpe={sharpe:.2f}  MDD={mdd:.3f}")

    trades_path = os.path.join(args.outdir, "trades.csv")
    equity_path = os.path.join(args.outdir, "equity.csv")
    trades_df.to_csv(trades_path, index=False)
    equity_df.to_csv(equity_path, index=False)
    print(f"Saved: {trades_path}, {equity_path}")

if __name__ == "__main__":
    main()
