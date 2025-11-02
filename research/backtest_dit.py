# -*- coding: utf-8 -*-
"""
research/backtest_dit.py  (SR Both-Side Episodes, One-Signal-Per-Interval)

Pipeline:
1) Load OHLCV & model, run logits over full dataset
2) Time split 80/20 -> validation/test
3) Probability calibration on validation (isotonic/platt/temperature/none)
4) Build SR episodes on BOTH sides:
   - Resistance R_t = rolling max(high, L)
   - Support    S_t = rolling min(low,  L)
   Episode_R: close in [R_t - band, R_t + eps]
   Episode_S: close in [S_t - eps, S_t + band]
5) In each episode, pick AT MOST ONE trade with grammar:
   - Direction: Resistance→SHORT, Support→LONG (可用 --force-long/--force-short 覆盖)
   - Hysteresis (t_hi > t_lo)、Persistence (m bars)、Slope (Δ over M bars)
   - Near-boundary filter: distance to (R or S) ≤ near_atr * ATR
   - Selection: 'first' or 'maxprob'
6) Execute: next-open entry (with slip/cost), ATR-adaptive SL/TP, timeout, cooldown
7) Report metrics + save trades/equity

Author: you :)
"""
import os, sys, json, math, argparse
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss, accuracy_score
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from scipy.special import softmax

# ---------------- Project imports ----------------
ROOT = os.path.abspath(".")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from research.Simple_DiT import MarketDiT
from research.prepare_dataset import make_market_tensor


# ---------------- IO ----------------
def read_any_df(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
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
    elif "timestamp" in df.columns:
        ts = pd.to_numeric(df["timestamp"], errors="coerce")
        v = ts.dropna().iloc[0] if len(ts.dropna()) else 0
        unit = "ns" if v > 1e13 else ("ms" if v > 1e10 else "s")
        df["date"] = pd.to_datetime(ts, unit=unit, utc=True, errors="coerce")
    else:
        df["date"] = pd.date_range("2000-01-01", periods=len(df), tz="UTC", freq="D")

    need = ["open","high","low","close","volume"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}. Columns={df.columns.tolist()}")

    df = (df.dropna(subset=["date"])
            .sort_values("date")
            .drop_duplicates("date")
            .reset_index(drop=True))
    return df


def logits_labels_indices(df: pd.DataFrame, win: int, batch_size: int,
                          device: str, model_ckpt: str, in_ch: int | None = None):
    """
    Forward the whole dataset to get logits/labels aligned to df.
    Returned idx_in_df aligns the first label to df row 'win'.
    """
    X, Y = make_market_tensor(df, win=win)  # X:(N,C,H,W), Y:(N,)
    if len(X) == 0:
        raise RuntimeError("make_market_tensor returned empty tensors")
    X = X.astype(np.float32)

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = MarketDiT(in_ch=in_ch or X.shape[1]).to(dev).eval()
    model.load_state_dict(torch.load(model_ckpt, map_location=dev))

    dl = DataLoader(TensorDataset(torch.tensor(X), torch.tensor(Y).long()),
                    batch_size=batch_size, shuffle=False)

    logits, labels = [], []
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(dev, non_blocking=True)
            lg = model(xb)
            logits.append(lg.detach().cpu().numpy())
            labels.append(yb.numpy())
    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0).astype(int)

    idx_in_df = np.arange(win, win + len(labels))
    return logits, labels, idx_in_df


# ---------------- Calibration & metrics ----------------
def summarize_probs(y: np.ndarray, p: np.ndarray, name: str = ""):
    auc = roc_auc_score(y, p)
    brier = brier_score_loss(y, p)
    acc = accuracy_score(y, (p > 0.5).astype(int))
    ll = log_loss(y, p, labels=[0, 1])
    print(f"[{name}] AUC={auc:.4f}  Brier={brier:.4f}  Acc@0.5={acc:.4f}  LogLoss={ll:.4f}")


def fit_calibrator(logits_val: np.ndarray, y_val: np.ndarray, method: str = "isotonic"):
    if method == "none":
        return lambda lg: softmax(lg, axis=1)[:, 1]

    if method == "temperature":
        lg = torch.as_tensor(logits_val, dtype=torch.float32)
        y = torch.as_tensor(y_val, dtype=torch.long)
        T = torch.nn.Parameter(torch.ones(()))
        opt = torch.optim.LBFGS([T], lr=0.1, max_iter=100, line_search_fn="strong_wolfe")
        def closure():
            opt.zero_grad()
            p = F.log_softmax(lg / torch.abs(T), dim=-1)
            loss = F.nll_loss(p, y)
            loss.backward()
            return loss
        opt.step(closure)
        T_star = float(torch.abs(T).detach().cpu())
        print(f"[calib] temperature T={T_star:.3f}")
        return lambda lg: softmax(lg / T_star, axis=1)[:, 1]

    if method == "platt":
        margin = (logits_val[:, 1] - logits_val[:, 0]).reshape(-1, 1)
        lr = LogisticRegression(max_iter=1000)
        lr.fit(margin, y_val)
        w = lr.coef_.ravel()[0]
        b = lr.intercept_.ravel()[0]
        print("[calib] platt logistic fitted")
        return lambda lg: 1.0 / (1.0 + np.exp(-(((lg[:, 1] - lg[:, 0]) * w) + b)))

    if method == "isotonic":
        p0 = softmax(logits_val, axis=1)[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p0, y_val)
        print("[calib] isotonic fitted")
        return lambda lg: iso.transform(softmax(lg, axis=1)[:, 1])

    raise ValueError(f"Unknown calibration method: {method}")


# ---------------- Technicals / Episodes ----------------
def compute_atr(df: pd.DataFrame, n: int = 20) -> np.ndarray:
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    prev_c = np.roll(c, 1); prev_c[0] = c[0]
    tr = np.maximum.reduce([h - l, np.abs(h - prev_c), np.abs(l - prev_c)])
    # EMA for ATR
    alpha = 2.0 / (n + 1.0)
    atr = np.empty_like(tr)
    atr[0] = tr[:n].mean() if len(tr) >= n else tr[0]
    for i in range(1, len(tr)):
        atr[i] = alpha * tr[i] + (1 - alpha) * atr[i-1]
    return atr


def rolling_max(x: np.ndarray, L: int) -> np.ndarray:
    from collections import deque
    out = np.empty_like(x)
    dq = deque()
    for i, v in enumerate(x):
        while dq and dq[-1][0] <= v: dq.pop()
        dq.append((v, i))
        while dq and dq[0][1] <= i - L: dq.popleft()
        out[i] = dq[0][0]
    return out


def rolling_min(x: np.ndarray, L: int) -> np.ndarray:
    from collections import deque
    out = np.empty_like(x)
    dq = deque()
    for i, v in enumerate(x):
        while dq and dq[-1][0] >= v: dq.pop()
        dq.append((v, i))
        while dq and dq[0][1] <= i - L: dq.popleft()
        out[i] = dq[0][0]
    return out


def build_sr_episodes(df: pd.DataFrame, lookback: int, band_mult_atr: float,
                      atr: np.ndarray, use_resistance: bool = True):
    """
    Single-side SR episodes.
    If use_resistance=True: close in [R - band, R + eps]
    else (support):         close in [S - eps, S + band]
    """
    close = df["close"].values
    high  = df["high"].values
    low   = df["low"].values

    R = rolling_max(high, lookback)
    S = rolling_min(low,  lookback)

    episodes = []
    in_ep = False
    s = None
    eps = 1e-12

    for i in range(len(df)):
        band = band_mult_atr * atr[i]
        if use_resistance:
            in_band = (close[i] >= (R[i] - band)) and (close[i] <= R[i] + eps)
        else:
            in_band = (close[i] <= (S[i] + band)) and (close[i] >= S[i] - eps)

        if in_band and not in_ep:
            in_ep = True
            s = i
        elif not in_band and in_ep:
            episodes.append((s, i-1))
            in_ep = False

    if in_ep:
        episodes.append((s, len(df)-1))

    return episodes, R, S


def build_sr_episodes_both(df: pd.DataFrame, lookback: int, band_mult_atr: float,
                           atr: np.ndarray):
    """
    Build BOTH resistance and support episodes, merge by time.
    Returns merged episodes (list of (s,e)), and the R,S arrays.
    """
    eps_R, R, S = build_sr_episodes(df, lookback, band_mult_atr, atr, use_resistance=True)
    eps_S, _, _ = build_sr_episodes(df, lookback, band_mult_atr, atr, use_resistance=False)

    # Tag episodes with side: +1 = resistance, -1 = support
    tagged = [(s,e, +1) for (s,e) in eps_R] + [(s,e, -1) for (s,e) in eps_S]
    tagged.sort(key=lambda x: x[0])

    # Merge overlaps but keep side preference per sub-interval:
    # 我们选择“按时间排序后一段一段处理”，每段内部仍然只打一笔。
    merged = []
    for s,e,side in tagged:
        if not merged or s > merged[-1][1] + 1:
            merged.append([s,e,{side}])
        else:
            merged[-1][1] = max(merged[-1][1], e)
            merged[-1][2].add(side)

    # 返回 episodes: [(s,e,set_of_sides)]
    return [(s,e, sides) for s,e,sides in merged], R, S


# ---------------- Backtest core (episode-aware, long/short) ----------------
def _safe_price(px, fallback):
    if px is None or (isinstance(px, float) and (np.isnan(px) or np.isinf(px))):
        return float(fallback)
    return float(px)


def backtest_episode_longshort(df: pd.DataFrame,
                               idx_tst: np.ndarray,
                               probs_tst: np.ndarray,
                               atr: np.ndarray,
                               R: np.ndarray,
                               S: np.ndarray,
                               episodes_both: list[tuple],
                               *,
                               t_hi: float,
                               t_lo: float,
                               persist: int,
                               slope_M: int,
                               slope_delta: float,
                               near_atr: float,
                               select_rule: str = "maxprob",
                               # execution
                               hold: int = 16,
                               atr_sl_mult: float = 1.0,
                               atr_tp_mult: float = 1.5,
                               cost: float = 0.0006,
                               slip: float = 0.0002,
                               cooldown: int = 0,
                               # overrides (optional): force to long/short regardless of side
                               force_long: bool = False,
                               force_short: bool = False):
    """
    Episode-aware trading with BOTH sides:
      - If episode contains resistance (side=+1) => prefer SHORT
      - If episode contains support    (side=-1) => prefer LONG
      - If both, 默认：先看 resistance->SHORT，再看 support->LONG（或根据 force_* 覆盖）
    One-trade-per-episode, with grammar filters (persist/slope/near/hysteresis).
    """
    opens = df["open"].values
    highs = df["high"].values
    lows  = df["low"].values
    close = df["close"].values
    dates = df["date"].values

    # Map df-index -> test-array-position
    mask_tst = np.zeros(len(df), dtype=bool)
    mask_tst[idx_tst] = True
    pos_of = -np.ones(len(df), dtype=int)
    pos_of[idx_tst] = np.arange(len(idx_tst))

    trades = []
    equity = []
    eq = 1.0
    next_allowed_idx = 0  # cooldown across episodes

    def collect_candidates(ks, side_tag):
        """
        side_tag: +1 for resistance->SHORT, -1 for support->LONG
        Build candidate entries satisfying grammar filters.
        """
        cands = []
        for k in ks:
            tpos = pos_of[k]
            if tpos < 0:  # not in test
                continue
            p = probs_tst[tpos]

            # Persistence
            ok_persist = True
            if persist > 1:
                start = tpos - persist + 1
                if start < 0:
                    ok_persist = False
                else:
                    ok_persist = np.all(probs_tst[start:tpos+1] >= t_hi)

            # Slope
            ok_slope = True
            if slope_M > 0:
                if tpos - slope_M < 0:
                    ok_slope = False
                else:
                    ok_slope = (probs_tst[tpos] - probs_tst[tpos - slope_M]) >= slope_delta

            # Near-boundary
            if side_tag == +1:   # resistance
                dist = abs(R[k] - close[k])
            else:                 # support
                dist = abs(S[k] - close[k])
            ok_near = (dist <= near_atr * atr[k])

            # Hysteresis enter condition: p >= t_hi
            ok_hyst = (p >= t_hi)

            if ok_persist and ok_slope and ok_near and ok_hyst:
                cands.append((k, p))
        return cands

    for (s, e, sides) in episodes_both:
        # episode timeline intersect test + cooldown
        ks_base = [k for k in range(max(s, next_allowed_idx), e+1) if mask_tst[k]]
        if not ks_base:
            continue

        # Determine preferred side order
        order = []
        if force_short and not force_long:
            order = [+1]  # resistance->short
        elif force_long and not force_short:
            order = [-1]  # support->long
        else:
            # default: if episode includes resistance, try short first; then support
            if +1 in sides:
                order.append(+1)
            if -1 in sides:
                order.append(-1)
            if not order:
                # fallback: allow both
                order = [+1, -1]

        executed = False
        for side_tag in order:
            cands = collect_candidates(ks_base, side_tag)
            if not cands:
                continue

            # pick ONE candidate in this episode
            if select_rule == "first":
                k_star, p_star = cands[0]
            else:  # maxprob
                k_star, p_star = max(cands, key=lambda x: x[1])

            # Execute trade with given direction
            if k_star + 1 >= len(df):
                next_allowed_idx = k_star + 1 + cooldown
                executed = True
                break

            entry_idx = k_star + 1
            entry_px  = _safe_price(opens[entry_idx] * (1 + slip), opens[entry_idx])
            entry_dt  = dates[entry_idx]
            atr_now   = atr[entry_idx]

            # Direction: +1 resistance->SHORT, -1 support->LONG
            # We encode dir = +1 for LONG, -1 for SHORT
            direction = -1 if side_tag == +1 else +1

            if direction == +1:
                stop_px = entry_px - atr_sl_mult * atr_now
                take_px = entry_px + atr_tp_mult * atr_now
            else:
                stop_px = entry_px + atr_sl_mult * atr_now
                take_px = entry_px - atr_tp_mult * atr_now

            # entry cost
            eq *= (1 - cost)

            exit_idx = None
            exit_px  = None
            exit_reason = "timeout"

            for h in range(hold):
                j = entry_idx + h
                if j >= len(df):
                    exit_idx = len(df) - 1
                    exit_px  = _safe_price(opens[exit_idx] * (1 - slip), opens[exit_idx])
                    exit_reason = "eod"
                    break

                if direction == +1:
                    # LONG: SL first, then TP
                    if lows[j] <= stop_px:
                        exit_idx = j
                        exit_px  = max(stop_px, opens[j] * (1 - slip))
                        exit_reason = "SL"
                        break
                    if highs[j] >= take_px:
                        exit_idx = j
                        exit_px  = min(take_px, opens[j] * (1 + slip))
                        exit_reason = "TP"
                        break
                else:
                    # SHORT: SL first (price > stop), then TP (price < take)
                    if highs[j] >= stop_px:
                        exit_idx = j
                        exit_px  = min(stop_px, opens[j] * (1 + slip))
                        exit_reason = "SL"
                        break
                    if lows[j] <= take_px:
                        exit_idx = j
                        exit_px  = max(take_px, opens[j] * (1 - slip))
                        exit_reason = "TP"
                        break

            if exit_idx is None:
                exit_idx = min(entry_idx + hold, len(df) - 1)
                # neutral exit
                exit_px  = _safe_price(opens[exit_idx] * (1 - slip), opens[exit_idx])
                exit_reason = "timeout"

            # exit cost
            eq *= (1 - cost)

            # return
            if direction == +1:
                ret = (exit_px - entry_px) / entry_px
            else:
                ret = (entry_px - exit_px) / entry_px

            eq  *= (1 + ret)

            trades.append(dict(
                entry_date=str(entry_dt), entry_idx=int(entry_idx), entry=float(entry_px),
                exit_date=str(dates[exit_idx]), exit_idx=int(exit_idx), exit=float(exit_px),
                ret=float(ret), reason=exit_reason, p=float(p_star),
                k_signal=int(k_star),
                side="SHORT" if side_tag == +1 else "LONG"
            ))
            equity.append((dates[exit_idx], eq))

            next_allowed_idx = exit_idx + cooldown
            executed = True
            break  # one trade per episode

        if not executed:
            # no trade in this episode, equity stays flat implicitly
            pass

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity, columns=["date", "equity"]).drop_duplicates("date")
    return trades_df, equity_df


# ---------------- Risk metrics ----------------
def sharpe_from_equity(equity_df: pd.DataFrame, bars_per_day: int = 96):
    if equity_df.empty:
        return 0.0
    eq = equity_df["equity"].values
    r = np.diff(np.log(eq), prepend=np.log(eq[0]))
    ann = r.mean() * (bars_per_day * 252)
    vol = r.std(ddof=1) * np.sqrt(bars_per_day * 252)
    return float(ann / (vol + 1e-12))


def max_drawdown(equity_df: pd.DataFrame):
    if equity_df.empty:
        return 0.0
    eq = equity_df["equity"].values
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / (peak + 1e-12)
    return float(dd.min())


# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser("Backtest DiT (SR both-side, episode-aware, one-signal-per-interval)")
    # data & model
    ap.add_argument("--data", required=True, help="Path to OHLCV (.feather/.parquet/.csv)")
    ap.add_argument("--ckpt", required=True, help="Trained MarketDiT checkpoint (.pt)")
    ap.add_argument("--win", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--device", default=None)
    ap.add_argument("--calib", default="isotonic", choices=["isotonic","platt","temperature","none"])
    # SR episodes
    ap.add_argument("--interval", default="sr", choices=["sr"])  # reserved for future
    ap.add_argument("--sr-lookback", type=int, default=64)
    ap.add_argument("--atr-n", type=int, default=20)
    ap.add_argument("--sr-band-mult", type=float, default=0.8, help="band = sr-band-mult * ATR")
    # signal grammar
    ap.add_argument("--t-hi", type=float, default=0.60)
    ap.add_argument("--t-lo", type=float, default=0.55)   # kept for future hysteresis reset if needed
    ap.add_argument("--persist", type=int, default=2)
    ap.add_argument("--slope-M", type=int, default=2)
    ap.add_argument("--slope-delta", type=float, default=0.02)
    ap.add_argument("--near-atr", type=float, default=0.8, help="distance to (R or S) <= near-atr * ATR")
    ap.add_argument("--select", default="maxprob", choices=["maxprob","first"])
    # execution
    ap.add_argument("--hold", type=int, default=16)
    ap.add_argument("--atr-sl-mult", type=float, default=1.0)
    ap.add_argument("--atr-tp-mult", type=float, default=1.5)
    ap.add_argument("--cost", type=float, default=0.0006)
    ap.add_argument("--slip", type=float, default=0.0002)
    ap.add_argument("--cooldown", type=int, default=8)
    ap.add_argument("--bars-per-day", type=int, default=96)
    ap.add_argument("--outdir", default="logs")
    # optional overrides
    ap.add_argument("--force-long", action="store_true", help="Force LONG regardless of SR side")
    ap.add_argument("--force-short", action="store_true", help="Force SHORT regardless of SR side")

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # 1) load data
    df = read_any_df(args.data)

    # 2) logits & labels
    logits, y, y_df_idx = logits_labels_indices(
        df, win=args.win, batch_size=args.batch_size,
        device=args.device, model_ckpt=args.ckpt
    )

    # 3) split
    N = len(logits)
    split = int(N * 0.8)
    lg_val, y_val, idx_val = logits[:split], y[:split], y_df_idx[:split]
    lg_tst, y_tst, idx_tst = logits[split:], y[split:], y_df_idx[split:]

    # 4) calibration
    calibrator = fit_calibrator(lg_val, y_val, method=args.calib)
    p_val = calibrator(lg_val)
    p_tst = calibrator(lg_tst)

    summarize_probs(y_val, p_val, f"VAL {args.calib}")
    summarize_probs(y_tst, p_tst, f"TST {args.calib}")

    # 5) episodes (both sides)
    atr = compute_atr(df, n=args.atr_n)
    episodes_both, R, S = build_sr_episodes_both(
        df, lookback=args.sr_lookback, band_mult_atr=args.sr_band_mult, atr=atr
    )

    # 6) backtest (episode-aware long/short)
    trades_df, equity_df = backtest_episode_longshort(
        df, idx_tst, p_tst, atr, R, S, episodes_both,
        t_hi=args.t_hi, t_lo=args.t_lo, persist=args.persist,
        slope_M=args.slope_M, slope_delta=args.slope_delta,
        near_atr=args.near_atr, select_rule=args.select,
        hold=args.hold, atr_sl_mult=args.atr_sl_mult, atr_tp_mult=args.atr_tp_mult,
        cost=args.cost, slip=args.slip, cooldown=args.cooldown,
        force_long=args.force_long, force_short=args.force_short
    )

    # 7) metrics
    sharpe = sharpe_from_equity(equity_df, bars_per_day=args.bars_per_day)
    mdd    = max_drawdown(equity_df)
    total_ret = float(equity_df["equity"].iloc[-1] - 1.0) if not equity_df.empty else 0.0
    n_trades = len(trades_df)
    win_rate = float((trades_df["ret"] > 0).mean()) if n_trades else 0.0

    print("\n===== BACKTEST (TEST, SR BOTH-SIDE) =====")
    print(f"trades={n_trades}  win_rate={win_rate:.3f}")
    print(f"total_return={total_ret:.3f}  Sharpe={sharpe:.2f}  MDD={mdd:.3f}")

    # 8) save
    trades_path = os.path.join(args.outdir, "trades.csv")
    equity_path = os.path.join(args.outdir, "equity.csv")
    trades_df.to_csv(trades_path, index=False)
    equity_df.to_csv(equity_path, index=False)
    print(f"Saved: {trades_path}, {equity_path}")


if __name__ == "__main__":
    main()
