print("[BOOT] train_sr.py starting ...")

import os, json, gzip, math, argparse, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import classification_report

def _die(msg): print("[FATAL]", msg); raise SystemExit(1)

def _to_datetime(ts):
    ts = int(ts)
    return pd.to_datetime(ts, unit=("ms" if ts>10_000_000_000 else "s"), utc=True)

def find_file(data_root, exchange, pair, timeframe):
    print(f"[STEP] find_file root={data_root} ex={exchange} pair={pair} tf={timeframe}")
    pair_fn = pair.replace('/','_')
    cands = []
    for base in [os.path.join(data_root, exchange), data_root]:
        for pat in [f"{pair_fn}-{timeframe}.json", f"{pair_fn}-{timeframe}.json.gz",
                    f"spot/{pair_fn}-{timeframe}.json", f"spot/{pair_fn}-{timeframe}.json.gz"]:
            p = os.path.join(base, pat)
            if os.path.isfile(p): cands.append(p)
    if not cands:
        for r,_,fs in os.walk(data_root):
            for fn in fs:
                if fn.startswith(f"{pair_fn}-{timeframe}") and (fn.endswith(".json") or fn.endswith(".json.gz")):
                    cands.append(os.path.join(r, fn))
    if not cands: _die("No data file found")
    p = sorted(cands)[0]
    print("[INFO] data file:", p)
    return p

def read_ohlcv(path):
    print("[STEP] read_ohlcv:", path)
    raw = gzip.open(path,'rt').read() if path.endswith('.gz') else open(path).read()
    js = json.loads(raw)
    if isinstance(js, dict):
        for k in ["data","ohlcv","result"]:
            if k in js: js = js[k]
    if not isinstance(js, list) or not js: _die("Bad JSON list")
    first = js[0]
    if isinstance(first,(list,tuple)) and len(first)>=6:
        arr = np.array(js, dtype=object)[:,:6]
        df = pd.DataFrame(arr, columns=["ts","open","high","low","close","volume"])
        df["date"] = [_to_datetime(x) for x in df["ts"]]; df = df.drop(columns=["ts"])
    elif isinstance(first, dict):
        rows=[]
        for d in js:
            t = d.get("date", d.get("timestamp", d.get("time", d.get("ts"))))
            if t is None: continue
            rows.append({
                "date": (pd.to_datetime(t, utc=True) if not isinstance(t,(int,float)) else _to_datetime(t)),
                "open": float(d.get("open", d.get("o", np.nan))),
                "high": float(d.get("high", d.get("h", np.nan))),
                "low":  float(d.get("low",  d.get("l", np.nan))),
                "close":float(d.get("close",d.get("c", np.nan))),
                "volume":float(d.get("volume",d.get("v", 0.0))),
            })
        df = pd.DataFrame(rows)
    else:
        _die("Unknown JSON structure")
    df = df.sort_values("date").set_index("date")[["open","high","low","close","volume"]].astype(float).dropna()
    print("[INFO] ohlcv rows:", len(df))
    return df

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

def build_zones_incremental(df, piv_low, piv_high, lookback=800, merge_k=1.2, atr_series=None, min_touches=3, min_span_k=0.5, min_retest_gap=10):
    if atr_series is None: atr_series = atr(df,14).fillna(method="bfill")
    highs = df['high'].values; lows = df['low'].values; closes = df['close'].values
    # closes 已上移到上方
    def _cluster(vals, thr):
        # 将触碰点聚为带，返回 [(center, touches, span), ...]
        if len(vals)==0: return []
        vals = np.sort(vals); groups=[[vals[0]]]
        for v in vals[1:]:
            if abs(v - np.mean(groups[-1])) <= thr: groups[-1].append(v)
            else: groups.append([v])
        zones=[]
        for g in groups:
            center = float(np.mean(g)); span = max(1e-9, float(np.max(g)-np.min(g)))
            zones.append((center, len(g), span))
        return zones
        if len(vals)==0: return []
        vals = np.sort(vals); zones=[]; cur=[vals[0]]
        for v in vals[1:]:
            if abs(v - np.mean(cur)) <= thr: cur.append(v)
            else: zones.append(np.mean(cur)); cur=[v]
        zones.append(np.mean(cur)); return zones
    def past_zones(t):
        s = max(0, t-lookback)
        atr_med = float(np.nanmedian(atr_series.iloc[s:t+1])) if t>s else float(atr_series.iloc[max(0,t)])
        thr = merge_k * atr_med
        lows_vals = closes[s:t+1][piv_low[s:t+1]]
        highs_vals= closes[s:t+1][piv_high[s:t+1]]
        # 过滤规则：触碰次数>=min_touches；带跨度>= min_span_k*thr；同带内触碰至少相隔 min_retest_gap 根
        def _filter(zs, vals):
            kept=[]
            for center,touches,span in zs:
                if touches < min_touches: continue
                if span < min_span_k*thr: continue
                # 简单重测间隔检查：统计与center接近的索引是否有足够间隔
                idxs = [i for i in range(s,t+1) if abs(closes[i]-center)<=thr]
                ok=True; last=-1e9; cnt=0
                for i0 in idxs:
                    if i0-last >= min_retest_gap: cnt+=1; last=i0
                if cnt < min_touches: ok=False
                if ok:
                    score = cnt  # 可扩展为触碰强度/成交量等
                    kept.append((center, score))
            return kept
        lows_z  = _filter(_cluster(lows_vals,thr),  lows_vals)
        highs_z = _filter(_cluster(highs_vals,thr), highs_vals)
        return lows_z, highs_z, thr
    return past_zones

def build_samples(df, seq_len=128, horizon=1, left=3, right=3, lookback=800, merge_k=1.2, eps_k=0.6, min_touches=3, min_span_k=0.5, min_retest_gap=10, time_nms=8):
    print("[STEP] build_samples")
    df = df.copy()
    df["ret"] = df["close"].pct_change()
    df["hl_spread"] = (df["high"]-df["low"])/df["close"]
    df["oc_spread"] = (df["open"]-df["close"])/df["close"]
    df["ema12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["sma20"] = df["close"].rolling(20).mean()
    df["atr14"] = atr(df,14)
    df["vol_chg"] = df["volume"].pct_change()
    df = df.dropna()
    feats_cols = ["ret","hl_spread","oc_spread","ema12","sma20","atr14","vol_chg","close"]
    piv_low, piv_high = find_pivots(df, left=left, right=right)
    zones_fn = build_zones_incremental(df, piv_low, piv_high, lookback=lookback, merge_k=merge_k, atr_series=df['atr14'], min_touches=min_touches, min_span_k=min_span_k, min_retest_gap=min_retest_gap)
    X,y,ts=[],[],[]
    vals = df[feats_cols].values; closes=df["close"].values; atrs=df["atr14"].values; dates=df.index
    last_fire = -10**9
    for t in range(seq_len-1, len(df)-horizon):
        lows, highs, thr = zones_fn(t)
        # 简单时间NMS：如果距离上次发射少于 time_nms 根，则跳过
        if t - last_fire < time_nms: 
            pass
        if len(lows)==0 and len(highs)==0: continue
        target = closes[t+1]
        atr_med = np.nanmedian(atrs[max(0,t-200):t+1])
        eps = (eps_k * (atr_med if not np.isnan(atr_med) and atr_med>0 else thr))
        lab = 0
        if any(abs(target - z) <= eps for z,_ in lows): lab = 1
        if any(abs(target - z) <= eps for z,_ in highs): lab = 2
        if lab!=0: last_fire = t
        X.append(vals[t-seq_len+1:t+1]); y.append(lab); ts.append(dates[t+1])
    X = np.asarray(X, np.float32); y = np.asarray(y, np.int64); ts = pd.to_datetime(ts)
    print(f"[INFO] samples={X.shape}, label_dist={np.bincount(y, minlength=3)}")
    return X,y,ts,feats_cols

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

def main():
    print("[STEP] parse args")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="user_data/data/binance")
    ap.add_argument("--exchange", default="binance")
    ap.add_argument("--pair", default="BTC/USDT")
    ap.add_argument("--timeframe", default="15m")
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--lookback", type=int, default=800)
    ap.add_argument("--left", type=int, default=3)
    ap.add_argument("--right", type=int, default=3)
    ap.add_argument("--merge-k", type=float, default=1.5)
    ap.add_argument("--eps-k", type=float, default=0.8)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--ffn", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--save-dir", default="models")
    args = ap.parse_args()
    print("[INFO] args:", args)

    os.makedirs(args.save_dir, exist_ok=True)
    path = find_file(args.data_root, args.exchange, args.pair, args.timeframe)
    df = read_ohlcv(path)

    X,y,ts,feats = build_samples(df, seq_len=args.seq_len, left=args.left, right=args.right,
                                 lookback=args.lookback, merge_k=args.merge_k, eps_k=args.eps_k)
    n=len(X); cut=int(n*0.8)
    Xtr,ytr = X[:cut], y[:cut]; Xte,yte = X[cut:], y[cut:]
    print(f"[INFO] split: train={len(Xtr)} test={len(Xte)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SRTransformer(input_dim=X.shape[-1], d_model=args.d_model, nhead=args.nhead,
                          layers=args.layers, ffn=args.ffn, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(weight=torch.tensor([1.0,2.0,2.0], device=device))

    def batches(A,B,bs):
        for i in range(0,len(A),bs): yield A[i:i+bs], B[i:i+bs]

    print("[STEP] training ...")
    for ep in range(1, args.epochs+1):
        model.train(); tot=0; cnt=0
        for xb,yb in batches(Xtr,ytr,args.batch_size):
            xb = torch.from_numpy(xb).to(device); yb=torch.from_numpy(yb).to(device)
            with torch.autocast(device_type=("cuda" if device=="cuda" else "cpu"),
                                dtype=torch.bfloat16, enabled=(device=="cuda")):
                loss = crit(model(xb), yb)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            tot += float(loss.item())*len(xb); cnt += len(xb)
        print(f"[EPOCH {ep:03d}] train_loss={tot/max(1,cnt):.5f}")

    print("[STEP] evaluating ...")
    model.eval(); preds=[]
    with torch.no_grad():
        for i in range(0,len(Xte),1024):
            xb = torch.from_numpy(Xte[i:i+1024]).to(device)
            preds.append(model(xb).argmax(-1).cpu().numpy())
    yhat = np.concatenate(preds,0)
    print("\n======== Test Report ========")
    print(classification_report(yte, yhat, digits=3, labels=[1,2,0], target_names=["support(1)","resistance(2)","none(0)"]))

    torch.save(model.state_dict(), os.path.join(args.save_dir, "sr_transformer.pth"))
    with open(os.path.join(args.save_dir, "sr_meta.json"), "w") as f:
        json.dump({"task":"sr-detection","seq_len":args.seq_len,"timeframe":args.timeframe,"pair":args.pair,"features":feats}, f, ensure_ascii=False, indent=2)
    print("[OK] saved: models/sr_transformer.pth, models/sr_meta.json")

if __name__ == "__main__":
    main()
