# research/prepare_dataset.py 关键片段（示例与你之前的函数一致）
import numpy as np
import pandas as pd

def make_market_tensor(df, win=128):
    o,h,l,c,v = [df[x].values for x in ['open','high','low','close','volume']]
    ret  = np.diff(np.log(c), prepend=np.log(c[0]))
    vola = pd.Series(ret).rolling(24).std().fillna(0).values
    v_ma = pd.Series(v).rolling(24).mean().bfill().values   # ← 修复 FutureWarning

    feats = [o,h,l,c,v,ret,vola,v_ma]  # C ≈ 8
    X, Y = [], []

    H, W = 16, 16
    for i in range(win, len(df)-1):
        patch = np.stack([f[i-win:i] for f in feats], axis=0)  # (C, win)
        idx = np.linspace(0, patch.shape[1]-1, H*W).astype(int)
        img = patch[:, idx].reshape(len(feats), H, W).astype(np.float32)

        # ★★★ 标准化 + 去 NaN/Inf + 截断（数值稳定的关键）★★★
        mean = img.mean(axis=(1,2), keepdims=True)
        std  = img.std(axis=(1,2), keepdims=True)
        img  = (img - mean) / (std + 1e-6)
        img  = np.nan_to_num(img, nan=0.0, posinf=10.0, neginf=-10.0)
        img  = np.clip(img, -8.0, 8.0)  # 防止极端值在 AMP 下溢出

        X.append(img)

        # 监督目标：未来4根收益之和 > 0
        future = df['close'].values[i+1:i+1+4]
        y = float(np.log(future[-1] / df['close'].values[i]) > 0)
        Y.append(y)

    X = np.array(X, dtype=np.float32)   # ← 明确 float32
    Y = np.array(Y, dtype=np.float32)
    return X, Y
