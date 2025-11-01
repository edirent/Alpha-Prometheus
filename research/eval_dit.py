# research/eval_prob.py（示意）
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score, log_loss
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from research.Simple_DiT import MarketDiT
from research.prepare_dataset import make_market_tensor

def eval_model(df, ckpt_path, win=128, batch_size=256, device='cuda'):
    X, Y = make_market_tensor(df, win=win)   # (N,C,H,W), (N,)
    N = len(X); split = int(N*0.8)
    Xtr, Xte = X[:split], X[split:]
    Ytr, Yte = Y[:split], Y[split:]

    ds = TensorDataset(torch.tensor(Xte, dtype=torch.float32),
                       torch.tensor(Yte).long())
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    model = MarketDiT(in_ch=X.shape[1]).to(device).eval()
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    ps, ys = [], []
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(device)
            logits = model(xb)
            p = F.softmax(logits, dim=-1)[:,1].detach().cpu().numpy()
            ps.append(p); ys.append(yb.numpy())
    p = np.concatenate(ps); y = np.concatenate(ys)

    auc   = roc_auc_score(y, p)
    brier = brier_score_loss(y, p)
    acc   = accuracy_score(y, (p>0.5).astype(int))
    ll    = log_loss(y, p, labels=[0,1])

    print(f"AUC={auc:.4f}  Brier={brier:.4f}  Acc@0.5={acc:.4f}  LogLoss={ll:.4f}")
    # 也可导出 p,y 做 Reliability diagram
    return p, y
