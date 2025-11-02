# -*- coding: utf-8 -*-
import numpy as np, torch, torch.nn.functional as F
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss, accuracy_score

def probs_from_logits(logits, T=1.0):
    logits = torch.as_tensor(logits, dtype=torch.float32)
    p = F.softmax(logits / T, dim=-1)[:, 1].detach().cpu().numpy()
    return p

def fit_temperature_T(logits, y, max_iter=100):
    """用 LBFGS 最小化 NLL 拟合温度 T>0."""
    y = torch.as_tensor(y, dtype=torch.long)
    lg = torch.as_tensor(logits, dtype=torch.float32)

    T = torch.nn.Parameter(torch.ones(()))  # 初值 1.0
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        p = F.log_softmax(lg / torch.abs(T), dim=-1)  # 保证正温度
        loss = F.nll_loss(p, y)
        loss.backward()
        return loss

    opt.step(closure)
    T_star = float(torch.abs(T).detach().cpu())
    return T_star

def summarize(y, p, name=""):
    auc = roc_auc_score(y, p)
    brier = brier_score_loss(y, p)
    acc = accuracy_score(y, (p>0.5).astype(int))
    ll = log_loss(y, p, labels=[0,1])
    print(f"[{name}] AUC={auc:.4f}  Brier={brier:.4f}  Acc@0.5={acc:.4f}  LogLoss={ll:.4f}")
    return dict(auc=auc, brier=brier, acc=acc, logloss=ll)

def grid_threshold_search(p, y, cost=0.0006, metric="pnl"):
    """
    简化阈值搜索：
    - metric='pnl'：多头当期方向收益 - 成本。y∈{0,1}映射 s=2y-1 ∈{-1,+1}。
    - 你也可以用 F1/MCC 等。
    """
    best_t, best = 0.5, -1e9
    ts = np.linspace(0.3, 0.7, 81)
    s = 2*y - 1
    for t in ts:
        long_ = (p > t).astype(int)
        score = (long_ * s - long_ * cost).mean()
        if score > best:
            best, best_t = score, t
    return best_t, best

def main():
    import argparse, os
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="logs/val_logits.npz")
    ap.add_argument("--cost", type=float, default=0.0006)
    ap.add_argument("--metric", default="pnl", choices=["pnl","f1"])
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    logits, y = z["logits"], z["y"].astype(int)

    # 1) 校准前
    p0 = probs_from_logits(logits, T=1.0)
    summarize(y, p0, "uncalibrated")

    # 2) 温度缩放
    T_star = fit_temperature_T(logits, y)
    pT = probs_from_logits(logits, T=T_star)
    summarize(y, pT, f"calibrated(T={T_star:.3f})")

    # 3) 阈值搜索（用校准后的概率）
    t_star, score = grid_threshold_search(pT, y, cost=args.cost, metric=args.metric)
    print(f"[threshold] best t*={t_star:.3f}  {args.metric}={score:.6f}")

    # 保存结果供策略使用
    os.makedirs("user_data/calib", exist_ok=True)
    np.savez("user_data/calib/temperature_threshold.npz", T=T_star, t=t_star)
    print("Saved: user_data/calib/temperature_threshold.npz")

if __name__ == "__main__":
    main()
