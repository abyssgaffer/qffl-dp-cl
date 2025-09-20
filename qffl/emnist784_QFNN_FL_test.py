# qffl/mnist_QFNN_FL_test.py
import os, re, glob, argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.mixture import GaussianMixture

from common.emnist784_QFNN import Qfnn

DEVICE = torch.device('cpu')
DATA_DIR = "../data/emnist784"        # ★ 改这里
RESULT_DIR = "../result"
NAME = "emnist784_qffl_qfi"           # ★ 和训练一致
BATCH = 1024
TAU = 2.3
UNIFORM_MIX = 0.10
TOPK = 3
PROB_ENSEMBLE = False
LOGIT_T = 1.2

def load_test():
    x = torch.load(os.path.join(DATA_DIR, "test_data.pkl")).float()
    y = torch.load(os.path.join(DATA_DIR, "test_label.pkl")).long()
    assert x.ndim == 2 and y.ndim == 1 and y.shape[0] == x.shape[0]
    return x, y

def load_scaler():
    d = torch.load(os.path.join(RESULT_DIR, "data", f"{NAME}_scaler.pt"))
    mu, sigma = d["mu"].float(), d["sigma"].float().clamp_min(1e-6)
    return mu, sigma

def find_models():
    pattern = os.path.join(RESULT_DIR, "model", f"{NAME}_n*.pth")
    paths = glob.glob(pattern)
    if not paths: raise FileNotFoundError(pattern)
    paths.sort(key=lambda p: int(re.search(r"_n(\d+)\.pth$", p).group(1)))
    return paths

def safe_load_gmms():
    path = os.path.join(RESULT_DIR, "data", f"{NAME}_gmm_list")
    try:
        from torch.serialization import add_safe_globals
        add_safe_globals([GaussianMixture])
        gmms = torch.load(path)
    except Exception:
        gmms = torch.load(path, weights_only=False)
    return gmms

def infer_num_classes_from_ckpt(ckpt_path: str) -> int:
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        sd = state["model"]
    elif isinstance(state, dict) and "state_dict" in state:
        sd = state["state_dict"]
    else:
        sd = state
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    w = sd.get("softmax_linear.weight", None)
    if w is None:
        raise RuntimeError(f"无法在 {ckpt_path} 找到 softmax_linear.weight")
    return int(w.shape[0])

def forward_logits(model, x_std):
    dl = DataLoader(TensorDataset(x_std), batch_size=BATCH, shuffle=False)
    outs = []
    model.eval()
    with torch.no_grad():
        for (xb,) in dl:
            outs.append(model(xb.to(DEVICE)).cpu())
    return torch.cat(outs, 0)

def main():
    # 1) 数据 + 标准化
    x_te, y_te = load_test()
    in_dim = x_te.shape[1]             # ★ 自动推断输入维度（应为 784）
    mu, sigma = load_scaler()
    x_std = (x_te - mu) / sigma
    x_np = x_std.numpy()

    # 2) 找到客户端模型与 GMM，并从 ckpt 推断类别数
    model_paths = find_models()
    gmm_list = safe_load_gmms()
    assert len(gmm_list) == len(model_paths), "gmm_list 数量与客户端模型数不一致"

    ckpt_num_classes = infer_num_classes_from_ckpt(model_paths[0])

    # 3) 逐客户端 logits
    logits_list = []
    print(f"Infer logits from {len(model_paths)} clients ...")
    for p in tqdm(model_paths, ncols=80):
        m = Qfnn(device=DEVICE, num_classes=ckpt_num_classes, in_dim=in_dim).to(DEVICE)  # ★ 传 in_dim
        state = torch.load(p, map_location="cpu")
        try:
            m.load_state_dict(state, strict=False)
        except Exception:
            if isinstance(state, dict) and "model" in state:
                m.load_state_dict(state["model"], strict=False)
            else:
                raise
        logits_list.append(forward_logits(m, x_std))

    # 4) GMM 权重（稳定 softmax）
    scores = np.stack([g.score_samples(x_np) for g in gmm_list], axis=1)  # [N,K]
    scores = scores - scores.max(axis=1, keepdims=True)
    w = torch.from_numpy(scores).float()
    w = F.softmax(w / TAU, dim=1)

    if UNIFORM_MIX > 0:
        K = w.size(1)
        w = (1.0 - UNIFORM_MIX) * w + UNIFORM_MIX * (1.0 / K)
    if TOPK is not None and int(TOPK) > 0 and int(TOPK) < w.size(1):
        k = int(TOPK)
        vals, idxs = torch.topk(w, k=k, dim=1)
        mask = torch.zeros_like(w).scatter_(1, idxs, 1.0)
        w = w * mask
        w = w / (w.sum(dim=1, keepdim=True) + 1e-12)

    # 5) 加权集成（logit 融合/概率融合）
    N, C = logits_list[0].shape
    if PROB_ENSEMBLE:
        probs = torch.zeros(N, C)
        for k in range(len(logits_list)):
            probs += F.softmax(logits_list[k] / LOGIT_T, dim=1) * w[:, k:k+1]
        pred = probs.argmax(1)
    else:
        ens = torch.zeros(N, C)
        for k in range(len(logits_list)):
            ens += logits_list[k] * w[:, k:k+1]
        pred = ens.argmax(1)

    # 6) 评估
    acc = accuracy_score(y_te.numpy(), pred.numpy())
    P, R, F1, _ = precision_recall_fscore_support(y_te.numpy(), pred.numpy(), average='macro', zero_division=0)
    print(f"acc:{acc:.4f}  precision:{P:.4f}  recall:{R:.4f}  f1:{F1:.4f}")
    hist = torch.bincount(pred, minlength=C).tolist()
    print(f"pred hist ({C}):", hist)
    try:
        print(classification_report(y_te.numpy(), pred.numpy(), digits=4))
    except Exception:
        pass

    # 均匀集成基线（消融）
    logits_avg = sum(logits_list) / len(logits_list)
    pred_avg = logits_avg.argmax(1)
    acc_avg = accuracy_score(y_te.numpy(), pred_avg.numpy())
    print(f"[uniform-avg baseline] acc:{acc_avg:.4f}")

if __name__ == "__main__":
    main()