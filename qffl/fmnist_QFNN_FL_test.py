# qffl/fmnist_QFNN_FL_test.py
import os, re, glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.mixture import GaussianMixture

from common.mni_QFNN import Qfnn

DEVICE = torch.device('cpu')
DATA_DIR = "../data/fmnist"
RESULT_DIR = "../result"
NAME = "fmnist_qffl_gas_q4_star"
BATCH = 1024
TAU = 2.3   # 比 1.0 更平滑，避免门控一边倒

def load_test():
    x = torch.load(os.path.join(DATA_DIR, "test_data.pkl")).float()
    y = torch.load(os.path.join(DATA_DIR, "test_label.pkl")).long()
    assert x.ndim == 2 and x.shape[1] == 10 and y.ndim == 1 and y.shape[0] == x.shape[0]
    return x, y

def load_scaler():
    path = os.path.join(RESULT_DIR, "data", f"{NAME}_scaler.pt")
    d = torch.load(path)
    mu, sigma = d["mu"].float(), d["sigma"].float().clamp_min(1e-6)
    return mu, sigma

def find_models():
    pattern = os.path.join(RESULT_DIR, "model", f"{NAME}_n*.pth")
    paths = glob.glob(pattern)
    paths.sort(key=lambda p: int(re.search(r"_n(\d+)\.pth$", p).group(1)))
    if not paths: raise FileNotFoundError(pattern)
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

def forward_logits(model, x_std):
    # 按批推理（主体逻辑不变）
    dl = DataLoader(TensorDataset(x_std), batch_size=BATCH, shuffle=False)
    outs = []
    model.eval()
    with torch.no_grad():
        for (xb,) in dl:
            xb = xb.to(DEVICE)
            outs.append(model(xb).cpu())
    return torch.cat(outs, 0)

def main():
    # 1) 数据 + 标准化
    x_te, y_te = load_test()
    mu, sigma = load_scaler()
    x_std = (x_te - mu) / sigma
    x_np = x_std.cpu().numpy()

    # 2) 客户端模型与 GMM
    model_paths = find_models()
    gmm_list = safe_load_gmms()
    assert len(gmm_list) == len(model_paths)

    # 3) 逐客户端 logits
    logits_list = []
    for p in tqdm(model_paths, desc="Infer logits", ncols=80):
        model = Qfnn(DEVICE).to(DEVICE)
        state = torch.load(p, map_location="cpu")
        try:
            model.load_state_dict(state, strict=False)
        except Exception:
            if isinstance(state, dict) and "model" in state:
                model.load_state_dict(state["model"], strict=False)
            else:
                raise
        logits_list.append(forward_logits(model, x_std))

    # 4) GMM 权重（稳定 softmax）
    scores = np.stack([g.score_samples(x_np) for g in gmm_list], axis=1)  # [N,K]
    scores = scores - scores.max(axis=1, keepdims=True)
    w = torch.from_numpy(scores).float()
    w = F.softmax(w / TAU, dim=1)  # [N,K]

    # 5) 样本级加权集成（主体逻辑不变）
    N, C = logits_list[0].shape
    logits_ens = torch.zeros(N, C)
    for k in range(len(logits_list)):
        logits_ens += logits_list[k] * w[:, k:k+1]
    pred = logits_ens.argmax(1)

    # 6) 评估与诊断
    acc = accuracy_score(y_te.numpy(), pred.numpy())
    p, r, f1, _ = precision_recall_fscore_support(y_te.numpy(), pred.numpy(), average='macro', zero_division=0)
    print(f"acc:{acc:.4f}  precision:{p:.4f}  recall:{r:.4f}  f1:{f1:.4f}")
    hist = torch.bincount(pred, minlength=10).tolist()
    print("pred hist:", hist)
    ent = (-(w * (w + 1e-12).log()).sum(dim=1)).mean().item()
    print(f"avg weight entropy: {ent:.3f}")
    try:
        print(classification_report(y_te.numpy(), pred.numpy(), digits=4))
    except Exception:
        pass

if __name__ == "__main__":
    main()