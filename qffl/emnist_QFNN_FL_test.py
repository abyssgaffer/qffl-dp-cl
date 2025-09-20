# qffl/emnist_QFNN_FL_test.py
import os, re, glob, argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.mixture import GaussianMixture

# 如你是放在 common/mni_QFNN.py，请改成 from common.mni_QFNN import Qfnn
from common.emnist_QFNN import Qfnn


# ----------------------------- 工具函数 -----------------------------
def load_test(data_dir: str):
    x = torch.load(os.path.join(data_dir, "test_data.pkl")).float()
    y = torch.load(os.path.join(data_dir, "test_label.pkl")).long()
    assert x.ndim == 2 and x.shape[1] == 10 and y.ndim == 1 and y.shape[0] == x.shape[0], \
        f"测试数据维度不匹配：x{tuple(x.shape)}, y{tuple(y.shape)}"
    return x, y

def load_scaler(result_dir: str, name: str):
    path = os.path.join(result_dir, "data", f"{name}_scaler.pt")
    d = torch.load(path)
    mu, sigma = d["mu"].float(), d["sigma"].float().clamp_min(1e-6)
    return mu, sigma

def find_models(result_dir: str, name: str):
    pattern = os.path.join(result_dir, "model", f"{name}_n*.pth")
    paths = glob.glob(pattern)
    if not paths:
        raise FileNotFoundError(f"未找到客户端模型：{pattern}")
    paths.sort(key=lambda p: int(re.search(r"_n(\d+)\.pth$", p).group(1)))
    return paths

def safe_load_gmms(result_dir: str, name: str):
    path = os.path.join(result_dir, "data", f"{name}_gmm_list")
    try:
        from torch.serialization import add_safe_globals
        add_safe_globals([GaussianMixture])
        gmms = torch.load(path)
    except Exception:
        gmms = torch.load(path, weights_only=False)
    if not isinstance(gmms, (list, tuple)) or not isinstance(gmms[0], GaussianMixture):
        raise TypeError("gmm_list 类型不正确（应为 GaussianMixture 的列表）")
    return gmms

def maybe_load_priors(result_dir: str, name: str, enabled: bool):
    if not enabled:
        return None
    path = os.path.join(result_dir, "data", f"{name}_data_weights")
    if not os.path.exists(path):
        return None
    w = torch.load(path)
    w = np.asarray(w, dtype=np.float64)
    s = w.sum()
    if s <= 0:
        return None
    return w / s

def fwd_logits(model: Qfnn, x_std: torch.Tensor, device: torch.device, batch: int):
    dl = DataLoader(TensorDataset(x_std), batch_size=batch, shuffle=False)
    outs = []
    model.eval()
    with torch.no_grad():
        for (xb,) in dl:
            outs.append(model(xb.to(device)).cpu())
    return torch.cat(outs, 0)

def infer_num_classes_from_ckpt(ckpt_path: str) -> int:
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        sd = state["model"]
    elif isinstance(state, dict) and "state_dict" in state:
        sd = state["state_dict"]
    else:
        sd = state
    # 去掉 'module.' 前缀
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    w = sd.get("softmax_linear.weight", None)
    if w is None:
        raise RuntimeError(f"无法在 {ckpt_path} 找到 softmax_linear.weight")
    return int(w.shape[0])

def assert_data_matches_num_classes(y_tensor: torch.Tensor, num_classes: int):
    k_data = int(y_tensor.max().item() + 1)
    if k_data != num_classes:
        raise RuntimeError(
            f"数据/模型类别数不一致: checkpoint={num_classes}, data={k_data}。\n"
            f"请确保一致：要么重训 {k_data} 类模型，要么把 DATA_DIR 指向 {num_classes} 类的数据目录。"
        )


# ----------------------------- 主程序 -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data_dir", default="../data/emnist")      # 你的 EMNIST（balanced/letters）10D 特征目录
    parser.add_argument("--result_dir", default="../result")
    parser.add_argument("--name", default="emnist47_qffl_qfi")       # 和训练脚本一致！
    parser.add_argument("--batch", type=int, default=1024)

    # QFI 门控与集成的可调参数（不改架构）
    parser.add_argument("--tau", type=float, default=2.3)            # 门控温度
    parser.add_argument("--uniform_mix", type=float, default=0.10)   # 与均匀分布混合
    parser.add_argument("--topk", type=int, default=3)               # Top-k 门控（<=0 表示关闭）
    parser.add_argument("--eps_floor", type=float, default=0.00)     # 每专家最小权重下限
    parser.add_argument("--prob_ensemble", action="store_true", default=False)  # 概率空间融合
    parser.add_argument("--logit_t", type=float, default=1.2)        # 概率融合时的温度
    parser.add_argument("--use_prior", action="store_true", default=False)      # 是否叠加样本量先验
    parser.add_argument("--prior_strength", type=float, default=0.5)            # 先验强度
    args = parser.parse_args()

    device = torch.device(args.device)

    # 1) 加载数据与标准化
    x_te, y_te = load_test(args.data_dir)
    mu, sigma = load_scaler(args.result_dir, args.name)
    x_std = (x_te - mu) / sigma
    x_np = x_std.numpy()

    # 2) 找到模型与 GMM；自动推断 checkpoint 类别数并与数据校验
    model_paths = find_models(args.result_dir, args.name)
    gmms = safe_load_gmms(args.result_dir, args.name)
    assert len(gmms) == len(model_paths), "gmm_list 数量与客户端模型数不一致"

    ckpt_num_classes = infer_num_classes_from_ckpt(model_paths[0])
    assert_data_matches_num_classes(y_te, ckpt_num_classes)

    # 3) 逐客户端推理 logits
    logits_list = []
    print(f"Infer logits from {len(model_paths)} clients ...")
    for p in tqdm(model_paths, ncols=80):
        m = Qfnn(device=device, num_classes=ckpt_num_classes).to(device)
        state = torch.load(p, map_location="cpu")
        try:
            m.load_state_dict(state, strict=False)
        except Exception:
            if isinstance(state, dict) and "model" in state:
                m.load_state_dict(state["model"], strict=False)
            else:
                raise
        logits_list.append(fwd_logits(m, x_std, device=device, batch=args.batch))

    # 4) QFI 门控权重（稳定 softmax + 可选先验/均匀混合/Top-k/下限）
    scores = np.stack([g.score_samples(x_np) for g in gmms], axis=1)         # [N, K]
    if args.use_prior:
        priors = maybe_load_priors(args.result_dir, args.name, enabled=True)
        if priors is not None:
            log_pi = np.log(priors + 1e-12) * float(args.prior_strength)
            scores = scores + log_pi[None, :]
    scores = scores - scores.max(axis=1, keepdims=True)
    w = torch.from_numpy(scores).float()
    w = F.softmax(w / float(args.tau), dim=1)

    if args.uniform_mix > 0:
        K = w.size(1)
        w = (1.0 - args.uniform_mix) * w + args.uniform_mix * (1.0 / K)

    if args.eps_floor > 0:
        w = torch.clamp(w, min=float(args.eps_floor))
        w = w / (w.sum(dim=1, keepdim=True) + 1e-12)

    if args.topk is not None and int(args.topk) > 0 and int(args.topk) < w.size(1):
        k = int(args.topk)
        vals, idxs = torch.topk(w, k=k, dim=1)
        mask = torch.zeros_like(w).scatter_(1, idxs, 1.0)
        w = w * mask
        w = w / (w.sum(dim=1, keepdim=True) + 1e-12)

    # 5) 样本级加权集成（logit 融合或概率空间融合）
    N, C = logits_list[0].shape
    if args.prob_ensemble:
        probs_ens = torch.zeros(N, C)
        for k in range(len(logits_list)):
            probs_k = F.softmax(logits_list[k] / float(args.logit_t), dim=1)
            probs_ens += probs_k * w[:, k:k+1]
        pred = probs_ens.argmax(1)
    else:
        logits_ens = torch.zeros(N, C)
        for k in range(len(logits_list)):
            logits_ens += logits_list[k] * w[:, k:k+1]
        pred = logits_ens.argmax(1)

    # 6) 评估与诊断
    y_np, pred_np = y_te.numpy(), pred.numpy()
    acc = accuracy_score(y_np, pred_np)
    P, R, F1, _ = precision_recall_fscore_support(y_np, pred_np, average="macro", zero_division=0)
    print(f"acc:{acc:.4f}  precision:{P:.4f}  recall:{R:.4f}  f1:{F1:.4f}")

    hist = torch.bincount(pred, minlength=ckpt_num_classes).tolist()
    print(f"pred hist ({ckpt_num_classes}):", hist)

    ent = (-(w * (w + 1e-12).log()).sum(dim=1)).mean().item()
    print(f"avg weight entropy: {ent:.3f}")

    try:
        print(classification_report(y_np, pred_np, digits=4))
    except Exception:
        pass

    # 均匀集成基线（消融）
    logits_avg = sum(logits_list) / len(logits_list)
    pred_avg = logits_avg.argmax(1)
    acc_avg = accuracy_score(y_np, pred_avg.numpy())
    print(f"[uniform-avg baseline] acc:{acc_avg:.4f}")


if __name__ == "__main__":
    main()