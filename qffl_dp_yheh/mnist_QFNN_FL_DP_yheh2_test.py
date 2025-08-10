# -*- coding: utf-8 -*-
import math
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ============ 设备：强制 CPU，避免量子层在 MPS/CUDA 上的不确定性 ============
DEVICE = torch.device('cpu')
torch.manual_seed(777)

# ============ 你项目里的模型 ============
# from common.mni_QFNN import Qfnn  # QFNN 模型下
from common.mni_QFNN_adapter import Qfnn

# ============ 路径 / 名称 ============
# 先指向你“已验证能到 acc≈0.8845”的那套产物
NAME_MODELS = 'pmnist_qffl_cl_er_si_adapter_lwf_gas_q4_star'   # 模型权重
NAME_GMM    = 'pmnist_qffl_cl_er_si_adapter_lwf_gas_q4_star'   # GMM 与 data_weights
NUM_NODES   = 9

# ============ 数据 ============
test_data = torch.load('../data/pmnist/test_data.pkl').to(DEVICE)[:2000]   # [B, D]
label     = torch.load('../data/pmnist/test_label.pkl').to(DEVICE)[:2000]  # [B]

# ============ 载入模型 ============
local_models = []
for h in range(NUM_NODES):
    m = Qfnn(DEVICE).to(DEVICE)
    state = torch.load(f'../result/model/{NAME_MODELS}_n{h}.pth', map_location=DEVICE)
    m.load_state_dict(state)
    m.eval()
    local_models.append(m)

# ============ 载入 GMM 与 data_weights ============
gmm_list     = torch.load(f'../result/data/{NAME_GMM}_gmm_list')
data_weights = torch.load(f'../result/data/{NAME_GMM}_data_weights')  # python list
data_weights = torch.tensor(data_weights, dtype=torch.float32, device=DEVICE)  # [H]

# ============ DP 机制工具（内置实现，独立可跑） ============
def gaussian_sigma(epsilon, delta, sensitivity):
    # σ = sqrt(2 ln(1.25/δ)) * Δ / ε
    return math.sqrt(2 * math.log(1.25 / delta)) * (sensitivity / epsilon)

def laplace_b(epsilon, sensitivity):
    # b = Δ / ε
    return sensitivity / epsilon

def add_gaussian_like(x: torch.Tensor, sigma: float, use_qrng=False) -> torch.Tensor:
    """对 tensor 加高斯噪声；use_qrng=True 时尝试用量子随机，失败回退到 torch。"""
    if not use_qrng:
        return x + torch.randn_like(x) * sigma
    try:
        from qrandom.numpy import quantum_rng
        rng = quantum_rng()
        noise = rng.normal(0.0, sigma, size=x.numel()).astype(np.float32)
        noise = torch.from_numpy(noise).to(x.device).view_as(x)
        return x + noise
    except Exception:
        return x + torch.randn_like(x) * sigma

def add_laplace_like(x: torch.Tensor, b: float, use_qrng=False) -> torch.Tensor:
    """对 tensor 加拉普拉斯噪声；use_qrng=True 时尝试用量子随机，失败回退到 torch。"""
    if not use_qrng:
        u = torch.rand_like(x) - 0.5
        noise = -b * torch.sign(u) * torch.log1p(-2 * torch.abs(u))
        return x + noise
    try:
        from qrandom.numpy import quantum_rng
        rng = quantum_rng()
        u = rng.random(size=x.numel()).astype(np.float32) - 0.5
        u = torch.from_numpy(u).to(x.device).view_as(x)
        noise = -b * torch.sign(u) * torch.log1p(-2 * torch.abs(u))
        return x + noise
    except Exception:
        u = torch.rand_like(x) - 0.5
        noise = -b * torch.sign(u) * torch.log1p(-2 * torch.abs(u))
        return x + noise

def clip_l2_lastdim(x: torch.Tensor, max_norm: float) -> torch.Tensor:
    """对最后一维做逐样本 L2 裁剪。"""
    with torch.no_grad():
        n = torch.norm(x, p=2, dim=-1, keepdim=True).clamp(min=1e-12)
        scale = (max_norm / n).clamp(max=1.0)
    return x * scale

# ============ 先做 baseline（完全复刻你原 test.py 的聚合路径） ============
with torch.no_grad():
    # 1) GMM 分数（log-likelihood）
    X_np = test_data.detach().cpu().numpy()  # [B, D]
    gmm_scores_np = np.stack([gmm_list[i].score_samples(X_np) for i in range(NUM_NODES)], axis=1)  # [B, H]
    gmm_scores = torch.from_numpy(gmm_scores_np).to(DEVICE).float()  # [B, H]

    # 2) 乘以 data_weights（线性，不是 softmax）
    for i in range(NUM_NODES):
        gmm_scores[:, i] = gmm_scores[:, i] * data_weights[i]

    # 3) 线性归一化（按你原来的实现：除以行和，且行和不是 keepdim）
    row_sum = torch.sum(gmm_scores, dim=1)  # [B]
    row_sum = torch.where(row_sum.abs() < 1e-12, torch.full_like(row_sum, 1e-12), row_sum)
    for i in range(NUM_NODES):
        gmm_scores[:, i] = gmm_scores[:, i] / row_sum  # 现在 gmm_scores 语义是 λ_h

    # 4) 逐节点前向，堆叠
    y_list = []
    for i in tqdm(range(NUM_NODES), desc="Forward each node (baseline)"):
        m = local_models[i]
        y_list.append(m(test_data))          # [B, C]，QFNN 输出“代价/距离”，越小越好
    y_stack = torch.stack(y_list, dim=1)     # [B, H, C]

    # 5) 节点维 softmax → 乘 λ → 求和 → argmin
    out_put = torch.softmax(y_stack, dim=1)  # [B, H, C]
    for i in range(NUM_NODES):
        out_put[:, i, :] = out_put[:, i, :] * gmm_scores[:, i].unsqueeze(1)
    output = torch.sum(out_put, dim=1)       # [B, C]
    pred_base = torch.argmin(output, dim=1)

    # baseline 指标
    y_true = label.detach().cpu().numpy()
    y_pred = pred_base.detach().cpu().numpy()
    acc_base = accuracy_score(y_true, y_pred)
    prec_base = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec_base  = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_base   = f1_score(y_true, y_pred, average='macro', zero_division=0)
    print(f"[Baseline/Noise=None] -> Acc={acc_base:.4f}  Prec={prec_base:.4f}  Rec={rec_base:.4f}  F1={f1_base:.4f}")

    # 若 baseline 不对，给点诊断
    if acc_base < 0.1:
        rs = torch.sum(gmm_scores, dim=1)
        print("Diag λ row-sum (min, max):", rs.min().item(), rs.max().item())
        print("Diag any NaN in output?  ", torch.isnan(output).any().item())
        vals, cnts = torch.unique(pred_base, return_counts=True)
        print("Diag pred histogram:     ", {int(v): int(c) for v, c in zip(vals.tolist(), cnts.tolist())})

# ============ 预计算：y_all（之后 DP 直接复用）、log_e ============
with torch.no_grad():
    # y_all：一次前向，后面复用
    y_all = []
    for i in tqdm(range(NUM_NODES), desc="Forward each node (cache y_all)"):
        m = local_models[i]
        y_all.append(m(test_data))                 # [B, C]
    y_all = torch.stack(y_all, dim=1)              # [B, H, C]

    # log_e：gmm score_samples
    log_e_np = np.stack([gmm_list[i].score_samples(X_np) for i in range(NUM_NODES)], axis=1)  # [B, H]
    log_e = torch.from_numpy(log_e_np).to(DEVICE).float()                                      # [B, H]

# ============ DP 实验配置 ============
mechanisms       = ['none', 'gaussian', 'laplace', 'quantum']  # 量子=高斯+QRNG
# inject_positions = ['y', 'e', 'both']
inject_positions = ['y']
epsilons         = [0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 1000.0, 1e6]  # 含超大值，验证能否逼近 baseline
delta            = 1e-5    # 高斯的 δ
# 灵敏度（裁剪阈值）
Y_CLIP_L2        = 1.0     # 对 p_h（概率向量）逐样本 L2 裁剪
Y_LOGIT_CLIP_L2 = 0.8
E_CLIP_ABS       = 8.0     # 对 log_e 做绝对值裁剪（标量灵敏度）；仍用你的“线性归一化”来得到 λ

results = []
max_eps = max(epsilons)

# ============ 实验主循环 ============
for mech in mechanisms:
    pos_list = ['none'] if mech == 'none' else inject_positions
    for pos in pos_list:
        for eps in ([None] if mech == 'none' else epsilons):

            if mech == 'none':
                # 直接复用 baseline
                acc, prec, rec, f1 = acc_base, prec_base, rec_base, f1_base
                scale_y = None; scale_e = None
            else:
                # 计算噪声尺度（展示用）
                if mech in ('gaussian', 'quantum'):
                    scale_y = gaussian_sigma(eps, delta, Y_LOGIT_CLIP_L2)
                    scale_e = gaussian_sigma(eps, delta, E_CLIP_ABS)
                elif mech == 'laplace':
                    scale_y = laplace_b(eps, Y_LOGIT_CLIP_L2)
                    scale_e = laplace_b(eps, E_CLIP_ABS)
                use_qrng = (mech == 'quantum')

                with torch.no_grad():
                    # ----- e 通道：在 log 域做 clip+noise，然后“线性归一化”回 λ（保持原结构）-----
                    if pos == 'y':
                        lam = gmm_scores  # 用 baseline 的 λ
                    else:
                        le = log_e.clone()                                 # [B, H]
                        le = torch.clamp(le, -E_CLIP_ABS, E_CLIP_ABS)      # 绝对值裁剪
                        if mech in ('gaussian', 'quantum'):
                            le = add_gaussian_like(le, sigma=scale_e, use_qrng=use_qrng)
                        else:  # laplace
                            le = add_laplace_like(le, b=scale_e, use_qrng=use_qrng)
                        # 乘权重 → 线性归一化（与你原 test 完全一致）
                        lam = le.clone()
                        for i in range(NUM_NODES):
                            lam[:, i] = lam[:, i] * data_weights[i]
                        rs = torch.sum(lam, dim=1)
                        rs = torch.where(rs.abs() < 1e-12, torch.full_like(rs, 1e-12), rs)
                        for i in range(NUM_NODES):
                            lam[:, i] = lam[:, i] / rs

                    # ----- y 通道：在概率域做裁剪+噪声，再映回代价域 -----
                    if pos == 'e':
                        y_all_used = y_all
                    else:
                        # 1) 取 logit：z = -y
                        z = -y_all.clone()  # [B, H, C]
                        # 2) 可选去均值，帮助缩小范数（不影响 DP，因为后面有 clip）
                        z = z - z.mean(dim=2, keepdim=True)
                        # 3) 逐样本 L2 裁剪（Δ = Y_LOGIT_CLIP_L2）
                        z = clip_l2_lastdim(z, Y_LOGIT_CLIP_L2)
                        # 4) 加噪
                        if mech in ('gaussian', 'quantum'):
                            z = add_gaussian_like(z, sigma=scale_y, use_qrng=use_qrng)
                        else:  # laplace
                            z = add_laplace_like(z, b=scale_y, use_qrng=use_qrng)
                        # 5) 回到概率，再映回“代价”
                        p = torch.softmax(z, dim=2)  # [B, H, C]
                        y_all_used = -torch.log(p.clamp_min(1e-8))  # [B, H, C]

                    # ----- 聚合（保持你原结构）-----
                    out_put_dp = torch.softmax(y_all_used, dim=1)  # 节点维 softmax
                    out_put_dp = out_put_dp * lam.unsqueeze(2)
                    output_dp  = out_put_dp.sum(dim=1)
                    pred_dp    = torch.argmin(output_dp, dim=1)

                    # 评估
                    y_pred = pred_dp.detach().cpu().numpy()
                    y_true = label.detach().cpu().numpy()
                    acc = accuracy_score(y_true, y_pred)
                    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
                    rec  = recall_score(y_true, y_pred, average='macro', zero_division=0)
                    f1   = f1_score(y_true, y_pred, average='macro', zero_division=0)

            results.append({
                "mechanism": mech,
                "inject_pos": pos,
                "epsilon": (None if mech == 'none' else eps),
                "scale_y": scale_y,
                "scale_e": scale_e,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1
            })

            # 打印
            if mech == 'none':
                print(f"[Noise=None] -> Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}")
            else:
                s_y = "-" if scale_y is None else f"{scale_y:.6f}"
                s_e = "-" if scale_e is None else f"{scale_e:.6f}"
                extra = ""
                if eps == max_eps:
                    extra = f" | ΔAcc={acc-acc_base:+.4f}"
                print(f"[Noise={mech:8s} | Inject={pos:4s} | ε={eps:>8}] "
                      f"σ/b_y={s_y}  σ/b_e={s_e}  -> Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}{extra}")

# ============ 汇总 ============
print("\n=== Summary ===")
print("Mechanism\tInjectAt\tε\tσ/b_y\tσ/b_e\tAcc\tPrec\tRec\tF1")
for r in results:
    mech = r['mechanism']; pos = r['inject_pos']; eps = r['epsilon']
    sy = r['scale_y']; se = r['scale_e']
    print(f"{mech}\t{pos}\t{('-' if eps is None else eps)}\t"
          f"{('-' if sy is None else f'{sy:.6f}')}\t{('-' if se is None else f'{se:.6f}')}\t"
          f"{r['accuracy']:.4f}\t{r['precision']:.4f}\t{r['recall']:.4f}\t{r['f1']:.4f}")