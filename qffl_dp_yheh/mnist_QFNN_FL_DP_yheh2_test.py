import math
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from common.mni_QFNN import Qfnn
from common.mni_QFNN_adapter import Qfnn

DEVICE = torch.device('cpu')
torch.manual_seed(777)

NAME_MODELS = 'pmnist_qffl_cl_er_si_adapter_lwf_gas_q4_star'
NUM_NODES = 9

# ======== 仅后处理超参（不影响 DP） ========
SHARPEN_T = 0.7       # 类别softmax温度（<1更尖）
LAM_MIN   = 1e-3      # 节点权重阈值，小于阈值置0再归一化

# ======== LDP 标定超参 ========
Y_LOGIT_CLIP_L2 = 0.8 # τ：逐样本logit L2裁剪（Δ2=2τ）
# 每节点 ε 分配：  ε_h = ε * (alpha * w_h + (1-alpha)/H)
EPS_ALPHA       = 0.7 # 加权分配的权重比例（0~1）
DELTA_PER_NODE  = None  # 若为 None，则 delta 均分：delta_h = delta / H

# ======== 数据 ========
test_data = torch.load('../data/pmnist/test_data.pkl').to(DEVICE)[:2000]
label     = torch.load('../data/pmnist/test_label.pkl').to(DEVICE)[:2000]

# ======== 载入模型 ========
local_models = []
for h in range(NUM_NODES):
    m = Qfnn(DEVICE).to(DEVICE)
    state = torch.load(f'../result/model/{NAME_MODELS}_n{h}.pth', map_location=DEVICE)
    m.load_state_dict(state)
    m.eval()
    local_models.append(m)

# ======== 载入 GMM 与 data_weights ========
gmm_list     = torch.load(f'../result/data/{NAME_MODELS}_gmm_list')
data_weights = torch.load(f'../result/data/{NAME_MODELS}_data_weights')
data_weights = torch.tensor(data_weights, dtype=torch.float32, device=DEVICE)  # [H]

# ======== 工具 ========
def add_gaussian_like(x: torch.Tensor, sigma: torch.Tensor, use_qrng=False) -> torch.Tensor:
    # sigma: 可为标量或 [1,H,1] 逐节点
    if not use_qrng:
        return x + torch.randn_like(x) * sigma
    try:
        from qrandom.numpy import quantum_rng
        rng = quantum_rng()
        noise = rng.normal(0.0, 1.0, size=x.numel()).astype(np.float32)
        noise = torch.from_numpy(noise).to(x.device).view_as(x)
        return x + noise * sigma
    except Exception:
        return x + torch.randn_like(x) * sigma

def add_laplace_like(x: torch.Tensor, b: torch.Tensor, use_qrng=False) -> torch.Tensor:
    # b: 可为标量或 [1,H,1]
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
    with torch.no_grad():
        n = torch.norm(x, p=2, dim=-1, keepdim=True).clamp(min=1e-12)
        scale = (max_norm / n).clamp(max=1.0)
    return x * scale

def sharpen(p: torch.Tensor, t: float) -> torch.Tensor:
    if t is None or abs(t-1.0) < 1e-8:
        return p
    q = (p.clamp_min(1e-8))**(1.0/t)
    return q / q.sum(dim=-1, keepdim=True)

# ======== Baseline（与原聚合完全一致，H 个节点） ========
with torch.no_grad():
    X_np = test_data.detach().cpu().numpy()
    gmm_scores_np = np.stack([gmm_list[i].score_samples(X_np) for i in range(NUM_NODES)], axis=1)  # [B,H]
    gmm_scores = torch.from_numpy(gmm_scores_np).to(DEVICE).float()
    for i in range(NUM_NODES):
        gmm_scores[:, i] = gmm_scores[:, i] * data_weights[i]
    row_sum = torch.sum(gmm_scores, dim=1)
    row_sum = torch.where(row_sum.abs() < 1e-12, torch.full_like(row_sum, 1e-12), row_sum)
    lam_all = gmm_scores / row_sum.unsqueeze(1)  # [B,H]

    # 逐节点前向
    y_list = []
    for i in tqdm(range(NUM_NODES), desc="Forward each node (baseline)"):
        m = local_models[i]
        y_list.append(m(test_data))            # [B,C]
    y_all = torch.stack(y_list, dim=1)         # [B,H,C]

    # 聚合
    out_put = torch.softmax(y_all, dim=1)
    out_put = out_put * lam_all.unsqueeze(2)
    output  = out_put.sum(dim=1)
    pred_base = torch.argmin(output, dim=1)

    y_true = label.detach().cpu().numpy()
    y_pred = pred_base.detach().cpu().numpy()
    acc_base = accuracy_score(y_true, y_pred)
    prec_base = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec_base  = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_base   = f1_score(y_true, y_pred, average='macro', zero_division=0)
    print(f"[Baseline/Noise=None] -> Acc={acc_base:.4f}  Prec={prec_base:.4f}  Rec={rec_base:.4f}  F1={f1_base:.4f}")

# ======== 预计算 log_e（如需 e 通道可用） ========
with torch.no_grad():
    log_e_np = np.stack([gmm_list[i].score_samples(X_np) for i in range(NUM_NODES)], axis=1)
    log_e = torch.from_numpy(log_e_np).to(DEVICE).float()  # [B,H]

# ======== DP 实验配置（仅 y 做 LDP） ========
mechanisms = ['none', 'gaussian', 'laplace', 'quantum']  # 'quantum' = 高斯+QRNG
epsilons   = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 1000.0, 1e6]
delta      = 1e-5
H          = NUM_NODES
d          = y_all.size(2)
tau        = Y_LOGIT_CLIP_L2

# 预算分配权重（与输入无关）：w>=0, sum(w)=1
w = data_weights.clamp_min(1e-12)
w = w / w.sum()

results = []
max_eps = max(epsilons)

for mech in mechanisms:
    for eps in ([None] if mech == 'none' else epsilons):

        if mech == 'none':
            acc, prec, rec, f1 = acc_base, prec_base, rec_base, f1_base
            scale_vec = None
        else:
            # per-node epsilon 配置： ε_h = ε * (alpha*w_h + (1-alpha)/H)
            # eps_h = eps * (EPS_ALPHA * w + (1.0 - EPS_ALPHA) / H)  # [H]
            # # per-node delta（可均分或按权重分配）
            # if DELTA_PER_NODE is None:
            #     delta_h = (delta / H) * torch.ones_like(eps_h)
            # else:
            #     delta_h = delta * (EPS_ALPHA * w + (1.0 - EPS_ALPHA) / H)

            # 每个客户端独立用同一个 (ε, δ)：不再按 H 或权重分摊
            eps_h = torch.full((H,), float(eps), device=DEVICE)  # [H]
            delta_h = torch.full((H,), float(delta), device=DEVICE)  # [H]


            # 逐节点噪声尺度
            if mech in ('gaussian', 'quantum'):
                # σ_h = sqrt(2 ln(1.25/δ_h)) * (2τ) / ε_h
                sigmas = []
                for eh, dh in zip(eps_h.tolist(), delta_h.tolist()):
                    dh = max(dh, 1e-12)
                    sigmas.append(math.sqrt(2 * math.log(1.25 / dh)) * (2.0 * tau) / eh)
                scale_vec = torch.tensor(sigmas, dtype=torch.float32, device=DEVICE).view(1, H, 1)  # [1,H,1]
            else:
                # b_h = (2τ√d) / ε_h
                bs = (2.0 * tau * math.sqrt(d)) / eps_h
                scale_vec = bs.view(1, H, 1)  # [1,H,1]

            use_qrng = (mech == 'quantum')

            with torch.no_grad():
                # λ 小权重截断 + 重归一化（后处理，DP 不变）
                lam = lam_all.clone()
                if LAM_MIN is not None and LAM_MIN > 0:
                    mask = (lam < LAM_MIN).float()
                    lam = lam * (1.0 - mask)
                    lam_sum = lam.sum(dim=1, keepdim=True).clamp_min(1e-12)
                    lam = lam / lam_sum

                # y -> z（logit）、去均值、裁剪
                z = -y_all.clone()                   # [B,H,C]
                z = z - z.mean(dim=2, keepdim=True)
                z = clip_l2_lastdim(z, tau)

                # 加噪（逐节点不同尺度）
                if mech in ('gaussian', 'quantum'):
                    z = add_gaussian_like(z, sigma=scale_vec, use_qrng=use_qrng)
                else:
                    z = add_laplace_like(z, b=scale_vec, use_qrng=False)

                # 回到概率 -> 锐化 -> 代价
                p = torch.softmax(z, dim=2)
                p = sharpen(p, SHARPEN_T)
                y_used = -torch.log(p.clamp_min(1e-8))

                # 聚合（与你原结构一致）
                out_put_dp = torch.softmax(y_used, dim=1)
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
            "epsilon": (None if mech == 'none' else eps),
            "scale": (None if mech == 'none' else scale_vec.mean().item()),
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1
        })

        if mech == 'none':
            print(f"[Noise=None] -> Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}")
        else:
            s = f"{results[-1]['scale']:.6f}"
            extra = " | ΔAcc={:+.4f}".format(acc - acc_base) if eps == max_eps else ""
            print(f"[Noise={mech:8s} | ε={eps:>8}] avg_scale={s} -> "
                  f"Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}{extra}")

# ======== 汇总 ========
print(f"\n=== Summary (H={H}, tau={tau}, t={SHARPEN_T}, eps_alpha={EPS_ALPHA}) ===")
print("Mechanism\tε\tavg(σ|b)\tAcc\tPrec\tRec\tF1")
for r in results:
    mech = r['mechanism']; eps = r['epsilon']; s = r['scale']
    print(f"{mech}\t{('-' if eps is None else eps)}\t"
          f"{('-' if s is None else f'{s:.6f}')}\t"
          f"{r['accuracy']:.4f}\t{r['precision']:.4f}\t{r['recall']:.4f}\t{r['f1']:.4f}")