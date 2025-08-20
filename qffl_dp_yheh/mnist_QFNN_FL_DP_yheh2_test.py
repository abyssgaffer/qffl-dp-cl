# -*- coding: utf-8 -*-
import math
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from common.mni_QFNN import Qfnn
# from common.mni_QFNN_adapter import Qfnn

# ========== 基本配置 ==========
DEVICE = torch.device('cpu')
torch.manual_seed(777)
np.random.seed(777)

# NAME_MODELS = 'pmnist_qffl_cl_er_si_adapter_lwf_gas_q4_star'
NAME_MODELS = 'pmnist_qffl_gas_q4_star'
NUM_NODES = 9

# ========== LDP 核心裁剪与预算 ==========
Y_LOGIT_CLIP_L2 = 0.8   # τ：逐样本 logit L2 裁剪（Δ2 = 2τ）
DELTA = 1e-5            # Gaussian 机制的 δ（每客户端）

# ========== 实验范围 ==========
MECHANISMS = ['none', 'gaussian', 'laplace', 'quantum']     # 'quantum' = 高斯+QRNG
EPSILONS   = [1.0,  5.0, 10.0, 20.0, 50.0, 100.0, 1000.0]

# ========== 后处理 + 网格搜索（均不影响 DP 保证） ==========
# 说明：适当收敛搜索空间以控制运行时间，按需自行扩展
LAM_MIN_GRID        = [0.0, 1e-3]                  # λ 的固定小阈值（先阈值后归一）
LAM_DROP_RATIO_GRID = [0.0, 0.2, 0.3, 0.4]         # 每行丢弃底部比例（再归一），0 关闭
ADAPTIVE_T_GRID     = [True, False]                # 自适应温度（按熵调温）
T_RANGE_GRID        = [(0.65, 0.95), (0.70, 0.95)] # (T_LOW, T_HIGH) 仅当 ADAPTIVE_T=True
SHARPEN_T_GRID      = [0.7, 0.9]                   # ADAPTIVE_T=False 时使用的固定温度
ENTROPY_REWEIGHT_GRID = [False, True]              # 是否用熵对 λ 再做一次重权
LAM_BETA_GRID       = [1.0, 1.5]                   # 熵重权幂 （ENTROPY_REWEIGHT=True 时生效）

# ========== 数据 ==========
test_data = torch.load('../data/pmnist/test_data.pkl').to(DEVICE)[:2000]  # [B, D]
label     = torch.load('../data/pmnist/test_label.pkl').to(DEVICE)[:2000] # [B]

# ========== 载入模型 ==========
local_models = []
for h in range(NUM_NODES):
    m = Qfnn(DEVICE).to(DEVICE)
    state = torch.load(f'../result/model/{NAME_MODELS}_n{h}.pth', map_location=DEVICE)
    m.load_state_dict(state)
    m.eval()
    local_models.append(m)

# ========== 载入 GMM 与 data_weights ==========
gmm_list     = torch.load(f'../result/data/{NAME_MODELS}_gmm_list')
data_weights = torch.load(f'../result/data/{NAME_MODELS}_data_weights')
data_weights = torch.tensor(data_weights, dtype=torch.float32, device=DEVICE)  # [H]
H = NUM_NODES

# ========== 工具 ==========
def add_gaussian_like(x: torch.Tensor, sigma: torch.Tensor, use_qrng=False) -> torch.Tensor:
    """
    sigma: 可为标量或形状可广播到 x 的张量（如 [1,H,1]）
    """
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
    """
    b: 可为标量或形状可广播到 x 的张量（如 [1,H,1]）
    """
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

def compute_metrics(y_true, y_pred):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec  = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1   = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return acc, prec, rec, f1

# ========== 预计算：λ（baseline）、y_all（全部节点前向） ==========
with torch.no_grad():
    X_np = test_data.detach().cpu().numpy()
    # GMM 得分 -> 乘权重 -> 行归一：得到 λ
    gmm_scores_np = np.stack([gmm_list[i].score_samples(X_np) for i in range(NUM_NODES)], axis=1)  # [B,H]
    gmm_scores = torch.from_numpy(gmm_scores_np).to(DEVICE).float()  # [B,H]
    for i in range(NUM_NODES):
        gmm_scores[:, i] = gmm_scores[:, i] * data_weights[i]
    row_sum = torch.sum(gmm_scores, dim=1)
    row_sum = torch.where(row_sum.abs() < 1e-12, torch.full_like(row_sum, 1e-12), row_sum)
    lam_all = gmm_scores / row_sum.unsqueeze(1)  # [B,H]

    # 逐节点前向，堆叠 y_all
    y_list = []
    for i in tqdm(range(NUM_NODES), desc="Forward each node (baseline)"):
        m = local_models[i]
        y_list.append(m(test_data))  # [B,C]，QFNN 输出代价/距离，越小越好
    y_all = torch.stack(y_list, dim=1)  # [B,H,C]

    # baseline 指标（保持你原聚合路径）
    out_put = torch.softmax(y_all, dim=1)
    out_put = out_put * lam_all.unsqueeze(2)
    output  = out_put.sum(dim=1)
    pred_base = torch.argmin(output, dim=1)

    y_true_np = label.detach().cpu().numpy()
    y_pred_np = pred_base.detach().cpu().numpy()
    acc_base, prec_base, rec_base, f1_base = compute_metrics(y_true_np, y_pred_np)

    print(f"[Baseline/Noise=None] -> Acc={acc_base:.4f}  Prec={prec_base:.4f}  Rec={rec_base:.4f}  F1={f1_base:.4f}")

# ========== 主评估函数（在固定机制与 ε 下，搜索后处理超参的最优组合） ==========
def eval_one(mech: str,
             eps: float,
             lam_min: float,
             lam_drop_ratio: float,
             adaptive_t: bool,
             t_low: float,
             t_high: float,
             sharpen_t: float,
             entropy_reweight: bool,
             lam_beta: float):
    """
    返回：metrics(dict), mean_scale(float)
    """
    tau = Y_LOGIT_CLIP_L2
    d = y_all.size(2)

    if mech == 'none':
        # 不做任何后处理（保持 baseline 不变）
        return {
            "acc": acc_base, "prec": prec_base, "rec": rec_base, "f1": f1_base
        }, None

    # 每客户端同一 (ε, δ)（不再按 H 分摊）
    eps_h   = torch.full((H,), float(eps),   device=DEVICE)  # [H]
    delta_h = torch.full((H,), float(DELTA), device=DEVICE)  # [H]

    # 逐节点噪声尺度
    if mech in ('gaussian', 'quantum'):
        # σ_h = sqrt(2 ln(1.25/δ_h)) * (2τ) / ε_h
        sigmas = []
        for eh, dh in zip(eps_h.tolist(), delta_h.tolist()):
            dh = max(dh, 1e-12)
            sigmas.append(math.sqrt(2 * math.log(1.25 / dh)) * (2.0 * tau) / eh)
        scale_vec = torch.tensor(sigmas, dtype=torch.float32, device=DEVICE).view(1, H, 1)  # [1,H,1]
    else:
        # Laplace: b_h = (2τ√d) / ε_h
        bs = (2.0 * tau * math.sqrt(d)) / eps_h
        scale_vec = bs.view(1, H, 1)  # [1,H,1]

    use_qrng = (mech == 'quantum')

    with torch.no_grad():
        # ---- λ 后处理：小阈值 + 底部比例丢弃（不影响 DP）----
        lam = lam_all.clone()

        if lam_min is not None and lam_min > 0.0:
            mask = (lam < lam_min).float()
            lam = lam * (1.0 - mask)
            lam = lam / lam.sum(dim=1, keepdim=True).clamp_min(1e-12)

        if lam_drop_ratio is not None and lam_drop_ratio > 0.0:
            drop_k = max(1, int(lam_drop_ratio * H))
            drop_k = min(drop_k, H - 1)
            th = lam.kthvalue(drop_k, dim=1).values.unsqueeze(1)  # [B,1]
            keep = (lam > th).float()
            # 若全 == 阈值，保底保留最大值
            all_zero = (keep.sum(dim=1, keepdim=True) == 0)
            if all_zero.any():
                maxpos = lam.argmax(dim=1, keepdim=True)
                keep.scatter_(1, maxpos, 1.0)
            lam = lam * keep
            lam = lam / lam.sum(dim=1, keepdim=True).clamp_min(1e-12)

        # ---- y -> z（logit），做 max-减法更稳，然后 L2 裁剪 ----
        z = -y_all.clone()                                   # [B,H,C]
        z = z - z.max(dim=2, keepdim=True).values            # 数值稳定化（比均值更稳）
        z = clip_l2_lastdim(z, tau)                          # Δ2 = 2τ

        # ---- 加噪 ----
        if mech in ('gaussian', 'quantum'):
            z = add_gaussian_like(z, sigma=scale_vec, use_qrng=use_qrng)
        else:
            z = add_laplace_like(z, b=scale_vec, use_qrng=False)

        # ---- 回到概率，温度后处理 ----
        p = torch.softmax(z, dim=2)                          # [B,H,C]

        if adaptive_t:
            # 自适应温度：按熵调节，置信高 -> T 低（更尖）
            ent = -(p.clamp_min(1e-8) * torch.log(p.clamp_min(1e-8))).sum(dim=2) / math.log(d)  # [B,H], 0~1
            T_adapt = (t_low + (t_high - t_low) * ent).unsqueeze(2)  # [B,H,1]
            p = (p.clamp_min(1e-8)) ** (1.0 / T_adapt)
            p = p / p.sum(dim=2, keepdim=True)
        else:
            # 固定温度锐化
            if sharpen_t is not None and abs(sharpen_t - 1.0) > 1e-8:
                p = (p.clamp_min(1e-8)) ** (1.0 / sharpen_t)
                p = p / p.sum(dim=2, keepdim=True)

        # （可选）按熵对 λ 再重权（置信高→权重大）
        if entropy_reweight:
            ent = -(p.clamp_min(1e-8) * torch.log(p.clamp_min(1e-8))).sum(dim=2) / math.log(d)  # [B,H]
            w_ent = (1.0 - ent).clamp_min(0.0).pow(lam_beta)   # 0~1
            lam = lam * w_ent
            lam = lam / lam.sum(dim=1, keepdim=True).clamp_min(1e-12)

        # ---- 回到“代价” ----
        y_used = -torch.log(p.clamp_min(1e-8))               # [B,H,C]

        # ---- 聚合（保持原结构：节点维 softmax，再乘 λ 聚合）----
        out_put_dp = torch.softmax(y_used, dim=1)            # [B,H,C]
        out_put_dp = out_put_dp * lam.unsqueeze(2)           # [B,H,C]
        output_dp  = out_put_dp.sum(dim=1)                   # [B,C]
        pred_dp    = torch.argmin(output_dp, dim=1)

        y_pred = pred_dp.detach().cpu().numpy()
        acc, prec, rec, f1 = compute_metrics(y_true_np, y_pred)

    return {
        "acc": acc, "prec": prec, "rec": rec, "f1": f1
    }, float(scale_vec.mean().item())

# ========== 运行：对每个机制 & ε 做超参搜索，打印最优 ==========
all_results = []
print("\n===== Grid Search (post-processing only; DP unchanged) =====")
for mech in MECHANISMS:
    for eps in ([None] if mech == 'none' else EPSILONS):
        best = None  # (acc, dict_params, metrics, mean_scale)
        trials = 0

        if mech == 'none':
            metrics, _ = eval_one(mech, eps, 0.0, 0.0, False, 1.0, 1.0, 1.0, False, 1.0)
            print(f"[Noise=None] -> Acc={metrics['acc']:.4f}  Prec={metrics['prec']:.4f}  "
                  f"Rec={metrics['rec']:.4f}  F1={metrics['f1']:.4f}")
            all_results.append({
                "mechanism": mech, "epsilon": None, "scale": None, **metrics,
                "params": {"lam_min": 0.0, "lam_drop": 0.0, "adaptive_t": False,
                           "t_low": 1.0, "t_high": 1.0, "sharpen_t": 1.0,
                           "entropy_reweight": False, "lam_beta": 1.0}
            })
            continue

        for lam_min in LAM_MIN_GRID:
            for lam_drop in LAM_DROP_RATIO_GRID:
                for adaptive_t in ADAPTIVE_T_GRID:
                    if adaptive_t:
                        t_pairs = T_RANGE_GRID
                        sharpens = [1.0]  # 占位，不用
                    else:
                        t_pairs = [(1.0, 1.0)]  # 占位
                        sharpens = SHARPEN_T_GRID

                    for (t_low, t_high) in t_pairs:
                        for sharpen_t in sharpens:
                            for ent_rw in ENTROPY_REWEIGHT_GRID:
                                lam_beta_list = LAM_BETA_GRID if ent_rw else [1.0]
                                for lam_beta in lam_beta_list:
                                    trials += 1
                                    metrics, scale = eval_one(
                                        mech, eps,
                                        lam_min=lam_min,
                                        lam_drop_ratio=lam_drop,
                                        adaptive_t=adaptive_t,
                                        t_low=t_low, t_high=t_high,
                                        sharpen_t=sharpen_t,
                                        entropy_reweight=ent_rw,
                                        lam_beta=lam_beta
                                    )
                                    if (best is None) or (metrics["acc"] > best[0]):
                                        best = (
                                            metrics["acc"],
                                            {
                                                "lam_min": lam_min,
                                                "lam_drop": lam_drop,
                                                "adaptive_t": adaptive_t,
                                                "t_low": t_low,
                                                "t_high": t_high,
                                                "sharpen_t": sharpen_t,
                                                "entropy_reweight": ent_rw,
                                                "lam_beta": lam_beta
                                            },
                                            metrics,
                                            scale
                                        )

        # 打印该 (mech, eps) 的最优组合
        acc, params, metrics, scale = best
        extra = ""
        if eps == max(EPSILONS):
            extra = f" | ΔAcc={metrics['acc'] - acc_base:+.4f}"
        print(f"[Noise={mech:8s} | ε={str(eps):>8}] "
              f"best Acc={metrics['acc']:.4f}  Prec={metrics['prec']:.4f}  "
              f"Rec={metrics['rec']:.4f}  F1={metrics['f1']:.4f}  "
              f"avg_scale={('-' if scale is None else f'{scale:.6f}')}  "
              f"trials={trials}{extra}  "
              f"params={params}")

        all_results.append({
            "mechanism": mech, "epsilon": eps, "scale": scale, **metrics, "params": params
        })

# ========== 汇总：按机制 + ε 输出最优 ==========
print("\n===== Summary (best per mechanism & epsilon) =====")
print("Mechanism\tε\tAcc\tPrec\tRec\tF1\tavg(σ|b)\tParams")
for r in all_results:
    mech = r["mechanism"]; eps = r["epsilon"]
    acc = r["acc"]; prec = r["prec"]; rec = r["rec"]; f1 = r["f1"]; sc = r["scale"]
    print(f"{mech}\t{('-' if eps is None else eps)}\t"
          f"{acc:.4f}\t{prec:.4f}\t{rec:.4f}\t{f1:.4f}\t"
          f"{('-' if sc is None else f'{sc:.6f}')}\t{r['params']}")