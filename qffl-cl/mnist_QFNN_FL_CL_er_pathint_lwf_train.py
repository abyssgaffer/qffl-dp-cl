import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from common.mni_QFNN import Qfnn
from common.utils import acc_cal, setup_seed
from torch.utils.data import DataLoader, TensorDataset
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
import copy

# ======================== 参数配置 ========================
BATCH_SIZE = 600
EPOCH = 5
LR = 0.1
LAMBDA_PI = 10.0
ALPHA_LWF = 0.5
TEMP = 2.0
REPLAY_SIZE = 20
EPSILON = 1e-6

DEVICE = torch.device('cpu')
setup_seed(777)

# ======================== 数据加载 ========================
train_data = torch.load('../data/pmnist/train_data.pkl').cpu().numpy()
train_label = torch.load('../data/pmnist/train_label.pkl').cpu().numpy()
all_len = len(train_label)

# ======================== 初始化 ========================
gmm_list = []
weights = []
NAME = 'pmnist_qffl_cl_er_pathint_lwf_gas_q4_star'
node = 9
keep_list = [[0,1],[0,2],[0,3],[0,4],[0,5],[0,6],[0,7],[0,8],[0,9]]

# ======================== 每个客户端 ========================
for i in range(node):
    model = Qfnn(DEVICE).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()

    train_loss_list = []
    train_acc_list = []

    # 选择子数据集
    keep = np.isin(train_label, keep_list[i])
    data = train_data[keep]
    labels = train_label[keep]
    weights.append(len(data) / all_len)

    # GMM 拟合（用于联邦加权）
    gmm = GaussianMixture(n_components=5, max_iter=100, random_state=42)
    gmm.fit(data)
    gmm_list.append(gmm)

    # ===== 经验回放构建 =====
    buffer_x, buffer_y = [], []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        selected = np.random.choice(idx, min(REPLAY_SIZE, len(idx)), replace=False)
        buffer_x.extend(data[selected])
        buffer_y.extend(labels[selected])
    replay_x = torch.tensor(np.stack(buffer_x), dtype=torch.float32)
    replay_y = torch.tensor(np.array(buffer_y), dtype=torch.long)

    # 合并当前数据 + 回放数据
    all_x = torch.cat([torch.tensor(data, dtype=torch.float32), replay_x], dim=0)
    all_y = torch.cat([torch.tensor(labels, dtype=torch.long), replay_y], dim=0)
    train_set = TensorDataset(all_x, all_y)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

    # Path Integral相关变量初始化
    prev_params = {name: p.clone().detach() for name, p in model.named_parameters()}
    omega = {name: torch.zeros_like(p) for name, p in model.named_parameters()}
    W = {name: torch.zeros_like(p) for name, p in model.named_parameters()}

    # 保存旧模型用于LwF
    old_model = copy.deepcopy(model)
    old_model.eval()

    for epoch in range(EPOCH):
        print(f'===== node:{i} epoch:{epoch} =====')
        for x, y in tqdm(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            model.train()
            optimizer.zero_grad()
            output = model(x)
            ce_loss = loss_func(output, y)

            # LwF蒸馏损失
            with torch.no_grad():
                old_logits = old_model(x)
            distil_loss = F.kl_div(
                F.log_softmax(output / TEMP, dim=1),
                F.softmax(old_logits / TEMP, dim=1),
                reduction='batchmean'
            ) * (TEMP ** 2)

            # Path Integral正则项
            pi_loss = 0.0
            for name, param in model.named_parameters():
                pi_loss += torch.sum(omega[name] * (param - prev_params[name]) ** 2)
            total_loss = ce_loss + ALPHA_LWF * distil_loss + LAMBDA_PI * pi_loss
            total_loss.backward()

            # 累积W
            for name, param in model.named_parameters():
                if param.grad is not None:
                    W[name] += param.grad.detach() * (param.detach() - prev_params[name])
            optimizer.step()

            acc = acc_cal(output, y)
            train_loss_list.append(total_loss.item())
            train_acc_list.append(acc)
            tqdm.write(f'loss: {total_loss.item():.4f} acc: {acc:.4f}')

            # 更新prev_params
            for name, param in model.named_parameters():
                prev_params[name] = param.clone().detach()

        # 每个epoch后更新omega
        for name, param in model.named_parameters():
            delta = param.detach() - prev_params[name]
            omega[name] += W[name] / (delta ** 2 + EPSILON)
            W[name].zero_()

    # 保存结果
    torch.save(model.state_dict(), f'../result/model/{NAME}_n{i}.pth')
    torch.save(train_loss_list, f'../result/data/{NAME}_train_loss_n{i}')
    torch.save(train_acc_list, f'../result/data/{NAME}_train_acc_n{i}')

# 保存 GMM 和 权重信息
torch.save(gmm_list, f'../result/data/{NAME}_gmm_list')
torch.save(weights, f'../result/data/{NAME}_data_weights') 