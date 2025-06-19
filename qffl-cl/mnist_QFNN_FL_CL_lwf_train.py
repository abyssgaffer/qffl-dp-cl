import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from common.mni_QFNN import Qfnn
from common.utils import acc_cal, setup_seed
from torch.utils.data import DataLoader, TensorDataset
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

# ======================== 参数配置 ========================
BATCH_SIZE = 600
EPOCH = 5
LR = 0.1
ALPHA_LWF = 0.5
TEMP = 2.0

DEVICE = torch.device('cpu')
setup_seed(777)

# ======================== 数据加载 ========================
train_data = torch.load('../data/pmnist/train_data.pkl').cpu().numpy()
train_label = torch.load('../data/pmnist/train_label.pkl').cpu().numpy()
all_len = len(train_label)

# ======================== 初始化 ========================
gmm_list = []
weights = []
NAME = 'pmnist_qffl_cl_lwf_gas_q4_star'
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

    # 构建当前数据集（LwF 无需回放和 EWC）
    x_tensor = torch.tensor(data, dtype=torch.float32)
    y_tensor = torch.tensor(labels, dtype=torch.long)
    train_set = TensorDataset(x_tensor, y_tensor)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

    # 保存旧模型用于生成软标签
    old_model = copy.deepcopy(model)
    old_model.eval()

    # 正式训练（仅 LwF）
    for epoch in range(EPOCH):
        print(f'===== node:{i} epoch:{epoch} =====')
        for x, y in tqdm(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            model.train()
            output = model(x)
            ce_loss = loss_func(output, y)

            # LwF 蒸馏损失
            with torch.no_grad():
                old_logits = old_model(x)
            distil_loss = F.kl_div(
                F.log_softmax(output / TEMP, dim=1),
                F.softmax(old_logits / TEMP, dim=1),
                reduction='batchmean'
            ) * (TEMP ** 2)

            total_loss = ce_loss + ALPHA_LWF * distil_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            acc = acc_cal(output, y)
            train_loss_list.append(total_loss.item())
            train_acc_list.append(acc)
            tqdm.write(f'loss: {total_loss.item():.4f} acc: {acc:.4f}')

    # 保存结果
    torch.save(model.state_dict(), f'../result/model/{NAME}_n{i}.pth')
    torch.save(train_loss_list, f'../result/data/{NAME}_train_loss_n{i}')
    torch.save(train_acc_list, f'../result/data/{NAME}_train_acc_n{i}')

# 保存 GMM 和 权重信息
torch.save(gmm_list, f'../result/data/{NAME}_gmm_list')
torch.save(weights, f'../result/data/{NAME}_data_weights')
