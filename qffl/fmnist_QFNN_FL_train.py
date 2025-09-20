# qffl/fmnist_QFNN_FL_train.py
import os
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn.mixture import GaussianMixture

from common.mni_QFNN import Qfnn
from common.utils import acc_cal, setup_seed

# ===== 固定随机种子 =====
setup_seed(777)

# ===== 基本配置（只做参数级适配，不动主体逻辑）=====
DEVICE = torch.device('cpu')  # 保持和你原脚本一致；如需GPU可改成cuda
DATA_DIR = "../data/fmnist"
RESULT_DIR = "../result"
os.makedirs(os.path.join(RESULT_DIR, "model"), exist_ok=True)
os.makedirs(os.path.join(RESULT_DIR, "data"), exist_ok=True)

# 训练参数（比MNIST更稳）
BATCH_SIZE = 256
EPOCH = 8
LR = 1e-2
WEIGHT_DECAY = 1e-2
GRAD_CLIP = 1.0

# 任务名（沿用你之前的命名，保证测试脚本能直接找到）
NAME = "fmnist_qffl_gas_q4_star"

# 客户端划分（两类/客户端，但更均衡；避免“0类在所有客户端”）
keep_list = [
    [0, 1], [2, 3], [4, 5], [6, 7], [8, 9],
    [0, 2], [3, 4], [5, 6], [7, 8],
]
node = len(keep_list)

def main():
    # 1) 读取 10D 训练特征
    train_data = torch.load(os.path.join(DATA_DIR, "train_data.pkl")).float()
    train_label = torch.load(os.path.join(DATA_DIR, "train_label.pkl")).long()
    assert train_data.ndim == 2 and train_data.shape[1] == 10
    assert train_label.ndim == 1 and train_label.shape[0] == train_data.shape[0]
    all_len = len(train_label)

    # 2) 全局标准化参数（训练/测试共用）
    mu = train_data.mean(0)
    sigma = train_data.std(0).clamp_min(1e-6)
    torch.save({"mu": mu, "sigma": sigma}, os.path.join(RESULT_DIR, "data", f"{NAME}_scaler.pt"))

    gmm_list = []
    weights = []

    for i in range(node):
        print(f"\n=== Client {i} keep {keep_list[i]} ===")
        # 3) 子集筛选（两类/客户端）
        mask = torch.isin(train_label, torch.tensor(keep_list[i], device=train_label.device))
        x = train_data[mask]
        y = train_label[mask]
        weights.append(len(x) / all_len)

        # 4) 标准化 -> DataLoader
        def preprocess(t): return (t - mu) / sigma
        x_std = preprocess(x)

        ds = TensorDataset(x_std, y)
        dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

        # 5) 模型/优化器（不改结构）
        model = Qfnn(DEVICE).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        loss_fn = nn.CrossEntropyLoss()

        train_loss_list, train_acc_list = [], []

        # 6) 训练
        for ep in range(EPOCH):
            print(f"--- epoch {ep+1}/{EPOCH} ---")
            model.train()
            for xb, yb in tqdm(dl, ncols=80):
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)                 # 10类输出，主体逻辑不变
                loss = loss_fn(logits, yb)
                acc = acc_cal(logits, yb)

                optimizer.zero_grad()
                loss.backward()
                if GRAD_CLIP is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

                train_loss_list.append(loss.item())
                train_acc_list.append(acc)
                tqdm.write(f"loss:{loss.item():.4f} acc:{acc:.4f}")

        # 7) 保存每客户端模型与训练曲线
        torch.save(model.state_dict(), os.path.join(RESULT_DIR, "model", f"{NAME}_n{i}.pth"))
        torch.save(train_loss_list, os.path.join(RESULT_DIR, "data", f"{NAME}_train_loss_n{i}"))
        torch.save(train_acc_list, os.path.join(RESULT_DIR, "data", f"{NAME}_train_acc_n{i}"))

        # 8) 拟合 GMM（在标准化空间），保存用于门控
        gmm = GaussianMixture(n_components=5, covariance_type="full", max_iter=100, random_state=42)
        gmm.fit(x_std.cpu().numpy())
        gmm_list.append(gmm)

    # 9) 保存门控所需对象
    torch.save(gmm_list, os.path.join(RESULT_DIR, "data", f"{NAME}_gmm_list"))
    torch.save(weights,   os.path.join(RESULT_DIR, "data", f"{NAME}_data_weights"))
    print("\n✅ Training done.")

if __name__ == "__main__":
    main()