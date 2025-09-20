# qffl/mnist_QFNN_FL_train.py
import os, numpy as np, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn.mixture import GaussianMixture

from common.emnist784_QFNN import Qfnn
from common.utils import acc_cal, setup_seed

# ===== 基本配置（保持你原先的风格）=====
setup_seed(777)
DEVICE = torch.device('cpu')

DATA_DIR = "../data/emnist784"  # ★ 改这里：EMNIST-784 的四个 pkl 在这
RESULT_DIR = "../result"
os.makedirs(os.path.join(RESULT_DIR, "model"), exist_ok=True)
os.makedirs(os.path.join(RESULT_DIR, "data"), exist_ok=True)

NAME = "emnist784_qffl_qfi"  # ★ 建议用新名字避免和旧权重冲突

# 维持你原 MNIST 的超参（若你原来不同，按需改回）
BATCH_SIZE = 600
EPOCH = 5
LR = 0.1
WEIGHT_DECAY = 0.0
GRAD_CLIP = None

NUM_CLIENTS = 9
DIRICHLET_ALPHA = 0.5  # 非IID 程度，尽量接近你原先的分布；需要可改


def load_pkls(dir_):
    x_tr = torch.load(os.path.join(dir_, "train_data.pkl")).float()
    y_tr = torch.load(os.path.join(dir_, "train_label.pkl")).long()
    return x_tr, y_tr


def dirichlet_split(y, num_clients=NUM_CLIENTS, alpha=DIRICHLET_ALPHA, seed=777):
    rng = np.random.RandomState(seed)
    y_np = y.cpu().numpy()
    classes = np.unique(y_np)
    idx_by_c = {c: np.where(y_np == c)[0] for c in classes}
    for c in classes:
        rng.shuffle(idx_by_c[c])

    client_indices = [[] for _ in range(num_clients)]
    for c in classes:
        idx = idx_by_c[c]
        props = rng.dirichlet(alpha * np.ones(num_clients))
        props = (props + 1e-8) / (props.sum() + 1e-8)
        cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
        splits = np.split(idx, cuts)
        for i in range(num_clients):
            client_indices[i].extend(splits[i].tolist())
    return [np.array(sorted(ix)) for ix in client_indices]


def main():
    # 1) 读数据（784D）
    x_tr, y_tr = load_pkls(DATA_DIR)
    in_dim = x_tr.shape[1]  # ★ 自动推断输入维度（应为 784）
    num_classes = int(y_tr.max().item() + 1)  # ★ 自动推断类别数（balanced=47）
    all_len = len(y_tr)

    # 2) 标准化参数（测试共用）
    mu = x_tr.mean(0)
    sigma = x_tr.std(0).clamp_min(1e-6)
    torch.save({"mu": mu, "sigma": sigma}, os.path.join(RESULT_DIR, "data", f"{NAME}_scaler.pt"))

    # 3) 客户端划分（保持原结构：每客户端独立训练；这里用 Dirichlet 分布近似你的非IID）
    client_ids = dirichlet_split(y_tr, NUM_CLIENTS, DIRICHLET_ALPHA, seed=777)

    gmm_list, weights = [], []

    for i in range(NUM_CLIENTS):
        idxs = client_ids[i]
        x, y = x_tr[idxs], y_tr[idxs]
        weights.append(len(x) / all_len)

        # 标准化到训练统计
        x_std = (x - mu) / sigma

        # DataLoader
        dl = DataLoader(TensorDataset(x_std, y), batch_size=BATCH_SIZE, shuffle=True)

        # 4) 模型（仅把 in_dim/num_classes 传入；其它结构不变）
        model = Qfnn(device=DEVICE, num_classes=num_classes, in_dim=in_dim).to(DEVICE)
        opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)  # 贴近你原 MNIST 风格
        ce = nn.CrossEntropyLoss()

        # 训练
        loss_log, acc_log = [], []
        for ep in range(EPOCH):
            print(f"\n=== Client {i} | epoch {ep + 1}/{EPOCH} | samples {len(x)} ===")
            model.train()
            for xb, yb in tqdm(dl, ncols=80):
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                loss = ce(logits, yb)
                opt.zero_grad();
                loss.backward()
                if GRAD_CLIP:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step()
                loss_log.append(loss.item())
                acc_log.append(acc_cal(logits, yb))
                tqdm.write(f"loss:{loss.item():.4f} acc:{acc_log[-1]:.4f}")

        # 保存
        torch.save(model.state_dict(), os.path.join(RESULT_DIR, "model", f"{NAME}_n{i}.pth"))
        torch.save(loss_log, os.path.join(RESULT_DIR, "data", f"{NAME}_train_loss_n{i}"))
        torch.save(acc_log, os.path.join(RESULT_DIR, "data", f"{NAME}_train_acc_n{i}"))

        # 5) GMM（QFI 门控用）。784D 建议 diag 协方差更稳（最小改动）
        gmm = GaussianMixture(n_components=5, covariance_type="diag", reg_covar=1e-5,
                              max_iter=150, random_state=42)
        gmm.fit(x_std.cpu().numpy())
        gmm_list.append(gmm)

    # 汇总保存
    torch.save(gmm_list, os.path.join(RESULT_DIR, "data", f"{NAME}_gmm_list"))
    torch.save(weights, os.path.join(RESULT_DIR, "data", f"{NAME}_data_weights"))
    print("\n✅ EMNIST-784 训练完成。")


if __name__ == "__main__":
    main()
