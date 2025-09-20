# qffl/emnist47_QFNN_FL_train.py
import os, numpy as np, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn.mixture import GaussianMixture

from common.emnist_QFNN import Qfnn
from common.utils import acc_cal, setup_seed

# ========== 基本配置 ==========
setup_seed(777)
DEVICE = torch.device('cpu')
DATA_DIR = "../data/emnist"   # ★ 你的 47 类特征目录
RESULT_DIR = "../result"
os.makedirs(os.path.join(RESULT_DIR, "model"), exist_ok=True)
os.makedirs(os.path.join(RESULT_DIR, "data"), exist_ok=True)

NAME = "emnist47_qffl_qfi"

BATCH_SIZE   = 256
EPOCH        = 8
LR           = 1e-2
WEIGHT_DECAY = 1e-2
GRAD_CLIP    = 1.0

NUM_CLIENTS  = 9
DIRICHLET_ALPHA = 0.1   # 非 IID 程度（越小越偏）

# ========== 数据加载 ==========
def load_pkls(dir_):
    x_tr = torch.load(os.path.join(dir_, "train_data.pkl")).float()
    y_tr = torch.load(os.path.join(dir_, "train_label.pkl")).long()
    assert x_tr.ndim == 2 and x_tr.shape[1] == 10 and y_tr.ndim == 1
    return x_tr, y_tr

# ========== Dirichlet 非IID 划分 ==========
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
        # 防止某客户端拿不到该类
        props = (props + 1e-8) / (props.sum() + 1e-8)
        cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
        splits = np.split(idx, cuts)
        for i in range(num_clients):
            client_indices[i].extend(splits[i].tolist())
    return [np.array(sorted(ix)) for ix in client_indices]

def main():
    x_tr, y_tr = load_pkls(DATA_DIR)
    all_len    = len(y_tr)
    num_classes = int(y_tr.max().item() + 1)  # 应为 47

    # 标准化参数（测试共用）
    mu = x_tr.mean(0)
    sigma = x_tr.std(0).clamp_min(1e-6)
    torch.save({"mu":mu, "sigma":sigma}, os.path.join(RESULT_DIR, "data", f"{NAME}_scaler.pt"))

    # 客户端索引
    client_ids = dirichlet_split(y_tr, NUM_CLIENTS, DIRICHLET_ALPHA, seed=777)

    # 如果你更偏好“静态 keep_list”（每客户端固定若干类），可用下面示例：
    # keep_list = [
    #   list(range(0,10)), list(range(8,18)), list(range(16,26)),
    #   list(range(24,34)), list(range(32,42)), list(range(40,47))+list(range(0,5)),
    #   list(range(6,16)),  list(range(14,24)), list(range(22,32)),
    # ]
    # 然后把 client_ids = [...] 改成：对每个 keep 取这些类的所有样本索引。

    gmm_list, weights = [], []

    for i in range(NUM_CLIENTS):
        idxs = client_ids[i]
        x, y = x_tr[idxs], y_tr[idxs]
        weights.append(len(x) / all_len)

        # 标准化
        x_std = (x - mu) / sigma

        # DataLoader
        dl = DataLoader(TensorDataset(x_std, y), batch_size=BATCH_SIZE, shuffle=True)

        # 模型（仅把 num_classes 改为 47；主体结构不变）
        model = Qfnn(DEVICE, num_classes=num_classes).to(DEVICE)
        opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        ce    = nn.CrossEntropyLoss()

        loss_log, acc_log = [], []
        for ep in range(EPOCH):
            print(f"\n=== Client {i} | epoch {ep+1}/{EPOCH} | samples {len(x)} ===")
            model.train()
            for xb, yb in tqdm(dl, ncols=80):
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)               # [B, 47]
                loss   = ce(logits, yb)

                opt.zero_grad(); loss.backward()
                if GRAD_CLIP: torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step()

                loss_log.append(loss.item())
                acc_log.append(acc_cal(logits, yb))
                tqdm.write(f"loss:{loss.item():.4f} acc:{acc_cal(logits,yb):.4f}")

        # 保存
        torch.save(model.state_dict(), os.path.join(RESULT_DIR, "model", f"{NAME}_n{i}.pth"))
        torch.save(loss_log, os.path.join(RESULT_DIR, "data", f"{NAME}_train_loss_n{i}"))
        torch.save(acc_log,  os.path.join(RESULT_DIR, "data", f"{NAME}_train_acc_n{i}"))

        # GMM（QFI 门控用），在标准化空间拟合
        gmm = GaussianMixture(n_components=7, covariance_type="full", max_iter=150, random_state=42)
        gmm.fit(x_std.cpu().numpy())
        gmm_list.append(gmm)

    torch.save(gmm_list, os.path.join(RESULT_DIR, "data", f"{NAME}_gmm_list"))
    torch.save(weights,   os.path.join(RESULT_DIR, "data", f"{NAME}_data_weights"))
    print("\n✅ EMNIST-47 训练完成。")

if __name__ == "__main__":
    main()