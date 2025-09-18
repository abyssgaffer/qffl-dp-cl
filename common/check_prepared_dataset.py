# common/check_prepared_dataset.py
import os
import torch
from torch.utils.data import TensorDataset, DataLoader

# ========= 可配置区域（已“写死”路径）=========
DATA_DIRS = [
    "../data/fmnist",
    "../data/emnist",
    # "./data/pmnist",  # 需要时去掉注释
]
DO_QUICK_TRAIN = True     # 是否做线性分类器快速验证
EPOCHS = 5                # 快训轮数
BATCH_SIZE = 512
LR = 1e-2
# ============================================

def load_dir(d):
    x_tr = torch.load(os.path.join(d, "train_data.pkl"))
    y_tr = torch.load(os.path.join(d, "train_label.pkl"))
    x_te = torch.load(os.path.join(d, "test_data.pkl"))
    y_te = torch.load(os.path.join(d, "test_label.pkl"))
    return x_tr, y_tr, x_te, y_te

def basic_checks(name, x_tr, y_tr, x_te, y_te):
    print(f"\n=== {name} ===")
    print("train_data:", tuple(x_tr.shape), x_tr.dtype)
    print("train_label:", tuple(y_tr.shape), y_tr.dtype)
    print("test_data:", tuple(x_te.shape), x_te.dtype)
    print("test_label:", tuple(y_te.shape), y_te.dtype)

    # 形状一致性
    assert x_tr.ndim == 2 and x_te.ndim == 2, "data 应是二维 [N, D]"
    assert y_tr.ndim == 1 and y_te.ndim == 1, "label 应是一维 [N]"
    assert x_tr.shape[0] == y_tr.shape[0], "train 数量不一致"
    assert x_te.shape[0] == y_te.shape[0], "test 数量不一致"
    D = x_tr.shape[1]
    print(f"feature_dim: {D}")

    # 数值健康度
    def stats(x, tag):
        nan = torch.isnan(x).any().item()
        inf = torch.isinf(x).any().item()
        mean = x.mean().item()
        std = x.std().item()
        print(f"{tag}: nan={nan}, inf={inf}, mean={mean:.4f}, std={std:.4f}")
    stats(x_tr, "train_data")
    stats(x_te, "test_data")

    # 标签范围与类分布
    def label_info(y, tag):
        u = torch.unique(y)
        print(f"{tag}: classes={len(u)}, min={int(y.min())}, max={int(y.max())}")
        vals, cnts = torch.unique(y, return_counts=True)
        pairs = list(zip([int(v) for v in vals.tolist()], [int(c) for c in cnts.tolist()]))
        head = ", ".join([f"{v}:{c}" for v,c in pairs[:20]])
        print(f"{tag} distribution(head): {head}")
        return len(u)
    num_classes_tr = label_info(y_tr, "train_label")
    num_classes_te = label_info(y_te, "test_label")
    assert num_classes_tr == num_classes_te, "train/test 类别数不一致"

    # 合理性提示
    if num_classes_tr == 10:
        print("- 检测到 10 类：可能是 MNIST/FMNIST/EMNIST-digits。")
    elif num_classes_tr == 26:
        print("- 检测到 26 类：可能是 EMNIST-letters（我们应已做 1..26 → 0..25 的转换）。")
    else:
        print(f"- 检测到 {num_classes_tr} 类：请确认与期望 split 一致。")

    return D, num_classes_tr

class LinearHead(torch.nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = torch.nn.Linear(in_dim, num_classes)
    def forward(self, x):
        return self.fc(x)

@torch.no_grad()
def eval_acc(model, loader, device):
    correct = total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += yb.numel()
    return correct / max(1, total)

def quick_train(name, x_tr, y_tr, x_te, y_te, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D = x_tr.shape[1]
    K = int(torch.unique(y_tr).numel())
    model = LinearHead(D, K).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = torch.nn.CrossEntropyLoss()

    tr_loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=batch_size, shuffle=True)
    te_loader = DataLoader(TensorDataset(x_te, y_te), batch_size=batch_size)

    for ep in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = crit(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * yb.size(0)
        tr_acc = eval_acc(model, tr_loader, device)
        te_acc = eval_acc(model, te_loader, device)
        print(f"[{name}] epoch {ep}/{epochs}  loss={total_loss/len(tr_loader.dataset):.4f}  "
              f"train_acc={tr_acc*100:.2f}%  test_acc={te_acc*100:.2f}%")

def main():
    for d in DATA_DIRS:
        if not os.path.isdir(d):
            print(f"\n⚠️  跳过：找不到目录 {d}")
            continue
        name = os.path.basename(d.rstrip("/"))
        x_tr, y_tr, x_te, y_te = load_dir(d)
        D, K = basic_checks(name, x_tr, y_tr, x_te, y_te)
        if DO_QUICK_TRAIN:
            quick_train(name, x_tr, y_tr, x_te, y_te)

if __name__ == "__main__":
    main()