# common/emnist_data_prepare_by_cnn.py
# 固定配置：EMNIST balanced → 原始 784 维特征 → 存到 ../data/emnist/*.pkl
import os
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# ===== 固定参数（可按需改，但默认写死）=====
SPLIT = "balanced"           # 'balanced'（47类）
DATA_ROOT = "../data/oemnist" # torchvision 原始数据缓存目录
OUT_DIR = "../data/emnist784"   # 导出的 pkl 目录（train/test_*）
BATCH = 1024
WORKERS = 0                  # macOS 建议 0
SEED = 777

def setup_seed(seed: int = 777):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def emnist_fix_orientation():
    # 官方建议的方向修正：先 ToTensor -> (C,H,W)，再做转置+镜像
    return transforms.Lambda(lambda x: x.transpose(1, 2).flip(2))

def get_emnist(split: str, root: str, train: bool):
    tfm = transforms.Compose([
        transforms.ToTensor(),     # -> [1, 28, 28], [0,1]
        emnist_fix_orientation(),  # 方向修正
    ])
    ds = datasets.EMNIST(root=root, split=split, train=train, download=True, transform=tfm)
    return ds

@torch.no_grad()
def dump_flat(dl: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
    feats, labels = [], []
    for xb, yb in tqdm(dl, ncols=80, desc="flatten"):
        feats.append(xb.view(xb.size(0), -1).to(torch.float32))  # 展平到 784D
        labels.append(yb.to(torch.long))
    X = torch.cat(feats, 0)
    Y = torch.cat(labels, 0)
    return X, Y

def save_pkls(out_dir: str, x_tr: torch.Tensor, y_tr: torch.Tensor, x_te: torch.Tensor, y_te: torch.Tensor):
    os.makedirs(out_dir, exist_ok=True)
    torch.save(x_tr, os.path.join(out_dir, "train_data.pkl"))
    torch.save(y_tr, os.path.join(out_dir, "train_label.pkl"))
    torch.save(x_te, os.path.join(out_dir, "test_data.pkl"))
    torch.save(y_te, os.path.join(out_dir, "test_label.pkl"))
    print(f"\n✅ saved to {out_dir}")
    print(f"  train_data: {tuple(x_tr.shape)} {x_tr.dtype}")
    print(f"  train_label: {tuple(y_tr.shape)} {y_tr.dtype}")
    print(f"  test_data:  {tuple(x_te.shape)} {x_te.dtype}")
    print(f"  test_label: {tuple(y_te.shape)} {y_te.dtype}")

def main():
    setup_seed(SEED)

    # 1) 加载 EMNIST（Balanced，47 类）
    ds_tr = get_emnist(SPLIT, DATA_ROOT, train=True)
    ds_te = get_emnist(SPLIT, DATA_ROOT, train=False)
    num_classes = int(max(ds_tr.targets.max().item(), ds_te.targets.max().item()) + 1)
    print(f"Loaded EMNIST-{SPLIT}: train={len(ds_tr)} test={len(ds_te)} classes={num_classes}")

    # 2) DataLoader
    dl_tr = DataLoader(ds_tr, batch_size=BATCH, shuffle=False, num_workers=WORKERS, pin_memory=False)
    dl_te = DataLoader(ds_te, batch_size=BATCH, shuffle=False, num_workers=WORKERS, pin_memory=False)

    # 3) 展平为 784D，不做标准化（下游训练会自己算 mu/sigma）
    x_tr, y_tr = dump_flat(dl_tr)  # float32, int64
    x_te, y_te = dump_flat(dl_te)

    # 4) 保存为 pkl（与你项目一致的四件套）
    save_pkls(OUT_DIR, x_tr, y_tr, x_te, y_te)

if __name__ == "__main__":
    main()