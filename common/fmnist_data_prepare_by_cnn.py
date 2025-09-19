import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from CNN import CNN
from common.utils import setup_seed

import argparse

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
setup_seed(777)


def extract_features(dataloader: DataLoader, model: torch.nn.Module):
    """
    与 pmnist_data_prepare_by_cnn.py 同风格的特征抽取：
    - 输入：DataLoader (x: [B,1,28,28], y)
    - 输出：两个 tensor，分别是拼接后的 features 与 labels（都在 CPU）
    约定：CNN 的输出为 [B, 10]（与你 pmnist 的 10 维特征保持一致）
    """
    model.eval()
    feats, labels = [], []
    with torch.no_grad():
        for x, y in tqdm(dataloader, ncols=80, desc="Extracting"):
            x = x.to(DEVICE)
            out = model(x)  # 期望输出 [B, 10]
            feats.append(out.detach().cpu())
            labels.append(y.detach().cpu())
    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)
    return feats, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="./data", help="torchvision 下载/读取根目录")
    parser.add_argument("--out_dir", type=str, default="../data/fmnist", help="输出目录（与 pmnist 对齐）")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--cnn_ckpt", type=str, default=None, help="可选：CNN 权重路径（建议提供）")
    args = parser.parse_args()

    # 1) 准备 Fashion-MNIST 数据（与 pmnist 的 ToTensor 风格一致；如你在 pmnist 用了 Normalize，这里也可加上）
    transform = transforms.ToTensor()
    train_ds = datasets.FashionMNIST(root=args.root, train=True, download=True, transform=transform)
    test_ds = datasets.FashionMNIST(root=args.root, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    # 2) 准备 CNN（与 pmnist 保持一致）
    model = CNN().to(DEVICE)
    if args.cnn_ckpt is not None and os.path.isfile(args.cnn_ckpt):
        state = torch.load(args.cnn_ckpt, map_location="cpu")
        # 视你的 CNN 保存格式决定是否需要 state['model']；保持宽松以避免键不匹配
        try:
            model.load_state_dict(state, strict=False)
        except Exception:
            if isinstance(state, dict) and 'model' in state:
                model.load_state_dict(state['model'], strict=False)
            else:
                raise

    # 3) 抽取特征
    train_data, train_label = extract_features(train_loader, model)
    test_data, test_label = extract_features(test_loader, model)

    # 4) 保存到与 pmnist 完全一致的结构/文件名
    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(train_data, os.path.join(args.out_dir, "train_data.pkl"))
    torch.save(train_label, os.path.join(args.out_dir, "train_label.pkl"))
    torch.save(test_data, os.path.join(args.out_dir, "test_data.pkl"))
    torch.save(test_label, os.path.join(args.out_dir, "test_label.pkl"))

    print(f"✅ FMNIST CNN特征已保存到 {args.out_dir}")


if __name__ == "__main__":
    main()
