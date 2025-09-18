# common/emnist_data_prepare_by_cnn.py
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import argparse

from common.CNN import CNN
from common.utils import setup_seed

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
setup_seed(777)

def extract_features(dataloader: DataLoader, model: torch.nn.Module):
    model.eval()
    feats, labels = [], []
    with torch.no_grad():
        for x, y in tqdm(dataloader, ncols=80, desc="Extracting"):
            x = x.to(DEVICE)
            out = model(x)   # 期望输出 [B, 10]
            feats.append(out.detach().cpu())
            labels.append(y.detach().cpu())
    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)
    return feats, labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="./data/oemnist",
                        help="原始 EMNIST 数据目录（包含 raw/）")
    parser.add_argument("--out_dir", type=str, default="./data/emnist",
                        help="输出目录")
    parser.add_argument("--split", type=str, default="letters",
                        help="emnist split: letters/byclass/balanced/digits/bymerge/mnist")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--cnn_ckpt", type=str, default=None,
                        help="CNN 权重路径，可选")
    args = parser.parse_args()

    transform = transforms.ToTensor()
    train_ds = datasets.EMNIST(root=args.root, split=args.split,
                               train=True, download=True, transform=transform)
    test_ds = datasets.EMNIST(root=args.root, split=args.split,
                              train=False, download=True, transform=transform)

    # ⚠️ 注意：EMNIST Letters 标签默认是 1..26，需要减 1 → 0..25
    if args.split == "letters":
        train_ds.targets = train_ds.targets - 1
        test_ds.targets = test_ds.targets - 1

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = CNN().to(DEVICE)
    if args.cnn_ckpt and os.path.isfile(args.cnn_ckpt):
        state = torch.load(args.cnn_ckpt, map_location="cpu")
        try:
            model.load_state_dict(state, strict=False)
        except Exception:
            if isinstance(state, dict) and 'model' in state:
                model.load_state_dict(state['model'], strict=False)
            else:
                raise

    train_data, train_label = extract_features(train_loader, model)
    test_data, test_label = extract_features(test_loader, model)

    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(train_data, os.path.join(args.out_dir, "train_data.pkl"))
    torch.save(train_label, os.path.join(args.out_dir, "train_label.pkl"))
    torch.save(test_data, os.path.join(args.out_dir, "test_data.pkl"))
    torch.save(test_label, os.path.join(args.out_dir, "test_label.pkl"))

    print(f"✅ EMNIST({args.split}) 特征已保存到 {args.out_dir}")

if __name__ == "__main__":
    main()