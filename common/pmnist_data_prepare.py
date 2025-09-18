import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm
import os

# ========== 1. 固定像素置换序列 ==========
np.random.seed(42)
permute_idx = np.random.permutation(28 * 28)


# ========== 2. 定义 Permuted MNIST Dataset（原始像素） ==========
class PermutedMNIST(Dataset):
    def __init__(self, train=True):
        self.dataset = datasets.MNIST(root="./data", train=train, download=True, transform=transforms.ToTensor())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = img.view(-1).numpy()  # 展平成 [784]
        img = img[permute_idx]  # 应用固定置换
        img = torch.tensor(img, dtype=torch.float32)  # 保持 [784]
        return img, label


# ========== 3. 提取并保存 ==========
def extract_and_save(dataset, path_prefix):
    data_loader = DataLoader(dataset, batch_size=5000, shuffle=False, num_workers=0)
    all_data = []
    all_label = []
    for x, y in tqdm(data_loader):
        all_data.append(x)
        all_label.append(y)
    data = torch.cat(all_data)
    label = torch.cat(all_label)
    torch.save(data, f'{path_prefix}_data.pkl')
    torch.save(label, f'{path_prefix}_label.pkl')


# ========== 4. 主程序入口 ==========
def main():
    os.makedirs('../data/opmnist', exist_ok=True)
    extract_and_save(PermutedMNIST(train=True), '../data/opmnist/train')
    extract_and_save(PermutedMNIST(train=False), '../data/opmnist/test')
    print("✅ 原始 Permuted MNIST 数据已保存到 ../data/opmnist/")


# ========== 5. 启动 ==========
if __name__ == "__main__":
    main()
