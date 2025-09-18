import torch
import numpy as np
from common.utils import setup_seed
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm
import os

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
setup_seed(777)

# ========== 1. 固定像素置换序列 ==========
np.random.seed(42)
permute_idx = np.random.permutation(28 * 28)


# ========== 2. 定义 Permuted MNIST Dataset ==========
class PermutedMNIST(Dataset):
    def __init__(self, train=True):
        self.dataset = datasets.MNIST(root="./data", train=train, download=True, transform=transforms.ToTensor())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = img.view(-1).numpy()  # 展平成 [784]
        img = img[permute_idx]  # 应用固定置换
        img = torch.tensor(img, dtype=torch.float32).view(1, 28, 28)  # 变回 [1, 28, 28]
        return img, label


# ========== 3. 提取特征函数 ==========
def extract_features(data_loader, model):
    all_data = []
    all_label = []
    for x, y in tqdm(data_loader):
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        with torch.no_grad():
            out = model(x)
        all_data.append(out.cpu())
        all_label.append(y.cpu())
    return torch.cat(all_data), torch.cat(all_label)


# ========== 4. 主程序入口 ==========
def main():
    # 加载 CNN 模型
    CNN_model = torch.load('../result/model/CNN_PMNIST.pkl').to(DEVICE)
    CNN_model.eval()

    # 加载 PMNIST 数据
    train_dataset = PermutedMNIST(train=True)
    test_dataset = PermutedMNIST(train=False)

    data_loader_train = DataLoader(train_dataset, batch_size=5000, shuffle=True, num_workers=2)
    data_loader_test = DataLoader(test_dataset, batch_size=5000, shuffle=True, num_workers=2)

    # 提取特征
    train_data, train_label = extract_features(data_loader_train, CNN_model)
    test_data, test_label = extract_features(data_loader_test, CNN_model)

    # 保存特征数据
    os.makedirs('../data/pmnist', exist_ok=True)
    torch.save(train_data, '../data/pmnist/train_data.pkl')
    torch.save(train_label, '../data/pmnist/train_label.pkl')
    torch.save(test_data, '../data/pmnist/test_data.pkl')
    torch.save(test_label, '../data/pmnist/test_label.pkl')

    print("✅ PMNIST CNN特征已保存到 ../data/pmnist/")


# ========== 5. 启动入口 ==========
if __name__ == "__main__":
    main()
