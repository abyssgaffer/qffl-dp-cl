import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from CNN import CNN
from utils import get_logger, setup_seed, acc_cal
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm
import os

# ========== 设置 ==========
BATCH_SIZE = 5000
EPOCH = 50
LR = 0.001
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
logger = get_logger('CNN_PMNIST_train')
setup_seed(777)

# ========== 固定像素置换 ==========
permute_idx = np.random.RandomState(seed=42).permutation(28 * 28)

# ========== 自定义数据集：PMNIST ==========
class PermutedMNIST(Dataset):
    def __init__(self, train=True):
        self.dataset = datasets.MNIST(root="../data/opmnist", train=train, download=True, transform=transforms.ToTensor())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = img.view(-1).numpy()                   # 展平成 784
        img = img[permute_idx]                       # 打乱像素
        img = torch.tensor(img, dtype=torch.float32).view(1, 28, 28)  # 再 reshape 成图片格式
        return img, label

# ========== 加载数据 ==========
data_train = PermutedMNIST(train=True)
data_test = PermutedMNIST(train=False)

data_loader_train = DataLoader(dataset=data_train, batch_size=BATCH_SIZE, shuffle=True)
data_loader_test = DataLoader(dataset=data_test, batch_size=BATCH_SIZE, shuffle=False)

# ========== 初始化模型 ==========
model = CNN().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

# ========== 训练 ==========
for epoch in range(EPOCH):
    model.train()
    for its, (x, y) in enumerate(tqdm(data_loader_train)):
        x, y = x.to(DEVICE), y.to(DEVICE)
        output = model(x)
        loss = loss_func(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    logger.info(f'epoch:{epoch} train_loss:{loss.item()}')
    torch.save(model, f'../result/model/CNN_PMNIST_{epoch}.pkl')

    # ====== 测试 ======
    model.eval()
    with torch.no_grad():
        total, correct = 0, 0
        for x, y in data_loader_test:
            x, y = x.to(DEVICE), y.to(DEVICE)
            output = model(x)
            pred = torch.argmax(output, dim=1)
            total += y.size(0)
            correct += (pred == y).sum().item()
        acc = correct / total
        logger.info(f'epoch:{epoch} test_acc:{acc}')