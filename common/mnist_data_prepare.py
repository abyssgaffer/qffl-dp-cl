import torch
from common.utils import setup_seed
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import os


def extract_raw_images(data_loader):
    all_data = []
    all_label = []
    for x, y in tqdm(data_loader):
        x = x.view(x.size(0), -1)  # Flatten to 784
        all_data.append(x)
        all_label.append(y)
    return torch.cat(all_data), torch.cat(all_label)


if __name__ == "__main__":
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    setup_seed(777)

    transform = transforms.Compose([transforms.ToTensor()])

    data_train = datasets.MNIST(root="../data/mnist/train",
                                transform=transform,
                                train=True,
                                download=True)

    data_test = datasets.MNIST(root="../data/mnist/test",
                               transform=transform,
                               train=False,
                               download=True)

    data_loader_train = DataLoader(dataset=data_train,
                                   batch_size=5000,
                                   shuffle=True,
                                   num_workers=0)  # ✅ 单线程

    data_loader_test = DataLoader(dataset=data_test,
                                  batch_size=5000,
                                  shuffle=True,
                                  num_workers=0)

    train_data, train_label = extract_raw_images(data_loader_train)
    test_data, test_label = extract_raw_images(data_loader_test)

    os.makedirs("../data/omnist", exist_ok=True)
    torch.save(train_data, "../data/omnist/train_data.pkl")
    torch.save(train_label, "../data/omnist/train_label.pkl")
    torch.save(test_data, "../data/omnist/test_data.pkl")
    torch.save(test_label, "../data/omnist/test_label.pkl")

    print("✅ 原始 MNIST 数据已保存到 omnist 文件夹！")
