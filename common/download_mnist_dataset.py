from torchvision import datasets, transforms

transform = transforms.ToTensor()

# 下载 Fashion-MNIST 训练集
fmnist_train = datasets.FashionMNIST(
    root="./data/fmnist", train=True, download=True, transform=transform
)

# 下载 Fashion-MNIST 测试集
fmnist_test = datasets.FashionMNIST(
    root="./data/fmnist", train=False, download=True, transform=transform
)

print("Fashion-MNIST:", len(fmnist_train), len(fmnist_test))

from torchvision import datasets, transforms

transform = transforms.ToTensor()

# 下载 EMNIST Letters 训练集
emnist_train = datasets.EMNIST(
    root="./data/emnist", split="letters", train=True, download=True, transform=transform
)

# 下载 EMNIST Letters 测试集
emnist_test = datasets.EMNIST(
    root="./data/emnist", split="letters", train=False, download=True, transform=transform
)

print("EMNIST Letters:", len(emnist_train), len(emnist_test))
