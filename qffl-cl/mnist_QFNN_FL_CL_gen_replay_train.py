import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import numpy as np
from common.mni_QFNN import Qfnn
from common.utils import acc_cal, setup_seed
from torch.utils.data import DataLoader, TensorDataset
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

# ======================== 参数配置 ========================
BATCH_SIZE = 600
EPOCH = 5
LR = 0.1
REPLAY_SIZE = 20
VAE_EPOCH = 3
VAE_LR = 1e-3
LATENT_DIM = 10

DEVICE = torch.device('cpu')
setup_seed(777)

# ======================== VAE定义 ========================
class VAE(nn.Module):
    def __init__(self, input_dim=10, latent_dim=LATENT_DIM):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc21 = nn.Linear(32, latent_dim)
        self.fc22 = nn.Linear(32, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 32)
        self.fc4 = nn.Linear(32, input_dim)

    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        return self.fc4(h3)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    BCE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# ======================== 数据加载 ========================
train_data = torch.load('../data/pmnist/train_data.pkl').cpu().numpy()
train_label = torch.load('../data/pmnist/train_label.pkl').cpu().numpy()
all_len = len(train_label)

# ======================== 初始化 ========================
gmm_list = []
weights = []
NAME = 'pmnist_qffl_cl_gen_replay_gas_q4_star'
node = 9
keep_list = [[0,1],[0,2],[0,3],[0,4],[0,5],[0,6],[0,7],[0,8],[0,9]]

# ======================== 每个客户端 ========================
for i in range(node):
    model = Qfnn(DEVICE).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()

    train_loss_list = []
    train_acc_list = []

    # 选择子数据集
    keep = np.isin(train_label, keep_list[i])
    data = train_data[keep]
    labels = train_label[keep]
    weights.append(len(data) / all_len)

    # GMM 拟合（用于联邦加权）
    gmm = GaussianMixture(n_components=5, max_iter=100, random_state=42)
    gmm.fit(data)
    gmm_list.append(gmm)

    # ===== 训练VAE生成器 =====
    vae = VAE(input_dim=data.shape[1], latent_dim=LATENT_DIM).to(DEVICE)
    vae_opt = torch.optim.Adam(vae.parameters(), lr=VAE_LR)
    vae.train()
    vae_dataset = torch.tensor(data, dtype=torch.float32)
    vae_loader = DataLoader(vae_dataset, batch_size=128, shuffle=True)
    for epoch in range(VAE_EPOCH):
        for x in vae_loader:
            x = x.to(DEVICE)
            vae_opt.zero_grad()
            recon_x, mu, logvar = vae(x)
            loss = vae_loss(recon_x, x, mu, logvar)
            loss.backward()
            vae_opt.step()

    # ===== 用VAE生成Replay样本 =====
    vae.eval()
    replay_x = []
    replay_y = []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        if len(idx) == 0:
            continue
        z = torch.randn(REPLAY_SIZE, LATENT_DIM).to(DEVICE)
        with torch.no_grad():
            gen_x = vae.decode(z).cpu().numpy()
        replay_x.append(gen_x)
        replay_y.append(np.full(REPLAY_SIZE, cls))
    if len(replay_x) > 0:
        replay_x = np.concatenate(replay_x, axis=0)
        replay_y = np.concatenate(replay_y, axis=0)
    else:
        replay_x = np.empty((0, data.shape[1]))
        replay_y = np.empty((0,))

    # 合并当前数据 + 回放数据
    all_x = np.concatenate([data, replay_x], axis=0)
    all_y = np.concatenate([labels, replay_y], axis=0)
    train_set = TensorDataset(torch.tensor(all_x, dtype=torch.float32), torch.tensor(all_y, dtype=torch.long))
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(EPOCH):
        print(f'===== node:{i} epoch:{epoch} =====')
        for x, y in tqdm(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            model.train()
            output = model(x)
            ce_loss = loss_func(output, y)
            total_loss = ce_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            acc = acc_cal(output, y)
            train_loss_list.append(total_loss.item())
            train_acc_list.append(acc)
            tqdm.write(f'loss: {total_loss.item():.4f} acc: {acc:.4f}')

    # 保存结果
    torch.save(model.state_dict(), f'../result/model/{NAME}_n{i}.pth')
    torch.save(train_loss_list, f'../result/data/{NAME}_train_loss_n{i}')
    torch.save(train_acc_list, f'../result/data/{NAME}_train_acc_n{i}')

# 保存 GMM 和 权重信息
torch.save(gmm_list, f'../result/data/{NAME}_gmm_list')
torch.save(weights, f'../result/data/{NAME}_data_weights') 