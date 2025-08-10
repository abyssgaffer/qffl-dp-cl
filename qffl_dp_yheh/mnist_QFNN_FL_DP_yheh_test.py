import torch
from tqdm import tqdm
from common.mni_QFNN import Qfnn
from common.utils import setup_seed
from torch.distributions import Laplace

def add_laplace_noise(x: torch.Tensor, eps: float, sensitivity: float = 1.0) -> torch.Tensor:
    """
    Adds element-wise Laplace noise to tensor x for ε-DP.

    Args:
        x (torch.Tensor): Input tensor to privatize.
        eps (float): Privacy budget ε.
        sensitivity (float): Sensitivity Δ of the query function (default 1.0).

    Returns:
        torch.Tensor: Noisy tensor of same shape as x.
    """
    scale = sensitivity / eps
    lap = Laplace(loc=0.0, scale=scale)
    noise = lap.sample(x.shape).to(x.device)
    return x + noise


DEVICE = torch.device('cpu')
NAME = 'pmnist_qffl_dp_yheh_gas_q4_star'
# eps=1
# acc:0.121 precision:0.11691733263151508 recall:0.12081292196448248 f1:0.11785194121095435
# eps=10
# acc:0.232 precision:0.22196847426095112 recall:0.23218641174915405 f1:0.22432163455734452
# eps=100
# acc:0.8175 precision:0.7362794400066356 recall:0.8144183070508284 f1:0.7723118139034343

setup_seed(777)
node = 9
# #测试
test_data = torch.load('../data/pmnist/test_data.pkl').to(DEVICE)[:2000]
label = torch.load('../data/pmnist/test_label.pkl').to(DEVICE)[:2000]

gmm_list = torch.load(f'../result/data/{NAME}_gmm_list')
data_weights = torch.load(f'../result/data/{NAME}_data_weights')
gmm_scores = []
for i in range(node):
    gmm_scores.append(gmm_list[i].score_samples(test_data.cpu().numpy()))

gmm_scores = torch.tensor(gmm_scores).to(DEVICE).permute(1, 0)

for i in range(node):
    gmm_scores[:, i] = gmm_scores[:, i] * data_weights[i]
sum = torch.sum(gmm_scores, dim=1)
for i in range(node):
    gmm_scores[:, i] = gmm_scores[:, i] / sum

out_put = []

for i in tqdm(range(node)):
    model = Qfnn(DEVICE).to(DEVICE)
    model.load_state_dict(torch.load(f'../result/model/{NAME}_n{i}.pth'))
    model.eval()
    with torch.no_grad():
        out_put.append(model(test_data))

out_put = torch.stack(out_put, dim=1)
out_put = torch.softmax(out_put, dim=1)
for i in range(node):
    y_h = out_put[:, i, :]
    e_h = gmm_scores[:, i].unsqueeze(1)
    # y_h = add_laplace_noise(y_h, eps=10, sensitivity=0.95)
    e_h = add_laplace_noise(e_h, eps=1, sensitivity=0.95)
    out_put[:, i, :] = y_h * e_h
output = torch.sum(out_put, dim=1)
pred = torch.argmin(output, dim=1)

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

pred = pred.cpu().numpy()
label = label.cpu().numpy()
acc = accuracy_score(label, pred)
precision = precision_score(label, pred, average='macro')
recall = recall_score(label, pred, average='macro')
f1 = f1_score(label, pred, average='macro')
print(f'acc:{acc} precision:{precision} recall:{recall} f1:{f1}')
cm = confusion_matrix(label, pred)
