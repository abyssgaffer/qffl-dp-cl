import torch
from tqdm import tqdm
from common.mni_QFNN import Qfnn
from common.utils import setup_seed

DEVICE = torch.device('cpu')
NAME = 'fmnist_qffl_gas_q4_star'
setup_seed(777)
node = 9
# #测试
test_data = torch.load('../data/fmnist/test_data.pkl').to(DEVICE)[:2000]
label = torch.load('../data/fmnist/test_label.pkl').to(DEVICE)[:2000]

gmm_list = torch.load(f'../result/data/{NAME}_gmm_list', weights_only=False)
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
    m = out_put[:, i, :]
    n = gmm_scores[:, i].unsqueeze(1)
    out_put[:, i, :] = out_put[:, i, :] * gmm_scores[:, i].unsqueeze(1)
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
