import torch
from tqdm import tqdm
# from common.mni_QFNN import Qfnn
from common.mni_QFNN_adapter import Qfnn
from common.utils import setup_seed

DEVICE = torch.device('cpu')
# NAME = 'pmnist_qffl_cl_er_gas_q4_star'
# acc:0.8845 precision:0.8308508760440197 recall:0.881848228232888 f1:0.8474213330301545

# NAME = 'pmnist_qffl_cl_ewc_gas_q4_star'
# acc:0.844 precision:0.7691543535194488 recall:0.8371446424450035 f1:0.7977755079685818

# NAME = 'pmnist_qffl_cl_lwf_gas_q4_star'
# acc:0.8435 precision:0.7896707925659099 recall:0.8395408353248035 f1:0.8074503619964556

# NAME = 'pmnist_qffl_cl_er_ewc_lwf_gas_q4_star'
# acc:0.8895 precision:0.9084482649822562 recall:0.8884384144661862 f1:0.8813330660073329

# NAME = 'pmnist_qffl_cl_adapter_gas_q4_star'
# 这里需要更改上面引用的模型
# acc:0.884 precision:0.8289181178895019 recall:0.8813946513196946 f1:0.8462814124306925

# NAME = 'pmnist_qffl_cl_gen_replay_gas_q4_star'
# acc:0.8845 precision:0.8253112814118392 recall:0.8818971638825086 f1:0.8452099982493232

# NAME = 'pmnist_qffl_cl_l2_gas_q4_star'
# acc:0.0975 precision:0.00975 recall:0.1 f1:0.01776765375854214

# NAME = 'pmnist_qffl_cl_mas_gas_q4_star'
# acc:0.8845 precision:0.8068673732733325 recall:0.8818742760045011 f1:0.839300606664952

# NAME = 'pmnist_qffl_cl_pathint_gas_q4_star'
# acc:0.8845 precision:0.8068673732733325 recall:0.8818742760045011 f1:0.839300606664952

# NAME = 'pmnist_qffl_cl_si_gas_q4_star'
# acc:0.8845 precision:0.8068673732733325 recall:0.8818742760045011 f1:0.839300606664952

# NAME = 'pmnist_qffl_cl_er_ewc_adapter_lwf_gas_q4_star'
# 这里需要更改上面引用的模型
# acc:0.7915 precision:0.7988167432961062 recall:0.7897164134621029 f1:0.7441718042111889

# NAME = 'pmnist_qffl_cl_er_ewc_l2_lwf_gas_q4_star'
# acc:0.0975 precision:0.00975 recall:0.1 f1:0.01776765375854214

# NAME = 'pmnist_qffl_cl_er_mas_lwf_gas_q4_star'
# acc:0.842 precision:0.8384756290216167 recall:0.8367542095074112 f1:0.8075232615039909

# NAME = 'pmnist_qffl_cl_er_pathint_lwf_gas_q4_star'
# acc:0.842 precision:0.8384756290216167 recall:0.8367542095074112 f1:0.8075232615039909

NAME = 'pmnist_qffl_cl_er_si_adapter_lwf_gas_q4_star'
# 这里需要更改上面引用的模型
# acc:0.961 precision:0.9617563380261693 recall:0.9618491009102541 f1:0.9609408153310909

# NAME = 'pmnist_qffl_cl_er_si_lwf_gas_q4_star'
# acc:0.842 precision:0.8384756290216167 recall:0.8367542095074112 f1:0.8075232615039909

# NAME = 'pmnist_qffl_cl_er_si_mas_lwf_gas_q4_star'
# acc:0.842 precision:0.8384756290216167 recall:0.8367542095074112 f1:0.8075232615039909

# NAME = 'pmnist_qffl_cl_gen_replay_ewc_lwf_gas_q4_star'
# acc:0.8585 precision:0.8767650454651086 recall:0.8592136664464387 f1:0.8579910228898117

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
