# common/mni_QFNN.py
import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F

# ===== 量子/模糊层基础配置（保持不变）=====
n_qubits = 3
n_fuzzy_mem = 2
defuzz_qubits = n_qubits
defuzz_layer = 2

dev1 = qml.device('default.qubit', wires=2 * n_qubits - 1)
dev2 = qml.device('default.qubit', wires=defuzz_qubits)

@qml.qnode(dev1, interface='torch', diff_method='backprop')
def q_tnorm_node(inputs, weights=None):
    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
    qml.Toffoli(wires=[0, 1, n_qubits])
    for i in range(n_qubits - 2):
        qml.Toffoli(wires=[i + 2, n_qubits + i, i + n_qubits + 1])
    return qml.probs(wires=2 * n_qubits - 2)

@qml.qnode(dev2, interface='torch', diff_method='backprop')
def q_defuzz(inputs, weights=None):
    qml.AmplitudeEmbedding(inputs, wires=range(defuzz_qubits), normalize=True)
    for i in range(defuzz_layer):
        for j in range(defuzz_qubits - 1):
            qml.CNOT(wires=[j, j + 1])
        qml.CNOT(wires=[defuzz_qubits - 1, 0])
        for j in range(defuzz_qubits):
            qml.RX(weights[i, 3 * j], wires=j)
            qml.RZ(weights[i, 3 * j + 1], wires=j)
            qml.RX(weights[i, 3 * j + 2], wires=j)
    return [qml.expval(qml.PauliZ(j)) for j in range(defuzz_qubits)]

weight_shapes = {"weights": (1, 1)}
defuzz_weight_shapes = {"weights": (defuzz_layer, 3 * defuzz_qubits)}

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)

class Qfnn(nn.Module):
    """
    仅做“输入维度 in_dim、类别数 num_classes 可配置”的最小改动；
    量子/模糊/前向流程保持不变。
    """
    def __init__(self, device='cpu', num_classes: int = 10, in_dim: int = 10) -> None:
        super().__init__()
        self.device = device
        self.num_classes = int(num_classes)

        # ★ 改这里：10 -> in_dim
        self.linear = nn.Linear(in_dim, n_qubits)
        self.dropout = nn.Dropout(0.5)

        self.m = nn.Parameter(torch.randn(n_qubits, n_fuzzy_mem))
        self.theta = nn.Parameter(torch.randn(n_qubits, n_fuzzy_mem))

        # ★ 改这里：10 -> num_classes
        self.softmax_linear = nn.Linear(defuzz_qubits, self.num_classes)

        self.gn = nn.GroupNorm(1, n_qubits)
        self.gn2 = nn.BatchNorm1d(n_fuzzy_mem ** n_qubits)

        self.qlayer = qml.qnn.TorchLayer(q_tnorm_node, weight_shapes)
        self.defuzz = qml.qnn.TorchLayer(q_defuzz, defuzz_weight_shapes)

        self.apply(weights_init)

    def forward(self, x):
        x = self.linear(x)
        x = self.gn(x)

        # 数值稳健：theta 加安全下限，防止除 0/极小
        eps = 1e-6
        theta_safe = self.theta.abs() + eps

        fuzzy_list0 = torch.zeros_like(x)
        fuzzy_list1 = torch.zeros_like(x)
        for i in range(x.shape[1]):
            a = (-(x[:, i] - self.m[i, 0]) ** 2) / (2 * (theta_safe[i, 0] ** 2))
            b = (-(x[:, i] - self.m[i, 1]) ** 2) / (2 * (theta_safe[i, 1] ** 2))
            fuzzy_list0[:, i] = torch.exp(a)
            fuzzy_list1[:, i] = torch.exp(b)
        fuzzy_list = torch.stack([fuzzy_list0, fuzzy_list1], dim=1)

        q_in = torch.zeros_like(x)
        q_out = []
        for i in range(n_fuzzy_mem ** n_qubits):
            loc = list(bin(i))[2:]
            if len(loc) < n_qubits:
                loc = [0] * (n_qubits - len(loc)) + loc
            for j in range(n_qubits):
                idx = 0 if loc[j] == '0' else 1
                q_in = q_in.clone()
                q_in[:, j] = fuzzy_list[:, idx, j]

            sq = torch.sqrt(q_in + 1e-16)
            sq = torch.clamp(sq, -0.99999, 0.99999)
            q_angles = 2 * torch.arcsin(sq)
            Q_tnorm_out = self.qlayer(q_angles)[:, 1]
            q_out.append(Q_tnorm_out)

        out = torch.stack(q_out, dim=1)
        out = self.gn2(out)
        defuzz_out = self.defuzz(out)
        logits = self.softmax_linear(defuzz_out)
        return logits