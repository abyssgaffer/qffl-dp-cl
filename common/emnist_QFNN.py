# common/mni_QFNN.py  （或新建 common/mni_QFNN_emnist47.py）
import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F

# ===== 量子/模糊层基本结构（保持不变）=====
n_qubits = 3
n_fuzzy_mem = 2
defuzz_qubits = n_qubits
defuzz_layer = 2

# 量子设备
dev1 = qml.device('default.qubit', wires=2 * n_qubits - 1)
dev2 = qml.device('default.qubit', wires=defuzz_qubits)

@qml.qnode(dev1, interface='torch', diff_method='backprop')
def q_tnorm_node(inputs, weights=None):
    # inputs: [n_qubits] 角编码
    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
    qml.Toffoli(wires=[0, 1, n_qubits])
    for i in range(n_qubits - 2):
        qml.Toffoli(wires=[i + 2, n_qubits + i, i + n_qubits + 1])
    # 保持你原来的返回
    return qml.probs(wires=2 * n_qubits - 2)

@qml.qnode(dev2, interface='torch', diff_method='backprop')
def q_defuzz(inputs, weights=None):
    # inputs: 长度应为 2**defuzz_qubits（本模型里是 8）
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

# 与 TorchLayer 对接的形状定义（保持不变）
weight_shapes = {"weights": (1, 1)}
defuzz_weight_shapes = {"weights": (defuzz_layer, 3 * defuzz_qubits)}

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)

class Qfnn(nn.Module):
    """
    量子联邦模糊推理前端（QFNN）
    - 仅做“分类头维度可配置”的最小改动：num_classes 默认 10，可设为 47
    - 其余结构/前向流程/量子电路保持不变
    """
    def __init__(self, device: str = 'cpu', num_classes: int = 10) -> None:
        super().__init__()
        self.device = device
        self.num_classes = int(num_classes)

        # 线性降维到 n_qubits
        self.linear = nn.Linear(10, n_qubits)
        self.dropout = nn.Dropout(0.5)

        # 模糊隶属度参数（均值 m、尺度 theta）
        self.m = nn.Parameter(torch.randn(n_qubits, n_fuzzy_mem))
        self.theta = nn.Parameter(torch.randn(n_qubits, n_fuzzy_mem))

        # 读出头（仅此处：把 10 改为 num_classes）
        self.softmax_linear = nn.Linear(defuzz_qubits, self.num_classes)

        # 归一化层（保持不变）
        self.gn = nn.GroupNorm(1, n_qubits)
        self.gn2 = nn.BatchNorm1d(n_fuzzy_mem ** n_qubits)

        # PennyLane 封装的量子层
        self.qlayer = qml.qnn.TorchLayer(q_tnorm_node, weight_shapes)
        self.defuzz = qml.qnn.TorchLayer(q_defuzz, defuzz_weight_shapes)

        # 线性层权重初始化（参数 nn.Parameter 不会被 .apply 遍历，这是 PyTorch 的常规行为）
        self.apply(weights_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 10]  —— 你的特征维度
        返回: [B, num_classes]  —— EMNIST47 时设 num_classes=47
        """
        device = self.device
        x = self.linear(x)        # [B, n_qubits]
        x = self.gn(x)

        # ===== 模糊隶属度（最小数值修补：theta 加安全下限，避免除零）=====
        eps = 1e-6
        theta_safe = self.theta.abs() + eps  # 防止 0 或过小
        fuzzy_list0 = torch.zeros_like(x, device=x.device)
        fuzzy_list1 = torch.zeros_like(x, device=x.device)

        for i in range(x.shape[1]):
            a = (-(x[:, i] - self.m[i, 0]) ** 2) / (2 * (theta_safe[i, 0] ** 2))
            b = (-(x[:, i] - self.m[i, 1]) ** 2) / (2 * (theta_safe[i, 1] ** 2))
            fuzzy_list0[:, i] = torch.exp(a)
            fuzzy_list1[:, i] = torch.exp(b)

        fuzzy_list = torch.stack([fuzzy_list0, fuzzy_list1], dim=1)  # [B, 2, n_qubits]

        # ===== 构造量子输入并做 T-norm 组合（保持不变）=====
        q_in = torch.zeros_like(x)  # [B, n_qubits]
        q_out = []
        for i in range(n_fuzzy_mem ** n_qubits):
            loc = list(bin(i))[2:]
            if len(loc) < n_qubits:
                loc = [0] * (n_qubits - len(loc)) + loc
            for j in range(n_qubits):
                # 注意：loc[j] 是 '0'/'1' 字符；转成 int 再索引
                idx = 0 if loc[j] == '0' else 1
                q_in = q_in.clone()
                q_in[:, j] = fuzzy_list[:, idx, j]

            # 角度预处理（与你原逻辑一致）
            sq = torch.sqrt(q_in + 1e-16)
            sq = torch.clamp(sq, -0.99999, 0.99999)
            q_angles = 2 * torch.arcsin(sq)

            Q_tnorm_out = self.qlayer(q_angles)[:, 1]  # 取概率向量的第 1 位（保持不变）
            q_out.append(Q_tnorm_out)

        out = torch.stack(q_out, dim=1)    # [B, 2**n_qubits] 这里是 [B, 8]
        out = self.gn2(out)                # BN1d over 8

        # defuzz 量子层（保持不变）
        defuzz_out = self.defuzz(out)      # [B, defuzz_qubits] 这里是 [B, 3]

        # 分类头（仅维度可变）
        logits = self.softmax_linear(defuzz_out)  # [B, num_classes]
        return logits