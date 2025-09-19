import pennylane as qml

import torch
import torch.nn as nn

n_qubits = 3

n_fuzzy_mem = 2
defuzz_qubits = n_qubits
defuzz_layer = 2

dev1 = qml.device('default.qubit', wires=2 * n_qubits - 1)


@qml.qnode(dev1, interface='torch', diff_method='backprop')
def q_tnorm_node(inputs, weights=None):
    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
    qml.Toffoli(wires=[0, 1, n_qubits])
    for i in range(n_qubits - 2):
        qml.Toffoli(wires=[i + 2, n_qubits + i, i + n_qubits + 1])
    return qml.probs(wires=2 * n_qubits - 2)


dev2 = qml.device('default.qubit', wires=defuzz_qubits)


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
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif isinstance(m, torch.nn.Parameter):
        torch.nn.init.normal_(m.data)


class Qfnn(nn.Module):
    def __init__(self, device, use_adapter=True) -> None:
        super(Qfnn, self).__init__()
        self.device = device
        self.use_adapter = use_adapter
        self.adapter_in = nn.Linear(10, 10) if use_adapter else nn.Identity()
        self.linear = nn.Linear(10, n_qubits)
        self.m = nn.Parameter(torch.randn(n_qubits, n_fuzzy_mem))
        self.theta = nn.Parameter(torch.randn(n_qubits, n_fuzzy_mem))
        self.softmax_linear = nn.Linear(defuzz_qubits, 10)
        self.gn = nn.GroupNorm(1, n_qubits)
        self.gn2 = nn.BatchNorm1d(n_fuzzy_mem ** n_qubits)
        self.adapter_out = nn.Linear(10, 10) if use_adapter else nn.Identity()
        self.apply(weights_init)

        self.qlayer = qml.qnn.TorchLayer(q_tnorm_node, weight_shapes)
        self.defuzz = qml.qnn.TorchLayer(q_defuzz, defuzz_weight_shapes)

    def forward(self, x):
        device = self.device
        x = self.adapter_in(x)
        x = self.linear(x)
        x = self.gn(x)
        fuzzy_list0 = torch.zeros_like(x).to(device)
        fuzzy_list1 = torch.zeros_like(x).to(device)
        for i in range(x.shape[1]):
            a = (-(x[:, i] - self.m[i, 0]) ** 2) / (2 * self.theta[i, 0] ** 2)
            b = (-(x[:, i] - self.m[i, 1]) ** 2) / (2 * self.theta[i, 1] ** 2)
            fuzzy_list0[:, i] = torch.exp(a)
            fuzzy_list1[:, i] = torch.exp(b)

        fuzzy_list = torch.stack([fuzzy_list0, fuzzy_list1], dim=1)

        q_in = torch.zeros_like(x).to(device)
        q_out = []
        for i in range(n_fuzzy_mem ** n_qubits):
            loc = list(bin(i))[2:]
            if len(loc) < n_qubits:
                loc = [0] * (n_qubits - len(loc)) + loc
            for j in range(n_qubits):
                q_in = q_in.clone()
                q_in[:, j] = fuzzy_list[:, int(loc[j]), j]

            sq = torch.sqrt(q_in + 1e-16)
            sq = torch.clamp(sq, -0.99999, 0.99999)
            q_in = 2 * torch.arcsin(sq)
            Q_tnorm_out = self.qlayer(q_in)[:, 1]
            q_out.append(Q_tnorm_out)
        out = torch.stack(q_out, dim=1)
        out = self.gn2(out)
        defuzz_out = self.defuzz(out)
        out = self.softmax_linear(defuzz_out)
        out = self.adapter_out(out)
        return out
