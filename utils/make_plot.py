import pandas as pd
import matplotlib.pyplot as plt

# 数据（原始顺序保留）
data = {
    "structure": [
        "QFFL", "QFFL + ER", "QFFL + EWC", "QFFL + LwF", "QFFL + (ER + EWC + LwF)",
        "QFFL + Adapter", "QFFL + GenReplay", "QFFL + L2", "QFFL + MAS", "QFFL + PathInt",
        "QFFL + SI", "QFFL + (ER + EWC + Adapter + LwF)", "QFFL + (ER + EWC + L2 + LwF)",
        "QFFL + (ER + MAS + LwF)", "QFFL + (ER + PathInt + LwF)", "QFFL + (ER + SI + Adapter + LwF)",
        "QFFL + (ER + SI + LwF)", "QFFL + (ER + SI + MAS + LwF)", "QFFL + (GenReplay + EWC + LwF)"
    ],
    "accuracy": [
        0.8845, 0.8845, 0.844, 0.8435, 0.8895,
        0.884, 0.8845, 0.0975, 0.8845, 0.8845,
        0.8845, 0.7915, 0.0975, 0.842, 0.842,
        0.961, 0.842, 0.842, 0.8585
    ],
    "precision": [
        0.8069, 0.8309, 0.7692, 0.7897, 0.9084,
        0.8289, 0.8253, 0.0098, 0.8069, 0.8069,
        0.8069, 0.7988, 0.0098, 0.8385, 0.8385,
        0.9618, 0.8385, 0.8385, 0.8768
    ],
    "recall": [
        0.8819, 0.8818, 0.8371, 0.8395, 0.8884,
        0.8814, 0.8819, 0.1, 0.8819, 0.8819,
        0.8819, 0.7897, 0.1, 0.8368, 0.8368,
        0.9618, 0.8368, 0.8368, 0.8592
    ],
    "f1": [
        0.8393, 0.8474, 0.7978, 0.8075, 0.8813,
        0.8463, 0.8452, 0.0178, 0.8393, 0.8393,
        0.8393, 0.7442, 0.0178, 0.8075, 0.8075,
        0.9609, 0.8075, 0.8075, 0.858
    ]
}

df = pd.DataFrame(data)

# 绘图函数（突出最大和最小值）
def plot_metric(metric_name):
    values = df[metric_name]
    max_idx = values.idxmax()
    min_idx = values.idxmin()

    colors = ['gray'] * len(df)
    colors[max_idx] = 'green'  # 最高值绿色
    colors[min_idx] = 'red'    # 最低值红色

    plt.figure(figsize=(12, 6))
    bars = plt.barh(df["structure"], values, color=colors)
    plt.xlabel(metric_name.capitalize())
    plt.title(f'{metric_name.capitalize()} per Structure')

    # 标注数值
    for i, bar in enumerate(bars):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                 f'{values[i]:.4f}', va='center', fontsize=9)

    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

# 依次绘图
for metric in ["accuracy", "precision", "recall", "f1"]:
    plot_metric(metric)