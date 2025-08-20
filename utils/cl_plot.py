# -*- coding: utf-8 -*-
import os, re, textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ================= 数据 =================
data = {
    "structure": [
        "BaseLine", "ER", "EWC", "LwF", "ER + EWC+LwF)",
        "Adapter", "GenReplay", "L2", "MAS", "PathInt",
        "SI", "ER+EWC + Adapter+LwF)", "ER + EWC+L2+LwF",
        "ER + MAS+LwF", "ER+PathInt + LwF", "ER+SI+Adapter+LwF",
        "ER + SI+LwF", "ER+SI+MAS+LwF", "GenReplay+EWC+LwF"
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

# =============== 样式参数 ===============
OUT_DIR = "figs_cl_bars"
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.35,
    "figure.dpi": 240
})

COLOR_NORM = "#cfd8dc"   # 浅灰蓝
COLOR_MAX  = "#2ecc71"   # 绿色
COLOR_MIN  = "#e74c3c"   # 红色
EDGE_COLOR = "white"

# —— 标签精简：去掉 “”，合并空格、去掉多余括号中的空格 —— #
def shorten_label(s: str) -> str:
    if s.strip() == "QFFL":
        return "QFFL"
    s2 = s.replace("", "")
    s2 = re.sub(r"\s*\+\s*", "+", s2)       # ER + EWC + LwF -> ER+EWC+LwF
    s2 = re.sub(r"\(\s*", "(", s2)
    s2 = re.sub(r"\s*\)", ")", s2)
    # 去掉多余括号： (ER+EWC+LwF) -> ER+EWC+LwF
    s2 = s2[1:-1] if s2.startswith("(") and s2.endswith(")") else s2
    return s2

def annotate_bars(ax, rects, values):
    ymin, ymax = ax.get_ylim()
    for r, v in zip(rects, values):
        x = r.get_x() + r.get_width()/2
        y = r.get_height()
        # 顶部太近时向下挪一点，避免溢出
        dy = 0.012 if y < ymax - 0.05 else -0.030
        ax.annotate(f"{v:.4f}", (x, y), xytext=(0, dy),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=9, color="#222222")

def plot_metric_vertical(metric_name, savepath, angle=35):
    vals   = df[metric_name].values
    labels = [shorten_label(s) for s in df["structure"].tolist()]
    x = np.arange(len(labels))

    max_idx = int(np.nanargmax(vals))
    min_idx = int(np.nanargmin(vals))
    colors = [COLOR_NORM] * len(vals)
    colors[max_idx] = COLOR_MAX
    colors[min_idx] = COLOR_MIN

    fig, ax = plt.subplots(figsize=(16, 6.2))
    rects = ax.bar(x, vals, color=colors, edgecolor=EDGE_COLOR, linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=angle, ha="right", fontsize=10)
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel(metric_name.capitalize())
    ax.set_title(f"{metric_name.capitalize()} per Structure")
    ax.margins(x=0.03)

    annotate_bars(ax, rects, vals)

    # 更大底部留白以容纳斜体标签
    plt.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.33)
    fig.savefig(savepath, bbox_inches="tight")
    plt.close(fig)

# —— 四张单图 —— #
for m in ["accuracy", "precision", "recall", "f1"]:
    out = os.path.join(OUT_DIR, f"cl_bar_{m}_slanted.png")
    plot_metric_vertical(m, out, angle=35)
    print("[OK]", out)

# —— 2×2 概览（同样斜体标签、加留白） —— #
fig, axes = plt.subplots(2, 2, figsize=(18, 10))
axes = axes.ravel()
labels = [shorten_label(s) for s in df["structure"].tolist()]
x = np.arange(len(labels))

for ax, m in zip(axes, ["accuracy", "precision", "recall", "f1"]):
    vals = df[m].values
    max_idx = int(np.nanargmax(vals))
    min_idx = int(np.nanargmin(vals))
    colors = [COLOR_NORM] * len(vals)
    colors[max_idx] = COLOR_MAX
    colors[min_idx] = COLOR_MIN

    rects = ax.bar(x, vals, color=colors, edgecolor=EDGE_COLOR, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel(m.capitalize())
    ax.set_title(m.capitalize())
    ax.margins(x=0.03)

    # 简洁数值（保留三位小数），避免太密
    for r, v in zip(rects, vals):
        x0 = r.get_x() + r.get_width()/2
        y0 = r.get_height()
        dy = 0.010 if y0 < 0.95 else -0.028
        ax.annotate(f"{v:.3f}", (x0, y0), xytext=(0, dy),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=8, color="#222222")

fig.suptitle("CL variants — Bar overview (2×2, slanted labels)", y=0.99, fontsize=13)
plt.subplots_adjust(left=0.05, right=0.99, top=0.93, bottom=0.25, wspace=0.18, hspace=0.35)
out = os.path.join(OUT_DIR, "cl_bar_overview_2x2_slanted.png")
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print("[OK]", out)