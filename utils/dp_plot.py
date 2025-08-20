# -*- coding: utf-8 -*-
"""
更鲁棒的版本：自动从 Excel 推断 Structure / DP type / ε
输出三套图：
  1) figs_ch6                         — 按结构&指标，三条机制曲线（x=ε对数）+ baseline 水平线
  2) figs_ch6_cmp_struct_PACKED       — 按机制各一张 2×2 面板图，对比两种 Structure
  3) figs_ch6_combo_single_bar        — 按 (Structure×DP) 各一张，四指标分组柱状图
如果某结构没有任何带 ε 的 DP 行，会回退画 baseline 柱状图（避免空图）。
同时打印数据覆盖情况，便于排查“为什么没画出来”。
"""

import os, re, sys, pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ========= 修改这里 =========
XLSX_PATH = "/Users/jon/MyInfos/um/semester2/7023-research-project/QFFL+DP+CL_Result.xlsx"
# ===========================

OUT_DIR_A = "figs_ch6"
OUT_DIR_B = "figs_ch6_cmp_struct_PACKED"
OUT_DIR_C = "figs_ch6_combo_single_bar"

FIG_DPI   = 240
FONT_SIZE = 12
METRICS   = ["Acc", "precision", "recall", "f1"]
METRIC_TITLES = {"Acc": "Accuracy", "precision": "Precision", "recall": "Recall", "f1": "F1"}

LINE_COLORS  = {"QFFL+DP": "#1f77b4", "QFFL+DP+CL": "#ff7f0e"}
LINE_MARKERS = {"QFFL+DP": "o",        "QFFL+DP+CL": "s"}

# 分组柱状图：四指标不同纹理
METRIC_STYLE = {
    "Acc":       dict(hatch=""),
    "precision": dict(hatch=""),
    "recall":    dict(hatch=""),
    "f1":        dict(hatch=""),
}
ANNOTATE_METRICS = ["Acc", "f1"]  # 只标注这两个，避免太密

plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.35,
})


# ---------- 工具 ----------
def ensure_dir(p):
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)

def normalize_str(x):
    if pd.isna(x):
        return ""
    return str(x).strip()

def unify_structure_name(s):
    """容错合并：QFFL+CL+DP → QFFL+DP+CL；其它保持原样"""
    s = normalize_str(s)
    s = s.replace(" ", "")
    s_upper = s.upper()
    if s_upper in {"QFFL+CL+DP", "QFFL+DP+CL"}:
        return "QFFL+DP+CL"
    return s

def unify_dp_type(x):
    """容错合并：none/null/空 → none；大小写兼容"""
    s = normalize_str(x).lower()
    if s in {"", "none", "null", "nan"}:
        return "none"
    # 容错：quantum gaussian 的多写法
    s = s.replace("qrng-gaussian", "quantum").replace("qrng", "quantum")
    return s

def parse_epsilon(s):
    """从 'ε=20.0' / 'epsilon=20' / 'eps=20' / 'e=20' 提取数值"""
    if pd.isna(s):
        return np.nan
    txt = str(s)
    m = re.search(r'(?:ε|epsilon|eps|e)\s*=\s*([0-9]+(?:\.[0-9]+)?)', txt, flags=re.IGNORECASE)
    return float(m.group(1)) if m else np.nan

def normalize_df(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    # 标准化三列
    if "Structure" in df.columns:
        df["Structure"] = df["Structure"].apply(unify_structure_name)
    if "DP type" in df.columns:
        df["DP type"] = df["DP type"].apply(unify_dp_type)
    if "DP param" in df.columns:
        df["epsilon"] = df["DP param"].apply(parse_epsilon)
    else:
        df["epsilon"] = np.nan
    # 指标转数值
    for m in METRICS:
        if m in df.columns:
            df[m] = pd.to_numeric(df[m], errors="coerce")
    return df

def jitter(y, frac=0.004):
    y = np.array(y, dtype=float)
    if len(y) == 0:
        return y
    span = np.nanmax(y) - np.nanmin(y)
    if not np.isfinite(span) or span == 0:
        span = 1.0
    return y + (np.random.rand(len(y)) - 0.5) * 2 * frac * span

def format_eps(e):
    if pd.isna(e):
        return "None"
    if e >= 1000:
        return f"{int(e/1000)}k"
    if abs(e - int(e)) < 1e-9:
        return f"{int(e)}"
    return f"{e:g}"

def get_baseline_value(df, structure, metric):
    """优先取该 Structure 的 none；若 Structure=QFFL+DP 找不到，则回退用 QFFL 的 none。"""
    sub = df[(df["Structure"] == structure) & (df["DP type"] == "none")]
    if not sub.empty:
        return float(sub.iloc[0][metric])
    if structure == "QFFL+DP":
        sub2 = df[(df["Structure"].str.upper() == "QFFL") & (df["DP type"] == "none")]
        if not sub2.empty:
            return float(sub2.iloc[0][metric])
    return None


# ---------- 数据分布打印（排障用） ----------
def print_coverage(df):
    print("\n===== Data coverage (what the script sees) =====")
    structs = df["Structure"].dropna().unique().tolist()
    dps     = df["DP type"].dropna().unique().tolist()
    print("Structures:", structs)
    print("DP types  :", dps)
    for st in structs:
        for dp in dps:
            cnt = len(df[(df["Structure"] == st) & (df["DP type"] == dp)])
            eps_cnt = len(df[(df["Structure"] == st) & (df["DP type"] == dp) & (df["epsilon"].notna())])
            print(f"  - {st:12s} | {dp:8s}  rows={cnt:3d}, with ε={eps_cnt:3d}")
    print("===============================================")


# ---------- 套 1：figs_ch6（若某结构没有任何 ε，则退化画 baseline 柱状图） ----------
def plot_figs_ch6(df, out_dir):
    ensure_dir(out_dir)
    structures = [s for s in df["Structure"].dropna().unique().tolist()
                  if s.upper() in {"QFFL+DP", "QFFL+DP+CL"}]
    dp_types   = [d for d in df["DP type"].dropna().unique().tolist() if d != "none"]
    if not structures:
        print("[WARN] 未发现 QFFL+DP / QFFL+DP+CL 结构，figs_ch6 跳过。")
        return

    for st in structures:
        # 判断该结构是否至少有一条带 ε 的 DP 数据
        has_dp_eps = not df[(df["Structure"] == st) & (df["DP type"] != "none") & (df["epsilon"].notna())].empty
        if not has_dp_eps:
            # ——退化：只画 baseline 的四指标柱状图（你截图中的那张）
            base_row = df[(df["Structure"] == st) & (df["DP type"] == "none")]
            if base_row.empty and st == "QFFL+DP":
                base_row = df[(df["Structure"].str.upper() == "QFFL") & (df["DP type"] == "none")]
            if base_row.empty:
                print(f"[WARN] {st} 无任何可画数据。")
                continue
            base_row = base_row.iloc[0]
            vals = [base_row[m] for m in METRICS]
            labels = [METRIC_TITLES.get(m, m) for m in METRICS]
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.bar(labels, vals, color="#1f77b4", alpha=0.85, label=f"{st} (no DP)")
            for i, v in enumerate(vals):
                if pd.notna(v):
                    ax.annotate(f"{v:.3f}", xy=(i, v), xytext=(0, 6),
                                textcoords="offset points", ha="center", va="bottom", fontsize=FONT_SIZE-2)
            ax.set_ylim(0, 1.0)
            ax.set_ylabel("Score")
            ax.set_title(f"Baseline (No DP): {st}")
            ax.legend(loc="best", frameon=True)
            fig.tight_layout()
            out = os.path.join(out_dir, f"{st.replace('+','_')}_baseline_bar.png")
            fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight"); plt.close(fig)
            print("[OK]", out)
            continue

        # ——正常：每个指标一张图，三条机制曲线
        eps_all = sorted(df[(df["Structure"] == st) & (df["epsilon"].notna())]["epsilon"].unique())
        for metric in METRICS:
            fig, ax = plt.subplots(figsize=(8, 5))
            base_val = get_baseline_value(df, st, metric)
            if base_val is not None and np.isfinite(base_val):
                ax.axhline(base_val, ls="--", lw=1.2, c="#7f7f7f",
                           label=f"Baseline (DP=None) = {base_val:.3f}")
            for dp in dp_types:
                sub = df[(df["Structure"] == st) & (df["DP type"] == dp) & (df["epsilon"].notna())].copy()
                if sub.empty:
                    continue
                sub = sub.sort_values("epsilon")
                x = sub["epsilon"].to_numpy(float)
                y = sub[metric].to_numpy(float)
                ax.plot(x, jitter(y, 0.003), marker="o", lw=1.8, ms=5, label=dp.capitalize())
                # 标注两端
                for pick in [0, -1]:
                    if np.isfinite(y[pick]):
                        ax.annotate(f"{y[pick]:.3f}",
                                    xy=(x[pick], y[pick]),
                                    xytext=(0, 6), textcoords="offset points",
                                    ha="center", va="bottom", fontsize=FONT_SIZE-2)
            ax.set_xscale("log")
            ax.set_xticks(eps_all)
            ax.set_xticklabels([format_eps(e) for e in eps_all])
            ax.set_ylim(0, 1.0)
            ax.set_xlabel("ε (log scale)"); ax.set_ylabel("Metric")
            ax.set_title(f"{st} — {METRIC_TITLES.get(metric, metric)}")
            ax.legend(loc="best", frameon=True)
            fig.tight_layout()
            out = os.path.join(out_dir, f"{st.replace('+','_')}_{metric}.png")
            fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight"); plt.close(fig)
            print("[OK]", out)


# ---------- 套 2：每个 DP 机制一张 2×2 面板图，对比两种 Structure ----------
def plot_struct_compare_per_dp(df, out_dir):
    ensure_dir(out_dir)
    dp_types = [d for d in df["DP type"].dropna().unique().tolist() if d != "none"]
    if not dp_types:
        print("[WARN] 没有任何 DP 行（含 ε），cmp_struct_PACKED 跳过。")
        return

    eps_global = sorted(df[df["epsilon"].notna()]["epsilon"].unique())
    for dp in dp_types:
        sub = df[(df["DP type"] == dp) & (df["epsilon"].notna())]
        if sub.empty:
            continue
        eps_all = sorted(sub["epsilon"].unique())
        fig, axes = plt.subplots(2, 2, figsize=(12, 7)); axes = axes.ravel()
        for idx, metric in enumerate(METRICS):
            ax = axes[idx]
            for st in ["QFFL+DP", "QFFL+DP+CL"]:
                df_st = sub[sub["Structure"] == st].copy()
                if df_st.empty:
                    continue
                df_st = df_st.sort_values("epsilon")
                x = df_st["epsilon"].to_numpy(float)
                y = df_st[metric].to_numpy(float)
                ax.plot(x, jitter(y, 0.003),
                        marker=("o" if st=="QFFL+DP" else "s"),
                        lw=1.8, ms=5,
                        color=LINE_COLORS.get(st, None),
                        label=st)
                for pick in [0, -1]:
                    if np.isfinite(y[pick]):
                        ax.annotate(f"{y[pick]:.3f}",
                                    xy=(x[pick], y[pick]), xytext=(0, 6),
                                    textcoords="offset points",
                                    ha="center", va="bottom", fontsize=FONT_SIZE-2)
            ax.set_xscale("log")
            ax.set_xticks(eps_all)
            ax.set_xticklabels([format_eps(e) for e in eps_all])
            ax.set_ylim(0, 1.0)
            ax.set_title(METRIC_TITLES.get(metric, metric))
            ax.set_xlabel("ε (log scale)"); ax.set_ylabel("Metric")
            ax.grid(True, linestyle="--", alpha=0.35)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncols=2, frameon=True)
        fig.suptitle(f"{dp.capitalize()} DP — Structure Comparison", y=1.02, fontsize=FONT_SIZE+2)
        fig.tight_layout()
        out = os.path.join(out_dir, f"cmp_struct_{dp}.png")
        fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight"); plt.close(fig)
        print("[OK]", out)


# ---------- 套 3：每个 (Structure×DP) 一张，四指标分组柱状 ----------
def add_bar_labels(ax, rects, values, fontsize=10):
    for i, (r, v) in enumerate(zip(rects, values)):
        if v is None or not np.isfinite(v): continue
        h = r.get_height()
        dy = 6 if (i % 2 == 0) else -8
        ax.annotate(f"{v:.3f}",
                    xy=(r.get_x() + r.get_width()/2, h),
                    xytext=(0, dy),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=fontsize)

def grouped_bar(ax, x_labels, metric_to_values, bar_width=0.18):
    metrics  = list(metric_to_values.keys())
    n_groups = len(x_labels)
    n_metrics = len(metrics)
    x = np.arange(n_groups)
    offsets = np.linspace(-(n_metrics-1)/2, (n_metrics-1)/2, n_metrics) * (bar_width*1.1)
    rects_all = {}
    for idx, m in enumerate(metrics):
        ys = metric_to_values[m]
        style = METRIC_STYLE.get(m, {})
        rects = ax.bar(x + offsets[idx], ys, width=bar_width,
                       label=METRIC_TITLES.get(m, m), **style)
        rects_all[m] = rects
    ax.set_xticks(x); ax.set_xticklabels(x_labels)
    return rects_all

def plot_combo_bar(df, out_dir):
    ensure_dir(out_dir)
    structures = [s for s in df["Structure"].dropna().unique().tolist()
                  if s.upper() in {"QFFL+DP", "QFFL+DP+CL"}]
    dp_types   = [d for d in df["DP type"].dropna().unique().tolist() if d != "none"]

    for st in structures:
        for dp in dp_types:
            sub = df[(df["Structure"] == st) & (df["DP type"] == dp) & (df["epsilon"].notna())]
            if sub.empty:
                print(f"[WARN] 无数据：{st} / {dp}")
                continue
            sub = sub.sort_values("epsilon")
            eps = sub["epsilon"].tolist()
            x_labels = [format_eps(e) for e in eps]
            metric_to_values = {m: sub[m].tolist() for m in METRICS}

            fig, ax = plt.subplots(figsize=(10, 5.5))
            rects_all = grouped_bar(ax, x_labels, metric_to_values, bar_width=0.18)
            for m in ANNOTATE_METRICS:
                add_bar_labels(ax, rects_all[m], metric_to_values[m], fontsize=FONT_SIZE-2)

            ax.set_ylim(0.0, 1.0)
            ax.set_xlabel("ε"); ax.set_ylabel("Metric")
            ax.set_title(f"{st} — {dp.capitalize()} DP (Grouped Bars)")
            ax.legend(loc="best", frameon=True)
            ax.grid(True, axis="y", linestyle="--", alpha=0.35)
            fig.tight_layout()
            out = os.path.join(out_dir, f"{st.replace('+','_')}_{dp}.png")
            fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight"); plt.close(fig)
            print("[OK]", out)


# ---------- 主程 ----------
def main(xlsx_path):
    df = pd.read_excel(xlsx_path)
    df = normalize_df(df)
    print_coverage(df)
    plot_figs_ch6(df, OUT_DIR_A)
    plot_struct_compare_per_dp(df, OUT_DIR_B)
    plot_combo_bar(df, OUT_DIR_C)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        XLSX_PATH = sys.argv[1]
    main(XLSX_PATH)