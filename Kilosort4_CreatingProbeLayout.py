# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 15:31:16 2026

@author: ZYH
"""
# Creating a Kilosort4 probe dictionary
# 'chanMap': the channel indices that are included in the data.
# 'xc':      the x-coordinates (in micrometers) of the probe contact centers.
# 'yc':      the y-coordinates (in micrometers) of the probe contact centers.
# 'kcoords': shank or channel group of each contact.
# 'n_chan':  the number of channels.

# ========== Clean Start ==========
try:
    from IPython import get_ipython
    ipy = get_ipython()
    if ipy is not None:
        # 等价 %clear
        ipy.run_line_magic("clear", "-f")
        # 等价 %reset -f 强制清空所有变量
        ipy.run_line_magic("reset", "-f")
except ImportError:
    # 非IPython环境（命令行python）直接跳过
    pass
import matplotlib.pyplot as plt
plt.close('all') #关闭所有绘图窗口

# %% make 64通道的UFE
import numpy as np
import pandas as pd
from kilosort.io import save_probe
import matplotlib.pyplot as plt

df = pd.read_excel(
    r'C:\Users\ZYH\Desktop\UFE64\UFE_contact_coordinate.xlsx',
    header=None  # 如果没有表头
)

x = df.iloc[:64, 1]  # 第2列（B列）
y = df.iloc[:64, 2]  # 第3列（C列）

x_marker = -3999.929 #以植入孔中心位置为坐标原点
y_marker = 21689.844

x_new = x - x_marker
y_new = y - y_marker

chanMap64 = np.array(df.iloc[:64, 3])
kcoords = np.zeros(64)
n_chan = 64

xc0 = x_new.values
yc0 = y_new.values

probe0 = {
    'chanMap': chanMap64,
    'xc': xc0,
    'yc': yc0,
    'kcoords': kcoords,
    'n_chan': n_chan
}

save_probe(probe0, r'C:\Users\ZYH\.kilosort\probes\UFE64.json')

# %% make 128通道的UFE
chanMap128 = np.array(df.iloc[:128, 3])

kcoords = np.concatenate([
    np.zeros(64),
    np.ones(64)
]).astype(np.float32)

n_chan = 128

xc1 = np.concatenate([xc0, xc0 + 1000])
yc1 = np.concatenate([yc0, yc0])

probe1 = {
    'chanMap': chanMap128,
    'xc': xc1,
    'yc': yc1,
    'kcoords': kcoords,
    'n_chan': n_chan
}

save_probe(probe1, r'C:\Users\ZYH\.kilosort\probes\UFE128.json')

# %%  ========== 探针可视化绘图函数 ==========
def plot_probe(probe):
    x = probe["xc"]
    y = probe["yc"]
    shank_ids = probe["kcoords"]

    plt.figure(figsize=(8, 5), dpi=120)
    # 按shank分颜色绘制电极点
    unique_shanks = np.unique(shank_ids)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i, s in enumerate(unique_shanks):
        mask = shank_ids == s
        plt.scatter(x[mask], y[mask], c=colors[i], label=f"Shank {int(s)}", s=30)

    plt.xlabel("X coordinate (μm)")
    plt.ylabel("Y coordinate (μm)")
    plt.title("Probe Layout (colored by shank / kcoords)")
    plt.legend()
    plt.axis("equal")  # 等比例坐标轴，不会拉伸探针形状
    plt.grid(alpha=0.3)
    plt.show()

# 执行绘图
plot_probe(probe0)
plot_probe(probe1)









