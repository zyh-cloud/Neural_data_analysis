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
        ipy.run_line_magic("clear", "")
        # 等价 %reset -f 强制清空所有变量
        ipy.run_line_magic("reset", "-f")
except ImportError:
    # 非IPython环境（命令行python）直接跳过
    pass
import matplotlib.pyplot as plt
plt.close('all') #关闭所有绘图窗口
# ================================

import numpy as np
import pandas as pd

df = pd.read_excel(
    r'C:\Users\ZYH\Desktop\UFE_contact_coordinate.xlsx',
    header=None  # 如果没有表头
)

x = df.iloc[:, 1]  # 第2列（B列）
y = df.iloc[:, 2]  # 第3列（C列）

x_marker = -3999.929 #以植入孔中心位置为坐标原点
y_marker = 21689.844

x_new = x - x_marker
y_new = y - y_marker

chanMap = np.array([
    61, 33, 63, 35, 62, 34, 64, 36, 30, 2, 32, 4, 29, 1, 31, 3, 
    27, 7, 25, 5, 28, 8, 26, 6, 60, 40, 58, 38, 59, 39, 57, 37, 
    53, 41, 55, 43, 54, 42, 56, 44, 22, 10, 24, 12, 21, 9, 23, 11, 
    19, 15, 17, 13, 20, 16, 18, 14, 52, 48, 50, 46, 51, 47, 49, 45
])
kcoords = np.zeros(64)
n_chan = 64

xc = x_new.values
yc = y_new.values

probe = {
    'chanMap': chanMap,
    'xc': xc,
    'yc': yc,
    'kcoords': kcoords,
    'n_chan': n_chan
}

from kilosort.io import save_probe

save_probe(probe, r'C:\Users\ZYH\.kilosort\probes\UFE64.json')

