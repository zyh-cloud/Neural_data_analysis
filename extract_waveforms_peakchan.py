# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 22:50:01 2026

@author: zhouy
"""
# %%
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

# %%
import numpy as np
from pathlib import Path
from kilosort.data_tools import get_spike_waveforms
import warnings
# 只屏蔽 kilosort.io 模块的所有运行时警告，不影响其他正常警告
# 警告文字先说 numpy.int16 不支持，下方列表又写了 int16 在支持列表里，这是 Kilosort 内部代码的文案 bug，不是真的不兼容 int16 数据
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    module="kilosort.io"
)

# %% 参数配置 
res_dir = Path("C:\\Alldata\\test_data\\kilosort4")        # Phy校正后的结果目录
batch_size = 5000                           # 计算峰值通道时，每批全通道spike数量（控制内存）
save_file = "all_units_peakchan_waves.npz"  # 输出文件名


# %% 加载基础数据（Phy/AutoCurationKilosor​t校正后）
spike_times = np.load(res_dir / "spike_times.npy").flatten()
spike_clusters = np.load(res_dir / "spike_clusters.npy").flatten()
cluster_ids = np.unique(spike_clusters)

cluster_peakchan = {}       # key: cluster ID, value: 峰值通道
all_unit_waves = {}         # key: cluster_ID, value: 峰值通道波形 (nt, n_spikes)


# %% 全量计算每个单元的峰值通道
#    分批全通道提取 → 累加求平均 → 计算峰峰值最大通道
for cid in cluster_ids:
    unit_spike_t = spike_times[spike_clusters == cid]
    n_spikes = len(unit_spike_t)
    if n_spikes == 0:
        continue

    # 第一批获取维度信息（自动适配：spike数少于batch_size则取全部）
    first_batch = unit_spike_t[:batch_size]
    first_waves = get_spike_waveforms(spikes=first_batch, results_dir=res_dir)
    n_channels, nt, _ = first_waves.shape
    # 初始化累加器与有效计数
    sum_template = np.zeros((n_channels, nt), dtype=np.float64)
    total_valid = 0

    # 累加第一批
    sum_template += first_waves.sum(axis=2)
    total_valid += first_waves.shape[2]

    # 处理剩余批次：若n_spikes <= batch_size，range为空，循环自动不执行
    for i in range(batch_size, n_spikes, batch_size):
        batch_t = unit_spike_t[i:i+batch_size]
        batch_waves = get_spike_waveforms(spikes=batch_t, results_dir=res_dir)  #(n_chan, nt, n_spikes)
        sum_template += batch_waves.sum(axis=2)
        total_valid += batch_waves.shape[2]

    # 计算平均模板与峰值通道
    mean_template = sum_template / total_valid
    peak_chan = np.argmax(np.ptp(mean_template, axis=1))
    cluster_peakchan[int(cid)] = peak_chan

    print(f"单元 {cid:4d} | 峰值通道 {peak_chan:3d} | 有效spike: {total_valid:5d}")
    
# %% 提取每个单元峰值通道的全部spike波形
#    单通道提取，内存占用极低，直接一次性提取即可
for cid in cluster_ids:

    peak_chan = cluster_peakchan[cid]
    unit_spike_t = spike_times[spike_clusters == cid]

    # 提取峰值通道上的所有spike波形 (nt, n_spikes)
    waves = get_spike_waveforms(
        spikes=unit_spike_t,
        results_dir=res_dir,
        chan=peak_chan
    )
    
    all_unit_waves[f"cluster_{cid}"] = waves

# %% 统一保存到 npz 文件
save_path = res_dir / save_file
np.savez(
    save_path,
    **all_unit_waves,                          # 所有单元的峰值通道波形
    cluster_ids = np.array(list(cluster_peakchan.keys())),   # 单元ID列表
    peak_channels = np.array(list(cluster_peakchan.values()))# 对应峰值通道列表
)

print(f"\n全部处理完成，结果已保存至：{save_path}")




