# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 15:47:14 2026

@author: ZYH
"""
# %reset -f

# Loading other data formats with SpikeInterface -- kilosort4
import scipy.io
from matplotlib import gridspec, rcParams
import matplotlib.pyplot as plt
from kilosort.io import load_ops
import pandas as pd
from kilosort import run_kilosort
from kilosort import io
from pathlib import Path
import numpy as np
import os
import time
import traceback

start_time_total = time.perf_counter()

# =====================新增函数：计算单个bin采样点数=====================
def get_n_samples(bin_path: Path, n_chan: int, dtype=np.int16) -> int:
    """
    计算单个bin文件每个通道总采样数量
    二进制格式：[n_channels, n_samples], int16(2字节)
    """
    bytes_per_sample = np.dtype(dtype).itemsize * n_chan
    total_bytes = bin_path.stat().st_size
    n_samp = total_bytes // bytes_per_sample
    return n_samp
# =====================================================================

# 1.配置文件--------------------------------------------------------------------
# 原始数据文件所在的文件夹
n_chan = 128
fs = 30000
data_dir = Path('C:\\Users\\ZYH\\Desktop\\UFE64\\spikedata_dogs')
probe_path = r"C:\Users\ZYH\.kilosort\probes\UFE128.json"
assert os.path.exists(probe_path), f"Probe file not found: {probe_path}"
probe = io.load_probe(probe_path)  # Specify probe configuration
skip_existing = True  # 已处理过的文件自动跳过
# 遍历文件夹内所有原始数据文件
data_files = sorted(data_dir.glob("*.bin"))
print(f"找到 {len(data_files)} 个待处理文件：")
# 进入到该文件夹中
os.chdir(data_dir)
# 总输出目录：当前目录下的kilosort文件夹，如没有就创建
output_root = data_dir / "kilosort"
output_root.mkdir(exist_ok=True) 

success_count = 0
fail_count = 0
failed_files = []

#------------------------------------------------------------------------------------
for data_file in data_files[:]:
    # data_file = data_files[0]
    file_stem = data_file.stem  # 提取不带后缀的文件名，如 "20260601_220345"
    # 每个文件对应一个独立的输出子文件夹
    results_dir = output_root / file_stem
    # 断点续跑：判断是否已处理完成（以ops.npy为标志）
    if skip_existing and (results_dir / "ops.npy").exists():
        print(f"[跳过] {data_file.name} 已处理过")
        success_count += 1
        continue
    
    results_dir.mkdir(exist_ok=True)
    print(f"[开始处理] {data_file.name}\n")
    n_samples  = get_n_samples(data_file, n_chan, np.int16)
    dur_min = n_samples  / fs / 60
    
    print("\n===== Bin文件时长校验 =====")
    print(f"{data_file.name}: samples={n_samples }, duration={dur_min:.2f} min")
    print("================================\n")

    try:
        # 2.调参-------------------------------------------------------------------- 
        start_time = time.perf_counter()
        # NOTE: 'n_chan_bin' is a required setting, and should reflect the total number
        #       of channels in the binary file, while probe['n_chans'] should reflect
        #       the number of channels that contain ephys data. In many cases these will
        #       be the same, but not always. For example, neuropixels data often contains
        #       385 channels, where 384 channels are for ephys traces and 1 channel is
        #       for some other variable. In that case, you would specify
        #       'n_chan_bin': 385.
        settings = {'n_chan_bin': n_chan,
                    'fs': fs,
                    'batch_size': 300000,
                    'nblocks': 0,
                    'Th_universal': 8,
                    'Th_learned': 8,
                    'dmin': 90, #触点的垂直距离
                    'dminx': 112, #触点的水平距离
                    'min_template_size': 50, #默认值 10 μm 是为 Neuropixels（20μm 间距）优化的，对应有效宽度 20~30 μm，刚好覆盖 1~2 个相邻触点。
                    'nearest_chans': 6,
                    'nearest_templates': 12,
                    'x_centers': 2,
                    'ccg_threshold': 0.25,
                    'n_pcs': 8, #默认6
                    #'cluster_neighbors': 8,  # 默认10
                    #'cluster_downsampling': 5,
                    #'max_cluster_subset': 50000
                    }
        
        # 3.调用API运行KS4--------------------------------------------------------------------
        # This command will both run the spike-sorting analysis and save the results to
        # `data_dir`.
        ops, st, clu, tF, Wall, similar_templates, is_ref, \
            est_contam_rate, kept_spikes = run_kilosort(
                settings=settings,
                probe=probe,
                probe_name=None,
                filename=data_file,
                data_dir=None,
                file_object=None,
                results_dir=results_dir,     # 指定独立输出目录
                data_dtype=np.int16,
                do_CAR=True,
                invert_sign=False,
                device=None,
                progress_bar=None,
                save_extra_vars=False,
                clear_cache=False,
                save_preprocessed_copy=True, #保留预处理后的数据
                bad_channels=None,
                shank_idx=None, #  shank_idx=[0, 1]  # 自动依次处理shank0和shank1
                verbose_console=False,
                verbose_log=False,
                torch_thread_lim=None
            )
        print(f"[完成] {data_file.name}")
        success_count += 1
    except Exception as e:
        print(f"[失败] {data_file.name} 处理出错：{str(e)}")
        traceback.print_exc()
        fail_count += 1
        failed_files.append(data_file.name)
        continue
    # 输出标记：是否多文件拼接
    np.save(results_dir / "is_multiflag.npy", np.array([0], dtype=np.int8))
    np.save(results_dir / "N_bin.npy", np.array(n_samples, dtype=np.int64))
    print(f"\n标记文件已保存至：{results_dir}")
    # 记录结束时间并计算
    end_time = time.perf_counter()
    cost_sec = end_time - start_time
    print(f"[{data_file.name}]总耗时：{cost_sec:.2f} 秒，折合时长：{cost_sec / 60:.2f} 分钟")
    
# 4.汇总统计--------------------------------------------------------------------    
print(f"\n{'='*50}")
print(f"批处理结束：成功 {success_count} 个，失败 {fail_count} 个")
if failed_files:
    print("失败文件列表：")
    for f in failed_files:
        print(f"  - {f}")

end_time_total = time.perf_counter()
cost_sec_total = end_time_total - start_time_total
print(f"总耗时：{cost_sec_total:.2f} 秒，折合时长：{cost_sec_total / 60:.2f} 分钟")



# %% Load outputs 导入结果
# outputs saved to results_dir
# results_dir = Path('C:/Alldata/test_data/kilosort4') #自定义想查看的结果
ops = load_ops(results_dir / 'ops.npy')
camps = pd.read_csv(results_dir / 'cluster_Amplitude.tsv',
                    sep='\t')['Amplitude'].values
contam_pct = pd.read_csv(
    results_dir / 'cluster_ContamPct.tsv', sep='\t')['ContamPct'].values
chan_map = np.load(results_dir / 'channel_map.npy')
templates = np.load(results_dir / 'templates.npy')
chan_best = (templates**2).sum(axis=1).argmax(axis=-1)
chan_best = chan_map[chan_best]
amplitudes = np.load(results_dir / 'amplitudes.npy')
st = np.load(results_dir / 'spike_times.npy')
clu = np.load(results_dir / 'spike_clusters.npy')
firing_rates = np.unique(clu, return_counts=True)[1] * 30000 / st.max()
dshift = ops['dshift']


# %% Plot outputs 画图查看
# %matplotlib inline
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False
gray = .5 * np.ones(3)

fig = plt.figure(figsize=(10, 10), dpi=100)
grid = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.5)

ax = fig.add_subplot(grid[0, 0])
ax.plot(np.arange(0, ops['Nbatches'])*2, dshift)
ax.set_xlabel('time (sec.)')
ax.set_ylabel('drift (um)')

ax = fig.add_subplot(grid[0, 1:])
t0 = 0
t1 = np.nonzero(st > ops['fs']*5)[0][0]
ax.scatter(st[t0:t1]/30000., chan_best[clu[t0:t1]],
           s=0.5, color='k', alpha=0.25)
ax.set_xlim([0, 5])
ax.set_ylim([chan_map.max(), 0])
ax.set_xlabel('time (sec.)')
ax.set_ylabel('channel')
ax.set_title('spikes from units')

ax = fig.add_subplot(grid[1, 0])
nb = ax.hist(firing_rates, 20, color=gray)
ax.set_xlabel('firing rate (Hz)')
ax.set_ylabel('# of units')

ax = fig.add_subplot(grid[1, 1])
nb = ax.hist(camps, 20, color=gray)
ax.set_xlabel('amplitude')
ax.set_ylabel('# of units')

ax = fig.add_subplot(grid[1, 2])
nb = ax.hist(np.minimum(100, contam_pct), np.arange(0, 105, 5), color=gray)
ax.plot([10, 10], [0, nb[0].max()], 'k--')
ax.set_xlabel('% contamination')
ax.set_ylabel('# of units')
ax.set_title('< 10% = good units')

for k in range(2):
    ax = fig.add_subplot(grid[2, k])
    is_ref = contam_pct < 10.
    ax.scatter(firing_rates[~is_ref], camps[~is_ref],
               s=3, color='r', label='mua', alpha=0.25)
    ax.scatter(firing_rates[is_ref], camps[is_ref],
               s=3, color='b', label='good', alpha=0.25)
    ax.set_ylabel('amplitude (a.u.)')
    ax.set_xlabel('firing rate (Hz)')
    ax.legend()
    if k == 1:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title('loglog')


probe = ops['probe']
# x and y position of probe sites
xc, yc = probe['xc'], probe['yc']
nc = 16  # number of channels to show
good_units = np.nonzero(contam_pct <= 0.1)[0]
mua_units = np.nonzero(contam_pct > 0.1)[0]


gstr = ['good', 'mua']
for j in range(2):
    print(f'~~~~~~~~~~~~~~ {gstr[j]} units ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('title = number of spikes from each unit')
    units = good_units if j == 0 else mua_units
    fig = plt.figure(figsize=(12, 3), dpi=150)
    grid = gridspec.GridSpec(2, 20, figure=fig, hspace=0.25, wspace=0.5)

    for k in range(40):
        wi = units[np.random.randint(len(units))]
        wv = templates[wi].copy()
        cb = chan_best[wi]
        nsp = (clu == wi).sum()

        ax = fig.add_subplot(grid[k//20, k % 20])
        n_chan = wv.shape[-1]
        ic0 = max(0, cb-nc//2)
        ic1 = min(n_chan, cb+nc//2)
        wv = wv[:, ic0:ic1]
        x0, y0 = xc[ic0:ic1], yc[ic0:ic1]

        amp = 4
        for ii, (xi, yi) in enumerate(zip(x0, y0)):
            t = np.arange(-wv.shape[0]//2, wv.shape[0]//2, 1, 'float32')
            t /= wv.shape[0] / 20
            ax.plot(xi + t, yi + wv[:, ii]*amp, lw=0.5, color='k')

        ax.set_title(f'{nsp}', fontsize='small')
        ax.axis('off')
    plt.show()


# %% 将Kilosort输出目录中的ops.npy转存为ops.mat 适用于KS2.5输出的结果转换
# AutoCurationKilosort已更新到KS4，请忽略！

os.chdir(r'C:\Alldata\test_data\kilosort4')

ops = np.load('ops.npy', allow_pickle=True).item()


def clean_value(v):
    '''把单个值转成 scipy.io.savemat 能正确存储的类型'''
    if v is None:
        return np.nan
    elif isinstance(v, Path):
        return str(v)
    elif isinstance(v, dict):
        # 递归处理嵌套 dict
        cleaned = {}
        for kk, vv in v.items():
            new_kk = kk[:31] if len(kk) > 31 else kk
            cleaned[new_kk] = clean_value(vv)
        return cleaned
    elif isinstance(v, (list, tuple)):
        # 列表里的元素逐一处理
        result = []
        for item in v:
            converted = clean_value(item)
            result.append(converted)
        return result
    elif isinstance(v, np.ndarray):
        # 整数类型数组整体转 float64
        if np.issubdtype(v.dtype, np.integer):
            return v.astype(np.float64)
        return v
    elif isinstance(v, (np.integer, int)):
        # numpy整数 + Python原生int 统一转 float64 标量
        return np.float64(v)
    elif isinstance(v, (np.floating,)):
        return float(v)
    elif isinstance(v, (int, float, str)):
        return v
    else:
        return str(v)


# 顶层 dict 也处理
ops_clean = {}
for k, v in ops.items():
    new_k = k[:31] if len(k) > 31 else k
    ops_clean[new_k] = clean_value(v)

scipy.io.savemat('ops.mat', {'ops': ops_clean})
print('Done! ops.mat saved.')
