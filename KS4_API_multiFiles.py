# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 15:47:14 2026

@author: ZYH
"""
# %reset -f

# Loading other data formats with SpikeInterface -- kilosort4

from kilosort import run_kilosort
from kilosort import io
from pathlib import Path
import numpy as np
import os
import time

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
file_name = '20260630_220345'
n_chan = 128
fs = 30000
data_dir = Path('C:\\Users\\ZYH\\Desktop\\UFE64\\spike_20260630')
probe_path = r"C:\Users\ZYH\.kilosort\probes\UFE128.json"
assert os.path.exists(probe_path), f"Probe file not found: {probe_path}"
probe = io.load_probe(probe_path)  # Specify probe configuration

# 进入到该文件夹中
os.chdir(data_dir)
# 总输出目录：当前目录下的kilosort文件夹，如没有就创建
output_root = data_dir / "kilosort"
output_root.mkdir(exist_ok=True) 

bin_files = [
    Path(f"{file_name}_HF8_1.bin"),
    Path(f"{file_name}_HF8_2.bin"),
    Path(f"{file_name}_HF8_3.bin"),
    Path(f"{file_name}_HF24_1.bin"),
    Path(f"{file_name}_HF24_2.bin"),
    Path(f"{file_name}_HF24_3.bin")
]

# ==========批量计算所有bin采样点数【关键新增】==========
n_samples = [get_n_samples(f, n_chan, np.int16) for f in bin_files]
print("\n===== Bin文件时长校验 =====")
for f, ns in zip(bin_files, n_samples):
    dur_min = ns / fs / 60
    print(f"{f.name}: samples={ns}, duration={dur_min:.2f} min")
print("================================\n")
# =====================================================

# 每个文件对应一个独立的输出子文件夹
results_dir = output_root / file_name
print(f"[开始处理] {file_name}\n")

# 3.调参-------------------------------------------------------------------- 
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
            'batch_size': 30000*20,
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

# 4.调用API运行KS4--------------------------------------------------------------------
# This command will both run the spike-sorting analysis and save the results to `data_dir`.
ops, st, clu, tF, Wall, similar_templates, is_ref, \
    est_contam_rate, kept_spikes = run_kilosort(
        settings=settings,
        probe=probe,
        probe_name=None,
        filename=bin_files,
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
print(f"[完成] {file_name}")

end_time = time.perf_counter()
cost_sec = end_time - start_time
print(f"[{file_name}]总耗时：{cost_sec:.2f} 秒，折合时长：{cost_sec / 60:.2f} 分钟")
     
# 输出标记：是否多文件拼接
np.save(results_dir / "is_multiflag.npy", np.array([1], dtype=np.int8))
np.save(results_dir / "N_bin.npy", np.array(n_samples, dtype=np.int64))
print(f"\n标记文件已保存至：{results_dir}")




