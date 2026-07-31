# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 15:47:14 2026

@author: ZYH
"""
# %reset -f

# Loading other data formats with SpikeInterface -- kilosort4


from kilosort import io
from pathlib import Path
import numpy as np
import os
from spikeinterface.extractors import read_blackrock
import time


start_time_total = time.perf_counter()

# 1.配置文件--------------------------------------------------------------------
# 原始数据文件所在的文件夹
data_dir = Path('C:\\Users\\ZYH\\Desktop\\UFE64\\spike_20260630')
probe_path = r"C:\Users\ZYH\.kilosort\probes\UFE128.json"
assert os.path.exists(probe_path), f"Probe file not found: {probe_path}"
probe = io.load_probe(probe_path)  # Specify probe configuration

# 遍历文件夹内所有原始数据文件
data_files = sorted(data_dir.glob("*.ns6"))
print(f"找到 {len(data_files)} 个待处理文件：")
# 进入到该文件夹中
os.chdir(data_dir)


#------------------------------------------------------------------------------------
for data_file in data_files[:]:
    # data_file = data_files[0]
    file_stem = data_file.stem  # 提取不带后缀的文件名，如 "20260601_220345"
    print(f"[开始处理] {data_file.name}\n")
    
    # 2.转换为二进制数据格式------------------------------------------------------------
    # Load existing data with spikeinterface
    # NOTE: You may need to specify additional keyword arguments for
    #       `read_nwb_recording`, such as `electrical_series_name`. Any required
    #       arguments should be clearly spelled out by an error message.
    #通过文件后缀自动匹配 nsx_to_load—— 你传 .ns6 就只加载 ns6，传 .ns5 就只加载 ns5，无需额外参数。
    #读取的是原始Raw Counts（原始 ADC 整数），需要乘以 Blackrock 的增益系数（约0.25）才是 µV
    recording = read_blackrock(data_file) 
    data_name = file_stem + ".bin"
    
    # NOTE: Data will be saved as np.int16 by default since that is the standard
    #       for ephys data. If you need a different data type for whatever reason
    #       such as `np.uint16`, be sure to update this.
    #转化后的.bin数据是Raw Counts（原始 ADC 整数），非uV
    filename, N, c, s, fs, probe_path = io.spikeinterface_to_binary(
        recording, data_dir, data_name=data_name, dtype=np.int16,
        chunksize=180000, export_probe=True, probe_name='probe.prb'
    )     

end_time_total = time.perf_counter()
cost_sec_total = end_time_total - start_time_total
print(f"总耗时：{cost_sec_total:.2f} 秒，折合时长：{cost_sec_total / 60:.2f} 分钟")



