import os
import scipy.io.wavfile as wavfile
from scipy.signal import resample
import numpy as np

root_dir = './custom_character_voice'

# 备份原文件（可选）
for subdir, _, files in os.walk(root_dir):
    for file in files:
        if file.lower().endswith('.wav'):
            old_path = os.path.join(subdir, file)
            new_path = old_path.replace('.wav', '_backup.wav')
            os.rename(old_path, new_path)

# 重采样
for subdir, _, files in os.walk(root_dir):
    for file in files:
        if file.lower().endswith('_backup.wav'):  # 处理备份文件
            file_path = os.path.join(subdir, file)
            print(f"处理文件: {file_path}")
            
            try:
                original_rate, data = wavfile.read(file_path)
                print(f"  原采样率: {original_rate} Hz")
                
                if original_rate == 22050:
                    print(f"  已为 22050 Hz，跳过")
                    # 恢复原名
                    new_file = file.replace('_backup.wav', '.wav')
                    os.rename(file_path, os.path.join(subdir, new_file))
                    continue
                
                # 计算新长度
                new_length = int(len(data) * 22050 / original_rate)
                
                # 重采样（处理多声道）
                if len(data.shape) > 1:
                    resampled_data = np.zeros((new_length, data.shape[1]))
                    for ch in range(data.shape[1]):
                        resampled_data[:, ch] = resample(data[:, ch], new_length)
                else:
                    resampled_data = resample(data, new_length)
                
                # 确保 int16 类型
                resampled_data = np.int16(resampled_data)
                
                # 保存为原名
                new_file = file.replace('_backup.wav', '.wav')
                new_path = os.path.join(subdir, new_file)
                wavfile.write(new_path, 22050, resampled_data)
                print(f"  已重采样并保存为 22050 Hz")
                
                # 删除备份
                os.remove(file_path)
                
            except Exception as e:
                print(f"  错误: {e}")
                # 恢复备份
                new_file = file.replace('_backup.wav', '.wav')
                os.rename(file_path, os.path.join(subdir, new_file))