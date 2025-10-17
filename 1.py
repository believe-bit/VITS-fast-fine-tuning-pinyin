import os
import glob
from sklearn.model_selection import train_test_split

# 路径设置
input_dir = "/data/huangtianle/VITS-fast-fine-tuning/custom_character_voice/speaker0"
output_dir = "/data/huangtianle/VITS-fast-fine-tuning/segmented_character_voice"
train_file = os.path.join(output_dir, "yue_train.txt")
val_file = os.path.join(output_dir, "yue_val.txt")

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 获取所有 .wav 文件
wav_files = glob.glob(os.path.join(input_dir, "*.wav"))
print(f"找到 {len(wav_files)} 个音频文件")

# 存储数据集条目
dataset = []

for wav_path in wav_files:
    # 获取对应的 .txt 文件
    txt_path = wav_path.replace(".wav", ".txt")
    if not os.path.exists(txt_path):
        print(f"警告: {txt_path} 不存在，跳过")
        continue
    
    # 读取 .txt 文件的第二行（拼音）
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        if len(lines) < 2:
            print(f"警告: {txt_path} 格式错误，跳过")
            continue
        pinyin = lines[1].strip()  # 第二行是拼音
        
    # 构造 VITS 格式：audio_path|0|pinyin
    relative_path = os.path.join("custom_character_voice/speaker0", os.path.basename(wav_path))
    entry = f"{relative_path}|0|{pinyin}"
    dataset.append(entry)

# 按 9:1 分割数据集
train_data, val_data = train_test_split(dataset, test_size=0.1, random_state=42)

# 保存到 yue_train.txt 和 yue_val.txt
with open(train_file, 'w', encoding='utf-8') as f:
    for entry in train_data:
        f.write(f"{entry}\n")
print(f"保存训练集: {train_file}，共 {len(train_data)} 条")

with open(val_file, 'w', encoding='utf-8') as f:
    for entry in val_data:
        f.write(f"{entry}\n")
print(f"保存验证集: {val_file}，共 {len(val_data)} 条")