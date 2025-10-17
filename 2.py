import os
import glob
import re
from sklearn.model_selection import train_test_split
from pypinyin import pinyin, Style
import jieba

input_dir = "/data/huangtianle/VITS-fast-fine-tuning/custom_character_voice/sichuan"
output_dir = "/data/huangtianle/VITS-fast-fine-tuning/filelists"  # 更新路径
train_file = os.path.join(output_dir, "sichuan_train.txt")
val_file = os.path.join(output_dir, "sichuan_val.txt")

os.makedirs(output_dir, exist_ok=True)

wav_files = glob.glob(os.path.join(input_dir, "*.wav"))
print(f"找到 {len(wav_files)} 个音频文件")

dataset = []
for wav_path in wav_files:
    txt_path = wav_path.replace(".wav", ".txt")
    if not os.path.exists(txt_path):
        print(f"警告: {txt_path} 不存在，跳过")
        continue
    
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        if len(lines) < 1:
            print(f"警告: {txt_path} 格式错误（无文本），跳过")
            continue
        text = lines[0].strip()
    
    # 清理所有非中文字符，保留汉字和空格
    text = re.sub(r'[^\u4e00-\u9fff\s]', '', text)
    text = text.strip()
    
    if not text:
        print(f"警告: {txt_path} 清理后文本为空，跳过")
        continue
    
    # 转换为普通话拼音（Style.TONE3）
    words = jieba.lcut(text)
    pinyin_words = pinyin(words, style=Style.TONE3, heteronym=False, v_to_u=True)
    pinyin_text = ' '.join([item[0] for item in pinyin_words])
    
    # 再次清理拼音，确保无非法字符
    pinyin_text = re.sub(r'[^a-zA-Z0-9\s]', '', pinyin_text)
    
    # 验证拼音格式（只允许字母、数字、空格）
    if not re.match(r'^[a-zA-Z0-9\s]+$', pinyin_text):
        print(f"警告: {txt_path} 拼音包含非法字符: {pinyin_text}，跳过")
        continue
    
    # 验证音频文件存在
    if not os.path.exists(wav_path):
        print(f"警告: {wav_path} 不存在，跳过")
        continue
    
    relative_path = os.path.join("custom_character_voice/sichuan", os.path.basename(wav_path))
    entry = f"{relative_path}|0|{pinyin_text}"
    dataset.append(entry)

print(f"有效数据条数: {len(dataset)}")
if len(dataset) < 1000:
    print("错误: 有效数据量过少，请检查输入文件！")

train_data, val_data = train_test_split(dataset, test_size=0.1, random_state=42)

with open(train_file, 'w', encoding='utf-8') as f:
    for entry in train_data:
        f.write(f"{entry}\n")
print(f"保存训练集: {train_file}，共 {len(train_data)} 条")

with open(val_file, 'w', encoding='utf-8') as f:
    for entry in val_data:
        f.write(f"{entry}\n")
print(f"保存验证集: {val_file}，共 {len(val_data)} 条")