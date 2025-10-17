import os
import re
import json
import jieba
from pathlib import Path

def extract_jyutping_from_txt(directory, output_file="jyutping_dict.json"):
    """从指定目录的 .txt 文件中提取汉字和拼音，生成去重的拼音字典"""
    jyutping_dict = {}
    txt_files = Path(directory).glob("*.txt")
    
    for txt_file in txt_files:
        with open(txt_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if len(lines) < 2:
                print(f"Warning: {txt_file} has insufficient lines, skipping")
                continue
            
            # 假设第一行是汉字，第二行是拼音
            hanzi_line = lines[0].strip()
            jyutping_line = lines[1].strip()
            
            # 使用 jieba 分词
            hanzi_words = jieba.lcut(hanzi_line)
            jyutping_words = jyutping_line.split()
            
            # 过滤非汉字（如标点）
            hanzi_words = [w for w in hanzi_words if re.match(r'[\u4e00-\u9fff]+', w)]
            
            # 统计所需拼音数量（根据分词后的字符数）
            required_jyutpings = sum(len(word) for word in hanzi_words)
            
            # 确保拼音数量足够
            if required_jyutpings != len(jyutping_words):
                print(f"Warning: Mismatch in {txt_file}: {len(hanzi_words)} hanzi words ({required_jyutpings} chars), {len(jyutping_words)} jyutping")
                # 尝试逐字符配对
                hanzi_chars = list(''.join(hanzi_words))
                if len(hanzi_chars) == len(jyutping_words):
                    for char, jyutping in zip(hanzi_chars, jyutping_words):
                        if char not in jyutping_dict:
                            jyutping_dict[char] = jyutping
                            jieba.add_word(char)  # 动态添加单字到 jieba
                        elif jyutping_dict[char] != jyutping:
                            print(f"Warning: Duplicate char '{char}' with different jyutping '{jyutping}' vs '{jyutping_dict[char]}' in {txt_file}")
                else:
                    print(f"Error: Cannot align {txt_file}, skipping")
                continue
            
            # 按词配对，拼接拼音
            jyutping_index = 0
            for word in hanzi_words:
                word_len = len(word)
                if not re.match(r'[\u4e00-\u9fff]+', word):
                    continue
                if jyutping_index + word_len <= len(jyutping_words):
                    # 拼接词的拼音
                    jyutping = ''.join(jyutping_words[jyutping_index:jyutping_index + word_len])
                    if word not in jyutping_dict:
                        jyutping_dict[word] = jyutping
                        jieba.add_word(word)  # 动态添加词到 jieba
                    elif jyutping_dict[word] != jyutping:
                        print(f"Warning: Duplicate word '{word}' with different jyutping '{jyutping}' vs '{jyutping_dict[word]}' in {txt_file}")
                    jyutping_index += word_len
                else:
                    print(f"Error: Not enough jyutping for word '{word}' in {txt_file}")
            
    # 保存到文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(jyutping_dict, f, ensure_ascii=False, indent=2)
    print(f"Jyutping dictionary saved to {output_file}")
    return jyutping_dict

if __name__ == "__main__":
    directory = "/data/huangtianle/VITS-fast-fine-tuning/custom_character_voice/speaker0"
    extract_jyutping_from_txt(directory)