import os
from pathlib import Path
from pinyin_jyutping import PinyinJyutping

def text_to_jyutping(input_dir):
    """为指定目录中的所有TXT文件生成完整Jyutping拼音，保存到原TXT文件的第二行"""
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Directory {input_dir} does not exist")

      # 初始化Jyutping转换器
    jyutping_converter = PinyinJyutping()

    for txt_file in input_path.glob("*.txt"):
          # 读取原始文本（第一行）
        with open(txt_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            original_text = lines[0].strip() if lines else ""

        if not original_text:
            print(f"Empty text in {txt_file}")
            continue

          # 生成Jyutping（带音调数字，空格分隔）
        try:
            jyutping_text = jyutping_converter.jyutping(original_text, tone_numbers=True, spaces=True)
        except Exception as e:
            print(f"Error processing {txt_file}: {e}")
            jyutping_text = "ERROR_IN_JYUTPING"

          # 保存原始文本和Jyutping到原文件
        with open(txt_file, "w", encoding="utf-8") as f:
            f.write(f"{original_text}\n{jyutping_text}")
        print(f"Processed {txt_file}: {original_text} -> {jyutping_text}")

if __name__ == "__main__":
    input_dir = "custom_character_voice/speaker0"
    text_to_jyutping(input_dir)