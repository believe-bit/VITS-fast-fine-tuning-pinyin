# /data/huangtianle/VITS-fast-fine-tuning/text.py
def cleaned_text_to_sequence(cleaned_text, symbols):
    """将清理后的文本转换为符号序列"""
    symbol_to_id = {s: i for i, s in enumerate(symbols)}
    sequence = []
    for symbol in cleaned_text.split():
        if symbol in symbol_to_id:
            sequence.append(symbol_to_id[symbol])
        else:
            print(f"警告: 符号 {symbol} 不在符号表中，跳过")
    if not sequence:
        print(f"错误: 转换序列为空，输入: {cleaned_text}")
    return sequence

def chinese_cleaners(text):
    """处理拼音文本"""
    return text.strip()

def text_to_sequence(text, symbols, cleaners):
    """将文本转换为序列"""
    for cleaner in cleaners:
        if cleaner == "chinese_cleaners":
            cleaned_text = chinese_cleaners(text)
        else:
            raise ValueError(f"不支持的清理规则: {cleaner}")
    return cleaned_text_to_sequence(cleaned_text, symbols)