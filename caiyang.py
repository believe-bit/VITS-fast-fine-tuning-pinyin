import librosa
import os

audio_dir = "/data/huangtianle/VITS-fast-fine-tuning/custom_character_voice/sichuan"  # 替换为您的音频目录
for wav_file in os.listdir(audio_dir):
    if wav_file.endswith(".wav"):
        audio, sr = librosa.load(os.path.join(audio_dir, wav_file))
        print(f"文件: {wav_file}, 采样率: {sr}, 时长: {len(audio)/sr:.2f}秒")