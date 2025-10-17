import os
speaker_dir = "custom_character_voice/speaker0"
for file in os.listdir(speaker_dir):
    if file.endswith(".wav.trn"):
        wav_name = file.replace(".wav.trn", "")
        os.rename(
            os.path.join(speaker_dir, file),
            os.path.join(speaker_dir, f"{wav_name}.txt")
        )