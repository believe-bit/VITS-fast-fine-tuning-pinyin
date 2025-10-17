import os
import torchaudio
from torchaudio.transforms import Resample

def main():
    target_sr = 16000
    audio_dir = "custom_character_voice/speaker0"
    print(f"Scanning directory: {audio_dir}")
    
    if not os.path.exists(audio_dir):
        print(f"Error: Directory {audio_dir} does not exist!")
        return
    
    filelist = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
    print(f"Found {len(filelist)} audio files: {filelist[:5]}...")
    
    for wavfile in filelist:
        input_path = os.path.join(audio_dir, wavfile)
        print(f"Processing: {wavfile}")
        waveform, sr = torchaudio.load(input_path, normalize=True)
        if sr != target_sr:
            resampler = Resample(orig_freq=sr, new_freq=target_sr)
            waveform = resampler(waveform)
            torchaudio.save(input_path, waveform, target_sr)
            print(f"Resampled {wavfile} to {target_sr}Hz")
        else:
            print(f"Skipping {wavfile}: already at {target_sr}Hz")

if __name__ == "__main__":
    main()