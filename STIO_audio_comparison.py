import os
import numpy as np
import pandas as pd
import librosa
from pystoi.stoi import stoi

# Load dataset metadata
df = pd.read_csv('bangla_tts_dataset/metadata.tsv', sep='\t', header=None, names=["file", "text"])

# Paths to reference and generated audio
ref_dir = 'bangla_tts_dataset/wavs'
gen_dir = 'generated_audio'

stoi_scores = []

for idx, row in df.iterrows():
    ref_path = os.path.join(ref_dir, f"{row['file']}.wav")
    gen_path = os.path.join(gen_dir, f"{row['file']}_gen.wav")  # Adjust based on your file pattern

    if os.path.exists(ref_path) and os.path.exists(gen_path):
        try:
            # Load both audios and resample to 16kHz (recommended for STOI)
            ref_audio, _ = librosa.load(ref_path, sr=16000)
            gen_audio, _ = librosa.load(gen_path, sr=16000)

            # Match lengths
            min_len = min(len(ref_audio), len(gen_audio))
            ref_audio = ref_audio[:min_len]
            gen_audio = gen_audio[:min_len]

            # Compute STOI score
            score = stoi(ref_audio, gen_audio, fs=16000, extended=False)
            stoi_scores.append(score)

        except Exception as e:
            print(f"Error with file {row['file']}: {e}")
            continue

# Compute average STOI
avg_stoi = np.mean(stoi_scores)
print(f"Average STOI Score: {avg_stoi:.3f}")
