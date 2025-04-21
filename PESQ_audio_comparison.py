import os
import numpy as np
import pandas as pd
import librosa
from pesq import pesq

# Load dataset
df = pd.read_csv('bangla_tts_dataset/metadata.tsv', sep='\t', header=None, names=["file", "text"])

# Path to audio
ref_dir = 'bangla_tts_dataset/wavs'
gen_dir = 'generated_audio'

pesq_scores = []

for idx, row in df.iterrows():
    ref_path = os.path.join(ref_dir, f"{row['file']}.wav")
    gen_path = os.path.join(gen_dir, f"{row['file']}_gen.wav")  # Change to your filename pattern

    if os.path.exists(ref_path) and os.path.exists(gen_path):
        try:
            # Load and resample to 16kHz
            ref_audio, _ = librosa.load(ref_path, sr=16000)
            gen_audio, _ = librosa.load(gen_path, sr=16000)

            # Match lengths (required by PESQ)
            min_len = min(len(ref_audio), len(gen_audio))
            ref_audio = ref_audio[:min_len]
            gen_audio = gen_audio[:min_len]

            # Compute PESQ score (narrowband mode)
            score = pesq(16000, ref_audio, gen_audio, 'nb')
            pesq_scores.append(score)

        except Exception as e:
            print(f"Error with {row['file']}: {e}")
            continue

# Average PESQ
avg_pesq = np.mean(pesq_scores)
print(f"Average PESQ Score: {avg_pesq:.2f}")
