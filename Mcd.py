import os
import numpy as np
import pandas as pd
import librosa
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# MCD calculation function
def calculate_mcd(ref_audio, gen_audio, sr=22050, n_mfcc=13):
    # Extract MFCCs
    ref_mfcc = librosa.feature.mfcc(y=ref_audio, sr=sr, n_mfcc=n_mfcc)
    gen_mfcc = librosa.feature.mfcc(y=gen_audio, sr=sr, n_mfcc=n_mfcc)

    # Use DTW to align
    distance, path = fastdtw(ref_mfcc.T, gen_mfcc.T, dist=euclidean)
    
    # Compute MCD
    diff = ref_mfcc[:, path[0]].T - gen_mfcc[:, path[1]].T
    mcd = (10.0 / np.log(10)) * np.sqrt((diff ** 2).mean())

    return mcd

# Load metadata
df = pd.read_csv('bangla_tts_dataset/metadata.tsv', sep='\t', header=None, names=["file", "text"])

# Path settings
ref_dir = 'bangla_tts_dataset/wavs'
gen_dir = 'generated_audio'

mcd_scores = []

for idx, row in df.iterrows():
    ref_path = os.path.join(ref_dir, f"{row['file']}.wav")
    gen_path = os.path.join(gen_dir, f"{row['file']}_gen.wav")
    
    if os.path.exists(ref_path) and os.path.exists(gen_path):
        ref_audio, _ = librosa.load(ref_path, sr=22050)
        gen_audio, _ = librosa.load(gen_path, sr=22050)
        
        try:
            mcd = calculate_mcd(ref_audio, gen_audio)
            mcd_scores.append(mcd)
        except Exception as e:
            print(f"Error computing MCD for {row['file']}: {e}")
            continue

# Final average MCD
avg_mcd = np.mean(mcd_scores)
print(f"Average MCD: {avg_mcd:.2f} dB")
