import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import cosine

from pystoi.stoi import stoi
from pesq import pesq

import torch
from transformers import Wav2Vec2Model, Wav2Vec2Processor

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Wav2Vec2 model and processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)
model.eval()

def calculate_mcd(ref, gen, sr=22050, n_mfcc=13):
    ref_mfcc = librosa.feature.mfcc(ref, sr=sr, n_mfcc=n_mfcc)
    gen_mfcc = librosa.feature.mfcc(gen, sr=sr, n_mfcc=n_mfcc)
    min_frames = min(ref_mfcc.shape[1], gen_mfcc.shape[1])
    ref_mfcc = ref_mfcc[:, :min_frames]
    gen_mfcc = gen_mfcc[:, :min_frames]
    diff = ref_mfcc - gen_mfcc
    mcd = np.mean(np.sqrt(np.sum(diff**2, axis=0)))
    return mcd

def get_embedding(audio, sr):
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return torch.mean(outputs.last_hidden_state[0], dim=0).cpu().numpy()

def evaluate(reference_folder, generated_folder, sample_rate=16000):
    results = []

    ref_files = [f for f in os.listdir(reference_folder) if f.endswith(".wav")]

    for file in tqdm(ref_files):
        base_name = file.replace(".wav", "")
        gen_file = base_name + "_geb.wav"
        ref_path = os.path.join(reference_folder, file)
        gen_path = os.path.join(generated_folder, gen_file)

        if not os.path.exists(gen_path):
            print(f"Missing: {gen_path}")
            continue

        ref_audio, _ = librosa.load(ref_path, sr=sample_rate)
        gen_audio, _ = librosa.load(gen_path, sr=sample_rate)

        min_len = min(len(ref_audio), len(gen_audio))
        ref_audio = ref_audio[:min_len]
        gen_audio = gen_audio[:min_len]

        try:
            mcd_score = calculate_mcd(ref_audio, gen_audio, sr=sample_rate)
            pesq_score = pesq(sample_rate, ref_audio, gen_audio, 'wb')
            stoi_score = stoi(ref_audio, gen_audio, sample_rate, extended=True)
            emb_ref = get_embedding(ref_audio, sample_rate)
            emb_gen = get_embedding(gen_audio, sample_rate)
            cosine_sim = 1 - cosine(emb_ref, emb_gen)

            results.append({
                "file": base_name,
                "MCD": mcd_score,
                "PESQ": pesq_score,
                "STOI": stoi_score,
                "CosineSimilarity": cosine_sim
            })

        except Exception as e:
            print(f"Error processing {file}: {e}")

    df = pd.DataFrame(results)
    df.to_csv("audio_comparison_results.csv", index=False)
    print("\n--- Averages ---")
    print(df.mean(numeric_only=True))

if __name__ == "__main__":
    # Replace with actual paths before running
    reference_folder = "path/to/reference_folder"
    generated_folder = "path/to/generated_folder"

    evaluate(reference_folder, generated_folder)
