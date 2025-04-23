import librosa
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# Function to extract MFCC features from an audio file
def extract_mfcc(file_path, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=None)  # Load the audio file
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)  # Extract MFCC features
    return mfcc.T  # Transpose to have time dimension first

# Function to calculate similarity using DTW
def calculate_similarity(mfcc1, mfcc2):
    distance, _ = fastdtw(mfcc1, mfcc2, dist=euclidean)
    return distance

# Example usage:
file1 = 'audio_file_1.wav'  # Path to your first audio file
file2 = 'audio_file_2.wav'  # Path to your second audio file

# Extract MFCCs from both audio files
mfcc1 = extract_mfcc(file1)
mfcc2 = extract_mfcc(file2)

# Calculate similarity
similarity_score = calculate_similarity(mfcc1, mfcc2)
print(f"DTW Distance (lower is more similar): {similarity_score}")
