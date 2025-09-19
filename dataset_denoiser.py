import os
import pandas as pd
import librosa
import soundfile as sf
import noisereduce as nr
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Paths
DATASET_PATH = "data/oostfraeisk"          # base dataset folder
WAVS_PATH = os.path.join(DATASET_PATH, "wavs")
METADATA_PATH = os.path.join(DATASET_PATH, "metadata.csv")

# Output folder for denoised wavs
OUTPUT_WAVS_PATH = os.path.join(DATASET_PATH, "wavs_denoised")
os.makedirs(OUTPUT_WAVS_PATH, exist_ok=True)

# Load metadata
metadata = pd.read_csv(METADATA_PATH, sep="|", header=None, names=["file", "text_norm", "text_orig"])

# Process each file
for idx, row in metadata.iterrows():
    filename = row["file"] + ".wav"
    filepath = os.path.join(WAVS_PATH, filename)
    output_path = os.path.join(OUTPUT_WAVS_PATH, filename)

    if not os.path.exists(filepath):
        print(f"⚠️ File not found: {filepath}")
        continue

    # Load audio
    y, sr = librosa.load(filepath, sr=None)

    # Estimate noise from first 0.5s (adjust if silence is elsewhere)
    noise_clip = y[:int(sr * 0.5)]

    pro = 0.9 # noise reduction aggressiveness

    if idx in range(301, 401) or idx in range(501, 601) or idx in range(651, 680):  # Example indices for less aggressive reduction
        pro = 0.4

    # Reduce noise
    reduced = nr.reduce_noise(y=y, sr=sr, y_noise=noise_clip, prop_decrease=pro)

    # Peak normalize
    reduced = reduced / np.max(np.abs(reduced))

    # Optional: RMS normalize for more consistent perceived volume
    target_rms = 0.1
    rms = np.sqrt(np.mean(reduced**2))
    if rms > 0:
        reduced = reduced * (target_rms / rms)

    # Save denoised + normalized
    sf.write(output_path, reduced, sr)

    if idx % 50 == 0:
        print(f"[{idx}/{len(metadata)}] Processed {filename}")

print("✅ All files denoised and saved to:", OUTPUT_WAVS_PATH)