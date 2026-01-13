"""
Voice Normalizer for Multi-Speaker TTS Training

This script helps merge two speaker voices by normalizing their acoustic characteristics.
Run this AFTER denoising but BEFORE training.

Methods:
1. Pitch normalization - Shift F0 to match target speaker
2. Speaking rate normalization - Time-stretch to match average duration
3. Formant shifting (optional) - More aggressive voice matching
"""

import os
import numpy as np
import librosa
import soundfile as sf
import pyworld as pw
from scipy.signal import resample
from tqdm import tqdm

# =============================================================================
# Configuration
# =============================================================================

DATASET_PATH = "data/oostfraeisk"
WAVS_INPUT = os.path.join(DATASET_PATH, "wavs")  # or "wavs"
WAVS_OUTPUT = os.path.join(DATASET_PATH, "wavs_normalized")
METADATA_PATH = os.path.join(DATASET_PATH, "metadata.csv")

# Define speaker ranges (matching the denoiser.py settings)
# Speaker A: more aggressive denoising (pro=0.9) - this is the dominant voice
# Speaker B: less aggressive denoising (pro=0.4) - to be normalized to match A
SPEAKER_A_INDICES = list(range(0, 301)) + list(range(401, 501)) + list(range(601, 651))
SPEAKER_B_INDICES = list(range(301, 401)) + list(range(501, 601)) + list(range(651, 821))

# Normalization settings
NORMALIZE_PITCH = True
NORMALIZE_DURATION = True
FORMANT_SHIFT_RATIO = 1.0  # 1.0 = no shift, <1.0 = lower formants, >1.0 = higher

os.makedirs(WAVS_OUTPUT, exist_ok=True)


# =============================================================================
# Step 1: Analyze both speakers' characteristics
# =============================================================================

def analyze_speaker_f0(wav_files, wav_dir, sample_size=50):
    """Compute average F0 (pitch) for a set of files."""
    f0_values = []
    
    files_to_analyze = wav_files[:sample_size] if len(wav_files) > sample_size else wav_files
    
    for filename in tqdm(files_to_analyze, desc="Analyzing F0"):
        filepath = os.path.join(wav_dir, filename)
        if not os.path.exists(filepath):
            continue
            
        y, sr = librosa.load(filepath, sr=None)
        y = y.astype(np.float64)
        
        # Extract F0 using WORLD vocoder
        f0, _ = pw.harvest(y, sr)
        
        # Only count voiced frames
        voiced_f0 = f0[f0 > 0]
        if len(voiced_f0) > 0:
            f0_values.extend(voiced_f0)
    
    return np.median(f0_values), np.std(f0_values)


def analyze_speaking_rate(wav_files, wav_dir, texts, sample_size=50):
    """Compute average speaking rate (chars per second)."""
    rates = []
    
    for i, filename in enumerate(wav_files[:sample_size]):
        filepath = os.path.join(wav_dir, filename)
        if not os.path.exists(filepath):
            continue
            
        y, sr = librosa.load(filepath, sr=None)
        duration = len(y) / sr
        
        # Get text length (approximate syllables)
        if i < len(texts):
            text_len = len(texts[i].replace(" ", ""))
            if duration > 0:
                rates.append(text_len / duration)
    
    return np.mean(rates) if rates else 0


# =============================================================================
# Step 2: Pitch shifting using WORLD vocoder
# =============================================================================

def shift_pitch(y, sr, semitones):
    """Shift pitch by given semitones using WORLD vocoder."""
    y = y.astype(np.float64)
    
    # Extract features
    f0, t = pw.harvest(y, sr)
    sp = pw.cheaptrick(y, f0, t, sr)
    ap = pw.d4c(y, f0, t, sr)
    
    # Shift F0
    ratio = 2 ** (semitones / 12.0)
    f0_shifted = f0 * ratio
    
    # Synthesize
    y_shifted = pw.synthesize(f0_shifted, sp, ap, sr)
    
    return y_shifted.astype(np.float32)


def shift_formants(y, sr, ratio):
    """Shift formants by resampling the spectral envelope."""
    if ratio == 1.0:
        return y
        
    y = y.astype(np.float64)
    
    # Extract features
    f0, t = pw.harvest(y, sr)
    sp = pw.cheaptrick(y, f0, t, sr)
    ap = pw.d4c(y, f0, t, sr)
    
    # Shift formants by resampling spectral envelope
    new_sp = np.zeros_like(sp)
    for i in range(sp.shape[0]):
        old_len = sp.shape[1]
        new_len = int(old_len * ratio)
        resampled = resample(sp[i], new_len)
        if new_len > old_len:
            new_sp[i] = resampled[:old_len]
        else:
            new_sp[i, :new_len] = resampled
    
    # Synthesize
    y_shifted = pw.synthesize(f0, new_sp, ap, sr)
    
    return y_shifted.astype(np.float32)


# =============================================================================
# Step 3: Time-stretch for speaking rate normalization
# =============================================================================

def time_stretch(y, sr, rate):
    """Time-stretch audio by given rate (>1 = faster, <1 = slower)."""
    return librosa.effects.time_stretch(y, rate=rate)


# =============================================================================
# Main processing
# =============================================================================

def main():
    import pandas as pd
    
    # Load metadata
    metadata = pd.read_csv(METADATA_PATH, sep="|", header=None, 
                          names=["file", "text_norm", "text_orig"])
    
    # Get file lists for each speaker
    all_files = [f"{row['file']}.wav" for _, row in metadata.iterrows()]
    all_texts = [row['text_norm'] for _, row in metadata.iterrows()]
    
    speaker_a_files = [all_files[i] for i in SPEAKER_A_INDICES if i < len(all_files)]
    speaker_b_files = [all_files[i] for i in SPEAKER_B_INDICES if i < len(all_files)]
    speaker_a_texts = [all_texts[i] for i in SPEAKER_A_INDICES if i < len(all_texts)]
    speaker_b_texts = [all_texts[i] for i in SPEAKER_B_INDICES if i < len(all_texts)]
    
    print(f"Speaker A: {len(speaker_a_files)} files")
    print(f"Speaker B: {len(speaker_b_files)} files")
    
    # Analyze speakers
    print("\n📊 Analyzing Speaker A (target)...")
    f0_a_median, f0_a_std = analyze_speaker_f0(speaker_a_files, WAVS_INPUT)
    rate_a = analyze_speaking_rate(speaker_a_files, WAVS_INPUT, speaker_a_texts)
    print(f"   Median F0: {f0_a_median:.1f} Hz (std: {f0_a_std:.1f})")
    print(f"   Speaking rate: {rate_a:.1f} chars/sec")
    
    print("\n📊 Analyzing Speaker B (to normalize)...")
    f0_b_median, f0_b_std = analyze_speaker_f0(speaker_b_files, WAVS_INPUT)
    rate_b = analyze_speaking_rate(speaker_b_files, WAVS_INPUT, speaker_b_texts)
    print(f"   Median F0: {f0_b_median:.1f} Hz (std: {f0_b_std:.1f})")
    print(f"   Speaking rate: {rate_b:.1f} chars/sec")
    
    # Calculate adjustments
    if f0_a_median > 0 and f0_b_median > 0:
        f0_ratio = f0_a_median / f0_b_median
        semitones = 12 * np.log2(f0_ratio)
        print(f"\n🎵 Pitch shift needed: {semitones:+.1f} semitones")
    else:
        semitones = 0
        
    if rate_a > 0 and rate_b > 0:
        time_ratio = rate_b / rate_a  # If B is faster, slow it down
        print(f"⏱️  Time stretch needed: {time_ratio:.2f}x")
    else:
        time_ratio = 1.0
    
    # Process all files
    print("\n🔄 Processing files...")
    for idx, row in tqdm(metadata.iterrows(), total=len(metadata)):
        filename = f"{row['file']}.wav"
        input_path = os.path.join(WAVS_INPUT, filename)
        output_path = os.path.join(WAVS_OUTPUT, filename)
        
        if not os.path.exists(input_path):
            continue
        
        y, sr = librosa.load(input_path, sr=None)
        
        # Only normalize Speaker B files
        if idx in SPEAKER_B_INDICES:
            # Pitch shift
            if NORMALIZE_PITCH and abs(semitones) > 0.5:
                y = shift_pitch(y, sr, semitones)
            
            # Formant shift (optional)
            if FORMANT_SHIFT_RATIO != 1.0:
                y = shift_formants(y, sr, FORMANT_SHIFT_RATIO)
            
            # Time stretch
            if NORMALIZE_DURATION and abs(time_ratio - 1.0) > 0.05:
                y = time_stretch(y, sr, time_ratio)
        
        # Peak normalize
        y = y / (np.max(np.abs(y)) + 1e-8)
        
        # RMS normalize
        target_rms = 0.1
        rms = np.sqrt(np.mean(y**2))
        if rms > 0:
            y = y * (target_rms / rms)
        
        sf.write(output_path, y, sr)
    
    print(f"\n✅ Done! Normalized files saved to: {WAVS_OUTPUT}")
    print("\n📝 Next steps:")
    print("1. Listen to a few normalized files to verify quality")
    print("2. Update your training notebook to use 'wavs_normalized' folder")
    print("3. Re-train the model")


if __name__ == "__main__":
    # Install pyworld if needed: pip install pyworld
    main()
