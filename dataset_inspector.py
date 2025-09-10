import os
import random
import sounddevice as sd
import soundfile as sf
import pandas as pd

# -------- CONFIG --------
DATA_DIR = "data/oostfraeisk"         # Folder containing metadata.csv and wavs/
METADATA_FILE = os.path.join(DATA_DIR, "metadata.csv")
WAVS_DIR = os.path.join(DATA_DIR, "wavs")
SAMPLE_RATE = 22050
# -------------------------

def load_metadata():
    """Load metadata.csv into a DataFrame."""
    df = pd.read_csv(
        METADATA_FILE,
        sep="|",
        header=None,
        names=["filename", "text", "speaker"],
        quoting=3,
        engine="python"
    )
    return df

def play_audio(filepath):
    """Play an audio file with sounddevice."""
    data, samplerate = sf.read(filepath, dtype="float32")
    sd.play(data, samplerate)
    sd.wait()

def main():
    if not os.path.exists(METADATA_FILE):
        print(f"❌ No metadata file found at {METADATA_FILE}")
        return
    
    df = load_metadata()
    print(f"✅ Loaded {len(df)} recordings.")

    while True:
        input("\nPress ENTER to hear a random recording (CTRL+C to quit)...")
        row = df.sample(1).iloc[0]

        filename = row["filename"]
        text = row["text"]

        filepath = os.path.join(WAVS_DIR, f"{filename}.wav")
        if not os.path.exists(filepath):
            print(f"⚠️ Missing audio file: {filepath}")
            continue

        print(f"\n▶️  Playing: {filename}")
        print(f"📝  Text: {text}\n")

        play_audio(filepath)

if __name__ == "__main__":
    main()