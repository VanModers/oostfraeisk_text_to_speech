import os
import re
import sounddevice as sd
import soundfile as sf
import librosa
import numpy as np

# -------- CONFIG --------
TEXT_FILE = "texts/rec0.txt"   # Your input text file
OUTPUT_DIR = "data/oostfraeisk"                # Where recordings and metadata.csv will be saved
SAMPLE_RATE = 22050                      # Sample rate for recording (Hz)
CHANNELS = 1                             # Mono audio
MAX_WORDS = 15                           # Maximum words per segment
# -------------------------

def split_sentences(text):
    """Split into sentences, then break long ones into shorter phrases."""
    sentences = re.split(r'(?<=[.!?]) +', text.strip())
    processed = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        words = s.split()
        if len(words) > MAX_WORDS:
            # Break into chunks
            for i in range(0, len(words), MAX_WORDS):
                chunk = " ".join(words[i:i+MAX_WORDS])
                processed.append(chunk)
        else:
            processed.append(s)
    return processed

def trim_silence(audio, top_db=30):
    """Trim trailing silence from audio using librosa."""
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed

def record_sentence(sentence, filename):
    print(f"\nPlease read aloud:\n>>> {sentence}")
    input("Press ENTER when ready to record...")
    
    duration = max(3, len(sentence.split()) // 2 + 1)  # rough guess
    print(f"Recording... (approx {duration} seconds)")
    recording = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS)
    sd.wait()  # Wait until recording is finished
    
    # Convert to mono numpy array
    recording = recording.flatten()
    
    # Trim silence
    recording = trim_silence(recording)
    
    # Save file
    sf.write(filename, recording, SAMPLE_RATE)
    print(f"Saved recording: {filename}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Read input text
    with open(TEXT_FILE, "r", encoding="utf-8") as f:
        text = f.read()

    sentences = split_sentences(text)

    metadata_path = os.path.join(OUTPUT_DIR, "metadata.csv")
    with open(metadata_path, "a", encoding="utf-8") as meta:
        for i, sentence in enumerate(sentences, 1):
            filename = f"sentence_{i:04d}.wav"
            filepath = os.path.join(OUTPUT_DIR, filename)

            # Skip if already recorded
            if os.path.exists(filepath):
                print(f"Skipping already recorded: {filename}")
                continue

            record_sentence(sentence, filepath)
            meta.write(f"{filename}|{sentence}\n")

    print("\n✅ Recording session complete!")
    print(f"Metadata file: {metadata_path}")

if __name__ == "__main__":
    main()