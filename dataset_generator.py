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
    sentences = re.split(r'(?<=[.!?])+', text.strip())
    print(sentences)
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
                processed.append(chunk.strip())
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
    
    duration = max(3, len(sentence.split()) // 2 + 2)  # rough guess
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

def load_existing_metadata(metadata_path):
    """Load sentences and last index from metadata.csv if it exists."""
    existing_sentences = set()
    last_index = 0
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as meta:
            for line in meta:
                parts = line.strip().split("|")
                if len(parts) >= 2:
                    existing_sentences.add(parts[1])
                    # Extract index from filename (e.g., sentence_0005)
                    match = re.match(r"sentence_(\d+)", parts[0])
                    if match:
                        idx = int(match.group(1))
                        last_index = max(last_index, idx)
    return existing_sentences, last_index

def main():
    os.makedirs(os.path.join(OUTPUT_DIR, "wavs"), exist_ok=True)

    # Read input text
    with open(TEXT_FILE, "r", encoding="utf-8") as f:
        text = f.read()

    sentences = split_sentences(text)

    metadata_path = os.path.join(OUTPUT_DIR, "metadata.csv")
    existing_sentences, last_index = load_existing_metadata(metadata_path)

    with open(metadata_path, "a", encoding="utf-8") as meta:
        sentence_num = last_index + 1
        for sentence in sentences:
            if sentence in existing_sentences:
                print(f"Skipping already in metadata: {sentence}")
                continue

            filename = f"sentence_{sentence_num:04d}"
            filepath = os.path.join(OUTPUT_DIR, f"wavs/{filename}.wav")

            # Skip if already recorded
            if os.path.exists(filepath):
                print(f"Skipping already recorded: {filename}, {sentence}")
                sentence_num += 1
                continue

            record_sentence(sentence, filepath)
            meta.write(f"{filename}|{sentence}|speaker\n")
            sentence_num += 1

    print("\n✅ Recording session complete!")
    print(f"Metadata file: {metadata_path}")

if __name__ == "__main__":
    main()