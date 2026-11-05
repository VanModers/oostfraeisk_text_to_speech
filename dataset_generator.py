import os
import re
import sounddevice as sd
import soundfile as sf
import librosa
import numpy as np
import winsound

# -------- CONFIG --------
TEXT_FILE = "texts/rec0.txt"   # Your input text file
OUTPUT_DIR = "data/oostfraeisk"                # Where recordings and metadata.csv will be saved
SAMPLE_RATE = 22050                      # Sample rate for recording (Hz)
CHANNELS = 1                             # Mono audio
MAX_WORDS = 25                           # Maximum words per segment
DEFAULT_SPEAKER_ID = "speaker_2"         # Standard-Sprecher-ID für metadata_multispeaker.csv
# -------------------------

def split_sentences(text):
    """Split into sentences, then break long ones into shorter phrases."""
    sentences = re.split(r'(?<=[\n])+', text.strip())
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

def trim_silence(audio, top_db=35):
    """Trim trailing silence from audio using librosa."""
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed

def select_input_device():
    """Let the user pick an input device. Returns the device index or None for the system default."""
    devices = sd.query_devices()
    mme_index = next(i for i, api in enumerate(sd.query_hostapis()) if api["name"] == "MME")
    inputs = [(i, d) for i, d in enumerate(devices)
              if d["max_input_channels"] > 0 and d["hostapi"] == mme_index]
    if not inputs:
        raise RuntimeError("Kein Aufnahmegerät gefunden.")

    default_in = sd.default.device[0]
    print("\nVerfügbare Mikrofone:")
    for i, d in inputs:
        marker = " (Standard)" if i == default_in else ""
        print(f"  [{i}] {d['name']}{marker}")

    while True:
        choice = input("Mikrofon-Nummer eingeben (ENTER für Standard): ").strip()
        if not choice:
            return None
        if choice.isdigit() and int(choice) in [i for i, _ in inputs]:
            return int(choice)
        print("Ungültige Auswahl, bitte erneut versuchen.")

def record_sentence(sentence, filename, device=None):
    print(f"\nPlease read aloud:\n>>> {sentence}")
    input("Press ENTER when ready to record...")
    
    duration = max(3, len(sentence.split()) // 1.5 + 2)  # rough guess
    print(f"Recording... (approx {duration} seconds)")
    winsound.Beep(1000,500)
    recording = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, device=device)
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

    device = select_input_device()
    if device is None:
        print("Verwende Standard-Mikrofon.")
    else:
        print(f"Verwende Mikrofon [{device}]: {sd.query_devices(device)['name']}")

    speaker_id = input(f"Sprecher-ID eingeben (ENTER für {DEFAULT_SPEAKER_ID}): ").strip() or DEFAULT_SPEAKER_ID
    if speaker_id.isdigit():
        speaker_id = f"speaker_{speaker_id}"

    metadata_path = os.path.join(OUTPUT_DIR, "metadata.csv")
    multispeaker_path = os.path.join(OUTPUT_DIR, "metadata_multispeaker.csv")
    existing_sentences, last_index = load_existing_metadata(metadata_path)

    with open(metadata_path, "a", encoding="utf-8") as meta, \
         open(multispeaker_path, "a", encoding="utf-8") as multi:
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

            record_sentence(sentence, filepath, device)
            meta.write(f"{filename}|{sentence}|{sentence}\n")
            meta.flush()
            multi.write(f"{filename}|{speaker_id}|{sentence}\n")
            multi.flush()
            sentence_num += 1

    print("\n✅ Recording session complete!")
    print(f"Metadata file: {metadata_path}")

if __name__ == "__main__":
    main()