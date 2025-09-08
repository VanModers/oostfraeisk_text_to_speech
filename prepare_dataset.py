import whisper
import torchaudio
from pathlib import Path
import torch
import json

INPUT_DIR = Path("long_recordings")  # .wav/.mp3
OUTPUT_DIR = Path("data/transcripts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
whisper_model = whisper.load_model("medium", device=device)

for audio_file in INPUT_DIR.glob("*"):
    if audio_file.suffix.lower() not in [".wav", ".mp3"]:
        continue

    print(f"🎙️ Processing {audio_file.name}...")

    # Whisper transcription (with timestamps)
    whisper_result = whisper_model.transcribe(str(audio_file), word_timestamps=True, language="nl")
    segments = whisper_result['segments']

    # Save segments to JSON
    out_path = OUTPUT_DIR / f"{audio_file.stem}_segments.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved segments to {out_path}")