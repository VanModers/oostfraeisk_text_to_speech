import whisperx
from pathlib import Path
import torchaudio
import torch
import re

# -----------------------------
# CONFIG
# -----------------------------
INPUT_DIR = Path("long_recordings")  # .wav/.mp3 + .txt
OUTPUT_DIR = Path("data/oostfraeisk")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "wavs").mkdir(parents=True, exist_ok=True)

metadata_path = OUTPUT_DIR / "metadata.csv"

# -----------------------------
# LOAD MODEL
# -----------------------------
device = "cpu"
align_model, metadata = whisperx.load_align_model(language_code="de", device=device)

# -----------------------------
# HELPERS
# -----------------------------
def split_text_into_sentences(text):
    """Split transcript into sentences using punctuation."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

# -----------------------------
# PROCESS FILES
# -----------------------------
with open(metadata_path, "w", encoding="utf-8") as f_meta:
    for audio_file in INPUT_DIR.glob("*"):
        if audio_file.suffix.lower() not in [".wav", ".mp3"]:
            continue

        transcript_file = audio_file.with_suffix(".txt")
        if not transcript_file.exists():
            print(f"⚠️ No transcript found for {audio_file}, skipping...")
            continue

        print(f"🎙️ Aligning {audio_file.name} with transcript...")

        # Load transcript
        transcript = transcript_file.read_text(encoding="utf-8")
        sentences = split_text_into_sentences(transcript)

        # Prepare segments for alignment
        segments = [{"text": s, "start": 0.0, "end": 0.0} for s in sentences]

        # Load audio
        waveform, sr = torchaudio.load(audio_file)

        # Align sentences to audio
        aligned = whisperx.align(segments, align_model, metadata, waveform, sr, device)

        # Reload audio on CPU for slicing/export
        audio, sr = torchaudio.load(audio_file)

        # Export aligned clips + metadata
        base_id = audio_file.stem
        for i, seg in enumerate(aligned["segments"]):
            if "start" not in seg or "end" not in seg:
                continue  # skip unaligned

            start, end = seg["start"], seg["end"]
            text = seg["text"].strip()
            clip_name = f"{base_id}_{i:03d}.wav"

            torchaudio.save(
                OUTPUT_DIR / "wavs" / clip_name,
                audio[:, int(start * sr):int(end * sr)],
                sr,
            )

            f_meta.write(f"{clip_name}|{text}\n")

print("✅ Done! Dataset in", OUTPUT_DIR)