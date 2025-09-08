import whisperx
import os
from pathlib import Path
import torchaudio
import torch

# -----------------------------
# CONFIG
# -----------------------------
INPUT_DIR = Path("long_recordings")       # .wav/.mp3 + .txt
OUTPUT_DIR = Path("data/oostfraeisk")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "wavs").mkdir(parents=True, exist_ok=True)

metadata_path = OUTPUT_DIR / "metadata.csv"

# -----------------------------
# LOAD MODELS
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
align_model, metadata = whisperx.load_align_model(language_code="de", device=device)

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

        print(f"🎙️ Processing {audio_file.name}...")

        # Load transcript
        transcript = transcript_file.read_text(encoding="utf-8")

        # Split transcript into pseudo-sentences (simple split, can be improved)
        segments = [
            {"text": s.strip(), "start": 0.0, "end": 0.0}
            for s in transcript.replace("?", ".").replace("!", ".").split(".")
            if s.strip()
        ]

        waveform, sr = torchaudio.load(audio_file)
        waveform = waveform.to(device)

        aligned = whisperx.align(
            segments, align_model, metadata, waveform, sr, device
        )

        # Reload audio on CPU just for slicing/export
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