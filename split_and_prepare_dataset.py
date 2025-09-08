import whisperx
import torchaudio
import os
from pathlib import Path

# -----------------------------
# CONFIGURATION
# -----------------------------
INPUT_DIR = Path("long_recordings")       # folder with .wav/.mp3 + .txt files
OUTPUT_DIR = Path("data/oostfraeisk")     # dataset folder
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "wavs").mkdir(parents=True, exist_ok=True)

metadata_path = OUTPUT_DIR / "metadata.csv"

# -----------------------------
# LOAD MODELS
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisperx.load_model("small", device)  # try "medium" for better alignment
align_model, metadata = whisperx.load_align_model(language_code="de", device=device)

# -----------------------------
# PROCESS ALL FILES
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

        # Load audio + transcript
        audio, sr = torchaudio.load(audio_file)
        transcript = transcript_file.read_text(encoding="utf-8")

        # Run WhisperX transcription/alignment
        result = model.transcribe(audio, language="de")  # use German for alignment
        aligned = whisperx.align(
            result["segments"], align_model, metadata, audio, sr, device
        )

        # Export aligned clips + metadata
        base_id = audio_file.stem
        for i, seg in enumerate(aligned["segments"]):
            start, end = seg["start"], seg["end"]
            text = seg["text"].strip()
            clip_name = f"{base_id}_{i:03d}.wav"

            # Save audio segment
            torchaudio.save(
                OUTPUT_DIR / "wavs" / clip_name,
                audio[:, int(start * sr):int(end * sr)],
                sr,
            )

            # Write metadata entry
            f_meta.write(f"{clip_name}|{text}\n")

print("✅ All recordings processed!")
print(f"Output in: {OUTPUT_DIR}")