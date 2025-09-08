import torchaudio
from pathlib import Path
from faster_whisper import WhisperModel
import re

# -----------------------------
# CONFIG
# -----------------------------
INPUT_DIR = Path("long_recordings")  # .wav/.mp3
OUTPUT_DIR = Path("data/oostfraeisk")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "wavs").mkdir(parents=True, exist_ok=True)

metadata_path = OUTPUT_DIR / "metadata.csv"

# -----------------------------
# FUNCTIONS
# -----------------------------
def split_text_into_sentences(text):
    """Simple sentence splitter"""
    sentences = re.split(r'(?<=[.!?]) +', text)
    return [s.strip() for s in sentences if s.strip()]

# -----------------------------
# LOAD MODEL
# -----------------------------
device = "cuda"  # Colab GPU
model_size = "medium"  # change to "small" or "large" if desired
model = WhisperModel(model_size, device=device, compute_type="float16")

# -----------------------------
# PROCESS FILES
# -----------------------------
with open(metadata_path, "w", encoding="utf-8") as f_meta:
    for audio_file in INPUT_DIR.glob("*"):
        if audio_file.suffix.lower() not in [".wav", ".mp3"]:
            continue

        print(f"🎙️ Processing {audio_file.name}...")

        # Transcribe with faster-whisper
        segments, info = model.transcribe(str(audio_file), beam_size=5)
        
        # Load audio for slicing
        audio, sr = torchaudio.load(audio_file)

        clip_counter = 0
        for segment in segments:
            # Split segment text into sentences
            sentences = split_text_into_sentences(segment.text)
            seg_duration = segment.end - segment.start
            # Approximate sentence timestamps proportionally
            if len(sentences) == 1:
                starts = [segment.start]
                ends = [segment.end]
            else:
                starts = [segment.start + i * seg_duration / len(sentences) for i in range(len(sentences))]
                ends = [segment.start + (i + 1) * seg_duration / len(sentences) for i in range(len(sentences))]

            for s_text, s_start, s_end in zip(sentences, starts, ends):
                clip_name = f"{audio_file.stem}_{clip_counter:03d}.wav"
                clip_counter += 1
                torchaudio.save(
                    OUTPUT_DIR / "wavs" / clip_name,
                    audio[:, int(s_start*sr):int(s_end*sr)],
                    sr
                )
                f_meta.write(f"{clip_name}|{s_text}\n")

print("✅ Done! Dataset in", OUTPUT_DIR)