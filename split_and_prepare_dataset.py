import os
from pathlib import Path
import torchaudio
from aeneas.executetask import ExecuteTask
from aeneas.task import Task
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
# FUNCTIONS
# -----------------------------
def split_text_into_sentences(text):
    """Split transcript into sentences using simple punctuation split"""
    sentences = re.split(r'[.!?]', text)
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
            print(f"⚠️ No transcript for {audio_file}, skipping...")
            continue

        print(f"🎙️ Processing {audio_file.name}...")

        # Load transcript
        transcript = transcript_file.read_text(encoding="utf-8")
        sentences = split_text_into_sentences(transcript)

        # Prepare Aeneas input
        tmp_text_path = audio_file.with_suffix(".sentences.txt")
        tmp_text_path.write_text("\n".join(sentences), encoding="utf-8")

        # Configure Aeneas task
        config_string = "task_language=de|is_text_type=plain|os_task_file_format=json"
        task = Task(config_string=config_string)
        task.audio_file_path_absolute = str(audio_file)
        task.text_file_path_absolute = str(tmp_text_path)
        task.sync_map_file_path_absolute = str(tmp_text_path.with_suffix(".json"))

        # Run alignment
        ExecuteTask(task).execute()
        task.output_sync_map_file()
        import json
        with open(task.sync_map_file_path_absolute, "r", encoding="utf-8") as f:
            sync_data = json.load(f)

        # Load audio for slicing
        audio, sr = torchaudio.load(audio_file)

        # Export clips
        base_id = audio_file.stem
        for i, fragment in enumerate(sync_data["fragments"]):
            start = float(fragment["begin"])
            end = float(fragment["end"])
            text = fragment["lines"][0].strip()
            clip_name = f"{base_id}_{i:03d}.wav"

            torchaudio.save(
                OUTPUT_DIR / "wavs" / clip_name,
                audio[:, int(start*sr):int(end*sr)],
                sr
            )

            f_meta.write(f"{clip_name}|{text}\n")

        # Clean up temporary files
        tmp_text_path.unlink()
        tmp_text_path.with_suffix(".json").unlink()

print("✅ Done! Dataset in", OUTPUT_DIR)