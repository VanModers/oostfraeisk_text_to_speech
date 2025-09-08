import torchaudio
from pathlib import Path
import difflib
import json

# -----------------------------
# CONFIG
# -----------------------------
INPUT_DIR = Path("long_recordings")
SEGMENTS_DIR = Path("data/transcripts")
OUTPUT_DIR = Path("data/oostfraeisk_safe")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "wavs").mkdir(parents=True, exist_ok=True)
metadata_path = OUTPUT_DIR / "metadata.csv"

DEBUG = True
CHUNK_WORDS = 10  # split manual transcription into chunks for matching

# -----------------------------
# HELPERS
# -----------------------------
def debug_print(*args):
    if DEBUG:
        print(*args)

def split_into_chunks(text, max_words=10):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunks.append(" ".join(words[i:i+max_words]))
    return chunks

def find_best_matching_segment(chunk, segments, used_indices):
    """Return the index of the segment with the highest similarity to the chunk."""
    best_ratio = 0
    best_idx = None
    for i, seg in enumerate(segments):
        if i in used_indices:
            continue
        ratio = difflib.SequenceMatcher(None, chunk, seg['text']).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_idx = i
    return best_idx, best_ratio

def confirm_chunk(chunk, seg):
    """Ask user to confirm or correct the automatically assigned chunk."""
    print("\n---")
    print(f"Suggested chunk: {chunk}")
    print(f"Matched segment: {seg['text']}")
    print(f"Start: {seg['start']:.2f}, End: {seg['end']:.2f}")
    user_input = input("Press Enter to accept, or type correct text: ").strip()
    if user_input:
        return user_input
    return chunk

def align_manual_to_segments(manual_chunks, segments):
    aligned = []
    used_indices = set()
    for chunk_idx, chunk in enumerate(manual_chunks):
        best_idx, ratio = find_best_matching_segment(chunk, segments, used_indices)
        if best_idx is None:
            debug_print(f"⚠️ No matching segment for chunk {chunk_idx}")
            continue

        used_indices.add(best_idx)
        seg = segments[best_idx]

        # Ask user to confirm/correct
        final_text = confirm_chunk(chunk, seg)
        aligned.append({
            "text": final_text,
            "start": seg['start'],
            "end": seg['end']
        })

    return aligned

# -----------------------------
# PROCESS FILES
# -----------------------------
with open(metadata_path, "w", encoding="utf-8") as f_meta:
    for audio_file in INPUT_DIR.glob("*"):
        if audio_file.suffix.lower() not in [".wav", ".mp3"]:
            continue

        transcript_file = audio_file.with_suffix(".txt")
        if not transcript_file.exists():
            debug_print(f"⚠️ No transcript found for {audio_file.name}, skipping...")
            continue

        segments_file = SEGMENTS_DIR / f"{audio_file.stem}_segments.json"
        if not segments_file.exists():
            debug_print(f"⚠️ No saved segments for {audio_file.name}, skipping...")
            continue

        debug_print(f"🎙️ Processing {audio_file.name}...")

        # Load manual transcription
        transcript = transcript_file.read_text(encoding="utf-8")
        manual_chunks = split_into_chunks(transcript, max_words=CHUNK_WORDS)
        debug_print(f"Split transcript into {len(manual_chunks)} chunks")

        # Load Whisper segments
        segments = json.loads(segments_file.read_text(encoding="utf-8"))
        debug_print(f"Loaded {len(segments)} segments")

        if not segments:
            debug_print(f"⚠️ No segments in {segments_file.name}, skipping...")
            continue

        # Load audio
        waveform, sr = torchaudio.load(audio_file)
        debug_print(f"Audio loaded: {waveform.shape[1]/sr:.2f} seconds")

        # Align chunks using sequence matching + user confirmation
        aligned_segments = align_manual_to_segments(manual_chunks, segments)

        # Export WAV clips + save confirmed text
        for i, seg in enumerate(aligned_segments):
            start, end = seg.get("start"), seg.get("end")
            if start is None or end is None or end <= start:
                debug_print(f"⚠️ Skipping segment {i}: invalid timing ({start}, {end})")
                continue
            start_sample, end_sample = int(start * sr), int(end * sr)
            clip = waveform[:, start_sample:end_sample]
            if clip.shape[1] == 0:
                debug_print(f"⚠️ Skipping segment {i}: zero-length clip")
                continue
            clip_name = f"{audio_file.stem}_{i:03d}.wav"
            torchaudio.save(OUTPUT_DIR / "wavs" / clip_name, clip, sr)
            f_meta.write(f"{clip_name}|{seg['text']}\n")
            debug_print(f"✅ Saved {clip_name}: {seg['text']}")

print("✅ Done! All recordings have confirmed transcription.")