"""
East Frisian TTS Dataset Generator — HuggingFace Space Version

A browser-based recording tool for adding new audio to the East Frisian TTS dataset.
Sentences that already have recordings (listed in metadata.csv) are skipped.
Only newly recorded audio is included in the downloadable zip.

Deploy as a HuggingFace Space (Gradio SDK).
"""

import os
import re
import zipfile
import tempfile
import shutil
import numpy as np
import gradio as gr
import librosa
import soundfile as sf

# -------- CONFIG --------
TEXT_FILE = "texts/rec0.txt"
METADATA_FILE = "data/oostfraeisk/metadata.csv"
NEW_WAVS_DIR = "new_recordings/wavs"
NEW_METADATA_FILE = "new_recordings/metadata.csv"
SAMPLE_RATE = 22050
CHANNELS = 1
MAX_WORDS = 25
RERECORD_COUNT = 300          # how many sentences to re-record
RERECORD_DIR = "rerecord/wavs"  # where re-recordings are saved
# -------------------------

os.makedirs(NEW_WAVS_DIR, exist_ok=True)
os.makedirs(RERECORD_DIR, exist_ok=True)


# =========================================================================
# Text splitting (same logic as original dataset_generator.py)
# =========================================================================

def split_sentences(text):
    """Split into sentences by newlines, then break long ones into chunks."""
    sentences = re.split(r'(?<=[\n])+', text.strip())
    processed = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        words = s.split()
        if len(words) > MAX_WORDS:
            for i in range(0, len(words), MAX_WORDS):
                chunk = " ".join(words[i:i + MAX_WORDS])
                processed.append(chunk.strip())
        else:
            processed.append(s)
    return processed


# =========================================================================
# Metadata helpers
# =========================================================================

def load_existing_metadata(metadata_path):
    """Return (set of recorded sentences, last numeric index)."""
    existing_sentences = set()
    last_index = 0
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) >= 2:
                    existing_sentences.add(parts[1])
                    match = re.match(r"sentence_(\d+)", parts[0])
                    if match:
                        last_index = max(last_index, int(match.group(1)))
    return existing_sentences, last_index


def load_new_metadata(metadata_path):
    """Return (set of recorded sentences, last numeric index) from new recordings."""
    existing_sentences = set()
    last_index = 0
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) >= 2:
                    existing_sentences.add(parts[1])
                    match = re.match(r"sentence_(\d+)", parts[0])
                    if match:
                        last_index = max(last_index, int(match.group(1)))
    return existing_sentences, last_index


# Padding (in seconds) added after the trim point so that final
# consonants (e.g. trailing 't') are not clipped.
TAIL_PAD_SEC = 0.2


def trim_silence(audio, sr=SAMPLE_RATE, top_db=35):
    """Trim leading/trailing silence, keeping a small tail pad."""
    trimmed, (start, end) = librosa.effects.trim(audio, top_db=top_db)
    # Re-attach a small portion after the trim endpoint so final
    # plosives / fricatives are not cut off.
    pad_samples = int(TAIL_PAD_SEC * sr)
    end_padded = min(end + pad_samples, len(audio))
    return audio[start:end_padded]


# =========================================================================
# Build the list of sentences that still need recording
# =========================================================================

def build_pending_sentences():
    """Return list of sentences not yet in either the original or new metadata."""
    with open(TEXT_FILE, "r", encoding="utf-8") as f:
        text = f.read()
    all_sentences = split_sentences(text)

    orig_sentences, orig_last_idx = load_existing_metadata(METADATA_FILE)
    new_sentences, new_last_idx = load_new_metadata(NEW_METADATA_FILE)

    recorded = orig_sentences | new_sentences
    last_index = max(orig_last_idx, new_last_idx)

    pending = [s for s in all_sentences if s not in recorded]
    return pending, last_index


# =========================================================================
# Gradio app state & callbacks
# =========================================================================

def get_status():
    """Return a status string summarising progress."""
    with open(TEXT_FILE, "r", encoding="utf-8") as f:
        text = f.read()
    total = len(split_sentences(text))

    orig, _ = load_existing_metadata(METADATA_FILE)
    new, _ = load_new_metadata(NEW_METADATA_FILE)
    done = len(orig) + len(new)
    remaining = total - done

    new_count = len(new)
    return (
        f"**Total sentences:** {total}  \n"
        f"**Already recorded (original dataset):** {len(orig)}  \n"
        f"**Newly recorded in this session:** {new_count}  \n"
        f"**Remaining:** {remaining}"
    )


def get_next_sentence():
    """Fetch the next sentence that needs recording."""
    pending, _ = build_pending_sentences()
    if not pending:
        return "All sentences have been recorded!", ""
    return pending[0], f"({len(pending)} sentences remaining)"


def save_recording(audio, sentence_text):
    """
    Process and save a recording.

    Parameters
    ----------
    audio : tuple (sample_rate, numpy array)  — from gr.Audio(type="numpy")
    sentence_text : str — the sentence that was read

    Returns
    -------
    status_msg, next_sentence, remaining_info, updated_status
    """
    if audio is None:
        return "No audio received. Please record again.", sentence_text, "", get_status(), None

    if not sentence_text or sentence_text.startswith("All sentences"):
        return "No more sentences to record.", sentence_text, "", get_status(), None

    sr_in, data = audio

    # Convert to float32 mono
    if data.dtype != np.float32:
        data = data.astype(np.float32)
        # Gradio int16 range
        if np.max(np.abs(data)) > 2.0:
            data = data / 32768.0

    if data.ndim > 1:
        data = data.mean(axis=1)

    # Resample to target rate if needed
    if sr_in != SAMPLE_RATE:
        data = librosa.resample(data, orig_sr=sr_in, target_sr=SAMPLE_RATE)

    # Trim silence
    data = trim_silence(data)

    # Determine filename index
    _, last_index = build_pending_sentences()
    new_index = last_index + 1
    filename = f"sentence_{new_index:04d}"
    filepath = os.path.join(NEW_WAVS_DIR, f"{filename}.wav")

    # Save wav
    sf.write(filepath, data, SAMPLE_RATE)

    # Append to new metadata
    with open(NEW_METADATA_FILE, "a", encoding="utf-8") as f:
        f.write(f"{filename}|{sentence_text}|{sentence_text}\n")

    # Get next sentence
    next_sent, remaining = get_next_sentence()
    status = get_status()

    return f"Saved {filename}.wav", next_sent, remaining, status, None


def skip_sentence(sentence_text):
    """Skip the current sentence without recording and move to the next."""
    pending, _ = build_pending_sentences()

    if not pending:
        return "All sentences have been recorded!", "", get_status()

    # Find the current sentence in pending and return the next one
    try:
        idx = pending.index(sentence_text)
        if idx + 1 < len(pending):
            next_sent = pending[idx + 1]
            remaining = f"({len(pending)} sentences remaining)"
        else:
            # Wrap around to first
            next_sent = pending[0]
            remaining = f"({len(pending)} sentences remaining)"
    except ValueError:
        # Current sentence not found in pending (maybe already saved), just get next
        next_sent = pending[0]
        remaining = f"({len(pending)} sentences remaining)"

    return next_sent, remaining, get_status()


def build_combined_metadata():
    """Return the full combined metadata (original + new) as a string."""
    lines = []
    # Original entries
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.rstrip("\n")
                if stripped:
                    lines.append(stripped)
    # New entries
    if os.path.exists(NEW_METADATA_FILE):
        with open(NEW_METADATA_FILE, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.rstrip("\n")
                if stripped:
                    lines.append(stripped)
    return "\n".join(lines) + "\n" if lines else ""


def download_new_recordings():
    """Create a zip of new wav files + combined (old+new) metadata."""
    new_sents, _ = load_new_metadata(NEW_METADATA_FILE)
    if not new_sents:
        return None

    zip_path = os.path.join(tempfile.gettempdir(), "new_recordings.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        combined = build_combined_metadata()
        zf.writestr("new_recordings/metadata.csv", combined)

        if os.path.isdir(NEW_WAVS_DIR):
            for wav_file in sorted(os.listdir(NEW_WAVS_DIR)):
                if wav_file.endswith(".wav"):
                    full_path = os.path.join(NEW_WAVS_DIR, wav_file)
                    zf.write(full_path, f"new_recordings/wavs/{wav_file}")

    return zip_path


# =========================================================================
# Re-record helpers — work through the first RERECORD_COUNT metadata entries
# =========================================================================

def load_rerecord_sentences():
    """Return list of (sentence_id, text) for the first RERECORD_COUNT entries."""
    entries = []
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) >= 2:
                    entries.append((parts[0], parts[1]))
                if len(entries) >= RERECORD_COUNT:
                    break
    return entries


def get_rerecord_pending():
    """Return (sentence_id, text) pairs not yet saved in RERECORD_DIR."""
    done = {
        wav[:-4]
        for wav in os.listdir(RERECORD_DIR)
        if wav.endswith(".wav")
    }
    return [(sid, text) for sid, text in load_rerecord_sentences() if sid not in done]


def get_rerecord_status():
    pending = get_rerecord_pending()
    done = RERECORD_COUNT - len(pending)
    return f"**Re-record progress:** {done} / {RERECORD_COUNT} done  \u2014  {len(pending)} remaining"


def get_next_rerecord():
    pending = get_rerecord_pending()
    if not pending:
        return "All 300 sentences re-recorded!", "", f"(0 of {RERECORD_COUNT} remaining)"
    sid, text = pending[0]
    return text, sid, f"({len(pending)} of {RERECORD_COUNT} remaining)"


def save_rerecording(audio, sentence_text, sentence_id):
    if audio is None:
        next_text, next_id, remaining = get_next_rerecord()
        return "No audio received. Please record again.", sentence_text, sentence_id, remaining, get_rerecord_status(), None

    sr_in, data = audio

    if data.dtype != np.float32:
        data = data.astype(np.float32)
        if np.max(np.abs(data)) > 2.0:
            data = data / 32768.0

    if data.ndim > 1:
        data = data.mean(axis=1)

    if sr_in != SAMPLE_RATE:
        data = librosa.resample(data, orig_sr=sr_in, target_sr=SAMPLE_RATE)

    data = trim_silence(data)

    filepath = os.path.join(RERECORD_DIR, f"{sentence_id}.wav")
    sf.write(filepath, data, SAMPLE_RATE)

    next_text, next_id, remaining = get_next_rerecord()
    return f"Saved {sentence_id}.wav", next_text, next_id, remaining, get_rerecord_status(), None


def skip_rerecord(sentence_id):
    pending = get_rerecord_pending()
    if not pending:
        return "All done!", "", f"(0 of {RERECORD_COUNT} remaining)", get_rerecord_status()
    try:
        idx = next(i for i, (sid, _) in enumerate(pending) if sid == sentence_id)
        sid, text = pending[(idx + 1) % len(pending)]
    except StopIteration:
        sid, text = pending[0]
    remaining = f"({len(pending)} of {RERECORD_COUNT} remaining)"
    return text, sid, remaining, get_rerecord_status()


def download_rerecordings():
    """Zip all re-recordings (keeps original sentence IDs, ready to replace originals)."""
    wavs = [w for w in os.listdir(RERECORD_DIR) if w.endswith(".wav")] if os.path.isdir(RERECORD_DIR) else []
    if not wavs:
        return None
    zip_path = os.path.join(tempfile.gettempdir(), "rerecordings.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for wav in sorted(wavs):
            zf.write(os.path.join(RERECORD_DIR, wav), f"rerecord/wavs/{wav}")
    return zip_path


# =========================================================================
# Gradio UI
# =========================================================================


def build_ui():
    with gr.Blocks(title="East Frisian TTS — Dataset Recorder") as demo:
        gr.Markdown("# East Frisian TTS Dataset Recorder")

        with gr.Tabs():

            # ------------------------------------------------------------------
            # Tab 1 — Record new sentences
            # ------------------------------------------------------------------
            with gr.Tab("Record new sentences"):
                gr.Markdown(
                    "Record sentences for the East Frisian Low Saxon TTS dataset.  \n"
                    "Sentences already in the original dataset are skipped automatically.  \n"
                    "New recordings can be downloaded as a zip file."
                )

                status_md = gr.Markdown(value=get_status)

                with gr.Row():
                    with gr.Column(scale=2):
                        sentence_box = gr.Textbox(
                            label="Read this sentence aloud:",
                            interactive=False,
                            lines=3,
                        )
                        remaining_info = gr.Textbox(
                            label="",
                            interactive=False,
                            lines=1,
                        )
                    with gr.Column(scale=1):
                        audio_input = gr.Audio(
                            sources=["microphone"],
                            type="numpy",
                            label="Record your voice",
                        )

                with gr.Row():
                    save_btn = gr.Button("Save recording", variant="primary")
                    skip_btn = gr.Button("Skip sentence")
                    load_btn = gr.Button("Load first sentence")

                save_msg = gr.Textbox(label="Status", interactive=False, lines=1)

                gr.Markdown("---")
                download_btn = gr.Button("Download new recordings (zip)")
                download_file = gr.File(label="Download")

                def on_load():
                    sent, rem = get_next_sentence()
                    return sent, rem, get_status()

                load_btn.click(fn=on_load, inputs=[], outputs=[sentence_box, remaining_info, status_md])
                save_btn.click(fn=save_recording, inputs=[audio_input, sentence_box],
                               outputs=[save_msg, sentence_box, remaining_info, status_md, audio_input])
                skip_btn.click(fn=skip_sentence, inputs=[sentence_box],
                               outputs=[sentence_box, remaining_info, status_md])
                download_btn.click(fn=download_new_recordings, inputs=[], outputs=[download_file])
                demo.load(fn=on_load, inputs=[], outputs=[sentence_box, remaining_info, status_md])

            # ------------------------------------------------------------------
            # Tab 2 — Re-record first 300 sentences
            # ------------------------------------------------------------------
            with gr.Tab(f"Re-record first {RERECORD_COUNT} sentences"):
                gr.Markdown(
                    f"Re-record the first **{RERECORD_COUNT}** sentences from the original dataset.  \n"
                    "Re-recordings are saved with their original sentence IDs to `rerecord/wavs/`.  \n"
                    "Download the zip and replace the corresponding files in `data/oostfraeisk/wavs/`."
                )

                rr_status_md = gr.Markdown(value=get_rerecord_status)

                # Hidden state to carry the sentence ID between callbacks
                rr_sentence_id = gr.State("")

                with gr.Row():
                    with gr.Column(scale=2):
                        rr_sentence_box = gr.Textbox(
                            label="Read this sentence aloud:",
                            interactive=False,
                            lines=3,
                        )
                        rr_remaining_info = gr.Textbox(
                            label="",
                            interactive=False,
                            lines=1,
                        )
                    with gr.Column(scale=1):
                        rr_audio_input = gr.Audio(
                            sources=["microphone"],
                            type="numpy",
                            label="Record your voice",
                        )

                with gr.Row():
                    rr_save_btn = gr.Button("Save recording", variant="primary")
                    rr_skip_btn = gr.Button("Skip sentence")
                    rr_load_btn = gr.Button("Load first sentence")

                rr_save_msg = gr.Textbox(label="Status", interactive=False, lines=1)

                gr.Markdown("---")
                rr_download_btn = gr.Button("Download re-recordings (zip)")
                rr_download_file = gr.File(label="Download")

                def rr_on_load():
                    text, sid, remaining = get_next_rerecord()
                    return text, sid, remaining, get_rerecord_status()

                rr_load_btn.click(
                    fn=rr_on_load, inputs=[],
                    outputs=[rr_sentence_box, rr_sentence_id, rr_remaining_info, rr_status_md],
                )
                rr_save_btn.click(
                    fn=save_rerecording,
                    inputs=[rr_audio_input, rr_sentence_box, rr_sentence_id],
                    outputs=[rr_save_msg, rr_sentence_box, rr_sentence_id, rr_remaining_info, rr_status_md, rr_audio_input],
                )
                rr_skip_btn.click(
                    fn=skip_rerecord,
                    inputs=[rr_sentence_id],
                    outputs=[rr_sentence_box, rr_sentence_id, rr_remaining_info, rr_status_md],
                )
                rr_download_btn.click(fn=download_rerecordings, inputs=[], outputs=[rr_download_file])

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch()
