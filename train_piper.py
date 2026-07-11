#!/usr/bin/env python3
"""
Standalone Piper TTS training script for East Frisian.

Run from the repo root after the environment is set up (piper1-gpl installed).
Works on Google Colab or any machine with a CUDA GPU.

Usage:
    python train_piper.py                       # single-speaker (default)
    python train_piper.py --mode multi          # multi-speaker + averaged export
    python train_piper.py --mode both           # train both and compare
    python train_piper.py --mode both --batch-size 16 --max-epochs 1000
    python train_piper.py --mode single --pretrained-ckpt /path/to/checkpoint.ckpt
"""

import argparse
import csv
import glob
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Fixed configuration ────────────────────────────────────────────────────────
DATASET_DIR        = Path("data/oostfraeisk")
PRETRAINED_REPO    = "rhasspy/piper-checkpoints"
PRETRAINED_FILE    = "de/de_DE/thorsten/medium/epoch=3135-step=2702056.ckpt"
SAMPLE_RATE        = 22050
ESPEAK_VOICE       = "de"
OPSET_VERSION      = 15

# ── Helpers ───────────────────────────────────────────────────────────────────

def apply_patches() -> None:
    """Patch piper1-gpl bugs (idempotent — safe to run multiple times)."""
    try:
        import piper
    except ImportError as exc:
        raise RuntimeError(
            "piper is not importable; activate the Piper training environment first"
        ) from exc

    piper_dir = Path(piper.__file__).resolve().parent
    dataset_py = piper_dir / "train" / "vits" / "dataset.py"
    export_py = piper_dir / "train" / "export_onnx.py"
    lightning_py = piper_dir / "train" / "vits" / "lightning.py"

    if not dataset_py.exists():
        raise FileNotFoundError(f"Piper training source not found: {dataset_py}")

    # Bug 1: custom phoneme map is loaded but never passed to phonemes_to_ids()
    PATCH_MARKER = "# PATCH-PHONEME-MAP: Use custom phoneme map for phoneme-to-ID conversion"
    lines = dataset_py.read_text().splitlines(keepends=True)
    if not any(PATCH_MARKER in line for line in lines):
        insert_idx = next(
            (i for i, l in enumerate(lines) if "elif self.phoneme_type == PhonemeType.PINYIN:" in l),
            None,
        )
        if insert_idx is not None:
            patch = [
                "\n",
                f"            {PATCH_MARKER}\n",
                "            _custom_map = phoneme_id_map\n",
                "            phonemes_to_ids = lambda phonemes, **kwargs: default_phonemes_to_ids(phonemes, id_map=_custom_map)\n",
                "\n",
            ]
            for j, pl in enumerate(patch):
                lines.insert(insert_idx + j, pl)
            dataset_py.write_text("".join(lines))
            log.info("Patched dataset.py: phonemes_to_ids() now uses custom map")
        else:
            log.error("Could not find insertion point in dataset.py")
    else:
        log.info("dataset.py already patched")

    # Bug 2: PyTorch ≥2.6 defaults to dynamo ONNX exporter — force legacy
    src = export_py.read_text()
    if "dynamo=False" not in src:
        src = src.replace("verbose=False,", "dynamo=False,\n        verbose=False,")
        export_py.write_text(src)
        log.info("Patched export_onnx.py: using legacy TorchScript exporter")
    else:
        log.info("export_onnx.py already patched")

    # Bug 3: Piper logs every test utterance as audio after every epoch. This
    # makes long TensorBoard event files several gigabytes large.
    AUDIO_MARKER = "# PATCH-SPARSE-AUDIO-LOGGING"
    src = lightning_py.read_text()
    if AUDIO_MARKER not in src:
        if "import os\n" not in src:
            src = src.replace("import ast\n", "import ast\nimport os\n", 1)
        target = (
            "        if self.trainer.sanity_checking:\n"
            "            return super().on_validation_end()\n"
        )
        replacement = target + (
            "\n"
            f"        {AUDIO_MARKER}\n"
            "        audio_every = max(1, int(os.environ.get(\"PIPER_AUDIO_LOG_EVERY\", \"25\")))\n"
            "        is_last_epoch = self.current_epoch == (self.trainer.max_epochs - 1)\n"
            "        if (self.current_epoch % audio_every != 0) and not is_last_epoch:\n"
            "            return super().on_validation_end()\n"
        )
        if target not in src:
            raise RuntimeError("Could not patch Piper audio logging frequency")
        lightning_py.write_text(src.replace(target, replacement, 1))
        log.info("Patched lightning.py: sparse validation audio logging")
    else:
        log.info("lightning.py audio logging already patched")


def download_pretrained_ckpt() -> str:
    """Download the German Thorsten medium checkpoint from HuggingFace Hub."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        log.error("huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    log.info("Downloading pretrained checkpoint...")
    ckpt = hf_hub_download(
        repo_id=PRETRAINED_REPO,
        filename=PRETRAINED_FILE,
        repo_type="dataset",
    )
    log.info("Pretrained checkpoint: %s", ckpt)
    return ckpt


def build_phoneme_map(out_dir: Path) -> Tuple[Path, int]:
    """
    Scan metadata.csv for all NFD-normalised characters and build a custom
    phoneme-ID map that extends Piper's default map with any missing symbols.
    Saves to out_dir/phoneme_id_map.json and returns (path, num_symbols).
    """
    from collections import Counter
    try:
        from piper.phoneme_ids import DEFAULT_PHONEME_ID_MAP
    except ImportError:
        log.error("piper package not importable. Install piper1-gpl first.")
        sys.exit(1)

    metadata_path = DATASET_DIR / "metadata.csv"
    all_chars: Counter = Counter()

    with open(metadata_path, "r", encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="|"):
            text = row[-1]
            all_chars.update(unicodedata.normalize("NFD", text))

    phoneme_id_map = dict(DEFAULT_PHONEME_ID_MAP)
    next_id = max(max(ids) for ids in phoneme_id_map.values()) + 1
    added = []
    for char in sorted(all_chars):
        if char not in phoneme_id_map:
            phoneme_id_map[char] = [next_id]
            added.append((char, next_id))
            next_id += 1

    if added:
        log.info("Added %d characters to phoneme map", len(added))

    out_dir.mkdir(parents=True, exist_ok=True)
    phonemes_path = out_dir / "phoneme_id_map.json"
    with open(phonemes_path, "w", encoding="utf-8") as f:
        json.dump(phoneme_id_map, f, ensure_ascii=False, indent=2)

    num_symbols = max(max(ids) for ids in phoneme_id_map.values()) + 1
    log.info("Phoneme map → %s  (num_symbols=%d)", phonemes_path, num_symbols)
    return phonemes_path, num_symbols


def validate_metadata(metadata_path: Path, *, multispeaker: bool = False) -> None:
    """Validate metadata, audio files, and checked-in speaker assignments."""
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8", newline="") as metadata_file:
        rows = list(csv.reader(metadata_file, delimiter="|"))

    if not rows:
        raise ValueError(f"Metadata is empty: {metadata_path}")

    seen_ids = set()
    speakers = set()
    for row_number, row in enumerate(rows, start=1):
        if len(row) != 3:
            raise ValueError(
                f"{metadata_path}:{row_number}: expected exactly 3 columns, "
                f"got {len(row)}; check CSV quoting"
            )

        utt_id = row[0].strip()
        text = row[-1].strip()
        if not utt_id or not text:
            raise ValueError(f"{metadata_path}:{row_number}: empty id or text")
        if utt_id in seen_ids:
            raise ValueError(f"{metadata_path}:{row_number}: duplicate id {utt_id!r}")
        seen_ids.add(utt_id)

        audio_path = DATASET_DIR / "wavs" / utt_id
        if not audio_path.exists():
            audio_path = audio_path.with_suffix(".wav")
        if not audio_path.is_file():
            raise FileNotFoundError(
                f"{metadata_path}:{row_number}: audio not found for {utt_id!r}"
            )

        if multispeaker:
            speaker = row[1].strip()
            if not speaker:
                raise ValueError(f"{metadata_path}:{row_number}: empty speaker")
            speakers.add(speaker)

    if multispeaker:
        expected_speakers = {"speaker_1", "speaker_2"}
        if speakers != expected_speakers:
            raise ValueError(
                f"Expected speakers {sorted(expected_speakers)}, got {sorted(speakers)}"
            )

        canonical_path = DATASET_DIR / "metadata.csv"
        with open(canonical_path, "r", encoding="utf-8", newline="") as canonical_file:
            canonical_rows = list(csv.reader(canonical_file, delimiter="|"))

        if len(rows) != len(canonical_rows):
            raise ValueError(
                f"Metadata row count mismatch: {metadata_path} has {len(rows)}, "
                f"{canonical_path} has {len(canonical_rows)}"
            )

        for row_number, (multi_row, canonical_row) in enumerate(
            zip(rows, canonical_rows), start=1
        ):
            if multi_row[0] != canonical_row[0]:
                raise ValueError(
                    f"{metadata_path}:{row_number}: id {multi_row[0]!r} does not "
                    f"match {canonical_row[0]!r}"
                )
            if multi_row[-1] != canonical_row[-1]:
                raise ValueError(
                    f"{metadata_path}:{row_number}: text differs from {canonical_path}"
                )

        log.info(
            "Validated %s: %d utterances, speakers=%s",
            metadata_path,
            len(rows),
            ", ".join(sorted(speakers)),
        )
    else:
        log.info("Validated %s: %d utterances", metadata_path, len(rows))


def build_callbacks(
    checkpoint_every: int,
    save_top_k: int,
    early_stopping_patience: int,
) -> list[dict]:
    """Build LightningCLI callback configuration for a comparable long run."""
    callbacks = [
        {
            "class_path": "piper_training_callbacks.ManualLRSchedulerStep",
            "init_args": {},
        },
        {
            "class_path": "lightning.pytorch.callbacks.ModelCheckpoint",
            "init_args": {
                "monitor": "val_loss",
                "mode": "min",
                "save_top_k": save_top_k,
                "save_last": True,
                "save_weights_only": False,
                "filename": "best-epoch={epoch:04d}-val_loss={val_loss:.4f}",
                "auto_insert_metric_name": False,
            },
        },
    ]

    if checkpoint_every > 0:
        callbacks.append(
            {
                "class_path": "lightning.pytorch.callbacks.ModelCheckpoint",
                "init_args": {
                    "monitor": None,
                    "save_top_k": -1,
                    "save_weights_only": True,
                    "every_n_epochs": checkpoint_every,
                    "filename": "periodic-epoch={epoch:04d}",
                    "auto_insert_metric_name": False,
                },
            }
        )

    if early_stopping_patience > 0:
        callbacks.append(
            {
                "class_path": "lightning.pytorch.callbacks.EarlyStopping",
                "init_args": {
                    "monitor": "val_loss",
                    "mode": "min",
                    "patience": early_stopping_patience,
                    "check_finite": True,
                },
            }
        )

    return callbacks


def run_training(
    metadata_path: Path,
    training_dir:  Path,
    logs_dir:      Path,
    phonemes_path: Path,
    num_symbols:   int,
    pretrained_ckpt: str,
    batch_size:    int,
    max_epochs:    int,
    num_workers:   int,
    checkpoint_every: int,
    save_top_k: int,
    early_stopping_patience: int,
    audio_log_every: int,
    num_speakers:  int = 1,
) -> None:
    """Run `python -m piper.train fit` with the given settings."""
    training_dir.mkdir(parents=True, exist_ok=True)
    cache_dir   = training_dir / "cache"
    config_path = training_dir / "config.json"

    # Clear cache so phoneme IDs are rebuilt with the patched code
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        log.info("Cleared cache at %s", cache_dir)

    cmd = [
        sys.executable, "-m", "piper.train", "fit",
        "--data.voice_name",            "oostfraeisk",
        "--data.csv_path",              str(metadata_path),
        "--data.audio_dir",             str(DATASET_DIR / "wavs"),
        "--model.sample_rate",          str(SAMPLE_RATE),
        "--data.espeak_voice",          ESPEAK_VOICE,
        "--data.cache_dir",             str(cache_dir),
        "--data.config_path",           str(config_path),
        "--data.batch_size",            str(batch_size),
        "--data.num_workers",           str(num_workers),
        "--data.validation_split",      "0.05",
        "--data.num_test_examples",     "5",
        "--trainer.max_epochs",         str(max_epochs),
        "--trainer.accelerator",        "gpu",
        "--trainer.devices",            "1",
        "--trainer.precision",          "32",
        "--trainer.default_root_dir",   str(logs_dir),
        "--trainer.callbacks", json.dumps(
            build_callbacks(
                checkpoint_every,
                save_top_k,
                early_stopping_patience,
            )
        ),
        "--data.phoneme_type",          "text",
        "--data.phonemes_path",         str(phonemes_path),
        "--data.num_symbols",           str(num_symbols),
        "--model.num_speakers",         str(num_speakers),
        "--model.vocoder_warmstart_ckpt", pretrained_ckpt,
    ]

    log.info("Starting training  (num_speakers=%d, max_epochs=%d) ...", num_speakers, max_epochs)
    env = os.environ.copy()
    env["PIPER_AUDIO_LOG_EVERY"] = str(audio_log_every)
    subprocess.run(cmd, check=True, env=env)
    log.info("Training complete!")


def find_latest_run_dir(logs_dir: Path) -> Optional[Path]:
    """Return the most recently modified Lightning version directory."""
    version_dirs = [
        Path(path)
        for pattern in (
            str(logs_dir / "lightning_logs" / "version_*"),
            str(logs_dir / "version_*"),
        )
        for path in glob.glob(pattern)
        if Path(path).is_dir()
    ]

    if not version_dirs:
        log.error("No Lightning runs found under %s", logs_dir)
        return None

    return max(version_dirs, key=lambda path: path.stat().st_mtime)


def list_run_checkpoints(logs_dir: Path) -> Tuple[Optional[Path], list[Path]]:
    """List checkpoints belonging only to the most recent Lightning run."""
    latest_version_dir = find_latest_run_dir(logs_dir)
    if latest_version_dir is None:
        return None, []

    ckpts = sorted(
        latest_version_dir.glob("checkpoints/*.ckpt"),
        key=lambda path: (checkpoint_epoch(path), path.name),
    )
    return latest_version_dir, ckpts


def checkpoint_epoch(checkpoint: Path) -> int:
    """Extract an epoch number for chronological checkpoint sorting."""
    match = re.search(r"epoch[=-](\d+)", checkpoint.name)
    return int(match.group(1)) if match else sys.maxsize


def find_checkpoint(logs_dir: Path, selection: str) -> Optional[Path]:
    """Find the requested checkpoint from the most recent training run."""
    latest_version_dir, ckpts = list_run_checkpoints(logs_dir)
    if latest_version_dir is None:
        return None

    if not ckpts:
        log.error("No checkpoints found in latest run %s", latest_version_dir)
        return None

    if selection == "last":
        last_ckpt = latest_version_dir / "checkpoints" / "last.ckpt"
        if last_ckpt.is_file():
            log.info("Using final checkpoint: %s", last_ckpt)
            return last_ckpt
        latest_ckpt = max(ckpts, key=lambda path: path.stat().st_mtime)
        log.info("No last.ckpt found; using newest checkpoint: %s", latest_ckpt)
        return latest_ckpt

    best_ckpt   = None
    best_loss   = float("inf")
    for ckpt in ckpts:
        m = re.search(r"val_loss=([0-9]+(?:\.[0-9]+)?)", str(ckpt))
        if m:
            loss = float(m.group(1))
            if loss < best_loss:
                best_loss = loss
                best_ckpt = ckpt

    if best_ckpt:
        log.info("Best checkpoint (val_loss=%.4f): %s", best_loss, best_ckpt)
    else:
        best_ckpt = ckpts[-1]
        log.info("Using latest checkpoint: %s", best_ckpt)

    return best_ckpt


def backup_existing_model(output_dir: Path) -> None:
    """Preserve an existing ONNX model before a new export overwrites it."""
    existing = [
        path
        for path in (
            output_dir / "oostfraeisk.onnx",
            output_dir / "oostfraeisk.onnx.json",
        )
        if path.is_file()
    ]
    if not existing:
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = output_dir / "archive" / timestamp
    backup_dir.mkdir(parents=True, exist_ok=False)
    for path in existing:
        shutil.copy2(path, backup_dir / path.name)
    log.info("Backed up existing model to %s", backup_dir)


def export_single_speaker(checkpoint: Path, output_dir: Path, config_src: Path) -> None:
    """Export a single-speaker ONNX model via piper.train.export_onnx."""
    backup_existing_model(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "oostfraeisk.onnx"

    result = subprocess.run([
        sys.executable, "-m", "piper.train.export_onnx",
        "--checkpoint",  str(checkpoint),
        "--output-file", str(onnx_path),
    ], check=False)

    if result.returncode != 0:
        log.error("ONNX export failed (exit code %d)", result.returncode)
        return

    if not config_src.exists():
        log.error("Config not found: %s  (did training complete successfully?)", config_src)
        return
    shutil.copy(config_src, output_dir / "oostfraeisk.onnx.json")
    log.info("Exported single-speaker model → %s", onnx_path)


def export_multi_averaged(checkpoint: Path, output_dir: Path, config_src: Path) -> None:
    """
    Export a multi-speaker model as a single-speaker ONNX by averaging the
    two learned speaker embeddings.

    Steps:
      1. Load the checkpoint
      2. Average emb_g[0] and emb_g[1]
      3. Wrap infer() so sid=0 (the averaged slot) is hardcoded — not an ONNX input
      4. Export without the 'sid' input → piper-tts treats it as single-speaker
      5. Write config JSON with num_speakers=1
    """
    import torch

    if not config_src.exists():
        log.error("Config not found: %s  (did training complete successfully?)", config_src)
        return

    try:
        from piper.train.vits.lightning import VitsModel
    except ImportError:
        log.error("Cannot import VitsModel — is piper1-gpl installed?")
        return

    backup_existing_model(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "oostfraeisk.onnx"

    log.info("Loading checkpoint for averaged export: %s", checkpoint)
    model   = VitsModel.load_from_checkpoint(str(checkpoint), map_location="cpu")
    model_g = model.model_g
    model_g.eval()

    with torch.no_grad():
        model_g.dec.remove_weight_norm()

    if model_g.n_speakers != 2:
        log.error("Expected n_speakers=2, got %d", model_g.n_speakers)
        return

    # Average the two speaker embeddings
    s0   = model_g.emb_g.weight.data[0]
    s1   = model_g.emb_g.weight.data[1]
    avg  = (s0 + s1) / 2.0
    log.info("Speaker 0 norm: %.4f | Speaker 1 norm: %.4f | Averaged norm: %.4f",
             s0.norm().item(), s1.norm().item(), avg.norm().item())

    model_g.emb_g.weight.data[0] = avg  # slot 0 now holds the averaged embedding

    # Wrapper with sid hardcoded to 0 — not exposed as an ONNX graph input
    def infer_forward_averaged(text, text_lengths, scales):
        audio = model_g.infer(
            text, text_lengths,
            noise_scale=scales[0],
            length_scale=scales[1],
            noise_scale_w=scales[2],
            sid=torch.LongTensor([0]),
        )[0].unsqueeze(1)
        return audio

    model_g.forward = infer_forward_averaged  # type: ignore

    dummy_len       = 50
    sequences       = torch.randint(0, model_g.n_vocab, (1, dummy_len), dtype=torch.long)
    seq_lengths     = torch.LongTensor([dummy_len])
    scales          = torch.FloatTensor([0.667, 1.0, 0.8])

    with torch.no_grad():
        torch.onnx.export(
            model=model_g,
            args=(sequences, seq_lengths, scales),
            f=str(onnx_path),
            dynamo=False,
            verbose=False,
            opset_version=OPSET_VERSION,
            input_names=["input", "input_lengths", "scales"],
            output_names=["output"],
            dynamic_axes={
                "input":         {0: "batch_size", 1: "phonemes"},
                "input_lengths": {0: "batch_size"},
                "output":        {0: "batch_size", 2: "time"},
            },
        )

    # Write config with num_speakers=1 so piper-tts treats the model as single-speaker
    with open(config_src) as f:
        config = json.load(f)
    config["num_speakers"] = 1
    config.pop("speaker_id_map", None)
    config_dst = output_dir / "oostfraeisk.onnx.json"
    with open(config_dst, "w") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    size_mb = os.path.getsize(onnx_path) / 1e6
    log.info("Exported averaged model → %s  (%.1f MB)", onnx_path, size_mb)
    log.info("Config → %s  (num_speakers=1)", config_dst)


# ── Test inference ────────────────────────────────────────────────────────────

TEST_SENTENCES = [
    "Moin, woo gaajt 't dii?",
    "Denkent jii, dat ik disser sats gaud uutprooten dau?",
    "Hest duu däi süen fandóóeğ al säin?",
    "Däi oorsprungelk tóól fan däi Fräisen tüsken Laauwers un Wäiser was dat Olfräisk.",
    "In disser sats gift dat kiin umluuden of anner roer dingen, man disser is trotsdeem langer as gewoon um dat tau testen.",
    "Föör disser ooversetter mussent wii eerst 'n paróleelkorpus uut düütsk un oostfräisk satsen móóken.",
    "Mii daajt d' rüeğ seer un ik góó nuu up bäer. Man eerst sal ik eeven däi tóóvel ofrüümen.",
    "Wielk Wilkens wichter willent wiet wuel waaskern - wiet waaske willent Wielk Wilkens wichter waaskern.",
    "In disser moment denk ik an früen un an 't kooken. Dan in miin läest leevend was ik 'n kok un kun gaud dansen.",
    "Däi letiinsk tóól, kört Letiin of Letiinsk, is 'n indoeuropeesk tóól, däi oorsprungelk fan däi Letiiners, däi bewooners fan Latium mit Room as tsentrum, proot't wur.",
    "Wen däi bossem soo ful, un dat läid is soo däip, un dat haart sülst man wul, dat 't doch stiel lağ un sleep. Och wat helpt dor 'n gaudiğ róód? Och wat helpt dor 'n gaud woord? Dat häiel lücht is ful kwóód! Un däi blöymtûun fersoort. Man 'n ooğstróól fan höör däi miin lecht is, miin süen. 'T is ful klööer un ful gööer! 'T is weer mâaj, wor ik bün.",
]


def synthesize_test_sentences(model_dir: Path, label: str) -> None:
    """
    Load the ONNX model from model_dir and synthesize TEST_SENTENCES.
    Output WAV files are written into model_dir as test_<label>_<n>.wav
    so single and multi results never overwrite each other.
    """
    import wave

    onnx_path = model_dir / "oostfraeisk.onnx"
    if not onnx_path.exists():
        log.error("Cannot run test inference — ONNX not found: %s", onnx_path)
        return

    try:
        from piper import PiperVoice
    except ImportError:
        log.warning("piper-tts not importable; skipping test inference.")
        return

    log.info("Loading model for test inference: %s", onnx_path)
    try:
        voice = PiperVoice.load(str(onnx_path))
    except Exception as exc:
        log.error("Failed to load model: %s", exc)
        return

    log.info("Synthesizing %d test sentences [%s]...", len(TEST_SENTENCES), label)
    for i, text in enumerate(TEST_SENTENCES):
        out_path = model_dir / f"test_{label}_{i}.wav"
        try:
            with wave.open(str(out_path), "w") as wav_file:
                if hasattr(voice, "synthesize_wav"):
                    voice.synthesize_wav(text, wav_file)
                else:
                    voice.synthesize(text, wav_file)
            log.info("  [%d] %s  →  %s", i, text, out_path)
        except Exception as exc:
            log.error("  [%d] synthesis failed: %s", i, exc)

    log.info("Test inference complete for [%s]. WAVs saved in %s/", label, model_dir)


def generate_checkpoint_comparisons(
    logs_dir: Path,
    model_dir: Path,
    config_src: Path,
    *,
    multispeaker: bool,
) -> None:
    """Export and synthesize examples for every checkpoint in the latest run."""
    run_dir, checkpoints = list_run_checkpoints(logs_dir)
    if run_dir is None or not checkpoints:
        log.warning("No checkpoints available for comparison under %s", logs_dir)
        return

    comparison_root = model_dir / "checkpoint_comparisons" / run_dir.name
    comparison_root.mkdir(parents=True, exist_ok=True)
    index = []

    log.info(
        "Generating comparison models and examples for %d checkpoint(s)...",
        len(checkpoints),
    )
    for checkpoint_number, checkpoint in enumerate(checkpoints, start=1):
        checkpoint_name = re.sub(r"[^A-Za-z0-9_.=-]+", "_", checkpoint.stem)
        output_dir = comparison_root / checkpoint_name
        log.info(
            "Checkpoint comparison %d/%d: %s",
            checkpoint_number,
            len(checkpoints),
            checkpoint.name,
        )

        if multispeaker:
            export_multi_averaged(checkpoint, output_dir, config_src)
        else:
            export_single_speaker(checkpoint, output_dir, config_src)

        onnx_path = output_dir / "oostfraeisk.onnx"
        exported = onnx_path.is_file()
        if exported:
            synthesize_test_sentences(output_dir, "comparison")

        index.append(
            {
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_name": checkpoint.name,
                "epoch": (
                    checkpoint_epoch(checkpoint)
                    if checkpoint_epoch(checkpoint) != sys.maxsize
                    else None
                ),
                "output_directory": str(output_dir.resolve()),
                "exported": exported,
            }
        )

    index_path = comparison_root / "index.json"
    with open(index_path, "w", encoding="utf-8") as index_file:
        json.dump(index, index_file, ensure_ascii=False, indent=2)
    log.info("Checkpoint comparison index → %s", index_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train East Frisian Piper TTS — single-speaker, multi-speaker, or both",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode", choices=["single", "multi", "both"], default="single",
        help="Training mode",
    )
    parser.add_argument("--batch-size",      type=int,  default=16)
    parser.add_argument("--max-epochs",      type=int,  default=3000)
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="Data-loader worker processes (0 is the safest setting)",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=250,
        help="Save a weights-only listening checkpoint every N epochs (0 disables)",
    )
    parser.add_argument(
        "--save-top-k", type=int, default=3,
        help="Keep this many full checkpoints with the lowest val_loss",
    )
    parser.add_argument(
        "--early-stopping-patience", type=int, default=0,
        help="Stop after N unimproved validation epochs (0 disables)",
    )
    parser.add_argument(
        "--audio-log-every", type=int, default=25,
        help="Log TensorBoard example audio every N epochs",
    )
    parser.add_argument(
        "--checkpoint-selection", choices=["last", "best"], default="last",
        help="Checkpoint exported after training; periodic checkpoints remain available",
    )
    parser.add_argument(
        "--pretrained-ckpt", type=str, default=None,
        help="Path to pretrained .ckpt (auto-downloaded if omitted)",
    )
    parser.add_argument(
        "--skip-patches", action="store_true",
        help="Skip piper1-gpl bug patches (use if already applied)",
    )
    parser.add_argument(
        "--skip-inference", action="store_true",
        help="Skip test-sentence synthesis after export",
    )
    parser.add_argument(
        "--skip-export", action="store_true",
        help="Train and save checkpoints without exporting/overwriting the ONNX model",
    )
    parser.add_argument(
        "--skip-checkpoint-samples", action="store_true",
        help="Do not export and synthesize examples for every saved checkpoint",
    )
    args = parser.parse_args()
    if args.checkpoint_every < 0:
        parser.error("--checkpoint-every cannot be negative")
    if args.save_top_k < 1:
        parser.error("--save-top-k must be at least 1")
    if args.early_stopping_patience < 0:
        parser.error("--early-stopping-patience cannot be negative")
    if args.audio_log_every < 1:
        parser.error("--audio-log-every must be at least 1")

    # Sanity-check working directory
    if not DATASET_DIR.exists():
        log.error(
            "Dataset directory '%s' not found. "
            "Run this script from the repo root (oostfraeisk_text_to_speech/).",
            DATASET_DIR,
        )
        sys.exit(1)

    run_single = args.mode in ("single", "both")
    run_multi = args.mode in ("multi", "both")

    validate_metadata(DATASET_DIR / "metadata.csv")
    if run_multi:
        validate_metadata(
            DATASET_DIR / "metadata_multispeaker.csv", multispeaker=True
        )

    # 1. Patch piper1-gpl
    if not args.skip_patches:
        apply_patches()

    # 2. Pretrained checkpoint
    pretrained_ckpt = args.pretrained_ckpt or download_pretrained_ckpt()

    # 3. Build shared phoneme map  (always in piper_training/)
    shared_training_dir = Path("piper_training")
    phonemes_path, num_symbols = build_phoneme_map(shared_training_dir)

    # ── Single-speaker ────────────────────────────────────────────────────────
    if run_single:
        log.info("=" * 60)
        log.info("SINGLE-SPEAKER TRAINING")
        log.info("=" * 60)
        logs_dir = Path("piper_training_single_logs")
        run_training(
            metadata_path  = DATASET_DIR / "metadata.csv",
            training_dir   = shared_training_dir,
            logs_dir       = logs_dir,
            phonemes_path  = phonemes_path,
            num_symbols    = num_symbols,
            pretrained_ckpt= pretrained_ckpt,
            batch_size     = args.batch_size,
            max_epochs     = args.max_epochs,
            num_workers    = args.num_workers,
            checkpoint_every = args.checkpoint_every,
            save_top_k     = args.save_top_k,
            early_stopping_patience = args.early_stopping_patience,
            audio_log_every = args.audio_log_every,
            num_speakers   = 1,
        )
        selected = (
            None
            if args.skip_export
            else find_checkpoint(logs_dir, args.checkpoint_selection)
        )
        if selected:
            export_single_speaker(
                selected,
                Path("model_piper"),
                shared_training_dir / "config.json",
            )
            if not args.skip_inference:
                synthesize_test_sentences(Path("model_piper"), "single")
                if not args.skip_checkpoint_samples:
                    generate_checkpoint_comparisons(
                        logs_dir,
                        Path("model_piper"),
                        shared_training_dir / "config.json",
                        multispeaker=False,
                    )

    # ── Multi-speaker ─────────────────────────────────────────────────────────
    if run_multi:
        log.info("=" * 60)
        log.info("MULTI-SPEAKER TRAINING  (2 speakers → averaged export)")
        log.info("=" * 60)
        multi_metadata  = DATASET_DIR / "metadata_multispeaker.csv"
        training_dir_m  = Path("piper_training_multi")
        logs_dir_m      = Path("piper_training_multi_logs")
        run_training(
            metadata_path  = multi_metadata,
            training_dir   = training_dir_m,
            logs_dir       = logs_dir_m,
            phonemes_path  = phonemes_path,
            num_symbols    = num_symbols,
            pretrained_ckpt= pretrained_ckpt,
            batch_size     = args.batch_size,
            max_epochs     = args.max_epochs,
            num_workers    = args.num_workers,
            checkpoint_every = args.checkpoint_every,
            save_top_k     = args.save_top_k,
            early_stopping_patience = args.early_stopping_patience,
            audio_log_every = args.audio_log_every,
            num_speakers   = 2,
        )
        selected_m = (
            None
            if args.skip_export
            else find_checkpoint(logs_dir_m, args.checkpoint_selection)
        )
        if selected_m:
            export_multi_averaged(
                selected_m,
                Path("model_piper_multi"),
                training_dir_m / "config.json",
            )
            if not args.skip_inference:
                synthesize_test_sentences(Path("model_piper_multi"), "multi")
                if not args.skip_checkpoint_samples:
                    generate_checkpoint_comparisons(
                        logs_dir_m,
                        Path("model_piper_multi"),
                        training_dir_m / "config.json",
                        multispeaker=True,
                    )


if __name__ == "__main__":
    main()
