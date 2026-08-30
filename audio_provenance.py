"""Machine-readable provenance and AudioSeal watermarking for generated WAV files."""

from __future__ import annotations

import json
import os
import struct
import threading
import wave
from pathlib import Path
from typing import Any

# AudioSeal 0.2 otherwise invokes torch.compile lazily on CPU. The eager path
# avoids a C++ compiler dependency and a costly first-request compilation on
# small Hugging Face CPU Spaces.
os.environ.setdefault("NO_TORCH_COMPILE", "1")


AIGC_CHUNK_ID = b"aigc"
PROVENANCE_VERSION = 1
DEFAULT_MESSAGE_BITS = "0100010101000110"  # ASCII "EF"
_AUDIOSEAL_SAMPLE_RATE = 16_000
_generator = None
_detector = None
_model_lock = threading.Lock()


def _chunk(chunk_id: bytes, payload: bytes) -> bytes:
    if len(chunk_id) != 4:
        raise ValueError("A RIFF chunk id must contain four bytes")
    padding = b"\x00" if len(payload) % 2 else b""
    return chunk_id + struct.pack("<I", len(payload)) + payload + padding


def _provenance_payload(watermark_applied: bool) -> dict[str, Any]:
    return {
        "schema": "org.oostfraeisk.ai-audio-provenance",
        "version": PROVENANCE_VERSION,
        "ai_generated": True,
        "generator": "Piper TTS",
        "model": os.environ.get(
            "MODEL_REPO_ID", "VanModers114/East_Frisian_TTS"
        ),
        "watermark": "AudioSeal" if watermark_applied else None,
        "watermark_message": (
            os.environ.get("AUDIOSEAL_MESSAGE_BITS", DEFAULT_MESSAGE_BITS)
            if watermark_applied
            else None
        ),
    }


def append_wav_provenance(path: str | Path, watermark_applied: bool) -> None:
    """Append an explicit AIGC JSON chunk and standard RIFF INFO comment."""
    wav_path = Path(path)
    payload = json.dumps(
        _provenance_payload(watermark_applied),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    info_text = (
        b"AI-generated speech; generator=Piper TTS; "
        + (b"watermark=AudioSeal" if watermark_applied else b"watermark=none")
        + b"\x00"
    )
    list_payload = b"INFO" + _chunk(b"ICMT", info_text)

    with wav_path.open("r+b") as wav_file:
        header = wav_file.read(12)
        if len(header) != 12 or header[:4] != b"RIFF" or header[8:] != b"WAVE":
            raise ValueError(f"Not a RIFF/WAVE file: {wav_path}")
        wav_file.seek(0, os.SEEK_END)
        wav_file.write(_chunk(AIGC_CHUNK_ID, payload))
        wav_file.write(_chunk(b"LIST", list_payload))
        file_size = wav_file.tell()
        wav_file.seek(4)
        wav_file.write(struct.pack("<I", file_size - 8))


def read_wav_provenance(path: str | Path) -> dict[str, Any] | None:
    """Return the JSON stored in this project's AIGC chunk, if present."""
    with Path(path).open("rb") as wav_file:
        header = wav_file.read(12)
        if len(header) != 12 or header[:4] != b"RIFF" or header[8:] != b"WAVE":
            raise ValueError("The supplied file is not a RIFF/WAVE file")

        while chunk_header := wav_file.read(8):
            if len(chunk_header) != 8:
                break
            chunk_id, chunk_size = struct.unpack("<4sI", chunk_header)
            payload = wav_file.read(chunk_size)
            if chunk_size % 2:
                wav_file.read(1)
            if chunk_id == AIGC_CHUNK_ID:
                return json.loads(payload.decode("utf-8"))
    return None


def _load_generator():
    global _generator
    with _model_lock:
        if _generator is None:
            from audioseal import AudioSeal
            from huggingface_hub import hf_hub_download

            checkpoint = hf_hub_download(
                repo_id="facebook/audioseal", filename="generator_base.pth"
            )
            _generator = AudioSeal.load_generator(checkpoint, nbits=16)
            _generator.eval()
    return _generator


def _load_detector():
    global _detector
    with _model_lock:
        if _detector is None:
            from audioseal import AudioSeal
            from huggingface_hub import hf_hub_download

            checkpoint = hf_hub_download(
                repo_id="facebook/audioseal", filename="detector_base.pth"
            )
            _detector = AudioSeal.load_detector(checkpoint, nbits=16)
            _detector.eval()
    return _detector


def _read_pcm16_mono(path: str | Path):
    import numpy as np
    import torch

    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())

    if channels != 1 or sample_width != 2:
        raise ValueError("AudioSeal integration expects mono 16-bit PCM WAV audio")
    samples = np.frombuffer(frames, dtype="<i2").copy()
    waveform = torch.from_numpy(samples).to(torch.float32).div_(32768.0)
    return waveform.reshape(1, 1, -1), sample_rate


def _resample(waveform, input_rate: int, output_rate: int, output_size: int | None = None):
    import torch.nn.functional as functional

    if output_size is None:
        output_size = max(1, round(waveform.shape[-1] * output_rate / input_rate))
    return functional.interpolate(
        waveform, size=output_size, mode="linear", align_corners=False
    )


def apply_audioseal_watermark(path: str | Path) -> None:
    """Embed an imperceptible AudioSeal watermark while retaining Piper's format."""
    import numpy as np
    import torch

    message_bits = os.environ.get("AUDIOSEAL_MESSAGE_BITS", DEFAULT_MESSAGE_BITS)
    if len(message_bits) != 16 or set(message_bits) - {"0", "1"}:
        raise ValueError("AUDIOSEAL_MESSAGE_BITS must contain exactly 16 zero/one digits")

    waveform, sample_rate = _read_pcm16_mono(path)
    audioseal_input = _resample(waveform, sample_rate, _AUDIOSEAL_SAMPLE_RATE)
    message = torch.tensor([[int(bit) for bit in message_bits]], dtype=torch.int32)
    generator = _load_generator()

    with _model_lock, torch.inference_mode():
        watermark_16k = generator.get_watermark(
            audioseal_input,
            sample_rate=_AUDIOSEAL_SAMPLE_RATE,
            message=message,
        )
    watermark = _resample(
        watermark_16k,
        _AUDIOSEAL_SAMPLE_RATE,
        sample_rate,
        output_size=waveform.shape[-1],
    )
    output = torch.clamp(waveform + watermark, -1.0, 1.0)
    pcm = output.squeeze().mul(32767.0).round().to(torch.int16).cpu().numpy()

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(np.asarray(pcm, dtype="<i2").tobytes())


def detect_audioseal_watermark(path: str | Path) -> tuple[float, str | None]:
    """Detect AudioSeal and return its probability and optional 16-bit message."""
    import torch

    waveform, sample_rate = _read_pcm16_mono(path)
    audioseal_input = _resample(waveform, sample_rate, _AUDIOSEAL_SAMPLE_RATE)
    detector = _load_detector()
    with _model_lock, torch.inference_mode():
        probability, message = detector.detect_watermark(audioseal_input)

    probability_value = float(torch.as_tensor(probability).mean().item())
    decoded = None
    if message is not None:
        bits = torch.as_tensor(message).reshape(-1)
        decoded = "".join("1" if float(bit) >= 0.5 else "0" for bit in bits[:16])
    return probability_value, decoded
