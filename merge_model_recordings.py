from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import wave
from pathlib import Path


def natural_sort_key(path: Path) -> list[object]:
    parts = re.split(r"(\d+)", path.stem)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def join_wavs(folder: Path, pattern: str, output_name: str, silence_seconds: float) -> Path:
    wav_files = sorted(folder.glob(pattern), key=natural_sort_key)

    if not wav_files:
        wav_files = sorted(
            [p for p in folder.glob("*.wav") if p.name != output_name],
            key=natural_sort_key,
        )

    if not wav_files:
        raise FileNotFoundError(f"No WAV files found in {folder}")

    first_file = wav_files[0]
    with wave.open(str(first_file), "rb") as first_wav:
        params = first_wav.getparams()

    silence_frame_count = int(params.framerate * silence_seconds)
    silence_bytes = b"\x00" * silence_frame_count * params.nchannels * params.sampwidth

    output_path = folder / output_name
    with wave.open(str(output_path), "wb") as out_wav:
        out_wav.setparams(params)

        for index, wav_path in enumerate(wav_files):
            with wave.open(str(wav_path), "rb") as in_wav:
                in_params = in_wav.getparams()
                # Frame counts can differ by file, but stream format must match.
                if (
                    in_params.nchannels != params.nchannels
                    or in_params.sampwidth != params.sampwidth
                    or in_params.framerate != params.framerate
                    or in_params.comptype != params.comptype
                ):
                    raise ValueError(
                        "WAV format mismatch: "
                        f"{wav_path.name} has "
                        f"(channels={in_params.nchannels}, sampwidth={in_params.sampwidth}, "
                        f"framerate={in_params.framerate}, comptype={in_params.comptype}), "
                        f"expected (channels={params.nchannels}, sampwidth={params.sampwidth}, "
                        f"framerate={params.framerate}, comptype={params.comptype})"
                    )

                out_wav.writeframes(in_wav.readframes(in_wav.getnframes()))

            if index < len(wav_files) - 1 and silence_frame_count > 0:
                out_wav.writeframes(silence_bytes)

    return output_path


def wav_to_mp3(wav_path: Path, bitrate: str) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is not installed or not found in PATH")

    mp3_path = wav_path.with_suffix(".mp3")
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(wav_path),
        "-codec:a",
        "libmp3lame",
        "-b:a",
        bitrate,
        str(mp3_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return mp3_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create one combined WAV for model_piper and one for model_piper_multi, "
            "with silence between each clip, then convert them to MP3."
        )
    )
    parser.add_argument(
        "--silence",
        type=float,
        default=0.4,
        help="Silence duration in seconds inserted between clips (default: 0.4)",
    )
    parser.add_argument(
        "--bitrate",
        type=str,
        default="192k",
        help="MP3 bitrate for conversion (default: 192k)",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    targets = [
        (root / "model_piper", "test_single_*.wav", "combined_model_piper.wav"),
        (root / "model_piper_multi", "test_multi_*.wav", "combined_model_piper_multi.wav"),
    ]

    for folder, pattern, output_name in targets:
        wav_output = join_wavs(folder, pattern, output_name, args.silence)
        print(f"Created: {wav_output}")

        mp3_output = wav_to_mp3(wav_output, args.bitrate)
        print(f"Created: {mp3_output}")


if __name__ == "__main__":
    main()