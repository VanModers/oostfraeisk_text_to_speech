import argparse
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, login


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload Piper ONNX model to Hugging Face model repo and update Space app files."
    )
    parser.add_argument("--token", default=os.getenv("HF_TOKEN"), help="HF token (or set HF_TOKEN)")
    parser.add_argument(
        "--model-repo-id",
        default="VanModers114/East_Frisian_TTS",
        help="Target Hugging Face model repo id",
    )
    parser.add_argument(
        "--space-repo-id",
        default="VanModers114/East_Frisian_TTS",
        help="Target Hugging Face Space repo id",
    )
    parser.add_argument(
        "--onnx-path",
        default="model_piper/oostfraeisk.onnx",
        help="Path to ONNX model file",
    )
    parser.add_argument(
        "--onnx-config-path",
        default="model_piper/oostfraeisk.onnx.json",
        help="Path to ONNX config json",
    )
    parser.add_argument(
        "--app-path",
        default="app_piper.py",
        help="Path to local app source file",
    )
    parser.add_argument(
        "--model-filename",
        default="oostfraeisk.onnx",
        help="Filename to use inside model repo",
    )
    return parser.parse_args()


def ensure_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def main() -> None:
    args = parse_args()

    onnx_path = Path(args.onnx_path)
    onnx_config_path = Path(args.onnx_config_path)
    app_path = Path(args.app_path)

    ensure_exists(onnx_path)
    ensure_exists(onnx_config_path)
    ensure_exists(app_path)

    if args.token:
        login(token=args.token)
    else:
        login()

    api = HfApi()

    # 1) Upload model assets to model repo
    api.create_repo(repo_id=args.model_repo_id, repo_type="model", exist_ok=True)

    api.upload_file(
        path_or_fileobj=str(onnx_path),
        path_in_repo=args.model_filename,
        repo_id=args.model_repo_id,
        repo_type="model",
        commit_message="Upload ONNX model",
    )
    api.upload_file(
        path_or_fileobj=str(onnx_config_path),
        path_in_repo=f"{args.model_filename}.json",
        repo_id=args.model_repo_id,
        repo_type="model",
        commit_message="Upload ONNX model config",
    )

    print(f"Model uploaded: https://huggingface.co/{args.model_repo_id}")

    # 2) Update Space runtime files (no model binaries in space repo)
    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)

        shutil.copy(app_path, tmpdir / "app.py")

        (tmpdir / "requirements.txt").write_text(
            "gradio\n"
            "piper-tts\n"
            "huggingface_hub\n",
            encoding="utf-8",
        )

        (tmpdir / "packages.txt").write_text("espeak-ng\n", encoding="utf-8")

        api.upload_folder(
            folder_path=str(tmpdir),
            repo_id=args.space_repo_id,
            repo_type="space",
            commit_message="Use model from HF model repo",
        )

    print(f"Space updated: https://huggingface.co/spaces/{args.space_repo_id}")
    print("Done. The Space app will download model files from the model repo at startup.")


if __name__ == "__main__":
    main()
