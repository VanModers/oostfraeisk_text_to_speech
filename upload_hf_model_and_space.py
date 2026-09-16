import argparse
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi


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
        default="model_piper_multi/oostfraeisk.onnx",
        help="Path to ONNX model file",
    )
    parser.add_argument(
        "--onnx-config-path",
        default="model_piper_multi/oostfraeisk.onnx.json",
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
    parser.add_argument(
        "--skip-model-binaries",
        action="store_true",
        help="Update documentation and Space files without re-uploading ONNX binaries",
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
    provenance_path = Path("audio_provenance.py")
    model_readme_path = Path("hf_model_README.md")
    space_readme_path = Path("hf_space_README.md")
    transparency_path = Path("AI_AUDIO_TRANSPARENCY.md")
    requirements_path = Path("hf_space_requirements.txt")
    packages_path = Path("hf_space_packages.txt")

    if not args.skip_model_binaries:
        ensure_exists(onnx_path)
        ensure_exists(onnx_config_path)
    ensure_exists(app_path)
    ensure_exists(provenance_path)
    ensure_exists(model_readme_path)
    ensure_exists(space_readme_path)
    ensure_exists(transparency_path)
    ensure_exists(requirements_path)
    ensure_exists(packages_path)

    # Passing the token directly avoids persisting deployment credentials to the
    # machine's Hugging Face login cache. HfApi also uses an existing cached
    # login when args.token is omitted.
    api = HfApi(token=args.token)

    # 1) Upload model assets to model repo
    api.create_repo(repo_id=args.model_repo_id, repo_type="model", exist_ok=True)

    if not args.skip_model_binaries:
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
    api.upload_file(
        path_or_fileobj=str(model_readme_path),
        path_in_repo="README.md",
        repo_id=args.model_repo_id,
        repo_type="model",
        commit_message="Document model and AI-audio provenance",
    )
    api.upload_file(
        path_or_fileobj=str(transparency_path),
        path_in_repo="TRANSPARENCY.md",
        repo_id=args.model_repo_id,
        repo_type="model",
        commit_message="Document AI-audio marking and detection",
    )

    print(f"Model uploaded: https://huggingface.co/{args.model_repo_id}")

    # 2) Update Space runtime files (no model binaries in space repo)
    api.create_repo(repo_id=args.space_repo_id, repo_type="space", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)

        shutil.copy(app_path, tmpdir / "app.py")
        shutil.copy(provenance_path, tmpdir / "audio_provenance.py")
        shutil.copy(space_readme_path, tmpdir / "README.md")
        shutil.copy(transparency_path, tmpdir / "TRANSPARENCY.md")
        shutil.copy(requirements_path, tmpdir / "requirements.txt")
        shutil.copy(packages_path, tmpdir / "packages.txt")

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
