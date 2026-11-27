"""Download the verified polyp checkpoints before starting the AI service.

Run ``python -m download_polyp_weights --help`` from the AI-service root.
Authentication uses HF_ACCESS_TOKEN, then the Hub's HF_TOKEN or saved login.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shlex
import tempfile

from huggingface_hub import hf_hub_download

from paths import hf_token


CHECKPOINTS = (
    {
        "repo_id": "iquana/iquana-cellpose-sam-polyp",
        "revision": "b66e63c09e893ac773a7aa43b7b95caf73bd53d8",
        "filename": "cpsam_best",
        "sha256": "8d17682741428d0c71b93dc6112de9ff9d6595fc586fe797bf4566423b788eba",
        "env": "IQUANA_CELLPOSE_POLYP_CHECKPOINT",
    },
    {
        "repo_id": "iquana/iquana-yolo-polyp",
        "revision": "15ffccd52e260f5096ddfbe20516c446ccc48b32",
        "filename": "yolo_polyp_finetuned.pt",
        "sha256": "7e9ac2eec159b4d227d1b8c6553de224a982f420f96b738362ef04bbfad08edf",
        "env": "IQUANA_YOLO_POLYP_CHECKPOINT",
    },
)


def download_weights(
    env_file: Path, *, cache_dir: Path | None = None, local_files_only: bool = False
) -> dict[str, str]:
    """Verify both artifacts before atomically replacing a shell configuration file."""
    configuration = {}
    for checkpoint in CHECKPOINTS:
        path = Path(hf_hub_download(
            repo_id=checkpoint["repo_id"],
            filename=checkpoint["filename"],
            revision=checkpoint["revision"],
            cache_dir=str(cache_dir.expanduser()) if cache_dir is not None else None,
            token=hf_token(),
            local_files_only=local_files_only,
        )).absolute()
        with path.open("rb") as stream:
            actual_sha = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual_sha != checkpoint["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {checkpoint['repo_id']}/{checkpoint['filename']}")
        configuration[checkpoint["env"]] = str(path)
    configuration["IQUANA_CELLPOSE_POLYP_CHECKPOINT_SHA256"] = CHECKPOINTS[0]["sha256"]

    env_file = env_file.expanduser().resolve()
    env_file.parent.mkdir(parents=True, exist_ok=True)
    # Keep an existing valid configuration if either download or verification fails.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=env_file.parent, delete=False) as stream:
            temporary = Path(stream.name)
            for name, value in configuration.items():
                stream.write(f"export {name}={shlex.quote(value)}\n")
        os.replace(temporary, env_file)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return configuration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True,
                        help="Output shell configuration; source it before starting the service.")
    parser.add_argument("--cache-dir", type=Path,
                        help="Persistent HF Hub cache directory (defaults to the Hub configuration).")
    parser.add_argument("--local-files-only", action="store_true",
                        help="Verify already-cached weights without contacting Hugging Face.")
    args = parser.parse_args()
    try:
        download_weights(args.env_file, cache_dir=args.cache_dir,
                         local_files_only=args.local_files_only)
    except Exception as exc:
        parser.exit(1, f"Checkpoint setup failed ({type(exc).__name__}). Check HF access, disk space and checkpoint integrity; configuration was not replaced.\n")
    print(f"Verified both checkpoints. Before starting the AI service, run:\nsource {shlex.quote(str(args.env_file.expanduser().resolve()))}")


if __name__ == "__main__":
    main()
