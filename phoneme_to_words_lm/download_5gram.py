"""Download KenLM language model files from HuggingFace Hub.

Repo: https://huggingface.co/nckcard/phoneme-lm-5gram

Available files:
    5gram_unpruned.bin  (~50 GB)  — full 5-gram KenLM binary
    5gram_pruned.bin    (~5 GB)   — pruned 5-gram KenLM binary
    lexicon.txt                   — word-to-phoneme lexicon
    tokens.txt                    — token list (BLANK, SIL, AA..ZH)

Usage:
    python -m phoneme_to_words_lm.download_5gram --output-dir /path/to/lm_files
    python -m phoneme_to_words_lm.download_5gram --pruned --output-dir /path/to/lm_files
"""

from __future__ import annotations

import argparse
import os
import sys

from huggingface_hub import hf_hub_download

from phoneme_to_words_lm.utils import HF_CACHE_DIR

REPO_ID = "nckcard/phoneme-lm-5gram"

# Files that are always downloaded
COMMON_FILES = ["lexicon.txt", "tokens.txt"]


def download_5gram_files(
    output_dir: str,
    pruned: bool = False,
    cache_dir: str | None = None,
    repo_id: str = REPO_ID,
) -> dict[str, str]:
    """Download LM files from HuggingFace Hub.

    Args:
        output_dir: Directory to symlink/copy downloaded files into.
        pruned: If True, download 5gram_pruned.bin instead of 5gram_unpruned.bin.
        cache_dir: HuggingFace cache directory. Defaults to HF_CACHE_DIR.
        repo_id: HuggingFace repo ID.

    Returns:
        Dict mapping filename to its local path.
    """
    cache_dir = os.path.expanduser(cache_dir or HF_CACHE_DIR)
    output_dir = os.path.expanduser(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    bin_file = "5gram_pruned.bin" if pruned else "5gram_unpruned.bin"
    files_to_download = COMMON_FILES + [bin_file]

    paths = {}
    for filename in files_to_download:
        print(f"Downloading {filename} from {repo_id}...")
        cached_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir,
        )

        # Create a symlink in output_dir pointing to the cached file.
        # On Windows where symlinks may not be available, copy instead.
        dest = os.path.join(output_dir, filename)
        if os.path.exists(dest) or os.path.islink(dest):
            os.remove(dest)
        try:
            os.symlink(cached_path, dest)
        except OSError:
            import shutil
            shutil.copy2(cached_path, dest)

        paths[filename] = dest
        print(f"  -> {dest}")

    print("\nDone. Files are ready in:", output_dir)
    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Download KenLM language model files from HuggingFace Hub.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to place downloaded files.",
    )
    parser.add_argument(
        "--pruned",
        action="store_true",
        help="Download the pruned 5-gram model instead of the unpruned (~50 GB) version.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help=f"HuggingFace cache directory (default: {HF_CACHE_DIR}).",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=REPO_ID,
        help=f"HuggingFace repo ID (default: {REPO_ID}).",
    )
    args = parser.parse_args()

    download_5gram_files(
        output_dir=args.output_dir,
        pruned=args.pruned,
        cache_dir=args.cache_dir,
        repo_id=args.repo_id,
    )


if __name__ == "__main__":
    main()
