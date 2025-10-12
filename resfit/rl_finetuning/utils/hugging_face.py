# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0

from __future__ import annotations

import multiprocessing as mp
import os
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from resfit.rl_finetuning.config.performance import PERF_CONFIG

_hf_api = HfApi()

# Performance optimization settings
COMPRESSION_LEVEL = PERF_CONFIG.compression_level
MAX_WORKERS = min(PERF_CONFIG.max_workers, mp.cpu_count())


def _create_archive_fast(cache_dir: Path, tar_path: Path) -> None:
    """Create tar.gz archive using optimized settings for speed."""
    print(f"[HF] Creating optimized archive {tar_path}")

    # Try using pigz (parallel gzip) if available for better performance
    if PERF_CONFIG.use_pigz:
        try:
            # Use pigz for parallel compression if available
            subprocess.run(["pigz", "--version"], capture_output=True, text=True, check=True)
            print("[HF] Using pigz for parallel compression")

            # Create tar with pigz compression
            with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as temp_tar:
                temp_tar_path = temp_tar.name

            # Create uncompressed tar first
            with tarfile.open(temp_tar_path, "w") as tar:
                tar.add(cache_dir, arcname=cache_dir.name)

            # Compress with pigz
            with open(tar_path, "wb") as f:
                subprocess.run(
                    ["pigz", f"-{COMPRESSION_LEVEL}", "-c", temp_tar_path], stdout=f, check=True
                )

            # Clean up temp file
            os.unlink(temp_tar_path)

        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback to standard tarfile with optimized settings
            print("[HF] Using standard tarfile with optimized compression")
            with tarfile.open(tar_path, "w:gz", compresslevel=COMPRESSION_LEVEL) as tar:
                tar.add(cache_dir, arcname=cache_dir.name)
    else:
        # Use standard tarfile with optimized settings
        print("[HF] Using standard tarfile with optimized compression")
        with tarfile.open(tar_path, "w:gz", compresslevel=COMPRESSION_LEVEL) as tar:
            tar.add(cache_dir, arcname=cache_dir.name)


def _extract_archive_fast(local_archive: str, target_parent: Path) -> None:
    """Extract tar.gz archive using optimized settings for speed."""
    print(f"[HF] Extracting archive {local_archive}")

    # Try using pigz for parallel decompression if available
    if PERF_CONFIG.use_pigz:
        try:
            subprocess.run(["pigz", "--version"], capture_output=True, text=True, check=True)
            print("[HF] Using pigz for parallel decompression")

            # Decompress with pigz and extract with tar
            with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as temp_tar:
                temp_tar_path = temp_tar.name

            # Decompress with pigz
            with open(temp_tar_path, "wb") as f:
                subprocess.run(["pigz", "-d", "-c", local_archive], stdout=f, check=True)

            # Extract tar
            with tarfile.open(temp_tar_path, "r") as tar:
                tar.extractall(path=target_parent)

            # Clean up temp file
            os.unlink(temp_tar_path)

        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback to standard tarfile
            print("[HF] Using standard tarfile for extraction")
            with tarfile.open(local_archive, "r:gz") as tar:
                tar.extractall(path=target_parent)
    else:
        # Use standard tarfile
        print("[HF] Using standard tarfile for extraction")
        with tarfile.open(local_archive, "r:gz") as tar:
            tar.extractall(path=target_parent)


def optimized_replay_buffer_dumps(replay_buffer, cache_dir: Path) -> None:
    """Optimized ReplayBuffer dumps with performance monitoring."""
    print(f"[HF] Starting optimized ReplayBuffer dumps to {cache_dir}")
    start_time = time.time()

    # Ensure cache directory exists
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Use the standard dumps method but with timing
    replay_buffer.dumps(cache_dir)

    end_time = time.time()
    print(f"[HF] ReplayBuffer dumps completed in {end_time - start_time:.2f} seconds")


def optimized_replay_buffer_loads(replay_buffer, cache_dir: Path) -> None:
    """Optimized ReplayBuffer loads with performance monitoring."""
    print(f"[HF] Starting optimized ReplayBuffer loads from {cache_dir}")
    start_time = time.time()

    # Use the standard loads method but with timing
    replay_buffer.loads(cache_dir)

    end_time = time.time()
    print(f"[HF] ReplayBuffer loads completed in {end_time - start_time:.2f} seconds")


def _hf_download_buffer(repo_id: str, hash_id: str, target_parent: Path) -> Path | None:
    """Download and extract a tar.gz buffer archive from *repo_id* if present.

    The archive must be named ``<hash_id>.tar.gz`` and contain a single directory
    with the same name as the hash.  Extraction happens into *target_parent*.
    Returns True if the download & extraction succeeded, False otherwise.
    """
    filename = f"{hash_id}.tar.gz"

    # Always attempt the download.  `huggingface_hub` will use its own disk
    # cache and skip the transfer if the file is already present locally.
    try:
        local_archive = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
        )
    except Exception as e:
        # Gracefully handle the common "repo or file doesn't exist" case so the
        # caller can decide to create & upload a fresh buffer.  Any other
        # exception will be bubbled up to make debugging easier.
        try:
            from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

            if isinstance(e, (RepositoryNotFoundError, EntryNotFoundError)):
                print(f"[HF] Buffer {filename} not found in repo {repo_id}")
                return None
        except ImportError:
            # Older hub versions -- fall back to string matching.
            if "404" in str(e):
                print(f"[HF] Buffer {filename} not found in repo {repo_id}")
                return None

        # Unexpected error → re-raise.
        raise

    # ------------------------------------------------------------------
    # Extract only if we haven't done so already ------------------------
    # ------------------------------------------------------------------
    extract_dir = target_parent / hash_id

    # Ensure parent directory exists
    target_parent.mkdir(parents=True, exist_ok=True)

    if not extract_dir.exists():
        _extract_archive_fast(local_archive, target_parent)
        print(f"[HF] Extracted buffer to {extract_dir}")

    print(f"[HF] Ready -- buffer {filename} available at {extract_dir}")
    return extract_dir


def _hf_upload_buffer(repo_id: str, cache_dir: Path, hash_id: str):
    """Create a tar.gz archive for *cache_dir* and upload it to Hugging Face Hub."""
    filename = f"{hash_id}.tar.gz"
    tar_path = cache_dir.parent / filename

    # Create archive if it doesn't exist locally
    if not tar_path.exists():
        _create_archive_fast(cache_dir, tar_path)
        print(f"[HF] Created archive {tar_path}")

    # Upload to HF Hub
    # First, ensure that the target repository exists (create it if needed).
    print(f"[HF] Creating repo {repo_id}")
    _hf_api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        exist_ok=True,  # no-op if the repo already exists
        private=False,  # default to public so collaborators can access
    )

    # Attempt the upload (retry once if it fails due to a missing repo)
    print(f"[HF] Uploading archive {filename} to repo {repo_id}")
    _hf_api.upload_file(
        repo_id=repo_id,
        path_in_repo=filename,
        path_or_fileobj=str(tar_path),
        repo_type="dataset",
    )
    print(f"[HF] Uploaded archive {filename} to repo {repo_id}")
